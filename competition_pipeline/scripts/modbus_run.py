#!/usr/bin/env python3
"""持久会话的 Modbus 远程作业运行（纳博特 22.07 标准流程）。

现场结论（2026-08-2x，MR07S-930 / C1102）：控制器 Modbus 从站的"已连接"状态
依赖主站**持续轮询**——官方手册 6.5.1 明确指出 Modbus Poll 的 Scan Rate 必须
改为 100ms，否则示教器"已连接/未连接"闪烁，且此状态下写入控制寄存器会被
**静默忽略**（可读写但无效）。本脚本按官方 6.5.5 流程实现：

  1. 建立并**保持**一个 Modbus-TCP 会话（每 100ms 轮询一个可读寄存器块）；
  2. 会话确认存活后，按序写入：45 选择程序 -> 61 运行次数 -> 71 确认次数
     -> 29 伺服就绪 -> 19 运行作业文件；
  3. 持续轮询 + 每秒回读用户坐标系1 TCP 位姿（7000 端口），观察机器人是否运动。

地址约定：手册地址为 PLC 1-based（45/61/71/29/19），线路上默认转换
``wire = manual - 1``（Modbus Poll 勾选 "PLC Addresses" 的约定）；
加 ``--raw-addresses`` 则手册数值直接作为 wire 地址。
"""

import argparse
import sys
import time

from competition_pipeline.controller_tcp import (
    ModbusTcpClient,
    TcpEndpoint,
)
from competition_pipeline.nexbot_tcp import (
    NexBotTcpEndpoint,
    NexBotTcpRobotController,
)

# ---- 手册地址码（PLC 1-based） -------------------------------------------
ADDR_SELECT_PROGRAM = 45   # 4x: 机器人X选中远程程序（1=程序1）
ADDR_RUN_COUNT = 61        # 4x: 运行次数（写入后不生效，需 71 确认）
ADDR_RUN_COUNT_CONFIRM = 71  # 4x: 确认修改运行次数（写 1 生效）
ADDR_SERVO_READY = 29      # 4x: 切至伺服就绪（1=就绪 / 2=停止）
ADDR_RUN_JOB = 19          # 4x: 运行作业文件（1=运行 / 0=停止 / 3=断点启动）

FC_READ_HOLDING = 3
FC_READ_INPUT = 4

# 心跳轮询的候选可读块（wire 地址, 功能码, 数量），取第一个成功的。
HEARTBEAT_CANDIDATES = (
    (18, FC_READ_HOLDING, 1),   # PLC 19（运行状态回读候选）
    (28, FC_READ_HOLDING, 1),   # PLC 29（伺服就绪回读候选）
    (44, FC_READ_HOLDING, 1),   # PLC 45（程序选择回读候选）
    (60, FC_READ_HOLDING, 1),   # PLC 61（运行次数回读候选）
    (0, FC_READ_INPUT, 16),
    (16, FC_READ_INPUT, 16),
    (100, FC_READ_INPUT, 16),
    (1000, FC_READ_INPUT, 8),
    (2000, FC_READ_INPUT, 8),
)


def plc_to_wire(plc_address, raw=False):
    """手册 PLC 1-based 地址 -> 线路 0-based 地址。"""
    return int(plc_address) if raw else int(plc_address) - 1


def build_sequence(program, runs, raw=False):
    """返回按序写入 (名称, plc地址, 值) 列表。"""
    return [
        ("选择程序 45", ADDR_SELECT_PROGRAM, program),
        ("运行次数 61", ADDR_RUN_COUNT, runs),
        ("确认次数 71", ADDR_RUN_COUNT_CONFIRM, 1),
        ("伺服就绪 29", ADDR_SERVO_READY, 1),
        ("运行作业 19", ADDR_RUN_JOB, 1),
    ]


class ModbusRunSession:
    """一个持续呼吸的 Modbus 会话 + 触发序列 + 位姿观察。"""

    def __init__(self, host, port, unit, poll_s=0.1, state_s=1.0,
                 raw_addresses=False, verbose=False):
        self.endpoint = TcpEndpoint(
            host, port,
            connect_timeout_s=2.0, io_timeout_s=1.0, keepalive=True,
        )
        self.client = ModbusTcpClient(self.endpoint, unit_id=unit)
        self.poll_s = float(poll_s)
        self.state_s = float(state_s)
        self.raw = bool(raw_addresses)
        self.verbose = bool(verbose)
        self._block = None
        self._state_client = None

    # -- 会话 ---------------------------------------------------------------
    def _read_block(self, spec):
        address, function, quantity = spec
        if function == FC_READ_HOLDING:
            return self.client.read_holding_registers(address, quantity)
        return self.client.read_input_registers(address, quantity)

    def open_heartbeat_block(self):
        """找第一个可读寄存器块作为心跳，返回描述。"""
        for spec in HEARTBEAT_CANDIDATES:
            try:
                values = self._read_block(spec)
            except Exception:
                continue
            self._block = (spec, values)
            return "0x{}/{}@{}(wire) 读取成功: {}".format(
                "03" if spec[1] == FC_READ_HOLDING else "04",
                spec[2], spec[0], values,
            )
        raise RuntimeError("没有找到可读的寄存器块（心跳无法保持）")

    def heartbeat_ok(self):
        """读一次心跳块，成功返回真。"""
        spec, _ = self._block
        values = self._read_block(spec)
        self._block = (spec, values)
        return True

    # -- 状态回读 -------------------------------------------------------------
    def ensure_state_client(self):
        if self._state_client is None:
            endpoint = NexBotTcpEndpoint(
                host=self.endpoint.host,
                port_state=7000,
                robot=1, channel=1,
                pose_frame="UCS", motion_coord=3, tool_id=1, user_id=1,
            )
            self._state_client = NexBotTcpRobotController(endpoint)
        return self._state_client

    def read_ucs(self):
        import numpy as np
        from competition_pipeline.geometry import inexbot_abc_from_transform
        state = self.ensure_state_client().read_state()
        xyz_m, abc_rad = inexbot_abc_from_transform(state.base_from_gripper)
        return tuple(xyz_m * 1000.0), tuple(np.degrees(abc_rad))

    # -- 流程 ---------------------------------------------------------------
    def _write(self, name, plc_addr, value):
        wire = plc_to_wire(plc_addr, self.raw)
        self.client.write_single_register(wire, value)
        print("  [OK] 写入 {} -> wire {} = {}".format(name, wire, value))

    def run_sequence(self, program, runs):
        print(">> 触发运行序列 (程序{})".format(program))
        for name, plc_addr, value in build_sequence(program, runs, self.raw):
            self._write(name, plc_addr, value)
            time.sleep(0.3)

    def watch(self, seconds):
        print(">> 观察 {}s（心跳保持每 {:.0f}ms 一次）...".format(
            seconds, self.poll_s * 1000))
        deadline = time.monotonic() + seconds
        last_ucs = None
        next_state = 0.0
        moved = []
        while time.monotonic() < deadline:
            heartbeat_ok = False
            try:
                self.heartbeat_ok()
                heartbeat_ok = True
            except Exception as error:
                print("  [心跳失败] {} (尝试重连)".format(error))
                try:
                    self.client.close()
                    self.client.connect()
                    self.open_heartbeat_block()
                    heartbeat_ok = True
                except Exception as reconnect_error:
                    print("  [重连失败] {}".format(reconnect_error))
            now = time.monotonic()
            if heartbeat_ok and now >= next_state:
                next_state = now + self.state_s
                try:
                    xyz, abc = self.read_ucs()
                    if last_ucs is None:
                        last_ucs = (xyz, abc)
                        moved.append((xyz, abc))
                        print("  UCS起点 x={:8.2f} y={:8.2f} z={:8.2f} "
                              "A={:7.2f} B={:7.2f} C={:7.2f}".format(*xyz, *abc))
                    else:
                        delta = max(abs(a - b) for a, b in zip(xyz, last_ucs[0]))
                        if delta > 0.5:
                            moved.append((xyz, abc))
                            last_ucs = (xyz, abc)
                            print("  UCS变化 x={:8.2f} y={:8.2f} z={:8.2f} "
                                  "A={:7.2f} B={:7.2f} C={:7.2f}".format(*xyz, *abc))
                except Exception as error:
                    print("  [位姿读取失败] {}".format(error))
            time.sleep(min(self.poll_s, 0.05))
        return len(moved) > 1

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
        if self._state_client is not None:
            try:
                self._state_client.close()
            except Exception:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.200")
    parser.add_argument("--port", type=int, default=502)
    parser.add_argument("--unit", type=int, default=1)
    parser.add_argument("--program", type=int, default=1,
                        help="远程程序编号（45 写入值）")
    parser.add_argument("--runs", type=int, default=1,
                        help="运行次数（61 写入值）")
    parser.add_argument("--poll-s", type=float, default=0.1,
                        help="心跳轮询周期（官方要求 0.1s=100ms）")
    parser.add_argument("--watch", type=float, default=60.0,
                        help="触发后观察秒数")
    parser.add_argument("--hold", action="store_true",
                        help="观察完后保持会话不退出")
    parser.add_argument("--no-stop", action="store_true",
                        help="退出前不写 19=0 停止")
    parser.add_argument("--session-only", action="store_true",
                        help="只保持会话不触发运行（验证示教器显示【已连接】）")
    parser.add_argument("--raw-addresses", action="store_true",
                        help="手册地址直接作为 wire 地址（不减 1）")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    session = ModbusRunSession(
        args.host, args.port, args.unit,
        poll_s=args.poll_s, raw_addresses=args.raw_addresses,
        verbose=args.verbose,
    )
    try:
        print(">> 连接 {host}:{port} (unit={unit})".format(
            host=args.host, port=args.port, unit=args.unit))
        session.client.connect()
        block_desc = session.open_heartbeat_block()
        print(">> 心跳块: {}".format(block_desc))
        # 确认会话存活（连读 3 次都成功才认为"已连接"稳定）
        for _ in range(3):
            session.heartbeat_ok()
            time.sleep(args.poll_s)
        print(">> 会话已保持（示教器 Modbus 页应显示【已连接】）")

        if args.session_only:
            print(">> 仅会话模式：保持 {}s 轮询，不触发运行".format(args.watch))
            deadline = time.monotonic() + args.watch
            count = 0
            while time.monotonic() < deadline:
                session.heartbeat_ok()
                count += 1
                time.sleep(args.poll_s)
            print(">> 会话保持结束，共轮询 {} 次".format(count))
            return 0

        session.run_sequence(args.program, args.runs)
        moved = session.watch(args.watch)

        print("")
        print("== 结果 ==")
        print("机器人是否发生运动: {}".format("是" if moved else "否/未观测到"))
        print("（若未运动，请对照示教器：是否在远程模式、报警是否已清、")
        print("   机器人在复位点/拍摄点、Modbus 参数页是否【已连接】且未闪烁）")
        if args.hold:
            print(">> 会话保持中，Ctrl+C 退出")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
        if not args.no_stop:
            try:
                wire = plc_to_wire(ADDR_RUN_JOB, args.raw_addresses)
                session.client.write_single_register(wire, 0)
                print(">> 已写 19=0 停止")
            except Exception as error:
                print(">> 写停止失败: {}".format(error))
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
