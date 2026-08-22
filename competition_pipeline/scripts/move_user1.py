#!/usr/bin/env python3
"""用户坐标系1 定点传送 + 到达验证（全部通道均现场实测）。

链路（MOKA MR07S-930 / C1102 21.05.23，2026-08-22 实测）：
  0x2311 使能(伺服上电, status=3) -> 0x4501/0x4502 运动
  {"robot","vel","acc","dec","coord":3,"pos":[X,Y,Z,A,B,C_弧度,0.0]}
  -> 位姿变化判停 -> 0x2A02 回读核对 |目标-实际| < 容差。

用法：
  # 查看计划（不发运动）
  python competition_pipeline/scripts/move_user1.py \
      --x -233.58 --y 63.43 --z 289.98 --a 3.11111 --b 0.388225 --c 3.13462

  # 实际传送并验证（角度默认弧度；--deg 用度）
  python competition_pipeline/scripts/move_user1.py \
      --x -233.58 --y 63.43 --z 289.98 --a 178.25 --b 22.24 --c 179.60 --deg --go

  # 从 pipeline 来的 JSON 目标（视觉抓取位姿直接投喂）
  python competition_pipeline/scripts/move_user1.py --json-pose '{
      "x":-233.58,"y":63.43,"z":289.98,"a":3.11111,"b":0.388225,"c":3.13462}' --go

  # 急停（任何情况下可用，单独执行）
  python competition_pipeline/scripts/move_user1.py --estop
"""

import argparse
import json
import sys
import time

import numpy as np

from competition_pipeline.geometry import inexbot_abc_from_transform
from competition_pipeline.nexbot_tcp import (
    build_frame,  # noqa: F401  (仅提示协议模块已就绪)
)
from competition_pipeline.controller_tcp import (
    ControllerConnectionError,
    InexbotPoint,
)
from competition_pipeline.tcp_pose import pose_endpoint_from_config
from competition_pipeline.nexbot_tcp import (
    NexBotTcpEndpoint,
    NexBotTcpRobotController,
    CMD_ENABLE,
    CMD_SERVO_INQUIRE,
    CMD_SERVO_RESPOND,
)


def make_endpoint():
    import yaml
    config = yaml.safe_load(
        open("competition_pipeline/config/competition.yaml", encoding="utf-8")
    )
    return pose_endpoint_from_config(config["controller"])


class User1Mover:
    """单次连接的"用户系1 定点传送 + 到达验证"。"""

    def __init__(self, endpoint: NexBotTcpEndpoint, verbose=True):
        self.endpoint = endpoint
        self._controller = None
        self.verbose = bool(verbose)

    @property
    def controller(self):
        if self._controller is None:
            self._controller = NexBotTcpRobotController(self.endpoint)
        return self._controller

    def connect(self):
        return self.controller

    def close(self):
        if self._controller is not None:
            self._controller.close()
            self._controller = None

    def log(self, message):
        if self.verbose:
            print(message)

    # -- 状态 ---------------------------------------------------------------
    def servo_status(self):
        self.controller.motion.send_frame(
            CMD_SERVO_INQUIRE, {"robot": self.endpoint.robot}
        )
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            command, data = self.controller.motion.read_frame()
            if command == CMD_SERVO_RESPOND:
                return int(data.get("status", 0))
        raise RuntimeError("伺服状态查询超时")

    def enable(self):
        """0x2311 上位机使能；返回状态（3=运行）。"""
        self.controller.motion.send_frame(CMD_ENABLE, {"robot": self.endpoint.robot})
        time.sleep(1.0)
        status = self.servo_status()
        if status != 3:
            raise RuntimeError(
                "上使能失败（status={}）：请确认示教器在示教/远程模式且无报警".format(status)
            )
        self.log("  伺服已上电 (status=3 运行)")
        return status

    def current_pose(self):
        """(xyz_mm, abc_rad) 当前用户系1 TCP 位姿。"""
        state = self.controller.read_state()
        xyz_m, abc_rad = inexbot_abc_from_transform(state.base_from_gripper)
        return tuple(xyz_m * 1000.0), tuple(abc_rad)

    # -- 运动 ---------------------------------------------------------------
    def move_to(self, xyz_mm, abc_rad, movl=True, vel=50.0, tolerance_mm=1.0,
                readback_timeout_s=15.0):
        """传送并验证。返回 (出发位姿, 到达位姿, 偏差mm)。"""

        def _read_pose():
            return self.current_pose()

        start_xyz, start_abc = _read_pose()
        start_xyz_arr = np.asarray(start_xyz, dtype=np.float64)
        target_arr = np.asarray(xyz_mm, dtype=np.float64)
        distance = float(np.linalg.norm(target_arr - start_xyz_arr))
        self.log(
            "  出发位姿 X={:.2f} Y={:.2f} Z={:.2f} mm｜目标 X={:.2f} Y={:.2f} Z={:.2f} mm｜直线距离 {:.2f} mm".format(
                *start_xyz[:3], *xyz_mm[:3], distance
            )
        )

        if self.servo_status() != 3:
            self.enable()
        else:
            self.log("  伺服已在运行状态")

        point = InexbotPoint(
            name="P0001",
            coordinate_system=3,      # 用户坐标
            angle_unit=1,             # 弧度
            shape=1,
            tool_id=int(self.endpoint.tool_id or 1),
            user_id=int(self.endpoint.user_id or 1),
            axes=[*xyz_mm, *abc_rad, 0.0],
        )
        target_4x4 = point_axes_matrix(xyz_mm, abc_rad)  # 仅用于显示/校验

        motion = self.controller.move_l if movl else self.controller.move_j
        motion([point], speed_mm_s=vel)

        # 位姿变化判停（等待实际到达）
        deadline = time.monotonic() + readback_timeout_s
        last_xyz = None
        settled = False
        while time.monotonic() < deadline:
            xyz, _abc = self.current_pose()
            arr = np.asarray(xyz, dtype=np.float64)
            if last_xyz is not None and float(np.max(np.abs(arr - last_xyz))) < 0.3:
                settled = True
                break
            last_xyz = arr
            time.sleep(0.3)
        if not settled:
            raise RuntimeError("运动完成检测超时（{}s）".format(int(readback_timeout_s)))

        final_xyz, final_abc = self.current_pose()
        final_arr = np.asarray(final_xyz, dtype=np.float64)
        deviation = float(np.linalg.norm(final_arr - target_arr))
        self.log(
            "  到达位姿 X={:.2f} Y={:.2f} Z={:.2f} mm｜偏差 {:.3f} mm".format(
                *final_xyz[:3], deviation
            )
        )
        if deviation > tolerance_mm:
            raise RuntimeError(
                "⚠️ 到达偏差 {:.2f} mm 超出容差 {:.1f} mm —— 未到位/被干涉".format(
                    deviation, tolerance_mm
                )
            )
        self.log("  ✅ 到达验证通过（偏差 {:.3f} mm ≤ {:.1f} mm）".format(deviation, tolerance_mm))
        return (start_xyz, start_abc), (final_xyz, final_abc), deviation

    def emergency_stop(self):
        self.log("  >> 发送急停 0x2314")
        self.controller.motion.send_frame(0x2314, {"robot": self.endpoint.robot})


def point_axes_matrix(xyz_mm, abc_rad):
    from competition_pipeline.geometry import transform_from_inexbot_abc
    return transform_from_inexbot_abc(
        np.asarray(xyz_mm) / 1000.0, np.asarray(abc_rad)
    )


def parse_pose(args):
    if args.json_pose:
        data = json.loads(args.json_pose)
        x, y, z = float(data["x"]), float(data["y"]), float(data["z"])
        a, b, c = float(data["a"]), float(data["b"]), float(data["c"])
    else:
        x, y, z = args.x, args.y, args.z
        a, b, c = args.a, args.b, args.c
    if args.deg:
        a, b, c = np.radians([a, b, c])
    return [x, y, z], [a, b, c]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--z", type=float)
    parser.add_argument("--a", type=float, help="弧度，或 --deg 时用度")
    parser.add_argument("--b", type=float)
    parser.add_argument("--c", type=float)
    parser.add_argument("--deg", action="store_true", help="ABC 用角度制")
    parser.add_argument("--json-pose", help='{"x":..,"y":..,"z":..,"a":..,"b":..,"c":..}')
    parser.add_argument("--go", action="store_true", help="实际执行运动（默认只打印计划）")
    parser.add_argument("--movl", action="store_true", default=True,
                        help="MOVL 直线（默认）；--movej 采用 MOVJ 关节插补")
    parser.add_argument("--movej", action="store_true")
    parser.add_argument("--vel", type=float, default=50.0, help="MOVL mm/s（MOVJ 用 %）")
    parser.add_argument("--tolerance", type=float, default=1.0, help="到达容差 mm")
    parser.add_argument("--estop", action="store_true", help="立即急停并退出")
    parser.add_argument("--readback", action="store_true",
                        help="只回读当前位姿（不动）")
    args = parser.parse_args(argv)

    mover = User1Mover(make_endpoint())
    try:
        if args.estop:
            mover.emergency_stop()
            return 0
        xyz, abc_rad = mover.current_pose()
        print("当前用户1系 TCP：X={:.2f} Y={:.2f} Z={:.2f} mm "
              "A={:.4f} B={:.4f} C={:.4f} rad（{:.2f}° {:.2f}° {:.2f}°）".format(
                  *xyz[:3], *abc_rad, *np.degrees(abc_rad)))
        if args.readback:
            return 0
        if not args.go:
            if not (args.json_pose or (args.x is not None and args.y is not None and args.z is not None)):
                print("未提供目标坐标（--x/--y/--z/--a/--b/--c 或 --json-pose）；"
                      "当前仅回读。")
                return 1
            tx, ta = parse_pose(args)
            print("计划：MOVL 至 X={:.2f} Y={:.2f} Z={:.2f} mm "
                  "ABC={:.4f},{:.4f},{:.4f} rad（加 --go 执行）".format(
                      *tx, *ta))
            return 0
        tx, ta = parse_pose(args)
        mover.move_to(tx, ta, movl=not args.movej, vel=args.vel,
                      tolerance_mm=args.tolerance)
        print("✅ 传送完成并验证到位")
        return 0
    except ControllerConnectionError as error:
        print("连接失败：{}（6001 单客户端，请确认无其他上位机占用）".format(error))
        return 2
    except Exception as error:
        print("失败：{}".format(error))
        return 1
    finally:
        mover.close()


if __name__ == "__main__":
    sys.exit(main())
