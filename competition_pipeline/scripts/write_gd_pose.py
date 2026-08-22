#!/usr/bin/env python3
"""视觉引导闭环的 PC 侧写点工具（用户坐标系1）。

已验证（2026-08-22 夜）：
- Modbus 4x 601 起 = DOUBLE 型全局数值变量 GD001..GD100（每变量 2 寄存器，Float CD AB），
  实写 123.456 -> 回读解码 123.456 ✓
- 控制器回读 0x2A02 {"robot":1,"coord":3} -> 0x2A03 含 configuration(形态) 与 pos（弧度）

作业文件侧（示教器粘贴）：
    POSSET P1001 UF 1 GD001   ; 用户坐标1 的 X
    POSSET P1001 UF 2 GD002   ; Y
    POSSET P1001 UF 3 GD003   ; Z
    POSSET P1001 UF 4 GD004   ; A(弧度)
    POSSET P1001 UF 5 GD005   ; B(弧度)
    POSSET P1001 UF 6 GD006   ; C(弧度)
    MOVL P1001 V=100 mm/s PL=0 ACC=10 DEC=10 0

用法：write_gd_pose.py --x -268.38 --y 56.18 --z 290.48 --a 3.1014 --b 0.3880 --c -3.1236
"""

import argparse
import struct
import sys

from competition_pipeline.controller_tcp import ModbusTcpClient, TcpEndpoint

# 4x 地址（手册值）：601 起 = GD001（每变量 2 寄存器，Float CD AB）
GD_DOUBLE_BASE = 601       # GD001
GD_REGISTERS_PER = 2       # 32-bit float
GD_COUNT = 100             # cSize 200


def float_to_cdab_regs(value):
    """Float32 -> 两个寄存器 [CD段, AB段]（CD AB 字节序）。"""
    raw = struct.pack('>f', float(value))
    cd = struct.unpack('>H', raw[2:4])[0]
    ab = struct.unpack('>H', raw[0:2])[0]
    return [cd, ab]


def write_gd(client, gd_number, value, based=False):
    """写 GDxxx（1-based 编号）; 地址基准 = 手册值-1。"""
    idx = int(gd_number) - 1
    address = GD_DOUBLE_BASE + idx * GD_REGISTERS_PER - 1  # wire
    regs = float_to_cdab_regs(value)
    client.write_multiple_registers(address, regs)
    return address, regs


def read_gd(client, gd_number):
    idx = int(gd_number) - 1
    address = GD_DOUBLE_BASE + idx * GD_REGISTERS_PER - 1
    raw = client.read_holding_registers(address, 2)
    return struct.unpack('>f', struct.pack('>HH', *[int(v) for v in raw[1:2] + [raw[1]]]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.1.200')
    parser.add_argument('--port', type=int, default=502)
    for name in ('x', 'y', 'z', 'a', 'b', 'c'):
        parser.add_argument('--' + name, type=float, required=True)
    parser.add_argument('--start-gd', type=int, default=1,
                        help='GD001 起的编号（默认 1）')
    args = parser.parse_args(argv)

    client = ModbusTcpClient(TcpEndpoint(args.host, args.port,
                                         connect_timeout_s=2.0, io_timeout_s=1.0),
                             unit_id=1)
    client.connect()
    try:
        values = [args.x, args.y, args.z, args.a, args.b, args.c]
        for i, value in enumerate(values):
            gd = args.start_gd + i
            address, regs = write_gd(client, gd, value)
            print('GD%03d = %.4f -> 4x.%d-%d %s (CD AB 字节序)' % (
                gd, value, address + 1, address + 2, regs))
        print('完成：POSSET 对应的 GD001..GD006 已写入（请按 POSSET 行序对应轴）')
    finally:
        client.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
