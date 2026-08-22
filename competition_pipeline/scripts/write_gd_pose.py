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

单位（写错会摔机械臂，看清楚）
------------------------------
``--x/--y/--z`` 是**毫米**；``--a/--b/--c`` 是**弧度**，因为 POSSET 的 UF 4/5/6
就是按弧度解释的。控制器**回读**给出的 A/B/C 在示教器上显示为**度**，直接抄
到这里就是把度当弧度 —— 2026-08-22 就是这个错误让腕部在 30mm 直线上摆 119.6°，
六轴同时 0F15 故障、控制器下电、机械臂坠落。

本脚本用 ``--allow-large-angle`` 之外的一切输入都会被 :func:`check_abc_is_radians`
拦下，并在写入前打印度数等价值供人工核对。
"""

import argparse
import struct
import sys

import numpy as np

from competition_pipeline.controller_tcp import ModbusTcpClient, TcpEndpoint
from competition_pipeline.nexbot_jog import MAX_ABC_RAD, check_abc_is_radians

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


def cdab_regs_to_float(regs):
    """两个寄存器 [CD段, AB段] -> Float32（:func:`float_to_cdab_regs` 的逆）。"""
    cd, ab = (int(regs[0]) & 0xFFFF, int(regs[1]) & 0xFFFF)
    return struct.unpack('>f', struct.pack('>HH', ab, cd))[0]


def read_gd(client, gd_number):
    """回读 GDxxx。返回 float。

    旧实现是 ``struct.pack('>HH', *[int(v) for v in raw[1:2] + [raw[1]]])`` ——
    把同一个寄存器取了两遍、还漏掉了 raw[0]，解出来的数与写进去的无关。
    没人调用它，所以一直没暴露；现在写入后要靠它做回读校验。
    """
    idx = int(gd_number) - 1
    address = GD_DOUBLE_BASE + idx * GD_REGISTERS_PER - 1
    raw = client.read_holding_registers(address, 2)
    return cdab_regs_to_float(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.1.200')
    parser.add_argument('--port', type=int, default=502)
    for name in ('x', 'y', 'z'):
        parser.add_argument('--' + name, type=float, required=True,
                            help='用户坐标系1 的 %s，单位 **毫米**' % name.upper())
    for name in ('a', 'b', 'c'):
        parser.add_argument('--' + name, type=float, required=True,
                            help='用户坐标系1 的 %s，单位 **弧度**（不是度！）'
                                 % name.upper())
    parser.add_argument('--start-gd', type=int, default=1,
                        help='GD001 起的编号（默认 1）')
    parser.add_argument('--allow-large-angle', action='store_true',
                        help='放行 |A/B/C| > %.1f rad 的输入。只有在你确认这'
                             '真的是弧度、而不是把度数抄进来时才用。' % MAX_ABC_RAD)
    parser.add_argument('--no-verify', action='store_true',
                        help='跳过写入后的回读校验（默认回读并比对）')
    args = parser.parse_args(argv)

    abc = [args.a, args.b, args.c]
    if not args.allow_large_angle:
        # 度当弧度是这条通道上最容易犯、后果最重的错误。写进 GD 变量之后
        # 由示教器作业里的 MOVL 执行，PC 侧再没有第二道闸门。
        check_abc_is_radians(abc, where='write_gd_pose')
    print('将写入 用户坐标系1：')
    print('  XYZ = %.3f, %.3f, %.3f mm' % (args.x, args.y, args.z))
    print('  ABC = %.6f, %.6f, %.6f rad  (= %.3f°, %.3f°, %.3f°)'
          % (args.a, args.b, args.c, *np.degrees(abc)))
    print('  ↑ 请核对度数一列是否与示教器显示一致；不一致说明单位搞反了。')

    client = ModbusTcpClient(TcpEndpoint(args.host, args.port,
                                         connect_timeout_s=2.0, io_timeout_s=1.0),
                             unit_id=1)
    client.connect()
    try:
        values = [args.x, args.y, args.z, args.a, args.b, args.c]
        mismatched = []
        for i, value in enumerate(values):
            gd = args.start_gd + i
            address, regs = write_gd(client, gd, value)
            note = ''
            if not args.no_verify:
                # Modbus 写入没有应答内容可言，不回读就等于"发了就当成了"。
                readback = read_gd(client, gd)
                # float32 有效位约 7 位，按相对误差比对。
                if abs(readback - float(value)) > max(1e-3, abs(value) * 1e-6):
                    mismatched.append((gd, value, readback))
                    note = '  ❌ 回读 %.6f' % readback
                else:
                    note = '  ✓ 回读 %.6f' % readback
            print('GD%03d = %.6f -> 4x.%d-%d %s (CD AB 字节序)%s' % (
                gd, value, address + 1, address + 2, regs, note))
        if mismatched:
            for gd, wrote, got in mismatched:
                print('❌ GD%03d 写入 %.6f 但回读 %.6f' % (gd, wrote, got),
                      file=sys.stderr)
            print('写入未通过回读校验，**不要**在示教器上运行该作业。',
                  file=sys.stderr)
            return 1
        print('完成：POSSET 对应的 GD001..GD006 已写入并回读一致'
              '（请按 POSSET 行序对应轴）')
    finally:
        client.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
