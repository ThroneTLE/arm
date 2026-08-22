#!/usr/bin/env python3
"""FoundationPose 输出 → 用户坐标系1 适配器（纯数学转换，不连控制器）。

链路（与 fp_closed_loop 相同，把"相机系物体位姿"转到"用户1系"）：

    T_user1_obj = T_user1_tcp · T_tcp_cam · T_cam_obj

- T_cam_obj  ：FoundationPose 输出（物体在相机系下的位姿；队友上传格式见下）
- T_tcp_cam  ：已标定手眼矩阵（competition.yaml hand_eye.tcp_from_color_camera，
               现场 15/16 内点 / 3.02mm；相机挂装工具手1 下方）
- T_user1_tcp：机器人 TCP 在用户坐标系1 的当前位姿（7040/0x2A02 coord=3 回读，
               XYZ mm + ABC 弧度）

**不触碰控制器**：默认从文件/参数取 T_user1_tcp；只有显式 --read-tcp 才会读控制器
（6001/7000 单客户端，其他 AI 联调时请勿同时使用）。

支持的输入格式（--cam-obj / --tcp 可用 json 或 yaml 文件，或直接 JSON 字符串）：
  1) ROS 风格:  {"translation":[x,y,z]m, "rotation_quaternion":[w,x,y,z]}
  2) 4x4 矩阵: {"matrix":[[...],[...],[...],[...]]}（默认米；--units-mm 改毫米）
  3) 本栈风格: {"xyz_mm":[...], "abc_rad":[...]} 或 {"xyz_mm":[...],"abc_deg":[...]}
  4) 纯文本 16 个浮点（行主序 4x4，米）

用法：
  # 队友的 FP 结果 + 机器人当前 TCP（从 0x2A02 回读存成文件）→ 转换
  python competition_pipeline/scripts/fp_to_user1.py \
      --cam-obj fp_result.json --tcp tcp_ucs.json

  # 转换并直接打印 用户1系 XYZ + ABC（用于 move_user1 --go 传送）
  python competition_pipeline/scripts/fp_to_user1.py \
      --cam-obj '{"translation":[0.02,-0.03,0.45],"rotation_quaternion":[0.9999,0,0,0.01]}' \
      --tcp '{"xyz_mm":[-115.85,44.98,201.87],"abc_rad":[3.1044,0.2402,-3.1415]}'

  # 直接读控制器（其他 AI 未占用时才用）
  python competition_pipeline/scripts/fp_to_user1.py --cam-obj fp_result.json --read-tcp

  # 离线自检（纯数学，不连任何设备）
  python competition_pipeline/scripts/fp_to_user1.py --self-test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# 姿态换算只有一份权威实现。这个脚本以前自带一份副本，而那份副本的逆解写错了
# （用了 Rz·Ry·Rx 的公式去逆 Rx·Ry·Rz），随机姿态往返最坏偏差 179.97°，
# 现场那条位姿偏 4.47°。--self-test 用的是单位旋转，所以一直测不出来。
# 教训：不要给同一个数学约定留第二份实现。
from competition_pipeline.geometry import (  # noqa: E402
    inexbot_abc_from_transform,
    transform_from_inexbot_abc,
)

# 手眼矩阵（competition.yaml hand_eye.tcp_from_color_camera；缓存值）
HAND_EYE_TCP_FROM_CAM = np.array([
    [0.000634, 0.9953, -0.096835, -0.102625],
    [-1.0, 0.000649, 0.000131, 0.004364],
    [0.000194, 0.096834, 0.9953, -0.161104],
    [0.0, 0.0, 0.0, 1.0],
])


def load_input(source):
    """文件路径或 JSON 字符串 -> dict。"""
    if source is None:
        return None
    text = None
    path = Path(source)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        text = source
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)


def quaternion_to_matrix(q):
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        raise ValueError("quaternion is zero")
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def parse_pose(data, units_mm=False, label="pose"):
    """dict -> (xyz_mm[3], R 3x3)。支持多种格式。"""
    if data is None:
        return None, None
    # 1) ROS 风格
    if "translation" in data and "rotation_quaternion" in data:
        xyz = np.asarray(data["translation"], dtype=float) * (
            1000.0 if not units_mm else 1.0)
        R = quaternion_to_matrix(data["rotation_quaternion"])
        return xyz, R
    # 2) 4x4 矩阵（row-major 嵌套或 {"matrix": [[..]]}）
    matrix = data.get("matrix") if isinstance(data, dict) and "matrix" in data else data
    if isinstance(matrix, (list, tuple)) and len(matrix) == 4:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (4, 4):
            raise ValueError("{}: matrix 必须是 4x4，得到 {}".format(label, arr.shape))
        xyz = arr[:3, 3].copy()
        if not units_mm:
            xyz = xyz * 1000.0
        return xyz, arr[:3, :3].copy()
    # 3) 本栈风格
    if "xyz_mm" in data or "xyz" in data or "abc_rad" in data or "abc_deg" in data:
        xyz = np.asarray(data.get("xyz_mm") or data.get("xyz"), dtype=float)
        if "abc_rad" in data:
            abc = np.asarray(data["abc_rad"], dtype=float)
        elif "abc_deg" in data:
            abc = np.radians(np.asarray(data["abc_deg"], dtype=float))
        else:
            raise ValueError("{}: 缺少 abc_rad/abc_deg".format(label))
        R = rotation_from_inexbot_abc(abc)
        return xyz, R
    raise ValueError("{}: 无法识别的位姿格式: {}".format(label, list(data.keys())))


def rotation_from_inexbot_abc(abc_rad):
    """控制器原生 A/B/C（内旋 X'Y'Z'）：R = Rx(A)Ry(B)Rz(C)。

    薄封装，实现来自 ``competition_pipeline.geometry``（唯一权威）。
    """
    return transform_from_inexbot_abc(
        np.zeros(3), np.asarray(abc_rad, dtype=float)
    )[:3, :3]


def inexbot_abc_from_rotation(R):
    """R -> (A, B, C) 弧度，是 :func:`rotation_from_inexbot_abc` 的**真正**逆。

    薄封装，实现来自 ``competition_pipeline.geometry``（唯一权威）。

    历史：这里曾有一份自己写的逆解 ``b = asin(-R[2,0]); a = atan2(R[2,1], R[2,2]);
    c = atan2(R[1,0], R[0,0])``，那是 Rz·Ry·Rx（外旋 ZYX/RPY）的公式，
    用来逆 Rx·Ry·Rz 是错的。随机姿态往返最坏偏差 179.97°；对现场那条位姿偏 4.47°。
    它的输出会经 --save-json 流到 move_user1 再发 0x4502 ——
    与 2026-08-22 摔臂完全同一条"XYZ 对、姿态被整体改写"的路径。
    """
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(R, dtype=float)
    return inexbot_abc_from_transform(transform)[1]


def convert(T_cam_obj, T_user1_tcp, T_tcp_cam=HAND_EYE_TCP_FROM_CAM):
    """返回 T_user1_obj (4x4)。"""
    result = T_user1_tcp @ T_tcp_cam @ T_cam_obj
    return result


def report(label, T, expecting_origin=False):
    xyz_mm = T[:3, 3] * 1000.0
    abc = inexbot_abc_from_rotation(T[:3, :3])
    print("{}  T =".format(label))
    for row in T:
        print("      [{: .6f}, {: .6f}, {: .6f}, {: .6f}]".format(*row))
    print("{}  XYZ = {:.2f}, {:.2f}, {:.2f} mm".format(label, *xyz_mm))
    print("{}  ABC = {:.4f}, {:.4f}, {:.4f} rad ({:.2f}°, {:.2f}°, {:.2f}°)".format(
        label, *abc, *np.degrees(abc)))
    if expecting_origin:
        ok = abs(xyz_mm[0]) < 30.0 and abs(xyz_mm[1]) < 30.0 and xyz_mm[2] > 0.0
        print("{}  原点核对：X={:.1f} Y={:.1f}（应≈0）Z={:.1f}(应≈柠檬高度) => {}".format(
            label, xyz_mm[0], xyz_mm[1], xyz_mm[2],
            "✅ 符合（柠檬在用户原点上方）" if ok else "⚠️ 偏离预期，检查 T_cam_obj/T_tcp_cam 方向约定"))
    return xyz_mm, abc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam-obj", help="T_cam_obj：相机系物体位姿（json/yaml 文件或字符串）")
    parser.add_argument("--tcp", help="T_user1_tcp：机器人 TCP 的用户1系位姿（文件或字符串）")
    parser.add_argument("--read-tcp", action="store_true",
                        help="从控制器实时回读 TCP（会占用单客户端端口，联调时勿用）")
    parser.add_argument("--units-mm", action="store_true",
                        help="矩阵/translation 单位为毫米（默认米）")
    parser.add_argument("--expect-origin", action="store_true",
                        help="核对输出是否≈用户1原点上方（柠檬测试）")
    parser.add_argument("--save-json", help="把 T_user1_obj 存成 {xyz_mm, abc_rad} json 供 move_user1 使用")
    parser.add_argument("--self-test", action="store_true", help="离线自检（不连设备）")
    args = parser.parse_args(argv)

    if args.self_test:
        # 合成：物体恰在用户1原点上方 60mm（TCP 在原点上方 500mm，相机下挂）
        tcp_xyz = np.array([0.0, 0.0, 500.0])
        tcp_R = np.eye(3)
        T_tcp = np.eye(4)
        T_tcp[:3, :3] = tcp_R
        T_tcp[:3, 3] = tcp_xyz / 1000.0
        # T_cam_obj = (T_tcp_cam)^-1 · T_tcp^-1 · T_user1_obj（用户原点上方60mm）
        T_obj_expected = np.eye(4)
        T_obj_expected[:3, 3] = np.array([0.0, 0.0, 60.0]) / 1000.0
        T_cam = np.linalg.inv(HAND_EYE_TCP_FROM_CAM) @ np.linalg.inv(T_tcp) @ T_obj_expected
        out = convert(T_cam, T_tcp)
        xyz = out[:3, 3] * 1000.0
        err = float(np.abs(xyz - np.array([0.0, 0.0, 60.0])).max())
        rot_err = float(np.abs(out[:3, :3] - np.eye(3)).max())
        print("自检：期望 (0, 0, 60)mm；得到 ({:.3f}, {:.3f}, {:.3f})mm；"
              "平移最大误差 {:.6f}mm，旋转最大误差 {:.6f}".format(*xyz, err, rot_err))
        # 姿态往返自检：以前这里只有单位旋转，所以逆解写错了 179.97° 也测不出来。
        # 现在覆盖现场真实位姿 + 一批随机姿态。
        probes = [
            np.array([3.104412961019, 0.240200009301, -3.141491526695]),  # 现场拍摄点
            np.array([3.045, -0.078, -3.044]),                            # 用户1原点姿态
            np.array([0.0, 0.0, 0.0]),
            np.array([0.3, 0.4, 0.5]),
            np.array([-2.9, 1.2, 2.7]),
            np.array([1.0, -1.5, -2.0]),
        ]
        worst_abc = 0.0
        for abc in probes:
            R = rotation_from_inexbot_abc(abc)
            R_back = rotation_from_inexbot_abc(inexbot_abc_from_rotation(R))
            cosine = np.clip((np.trace(R.T @ R_back) - 1.0) * 0.5, -1.0, 1.0)
            worst_abc = max(worst_abc, float(np.degrees(np.arccos(cosine))))
        print("姿态往返自检（{} 个姿态，含现场实测两点）：最坏误差 {:.9f}°".format(
            len(probes), worst_abc))
        if err < 1e-6 and rot_err < 1e-9 and worst_abc < 1e-6:
            print("✅ 转换链路自检通过（T_user1_obj = T_user1_tcp · T_tcp_cam · T_cam_obj）")
            print("✅ A/B/C 正逆解互为逆运算")
            return 0
        print("❌ 自检失败")
        return 1

    cam_data = load_input(args.cam_obj)
    if cam_data is None:
        print("请提供 --cam-obj（FoundationPose 输出）或 --self-test")
        return 2
    xyz_c_mm, R_c = parse_pose(cam_data, units_mm=args.units_mm, label="cam_obj")
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = R_c
    T_cam_obj[:3, 3] = xyz_c_mm / 1000.0

    if args.read_tcp:
        # ⚠️ 占用 6001/7000 单客户端端口
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from competition_pipeline.scripts.move_user1 import User1Mover, make_endpoint
        mover = User1Mover(make_endpoint())
        try:
            xyz_t_mm, abc_t_rad = mover.current_pose()
        finally:
            mover.close()
        R_t = rotation_from_inexbot_abc(np.asarray(abc_t_rad, dtype=float))
    else:
        tcp_data = load_input(args.tcp)
        if tcp_data is None:
            print("请提供 --tcp（机器人 TCP 用户1系位姿）或 --read-tcp")
            return 2
        xyz_t_mm, R_t = parse_pose(tcp_data, units_mm=args.units_mm, label="tcp")

    T_tcp = np.eye(4)
    T_tcp[:3, :3] = R_t
    T_tcp[:3, 3] = np.asarray(xyz_t_mm, dtype=float) / 1000.0

    T_out = convert(T_cam_obj, T_tcp)
    report("用户系1物体", T_out, expecting_origin=args.expect_origin)
    xyz_out_mm = T_out[:3, 3] * 1000.0
    abc_out_rad = inexbot_abc_from_rotation(T_out[:3, :3])
    if args.save_json:
        out = {"xyz_mm": [round(float(v), 3) for v in xyz_out_mm],
               "abc_rad": [round(float(v), 6) for v in abc_out_rad]}
        Path(args.save_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("已保存 {}".format(args.save_json))
        # move_user1 的 --json-pose 只认 x/y/z/a/b/c 六个键（parse_pose），
        # 直接把上面的 {xyz_mm, abc_rad} 贴过去会 KeyError。这里输出它真正吃的格式。
        move_pose = {
            "x": round(float(xyz_out_mm[0]), 3),
            "y": round(float(xyz_out_mm[1]), 3),
            "z": round(float(xyz_out_mm[2]), 3),
            "a": round(float(abc_out_rad[0]), 6),
            "b": round(float(abc_out_rad[1]), 6),
            "c": round(float(abc_out_rad[2]), 6),
        }
        print("传送到机器人（a/b/c 为弧度）：")
        print("  python -m competition_pipeline.scripts.move_user1 "
              "--json-pose '{}' --go".format(json.dumps(move_pose)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
