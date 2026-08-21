#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AprilTag 工作台坐标系：把 FoundationPose 的物体位姿转到 tag 坐标系下。

流程:
  1. 抓帧（capture_frame.py）+ 物体检测（detect_multi.py 生成 objects_poses.npy）
  2. 运行本脚本：检测桌面 tag，取最可靠的一张建立工作台坐标系
  3. 输出每个物体在工作台坐标系下的位姿

用法:
  conda activate foundationpose
  python ~/tag_workspace.py --tag-size 80

参数:
  --tag-size    tag 码边长，单位 mm（默认 80）
"""

import argparse
import os

import cv2
import numpy as np

CAPTURE_DIR = os.path.expanduser("~/fp_capture")
DEBUG_DIR = os.path.expanduser("~/fp_debug")

# 常见 AprilTag 家族，自动探测
APRILTAG_DICTS = [
    (cv2.aruco.DICT_APRILTAG_16h5, "APRILTAG_16h5"),
    (cv2.aruco.DICT_APRILTAG_25h9, "APRILTAG_25h9"),
    (cv2.aruco.DICT_APRILTAG_36h11, "APRILTAG_36h11"),
]


def detect_tags(rgb, tag_size_mm):
    """尝试多个字典检测 AprilTag，返回 [(id, T_cam_tag(4x4), 面积)]。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    s = tag_size_mm / 1000.0  # 转米，与 FoundationPose 位姿单位一致
    obj_pts = np.array([
        [-s / 2, -s / 2, 0], [s / 2, -s / 2, 0],
        [s / 2, s / 2, 0], [-s / 2, s / 2, 0],
    ], dtype=np.float32)
    K = np.loadtxt(os.path.join(CAPTURE_DIR, "cam_K.txt")).astype(np.float64)
    dist = np.zeros((4, 1))

    tags = []
    for dict_id, name in APRILTAG_DICTS:
        try:
            aruco_dict = cv2.aruco.Dictionary_get(dict_id)
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=params
            )
        except AttributeError:  # 新版 OpenCV API
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for i, tag_id in enumerate(ids.flatten()):
            c = corners[i][0]
            ok, rvec, tvec = cv2.solvePnP(obj_pts, c, K, dist)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            # 检测出的 z 轴可能指向桌面内部（角点顺序约定），统一翻转为朝上
            # 相机系 y 向下，朝上 => z 轴的 y 分量应为负
            if R[1, 2] > 0:
                R = R @ np.diag([1.0, -1.0, -1.0])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            area = cv2.contourArea(c)
            tags.append((int(tag_id), T, area))
        if tags:
            print(f"字典 {name}: 检测到 {len(tags)} 个 tag")
            break
    return tags


def build_world_frame(tags):
    """多 tag 融合工作台坐标系：
    用所有 tag 中心做平面拟合得到桌面法向（z 轴），原点取中心平均，
    x/y 取投影最大 tag 的轴投影到桌面平面。
    """
    Ts = [T for _, T, _ in tags]
    pts = np.array([T[:3, 3] for T in Ts])
    origin = pts.mean(axis=0)

    # 平面拟合：SVD 最小奇异值方向 = 平面法向
    _, _, vh = np.linalg.svd(pts - origin)
    z_axis = vh[-1]
    # 相机系 y 向下，桌面法向朝上 => y 分量应为负
    if z_axis[1] > 0:
        z_axis = -z_axis
    z_axis /= np.linalg.norm(z_axis)

    dists = np.abs((pts - origin) @ z_axis) * 1000.0
    print(f"tag 中心到拟合平面的距离 (mm): {np.round(dists, 1)}")

    base = max(Ts, key=lambda T: np.linalg.norm(T[:3, 3] - origin))
    x_axis = base[:3, 0]
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = origin
    return T, len(Ts)


def draw_tag(rgb, tags):
    """在图上画出 tag 边框、ID 和坐标轴。"""
    K = np.loadtxt(os.path.join(CAPTURE_DIR, "cam_K.txt")).astype(np.float64)
    for tag_id, T, _ in tags:
        origin = T[:3, 3]
        axis_len = 0.04  # 40mm
        axes = {
            "x": T[:3, :3] @ np.array([axis_len, 0, 0]),
            "y": T[:3, :3] @ np.array([0, axis_len, 0]),
            "z": T[:3, :3] @ np.array([0, 0, axis_len]),
        }
        def proj(p):
            p = p / p[2]
            return (int(K[0, 0] * p[0] + K[0, 2]),
                    int(K[1, 1] * p[1] + K[1, 2]))
        p0 = proj(origin)
        colors = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}
        for name, vec in axes.items():
            p1 = proj(origin + vec)
            cv2.arrowedLine(rgb, p0, p1, colors[name], 2)
        cv2.putText(rgb, f"tag{tag_id}", p0, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-size", type=float, default=80.0, help="tag 边长 mm")
    ap.add_argument("--base-tag", type=int, default=None,
                    help="指定工作台原点 tag ID（默认多点融合）")
    args = ap.parse_args()

    rgb = cv2.imread(os.path.join(CAPTURE_DIR, "rgb.png"))
    if rgb is None:
        print("请先运行 capture_frame.py 抓帧")
        return 1

    tags = detect_tags(rgb, args.tag_size)
    if not tags:
        print("没有检测到 AprilTag，请确认 tag 贴在画面内、尺寸参数正确")
        return 1

    tags.sort(key=lambda t: t[2], reverse=True)
    ids = [t[0] for t in tags]
    print(f"检测到 {len(tags)} 个 tag: ID {ids}")

    if args.base_tag is not None:
        base = [t for t in tags if t[0] == args.base_tag]
        if not base:
            print(f"画面里没有检测到 tag {args.base_tag}，可用: {[t[0] for t in tags]}")
            return 1
        T_cam_world = base[0][1]
        n_kept = 1
        print(f"工作台坐标系 = tag {args.base_tag} 的坐标系")
    else:
        T_cam_world, n_kept = build_world_frame(tags)
        print(f"工作台坐标系由 {n_kept}/{len(tags)} 个 tag 融合")
    print(f"相机→tag 位姿:\n{np.round(T_cam_world, 4)}")

    T_world_cam = np.linalg.inv(T_cam_world)

    pose_file = os.path.join(DEBUG_DIR, "objects_poses.npy")
    if not os.path.exists(pose_file):
        print(f"找不到 {pose_file}，请先运行 detect_multi.py")
        return 1
    cam_poses = np.load(pose_file)

    world_poses = []
    for i, M in enumerate(cam_poses):
        T_world_obj = T_world_cam @ M
        world_poses.append(T_world_obj)
        t = T_world_obj[:3, 3] * 1000.0
        print(f"物体 #{i} 工作台坐标 (mm): x={t[0]:.1f} y={t[1]:.1f} z={t[2]:.1f}")

    np.save(os.path.join(DEBUG_DIR, "objects_poses_world.npy"), np.array(world_poses))
    print("已保存: ~/fp_debug/objects_poses_world.npy")

    viz = draw_tag(rgb, tags)
    out = os.path.join(DEBUG_DIR, "viz_world.png")
    cv2.imwrite(out, viz)
    cv2.imwrite("/mnt/c/Users/Administrator/Desktop/viz_world.png", viz)
    print("可视化:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
