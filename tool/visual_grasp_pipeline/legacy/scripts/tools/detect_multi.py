#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单帧多目标检测：抓帧数据 + YOLO 多框 + 逐目标 FoundationPose 注册。

考核流程：
  1. 相机启动，机械臂固定姿态，罐子摆好
  2. /usr/bin/python3 ~/capture_frame.py ~/fp_capture   （抓一帧）
  3. conda activate foundationpose && python ~/detect_multi.py
  4. 输出所有罐子位姿，机械臂按位姿执行

输出:
  ~/fp_debug/objects_poses.npy   (N,4,4) 每个罐子的位姿矩阵
  ~/fp_debug/objects_boxes.npy   (N,4)   每个罐子的 2D 框
  ~/fp_debug/viz_multi.png       可视化（桌面也有一份）
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import trimesh

FP_DIR = os.path.expanduser("~/FoundationPose")
CAPTURE_DIR = os.path.expanduser("~/fp_capture")
MESH_FILE = os.path.join(FP_DIR, "demo_data/can/mesh/can.obj")
DEBUG_DIR = os.path.expanduser("~/fp_debug")
VIZ_FILE = "/mnt/c/Users/Administrator/Desktop/viz_multi.png"
YOLO_WEIGHTS = os.path.expanduser("~/yolov8s-worldv2.pt")
YOLO_CLASSES = [
    "canned drink", "drink can", "soda can", "cola can", "coke can",
    "red can", "red tin can", "milk can", "beverage can", "tin can",
    "milk drink", "can of drink",
]
YOLO_CONF = 0.05
YOLO_IMGSZ = 1280

sys.path.insert(0, FP_DIR)
os.chdir(FP_DIR)

from estimater import *  # noqa: E402
from Utils import (  # noqa: E402
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
)


def detect_boxes(rgb, model):
    """YOLO 检测所有罐子框，过滤异常框并做简单去重。"""
    H, W = rgb.shape[:2]
    res = model.predict(rgb, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)[0]
    raw = []
    img_area = W * H
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, confs):
            x1, y1, x2, y2 = b.astype(int)
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if area < 500 or area > img_area * 0.5:
                continue
            if h > 0 and not (0.3 < w / h < 2.5):
                continue
            raw.append((max(0, x1), max(0, y1), min(W, x2), min(H, y2), float(c)))

    # 按面积从大到小，剔除与更大框重叠超过一半的框
    raw.sort(key=lambda t: (t[3] - t[1]) * (t[2] - t[0]), reverse=True)
    keep = []
    for b in raw:
        x1, y1, x2, y2, _ = b
        area = (x2 - x1) * (y2 - y1)
        dup = False
        for k in keep:
            ix1, iy1, ix2, iy2, _ = k
            xx1, yy1 = max(x1, ix1), max(y1, iy1)
            xx2, yy2 = min(x2, ix2), min(y2, iy2)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            if area > 0 and inter / area > 0.5:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


def draw_one(rgb, mesh, R, t, K, bbox, label, color=(0, 255, 0)):
    """画单个目标的轮廓 + 包围盒 + 坐标轴 + 编号。"""
    H, W = rgb.shape[:2]
    P = mesh.vertices @ R.T + t
    z = P[:, 2]
    pts = np.column_stack([
        P[:, 0] * K[0, 0] / z + K[0, 2],
        P[:, 1] * K[1, 1] / z + K[1, 2],
    ])
    mm = np.zeros((H, W), np.uint8)
    for f in mesh.faces:
        if np.any(z[f] < 0.02):
            continue
        cv2.fillConvexPoly(mm, pts[f].astype(np.int32), 255)
    contours, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, color, 2)

    ob_in_cam = np.eye(4)
    ob_in_cam[:3, :3] = R
    ob_in_cam[:3, 3] = t
    rgb = draw_xyz_axis(rgb, ob_in_cam, scale=0.05, K=K, thickness=3, is_input_rgb=False)
    rgb = draw_posed_3d_box(K, rgb, ob_in_cam, bbox, line_color=(0, 255, 255), linewidth=2)
    cv2.putText(rgb, f"#{label}", (10, 30 + label * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=5, help="每个目标注册迭代次数，默认5")
    ap.add_argument("--no-yolo", action="store_true", help="不用YOLO，全图当单个目标")
    args = ap.parse_args()

    os.makedirs(DEBUG_DIR, exist_ok=True)
    set_logging_format()
    set_seed(0)

    rgb = cv2.imread(os.path.join(CAPTURE_DIR, "rgb.png"))
    depth = cv2.imread(os.path.join(CAPTURE_DIR, "depth.png"), cv2.IMREAD_UNCHANGED)
    K = np.loadtxt(os.path.join(CAPTURE_DIR, "cam_K.txt")).astype(np.float64)
    if rgb is None or depth is None:
        print(f"读取失败，先运行 capture_frame.py 抓帧到 {CAPTURE_DIR}")
        return 1
    H, W = rgb.shape[:2]
    if depth.shape[:2] != (H, W):
        aligned = np.zeros((H, W), dtype=np.uint16)
        aligned[: depth.shape[0], :] = depth  # 顶部对齐（已验证正确）
        depth = aligned
    depth_m = depth.astype(np.float32) / 1000.0

    mesh = trimesh.load(MESH_FILE)
    mesh.apply_scale(0.001)
    _, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    print(f"mesh: 顶点 {len(mesh.vertices)}, 面 {len(mesh.faces)}")

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=DEBUG_DIR,
        debug=1,
        glctx=glctx,
    )

    if args.no_yolo:
        boxes = [(0, 0, W - 1, H - 1, 1.0)]
    else:
        from ultralytics import YOLO
        model = YOLO(YOLO_WEIGHTS)
        model.set_classes(YOLO_CLASSES)
        boxes = detect_boxes(rgb, model)
        print(f"YOLO 检测到 {len(boxes)} 个目标")

    poses = []
    viz = rgb.copy()
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        mask = np.zeros((H, W), np.uint8)
        mask[y1:y2, x1:x2] = 255
        print(f"\n--- 目标 #{i}: 框 ({x1},{y1})-({x2},{y2}), conf={conf:.2f} ---")
        t0 = time.time()
        res = est.register(K, rgb, depth_m, mask, iteration=args.iter)
        if isinstance(res, tuple) and len(res) >= 2:
            R, t = res[0], res[1]
        else:
            R = res[:3, :3]
            t = res[:3, 3]
        print(f"注册耗时 {time.time() - t0:.1f}s")
        if not (0.3 < t[2] < 1.5):
            print(f"[警告] 目标 #{i} 距离异常 z={t[2]*1000:.0f}mm，跳过")
            continue
        poses.append((R, t, (x1, y1, x2, y2)))
        print(f"  pos (mm): x={t[0]*1000:.1f} y={t[1]*1000:.1f} z={t[2]*1000:.1f}")
        viz = draw_one(viz, mesh, R, t, K, bbox, i)

    print("\n" + "=" * 50)
    print(f"有效位姿: {len(poses)} 个")
    mats = []
    boxes_np = []
    for R, t, box in poses:
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        mats.append(M)
        boxes_np.append(box)
    if mats:
        np.save(os.path.join(DEBUG_DIR, "objects_poses.npy"), np.array(mats))
        np.save(os.path.join(DEBUG_DIR, "objects_boxes.npy"), np.array(boxes_np))
        print("已保存 ~/fp_debug/objects_poses.npy (N,4,4) 和 objects_boxes.npy")
    cv2.imwrite(os.path.join(DEBUG_DIR, "viz_multi.png"), viz)
    cv2.imwrite(VIZ_FILE, viz)
    print("可视化:", VIZ_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
