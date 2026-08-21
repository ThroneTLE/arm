#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证深度 640x400 与彩色 640x480 的正确对齐方式。

原理：YOLO 在彩色图上找到罐子框（y 范围），
分别用三种方式把深度 400 行映射到 480 行，
看哪种方式下"罐子的深度区域"的 y 范围与 YOLO 框最吻合。
"""

import os

import cv2
import numpy as np

CAPTURE = os.path.expanduser("~/fp_capture")
DEBUG = os.path.expanduser("~/fp_debug")
YOLO_WEIGHTS = os.path.expanduser("~/yolov8s-worldv2.pt")
YOLO_CLASSES = [
    "canned drink", "drink can", "soda can", "cola can", "coke can",
    "red can", "red tin can", "milk can", "beverage can", "tin can",
    "milk drink", "can of drink",
]


def main():
    rgb = cv2.imread(os.path.join(CAPTURE, "rgb.png"))
    depth = cv2.imread(os.path.join(CAPTURE, "depth.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        print("读取失败")
        return 1
    H, W = rgb.shape[:2]
    print(f"rgb {rgb.shape}, depth 原始 {depth.shape}")

    # YOLO 找罐子框
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)
    model.set_classes(YOLO_CLASSES)
    res = model.predict(rgb, conf=0.05, imgsz=1280, verbose=False)[0]
    boxes = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        for b in xyxy:
            x1, y1, x2, y2 = b.astype(int)
            if (x2 - x1) * (y2 - y1) > 500:
                boxes.append((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
    if not boxes:
        print("YOLO 没检测到罐子，无法验证")
        return 1
    x1, y1, x2, y2 = max(boxes, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
    print(f"YOLO 罐子框: ({x1},{y1})-({x2},{y2})")

    # 三种对齐方式
    aligns = {
        "top(顶部对齐)": np.zeros((H, W), np.uint16),
        "center(居中)": np.zeros((H, W), np.uint16),
        "stretch(拉伸)": np.zeros((H, W), np.uint16),
    }
    aligns["top(顶部对齐)"][:400, :] = depth
    aligns["center(居中)"][40:440, :] = depth
    aligns["stretch(拉伸)"] = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    print("-" * 60)
    for name, da in aligns.items():
        roi = da[y1:y2, x1:x2]
        valid = roi > 0
        n_valid = int(valid.sum())
        if n_valid < 100:
            print(f"{name}: 框内有效深度太少 ({n_valid})")
            continue
        vals = roi[valid]
        med = float(np.median(vals))
        th = float(np.percentile(vals, 50))  # 近处一半
        near = (roi > 0) & (roi <= th)
        ys = np.where(near)[0]  # 框内行号
        y_near_min = y1 + int(ys.min())
        y_near_max = y1 + int(ys.max())
        overlap = max(0, min(y2, y_near_max) - max(y1, y_near_min))
        ybox_h = y2 - y1
        print(f"{name}:")
        print(f"  框内深度中位数 {med:.0f}mm, 近处像素 y 范围 "
              f"[{y_near_min},{y_near_max}]")
        print(f"  YOLO 框 y 范围 [{y1},{y2}] (高 {ybox_h}px)")
        print(f"  近处范围与框 y 的重叠: {overlap}px "
              f"({100.0 * overlap / ybox_h:.0f}%)")
    print("-" * 60)
    print("重叠率最高的那个 = 正确的对齐方式")

    # 保存三种对齐的可视化
    os.makedirs(DEBUG, exist_ok=True)
    for i, (name, da) in enumerate(aligns.items()):
        vis = np.zeros((H, W, 3), np.uint8)
        d = da[da > 0]
        if d.size:
            norm = np.clip(da / 2500.0 * 255, 0, 255).astype(np.uint8)
            vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            vis[da == 0] = 0
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.imwrite(os.path.join(DEBUG, f"align_{i}.png"), vis)
        print("已保存:", os.path.join(DEBUG, f"align_{i}.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
