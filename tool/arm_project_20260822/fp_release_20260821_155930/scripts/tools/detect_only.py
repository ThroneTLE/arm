#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用自定义 YOLOv8 模型识别图片中的物体（仅检测，不做姿态估计）。

用法（foundationpose 环境）:
  python ~/detect_only.py --model ~/yolo_model.pt --image ~/fp_capture/rgb.png

参数:
  --model  模型文件路径（默认 ~/yolo_model.pt）
  --image  要识别的图片（默认上次抓帧的 rgb.png）
  --conf   置信度阈值（默认 0.25）
  --out    结果图保存路径（默认桌面 detect_result.png）
"""

import argparse
import os

import cv2

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/yolo_model.pt"))
    ap.add_argument("--image", default=os.path.expanduser("~/fp_capture/rgb.png"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="/mnt/c/Users/Administrator/Desktop/detect_result.png")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"模型不存在: {args.model}")
        return 1
    if not os.path.exists(args.image):
        print(f"图片不存在: {args.image}")
        return 1

    model = YOLO(args.model)
    img = cv2.imread(args.image)
    print(f"模型: {args.model}")
    print(f"图片: {args.image} ({img.shape[1]}x{img.shape[0]})")

    res = model.predict(img, conf=args.conf, verbose=False)[0]
    boxes = res.boxes
    viz = img.copy()

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        print(f"检测到 {len(xyxy)} 个目标:")
        for b, c, s in zip(xyxy, cls, confs):
            x1, y1, x2, y2 = b.astype(int)
            name = model.names[int(c)]
            print(f"  [{name}] conf={s:.2f} bbox=({x1},{y1})-({x2},{y2})")
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(viz, f"{name} {s:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print(f"没有检测到目标（conf阈值 {args.conf}）")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, viz)
    print("结果图:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
