#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时用自定义 YOLOv8 模型检测物体（仅检测，不做姿态估计）。

配合 fp_bridge.py 使用：
  终端1(系统python): /usr/bin/python3 ~/fp_bridge.py
  终端2(conda)     : conda activate foundationpose && python ~/yolo_live.py

输出：桌面 live_yolo.png 实时刷新；终端打印检测结果。
"""

import argparse
import os
import time

import cv2
import numpy as np
import zmq

from ultralytics import YOLO

ZMQ_ADDR = "tcp://127.0.0.1:5555"
VIZ_FILE = "/mnt/c/Users/Administrator/Desktop/live_yolo.png"


def recv_latest(sub):
    parts = sub.recv_multipart()
    while True:
        try:
            parts = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/yolo_model.pt"))
    ap.add_argument("--conf", type=float, default=0.85)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"模型不存在: {args.model}")
        return 1
    model = YOLO(args.model)
    print(f"已加载模型: {args.model}")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_ADDR)
    sub.setsockopt_string(zmq.SUBSCRIBE, "FRAME")
    print("等待 fp_bridge 画面...")

    frame_id = 0
    last_print = 0.0
    while True:
        try:
            parts = recv_latest(sub)
        except KeyboardInterrupt:
            break
        frame_id += 1
        jpg = parts[1]
        rgb = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

        res = model.predict(rgb, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        viz = rgb.copy()
        boxes = res.boxes
        n = 0
        # 分割掩膜：半透明填充 + 轮廓
        if res.masks is not None and len(res.masks) > 0:
            mask_data = res.masks.data.cpu().numpy()
            for m in mask_data:
                m_bin = (m > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                overlay = viz.copy()
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
                viz = cv2.addWeighted(overlay, 0.35, viz, 0.65, 0)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            n = len(xyxy)
            for b, c, s in zip(xyxy, cls, confs):
                x1, y1, x2, y2 = b.astype(int)
                name = model.names[int(c)]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz, f"{name} {s:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        now = time.time()
        if now - last_print >= 2.0:
            print(f"[检测] {n} 个目标")
            last_print = now

        if frame_id % 3 == 0:
            cv2.imwrite(VIZ_FILE, viz)
        try:
            cv2.imshow("yolo live (q: quit)", viz)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception:
            pass

    print("退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
