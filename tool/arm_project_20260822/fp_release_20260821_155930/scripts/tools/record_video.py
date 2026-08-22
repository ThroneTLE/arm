#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""录制相机彩色视频，用于制作训练数据集。

用法（系统 python3，相机需在跑）:
  /usr/bin/python3 ~/record_video.py --out ~/fp_dataset/scene1.mp4 --duration 20

参数:
  --out      保存路径（默认 ~/fp_dataset/video.mp4）
  --duration 录制秒数（默认 20），到时间自动停止；Ctrl+C 也可提前停止
"""

import argparse
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class Recorder(Node):
    def __init__(self, out_path, duration):
        super().__init__("record_video")
        self.out_path = out_path
        self.duration = duration
        self.writer = None
        self.count = 0
        self.t0 = None
        self.sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.cb, 10
        )
        self.timer = self.create_timer(0.5, self.check)
        self.get_logger().info(f"录制目标: {out_path}，时长 {duration}s")
        self.get_logger().info("开始后请移动罐子/相机，覆盖不同角度和位置...")

    def cb(self, msg):
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        if msg.encoding == "rgb8":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.writer is None:
            h, w = rgb.shape[:2]
            out_dir = os.path.dirname(os.path.abspath(self.out_path))
            os.makedirs(out_dir, exist_ok=True)
            self.writer = cv2.VideoWriter(
                self.out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                15.0,
                (w, h),
            )
            self.t0 = time.time()
            self.get_logger().info("开始写入视频...")

        self.writer.write(rgb)
        self.count += 1

    def check(self):
        if self.writer is not None and time.time() - self.t0 >= self.duration:
            self.stop()

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.get_logger().info(
                f"保存完成: {self.out_path}（{self.count} 帧）"
            )
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/fp_dataset/video.mp4"))
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args.out, args.duration)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass  # stop() 可能已关闭，重复关闭会报错，忽略即可


if __name__ == "__main__":
    main()
