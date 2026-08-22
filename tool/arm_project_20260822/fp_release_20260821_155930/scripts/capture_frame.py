#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取一帧 彩色图 + 深度图 + 彩色/深度相机内参，保存为文件。

用法（必须用系统 python3，不能用 conda 环境）:
  /usr/bin/python3 ~/capture_frame.py ~/fp_capture

输出:
  ~/fp_capture/rgb.png      彩色图 (BGR)
  ~/fp_capture/depth.png    深度图 (16bit, 单位 mm, 0=无效)
  ~/fp_capture/cam_K.txt       彩色相机内参 3x3
  ~/fp_capture/cam_K_depth.txt 深度相机内参 3x3（做深度对齐用）
"""

import os
import sys

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class CaptureNode(Node):
    def __init__(self, outdir):
        super().__init__("capture_frame")
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.saved = False

        sub_rgb = Subscriber(self, Image, "/camera/color/image_raw")
        sub_depth = Subscriber(self, Image, "/camera/depth/image_raw")
        sub_info = Subscriber(self, CameraInfo, "/camera/color/camera_info")
        sub_depth_info = Subscriber(self, CameraInfo, "/camera/depth/camera_info")
        self.sync = ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_info, sub_depth_info], queue_size=20, slop=0.2
        )
        self.sync.registerCallback(self.cb)

        self.timer = self.create_timer(0.5, self.check)
        self.get_logger().info("等待相机话题数据，罐子请放在画面中央...")

    def cb(self, rgb_msg, depth_msg, info_msg, depth_info_msg):
        if self.saved:
            return

        # 彩色图
        rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
            rgb_msg.height, rgb_msg.width, -1
        )
        if rgb_msg.encoding == "rgb8":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif rgb_msg.encoding not in ("bgr8", "bgra8"):
            self.get_logger().warn(f"未知彩色编码: {rgb_msg.encoding}")

        # 深度图
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
            depth_msg.height, depth_msg.width
        )

        # 内参
        K = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)
        Kd = np.array(depth_info_msg.k, dtype=np.float64).reshape(3, 3)

        cv2.imwrite(os.path.join(self.outdir, "rgb.png"), rgb)
        cv2.imwrite(os.path.join(self.outdir, "depth.png"), depth)
        np.savetxt(os.path.join(self.outdir, "cam_K.txt"), K)
        np.savetxt(os.path.join(self.outdir, "cam_K_depth.txt"), Kd)

        print("=" * 60)
        print("已保存:")
        print(f"  rgb   : {rgb.shape}  编码 {rgb_msg.encoding}")
        print(f"  depth : {depth.shape}  有效像素 {np.count_nonzero(depth)}")
        print(f"  K     : fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        print(f"  K_depth: fx={Kd[0,0]:.2f} fy={Kd[1,1]:.2f} cx={Kd[0,2]:.2f} cy={Kd[1,2]:.2f}")
        print(f"  时间戳: rgb {rgb_msg.header.stamp}, depth {depth_msg.header.stamp}")
        print("=" * 60)
        self.saved = True

    def check(self):
        if self.saved:
            self.get_logger().info("抓帧完成，退出")
            rclpy.shutdown()


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/fp_capture")
    rclpy.init()
    node = CaptureNode(outdir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
