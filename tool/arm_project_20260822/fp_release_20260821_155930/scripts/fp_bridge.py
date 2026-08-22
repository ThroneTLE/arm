#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 -> ZeroMQ 桥接 + 位姿发布节点（系统 python3 运行，不要用 conda）。

功能:
  1. 订阅相机彩色/深度/内参，同步后压缩发给推理端 (ZMQ PUB, 端口 5555)
  2. 每 0.2s 读取推理端写出的位姿文件，发布为 ROS2 话题 /can_pose (PoseStamped)

用法:
  /usr/bin/python3 ~/fp_bridge.py
"""

import os
import time

import cv2
import numpy as np
import rclpy
import zmq
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

ZMQ_ADDR = "tcp://127.0.0.1:5555"
POSE_FILE = "/tmp/can_pose.npy"


def rot2quat(R):
    """旋转矩阵 -> 四元数 [x, y, z, w]"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return [float(qx), float(qy), float(qz), float(qw)]


class Bridge(Node):
    def __init__(self):
        super().__init__("fp_bridge")
        self.ctx = zmq.Context()
        self.zpub = self.ctx.socket(zmq.PUB)
        self.zpub.bind(ZMQ_ADDR)
        self.pose_pub = self.create_publisher(PoseStamped, "/can_pose", 10)

        self.frame_count = 0
        self.t_last = time.time()

        sub_rgb = Subscriber(self, Image, "/camera/color/image_raw")
        sub_depth = Subscriber(self, Image, "/camera/depth/image_raw")
        sub_info = Subscriber(self, CameraInfo, "/camera/color/camera_info")
        self.sync = ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_info], queue_size=20, slop=0.2
        )
        self.sync.registerCallback(self.cb)
        self.timer = self.create_timer(0.2, self.publish_pose)
        self.get_logger().info("桥接启动: 等待相机话题...")

    def cb(self, rgb_msg, depth_msg, info_msg):
        rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
            rgb_msg.height, rgb_msg.width, -1
        )
        if rgb_msg.encoding == "rgb8":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
            depth_msg.height, depth_msg.width
        )
        K = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)

        ok, jpg = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        self.zpub.send_multipart(
            [b"FRAME", jpg.tobytes(), depth.tobytes(), K.tobytes()]
        )

        self.frame_count += 1
        now = time.time()
        if now - self.t_last >= 5.0:
            fps = self.frame_count / (now - self.t_last)
            self.get_logger().info(f"发送帧率: {fps:.1f} Hz")
            self.frame_count = 0
            self.t_last = now

    def publish_pose(self):
        if not os.path.exists(POSE_FILE):
            return
        try:
            M = np.load(POSE_FILE)
            R = M[:, :3]
            t = M[:, 3]
            msg = PoseStamped()
            msg.header.frame_id = "camera_color_optical_frame"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = float(t[0])
            msg.pose.position.y = float(t[1])
            msg.pose.position.z = float(t[2])
            qx, qy, qz, qw = rot2quat(R)
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw
            self.pose_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"位姿发布失败: {e}")


def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
