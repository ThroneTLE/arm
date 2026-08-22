#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机械臂模拟节点（独立进程）：接收视觉节点的抓取请求，模拟执行。

通信：ZMQ REP 服务，端口 5556（模拟 ROS2 service/action）。
状态：无抓取信号 / 抓取进行中；只有点『模拟抓取完成』才向视觉反馈完成。

用法:
  conda activate foundationpose
  python ~/arm_node.py
"""

import os
import queue
import threading

import numpy as np
import tkinter as tk
import zmq

import ui_font

SERVICE_ADDR = "tcp://127.0.0.1:5556"


class ArmNode:
    def __init__(self, root):
        self.root = root
        root.title("机械臂（模拟节点）")
        root.geometry("420x260")
        self.cn = ui_font.setup_cn_font(root) or ""

        self.ctx = zmq.Context()
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.bind(SERVICE_ADDR)

        self.done_event = threading.Event()
        self.ui_queue = queue.Queue()

        tk.Label(root, text="机械臂节点", font=(self.cn, 15)).pack(pady=8)
        self.state_var = tk.StringVar(value="无抓取信号")
        tk.Label(root, textvariable=self.state_var,
                 font=(self.cn, 16), fg="#0b5394").pack(pady=6)
        self.pose_var = tk.StringVar(value="等待视觉节点调用...")
        tk.Label(root, textvariable=self.pose_var,
                 font=(self.cn, 12), justify="left").pack(pady=6)
        self.btn = tk.Button(root, text="模拟抓取完成",
                             font=(self.cn, 12),
                             command=self.on_done, state="disabled")
        self.btn.pack(pady=10)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        threading.Thread(target=self.serve, daemon=True).start()
        root.after(50, self.update_ui)

    def on_done(self):
        self.done_event.set()

    def serve(self):
        """服务循环：收抓取请求 -> 状态'抓取进行中' -> 等用户点完成 -> 回复。"""
        while True:
            try:
                T_grasp = self.rep.recv_pyobj()
            except Exception:
                break
            g = np.asarray(T_grasp)[:3, 3] * 1000.0
            pose_str = (f"抓取位姿 (mm):\nx={g[0]:.1f}  y={g[1]:.1f}  z={g[2]:.1f}")
            self.ui_queue.put(("pose", pose_str))
            self.done_event.clear()
            self.done_event.wait(3600)  # 等用户点『模拟抓取完成』
            try:
                self.rep.send_pyobj({"status": "done"})
            except Exception:
                break
            self.ui_queue.put(("pose", "等待视觉节点调用..."))

    def update_ui(self):
        try:
            while True:
                kind, val = self.ui_queue.get_nowait()
                if kind == "pose":
                    self.state_var.set("抓取进行中")
                    self.pose_var.set(val)
                    self.btn.config(state="normal")
                elif kind == "reset":
                    self.state_var.set("无抓取信号")
                    self.pose_var.set(val)
                    self.btn.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(50, self.update_ui)

    def on_close(self):
        try:
            self.rep.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        finally:
            os._exit(0)


def main():
    root = tk.Tk()
    ArmNode(root)
    root.mainloop()


if __name__ == "__main__":
    main()
