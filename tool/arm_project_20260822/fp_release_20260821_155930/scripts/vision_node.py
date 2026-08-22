#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉抓取节点（独立进程）：输入抓取序列，调用机械臂节点服务抓取。

流程：拍照识别(编号与上次比对) -> 找序列目标 -> 静态姿态 ->
      调用机械臂服务发送抓取位姿 -> 等待机械臂反馈完成 -> 下一个目标。

通信：ZMQ REQ 客户端，连机械臂节点服务端口 5556。

取帧：先等相机帧（fp_bridge 在跑），等不到就自动改用静态照片
      ~/fp_capture/rgb.png（连同 depth.png / cam_K.txt，即 capture_frame.py 的输出），
      后续所有“抓一帧”都用这张照片。

用法（先启动 arm_node.py，再启动本节点; fp_bridge 在跑）:
  conda activate foundationpose
  python ~/arm_node.py     # 终端1
  python ~/vision_node.py  # 终端2

序列格式：逗号分隔 类别#编号，例如 can#2, red_apple#1；只写类别=取该类第一个。
"""

import os
import queue
import sys
import threading

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import zmq
from PIL import Image, ImageTk

sys.path.insert(0, os.path.expanduser("~"))

import fp_pipeline as fp  # noqa: E402
import grasp_ui as gu     # noqa: E402
import ui_font            # noqa: E402
from ultralytics import YOLO  # noqa: E402

ZMQ_ADDR = "tcp://127.0.0.1:5555"       # bridge 画面
SERVICE_ADDR = "tcp://127.0.0.1:5556"   # 机械臂节点服务
CAMERA_TIMEOUT_MS = 300                 # 等相机帧的最长时间；超时改用静态照片
STATIC_FRAME_DIR = os.path.expanduser("~/fp_capture")
STATIC_RGB = os.path.join(STATIC_FRAME_DIR, "rgb.png")
STATIC_DEPTH = os.path.join(STATIC_FRAME_DIR, "depth.png")   # 16bit, 单位 mm
STATIC_K = os.path.join(STATIC_FRAME_DIR, "cam_K.txt")
DEFAULT_K = np.array([[451.92, 0.0, 326.25],
                      [0.0, 451.92, 245.32],
                      [0.0, 0.0, 1.0]])


def recv_latest(sub, timeout_ms=0):
    """等一帧；timeout_ms>0 且超时返回 None（相机不在线时用静态照片）。"""
    if timeout_ms > 0 and not sub.poll(timeout_ms):
        return None
    parts = sub.recv_multipart()
    while True:
        try:
            parts = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return parts


def parse_sequence(text):
    seq = []
    for item in text.replace("，", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "#" in item:
            name, tid = item.split("#", 1)
            seq.append((name.strip(), int(tid.strip())))
        else:
            seq.append((item, None))
    return seq


class VisionNode:
    def __init__(self, root):
        self.root = root
        root.title("视觉抓取节点")
        # 中文字体：统一应用到 Tk + ttk 控件（找不到会打印安装提示）
        ui_font.setup_cn_font(root)

        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(ZMQ_ADDR)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "FRAME")
        self.req = self.ctx.socket(zmq.REQ)
        self.req.connect(SERVICE_ADDR)

        self.yolo = YOLO(fp.MODEL_PATH)
        self.tracker = gu.StableTracker(max_miss=0)
        self.objs = []
        self.ui_queue = queue.Queue()
        self.busy = False
        self.seq_items = []
        self.frame_src = "camera"   # 最近一帧来源：camera / static
        self.depth_valid = 0        # 最近一帧深度有效像素数

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="拍照识别", command=self.on_capture).pack(side="left")
        self.obj_combo = ttk.Combobox(top, state="readonly", width=26)
        self.obj_combo.pack(side="left", padx=6)
        ttk.Button(top, text="＋", width=3, command=self.on_add).pack(side="left")
        ttk.Button(top, text="清空序列", command=self.on_clear).pack(side="left", padx=4)

        row2 = ttk.Frame(root, padding=(8, 0, 8, 4))
        row2.pack(fill="x")
        ttk.Label(row2, text="序列:").pack(side="left")
        self.seq_entry = ttk.Entry(row2, width=40)
        self.seq_entry.pack(side="left", padx=6)
        self.start_btn = ttk.Button(row2, text="开始抓取", command=self.on_start)
        self.start_btn.pack(side="left")

        images = tk.Frame(root)
        images.pack()
        self.img_label = tk.Label(images, bg="black")
        self.img_label.pack(side="left", padx=4)
        self.img3d_label = tk.Label(images, bg="black")
        self.img3d_label.pack(side="left", padx=4)
        self.status = tk.Label(root, text="先启动 arm_node.py，再点『开始抓取』",
                               anchor="w", padx=8)
        self.status.pack(fill="x")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(50, self.update_ui)

    def on_capture(self):
        """拍照识别：显示所有物体，填入下拉框供选择加入序列。"""
        try:
            rgb, _, _ = self.grab_frame()
        except Exception as e:
            self.status.config(text=f"取帧失败: {e}")
            return
        self.objs = gu.detect_all_track(rgb, self.yolo, self.tracker)
        if not self.objs:
            self.status.config(text=f"[{self.frame_src}] 没有检测到物体")
            self.obj_combo["values"] = []
            self.ui_queue.put(("img", rgb))
            return
        self.obj_combo["values"] = [
            f"{o['name']} #{o.get('id')}" for o in self.objs
        ]
        self.obj_combo.current(0)
        self.ui_queue.put(("img", gu.draw_boxes(rgb, self.objs)))
        self.status.config(
            text=f"[{self.frame_src}] 识别到 {len(self.objs)} 个物体。"
                 "选择物体点『＋』加入序列，或直接在序列框输入")

    def on_add(self):
        if not self.objs or self.obj_combo.current() < 0:
            self.status.config(text="先『拍照识别』再选择物体")
            return
        o = self.objs[self.obj_combo.current()]
        self.seq_items.append(f"{o['name']}#{o.get('id')}")
        self.seq_entry.delete(0, tk.END)
        self.seq_entry.insert(0, ", ".join(self.seq_items))

    def on_clear(self):
        self.seq_items = []
        self.seq_entry.delete(0, tk.END)

    def grab_frame(self):
        """抓一帧：优先相机实时帧；相机不在线则用静态照片。"""
        parts = recv_latest(self.sub, timeout_ms=CAMERA_TIMEOUT_MS)
        if parts is not None:
            rgb = cv2.imdecode(np.frombuffer(parts[1], np.uint8),
                               cv2.IMREAD_COLOR)
            depth = np.frombuffer(parts[2], np.uint16)
            H, W = rgb.shape[:2]
            try:
                depth = depth.reshape(400, 640)
            except ValueError:
                depth = depth.reshape(H, W)
            K = np.frombuffer(parts[3], np.float64).reshape(3, 3)
            if depth.shape[:2] != (H, W):
                aligned = np.zeros((H, W), np.uint16)
                aligned[: depth.shape[0], :] = depth
                depth = aligned
            self.frame_src = "camera"
            self.depth_valid = int(np.count_nonzero(depth))
            return rgb, depth.astype(np.float32) / 1000.0, K

        # ---- 相机不在线：改用静态照片 ----
        if not os.path.exists(STATIC_RGB):
            raise RuntimeError(
                "相机未连接，且没有静态照片。请先运行 "
                "/usr/bin/python3 ~/capture_frame.py ~/fp_capture "
                f"生成 {STATIC_FRAME_DIR}/rgb.png，或修改脚本里的 STATIC_RGB")
        rgb = cv2.imread(STATIC_RGB)
        if rgb is None:
            raise RuntimeError(f"静态照片读取失败: {STATIC_RGB}")
        H, W = rgb.shape[:2]
        if os.path.exists(STATIC_DEPTH):
            depth = cv2.imread(STATIC_DEPTH, cv2.IMREAD_UNCHANGED)
        else:
            depth = np.zeros((H, W), np.uint16)
        if depth.shape[:2] != (H, W):
            aligned = np.zeros((H, W), np.uint16)
            aligned[: depth.shape[0], :] = depth
            depth = aligned
        K = DEFAULT_K.copy()
        if os.path.exists(STATIC_K):
            try:
                K = np.loadtxt(STATIC_K).reshape(3, 3)
            except Exception:
                pass
        self.frame_src = "static"
        self.depth_valid = int(np.count_nonzero(depth))
        print(f"[帧源] 相机未连接（{CAMERA_TIMEOUT_MS}ms 无帧），"
              f"改用静态照片: {STATIC_RGB}")
        return rgb, depth.astype(np.float32) / 1000.0, K

    def on_start(self):
        if self.busy:
            return
        seq = parse_sequence(self.seq_entry.get())
        if not seq:
            self.status.config(text="序列格式不对，示例: can#1, can#2")
            return
        self.busy = True
        self.start_btn.config(state="disabled")
        threading.Thread(target=self.run_pipeline, args=(seq,),
                         daemon=True).start()

    def run_pipeline(self, seq):
        try:
            for name, tid in seq:
                self.ui_queue.put(
                    ("status", f"[{self.frame_src}] 拍照识别，寻找 {name}#{tid}..."))
                rgb, depth_m, K = self.grab_frame()
                if self.frame_src == "static" and self.depth_valid == 0:
                    self.ui_queue.put(
                        ("status", "静态照片没有深度数据，无法计算姿态，流程停止"))
                    break
                self.objs = gu.detect_all_track(rgb, self.yolo, self.tracker)
                target = None
                if tid is not None:
                    target = next((o for o in self.objs
                                   if o["name"] == name and o.get("id") == tid),
                                  None)
                if target is None:
                    target = next((o for o in self.objs if o["name"] == name),
                                  None)
                if target is None:
                    self.ui_queue.put(
                        ("status",
                         f"序列目标 {name}#{tid} 不在画面，流程停止"))
                    break
                self.ui_queue.put(
                    ("status", f"找到 {name} #{target.get('id')}，计算姿态..."))

                obj_key = fp.YOLO_TO_OBJECT.get(name)
                if obj_key is None:
                    self.ui_queue.put(("status", f"类别 {name} 未配置，停止"))
                    break
                mesh_file = fp.OBJECT_MODELS.get(obj_key, "")
                if not mesh_file or not os.path.exists(mesh_file):
                    self.ui_queue.put(("status", f"物体 {obj_key} 模型未配置"))
                    break
                est = fp.PoseEstimator(mesh_file)

                # 3D 视图：Agg 渲染成图片，显示在 2D 画面右侧
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                if getattr(self, "_fig3d", None) is not None:
                    try:
                        plt.close(self._fig3d)
                    except Exception:
                        pass
                self._fig3d = plt.figure(figsize=(5, 5))
                self._ax3d = self._fig3d.add_subplot(111, projection="3d")
                self._plt3d = plt
                self._viz3d_path = os.path.join(fp.DEBUG_DIR,
                                                "vision_node_3d.png")
                self.ui_queue.put(("status", "3D 视图将显示在 2D 画面右侧"))

                mask = target["mask"] if target["mask"] is not None \
                    else np.zeros(rgb.shape[:2], np.uint8)
                if mask.sum() == 0:
                    x1, y1, x2, y2 = target["xyxy"]
                    mask[y1:y2, x1:x2] = 255
                M4 = est.est.register(
                    K, rgb, fp.fill_depth_roi(depth_m, mask), mask, iteration=5)
                R, t = M4[:3, :3], M4[:3, 3]
                if not (0.3 < t[2] < 1.5):
                    self.ui_queue.put(("status", f"姿态距离异常 z={t[2]*1000:.0f}mm"))
                    break
                tags = fp.detect_tags(rgb, fp.TAG_SIZE_MM)
                T_world = fp.build_world(tags) if tags else None
                if T_world is None:
                    self.ui_queue.put(("status", "未检测到 tag0，停止"))
                    break
                M_cam = np.eye(4)
                M_cam[:3, :3] = R
                M_cam[:3, 3] = t
                M_world = fp.to_world_and_compensate(M_cam, np.linalg.inv(T_world))
                rule = fp.GRASP_RULES.get(obj_key, fp.GRASP_RULES[fp.DEFAULT_OBJECT])
                if rule.get("type") == "sphere":
                    T_grasp = fp.compute_grasp_sphere(
                        M_world, rule.get("offset_mm", 0.0))
                else:
                    T_grasp = fp.compute_grasp(M_world)
                w = M_world[:3, 3] * 1000.0
                g = T_grasp[:3, 3] * 1000.0
                np.save(fp.GRASP_FILE, T_grasp)

                viz = fp.draw_2d(rgb, est.mesh, R, t, K, est.bbox,
                                 target, [o["xyxy"] for o in self.objs])
                viz = gu.draw_boxes(viz, self.objs)
                cv2.putText(viz,
                            f"目标 {name} #{target.get('id')} | "
                            f"world(mm): x={w[0]:.1f} y={w[1]:.1f} z={w[2]:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                self.ui_queue.put(("img", viz))

                # 3D 工作台视图（物体姿态框 + 抓取三叉戟）
                mn = est.mesh.bounds[0]
                mx = est.mesh.bounds[1]
                fp.draw_3d(self._ax3d, self._fig3d, self._plt3d,
                           M_world, T_grasp, self._viz3d_path, mn, mx)
                bgr3 = cv2.imread(self._viz3d_path)
                if bgr3 is not None:
                    self.ui_queue.put(("img3d", bgr3))

                # 调用机械臂节点服务：发送抓取位姿，阻塞等完成
                self.ui_queue.put(
                    ("status",
                     f"调用机械臂服务: {name} #{target.get('id')}，等待执行..."))
                self.req.send_pyobj(T_grasp)
                resp = self.req.recv_pyobj()  # 机械臂点『完成』后才返回
                self.ui_queue.put(
                    ("status",
                     f"{name} #{target.get('id')} 抓取完成 "
                     f"（机械臂反馈: {resp.get('status')}）"))

            self.ui_queue.put(("status", "抓取序列全部完成"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.ui_queue.put(("status", f"流程异常: {e}"))
        finally:
            self.busy = False
            self.start_btn.config(state="normal")

    def update_ui(self):
        try:
            while True:
                kind, val = self.ui_queue.get_nowait()
                if kind == "img":
                    rgb = cv2.cvtColor(val, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    photo = ImageTk.PhotoImage(img)
                    self.img_label.config(image=photo)
                    self.img_label.image = photo
                elif kind == "img3d":
                    rgb = cv2.cvtColor(val, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    photo = ImageTk.PhotoImage(img)
                    self.img3d_label.config(image=photo)
                    self.img3d_label.image = photo
                elif kind == "status":
                    self.status.config(text=val)
                    print(val)
        except queue.Empty:
            pass
        self.root.after(50, self.update_ui)

    def on_close(self):
        try:
            self.req.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        finally:
            os._exit(0)


def main():
    root = tk.Tk()
    VisionNode(root)
    root.mainloop()


if __name__ == "__main__":
    main()
