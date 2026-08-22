#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取流水线图形界面：重新识别 -> 选择目标 -> 姿态检测。

用法（fp_bridge.py 必须先在系统 python 里跑着）:
  conda activate foundationpose
  python ~/grasp_ui.py

功能:
  1. 点击"重新识别"：从相机取一帧，YOLO 识别所有物体，显示结果图；
  2. 下拉框选择要抓的目标；
  3. 点击"开始姿态检测"：对选定目标跑 FoundationPose + tag 定位 + 抓取位姿，
     画面实时刷新，world/grasp 打印在终端。
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

import fp_pipeline as fp  # noqa: E402  复用检测/位姿/抓取逻辑
import ui_font            # noqa: E402  中文字体（须在 tk.Tk() 之前 import）
from ultralytics import YOLO  # noqa: E402

ZMQ_ADDR = "tcp://127.0.0.1:5555"
YOLO_CONF = 0.85
YOLO_IMGSZ = 640
SCALE = 1.0  # UI 显示缩放


def recv_latest(sub):
    parts = sub.recv_multipart()
    while True:
        try:
            parts = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return parts


def draw_boxes(rgb, objs):
    """在识别结果图上画所有物体的框和标签（类别 + ID/序号）。"""
    out = rgb.copy()
    for o in objs:
        x1, y1, x2, y2 = o["xyxy"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        tag = o.get("id") if o.get("id") is not None else o.get("seq")
        cv2.putText(out, f"{o['name']} #{tag}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return out


def add_seq(objs):
    """给同类别的多个实例编号：#1, #2, ...（ID 为空时的回退）"""
    counter = {}
    for o in objs:
        counter[o["name"]] = counter.get(o["name"], 0) + 1
        o["seq"] = counter[o["name"]]
    return objs


def detect_all_track(rgb, model, tracker):
    """YOLO 检测 + 帧间 IOU 匹配分配稳定 ID（漏检编号不变）。"""
    H, W = rgb.shape[:2]
    res = model.predict(rgb, conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                        verbose=False)[0]
    objs = []
    masks = res.masks.data.cpu().numpy() if res.masks is not None else None
    if res.boxes is None or len(res.boxes) == 0:
        tracker.update([])  # 空帧也推进跟踪状态（编号保留）
        return objs
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else None
    for i, (b, c, cl) in enumerate(zip(xyxy, confs, cls)):
        x1, y1, x2, y2 = b.astype(int)
        mask = None
        if masks is not None and i < len(masks):
            m = (masks[i] > 0.5).astype(np.uint8) * 255
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = m
        objs.append({
            "xyxy": (max(0, x1), max(0, y1), min(W, x2), min(H, y2)),
            "cls": int(cl),
            "name": model.names[cl],
            "conf": float(c),
            "mask": mask,
            "id": int(ids[i]) if ids is not None else None,
        })
    objs = tracker.update(objs)
    return add_seq(objs)


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class StableTracker:
    """帧间 IOU 匹配保持 ID：漏检几帧编号不变，新物体才分配新号。"""
    def __init__(self, max_miss=15, match_iou=0.1):
        self.tracks = {}      # id -> (xyxy, name, miss_count)
        self.next_id = 1
        self.max_miss = max_miss
        self.match_iou = match_iou

    def update(self, objs):
        unmatched = list(range(len(objs)))
        new_tracks = {}
        for tid in list(self.tracks):
            txy, tname, miss = self.tracks[tid]
            best_i = -1
            best_v = self.match_iou
            for i in unmatched:
                o = objs[i]
                if o["name"] != tname:
                    continue
                v = box_iou(txy, o["xyxy"])
                if v > best_v:
                    best_v = v
                    best_i = i
            if best_i >= 0:
                objs[best_i]["id"] = tid
                new_tracks[tid] = (objs[best_i]["xyxy"], tname, 0)
                unmatched.remove(best_i)
            else:
                new_tracks[tid] = (txy, tname, miss + 1)  # 本帧漏检，编号保留
        for i in unmatched:
            o = objs[i]
            o["id"] = self.next_id
            new_tracks[self.next_id] = (o["xyxy"], o["name"], 0)
            self.next_id += 1
        # 清理连续丢失超过 max_miss 帧的编号
        self.tracks = {tid: v for tid, v in new_tracks.items()
                       if v[2] <= self.max_miss}
        return objs


class App:
    def __init__(self, root):
        self.root = root
        root.title("抓取流水线")

        # ZMQ 帧源
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(ZMQ_ADDR)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "FRAME")

        # YOLO
        self.yolo = YOLO(fp.MODEL_PATH)
        self.tracker = StableTracker()
        print("YOLO 类别:", self.yolo.names)

        self.objs = []
        self.ui_queue = queue.Queue()
        self.pose_thread = None
        self.stop_flag = threading.Event()

        # ---- UI 控件 ----
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="重新识别", command=self.on_recognize).pack(side="left")
        self.combo = ttk.Combobox(top, state="readonly", width=30)
        self.combo.pack(side="left", padx=8)
        ttk.Button(top, text="开始姿态检测", command=self.on_start).pack(side="left")

        images = tk.Frame(root)
        images.pack()
        self.img_label = tk.Label(images, bg="black")
        self.img_label.pack(side="left", padx=4)
        self.img3d_label = tk.Label(images, bg="black")
        self.img3d_label.pack(side="left", padx=4)

        self.status = tk.Label(root, text="等待操作：先点『重新识别』",
                               anchor="w", padx=8)
        self.status.pack(fill="x")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(50, self.update_ui)

    def grab_frame(self):
        parts = recv_latest(self.sub)
        rgb = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
        depth = np.frombuffer(parts[2], np.uint16).reshape(400, 640)
        K = np.frombuffer(parts[3], np.float64).reshape(3, 3)
        H, W = rgb.shape[:2]
        if depth.shape[:2] != (H, W):
            aligned = np.zeros((H, W), np.uint16)
            aligned[: depth.shape[0], :] = depth
            depth = aligned
        return rgb, depth.astype(np.float32) / 1000.0, K

    def show_img(self, bgr):
        self.ui_queue.put(("img", bgr.copy()))

    def show_img3d(self, bgr):
        self.ui_queue.put(("img3d", bgr.copy()))

    def on_recognize(self):
        self.stop_pose()
        self.status.config(text="识别中...")
        try:
            rgb, _, _ = self.grab_frame()
        except Exception as e:
            self.status.config(text=f"取帧失败: {e}（确认 fp_bridge 在跑）")
            return
        self.objs = detect_all_track(rgb, self.yolo, self.tracker)
        if not self.objs:
            self.status.config(text="没有检测到物体")
            self.combo["values"] = []
            self.show_img(rgb)
            return
        self.combo["values"] = [
            f"{o['name']} #{o.get('id') if o.get('id') is not None else o['seq']}  "
            f"conf={o['conf']:.2f}" for o in self.objs
        ]
        self.combo.current(0)
        self.show_img(draw_boxes(rgb, self.objs))
        self.status.config(text=f"识别到 {len(self.objs)} 个物体，请选择目标")

    def on_start(self):
        if not self.objs or self.combo.current() < 0:
            self.status.config(text="请先『重新识别』并选择目标")
            return
        self.stop_pose()  # 先停旧线程，再以新目标重新注册
        idx = self.combo.current()
        old_obj = self.objs[idx]
        # 抓最新一帧重新检测，用与选中框最近的同类作为注册目标
        init_obj = old_obj
        try:
            rgb, _, _ = self.grab_frame()
            objs_new = detect_all_track(rgb, self.yolo, self.tracker)
            same_cls = [o for o in objs_new if o["name"] == old_obj["name"]]
            if same_cls:
                old_box = np.array(old_obj["xyxy"], dtype=float)
                rc = np.array([(old_box[0] + old_box[2]) / 2.0,
                               (old_box[1] + old_box[3]) / 2.0])
                centers = np.array([[
                    (o["xyxy"][0] + o["xyxy"][2]) / 2.0,
                    (o["xyxy"][1] + o["xyxy"][3]) / 2.0] for o in same_cls])
                init_obj = same_cls[int(np.argmin(
                    np.linalg.norm(centers - rc, axis=1)))]
            self.objs = objs_new if objs_new else self.objs
            self.combo["values"] = [
                f"{o['name']} #{o.get('id') if o.get('id') is not None else o['seq']}  "
                f"conf={o['conf']:.2f}" for o in self.objs
            ]
        except Exception:
            pass
        label = str(init_obj["name"])
        self.status.config(text=f"开始姿态检测: {label} #{init_obj['seq']}")
        self.stop_flag.clear()
        self.pose_thread = threading.Thread(
            target=self.pose_worker, args=(label, init_obj), daemon=True)
        self.pose_thread.start()

    def stop_pose(self):
        self.stop_flag.set()
        if self.pose_thread is not None and self.pose_thread.is_alive():
            self.pose_thread.join(timeout=3)
        self.pose_thread = None

    def pose_worker(self, label, init_obj):
        """姿态检测循环：只处理选定类别；找不到就跳过，不回退。"""
        try:
            obj_key = fp.YOLO_TO_OBJECT.get(label)
            if obj_key is None:
                self.ui_queue.put(
                    ("status", f"类别 {label} 未在 YOLO_TO_OBJECT 配置，跳过"))
                return
            mesh_file = fp.OBJECT_MODELS.get(obj_key, "")
            if not mesh_file or not os.path.exists(mesh_file):
                self.ui_queue.put(("status", f"物体 {obj_key} 模型未配置"))
                return
            rule = fp.GRASP_RULES.get(obj_key, fp.GRASP_RULES[fp.DEFAULT_OBJECT])
            self.ui_queue.put(("status", f"加载模型: {mesh_file}"))
            est = fp.PoseEstimator(mesh_file)

            # 3D 视图：渲染成图片显示在 UI 里（避免线程弹窗冲突）
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            _fig3d = plt.figure(figsize=(6, 6))
            _ax3d = _fig3d.add_subplot(111, projection="3d")
            _plt3d = plt
            path3d = os.path.join(fp.DEBUG_DIR, "live_ui_3d.png")
            self.ui_queue.put(("status", "3D 视图将显示在界面下方"))

            smooth_q = smooth_t = None
            tag_q = tag_t = None
            first = True
            # 用选中物体的 2D 框作为参考，每帧选最近的同类物体跟随
            ref_box = np.array(init_obj["xyxy"], dtype=float)
            init_id = init_obj.get("id")
            while not self.stop_flag.is_set():
                rgb, depth_m, K = self.grab_frame()

                # tag 工作台系
                tags = fp.detect_tags(rgb, fp.TAG_SIZE_MM)
                T_world = fp.build_world(tags) if tags else None
                if T_world is None:
                    continue
                if fp.TAG_ALPHA > 0:
                    from scipy.spatial.transform import Rotation as R_util
                    q = R_util.from_matrix(T_world[:3, :3]).as_quat()
                    tv = T_world[:3, 3].copy()
                    if tag_q is None:
                        tag_q, tag_t = q, tv
                    else:
                        if np.dot(tag_q, q) < 0:
                            q = -q
                        tag_q = tag_q + fp.TAG_ALPHA * (q - tag_q)
                        tag_q /= np.linalg.norm(tag_q)
                        tag_t = tag_t + fp.TAG_ALPHA * (tv - tag_t)
                    T_world = np.eye(4)
                    T_world[:3, :3] = R_util.from_quat(tag_q).as_matrix()
                    T_world[:3, 3] = tag_t
                T_world_cam = np.linalg.inv(T_world)

                objs = detect_all_track(rgb, self.yolo, self.tracker)
                cands = [o for o in objs if o["name"] == label]
                target = None
                if cands and init_id is not None:
                    target = next(
                        (o for o in cands if o.get("id") == init_id), None)
                if target is None and cands:
                    centers = np.array([[
                        (o["xyxy"][0] + o["xyxy"][2]) / 2.0,
                        (o["xyxy"][1] + o["xyxy"][3]) / 2.0] for o in cands])
                    rc = np.array([(ref_box[0] + ref_box[2]) / 2.0,
                                   (ref_box[1] + ref_box[3]) / 2.0])
                    target = cands[int(np.argmin(
                        np.linalg.norm(centers - rc, axis=1)))]
                if target is None:
                    # 目标丢失：更新画面提示，不检测姿态、不残留旧框
                    smooth_q = smooth_t = None
                    viz = draw_boxes(rgb.copy(), self.objs)
                    cv2.putText(viz, "目标丢失", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    self.show_img(viz)
                    continue
                ref_box = np.array(target["xyxy"], dtype=float)  # 跟随目标

                # FoundationPose
                # 点"开始"时注册一次；之后跟踪，偏差大才用 ROI 重注册
                mask = target["mask"] if target["mask"] is not None \
                    else np.zeros(rgb.shape[:2], np.uint8)
                if mask.sum() == 0:
                    x1, y1, x2, y2 = target["xyxy"]
                    mask[y1:y2, x1:x2] = 255
                if first:
                    M4 = est.est.register(
                        K, rgb, fp.fill_depth_roi(depth_m, mask), mask,
                        iteration=5)
                    R, t = M4[:3, :3], M4[:3, 3]
                    first = False
                    smooth_q = smooth_t = None
                    print(f"[调试] 注册: 目标 {label} id={target.get('id')} "
                          f"框={target['xyxy']} mask面积={mask.sum()}")
                else:
                    R, t = est.track(K, rgb, depth_m)
                    # 核对：投影中心与目标框中心偏差 >35px 则重注册
                    u = K[0, 0] * t[0] / t[2] + K[0, 2]
                    v = K[1, 1] * t[1] / t[2] + K[1, 2]
                    x1, y1, x2, y2 = target["xyxy"]
                    bc = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    dist = np.hypot(u - bc[0], v - bc[1])
                    print(f"[调试] 跟踪: 目标 id={target.get('id')} "
                          f"框中心=({bc[0]:.0f},{bc[1]:.0f}) "
                          f"投影=({u:.0f},{v:.0f}) 偏差={dist:.0f}px")
                    if dist > 35:
                        first = True
                        continue
                if not (0.3 < t[2] < 1.5):
                    first = True
                    continue

                # 坐标转换 + 平滑
                M_cam = np.eye(4)
                M_cam[:3, :3] = R
                M_cam[:3, 3] = t
                M_world = fp.to_world_and_compensate(M_cam, T_world_cam)
                if fp.ALPHA > 0:
                    from scipy.spatial.transform import Rotation as R_util
                    q = R_util.from_matrix(M_world[:3, :3]).as_quat()
                    tv = M_world[:3, 3].copy()
                    if smooth_q is None:
                        smooth_q, smooth_t = q, tv
                    else:
                        if np.dot(smooth_q, q) < 0:
                            q = -q
                        smooth_q = smooth_q + fp.ALPHA * (q - smooth_q)
                        smooth_q /= np.linalg.norm(smooth_q)
                        smooth_t = smooth_t + fp.ALPHA * (tv - smooth_t)
                    M_world = np.eye(4)
                    M_world[:3, :3] = R_util.from_quat(smooth_q).as_matrix()
                    M_world[:3, 3] = smooth_t
                w = M_world[:3, 3] * 1000.0

                # 抓取位姿
                if rule.get("type") == "sphere":
                    T_grasp = fp.compute_grasp_sphere(
                        M_world, rule.get("offset_mm", 0.0))
                else:
                    T_grasp = fp.compute_grasp(M_world)
                g = T_grasp[:3, 3] * 1000.0

                # 2D 画面（目标掩膜 + 姿态框）
                R_cam, t_cam = R, t
                # 其他物体用识别快照的框（保持不动），目标用当前帧
                snap_others = [o for o in self.objs
                               if not (o.get("id") == init_id
                                       and o["name"] == label)]
                display_objs = snap_others + [target]
                viz = fp.draw_2d(rgb, est.mesh, R_cam, t_cam, K, est.bbox,
                                 target, [o["xyxy"] for o in display_objs])
                viz = draw_boxes(viz, display_objs)
                self.show_img(viz)
                mn = est.mesh.bounds[0]
                mx = est.mesh.bounds[1]
                fp.draw_3d(_ax3d, _fig3d, _plt3d, M_world, T_grasp,
                           path3d, mn, mx)
                bgr3 = cv2.imread(path3d)
                if bgr3 is not None:
                    self.show_img3d(bgr3)
                self.ui_queue.put((
                    "status",
                    f"world(mm): x={w[0]:.1f} y={w[1]:.1f} z={w[2]:.1f}  |  "
                    f"grasp(mm): x={g[0]:.1f} y={g[1]:.1f} z={g[2]:.1f}"))
                print(f"world(mm): x={w[0]:.1f} y={w[1]:.1f} z={w[2]:.1f} | "
                      f"grasp(mm): x={g[0]:.1f} y={g[1]:.1f} z={g[2]:.1f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.ui_queue.put(("status", f"姿态检测异常: {e}"))

    def update_ui(self):
        try:
            while True:
                kind, val = self.ui_queue.get_nowait()
                if kind == "img":
                    rgb = cv2.cvtColor(val, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    img = img.resize(
                        (int(img.width * SCALE), int(img.height * SCALE)))
                    photo = ImageTk.PhotoImage(img)
                    self.img_label.config(image=photo)
                    self.img_label.image = photo
                elif kind == "img3d":
                    rgb = cv2.cvtColor(val, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    img = img.resize(
                        (int(img.width * SCALE), int(img.height * SCALE)))
                    photo = ImageTk.PhotoImage(img)
                    self.img3d_label.config(image=photo)
                    self.img3d_label.image = photo
                elif kind == "status":
                    self.status.config(text=val)
        except queue.Empty:
            pass
        self.root.after(50, self.update_ui)

    def on_close(self):
        self.stop_pose()
        try:
            self.root.destroy()
        finally:
            os._exit(0)  # 确保进程彻底退出（远程桌面下 Tk 窗口可能不响应）


def main():
    root = tk.Tk()
    ui_font.setup_cn_font(root)
    app = App(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()


if __name__ == "__main__":
    main()
