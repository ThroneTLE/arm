#!/usr/bin/env python3
"""Tkinter UI for the migrated visual grasping pipeline.

The UI shows YOLO recognition results on the image, lets you pick a target, and
can run the offline FoundationPose + grasp calculation.  It supports two frame
sources:

* static frame from ``static_frame_dir`` (always available, no camera needed);
* live camera frames from the original ``fp_bridge`` ZMQ publisher on
  ``tcp://127.0.0.1:5555`` (optional).

Run with::

    ./tool/visual_grasp_pipeline/run_ui.sh
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from .config import VisualGraspConfig
from .detection import detect_all_track, detect_tags, draw_boxes, select_target
from .foundationpose import FoundationPosePoseEstimator
from .geometry import (
    build_world_from_tags,
    compute_grasp,
    compute_grasp_sphere,
    fill_depth_roi,
    to_world_and_compensate,
)
from .offline import load_static_frame
from .tracking import StableTracker

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - Pillow is required for image display
    Image = None
    ImageTk = None

ZMQ_ADDR = "tcp://127.0.0.1:5555"
CAMERA_TIMEOUT_MS = 500


def recv_latest(sub, timeout_ms=0):
    """Receive the newest ZMQ multipart frame; return None on timeout."""
    if timeout_ms > 0 and not sub.poll(timeout_ms):
        return None
    parts = sub.recv_multipart()
    try:
        import zmq
        no_block = zmq.NOBLOCK
    except Exception:
        no_block = 0x1
    while True:
        try:
            parts = sub.recv_multipart(flags=no_block)
        except Exception:
            break
    return parts


class VisualGraspUI:
    def __init__(self, root: tk.Tk, config: VisualGraspConfig):
        self.root = root
        self.config = config
        self.model = None
        self.tracker = StableTracker(max_miss=10)
        self.objects = []
        self.latest_frame = None
        self.latest_depth_m = None
        self.latest_K = None
        self.pose_estimators = {}
        self.ui_queue = queue.Queue()
        self.ctx = None
        self.sub = None

        root.title("视觉抓取流水线 - 识别结果")
        root.geometry("1100x760")

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="静态帧识别", command=self.on_recognize_static).pack(side="left")
        ttk.Button(top, text="相机识别", command=self.on_recognize_camera).pack(side="left", padx=6)
        ttk.Button(top, text="识别并计算位姿", command=self.on_pose).pack(side="left", padx=6)

        self.source_var = tk.StringVar(value="static")
        ttk.Label(top, text="数据源:").pack(side="left", padx=(12, 2))
        ttk.Combobox(
            top,
            textvariable=self.source_var,
            values=["static", "camera"],
            state="readonly",
            width=8,
        ).pack(side="left")

        self.obj_combo = ttk.Combobox(top, state="readonly", width=28)
        self.obj_combo.pack(side="left", padx=12)

        self.status = tk.Label(root, text="就绪", anchor="w", padx=8)
        self.status.pack(fill="x")

        image_frame = tk.Frame(root)
        image_frame.pack(fill="both", expand=True)
        self.image_label = tk.Label(image_frame, bg="black")
        self.image_label.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(root, padding=4)
        right.pack(side="right", fill="y")
        ttk.Label(right, text="识别结果").pack(anchor="w")
        self.result_text = tk.Text(right, width=46, height=30)
        self.result_text.pack(fill="both", expand=True)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(50, self.update_ui)

    def _ensure_model(self):
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.config.yolo_weights)
        return self.model

    def _ensure_camera(self):
        if self.sub is not None:
            return self.sub
        try:
            import zmq
        except Exception as exc:
            raise RuntimeError("pyzmq 未安装，无法使用相机识别") from exc
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(ZMQ_ADDR)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "FRAME")
        return self.sub

    def grab_frame(self, source: str):
        if source == "camera":
            sub = self._ensure_camera()
            parts = recv_latest(sub, timeout_ms=CAMERA_TIMEOUT_MS)
            if parts is None:
                raise RuntimeError("相机 ZMQ 无数据，请先运行 fp_bridge.py")
            rgb = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
            depth = np.frombuffer(parts[2], np.uint16)
            height, width = rgb.shape[:2]
            try:
                depth = depth.reshape(400, 640)
            except ValueError:
                depth = depth.reshape(height, width)
            k = np.frombuffer(parts[3], np.float64).reshape(3, 3)
            if depth.shape[:2] != (height, width):
                aligned = np.zeros((height, width), dtype=np.uint16)
                aligned[: depth.shape[0], :] = depth
                depth = aligned
            return rgb, depth.astype(np.float32) / 1000.0, k
        return load_static_frame(self.config.static_frame_dir)

    def recognize(self, source: str):
        try:
            model = self._ensure_model()
            rgb, depth_m, k = self.grab_frame(source)
            objects = detect_all_track(rgb, model, self.tracker,
                                       conf=self.config.yolo_conf,
                                       imgsz=self.config.yolo_imgsz)
            self.objects = objects
            self.latest_frame = rgb
            self.latest_depth_m = depth_m
            self.latest_K = k
            viz = draw_boxes(rgb, objects)
            self.ui_queue.put(("image", viz))
            lines = []
            for obj in objects:
                lines.append(
                    f"{obj['name']} #{obj.get('id')} conf={obj['conf']:.3f} "
                    f"box={obj['xyxy']}"
                )
            if not lines:
                lines = ["没有识别到物体"]
            self.ui_queue.put(("objects", lines))
            self.ui_queue.put(("status", f"[{source}] 识别到 {len(objects)} 个物体"))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if source == "camera":
                self.ui_queue.put(("status", f"相机不可用（{exc}），自动改用静态帧"))
                self.ui_queue.put(("source", "static"))
                self.recognize("static")
                return
            self.ui_queue.put(("status", f"识别失败: {exc}"))

    def on_recognize_static(self):
        self.source_var.set("static")
        threading.Thread(target=self.recognize, args=("static",), daemon=True).start()

    def on_recognize_camera(self):
        self.source_var.set("camera")
        threading.Thread(target=self.recognize, args=("camera",), daemon=True).start()

    def on_pose(self):
        label = None
        if self.obj_combo.current() >= 0 and self.objects:
            label = self.objects[self.obj_combo.current()]["name"]
        threading.Thread(target=self._pose_worker, args=(label,), daemon=True).start()

    def _get_estimator(self, mesh_path, mesh_scale_to_meters):
        key = (str(mesh_path), float(mesh_scale_to_meters))
        estimator = self.pose_estimators.get(key)
        if estimator is None:
            estimator = FoundationPosePoseEstimator(
                foundationpose_root=self.config.foundationpose_root,
                mesh_path=mesh_path,
                mesh_scale_to_meters=mesh_scale_to_meters,
                debug_dir=self.config.debug_dir,
                est_refine_iter=self.config.est_refine_iter,
                track_refine_iter=self.config.track_refine_iter,
                device=self.config.device,
                use_mask_center_guidance=self.config.use_mask_center_guidance,
                registration_max_hypotheses=(
                    self.config.foundationpose_registration_hypotheses
                ),
            )
            self.pose_estimators[key] = estimator
        return estimator

    def _pose_worker(self, label):
        try:
            self.ui_queue.put(("status", "正在计算 FoundationPose 位姿（首次会较慢）..."))
            if self.latest_frame is None:
                rgb, depth_m, k = load_static_frame(self.config.static_frame_dir)
            else:
                rgb = self.latest_frame
                depth_m = self.latest_depth_m
                k = self.latest_K
                if depth_m is None or k is None:
                    rgb, depth_m, k = load_static_frame(self.config.static_frame_dir)

            # If nothing has been recognized yet, automatically run a static-frame
            # recognition first so the button can be used as a one-click flow.
            if not self.objects:
                self.recognize("static")
            # If no explicit label, use the currently selected object captured by
            # the main thread (on_pose already converted it to a class name).
            target = select_target(self.objects, label) if self.objects else None
            if target is None:
                self.ui_queue.put(("status", "没有可计算位姿的目标，请先识别"))
                return

            object_key = self.config.resolve_object_key(target["name"], target["cls"])
            mesh_path = self.config.mesh_for_object(object_key)
            if not mesh_path or not os.path.exists(mesh_path):
                self.ui_queue.put(("status", f"物体 {object_key} 的模型未配置: {mesh_path}"))
                return

            height, width = rgb.shape[:2]
            # 掩膜: 直接用 YOLO 识别框(矩形)作为掩膜, 不使用实例分割输出
            mask = np.zeros((height, width), np.uint8)
            bx1, by1, bx2, by2 = (int(round(v)) for v in target["xyxy"])
            mask[max(0, by1):min(height, by2), max(0, bx1):min(width, bx2)] = 255

            estimator = self._get_estimator(
                mesh_path, self.config.mesh_scale_for_object(object_key)
            )
            camera_from_object = estimator.register(
                rgb, fill_depth_roi(depth_m, mask), mask, k
            )

            tags = []
            try:
                tags = detect_tags(rgb, self.config.tag_size_mm, k)
            except Exception:
                tags = []
            world_from_object = camera_from_object.copy()
            if tags:
                tag_world = build_world_from_tags(tags)
                if tag_world is not None:
                    world_from_object = to_world_and_compensate(
                        camera_from_object,
                        np.linalg.inv(tag_world),
                        offset_xy_mm=self.config.offset_xy_mm,
                        center_offset_mm=self.config.center_offset_mm,
                        flip_x=self.config.flip_x,
                        flip_y=self.config.flip_y,
                    )

            rule = self.config.rule_for_object(object_key)
            if rule.type == "sphere":
                grasp = compute_grasp_sphere(world_from_object, rule.offset_mm)
            else:
                grasp = compute_grasp(world_from_object, rule.offset_mm)

            lines = [
                f"status: ok",
                f"target: {target['name']}",
                f"object_key: {object_key}",
                f"tag_count: {len(tags)}",
                f"camera pos (mm): {np.asarray(camera_from_object[:3, 3]) * 1000}",
                f"world pos (mm): {np.asarray(world_from_object[:3, 3]) * 1000}",
                f"grasp pos (mm): {np.asarray(grasp[:3, 3]) * 1000}",
            ]
            self.ui_queue.put(("result", "\n".join(lines)))
            self.ui_queue.put(("status", "位姿计算完成"))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.ui_queue.put(("status", f"位姿计算失败: {exc}"))

    def update_ui(self):
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "image":
                    self._show_image(value)
                elif kind == "objects":
                    self.obj_combo["values"] = value
                    if value:
                        self.obj_combo.current(0)
                elif kind == "source":
                    self.source_var.set(value)
                elif kind == "result":
                    self.result_text.delete("1.0", tk.END)
                    self.result_text.insert("1.0", value)
                elif kind == "status":
                    self.status.config(text=value)
        except queue.Empty:
            pass
        self.root.after(50, self.update_ui)

    def _show_image(self, bgr):
        if ImageTk is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        # Keep the image reasonably sized for the UI.
        max_width = 760
        max_height = 620
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        if scale < 1.0:
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.LANCZOS,
            )
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def on_close(self):
        try:
            for estimator in self.pose_estimators.values():
                try:
                    estimator.close()
                except Exception:
                    pass
            if self.sub is not None:
                self.sub.close()
            if self.ctx is not None:
                self.ctx.term()
        except Exception:
            pass
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml",
        help="path to visual_grasp_pipeline.yaml",
    )
    args = parser.parse_args(argv)

    config = VisualGraspConfig.from_yaml(args.config)
    root = tk.Tk()
    VisualGraspUI(root, config)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
