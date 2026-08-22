#!/usr/bin/env python3
"""Original vision-node workflow rebuilt for the active OAK-D-PRO-FF.

This is a single-process replacement for the legacy ``fp_bridge.py`` plus
``vision_node.py`` camera path. It connects to DepthAI directly, reads live
EEPROM intrinsics, consumes hardware-aligned depth, runs YOLO and
FoundationPose, and keeps the original target-sequence UI. Motion is dry-run
by default. The optional ZMQ endpoint is intended only for the legacy
simulated ``arm_node.py`` service.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import yaml

from tool.object_model_builder.camera_source import OakDProSource
from tool.object_model_builder.tag_pose_provider import TagPoseProvider
from tool.object_model_builder.rgbd_geometry import (
    CameraIntrinsics,
    depth_coverage,
    rectified_intrinsics,
    rectify_aligned_depth_image,
    rectify_color_image,
)
from tool.visual_grasp_pipeline.config import VisualGraspConfig
from tool.visual_grasp_pipeline.detection import (
    detect_all_track,
    detect_tags,
    draw_boxes,
)
from tool.visual_grasp_pipeline.foundationpose import FoundationPosePoseEstimator
from tool.visual_grasp_pipeline.geometry import (
    build_world_from_tags,
    compute_grasp,
    compute_grasp_sphere,
    fill_depth_roi,
    to_world_and_compensate,
)
from tool.visual_grasp_pipeline.tracking import StableTracker, parse_sequence

try:
    from PIL import Image, ImageTk
except ImportError as error:  # pragma: no cover - checked by the launcher
    raise RuntimeError("Pillow is required for the OAK vision-node UI") from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISUAL_CONFIG = (
    PROJECT_ROOT / "tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml"
)
DEFAULT_CAMERA_CONFIG = (
    PROJECT_ROOT / "tool/object_model_builder/config/object_model_builder.yaml"
)


@dataclass(frozen=True)
class OakSnapshot:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_s: float
    sync_delta_s: float


@dataclass(frozen=True)
class FoundationPoseInput:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    camera_matrix: np.ndarray
    roi_xyxy: tuple


def prepare_foundationpose_input(
    snapshot: OakSnapshot,
    target,
    mask,
    padding_pixels=24,
    maximum_size=640,
) -> FoundationPoseInput:
    """Crop a YOLO ROI and scale its intrinsics to bound CUDA memory."""

    height, width = snapshot.color_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, target["xyxy"])
    padding = max(0, int(padding_pixels))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("FoundationPose ROI is empty")
    color = np.ascontiguousarray(snapshot.color_bgr[y1:y2, x1:x2])
    depth = np.ascontiguousarray(snapshot.depth_m[y1:y2, x1:x2], dtype=np.float32)
    roi_mask = np.ascontiguousarray(np.asarray(mask[y1:y2, x1:x2]) > 0, dtype=np.uint8)
    matrix = np.asarray(snapshot.intrinsics.matrix, dtype=np.float64).copy()
    matrix[0, 2] -= x1
    matrix[1, 2] -= y1

    limit = max(160, int(maximum_size))
    scale = min(1.0, float(limit) / max(color.shape[:2]))
    if scale < 1.0:
        new_width = max(1, int(round(color.shape[1] * scale)))
        new_height = max(1, int(round(color.shape[0] * scale)))
        color = cv2.resize(color, (new_width, new_height), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        roi_mask = cv2.resize(
            roi_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        matrix[0, 0] *= scale
        matrix[0, 2] *= scale
        matrix[1, 1] *= scale
        matrix[1, 2] *= scale
    return FoundationPoseInput(
        color_bgr=color,
        depth_m=depth,
        mask=roi_mask,
        camera_matrix=matrix,
        roi_xyxy=(x1, y1, x2, y2),
    )


def load_oak_settings(path) -> tuple[dict, float]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    camera = data.get("camera", {})
    if camera.get("backend") != "oak_depthai":
        raise ValueError("camera.backend must be oak_depthai for oak_vision_node")
    oak = dict(camera.get("oak", {}))
    required = ("mxid", "color_width", "color_height", "fps")
    missing = [name for name in required if not str(oak.get(name, "")).strip()]
    if missing:
        raise ValueError("missing OAK settings: {}".format(", ".join(missing)))
    maximum_sync = float(camera.get("maximum_sync_delta_s", 0.03))
    if maximum_sync <= 0.0:
        raise ValueError("camera.maximum_sync_delta_s must be positive")
    return oak, maximum_sync


def build_oak_source(settings: dict) -> OakDProSource:
    return OakDProSource(
        color_width=settings.get("color_width", 1920),
        color_height=settings.get("color_height", 1080),
        fps=settings.get("fps", 10),
        mxid=settings.get("mxid", ""),
        dot_projector_mA=settings.get("dot_projector_mA", 800),
        floodlight_mA=settings.get("floodlight_mA", 0),
        mono_resolution=settings.get("mono_resolution", "800p"),
        extended_disparity=settings.get("extended_disparity", True),
        subpixel=settings.get("subpixel", False),
        left_right_check=settings.get("left_right_check", True),
        focus_mode=settings.get("focus_mode", "device_default"),
        manual_focus=settings.get("manual_focus"),
    )


def build_tag_provider(camera_config_path) -> TagPoseProvider:
    source = Path(camera_config_path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    paths = data.get("paths", {})
    settings = data.get("tag_pose", {})
    layout = paths.get("tag_layout")
    if not layout:
        raise ValueError("paths.tag_layout is required for workspace localization")
    return TagPoseProvider(
        layout,
        minimum_tags=settings.get("minimum_tags", 1),
        maximum_rms_px=settings.get("maximum_rms_px", 2.5),
    )


def find_sequence_target(objects, name: str, instance: Optional[int]):
    matching = [item for item in objects if item.get("name") == name]
    if instance is None:
        return matching[0] if matching else None
    return next(
        (
            item for item in matching
            if int(item.get("id", item.get("seq", -1))) == int(instance)
        ),
        None,
    )


def draw_pose_axes(image, camera_from_object, camera_matrix, length_m=0.06):
    output = np.asarray(image).copy()
    pose = np.asarray(camera_from_object, dtype=np.float64).reshape(4, 4)
    axes = np.asarray(
        [[0.0, 0.0, 0.0], [length_m, 0.0, 0.0],
         [0.0, length_m, 0.0], [0.0, 0.0, length_m]],
        dtype=np.float64,
    )
    points = axes @ pose[:3, :3].T + pose[:3, 3]
    if np.any(points[:, 2] <= 0.0):
        return output
    projected = points @ np.asarray(camera_matrix, dtype=np.float64).T
    pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
    origin = tuple(pixels[0])
    for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(output, origin, tuple(endpoint), color, 3, cv2.LINE_AA)
    return output


class LegacyArmClient:
    """Optional client for the legacy simulated arm_node.py only."""

    def __init__(self, endpoint=""):
        self.endpoint = str(endpoint or "").strip()
        self.context = None
        self.socket = None
        if not self.endpoint:
            return
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 3600 * 1000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.socket.connect(self.endpoint)

    @property
    def enabled(self):
        return self.socket is not None

    def execute(self, grasp):
        if not self.enabled:
            return {"status": "dry_run"}
        self.socket.send_pyobj(np.asarray(grasp, dtype=np.float64))
        return self.socket.recv_pyobj()

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None


class OakVisionNode:
    def __init__(
        self,
        root,
        visual_config: VisualGraspConfig,
        oak_settings: dict,
        maximum_sync_delta_s: float,
        tag_provider: TagPoseProvider,
        arm_service="",
    ):
        self.root = root
        self.config = visual_config
        self.maximum_sync_delta_s = float(maximum_sync_delta_s)
        self.tag_provider = tag_provider
        self.source = build_oak_source(oak_settings)
        self.arm = LegacyArmClient(arm_service)
        self.model = None
        self.tracker = StableTracker(max_miss=10)
        self.active_estimator = None
        self.active_estimator_key = None
        self.objects = []
        self.sequence_items = []
        self.latest_snapshot = None
        self.busy = False
        self.closed = False
        self.ui_queue = queue.Queue()

        root.title("OAK 视觉抓取节点（原版兼容 · 默认 Dry-run）")
        root.geometry("1420x840")
        root.minsize(1080, 680)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        root.after(50, self._update_ui)
        self._set_busy(True, "正在连接 OAK-D-PRO-FF……")
        threading.Thread(target=self._connect_camera, daemon=True).start()

    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill="x")
        self.capture_button = ttk.Button(
            toolbar, text="拍照识别", command=self.on_capture
        )
        self.capture_button.pack(side="left")
        self.object_combo = ttk.Combobox(toolbar, state="readonly", width=28)
        self.object_combo.pack(side="left", padx=8)
        ttk.Button(toolbar, text="＋加入序列", command=self.on_add).pack(side="left")
        ttk.Button(toolbar, text="清空序列", command=self.on_clear).pack(
            side="left", padx=8
        )

        sequence_row = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        sequence_row.pack(fill="x")
        ttk.Label(sequence_row, text="目标序列：").pack(side="left")
        self.sequence_entry = ttk.Entry(sequence_row, width=56)
        self.sequence_entry.pack(side="left", fill="x", expand=True, padx=6)
        button_text = (
            "开始序列（旧模拟机械臂）" if self.arm.enabled
            else "开始算法序列（Dry-run，不运动）"
        )
        self.start_button = ttk.Button(
            sequence_row, text=button_text, command=self.on_start
        )
        self.start_button.pack(side="left")

        content = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        content.pack(fill="both", expand=True)
        self.image_label = tk.Label(content, bg="#11181c")
        self.image_label.pack(side="left", fill="both", expand=True)
        result_frame = ttk.Frame(content, width=410)
        result_frame.pack(side="right", fill="y", padx=(10, 0))
        ttk.Label(result_frame, text="算法结果").pack(anchor="w")
        self.result_text = tk.Text(
            result_frame, width=48, height=32, wrap="word",
            background="#20272b", foreground="#e4ecef",
        )
        self.result_text.pack(fill="both", expand=True, pady=(5, 0))
        self.result_text.insert(
            "end",
            "相机：OAK-D-PRO-FF\n"
            "流程：YOLO → FoundationPose → AprilTag/相机系 → 抓取位姿\n"
            "安全：默认 Dry-run，不会发送真实机械臂运动。\n",
        )

        self.status = ttk.Label(
            self.root, text="准备启动", anchor="w", padding=(10, 5, 10, 10)
        )
        self.status.pack(fill="x")

    def _connect_camera(self):
        try:
            self.source.start()
            snapshot = self._grab_snapshot(timeout_s=12.0)
            self.latest_snapshot = snapshot
            self.ui_queue.put(("image", snapshot.color_bgr))
            self.ui_queue.put(("status", "OAK 已连接：1920×1080 RGB-D，点击“拍照识别”"))
        except Exception as error:
            self.ui_queue.put(("error", ("OAK 启动失败", str(error))))
        finally:
            self.ui_queue.put(("busy", False))

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.config.yolo_weights)
        return self.model

    def _grab_snapshot(self, timeout_s=5.0):
        deadline = time.monotonic() + float(timeout_s)
        bundle = None
        while time.monotonic() < deadline and not self.closed:
            bundle = self.source.latest()
            if bundle is not None and bundle.depth_m is not None:
                if bundle.sync_delta_s is not None and (
                    bundle.sync_delta_s <= self.maximum_sync_delta_s
                ):
                    break
            time.sleep(0.01)
        if bundle is None or bundle.depth_m is None:
            raise RuntimeError("等待 OAK RGB-D 帧超时")
        if bundle.sync_delta_s is None or bundle.sync_delta_s > self.maximum_sync_delta_s:
            raise RuntimeError(
                "OAK RGB-D 时间差 {:.1f} ms 超过 {:.1f} ms".format(
                    -1.0 if bundle.sync_delta_s is None else bundle.sync_delta_s * 1000.0,
                    self.maximum_sync_delta_s * 1000.0,
                )
            )
        intrinsics = bundle.color_intrinsics
        if intrinsics is None:
            raise RuntimeError("OAK EEPROM RGB 内参不可用")
        color = rectify_color_image(bundle.color_bgr, intrinsics)
        depth = rectify_aligned_depth_image(bundle.depth_m, intrinsics)
        return OakSnapshot(
            color_bgr=color,
            depth_m=depth,
            intrinsics=rectified_intrinsics(intrinsics),
            timestamp_s=float(bundle.color_timestamp_s),
            sync_delta_s=float(bundle.sync_delta_s),
        )

    def _detect(self, snapshot):
        return detect_all_track(
            snapshot.color_bgr,
            self._load_model(),
            self.tracker,
            conf=self.config.yolo_conf,
            imgsz=self.config.yolo_imgsz,
        )

    def on_capture(self):
        if self.busy:
            return
        self._set_busy(True, "正在拍照并运行 YOLO……")
        threading.Thread(target=self._capture_worker, daemon=True).start()

    def _capture_worker(self):
        try:
            snapshot = self._grab_snapshot()
            objects = self._detect(snapshot)
            self.latest_snapshot = snapshot
            self.objects = objects
            self.ui_queue.put(("objects", objects))
            self.ui_queue.put(("image", draw_boxes(snapshot.color_bgr, objects)))
            self.ui_queue.put((
                "status",
                "识别到 {} 个物体；RGB-D 时间差 {:.2f} ms".format(
                    len(objects), snapshot.sync_delta_s * 1000.0
                ),
            ))
        except Exception as error:
            self.ui_queue.put(("error", ("拍照识别失败", str(error))))
        finally:
            self.ui_queue.put(("busy", False))

    def on_add(self):
        index = self.object_combo.current()
        if index < 0 or index >= len(self.objects):
            self.status.configure(text="请先拍照识别并选择物体")
            return
        item = self.objects[index]
        self.sequence_items.append("{}#{}".format(item["name"], item.get("id")))
        self.sequence_entry.delete(0, tk.END)
        self.sequence_entry.insert(0, ", ".join(self.sequence_items))

    def on_clear(self):
        self.sequence_items = []
        self.sequence_entry.delete(0, tk.END)

    def on_start(self):
        if self.busy:
            return
        try:
            sequence = parse_sequence(self.sequence_entry.get())
        except Exception as error:
            messagebox.showerror("序列格式错误", str(error), parent=self.root)
            return
        if not sequence:
            messagebox.showerror(
                "序列为空", "示例：can#1, green_apple#2", parent=self.root
            )
            return
        self._set_busy(True, "开始运行目标序列……")
        threading.Thread(
            target=self._sequence_worker, args=(sequence,), daemon=True
        ).start()

    def _estimator(self, object_key, mesh_path):
        scale = self.config.mesh_scale_for_object(object_key)
        key = (str(mesh_path), float(scale))
        if self.active_estimator is None or self.active_estimator_key != key:
            if self.active_estimator is not None:
                self.active_estimator.close()
            self.active_estimator = FoundationPosePoseEstimator(
                foundationpose_root=self.config.foundationpose_root,
                mesh_path=mesh_path,
                mesh_scale_to_meters=scale,
                debug_dir=self.config.debug_dir,
                est_refine_iter=self.config.est_refine_iter,
                track_refine_iter=self.config.track_refine_iter,
                device=self.config.device,
                use_mask_center_guidance=self.config.use_mask_center_guidance,
                registration_max_hypotheses=(
                    self.config.foundationpose_registration_hypotheses
                ),
            )
            self.active_estimator_key = key
        return self.active_estimator

    def _release_yolo_cuda(self):
        """Keep YOLO available on CPU while reserving CUDA for FoundationPose."""

        if self.model is None:
            return
        try:
            self.model.to("cpu")
            # Ultralytics caches an AutoBackend on the predictor. Rebuilding
            # it later avoids retaining CUDA buffers after moving the model.
            self.model.predictor = None
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _process_target(self, name, instance):
        snapshot = self._grab_snapshot()
        objects = self._detect(snapshot)
        target = find_sequence_target(objects, name, instance)
        if target is None:
            raise RuntimeError("目标 {}#{} 不在当前画面".format(name, instance or "*"))
        object_key = self.config.resolve_object_key(target["name"], target["cls"])
        mesh_path = self.config.mesh_for_object(object_key)
        if not mesh_path or not Path(mesh_path).is_file():
            raise RuntimeError("物体 {} 没有可用 CAD 网格".format(object_key))
        mask = target.get("mask")
        if mask is None or not np.any(mask):
            mask = np.zeros(snapshot.depth_m.shape, dtype=np.uint8)
            x1, y1, x2, y2 = target["xyxy"]
            mask[y1:y2, x1:x2] = 255
        coverage = depth_coverage(snapshot.depth_m, mask)
        valid_depth = int(np.count_nonzero((snapshot.depth_m > 0.0) & (mask > 0)))
        if valid_depth < 30:
            raise RuntimeError("目标 Mask 内有效深度不足：{} 点".format(valid_depth))

        fp_input = prepare_foundationpose_input(
            snapshot,
            target,
            mask,
            padding_pixels=self.config.foundationpose_roi_padding_pixels,
            maximum_size=self.config.foundationpose_max_input_size,
        )
        self._release_yolo_cuda()
        estimator = self._estimator(object_key, mesh_path)
        try:
            camera_from_object = estimator.register(
                fp_input.color_bgr,
                fill_depth_roi(fp_input.depth_m, fp_input.mask),
                fp_input.mask,
                fp_input.camera_matrix,
            )
        except Exception as error:
            if "out of memory" in str(error).lower():
                estimator.close()
                self.active_estimator = None
                self.active_estimator_key = None
                raise RuntimeError(
                    "FoundationPose 显存不足；已释放运行时，请关闭其他 CUDA 程序后重试"
                ) from error
            raise
        distance = float(camera_from_object[2, 3])
        if not 0.10 < distance < 3.0:
            raise RuntimeError("FoundationPose 距离异常：{:.3f} m".format(distance))

        tag_estimate, mapped_detections = self.tag_provider.estimate(
            snapshot.color_bgr,
            snapshot.intrinsics.matrix,
            snapshot.intrinsics.distortion,
        )
        overlay_base = self.tag_provider.draw_status(
            snapshot.color_bgr, mapped_detections, tag_estimate
        )
        tag_ids = sorted(int(tag_id) for tag_id in mapped_detections)
        workspace_valid = bool(tag_estimate.valid)
        if workspace_valid:
            workspace_from_object = (
                np.asarray(tag_estimate.workspace_from_camera, dtype=np.float64)
                @ camera_from_object
            )
            pose_frame = str(
                self.tag_provider.layout.get("workspace_frame", "AprilTag 工作台")
            )
        else:
            # Compatibility fallback for the original single tag0 workspace.
            legacy_tags = detect_tags(
                snapshot.color_bgr,
                self.config.tag_size_mm,
                snapshot.intrinsics.matrix,
                snapshot.intrinsics.distortion.reshape(-1, 1),
            )
            tag_world = build_world_from_tags(legacy_tags)
            if tag_world is not None:
                workspace_valid = True
                workspace_from_object = to_world_and_compensate(
                    camera_from_object,
                    np.linalg.inv(tag_world),
                    offset_xy_mm=self.config.offset_xy_mm,
                    center_offset_mm=self.config.center_offset_mm,
                    flip_x=self.config.flip_x,
                    flip_y=self.config.flip_y,
                )
                pose_frame = "legacy tag0 工作台"
                tag_ids = [int(tag_id) for tag_id, _ in legacy_tags]
            else:
            # Useful for camera/algorithm verification, but never eligible for
            # an arm request because it is not expressed in a robot/workspace frame.
                workspace_from_object = camera_from_object.copy()
                pose_frame = "相机光学系（无有效 Tag 地图，禁止执行）"

        rule = self.config.rule_for_object(object_key)
        grasp = (
            compute_grasp_sphere(workspace_from_object, rule.offset_mm)
            if rule.type == "sphere"
            else compute_grasp(workspace_from_object, rule.offset_mm)
        )
        Path(self.config.pose_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.grasp_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        np.save(self.config.pose_file, workspace_from_object)
        np.save(self.config.grasp_file, grasp)

        overlay = draw_boxes(overlay_base, objects)
        overlay = draw_pose_axes(
            overlay, camera_from_object, snapshot.intrinsics.matrix
        )
        return {
            "target": target,
            "object_key": object_key,
            "camera_from_object": camera_from_object,
            "workspace_from_object": workspace_from_object,
            "grasp": grasp,
            "pose_frame": pose_frame,
            "workspace_valid": workspace_valid,
            "tag_ids": tag_ids,
            "depth_coverage": coverage,
            "valid_depth": valid_depth,
            "overlay": overlay,
        }

    def _sequence_worker(self, sequence):
        try:
            for name, instance in sequence:
                self.ui_queue.put((
                    "status", "计算目标 {}#{}……".format(name, instance or "*")
                ))
                result = self._process_target(name, instance)
                self.ui_queue.put(("image", result["overlay"]))
                camera_xyz = result["camera_from_object"][:3, 3] * 1000.0
                pose_xyz = result["workspace_from_object"][:3, 3] * 1000.0
                grasp_xyz = result["grasp"][:3, 3] * 1000.0
                text = (
                    "target: {name} #{instance}\n"
                    "object_key: {object_key}\n"
                    "pose_frame: {pose_frame}\n"
                    "tags: {tags}\n"
                    "depth coverage: {coverage:.1%} ({points} points)\n"
                    "camera XYZ mm: {camera}\n"
                    "pose XYZ mm: {pose}\n"
                    "grasp XYZ mm: {grasp}\n"
                ).format(
                    name=result["target"]["name"],
                    instance=result["target"].get("id"),
                    object_key=result["object_key"],
                    pose_frame=result["pose_frame"],
                    tags=result["tag_ids"],
                    coverage=result["depth_coverage"],
                    points=result["valid_depth"],
                    camera=np.round(camera_xyz, 2).tolist(),
                    pose=np.round(pose_xyz, 2).tolist(),
                    grasp=np.round(grasp_xyz, 2).tolist(),
                )
                if self.arm.enabled:
                    if not result["workspace_valid"]:
                        text += "arm: BLOCKED（缺少工作台坐标）\n"
                    else:
                        response = self.arm.execute(result["grasp"])
                        text += "arm simulator: {}\n".format(response)
                else:
                    text += "arm: DRY-RUN（未发送运动）\n"
                self.ui_queue.put(("result", text))
            self.ui_queue.put(("status", "目标序列算法运行完成"))
        except Exception as error:
            self.ui_queue.put(("error", ("序列运行失败", str(error))))
        finally:
            self.ui_queue.put(("busy", False))

    def _set_busy(self, busy, status=None):
        self.busy = bool(busy)
        state = "disabled" if self.busy else "normal"
        self.capture_button.configure(state=state)
        self.start_button.configure(state=state)
        if status is not None:
            self.status.configure(text=str(status))

    def _show_image(self, frame):
        rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_width, max_height = 920, 690
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        if scale < 1.0:
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)), Image.LANCZOS
            )
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def _update_ui(self):
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "image":
                    self._show_image(value)
                elif kind == "objects":
                    self.objects = list(value)
                    labels = [
                        "{} #{}  {:.3f}".format(
                            item["name"], item.get("id"), item["conf"]
                        )
                        for item in self.objects
                    ]
                    self.object_combo["values"] = labels
                    if labels:
                        self.object_combo.current(0)
                elif kind == "result":
                    self.result_text.delete("1.0", tk.END)
                    self.result_text.insert("1.0", value)
                elif kind == "status":
                    self.status.configure(text=str(value))
                elif kind == "busy":
                    self._set_busy(bool(value))
                elif kind == "error":
                    title, message = value
                    self.status.configure(text=message)
                    messagebox.showerror(title, message, parent=self.root)
        except queue.Empty:
            pass
        if not self.closed:
            self.root.after(50, self._update_ui)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.source.stop()
        finally:
            if self.active_estimator is not None:
                try:
                    self.active_estimator.close()
                except Exception:
                    pass
                self.active_estimator = None
            self.arm.close()
            self.root.destroy()


def camera_check(visual_config, oak_settings, maximum_sync_delta_s):
    source = build_oak_source(oak_settings)
    try:
        source.start()
        deadline = time.monotonic() + 12.0
        bundle = None
        while time.monotonic() < deadline:
            bundle = source.latest()
            if bundle is not None and bundle.depth_m is not None:
                break
            time.sleep(0.01)
        if bundle is None or bundle.depth_m is None:
            raise RuntimeError("OAK RGB-D frame timeout")
        if bundle.sync_delta_s > maximum_sync_delta_s:
            raise RuntimeError("OAK RGB-D synchronization failed")
        from ultralytics import YOLO

        objects = detect_all_track(
            bundle.color_bgr,
            YOLO(visual_config.yolo_weights),
            StableTracker(max_miss=0),
            conf=visual_config.yolo_conf,
            imgsz=visual_config.yolo_imgsz,
        )
        return {
            "status": "ok",
            "mxid": oak_settings["mxid"],
            "rgb_shape": list(bundle.color_bgr.shape),
            "depth_shape": list(bundle.depth_m.shape),
            "sync_delta_ms": round(bundle.sync_delta_s * 1000.0, 3),
            "camera_matrix": bundle.color_intrinsics.matrix.tolist(),
            "detections": [
                {
                    "name": item["name"],
                    "confidence": item["conf"],
                    "bbox": list(item["xyxy"]),
                }
                for item in objects
            ],
        }
    finally:
        source.stop()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_VISUAL_CONFIG))
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument(
        "--legacy-arm-service",
        default="",
        help="optional legacy simulated arm_node ZMQ endpoint, e.g. tcp://127.0.0.1:5556",
    )
    parser.add_argument(
        "--camera-check",
        action="store_true",
        help="capture one OAK frame, run YOLO, print JSON and exit without a UI",
    )
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    visual_config = VisualGraspConfig.from_yaml(arguments.config)
    oak_settings, maximum_sync = load_oak_settings(arguments.camera_config)
    tag_provider = build_tag_provider(arguments.camera_config)
    if arguments.camera_check:
        print(json.dumps(
            camera_check(visual_config, oak_settings, maximum_sync),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    root = tk.Tk()
    node = OakVisionNode(
        root,
        visual_config,
        oak_settings,
        maximum_sync,
        tag_provider,
        arm_service=arguments.legacy_arm_service,
    )
    signal.signal(signal.SIGINT, lambda *_args: root.after(0, node.close))
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
