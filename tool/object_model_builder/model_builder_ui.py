#!/usr/bin/env python3
"""Single-entry desktop UI for RGB-D capture and FoundationPose mesh export."""

import argparse
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .camera_source import AstraRosSource, FrameBundle, OakDProSource
from .capture_session import CaptureSession
from .environment_check import run_checks
from .foundationpose_export import export_foundationpose_model, validate_mesh
from .foundationpose_live import (
    FoundationPoseLiveConfig,
    FoundationPoseLiveFrame,
    FoundationPoseLiveWorker,
    draw_pose_overlay,
)
from .mesh_fusion import fuse_session
from .rgbd_calibration import (
    CalibrationTarget,
    StereoCalibrationResult,
    calibration_target_from_mapping,
    calibrate_color_from_depth,
    calibration_view_signature,
    calibration_view_change,
    detect_calibration_target,
    infrared_to_uint8,
    load_image_pairs,
    update_runtime_calibration,
)
from .rgbd_geometry import (
    CameraIntrinsics,
    DepthToColorAligner,
    RgbdCalibration,
    depth_coverage,
    load_runtime_calibration,
    masked_depth_centroid,
    rectify_aligned_depth_image,
    rectified_intrinsics,
    rectify_color_image,
    save_rgbd_calibration,
    transform_points,
)
from .tag_pose_provider import TagPoseProvider
from .yolo_segmenter import MaskResult, YoloMaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "object_model_builder.yaml"


def load_config(path: str) -> dict:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported object model builder config schema")
    return config


def load_color_intrinsics(path: str) -> CameraIntrinsics:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as handle:
        return CameraIntrinsics.from_mapping(yaml.safe_load(handle) or {})


def pose_difference(first: np.ndarray, second: np.ndarray):
    relative = np.linalg.inv(first) @ second
    translation = float(np.linalg.norm(relative[:3, 3]))
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    rotation_deg = math.degrees(math.acos(cosine))
    return translation, rotation_deg


@dataclass
class CaptureAnalysisResult:
    generation: int
    bundle: FrameBundle
    rectified_color: np.ndarray
    tag_estimate: object = None
    tag_detections: Optional[dict] = None
    mask_result: Optional[MaskResult] = None
    aligned_depth: Optional[np.ndarray] = None
    depth_preview_bgr: Optional[np.ndarray] = None
    aligned_preview_bgr: Optional[np.ndarray] = None
    device_calibration: Optional[RgbdCalibration] = None
    completed_monotonic_s: float = 0.0
    error: Optional[Exception] = None


class ModelBuilderApp:
    def __init__(self, root, config_path: str):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self._initialize_backend(config_path)

        root.title("物体三维模型工作台")
        root.geometry("1460x900")
        root.minsize(1180, 760)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()
        self._build_ui()
        self._reload_rgbd_calibration(show_dialog=False)
        self._run_environment_check()
        root.after(80, self._tick)

    def _initialize_backend(self, config_path: str):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = load_config(str(self.config_path))
        self.paths = self.config["paths"]
        self.camera_config = self.config["camera"]
        target_mapping = self.config.get(
            "rgbd_calibration_target",
            self.camera_config.get("rgbd_calibration_target", {}),
        )
        self.calibration_target: CalibrationTarget = calibration_target_from_mapping(
            target_mapping
        )
        self.segmentation_config = self.config["segmentation"]
        self.tag_config = self.config["tag_pose"]
        self.capture_config = self.config["capture"]
        self.fusion_config = self.config["fusion"]
        self.foundationpose_live_config = self.config.get("foundationpose_live", {})
        self.source: Optional[AstraRosSource] = None
        self.bundle: Optional[FrameBundle] = None
        self.analysis_bundle: Optional[FrameBundle] = None
        self.analysis_rectified_color: Optional[np.ndarray] = None
        self.analysis_completed_monotonic_s = 0.0
        self.color_intrinsics = load_color_intrinsics(self.paths["color_intrinsics"])
        self.rgbd_calibration: Optional[RgbdCalibration] = None
        self.depth_aligner: Optional[DepthToColorAligner] = None
        self.device_rgbd_calibration: Optional[RgbdCalibration] = None
        self.stereo_result: Optional[StereoCalibrationResult] = None
        self.tag_provider = TagPoseProvider(
            self.paths["tag_layout"],
            minimum_tags=self.tag_config["minimum_tags"],
            maximum_rms_px=self.tag_config["maximum_rms_px"],
        )
        self.tag_estimate = None
        self.tag_detections = {}
        self.yolo_provider: Optional[YoloMaskProvider] = None
        self.mask_result = MaskResult(False, reason="YOLO weights not loaded")
        self.aligned_depth = None
        self.rectified_color = None
        self.capture_session: Optional[CaptureSession] = None
        self.last_capture_pose = None
        self._captured_object_centroids_workspace = []
        self.fusion_result = None
        self._frame_counter = 0
        self._last_processed_anchor_timestamp_s = None
        self._last_displayed_depth_timestamp_s = None
        self._last_displayed_ir_timestamp_s = None
        self._last_ir_detection_monotonic_s = 0.0
        self._ir_target_timestamp_s = None
        self._busy = False
        self._preview_refs = {}
        self._calibration_pair_root = None
        self._active_stage_index = 0
        self._ir_target_count = 0
        self._ir_target_corners = np.empty((0, 2), dtype=np.float32)
        self._ir_target_ids = np.empty((0,), dtype=np.int32)
        # Keep legacy names for external scripts that inspect the preview state.
        self._ir_charuco_count = 0
        self._ir_charuco_corners = self._ir_target_corners
        self._ir_charuco_ids = self._ir_target_ids
        auto_config = self.camera_config.get("rgbd_auto_capture", {})
        self._auto_calibration_active = False
        self._auto_calibration_target_pairs = int(auto_config.get("target_pairs", 20))
        default_minimum_common = (
            min(40, self.calibration_target.point_count)
            if self.calibration_target.target_type == "checkerboard"
            else 12
        )
        self._auto_calibration_minimum_common = int(
            auto_config.get("minimum_common_corners", default_minimum_common)
        )
        self._auto_calibration_minimum_interval_s = float(
            auto_config.get("minimum_interval_s", 0.8)
        )
        self._auto_calibration_minimum_view_change = float(
            auto_config.get("minimum_view_change_normalized", 0.025)
        )
        self._auto_calibration_last_capture_s = 0.0
        self._auto_calibration_last_signature = None
        self._auto_calibration_last_attempt_s = 0.0
        self._preview_interval_ms = max(
            16, int(self.camera_config.get("preview_interval_ms", 33))
        )
        self._maximum_analysis_age_s = float(
            self.capture_config.get("maximum_analysis_age_s", 0.75)
        )
        self._yolo_preview_interval_s = float(
            self.segmentation_config.get("preview_interval_s", 0.20)
        )
        self._analysis_lock = threading.Lock()
        self._analysis_pending = None
        self._analysis_results = deque(maxlen=1)
        self._analysis_worker_active = False
        self._analysis_shutdown = False
        self._analysis_generation = 0
        self._analysis_cache_generation = None
        self._analysis_cached_depth_timestamp_s = None
        self._analysis_cached_aligned_depth = None
        self._analysis_cached_depth_preview_bgr = None
        self._analysis_cached_aligned_preview_bgr = None
        self._analysis_cached_device_calibration = None
        self._analysis_cached_mask_result = MaskResult(
            False, reason="等待后台分割"
        )
        self._analysis_last_yolo_monotonic_s = 0.0
        self.foundationpose_live_worker = FoundationPoseLiveWorker()
        self.foundationpose_live_active = False
        self.foundationpose_live_pose = None
        self.foundationpose_live_bounds_m = None
        self.foundationpose_live_mode = "stopped"
        self.foundationpose_live_inference_ms = 0.0
        self.foundationpose_live_timestamp_s = None
        self._foundationpose_live_last_frame_id = None
        self._capture_request_pending = False
        self._capture_request_deadline_s = 0.0
        self._capture_request_last_reason = ""
        self._capture_feedback_text = ""
        self._capture_feedback_until_s = 0.0
        self._last_captured_view_index = None
        self._view_mesh_after_fusion = False

    def _configure_style(self):
        from tkinter import font as tkfont

        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(name).configure(family="Microsoft YaHei", size=10)
            except self.tk.TclError:
                pass
        try:
            tkfont.nametofont("TkFixedFont").configure(
                family="Noto Sans Mono CJK SC", size=9
            )
        except self.tk.TclError:
            pass

        self.root.configure(background="#f4f6f7")
        style = self.ttk.Style()
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        style.configure("TFrame", background="#ffffff")
        style.configure("Workspace.TFrame", background="#f4f6f7")
        style.configure("Preview.TFrame", background="#14191d")
        style.configure("Metric.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure(
            "TLabel",
            background="#ffffff",
            foreground="#20282d",
            font=("Microsoft YaHei", 10),
        )
        style.configure("Workspace.TLabel", background="#f4f6f7", foreground="#20282d")
        style.configure(
            "ViewTitle.TLabel",
            background="#f4f6f7",
            foreground="#20282d",
            font=("Microsoft YaHei", 15, "bold"),
        )
        style.configure(
            "ViewState.TLabel",
            background="#f4f6f7",
            foreground="#0f766e",
            font=("Microsoft YaHei", 10, "bold"),
        )
        style.configure(
            "PanelTitle.TLabel",
            background="#ffffff",
            foreground="#20282d",
            font=("Microsoft YaHei", 15, "bold"),
        )
        style.configure("Muted.TLabel", foreground="#66737a")
        style.configure(
            "Section.TLabel",
            foreground="#647078",
            font=("Microsoft YaHei", 9, "bold"),
        )
        style.configure(
            "Result.TLabel",
            background="#eef5f3",
            foreground="#285c55",
            padding=(10, 9),
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Preview.TLabel",
            background="#14191d",
            foreground="#aab3ba",
            font=("Microsoft YaHei", 11),
        )
        style.configure(
            "PreviewTitle.TLabel",
            background="#20272b",
            foreground="#d7dee1",
            padding=(10, 7),
            font=("Microsoft YaHei", 9, "bold"),
        )
        style.configure(
            "MetricCaption.TLabel",
            background="#ffffff",
            foreground="#707b81",
            font=("Microsoft YaHei", 8),
        )
        style.configure(
            "MetricValue.TLabel",
            background="#ffffff",
            foreground="#20282d",
            font=("Microsoft YaHei", 11, "bold"),
        )
        style.configure(
            "Status.TLabel",
            padding=(12, 6),
            background="#eef1f2",
            foreground="#526068",
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TButton",
            background="#ffffff",
            foreground="#20282d",
            bordercolor="#cbd3d6",
            lightcolor="#ffffff",
            darkcolor="#cbd3d6",
            padding=(12, 7),
            font=("Microsoft YaHei", 10),
        )
        style.map(
            "TButton",
            background=[("active", "#f0f3f4"), ("pressed", "#e5e9eb")],
        )
        style.configure(
            "Primary.TButton",
            background="#176d64",
            foreground="#ffffff",
            bordercolor="#176d64",
            lightcolor="#176d64",
            darkcolor="#176d64",
            padding=(12, 8),
            font=("Microsoft YaHei", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#115c54"), ("pressed", "#0d5049")],
            foreground=[("disabled", "#eef3f4")],
        )
        style.configure("Accent.TButton", **style.configure("Primary.TButton"))
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground="#20282d",
            bordercolor="#cbd3d6",
            padding=(7, 6),
        )

    def _build_ui(self):
        self.connection_text = self.tk.StringVar(value="相机未连接")
        self.preview_state_text = self.tk.StringVar(value="待机")
        self.status_text = self.tk.StringVar(value="等待操作")

        header = self.tk.Frame(self.root, background="#20272b", height=66)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_group = self.tk.Frame(header, background="#20272b")
        title_group.pack(side="left", padx=(24, 0), pady=(9, 7))
        self.tk.Label(
            title_group,
            text="物体三维模型工作台",
            background="#20272b",
            foreground="#ffffff",
            font=("Microsoft YaHei", 15, "bold"),
            anchor="w",
        ).pack(anchor="w")
        self.tk.Label(
            title_group,
            text="Astra Pro · AprilTag · YOLO Mask · TSDF · FoundationPose",
            background="#20272b",
            foreground="#aeb8bd",
            font=("Microsoft YaHei", 9),
            anchor="w",
        ).pack(anchor="w")
        self.connect_button = self.ttk.Button(
            header,
            text="连接相机",
            style="Primary.TButton",
            command=self._toggle_camera,
        )
        self.connect_button.pack(side="right", padx=(10, 20), pady=14)
        self.tk.Label(
            header,
            textvariable=self.connection_text,
            background="#323b40",
            foreground="#d7dde0",
            highlightbackground="#465158",
            highlightthickness=1,
            padx=11,
            pady=6,
            font=("Microsoft YaHei", 9),
        ).pack(side="right", pady=17)

        self.ttk.Label(
            self.root,
            textvariable=self.status_text,
            style="Status.TLabel",
            anchor="w",
        ).pack(side="bottom", fill="x")

        body = self.tk.Frame(self.root, background="#f4f6f7")
        body.pack(fill="both", expand=True)
        sidebar = self.tk.Frame(
            body,
            background="#edf0f1",
            width=238,
            highlightbackground="#d6dcdf",
            highlightthickness=1,
        )
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        controls = self.tk.Frame(
            body,
            background="#ffffff",
            width=440,
            highlightbackground="#dce1e3",
            highlightthickness=1,
        )
        controls.pack(side="right", fill="y")
        controls.pack_propagate(False)
        preview = self.ttk.Frame(body, style="Workspace.TFrame", padding=(20, 18))
        preview.pack(side="left", fill="both", expand=True)

        self._build_sidebar(sidebar)
        self._build_preview(preview)
        self._build_controls(controls)

    def _build_sidebar(self, parent):
        self.tk.Label(
            parent,
            text="建模流程",
            background="#edf0f1",
            foreground="#647078",
            font=("Microsoft YaHei", 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(20, 10))
        self.stage_buttons = []
        for index, text in enumerate(
            (
                "01  环境与相机",
                "02  RGB-D 标定",
                "03  无模型拍照",
                "04  重建与导出",
            )
        ):
            button = self.tk.Button(
                parent,
                text=text,
                command=lambda current=index: self._change_stage(current),
                anchor="w",
                relief="flat",
                borderwidth=0,
                highlightthickness=1,
                padx=12,
                pady=10,
                background="#edf0f1",
                foreground="#435057",
                activebackground="#e2e7e9",
                activeforeground="#0b645c",
                font=("Microsoft YaHei", 10),
                cursor="hand2",
            )
            button.pack(fill="x", padx=16, pady=2)
            self.stage_buttons.append(button)

        self.tk.Label(
            parent,
            text="图像源",
            background="#edf0f1",
            foreground="#647078",
            font=("Microsoft YaHei", 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(22, 8))
        backend = self.camera_config.get("backend", "astra_ros")
        source_name = "Astra Pro · ROS" if backend == "astra_ros" else "OAK-D Pro · DepthAI"
        self.tk.Label(
            parent,
            text=source_name,
            background="#ffffff",
            foreground="#243139",
            highlightbackground="#cfd6d9",
            highlightthickness=1,
            padx=10,
            pady=8,
            font=("Microsoft YaHei", 9),
        ).pack(fill="x", padx=16)

        note = self.tk.Label(
            parent,
            text="工作坐标：ruler_workspace\n输出：FoundationPose CAD 模型",
            background="#e2e7e9",
            foreground="#66737a",
            justify="left",
            anchor="w",
            wraplength=186,
            padx=10,
            pady=10,
            font=("Microsoft YaHei", 9),
        )
        note.pack(side="bottom", fill="x", padx=16, pady=18)

    def _build_preview(self, parent):
        heading = self.ttk.Frame(parent, style="Workspace.TFrame")
        heading.pack(fill="x", pady=(0, 12))
        self.ttk.Label(
            heading, text="RGB-D 实时预览", style="ViewTitle.TLabel"
        ).pack(side="left")
        self.ttk.Label(
            heading, textvariable=self.preview_state_text, style="ViewState.TLabel"
        ).pack(side="right", pady=(4, 0))

        color_group = self._preview_panel(parent, "RGB / Tag / 分割结果")
        color_group.pack(fill="both", expand=True)
        self.color_preview = self.ttk.Label(
            color_group, text="相机未连接", anchor="center", style="Preview.TLabel"
        )
        self.color_preview.pack(fill="both", expand=True)

        bottom = self.ttk.Frame(parent, style="Workspace.TFrame")
        bottom.pack(fill="both", expand=True, pady=(10, 0))
        raw_group = self._preview_panel(bottom, "原始深度")
        ir_group = self._preview_panel(
            bottom, "红外原图 / {}".format(self.calibration_target.display_name)
        )
        aligned_group = self._preview_panel(bottom, "对齐深度 / Mask")
        raw_group.pack(side="left", fill="both", expand=True, padx=(0, 4))
        ir_group.pack(side="left", fill="both", expand=True, padx=4)
        aligned_group.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.depth_preview = self.ttk.Label(
            raw_group, text="等待深度图像", anchor="center", style="Preview.TLabel"
        )
        self.depth_preview.pack(fill="both", expand=True)
        self.ir_preview = self.ttk.Label(
            ir_group, text="等待红外图像", anchor="center", style="Preview.TLabel"
        )
        self.ir_preview.pack(fill="both", expand=True)
        self.aligned_preview = self.ttk.Label(
            aligned_group, text="等待彩深对齐", anchor="center", style="Preview.TLabel"
        )
        self.aligned_preview.pack(fill="both", expand=True)

        self.frame_metric_text = self.tk.StringVar(value="--")
        self.detection_metric_text = self.tk.StringVar(value="待机")
        self.capture_metric_text = self.tk.StringVar(value="0 帧")
        metrics = self.ttk.Frame(parent, style="Metric.TFrame", padding=(18, 9))
        metrics.pack(fill="x", pady=(10, 0))
        self._build_metric(metrics, "图像", self.frame_metric_text).pack(
            side="left", fill="x", expand=True
        )
        self.ttk.Separator(metrics, orient="vertical").pack(side="left", fill="y", padx=12)
        self._build_metric(metrics, "检测", self.detection_metric_text).pack(
            side="left", fill="x", expand=True
        )
        self.ttk.Separator(metrics, orient="vertical").pack(side="left", fill="y", padx=12)
        self._build_metric(metrics, "采集", self.capture_metric_text).pack(
            side="left", fill="x", expand=True
        )

    def _preview_panel(self, parent, title):
        panel = self.ttk.Frame(parent, style="Preview.TFrame")
        self.ttk.Label(panel, text=title, style="PreviewTitle.TLabel").pack(fill="x")
        return panel

    def _build_metric(self, parent, caption, variable):
        metric = self.ttk.Frame(parent)
        self.ttk.Label(metric, text=caption, style="MetricCaption.TLabel").pack(anchor="w")
        self.ttk.Label(metric, textvariable=variable, style="MetricValue.TLabel").pack(anchor="w")
        return metric

    def _build_controls(self, parent):
        host = self.tk.Frame(parent, background="#ffffff")
        host.pack(fill="both", expand=True, padx=18, pady=(18, 14))
        host.grid_rowconfigure(0, weight=1)
        host.grid_columnconfigure(0, weight=1)
        environment = self.ttk.Frame(host)
        calibration = self.ttk.Frame(host)
        capture = self.ttk.Frame(host)
        reconstruction = self.ttk.Frame(host)
        self.control_pages = (environment, calibration, capture, reconstruction)
        for page in self.control_pages:
            page.grid(row=0, column=0, sticky="nsew")
        self._build_environment_tab(environment)
        self._build_calibration_tab(calibration)
        self._build_capture_tab(capture)
        self._build_reconstruction_tab(reconstruction)
        self._change_stage(0)

    def _change_stage(self, index):
        if not hasattr(self, "control_pages"):
            return
        index = max(0, min(int(index), len(self.control_pages) - 1))
        if index != 1 and self._auto_calibration_active:
            self._stop_auto_calibration("离开标定页，自动采集已停止")
        if index != 2 and self.foundationpose_live_active:
            self._stop_foundationpose_live()
        if index != self._active_stage_index:
            self._invalidate_capture_analysis(clear_state=index != 2)
        self._active_stage_index = index
        self._last_processed_anchor_timestamp_s = None
        self.control_pages[index].tkraise()
        for current, button in enumerate(self.stage_buttons):
            selected = current == index
            button.configure(
                background="#ffffff" if selected else "#edf0f1",
                foreground="#0b645c" if selected else "#435057",
                font=("Microsoft YaHei", 10, "bold" if selected else "normal"),
                highlightbackground="#d3ddda" if selected else "#edf0f1",
            )
        self._sync_calibration_projector()

    def _page_heading(self, parent, title, subtitle):
        self.ttk.Label(parent, text=title, style="PanelTitle.TLabel").pack(anchor="w")
        self.ttk.Label(parent, text=subtitle, style="Muted.TLabel").pack(
            anchor="w", pady=(2, 14)
        )

    def _row_entry(self, parent, row, label, variable, browse=None):
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = self.ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 4), pady=4)
        if browse is not None:
            self.ttk.Button(parent, text="选择", command=browse).grid(
                row=row, column=2, sticky="e", pady=4
            )
        parent.columnconfigure(1, weight=1)
        return entry

    def _build_environment_tab(self, parent):
        self._page_heading(parent, "环境与相机", "运行依赖与设备状态")
        self.environment_text = self.tk.Text(
            parent,
            height=19,
            wrap="word",
            relief="flat",
            borderwidth=0,
            background="#20272b",
            foreground="#d7dee1",
            insertbackground="#d7dee1",
            selectbackground="#176d64",
            padx=9,
            pady=8,
            font=("Noto Sans Mono CJK SC", 9),
        )
        self.environment_text.pack(fill="both", expand=True)
        buttons = self.ttk.Frame(parent)
        buttons.pack(fill="x", pady=(10, 0))
        self.ttk.Button(buttons, text="重新检查", command=self._run_environment_check).pack(
            fill="x"
        )

    def _build_calibration_tab(self, parent):
        self._page_heading(
            parent,
            "RGB-D 标定",
            "{} 彩色与红外图像对".format(self.calibration_target.display_name),
        )
        explanation = (
            "RGB 与深度不可按分辨率缩放。这里用同一 {} 的 RGB/IR 图像求 "
            "T_color_depth。自动采集只在共同角点达到阈值且视角变化后记录。"
            "当前方格尺寸以配置中的 {:.1f} mm 为准。".format(
                self.calibration_target.display_name,
                self.calibration_target.square_size_m * 1000.0
                if self.calibration_target.target_type == "checkerboard"
                else 36.0,
            )
        )
        self.ttk.Label(
            parent, text=explanation, style="Muted.TLabel", wraplength=390, justify="left"
        ).pack(fill="x")
        self.calibration_pair_text = self.tk.StringVar(value="尚未采集 RGB/IR 图像对")
        self.rgbd_status_text = self.tk.StringVar(value="RGB-D 外参无效")
        self.projector_status_text = self.tk.StringVar(
            value="红外投影器：连接相机后自动控制"
        )
        self.ttk.Label(
            parent,
            textvariable=self.rgbd_status_text,
            style="Result.TLabel",
            wraplength=370,
            justify="left",
        ).pack(fill="x", pady=(12, 6))
        self.ttk.Label(parent, textvariable=self.calibration_pair_text, style="Muted.TLabel").pack(
            fill="x", pady=(0, 10)
        )
        self.ttk.Label(
            parent,
            textvariable=self.projector_status_text,
            style="Result.TLabel",
            wraplength=370,
            justify="left",
        ).pack(fill="x", pady=(0, 10))
        actions = self.ttk.Frame(parent)
        actions.pack(fill="x")
        settings = self.ttk.Frame(actions)
        settings.pack(fill="x", pady=(0, 5))
        self.auto_corner_threshold_var = self.tk.StringVar(
            value=str(self._auto_calibration_minimum_common)
        )
        self.auto_target_pairs_var = self.tk.StringVar(
            value=str(self._auto_calibration_target_pairs)
        )
        self.ttk.Label(settings, text="共同角点阈值").grid(row=0, column=0, sticky="w")
        self.ttk.Entry(
            settings, textvariable=self.auto_corner_threshold_var, width=7
        ).grid(row=0, column=1, padx=(6, 16))
        self.ttk.Label(settings, text="目标对数").grid(row=0, column=2, sticky="w")
        self.ttk.Entry(
            settings, textvariable=self.auto_target_pairs_var, width=7
        ).grid(row=0, column=3, padx=(6, 0))
        self.auto_calibration_button = self.ttk.Button(
            actions,
            text="开始自动采集",
            style="Primary.TButton",
            command=self._toggle_auto_calibration,
        )
        self.auto_calibration_button.pack(fill="x", pady=3)
        self.ttk.Button(actions, text="手动采集当前 RGB/IR 对", command=self._capture_calibration_pair).pack(
            fill="x", pady=3
        )
        self.ttk.Button(actions, text="计算 RGB-D 标定", command=self._solve_rgbd_calibration).pack(
            fill="x", pady=3
        )
        self.ttk.Button(
            actions,
            text="写入中央参数",
            style="Primary.TButton",
            command=self._write_rgbd_calibration,
        ).pack(
            fill="x", pady=3
        )
        self.ttk.Button(actions, text="重新载入中央参数", command=self._reload_rgbd_calibration).pack(
            fill="x", pady=3
        )
        self.calibration_log = self.tk.Text(
            parent,
            height=14,
            wrap="word",
            background="#20272b",
            foreground="#d7dee1",
            insertbackground="#d7dee1",
            relief="flat",
            borderwidth=0,
            padx=9,
            pady=8,
            font=("Noto Sans Mono CJK SC", 9),
        )
        self.calibration_log.pack(fill="both", expand=True, pady=(12, 0))

    def _build_capture_tab(self, parent):
        from tkinter import filedialog

        self._page_heading(
            parent,
            "无模型拍照与实时测试",
            "RGB-D 参考视图 · AprilTag · YOLO Mask · FoundationPose",
        )
        self.yolo_weights_var = self.tk.StringVar(value=self.paths.get("yolo_weights", ""))
        self.target_class_var = self.tk.StringVar(
            value=",".join(self.segmentation_config.get("target_classes", []))
        )
        self.session_var = self.tk.StringVar(value="")
        self.reference_object_id_var = self.tk.StringVar(value="1")
        self.reference_object_name_var = self.tk.StringVar(
            value=(self.segmentation_config.get("target_classes") or ["object"])[0]
        )
        self.foundationpose_mesh_var = self.tk.StringVar(
            value=str(self.foundationpose_live_config.get("mesh_path", ""))
        )
        self.foundationpose_mesh_scale_var = self.tk.StringVar(
            value=str(
                self.foundationpose_live_config.get("mesh_scale_to_meters", 1.0)
            )
        )

        def choose_weights():
            path = filedialog.askopenfilename(filetypes=[("PyTorch weights", "*.pt *.pth")])
            if path:
                self.yolo_weights_var.set(path)

        def choose_session():
            path = filedialog.askdirectory(initialdir=self.paths["capture_root"])
            if path:
                try:
                    self._open_capture_session(path)
                except Exception as error:
                    self._show_error("无法打开参考会话", error)

        def choose_mesh():
            path = filedialog.askopenfilename(
                initialdir=self.paths["mesh_root"],
                filetypes=[("Triangle mesh", "*.obj *.ply *.stl")],
            )
            if path:
                self.foundationpose_mesh_var.set(path)

        form = self.ttk.Frame(parent)
        form.pack(fill="x")
        self._row_entry(form, 0, "YOLO 权重", self.yolo_weights_var, choose_weights)
        self._row_entry(form, 1, "目标类别", self.target_class_var)
        self._row_entry(form, 2, "采集会话", self.session_var, choose_session)
        self._row_entry(form, 3, "参考物体 ID", self.reference_object_id_var)
        self._row_entry(form, 4, "参考物体名", self.reference_object_name_var)
        self._row_entry(form, 5, "实时网格", self.foundationpose_mesh_var, choose_mesh)
        self._row_entry(form, 6, "网格米制缩放", self.foundationpose_mesh_scale_var)
        self.ttk.Button(parent, text="1  加载 YOLO 分割模型", command=self._load_yolo).pack(
            fill="x", pady=(12, 3)
        )
        self.ttk.Button(parent, text="2  开始本次拍照（只需一次）", command=self._new_capture_session).pack(
            fill="x", pady=3
        )
        capture_button = self.ttk.Button(
            parent,
            text="3  拍摄参考图",
            style="Primary.TButton",
            command=self._capture_view,
        )
        capture_button.pack(fill="x", pady=3)
        self.capture_button = capture_button
        self.ttk.Button(
            parent,
            text="4  导出 FoundationPose 参考照片 ZIP",
            command=self._pack_foundationpose_reference_zip,
        ).pack(fill="x", pady=3)
        self.ttk.Button(
            parent,
            text="5  加载网格并实时测试",
            style="Primary.TButton",
            command=self._load_foundationpose_live,
        ).pack(fill="x", pady=(10, 3))
        self.ttk.Button(
            parent,
            text="初始化 / 重新初始化",
            command=self._reset_foundationpose_live,
        ).pack(fill="x", pady=3)
        self.ttk.Button(
            parent,
            text="停止实时测试",
            command=self._stop_foundationpose_live,
        ).pack(fill="x", pady=3)
        self.foundationpose_live_status_text = self.tk.StringVar(
            value="参考照片 ZIP 需先重建为 OBJ/PLY/STL"
        )
        self.ttk.Label(
            parent,
            textvariable=self.foundationpose_live_status_text,
            style="Result.TLabel",
            wraplength=370,
            justify="left",
        ).pack(fill="x", pady=(8, 0))
        self.foundationpose_pose_text = self.tk.Text(
            parent,
            height=6,
            wrap="none",
            background="#20272b",
            foreground="#d7dee1",
            relief="flat",
            borderwidth=0,
            padx=7,
            pady=6,
            font=("Noto Sans Mono CJK SC", 8),
        )
        self.foundationpose_pose_text.insert("end", "camera_from_object\n--")
        self.foundationpose_pose_text.configure(state="disabled")
        self.foundationpose_pose_text.pack(fill="x", pady=(6, 0))
        self.capture_status_text = self.tk.StringVar(value="等待 Tag、Mask 和对齐深度")
        self.ttk.Label(
            parent,
            textvariable=self.capture_status_text,
            style="Result.TLabel",
            wraplength=370,
            justify="left",
        ).pack(fill="x", pady=(14, 0))
        gates = "彩深不同步时按钮会等待下一组深度；深度覆盖率、物体固定和新视角仍需合格。"
        self.ttk.Label(parent, text=gates, style="Muted.TLabel", wraplength=370).pack(
            fill="x", pady=(12, 0)
        )

    def _build_reconstruction_tab(self, parent):
        from tkinter import filedialog

        self._page_heading(parent, "重建与导出", "TSDF 网格与 FoundationPose 模型")
        self.mesh_root_var = self.tk.StringVar(value=self.paths["mesh_root"])
        self.model_name_var = self.tk.StringVar(value="bottle")
        self.voxel_var = self.tk.StringVar(value=str(self.fusion_config["voxel_length_m"]))
        self.trunc_var = self.tk.StringVar(value=str(self.fusion_config["sdf_trunc_m"]))

        def choose_mesh_root():
            path = filedialog.askdirectory(initialdir=self.mesh_root_var.get())
            if path:
                self.mesh_root_var.set(path)

        form = self.ttk.Frame(parent)
        form.pack(fill="x")
        self._row_entry(form, 0, "模型输出目录", self.mesh_root_var, choose_mesh_root)
        self._row_entry(form, 1, "模型名称", self.model_name_var)
        self._row_entry(form, 2, "TSDF 体素 (m)", self.voxel_var)
        self._row_entry(form, 3, "TSDF 截断 (m)", self.trunc_var)
        self.ttk.Button(parent, text="融合采集会话", command=self._fuse_session).pack(
            fill="x", pady=(12, 3)
        )
        self.ttk.Button(parent, text="查看三维网格", command=self._view_mesh).pack(fill="x", pady=3)
        self.ttk.Button(
            parent,
            text="导出 FoundationPose 模型",
            style="Primary.TButton",
            command=self._export_model,
        ).pack(fill="x", pady=3)
        self.mesh_status_text = self.tk.StringVar(value="尚未生成网格")
        self.ttk.Label(
            parent,
            textvariable=self.mesh_status_text,
            style="Result.TLabel",
            wraplength=370,
            justify="left",
        ).pack(fill="x", pady=(14, 0))

    def _set_status(self, text: str):
        self.status_text.set(str(text))

    def _show_error(self, title: str, error):
        from tkinter import messagebox

        self._set_status(str(error))
        messagebox.showerror(title, str(error), parent=self.root)

    def _run_environment_check(self):
        results = run_checks(
            self.paths["foundationpose_root"],
            self.paths["runtime_calibration"],
            self.yolo_weights_var.get() if hasattr(self, "yolo_weights_var") else self.paths.get("yolo_weights"),
        )
        self.environment_text.configure(state="normal")
        self.environment_text.delete("1.0", "end")
        for result in results:
            marker = "[OK]" if result.ok else "[缺失]"
            self.environment_text.insert("end", "{} {}\n    {}\n\n".format(marker, result.name, result.detail))
        self.environment_text.configure(state="disabled")

    def _toggle_camera(self):
        if self.source is not None:
            if self._auto_calibration_active:
                self._stop_auto_calibration("相机断开，自动采集已停止")
            self.source.stop()
            self.source = None
            self._invalidate_capture_analysis(clear_state=True)
            self.connection_text.set("相机未连接")
            self.connect_button.configure(text="连接相机")
            self.preview_state_text.set("待机")
            self._reset_previews()
            self._sync_calibration_projector()
            self._set_status("相机已断开")
            return
        try:
            log_path = Path(self.paths["capture_root"]) / "orbbec_driver.log"
            backend = self.camera_config.get("backend", "astra_ros")
            if backend == "oak_depthai":
                oak = self.camera_config.get("oak", {})
                self.source = OakDProSource(
                    color_width=oak.get("color_width", 1280),
                    color_height=oak.get("color_height", 720),
                    fps=oak.get("fps", 30),
                )
            elif backend == "astra_ros":
                ros_driver = self.camera_config.get("ros_driver", {})
                self.source = AstraRosSource(
                    color_device=self.camera_config["color_device"],
                    color_width=self.camera_config["color_width"],
                    color_height=self.camera_config["color_height"],
                    color_fps=self.camera_config["color_fps"],
                    color_fourcc=self.camera_config["color_fourcc"],
                    depth_topic=self.camera_config["depth_topic"],
                    depth_info_topic=self.camera_config["depth_info_topic"],
                    ir_topic=self.camera_config["ir_topic"],
                    start_ros_driver=self.camera_config.get("start_ros_driver", True),
                    driver_log_path=str(log_path),
                    driver_package=ros_driver.get("package", "astra_camera"),
                    driver_launch_file=ros_driver.get("launch_file", "astra.launch"),
                    driver_arguments=ros_driver.get("arguments"),
                    driver_startup_timeout_s=ros_driver.get("startup_timeout_s", 2.0),
                    laser_service=self.camera_config.get(
                        "laser_service", "/camera/set_laser"
                    ),
                )
            else:
                raise ValueError("unsupported camera backend: {}".format(backend))
            self.source.start()
            self._invalidate_capture_analysis(clear_state=True)
            self._sync_calibration_projector(required=True)
            self.connection_text.set("已连接")
            self.connect_button.configure(text="断开相机")
            self.preview_state_text.set("实时")
            self._set_status("相机已启动，等待 RGB、深度和 IR 帧")
        except Exception as error:
            if self.source is not None:
                self.source.stop()
            self.source = None
            self.connection_text.set("相机未连接")
            self.connect_button.configure(text="连接相机")
            self.preview_state_text.set("连接失败")
            self._show_error("相机启动失败", error)

    def _reload_rgbd_calibration(self, show_dialog=True):
        try:
            self.rgbd_calibration = load_runtime_calibration(self.paths["runtime_calibration"])
            self.rgbd_calibration.require_valid()
            self.depth_aligner = DepthToColorAligner(self.rgbd_calibration)
            self._invalidate_capture_analysis(clear_state=True)
            rms = self.rgbd_calibration.rms_reprojection_error_px
            detail = "RGB-D 标定有效"
            if rms is not None:
                detail += "，RMS {:.3f} px".format(float(rms))
            self.rgbd_status_text.set(detail) if hasattr(self, "rgbd_status_text") else None
            self._set_status(detail) if hasattr(self, "status_text") else None
        except Exception as error:
            self.rgbd_calibration = None
            self.depth_aligner = None
            self._invalidate_capture_analysis(clear_state=True)
            if hasattr(self, "rgbd_status_text"):
                self.rgbd_status_text.set("RGB-D 标定无效：{}".format(error))
            if show_dialog:
                self._show_error("RGB-D 标定尚不可用", error)

    def _load_yolo(self):
        try:
            classes = [value.strip() for value in self.target_class_var.get().split(",") if value.strip()]
            self.root.configure(cursor="watch")
            self.root.update_idletasks()
            self.yolo_provider = YoloMaskProvider(
                self.yolo_weights_var.get(),
                target_classes=classes,
                confidence_threshold=self.segmentation_config["confidence_threshold"],
                device=self.segmentation_config.get("device", "0"),
            )
            self._invalidate_capture_analysis(clear_state=True)
            self._set_status("YOLO 分割模型已加载：{}".format(self.yolo_weights_var.get()))
            self._run_environment_check()
        except Exception as error:
            self.yolo_provider = None
            self._invalidate_capture_analysis(clear_state=True)
            self._show_error("YOLO 加载失败", error)
        finally:
            self.root.configure(cursor="")

    def _invalidate_capture_analysis(self, clear_state: bool) -> None:
        with self._analysis_lock:
            self._analysis_generation += 1
            self._analysis_pending = None
            self._analysis_results.clear()
        if getattr(self, "foundationpose_live_active", False):
            self._reset_foundationpose_live("分析链路已刷新，FoundationPose 已重新初始化")
        if clear_state:
            self.analysis_bundle = None
            self.analysis_rectified_color = None
            self.analysis_completed_monotonic_s = 0.0
            self.tag_estimate = None
            self.tag_detections = {}
            self.mask_result = MaskResult(False, reason="等待后台分析")
            self.aligned_depth = None
            self.device_rgbd_calibration = None

    def _shutdown_capture_analysis(self) -> None:
        with self._analysis_lock:
            self._analysis_shutdown = True
            self._analysis_generation += 1
            self._analysis_pending = None
            self._analysis_results.clear()
        worker = getattr(self, "foundationpose_live_worker", None)
        if worker is not None:
            worker.close()

    def _queue_capture_analysis(
        self,
        bundle: FrameBundle,
        rectified_color: np.ndarray,
        rectified_color_intrinsics: CameraIntrinsics,
    ) -> None:
        task = (
            self._analysis_generation,
            bundle,
            rectified_color,
            rectified_color_intrinsics,
        )
        with self._analysis_lock:
            if self._analysis_shutdown:
                return
            self._analysis_pending = task
            if self._analysis_worker_active:
                return
            self._analysis_worker_active = True
        threading.Thread(target=self._capture_analysis_loop, daemon=True).start()

    def _capture_analysis_loop(self) -> None:
        while True:
            with self._analysis_lock:
                if self._analysis_shutdown:
                    self._analysis_worker_active = False
                    return
                task = self._analysis_pending
                self._analysis_pending = None
                if task is None:
                    self._analysis_worker_active = False
                    return
            generation, bundle, rectified_color, color_intrinsics = task
            try:
                result = self._analyze_capture_frame(
                    generation,
                    bundle,
                    rectified_color,
                    color_intrinsics,
                )
            except Exception as error:
                result = CaptureAnalysisResult(
                    generation=generation,
                    bundle=bundle,
                    rectified_color=rectified_color,
                    completed_monotonic_s=time.monotonic(),
                    error=error,
                )
            with self._analysis_lock:
                if not self._analysis_shutdown:
                    self._analysis_results.clear()
                    self._analysis_results.append(result)

    def _analyze_capture_frame(
        self,
        generation: int,
        bundle: FrameBundle,
        rectified_color: np.ndarray,
        color_intrinsics: CameraIntrinsics,
    ) -> CaptureAnalysisResult:
        if self._analysis_cache_generation != generation:
            self._analysis_cache_generation = generation
            self._analysis_cached_depth_timestamp_s = None
            self._analysis_cached_aligned_depth = None
            self._analysis_cached_depth_preview_bgr = None
            self._analysis_cached_aligned_preview_bgr = None
            self._analysis_cached_device_calibration = None
            self._analysis_cached_mask_result = MaskResult(
                False, reason="等待后台分割"
            )
            self._analysis_last_yolo_monotonic_s = 0.0

        estimate, detections = self.tag_provider.estimate(
            rectified_color,
            color_intrinsics.matrix,
            color_intrinsics.distortion,
        )
        mask_result = self._analysis_cached_mask_result
        now = time.monotonic()
        provider = self.yolo_provider
        if (
            provider is not None
            and now - self._analysis_last_yolo_monotonic_s
            >= self._yolo_preview_interval_s
        ):
            mask_result = provider.predict(rectified_color)
            self._analysis_cached_mask_result = mask_result
            self._analysis_last_yolo_monotonic_s = now

        device_calibration = self._analysis_cached_device_calibration
        aligned_depth = self._analysis_cached_aligned_depth
        if (
            bundle.depth_m is not None
            and bundle.depth_timestamp_s != self._analysis_cached_depth_timestamp_s
        ):
            if bundle.depth_aligned_to_color:
                if bundle.depth_m.shape != rectified_color.shape[:2]:
                    raise ValueError(
                        "device-aligned depth dimensions do not match RGB"
                    )
                aligned_depth = (
                    bundle.depth_m.copy()
                    if bundle.color_is_rectified
                    else rectify_aligned_depth_image(
                        bundle.depth_m,
                        bundle.color_intrinsics or self.color_intrinsics,
                    )
                )
                device_calibration = RgbdCalibration(
                    color=color_intrinsics,
                    depth=color_intrinsics,
                    color_from_depth=np.eye(4),
                    valid=True,
                    source="DepthAI factory calibration; depth aligned to RGB",
                )
            elif self.depth_aligner is not None:
                aligned_depth = self.depth_aligner.align(bundle.depth_m)
                device_calibration = None
            else:
                aligned_depth = None
                device_calibration = None
            self._analysis_cached_depth_timestamp_s = bundle.depth_timestamp_s
            self._analysis_cached_aligned_depth = aligned_depth
            self._analysis_cached_device_calibration = device_calibration
            self._analysis_cached_depth_preview_bgr = self._depth_colormap(
                bundle.depth_m
            )
            self._analysis_cached_aligned_preview_bgr = (
                None
                if aligned_depth is None
                else self._depth_colormap(aligned_depth)
            )

        aligned_preview = self._analysis_cached_aligned_preview_bgr
        if aligned_preview is not None:
            aligned_preview = aligned_preview.copy()
            if mask_result.valid and mask_result.mask is not None:
                outside = ~mask_result.mask.astype(bool)
                aligned_preview[outside] = (
                    aligned_preview[outside] * 0.22
                ).astype(np.uint8)

        return CaptureAnalysisResult(
            generation=generation,
            bundle=bundle,
            rectified_color=rectified_color,
            tag_estimate=estimate,
            tag_detections=detections,
            mask_result=mask_result,
            aligned_depth=aligned_depth,
            depth_preview_bgr=self._analysis_cached_depth_preview_bgr,
            aligned_preview_bgr=aligned_preview,
            device_calibration=device_calibration,
            completed_monotonic_s=time.monotonic(),
        )

    def _consume_capture_analysis(self) -> None:
        with self._analysis_lock:
            if not self._analysis_results:
                return
            result = self._analysis_results.pop()
            self._analysis_results.clear()
            generation = self._analysis_generation
        if result.generation != generation or self._active_stage_index != 2:
            return
        if result.error is not None:
            self.analysis_bundle = None
            self.analysis_rectified_color = None
            self.analysis_completed_monotonic_s = 0.0
            self.tag_estimate = None
            self.tag_detections = {}
            self.mask_result = MaskResult(
                False, reason="后台分析失败：{}".format(result.error)
            )
            self.aligned_depth = None
            self.device_rgbd_calibration = None
            self._set_status("后台分析错误：{}".format(result.error))
            return
        self.analysis_bundle = result.bundle
        self.analysis_rectified_color = result.rectified_color
        self.analysis_completed_monotonic_s = result.completed_monotonic_s
        self.tag_estimate = result.tag_estimate
        self.tag_detections = result.tag_detections or {}
        self.mask_result = result.mask_result or MaskResult(
            False, reason="后台分割尚无结果"
        )
        self.aligned_depth = result.aligned_depth
        self.device_rgbd_calibration = result.device_calibration
        if result.depth_preview_bgr is not None:
            self._set_preview(
                self.depth_preview,
                result.depth_preview_bgr,
                (390, 270),
                "depth",
            )
            self._last_displayed_depth_timestamp_s = result.bundle.depth_timestamp_s
        if result.aligned_preview_bgr is not None:
            self._set_preview(
                self.aligned_preview,
                result.aligned_preview_bgr,
                (390, 270),
                "aligned",
            )
        self._queue_foundationpose_live_frame()

    def _set_foundationpose_live_status(self, text: str) -> None:
        if hasattr(self, "foundationpose_live_status_text"):
            self.foundationpose_live_status_text.set(str(text))

    def _set_foundationpose_pose_matrix(self, pose) -> None:
        widget = getattr(self, "foundationpose_pose_text", None)
        if widget is None:
            return
        if pose is None:
            text = "camera_from_object\n--"
        else:
            matrix = np.asarray(pose, dtype=np.float64).reshape(4, 4)
            rows = ["[" + "  ".join("{:+.5f}".format(value) for value in row) + "]" for row in matrix]
            text = "camera_from_object (m)\n" + "\n".join(rows)
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _load_foundationpose_live(self) -> None:
        try:
            mesh_path = self.foundationpose_mesh_var.get().strip()
            if not mesh_path:
                raise ValueError("请先选择 FoundationPose OBJ/PLY/STL 网格")
            scale = float(self.foundationpose_mesh_scale_var.get())
            live = self.foundationpose_live_config
            debug_dir = self.paths.get(
                "foundationpose_debug_root",
                str(Path(self.paths["capture_root"]) / "foundationpose_debug"),
            )
            config = FoundationPoseLiveConfig(
                foundationpose_root=self.paths["foundationpose_root"],
                mesh_path=mesh_path,
                mesh_scale_to_meters=scale,
                debug_dir=debug_dir,
                debug=int(live.get("debug", 0)),
                est_refine_iter=int(live.get("est_refine_iter", 5)),
                track_refine_iter=int(live.get("track_refine_iter", 2)),
                device=str(live.get("device", "cuda:0")),
                use_mask_center_guidance=bool(
                    live.get("use_mask_center_guidance", True)
                ),
            )
            self.foundationpose_live_worker.configure(config)
            self.foundationpose_live_active = True
            self.foundationpose_live_pose = None
            self.foundationpose_live_bounds_m = None
            self.foundationpose_live_mode = "loading"
            self.foundationpose_live_timestamp_s = None
            self._foundationpose_live_last_frame_id = None
            self._set_foundationpose_pose_matrix(None)
            text = "正在后台加载 FoundationPose 和网格；RGB 预览不会暂停"
            self._set_foundationpose_live_status(text)
            self._set_status(text)
        except Exception as error:
            self._show_error("FoundationPose 加载失败", error)

    def _reset_foundationpose_live(self, reason=None) -> None:
        worker = getattr(self, "foundationpose_live_worker", None)
        if worker is None or not getattr(self, "foundationpose_live_active", False):
            return
        if not worker.reset():
            return
        self.foundationpose_live_pose = None
        self.foundationpose_live_bounds_m = None
        self.foundationpose_live_mode = "resetting"
        self.foundationpose_live_timestamp_s = None
        self._foundationpose_live_last_frame_id = None
        self._set_foundationpose_pose_matrix(None)
        text = reason or "FoundationPose 已请求重新初始化；下一帧执行注册"
        self._set_foundationpose_live_status(text)
        self._set_status(text)

    def _stop_foundationpose_live(self) -> None:
        worker = getattr(self, "foundationpose_live_worker", None)
        if worker is not None:
            worker.stop()
        self.foundationpose_live_active = False
        self.foundationpose_live_pose = None
        self.foundationpose_live_bounds_m = None
        self.foundationpose_live_mode = "stopped"
        self.foundationpose_live_timestamp_s = None
        self._foundationpose_live_last_frame_id = None
        self._set_foundationpose_pose_matrix(None)
        self._set_foundationpose_live_status("实时测试已停止")
        self._set_status("FoundationPose 实时测试已停止")

    def _queue_foundationpose_live_frame(self) -> None:
        if not self.foundationpose_live_active:
            return
        bundle = self.analysis_bundle
        color = self.analysis_rectified_color
        mask_result = self.mask_result
        if bundle is None or color is None or self.aligned_depth is None:
            self._set_foundationpose_live_status("等待成对 RGB 和对齐深度")
            return
        if not mask_result.valid or mask_result.mask is None:
            self._set_foundationpose_live_status("等待有效 YOLO Mask")
            return
        if bundle.sync_delta_s is None or bundle.sync_delta_s > float(
            self.camera_config["maximum_sync_delta_s"]
        ):
            self._set_foundationpose_live_status("彩深时间差超限，实时帧已拒绝")
            return
        analysis_age_s = time.monotonic() - self.analysis_completed_monotonic_s
        if analysis_age_s > self._maximum_analysis_age_s:
            self._set_foundationpose_live_status("后台分析结果过期，实时帧已拒绝")
            return
        calibration = self._effective_rgbd_calibration()
        if calibration is None:
            self._set_foundationpose_live_status("等待有效 RGB-D 标定")
            return
        frame_id = (bundle.color_timestamp_s, bundle.depth_timestamp_s)
        if frame_id == self._foundationpose_live_last_frame_id:
            return
        frame = FoundationPoseLiveFrame(
            frame_id=frame_id,
            timestamp_s=float(bundle.color_timestamp_s),
            color_bgr=color,
            depth_m=self.aligned_depth,
            mask=mask_result.mask,
            camera_matrix=rectified_intrinsics(calibration.color).matrix,
        )
        try:
            accepted = self.foundationpose_live_worker.submit(frame)
        except Exception as error:
            self._set_foundationpose_live_status("实时帧被拒绝：{}".format(error))
            return
        if accepted:
            self._foundationpose_live_last_frame_id = frame_id

    def _consume_foundationpose_live_result(self) -> None:
        result = self.foundationpose_live_worker.poll()
        if result is None:
            return
        if result.status == "ready":
            self.foundationpose_live_mode = "ready"
            self._set_foundationpose_live_status(
                "FoundationPose 已加载；等待有效 RGB-D + Mask 执行首帧注册"
            )
            return
        if result.status == "reset":
            self.foundationpose_live_mode = "ready"
            self._set_foundationpose_live_status("已重置；下一帧执行 REGISTER")
            return
        if result.status == "stopped":
            return
        if result.status == "error":
            self.foundationpose_live_pose = None
            self.foundationpose_live_bounds_m = None
            self.foundationpose_live_mode = "error"
            self.foundationpose_live_timestamp_s = None
            text = "FoundationPose 失败：{}；请点击初始化/重新初始化".format(
                result.error
            )
            self._set_foundationpose_live_status(text)
            self._set_status(text)
            self._set_foundationpose_pose_matrix(None)
            return
        self.foundationpose_live_pose = result.camera_from_object
        self.foundationpose_live_bounds_m = result.mesh_bounds_m
        self.foundationpose_live_mode = result.status
        self.foundationpose_live_inference_ms = float(result.inference_ms)
        self.foundationpose_live_timestamp_s = result.timestamp_s
        mode = "REGISTER" if result.status == "registered" else "TRACK"
        text = "{} 成功 · {:.1f} ms · camera_from_object 已更新".format(
            mode, result.inference_ms
        )
        self._set_foundationpose_live_status(text)
        self._set_foundationpose_pose_matrix(result.camera_from_object)

    def _draw_foundationpose_live_overlay(
        self,
        color_view: np.ndarray,
        bundle: FrameBundle,
        camera_matrix: np.ndarray,
    ) -> np.ndarray:
        if (
            not self.foundationpose_live_active
            or self.foundationpose_live_pose is None
            or self.foundationpose_live_bounds_m is None
            or self.foundationpose_live_timestamp_s is None
        ):
            return color_view
        maximum_age_s = float(
            self.foundationpose_live_config.get("maximum_pose_age_s", 1.0)
        )
        if abs(float(bundle.color_timestamp_s) - float(self.foundationpose_live_timestamp_s)) > maximum_age_s:
            return color_view
        mode = "REGISTER" if self.foundationpose_live_mode == "registered" else "TRACK"
        return draw_pose_overlay(
            color_view,
            self.foundationpose_live_pose,
            camera_matrix,
            self.foundationpose_live_bounds_m,
            mode=mode,
            inference_ms=self.foundationpose_live_inference_ms,
        )

    def _tick(self):
        try:
            self._consume_capture_analysis()
            self._consume_foundationpose_live_result()
            self._try_pending_capture()
            if self.source is not None:
                if self._active_stage_index == 1:
                    anchor = "ir"
                else:
                    anchor = "color"
                bundle = self.source.latest(anchor=anchor)
                if bundle is not None:
                    self.bundle = bundle
                    if anchor == "ir":
                        anchor_timestamp = bundle.ir_timestamp_s
                    elif anchor == "depth":
                        anchor_timestamp = bundle.depth_timestamp_s
                    else:
                        anchor_timestamp = bundle.color_timestamp_s
                    if (
                        anchor_timestamp is not None
                        and anchor_timestamp == self._last_processed_anchor_timestamp_s
                    ):
                        return
                    self._last_processed_anchor_timestamp_s = anchor_timestamp
                    self._process_preview(bundle)
        except Exception as error:
            self._set_status("实时预览错误：{}".format(error))
        finally:
            self.root.after(self._preview_interval_ms, self._tick)

    def _process_preview(self, bundle: FrameBundle):
        self._frame_counter += 1
        calibration_stage = self._active_stage_index == 1
        capture_stage = self._active_stage_index == 2
        active_color_intrinsics = bundle.color_intrinsics or self.color_intrinsics
        self.rectified_color = (
            bundle.color_bgr.copy()
            if bundle.color_is_rectified
            else rectify_color_image(bundle.color_bgr, active_color_intrinsics)
        )
        rectified_color_intrinsics = rectified_intrinsics(active_color_intrinsics)
        self.color_intrinsics = active_color_intrinsics
        if capture_stage:
            estimate = self.tag_estimate
            detections = self.tag_detections
            color_view = self.rectified_color
            if estimate is not None:
                color_view = self.tag_provider.draw_status(
                    color_view, detections, estimate
                )
            color_view = (
                self.yolo_provider.overlay(color_view, self.mask_result)
                if self.yolo_provider
                else color_view
            )
            color_view = self._draw_foundationpose_live_overlay(
                color_view,
                bundle,
                rectified_color_intrinsics.matrix,
            )
            self._queue_capture_analysis(
                bundle,
                self.rectified_color,
                rectified_color_intrinsics,
            )
        else:
            estimate = None
            detections = {}
            color_view = self.rectified_color
        self._set_preview(self.color_preview, color_view, (790, 445), "color")
        if (
            bundle.depth_m is not None
            and not capture_stage
            and bundle.depth_timestamp_s != self._last_displayed_depth_timestamp_s
        ):
            self._set_preview(
                self.depth_preview,
                self._depth_colormap(bundle.depth_m),
                (390, 270),
                "depth",
            )
            self._last_displayed_depth_timestamp_s = bundle.depth_timestamp_s
        if (
            bundle.ir_image is not None
            and hasattr(self, "ir_preview")
            and bundle.ir_timestamp_s != self._last_displayed_ir_timestamp_s
        ):
            infrared = infrared_to_uint8(bundle.ir_image)
            ir_view = cv2.cvtColor(infrared, cv2.COLOR_GRAY2BGR)
            if calibration_stage:
                now = time.monotonic()
                detection_is_current = (
                    self._ir_target_timestamp_s == bundle.ir_timestamp_s
                )
                if not detection_is_current and now - self._last_ir_detection_monotonic_s >= 0.25:
                    corners, ids = detect_calibration_target(
                        infrared, self.calibration_target
                    )
                    self._ir_target_corners = corners
                    self._ir_target_ids = ids
                    self._ir_target_count = len(ids)
                    self._ir_charuco_corners = corners
                    self._ir_charuco_ids = ids
                    self._ir_charuco_count = len(ids)
                    self._ir_target_timestamp_s = bundle.ir_timestamp_s
                    self._last_ir_detection_monotonic_s = now
                corners = self._ir_target_corners
                laser_state = getattr(self.source, "laser_enabled", None)
                laser_text = "CMD OFF" if laser_state is False else "CMD ON" if laser_state else "UNKNOWN"
                color = (0, 210, 230) if laser_state is False else (0, 80, 255)
                for x, y in corners:
                    cv2.circle(ir_view, (int(round(x)), int(round(y))), 3, (0, 220, 0), -1)
                cv2.rectangle(ir_view, (0, 0), (ir_view.shape[1], 30), (20, 24, 27), -1)
                cv2.putText(
                    ir_view,
                    "{}: {}  PROJECTOR: {}".format(
                        self.calibration_target.display_name,
                        self._ir_target_count,
                        laser_text,
                    ),
                    (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                self._ir_target_corners = np.empty((0, 2), dtype=np.float32)
                self._ir_target_ids = np.empty((0,), dtype=np.int32)
                self._ir_target_count = 0
                self._ir_target_timestamp_s = None
                self._ir_charuco_corners = self._ir_target_corners
                self._ir_charuco_ids = self._ir_target_ids
                self._ir_charuco_count = 0
            self._set_preview(self.ir_preview, ir_view, (390, 270), "ir")
            self._last_displayed_ir_timestamp_s = bundle.ir_timestamp_s
        elif bundle.ir_image is None:
            self._ir_target_corners = np.empty((0, 2), dtype=np.float32)
            self._ir_target_ids = np.empty((0,), dtype=np.int32)
            self._ir_target_count = 0
            self._ir_charuco_corners = self._ir_target_corners
            self._ir_charuco_ids = self._ir_target_ids
            self._ir_charuco_count = 0
        status_bundle = (
            self.analysis_bundle
            if capture_stage and self.analysis_bundle is not None
            else bundle
        )
        sync = (
            "--"
            if status_bundle.sync_delta_s is None
            else "{:.0f} ms".format(status_bundle.sync_delta_s * 1000.0)
        )
        tag = "有效" if capture_stage and estimate is not None and estimate.valid else "待机"
        mask = "有效" if capture_stage and self.mask_result.valid else "待机"
        coverage_text = "--"
        if (
            capture_stage
            and self.aligned_depth is not None
            and self.mask_result.valid
            and self.mask_result.mask is not None
        ):
            coverage_text = "{:.0%}".format(
                depth_coverage(self.aligned_depth, self.mask_result.mask)
            )
        if self._capture_request_pending:
            remaining_s = max(0.0, self._capture_request_deadline_s - time.monotonic())
            self.capture_status_text.set(
                "等待拍摄 · {} · 剩余 {:.1f} 秒".format(
                    self._capture_request_last_reason, remaining_s
                )
            )
        elif time.monotonic() < self._capture_feedback_until_s:
            self.capture_status_text.set(self._capture_feedback_text)
        else:
            self.capture_status_text.set(
                "Tag {} | Mask {} | 深度 {}（需 {:.0%}） | 彩深差 {} | 已拍 {} 张".format(
                    tag,
                    mask,
                    coverage_text,
                    float(self.capture_config["minimum_mask_depth_coverage"]),
                    sync,
                    len(self.capture_session) if self.capture_session is not None else 0,
                )
            )
        height, width = self.rectified_color.shape[:2]
        frame_text = "RGB {} × {}".format(width, height)
        if bundle.ir_image is not None:
            ir_height, ir_width = bundle.ir_image.shape[:2]
            frame_text += " · IR {} × {}".format(ir_width, ir_height)
        self.frame_metric_text.set(frame_text)
        if calibration_stage:
            self.detection_metric_text.set(
                "IR {} {}".format(
                    self.calibration_target.display_name, self._ir_target_count
                )
            )
        else:
            self.detection_metric_text.set(
                "Tag {} · Mask {}".format(tag, mask)
            )
        self.capture_metric_text.set(
            "{} 帧".format(
                len(self.capture_session) if self.capture_session is not None else 0
            )
        )
        if calibration_stage:
            self._try_auto_calibration_capture(bundle)

    @staticmethod
    def _depth_colormap(depth_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = depth > 0.0
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        if valid.any():
            near = max(0.1, float(np.percentile(depth[valid], 2)))
            far = max(near + 0.05, float(np.percentile(depth[valid], 98)))
            normalized[valid] = np.clip((depth[valid] - near) / (far - near) * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = (28, 32, 35)
        return colored

    def _set_preview(self, widget, bgr: np.ndarray, maximum_size, key):
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(maximum_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        widget.configure(image=photo, text="")
        self._preview_refs[key] = photo

    def _reset_previews(self):
        self._preview_refs.clear()
        self._last_displayed_depth_timestamp_s = None
        self._last_displayed_ir_timestamp_s = None
        self.color_preview.configure(image="", text="相机未连接")
        self.depth_preview.configure(image="", text="等待深度图像")
        self.aligned_preview.configure(image="", text="等待彩深对齐")
        if hasattr(self, "ir_preview"):
            self.ir_preview.configure(image="", text="等待红外图像")
        self.frame_metric_text.set("--")
        self.detection_metric_text.set("待机")
        self.capture_metric_text.set("0 帧")

    def _calibration_pair_count(self) -> int:
        if self._calibration_pair_root is None:
            return 0
        return len(list((self._calibration_pair_root / "color").glob("*.png")))

    def _ensure_calibration_pair_root(self) -> Path:
        if self._calibration_pair_root is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._calibration_pair_root = (
                Path(self.paths["rgbd_calibration_root"]) / ("rgbd_" + timestamp)
            )
            (self._calibration_pair_root / "color").mkdir(parents=True, exist_ok=True)
            (self._calibration_pair_root / "ir").mkdir(parents=True, exist_ok=True)
        return self._calibration_pair_root

    def _calibration_observation(
        self,
        bundle,
        ir_corners=None,
        ir_ids=None,
    ):
        if bundle is None or bundle.ir_image is None:
            return 0, None, "当前没有同步的 RGB 和 IR 图像"
        if bundle.ir_timestamp_s is None:
            return 0, None, "IR 时间戳不可用"
        delta = abs(bundle.color_timestamp_s - bundle.ir_timestamp_s)
        maximum_delta = float(self.camera_config["maximum_sync_delta_s"])
        if delta > maximum_delta:
            return 0, None, "RGB/IR 时间差 {:.0f} ms 超限".format(delta * 1000.0)
        color_corners, color_ids = detect_calibration_target(
            bundle.color_bgr, self.calibration_target
        )
        if ir_corners is None or ir_ids is None:
            ir_corners, ir_ids = detect_calibration_target(
                bundle.ir_image, self.calibration_target
            )
        color_lookup = {
            int(identifier): corner
            for identifier, corner in zip(color_ids, color_corners)
        }
        ir_lookup = {
            int(identifier): corner
            for identifier, corner in zip(ir_ids, ir_corners)
        }
        common_ids = sorted(set(color_lookup).intersection(ir_lookup))
        if not common_ids:
            return 0, None, "RGB/IR 暂无共同{}角点".format(
                self.calibration_target.display_name
            )
        common_ir_corners = np.asarray(
            [ir_lookup[identifier] for identifier in common_ids], dtype=np.float32
        )
        signature = calibration_view_signature(
            common_ir_corners,
            np.asarray(common_ids, dtype=np.int32),
            bundle.ir_image.shape[:2],
            self.calibration_target,
        )
        return len(common_ids), signature, ""

    def _save_calibration_pair(self, bundle) -> int:
        root = self._ensure_calibration_pair_root()
        index = self._calibration_pair_count()
        name = "{:04d}.png".format(index)
        color_path = root / "color" / name
        ir_path = root / "ir" / name
        color_ok = cv2.imwrite(str(color_path), bundle.color_bgr)
        ir_ok = cv2.imwrite(str(ir_path), bundle.ir_image)
        if not color_ok or not ir_ok:
            color_path.unlink(missing_ok=True)
            ir_path.unlink(missing_ok=True)
            raise OSError("RGB/IR 标定图像写入失败")
        return index + 1

    def _stop_auto_calibration(self, message=None):
        self._auto_calibration_active = False
        if hasattr(self, "auto_calibration_button"):
            self.auto_calibration_button.configure(text="开始自动采集")
        if message and hasattr(self, "calibration_pair_text"):
            self.calibration_pair_text.set(message)

    def _toggle_auto_calibration(self, *_args):
        if self._auto_calibration_active:
            count = self._calibration_pair_count()
            self._stop_auto_calibration(
                "自动采集已暂停：{}/{} 对".format(
                    count, self._auto_calibration_target_pairs
                )
            )
            return
        try:
            self._sync_calibration_projector(required=True)
            if self.source is None or self.bundle is None:
                raise ValueError("请先连接相机并等待 RGB/IR 图像")
            minimum_common = int(self.auto_corner_threshold_var.get())
            target_pairs = int(self.auto_target_pairs_var.get())
            solver_minimum = int(
                self.camera_config.get(
                    "minimum_common_corners",
                    self.camera_config.get("minimum_common_charuco_corners", 10),
                )
            )
            if minimum_common < solver_minimum or minimum_common > self.calibration_target.point_count:
                raise ValueError(
                    "共同角点阈值必须在 {}～{} 之间".format(
                        solver_minimum, self.calibration_target.point_count
                    )
                )
            if target_pairs < 10 or target_pairs > 100:
                raise ValueError("目标图像对必须在 10～100 之间")
            self._auto_calibration_minimum_common = minimum_common
            self._auto_calibration_target_pairs = target_pairs
            current_count = self._calibration_pair_count()
            if current_count >= target_pairs:
                self._calibration_pair_root = None
                current_count = 0
            self._auto_calibration_last_capture_s = 0.0
            self._auto_calibration_last_signature = None
            self._auto_calibration_last_attempt_s = 0.0
            self._auto_calibration_active = True
            self.auto_calibration_button.configure(text="停止自动采集")
            self.calibration_pair_text.set(
                "自动采集 {}/{} 对｜等待共同角点 ≥ {}".format(
                    current_count, target_pairs, minimum_common
                )
            )
            self._set_status("自动 RGB/IR 标定采集已开始，请缓慢移动标定板")
        except Exception as error:
            self._stop_auto_calibration()
            self._show_error("无法开始自动采集", error)

    def _try_auto_calibration_capture(self, bundle):
        if not self._auto_calibration_active or self._active_stage_index != 1:
            return
        if self._ir_target_timestamp_s != bundle.ir_timestamp_s:
            return
        now = time.monotonic()
        if now - self._auto_calibration_last_attempt_s < 0.25:
            return
        self._auto_calibration_last_attempt_s = now
        count = self._calibration_pair_count()
        target = self._auto_calibration_target_pairs
        if count >= target:
            self._stop_auto_calibration(
                "自动采集完成：{}/{} 对，可以计算 RGB-D 标定".format(count, target)
            )
            return
        common_count, signature, reason = self._calibration_observation(
            bundle,
            self._ir_target_corners,
            self._ir_target_ids,
        )
        minimum = self._auto_calibration_minimum_common
        if reason:
            self.calibration_pair_text.set(
                "自动采集 {}/{} 对｜{}".format(count, target, reason)
            )
            return
        if common_count < minimum:
            self.calibration_pair_text.set(
                "自动采集 {}/{} 对｜共同角点 {}/{}，等待稳定识别".format(
                    count, target, common_count, minimum
                )
            )
            return
        if signature is None:
            self.calibration_pair_text.set(
                "自动采集 {}/{} 对｜角点几何无效，请调整板面角度".format(
                    count, target
                )
            )
            return
        elapsed = now - self._auto_calibration_last_capture_s
        if count > 0 and elapsed < self._auto_calibration_minimum_interval_s:
            return
        view_change = calibration_view_change(
            self._auto_calibration_last_signature, signature
        )
        if count > 0 and view_change < self._auto_calibration_minimum_view_change:
            self.calibration_pair_text.set(
                "自动采集 {}/{} 对｜共同角点 {}，请移动或旋转标定板".format(
                    count, target, common_count
                )
            )
            return
        try:
            count = self._save_calibration_pair(bundle)
        except Exception as error:
            self._stop_auto_calibration("自动采集因写入失败而停止")
            self._show_error("自动采集失败", error)
            return
        self._auto_calibration_last_capture_s = now
        self._auto_calibration_last_signature = signature
        if count >= target:
            self._stop_auto_calibration(
                "自动采集完成：{}/{} 对，可以计算 RGB-D 标定".format(count, target)
            )
            self._set_status("自动 RGB/IR 图像采集完成")
        else:
            self.calibration_pair_text.set(
                "自动采集 {}/{} 对｜本帧共同角点 {}，请继续移动标定板".format(
                    count, target, common_count
                )
            )
            self._set_status(
                "已自动保存第 {} 对 RGB/IR 图像，共同角点 {}".format(
                    count, common_count
                )
            )

    def _capture_calibration_pair(self):
        if self._auto_calibration_active:
            self._show_error("无法手动采集", "请先停止自动采集")
            return
        try:
            self._sync_calibration_projector(required=True)
        except Exception as error:
            self._show_error("无法采集", error)
            return
        common_count, _, reason = self._calibration_observation(self.bundle)
        if reason:
            self._show_error("无法采集", reason)
            return
        minimum_common = int(
            self.camera_config.get(
                "minimum_common_corners",
                self.camera_config.get("minimum_common_charuco_corners", 10),
            )
        )
        if common_count < minimum_common:
            self._show_error(
                "无法采集",
                "当前 RGB/IR 共同{}角点只有 {} 个，至少需要 {} 个。"
                "请确认红外预览中没有散斑、标定板清晰且完整可见。".format(
                    self.calibration_target.display_name,
                    common_count,
                    minimum_common,
                ),
            )
            return
        count = self._save_calibration_pair(self.bundle)
        self.calibration_pair_text.set(
            "已采集 {} 对：{}".format(count, self._calibration_pair_root)
        )
        self._set_status(
            "RGB/IR 标定图像对已保存，共同{}角点 {} 个".format(
                self.calibration_target.display_name, common_count
            )
        )

    def _sync_calibration_projector(self, required=False):
        if not hasattr(self, "projector_status_text"):
            return
        source = self.source
        if source is None:
            self.projector_status_text.set("红外投影器：连接相机后自动控制")
            return
        if not isinstance(source, AstraRosSource):
            self.projector_status_text.set("红外投影器：当前相机无需软件控制")
            return
        desired_enabled = self._active_stage_index != 1
        if source.laser_enabled is desired_enabled:
            state = (
                "已请求打开（深度模式）"
                if desired_enabled
                else "已请求关闭（请以 IR 画面为准）"
            )
            self.projector_status_text.set("红外投影器：{}".format(state))
            return
        try:
            source.set_laser_enabled(desired_enabled)
            state = (
                "已请求打开（深度模式）"
                if desired_enabled
                else "已请求关闭（请以 IR 画面为准）"
            )
            self.projector_status_text.set("红外投影器：{}".format(state))
            self._set_status(
                "红外投影器{}指令已发送".format(
                    "打开" if desired_enabled else "关闭"
                )
            )
        except Exception as error:
            self.projector_status_text.set("红外投影器控制失败：{}".format(error))
            if required:
                raise
            self._set_status("红外投影器控制失败：{}".format(error))

    def _solve_rgbd_calibration(self):
        if self._auto_calibration_active:
            self._stop_auto_calibration("自动采集已停止，正在计算 RGB-D 标定")
        if self._calibration_pair_root is None:
            self._show_error("无法标定", "请先采集 RGB/IR 图像对")
            return
        try:
            pairs = load_image_pairs(str(self._calibration_pair_root))
            minimum_common = int(
                self.camera_config.get(
                    "minimum_common_corners",
                    self.camera_config.get("minimum_common_charuco_corners", 10),
                )
            )
            result = calibrate_color_from_depth(
                pairs,
                self.color_intrinsics,
                minimum_common_corners=minimum_common,
                target=self.calibration_target,
            )
            self.stereo_result = result
            output = Path(__file__).resolve().parent / "output" / "rgbd_calibration.yaml"
            save_rgbd_calibration(str(output), result.calibration)
            text = (
                "标定成功\n有效图像对：{}\n共同角点：{}\nStereo RMS：{:.4f} px\n"
                "结果：{}"
            ).format(
                result.pairs_used,
                list(result.common_corner_counts),
                result.calibration.rms_reprojection_error_px,
                output,
            )
            self.calibration_log.delete("1.0", "end")
            self.calibration_log.insert("end", text)
            self.rgbd_status_text.set(
                "待写入：Stereo RMS {:.4f} px".format(
                    result.calibration.rms_reprojection_error_px
                )
            )
            self._set_status("RGB-D 标定计算完成，尚未写入中央参数")
        except Exception as error:
            self._show_error("RGB-D 标定失败", error)

    def _write_rgbd_calibration(self):
        if self.stereo_result is None:
            self._show_error("无法写入", "当前没有新计算的 RGB-D 标定结果")
            return
        try:
            suffix = time.strftime("%Y%m%d_%H%M%S")
            backup = update_runtime_calibration(
                self.paths["runtime_calibration"],
                self.stereo_result.calibration,
                suffix,
            )
            self._reload_rgbd_calibration(show_dialog=False)
            self._set_status("RGB-D 标定已写入中央参数；备份：{}".format(backup))
        except Exception as error:
            self._show_error("写入失败", error)

    def _new_capture_session(self):
        calibration = self._effective_rgbd_calibration()
        if calibration is None:
            self._show_error("无法新建", "RGB-D 标定无效")
            return
        if self.yolo_provider is None:
            self._show_error("无法新建", "请先加载 YOLO 分割权重")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_path = Path(self.paths["capture_root"]) / (
            "foundationpose_reference_" + timestamp
        )
        classes = [value.strip() for value in self.target_class_var.get().split(",") if value.strip()]
        self.capture_session = CaptureSession.create(
            str(session_path),
            color_intrinsics=rectified_intrinsics(calibration.color),
            depth_intrinsics=calibration.depth,
            color_from_depth=calibration.color_from_depth,
            calibration_source=calibration.source or self.paths["runtime_calibration"],
            tag_layout_source=self.paths["tag_layout"],
            yolo_weights=self.yolo_weights_var.get(),
            target_class=",".join(classes),
        )
        self.session_var.set(str(session_path))
        self.last_capture_pose = None
        self._captured_object_centroids_workspace = []
        self.fusion_result = None
        self._capture_request_pending = False
        self._last_captured_view_index = None
        self._refresh_capture_gallery_count()
        text = "参考拍照会话已创建，可以拍摄第 1 张"
        self._set_capture_feedback(text, 5.0)
        self._set_status(text)

    def _open_capture_session(self, session_path: str) -> None:
        session = CaptureSession.open(session_path)
        last_pose = None
        centroids = []
        for view in session.iter_views():
            last_pose = view.workspace_from_color.copy()
            centroid = view.metadata.get("object_centroid_workspace_m")
            if centroid is not None:
                point = np.asarray(centroid, dtype=np.float64).reshape(3)
                if np.isfinite(point).all():
                    centroids.append(point)
        self.capture_session = session
        self.session_var.set(str(session.root))
        self.last_capture_pose = last_pose
        self._captured_object_centroids_workspace = centroids
        self._capture_request_pending = False
        self._last_captured_view_index = len(session) - 1 if len(session) else None
        self._refresh_capture_gallery_count()
        text = "已打开参考会话，共 {} 张照片".format(len(session))
        self._set_capture_feedback(text, 5.0)
        self._set_status(text)

    def _capture_view(self):
        if self._capture_request_pending:
            self._cancel_capture_request("已取消本次拍摄等待")
            return
        if self.capture_session is None:
            self._show_error("无法拍摄", "请先点击“新建参考拍照会话”")
            return
        self._capture_request_pending = True
        self._capture_request_deadline_s = time.monotonic() + float(
            self.capture_config.get("capture_request_timeout_s", 5.0)
        )
        self._capture_request_last_reason = "正在等待下一组同步 RGB-D"
        if hasattr(self, "capture_button"):
            self.capture_button.configure(text="取消本次拍摄等待")
        self._set_capture_feedback("正在等待合格的 RGB-D、Tag 和 Mask...", 6.0)
        self._try_pending_capture()

    def _try_pending_capture(self) -> None:
        if not self._capture_request_pending:
            return
        remaining_s = self._capture_request_deadline_s - time.monotonic()
        if remaining_s <= 0.0:
            reason = self._capture_request_last_reason or "没有收到合格帧"
            self._cancel_capture_request("拍摄超时：{}".format(reason), duration_s=6.0)
            return
        try:
            index, coverage, sync_delta_s = self._save_current_capture_view()
        except Exception as error:
            self._capture_request_last_reason = str(error)
            self._set_capture_feedback(
                "等待拍摄 · {} · 剩余 {:.1f} 秒".format(
                    error, max(0.0, remaining_s)
                ),
                max(1.0, remaining_s),
            )
            return
        self._capture_request_pending = False
        if hasattr(self, "capture_button"):
            self.capture_button.configure(text="3  拍摄参考图")
        self._last_captured_view_index = index
        self._refresh_capture_gallery_count()
        text = "已拍摄第 {} 张 · 深度 {:.0%} · 彩深差 {:.0f} ms".format(
            index + 1, coverage, sync_delta_s * 1000.0
        )
        self._set_capture_feedback(text, 6.0)
        self._set_status(text)

    def _save_current_capture_view(self):
        calibration = self._effective_rgbd_calibration()
        bundle = self.analysis_bundle
        rectified_color = self.analysis_rectified_color
        if self.capture_session is None:
            raise ValueError("尚未新建参考拍照会话")
        if bundle is None or bundle.depth_m is None or rectified_color is None:
            raise ValueError("等待成对 RGB/深度")
        analysis_age_s = time.monotonic() - self.analysis_completed_monotonic_s
        if analysis_age_s > self._maximum_analysis_age_s:
            raise ValueError("后台分析暂未跟上")
        if calibration is None:
            raise ValueError("RGB-D 标定无效")
        if self.tag_estimate is None or not self.tag_estimate.valid:
            raise ValueError("画面中没有有效 AprilTag 位姿")
        if not self.mask_result.valid or self.mask_result.mask is None:
            raise ValueError("画面中没有有效 YOLO Mask")
        if bundle.sync_delta_s is None or bundle.sync_delta_s > float(
            self.camera_config["maximum_sync_delta_s"]
        ):
            delta_ms = (
                "--"
                if bundle.sync_delta_s is None
                else "{:.0f} ms".format(bundle.sync_delta_s * 1000.0)
            )
            raise ValueError("等待下一组同步深度（当前 {}）".format(delta_ms))
        if self.aligned_depth is None:
            raise ValueError("深度正在对齐")
        aligned = self.aligned_depth.copy()
        coverage = depth_coverage(aligned, self.mask_result.mask)
        minimum_coverage = float(
            self.capture_config["minimum_mask_depth_coverage"]
        )
        if coverage < minimum_coverage:
            raise ValueError(
                "物体深度覆盖 {:.0%}，需要至少 {:.0%}".format(
                    coverage, minimum_coverage
                )
            )
        pose = self.tag_estimate.workspace_from_camera
        centroid_camera = masked_depth_centroid(
            aligned,
            self.mask_result.mask,
            rectified_intrinsics(calibration.color),
        )
        centroid_workspace = transform_points(
            centroid_camera.reshape(1, 3), pose
        )[0]
        maximum_object_shift_m = float(
            self.capture_config.get("maximum_object_centroid_shift_m", 0.07)
        )
        if self._captured_object_centroids_workspace:
            observed_shift_m = max(
                float(np.linalg.norm(centroid_workspace - previous))
                for previous in self._captured_object_centroids_workspace
            )
            if observed_shift_m > maximum_object_shift_m:
                raise ValueError(
                    "物体相对 Tag 移动 {:.0f} mm；请只移动相机".format(
                        observed_shift_m * 1000.0
                    )
                )
        if self.last_capture_pose is not None:
            translation, rotation_deg = pose_difference(self.last_capture_pose, pose)
            if (
                translation < float(self.capture_config["minimum_translation_m"])
                and rotation_deg < float(self.capture_config["minimum_rotation_deg"])
            ):
                raise ValueError(
                    "请移动相机到新角度（当前变化 {:.0f} mm / {:.1f}°）".format(
                        translation * 1000.0, rotation_deg
                    )
                )
        index = self.capture_session.add_view(
            color_bgr=rectified_color,
            depth_raw_m=bundle.depth_m,
            depth_aligned_m=aligned,
            mask=self.mask_result.mask,
            workspace_from_color=pose,
            timestamp_s=bundle.color_timestamp_s,
            metadata={
                "tag_ids": list(self.tag_estimate.visible_tag_ids),
                "tag_rms_px": self.tag_estimate.rms_reprojection_error_px,
                "yolo_class": self.mask_result.class_name,
                "yolo_confidence": self.mask_result.confidence,
                "mask_depth_coverage": coverage,
                "rgb_depth_sync_delta_s": bundle.sync_delta_s,
                "analysis_age_s": analysis_age_s,
                "object_centroid_workspace_m": centroid_workspace.tolist(),
            },
        )
        self.last_capture_pose = pose.copy()
        self._captured_object_centroids_workspace.append(centroid_workspace.copy())
        return index, coverage, float(bundle.sync_delta_s)

    def _cancel_capture_request(self, text: str, duration_s: float = 3.0) -> None:
        self._capture_request_pending = False
        self._capture_request_deadline_s = 0.0
        if hasattr(self, "capture_button"):
            self.capture_button.configure(text="3  拍摄参考图")
        self._set_capture_feedback(text, duration_s)
        self._set_status(text)

    def _set_capture_feedback(self, text: str, duration_s: float = 3.0) -> None:
        self._capture_feedback_text = str(text)
        self._capture_feedback_until_s = time.monotonic() + float(duration_s)
        if hasattr(self, "capture_status_text"):
            self.capture_status_text.set(self._capture_feedback_text)

    def _refresh_capture_gallery_count(self) -> None:
        count = 0
        session_path = self.session_var.get().strip() if hasattr(self, "session_var") else ""
        if session_path:
            try:
                count = len(CaptureSession.open(session_path))
            except Exception:
                count = 0
        if hasattr(self, "capture_gallery_button"):
            self.capture_gallery_button.configure(
                text="查看已拍照片（{} 张）".format(count)
            )

    def _preview_captured_model(self) -> None:
        if self._busy:
            self._set_status("已有任务正在运行")
            return
        session_path = self.session_var.get().strip()
        if not session_path:
            self._show_error("无法预览", "请先拍摄参考照片")
            return
        try:
            session = CaptureSession.open(session_path)
            count = len(session)
        except Exception as error:
            self._show_error("无法预览", error)
            return
        minimum_views = int(self.fusion_config["minimum_views"])
        if count < minimum_views:
            text = "三维快速预览至少需要 {} 张不同角度照片；当前 {} 张".format(
                minimum_views, count
            )
            self._set_capture_feedback(text, 6.0)
            self._set_status(text)
            return
        self._view_mesh_after_fusion = True
        if hasattr(self, "model_name_var"):
            name = self.reference_object_name_var.get().strip()
            if name:
                self.model_name_var.set(name)
        if hasattr(self, "stage_list"):
            self.stage_list.setCurrentRow(3)
        elif hasattr(self, "control_pages"):
            self._change_stage(3)
        self._fuse_session()

    def _effective_rgbd_calibration(self):
        return self.device_rgbd_calibration or self.rgbd_calibration

    def _pack_foundationpose_reference_zip(self):
        from tkinter import filedialog

        from .session_archive import create_foundationpose_reference_archive

        if self._busy:
            self._set_status("已有任务正在运行")
            return
        session_path = self.session_var.get().strip()
        if not session_path:
            self._show_error("无法打包", "请先新建或选择参考拍照会话")
            return
        session = Path(session_path).expanduser()
        destination = filedialog.asksaveasfilename(
            initialdir=str(session.parent),
            initialfile=session.name + "_foundationpose_reference.zip",
            defaultextension=".zip",
            filetypes=[("FoundationPose reference ZIP", "*.zip")],
        )
        if not destination:
            return
        try:
            object_id = int(self.reference_object_id_var.get())
            object_name = self.reference_object_name_var.get().strip() or "object"
        except ValueError as error:
            self._show_error("无法打包", "参考物体 ID 必须是正整数")
            return
        self._busy = True
        self._set_status("正在生成 FoundationPose 无模型参考 ZIP...")

        def worker():
            try:
                path = create_foundationpose_reference_archive(
                    session_path,
                    destination,
                    object_id=object_id,
                    object_name=object_name,
                )
                self.root.after(0, lambda: self._reference_zip_finished(path))
            except Exception as error:
                self.root.after(
                    0,
                    lambda current=error: self._reference_zip_failed(current),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _reference_zip_finished(self, path):
        self._busy = False
        text = "FoundationPose 无模型参考 ZIP 已生成：{}".format(path)
        self.capture_status_text.set(text)
        self._set_status(text)

    def _reference_zip_failed(self, error):
        self._busy = False
        self._show_error("无模型参考 ZIP 打包失败", error)

    def _fuse_session(self):
        if self._busy:
            return
        session_path = self.session_var.get().strip()
        if not session_path:
            self._show_error("无法重建", "未选择采集会话")
            return
        self._busy = True
        self.mesh_status_text.set("正在执行 TSDF 融合...")

        def worker():
            try:
                result = fuse_session(
                    session_path,
                    voxel_length_m=float(self.voxel_var.get()),
                    sdf_trunc_m=float(self.trunc_var.get()),
                    maximum_depth_m=float(self.fusion_config["maximum_depth_m"]),
                    minimum_views=int(self.fusion_config["minimum_views"]),
                    mask_erosion_pixels=int(self.fusion_config["mask_erosion_pixels"]),
                    simplify_triangles=int(self.fusion_config["simplify_triangles"]),
                    workspace_up=self.fusion_config["workspace_up"],
                    minimum_mask_depth_coverage=float(
                        self.capture_config["minimum_mask_depth_coverage"]
                    ),
                    maximum_object_centroid_shift_m=float(
                        self.capture_config.get(
                            "maximum_object_centroid_shift_m", 0.07
                        )
                    ),
                )
                self.root.after(0, lambda: self._fusion_finished(result))
            except Exception as error:
                self.root.after(0, lambda current=error: self._fusion_failed(current))

        threading.Thread(target=worker, daemon=True).start()

    def _fusion_finished(self, result):
        self._busy = False
        self.fusion_result = result
        mesh_quality = self.fusion_config.get("mesh_quality", {})
        validation = validate_mesh(
            result.mesh,
            minimum_dimensions_m=mesh_quality.get("minimum_dimensions_m"),
            maximum_dimensions_m=mesh_quality.get("maximum_dimensions_m"),
            require_watertight=bool(
                mesh_quality.get("require_watertight", False)
            ),
        )
        dimensions = [value * 1000.0 for value in result.dimensions_m]
        prefix = "融合完成" if validation.valid else "融合完成，但质量门禁未通过"
        text = (
            "{}：{} 个视角，{} 顶点，{} 三角面，尺寸 "
            "{:.1f} x {:.1f} x {:.1f} mm"
        ).format(
            prefix,
            result.views_integrated,
            result.vertex_count,
            result.triangle_count,
            dimensions[0],
            dimensions[1],
            dimensions[2],
        )
        if not validation.valid:
            text += "；{}".format(validation.reason)
        self.mesh_status_text.set(text)
        self._set_status(text)
        if self._view_mesh_after_fusion:
            self._view_mesh_after_fusion = False
            self._view_mesh()

    def _fusion_failed(self, error):
        self._busy = False
        self._view_mesh_after_fusion = False
        self.mesh_status_text.set("重建失败：{}".format(error))
        self._show_error("TSDF 重建失败", error)

    def _view_mesh(self):
        if self.fusion_result is None:
            self._show_error("无法查看", "尚未生成三维网格")
            return

        def worker():
            import open3d as o3d

            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            o3d.visualization.draw_geometries(
                [self.fusion_result.mesh, axes],
                window_name="FoundationPose 模型预览",
                width=1000,
                height=760,
                mesh_show_back_face=True,
            )

        threading.Thread(target=worker, daemon=True).start()

    def _export_model(self):
        if self.fusion_result is None:
            self._show_error("无法导出", "尚未生成三维网格")
            return
        try:
            path, validation = export_foundationpose_model(
                self.fusion_result.mesh,
                self.mesh_root_var.get(),
                self.model_name_var.get(),
                self.fusion_result.workspace_from_object,
                self.session_var.get(),
                quality_config=self.fusion_config.get("mesh_quality"),
            )
            warning = ""
            if validation.warnings:
                warning = "；" + "；".join(validation.warnings)
            text = "FoundationPose 模型已导出：{}{}".format(path, warning)
            self.mesh_status_text.set(text)
            self._set_status(text)
        except Exception as error:
            self._show_error("模型导出失败", error)

    def close(self):
        self._shutdown_capture_analysis()
        if self.source is not None:
            self.source.stop()
            self.source = None
        self.root.destroy()


def check_environment(config_path: str) -> int:
    config = load_config(config_path)
    results = run_checks(
        config["paths"]["foundationpose_root"],
        config["paths"]["runtime_calibration"],
        config["paths"].get("yolo_weights"),
    )
    for result in results:
        print("{} {:<36} {}".format("OK" if result.ok else "MISSING", result.name, result.detail))
    return 0 if all(result.ok for result in results if result.required) else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check", action="store_true", help="run dependency checks without opening the UI")
    parser.add_argument("--screenshot", help="save a UI screenshot and exit")
    parser.add_argument(
        "--stage",
        choices=("environment", "calibration", "capture", "reconstruction"),
        default="environment",
        help="initial workflow page",
    )
    parser.add_argument("--pack-session", help="capture session directory to package")
    parser.add_argument("--archive-output", help="destination ZIP for --pack-session")
    parser.add_argument("--reconstruct-zip", help="capture ZIP to reconstruct without a camera or UI")
    parser.add_argument("--model-name", default="bottle", help="offline reconstruction model name")
    parser.add_argument("--output-root", help="offline model job output directory")
    parser.add_argument("--work-root", help="temporary capture extraction directory")
    parser.add_argument("--result-zip", help="offline CAD result ZIP path")
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument("--voxel-length-m", type=float)
    parser.add_argument("--sdf-trunc-m", type=float)
    args = parser.parse_args(argv)
    if args.check:
        return check_environment(args.config)
    if args.pack_session:
        if not args.archive_output:
            parser.error("--archive-output is required with --pack-session")
        from .session_archive import create_capture_archive

        output = create_capture_archive(args.pack_session, args.archive_output)
        print("capture_zip: {}".format(output))
        return 0
    if args.reconstruct_zip:
        from .offline_reconstruction import reconstruct_capture_archive

        result = reconstruct_capture_archive(
            args.reconstruct_zip,
            args.model_name,
            config_path=args.config,
            output_root=args.output_root,
            work_root=args.work_root,
            result_zip=args.result_zip,
            keep_extracted=args.keep_extracted,
            voxel_length_m=args.voxel_length_m,
            sdf_trunc_m=args.sdf_trunc_m,
        )
        print("status: ok")
        print("model_directory: {}".format(result.model_directory))
        print("model_obj: {}".format(result.model_obj))
        print("result_zip: {}".format(result.result_zip))
        print("views_integrated: {}".format(result.fusion_result.views_integrated))
        return 0
    from .model_builder_qt_ui import run_qt_ui

    stage_indices = {
        "environment": 0,
        "calibration": 1,
        "capture": 2,
        "reconstruction": 3,
    }
    return run_qt_ui(
        args.config,
        screenshot=args.screenshot,
        initial_stage=stage_indices[args.stage],
    )


if __name__ == "__main__":
    raise SystemExit(main())
