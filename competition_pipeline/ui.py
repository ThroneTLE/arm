#!/usr/bin/env python3
"""Competition workbench for one RGB-D camera and an eye-in-hand robot."""

import argparse
from dataclasses import replace
import glob
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import QRectF, QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox,
    QScrollArea, QSizePolicy, QStackedWidget, QStyle, QTableWidget,
    QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from tool.camera_calibration.calib_common import charuco_board, save_camera_yaml
from tool.camera_calibration.calibrate_intrinsics import calibrate, save_report
from tool.object_model_builder.camera_source import (
    AstraRosSource, OakDProSource, OrbbecRosSource,
)
from tool.object_model_builder.rgbd_geometry import RgbdCalibration

from .configuration import CompetitionConfig, load_camera_intrinsics
from .checkerboard_target import CHECKERBOARD_TARGET, CheckerboardTarget
from .controller_state import ControllerState
from .controller_state_reader import ControllerStateReader
from .controller_tcp import InexbotPoint, modbus_client_from_config
from .shape_latch import ShapeLatch
from .tcp_pose import NexBotTcpPoseSource, pose_endpoint_from_config
from .geometry import transform_from_xyz_rpy_mm, xyz_rpy_from_transform
from .hand_eye import APRILTAG_MAP_TARGET, HandEyeCalibrator
from .localization import (
    HybridLocalizer, SOURCE_TAG_VISUAL, SOURCE_TAG_VISUAL_HELD,
    SOURCE_TCP_FALLBACK,
)
from .oak_calibration import (
    export_connected_oak_eeprom, format_oak_summary, import_oak_calibration,
    inspect_oak_calibration,
)
from .object_localization import ObjectCloudSettings, localize_segmented_instances
from .rgbd_calibration import (
    DepthToColorAligner, calibrate_rgb_ir_pairs,
    calibration_for_depth_frame, camera_intrinsics_from_file, depth_colormap,
    detect_charuco, infrared_preview, intrinsics_signature, load_rgbd_result,
    save_rgbd_result,
)
from .sample_store import HandEyeSampleStore
from .segmentation_validation import SegmentationModel, file_sha256
from .tag_map import TagMap
from .rviz_visualization import (
    CompetitionRvizVisualizer, ensure_ros_master, launch_rviz, stop_process,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "competition.yaml"
TARGET_PDF = ROOT.parent / "tool" / "camera_calibration" / "targets" / "charuco_intrinsics_A4.pdf"


class VideoCanvas(QWidget):
    def __init__(self, empty_text, minimum=(250, 180)):
        super().__init__()
        self.image = QImage()
        self.empty_text = empty_text
        self.setMinimumSize(*minimum)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, frame):
        image = np.asarray(frame)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self.image = QImage(
            rgb.data, width, height, channels * width, QImage.Format_RGB888
        ).copy()
        self.update()

    def clear(self, text=None):
        self.image = QImage()
        if text is not None:
            self.empty_text = text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#14191d"))
        if self.image.isNull():
            painter.setPen(QColor("#aab3ba"))
            font = QFont()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, self.empty_text)
            return
        target = QSize(max(1, self.width() - 18), max(1, self.height() - 18))
        scaled = self.image.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawImage(
            QRectF(
                (self.width() - scaled.width()) / 2.0,
                (self.height() - scaled.height()) / 2.0,
                scaled.width(), scaled.height(),
            ),
            scaled,
        )


class RgbdCameraWorker(QThread):
    bundle_ready = pyqtSignal(object)
    connected = pyqtSignal()
    laser_changed = pyqtSignal(bool)
    failed = pyqtSignal(str)

    def __init__(
        self, camera_config, color_device=None,
        initial_laser_enabled=True, initial_anchor="color",
    ):
        super().__init__()
        self.camera_config = dict(camera_config)
        self.color_device = str(color_device)
        self.stopping = False
        self.laser_lock = threading.Lock()
        self.requested_laser = bool(initial_laser_enabled)
        self.requested_anchor = str(initial_anchor)
        self.applied_laser = None

    def stop(self):
        self.stopping = True

    def request_laser(self, enabled, anchor=None):
        with self.laser_lock:
            self.requested_laser = bool(enabled)
            if anchor is not None:
                if anchor not in ("depth", "ir", "color"):
                    raise ValueError("unsupported RGB-D stream anchor: {}".format(anchor))
                self.requested_anchor = str(anchor)

    def run(self):
        source = None
        try:
            backend = str(self.camera_config.get("backend"))
            if backend == "astra_ros":
                driver = self.camera_config.get("ros_driver", {})
                source = AstraRosSource(
                    color_device=self.color_device,
                    color_width=int(self.camera_config["color_width"]),
                    color_height=int(self.camera_config["color_height"]),
                    color_fps=int(self.camera_config["color_fps"]),
                    color_fourcc=str(self.camera_config.get("color_fourcc", "MJPG")),
                    depth_topic=str(self.camera_config["depth_topic"]),
                    depth_info_topic=str(self.camera_config["depth_info_topic"]),
                    ir_topic=str(self.camera_config["ir_topic"]),
                    start_ros_driver=bool(self.camera_config.get("start_ros_driver", True)),
                    driver_log_path=str(ROOT / "output" / "astra_driver.log"),
                    driver_package=str(driver.get("package", "astra_camera")),
                    driver_launch_file=str(driver.get("launch_file", "astra.launch")),
                    driver_arguments=driver.get("arguments"),
                    driver_startup_timeout_s=float(driver.get("startup_timeout_s", 2.0)),
                    laser_service=str(self.camera_config.get("laser_service", "/camera/set_laser")),
                    ros_node_name="competition_calibration_ui",
                )
            elif backend == "oak_depthai":
                source = OakDProSource(
                    color_width=int(self.camera_config["color_width"]),
                    color_height=int(self.camera_config["color_height"]),
                    fps=int(self.camera_config["color_fps"]),
                    mxid=str(self.camera_config.get("mxid", "")),
                    dot_projector_mA=int(self.camera_config.get("dot_projector_mA", 800)),
                    floodlight_mA=int(self.camera_config.get("floodlight_mA", 0)),
                    mono_resolution=str(self.camera_config.get("mono_resolution", "800p")),
                    extended_disparity=bool(self.camera_config.get("extended_disparity", True)),
                    subpixel=bool(self.camera_config.get("subpixel", False)),
                    left_right_check=bool(self.camera_config.get("left_right_check", True)),
                    focus_mode=str(self.camera_config.get("focus_mode", "device_default")),
                    manual_focus=self.camera_config.get("manual_focus"),
                )
            elif backend == "orbbec_ros":
                driver = self.camera_config.get("ros_driver", {})
                source = OrbbecRosSource(
                    color_topic=str(self.camera_config["color_topic"]),
                    color_info_topic=str(self.camera_config["color_info_topic"]),
                    depth_topic=str(self.camera_config["depth_topic"]),
                    depth_info_topic=str(self.camera_config["depth_info_topic"]),
                    ir_topic=str(self.camera_config["ir_topic"]),
                    start_ros_driver=bool(self.camera_config.get("start_ros_driver", True)),
                    driver_log_path=str(ROOT / "output" / "orbbec_driver.log"),
                    driver_package=str(driver.get("package", "orbbec_camera")),
                    driver_launch_file=str(driver.get("launch_file", "gemini.launch")),
                    driver_arguments=driver.get("arguments"),
                    driver_startup_timeout_s=float(driver.get("startup_timeout_s", 3.0)),
                    laser_service=str(self.camera_config.get("laser_service", "/camera/set_laser")),
                    depth_aligned_to_color=bool(
                        self.camera_config.get("depth_aligned_to_color", True)
                    ),
                    expected_serial=self.camera_config.get("expected_serial"),
                    ros_node_name="competition_orbbec_gemini",
                )
            else:
                raise ValueError("unsupported camera backend: {}".format(backend))
            source.start()
            self.connected.emit()
            last_key = None
            while not self.stopping:
                with self.laser_lock:
                    requested = self.requested_laser
                    anchor = self.requested_anchor
                if requested is not self.applied_laser:
                    source.set_laser_enabled(requested)
                    self.applied_laser = requested
                    self.laser_changed.emit(requested)
                # Histories contain immutable frame arrays.  Passing references
                # avoids copying the same 1280x1024 Depth/IR frames at RGB rate;
                # consumers that retain/modify an image make their own copy.
                bundle = source.latest(anchor=anchor, copy_frames=False)
                if bundle is not None:
                    key = (
                        bundle.color_timestamp_s,
                        bundle.ir_timestamp_s,
                        bundle.depth_timestamp_s,
                    )
                    if key != last_key:
                        self.bundle_ready.emit(bundle)
                        last_key = key
                self.msleep(12)
        except Exception as error:
            if not self.stopping:
                self.failed.emit(str(error))
        finally:
            if source is not None:
                source.stop()


class ControllerStateWorker(QThread):
    """Read-only controller status worker used by the field test page."""

    state_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, settings, interval_s=0.5):
        super().__init__()
        self.settings = dict(settings)
        self.interval_s = max(float(interval_s), 0.1)
        self.stopping = False
        self.client = None
        self.shape_latch = ShapeLatch(settings.get("initial_shape"))

    def stop(self):
        self.stopping = True
        if self.client is not None:
            self.client.close()

    def run(self):
        try:
            if not bool(self.settings.get("enabled", False)):
                self.state_ready.emit(ControllerState(connected=False, error="controller.enabled=false（只读测试未启用）"))
                return
            self.client = modbus_client_from_config({"controller": self.settings})
            if self.client is None:
                raise RuntimeError("controller configuration is disabled or incomplete")
            self.client.connect()
            reader = ControllerStateReader(self.client, {"controller": self.settings})
            while not self.stopping:
                try:
                    state = reader.read()
                    latch = self.shape_latch.observe(state.shape)
                    self.state_ready.emit(replace(
                        state,
                        initial_shape=latch.initial_shape,
                        shape_changed=latch.changed,
                        raw_registers={
                            **state.raw_registers,
                            "observed_shape": latch.observed_shape,
                            "initial_shape": latch.initial_shape,
                            "shape_changed": latch.changed,
                        },
                    ))
                except Exception as error:
                    self.state_ready.emit(ControllerState(connected=False, error=str(error)))
                self.msleep(int(self.interval_s * 1000.0))
        except Exception as error:
            if not self.stopping:
                self.failed.emit(str(error))
        finally:
            if self.client is not None:
                self.client.close()
                self.client = None


class NexBotPoseWorker(QThread):
    """Live controller TCP pose via the official NexBot 7000-port protocol.

    Used by the hand-eye page so ``T_base_tcp`` is read from the controller
    directly instead of transcribed from the teach pendant.  This is the
    verification step placed before every hand-eye sample: connect first,
    confirm the pose matches the teach pendant, then sample.
    """

    pose_ready = pyqtSignal(bool, object)
    failed = pyqtSignal(str)

    def __init__(self, endpoint, interval_s=0.2):
        super().__init__()
        self.endpoint = endpoint
        self.interval_s = max(float(interval_s), 0.1)
        self.stopping = False
        self.source = None

    def stop(self):
        self.stopping = True
        if self.source is not None:
            self.source.close()

    def run(self):
        try:
            self.source = NexBotTcpPoseSource(self.endpoint)
            while not self.stopping:
                try:
                    xyz_mm, rpy_deg = self.source.read()
                    if self.stopping:
                        break
                    self.pose_ready.emit(True, (tuple(xyz_mm), tuple(rpy_deg)))
                except Exception as error:
                    if self.stopping:
                        break
                    self.pose_ready.emit(False, str(error))
                self.msleep(int(self.interval_s * 1000.0))
        except Exception as error:
            if not self.stopping:
                self.failed.emit(str(error))
        finally:
            if self.source is not None:
                self.source.close()
                self.source = None


class TagLocalizationWorker(QThread):
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config_path):
        super().__init__()
        self.config_path = Path(config_path).expanduser().resolve()
        self.stopping = False
        self.pending_lock = threading.Lock()
        self.pending = None

    def submit(
        self, frame, timestamp_s, base_from_tcp=None,
        robot_timestamp_s=None, hide_tags=False,
    ):
        with self.pending_lock:
            self.pending = {
                "frame": np.asarray(frame).copy(),
                "timestamp_s": float(timestamp_s),
                "base_from_tcp": base_from_tcp,
                "robot_timestamp_s": robot_timestamp_s,
                "hide_tags": bool(hide_tags),
            }

    def stop(self):
        self.stopping = True

    def run(self):
        try:
            config = CompetitionConfig(self.config_path)
            matrix, distortion, size = load_camera_intrinsics(
                config.resolve_path(config.camera["color_intrinsics_file"])
            )
            localizer = HybridLocalizer(config)
            while not self.stopping:
                with self.pending_lock:
                    pending = self.pending
                    self.pending = None
                if pending is None:
                    self.msleep(5)
                    continue
                frame = pending["frame"]
                if (frame.shape[1], frame.shape[0]) != size:
                    raise ValueError("RGB 分辨率与当前定位内参不一致")
                started_s = time.perf_counter()
                use_tags = bool(
                    config.data["localization"].get("use_apriltag_runtime", False)
                )
                detections = localizer.visual.detect(frame) if use_tags else {}
                pose = localizer.localize(
                    frame,
                    matrix,
                    distortion,
                    base_from_tcp=pending["base_from_tcp"],
                    image_timestamp_s=pending["timestamp_s"],
                    robot_timestamp_s=pending["robot_timestamp_s"],
                    detections_override={} if pending["hide_tags"] else detections,
                )
                self.result_ready.emit({
                    "frame": frame,
                    "timestamp_s": pending["timestamp_s"],
                    "detections": detections,
                    "pose": pose,
                    "elapsed_ms": (time.perf_counter() - started_s) * 1000.0,
                })
        except Exception as error:
            if not self.stopping:
                self.failed.emit(str(error))


class CheckerboardPreviewWorker(QThread):
    """Run checkerboard detection/PnP off the UI thread for live hand-eye preview."""

    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config_path):
        super().__init__()
        self.config_path = Path(config_path).expanduser().resolve()
        self.stopping = False
        self.pending_lock = threading.Lock()
        self.pending = None

    def submit(self, frame, timestamp_s, target_settings):
        with self.pending_lock:
            # Keep only the newest frame so a slow detector never builds a backlog.
            self.pending = {
                "frame": np.asarray(frame).copy(),
                "timestamp_s": float(timestamp_s),
                "target": dict(target_settings or {}),
            }

    def stop(self):
        self.stopping = True

    def run(self):
        try:
            config = CompetitionConfig(self.config_path)
            while not self.stopping:
                with self.pending_lock:
                    pending = self.pending
                    self.pending = None
                if pending is None:
                    self.msleep(8)
                    continue
                frame = pending["frame"]
                started_s = time.perf_counter()
                try:
                    matrix, distortion, size = load_camera_intrinsics(
                        config.resolve_path(config.camera["color_intrinsics_file"])
                    )
                    checkerboard = CheckerboardTarget(pending["target"])
                    if (frame.shape[1], frame.shape[0]) != size:
                        raise ValueError("当前 RGB 分辨率与内参不一致")
                    observation = checkerboard.estimate(frame, matrix, distortion)
                    preview = checkerboard.draw(frame, observation)
                    self.result_ready.emit({
                        "timestamp_s": pending["timestamp_s"],
                        "preview": preview,
                        "observation": observation,
                        "corner_count": checkerboard.corner_count,
                        "found": (
                            checkerboard.corner_count
                            if observation.corners is not None else 0
                        ),
                        "elapsed_ms": (time.perf_counter() - started_s) * 1000.0,
                    })
                except Exception as error:
                    self.result_ready.emit({
                        "timestamp_s": pending["timestamp_s"],
                        "preview": np.asarray(frame).copy(),
                        "error": str(error),
                    })
        except Exception as error:
            if not self.stopping:
                self.failed.emit(str(error))


class DepthPreviewWorker(QThread):
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config_path):
        super().__init__()
        self.config_path = Path(config_path).expanduser().resolve()
        self.stopping = False
        self.pending_lock = threading.Lock()
        self.pending = None

    def submit(self, bundle):
        with self.pending_lock:
            # Frame arrays are immutable snapshots owned by their history or
            # by this bundle, so only the latest reference needs to be retained.
            self.pending = bundle

    def stop(self):
        self.stopping = True

    def run(self):
        try:
            config = CompetitionConfig(self.config_path)
            camera = config.camera
            calibration = None
            calibration_key = None
            aligner = None
            aligner_signature = None
            while not self.stopping:
                with self.pending_lock:
                    bundle = self.pending
                    self.pending = None
                if bundle is None:
                    self.msleep(8)
                    continue
                if bundle.depth_m is None:
                    continue
                try:
                    started_s = time.perf_counter()
                    if bundle.depth_aligned_to_color:
                        aligned = np.asarray(bundle.depth_m)
                        caption = "DepthAI 已对齐到 RGB"
                    else:
                        path = config.resolve_path(camera["rgbd_calibration_file"])
                        key = (path, path.stat().st_mtime)
                        if calibration is None or key != calibration_key:
                            calibration = load_rgbd_result(
                                path, camera["max_rgbd_rms_px"]
                            )
                            calibration_key = key
                            aligner = None
                            aligner_signature = None
                        runtime = calibration_for_depth_frame(
                            calibration,
                            bundle.depth_intrinsics,
                            bundle.depth_m.shape,
                        )
                        signature = intrinsics_signature(runtime.depth)
                        if aligner is None or signature != aligner_signature:
                            aligner = DepthToColorAligner(runtime)
                            aligner_signature = signature
                        aligned = aligner.align(
                            bundle.depth_m,
                            splat_radius_pixels=int(
                                camera.get("alignment_splat_radius_pixels", 0)
                            ),
                        )
                        caption = "对齐到 RGB 的 Depth · {}x{}".format(
                            bundle.depth_m.shape[1], bundle.depth_m.shape[0]
                        )
                    self.result_ready.emit({
                        "stamp": bundle.depth_timestamp_s,
                        "preview": depth_colormap(aligned),
                        "caption": caption,
                        "elapsed_ms": (time.perf_counter() - started_s) * 1000.0,
                    })
                except Exception as error:
                    if not self.stopping:
                        self.failed.emit(str(error))
        except Exception as error:
            if not self.stopping:
                self.failed.emit(str(error))


class SegmentationWorker(QThread):
    model_ready = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = dict(settings)
        self.stopping = False
        self.frame_lock = threading.Lock()
        self.pending_frame = None

    def submit(self, frame, timestamp_s, context=None):
        with self.frame_lock:
            # Keep only the newest frame. Inference must never build a stale
            # backlog behind the live camera.
            self.pending_frame = (
                np.asarray(frame).copy(), float(timestamp_s), context
            )

    def stop(self):
        self.stopping = True

    def run(self):
        try:
            model = SegmentationModel(
                self.settings["weights_file"],
                target_classes=self.settings.get("target_classes", []),
                confidence_threshold=float(self.settings["confidence_threshold"]),
                device=str(self.settings.get("device", "auto")),
                iou_threshold=float(self.settings.get("iou_threshold", 0.45)),
                image_size=int(self.settings.get("image_size", 640)),
                agnostic_nms=bool(self.settings.get("agnostic_nms", True)),
                deduplicate_instances=bool(
                    self.settings.get("deduplicate_instances", True)
                ),
                duplicate_mask_iou_threshold=float(
                    self.settings.get("duplicate_mask_iou_threshold", 0.50)
                ),
                duplicate_mask_containment_threshold=float(
                    self.settings.get(
                        "duplicate_mask_containment_threshold", 0.80
                    )
                ),
                duplicate_center_distance_ratio=float(
                    self.settings.get("duplicate_center_distance_ratio", 0.35)
                ),
                duplicate_confidence_tie_margin=float(
                    self.settings.get("duplicate_confidence_tie_margin", 0.05)
                ),
                maximum_detections=int(
                    self.settings.get("maximum_detections", 50)
                ),
            )
            self.model_ready.emit(model.weights)
            gates = {
                "minimum_confidence": float(self.settings["minimum_confidence"]),
                "minimum_mask_area_ratio": float(self.settings["minimum_mask_area_ratio"]),
                "maximum_mask_area_ratio": float(self.settings["maximum_mask_area_ratio"]),
            }
            while not self.stopping:
                with self.frame_lock:
                    pending = self.pending_frame
                    self.pending_frame = None
                if pending is None:
                    self.msleep(8)
                    continue
                frame, timestamp_s, context = pending
                (
                    result, quality, overlay, elapsed_ms, instances, statistics
                ) = model.predict(frame, gates)
                self.result_ready.emit(
                    {
                        "timestamp_s": timestamp_s,
                        "result": result,
                        "quality": quality,
                        "overlay": overlay,
                        "elapsed_ms": elapsed_ms,
                        "instances": instances,
                        "statistics": statistics,
                        "context": context,
                    }
                )
        except Exception as error:
            if not self.stopping:
                self.failed.emit(str(error))


class RvizVisualizationWorker(QThread):
    ready = pyqtSignal()
    observation_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config_path):
        super().__init__()
        self.config_path = Path(config_path).expanduser().resolve()
        self.stopping = False
        self.pending_lock = threading.Lock()
        self.pending = None
        self.pending_pose = None
        self.pending_image = None

    def submit(self, payload):
        with self.pending_lock:
            # As with inference, visualization should never process a stale queue.
            self.pending = payload
            self.pending_image = payload

    def submit_pose(self, pose):
        if pose is None or not pose.valid:
            return
        with self.pending_lock:
            self.pending_pose = pose

    def stop(self):
        self.stopping = True

    @staticmethod
    def _settings(config):
        entry = config.data.get("planning_validation", {}).get(
            "visualization", {}
        )
        safety = config.data.get("safety", {})
        return ObjectCloudSettings(
            minimum_depth_m=float(entry.get("minimum_depth_m", 0.10)),
            maximum_depth_m=float(entry.get("maximum_depth_m", 3.0)),
            minimum_depth_coverage=float(
                entry.get("minimum_depth_coverage", 0.35)
            ),
            minimum_valid_points=int(entry.get("minimum_valid_points", 80)),
            mask_erosion_pixels=int(entry.get("mask_erosion_pixels", 2)),
            keep_largest_depth_component=bool(
                entry.get("keep_largest_depth_component", True)
            ),
            maximum_points_per_instance=int(
                entry.get("maximum_points_per_instance", 4000)
            ),
            workspace_min_m=tuple(
                np.asarray(safety["workspace_min_mm"], dtype=np.float64) / 1000.0
            ),
            workspace_max_m=tuple(
                np.asarray(safety["workspace_max_mm"], dtype=np.float64) / 1000.0
            ),
            assume_supported_objects=bool(
                entry.get("assume_objects_on_support_plane", True)
            ),
            support_plane_z_m=(
                float(entry.get("support_plane_z_mm", 0.0)) / 1000.0
                if entry.get("support_plane_z_mm") is not None else None
            ),
            maximum_support_gap_m=float(
                entry.get("maximum_support_gap_mm", 150.0)
            ) / 1000.0,
            support_plane_tolerance_m=float(
                entry.get("support_plane_tolerance_mm", 30.0)
            ) / 1000.0,
        )

    def run(self):
        visualizer = None
        try:
            config = CompetitionConfig(self.config_path)
            matrix, distortion, size = load_camera_intrinsics(
                config.resolve_path(config.camera["color_intrinsics_file"])
            )
            active_intrinsics = camera_intrinsics_from_file(
                config.resolve_path(config.camera["color_intrinsics_file"])
            )
            aligner = None
            aligner_depth_signature = None
            active_calibration = None
            intrinsics_warning = ""
            if config.camera["backend"] == "astra_ros":
                calibration = load_rgbd_result(
                    config.resolve_path(config.camera["rgbd_calibration_file"]),
                    config.camera["max_rgbd_rms_px"],
                )
                focal_delta = max(
                    abs(calibration.color.matrix[0, 0] / matrix[0, 0] - 1.0),
                    abs(calibration.color.matrix[1, 1] / matrix[1, 1] - 1.0),
                )
                principal_delta = float(
                    np.max(np.abs(calibration.color.matrix[:2, 2] - matrix[:2, 2]))
                )
                if focal_delta > 0.02 or principal_delta > 10.0:
                    intrinsics_warning = (
                        "RGB-D 文件内 RGB 内参与当前 RGB 内参不一致；"
                        "已按当前内参重投影，请尽快重新做 RGB-IR 标定"
                    )
                active_calibration = RgbdCalibration(
                    color=active_intrinsics,
                    depth=calibration.depth,
                    color_from_depth=calibration.color_from_depth,
                    valid=calibration.valid,
                    source=calibration.source,
                    rms_reprojection_error_px=calibration.rms_reprojection_error_px,
                )
                # The actual Depth mode is known only after the first frame and
                # CameraInfo arrive. Build the heavy projection cache lazily.
                aligner = None
            localizer = HybridLocalizer(config)
            settings = self._settings(config)
            visualizer = CompetitionRvizVisualizer(config)
            visualization = config.data.get("planning_validation", {}).get(
                "visualization", {}
            )
            point_cloud_fps = float(
                visualization.get("maximum_update_fps", 5.0)
            )
            if config.camera.get("backend") == "astra_ros":
                mode = config.camera.get("depth_modes", {}).get(
                    config.camera.get("depth_mode"), {}
                )
                point_cloud_fps = float(
                    mode.get("point_cloud_fps", point_cloud_fps)
                )
            point_cloud_interval_s = 1.0 / max(point_cloud_fps, 0.1)
            last_observation_s = 0.0
            self.ready.emit()
            while not self.stopping:
                with self.pending_lock:
                    pending_pose = self.pending_pose
                    self.pending_pose = None
                    pending_image = self.pending_image
                    self.pending_image = None
                if pending_pose is not None:
                    visualizer.publish_camera_pose(
                        pending_pose.base_from_camera
                    )
                if pending_image is not None:
                    image_context = pending_image.get("context") or {}
                    image_bundle = image_context.get("bundle")
                    frame_intrinsics = (
                        None if image_bundle is None
                        else image_bundle.color_intrinsics
                    ) or active_intrinsics
                    overlay = pending_image.get("overlay")
                    if overlay is not None:
                        visualizer.publish_camera_image(
                            overlay, frame_intrinsics
                        )
                remaining = point_cloud_interval_s - (
                    time.monotonic() - last_observation_s
                )
                if remaining > 0.0:
                    self.msleep(max(1, min(20, int(remaining * 1000.0))))
                    continue
                with self.pending_lock:
                    payload = self.pending
                    self.pending = None
                if payload is None:
                    self.msleep(10)
                    continue
                last_observation_s = time.monotonic()
                context = payload.get("context") or {}
                bundle = context.get("bundle")
                if bundle is None or bundle.depth_m is None:
                    visualizer.publish_invalid("当前分割帧没有 Depth")
                    self.observation_ready.emit({"valid": False, "reason": "当前分割帧没有 Depth"})
                    continue
                maximum_sync = float(
                    config.camera.get("maximum_sync_delta_s", 0.20)
                )
                if bundle.sync_delta_s is None or bundle.sync_delta_s > maximum_sync:
                    reason = "RGB/Depth 时间差超过 {:.0f} ms".format(
                        maximum_sync * 1000.0
                    )
                    visualizer.publish_invalid(reason)
                    self.observation_ready.emit({"valid": False, "reason": reason})
                    continue
                if (bundle.color_bgr.shape[1], bundle.color_bgr.shape[0]) != size:
                    raise ValueError("RGB 分辨率与当前内参不一致")
                pose = context.get("localized_pose")
                maximum_cached_age = float(
                    config.data.get("localization", {}).get(
                        "cached_pose_max_age_s", 0.25
                    )
                )
                cache_is_current = (
                    pose is not None
                    and pose.valid
                    and abs(
                        float(bundle.color_timestamp_s) - float(pose.timestamp_s)
                    ) <= maximum_cached_age
                )
                if not cache_is_current:
                    pose = localizer.localize(
                        bundle.color_bgr,
                        matrix,
                        distortion,
                        base_from_tcp=context.get("base_from_tcp"),
                        image_timestamp_s=float(bundle.color_timestamp_s),
                        robot_timestamp_s=context.get("robot_timestamp_s"),
                        detections_override=(
                            {} if context.get("hide_tags", False) else None
                        ),
                    )
                if not pose.valid:
                    reason = "相机定位无效：{}".format(pose.reason)
                    visualizer.publish_invalid(reason)
                    self.observation_ready.emit({"valid": False, "reason": reason})
                    continue
                if bundle.depth_aligned_to_color:
                    aligned_depth = np.asarray(bundle.depth_m).copy()
                elif active_calibration is not None:
                    runtime_calibration = calibration_for_depth_frame(
                        active_calibration,
                        bundle.depth_intrinsics,
                        bundle.depth_m.shape,
                    )
                    runtime_signature = intrinsics_signature(
                        runtime_calibration.depth
                    )
                    if (
                        aligner is None
                        or aligner_depth_signature != runtime_signature
                    ):
                        aligner = DepthToColorAligner(runtime_calibration)
                        aligner_depth_signature = runtime_signature
                    aligned_depth = aligner.align(
                        bundle.depth_m,
                        minimum_depth_m=settings.minimum_depth_m,
                        maximum_depth_m=settings.maximum_depth_m,
                        splat_radius_pixels=int(
                            config.camera.get(
                                "alignment_splat_radius_pixels", 0
                            )
                        ),
                    )
                else:
                    raise ValueError("当前相机没有可用的 Depth→RGB 对齐方式")
                if aligned_depth.shape != bundle.color_bgr.shape[:2]:
                    raise ValueError("对齐 Depth 尺寸与 RGB 不一致")
                frame_intrinsics = bundle.color_intrinsics or active_intrinsics
                objects = localize_segmented_instances(
                    aligned_depth,
                    payload.get("instances", []),
                    frame_intrinsics,
                    pose.base_from_camera,
                    settings,
                )
                diagnostics = {
                    "pose_source": pose.source,
                    "pose_reason": pose.reason,
                    "visible_tag_ids": list(pose.visible_tag_ids),
                    "used_tag_ids": list(pose.used_tag_ids),
                    "tag_rms_px": pose.rms_reprojection_error_px,
                    "sync_delta_ms": float(bundle.sync_delta_s * 1000.0),
                    "intrinsics_warning": intrinsics_warning,
                    "pose_cache_reused": bool(cache_is_current),
                }
                visualizer.publish_observation(
                    pose.base_from_camera,
                    objects,
                    diagnostics=diagnostics,
                    annotated_bgr=None,
                    camera_intrinsics=None,
                )
                valid_count = sum(item.valid for item in objects)
                rejected = [item.reason for item in objects if not item.valid]
                self.observation_ready.emit({
                    "valid": valid_count > 0,
                    "object_count": int(valid_count),
                    "support_constrained_count": int(sum(
                        item.valid and item.support_constrained for item in objects
                    )),
                    "pose_source": pose.source,
                    "warning": intrinsics_warning,
                    "rejected": rejected,
                    "reason": (
                        "已发布 {} 个物体点云".format(valid_count)
                        if valid_count else "没有物体通过 Depth 点云质量门"
                    ),
                })
        except Exception as error:
            if not self.stopping:
                self.failed.emit(str(error))
        finally:
            if visualizer is not None:
                visualizer.close()


def action_button(parent, text, icon, primary=False):
    button = QPushButton(text, parent)
    button.setObjectName("primaryButton" if primary else "secondaryButton")
    button.setIcon(parent.style().standardIcon(getattr(QStyle, icon)))
    button.setMinimumHeight(36)
    return button


class CompetitionCalibrationWindow(QMainWindow):
    RGB, RGBD, TAGS, HAND_EYE, LOCALIZATION, SEGMENTATION, PLANNING, GRASP, CONTROLLER = range(9)

    def __init__(self, config_path=CONFIG_PATH, initial_stage=0, auto_connect=False):
        super().__init__()
        self.setWindowTitle("RGB-D 机械臂比赛工作台")
        screen = QApplication.primaryScreen()
        available = None if screen is None else screen.availableGeometry()
        self.setMinimumSize(960, 600)
        self.resize(
            1540 if available is None else min(1540, max(960, available.width() - 32)),
            920 if available is None else min(920, max(600, available.height() - 32)),
        )
        self.config = CompetitionConfig(config_path)
        self.sample_store = HandEyeSampleStore(self._sample_path(), self.config)
        self.camera_worker = None
        self.tag_pose_worker = None
        self.depth_preview_worker = None
        self.checkerboard_preview_worker = None
        self.segmentation_worker = None
        self.segmentation_model_ready = False
        self.segmentation_last_submit_s = 0.0
        self.segmentation_consecutive_valid = 0
        self.segmentation_last_result = None
        self.rviz_worker = None
        self.rviz_process = None
        self.roscore_process = None
        self.controller_worker = None
        self.controller_state = ControllerState()
        self.camera_connected = False
        self.bundle = None
        self.current_stage = 0
        self.board = charuco_board()
        self.board_points = np.asarray(self.board.getChessboardCorners(), dtype=np.float32)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        self.color_detection = self._empty_detection()
        self.ir_detection = self._empty_detection()
        self.tag_localizer = None
        self.hybrid_localizer = None
        self.color_intrinsics_cache = None
        self.orbbec_intrinsics_synced = False
        self.depth_aligner = None
        self.depth_aligner_source = None
        self.last_depth_stamp = None
        self.last_depth_preview_s = 0.0
        self.last_aligned_preview = None
        self.last_ir_preview_stamp = None
        self.last_ir_preview_s = 0.0
        self.last_raw_depth_preview_stamp = None
        self.last_tag_submit_s = 0.0
        self.latest_tag_payload = None
        self.last_stage_process_s = {}
        self.rgb_object_points = []
        self.rgb_image_points = []
        self.rgb_all_pixels = []
        self.rgb_image_size = None
        self.rgbd_pairs = []
        self._build_ui()
        self._apply_style()
        self._apply_camera_profile_ui()
        self._load_tag_table()
        self._refresh_samples()
        self._refresh_readiness()
        self.stage_list.setCurrentRow(initial_stage)
        self._log("RGB-D 比赛配置已载入")
        if auto_connect:
            QTimer.singleShot(0, self.toggle_camera)

    def _sample_path(self):
        return self.config.resolve_path(self.config.camera["hand_eye_samples_file"])

    def _color_output_path(self):
        return self.config.resolve_path(self.config.camera["color_intrinsics_file"])

    def _rgbd_output_path(self):
        value = self.config.camera.get("rgbd_calibration_file")
        return None if not value else self.config.resolve_path(value)

    def _oak_factory_output_path(self):
        value = self.config.camera.get("factory_calibration_file")
        return None if not value else self.config.resolve_path(value)

    @staticmethod
    def _empty_detection():
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_preview(), 1)
        body.addWidget(self._build_controls())
        outer.addLayout(body, 1)
        self.statusBar().showMessage("就绪")
        self.statusBar().setSizeGripEnabled(False)

    def _build_header(self):
        frame = QFrame()
        frame.setObjectName("header")
        frame.setFixedHeight(66)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(24, 0, 20, 0)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        title = QLabel("RGB-D 机械臂比赛工作台")
        title.setObjectName("appTitle")
        subtitle = QLabel("相机标定 · AprilTag/TCP 定位 · 实例分割 · 规划与抓取验证")
        subtitle.setObjectName("appSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        layout.addLayout(titles)
        layout.addStretch()
        self.connection_badge = QLabel("未连接")
        self.connection_badge.setObjectName("connectionBadge")
        self.connect_button = action_button(self, "连接深度相机", "SP_MediaPlay", True)
        self.connect_button.clicked.connect(self.toggle_camera)
        output = QToolButton()
        output.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        output.setToolTip("打开比赛输出目录")
        output.setFixedSize(36, 36)
        output.clicked.connect(self.open_output)
        layout.addWidget(self.connection_badge)
        layout.addSpacing(10)
        layout.addWidget(self.connect_button)
        layout.addWidget(output)
        return frame

    def _build_sidebar(self):
        frame = QFrame()
        frame.setObjectName("sidebar")
        frame.setFixedWidth(250)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setObjectName("sidebarScroll")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_content = QWidget()
        sidebar_scroll.setWidget(sidebar_content)
        frame_layout.addWidget(sidebar_scroll)
        layout = QVBoxLayout(sidebar_content)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(9)
        label = QLabel("比赛流程")
        label.setObjectName("sectionLabel")
        layout.addWidget(label)
        self.stage_list = QListWidget()
        self.stage_list.setObjectName("stageList")
        self.stage_list.setSpacing(3)
        self.stage_list.setFixedHeight(350)
        self.stage_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for text in (
            "01  RGB 内参", "02  RGB-IR 标定", "03  Tag 地图",
            "04  眼在手上", "05  定位验证", "06  分割验证",
            "07  抓取规划", "08  抓取执行", "09  控制器/TCP 测试",
        ):
            item = QListWidgetItem(text)
            item.setSizeHint(QSize(210, 32))
            self.stage_list.addItem(item)
        self.stage_list.currentRowChanged.connect(self.change_stage)
        layout.addWidget(self.stage_list)
        source = QLabel("相机配置")
        source.setObjectName("sectionLabel")
        layout.addWidget(source)
        self.camera_profile = QComboBox()
        for profile_name, profile in self.config.camera_profiles.items():
            self.camera_profile.addItem(
                str(profile.get("label", profile_name)), profile_name
            )
        active_index = self.camera_profile.findData(self.config.active_camera_profile)
        self.camera_profile.setCurrentIndex(max(0, active_index))
        self.camera_profile.currentIndexChanged.connect(self.change_camera_profile)
        layout.addWidget(self.camera_profile)
        self.depth_mode_label = QLabel("Depth / IR 模式")
        layout.addWidget(self.depth_mode_label)
        self.depth_mode = QComboBox()
        self.depth_mode.setToolTip(
            "640×480 延迟更低；切换模式前会断开相机，RGB 分辨率不变"
        )
        self.depth_mode.currentIndexChanged.connect(self.change_depth_mode)
        layout.addWidget(self.depth_mode)
        self.source_banner = QLabel()
        self.source_banner.setObjectName("sourceBanner")
        self.source_banner.setWordWrap(True)
        layout.addWidget(self.source_banner)
        astra = self.config.camera_profiles.get("astra_validation", {})
        configured = str(astra.get("color_device", ""))
        devices = sorted(glob.glob("/dev/video*"))
        self.color_device = QComboBox()
        self.color_device.setEditable(True)
        self.color_device.addItems([configured] + [item for item in devices if item != configured])
        self.color_device_label = QLabel("RGB UVC 设备")
        layout.addWidget(self.color_device_label)
        layout.addWidget(self.color_device)
        self.source_metadata = QLabel()
        self.source_metadata.setObjectName("sourceMeta")
        self.source_metadata.setWordWrap(True)
        layout.addWidget(self.source_metadata)
        self.projector_state = QLabel("IR 投影器：待连接")
        self.projector_state.setObjectName("projectorState")
        self.projector_state.setWordWrap(True)
        layout.addWidget(self.projector_state)
        self.ir_emitter_toggle = QCheckBox("启用 IR 投影器 / 点阵")
        self.ir_emitter_toggle.setChecked(True)
        self.ir_emitter_toggle.setToolTip(
            "室内 Depth 通常开启；太阳光下做 RGB-IR 标定时可关闭，避免点阵与环境红外互相干扰"
        )
        self.ir_emitter_toggle.toggled.connect(self.set_ir_emitter_enabled)
        layout.addWidget(self.ir_emitter_toggle)
        layout.addStretch()
        self.readiness = QLabel()
        self.readiness.setObjectName("sidebarNote")
        self.readiness.setWordWrap(True)
        layout.addWidget(self.readiness)
        return frame

    def _preview_panel(self, caption, canvas):
        panel = QFrame()
        panel.setObjectName("previewPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        label = QLabel(caption)
        label.setObjectName("previewCaption")
        layout.addWidget(label)
        layout.addWidget(canvas, 1)
        return panel, label

    def _build_preview(self):
        widget = QWidget()
        widget.setObjectName("previewArea")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        heading = QHBoxLayout()
        self.preview_title = QLabel("RGB 内参")
        self.preview_title.setObjectName("viewTitle")
        self.preview_state = QLabel("待连接")
        self.preview_state.setObjectName("viewState")
        heading.addWidget(self.preview_title)
        heading.addStretch()
        heading.addWidget(self.preview_state)
        layout.addLayout(heading)
        views = QHBoxLayout()
        views.setSpacing(10)
        self.color_canvas = VideoCanvas("等待 RGB", (190, 180))
        color_panel, self.color_caption = self._preview_panel(
            "RGB / 手眼与 AprilTag 主相机", self.color_canvas
        )
        views.addWidget(color_panel, 3)
        right = QVBoxLayout()
        right.setSpacing(10)
        self.ir_canvas = VideoCanvas("等待 IR", (110, 90))
        ir_panel, self.ir_caption = self._preview_panel("IR 原图", self.ir_canvas)
        self.depth_canvas = VideoCanvas("等待 Depth", (110, 90))
        depth_panel, self.depth_caption = self._preview_panel("原始 Depth", self.depth_canvas)
        right.addWidget(ir_panel, 1)
        right.addWidget(depth_panel, 1)
        views.addLayout(right, 2)
        layout.addLayout(views, 1)
        metric = QFrame()
        metric.setObjectName("metricBar")
        bar = QHBoxLayout(metric)
        bar.setContentsMargins(16, 9, 16, 9)
        self.metric_source = QLabel("图像  --")
        self.metric_detection = QLabel("检测  --")
        self.metric_progress = QLabel("进度  --")
        for value in (self.metric_source, self.metric_detection, self.metric_progress):
            value.setObjectName("metricValue")
            bar.addWidget(value)
            bar.addStretch()
        layout.addWidget(metric)
        return widget

    def _build_controls(self):
        frame = QFrame()
        frame.setObjectName("controlArea")
        frame.setMinimumWidth(340)
        frame.setMaximumWidth(475)
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(9)
        self.stack = QStackedWidget()
        for page in (
            self._build_rgb_page(), self._build_rgbd_page(), self._build_tag_page(),
            self._build_hand_eye_page(), self._build_localization_page(),
            self._build_segmentation_page(), self._build_planning_page(),
            self._build_grasp_page(), self._build_controller_page(),
        ):
            self.stack.addWidget(self._scroll(page))
        layout.addWidget(self.stack, 1)
        label = QLabel("运行日志")
        label.setObjectName("sectionLabel")
        layout.addWidget(label)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(300)
        self.log_output.setFixedHeight(122)
        layout.addWidget(self.log_output)
        return frame

    def _page(self, title, subtitle):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        heading = QLabel(title)
        heading.setObjectName("panelTitle")
        detail = QLabel(subtitle)
        detail.setObjectName("panelSubtitle")
        detail.setWordWrap(True)
        detail.setMinimumWidth(0)
        detail.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(heading)
        layout.addWidget(detail)
        return page, layout

    def _build_rgb_page(self):
        page, layout = self._page("RGB 内参", "ChArUco 5 x 7 · 36 mm 方格")
        self.rgb_manual_controls = QWidget()
        manual_layout = QVBoxLayout(self.rgb_manual_controls)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(9)
        target = action_button(self, "打开 ChArUco 标定板", "SP_DialogOpenButton")
        target.clicked.connect(self.open_target)
        manual_layout.addWidget(target)
        box = QGroupBox("当前 RGB")
        grid = QGridLayout(box)
        self.rgb_corner_value = self._strong("0")
        self.rgb_view_value = self._strong("0 / 20")
        grid.addWidget(QLabel("ChArUco 角点"), 0, 0)
        grid.addWidget(self.rgb_corner_value, 0, 1)
        grid.addWidget(QLabel("已采视角"), 1, 0)
        grid.addWidget(self.rgb_view_value, 1, 1)
        manual_layout.addWidget(box)
        self.rgb_progress = QProgressBar()
        self.rgb_progress.setRange(0, 40)
        manual_layout.addWidget(self.rgb_progress)
        for text, callback, icon, primary in (
            ("采集当前 RGB 姿态", self.capture_rgb_view, "SP_DialogSaveButton", True),
            ("计算并启用 RGB 内参", self.solve_rgb_intrinsics, "SP_DialogApplyButton", True),
            ("清空 RGB 采样", self.clear_rgb_samples, "SP_BrowserReload", False),
        ):
            button = action_button(self, text, icon, primary)
            button.clicked.connect(callback)
            manual_layout.addWidget(button)
        layout.addWidget(self.rgb_manual_controls)
        self.rgb_factory_note = self._result(
            "Gemini 连接后会从 /camera/color/camera_info 读取当前设备的出厂内参。"
            "不会加载 Astra 参数，也不需要采集 ChArUco 内参图像。"
        )
        self.rgb_factory_note.setVisible(False)
        layout.addWidget(self.rgb_factory_note)
        self.rgb_result = self._result("可直接复用已有 RGB 内参，也可在此重新标定")
        layout.addWidget(self.rgb_result)
        layout.addStretch()
        return page

    def _build_rgbd_page(self):
        page, layout = self._page("RGB-IR 标定", "IR 内参 + T_color_depth")
        self.rgbd_page_title = layout.itemAt(0).widget()
        self.rgbd_page_subtitle = layout.itemAt(1).widget()
        self.astra_rgbd_controls = QWidget()
        astra_layout = QVBoxLayout(self.astra_rgbd_controls)
        astra_layout.setContentsMargins(0, 0, 0, 0)
        astra_layout.setSpacing(9)
        box = QGroupBox("当前同步图像对")
        grid = QGridLayout(box)
        self.rgbd_color_corner_value = self._strong("0")
        self.rgbd_ir_corner_value = self._strong("0")
        self.rgbd_common_value = self._strong("0")
        self.rgbd_pair_value = self._strong("0 / 12")
        for row, (text, value) in enumerate((
            ("RGB 角点", self.rgbd_color_corner_value),
            ("IR 角点", self.rgbd_ir_corner_value),
            ("共同角点", self.rgbd_common_value),
            ("已采图像对", self.rgbd_pair_value),
        )):
            grid.addWidget(QLabel(text), row, 0)
            grid.addWidget(value, row, 1)
        astra_layout.addWidget(box)
        self.rgbd_progress = QProgressBar()
        self.rgbd_progress.setRange(0, 30)
        astra_layout.addWidget(self.rgbd_progress)
        for text, callback, icon, primary in (
            ("采集当前 RGB / IR 对", self.capture_rgbd_pair, "SP_DialogSaveButton", True),
            ("计算并启用 RGB-D 标定", self.solve_rgbd, "SP_DialogApplyButton", True),
            ("清空 RGB / IR 采样", self.clear_rgbd_samples, "SP_BrowserReload", False),
        ):
            button = action_button(self, text, icon, primary)
            button.clicked.connect(callback)
            astra_layout.addWidget(button)
        self.rgbd_result = self._result(
            "室内默认开启 IR 投影器；太阳光标定可用侧栏开关关闭。"
            "无论开关状态，都必须确认 IR 角点清晰且画面未饱和。"
        )
        astra_layout.addWidget(self.rgbd_result)
        layout.addWidget(self.astra_rgbd_controls)

        self.oak_calibration_controls = QWidget()
        oak_layout = QVBoxLayout(self.oak_calibration_controls)
        oak_layout.setContentsMargins(0, 0, 0, 0)
        oak_layout.setSpacing(9)
        oak_note = self._result(
            "OAK-D Pro 的 RGB、左右 OV9282 内参和外参存放在 EEPROM。比赛现场优先"
            "导入官方 JSON 或直接从相机导出，不执行 Astra 的 RGB/IR 手工双目标定。"
        )
        oak_layout.addWidget(oak_note)
        tool_box = QGroupBox("Luxonis 官方标定工具参数")
        tool_grid = QGridLayout(tool_box)
        self.oak_square_size = self._spin(0.1, 20.0, 3.0, " cm")
        self.oak_marker_size = self._spin(0.1, 20.0, 2.25, " cm")
        tool_grid.addWidget(QLabel("方格边长"), 0, 0)
        tool_grid.addWidget(self.oak_square_size, 0, 1)
        tool_grid.addWidget(QLabel("Marker 边长"), 1, 0)
        tool_grid.addWidget(self.oak_marker_size, 1, 1)
        oak_layout.addWidget(tool_box)
        for text, callback, icon, primary in (
            ("导入 OAK 官方标定 JSON", self.import_oak_json, "SP_DialogOpenButton", True),
            ("从已连接 OAK 导出 EEPROM", self.export_oak_eeprom, "SP_DialogSaveButton", True),
            ("启动 Luxonis 官方标定程序", self.launch_oak_calibration_tool, "SP_ComputerIcon", False),
        ):
            button = action_button(self, text, icon, primary)
            button.clicked.connect(callback)
            oak_layout.addWidget(button)
        self.oak_result = self._result("尚未导入比赛相机标定；无实物时可先保留为待导入状态")
        oak_layout.addWidget(self.oak_result)
        layout.addWidget(self.oak_calibration_controls)

        self.orbbec_calibration_controls = QWidget()
        orbbec_layout = QVBoxLayout(self.orbbec_calibration_controls)
        orbbec_layout.setContentsMargins(0, 0, 0, 0)
        orbbec_layout.setSpacing(9)
        orbbec_layout.addWidget(self._result(
            "Gemini 使用 Orbbec SDK 从设备读取 RGB/Depth 内参和畸变参数。"
            "Depth 在驱动中对齐到 RGB；更换实物或分辨率后会在下次连接时刷新。"
        ))
        self.orbbec_result = self._result("等待连接 Gemini 并读取 CameraInfo")
        orbbec_layout.addWidget(self.orbbec_result)
        layout.addWidget(self.orbbec_calibration_controls)
        layout.addStretch()
        return page

    def _build_tag_page(self):
        page, layout = self._page(
            "Tag 地图", "BR 原点 · +X/+Y 向板内 · 水平放置时 +Z 竖直向上"
        )
        axis_note = self._result(
            "局部坐标：红色 +X 沿 BR→TR；绿色 +Y 沿 BR→BL；蓝色 +Z = +X×+Y，"
            "垂直标定板平面。标定板水平且正面朝上时，+Z 为竖直向上。"
        )
        layout.addWidget(axis_note)
        defaults = QGroupBox("共用参数")
        grid = QGridLayout(defaults)
        self.tag_size = self._spin(1, 500, 70, " mm")
        grid.addWidget(QLabel("黑框边长"), 0, 0)
        grid.addWidget(self.tag_size, 0, 1, 1, 2)
        self.default_rpy = []
        for column, text in enumerate(("R", "P", "Y")):
            spin = self._spin(-180, 180, 0, " deg")
            self.default_rpy.append(spin)
            grid.addWidget(QLabel(text), 1, column)
            grid.addWidget(spin, 2, column)
        layout.addWidget(defaults)
        apply_default = action_button(self, "默认 RPY 填充全部 Tag", "SP_ArrowDown")
        apply_default.clicked.connect(self.apply_default_rpy_to_rows)
        layout.addWidget(apply_default)
        self.tag_table = QTableWidget(0, 7)
        self.tag_table.setHorizontalHeaderLabels(("ID", "X", "Y", "Z", "R", "P", "Y"))
        self.tag_table.verticalHeader().setVisible(False)
        self.tag_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tag_table.setMinimumHeight(210)
        layout.addWidget(self.tag_table)
        buttons = QHBoxLayout()
        add = action_button(self, "新增 ID", "SP_FileDialogNewFolder")
        add.clicked.connect(self.add_tag_row)
        remove = action_button(self, "删除选中", "SP_TrashIcon")
        remove.clicked.connect(self.remove_tag_row)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        layout.addLayout(buttons)
        save = action_button(self, "保存 Tag 地图", "SP_DialogSaveButton", True)
        save.clicked.connect(self.save_tag_map)
        layout.addWidget(save)
        self.tag_result = self._result("地图尚未修改")
        layout.addWidget(self.tag_result)
        layout.addStretch()
        return page

    def _pose_box(self, title):
        box = QGroupBox(title)
        grid = QGridLayout(box)
        spins = []
        for column, text in enumerate(("X", "Y", "Z")):
            spin = self._spin(-10000, 10000, 0, " mm")
            spins.append(spin)
            grid.addWidget(QLabel(text), 0, column)
            grid.addWidget(spin, 1, column)
        for column, text in enumerate(("R", "P", "Y")):
            spin = self._spin(-180, 180, 0, " deg")
            spins.append(spin)
            grid.addWidget(QLabel(text), 2, column)
            grid.addWidget(spin, 3, column)
        return box, spins

    def _build_hand_eye_page(self):
        page, layout = self._page("眼在手上", "T_tcp_color_camera · RGB 主相机")
        target_box = QGroupBox("标定靶标（切换或改尺寸会归档旧样本）")
        target_layout = QGridLayout(target_box)
        self.hand_target_combo = QComboBox()
        self.hand_target_combo.addItem("已建图 AprilTag（原有兼容流程）", APRILTAG_MAP_TARGET)
        self.hand_target_combo.addItem("张正友棋盘格（固定板，多姿态手眼）", CHECKERBOARD_TARGET)
        checkerboard = self.config.data["hand_eye"]["calibration_target"]["checkerboard"]
        active_target = self.config.data["hand_eye"]["calibration_target"]["type"]
        self.hand_target_combo.setCurrentIndex(
            max(0, self.hand_target_combo.findData(active_target))
        )
        self.hand_board_width = self._spin(1, 1000, checkerboard["board_width_mm"], " mm")
        self.hand_board_height = self._spin(1, 1000, checkerboard["board_height_mm"], " mm")
        self.hand_square_size = self._spin(0.1, 100, checkerboard["square_size_mm"], " mm")
        self.hand_squares_x = QSpinBox()
        self.hand_squares_x.setRange(0, 100)
        self.hand_squares_x.setSpecialValueText("待填写")
        self.hand_squares_x.setValue(int(checkerboard.get("squares_x") or 0))
        self.hand_squares_x.setButtonSymbols(QSpinBox.NoButtons)
        self.hand_squares_y = QSpinBox()
        self.hand_squares_y.setRange(0, 100)
        self.hand_squares_y.setSpecialValueText("待填写")
        self.hand_squares_y.setValue(int(checkerboard.get("squares_y") or 0))
        self.hand_squares_y.setButtonSymbols(QSpinBox.NoButtons)
        self.hand_target_info = self._result("")
        apply_target = action_button(self, "应用标定靶标设置并归档样本", "SP_DialogApplyButton")
        apply_target.clicked.connect(self.save_hand_eye_target)
        target_layout.addWidget(QLabel("类型"), 0, 0)
        target_layout.addWidget(self.hand_target_combo, 0, 1, 1, 2)
        target_layout.addWidget(QLabel("外板长边"), 1, 0)
        target_layout.addWidget(self.hand_board_width, 1, 1, 1, 2)
        target_layout.addWidget(QLabel("外板短边"), 2, 0)
        target_layout.addWidget(self.hand_board_height, 2, 1, 1, 2)
        target_layout.addWidget(QLabel("单个格子"), 3, 0)
        target_layout.addWidget(self.hand_square_size, 3, 1, 1, 2)
        target_layout.addWidget(QLabel("长边总格数（黑+白）"), 4, 0)
        target_layout.addWidget(self.hand_squares_x, 4, 1, 1, 2)
        target_layout.addWidget(QLabel("短边总格数（黑+白）"), 5, 0)
        target_layout.addWidget(self.hand_squares_y, 5, 1, 1, 2)
        target_layout.addWidget(apply_target, 6, 0, 1, 3)
        target_layout.addWidget(self.hand_target_info, 7, 0, 1, 3)
        layout.addWidget(target_box)
        self.hand_target_combo.currentIndexChanged.connect(self._refresh_hand_target_hint)
        self.hand_board_width.valueChanged.connect(self._refresh_hand_target_hint)
        self.hand_board_height.valueChanged.connect(self._refresh_hand_target_hint)
        self.hand_square_size.valueChanged.connect(self._refresh_hand_target_hint)
        self.hand_squares_x.valueChanged.connect(self._refresh_hand_target_hint)
        self.hand_squares_y.valueChanged.connect(self._refresh_hand_target_hint)
        self._refresh_hand_target_hint()
        layout.addWidget(self._build_tcp_pose_box())
        box, self.hand_pose_spins = self._pose_box("当前 T_base_tcp")
        layout.addWidget(box)
        capture = action_button(self, "采集当前 RGB 与 TCP", "SP_DialogSaveButton", True)
        capture.clicked.connect(self.capture_hand_eye)
        layout.addWidget(capture)
        self.hand_table = QTableWidget(0, 3)
        self.hand_table.setHorizontalHeaderLabels(("序号", "靶标", "RMS / px"))
        self.hand_table.verticalHeader().setVisible(False)
        self.hand_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hand_table.setMinimumHeight(175)
        layout.addWidget(self.hand_table)
        solve = action_button(self, "求解并启用手眼矩阵", "SP_DialogApplyButton", True)
        solve.clicked.connect(self.solve_hand_eye)
        reset = action_button(self, "归档并清空样本", "SP_BrowserReload")
        reset.clicked.connect(self.reset_hand_eye_samples)
        layout.addWidget(solve)
        layout.addWidget(reset)
        self.hand_result = self._result("等待采样")
        layout.addWidget(self.hand_result)
        layout.addStretch()
        return page

    def _hand_eye_ui_target_settings(self, require_complete=False):
        target_type = str(self.hand_target_combo.currentData())
        checkerboard = dict(
            self.config.data["hand_eye"]["calibration_target"]["checkerboard"]
        )
        width = float(self.hand_board_width.value())
        height = float(self.hand_board_height.value())
        square = float(self.hand_square_size.value())
        squares_x = int(self.hand_squares_x.value())
        squares_y = int(self.hand_squares_y.value())
        configured = squares_x >= 3 and squares_y >= 3
        checkerboard.update({
            "board_width_mm": width,
            "board_height_mm": height,
            "square_size_mm": square,
            "configured": configured,
            "squares_x": squares_x if configured else None,
            "squares_y": squares_y if configured else None,
            "inner_corners": [squares_x - 1, squares_y - 1] if configured else None,
        })
        if target_type == CHECKERBOARD_TARGET and require_complete and not configured:
            raise ValueError("请先填写长边和短边的黑白方格总数（每边至少 3 格）")
        if configured:
            CheckerboardTarget(checkerboard)
        return {"type": target_type, "checkerboard": checkerboard}

    def _refresh_hand_target_hint(self, *_unused):
        target_type = self.hand_target_combo.currentData()
        if target_type == CHECKERBOARD_TARGET:
            squares_x = int(self.hand_squares_x.value())
            squares_y = int(self.hand_squares_y.value())
            if squares_x < 3 or squares_y < 3:
                self.hand_target_info.setText(
                    "待填写：现场数长边、短边的黑白格总数；不是只数黑格。填完后实时预览立即按该棋盘识别。"
                )
                return
            try:
                settings = self._hand_eye_ui_target_settings(require_complete=True)
                checkerboard = settings["checkerboard"]
                inner = checkerboard["inner_corners"]
                self.hand_target_info.setText(
                    "外板 {:.1f}×{:.1f} mm；印刷网格 {}×{} 格 = {:.1f}×{:.1f} mm；OpenCV 内角点 {}×{}（共 {} 个）".format(
                        checkerboard["board_width_mm"], checkerboard["board_height_mm"],
                        squares_x, squares_y,
                        squares_x * checkerboard["square_size_mm"],
                        squares_y * checkerboard["square_size_mm"],
                        inner[0], inner[1], inner[0] * inner[1],
                    )
                )
            except Exception as error:
                self.hand_target_info.setText("棋盘参数无效：{}".format(error))
        else:
            self.hand_target_info.setText(
                "沿用已登记 AprilTag 地图：每帧 PnP 需可见至少一个已登记 Tag"
            )

    def _apply_hand_eye_target_settings(self):
        settings = self._hand_eye_ui_target_settings(require_complete=True)
        current = self.config.data["hand_eye"]["calibration_target"]
        if current == settings:
            return None, False
        current.clear()
        current.update(settings)
        self.config.data["hand_eye"]["tcp_from_color_camera"]["valid"] = False
        self.config.save()
        backup = self.sample_store.reset()
        self.tag_localizer = None
        self.hybrid_localizer = None
        self._restart_live_processing_workers()
        self._refresh_samples()
        self._refresh_readiness()
        return backup, True

    def save_hand_eye_target(self):
        try:
            backup, changed = self._apply_hand_eye_target_settings()
            if changed:
                self.hand_result.setText(
                    "标定靶标已应用；旧样本已归档 {}".format(backup or "（无旧样本）")
                )
            else:
                self.hand_result.setText("标定靶标设置未变化")
            self._refresh_hand_target_hint()
        except Exception as error:
            self._show_error("保存手眼标定靶标失败", error)

    def _build_localization_page(self):
        page, layout = self._page("定位验证", "TCP + 手眼实时定位 · Tag 仅标定/诊断")
        self.robot_available = QCheckBox("提供当前 TCP 数据")
        self.robot_available.setChecked(True)
        self.hide_tags = QCheckBox("模拟 Tag 丢失")
        layout.addWidget(self.robot_available)
        layout.addWidget(self.hide_tags)
        box, self.verify_pose_spins = self._pose_box("当前 T_base_tcp")
        layout.addWidget(box)
        layout.addWidget(self._result(
            "实时 TCP 位姿由“眼在手上”页的 TCP 通信验证面板（NexBot 7000 端口）提供；"
            "连接后自动填入本页，无需手抄示教器。"
        ))
        state = QGroupBox("T_base_color_camera")
        grid = QGridLayout(state)
        self.loc_source = self._strong("无有效位姿")
        self.loc_tags = self._strong("--")
        self.loc_quality = self._strong("--")
        self.loc_xyz = self._strong("--")
        self.loc_rpy = self._strong("--")
        for row, (text, value) in enumerate((
            ("来源", self.loc_source), ("Tag", self.loc_tags),
            ("质量", self.loc_quality), ("XYZ / mm", self.loc_xyz),
            ("RPY / deg", self.loc_rpy),
        )):
            grid.addWidget(QLabel(text), row, 0)
            grid.addWidget(value, row, 1)
        layout.addWidget(state)
        self.localization_result = self._result("等待相机")
        layout.addWidget(self.localization_result)
        layout.addStretch()
        return page

    def _build_segmentation_page(self):
        page, layout = self._page(
            "分割模型验证", "复用 bottle_localization 的 YOLO 实例 Mask 接口"
        )
        note = self._result(
            "选择实例分割 .pt 权重后加载模型。推理在后台线程运行，只保留最新 RGB 帧，"
            "不会阻塞相机或累积旧帧。更换实物后在这里替换权重和目标类别。"
        )
        layout.addWidget(note)
        settings = self.config.data["segmentation_validation"]
        model_box = QGroupBox("模型")
        model_layout = QVBoxLayout(model_box)
        self.seg_weights = QLineEdit(str(settings.get("weights_file", "")))
        self.seg_weights.setCursorPosition(0)
        self.seg_weights.setPlaceholderText("选择 Ultralytics 实例分割 .pt 权重")
        model_layout.addWidget(self.seg_weights)
        browse = action_button(self, "选择权重文件", "SP_DialogOpenButton")
        browse.clicked.connect(self.browse_segmentation_weights)
        model_layout.addWidget(browse)
        model_grid = QGridLayout()
        self.seg_classes = QLineEdit(",".join(settings.get("target_classes", [])))
        self.seg_classes.setPlaceholderText("留空接受全部类别，例如 bottle,can")
        self.seg_device = QComboBox()
        self.seg_device.setEditable(True)
        configured_device = str(settings.get("device", "0"))
        self.seg_device.addItems([configured_device] + [
            value for value in ("auto", "0", "cpu") if value != configured_device
        ])
        self.seg_confidence = self._spin(0.05, 1.0, settings["confidence_threshold"], "")
        self.seg_iou = self._spin(0.0, 1.0, settings.get("iou_threshold", 0.45), "")
        self.seg_image_size = QSpinBox()
        self.seg_image_size.setRange(64, 4096)
        self.seg_image_size.setSingleStep(32)
        self.seg_image_size.setValue(int(settings.get("image_size", 640)))
        self.seg_agnostic_nms = QCheckBox("跨类别 NMS")
        self.seg_agnostic_nms.setChecked(bool(settings.get("agnostic_nms", True)))
        self.seg_deduplicate = QCheckBox("Mask 二次去重")
        self.seg_deduplicate.setChecked(
            bool(settings.get("deduplicate_instances", True))
        )
        for row, (text, widget) in enumerate((
            ("目标类别", self.seg_classes), ("设备", self.seg_device),
            ("置信度阈值", self.seg_confidence), ("IOU 阈值", self.seg_iou),
            ("推理尺寸", self.seg_image_size),
        )):
            model_grid.addWidget(QLabel(text), row, 0)
            model_grid.addWidget(widget, row, 1)
        model_grid.setColumnStretch(1, 1)
        model_layout.addLayout(model_grid)
        nms_row = QHBoxLayout()
        nms_row.addWidget(self.seg_agnostic_nms)
        nms_row.addWidget(self.seg_deduplicate)
        model_layout.addLayout(nms_row)
        dedup_grid = QGridLayout()
        self.seg_mask_iou = self._spin(
            0.0, 1.0, settings.get("duplicate_mask_iou_threshold", 0.50), ""
        )
        self.seg_mask_containment = self._spin(
            0.0, 1.0,
            settings.get("duplicate_mask_containment_threshold", 0.80), "",
        )
        self.seg_center_ratio = self._spin(
            0.0, 2.0, settings.get("duplicate_center_distance_ratio", 0.35), ""
        )
        self.seg_confidence_tie = self._spin(
            0.0, 0.25, settings.get("duplicate_confidence_tie_margin", 0.05), ""
        )
        for row, (text, widget) in enumerate((
            ("Mask IoU 去重", self.seg_mask_iou),
            ("小 Mask 包含率", self.seg_mask_containment),
            ("中心距离/尺寸", self.seg_center_ratio),
            ("置信度近似范围", self.seg_confidence_tie),
        )):
            dedup_grid.addWidget(QLabel(text), row, 0)
            dedup_grid.addWidget(widget, row, 1)
        dedup_grid.setColumnStretch(1, 1)
        model_layout.addLayout(dedup_grid)
        layout.addWidget(model_box)

        quality_box = QGroupBox("实时质量")
        quality_grid = QGridLayout(quality_box)
        self.seg_state_value = self._strong("未加载")
        self.seg_class_value = self._strong("--")
        # 类别名和去重统计会在模型开始返回结果后突然变长。允许该值在
        # 当前控制栏宽度内换行，避免它反向撑宽整个滚动页面。
        self.seg_class_value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.seg_conf_value = self._strong("--")
        self.seg_area_value = self._strong("--")
        self.seg_time_value = self._strong("--")
        self.seg_streak_value = self._strong("0")
        for row, (text, widget) in enumerate((
            ("状态", self.seg_state_value), ("类别", self.seg_class_value),
            ("置信度", self.seg_conf_value), ("Mask 面积", self.seg_area_value),
            ("推理耗时", self.seg_time_value), ("连续合格帧", self.seg_streak_value),
        )):
            quality_grid.addWidget(QLabel(text), row, 0)
            quality_grid.addWidget(widget, row, 1)
        quality_grid.setColumnStretch(1, 1)
        layout.addWidget(quality_box)
        self.seg_progress = QProgressBar()
        self.seg_progress.setRange(
            0, int(settings.get("required_consecutive_valid_frames", 5))
        )
        layout.addWidget(self.seg_progress)
        load = action_button(self, "保存配置并加载模型", "SP_MediaPlay", True)
        load.clicked.connect(self.load_segmentation_model)
        self.seg_unload_button = action_button(self, "卸载分割模型", "SP_MediaStop")
        self.seg_unload_button.clicked.connect(self.stop_segmentation_model)
        self.seg_confirm_button = action_button(
            self, "确认当前模型分割通过", "SP_DialogApplyButton", True
        )
        self.seg_confirm_button.setEnabled(False)
        self.seg_confirm_button.clicked.connect(self.confirm_segmentation_validation)
        layout.addWidget(load)
        layout.addWidget(self.seg_unload_button)
        layout.addWidget(self.seg_confirm_button)
        self.seg_result = self._result("模型尚未加载；已有示例权重仅用于验证工作流")
        layout.addWidget(self.seg_result)
        layout.addStretch()
        return page

    def _build_planning_page(self):
        page, layout = self._page("抓取规划验证", "阶段 07 · 确定性俯视规划后端已就绪")
        contract = self._result(
            "规划输入契约：已验证实例 Mask + 对齐 Depth + T_base_camera → 目标点云；"
            "规划输出契约：候选抓取位姿、评分、夹爪宽度、碰撞检查结果。"
        )
        layout.addWidget(contract)
        box = QGroupBox("前置条件")
        grid = QGridLayout(box)
        self.plan_seg_state = self._strong("待检查")
        self.plan_depth_state = self._strong("待检查")
        self.plan_pose_state = self._strong("待检查")
        for row, (text, value) in enumerate((
            ("分割模型", self.plan_seg_state),
            ("RGB-D", self.plan_depth_state),
            ("相机定位", self.plan_pose_state),
        )):
            grid.addWidget(QLabel(text), row, 0)
            grid.addWidget(value, row, 1)
        layout.addWidget(box)
        refresh = action_button(self, "刷新前置条件", "SP_BrowserReload")
        refresh.clicked.connect(self._refresh_readiness)
        layout.addWidget(refresh)
        self.rviz_button = action_button(
            self, "打开 RViz 点云验证", "SP_ComputerIcon", True
        )
        self.rviz_button.clicked.connect(self.toggle_rviz_visualization)
        layout.addWidget(self.rviz_button)
        rviz_contract = self._result(
            "RViz 复用本 UI 的同一组 RGB、Depth、Mask 与 AprilTag/TCP 相机位姿；"
            "不会启动第二个相机节点。已预留 grasp_candidates 与 planned_path 话题。"
        )
        layout.addWidget(rviz_contract)
        self.planning_result = self._result(
            "competition_pipeline.planning 已提供点云边界→俯视抓取→六段 TCP 位姿；"
            "本页先完成 RGB-D/RViz 输入验收，实物 TCP 挂载复测前不会发送运动命令。"
        )
        layout.addWidget(self.planning_result)
        layout.addStretch()
        return page

    def _build_grasp_page(self):
        page, layout = self._page("抓取执行验证", "阶段 08 · 状态机已实现，等待实机适配验收")
        contract = self._result(
            "执行输入契约：经人工确认的规划位姿 → TCP 预抓取位姿 → 直线接近 → 闭合夹爪 →"
            "抬升验证。正式执行前必须先通过 dry-run、工作空间和单步位移限制。"
        )
        layout.addWidget(contract)
        box = QGroupBox("安全状态")
        grid = QGridLayout(box)
        self.grasp_plan_state = self._strong("规划未验证")
        self.grasp_dry_run_state = self._strong("开启")
        self.grasp_motion_state = self._strong("禁止")
        for row, (text, value) in enumerate((
            ("规划结果", self.grasp_plan_state),
            ("Dry-run", self.grasp_dry_run_state),
            ("机器人运动", self.grasp_motion_state),
        )):
            grid.addWidget(QLabel(text), row, 0)
            grid.addWidget(value, row, 1)
        layout.addWidget(box)
        self.grasp_result = self._result(
            "执行状态机已实现并通过 UR5e+Robotiq Gazebo 闭环；当前正式配置仍保持 "
            "fail-closed。接入真实 RobotController/GripperController 并完成 dry-run 后才能启用。"
        )
        layout.addWidget(self.grasp_result)
        layout.addStretch()
        return page

    def _build_controller_page(self):
        page, layout = self._page(
            "控制器/TCP 测试", "现场接入前的只读通信与状态验收；不会发送 MOVJ/MOVL 或 IO 写入"
        )
        box = QGroupBox("连接参数（来自 competition.yaml）")
        grid = QGridLayout(box)
        controller = self.config.data.get("controller", {})
        self.controller_enabled = QCheckBox("启用 Modbus-TCP 只读测试")
        self.controller_enabled.setChecked(bool(controller.get("enabled", False)))
        self.controller_host = QLineEdit(str(controller.get("host") or ""))
        self.controller_host.setPlaceholderText("控制器 IP，例如 192.168.1.10")
        self.controller_port = QSpinBox()
        self.controller_port.setRange(1, 65535)
        self.controller_port.setValue(int(controller.get("port") or 502))
        self.controller_unit = QSpinBox()
        self.controller_unit.setRange(1, 247)
        self.controller_unit.setValue(int(controller.get("unit_id") or 1))
        self.controller_connect_button = action_button(self, "保存并开始只读测试", "SP_MediaPlay", True)
        self.controller_connect_button.clicked.connect(self.toggle_controller_test)
        for row, (title, value) in enumerate((("协议", self.controller_enabled), ("IP", self.controller_host), ("Port", self.controller_port), ("Unit ID", self.controller_unit))):
            grid.addWidget(QLabel(title), row, 0)
            grid.addWidget(value, row, 1)
        grid.addWidget(self.controller_connect_button, 4, 0, 1, 2)
        layout.addWidget(box)
        mapping_box = QGroupBox("状态寄存器映射（YAML，只读）")
        mapping_layout = QVBoxLayout(mapping_box)
        self.controller_mapping = QPlainTextEdit()
        configured_mapping = controller.get("state_registers", {}) or {}
        if configured_mapping:
            mapping_text = yaml.safe_dump({
                "state_registers": configured_mapping,
                "state_codec": controller.get("state_codec", {}) or {},
            }, sort_keys=False, allow_unicode=True)
        else:
            mapping_text = """# 仅填厂家确认的零基地址；以下只是字段模板，不是地址示例。
state_registers:
  # joint_deg_1: {address: <官方地址>, source: holding, encoding: s32, scale: <倍率>}
  # joint_deg_2: {address: <官方地址>, source: holding, encoding: s32, scale: <倍率>}
  # joint_deg_3 .. joint_deg_6；axis_1 .. axis_7；reserved_1 / reserved_2
  # tcp_x_mm / tcp_y_mm / tcp_z_mm；tcp_rx_deg / tcp_ry_deg / tcp_rz_deg
  # coordinate_system / angle_unit / shape / tool_id / user_id
  # servo_on / emergency_stop / moving / alarm_code / alarm_active
  {}
state_codec:
  # alarm_texts: {'<报警码>': '厂家报警文本'}
  alarm_severity: error
"""
        self.controller_mapping.setPlainText(mapping_text)
        self.controller_mapping.setPlaceholderText(
            "按官方表填写 address/source/encoding/scale；禁止猜测地址"
        )
        self.controller_mapping.setFixedHeight(180)
        mapping_layout.addWidget(self.controller_mapping)
        layout.addWidget(mapping_box)
        state_box = QGroupBox("控制器状态（手册字段）")
        state_grid = QGridLayout(state_box)
        self.controller_state_labels = {}
        for row, name in enumerate((
            "connected", "servo_on", "emergency_stop", "moving", "joint_deg",
            "tcp_xyz_mm", "tcp_rpy_deg", "point_name", "coordinate_system",
            "angle_unit", "reserved", "axes", "tool_id", "user_id", "shape", "alarm",
            "initial_shape", "shape_changed",
        )):
            label = self._strong("--")
            self.controller_state_labels[name] = label
            state_grid.addWidget(QLabel(name), row, 0)
            state_grid.addWidget(label, row, 1)
        layout.addWidget(state_box)
        self.controller_result = self._result(
            "配置 state_registers 后会周期读取，并按 XYZ mm、RPY deg、Tool/User/shape 显示。"
        )
        layout.addWidget(self.controller_result)
        self.safe_point_button = action_button(self, "保存当前无奇异安全点", "SP_DialogSaveButton")
        self.safe_point_button.clicked.connect(self.save_controller_safe_point)
        layout.addWidget(self.safe_point_button)
        self.safe_point_result = self._result(
            "故障时正式执行器先 STOP，再按配置决定是否用低速 MOVJ 回退。默认禁止自动回退；"
            "急停、TCP/关节状态失效或没有安全点时保持锁定。"
        )
        layout.addWidget(self.safe_point_result)
        layout.addStretch()
        return page

    def _scroll(self, page):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        # 正常情况下内容会随控制栏压缩；若驱动返回了异常长的设备名或
        # 模型类别名，仍保留横向滚动作为兜底，不能把右半边永久裁掉。
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(page)
        return scroll

    def _spin(self, minimum, maximum, value, suffix):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        return spin

    def _strong(self, text):
        label = QLabel(text)
        label.setObjectName("strongValue")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        return label

    def _result(self, text):
        label = QLabel(text)
        label.setObjectName("resultBanner")
        label.setWordWrap(True)
        return label

    def browse_segmentation_weights(self):
        current = Path(self.seg_weights.text().strip() or str(Path.home())).expanduser()
        start = current.parent if current.suffix else current
        source, _ = QFileDialog.getOpenFileName(
            self, "选择实例分割权重", str(start),
            "Ultralytics weights (*.pt *.pth);;All files (*)",
        )
        if source:
            self.seg_weights.setText(source)

    def _segmentation_ui_settings(self):
        weights = Path(self.seg_weights.text().strip()).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError("分割权重不存在：{}".format(weights))
        classes = [
            value.strip() for value in self.seg_classes.text().split(",")
            if value.strip()
        ]
        current = self.config.data["segmentation_validation"]
        return {
            "weights_file": str(weights),
            "target_classes": classes,
            "confidence_threshold": self.seg_confidence.value(),
            "iou_threshold": self.seg_iou.value(),
            "image_size": self.seg_image_size.value(),
            "device": self.seg_device.currentText().strip() or "auto",
            "agnostic_nms": self.seg_agnostic_nms.isChecked(),
            "deduplicate_instances": self.seg_deduplicate.isChecked(),
            "duplicate_mask_iou_threshold": self.seg_mask_iou.value(),
            "duplicate_mask_containment_threshold": self.seg_mask_containment.value(),
            "duplicate_center_distance_ratio": self.seg_center_ratio.value(),
            "duplicate_confidence_tie_margin": self.seg_confidence_tie.value(),
            "maximum_detections": int(current.get("maximum_detections", 50)),
            "preview_interval_s": float(current.get("preview_interval_s", 0.2)),
            "minimum_confidence": float(current.get("minimum_confidence", 0.25)),
            "minimum_mask_area_ratio": float(current.get("minimum_mask_area_ratio", 0.002)),
            "maximum_mask_area_ratio": float(current.get("maximum_mask_area_ratio", 0.8)),
            "required_consecutive_valid_frames": int(
                current.get("required_consecutive_valid_frames", 5)
            ),
            "validation": {
                "valid": False,
                "weights_sha256": "",
                "camera_profile": "",
                "confirmed_at": "",
            },
        }

    def load_segmentation_model(self):
        try:
            settings = self._segmentation_ui_settings()
            self.stop_segmentation_model(silent=True)
            self.config.data["segmentation_validation"] = settings
            self.config.save()
            self.segmentation_consecutive_valid = 0
            self.segmentation_last_result = None
            self.segmentation_model_ready = False
            self.seg_progress.setRange(
                0, settings["required_consecutive_valid_frames"]
            )
            self.seg_progress.setValue(0)
            self.seg_confirm_button.setEnabled(False)
            self.seg_state_value.setText("加载中")
            self.seg_result.setText("正在加载 {}".format(Path(settings["weights_file"]).name))
            worker = SegmentationWorker(settings)
            worker.model_ready.connect(self._segmentation_model_ready)
            worker.result_ready.connect(self._receive_segmentation_result)
            worker.failed.connect(self._segmentation_failed)
            self.segmentation_worker = worker
            worker.start()
            self._refresh_readiness()
        except Exception as error:
            self._show_error("分割模型加载失败", error)

    def stop_segmentation_model(self, checked=False, silent=False):
        worker = self.segmentation_worker
        if worker is not None:
            for signal, slot in (
                (worker.result_ready, self._receive_segmentation_result),
                (worker.model_ready, self._segmentation_model_ready),
                (worker.failed, self._segmentation_failed),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            worker.stop()
            worker.wait(5000)
        self.segmentation_worker = None
        self.segmentation_model_ready = False
        self.segmentation_last_submit_s = 0.0
        self.segmentation_consecutive_valid = 0
        self.segmentation_last_result = None
        self.seg_confirm_button.setEnabled(False)
        self.seg_progress.setValue(0)
        if self.bundle is not None and self.current_stage in (
            self.SEGMENTATION, self.PLANNING
        ):
            self.color_canvas.set_frame(self.bundle.color_bgr)
        if not silent:
            self.seg_state_value.setText("已卸载")
            self.seg_result.setText("分割模型已卸载；已确认记录仍保留在配置中")

    def _segmentation_model_ready(self, weights):
        self.segmentation_model_ready = True
        self.seg_state_value.setText("已加载")
        self.seg_result.setText(
            "模型已加载：{}；连接相机并保持目标在不同位置/距离下连续观察".format(
                Path(weights).name
            )
        )
        self._log("分割模型已加载：{}".format(weights))

    def _segmentation_failed(self, message):
        self.segmentation_model_ready = False
        self.segmentation_last_result = None
        self.seg_state_value.setText("失败")
        self.seg_result.setText("分割推理失败：{}".format(message))
        self._log("分割推理失败：{}".format(message))
        self.statusBar().showMessage(str(message), 6000)

    def _receive_segmentation_result(self, payload):
        self.segmentation_last_result = payload
        quality = payload["quality"]
        if quality.valid:
            self.segmentation_consecutive_valid += 1
        else:
            self.segmentation_consecutive_valid = 0
        required = int(
            self.config.data["segmentation_validation"][
                "required_consecutive_valid_frames"
            ]
        )
        streak = self.segmentation_consecutive_valid
        self.seg_progress.setValue(min(streak, required))
        self.seg_state_value.setText("合格" if quality.valid else "拒绝")
        instances = payload.get("instances", [])
        statistics = payload.get("statistics", {})
        model_instances = int(statistics.get("model_instances", len(instances)))
        suppressed = int(statistics.get("suppressed_duplicates", 0))
        quality_rejected = int(statistics.get("quality_rejected", 0))
        class_names = sorted({item.class_name for item in instances if item.class_name})
        self.seg_class_value.setText(
            "{} · {} 个（模型 {}，合并 {}，质量拒绝 {}）".format(
                ", ".join(class_names) or "--",
                len(instances), model_instances, suppressed, quality_rejected,
            )
        )
        self.seg_conf_value.setText("{:.1f}%".format(quality.confidence * 100.0))
        self.seg_area_value.setText("{:.2f}%".format(quality.mask_area_ratio * 100.0))
        self.seg_time_value.setText("{:.1f} ms".format(payload["elapsed_ms"]))
        self.seg_streak_value.setText("{} / {}".format(streak, required))
        self.seg_confirm_button.setEnabled(quality.valid and streak >= required)
        self.seg_result.setText(quality.reason)
        if self.current_stage in (self.SEGMENTATION, self.PLANNING):
            self.color_canvas.set_frame(payload["overlay"])
        if self.current_stage == self.SEGMENTATION:
            self.preview_state.setText(
                "分割{} · {} {:.1f}%".format(
                    "合格" if quality.valid else "拒绝",
                    quality.class_name or "无目标", quality.confidence * 100.0,
                )
            )
            self.metric_detection.setText(
                "Mask  {:.2f}%".format(quality.mask_area_ratio * 100.0)
            )
            self.metric_progress.setText("连续  {} / {}".format(streak, required))
        if self.rviz_worker is not None:
            self.rviz_worker.submit(payload)

    def confirm_segmentation_validation(self):
        payload = self.segmentation_last_result
        required = int(
            self.config.data["segmentation_validation"][
                "required_consecutive_valid_frames"
            ]
        )
        if (
            payload is None or not payload["quality"].valid
            or self.segmentation_consecutive_valid < required
        ):
            self._show_error("不能确认分割模型", "连续合格帧尚未达到质量门")
            return
        try:
            from datetime import datetime

            settings = self.config.data["segmentation_validation"]
            quality = payload["quality"]
            settings["validation"] = {
                "valid": True,
                "weights_sha256": file_sha256(settings["weights_file"]),
                "weights_file": settings["weights_file"],
                "camera_profile": self.config.active_camera_profile,
                "target_classes": list(settings.get("target_classes", [])),
                "image_width": int(self.bundle.color_bgr.shape[1]) if self.bundle is not None else 0,
                "image_height": int(self.bundle.color_bgr.shape[0]) if self.bundle is not None else 0,
                "class_name": quality.class_name,
                "confidence": quality.confidence,
                "mask_area_ratio": quality.mask_area_ratio,
                "confirmed_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.config.save()
            self.seg_result.setText("分割模型已确认并冻结 SHA256；更换相机或模型后必须重新验证")
            self._refresh_readiness()
            self._log("分割模型验证已确认")
        except Exception as error:
            self._show_error("分割验证保存失败", error)

    def _apply_camera_profile_ui(self):
        camera = self.config.camera
        backend = camera["backend"]
        is_astra = backend == "astra_ros"
        is_orbbec = backend == "orbbec_ros"
        label = str(camera.get("label", self.config.active_camera_profile))
        self.source_banner.setText(
            "{}\n{}".format(
                label,
                "RGB UVC + IR/Depth ROS" if is_astra else (
                    "Orbbec SDK ROS · 实机 CameraInfo / 出厂内参"
                    if is_orbbec
                    else "DepthAI · RGB + 双 OV9282 · Depth 对齐 RGB"
                ),
            )
        )
        self.color_device_label.setVisible(is_astra)
        self.color_device.setVisible(is_astra)
        self.depth_mode_label.setVisible(is_astra)
        self.depth_mode.setVisible(is_astra)
        self.rgb_manual_controls.setVisible(not is_orbbec)
        self.rgb_factory_note.setVisible(is_orbbec)
        self.astra_rgbd_controls.setVisible(is_astra)
        self.oak_calibration_controls.setVisible(backend == "oak_depthai")
        self.orbbec_calibration_controls.setVisible(is_orbbec)
        if is_astra:
            self.depth_mode.blockSignals(True)
            self.depth_mode.clear()
            for mode_name, mode in camera.get("depth_modes", {}).items():
                self.depth_mode.addItem(
                    str(mode.get("label", mode_name)), mode_name
                )
            mode_index = self.depth_mode.findData(camera.get("depth_mode"))
            self.depth_mode.setCurrentIndex(max(0, mode_index))
            self.depth_mode.blockSignals(False)
            self.color_device.setCurrentText(str(camera["color_device"]))
            mode = camera["depth_modes"][camera["depth_mode"]]
            self.source_metadata.setText(
                "RGB  {} x {} @ {}\nDepth {} x {} @ {} · IR {} x {} @ {}".format(
                    camera["color_width"], camera["color_height"], camera["color_fps"],
                    mode["depth_width"], mode["depth_height"], mode["depth_fps"],
                    mode["ir_width"], mode["ir_height"], mode["ir_fps"],
                )
            )
            self.projector_state.setText("Astra IR 投影器：待连接")
            self.stage_list.item(self.RGBD).setText("02  RGB-IR 标定")
            self.rgbd_page_title.setText("RGB-IR 标定")
            self.rgbd_page_subtitle.setText("IR 内参 + T_color_depth")
            self.stage_list.item(self.RGB).setText("01  RGB 内参")
        elif is_orbbec:
            self.source_metadata.setText(
                "RGB  {} x {} @ {}\nDepth 对齐 RGB · CameraInfo 实时读取\n序列号 {}".format(
                    camera["color_width"], camera["color_height"],
                    camera["color_fps"], camera.get("expected_serial", "未限制"),
                )
            )
            self.projector_state.setText("Gemini IR 投影器：待连接")
            self.stage_list.item(self.RGB).setText("01  RGB 工厂内参")
            self.stage_list.item(self.RGBD).setText("02  Gemini 工厂 RGB-D 参数")
            self.rgbd_page_title.setText("Gemini 工厂 RGB-D 参数")
            self.rgbd_page_subtitle.setText("Orbbec SDK · 设备内参 · 驱动 Depth→RGB 对齐")
        else:
            self.source_metadata.setText(
                "RGB/Depth  {} x {} @ {}\nMono {} · 点阵 {} mA\n泛光灯 {} mA".format(
                    camera["color_width"], camera["color_height"], camera["color_fps"],
                    camera.get("mono_resolution", "800p"),
                    camera.get("dot_projector_mA", 0), camera.get("floodlight_mA", 0),
                )
            )
            self.projector_state.setText("OAK IR 投影器：待连接")
            self.stage_list.item(self.RGBD).setText("02  OAK 标定导入")
            self.rgbd_page_title.setText("OAK-D Pro 工厂标定")
            self.rgbd_page_subtitle.setText("DepthAI EEPROM JSON · RGB 内参 · 双目外参")
            tool = camera.get("calibration_tool", {})
            self.oak_square_size.setValue(float(tool.get("square_size_cm", 3.0)))
            self.oak_marker_size.setValue(float(tool.get("marker_size_cm", 2.25)))
            self.stage_list.item(self.RGB).setText("01  RGB 内参")
        self.connect_button.setText(
            "连接 {}".format("Astra" if is_astra else ("Gemini" if is_orbbec else "OAK-D Pro"))
        )

    def change_depth_mode(self, index):
        if self.config.camera.get("backend") != "astra_ros":
            return
        mode_name = self.depth_mode.itemData(index)
        if not mode_name or mode_name == self.config.camera.get("depth_mode"):
            return
        if self.camera_worker is not None:
            self.disconnect_camera()
        self.stop_rviz_visualization()
        try:
            previous = str(self.config.camera.get("depth_mode"))
            self.config.set_active_depth_mode(mode_name)
            self.bundle = None
            self.last_depth_stamp = None
            self.last_depth_preview_s = 0.0
            self.last_aligned_preview = None
            self.depth_aligner = None
            self.depth_aligner_source = None
            self.clear_rgbd_samples()
            self.ir_canvas.clear("等待 IR")
            self.depth_canvas.clear("等待 Depth")
            self._apply_camera_profile_ui()
            self._log(
                "Depth/IR 模式 {} → {}；RGB 内参、手眼和刚性外参保持不变".format(
                    previous, mode_name
                )
            )
        except Exception as error:
            self._apply_camera_profile_ui()
            self._show_error("Depth/IR 模式切换失败", error)

    def change_camera_profile(self, index):
        profile_name = self.camera_profile.itemData(index)
        if not profile_name or profile_name == self.config.active_camera_profile:
            return
        if self.camera_worker is not None:
            self.disconnect_camera()
        self.stop_rviz_visualization()
        self.stop_segmentation_model(silent=True)
        try:
            previous = self.config.active_camera_profile
            self.config.set_active_camera_profile(profile_name)
            self.sample_store = HandEyeSampleStore(self._sample_path(), self.config)
            self.bundle = None
            self.color_intrinsics_cache = None
            self.orbbec_intrinsics_synced = False
            self.depth_aligner = None
            self.depth_aligner_source = None
            self.last_aligned_preview = None
            self.tag_localizer = None
            self.hybrid_localizer = None
            self.clear_rgb_samples()
            self.clear_rgbd_samples()
            self.color_canvas.clear("等待 RGB")
            self.ir_canvas.clear("等待 IR")
            self.depth_canvas.clear("等待 Depth")
            self._apply_camera_profile_ui()
            self._refresh_samples()
            self._refresh_readiness()
            self.change_stage(self.current_stage)
            self._log(
                "相机配置 {} → {}；旧手眼矩阵已失效".format(previous, profile_name)
            )
        except Exception as error:
            old_index = self.camera_profile.findData(self.config.active_camera_profile)
            self.camera_profile.blockSignals(True)
            self.camera_profile.setCurrentIndex(old_index)
            self.camera_profile.blockSignals(False)
            self._show_error("相机配置切换失败", error)

    def _accept_oak_calibration(self, info):
        self.config.data["hand_eye"]["tcp_from_color_camera"]["valid"] = False
        self.config.save()
        self.color_intrinsics_cache = None
        self.hybrid_localizer = None
        self._restart_live_processing_workers()
        self.oak_result.setText("已导入 · {} · 旧手眼结果已失效".format(
            format_oak_summary(info)
        ))
        self._refresh_readiness()

    def import_oak_json(self):
        source, _ = QFileDialog.getOpenFileName(
            self, "选择 Luxonis/DepthAI 标定 JSON", str(Path.home()),
            "DepthAI calibration (*.json);;All files (*)",
        )
        if not source:
            return
        camera = self.config.camera
        try:
            info = import_oak_calibration(
                source, self._oak_factory_output_path(), self._color_output_path(),
                camera["color_width"], camera["color_height"],
            )
            self._accept_oak_calibration(info)
            self._log("OAK 官方标定已导入：{}".format(source))
        except Exception as error:
            self._show_error("OAK 标定导入失败", error)

    def export_oak_eeprom(self):
        if self.camera_worker is not None:
            self._show_error("无法导出 OAK EEPROM", "请先断开实时相机连接")
            return
        camera = self.config.camera
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            info = export_connected_oak_eeprom(
                self._oak_factory_output_path(), self._color_output_path(),
                camera["color_width"], camera["color_height"],
                mxid=camera.get("mxid", ""),
            )
            self._accept_oak_calibration(info)
            self._log("已从 OAK EEPROM 导出官方 JSON")
        except Exception as error:
            self._show_error("OAK EEPROM 导出失败", error)
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

    def launch_oak_calibration_tool(self):
        if self.camera_worker is not None:
            self.disconnect_camera()
        camera = self.config.camera
        tool = camera.get("calibration_tool", {})
        python = Path(str(tool.get("python", ""))).expanduser()
        script = Path(str(tool.get("script", ""))).expanduser()
        if not python.is_file() or not script.is_file():
            self._show_error("官方标定工具不可用", "未找到独立 OAK 标定环境或 calibrate.py")
            return
        answer = QMessageBox.question(
            self, "启动 Luxonis 标定",
            "官方程序完成求解后会写入 OAK 用户 EEPROM。请确认方格/Marker 实测尺寸"
            "与界面一致，并确保相机供电稳定。继续吗？",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        tool["square_size_cm"] = self.oak_square_size.value()
        tool["marker_size_cm"] = self.oak_marker_size.value()
        camera["calibration_tool"] = tool
        self.config.save()
        command = [
            str(python), str(script),
            "-s", str(self.oak_square_size.value()),
            "-ms", str(self.oak_marker_size.value()),
            "-brd", str(tool.get("board", "OAK-D-PRO")),
            "-scp", str(self._oak_factory_output_path()),
        ]
        try:
            subprocess.Popen(command, cwd=str(script.parent))
            self.oak_result.setText(
                "官方标定程序已启动；完成后点击“导入 OAK 官方标定 JSON”刷新内参"
            )
            self._log("已启动 Luxonis calibrate.py")
        except Exception as error:
            self._show_error("官方标定程序启动失败", error)

    def _stream_anchor(self, stage=None):
        stage = self.current_stage if stage is None else int(stage)
        if self.config.camera["backend"] == "astra_ros" and stage == self.RGBD:
            return "ir"
        # RGB localization and YOLO must not inherit the much lower Depth FPS.
        # Their bundles still carry the nearest immutable Depth frame for RViz.
        return "color"

    def _active_depth_mode_value(self, field, default):
        camera = self.config.camera
        if camera.get("backend") == "astra_ros":
            mode = camera.get("depth_modes", {}).get(camera.get("depth_mode"), {})
            if field in mode:
                return mode[field]
        return camera.get(field, default)

    def _stage_due(self, name, fps):
        now = time.monotonic()
        previous = float(self.last_stage_process_s.get(name, 0.0))
        if now - previous < 1.0 / max(float(fps), 0.1):
            return False
        self.last_stage_process_s[name] = now
        return True

    def _start_live_processing_workers(self):
        self._stop_live_processing_workers()
        tag_worker = TagLocalizationWorker(self.config.path)
        tag_worker.result_ready.connect(self._receive_tag_localization)
        tag_worker.failed.connect(self._tag_localization_failed)
        self.tag_pose_worker = tag_worker
        tag_worker.start()
        depth_worker = DepthPreviewWorker(self.config.path)
        depth_worker.result_ready.connect(self._receive_depth_preview)
        depth_worker.failed.connect(self._depth_preview_failed)
        self.depth_preview_worker = depth_worker
        depth_worker.start()
        checker_worker = CheckerboardPreviewWorker(self.config.path)
        checker_worker.result_ready.connect(self._receive_checkerboard_preview)
        checker_worker.failed.connect(self._checkerboard_preview_failed)
        self.checkerboard_preview_worker = checker_worker
        checker_worker.start()

    def _restart_live_processing_workers(self):
        if self.camera_connected:
            self._start_live_processing_workers()

    def _stop_live_processing_workers(self):
        for name in (
            "tag_pose_worker", "depth_preview_worker",
            "checkerboard_preview_worker",
        ):
            worker = getattr(self, name, None)
            if worker is not None:
                worker.stop()
                worker.wait(5000)
            setattr(self, name, None)
        self.latest_tag_payload = None
        self.last_tag_submit_s = 0.0

    def _tag_localization_failed(self, message):
        self._log("后台 Tag 定位失败：{}".format(message))
        self.statusBar().showMessage(str(message), 5000)

    def _depth_preview_failed(self, message):
        self._log("后台 Depth 预览失败：{}".format(message))
        self.statusBar().showMessage(str(message), 5000)

    def _checkerboard_preview_failed(self, message):
        self._log("后台棋盘格预览失败：{}".format(message))
        self.statusBar().showMessage(str(message), 5000)
        self.checkerboard_preview_worker = None
        if self.current_stage == self.HAND_EYE:
            self.metric_detection.setText("棋盘角点  后台失败")
            self.preview_state.setText("棋盘格后台线程失败：{}".format(message))

    def _receive_checkerboard_preview(self, payload):
        error = payload.get("error")
        self.color_canvas.set_frame(payload["preview"])
        if error:
            self.metric_detection.setText("棋盘角点  识别失败")
            self.preview_state.setText("棋盘格检测失败：{}".format(error))
            return
        observation = payload.get("observation")
        found = int(payload.get("found", 0))
        total = int(payload.get("corner_count", 0))
        self.metric_detection.setText("棋盘角点  {} / {}".format(found, total))
        try:
            sample_count = len(self.sample_store.entries())
        except ValueError:
            sample_count = 0
        self.metric_progress.setText(
            "手眼样本  {} / {}".format(
                sample_count,
                int(self.config.data["hand_eye"].get("minimum_samples", 8)),
            )
        )
        if observation is not None:
            valid = bool(getattr(observation, "valid", False))
            reason = getattr(observation, "reason", "")
            self.preview_state.setText(
                "棋盘格{} · {} · 后台 {:.0f} ms".format(
                    "有效" if valid else "未通过",
                    reason,
                    float(payload.get("elapsed_ms", 0.0)),
                )
            )
        else:
            self.preview_state.setText(str(error or "棋盘格检测无结果"))

    def _receive_depth_preview(self, payload):
        self.last_aligned_preview = payload["preview"]
        self.last_depth_stamp = payload.get("stamp")
        self.depth_caption.setText(
            "{} · {:.0f} ms 后台".format(
                payload["caption"], float(payload.get("elapsed_ms", 0.0))
            )
        )
        if self.current_stage in (
            self.LOCALIZATION, self.SEGMENTATION, self.PLANNING
        ):
            self.depth_canvas.set_frame(self.last_aligned_preview)

    def _submit_tag_localization(self, bundle):
        if self.tag_pose_worker is None:
            return
        fps = float(self.config.data["localization"].get("processing_fps", 12.0))
        now = time.monotonic()
        if now - self.last_tag_submit_s < 1.0 / max(fps, 0.1):
            return
        tcp = None
        robot_timestamp = None
        if self.robot_available.isChecked():
            tcp = self._pose_from_spins(self.verify_pose_spins)
            robot_timestamp = float(bundle.color_timestamp_s)
        self.tag_pose_worker.submit(
            bundle.color_bgr,
            bundle.color_timestamp_s,
            base_from_tcp=tcp,
            robot_timestamp_s=robot_timestamp,
            hide_tags=self.hide_tags.isChecked(),
        )
        self.last_tag_submit_s = now

    def _receive_tag_localization(self, payload):
        self.latest_tag_payload = payload
        detections = payload["detections"]
        pose = payload["pose"]
        if self.rviz_worker is not None and pose.valid:
            self.rviz_worker.submit_pose(pose)
        hand_eye_uses_tags = (
            self.current_stage == self.HAND_EYE
            and self.hand_target_combo.currentData() == APRILTAG_MAP_TARGET
        )
        if self.current_stage == self.TAGS or hand_eye_uses_tags:
            mapped = sorted(set(detections).intersection(TagMap(self.config).ids))
            self.color_canvas.set_frame(
                self._draw_tags(payload["frame"], detections)
            )
            self.metric_detection.setText("RGB Tag  {} 个".format(len(detections)))
            self.metric_progress.setText(
                "已登记  {} 个 ID".format(len(TagMap(self.config).ids))
            )
            self.preview_state.setText(
                "可见 {} · 已登记 {}".format(sorted(detections), mapped)
            )
        elif self.current_stage == self.LOCALIZATION:
            preview = self._draw_tags(payload["frame"], detections)
            cv2.putText(
                preview, pose.source, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (30, 210, 120) if pose.valid else (30, 30, 230), 2, cv2.LINE_AA,
            )
            self.color_canvas.set_frame(preview)
            self.metric_detection.setText("RGB Tag  {} 个".format(len(detections)))
            self.preview_state.setText(
                "Tag 定位 {:.1f} ms · 后台 {:.0f} Hz 上限".format(
                    float(payload.get("elapsed_ms", 0.0)),
                    float(self.config.data["localization"].get("processing_fps", 12.0)),
                )
            )
            self._present_localization(pose)

    def toggle_camera(self):
        if self.camera_worker is not None:
            self.disconnect_camera()
            return
        self.connect_button.setEnabled(False)
        self.orbbec_intrinsics_synced = False
        worker = RgbdCameraWorker(
            self.config.runtime_camera(), self.color_device.currentText(),
            initial_laser_enabled=self.ir_emitter_toggle.isChecked(),
            initial_anchor=self._stream_anchor(),
        )
        worker.connected.connect(self._camera_connected)
        worker.bundle_ready.connect(self._receive_bundle)
        worker.laser_changed.connect(self._laser_changed)
        worker.failed.connect(self._camera_failed)
        worker.finished.connect(self._camera_finished)
        self.camera_worker = worker
        worker.start()

    def toggle_controller_test(self):
        if self.controller_worker is not None:
            worker, self.controller_worker = self.controller_worker, None
            worker.stop()
            worker.wait(3000)
            self.controller_connect_button.setText("保存并开始只读测试")
            self.controller_result.setText("控制器只读测试已停止")
            return
        try:
            mapping_document = yaml.safe_load(
                self.controller_mapping.toPlainText()
            ) or {}
            if not isinstance(mapping_document, dict):
                raise ValueError("寄存器 YAML 根节点必须是 mapping")
            state_registers = mapping_document.get("state_registers", {}) or {}
            state_codec = mapping_document.get("state_codec", {}) or {}
            if not isinstance(state_registers, dict) or not isinstance(state_codec, dict):
                raise ValueError("state_registers 和 state_codec 必须是 mapping")
            settings = dict(self.config.data.get("controller", {}))
            settings.update({
                "enabled": self.controller_enabled.isChecked(),
                "host": self.controller_host.text().strip(),
                "port": int(self.controller_port.value()),
                "unit_id": int(self.controller_unit.value()),
                "state_registers": state_registers,
                "state_codec": state_codec,
            })
            self.config.data["controller"] = settings
            self.config.save()
        except Exception as error:
            self._show_error("控制器配置无效", error)
            return
        poll_hz = max(float(settings.get("state_poll_hz", 2.0)), 0.1)
        worker = ControllerStateWorker(settings, interval_s=1.0 / poll_hz)
        worker.state_ready.connect(self._receive_controller_state)
        worker.failed.connect(self._controller_test_failed)
        worker.finished.connect(self._controller_test_finished)
        self.controller_worker = worker
        self.controller_connect_button.setText("停止测试")
        self.controller_result.setText("正在连接控制器，只执行状态读取……")
        worker.start()

    @staticmethod
    def _controller_value(value, precision=3):
        if value is None:
            return "未知/未映射"
        if isinstance(value, (tuple, list)):
            return "[{}]".format(", ".join("{:.{}f}".format(float(item), precision) for item in value))
        if isinstance(value, bool):
            return "是" if value else "否"
        return str(value)

    def _receive_controller_state(self, state):
        self.controller_state = state
        values = {
            "connected": state.connected,
            "servo_on": state.servo_on,
            "emergency_stop": state.emergency_stop,
            "moving": state.moving,
            "joint_deg": state.joint_deg,
            "tcp_xyz_mm": state.tcp_xyz_mm,
            "tcp_rpy_deg": state.tcp_rpy_deg,
            "point_name": state.point_name or "未知/未映射",
            "coordinate_system": {
                0: "0 关节", 1: "1 直角", 2: "2 工具", 3: "3 用户",
            }.get(state.coordinate_system, "未知/未映射"),
            "angle_unit": {0: "0 度", 1: "1 弧度"}.get(
                state.angle_unit, "未知/未映射"
            ),
            "reserved": state.reserved,
            "axes": state.axes,
            "tool_id": state.tool_id,
            "user_id": state.user_id,
            "shape": state.shape,
            "initial_shape": state.initial_shape,
            "shape_changed": state.shape_changed,
            "alarm": "{} {}".format(state.alarm.code if state.alarm.code is not None else "--", state.alarm.text or "无"),
        }
        for name, value in values.items():
            self.controller_state_labels[name].setText(self._controller_value(value))
        if state.tcp_xyz_mm is not None and state.tcp_rpy_deg is not None:
            values = tuple(state.tcp_xyz_mm) + tuple(state.tcp_rpy_deg)
            for spins in (self.verify_pose_spins, self.hand_pose_spins):
                for spin, value in zip(spins, values):
                    spin.setValue(float(value))
        if state.error:
            self.controller_result.setText("读取失败：{}".format(state.error))
        elif not state.raw_registers:
            self.controller_result.setText("已连接，但 state_registers 为空；请按官方寄存器表填写，当前没有猜测地址。")
        else:
            self.controller_result.setText("状态已刷新；原始字段：{}".format(state.raw_registers))

    def _controller_test_failed(self, message):
        self.controller_result.setText("连接失败：{}".format(message))
        self._log("控制器只读测试失败：{}".format(message))

    def _controller_test_finished(self):
        self.controller_worker = None
        self.controller_connect_button.setText("保存并开始只读测试")

    def save_controller_safe_point(self):
        state = self.controller_state
        try:
            if (
                not state.connected or state.joint_deg is None or state.axes is None
                or state.tcp_xyz_mm is None
            ):
                raise ValueError("控制器、Axis1..7、关节角和 TCP 状态必须全部有效")
            if state.moving is not False:
                raise ValueError("仅允许在控制器明确报告已停止时保存安全点")
            if state.emergency_stop is not False:
                raise ValueError("急停状态未知或已触发，不能保存安全点")
            if state.alarm.active:
                raise ValueError("当前仍有活动报警，不能标记为无奇异安全点")
            if (
                state.shape is None or state.tool_id is None or state.user_id is None
                or state.reserved is None
            ):
                raise ValueError(
                    "必须先从控制器读取 shape、Tool ID、User ID 和两个保留字段"
                )
            point = InexbotPoint(
                name="P9000", coordinate_system=0, angle_unit=0,
                shape=state.shape, tool_id=state.tool_id, user_id=state.user_id,
                reserved=state.reserved,
                axes=state.axes,
            )
            self.config.data.setdefault("safety", {}).setdefault("recovery", {})["safe_movej_points"] = [{
                "name": point.name,
                "coordinate_system": point.coordinate_system,
                "angle_unit": point.angle_unit,
                "shape": point.shape,
                "tool_id": point.tool_id,
                "user_id": point.user_id,
                "reserved": list(point.reserved),
                "axes": list(point.axes),
            }]
            self.config.save()
            self.safe_point_result.setText(
                "已保存 P9000：shape={} Tool={} User={}，Axis1..7={}。"
                "现场确认路径无碰撞后，才能开启 auto_recover。".format(
                    point.shape, point.tool_id, point.user_id, list(point.axes)
                )
            )
        except Exception as error:
            self._show_error("安全点保存失败", error)

    def disconnect_camera(self):
        if self.camera_worker is None:
            return
        self._stop_live_processing_workers()
        self.camera_worker.stop()
        self.camera_worker.wait(5000)
        self.camera_worker = None
        self.camera_connected = False
        self.connect_button.setText(
            "连接 {}".format(
                "Astra" if self.config.camera["backend"] == "astra_ros" else (
                    "Gemini" if self.config.camera["backend"] == "orbbec_ros"
                    else "OAK-D Pro"
                )
            )
        )
        self.connect_button.setEnabled(True)
        self.connection_badge.setText("未连接")
        self.projector_state.setText("IR 投影器：待连接")

    def _camera_connected(self):
        self.camera_connected = True
        if self.config.camera["backend"] != "orbbec_ros":
            self._start_live_processing_workers()
        self.connect_button.setEnabled(True)
        self.connect_button.setText("断开相机")
        self.connection_badge.setText("RGB-D 已连接")
        self.preview_state.setText("等待 RGB / IR / Depth")
        if self.config.camera["backend"] == "astra_ros":
            self._log("Astra RGB UVC 与 IR/Depth ROS 已启动")
        elif self.config.camera["backend"] == "orbbec_ros":
            self._log("Gemini ROS 已启动，等待实机 CameraInfo 后启用定位")
        else:
            self._log("OAK-D Pro DepthAI RGB/IR/Depth 已启动（Depth 对齐 RGB）")

    def _laser_changed(self, enabled):
        prefix = {
            "astra_ros": "Astra IR",
            "orbbec_ros": "Gemini IR",
            "oak_depthai": "OAK 点阵",
        }.get(self.config.camera["backend"], "IR")
        self.projector_state.setText(
            "{}发射器：{}".format(prefix, "已打开（IR / Depth 可用）" if enabled else "已关闭")
        )

    def set_ir_emitter_enabled(self, enabled):
        enabled = bool(enabled)
        if self.camera_worker is not None:
            self.camera_worker.request_laser(
                enabled, anchor=self._stream_anchor()
            )
            self.projector_state.setText(
                "IR 投影器：正在{}".format("打开" if enabled else "关闭")
            )
        else:
            self.projector_state.setText(
                "IR 投影器：连接相机后将{}".format(
                    "打开" if enabled else "保持关闭"
                )
            )
        if hasattr(self, "log_output"):
            self._log(
                "IR 投影器请求：{}{}".format(
                    "开启" if enabled else "关闭",
                    "（太阳光标定模式）" if not enabled else "",
                )
            )

    def _camera_failed(self, message):
        self._show_error("深度相机启动失败", message)

    def _camera_finished(self):
        self._stop_live_processing_workers()
        self.camera_worker = None
        self.camera_connected = False
        self.connect_button.setText(
            "连接 {}".format(
                "Astra" if self.config.camera["backend"] == "astra_ros" else (
                    "Gemini" if self.config.camera["backend"] == "orbbec_ros"
                    else "OAK-D Pro"
                )
            )
        )
        self.connect_button.setEnabled(True)
        self.connection_badge.setText("未连接")

    def _receive_bundle(self, bundle):
        if self.config.camera["backend"] == "orbbec_ros":
            self._sync_orbbec_factory_intrinsics(bundle)
        self.bundle = bundle
        self.metric_source.setText(
            "RGB {}x{} · Depth {}".format(
                bundle.color_bgr.shape[1], bundle.color_bgr.shape[0],
                "--" if bundle.depth_m is None else "{}x{}".format(
                    bundle.depth_m.shape[1], bundle.depth_m.shape[0]
                ),
            )
        )
        processors = (
            self._process_rgb, self._process_rgbd, self._process_tags,
            self._process_hand_eye, self._process_localization,
            self._process_segmentation, self._process_future_stage,
            self._process_future_stage,
        )
        processors[self.current_stage](bundle)
        self._show_auxiliary(bundle)

    def _sync_orbbec_factory_intrinsics(self, bundle):
        if self.orbbec_intrinsics_synced:
            return
        intrinsics = bundle.color_intrinsics
        if intrinsics is None:
            return
        expected = (
            int(self.config.camera["color_width"]),
            int(self.config.camera["color_height"]),
        )
        actual = (int(intrinsics.width), int(intrinsics.height))
        if actual != expected:
            raise ValueError(
                "Gemini CameraInfo 为 {}x{}，当前配置要求 {}x{}".format(
                    actual[0], actual[1], expected[0], expected[1]
                )
            )
        save_camera_yaml(
            str(self._color_output_path()),
            intrinsics.matrix,
            intrinsics.distortion,
            actual,
            "orbbec_gemini_{}".format(
                self.config.camera.get("expected_serial", "device")
            ),
            distortion_model="rational_polynomial",
        )
        self.color_intrinsics_cache = None
        self.orbbec_intrinsics_synced = True
        self.rgb_result.setText(
            "已读取 Gemini 出厂内参 · {}x{} · fx {:.3f} · fy {:.3f}".format(
                actual[0], actual[1], intrinsics.matrix[0, 0], intrinsics.matrix[1, 1]
            )
        )
        self.orbbec_result.setText(
            "CameraInfo 已读取并保存 · Depth 已由驱动对齐到 RGB"
        )
        self._start_live_processing_workers()
        self._refresh_readiness()
        self._log("Gemini 实机 CameraInfo 已刷新：{}".format(self._color_output_path()))

    def _show_auxiliary(self, bundle):
        now = time.monotonic()
        if bundle.ir_image is None:
            self.ir_canvas.clear("等待 IR")
        elif (
            self.current_stage != self.RGBD
            and bundle.ir_timestamp_s != self.last_ir_preview_stamp
            and now - self.last_ir_preview_s >= 0.1
        ):
            self.ir_canvas.set_frame(infrared_preview(bundle.ir_image))
            self.last_ir_preview_stamp = bundle.ir_timestamp_s
            self.last_ir_preview_s = now
        if bundle.depth_m is None:
            self.depth_canvas.clear("等待 Depth")
        elif (
            self.current_stage not in (
                self.LOCALIZATION, self.SEGMENTATION, self.PLANNING
            )
            and bundle.depth_timestamp_s != self.last_raw_depth_preview_stamp
            and now - self.last_depth_preview_s >= 0.1
        ):
            self.depth_caption.setText("原始 Depth")
            self.depth_canvas.set_frame(depth_colormap(bundle.depth_m))
            self.last_raw_depth_preview_stamp = bundle.depth_timestamp_s
            self.last_depth_preview_s = now

    def _detect_color_charuco(self, color):
        corners, ids, _, _ = self.charuco_detector.detectBoard(
            cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        )
        if ids is None:
            return self._empty_detection()
        return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1).astype(np.int32)

    @staticmethod
    def _draw_charuco(image, detection):
        preview = image.copy()
        corners, ids = detection
        if len(ids):
            cv2.aruco.drawDetectedCornersCharuco(
                preview, corners.reshape(-1, 1, 2), ids.reshape(-1, 1)
            )
        return preview

    def _process_rgb(self, bundle):
        if not self._stage_due("rgb_intrinsics", 10.0):
            return
        if self.config.camera["backend"] == "orbbec_ros":
            self.color_canvas.set_frame(bundle.color_bgr)
            intrinsics = bundle.color_intrinsics
            self.metric_detection.setText("Gemini CameraInfo")
            self.metric_progress.setText("出厂内参  已读取")
            if intrinsics is not None:
                self.preview_state.setText(
                    "出厂内参 · fx {:.2f} · fy {:.2f}".format(
                        intrinsics.matrix[0, 0], intrinsics.matrix[1, 1]
                    )
                )
            return
        self.color_detection = self._detect_color_charuco(bundle.color_bgr)
        count = len(self.color_detection[1])
        self.rgb_corner_value.setText(str(count))
        self.metric_detection.setText("RGB  {} 个角点".format(count))
        self.metric_progress.setText("内参  {} / 20".format(len(self.rgb_object_points)))
        self.color_canvas.set_frame(self._draw_charuco(bundle.color_bgr, self.color_detection))
        self.preview_state.setText("RGB 内参采集")

    def _process_rgbd(self, bundle):
        if not self._stage_due("rgb_ir_calibration", 8.0):
            return
        if self.config.camera["backend"] == "orbbec_ros":
            self.color_canvas.set_frame(bundle.color_bgr)
            if bundle.ir_image is not None:
                self.ir_canvas.set_frame(infrared_preview(bundle.ir_image))
            self.metric_detection.setText("Gemini 工厂 RGB-D")
            self.metric_progress.setText(
                "Depth→RGB  {}".format(
                    "已对齐" if bundle.depth_aligned_to_color else "未对齐"
                )
            )
            self.preview_state.setText("Orbbec SDK CameraInfo / 工厂参数")
            return
        if self.config.camera["backend"] == "oak_depthai":
            self.color_canvas.set_frame(bundle.color_bgr)
            if bundle.ir_image is not None:
                self.ir_canvas.set_frame(infrared_preview(bundle.ir_image))
            self.metric_detection.setText("OAK EEPROM 标定")
            self.metric_progress.setText(
                "JSON  {}".format(
                    "已导入" if self._oak_factory_output_path().is_file() else "待导入"
                )
            )
            self.preview_state.setText("OAK 工厂标定 / 官方 JSON")
            return
        self.color_detection = self._detect_color_charuco(bundle.color_bgr)
        if bundle.ir_image is None:
            self.ir_detection = self._empty_detection()
            ir_preview = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            self.ir_detection = detect_charuco(bundle.ir_image)
            ir_preview = infrared_preview(bundle.ir_image)
        common = set(self.color_detection[1]).intersection(self.ir_detection[1])
        self.rgbd_color_corner_value.setText(str(len(self.color_detection[1])))
        self.rgbd_ir_corner_value.setText(str(len(self.ir_detection[1])))
        self.rgbd_common_value.setText(str(len(common)))
        self.metric_detection.setText("共同角点  {}".format(len(common)))
        self.metric_progress.setText("RGB/IR  {} / 12 对".format(len(self.rgbd_pairs)))
        self.color_canvas.set_frame(self._draw_charuco(bundle.color_bgr, self.color_detection))
        self.ir_canvas.set_frame(self._draw_charuco(ir_preview, self.ir_detection))
        self.preview_state.setText(
            "RGB-IR 标定 · IR 投影器{}".format(
                "开启" if self.ir_emitter_toggle.isChecked() else "关闭 / 太阳光"
            )
        )

    def _tag_detections(self, color):
        if self.tag_localizer is None:
            self.tag_localizer = HybridLocalizer(self.config).visual
        return self.tag_localizer.detect(color)

    @staticmethod
    def _draw_tags(color, detections):
        preview = color.copy()
        if not detections:
            return preview
        ids = np.asarray(sorted(detections), dtype=np.int32).reshape(-1, 1)
        corners = [
            np.asarray(detections[int(tag_id)], dtype=np.float32).reshape(1, 4, 2)
            for tag_id in ids.reshape(-1)
        ]
        cv2.aruco.drawDetectedMarkers(preview, corners, ids)
        for tag_id in ids.reshape(-1):
            tl, tr, br, bl = np.asarray(detections[int(tag_id)]).reshape(4, 2)
            origin = tuple(np.rint(br).astype(int))
            x_end = tuple(np.rint(br + 0.55 * (tr - br)).astype(int))
            y_end = tuple(np.rint(br + 0.55 * (bl - br)).astype(int))
            cv2.circle(preview, origin, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.arrowedLine(preview, origin, x_end, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(preview, origin, y_end, (0, 210, 0), 2, cv2.LINE_AA)
            # A circle with a center dot is the standard 2-D symbol for an
            # axis pointing out of the Tag plane toward the viewer.
            cv2.circle(preview, origin, 10, (255, 100, 0), 2, cv2.LINE_AA)
            cv2.circle(preview, origin, 3, (255, 100, 0), -1, cv2.LINE_AA)
            cv2.putText(
                preview, "+X", (x_end[0] + 5, x_end[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                preview, "+Y", (y_end[0] + 5, y_end[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                preview, "+Z UP", (origin[0] + 12, origin[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 100, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                preview, "BR {}".format(int(tag_id)), (origin[0] + 8, origin[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
            )
        return preview

    def _process_tags(self, bundle):
        self._submit_tag_localization(bundle)
        if self.latest_tag_payload is None and self._stage_due("tag_raw_preview", 15.0):
            self.color_canvas.set_frame(bundle.color_bgr)
            self.preview_state.setText("后台检测 Tag")

    def _process_hand_eye(self, bundle):
        target = self._hand_eye_ui_target_settings(require_complete=False)
        if target["type"] != CHECKERBOARD_TARGET:
            self._process_tags(bundle)
            return
        if not self._stage_due("hand_eye_checkerboard", 5.0):
            return
        if not target["checkerboard"].get("configured", False):
            self.color_canvas.set_frame(bundle.color_bgr)
            self.metric_detection.setText("棋盘角点  待设置")
            self.metric_progress.setText("长短边格数未填写")
            self.preview_state.setText("请填写长边和短边的黑白方格总数")
            return
        if self.checkerboard_preview_worker is None:
            self.color_canvas.set_frame(bundle.color_bgr)
            self.metric_detection.setText("棋盘角点  后台未启动")
            self.preview_state.setText("棋盘格后台检测线程尚未启动")
            return
        # Detection/PnP is intentionally off the UI thread.  The worker keeps
        # only the newest frame, so a 1920x1080 checkerboard search cannot
        # freeze the hand-eye page while the user is dragging/sliding widgets.
        self.checkerboard_preview_worker.submit(
            bundle.color_bgr,
            bundle.color_timestamp_s,
            target["checkerboard"],
        )

    def _load_color_intrinsics(self):
        path = self._color_output_path()
        key = (path, path.stat().st_mtime)
        if self.color_intrinsics_cache is None or self.color_intrinsics_cache[0] != key:
            self.color_intrinsics_cache = (key,) + load_camera_intrinsics(path)
        return self.color_intrinsics_cache[1:]

    def _process_localization(self, bundle):
        self._submit_tag_localization(bundle)
        if self.latest_tag_payload is None and self._stage_due("localization_raw_preview", 15.0):
            self.color_canvas.set_frame(bundle.color_bgr)
            self.preview_state.setText("后台计算相机位姿")
        self._update_aligned_depth(bundle)

    def _process_segmentation(self, bundle):
        self._submit_tag_localization(bundle)
        if (
            self.segmentation_last_result is None
            and self._stage_due("segmentation_raw_preview", 15.0)
        ):
            self.color_canvas.set_frame(bundle.color_bgr)
        if self.segmentation_worker is None:
            self.preview_state.setText("待加载分割模型")
            self.metric_detection.setText("模型  未加载")
            self.metric_progress.setText("连续  0")
            return
        if not self.segmentation_model_ready:
            self.preview_state.setText("正在加载分割模型")
            self.metric_detection.setText("模型  加载中")
            return
        self._submit_segmentation(bundle)
        self._update_aligned_depth(bundle)
        if self.segmentation_last_result is None:
            self.preview_state.setText("等待首帧分割结果")

    def _submit_segmentation(self, bundle):
        if self.segmentation_worker is None or not self.segmentation_model_ready:
            return
        now = time.monotonic()
        interval = float(
            self.config.data["segmentation_validation"].get(
                "preview_interval_s", 0.2
            )
        )
        if now - self.segmentation_last_submit_s < interval:
            return
        tcp = None
        robot_timestamp = None
        if self.robot_available.isChecked():
            tcp = self._pose_from_spins(self.verify_pose_spins)
            robot_timestamp = now
        tag_payload = self.latest_tag_payload
        self.segmentation_worker.submit(
            bundle.color_bgr,
            bundle.color_timestamp_s,
            context={
                "bundle": bundle,
                "base_from_tcp": tcp,
                "robot_timestamp_s": robot_timestamp,
                "hide_tags": self.hide_tags.isChecked(),
                "localized_pose": (
                    None if tag_payload is None else tag_payload.get("pose")
                ),
            },
        )
        self.segmentation_last_submit_s = now

    def _process_future_stage(self, bundle):
        if (
            self.segmentation_last_result is None
            and self._stage_due("future_raw_preview", 15.0)
        ):
            self.color_canvas.set_frame(bundle.color_bgr)
        if self.current_stage == self.PLANNING:
            self._submit_tag_localization(bundle)
            self._submit_segmentation(bundle)
            self._update_aligned_depth(bundle)
            self.preview_state.setText("分割定位点云验证 · 确定性俯视后端已就绪")
            count = len(
                (self.segmentation_last_result or {}).get("instances", [])
            )
            self.metric_detection.setText("物体  {} 个".format(count))
            self.metric_progress.setText(
                "分割  {}".format("已验证" if self.config.segmentation_valid else "未验证")
            )
        else:
            self.preview_state.setText("执行后端就绪 · 实机仍保持 fail-closed")
            self.metric_detection.setText("运动  禁止")
            self.metric_progress.setText("Dry-run  开启")

    def _update_aligned_depth(self, bundle):
        if bundle.depth_m is None or self.depth_preview_worker is None:
            return
        now = time.monotonic()
        preview_interval = 1.0 / float(
            self._active_depth_mode_value("depth_preview_fps", 8.0)
        )
        new_depth = bundle.depth_timestamp_s != self.last_depth_stamp
        should_refresh = new_depth and (
            self.last_aligned_preview is None
            or now - self.last_depth_preview_s >= preview_interval
        )
        if should_refresh:
            self.depth_preview_worker.submit(bundle)
            # Mark submission time immediately so a 30 FPS RGB stream cannot
            # enqueue the same slow high-resolution Depth frame repeatedly.
            self.last_depth_stamp = bundle.depth_timestamp_s
            self.last_depth_preview_s = now

    def _present_localization(self, result):
        names = {
            SOURCE_TAG_VISUAL: "Tag 视觉",
            SOURCE_TAG_VISUAL_HELD: "Tag 视觉保持",
            SOURCE_TCP_FALLBACK: "TCP + 手眼",
        }
        self.loc_source.setText(names.get(result.source, "无有效位姿"))
        self.loc_tags.setText(", ".join(map(str, result.used_tag_ids)) or "--")
        self.loc_quality.setText(
            "--" if result.rms_reprojection_error_px is None
            else "RMS {:.3f} px".format(result.rms_reprojection_error_px)
        )
        if result.valid:
            xyz_m, rpy = xyz_rpy_from_transform(result.base_from_camera)
            self.loc_xyz.setText("{:.1f}, {:.1f}, {:.1f}".format(*(xyz_m * 1000.0)))
            self.loc_rpy.setText("{:.2f}, {:.2f}, {:.2f}".format(*rpy))
            self.localization_result.setText("定位有效 · {}".format(result.reason))
        else:
            self.loc_xyz.setText("--")
            self.loc_rpy.setText("--")
            self.localization_result.setText("定位无效 · {}".format(result.reason))

    def capture_rgb_view(self):
        if self.bundle is None:
            self._show_error("无法采样", "尚无 RGB 图像")
            return
        corners, ids = self.color_detection
        if len(ids) < 12:
            self._show_error("RGB 姿态被拒绝", "至少需要 12 个 ChArUco 角点")
            return
        size = (self.bundle.color_bgr.shape[1], self.bundle.color_bgr.shape[0])
        if self.rgb_image_size is not None and size != self.rgb_image_size:
            self._show_error("RGB 分辨率变化", "内参采样期间不能改变分辨率")
            return
        self.rgb_image_size = size
        self.rgb_object_points.append(self.board_points[ids].reshape(-1, 1, 3).copy())
        self.rgb_image_points.append(corners.reshape(-1, 1, 2).astype(np.float32).copy())
        self.rgb_all_pixels.append(corners.copy())
        count = len(self.rgb_object_points)
        self.rgb_view_value.setText("{} / 20".format(count))
        self.rgb_progress.setValue(min(count, 40))
        self._log("已采集 RGB 内参姿态 {}，角点 {}".format(count, len(ids)))

    def clear_rgb_samples(self):
        self.rgb_object_points.clear()
        self.rgb_image_points.clear()
        self.rgb_all_pixels.clear()
        self.rgb_image_size = None
        self.rgb_view_value.setText("0 / 20")
        self.rgb_progress.setValue(0)
        self.rgb_result.setText("RGB 内参采样已清空")

    def solve_rgb_intrinsics(self):
        if len(self.rgb_object_points) < 20:
            self._show_error("RGB 内参无法计算", "至少需要 20 个不同姿态")
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            result = calibrate(self.rgb_object_points, self.rgb_image_points, self.rgb_image_size)
            maximum = float(self.config.camera.get("max_color_intrinsic_rms_px", 0.8))
            if result["rms"] > maximum:
                raise ValueError("RGB RMS {:.4f} px 超过 {:.3f} px".format(result["rms"], maximum))
            pixels = np.concatenate(self.rgb_all_pixels, axis=0)
            coverage = (
                float(np.ptp(pixels[:, 0]) / self.rgb_image_size[0]),
                float(np.ptp(pixels[:, 1]) / self.rgb_image_size[1]),
            )
            output = self._color_output_path()
            save_camera_yaml(
                str(output), result["camera_matrix"], result["distortion"],
                self.rgb_image_size, "competition_rgb",
            )
            save_report(
                output.with_name(output.stem + "_report.yaml"), result,
                len(self.rgb_object_points), coverage,
            )
            self.config.data["hand_eye"]["tcp_from_color_camera"]["valid"] = False
            self.config.save()
            self.color_intrinsics_cache = None
            self._restart_live_processing_workers()
            self.rgb_result.setText(
                "RGB 有效 · RMS {:.4f} px · 覆盖 X {:.0%} / Y {:.0%}".format(
                    result["rms"], coverage[0], coverage[1]
                )
            )
            self._refresh_readiness()
        except Exception as error:
            self._show_error("RGB 内参失败", error)
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

    def capture_rgbd_pair(self):
        if self.config.camera["backend"] != "astra_ros":
            self._show_error("无需采集 RGB/IR", "OAK-D Pro 请导入官方 EEPROM 标定 JSON")
            return
        if self.bundle is None or self.bundle.ir_image is None:
            self._show_error("无法采集 RGB/IR", "等待同步 RGB 与 IR 图像")
            return
        camera = self.config.camera
        common = set(self.color_detection[1]).intersection(self.ir_detection[1])
        minimum = int(camera["minimum_rgbd_common_corners"])
        if len(common) < minimum:
            self._show_error(
                "RGB/IR 图像对被拒绝",
                "共同角点 {} 少于 {}。请让标定板同时完整出现在 RGB 和 IR 中，"
                "靠近一些并保持静止清晰。".format(len(common), minimum),
            )
            return
        delta = abs(float(self.bundle.color_timestamp_s) - float(self.bundle.ir_timestamp_s))
        maximum_delta = float(camera["maximum_sync_delta_s"])
        if delta > maximum_delta:
            self._show_error("RGB/IR 不同步", "时间差 {:.1f} ms 超限".format(delta * 1000.0))
            return
        color = self.bundle.color_bgr.copy()
        infrared = self.bundle.ir_image.copy()
        self.rgbd_pairs.append((color, infrared))
        pair_dir = ROOT / "output" / "rgbd_frames"
        pair_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + "_{:06d}".format(int(time.time_ns() % 1000000))
        cv2.imwrite(str(pair_dir / (stamp + "_rgb.png")), color)
        cv2.imwrite(str(pair_dir / (stamp + "_ir.png")), infrared)
        count = len(self.rgbd_pairs)
        self.rgbd_pair_value.setText("{} / 12".format(count))
        self.rgbd_progress.setValue(min(count, 30))
        self._log("RGB/IR 对 {} · 共同角点 {} · {:.1f} ms".format(count, len(common), delta * 1000.0))

    def clear_rgbd_samples(self):
        self.rgbd_pairs.clear()
        self.rgbd_pair_value.setText("0 / 12")
        self.rgbd_progress.setValue(0)
        self.rgbd_result.setText("本次 RGB/IR 内存采样已清空")

    def solve_rgbd(self):
        camera = self.config.camera
        if camera["backend"] != "astra_ros":
            self._show_error("无需 RGB/IR 求解", "OAK-D Pro 使用 DepthAI 工厂/官方标定")
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            color_intrinsics = camera_intrinsics_from_file(
                self.config.resolve_path(camera["color_intrinsics_file"])
            )
            result = calibrate_rgb_ir_pairs(
                self.rgbd_pairs, color_intrinsics,
                minimum_common_corners=int(camera["minimum_rgbd_common_corners"]),
                minimum_pairs=int(camera["minimum_rgbd_pairs"]),
                maximum_rms_px=float(camera["max_rgbd_rms_px"]),
            )
            save_rgbd_result(self._rgbd_output_path(), result)
            self.config.data["hand_eye"]["tcp_from_color_camera"]["valid"] = False
            self.config.save()
            self.depth_aligner = None
            calibration = result.calibration
            baseline = np.linalg.norm(calibration.color_from_depth[:3, 3]) * 1000.0
            rejected = len(getattr(result, "rejected_pair_indices", ()))
            self.rgbd_result.setText(
                "RGB-D 有效 · 使用 {} 对 · 剔除 {} 对 · RMS {:.4f} px · 基线 {:.2f} mm".format(
                    result.pairs_used, rejected,
                    calibration.rms_reprojection_error_px, baseline,
                )
            )
            if rejected:
                self._log(
                    "RGB-D 自动剔除离群图像对索引：{}".format(
                        list(result.rejected_pair_indices)
                    )
                )
            self._refresh_readiness()
        except Exception as error:
            self._show_error("RGB-D 标定失败", error)
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

    def _load_tag_table(self):
        tag_map = TagMap(self.config)
        self.tag_size.setValue(float(self.config.tag_map["tag_size_mm"]))
        for spin, value in zip(self.default_rpy, tag_map.default_rpy_deg):
            spin.setValue(float(value))
        self.tag_table.setRowCount(0)
        for tag_id in tag_map.ids:
            entry = tag_map.entry(tag_id)
            rpy = entry.get("base_from_tag_rpy_deg", tag_map.default_rpy_deg)
            self._append_tag_row([tag_id] + list(entry["bottom_right_xyz_mm"]) + list(rpy))

    def _append_tag_row(self, values):
        row = self.tag_table.rowCount()
        self.tag_table.insertRow(row)
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(int(value)) if column == 0 else "{:.2f}".format(float(value)))
            item.setTextAlignment(Qt.AlignCenter)
            self.tag_table.setItem(row, column, item)

    def add_tag_row(self):
        used = {
            int(self.tag_table.item(row, 0).text())
            for row in range(self.tag_table.rowCount())
            if self.tag_table.item(row, 0) is not None
        }
        tag_id = next(value for value in range(100, 10000) if value not in used)
        self._append_tag_row([tag_id, 0, 0, 0] + [spin.value() for spin in self.default_rpy])

    def remove_tag_row(self):
        if self.tag_table.currentRow() >= 0:
            self.tag_table.removeRow(self.tag_table.currentRow())

    def apply_default_rpy_to_rows(self):
        for row in range(self.tag_table.rowCount()):
            for offset, spin in enumerate(self.default_rpy):
                item = QTableWidgetItem("{:.2f}".format(spin.value()))
                item.setTextAlignment(Qt.AlignCenter)
                self.tag_table.setItem(row, 4 + offset, item)

    def save_tag_map(self):
        try:
            tags = {}
            for row in range(self.tag_table.rowCount()):
                values = []
                for column in range(7):
                    item = self.tag_table.item(row, column)
                    if item is None or not item.text().strip():
                        raise ValueError("第 {} 行存在空字段".format(row + 1))
                    values.append(float(item.text()))
                tag_id = int(values[0])
                if values[0] != tag_id or tag_id in tags:
                    raise ValueError("Tag ID 必须是唯一整数")
                tags[tag_id] = {
                    "bottom_right_xyz_mm": values[1:4],
                    "base_from_tag_rpy_deg": values[4:7],
                }
            self.config.tag_map["tag_size_mm"] = self.tag_size.value()
            self.config.tag_map["default_base_from_tag_rpy_deg"] = [spin.value() for spin in self.default_rpy]
            self.config.tag_map["tags"] = tags
            self.config.data["hand_eye"]["tcp_from_color_camera"]["valid"] = False
            self.config.save()
            self.tag_localizer = None
            self.hybrid_localizer = None
            self._restart_live_processing_workers()
            self.tag_result.setText("已保存 {} 个 Tag · 旧手眼结果已失效".format(len(tags)))
            self._refresh_samples()
            self._refresh_readiness()
        except Exception as error:
            self._show_error("Tag 地图保存失败", error)

    @staticmethod
    def _pose_from_spins(spins):
        values = [spin.value() for spin in spins]
        return transform_from_xyz_rpy_mm(values[:3], values[3:])

    def _build_tcp_pose_box(self):
        """NexBot TCP verification panel: read T_base_tcp before sampling."""
        box = QGroupBox(
            "TCP 通信验证（NexBot 官方协议 7000 端口）· 采样前先连接并核对示教器"
        )
        grid = QGridLayout(box)
        controller = self.config.data.get("controller", {}) or {}
        tcp_config = controller.get("nexbot_tcp", {}) or {}
        self.nexbot_host = QLineEdit(str(tcp_config.get("host") or ""))
        self.nexbot_host.setPlaceholderText("控制器 IP，例如 192.168.1.10")
        self.nexbot_port = QSpinBox()
        self.nexbot_port.setRange(1, 65535)
        self.nexbot_port.setValue(int(tcp_config.get("port_state") or 7000))
        self.nexbot_robot = QSpinBox()
        self.nexbot_robot.setRange(1, 4)
        self.nexbot_robot.setValue(int(tcp_config.get("robot") or 1))
        self.nexbot_autofill = QCheckBox("读取值自动填入“当前 T_base_tcp”")
        self.nexbot_autofill.setChecked(True)
        self.nexbot_toggle = action_button(
            self, "保存参数并连接读取 TCP", "SP_MediaPlay", True
        )
        self.nexbot_toggle.clicked.connect(self.toggle_nexbot_pose)
        self.nexbot_status = self._strong("未连接")
        for row, (title, value) in enumerate((
            ("IP", self.nexbot_host),
            ("7000 端口", self.nexbot_port),
            ("Robot 号", self.nexbot_robot),
        )):
            grid.addWidget(QLabel(title), row, 0)
            grid.addWidget(value, row, 1)
        grid.addWidget(self.nexbot_autofill, 3, 0, 1, 2)
        grid.addWidget(self.nexbot_toggle, 4, 0, 1, 2)
        grid.addWidget(self.nexbot_status, 5, 0, 1, 2)
        return box

    def toggle_nexbot_pose(self):
        worker = getattr(self, "nexbot_pose_worker", None)
        if worker is not None:
            worker.stop()
            worker.wait(1000)
            self.nexbot_pose_worker = None
            self.nexbot_toggle.setText("保存参数并连接读取 TCP")
            self.nexbot_status.setText("已断开")
            return
        self.save_nexbot_pose_settings()
        try:
            endpoint = pose_endpoint_from_config(self.config.data.get("controller", {}))
        except Exception as error:
            self._show_error("TCP 连接参数无效", error)
            return
        worker = NexBotPoseWorker(endpoint)
        worker.pose_ready.connect(self._receive_nexbot_pose)
        worker.failed.connect(self._nexbot_pose_failed)
        worker.finished.connect(self._nexbot_pose_finished)
        self.nexbot_pose_worker = worker
        self.nexbot_toggle.setText("断开读取")
        self.nexbot_status.setText("连接中……")
        worker.start()

    def save_nexbot_pose_settings(self):
        controller = self.config.data.setdefault("controller", {})
        tcp_config = controller.setdefault("nexbot_tcp", {})
        tcp_config.update({
            "host": self.nexbot_host.text().strip(),
            "port_motion": 6000,
            "port_state": int(self.nexbot_port.value()),
            "robot": int(self.nexbot_robot.value()),
        })
        self.config.save()

    def _receive_nexbot_pose(self, ok, payload):
        if ok:
            xyz_mm, rpy_deg = payload
            self.nexbot_status.setText(
                "已连接 · TCP X/Y/Z {:.1f} {:.1f} {:.1f} mm · R/P/Y {:.2f} {:.2f} {:.2f}°"
                .format(*xyz_mm, *rpy_deg)
            )
            if self.nexbot_autofill.isChecked():
                values = tuple(xyz_mm) + tuple(rpy_deg)
                for spins in (self.hand_pose_spins, self.verify_pose_spins):
                    for spin, value in zip(spins, values):
                        spin.setValue(float(value))
        else:
            self.nexbot_status.setText("读取失败：{}".format(payload))

    def _nexbot_pose_failed(self, message):
        self.nexbot_status.setText("连接失败：{}".format(message))

    def _nexbot_pose_finished(self):
        self.nexbot_pose_worker = None
        self.nexbot_toggle.setText("保存参数并连接读取 TCP")

    def capture_hand_eye(self):
        if self.bundle is None:
            self._show_error("无法采样", "等待 RGB 图像")
            return
        try:
            settings = self._hand_eye_ui_target_settings(require_complete=True)
            if self.config.data["hand_eye"]["calibration_target"] != settings:
                backup, _changed = self._apply_hand_eye_target_settings()
                self.hand_result.setText(
                    "已自动应用界面靶标参数；旧样本归档 {}".format(
                        backup or "（无旧样本）"
                    )
                )
            matrix, distortion, size = self._load_color_intrinsics()
            color = self.bundle.color_bgr
            if (color.shape[1], color.shape[0]) != size:
                raise ValueError("当前 RGB 分辨率与内参不一致")
            sample = HandEyeCalibrator(self.config).add_image_sample(
                color, self._pose_from_spins(self.hand_pose_spins), matrix, distortion
            )
            frames = ROOT / "output" / "hand_eye_frames"
            frames.mkdir(parents=True, exist_ok=True)
            path = frames / "sample_{}_{}.png".format(
                time.strftime("%Y%m%d_%H%M%S"), time.time_ns() % 1000000
            )
            if not cv2.imwrite(str(path), color):
                raise RuntimeError("手眼样本图像保存失败")
            count = self.sample_store.append(sample, path)
            self._refresh_samples()
            self.hand_result.setText(
                "样本 {} · {} · RMS {:.3f} px".format(
                    count, sample.target_label, sample.rms_reprojection_error_px
                )
            )
        except Exception as error:
            self._show_error("手眼样本被拒绝", error)

    def _refresh_samples(self):
        self.hand_table.setRowCount(0)
        try:
            entries = self.sample_store.entries()
        except Exception as error:
            self.hand_result.setText(str(error))
            return
        for index, entry in enumerate(entries, 1):
            row = self.hand_table.rowCount()
            self.hand_table.insertRow(row)
            values = (
                str(index), (
                    "棋盘 {} 角点".format(entry.get("target_corner_count", 0))
                    if entry.get("target_type", APRILTAG_MAP_TARGET) == CHECKERBOARD_TARGET
                    else "Tag {}".format(",".join(map(str, entry.get("visible_tag_ids", []))))
                ),
                "{:.3f}".format(float(entry.get("rms_reprojection_error_px", 0))),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.hand_table.setItem(row, column, item)
        if entries:
            self.hand_result.setText("已加载 {} 个样本".format(len(entries)))

    def reset_hand_eye_samples(self):
        answer = QMessageBox.question(
            self, "清空手眼样本", "当前样本会先归档，然后建立空会话。",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if answer == QMessageBox.Yes:
            backup = self.sample_store.reset()
            self._refresh_samples()
            self.hand_result.setText("样本已清空 · 归档 {}".format(backup or "无"))

    def solve_hand_eye(self):
        try:
            calibrator = HandEyeCalibrator(self.config)
            for entry in self.sample_store.load()["samples"]:
                calibrator.add_stored_sample(entry)
            result = calibrator.solve()
            calibrator.promote(result)
            self.hybrid_localizer = None
            self._restart_live_processing_workers()
            self.hand_result.setText(
                "已启用 · 内点 {}/{} · {:.3f} mm · {:.3f} deg".format(
                    len(result.inlier_indices), result.total_samples,
                    result.translation_rms_mm, result.rotation_rms_deg,
                )
            )
            self._refresh_readiness()
        except Exception as error:
            self._show_error("手眼求解失败", error)

    def change_stage(self, index):
        if index < 0:
            return
        self.current_stage = int(index)
        self.stack.setCurrentIndex(index)
        second_stage = {
            "astra_ros": "RGB-IR / Depth 标定",
            "orbbec_ros": "Gemini 工厂 RGB-D 参数",
            "oak_depthai": "OAK 工厂标定 / EEPROM 导入",
        }.get(self.config.camera["backend"], "RGB-D 参数")
        self.preview_title.setText((
            "RGB 内参", second_stage, "Tag 右下角地图",
            "眼在手上标定", "混合定位验证", "实例分割模型验证",
            "抓取规划验证", "抓取执行验证", "控制器/TCP 测试",
        )[index])
        self.preview_state.setText("实时" if self.camera_connected else "待连接")
        if self.camera_worker is not None:
            self.camera_worker.request_laser(
                self.ir_emitter_toggle.isChecked(),
                anchor=self._stream_anchor(index),
            )
        if self.bundle is not None:
            self._receive_bundle(self.bundle)

    def _refresh_readiness(self):
        camera = self.config.camera
        color_ready = False
        try:
            load_camera_intrinsics(self._color_output_path())
            color_ready = True
        except Exception:
            pass
        depth_ready = False
        depth_label = "RGB-D"
        if camera["backend"] == "astra_ros":
            try:
                load_rgbd_result(
                    self._rgbd_output_path(), camera["max_rgbd_rms_px"],
                )
                depth_ready = True
            except Exception:
                pass
        else:
            depth_label = "OAK EEPROM"
            try:
                info = inspect_oak_calibration(
                    self._oak_factory_output_path(),
                    camera["color_width"], camera["color_height"],
                )
                depth_ready = True
                self.oak_result.setText("已导入 · {}".format(format_oak_summary(info)))
            except Exception:
                pass
        segmentation_ready = self.config.segmentation_valid
        tag_count = len(TagMap(self.config).ids)
        pose_ready = color_ready and tag_count > 0
        self.plan_seg_state.setText("有效" if segmentation_ready else "未验证")
        self.plan_depth_state.setText("有效" if depth_ready else "未就绪")
        self.plan_pose_state.setText("可配置" if pose_ready else "缺少内参/Tag")
        planning_valid = bool(
            self.config.data.get("planning_validation", {}).get("valid", False)
        )
        self.grasp_plan_state.setText("有效" if planning_valid else "规划未验证")
        dry_run = bool(self.config.data["safety"].get("dry_run", True))
        motion_allowed = bool(self.config.data["safety"].get("allow_robot_motion", False))
        self.grasp_dry_run_state.setText("开启" if dry_run else "关闭")
        self.grasp_motion_state.setText("允许" if motion_allowed else "禁止")
        self.readiness.setText(
            "配置 {}\nRGB 内参 {}\n{} {}\nTag {} 个\n手眼 {}\n分割 {}\n规划 {}\n抓取 {}".format(
                self.config.active_camera_profile,
                "有效" if color_ready else "无效", depth_label,
                "有效" if depth_ready else "待导入",
                tag_count, "有效" if self.config.hand_eye_valid else "无效",
                "有效" if segmentation_ready else "待验证",
                "有效" if planning_valid else "后端就绪/待验证",
                "有效" if bool(self.config.data.get("grasp_execution_validation", {}).get("valid", False)) else "后端就绪/待实机验证",
            )
        )

    def open_target(self):
        from PyQt5.QtGui import QDesktopServices

        if TARGET_PDF.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(TARGET_PDF)))
        else:
            self._show_error("标定板不存在", TARGET_PDF)

    def open_output(self):
        from PyQt5.QtGui import QDesktopServices

        output = ROOT / "output"
        output.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))

    def toggle_rviz_visualization(self):
        if self.rviz_process is not None and self.rviz_process.poll() is None:
            self.stop_rviz_visualization()
            return
        # Clean up a worker left behind when the user closed RViz directly.
        self.stop_rviz_visualization()
        try:
            if self.roscore_process is None or self.roscore_process.poll() is not None:
                started_core = ensure_ros_master()
                if started_core is not None:
                    self.roscore_process = started_core
                    self._log("未发现 ROS master，已由比赛 UI 启动 roscore")
            visualization = self.config.data["planning_validation"].get(
                "visualization", {}
            )
            rviz_config = self.config.resolve_path(
                visualization.get(
                    "rviz_config", "competition_planning.rviz"
                )
            )
            worker = RvizVisualizationWorker(self.config.path)
            worker.ready.connect(self._rviz_worker_ready)
            worker.observation_ready.connect(self._rviz_observation_ready)
            worker.failed.connect(self._rviz_worker_failed)
            self.rviz_worker = worker
            worker.start()
            self.rviz_process = launch_rviz(rviz_config)
            self.rviz_button.setText("关闭 RViz 点云验证")
            self.planning_result.setText(
                "RViz 已启动；等待同一帧的分割 Mask、对齐 Depth 和相机定位。"
            )
            self._log("已启动 RViz 分割定位点云验证")
            if self.segmentation_last_result is not None:
                worker.submit(self.segmentation_last_result)
        except Exception as error:
            self.stop_rviz_visualization()
            self._show_error("RViz 启动失败", error)

    def _rviz_worker_ready(self):
        self.statusBar().showMessage("RViz ROS 发布器已就绪", 4000)

    def _rviz_observation_ready(self, result):
        if result.get("valid"):
            text = "RViz 已发布 {} 个物体点云 · 相机位姿来源 {}".format(
                result.get("object_count", 0), result.get("pose_source", "--")
            )
            supported = int(result.get("support_constrained_count", 0))
            if supported:
                text += "\n{} 个物体已按 Tag 台面修正中心与接地包围框；彩色点仍为原始 Depth".format(
                    supported
                )
            if result.get("warning"):
                text += "\n警告：{}".format(result["warning"])
            self.planning_result.setText(text)
        else:
            reason = result.get("reason", "点云定位无效")
            rejected = result.get("rejected") or []
            if rejected:
                reason += "；{}".format("；".join(rejected[:2]))
            self.planning_result.setText("RViz 点云未发布：{}".format(reason))

    def _rviz_worker_failed(self, message):
        self.planning_result.setText("RViz 发布器失败：{}".format(message))
        self.statusBar().showMessage(str(message), 6000)
        self._log("RViz 发布器失败：{}".format(message))
        self.stop_rviz_visualization()

    def stop_rviz_visualization(self, stop_master=False):
        worker = self.rviz_worker
        if worker is not None:
            worker.stop()
            worker.wait(5000)
        self.rviz_worker = None
        stop_process(self.rviz_process)
        self.rviz_process = None
        if stop_master:
            stop_process(self.roscore_process)
            self.roscore_process = None
        if hasattr(self, "rviz_button"):
            self.rviz_button.setText("打开 RViz 点云验证")

    def _show_error(self, title, message):
        self.statusBar().showMessage(str(message), 6000)
        self._log("{}：{}".format(title, str(message).replace("\n", " ")))
        QMessageBox.critical(self, title, str(message))

    def _log(self, message):
        from datetime import datetime

        self.log_output.appendPlainText("{}  {}".format(datetime.now().strftime("%H:%M:%S"), message))

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6f7; color: #20282d; font-size: 13px; }
            QFrame#header { background: #20272b; border: 0; }
            QLabel#appTitle { color: white; background: transparent; font-size: 18px; font-weight: 650; }
            QLabel#appSubtitle { color: #aeb8bd; background: transparent; font-size: 11px; }
            QLabel#connectionBadge { color: #d7dde0; background: #323b40; border: 1px solid #465158; padding: 6px 11px; border-radius: 4px; }
            QFrame#sidebar { background: #edf0f1; border-right: 1px solid #d6dcdf; }
            QFrame#controlArea { background: white; border-left: 1px solid #dce1e3; }
            QLabel#sectionLabel { color: #647078; font-size: 11px; font-weight: 650; }
            QLabel#sourceBanner { background: white; border: 1px solid #cfd6d9; border-radius: 4px; padding: 9px; }
            QLabel#sourceMeta { color: #5f6d74; font-size: 11px; }
            QLabel#projectorState { background: #f6f0dd; color: #725c19; border: 1px solid #dfd09e; padding: 7px; border-radius: 4px; }
            QLabel#viewTitle { font-size: 17px; font-weight: 650; }
            QLabel#viewState { color: #0f766e; font-weight: 650; }
            QLabel#panelTitle { font-size: 18px; font-weight: 650; }
            QLabel#panelSubtitle { color: #6b767c; }
            QLabel#metricValue, QLabel#strongValue { font-weight: 650; color: #243139; }
            QLabel#sidebarNote { color: #526169; background: #e1e7e8; padding: 10px; border-radius: 4px; }
            QLabel#resultBanner { background: #eef5f3; color: #285c55; border: 1px solid #c8ddd8; padding: 10px; border-radius: 5px; }
            QListWidget#stageList { background: transparent; border: 0; outline: 0; }
            QListWidget#stageList::item { padding: 8px 10px; border-radius: 5px; color: #435057; }
            QListWidget#stageList::item:selected { background: white; color: #0b645c; font-weight: 650; border: 1px solid #d3ddda; }
            QPushButton#primaryButton { background: #176d64; color: white; border: 1px solid #176d64; border-radius: 5px; padding: 7px 14px; font-weight: 650; }
            QPushButton#primaryButton:hover { background: #115c54; }
            QPushButton#secondaryButton { background: white; border: 1px solid #cbd3d6; border-radius: 5px; padding: 7px 14px; }
            QToolButton { background: white; border: 1px solid #cbd3d6; border-radius: 5px; }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { background: white; border: 1px solid #cbd3d6; border-radius: 4px; padding: 5px; min-height: 23px; }
            QGroupBox { border: 1px solid #d9dfe1; border-radius: 6px; margin-top: 12px; padding: 12px 8px 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; color: #58656c; }
            QProgressBar { background: #e5e9eb; border: 0; border-radius: 3px; height: 8px; }
            QProgressBar::chunk { background: #238176; border-radius: 3px; }
            QTableWidget { background: white; border: 1px solid #d7dde0; gridline-color: #e5e9eb; }
            QHeaderView::section { background: #eef1f2; border: 0; border-bottom: 1px solid #d7dde0; padding: 5px; font-weight: 650; }
            QFrame#previewPanel { background: #14191d; }
            QLabel#previewCaption { background: #20272b; color: #d7dee1; padding: 7px 10px; font-weight: 650; }
            QFrame#metricBar { background: white; border: 1px solid #dce1e3; border-radius: 6px; }
            QPlainTextEdit { background: #20272b; color: #d7dee1; border: 0; border-radius: 5px; font-family: monospace; font-size: 11px; padding: 7px; }
            QScrollArea { background: white; border: 0; }
            QScrollArea > QWidget > QWidget { background: white; }
            QScrollArea#sidebarScroll, QScrollArea#sidebarScroll > QWidget > QWidget { background: #edf0f1; }
            QStatusBar { background: #eef1f2; color: #526068; border-top: 1px solid #d7dde0; }
        """)

    def closeEvent(self, event):
        self.stop_rviz_visualization(stop_master=False)
        self._stop_live_processing_workers()
        if self.controller_worker is not None:
            self.controller_worker.stop()
            self.controller_worker.wait(3000)
            self.controller_worker = None
        if self.segmentation_worker is not None:
            self.segmentation_worker.stop()
            self.segmentation_worker.wait(5000)
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker.wait(5000)
        stop_process(self.roscore_process)
        self.roscore_process = None
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--stage", choices=(
            "rgb", "rgbd", "tags", "hand-eye", "localization",
            "segmentation", "planning", "grasp", "controller",
        ),
        default="rgb",
    )
    parser.add_argument("--auto-connect", action="store_true")
    parser.add_argument("--screenshot")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("RGB-D Competition Calibration Workbench")
    indices = {
        "rgb": 0, "rgbd": 1, "tags": 2, "hand-eye": 3,
        "localization": 4, "segmentation": 5, "planning": 6, "grasp": 7,
        "controller": 8,
    }
    window = CompetitionCalibrationWindow(args.config, indices[args.stage], args.auto_connect)
    window.show()
    if args.screenshot:
        destination = Path(args.screenshot).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        def save_and_quit():
            window.grab().save(str(destination))
            app.quit()

        QTimer.singleShot(1000 if args.auto_connect else 450, save_and_quit)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
