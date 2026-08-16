#!/usr/bin/env python3
"""Live AprilTag + RGB-D + YOLO bottle localization and RViz publishing."""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import yaml

from tool.camera_calibration.rviz_visualization import RosPoseVisualizer
from tool.object_model_builder.camera_source import (
    AstraRosSource,
    OakDProSource,
    native_ros_environment,
)
from tool.object_model_builder.rgbd_geometry import (
    DepthToColorAligner,
    load_runtime_calibration,
    rectified_intrinsics,
    rectify_aligned_depth_image,
    rectify_color_image,
)
from tool.object_model_builder.tag_pose_provider import TagPoseProvider
from tool.object_model_builder.yolo_segmenter import MaskResult, YoloMaskProvider

from .estimator import BottleEstimate, BottlePositionEstimator, BottlePositionSettings
from .ros_visualization import BottleRosVisualizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "bottle_localization.yaml"


def _load_yaml(path: Path) -> dict:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping: {}".format(path))
    return data


def load_config(path: str) -> tuple:
    source = Path(path).expanduser().resolve()
    config = _load_yaml(source)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported bottle localization config schema")
    base_path = Path(config["base_config"]).expanduser()
    if not base_path.is_absolute():
        base_path = source.parent / base_path
    base = _load_yaml(base_path.resolve())
    if int(base.get("schema_version", 0)) != 1:
        raise ValueError("unsupported object model builder config schema")
    return config, base, source


def _depth_colormap(depth_m: np.ndarray, maximum_depth_m: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros_like(depth, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth, 0.0, float(maximum_depth_m))
        normalized[valid] = np.rint(
            (1.0 - clipped[valid] / float(maximum_depth_m)) * 255.0
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _draw_estimate(
    image: np.ndarray,
    estimate: BottleEstimate,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    output = np.asarray(image).copy()
    if not estimate.valid or estimate.center_camera_m is None:
        return output
    point = np.asarray(estimate.center_camera_m, dtype=np.float64).reshape(3)
    if point[2] <= 0.0:
        return output
    pixel = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3) @ point
    u = int(round(pixel[0] / pixel[2]))
    v = int(round(pixel[1] / pixel[2]))
    if 0 <= u < output.shape[1] and 0 <= v < output.shape[0]:
        cv2.drawMarker(
            output,
            (u, v),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=3,
        )
        base = np.asarray(estimate.base_center_workspace_m) * 1000.0
        text = "base XYZ mm: {:.1f}, {:.1f}, {:.1f}".format(*base)
        cv2.rectangle(output, (8, output.shape[0] - 45), (520, output.shape[0] - 8), (20, 24, 28), -1)
        cv2.putText(
            output,
            text,
            (18, output.shape[0] - 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


class BottleLocalizationNode:
    def __init__(self, arguments):
        self.arguments = arguments
        self.config, self.base_config, self.config_path = load_config(arguments.config)
        self.paths = self.base_config["paths"]
        self.camera_config = dict(self.base_config["camera"])
        if arguments.no_driver:
            self.camera_config["start_ros_driver"] = False
        self.calibration = load_runtime_calibration(self.paths["runtime_calibration"])
        self.calibration.require_valid()
        self.depth_aligner = DepthToColorAligner(self.calibration)

        tag_config = self.base_config.get("tag_pose", {})
        self.tag_provider = TagPoseProvider(
            self.paths["tag_layout"],
            minimum_tags=tag_config.get("minimum_tags", 1),
            maximum_rms_px=tag_config.get("maximum_rms_px", 2.5),
        )
        segmentation = self.base_config.get("segmentation", {})
        weights = arguments.weights or self.paths.get("yolo_weights", "")
        if not weights:
            raise ValueError(
                "YOLO weights are not configured; pass --weights /absolute/path/to/best.pt"
            )
        classes = arguments.target_class or segmentation.get("target_classes", ["bottle"])
        self.yolo_provider = YoloMaskProvider(
            weights,
            target_classes=classes,
            confidence_threshold=segmentation.get("confidence_threshold", 0.5),
            device=arguments.device or segmentation.get("device", "0"),
        )

        settings_mapping = dict(self.config.get("localization", {}))
        if arguments.bottle_height_mm is not None:
            settings_mapping["nominal_bottle_height_m"] = arguments.bottle_height_mm / 1000.0
        if arguments.bottle_diameter_mm is not None:
            settings_mapping["nominal_bottle_diameter_m"] = arguments.bottle_diameter_mm / 1000.0
        if settings_mapping.get("workspace_plane_z_m") is None:
            settings_mapping["workspace_plane_z_m"] = (
                float(self.tag_provider.layout.get("workspace_plane_z_mm", 0.0)) / 1000.0
            )
        self.settings = BottlePositionSettings(**settings_mapping)
        self.estimator = BottlePositionEstimator(self.settings)
        self.source = self._build_source()
        self.camera_visualizer = None
        self.bottle_visualizer = None
        self.rviz_process = None
        self._last_depth_timestamp = None

    def _build_source(self):
        backend = self.camera_config.get("backend", "astra_ros")
        if backend == "oak_depthai":
            oak = self.camera_config.get("oak", {})
            return OakDProSource(
                color_width=oak.get("color_width", 1280),
                color_height=oak.get("color_height", 720),
                fps=oak.get("fps", 30),
            )
        if backend != "astra_ros":
            raise ValueError("unsupported camera backend: {}".format(backend))
        ros_driver = self.camera_config.get("ros_driver", {})
        log_root = Path(self.paths.get("capture_root", PROJECT_ROOT / "Log"))
        return AstraRosSource(
            color_device=self.camera_config["color_device"],
            color_width=self.camera_config["color_width"],
            color_height=self.camera_config["color_height"],
            color_fps=self.camera_config["color_fps"],
            color_fourcc=self.camera_config["color_fourcc"],
            depth_topic=self.camera_config["depth_topic"],
            depth_info_topic=self.camera_config["depth_info_topic"],
            ir_topic=self.camera_config["ir_topic"],
            start_ros_driver=self.camera_config.get("start_ros_driver", True),
            driver_log_path=str(log_root / "bottle_localization_driver.log"),
            driver_package=ros_driver.get("package", "astra_camera"),
            driver_launch_file=ros_driver.get("launch_file", "astra.launch"),
            driver_arguments=ros_driver.get("arguments"),
            driver_startup_timeout_s=ros_driver.get("startup_timeout_s", 2.0),
            laser_service=self.camera_config.get("laser_service", "/camera/set_laser"),
            ros_node_name="bottle_localization",
        )

    def run(self) -> None:
        try:
            self.source.start()
            import rospy

            if not rospy.core.is_initialized():
                rospy.init_node("bottle_localization", anonymous=True, disable_signals=True)
            workspace_frame = str(self.tag_provider.layout["workspace_frame"])
            camera_frame = str(self.tag_provider.layout["camera_frame"])
            visualization = self.config.get("visualization", {})
            self.camera_visualizer = RosPoseVisualizer(workspace_frame, camera_frame)
            self.camera_visualizer.publish_scene(self.tag_provider.layout)
            self.bottle_visualizer = BottleRosVisualizer(
                workspace_frame=workspace_frame,
                bottle_frame=visualization.get("bottle_frame", "bottle_estimated"),
                topic_prefix=visualization.get("topic_prefix", "/bottle_localization"),
            )
            if self.arguments.rviz:
                self._launch_rviz()
            rospy.loginfo("Bottle localization is ready; waiting for synchronized RGB-D frames")
            rate = rospy.Rate(float(self.config.get("runtime", {}).get("rate_hz", 7.0)))
            while not rospy.is_shutdown():
                self._process_latest()
                rate.sleep()
        finally:
            self.close()

    def _process_latest(self) -> None:
        bundle = self.source.latest(anchor="depth")
        if bundle is None or bundle.depth_m is None or bundle.depth_timestamp_s is None:
            return
        if bundle.depth_timestamp_s == self._last_depth_timestamp:
            return
        self._last_depth_timestamp = bundle.depth_timestamp_s
        maximum_sync = float(self.camera_config.get("maximum_sync_delta_s", 0.12))
        if bundle.sync_delta_s is None or bundle.sync_delta_s > maximum_sync:
            self._publish_invalid(
                "RGB/depth time difference exceeds {:.0f} ms".format(maximum_sync * 1000.0),
                bundle.color_bgr,
                diagnostics={"sync_delta_ms": None if bundle.sync_delta_s is None else bundle.sync_delta_s * 1000.0},
            )
            return

        active_intrinsics = bundle.color_intrinsics or self.calibration.color
        rectified_color = (
            bundle.color_bgr.copy()
            if bundle.color_is_rectified
            else rectify_color_image(bundle.color_bgr, active_intrinsics)
        )
        intrinsics = rectified_intrinsics(active_intrinsics)
        if bundle.depth_aligned_to_color:
            if bundle.depth_m.shape != rectified_color.shape[:2]:
                self._publish_invalid(
                    "device-aligned depth dimensions do not match RGB",
                    rectified_color,
                )
                return
            aligned_depth = (
                bundle.depth_m.copy()
                if bundle.color_is_rectified
                else rectify_aligned_depth_image(bundle.depth_m, active_intrinsics)
            )
        else:
            aligned_depth = self.depth_aligner.align(
                bundle.depth_m,
                minimum_depth_m=self.settings.minimum_depth_m,
                maximum_depth_m=self.settings.maximum_depth_m,
            )

        tag_estimate, detections = self.tag_provider.estimate(
            rectified_color, intrinsics.matrix, intrinsics.distortion
        )
        annotated = self.tag_provider.draw_status(
            rectified_color, detections, tag_estimate
        )
        mask_result = self.yolo_provider.predict(rectified_color)
        annotated = self.yolo_provider.overlay(annotated, mask_result)
        depth_preview = _depth_colormap(aligned_depth, self.settings.maximum_depth_m)
        if not tag_estimate.valid:
            self._publish_invalid(
                "AprilTag camera pose is invalid: {}".format(tag_estimate.reason),
                annotated,
                mask_result=mask_result,
                depth_preview=depth_preview,
                diagnostics=self._diagnostics(bundle, tag_estimate, mask_result),
            )
            return
        self.camera_visualizer.publish_pose(tag_estimate.workspace_from_camera)
        if not mask_result.valid or mask_result.mask is None:
            self._publish_invalid(
                "YOLO bottle mask is invalid: {}".format(mask_result.reason),
                annotated,
                mask_result=mask_result,
                depth_preview=depth_preview,
                diagnostics=self._diagnostics(bundle, tag_estimate, mask_result),
            )
            return

        estimate = self.estimator.estimate(
            aligned_depth,
            mask_result.mask,
            intrinsics,
            tag_estimate.workspace_from_camera,
        )
        annotated = _draw_estimate(annotated, estimate, intrinsics.matrix)
        if mask_result.mask is not None:
            outside = ~np.asarray(mask_result.mask).astype(bool)
            depth_preview[outside] = (depth_preview[outside] * 0.18).astype(np.uint8)
        self.bottle_visualizer.publish(
            estimate,
            annotated_bgr=annotated,
            mask=mask_result.mask,
            depth_preview_bgr=depth_preview,
            diagnostics=self._diagnostics(bundle, tag_estimate, mask_result),
        )

    def _publish_invalid(
        self,
        reason: str,
        annotated: np.ndarray,
        mask_result: Optional[MaskResult] = None,
        depth_preview: Optional[np.ndarray] = None,
        diagnostics: Optional[dict] = None,
    ) -> None:
        if self.bottle_visualizer is None:
            return
        result = mask_result or MaskResult(False, reason="YOLO not evaluated")
        estimate = BottleEstimate(False, reason=str(reason))
        self.bottle_visualizer.publish(
            estimate,
            annotated_bgr=annotated,
            mask=result.mask,
            depth_preview_bgr=depth_preview,
            diagnostics=diagnostics,
        )

    def _diagnostics(self, bundle, tag_estimate, mask_result) -> dict:
        return {
            "sync_delta_ms": None if bundle.sync_delta_s is None else float(bundle.sync_delta_s * 1000.0),
            "visible_tag_ids": list(tag_estimate.visible_tag_ids),
            "tag_rms_px": tag_estimate.rms_reprojection_error_px,
            "rgbd_stereo_rms_px": self.calibration.rms_reprojection_error_px,
            "yolo_class": mask_result.class_name,
            "yolo_confidence": float(mask_result.confidence),
        }

    def _launch_rviz(self) -> None:
        rviz_config = Path(
            self.config.get("visualization", {}).get(
                "rviz_config",
                Path(__file__).resolve().parent / "config" / "bottle_localization.rviz",
            )
        ).expanduser()
        if not rviz_config.is_absolute():
            rviz_config = self.config_path.parent / rviz_config
        if shutil.which("rviz") is None:
            raise RuntimeError("rviz is not installed or is not on PATH")
        self.rviz_process = subprocess.Popen(
            ["rviz", "-d", str(rviz_config.resolve())],
            env=native_ros_environment(),
        )

    def close(self) -> None:
        if self.rviz_process is not None and self.rviz_process.poll() is None:
            self.rviz_process.terminate()
            try:
                self.rviz_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.rviz_process.kill()
        self.rviz_process = None
        if self.source is not None:
            self.source.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locate a bottle from fixed AprilTags, calibrated RGB-D, and YOLO segmentation"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weights", help="Ultralytics segmentation .pt/.pth weights")
    parser.add_argument(
        "--target-class",
        action="append",
        help="YOLO class to accept; repeat for more than one class",
    )
    parser.add_argument("--device", help="Ultralytics device, for example 0 or cpu")
    parser.add_argument("--bottle-height-mm", type=float, help="measured bottle height")
    parser.add_argument("--bottle-diameter-mm", type=float, help="measured bottle diameter")
    parser.add_argument("--no-driver", action="store_true", help="reuse an already running camera ROS driver")
    parser.add_argument("--rviz", action="store_true", help="launch RViz with the supplied view")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        BottleLocalizationNode(arguments).run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print("瓶子定位启动失败：{}".format(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
