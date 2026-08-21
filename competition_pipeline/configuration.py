"""Configuration loading, validation, and atomic updates."""

import copy
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .geometry import as_transform, transform_from_xyz_rpy_mm


SCHEMA_VERSION = 1
TAG_CONVENTION_ID = "tag_bottom_right_x_to_top_right_y_to_bottom_left_v3"
DEFAULT_ASTRA_DEPTH_MODES = {
    "smooth_640x480": {
        "label": "640 x 480 @ 30（流畅，推荐）",
        "depth_width": 640,
        "depth_height": 480,
        "depth_fps": 30,
        "ir_width": 640,
        "ir_height": 480,
        "ir_fps": 30,
        "depth_preview_fps": 8,
        "point_cloud_fps": 5,
    },
    "detail_1280x1024": {
        "label": "1280 x 1024 @ 7（高细节）",
        "depth_width": 1280,
        "depth_height": 1024,
        "depth_fps": 7,
        "ir_width": 1280,
        "ir_height": 1024,
        "ir_fps": 30,
        "depth_preview_fps": 2,
        "point_cloud_fps": 2,
    },
}


def _read_yaml(path):
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping: {}".format(resolved))
    return resolved, data


def atomic_write_yaml(path, data):
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(destination.parent),
        prefix=".{}-".format(destination.name), suffix=".tmp", delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


class CompetitionConfig:
    def __init__(self, path):
        self.path, self.data = _read_yaml(path)
        self._add_legacy_depth_modes()
        self._add_legacy_hand_eye_target()
        self.validate()

    def _add_legacy_depth_modes(self):
        """Keep configurations written by the pre-resolution UI loadable."""
        profiles = self.data.get("camera", {}).get("profiles", {})
        for camera in profiles.values():
            if camera.get("backend") != "astra_ros":
                continue
            if not camera.get("depth_modes"):
                camera["depth_modes"] = copy.deepcopy(DEFAULT_ASTRA_DEPTH_MODES)
            else:
                for mode_name, mode in camera["depth_modes"].items():
                    defaults = DEFAULT_ASTRA_DEPTH_MODES.get(mode_name, {})
                    for field in ("depth_preview_fps", "point_cloud_fps"):
                        if field in defaults:
                            mode.setdefault(field, defaults[field])
            if camera.get("depth_mode") in camera["depth_modes"]:
                continue
            arguments = camera.get("ros_driver", {}).get("arguments", {})
            stream_fields = (
                "depth_width", "depth_height", "depth_fps",
                "ir_width", "ir_height", "ir_fps",
            )
            selected = None
            if all(field in arguments for field in stream_fields):
                for mode_name, mode in camera["depth_modes"].items():
                    if all(
                        int(arguments[field]) == int(mode[field])
                        for field in stream_fields
                    ):
                        selected = mode_name
                        break
            camera["depth_mode"] = selected or next(iter(camera["depth_modes"]))
            camera.setdefault("alignment_splat_radius_pixels", 0)

    def _add_legacy_hand_eye_target(self):
        """Keep old Tag-only competition YAML files valid."""
        hand_eye = self.data.setdefault("hand_eye", {})
        target = hand_eye.setdefault("calibration_target", {})
        target.setdefault("type", "apriltag_map")
        checkerboard = target.setdefault("checkerboard", {})
        checkerboard.setdefault("board_width_mm", 60.0)
        checkerboard.setdefault("board_height_mm", 45.0)
        checkerboard.setdefault("square_size_mm", 5.0)
        # Never guess the printed grid from the outer-board dimensions. Legacy
        # files did not contain explicit cell counts, so leave them unconfigured
        # until the real board is counted at the venue.
        checkerboard.setdefault(
            "configured",
            checkerboard.get("squares_x") is not None
            and checkerboard.get("squares_y") is not None,
        )
        checkerboard.setdefault("squares_x", None)
        checkerboard.setdefault("squares_y", None)
        checkerboard.setdefault("inner_corners", None)
        checkerboard.setdefault("maximum_rms_px", 1.5)
        checkerboard.setdefault("prefer_find_chessboard_sb", True)
        checkerboard.setdefault("minimum_tcp_translation_span_mm", 30.0)
        checkerboard.setdefault("minimum_tcp_rotation_span_deg", 15.0)

    def validate(self):
        if int(self.data.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("unsupported competition config schema")
        tag_map = self.data.get("tag_map", {})
        convention = tag_map.get("coordinate_convention", {})
        if convention.get("id") != TAG_CONVENTION_ID:
            raise ValueError(
                "Tag map must use {}; legacy tool maps cannot be mixed".format(TAG_CONVENTION_ID)
            )
        if float(tag_map.get("tag_size_mm", 0.0)) <= 0.0:
            raise ValueError("tag_size_mm must be positive")
        localization = self.data.get("localization", {})
        if float(localization.get("processing_fps", 12.0)) <= 0.0:
            raise ValueError("localization.processing_fps must be positive")
        if float(localization.get("cached_pose_max_age_s", 0.25)) <= 0.0:
            raise ValueError("localization.cached_pose_max_age_s must be positive")
        default_rpy = tag_map.get("default_base_from_tag_rpy_deg", [])
        if len(default_rpy) != 3:
            raise ValueError("default_base_from_tag_rpy_deg must contain three values")
        seen = set()
        for raw_id, entry in tag_map.get("tags", {}).items():
            tag_id = int(raw_id)
            if tag_id in seen:
                raise ValueError("duplicate Tag ID {}".format(tag_id))
            seen.add(tag_id)
            if len(entry.get("bottom_right_xyz_mm", [])) != 3:
                raise ValueError("Tag {} bottom_right_xyz_mm must contain three values".format(tag_id))
            rpy = entry.get("base_from_tag_rpy_deg", default_rpy)
            if len(rpy) != 3:
                raise ValueError("Tag {} RPY must contain three values".format(tag_id))
            transform_from_xyz_rpy_mm(entry["bottom_right_xyz_mm"], rpy)
        localization = self.data.get("localization", {})
        if int(localization.get("minimum_visible_tags", 1)) < 1:
            raise ValueError("minimum_visible_tags must be at least one")
        for name in ("single_tag_after_multi_hold_s", "invalid_visual_hold_s"):
            if float(localization.get(name, 0.0)) < 0.0:
                raise ValueError("localization.{} cannot be negative".format(name))
        segmentation = self.data.get("segmentation_validation", {})
        for name in (
            "iou_threshold", "minimum_confidence",
            "minimum_mask_area_ratio", "maximum_mask_area_ratio",
            "duplicate_mask_iou_threshold",
            "duplicate_mask_containment_threshold",
            "duplicate_confidence_tie_margin",
        ):
            value = float(segmentation.get(name, -1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError("segmentation_validation.{} must be within 0..1".format(name))
        confidence = float(segmentation.get("confidence_threshold", -1.0))
        if not 0.05 <= confidence <= 1.0:
            raise ValueError(
                "segmentation_validation.confidence_threshold must be within 0.05..1"
            )
        tie_margin = float(
            segmentation.get("duplicate_confidence_tie_margin", -1.0)
        )
        if not 0.0 <= tie_margin <= 0.25:
            raise ValueError(
                "segmentation_validation.duplicate_confidence_tie_margin "
                "must be within 0..0.25"
            )
        maximum_detections = int(segmentation.get("maximum_detections", 0))
        if not 1 <= maximum_detections <= 300:
            raise ValueError(
                "segmentation_validation.maximum_detections must be within 1..300"
            )
        if float(segmentation["minimum_mask_area_ratio"]) >= float(
            segmentation["maximum_mask_area_ratio"]
        ):
            raise ValueError("segmentation mask-area minimum must be below maximum")
        if int(segmentation.get("image_size", 0)) <= 0:
            raise ValueError("segmentation_validation.image_size must be positive")
        if float(segmentation.get("preview_interval_s", 0.0)) <= 0.0:
            raise ValueError("segmentation_validation.preview_interval_s must be positive")
        if int(segmentation.get("required_consecutive_valid_frames", 0)) < 1:
            raise ValueError("segmentation validation requires at least one valid frame")
        if not isinstance(segmentation.get("target_classes", []), list):
            raise ValueError("segmentation_validation.target_classes must be a list")
        center_ratio = float(
            segmentation.get("duplicate_center_distance_ratio", -1.0)
        )
        if not 0.0 <= center_ratio <= 2.0:
            raise ValueError(
                "segmentation_validation.duplicate_center_distance_ratio "
                "must be within 0..2"
            )
        planning = self.data.get("planning_validation", {})
        grasp = self.data.get("grasp_execution_validation", {})
        for name, entry in (("planning_validation", planning), ("grasp_execution_validation", grasp)):
            if entry.get("state") not in ("not_implemented", "ready", "validated"):
                raise ValueError("{}.state is unsupported".format(name))
            if bool(entry.get("valid", False)) and entry.get("state") != "validated":
                raise ValueError("{} cannot be valid unless state is validated".format(name))
        if bool(grasp.get("valid", False)) and not bool(planning.get("valid", False)):
            raise ValueError("grasp execution cannot be valid before planning validation")
        grasp_planning = self.data.get("grasp_planning", {})
        backend = str(grasp_planning.get("backend", "deterministic_top_down")).strip()
        if backend != "deterministic_top_down":
            raise ValueError(
                "grasp_planning.backend must remain deterministic_top_down; "
                "configure AnyGrasp under grasp_planning.fallback"
            )
        fallback = grasp_planning.get("fallback", {})
        if not isinstance(fallback, dict):
            raise ValueError("grasp_planning.fallback must be a mapping")
        if bool(fallback.get("enabled", False)):
            fallback_backend = str(fallback.get("backend", "anygrasp")).strip().lower()
            if fallback_backend != "anygrasp":
                raise ValueError(
                    "unsupported grasp_planning.fallback.backend: {}".format(
                        fallback_backend
                    )
                )
            if not str(fallback.get("sdk_grasp_dir", "")).strip():
                raise ValueError("AnyGrasp fallback sdk_grasp_dir is required when enabled")
            if not str(fallback.get("checkpoint_path", "")).strip():
                raise ValueError("AnyGrasp fallback checkpoint_path is required when enabled")
            if float(fallback.get("minimum_score", -1.0)) < 0.0:
                raise ValueError("AnyGrasp fallback minimum_score cannot be negative")
            if not 0.0 < float(fallback.get("maximum_gripper_width_m", 0.0)) <= 0.1:
                raise ValueError(
                    "AnyGrasp fallback maximum_gripper_width_m must be within (0, 0.1]"
                )
            if int(fallback.get("top_k", 0)) < 1:
                raise ValueError("AnyGrasp fallback top_k must be positive")
        as_transform(
            grasp_planning.get("tcp_from_grasp", {}).get("matrix"),
            "grasp_planning.tcp_from_grasp",
        )
        minimum_width = float(grasp_planning.get("minimum_grasp_width_mm", -1.0))
        maximum_width = float(grasp_planning.get("maximum_grasp_width_mm", -1.0))
        if minimum_width < 0.0 or maximum_width <= minimum_width:
            raise ValueError("configured grasp width range is invalid")
        for name in (
            "pregrasp_clearance_mm", "lift_distance_mm", "place_clearance_mm"
        ):
            if float(grasp_planning.get(name, 0.0)) <= 0.0:
                raise ValueError("grasp_planning.{} must be positive".format(name))
        execution = self.data.get("grasp_execution", {})
        for name in (
            "speed_scale", "maximum_segment_mm", "maximum_segment_rotation_deg",
            "minimum_verified_lift_mm", "placement_tolerance_mm",
            "maximum_object_pose_age_s",
        ):
            if float(execution.get(name, 0.0)) <= 0.0:
                raise ValueError("grasp_execution.{} must be positive".format(name))
        visualization = planning.get("visualization", {})
        if visualization:
            minimum_depth = float(visualization.get("minimum_depth_m", 0.0))
            maximum_depth = float(visualization.get("maximum_depth_m", 0.0))
            if minimum_depth <= 0.0 or maximum_depth <= minimum_depth:
                raise ValueError("planning visualization depth range is invalid")
            coverage = float(visualization.get("minimum_depth_coverage", -1.0))
            if not 0.0 <= coverage <= 1.0:
                raise ValueError(
                    "planning visualization minimum_depth_coverage must be within 0..1"
                )
            for name in ("minimum_valid_points", "maximum_points_per_instance"):
                if int(visualization.get(name, 0)) < 1:
                    raise ValueError(
                        "planning visualization {} must be positive".format(name)
                    )
            if int(visualization.get("mask_erosion_pixels", -1)) < 0:
                raise ValueError(
                    "planning visualization mask_erosion_pixels cannot be negative"
                )
        hand_eye = self.data.get("hand_eye", {})
        as_transform(
            hand_eye.get("tcp_from_color_camera", {}).get("matrix"),
            "tcp_from_color_camera",
        )
        target = hand_eye.get("calibration_target", {}) or {}
        target_type = str(target.get("type", "apriltag_map"))
        if target_type not in ("apriltag_map", "checkerboard"):
            raise ValueError(
                "hand_eye.calibration_target.type must be apriltag_map or checkerboard"
            )
        checkerboard = target.get("checkerboard", {}) or {}
        for name in (
            "board_width_mm", "board_height_mm", "square_size_mm", "maximum_rms_px",
            "minimum_tcp_translation_span_mm", "minimum_tcp_rotation_span_deg",
        ):
            if float(checkerboard.get(name, 0.0)) <= 0.0:
                raise ValueError(
                    "hand_eye.calibration_target.checkerboard.{} must be positive".format(name)
                )
        if not bool(checkerboard.get("configured", False)):
            squares = None
        else:
            squares = []
            for name, limit_name in (
                ("squares_x", "board_width_mm"),
                ("squares_y", "board_height_mm"),
            ):
                try:
                    count = int(checkerboard[name])
                except (TypeError, ValueError):
                    raise ValueError("checkerboard.{} must be an integer".format(name))
                if count < 3:
                    raise ValueError("checkerboard needs at least 3 squares on every side")
                if count * float(checkerboard["square_size_mm"]) > float(checkerboard[limit_name]) + 1e-9:
                    raise ValueError(
                        "checkerboard printed grid is larger than physical board {}".format(limit_name)
                    )
                squares.append(count)
            expected_corners = [squares[0] - 1, squares[1] - 1]
            inner_corners = checkerboard.get("inner_corners")
            if inner_corners is None or [int(value) for value in inner_corners] != expected_corners:
                raise ValueError(
                    "checkerboard.inner_corners must be {} (inner corners, not square count)".format(
                        expected_corners
                    )
                )
        camera_root = self.data.get("camera", {})
        profiles = camera_root.get("profiles", {})
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("camera.profiles must contain at least one camera profile")
        active_profile = str(camera_root.get("active_profile", "")).strip()
        if active_profile not in profiles:
            raise ValueError("camera.active_profile does not name a configured profile")
        for profile_name, camera in profiles.items():
            backend = str(camera.get("backend", "")).strip()
            if backend not in ("astra_ros", "orbbec_ros", "oak_depthai"):
                raise ValueError(
                    "camera profile {} has unsupported backend {}".format(
                        profile_name, backend
                    )
                )
            for name in ("color_intrinsics_file", "hand_eye_samples_file"):
                if not str(camera.get(name, "")).strip():
                    raise ValueError(
                        "camera.profiles.{}.{} is required".format(profile_name, name)
                    )
            if backend == "astra_ros" and not str(
                camera.get("rgbd_calibration_file", "")
            ).strip():
                raise ValueError(
                    "camera.profiles.{}.rgbd_calibration_file is required".format(
                        profile_name
                    )
                )
            if backend == "astra_ros":
                modes = camera.get("depth_modes", {})
                active_mode = str(camera.get("depth_mode", "")).strip()
                if not isinstance(modes, dict) or not modes:
                    raise ValueError(
                        "camera profile {} must define depth_modes".format(
                            profile_name
                        )
                    )
                if active_mode not in modes:
                    raise ValueError(
                        "camera profile {} depth_mode is not configured".format(
                            profile_name
                        )
                    )
                for mode_name, mode in modes.items():
                    for field in (
                        "depth_width", "depth_height", "depth_fps",
                        "ir_width", "ir_height", "ir_fps",
                    ):
                        if int(mode.get(field, 0)) <= 0:
                            raise ValueError(
                                "camera profile {} depth mode {} field {} must "
                                "be positive".format(profile_name, mode_name, field)
                            )
                    for field in ("depth_preview_fps", "point_cloud_fps"):
                        if field in mode and float(mode[field]) <= 0.0:
                            raise ValueError(
                                "camera profile {} depth mode {} field {} must "
                                "be positive".format(profile_name, mode_name, field)
                            )
                if int(camera.get("alignment_splat_radius_pixels", 0)) < 0:
                    raise ValueError(
                        "camera profile {} alignment splat radius cannot be "
                        "negative".format(profile_name)
                    )
            if backend == "orbbec_ros":
                for field in (
                    "color_topic", "color_info_topic", "depth_topic",
                    "depth_info_topic", "ir_topic",
                ):
                    if not str(camera.get(field, "")).strip():
                        raise ValueError(
                            "camera profile {} {} is required".format(
                                profile_name, field
                            )
                        )
                if int(camera.get("color_width", 0)) <= 0 or int(
                    camera.get("color_height", 0)
                ) <= 0:
                    raise ValueError(
                        "camera profile {} Orbbec RGB size must be positive".format(
                            profile_name
                        )
                    )
                if float(camera.get("color_fps", 0.0)) <= 0.0:
                    raise ValueError(
                        "camera profile {} Orbbec RGB FPS must be positive".format(
                            profile_name
                        )
                    )
                driver = camera.get("ros_driver", {}) or {}
                for field in ("package", "launch_file"):
                    if not str(driver.get(field, "")).strip():
                        raise ValueError(
                            "camera profile {} ros_driver.{} is required".format(
                                profile_name, field
                            )
                        )
            if float(camera.get("depth_preview_fps", 10.0)) <= 0.0:
                raise ValueError(
                    "camera profile {} depth_preview_fps must be positive".format(
                        profile_name
                    )
                )
            if backend == "oak_depthai" and not str(
                camera.get("factory_calibration_file", "")
            ).strip():
                raise ValueError(
                    "camera.profiles.{}.factory_calibration_file is required".format(
                        profile_name
                    )
                )
            if backend == "oak_depthai":
                size = (
                    int(camera.get("color_width", 0)),
                    int(camera.get("color_height", 0)),
                )
                if size != (1920, 1080):
                    raise ValueError(
                        "camera profile {} must use the validated OAK RGB "
                        "1920x1080 competition profile".format(profile_name)
                    )
                if not 1 <= int(camera.get("color_fps", 0)) <= 30:
                    raise ValueError(
                        "camera profile {} OAK FPS must be within 1..30".format(
                            profile_name
                        )
                    )
                if str(camera.get("mono_resolution", "")).lower() != "800p":
                    raise ValueError(
                        "camera profile {} must use OAK mono 800p".format(profile_name)
                    )
                extended = bool(camera.get("extended_disparity", True))
                subpixel = bool(camera.get("subpixel", False))
                if extended and subpixel:
                    raise ValueError(
                        "camera profile {} cannot enable OAK extended disparity "
                        "and subpixel together".format(profile_name)
                    )
                imu = camera.get("imu", {}) or {}
                if not isinstance(imu, dict):
                    raise ValueError("camera profile {} imu must be a mapping".format(profile_name))
                if float(imu.get("report_rate_hz", 100.0)) <= 0.0:
                    raise ValueError("camera profile {} IMU report rate must be positive".format(profile_name))
                camera_from_imu = imu.get("camera_from_imu")
                if camera_from_imu is not None:
                    matrix = np.asarray(camera_from_imu, dtype=np.float64)
                    if matrix.size != 16:
                        raise ValueError("camera profile {} camera_from_imu must be 4x4".format(profile_name))
                focus_mode = str(camera.get("focus_mode", "device_default")).strip().lower()
                if focus_mode not in ("device_default", "continuous_auto", "manual"):
                    raise ValueError(
                        "camera profile {} has invalid OAK focus_mode".format(profile_name)
                    )
                if focus_mode == "manual" and (
                    camera.get("manual_focus") is None
                    or not 0 <= int(camera["manual_focus"]) <= 255
                ):
                    raise ValueError(
                        "camera profile {} manual OAK focus must be within 0..255".format(
                            profile_name
                        )
                    )
                minimum_depth = float(camera.get("minimum_depth_m", 0.0))
                maximum_depth = float(camera.get("maximum_depth_m", 0.0))
                if not 0.0 < minimum_depth < maximum_depth:
                    raise ValueError(
                        "camera profile {} OAK depth range is invalid".format(
                            profile_name
                        )
                    )
        controller = self.data.get("controller", {})
        if not isinstance(controller, dict):
            raise ValueError("controller must be a mapping")
        transport = str(controller.get("transport", "modbus_tcp")).strip().lower()
        if transport != "modbus_tcp":
            raise ValueError(
                "controller.transport must be modbus_tcp until the official motion protocol is available"
            )
        unit_id = controller.get("unit_id")
        if unit_id is not None and not 1 <= int(unit_id) <= 247:
            raise ValueError("controller.unit_id must be within 1..247")
        for name in ("connect_timeout_s", "io_timeout_s"):
            if float(controller.get(name, 0.0)) <= 0.0:
                raise ValueError("controller.{} must be positive".format(name))
        if not 8 <= int(controller.get("max_frame_bytes", 0)) <= 260:
            raise ValueError("controller.max_frame_bytes must be within 8..260")
        if float(controller.get("state_poll_hz", 1.0)) <= 0.0:
            raise ValueError("controller.state_poll_hz must be positive")
        state_registers = controller.get("state_registers", {}) or {}
        state_codec = controller.get("state_codec", {}) or {}
        if not isinstance(state_registers, dict) or not isinstance(state_codec, dict):
            raise ValueError("controller state_registers/state_codec must be mappings")
        register_encodings = {
            "u16", "s16", "u32", "s32", "f32_be",
            "u32_le_words", "s32_le_words", "f32_le_words",
        }
        for field, spec in state_registers.items():
            if not isinstance(spec, dict):
                raise ValueError("controller.state_registers.{} must be a mapping".format(field))
            if "address" not in spec or not 0 <= int(spec["address"]) <= 65535:
                raise ValueError("controller.state_registers.{} address is invalid".format(field))
            source = str(spec.get("source", "holding")).lower()
            if source not in ("holding", "input", "coil", "coils", "discrete", "discrete_input", "discrete_inputs"):
                raise ValueError("controller.state_registers.{} source is invalid".format(field))
            encoding = str(spec.get("encoding", "u16")).lower()
            if source in ("holding", "input") and encoding not in register_encodings:
                raise ValueError("controller.state_registers.{} encoding is invalid".format(field))
            minimum_quantity = 2 if encoding in {
                "u32", "s32", "f32_be", "u32_le_words", "s32_le_words", "f32_le_words",
            } else 1
            if int(spec.get("quantity", minimum_quantity)) < minimum_quantity:
                raise ValueError("controller.state_registers.{} quantity is too small".format(field))
        if bool(controller.get("enabled", False)):
            if not str(controller.get("host", "")).strip():
                raise ValueError("enabled controller requires controller.host")
            port = controller.get("port")
            if port is None or not 1 <= int(port) <= 65535:
                raise ValueError("enabled controller.port must be within 1..65535")
            if unit_id is None:
                raise ValueError("enabled controller requires controller.unit_id")
        motion = controller.get("motion", {})
        if not isinstance(motion, dict):
            raise ValueError("controller.motion must be a mapping")
        if bool(motion.get("enabled", False)):
            raise ValueError(
                "controller.motion is fail-closed: enable it only after the official motion protocol is implemented"
            )
        remote_io = controller.get("remote_io", {})
        if not isinstance(remote_io, dict):
            raise ValueError("controller.remote_io must be a mapping")
        for direction in ("outputs", "inputs"):
            mapping = remote_io.get(direction, {})
            if not isinstance(mapping, dict):
                raise ValueError("controller.remote_io.{} must be a mapping".format(direction))
            for name, address in mapping.items():
                if not 0 <= int(address) <= 65535:
                    raise ValueError(
                        "controller.remote_io.{}[{}] must be within 0..65535".format(
                            direction, name
                        )
                    )
        fallback = controller.get("modbus_global_point_fallback", {}) or {}
        if not isinstance(fallback, dict):
            raise ValueError("controller.modbus_global_point_fallback must be a mapping")
        if bool(fallback.get("enabled", False)):
            if not bool(fallback.get("local_program_verified", False)):
                raise ValueError("enabled Modbus fallback requires local_program_verified")
            point_fields = fallback.get("global_point_fields", {}) or {}
            required_fields = {
                "coordinate_system", "angle_unit", "shape", "tool_id", "user_id",
                "reserved_1", "reserved_2", "axis_1", "axis_2", "axis_3", "axis_4",
                "axis_5", "axis_6", "axis_7",
            }
            if not required_fields.issubset(point_fields):
                raise ValueError(
                    "enabled Modbus fallback is missing GP fields: {}".format(
                        ", ".join(sorted(required_fields.difference(point_fields)))
                    )
                )
            for signal_name in (
                "command_fields", "start_signal", "complete_signal", "stop_signal",
            ):
                signal = fallback.get(signal_name, {}) or {}
                if signal_name == "command_fields":
                    if not {"motion_code", "sequence_id"}.issubset(signal):
                        raise ValueError("fallback command_fields needs motion_code and sequence_id")
                elif "address" not in signal:
                    raise ValueError("fallback {} needs address".format(signal_name))
        safety = self.data.get("safety", {})
        for name in ("workspace_min_mm", "workspace_max_mm"):
            if len(safety.get(name, [])) != 3:
                raise ValueError("{} must contain three values".format(name))
        if np.any(np.asarray(safety["workspace_min_mm"]) >= np.asarray(safety["workspace_max_mm"])):
            raise ValueError("workspace minimum must be below maximum on every axis")
        return self

    @property
    def tag_map(self):
        return self.data["tag_map"]

    @property
    def camera_profiles(self):
        return self.data["camera"]["profiles"]

    @property
    def active_camera_profile(self):
        return str(self.data["camera"]["active_profile"])

    @property
    def camera(self):
        return self.camera_profiles[self.active_camera_profile]

    @property
    def controller(self):
        """Detached controller settings; endpoint values may remain unset."""
        return self.data.get("controller", {})

    def set_active_camera_profile(self, profile_name, save=True):
        profile_name = str(profile_name)
        if profile_name not in self.camera_profiles:
            raise ValueError("unknown camera profile: {}".format(profile_name))
        changed = profile_name != self.active_camera_profile
        if changed:
            self.data["camera"]["active_profile"] = profile_name
            # This transform depends on both the physical camera and its mount.
            self.data["hand_eye"]["tcp_from_color_camera"]["valid"] = False
            self.data["segmentation_validation"]["validation"] = {
                "valid": False,
                "weights_sha256": "",
                "camera_profile": "",
                "confirmed_at": "",
            }
            if save:
                self.save()
        return changed

    def set_active_depth_mode(self, mode_name, save=True):
        """Select an Astra Depth/IR stream mode without invalidating RGB work."""
        camera = self.camera
        if camera.get("backend") != "astra_ros":
            raise ValueError("depth mode selection is only available for Astra")
        mode_name = str(mode_name)
        if mode_name not in camera.get("depth_modes", {}):
            raise ValueError("unknown Astra depth mode: {}".format(mode_name))
        changed = mode_name != str(camera.get("depth_mode", ""))
        if changed:
            camera["depth_mode"] = mode_name
            if save:
                self.save()
        return changed

    def runtime_camera(self):
        """Return a detached camera config with the selected stream mode applied."""
        camera = copy.deepcopy(self.camera)
        if camera.get("backend") == "astra_ros":
            mode_name = str(camera["depth_mode"])
            mode = camera["depth_modes"][mode_name]
            driver = camera.setdefault("ros_driver", {})
            arguments = driver.setdefault("arguments", {})
            for field in (
                "depth_width", "depth_height", "depth_fps",
                "ir_width", "ir_height", "ir_fps",
            ):
                arguments[field] = int(mode[field])
        return camera

    @property
    def hand_eye_valid(self):
        return bool(self.data["hand_eye"]["tcp_from_color_camera"].get("valid", False))

    @property
    def segmentation_valid(self):
        segmentation = self.data["segmentation_validation"]
        validation = segmentation.get("validation", {})
        return bool(validation.get("valid", False)) and (
            validation.get("camera_profile") == self.active_camera_profile
        ) and bool(validation.get("weights_sha256")) and (
            str(validation.get("weights_file", "")) == str(segmentation.get("weights_file", ""))
        )

    @property
    def tcp_from_color_camera(self):
        return as_transform(
            self.data["hand_eye"]["tcp_from_color_camera"]["matrix"],
            "tcp_from_color_camera",
        )

    def resolve_path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.path.parent / path).resolve()

    def save(self):
        self.validate()
        if self.path.exists():
            backup_dir = self.path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            shutil.copy2(
                str(self.path),
                str(backup_dir / "{}_{}{}".format(self.path.stem, stamp, self.path.suffix)),
            )
        atomic_write_yaml(self.path, self.data)


def _matrix_data(entry, shape):
    values = entry.get("data") if isinstance(entry, dict) else entry
    return np.asarray(values, dtype=np.float64).reshape(shape)


def load_camera_intrinsics(path, camera="color"):
    """Load RGB/color intrinsics from a pipeline or ROS camera YAML."""
    resolved, data = _read_yaml(path)
    entry = data.get("cameras", {}).get(camera, data)
    matrix = _matrix_data(entry["camera_matrix"], (3, 3))
    distortion = _matrix_data(entry["distortion_coefficients"], (-1, 1))
    width = int(entry["image_width"])
    height = int(entry["image_height"])
    if width <= 0 or height <= 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid camera intrinsics in {}".format(resolved))
    return matrix, distortion, (width, height)
