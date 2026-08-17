"""Competition RGB/IR calibration with fail-closed quality checks."""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from tool.object_model_builder.rgbd_calibration import (
    StereoCalibrationResult,
    calibrate_color_from_depth,
    detect_charuco,
    infrared_to_uint8,
)
from tool.object_model_builder.rgbd_geometry import (
    CameraIntrinsics,
    DepthToColorAligner,
    RgbdCalibration,
)
from tool.camera_calibration.calib_common import charuco_board

from .configuration import atomic_write_yaml, load_camera_intrinsics


@dataclass(frozen=True)
class RobustRgbdCalibrationResult:
    calibration: RgbdCalibration
    pairs_used: int
    common_corner_counts: tuple
    rejected_pair_indices: tuple
    retained_pair_errors_px: tuple


def camera_intrinsics_from_file(path):
    matrix, distortion, size = load_camera_intrinsics(path)
    return CameraIntrinsics(size[0], size[1], matrix, distortion.reshape(-1))


def intrinsics_signature(intrinsics):
    """Stable cache key for a CameraInfo/calibration intrinsic model."""
    if intrinsics is None:
        return None
    return (
        int(intrinsics.width),
        int(intrinsics.height),
        np.asarray(intrinsics.matrix, dtype=np.float64).tobytes(),
        np.asarray(intrinsics.distortion, dtype=np.float64).tobytes(),
    )


def calibration_for_depth_frame(calibration, depth_intrinsics, depth_shape):
    """Adapt a calibrated RGB-D transform to the active raw Depth mode.

    RGB↔Depth extrinsics are rigid and do not change with stream resolution.
    Calibrated Depth intrinsics remain preferable when their dimensions match;
    otherwise the active ROS CameraInfo model is required.
    """
    height, width = map(int, tuple(depth_shape)[:2])
    calibrated = calibration.depth
    if (calibrated.width, calibrated.height) == (width, height):
        depth = calibrated
    elif (
        depth_intrinsics is not None
        and (depth_intrinsics.width, depth_intrinsics.height) == (width, height)
    ):
        depth = depth_intrinsics
    else:
        live_size = (
            "无"
            if depth_intrinsics is None
            else "{}x{}".format(
                depth_intrinsics.width, depth_intrinsics.height
            )
        )
        raise ValueError(
            "Depth 当前为 {}x{}，标定内参为 {}x{}，实时 CameraInfo 为 {}；"
            "无法安全对齐".format(
                width, height, calibrated.width, calibrated.height, live_size
            )
        )
    return RgbdCalibration(
        color=calibration.color,
        depth=depth,
        color_from_depth=calibration.color_from_depth,
        valid=calibration.valid,
        source=calibration.source,
        rms_reprojection_error_px=calibration.rms_reprojection_error_px,
    )


def rotation_magnitude_deg(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def validate_rgbd_calibration(calibration, maximum_rms_px=2.0):
    calibration.require_valid()
    rms = calibration.rms_reprojection_error_px
    if rms is None or not np.isfinite(rms) or float(rms) > float(maximum_rms_px):
        raise ValueError(
            "RGB/IR stereo RMS is invalid or exceeds {:.3f} px: {}".format(
                float(maximum_rms_px), rms
            )
        )
    depth = calibration.depth
    largest_dimension = float(max(depth.width, depth.height))
    fx, fy = float(depth.matrix[0, 0]), float(depth.matrix[1, 1])
    if not (0.2 * largest_dimension <= fx <= 10.0 * largest_dimension):
        raise ValueError("IR focal length fx is physically implausible: {}".format(fx))
    if not (0.2 * largest_dimension <= fy <= 10.0 * largest_dimension):
        raise ValueError("IR focal length fy is physically implausible: {}".format(fy))
    cx, cy = float(depth.matrix[0, 2]), float(depth.matrix[1, 2])
    if not (-0.5 * depth.width <= cx <= 1.5 * depth.width):
        raise ValueError("IR principal point cx is physically implausible: {}".format(cx))
    if not (-0.5 * depth.height <= cy <= 1.5 * depth.height):
        raise ValueError("IR principal point cy is physically implausible: {}".format(cy))
    transform = calibration.color_from_depth
    baseline_m = float(np.linalg.norm(transform[:3, 3]))
    if baseline_m > 0.25:
        raise ValueError("RGB/IR baseline is physically implausible: {:.1f} mm".format(baseline_m * 1000.0))
    angle_deg = rotation_magnitude_deg(transform[:3, :3])
    if angle_deg > 30.0:
        raise ValueError("RGB/IR relative rotation is physically implausible: {:.2f} deg".format(angle_deg))
    return calibration


def calibrate_rgb_ir_pairs(
    pairs,
    color_intrinsics,
    minimum_common_corners=10,
    minimum_pairs=12,
    maximum_rms_px=2.0,
    maximum_rejection_fraction=0.40,
):
    valid_pairs = []
    valid_indices = []
    for index, pair in enumerate(pairs):
        color_corners, color_ids = detect_charuco(pair[0])
        ir_corners, ir_ids = detect_charuco(pair[1])
        del color_corners, ir_corners
        common = set(color_ids.tolist()).intersection(ir_ids.tolist())
        if len(common) >= int(minimum_common_corners):
            valid_pairs.append(pair)
            valid_indices.append(index)
    if len(valid_pairs) < int(minimum_pairs):
        raise ValueError(
            "only {} valid RGB/IR pairs; at least {} are required".format(
                len(valid_pairs), minimum_pairs
            )
        )

    active_pairs = list(valid_pairs)
    active_indices = list(valid_indices)
    rejected = []
    maximum_rejections = min(
        len(active_pairs) - int(minimum_pairs),
        int(math.floor(len(active_pairs) * float(maximum_rejection_fraction))),
    )

    while True:
        result = calibrate_color_from_depth(
            active_pairs,
            color_intrinsics,
            minimum_common_corners=minimum_common_corners,
            minimum_pairs=minimum_pairs,
        )
        # Reject physically impossible solutions before any statistical trimming.
        validate_rgbd_calibration(result.calibration, float("inf"))
        errors = tuple(
            _pair_color_reprojection_rms(pair, result.calibration)
            for pair in active_pairs
        )
        rms = float(result.calibration.rms_reprojection_error_px)
        if rms <= float(maximum_rms_px):
            validate_rgbd_calibration(result.calibration, maximum_rms_px)
            return RobustRgbdCalibrationResult(
                calibration=result.calibration,
                pairs_used=result.pairs_used,
                common_corner_counts=tuple(result.common_corner_counts),
                rejected_pair_indices=tuple(rejected),
                retained_pair_errors_px=errors,
            )
        if len(rejected) >= maximum_rejections:
            raise ValueError(
                "RGB/IR RMS {:.4f} px still exceeds {:.3f} px after rejecting "
                "{} of {} outlier pairs; capture more varied, sharp views".format(
                    rms, float(maximum_rms_px), len(rejected), len(valid_pairs)
                )
            )
        worst = int(np.argmax(np.asarray(errors, dtype=np.float64)))
        rejected.append(active_indices.pop(worst))
        active_pairs.pop(worst)


def _pair_color_reprojection_rms(pair, calibration):
    color_corners, color_ids = detect_charuco(pair[0])
    ir_corners, ir_ids = detect_charuco(pair[1])
    common_ids = sorted(set(color_ids.tolist()).intersection(ir_ids.tolist()))
    if len(common_ids) < 4:
        return float("inf")
    board_points = np.asarray(
        charuco_board().getChessboardCorners(), dtype=np.float32
    )
    color_lookup = {
        int(tag_id): point for tag_id, point in zip(color_ids, color_corners)
    }
    ir_lookup = {
        int(tag_id): point for tag_id, point in zip(ir_ids, ir_corners)
    }
    object_points = np.asarray(
        [board_points[tag_id] for tag_id in common_ids], dtype=np.float32
    )
    color_points = np.asarray(
        [color_lookup[tag_id] for tag_id in common_ids], dtype=np.float32
    )
    infrared_points = np.asarray(
        [ir_lookup[tag_id] for tag_id in common_ids], dtype=np.float32
    )
    ok, rotation_vector, translation = cv2.solvePnP(
        object_points,
        infrared_points,
        calibration.depth.matrix,
        calibration.depth.distortion,
    )
    if not ok:
        return float("inf")
    rotation = cv2.Rodrigues(rotation_vector)[0]
    points_depth = (
        rotation @ object_points.astype(np.float64).T + translation
    ).T
    points_color = (
        calibration.color_from_depth[:3, :3] @ points_depth.T
        + calibration.color_from_depth[:3, 3:4]
    ).T
    projected, _ = cv2.projectPoints(
        points_color,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        calibration.color.matrix,
        calibration.color.distortion,
    )
    residual = projected.reshape(-1, 2) - color_points
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def calibration_to_mapping(calibration):
    return {
        "schema_version": 1,
        "frames": {
            "color": "camera_color_optical_frame",
            "depth": "camera_depth_optical_frame",
        },
        "color": calibration.color.to_mapping(),
        "depth": calibration.depth.to_mapping(),
        "color_from_depth": {
            "valid": bool(calibration.valid),
            "description": "p_color = T_color_depth * p_depth; translation is meters",
            "matrix": calibration.color_from_depth.tolist(),
        },
        "quality": {
            "stereo_rms_reprojection_error_px": calibration.rms_reprojection_error_px,
        },
    }


def save_rgbd_result(path, result):
    mapping = calibration_to_mapping(result.calibration)
    mapping["quality"].update({
        "pairs_used": int(result.pairs_used),
        "common_corner_counts": [int(value) for value in result.common_corner_counts],
        "rejected_pair_indices": [
            int(value) for value in getattr(result, "rejected_pair_indices", ())
        ],
        "retained_pair_errors_px": [
            float(value) for value in getattr(result, "retained_pair_errors_px", ())
        ],
    })
    atomic_write_yaml(path, mapping)


def load_rgbd_result(path, maximum_rms_px=2.0):
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    color = CameraIntrinsics.from_mapping(data["color"])
    depth = CameraIntrinsics.from_mapping(data["depth"])
    transform = data["color_from_depth"]
    calibration = RgbdCalibration(
        color=color,
        depth=depth,
        color_from_depth=transform["matrix"],
        valid=bool(transform.get("valid", False)),
        source=str(resolved),
        rms_reprojection_error_px=data.get("quality", {}).get(
            "stereo_rms_reprojection_error_px"
        ),
    )
    return validate_rgbd_calibration(calibration, maximum_rms_px)


def depth_colormap(depth_m):
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], (2.0, 98.0))
        if high > low:
            normalized[valid] = np.clip(
                (depth[valid] - low) * (255.0 / (high - low)), 0, 255
            ).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def infrared_preview(ir_image):
    gray = infrared_to_uint8(ir_image)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


__all__ = [
    "CameraIntrinsics",
    "DepthToColorAligner",
    "RobustRgbdCalibrationResult",
    "StereoCalibrationResult",
    "calibrate_rgb_ir_pairs",
    "calibration_for_depth_frame",
    "camera_intrinsics_from_file",
    "depth_colormap",
    "detect_charuco",
    "infrared_preview",
    "intrinsics_signature",
    "load_rgbd_result",
    "save_rgbd_result",
    "validate_rgbd_calibration",
]
