#!/usr/bin/env python3
"""Geometry primitives for calibrated RGB-D registration."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml


def _as_transform(matrix, name="transform") -> np.ndarray:
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(transform)):
        raise ValueError("{} contains non-finite values".format(name))
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("{} has an invalid homogeneous row".format(name))
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("{} rotation is not orthonormal".format(name))
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("{} rotation determinant is not +1".format(name))
    return transform.copy()


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray

    def __post_init__(self) -> None:
        width = int(self.width)
        height = int(self.height)
        matrix = np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if width <= 0 or height <= 0:
            raise ValueError("camera image size must be positive")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
            raise ValueError("camera intrinsics contain non-finite values")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "distortion", distortion)

    @classmethod
    def from_mapping(cls, entry: dict) -> "CameraIntrinsics":
        if not isinstance(entry, dict):
            raise ValueError("camera intrinsics entry must be a mapping")
        matrix_entry = entry.get("camera_matrix", entry.get("K"))
        distortion_entry = entry.get(
            "distortion_coefficients", entry.get("distortion", entry.get("D", []))
        )
        if isinstance(matrix_entry, dict):
            matrix_entry = matrix_entry.get("data")
        if isinstance(distortion_entry, dict):
            distortion_entry = distortion_entry.get("data")
        if matrix_entry is None:
            raise ValueError("camera_matrix is missing")
        return cls(
            width=int(entry.get("image_width", entry.get("width", 0))),
            height=int(entry.get("image_height", entry.get("height", 0))),
            matrix=np.asarray(matrix_entry, dtype=np.float64),
            distortion=np.asarray(distortion_entry or [], dtype=np.float64),
        )

    def to_mapping(self) -> dict:
        return {
            "image_width": self.width,
            "image_height": self.height,
            "distortion_model": "plumb_bob",
            "camera_matrix": self.matrix.tolist(),
            "distortion_coefficients": self.distortion.tolist(),
        }


@dataclass(frozen=True)
class RgbdCalibration:
    color: CameraIntrinsics
    depth: CameraIntrinsics
    color_from_depth: np.ndarray
    valid: bool
    source: str = ""
    rms_reprojection_error_px: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "color_from_depth",
            _as_transform(self.color_from_depth, "color_from_depth"),
        )

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError(
                "RGB-D calibration is invalid; calibrate depth/IR intrinsics and "
                "T_color_depth before alignment"
            )


def load_runtime_calibration(path: str) -> RgbdCalibration:
    calibration_path = Path(path).expanduser().resolve()
    with open(calibration_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    camera = data.get("camera", {})
    color_entry = camera.get("color", {})
    depth_entry = camera.get("depth", {})
    color = CameraIntrinsics.from_mapping(color_entry)
    depth = CameraIntrinsics.from_mapping(depth_entry)
    transform_entry = data.get("transforms", {}).get("color_from_depth", {})
    if "matrix" not in transform_entry:
        raise ValueError("transforms.color_from_depth.matrix is missing")
    valid = bool(transform_entry.get("valid", False))
    valid = valid and bool(depth_entry.get("intrinsics_valid", depth_entry.get("valid", False)))
    quality = data.get("quality", {})
    return RgbdCalibration(
        color=color,
        depth=depth,
        color_from_depth=transform_entry["matrix"],
        valid=valid,
        source=str(calibration_path),
        rms_reprojection_error_px=quality.get("rgbd_stereo_rms_px"),
    )


def save_rgbd_calibration(path: str, calibration: RgbdCalibration) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "frames": {
            "color": "camera_color_optical_frame",
            "depth": "camera_depth_optical_frame",
        },
        "color": calibration.color.to_mapping(),
        "depth": calibration.depth.to_mapping(),
        "color_from_depth": {
            "valid": bool(calibration.valid),
            "description": "p_color = T_color_depth * p_depth",
            "matrix": calibration.color_from_depth.tolist(),
        },
        "quality": {
            "stereo_rms_reprojection_error_px": calibration.rms_reprojection_error_px,
        },
    }
    with open(destination, "w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False, allow_unicode=True)


def depth_pixels_to_points(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project valid raw depth pixels into the depth optical frame."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            "depth shape {} does not match calibrated {}x{}".format(
                depth.shape, intrinsics.width, intrinsics.height
            )
        )
    valid = np.isfinite(depth) & (depth >= minimum_depth_m) & (depth <= maximum_depth_m)
    rows, columns = np.nonzero(valid)
    if columns.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
    pixels = np.column_stack([columns, rows]).astype(np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(
        pixels,
        intrinsics.matrix,
        intrinsics.distortion.reshape(-1, 1),
    ).reshape(-1, 2)
    z = depth[rows, columns].astype(np.float64)
    points = np.column_stack([normalized[:, 0] * z, normalized[:, 1] * z, z])
    return points, np.column_stack([columns, rows]).astype(np.int32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rigid = _as_transform(transform)
    return points_array @ rigid[:3, :3].T + rigid[:3, 3]


def project_points(
    points_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    apply_distortion: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)
    positive = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    if positive.any():
        pixels, _ = cv2.projectPoints(
            points[positive],
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            intrinsics.matrix,
            (
                intrinsics.distortion.reshape(-1, 1)
                if apply_distortion
                else np.zeros((5, 1), dtype=np.float64)
            ),
        )
        projected[positive] = pixels.reshape(-1, 2)
    return projected, positive


class DepthToColorAligner:
    """Reusable rectified depth projector with calibration geometry cached.

    Depth values change for every frame, but the undistorted depth rays and the
    rigid/color projection coefficients do not.  Precomputing those coefficients
    avoids running ``cv2.undistortPoints`` over every depth pixel on every frame.
    Float64 coefficients intentionally preserve the pixel-for-pixel output of the
    reference OpenCV projection path.
    """

    def __init__(self, calibration: RgbdCalibration):
        calibration.require_valid()
        self.calibration = calibration
        depth = calibration.depth
        color = calibration.color
        columns, rows = np.meshgrid(
            np.arange(depth.width, dtype=np.float64),
            np.arange(depth.height, dtype=np.float64),
        )
        pixels = np.column_stack([columns.reshape(-1), rows.reshape(-1)]).reshape(
            -1, 1, 2
        )
        normalized = cv2.undistortPoints(
            pixels,
            depth.matrix,
            depth.distortion.reshape(-1, 1),
        ).reshape(-1, 2)
        depth_rays = np.column_stack(
            [normalized, np.ones(len(normalized), dtype=np.float64)]
        )
        rotated_rays = depth_rays @ calibration.color_from_depth[:3, :3].T
        translation = calibration.color_from_depth[:3, 3]
        matrix = color.matrix

        self._color_z_coefficient = rotated_rays[:, 2]
        self._column_numerator_coefficient = (
            matrix[0, 0] * rotated_rays[:, 0]
            + matrix[0, 2] * rotated_rays[:, 2]
        )
        self._row_numerator_coefficient = (
            matrix[1, 1] * rotated_rays[:, 1]
            + matrix[1, 2] * rotated_rays[:, 2]
        )
        self._color_z_offset = float(translation[2])
        self._column_numerator_offset = float(
            matrix[0, 0] * translation[0] + matrix[0, 2] * translation[2]
        )
        self._row_numerator_offset = float(
            matrix[1, 1] * translation[1] + matrix[1, 2] * translation[2]
        )
        scale = max(color.width / depth.width, color.height / depth.height)
        self._default_splat_radius_pixels = max(
            0, int(math.ceil((scale - 1.0) * 0.5))
        )

    def align(
        self,
        depth_m: np.ndarray,
        minimum_depth_m: float = 0.05,
        maximum_depth_m: float = 3.0,
        splat_radius_pixels: Optional[int] = None,
    ) -> np.ndarray:
        """Project one raw depth frame into the rectified color image."""
        calibration = self.calibration
        depth = np.asarray(depth_m, dtype=np.float32)
        expected_shape = (calibration.depth.height, calibration.depth.width)
        if depth.shape != expected_shape:
            raise ValueError(
                "depth shape {} does not match calibrated {}x{}".format(
                    depth.shape,
                    calibration.depth.width,
                    calibration.depth.height,
                )
            )
        flat_depth = depth.reshape(-1)
        finite_depth = np.isfinite(flat_depth)
        with np.errstate(invalid="ignore"):
            valid_depth = (
                finite_depth
                & (flat_depth >= float(minimum_depth_m))
                & (flat_depth <= float(maximum_depth_m))
            )
        projection_depth = (
            flat_depth
            if finite_depth.all()
            else np.where(finite_depth, flat_depth, 0.0)
        )
        color_z = (
            self._color_z_coefficient * projection_depth + self._color_z_offset
        )
        positive = valid_depth & np.isfinite(color_z) & (color_z > 0.0)
        nonzero_color_z = np.isfinite(color_z) & (color_z != 0.0)
        safe_color_z = (
            color_z
            if nonzero_color_z.all()
            else np.where(nonzero_color_z, color_z, 1.0)
        )
        columns = np.rint(
            (
                self._column_numerator_coefficient * projection_depth
                + self._column_numerator_offset
            )
            / safe_color_z
        ).astype(np.int32)
        rows = np.rint(
            (
                self._row_numerator_coefficient * projection_depth
                + self._row_numerator_offset
            )
            / safe_color_z
        ).astype(np.int32)
        inside = (
            positive
            & (columns >= 0)
            & (columns < calibration.color.width)
            & (rows >= 0)
            & (rows < calibration.color.height)
        )
        z_buffer = np.full(
            calibration.color.width * calibration.color.height,
            np.inf,
            dtype=np.float32,
        )
        radius = (
            self._default_splat_radius_pixels
            if splat_radius_pixels is None
            else max(0, int(splat_radius_pixels))
        )
        if inside.any():
            valid_rows = rows[inside]
            valid_columns = columns[inside]
            valid_z = color_z[inside].astype(np.float32)
            for offset_y in range(-radius, radius + 1):
                for offset_x in range(-radius, radius + 1):
                    target_rows = valid_rows + offset_y
                    target_columns = valid_columns + offset_x
                    target_inside = (
                        (target_rows >= 0)
                        & (target_rows < calibration.color.height)
                        & (target_columns >= 0)
                        & (target_columns < calibration.color.width)
                    )
                    flat_indices = (
                        target_rows[target_inside] * calibration.color.width
                        + target_columns[target_inside]
                    )
                    np.minimum.at(z_buffer, flat_indices, valid_z[target_inside])
        z_buffer[~np.isfinite(z_buffer)] = 0.0
        return z_buffer.reshape(calibration.color.height, calibration.color.width)


def align_depth_to_color(
    depth_m: np.ndarray,
    calibration: RgbdCalibration,
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = 3.0,
    rectified_output: bool = True,
    splat_radius_pixels: Optional[int] = None,
) -> np.ndarray:
    """Create color-sized Z depth using calibrated 3D reprojection and a Z-buffer.

    The default output corresponds to an RGB image rectified with the original K as
    its new camera matrix. A small resolution-aware splat fills the color pixels
    represented by each lower-resolution depth sample; it is not image scaling.
    """
    calibration.require_valid()
    if rectified_output:
        return DepthToColorAligner(calibration).align(
            depth_m,
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
            splat_radius_pixels=splat_radius_pixels,
        )
    depth_points, _ = depth_pixels_to_points(
        depth_m,
        calibration.depth,
        minimum_depth_m=minimum_depth_m,
        maximum_depth_m=maximum_depth_m,
    )
    color_points = transform_points(depth_points, calibration.color_from_depth)
    pixels, positive = project_points(
        color_points,
        calibration.color,
        apply_distortion=not rectified_output,
    )
    finite_pixels = np.isfinite(pixels).all(axis=1)
    columns = np.zeros(len(pixels), dtype=np.int64)
    rows = np.zeros(len(pixels), dtype=np.int64)
    columns[finite_pixels] = np.rint(pixels[finite_pixels, 0]).astype(np.int64)
    rows[finite_pixels] = np.rint(pixels[finite_pixels, 1]).astype(np.int64)
    inside = (
        positive
        & finite_pixels
        & (columns >= 0)
        & (columns < calibration.color.width)
        & (rows >= 0)
        & (rows < calibration.color.height)
    )
    output_size = calibration.color.width * calibration.color.height
    z_buffer = np.full(output_size, np.inf, dtype=np.float32)
    if splat_radius_pixels is None:
        scale = max(
            calibration.color.width / calibration.depth.width,
            calibration.color.height / calibration.depth.height,
        )
        splat_radius_pixels = max(0, int(math.ceil((scale - 1.0) * 0.5)))
    radius = max(0, int(splat_radius_pixels))
    if inside.any():
        valid_rows = rows[inside]
        valid_columns = columns[inside]
        valid_z = color_points[inside, 2].astype(np.float32)
        for offset_y in range(-radius, radius + 1):
            for offset_x in range(-radius, radius + 1):
                target_rows = valid_rows + offset_y
                target_columns = valid_columns + offset_x
                target_inside = (
                    (target_rows >= 0)
                    & (target_rows < calibration.color.height)
                    & (target_columns >= 0)
                    & (target_columns < calibration.color.width)
                )
                flat_indices = (
                    target_rows[target_inside] * calibration.color.width
                    + target_columns[target_inside]
                )
                np.minimum.at(z_buffer, flat_indices, valid_z[target_inside])
    z_buffer[~np.isfinite(z_buffer)] = 0.0
    return z_buffer.reshape(calibration.color.height, calibration.color.width)


def rectified_intrinsics(intrinsics: CameraIntrinsics) -> CameraIntrinsics:
    return CameraIntrinsics(
        intrinsics.width,
        intrinsics.height,
        intrinsics.matrix.copy(),
        np.zeros_like(intrinsics.distortion),
    )


def rectify_color_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    color = np.asarray(image)
    if color.shape[:2] != (intrinsics.height, intrinsics.width):
        raise ValueError("color image dimensions do not match calibrated intrinsics")
    return cv2.undistort(
        color,
        intrinsics.matrix,
        intrinsics.distortion.reshape(-1, 1),
        None,
        intrinsics.matrix,
    )


def rectify_aligned_depth_image(
    aligned_depth_m: np.ndarray,
    color_intrinsics: CameraIntrinsics,
) -> np.ndarray:
    depth = np.asarray(aligned_depth_m, dtype=np.float32)
    if depth.shape != (color_intrinsics.height, color_intrinsics.width):
        raise ValueError("aligned depth dimensions do not match color intrinsics")
    map_x, map_y = cv2.initUndistortRectifyMap(
        color_intrinsics.matrix,
        color_intrinsics.distortion.reshape(-1, 1),
        np.eye(3, dtype=np.float64),
        color_intrinsics.matrix,
        (color_intrinsics.width, color_intrinsics.height),
        cv2.CV_32FC1,
    )
    return cv2.remap(
        depth,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def mask_aligned_depth(
    aligned_depth_m: np.ndarray,
    mask: np.ndarray,
    erosion_pixels: int = 1,
) -> np.ndarray:
    depth = np.asarray(aligned_depth_m, dtype=np.float32)
    binary = np.asarray(mask).astype(bool)
    if depth.shape != binary.shape:
        raise ValueError("aligned depth and mask dimensions do not match")
    if erosion_pixels > 0:
        size = int(erosion_pixels) * 2 + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        binary = cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    output = np.zeros_like(depth)
    output[binary] = depth[binary]
    return output


def depth_coverage(aligned_depth_m: np.ndarray, mask: np.ndarray) -> float:
    depth = np.asarray(aligned_depth_m)
    binary = np.asarray(mask).astype(bool)
    if depth.shape != binary.shape:
        raise ValueError("aligned depth and mask dimensions do not match")
    count = int(binary.sum())
    if count == 0:
        return 0.0
    return float(np.count_nonzero((depth > 0.0) & binary) / count)


def masked_depth_centroid(
    aligned_depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = 3.0,
    minimum_valid_points: int = 50,
) -> np.ndarray:
    """Return the robust median 3D point inside a color-aligned object mask."""
    depth = np.asarray(aligned_depth_m, dtype=np.float32)
    binary = np.asarray(mask).astype(bool)
    expected_shape = (intrinsics.height, intrinsics.width)
    if depth.shape != expected_shape or binary.shape != expected_shape:
        raise ValueError("aligned depth, mask, and color intrinsics must match")
    valid = (
        binary
        & np.isfinite(depth)
        & (depth >= float(minimum_depth_m))
        & (depth <= float(maximum_depth_m))
    )
    rows, columns = np.nonzero(valid)
    if len(columns) < int(minimum_valid_points):
        raise ValueError(
            "object mask contains only {} valid depth points; at least {} are required".format(
                len(columns), minimum_valid_points
            )
        )
    pixels = np.column_stack([columns, rows]).astype(np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(
        pixels,
        intrinsics.matrix,
        intrinsics.distortion.reshape(-1, 1),
    ).reshape(-1, 2)
    z = depth[rows, columns].astype(np.float64)
    points = np.column_stack([normalized[:, 0] * z, normalized[:, 1] * z, z])
    return np.median(points, axis=0)
