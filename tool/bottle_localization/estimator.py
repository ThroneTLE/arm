#!/usr/bin/env python3
"""Calibrated bottle position estimation without an object CAD model."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from tool.object_model_builder.rgbd_geometry import CameraIntrinsics, transform_points


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("{} must be a finite non-zero vector".format(name))
    return value / norm


def workspace_plane_basis(workspace_up: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right-handed X/Y/Z axes with Z aligned to the workspace up vector."""
    z_axis = _unit(workspace_up, "workspace_up")
    preferred = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(preferred, z_axis))) > 0.95:
        preferred = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = _unit(preferred - np.dot(preferred, z_axis) * z_axis, "plane_x")
    y_axis = _unit(np.cross(z_axis, x_axis), "plane_y")
    return x_axis, y_axis, z_axis


def fit_fixed_radius_circle(
    points_xy: np.ndarray,
    radius_m: float,
    initial_center_xy: Optional[np.ndarray] = None,
    maximum_iterations: int = 20,
) -> Tuple[np.ndarray, float]:
    """Robustly fit a known-radius circle to a visible cylinder surface arc."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    radius = float(radius_m)
    if len(points) < 3:
        raise ValueError("circle fitting requires at least three points")
    if not np.all(np.isfinite(points)) or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("circle points and radius must be finite and radius must be positive")

    if initial_center_xy is None:
        # Algebraic circle fit supplies a deterministic seed. The known-radius
        # geometric iterations below correct its radius and reduce partial-arc bias.
        design = np.column_stack(
            [2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))]
        )
        target = np.sum(points * points, axis=1)
        solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        center = solution[:2]
    else:
        center = np.asarray(initial_center_xy, dtype=np.float64).reshape(2).copy()
    if not np.all(np.isfinite(center)):
        center = np.median(points, axis=0)

    for _ in range(max(1, int(maximum_iterations))):
        offset = points - center
        distance = np.linalg.norm(offset, axis=1)
        valid = distance > 1e-9
        if int(np.count_nonzero(valid)) < 3:
            break
        residual = distance[valid] - radius
        jacobian = -offset[valid] / distance[valid, None]
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = max(1.4826 * mad, radius * 0.03, 1e-5)
        normalized = np.abs(residual - median) / (1.5 * scale)
        weights = np.ones_like(normalized)
        outside = normalized > 1.0
        weights[outside] = 1.0 / normalized[outside]
        weighted_jacobian = jacobian * np.sqrt(weights)[:, None]
        weighted_target = -residual * np.sqrt(weights)
        delta, _, _, _ = np.linalg.lstsq(
            weighted_jacobian, weighted_target, rcond=None
        )
        if not np.all(np.isfinite(delta)):
            break
        center += delta
        if float(np.linalg.norm(delta)) < 1e-7:
            break

    residual = np.linalg.norm(points - center, axis=1) - radius
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    scale = max(1.4826 * mad, radius * 0.03, 1e-5)
    inliers = np.abs(residual - median) <= 3.0 * scale
    quality_residual = residual[inliers] if int(np.count_nonzero(inliers)) >= 3 else residual
    rms = float(np.sqrt(np.mean(quality_residual * quality_residual)))
    return center, rms


@dataclass(frozen=True)
class BottlePositionSettings:
    workspace_up: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    workspace_plane_z_m: float = 0.0
    minimum_depth_m: float = 0.15
    maximum_depth_m: float = 2.0
    minimum_object_height_m: float = 0.004
    maximum_object_height_m: float = 0.6
    minimum_depth_coverage: float = 0.5
    minimum_valid_points: int = 100
    mask_erosion_pixels: int = 2
    keep_largest_depth_component: bool = True
    nominal_bottle_height_m: float = 0.0
    nominal_bottle_diameter_m: float = 0.0
    cylinder_fit_seed_radius_multiplier: float = 2.0
    maximum_radial_excess_m: float = 0.010
    maximum_height_excess_m: float = 0.015
    smoothing_alpha: float = 0.35
    maximum_cloud_points: int = 6000

    def __post_init__(self) -> None:
        _unit(np.asarray(self.workspace_up), "workspace_up")
        if not 0.0 <= float(self.minimum_depth_coverage) <= 1.0:
            raise ValueError("minimum_depth_coverage must be within [0, 1]")
        if float(self.maximum_depth_m) <= float(self.minimum_depth_m):
            raise ValueError("maximum_depth_m must exceed minimum_depth_m")
        if float(self.maximum_object_height_m) <= float(self.minimum_object_height_m):
            raise ValueError("maximum_object_height_m must exceed minimum_object_height_m")
        if int(self.minimum_valid_points) < 3:
            raise ValueError("minimum_valid_points must be at least three")
        if not 0.0 < float(self.smoothing_alpha) <= 1.0:
            raise ValueError("smoothing_alpha must be within (0, 1]")
        if float(self.nominal_bottle_height_m) < 0.0:
            raise ValueError("nominal_bottle_height_m cannot be negative")
        if float(self.nominal_bottle_diameter_m) < 0.0:
            raise ValueError("nominal_bottle_diameter_m cannot be negative")
        if float(self.cylinder_fit_seed_radius_multiplier) <= 1.0:
            raise ValueError("cylinder_fit_seed_radius_multiplier must exceed one")
        if float(self.maximum_radial_excess_m) < 0.0:
            raise ValueError("maximum_radial_excess_m cannot be negative")
        if float(self.maximum_height_excess_m) < 0.0:
            raise ValueError("maximum_height_excess_m cannot be negative")


@dataclass
class BottleEstimate:
    valid: bool
    workspace_from_bottle: Optional[np.ndarray] = None
    workspace_from_bottle_base: Optional[np.ndarray] = None
    camera_from_bottle: Optional[np.ndarray] = None
    center_workspace_m: Optional[np.ndarray] = None
    base_center_workspace_m: Optional[np.ndarray] = None
    center_camera_m: Optional[np.ndarray] = None
    cloud_workspace_m: Optional[np.ndarray] = None
    depth_coverage: float = 0.0
    valid_point_count: int = 0
    observed_height_m: float = 0.0
    observed_diameter_m: float = 0.0
    nominal_height_m: float = 0.0
    nominal_diameter_m: float = 0.0
    circle_fit_rms_m: Optional[float] = None
    method: str = ""
    reason: str = ""


class BottlePositionEstimator:
    """Estimate an upright bottle base and center in the AprilTag workspace."""

    def __init__(self, settings: BottlePositionSettings):
        self.settings = settings
        self._filtered_base = None
        self._filtered_center = None

    def reset(self) -> None:
        self._filtered_base = None
        self._filtered_center = None

    def estimate(
        self,
        aligned_depth_m: np.ndarray,
        mask: np.ndarray,
        color_intrinsics: CameraIntrinsics,
        workspace_from_camera: np.ndarray,
    ) -> BottleEstimate:
        settings = self.settings
        depth = np.asarray(aligned_depth_m, dtype=np.float32)
        binary = np.asarray(mask).astype(bool)
        expected = (color_intrinsics.height, color_intrinsics.width)
        if depth.shape != expected or binary.shape != expected:
            return BottleEstimate(
                False,
                reason="aligned depth, YOLO mask, and color intrinsics must have identical dimensions",
            )
        mask_count = int(np.count_nonzero(binary))
        if mask_count == 0:
            return BottleEstimate(False, reason="YOLO mask is empty")
        if settings.mask_erosion_pixels > 0:
            size = int(settings.mask_erosion_pixels) * 2 + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            binary = cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        eroded_count = int(np.count_nonzero(binary))
        if eroded_count == 0:
            return BottleEstimate(False, reason="YOLO mask vanished after erosion")

        valid_depth = (
            binary
            & np.isfinite(depth)
            & (depth >= float(settings.minimum_depth_m))
            & (depth <= float(settings.maximum_depth_m))
        )
        if settings.keep_largest_depth_component and np.any(valid_depth):
            component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
                valid_depth.astype(np.uint8), connectivity=8
            )
            if component_count > 2:
                largest_label = 1 + int(
                    np.argmax(statistics[1:, cv2.CC_STAT_AREA])
                )
                valid_depth &= labels == largest_label
        depth_count = int(np.count_nonzero(valid_depth))
        coverage = float(depth_count / eroded_count)
        if coverage < float(settings.minimum_depth_coverage):
            return BottleEstimate(
                False,
                depth_coverage=coverage,
                valid_point_count=depth_count,
                reason="valid depth coverage {:.1%} is below the configured {:.1%}".format(
                    coverage, settings.minimum_depth_coverage
                ),
            )
        if depth_count < int(settings.minimum_valid_points):
            return BottleEstimate(
                False,
                depth_coverage=coverage,
                valid_point_count=depth_count,
                reason="only {} valid depth points are available".format(depth_count),
            )

        rows, columns = np.nonzero(valid_depth)
        pixels = np.column_stack([columns, rows]).astype(np.float64).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(
            pixels,
            color_intrinsics.matrix,
            color_intrinsics.distortion.reshape(-1, 1),
        ).reshape(-1, 2)
        z = depth[rows, columns].astype(np.float64)
        points_camera = np.column_stack([normalized[:, 0] * z, normalized[:, 1] * z, z])
        workspace_from_camera = np.asarray(
            workspace_from_camera, dtype=np.float64
        ).reshape(4, 4)
        points_workspace = transform_points(points_camera, workspace_from_camera)

        x_axis, y_axis, up = workspace_plane_basis(np.asarray(settings.workspace_up))
        plane_origin = np.asarray(
            [0.0, 0.0, float(settings.workspace_plane_z_m)], dtype=np.float64
        )
        relative = points_workspace - plane_origin
        height = relative @ up
        object_points = (
            np.isfinite(points_workspace).all(axis=1)
            & (height >= float(settings.minimum_object_height_m))
            & (height <= float(settings.maximum_object_height_m))
        )
        points_camera = points_camera[object_points]
        points_workspace = points_workspace[object_points]
        height = height[object_points]
        if len(points_workspace) < int(settings.minimum_valid_points):
            return BottleEstimate(
                False,
                depth_coverage=coverage,
                valid_point_count=len(points_workspace),
                reason="too few Mask points remain above the calibrated table plane",
            )

        relative = points_workspace - plane_origin
        plane_xy = np.column_stack([relative @ x_axis, relative @ y_axis])
        center_xy = np.median(plane_xy, axis=0)
        method = "visible_surface_median"
        circle_rms = None
        nominal_radius = float(settings.nominal_bottle_diameter_m) * 0.5
        if nominal_radius > 0.0:
            camera_relative = workspace_from_camera[:3, 3] - plane_origin
            camera_xy = np.asarray(
                [camera_relative @ x_axis, camera_relative @ y_axis], dtype=np.float64
            )
            direction = center_xy - camera_xy
            direction_norm = float(np.linalg.norm(direction))
            seed = center_xy.copy()
            if direction_norm > 1e-9:
                seed += nominal_radius * direction / direction_norm
            # Prefer the central body band so a narrow neck/cap does not pull
            # the known-radius cylinder center away from the bottle body.
            if float(settings.nominal_bottle_height_m) > 0.0:
                band = (
                    (height >= 0.15 * float(settings.nominal_bottle_height_m))
                    & (height <= 0.75 * float(settings.nominal_bottle_height_m))
                )
            else:
                band_limits = np.percentile(height, [20.0, 80.0])
                band = (height >= band_limits[0]) & (height <= band_limits[1])
            sample = plane_xy[band]
            if len(sample) < int(settings.minimum_valid_points):
                sample = plane_xy
            seed_distance = np.linalg.norm(sample - seed, axis=1)
            seed_limit = max(
                float(settings.cylinder_fit_seed_radius_multiplier) * nominal_radius,
                nominal_radius + 2.0 * float(settings.maximum_radial_excess_m),
            )
            seed_inliers = seed_distance <= seed_limit
            if int(np.count_nonzero(seed_inliers)) >= int(settings.minimum_valid_points):
                sample = sample[seed_inliers]
            if len(sample) > int(settings.maximum_cloud_points):
                stride = int(np.ceil(len(sample) / float(settings.maximum_cloud_points)))
                sample = sample[::stride]
            fitted, circle_rms = fit_fixed_radius_circle(
                sample, nominal_radius, initial_center_xy=seed
            )
            # A partial arc can be ill-conditioned. Keep the physically seeded
            # estimate if optimization escapes well beyond one bottle diameter.
            if (
                np.all(np.isfinite(fitted))
                and float(np.linalg.norm(fitted - seed))
                <= max(2.0 * nominal_radius, 0.02)
            ):
                center_xy = fitted
                method = "known_radius_cylinder_fit"
            else:
                center_xy = seed
                method = "known_radius_camera_ray"

            radial_distance = np.linalg.norm(plane_xy - center_xy, axis=1)
            envelope = radial_distance <= (
                nominal_radius + float(settings.maximum_radial_excess_m)
            )
            if float(settings.nominal_bottle_height_m) > 0.0:
                envelope &= height <= (
                    float(settings.nominal_bottle_height_m)
                    + float(settings.maximum_height_excess_m)
                )
            envelope_count = int(np.count_nonzero(envelope))
            if envelope_count < int(settings.minimum_valid_points):
                return BottleEstimate(
                    False,
                    depth_coverage=coverage,
                    valid_point_count=envelope_count,
                    circle_fit_rms_m=circle_rms,
                    method=method,
                    reason="too few depth points remain inside the measured bottle envelope",
                )
            points_workspace = points_workspace[envelope]
            height = height[envelope]
            plane_xy = plane_xy[envelope]

        base = plane_origin + center_xy[0] * x_axis + center_xy[1] * y_axis
        lower, upper = np.percentile(height, [5.0, 95.0])
        observed_height = max(0.0, float(upper - max(0.0, lower)))
        plane_span = np.percentile(plane_xy, 95.0, axis=0) - np.percentile(
            plane_xy, 5.0, axis=0
        )
        observed_diameter = max(0.0, float(np.max(plane_span)))
        if float(settings.nominal_bottle_height_m) > 0.0:
            center_height = 0.5 * float(settings.nominal_bottle_height_m)
        else:
            center_height = float(np.median(height))
        center = base + center_height * up
        base, center = self._smooth(base, center)

        rotation = np.column_stack([x_axis, y_axis, up])
        workspace_from_bottle = np.eye(4, dtype=np.float64)
        workspace_from_bottle[:3, :3] = rotation
        workspace_from_bottle[:3, 3] = center
        workspace_from_base = workspace_from_bottle.copy()
        workspace_from_base[:3, 3] = base
        camera_from_workspace = np.linalg.inv(workspace_from_camera)
        camera_from_bottle = camera_from_workspace @ workspace_from_bottle

        maximum = max(1, int(settings.maximum_cloud_points))
        cloud = points_workspace
        if len(cloud) > maximum:
            stride = int(np.ceil(len(cloud) / float(maximum)))
            cloud = cloud[::stride]
        return BottleEstimate(
            True,
            workspace_from_bottle=workspace_from_bottle,
            workspace_from_bottle_base=workspace_from_base,
            camera_from_bottle=camera_from_bottle,
            center_workspace_m=center,
            base_center_workspace_m=base,
            center_camera_m=camera_from_bottle[:3, 3].copy(),
            cloud_workspace_m=cloud,
            depth_coverage=coverage,
            valid_point_count=len(points_workspace),
            observed_height_m=observed_height,
            observed_diameter_m=observed_diameter,
            nominal_height_m=float(settings.nominal_bottle_height_m),
            nominal_diameter_m=float(settings.nominal_bottle_diameter_m),
            circle_fit_rms_m=circle_rms,
            method=method,
            reason="bottle position accepted",
        )

    def _smooth(self, base: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha = float(self.settings.smoothing_alpha)
        if self._filtered_base is None:
            self._filtered_base = np.asarray(base, dtype=np.float64).copy()
            self._filtered_center = np.asarray(center, dtype=np.float64).copy()
        else:
            self._filtered_base = alpha * base + (1.0 - alpha) * self._filtered_base
            self._filtered_center = alpha * center + (1.0 - alpha) * self._filtered_center
        return self._filtered_base.copy(), self._filtered_center.copy()
