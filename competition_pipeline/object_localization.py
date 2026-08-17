"""Generic segmented-object point clouds in the robot base frame."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from tool.object_model_builder.rgbd_geometry import CameraIntrinsics, transform_points


@dataclass(frozen=True)
class ObjectCloudSettings:
    minimum_depth_m: float = 0.10
    maximum_depth_m: float = 3.0
    minimum_depth_coverage: float = 0.35
    minimum_valid_points: int = 80
    mask_erosion_pixels: int = 2
    keep_largest_depth_component: bool = True
    maximum_points_per_instance: int = 4000
    workspace_min_m: Optional[Tuple[float, float, float]] = None
    workspace_max_m: Optional[Tuple[float, float, float]] = None
    assume_supported_objects: bool = False
    support_plane_z_m: Optional[float] = None
    maximum_support_gap_m: float = 0.15
    support_plane_tolerance_m: float = 0.03

    def __post_init__(self):
        if float(self.maximum_depth_m) <= float(self.minimum_depth_m):
            raise ValueError("maximum_depth_m must exceed minimum_depth_m")
        if not 0.0 <= float(self.minimum_depth_coverage) <= 1.0:
            raise ValueError("minimum_depth_coverage must be within [0, 1]")
        if int(self.minimum_valid_points) < 1:
            raise ValueError("minimum_valid_points must be positive")
        if int(self.mask_erosion_pixels) < 0:
            raise ValueError("mask_erosion_pixels cannot be negative")
        if int(self.maximum_points_per_instance) < 1:
            raise ValueError("maximum_points_per_instance must be positive")
        if float(self.maximum_support_gap_m) < 0.0:
            raise ValueError("maximum_support_gap_m cannot be negative")
        if float(self.support_plane_tolerance_m) < 0.0:
            raise ValueError("support_plane_tolerance_m cannot be negative")


@dataclass
class SegmentedObjectCloud:
    valid: bool
    class_id: Optional[int] = None
    class_name: str = ""
    confidence: float = 0.0
    center_base_m: Optional[np.ndarray] = None
    bounds_min_base_m: Optional[np.ndarray] = None
    bounds_max_base_m: Optional[np.ndarray] = None
    points_base_m: Optional[np.ndarray] = None
    depth_coverage: float = 0.0
    valid_point_count: int = 0
    observed_bounds_min_base_m: Optional[np.ndarray] = None
    observed_bounds_max_base_m: Optional[np.ndarray] = None
    support_plane_z_m: Optional[float] = None
    support_constrained: bool = False
    reason: str = ""


def _largest_component(binary: np.ndarray) -> np.ndarray:
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        np.asarray(binary).astype(np.uint8), connectivity=8
    )
    if count <= 2:
        return np.asarray(binary).astype(bool)
    label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _downsample(points: np.ndarray, maximum: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) <= int(maximum):
        return points
    indices = np.linspace(0, len(points) - 1, int(maximum), dtype=np.int64)
    return points[indices]


def localize_segmented_instance(
    aligned_depth_m: np.ndarray,
    mask_result,
    color_intrinsics: CameraIntrinsics,
    base_from_camera: np.ndarray,
    settings: ObjectCloudSettings,
) -> SegmentedObjectCloud:
    depth = np.asarray(aligned_depth_m, dtype=np.float32)
    mask = None if mask_result.mask is None else np.asarray(mask_result.mask).astype(bool)
    expected = (int(color_intrinsics.height), int(color_intrinsics.width))
    description = {
        "class_id": mask_result.class_id,
        "class_name": str(mask_result.class_name),
        "confidence": float(mask_result.confidence),
    }
    if depth.shape != expected or mask is None or mask.shape != expected:
        return SegmentedObjectCloud(
            False, **description,
            reason="aligned Depth, Mask and RGB intrinsics must have identical dimensions",
        )
    if not np.any(mask):
        return SegmentedObjectCloud(False, **description, reason="instance Mask is empty")
    if int(settings.mask_erosion_pixels) > 0:
        radius = int(settings.mask_erosion_pixels)
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    mask_count = int(np.count_nonzero(mask))
    if mask_count == 0:
        return SegmentedObjectCloud(
            False, **description, reason="instance Mask vanished after erosion"
        )
    valid = (
        mask
        & np.isfinite(depth)
        & (depth >= float(settings.minimum_depth_m))
        & (depth <= float(settings.maximum_depth_m))
    )
    if settings.keep_largest_depth_component and np.any(valid):
        valid = _largest_component(valid)
    count = int(np.count_nonzero(valid))
    coverage = float(count / mask_count)
    if coverage < float(settings.minimum_depth_coverage):
        return SegmentedObjectCloud(
            False, **description, depth_coverage=coverage,
            valid_point_count=count,
            reason="valid depth coverage {:.1%} is below {:.1%}".format(
                coverage, settings.minimum_depth_coverage
            ),
        )
    if count < int(settings.minimum_valid_points):
        return SegmentedObjectCloud(
            False, **description, depth_coverage=coverage,
            valid_point_count=count,
            reason="only {} valid depth points are available".format(count),
        )

    rows, columns = np.nonzero(valid)
    pixels = np.column_stack([columns, rows]).astype(np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(
        pixels,
        color_intrinsics.matrix,
        color_intrinsics.distortion.reshape(-1, 1),
    ).reshape(-1, 2)
    z = depth[rows, columns].astype(np.float64)
    points_camera = np.column_stack([normalized[:, 0] * z, normalized[:, 1] * z, z])
    points_base = transform_points(points_camera, base_from_camera)
    finite = np.isfinite(points_base).all(axis=1)
    if settings.workspace_min_m is not None:
        finite &= np.all(
            points_base >= np.asarray(settings.workspace_min_m, dtype=np.float64), axis=1
        )
    if settings.workspace_max_m is not None:
        finite &= np.all(
            points_base <= np.asarray(settings.workspace_max_m, dtype=np.float64), axis=1
        )
    points_base = points_base[finite]
    if len(points_base) < int(settings.minimum_valid_points):
        return SegmentedObjectCloud(
            False, **description, depth_coverage=coverage,
            valid_point_count=int(len(points_base)),
            reason="only {} points remain inside the configured workspace".format(
                len(points_base)
            ),
        )
    center = np.median(points_base, axis=0)
    observed_min = np.percentile(points_base, 2.0, axis=0)
    observed_max = np.percentile(points_base, 98.0, axis=0)
    bounds_min = observed_min.copy()
    bounds_max = observed_max.copy()
    support_plane_z = None
    support_constrained = False
    if settings.assume_supported_objects and settings.support_plane_z_m is not None:
        support_plane_z = float(settings.support_plane_z_m)
        support_gap = float(observed_min[2] - support_plane_z)
        top_height = float(observed_max[2] - support_plane_z)
        if (
            support_gap >= -float(settings.support_plane_tolerance_m)
            and support_gap <= float(settings.maximum_support_gap_m)
            and top_height > float(settings.support_plane_tolerance_m)
        ):
            # Structured-light depth is often absent on dark/reflective lower
            # bottle walls.  Keep the measured points untouched, but use the
            # known table support for the object's occupancy and pose instead
            # of treating the sparse visible-point median as its center.
            bounds_min[2] = support_plane_z
            center[2] = 0.5 * (support_plane_z + observed_max[2])
            support_constrained = True
    cloud = _downsample(points_base, settings.maximum_points_per_instance)
    return SegmentedObjectCloud(
        True,
        **description,
        center_base_m=center,
        bounds_min_base_m=bounds_min,
        bounds_max_base_m=bounds_max,
        points_base_m=cloud,
        depth_coverage=coverage,
        valid_point_count=int(len(points_base)),
        observed_bounds_min_base_m=observed_min,
        observed_bounds_max_base_m=observed_max,
        support_plane_z_m=support_plane_z,
        support_constrained=support_constrained,
        reason=(
            "segmented RGB-D cloud accepted with support-plane constraint"
            if support_constrained else "segmented RGB-D cloud accepted"
        ),
    )


def localize_segmented_instances(
    aligned_depth_m: np.ndarray,
    mask_results: Sequence,
    color_intrinsics: CameraIntrinsics,
    base_from_camera: np.ndarray,
    settings: ObjectCloudSettings,
):
    return [
        localize_segmented_instance(
            aligned_depth_m,
            result,
            color_intrinsics,
            base_from_camera,
            settings,
        )
        for result in mask_results
    ]


__all__ = [
    "ObjectCloudSettings",
    "SegmentedObjectCloud",
    "localize_segmented_instance",
    "localize_segmented_instances",
]
