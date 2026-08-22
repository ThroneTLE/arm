"""Pure geometry and grasp-pose helpers for the visual grasping pipeline.

These functions are copied/refactored from the released ``fp_pipeline.py`` so the
logic can be unit-tested without a camera, GPU, YOLO or FoundationPose.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

def fill_depth_roi(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill zero-depth holes inside a mask with the median valid depth.

    FoundationPose erodes the depth in the mask; reflective surfaces often have
    many zeros and would otherwise produce an empty valid region.
    """
    roi = np.asarray(depth_m, dtype=np.float32).copy()
    mask = np.asarray(mask)
    vals = roi[mask > 0]
    valid = vals[np.isfinite(vals) & (vals > 0.05)]
    if len(valid) == 0:
        return roi
    median = float(np.median(valid))
    hole = (mask > 0) & (roi <= 0.05)
    roi[hole] = median
    return roi


def build_world_from_tags(
    tags: Sequence[Tuple[int, np.ndarray]], base_id: int = 0
) -> Optional[np.ndarray]:
    """Build a 4x4 ``T_cam_world`` from AprilTag camera poses.

    ``tags`` is a list of ``(tag_id, T_cam_tag)``.  The tag with ``base_id``
    defines the origin; additional tags are used only to refine the plane
    normal.  This mirrors the original fp_pipeline behaviour.
    """
    if not tags:
        return None
    base = [item for item in tags if item[0] == base_id]
    if not base:
        return None
    t0 = base[0][1]
    origin = np.asarray(t0[:3, 3], dtype=np.float64).copy()
    transforms = [np.asarray(t, dtype=np.float64) for _, t in tags]

    if len(transforms) >= 2:
        points = np.asarray([t[:3, 3] for t in transforms], dtype=np.float64)
        center = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - center)
        z = vh[-1].astype(np.float64)
        if z[1] > 0:
            z = -z
        z /= np.linalg.norm(z)
        if np.dot(z, t0[:3, 2]) < 0.7:
            z = np.asarray(t0[:3, 2], dtype=np.float64)
    else:
        z = np.asarray(t0[:3, 2], dtype=np.float64)

    if z[1] > 0:
        z = -z
    z /= np.linalg.norm(z)

    x = np.asarray(t0[:3, 0], dtype=np.float64) - np.dot(t0[:3, 0], z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    world = np.eye(4, dtype=np.float64)
    world[:3, 0] = x
    world[:3, 1] = y
    world[:3, 2] = z
    world[:3, 3] = origin
    return world


def to_world_and_compensate(
    camera_from_object: np.ndarray,
    world_from_camera: np.ndarray,
    offset_xy_mm: Tuple[float, float] = (0.0, 0.0),
    center_offset_mm: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    """Transform an object pose from camera to workspace and apply compensation.

    The original pipeline used a measured offset plus optional X/Y polarity
    flips.  Keeping those as explicit arguments makes the migration testable.
    """
    result = np.asarray(world_from_camera, dtype=np.float64) @ np.asarray(
        camera_from_object, dtype=np.float64
    )
    result = result.copy()
    ox, oy = offset_xy_mm
    result[:3, 3] += np.array([ox / 1000.0, oy / 1000.0, -center_offset_mm / 1000.0])

    flip = np.eye(3)
    if flip_x:
        flip[0, 0] = -1.0
    if flip_y:
        flip[1, 1] = -1.0
    if flip_x or flip_y:
        result[:3, :3] = flip @ result[:3, :3] @ flip
        result[:3, 3] = flip @ result[:3, 3]
    return result


def compute_grasp(world_from_object: np.ndarray, offset_mm: float = 5.0) -> np.ndarray:
    """Cylinder grasp: grip the middle, keep the handle aligned with workspace Y.

    The returned frame's ``-X`` axis is the fork/handle direction and lies in the
    horizontal plane as close as possible to the workspace ``+Y`` direction.
    """
    source = np.asarray(world_from_object, dtype=np.float64)
    rotation = source[:3, :3]
    translation = source[:3, 3].copy() - rotation[:, 2] * (offset_mm / 1000.0)

    a, b = rotation[0, 0], rotation[0, 1]
    norm = math.hypot(a, b)
    if norm > 1e-6:
        ca, sa = b / norm, -a / norm
        yaw = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        grasp_rotation = rotation @ yaw
    else:
        grasp_rotation = rotation

    grasp = np.eye(4, dtype=np.float64)
    grasp[:3, :3] = grasp_rotation
    grasp[:3, 3] = translation
    return grasp


def compute_grasp_sphere(
    world_from_object: np.ndarray, offset_mm: float = 0.0
) -> np.ndarray:
    """Sphere/symmetric-object grasp: approach vertically from above.

    The fork direction is ``-Z`` (downwards) and the fork face normal is ``+Y``.
    """
    source = np.asarray(world_from_object, dtype=np.float64)
    translation = source[:3, 3].copy()
    translation[2] += offset_mm / 1000.0

    approach = np.array([0.0, 0.0, -1.0])
    normal = np.array([0.0, 1.0, 0.0])
    rotation = np.column_stack([approach, np.cross(normal, approach), normal])
    grasp = np.eye(4, dtype=np.float64)
    grasp[:3, :3] = rotation
    grasp[:3, 3] = translation
    return grasp


def smooth_pose(
    current: Optional[np.ndarray],
    new_pose: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exponentially smooth a 4x4 pose using quaternion SLERP-like averaging.

    ``current`` may be ``None``; in that case ``new_pose`` is returned unchanged.
    """
    new_pose = np.asarray(new_pose, dtype=np.float64)
    if current is None:
        return new_pose.copy()
    current = np.asarray(current, dtype=np.float64)
    new_q = np.asarray(rot2quat(new_pose[:3, :3]), dtype=np.float64)
    old_q = np.asarray(rot2quat(current[:3, :3]), dtype=np.float64)
    if np.dot(old_q, new_q) < 0.0:
        new_q = -new_q
    q = old_q + alpha * (new_q - old_q)
    q /= np.linalg.norm(q)
    t = current[:3, 3] + alpha * (new_pose[:3, 3] - current[:3, 3])
    x, y, z, w = q
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])
    out[:3, 3] = t
    return out


def rot2quat(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a rotation matrix to ``[x, y, z, w]`` quaternion."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)
