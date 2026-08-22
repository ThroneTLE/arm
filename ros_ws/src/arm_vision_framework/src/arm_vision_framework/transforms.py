"""Rigid-transform helpers shared by ROS and offline tests."""

import math
from typing import Sequence, Tuple

import numpy as np


def as_transform(matrix, name="transform"):
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


def transform_from_xyz_rpy(xyz_m: Sequence[float], rpy_deg: Sequence[float]):
    """Build a transform with fixed-frame ZYX yaw-pitch-roll convention."""
    roll, pitch, yaw = np.radians(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    rotation_y = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rotation_z = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    return transform


def transform_from_inexbot_abc(xyz_m: Sequence[float], abc_rad: Sequence[float]):
    """Build a parent_from_child matrix from an Inexbot/NexBot pose readback.

    The NexBot controller reports A/B/C as intrinsic X'Y'Z' Euler angles,
    which by the duality theorem equal fixed-frame ZYX, so the rotation is
    ``R = Rx(A) @ Ry(B) @ Rz(C)`` -- the REVERSE composition order of
    :func:`transform_from_xyz_rpy` (fixed-frame XYZ / intrinsic ZYX).

    Field-verified 2026-08-22 on MOKA MR07S-930 / Inexbot C1102 (RTL-22.07,
    state service ``realPosPCS``): using this order collapsed the checkerboard
    hand-eye residuals from ~190 mm / 170 deg to <= 2.4 mm / <= 0.74 deg.
    """
    a, b, c = np.asarray(abc_rad, dtype=np.float64).reshape(3)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    rotation_x = np.asarray([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)
    rotation_y = np.asarray([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
    rotation_z = np.asarray([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_x @ rotation_y @ rotation_z
    transform[:3, 3] = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    return transform


def inexbot_abc_from_transform(matrix) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`transform_from_inexbot_abc`: extract ``(xyz_m, abc_rad)``.

    ``R = Rx(A) Ry(B) Rz(C)``, so with ``M = Rx(A)^T @ R`` one reads
    ``A = atan2(-R[1,2], R[2,2])``, ``B = asin(R[0,2])`` and
    ``C = atan2(M[1,0], M[1,1])`` (angles in radians).
    """
    transform = as_transform(matrix)
    rotation = transform[:3, :3]
    r11, r12, r21, r22 = rotation[1, 1], rotation[1, 2], rotation[2, 1], rotation[2, 2]
    a = math.atan2(-r12, r22)
    sa, ca = math.sin(a), math.cos(a)
    b = math.asin(np.clip(rotation[0, 2], -1.0, 1.0))
    c = math.atan2(ca * rotation[1, 0] + sa * rotation[2, 0],
                   ca * rotation[1, 1] + sa * rotation[2, 1])
    return transform[:3, 3].copy(), np.asarray([a, b, c], dtype=np.float64)


def xyz_rpy_from_transform(matrix) -> Tuple[np.ndarray, np.ndarray]:
    transform = as_transform(matrix)
    rotation = transform[:3, :3]
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return transform[:3, 3].copy(), np.degrees([roll, pitch, yaw])


def transform_from_quaternion(xyz_m, quaternion_xyzw):
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    return transform


def quaternion_from_transform(matrix):
    rotation = as_transform(matrix)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    return np.asarray([x, y, z, w], dtype=np.float64)
