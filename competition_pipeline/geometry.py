"""Rigid-transform helpers. Runtime lengths are meters and angles are degrees."""

import math

import numpy as np


def as_transform(value, name="transform"):
    transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
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


def transform_from_xyz_rpy(xyz_m, rpy_deg):
    """Return parent_from_child using fixed-frame Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.radians(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    return result


def transform_from_xyz_rpy_mm(xyz_mm, rpy_deg):
    return transform_from_xyz_rpy(np.asarray(xyz_mm, dtype=np.float64) / 1000.0, rpy_deg)


def xyz_rpy_from_transform(value):
    transform = as_transform(value)
    rotation = transform[:3, :3]
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return transform[:3, 3].copy(), np.degrees([roll, pitch, yaw])


def rotation_angle_deg(first, second):
    relative = as_transform(first)[:3, :3].T @ as_transform(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def interpolate_transforms(first, second, maximum_translation_m, maximum_rotation_deg):
    """Split a rigid motion so every returned step respects both limits.

    The first pose is not returned; the final pose is always returned. Rotation
    uses shortest-path quaternion interpolation and translation is linear.
    """
    start = as_transform(first, "first transform")
    target = as_transform(second, "second transform")
    translation_distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    rotation_distance = rotation_angle_deg(start, target)
    maximum_translation_m = float(maximum_translation_m)
    maximum_rotation_deg = float(maximum_rotation_deg)
    if maximum_translation_m <= 0.0 or maximum_rotation_deg <= 0.0:
        raise ValueError("interpolation limits must be positive")
    steps = max(
        1,
        int(math.ceil(translation_distance / maximum_translation_m)),
        int(math.ceil(rotation_distance / maximum_rotation_deg)),
    )
    q0 = _quaternion_wxyz(start[:3, :3])
    q1 = _quaternion_wxyz(target[:3, :3])
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    poses = []
    for index in range(1, steps + 1):
        fraction = float(index) / float(steps)
        if dot > 0.9995:
            quaternion = q0 + fraction * (q1 - q0)
            quaternion /= np.linalg.norm(quaternion)
        else:
            theta = math.acos(dot)
            denominator = math.sin(theta)
            quaternion = (
                math.sin((1.0 - fraction) * theta) / denominator * q0
                + math.sin(fraction * theta) / denominator * q1
            )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _rotation_from_quaternion_wxyz(quaternion)
        pose[:3, 3] = (
            (1.0 - fraction) * start[:3, 3] + fraction * target[:3, 3]
        )
        poses.append(pose)
    return poses


def _quaternion_wxyz(rotation):
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    return np.asarray([w, x, y, z], dtype=np.float64)


def _rotation_from_quaternion_wxyz(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_transforms(transforms):
    items = [as_transform(item) for item in transforms]
    if not items:
        raise ValueError("at least one transform is required")
    quaternions = [_quaternion_wxyz(item[:3, :3]) for item in items]
    reference = quaternions[0]
    quaternions = [q if np.dot(q, reference) >= 0.0 else -q for q in quaternions]
    accumulator = sum(np.outer(q, q) for q in quaternions)
    values, vectors = np.linalg.eigh(accumulator)
    quaternion = vectors[:, np.argmax(values)]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation_from_quaternion_wxyz(quaternion)
    result[:3, 3] = np.median(np.asarray([item[:3, 3] for item in items]), axis=0)
    return result
