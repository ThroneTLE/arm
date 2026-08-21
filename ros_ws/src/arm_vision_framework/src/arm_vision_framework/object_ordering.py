"""Deterministic competition candidate ordering."""

import numpy as np


def sort_workspace_objects(objects, reference_position=None):
    """Sort candidates near-to-far, then high-to-low.

    ``objects`` is an iterable of ``(pose, payload)`` pairs; pose is a 4x4
    base/workspace transform. The default reference is the workspace origin.
    The result is stable for equal keys and leaves payloads untouched.
    """
    reference = np.zeros(3, dtype=np.float64) if reference_position is None else np.asarray(
        reference_position, dtype=np.float64
    ).reshape(3)
    entries = list(objects)
    decorated = []
    for index, (pose, payload) in enumerate(entries):
        matrix = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        position = matrix[:3, 3]
        distance = float(np.linalg.norm(position - reference))
        height = float(position[2])
        decorated.append((distance, -height, index, pose, payload))
    decorated.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(item[3], item[4]) for item in decorated]


__all__ = ["sort_workspace_objects"]
