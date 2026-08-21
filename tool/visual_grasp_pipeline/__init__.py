"""Visual grasping pipeline tool migrated from the fp_release package.

This package contains the reusable, hardware-independent parts of the original
FoundationPose + YOLO + AprilTag grasping demo.  Vendor/model files stay outside
the repository; configuration points at the local data roots.
"""

from .detection import (
    detect_all_objects,
    detect_all_track,
    detect_tags,
    draw_boxes,
    select_target,
)
from .geometry import (
    build_world_from_tags,
    compute_grasp,
    compute_grasp_sphere,
    fill_depth_roi,
    smooth_pose,
    to_world_and_compensate,
)
from .tracking import StableTracker, parse_sequence

__all__ = [
    "StableTracker",
    "build_world_from_tags",
    "compute_grasp",
    "compute_grasp_sphere",
    "detect_all_objects",
    "detect_all_track",
    "detect_tags",
    "draw_boxes",
    "fill_depth_roi",
    "parse_sequence",
    "select_target",
    "smooth_pose",
    "to_world_and_compensate",
]
