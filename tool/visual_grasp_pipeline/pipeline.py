"""High-level imports for the migrated visual grasping pipeline.

This module keeps the public API in one place for callers that want the
original ``fp_pipeline``-style entry points without importing FoundationPose.
"""

from .config import GraspRule, VisualGraspConfig
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
from .tracking import StableTracker, add_seq, parse_sequence

__all__ = [
    "GraspRule",
    "StableTracker",
    "VisualGraspConfig",
    "add_seq",
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
