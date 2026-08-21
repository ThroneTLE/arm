"""Typed data exchanged between framework components."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class FrameData:
    color_bgr: np.ndarray
    camera_matrix: np.ndarray
    distortion: np.ndarray
    timestamp_s: float
    frame_id: str
    depth_m: Optional[np.ndarray] = None
    depth_aligned_to_color: bool = False


@dataclass
class DetectionResult:
    bbox_xyxy: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    mask: Optional[np.ndarray] = None


@dataclass
class SegmentationResult:
    valid: bool
    mask: Optional[np.ndarray] = None
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    class_id: Optional[int] = None
    class_name: str = ""
    confidence: float = 0.0
    simulated: bool = False
    reason: str = ""
    detections: Tuple[DetectionResult, ...] = field(default_factory=tuple)


@dataclass
class ObjectPoseEstimate:
    valid: bool
    camera_from_object: Optional[np.ndarray] = None
    score: float = 0.0
    tracking: bool = False
    simulated: bool = False
    reason: str = ""
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    class_id: Optional[int] = None
    class_name: str = ""
    confidence: float = 0.0


@dataclass
class RobotState:
    valid: bool
    base_from_gripper: Optional[np.ndarray]
    timestamp_s: float
    simulated: bool = False
    reason: str = ""


@dataclass
class CameraLocalization:
    valid: bool
    workspace_from_camera: Optional[np.ndarray]
    source: str
    timestamp_s: float
    visible_tag_ids: Tuple[int, ...] = field(default_factory=tuple)
    rms_reprojection_error_px: Optional[float] = None
    simulated: bool = False
    reason: str = ""


@dataclass
class PipelineResult:
    valid: bool
    timestamp_s: float
    workspace_from_object: Optional[np.ndarray] = None
    camera_localization: Optional[CameraLocalization] = None
    segmentation: Optional[SegmentationResult] = None
    object_pose: Optional[ObjectPoseEstimate] = None
    simulated: bool = False
    reason: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    workspace_from_objects: Tuple[np.ndarray, ...] = field(default_factory=tuple)
    object_poses: Tuple[ObjectPoseEstimate, ...] = field(default_factory=tuple)
