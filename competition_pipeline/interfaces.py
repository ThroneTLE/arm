"""Stable hardware boundaries for later camera and robot vendor adapters."""

from dataclasses import dataclass

import numpy as np

from .geometry import as_transform


@dataclass
class RobotPoseSample:
    base_from_tcp: np.ndarray
    timestamp_s: float

    def __post_init__(self):
        self.base_from_tcp = as_transform(self.base_from_tcp, "base_from_tcp")
        self.timestamp_s = float(self.timestamp_s)


class RobotPoseProvider:
    """A vendor bridge only needs to return the latest base-from-TCP pose."""

    def latest_pose(self):
        raise NotImplementedError


class RobotController(RobotPoseProvider):
    """Vendor adapter contract. Pose and target are both T_base_tcp."""

    def move_tcp(self, base_from_tcp, speed_scale):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class GripperController:
    """Parallel-jaw gripper boundary; all widths are physical metres."""

    def open(self):
        raise NotImplementedError

    def close(self, width_m, maximum_effort=None):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


@dataclass
class ObjectPoseSample:
    object_id: str
    base_from_object: np.ndarray
    timestamp_s: float

    def __post_init__(self):
        self.object_id = str(self.object_id)
        self.base_from_object = as_transform(
            self.base_from_object, "base_from_object"
        )
        self.timestamp_s = float(self.timestamp_s)


class ObjectPoseProvider:
    """Optional verification boundary used by simulation or a live tracker."""

    def latest_object_pose(self, object_id):
        raise NotImplementedError


@dataclass
class RgbdFrame:
    """One synchronized sample from the single RGB-D camera."""

    color_bgr: np.ndarray
    timestamp_s: float
    ir_image: np.ndarray = None
    depth_m: np.ndarray = None


class RgbdFrameProvider:
    """Optional vendor boundary for a color + IR/depth frame source."""

    def read(self):
        raise NotImplementedError
