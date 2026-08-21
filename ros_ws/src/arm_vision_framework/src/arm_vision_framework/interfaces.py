"""Stable component interfaces for vendor and algorithm adapters."""

from abc import ABC, abstractmethod

from .types import FrameData, ObjectPoseEstimate, RobotState, SegmentationResult


class Segmenter(ABC):
    @abstractmethod
    def segment(self, frame: FrameData) -> SegmentationResult:
        raise NotImplementedError


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(
        self, frame: FrameData, segmentation: SegmentationResult
    ) -> ObjectPoseEstimate:
        raise NotImplementedError

    def reset(self):
        pass


class RobotController(ABC):
    @abstractmethod
    def read_state(self, now_s=None) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def move_to(self, base_from_gripper, speed_scale=0.1):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    def move_j(self, points, speed_scale=0.1):
        """Execute a validated joint-point list using vendor MOVJ semantics."""
        raise NotImplementedError("MOVJ is not implemented by this adapter")

    def move_l(self, points, speed_mm_s=30.0):
        """Execute a validated Cartesian-point list using vendor MOVL semantics."""
        raise NotImplementedError("MOVL is not implemented by this adapter")

    def set_gripper(self, closed):
        raise NotImplementedError("gripper control is not implemented by this adapter")


class GripperController(ABC):
    """Formal gripper boundary; addresses and feedback stay vendor-specific."""

    @abstractmethod
    def open(self):
        raise NotImplementedError

    @abstractmethod
    def close(self, width_mm, maximum_effort=None):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError


__all__ = [
    "Segmenter", "PoseEstimator", "RobotController", "GripperController",
]
