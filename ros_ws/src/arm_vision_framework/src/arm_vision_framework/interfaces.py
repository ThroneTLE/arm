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

    def set_gripper(self, closed):
        raise NotImplementedError("gripper control is not implemented by this adapter")
