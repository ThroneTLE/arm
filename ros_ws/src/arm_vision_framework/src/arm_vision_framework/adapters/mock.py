"""Deterministic no-hardware adapters for framework and CI smoke tests."""

import time

import numpy as np

from ..errors import SafetyInterlockError
from ..interfaces import PoseEstimator, RobotController, Segmenter
from ..transforms import as_transform, transform_from_xyz_rpy
from ..types import ObjectPoseEstimate, RobotState, SegmentationResult


class MockSegmenter(Segmenter):
    def segment(self, frame):
        height, width = frame.color_bgr.shape[:2]
        x1, x2 = int(width * 0.4), int(width * 0.6)
        y1, y2 = int(height * 0.3), int(height * 0.75)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return SegmentationResult(
            True,
            mask=mask,
            bbox_xyxy=(x1, y1, x2, y2),
            class_id=0,
            class_name="simulated_can",
            confidence=1.0,
            simulated=True,
            reason="mock segmentation",
        )


class MockPoseEstimator(PoseEstimator):
    def __init__(self, camera_from_object=None):
        self.camera_from_object = as_transform(
            camera_from_object
            if camera_from_object is not None
            else transform_from_xyz_rpy([0.0, 0.0, 0.5], [0.0, 0.0, 0.0]),
            "mock_camera_from_object",
        )

    def estimate(self, frame, segmentation):
        if not segmentation.valid:
            return ObjectPoseEstimate(False, reason="segmentation is invalid")
        return ObjectPoseEstimate(
            True,
            self.camera_from_object.copy(),
            score=1.0,
            tracking=False,
            simulated=True,
            reason="mock 6D pose",
        )


class MockRobotController(RobotController):
    def __init__(self, base_from_gripper=None, allow_motion=False):
        self.base_from_gripper = as_transform(
            base_from_gripper
            if base_from_gripper is not None
            else transform_from_xyz_rpy([0.0, 0.0, 0.7], [180.0, 0.0, 0.0]),
            "mock_base_from_gripper",
        )
        self.allow_motion = bool(allow_motion)
        self.stopped = False

    def read_state(self, now_s=None):
        now_s = time.monotonic() if now_s is None else float(now_s)
        return RobotState(
            True,
            self.base_from_gripper.copy(),
            now_s,
            simulated=True,
            reason="mock robot state",
        )

    def move_to(self, base_from_gripper, speed_scale=0.1):
        if not self.allow_motion:
            raise SafetyInterlockError("mock robot motion is disabled by the safety interlock")
        self.base_from_gripper = as_transform(base_from_gripper, "base_from_gripper")
        return True

    def stop(self):
        self.stopped = True
        return True
