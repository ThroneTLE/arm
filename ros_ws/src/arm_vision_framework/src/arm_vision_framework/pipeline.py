"""Pure-Python competition pipeline independent of ROS transport."""

import time

from .errors import FrameworkError
from .types import PipelineResult


class CompetitionPipeline:
    def __init__(self, segmenter, pose_estimator, localizer, robot_controller):
        self.segmenter = segmenter
        self.pose_estimator = pose_estimator
        self.localizer = localizer
        self.robot_controller = robot_controller

    def process(self, frame):
        started = time.monotonic()
        robot_state = self.robot_controller.read_state(frame.timestamp_s)
        localization = self.localizer.localize(frame, robot_state)
        try:
            segmentation = self.segmenter.segment(frame)
        except FrameworkError as error:
            return PipelineResult(
                False,
                frame.timestamp_s,
                camera_localization=localization,
                reason=str(error),
                diagnostics={"elapsed_ms": (time.monotonic() - started) * 1000.0},
            )
        if not segmentation.valid:
            return PipelineResult(
                False,
                frame.timestamp_s,
                camera_localization=localization,
                segmentation=segmentation,
                simulated=localization.simulated,
                reason=segmentation.reason,
                diagnostics={"elapsed_ms": (time.monotonic() - started) * 1000.0},
            )
        try:
            object_pose = self.pose_estimator.estimate(frame, segmentation)
        except FrameworkError as error:
            return PipelineResult(
                False,
                frame.timestamp_s,
                camera_localization=localization,
                segmentation=segmentation,
                reason=str(error),
                diagnostics={"elapsed_ms": (time.monotonic() - started) * 1000.0},
            )
        if not object_pose.valid:
            return PipelineResult(
                False,
                frame.timestamp_s,
                camera_localization=localization,
                segmentation=segmentation,
                object_pose=object_pose,
                simulated=segmentation.simulated or object_pose.simulated,
                reason=object_pose.reason,
                diagnostics={"elapsed_ms": (time.monotonic() - started) * 1000.0},
            )
        if not localization.valid:
            return PipelineResult(
                False,
                frame.timestamp_s,
                camera_localization=localization,
                segmentation=segmentation,
                object_pose=object_pose,
                simulated=segmentation.simulated or object_pose.simulated,
                reason=localization.reason,
                diagnostics={"elapsed_ms": (time.monotonic() - started) * 1000.0},
            )
        workspace_from_object = (
            localization.workspace_from_camera @ object_pose.camera_from_object
        )
        simulated = (
            localization.simulated or segmentation.simulated or object_pose.simulated
        )
        return PipelineResult(
            True,
            frame.timestamp_s,
            workspace_from_object=workspace_from_object,
            camera_localization=localization,
            segmentation=segmentation,
            object_pose=object_pose,
            simulated=simulated,
            reason="simulated pipeline output" if simulated else "object pose accepted",
            diagnostics={"elapsed_ms": (time.monotonic() - started) * 1000.0},
        )
