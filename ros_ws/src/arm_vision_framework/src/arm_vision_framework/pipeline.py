"""Pure-Python competition pipeline independent of ROS transport."""

import time

from .errors import FrameworkError
from .types import PipelineResult
from .object_ordering import sort_workspace_objects


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
            if hasattr(self.pose_estimator, "estimate_all"):
                object_poses = tuple(self.pose_estimator.estimate_all(frame, segmentation))
            else:
                object_poses = (self.pose_estimator.estimate(frame, segmentation),)
        except FrameworkError as error:
            return PipelineResult(
                False,
                frame.timestamp_s,
                camera_localization=localization,
                segmentation=segmentation,
                reason=str(error),
                diagnostics={"elapsed_ms": (time.monotonic() - started) * 1000.0},
            )
        object_poses = tuple(pose for pose in object_poses if pose.valid)
        if not object_poses:
            reason = "no FoundationPose object pose accepted"
            return PipelineResult(
                False,
                frame.timestamp_s,
                camera_localization=localization,
                segmentation=segmentation,
                object_pose=None,
                object_poses=tuple(),
                simulated=segmentation.simulated,
                reason=reason,
                diagnostics={"elapsed_ms": (time.monotonic() - started) * 1000.0},
            )
        if not localization.valid:
            return PipelineResult(
                False,
                frame.timestamp_s,
                camera_localization=localization,
                segmentation=segmentation,
                object_pose=object_poses[0],
                object_poses=object_poses,
                simulated=segmentation.simulated or any(
                    pose.simulated for pose in object_poses
                ),
                reason=localization.reason,
                diagnostics={"elapsed_ms": (time.monotonic() - started) * 1000.0},
            )
        workspace_entries = [
            (
                localization.workspace_from_camera @ pose.camera_from_object,
                pose,
            )
            for pose in object_poses
        ]
        reference = localization.workspace_from_camera[:3, 3]
        workspace_entries = sort_workspace_objects(workspace_entries, reference)
        workspace_from_objects = tuple(item[0] for item in workspace_entries)
        ordered_poses = tuple(item[1] for item in workspace_entries)
        workspace_from_object = workspace_from_objects[0]
        simulated = (
            localization.simulated or segmentation.simulated or any(
                pose.simulated for pose in ordered_poses
            )
        )
        return PipelineResult(
            True,
            frame.timestamp_s,
            workspace_from_object=workspace_from_object,
            camera_localization=localization,
            segmentation=segmentation,
            object_pose=ordered_poses[0],
            simulated=simulated,
            reason="simulated pipeline output" if simulated else "object pose accepted",
            diagnostics={
                "elapsed_ms": (time.monotonic() - started) * 1000.0,
                "object_count": len(ordered_poses),
                "ordering": "near_to_far_then_high_to_low",
            },
            workspace_from_objects=workspace_from_objects,
            object_poses=ordered_poses,
        )
