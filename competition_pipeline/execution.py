"""Fail-closed grasp execution state machine with observable verification."""

from dataclasses import dataclass, field
import time
from typing import Callable, List, Optional

import numpy as np

from .geometry import interpolate_transforms


@dataclass(frozen=True)
class ExecutionEvent:
    state: str
    timestamp_s: float
    detail: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    state: str
    reason: str
    events: tuple = field(default_factory=tuple)


class GraspExecutor:
    STATES = (
        "opening", "moving_pregrasp", "approaching", "closing", "lifting",
        "verifying_lift", "moving_preplace", "placing", "releasing",
        "verifying_place", "retreating", "complete",
    )

    def __init__(
        self,
        robot,
        gripper,
        object_pose_provider=None,
        speed_scale=0.1,
        maximum_segment_mm=40.0,
        maximum_segment_rotation_deg=8.0,
        minimum_verified_lift_mm=30.0,
        placement_tolerance_mm=80.0,
        maximum_object_pose_age_s=0.5,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ):
        self.robot = robot
        self.gripper = gripper
        self.object_pose_provider = object_pose_provider
        self.speed_scale = float(speed_scale)
        self.maximum_segment_m = float(maximum_segment_mm) / 1000.0
        self.maximum_segment_rotation_deg = float(maximum_segment_rotation_deg)
        self.minimum_verified_lift_m = float(minimum_verified_lift_mm) / 1000.0
        self.placement_tolerance_m = float(placement_tolerance_mm) / 1000.0
        self.maximum_object_pose_age_s = float(maximum_object_pose_age_s)
        self.cancel_requested = cancel_requested or (lambda: False)
        self._events: List[ExecutionEvent] = []

    def _event(self, state, detail=""):
        self._events.append(ExecutionEvent(state, time.monotonic(), detail))

    def _check_cancelled(self):
        if self.cancel_requested():
            self.robot.stop()
            self.gripper.stop()
            raise RuntimeError("execution cancelled")

    def _move(self, state, target):
        self._event(state)
        current = self.robot.latest_pose()
        if current is None:
            raise RuntimeError("current TCP pose is unavailable")
        segments = interpolate_transforms(
            current.base_from_tcp,
            target,
            self.maximum_segment_m,
            self.maximum_segment_rotation_deg,
        )
        for segment in segments:
            self._check_cancelled()
            result = self.robot.move_tcp(segment, self.speed_scale)
            if result is False:
                raise RuntimeError("robot rejected {} motion".format(state))

    def _object_pose(self, object_id):
        if self.object_pose_provider is None:
            return None
        sample = self.object_pose_provider.latest_object_pose(object_id)
        if sample is None:
            raise RuntimeError("object pose is unavailable for verification")
        if abs(time.monotonic() - sample.timestamp_s) > self.maximum_object_pose_age_s:
            raise RuntimeError("object pose is stale")
        return sample

    def execute(self, plan):
        self._events = []
        initial_object = None
        try:
            self._check_cancelled()
            initial_object = self._object_pose(plan.target.object_id)
            self._event("opening")
            if self.gripper.open() is False:
                raise RuntimeError("gripper failed to open")
            self._move("moving_pregrasp", plan.base_from_tcp_pregrasp)
            self._move("approaching", plan.base_from_tcp_grasp)
            self._event("closing", "width_m={:.4f}".format(plan.target.width_m))
            if self.gripper.close(plan.target.width_m) is False:
                raise RuntimeError("gripper failed to close")
            self._move("lifting", plan.base_from_tcp_lift)
            self._event("verifying_lift")
            lifted = self._object_pose(plan.target.object_id)
            if initial_object is not None and (
                lifted.base_from_object[2, 3] - initial_object.base_from_object[2, 3]
                < self.minimum_verified_lift_m
            ):
                raise RuntimeError("object did not rise by the required lift distance")
            self._move("moving_preplace", plan.base_from_tcp_preplace)
            self._move("placing", plan.base_from_tcp_place)
            self._event("releasing")
            if self.gripper.open() is False:
                raise RuntimeError("gripper failed to release")
            self._event("verifying_place")
            placed = self._object_pose(plan.target.object_id)
            if placed is not None:
                error = float(
                    np.linalg.norm(
                        placed.base_from_object[:2, 3]
                        - plan.place_position_base_m[:2]
                    )
                )
                if error > self.placement_tolerance_m:
                    raise RuntimeError(
                        "object placement error {:.1f} mm exceeds tolerance".format(error * 1000.0)
                    )
            self._move("retreating", plan.base_from_tcp_retreat)
            self._event("complete")
            return ExecutionResult(True, "complete", "grasp and place completed", tuple(self._events))
        except Exception as error:
            try:
                self.robot.stop()
            except Exception:
                pass
            try:
                self.gripper.stop()
            except Exception:
                pass
            state = self._events[-1].state if self._events else "not_started"
            return ExecutionResult(False, state, str(error), tuple(self._events))


def executor_from_config(config, robot, gripper, object_pose_provider=None, cancel_requested=None):
    data = config.data if hasattr(config, "data") else config
    entry = data["grasp_execution"]
    return GraspExecutor(
        robot,
        gripper,
        object_pose_provider=object_pose_provider,
        speed_scale=entry["speed_scale"],
        maximum_segment_mm=entry["maximum_segment_mm"],
        maximum_segment_rotation_deg=entry["maximum_segment_rotation_deg"],
        minimum_verified_lift_mm=entry["minimum_verified_lift_mm"],
        placement_tolerance_mm=entry["placement_tolerance_mm"],
        maximum_object_pose_age_s=entry["maximum_object_pose_age_s"],
        cancel_requested=cancel_requested,
    )


__all__ = ["ExecutionEvent", "ExecutionResult", "GraspExecutor", "executor_from_config"]
