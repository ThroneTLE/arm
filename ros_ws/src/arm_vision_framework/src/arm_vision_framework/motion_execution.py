"""Formal MOVJ/MOVL pick-and-place execution boundary.

The controller manual defines point metadata and the MOVJ/MOVL concepts, but
does not publish the vendor network packets.  This module therefore fixes the
task order and safety gates while delegating the actual wire commands to a
future :class:`RobotController` adapter.
"""

from dataclasses import dataclass, field, replace
import time
from typing import Callable, Sequence, Tuple

import numpy as np

from .adapters.inexbot_modbus import InexbotPoint, shape_from_joint_degrees
from .errors import SafetyInterlockError
from .safety_recovery import SafeRecoveryManager
from .shape_latch import ShapeLatch


class PickPlaceError(RuntimeError):
    """A pick/place plan or execution step was rejected."""


@dataclass(frozen=True)
class MotionEvent:
    state: str
    timestamp_s: float
    detail: str = ""


@dataclass(frozen=True)
class PickPlaceResult:
    success: bool
    state: str
    reason: str
    events: Tuple[MotionEvent, ...] = field(default_factory=tuple)


def joint_point(name: str, joints_deg: Sequence[float], tool_id=0, user_id=0):
    """Build a manual-compatible MOVJ point from six degree joint values."""

    joints = tuple(float(value) for value in joints_deg)
    if len(joints) != 6:
        raise ValueError("MOVJ point requires six joint angles")
    return InexbotPoint(
        name=name,
        coordinate_system=0,
        angle_unit=0,
        shape=shape_from_joint_degrees(joints),
        tool_id=tool_id,
        user_id=user_id,
        axes=joints + (0.0,),
    )


def cartesian_point(
    name: str,
    xyz_mm: Sequence[float],
    rpy_deg: Sequence[float],
    shape: int,
    tool_id=0,
    user_id=0,
):
    """Build a manual-compatible MOVL point.

    The manual uses millimetres for X/Y/Z and radians for A/B/C in Cartesian,
    Tool, and User point records.  The public helper accepts degrees to match
    the competition UI and converts only at this boundary.
    """

    xyz = tuple(float(value) for value in xyz_mm)
    rpy = tuple(np.radians(np.asarray(rpy_deg, dtype=np.float64).reshape(3)))
    if len(xyz) != 3:
        raise ValueError("MOVL point requires XYZ")
    return InexbotPoint(
        name=name,
        coordinate_system=1,
        angle_unit=1,
        shape=shape,
        tool_id=tool_id,
        user_id=user_id,
        axes=xyz + rpy + (0.0,),
    )


def _points(value, label):
    points = tuple(value)
    if not points:
        raise PickPlaceError("{} point list cannot be empty".format(label))
    if not all(isinstance(point, InexbotPoint) for point in points):
        raise PickPlaceError("{} contains a non-Inexbot point".format(label))
    return points


@dataclass(frozen=True)
class PickPlacePlan:
    """Validated six-stage path with placement supplied by the caller."""

    movej_pregrasp: Tuple[InexbotPoint, ...]
    movel_grasp: Tuple[InexbotPoint, ...]
    movel_lift: Tuple[InexbotPoint, ...]
    movej_preplace: Tuple[InexbotPoint, ...]
    movel_place: Tuple[InexbotPoint, ...]
    movej_retreat: Tuple[InexbotPoint, ...]
    grasp_width_mm: float
    maximum_effort: float = None

    def __post_init__(self):
        for field_name in (
            "movej_pregrasp", "movel_grasp", "movel_lift",
            "movej_preplace", "movel_place", "movej_retreat",
        ):
            points = _points(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, points)
        width = float(self.grasp_width_mm)
        if not 0.0 < width <= 200.0:
            raise ValueError("grasp_width_mm must be within (0, 200]")
        object.__setattr__(self, "grasp_width_mm", width)
        if self.maximum_effort is not None:
            object.__setattr__(self, "maximum_effort", float(self.maximum_effort))
        self.validate()

    def validate(self):
        for field_name in ("movej_pregrasp", "movej_preplace", "movej_retreat"):
            for point in getattr(self, field_name):
                if point.coordinate_system != 0 or point.angle_unit != 0:
                    raise PickPlaceError(
                        "{} must contain joint degree points".format(field_name)
                    )
        for field_name in ("movel_grasp", "movel_lift", "movel_place"):
            for point in getattr(self, field_name):
                if point.coordinate_system not in (1, 2, 3) or point.angle_unit != 1:
                    raise PickPlaceError(
                        "{} must contain Cartesian radian points".format(field_name)
                    )
        return True


class PickPlaceExecutor:
    """Execute MOVJ/MOVL and gripper actions with fail-closed stopping."""

    def __init__(
        self,
        robot,
        gripper,
        dry_run=True,
        allow_motion=False,
        movej_speed_scale=0.1,
        movel_speed_mm_s=30.0,
        cancel_requested: Callable[[], bool] = None,
        placement_verifier: Callable[[], bool] = None,
        recovery_manager: SafeRecoveryManager = None,
        shape_latch: ShapeLatch = None,
        shape_provider: Callable[[], object] = None,
    ):
        self.robot = robot
        self.gripper = gripper
        self.dry_run = bool(dry_run)
        self.allow_motion = bool(allow_motion)
        self.movej_speed_scale = float(movej_speed_scale)
        self.movel_speed_mm_s = float(movel_speed_mm_s)
        if self.movej_speed_scale <= 0.0 or self.movel_speed_mm_s <= 0.0:
            raise ValueError("motion speeds must be positive")
        self.cancel_requested = cancel_requested or (lambda: False)
        self.placement_verifier = placement_verifier
        self.recovery_manager = recovery_manager
        # The controller's first valid shape is a match-start invariant.  The
        # provider is deliberately optional for simulation/unit tests; real
        # hardware construction supplies it and therefore fails closed when
        # no valid state has been read.
        self.shape_latch = shape_latch
        self.shape_provider = shape_provider
        self._events = []

    def _event(self, state, detail=""):
        self._events.append(MotionEvent(state, time.monotonic(), detail))

    def _check(self):
        if self.cancel_requested():
            raise PickPlaceError("execution cancelled")

    @staticmethod
    def _accepted(result, state):
        if result is False:
            raise PickPlaceError("robot rejected {}".format(state))

    def _latched_points(self, points):
        """Attach the locked controller shape to every outgoing point.

        Shape is not recomputed from each joint waypoint.  If the controller
        reports a different configuration after the initial read, the whole
        command is rejected so the caller can return to a known safe point and
        re-plan explicitly.
        """
        if self.shape_latch is None and self.shape_provider is None:
            return tuple(points)
        if self.shape_latch is None:
            self.shape_latch = ShapeLatch()
        state = self.shape_provider() if self.shape_provider is not None else None
        observed = None
        if state is not None:
            observed = getattr(state, "shape", None)
            if observed is None:
                observed = getattr(state, "initial_shape", None)
        self.shape_latch.observe(observed)
        latch = self.shape_latch.state
        if self.shape_latch.value is None:
            raise PickPlaceError("initial controller shape has not been read")
        if latch.changed:
            raise PickPlaceError(
                "controller shape changed from {} to {}; refuse motion".format(
                    latch.initial_shape, latch.observed_shape
                )
            )
        return tuple(replace(point, shape=self.shape_latch.value) for point in points)

    def _move_j(self, state, points):
        self._check()
        points = self._latched_points(points)
        self._event(state, "points={}".format(len(points)))
        self._accepted(
            self.robot.move_j(points, speed_scale=self.movej_speed_scale), state
        )

    def _move_l(self, state, points):
        self._check()
        points = self._latched_points(points)
        self._event(state, "points={}".format(len(points)))
        self._accepted(
            self.robot.move_l(points, speed_mm_s=self.movel_speed_mm_s), state
        )

    def execute(self, plan: PickPlacePlan):
        self._events = []
        try:
            plan.validate()
            if self.dry_run or not self.allow_motion:
                raise SafetyInterlockError(
                    "pick/place motion is disabled; complete dry-run and safety validation first"
                )
            self._event("opening")
            self._accepted(self.gripper.open(), "gripper open")
            self._move_j("movej_pregrasp", plan.movej_pregrasp)
            self._move_l("movel_grasp", plan.movel_grasp)
            self._event("closing", "width_mm={:.1f}".format(plan.grasp_width_mm))
            self._accepted(
                self.gripper.close(plan.grasp_width_mm, plan.maximum_effort),
                "gripper close",
            )
            self._move_l("movel_lift", plan.movel_lift)
            self._move_j("movej_preplace", plan.movej_preplace)
            self._move_l("movel_place", plan.movel_place)
            self._event("releasing")
            self._accepted(self.gripper.open(), "gripper release")
            if self.placement_verifier is not None:
                self._event("verifying_place")
                if not self.placement_verifier():
                    raise PickPlaceError("placement verifier rejected the result")
            self._move_j("movej_retreat", plan.movej_retreat)
            self._event("complete")
            return PickPlaceResult(True, "complete", "pick and place completed", tuple(self._events))
        except SafetyInterlockError as error:
            # 联锁在**任何运动下发之前**就拒绝了这次调用，机器人从头到尾没有动过，
            # 所以这里不发 stop()。在这台 C1102 上 robot.stop() 发的是 0x2314，
            # 实测映射到 Deadan_End -> PowerOff，即**直接下电**而不是受控停止；
            # 对伸展着的手臂下电会让它失力坠落（2026-08-22 摔臂正是 PowerOff 造成的）。
            # 「dry-run 模式下调一次执行器就把伺服打掉」是纯粹的自伤。
            # 返回值契约保持不变，仍然是 PickPlaceResult(False, ...)。
            state = self._events[-1].state if self._events else "not_started"
            return PickPlaceResult(False, state, str(error), tuple(self._events))
        except Exception as error:
            # 到这里说明运动已经开始，停机是必要的。仍需注意 stop() 是下电语义，
            # 真正的安全急停要靠示教器上的物理急停按钮。
            try:
                self.robot.stop()
            except Exception:
                pass
            try:
                self.gripper.stop()
            except Exception:
                pass
            if self.recovery_manager is not None:
                reason = str(error)
                # A recovery is attempted only when explicitly enabled.  The
                # manager itself also checks controller/TCP state and has no
                # effect for ordinary errors when auto_recover is disabled.
                if self.recovery_manager.reason_is_singularity(reason):
                    self._event("singularity_detected", reason)
                    if self.recovery_manager.recover():
                        self._event("safe_recovered")
                    else:
                        self._event("recovery_locked", self.recovery_manager.state.last_reason)
            state = self._events[-1].state if self._events else "not_started"
            return PickPlaceResult(False, state, str(error), tuple(self._events))


def pick_place_executor_from_config(settings, robot, gripper, **kwargs):
    """Construct the formal executor from ROS ``system_parameters.yaml``."""

    data = settings.data if hasattr(settings, "data") else settings
    entry = data.get("pick_place", {})
    motion = entry.get("motion", {})
    safety = data.get("safety", {})
    recovery_entry = safety.get("recovery", {}) or {}
    recovery_manager = kwargs.pop("recovery_manager", None)
    controller_state_provider = kwargs.pop("controller_state_provider", None)
    shape_provider = kwargs.pop("shape_provider", controller_state_provider)
    shape_latch = kwargs.pop("shape_latch", None)
    if shape_latch is None and shape_provider is not None:
        shape_latch = ShapeLatch(
            data.get("controller", {}).get("initial_shape")
        )
    if recovery_manager is None:
        recovery_manager = SafeRecoveryManager(
            robot,
            auto_recover=bool(recovery_entry.get("auto_recover", False)),
            state_provider=controller_state_provider,
            singularity_error_codes=recovery_entry.get("singularity_error_codes", ()),
        )
        configured_points = []
        for point in recovery_entry.get("safe_movej_points", []) or []:
            configured_points.append(InexbotPoint(**dict(point)))
        if configured_points:
            recovery_manager.save(configured_points)
    return PickPlaceExecutor(
        robot,
        gripper,
        dry_run=bool(safety.get("dry_run", True)),
        allow_motion=bool(safety.get("allow_robot_motion", False)),
        movej_speed_scale=motion.get("movej_speed_scale", 0.1),
        movel_speed_mm_s=motion.get("movel_speed_mm_s", 30.0),
        recovery_manager=recovery_manager,
        shape_latch=shape_latch,
        shape_provider=shape_provider,
        **kwargs
    )


@dataclass(frozen=True)
class ObservationPlan:
    """Two collision-checked MOVJ observation routes before a grasp.

    A camera on the TCP must only be sampled after the robot has stopped.  The
    two routes are normally generated by MoveIt and converted to joint point
    lists by the vendor bridge.  They are not Cartesian approaches and must
    therefore remain MOVJ.
    """

    movej_first_view: Tuple[InexbotPoint, ...]
    movej_second_view: Tuple[InexbotPoint, ...]
    settle_time_s: float = 0.25

    def __post_init__(self):
        for field_name in ("movej_first_view", "movej_second_view"):
            points = _points(getattr(self, field_name), field_name)
            for point in points:
                if point.coordinate_system != 0 or point.angle_unit != 0:
                    raise PickPlaceError("{} must contain joint degree points".format(field_name))
            object.__setattr__(self, field_name, points)
        settle = float(self.settle_time_s)
        if settle < 0.0:
            raise ValueError("settle_time_s cannot be negative")
        object.__setattr__(self, "settle_time_s", settle)


class TwoViewPickPlaceCoordinator:
    """Capture two stopped-arm RGB-D views, then execute the resolved plan.

    ``capture_snapshot`` must return only after one synchronized RGB + aligned
    depth frame has been obtained.  ``plan_from_snapshots`` receives the two
    frames and returns a fully validated :class:`PickPlacePlan`; target and
    placement selection stay outside this hardware-ordering boundary.
    """

    def __init__(
        self, executor, capture_snapshot, plan_from_snapshots,
        sleep=time.sleep, cancel_requested=None,
    ):
        self.executor = executor
        self.capture_snapshot = capture_snapshot
        self.plan_from_snapshots = plan_from_snapshots
        self._sleep = sleep
        self.cancel_requested = cancel_requested or (lambda: False)

    def _check(self):
        if self.cancel_requested():
            raise PickPlaceError("observation cancelled")

    def execute(self, observation: ObservationPlan):
        events = []
        try:
            if self.executor.dry_run or not self.executor.allow_motion:
                raise SafetyInterlockError(
                    "two-view grasp is disabled; complete dry-run and safety validation first"
                )
            self._check()
            self.executor._events = []
            self.executor._move_j("movej_first_view", observation.movej_first_view)
            events.extend(self.executor._events)
            self.executor._events = []
            if observation.settle_time_s:
                self._sleep(observation.settle_time_s)
            self._check()
            events.append(MotionEvent("capturing_first_view", time.monotonic()))
            first = self.capture_snapshot()
            self._check()
            self.executor._move_j("movej_second_view", observation.movej_second_view)
            events.extend(self.executor._events)
            self.executor._events = []
            if observation.settle_time_s:
                self._sleep(observation.settle_time_s)
            self._check()
            events.append(MotionEvent("capturing_second_view", time.monotonic()))
            second = self.capture_snapshot()
            self._check()
            events.append(MotionEvent("planning_grasp", time.monotonic()))
            plan = self.plan_from_snapshots(first, second)
            if not isinstance(plan, PickPlacePlan):
                raise PickPlaceError("plan_from_snapshots must return PickPlacePlan")
            result = self.executor.execute(plan)
            return PickPlaceResult(
                result.success, result.state, result.reason,
                tuple(events) + result.events,
            )
        except Exception as error:
            try:
                self.executor.robot.stop()
            except Exception:
                pass
            try:
                self.executor.gripper.stop()
            except Exception:
                pass
            recovery = self.executor.recovery_manager
            if recovery is not None and recovery.reason_is_singularity(str(error)):
                events.append(MotionEvent("singularity_detected", time.monotonic(), str(error)))
                if recovery.recover():
                    events.append(MotionEvent("safe_recovered", time.monotonic()))
                else:
                    events.append(MotionEvent(
                        "recovery_locked", time.monotonic(), recovery.state.last_reason
                    ))
            state = events[-1].state if events else "not_started"
            return PickPlaceResult(False, state, str(error), tuple(events))


__all__ = [
    "PickPlaceError", "MotionEvent", "PickPlaceResult", "PickPlacePlan",
    "PickPlaceExecutor", "pick_place_executor_from_config", "joint_point",
    "cartesian_point", "ObservationPlan", "TwoViewPickPlaceCoordinator",
]
