"""Robot adapter for the framework's canonical ROS pose topics."""

import time

from ..errors import SafetyInterlockError
from ..interfaces import RobotController
from ..transforms import as_transform
from ..types import RobotState


class TopicRobotController(RobotController):
    def __init__(self, allow_motion=False):
        self.allow_motion = bool(allow_motion)
        self._state = RobotState(False, None, 0.0, reason="no robot pose received")
        self._command_sink = None
        self._stop_sink = None

    def update_state(self, base_from_gripper, timestamp_s=None):
        timestamp_s = time.monotonic() if timestamp_s is None else float(timestamp_s)
        self._state = RobotState(
            True,
            as_transform(base_from_gripper, "base_from_gripper"),
            timestamp_s,
            simulated=False,
            reason="canonical robot pose topic",
        )

    def set_command_sinks(self, command_sink=None, stop_sink=None):
        self._command_sink = command_sink
        self._stop_sink = stop_sink

    def read_state(self, now_s=None):
        return self._state

    def move_to(self, base_from_gripper, speed_scale=0.1):
        if not self.allow_motion:
            raise SafetyInterlockError("robot motion is disabled by system_parameters.yaml")
        if self._command_sink is None:
            raise SafetyInterlockError("robot command topic is not connected")
        return self._command_sink(
            as_transform(base_from_gripper, "base_from_gripper"),
            float(speed_scale),
        )

    def move_j(self, points, speed_scale=0.1):
        """Forward a validated MOVJ point list to an optional vendor bridge."""
        if not self.allow_motion:
            raise SafetyInterlockError("robot motion is disabled by system_parameters.yaml")
        sink = getattr(self, "_move_j_sink", None)
        if sink is None:
            raise SafetyInterlockError("MOVJ command bridge is not connected")
        return sink(tuple(points), float(speed_scale))

    def move_l(self, points, speed_mm_s=30.0):
        """Forward a validated MOVL point list to an optional vendor bridge."""
        if not self.allow_motion:
            raise SafetyInterlockError("robot motion is disabled by system_parameters.yaml")
        sink = getattr(self, "_move_l_sink", None)
        if sink is None:
            raise SafetyInterlockError("MOVL command bridge is not connected")
        return sink(tuple(points), float(speed_mm_s))

    def set_motion_sinks(self, move_j_sink=None, move_l_sink=None):
        self._move_j_sink = move_j_sink
        self._move_l_sink = move_l_sink

    def stop(self):
        if self._stop_sink is not None:
            return self._stop_sink()
        return True
