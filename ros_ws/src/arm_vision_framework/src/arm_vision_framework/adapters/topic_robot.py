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

    def stop(self):
        if self._stop_sink is not None:
            return self._stop_sink()
        return True
