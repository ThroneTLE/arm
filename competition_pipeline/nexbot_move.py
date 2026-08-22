"""Pipeline RobotController contract over the formal NexBot TCP adapter.

The wire protocol (frame, CRC, MOVJ/MOVL command words) is owned by the
formal ROS adapter ``arm_vision_framework.adapters.nexbot_tcp``; this module
only maps the pipeline's ``T_base_tcp`` matrix / ``speed_scale`` vocabulary
onto the formal ``RobotController`` interface, and is the bridge
``SafeRobotController`` + ``GraspExecutor`` expect.

``move_tcp`` always reaches the controller as a single Cartesian MOVL
(``0x4502``); the executor already splits a path into <=40 mm segments, so
each call is a short straight-line move.  Every motion safety gate stays with
``SafeRobotController`` (dry-run, workspace, freshness, step limits).

Timestamps follow the pipeline convention (``time.monotonic``), which the
freshness gate in ``SafeRobotController`` compares against.
"""

import time

from .interfaces import RobotPoseSample
from .nexbot_tcp import NexBotTcpRobotController


class NexBotTcpMoveController:
    """Pipeline adapter: T_base_tcp matrices in, formal NexBot protocol out."""

    def __init__(self, controller: NexBotTcpRobotController):
        self._controller = controller

    def move_tcp(self, base_from_tcp, speed_scale):
        self._controller.move_to(base_from_tcp, speed_scale)

    def latest_pose(self):
        state = self._controller.read_state()
        return RobotPoseSample(state.base_from_gripper, time.monotonic())

    def stop(self):
        self._controller.stop()

    def close(self):
        self._controller.close()


__all__ = ["NexBotTcpMoveController"]
