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

``speed_scale`` 的量纲（这个名字在三处含义不同，看清楚再改）
-----------------------------------------------------------
本模块与 ``SafeRobotController.move_tcp`` 里的 ``speed_scale`` 是
**无量纲比例**，下游 ``NexBotTcpRobotController.move_to`` 把它乘 1000 得到
MOVL 的 ``vel``(mm/s)，并夹到 [1, 1000]：

    speed_scale 0.05 -> 50 mm/s      speed_scale 0.2 -> 200 mm/s

注意同名参数在 ``move_j`` 里是 **×100 的百分比**（0.1 -> vel=10%），
而 ``move_l`` 的 ``speed_mm_s`` 直接就是 mm/s。三者不可互相套用：
把一个"给 move_j 调好的 0.5"塞进 move_tcp 会变成 500 mm/s。

``safety.maximum_speed_scale``（默认 0.2 = 200 mm/s）只约束本条路径。
"""

import time

from .interfaces import RobotPoseSample
from .nexbot_tcp import NexBotTcpRobotController


class NexBotTcpMoveController:
    """Pipeline adapter: T_base_tcp matrices in, formal NexBot protocol out."""

    def __init__(self, controller: NexBotTcpRobotController):
        self._controller = controller

    def move_tcp(self, base_from_tcp, speed_scale):
        """一段直线 MOVL。``speed_scale`` 无量纲，×1000 = mm/s（见模块 docstring）。

        伺服使能前置条件由适配器的 ``_ensure_servo_enabled`` 保证：它挂在
        ``move_l`` 上，本方法经 ``move_to`` 调用它，因此自动继承。2026-08-22
        之前这条路径完全没有使能，只能蹭 ``NexBotTcpJog.step`` 残留的使能。
        """
        self._controller.move_to(base_from_tcp, speed_scale)

    def latest_pose(self):
        state = self._controller.read_state()
        return RobotPoseSample(state.base_from_gripper, time.monotonic())

    def stop(self):
        self._controller.stop()

    def close(self):
        self._controller.close()


__all__ = ["NexBotTcpMoveController"]
