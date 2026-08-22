"""Qt-free robot jog + gripper controls for the competition UI (NexBot).

Wraps ``NexBotTcpRobotController`` (official RTL-22.07 protocol) with the
teach-pendant style actions the verification page needs:

- 六方向步进（用户坐标系1，安全小步长）: 前进/后退/左/右/上/下；
- 夹爪开/关（DOUT16/DOUT15 双线圈，现场实测映射 (15,16)=(1,0)关 <-> (0,1)开）；
- 回零（0x3002 GO_HOME）与回复位点（0x3007 GO_RESET_POSITION，现场约定
  复位点=拍摄点）；
- 急停（0x2314）。

All poses are read/written in the active user coordinate frame (用户坐标系1),
so Y/X/Z values match the teach pendant's user-coordinate display.
"""

import time

import numpy as np

from .geometry import as_transform, inexbot_abc_from_transform
from .nexbot_tcp import (
    NexBotTcpEndpoint,
    NexBotTcpRobotController,
    ControllerConnectionError,
)

#: 现场实测 (2026-08-22): DOUT15/16 = 双线圈气阀; (15,16)=(1,0) 关闭 <-> (0,1) 打开
GRIPPER_PORT_CLOSE = 15
GRIPPER_PORT_OPEN = 16

#: 步进默认速度（速度比例 -> move_to 内部 mm/s = speed_scale*1000）
JOG_SPEED_SCALE = 0.05  # 50 mm/s，安全小步进


class NexBotTcpJog:
    """One controller instance used for jog/gripper/home/reset actions."""

    def __init__(self, endpoint: NexBotTcpEndpoint):
        self.endpoint = endpoint
        self._controller = None

    @property
    def controller(self):
        if self._controller is None:
            self._controller = NexBotTcpRobotController(self.endpoint)
        return self._controller

    def close(self):
        if self._controller is not None:
            self._controller.close()
            self._controller = None

    def current_pose(self):
        """(xyz_mm, abc_deg) 当前 TCP 位姿（用户坐标系1）。"""
        state = self.controller.read_state()
        xyz_m, abc_rad = inexbot_abc_from_transform(state.base_from_gripper)
        return tuple(xyz_m * 1000.0), tuple(np.degrees(abc_rad))

    def step(self, axis: int, step_mm: float):
        """在用户坐标系1 中沿 axis(0=X,1=Y,2=Z) 平移 step_mm（可负）。

        绝对运动：当前位姿 + 轴偏移 -> MOVL(coord=用户坐标)。小步长专为
        坐标核对设计，速度按 JOG_SPEED_SCALE。自动确保伺服上电（0x2311）。
        6001 为单客户端端口，连接被抢占时重连重试一次（绝对运动幂等）。
        """
        last_error = None
        for _attempt in (1, 2):
            try:
                controller = self.controller
                try:
                    if controller.servo_status() != 3:
                        controller.enable_servo()
                except Exception:
                    pass
                state = controller.read_state()
                matrix = as_transform(state.base_from_gripper, "world_from_gripper")
                delta = np.zeros(3, dtype=np.float64)
                delta[int(axis)] = float(step_mm) / 1000.0
                target = np.eye(4, dtype=np.float64)
                target[:3, :3] = matrix[:3, :3]
                target[:3, 3] = matrix[:3, 3] + delta
                controller.move_to(target, speed_scale=JOG_SPEED_SCALE)
                return
            except ControllerConnectionError as error:
                last_error = error
                self.close()
                time.sleep(0.5)
        raise last_error

    def gripper(self, open_: bool):
        """开/关夹爪：开=(15,16)=(0,1) 关=(15,16)=(1,0)。"""
        controller = self.controller
        if open_:
            controller.set_digital_output(GRIPPER_PORT_CLOSE, 0)
            controller.set_digital_output(GRIPPER_PORT_OPEN, 1)
        else:
            controller.set_digital_output(GRIPPER_PORT_CLOSE, 1)
            controller.set_digital_output(GRIPPER_PORT_OPEN, 0)

    def gripper_state(self):
        """返回 (DOUT15, DOUT16) 当前状态。"""
        status = self.controller.digital_output_states()
        if len(status) < 16:
            raise ValueError(
                "DOUT status has {} entries; expected >= 16".format(len(status))
            )
        return status[GRIPPER_PORT_CLOSE - 1], status[GRIPPER_PORT_OPEN - 1]

    def go_home(self):
        self.controller.go_home()

    def go_reset_position(self):
        self.controller.go_reset_position()

    def emergency_stop(self):
        self.controller.stop()


__all__ = [
    "GRIPPER_PORT_CLOSE",
    "GRIPPER_PORT_OPEN",
    "JOG_SPEED_SCALE",
    "NexBotTcpJog",
]
