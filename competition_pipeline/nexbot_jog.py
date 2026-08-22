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

import threading
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
    """One controller instance used for jog/gripper/home/reset actions.

    6001 是单客户端端口：旧连接被控制器释放前，新连接会 [Errno 111] 拒绝。
    ``_run`` 对每个动作做 关闭旧连接 -> 等待 -> 重连重试（运动为绝对/幂等，
    夹爪重试重发 15/16 全序列，不会出现只写一半的半开阀状态）。
    """

    #: 重连尝试次数与间隔（6001 槽位释放延迟通常 < 2s）
    MAX_RETRIES = 3
    RETRY_WAIT_S = 1.2

    def __init__(self, endpoint: NexBotTcpEndpoint, keepalive_s=None):
        """keepalive_s: 非空时启动持久保活线程（6001 单客户端：占住槽位，
        让控制器保持"已连接"；掉线后由 `_run` 自动重连）。"""
        self.endpoint = endpoint
        self._controller = None
        self._lock = threading.Lock()
        self._keepalive_s = keepalive_s
        self._keepalive_stop = threading.Event()
        self._keepalive_thread = None
        if keepalive_s is not None:
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, daemon=True,
                name="nexbot-keepalive",
            )
            self._keepalive_thread.start()

    def _keepalive_loop(self):
        while not self._keepalive_stop.wait(self._keepalive_s):
            try:
                self._run(self.controller.servo_status)
            except Exception:
                # 连接断了：关闭控制器，下一次动作自动重连
                self.close()

    @property
    def controller(self):
        if self._controller is None:
            self._controller = NexBotTcpRobotController(self.endpoint)
        return self._controller

    def _run(self, action, *args, **kwargs):
        with self._lock:
            return self._run_locked(action, *args, **kwargs)

    def _run_locked(self, action, *args, **kwargs):
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return action(*args, **kwargs)
            except ControllerConnectionError as error:
                last_error = error
                self.close()
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_WAIT_S * attempt)
        raise last_error

    def close(self):
        self._keepalive_stop.set()
        if self._controller is not None:
            self._controller.close()
            self._controller = None

    def current_pose(self):
        """(xyz_mm, abc_deg) 当前 TCP 位姿（用户坐标系1）。"""
        state = self._run(self.controller.read_state)
        xyz_m, abc_rad = inexbot_abc_from_transform(state.base_from_gripper)
        return tuple(xyz_m * 1000.0), tuple(np.degrees(abc_rad))

    def step(self, axis: int, step_mm: float):
        """在用户坐标系1 中沿 axis(0=X,1=Y,2=Z) 平移 step_mm（可负）。

        绝对运动：当前位姿 + 轴偏移 -> MOVL(coord=用户坐标)。小步长专为
        坐标核对设计，速度按 JOG_SPEED_SCALE。自动确保伺服上电（0x2311）。
        6001 为单客户端端口，连接被抢占时重连重试一次（绝对运动幂等）。
        """
        def _do_step():
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
        self._run(_do_step)

    def gripper(self, open_: bool):
        """开/关夹爪：开=(15,16)=(0,1) 关=(15,16)=(1,0)。"""
        def _do_gripper():
            controller = self.controller
            if open_:
                controller.set_digital_output(GRIPPER_PORT_CLOSE, 0)
                controller.set_digital_output(GRIPPER_PORT_OPEN, 1)
            else:
                controller.set_digital_output(GRIPPER_PORT_CLOSE, 1)
                controller.set_digital_output(GRIPPER_PORT_OPEN, 0)
        self._run(_do_gripper)

    def gripper_state(self):
        """返回 (DOUT15, DOUT16) 当前状态。"""
        status = self._run(self.controller.digital_output_states)
        if len(status) < 16:
            raise ValueError(
                "DOUT status has {} entries; expected >= 16".format(len(status))
            )
        return status[GRIPPER_PORT_CLOSE - 1], status[GRIPPER_PORT_OPEN - 1]

    def go_home(self):
        self._run(self.controller.go_home)

    def go_reset_position(self):
        self._run(self.controller.go_reset_position)

    def emergency_stop(self):
        self._run(self.controller.stop)


__all__ = [
    "GRIPPER_PORT_CLOSE",
    "GRIPPER_PORT_OPEN",
    "JOG_SPEED_SCALE",
    "NexBotTcpJog",
]
