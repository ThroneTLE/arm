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
        self._controller_lock = threading.RLock()
        self._lock = threading.Lock()
        self._estop_lock = threading.Lock()
        self._closed = False
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
            if not self._lock.acquire(blocking=False):
                continue
            try:
                self.controller.servo_status()
            except Exception:
                self._drop_controller()
            finally:
                self._lock.release()

    @property
    def controller(self):
        with self._controller_lock:
            if self._closed:
                raise ControllerConnectionError("NexBot jog controller is closed")
            if self._controller is None:
                self._controller = NexBotTcpRobotController(self.endpoint)
            return self._controller

    def _run(self, action, *args, **kwargs):
        """Run ``action(controller, ...)`` as one serialized transaction."""
        with self._lock:
            return self._run_locked(action, *args, **kwargs)

    def _run_locked(self, action, *args, **kwargs):
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            if self._closed:
                raise ControllerConnectionError("NexBot jog controller is closed")
            try:
                return action(self.controller, *args, **kwargs)
            except ControllerConnectionError as error:
                last_error = error
                self._drop_controller()
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_WAIT_S * attempt)
        raise last_error

    def _drop_controller(self):
        """Discard a failed connection without stopping keepalive."""
        with self._controller_lock:
            controller = self._controller
            self._controller = None
            # The controller accepts one client per port.  Release the old
            # slot before controller() may construct its replacement.
            if controller is not None:
                controller.close()

    def close(self):
        """Permanently stop keepalive and release the shared connections."""
        with self._controller_lock:
            if self._closed:
                return
            self._closed = True
        self._keepalive_stop.set()
        self._drop_controller()

    def current_pose(self):
        """(xyz_mm, abc_deg) 当前 TCP 位姿（用户坐标系1）。"""
        state = self._run(lambda controller: controller.read_state())
        xyz_m, abc_rad = inexbot_abc_from_transform(state.base_from_gripper)
        return tuple(xyz_m * 1000.0), tuple(np.degrees(abc_rad))

    def step(self, axis: int, step_mm: float):
        """在用户坐标系1 中沿 axis(0=X,1=Y,2=Z) 平移 step_mm（可负）。

        绝对运动：当前位姿 + 轴偏移 -> MOVL(coord=用户坐标)。小步长专为
        坐标核对设计，速度按 JOG_SPEED_SCALE。自动确保伺服上电（0x2311）。
        6001 为单客户端端口，连接被抢占时重连重试一次（绝对运动幂等）。
        """
        axis = int(axis)
        if axis not in (0, 1, 2):
            raise ValueError("jog axis must be 0, 1 or 2")
        step_mm = float(step_mm)
        if not np.isfinite(step_mm) or step_mm == 0.0:
            raise ValueError("jog step must be a finite non-zero distance")

        # Build the absolute destination once.  An ambiguous disconnect after
        # MOVL therefore resends the same target rather than adding the step
        # again from an already-advanced pose.
        with self._lock:
            status = self._run_locked(lambda controller: controller.servo_status())
            if status != 3:
                self._run_locked(lambda controller: controller.enable_servo())
            state = self._run_locked(lambda controller: controller.read_state())
            matrix = as_transform(state.base_from_gripper, "world_from_gripper")
            delta = np.zeros(3, dtype=np.float64)
            delta[axis] = step_mm / 1000.0
            target = np.eye(4, dtype=np.float64)
            target[:3, :3] = matrix[:3, :3]
            target[:3, 3] = matrix[:3, 3] + delta
            self._run_locked(
                lambda controller: controller.move_to(
                    target, speed_scale=JOG_SPEED_SCALE
                )
            )

    def move_to_ucs(self, xyz_mm, abc_rad, vel_mm_s=50.0, tolerance_mm=1.0):
        """用户坐标系1 绝对传送（与 0x45xx 实测链路一致）并核验到位。

        通过共享持久连接/_run 事务锁执行（不与步进/夹爪/位姿轮询抢线）；
        返回到达偏差 mm；超出 tolerance_mm 抛 RuntimeError。
        """
        from .geometry import transform_from_inexbot_abc

        def _read(c):
            return c.read_state()

        start_state = self._run(_read)
        target = transform_from_inexbot_abc(
            np.asarray(xyz_mm, dtype=float) / 1000.0,
            np.asarray(abc_rad, dtype=float),
        )
        start_xyz = start_state.base_from_gripper[:3, 3] * 1000.0
        distance = float(np.linalg.norm(
            np.asarray(xyz_mm, dtype=float) - start_xyz))
        if distance > 400.0:
            raise RuntimeError(
                "目标距当前 {:.0f} mm > 400mm：疑似未确认输入（默认 0,0,0）或目标超工作范围；"
                "请先【回读当前】核对（当前 {:.2f},{:.2f},{:.2f}）".format(
                    distance, *start_xyz)
            )

        def _go(c):
            c.move_to(target, speed_scale=vel_mm_s / 1000.0)

        # 竞态修复：控制器收到指令后有百毫秒级延迟才开始动，"静止判停"可能
        # 在启动前误判完成。改为"等到达目标"：|当前-目标|<容差 连续 2 次
        # 采样（0.25s 间隔）才算到位，上限 20s。
        self._run(_go)
        deadline = time.monotonic() + 20.0
        reached = 0
        final_state = None
        while time.monotonic() < deadline:
            final_state = self._run(_read)
            final_xyz = final_state.base_from_gripper[:3, 3] * 1000.0
            dist = float(np.linalg.norm(
                np.asarray(xyz_mm, dtype=float) - final_xyz))
            reached = reached + 1 if dist <= tolerance_mm else 0
            if reached >= 2:
                break
            time.sleep(0.25)
        if reached < 2:
            raise RuntimeError(
                "未在 20s 内到达目标（当前 {:.2f},{:.2f},{:.2f}，目标 {:.2f},{:.2f},{:.2f}）".format(
                    *final_xyz, *xyz_mm)
            )
        final_xyz = final_state.base_from_gripper[:3, 3] * 1000.0
        deviation = float(np.linalg.norm(
            np.asarray(xyz_mm, dtype=float) - final_xyz))
        if deviation > tolerance_mm:
            raise RuntimeError(
                "到达偏差 {:.2f} mm 超出容差 {:.1f} mm（起点 {:.2f},{:.2f},{:.2f}）".format(
                    deviation, tolerance_mm, *start_xyz)
            )
        return deviation

    def gripper(self, open_: bool):
        """开/关夹爪：开=(15,16)=(0,1) 关=(15,16)=(1,0)。"""
        def _do_gripper(controller):
            if open_:
                controller.set_digital_output(GRIPPER_PORT_CLOSE, 0)
                controller.set_digital_output(GRIPPER_PORT_OPEN, 1)
            else:
                controller.set_digital_output(GRIPPER_PORT_CLOSE, 1)
                controller.set_digital_output(GRIPPER_PORT_OPEN, 0)
        self._run(_do_gripper)

    def gripper_state(self):
        """返回 (DOUT15, DOUT16) 当前状态。"""
        status = self._run(lambda controller: controller.digital_output_states())
        if len(status) < 16:
            raise ValueError(
                "DOUT status has {} entries; expected >= 16".format(len(status))
            )
        return status[GRIPPER_PORT_CLOSE - 1], status[GRIPPER_PORT_OPEN - 1]

    def go_home(self):
        self._run(lambda controller: controller.go_home())

    def go_reset_position(self):
        self._run(lambda controller: controller.go_reset_position())

    def emergency_stop(self):
        """Send 0x2314 without waiting behind an active normal operation."""
        with self._estop_lock:
            last_error = None
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    self.controller.stop()
                    return
                except ControllerConnectionError as error:
                    last_error = error
                    # Leave the shared state port intact for the active motion
                    # worker; reopen only the failed 6001 transport.
                    try:
                        self.controller.motion.close()
                    except Exception:
                        pass
                    if attempt < self.MAX_RETRIES:
                        time.sleep(self.RETRY_WAIT_S * attempt)
            raise last_error


__all__ = [
    "GRIPPER_PORT_CLOSE",
    "GRIPPER_PORT_OPEN",
    "JOG_SPEED_SCALE",
    "NexBotTcpJog",
]
