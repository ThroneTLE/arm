"""Live TCP pose source for the competition UI (NexBot JSON-over-TCP).

Wraps the formal ``NexBotTcpRobotController`` (``ros_ws/.../adapters/nexbot_tcp.py``,
official RTL-22.07 protocol, 7000-port ``0x9512`` state service) so the
hand-eye and localization pages can read the current ``T_base_tcp`` directly
from the controller instead of transcribing teach-pendant values by hand.

This module is Qt-free: the UI thread worker lives in ``ui.py`` and only this
thin boundary (endpoint, read, close) is covered by offline tests.
"""

from .geometry import inexbot_abc_from_transform
from .nexbot_tcp import (
    NexBotTcpEndpoint,
    NexBotTcpRobotController,
)

import numpy as np


def pose_endpoint_from_config(controller_settings):
    """Build a ``NexBotTcpEndpoint`` from the ``controller.nexbot_tcp`` section.

    ``host`` stays optional here (the UI fills it live and saves it back);
    the endpoint constructor reports a missing host on connect, which the UI
    surfaces as a verification error.
    """
    config = dict(controller_settings or {}).get("nexbot_tcp", {}) or {}
    return NexBotTcpEndpoint(
        host=str(config.get("host", "")),
        # 6001, NOT 6000.  Field-verified on the MOKA MR07S-930 / Inexbot C1102:
        # 6000 never opens, 6001 is the real-time command port.  This default
        # used to be 6000 while ``nexbot_tcp_client_from_config`` already said
        # 6001 -- a config that omitted ``port_motion`` silently talked to a
        # dead port here and to the right one there.
        port_motion=int(config.get("port_motion", 6001)),
        port_state=int(config.get("port_state", 7000)),
        robot=int(config.get("robot", 1)),
        channel=int(config.get("channel", 1)),
        connect_timeout_s=float(config.get("connect_timeout_s", 2.0)),
        io_timeout_s=float(config.get("io_timeout_s", 1.0)),
        keepalive=bool(config.get("keepalive", True)),
        max_frame_bytes=int(config.get("max_frame_bytes", 1024 * 1024)),
        external_axes=int(config.get("external_axes", 0)),
        wait_for_finish=bool(config.get("wait_for_finish", True)),
        motion_finish_timeout_s=float(config.get("motion_finish_timeout_s", 60.0)),
        velocity_eps_rad_s=float(config.get("velocity_eps_rad_s", 0.02)),
        # heartbeat_s 必须保持 0：0x7266 心跳线程会在 MOVL 事务中间插帧,
        # 现场实测会让示教器同时报 0x4502 和 0x7266 错误。
        heartbeat_s=float(config.get("heartbeat_s", 0.0)),
        # 整条流水线以用户坐标系1 为基准：pose_frame=UCS、motion_coord=3(用户)。
        # 旧默认 PCS/1(直角) 与 competition.yaml 里写死的值不一致，配置缺项时
        # 会读到工具系位姿却按直角系发运动。
        pose_frame=str(config.get("pose_frame", "UCS")),
        motion_coord=int(config.get("motion_coord", 3)),
        tool_id=int(config.get("tool_id", 1)),
        user_id=int(config.get("user_id", 1)),
        motion_ack_timeout_s=float(config.get("motion_ack_timeout_s", 3.0)),
    )


class NexBotTcpPoseSource:
    """One-shot state reads as ``(xyz_mm, rpy_deg)`` tuples for the UI.

    ``controller`` may be shared (one persistent 7000 connection owned by the
    robot panel) so the pose poller and the jog actions never fight over the
    single-client 7000/6001 ports; the adapter transports are lock-protected.
    """

    def __init__(self, endpoint: NexBotTcpEndpoint, jog: object = None):
        """Share the jog's current controller and transaction lock."""
        self.endpoint = endpoint
        self._jog = jog
        self._controller = None
        self._owns_controller = jog is None

    @property
    def controller(self):
        return self._controller

    def connect(self):
        if self._controller is None:
            self._controller = NexBotTcpRobotController(self.endpoint)
        return self

    def read(self):
        """Return ``(xyz_mm, abc_deg)`` of the current controller TCP pose.

        Angles are the controller-native A/B/C (intrinsic X'Y'Z'), so the
        numbers match the teach pendant's tool-coordinate display -- this is
        the verification step before every hand-eye sample.
        """
        if self._jog is not None:
            return self._jog._run(
                lambda controller: self._read_controller(controller)
            )
        return self._read_controller(self.connect().controller)

    def try_read(self):
        """Read without queueing behind motion; return ``None`` while busy."""
        if self._jog is None:
            return self.read()
        if not self._jog._lock.acquire(blocking=False):
            return None
        try:
            return self._jog._run_locked(
                lambda controller: self._read_controller(controller)
            )
        finally:
            self._jog._lock.release()

    @staticmethod
    def _read_controller(controller):
        state = controller.read_state()
        xyz_m, abc_rad = inexbot_abc_from_transform(state.base_from_gripper)
        return tuple(xyz_m * 1000.0), tuple(np.degrees(abc_rad))

    def close(self):
        if self._owns_controller and self._controller is not None:
            self._controller.close()
            self._controller = None


__all__ = ["NexBotTcpPoseSource", "pose_endpoint_from_config"]
