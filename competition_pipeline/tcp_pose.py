"""Live TCP pose source for the competition UI (NexBot JSON-over-TCP).

Wraps the formal ``NexBotTcpRobotController`` (``ros_ws/.../adapters/nexbot_tcp.py``,
official RTL-22.07 protocol, 7000-port ``0x9512`` state service) so the
hand-eye and localization pages can read the current ``T_base_tcp`` directly
from the controller instead of transcribing teach-pendant values by hand.

This module is Qt-free: the UI thread worker lives in ``ui.py`` and only this
thin boundary (endpoint, read, close) is covered by offline tests.
"""

from .geometry import xyz_rpy_from_transform
from .nexbot_tcp import (
    NexBotTcpEndpoint,
    NexBotTcpRobotController,
)


def pose_endpoint_from_config(controller_settings):
    """Build a ``NexBotTcpEndpoint`` from the ``controller.nexbot_tcp`` section.

    ``host`` stays optional here (the UI fills it live and saves it back);
    the endpoint constructor reports a missing host on connect, which the UI
    surfaces as a verification error.
    """
    config = dict(controller_settings or {}).get("nexbot_tcp", {}) or {}
    return NexBotTcpEndpoint(
        host=str(config.get("host", "")),
        port_motion=int(config.get("port_motion", 6000)),
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
        heartbeat_s=float(config.get("heartbeat_s", 0.0)),
    )


class NexBotTcpPoseSource:
    """One-shot state reads as ``(xyz_mm, rpy_deg)`` tuples for the UI."""

    def __init__(self, endpoint: NexBotTcpEndpoint):
        self.endpoint = endpoint
        self._controller = None

    @property
    def controller(self):
        return self._controller

    def connect(self):
        if self._controller is None:
            self._controller = NexBotTcpRobotController(self.endpoint)
        return self

    def read(self):
        """Return ``(xyz_mm, rpy_deg)`` of the current controller TCP pose."""
        controller = self.connect().controller
        state = controller.read_state()
        xyz_m, rpy_deg = xyz_rpy_from_transform(state.base_from_gripper)
        return tuple(xyz_m * 1000.0), tuple(rpy_deg)

    def close(self):
        if self._controller is not None:
            self._controller.close()
            self._controller = None


__all__ = ["NexBotTcpPoseSource", "pose_endpoint_from_config"]
