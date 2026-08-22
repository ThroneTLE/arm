"""NexBot (纳博特) JSON-over-TCP protocol adapter (RTL-22.07).

Official protocol: open.inexbot.com SDK docs (RTL-22.07) and doc.inexbot.com
knowledge base, summarized in ``docs/纳博特通讯协议.md``.  Wire frame::

    0x4E66 | Length u16 BE | Command u16 BE | JSON data | CRC32 u32 BE

CRC32 covers Length+Command+data and is sent big-endian; verified against the
official example (``zlib.crc32`` of the documented query frame equals
``0x9090FA38``).  Port 6001 is the real-time command channel (field-tested on
the MOKA MR07S-930 / Inexbot C1102: 6000 stays closed, 6001 answers; MOVJ
``0x4501``, MOVL ``0x4502``, MOVC ``0x4503``, MOVS ``0x4504``, GO_HOME
``0x3002``, emergency stop ``0x2314``, DOUT set ``0x3601``); port 7000 is the
host-computer state service (``0x9512`` query / ``0x9513`` reply).

This adapter implements the framework ``RobotController`` boundary only.
Every motion safety gate (dry-run, workspace, segmentation, pose freshness)
stays with the caller; this class never moves the robot on its own.

Field-verification items (documented, not guessed): the exact ``pos`` array
length (official text says 7 body slots + 5 external-axis slots while the
example shows 7; ``external_axes`` selects), joint angle units (``joint_unit``),
and whether ``0x4501/0x4502`` accept Cartesian targets without a shape field.
"""

import json
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import RobotController
from ..transforms import (
    as_transform,
    inexbot_abc_from_transform,
    transform_from_inexbot_abc,
    transform_from_xyz_rpy,
    xyz_rpy_from_transform,
)
from ..types import RobotState
from .inexbot_modbus import (
    ControllerConnectionError,
    ControllerProtocolError,
    ControllerTimeout,
    InexbotPoint,
)

SYNC = b"\x4E\x66"
CMD_HEARTBEAT = 0x7266
CMD_HEARTBEAT_REPLY = 0x7267
CMD_MOVJ = 0x4501
CMD_MOVL = 0x4502
CMD_EMERGENCY_STOP = 0x2314
CMD_QUERY = 0x9512
CMD_QUERY_REPLY = 0x9513
CMD_ERRORS = frozenset({0x6010, 0x6020, 0x6030, 0x6040})
CMD_WARNINGS = frozenset({0x6110, 0x6210})

#: Coord value used by the protocol for joint / Cartesian coordinate systems.
COORD_JOINT = 0
COORD_CARTESIAN = 1


def build_frame(command: int, data: Optional[Dict[str, Any]] = None) -> bytes:
    """Build one wire frame (0x4E66 + length + command + JSON + CRC32)."""
    payload = (
        b""
        if data is None
        else json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    if len(payload) > 0xFFFF:
        raise ControllerProtocolError("frame data exceeds 0xFFFF bytes")
    length = struct.pack(">H", len(payload))
    command_bytes = struct.pack(">H", int(command))
    crc = zlib.crc32(length + command_bytes + payload) & 0xFFFFFFFF
    return SYNC + length + command_bytes + payload + struct.pack(">I", crc)


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ControllerConnectionError("connection closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket, max_frame_bytes: int) -> Tuple[int, Any]:
    """Read and validate one frame; returns ``(command, parsed_data)``."""
    sync = _recv_exact(sock, 2)
    if sync != SYNC:
        raise ControllerProtocolError(
            "frame sync mismatch: expected 0x4E66, got 0x{}".format(sync.hex())
        )
    length = struct.unpack(">H", _recv_exact(sock, 2))[0]
    if length > max_frame_bytes:
        raise ControllerProtocolError(
            "frame data length {} exceeds limit {}".format(length, max_frame_bytes)
        )
    command = struct.unpack(">H", _recv_exact(sock, 2))[0]
    payload = _recv_exact(sock, length)
    crc = struct.unpack(">I", _recv_exact(sock, 4))[0]
    expected = (
        zlib.crc32(struct.pack(">H", length) + struct.pack(">H", command) + payload)
        & 0xFFFFFFFF
    )
    if crc != expected:
        raise ControllerProtocolError(
            "frame CRC mismatch: got 0x{:08X}, expected 0x{:08X}".format(crc, expected)
        )
    parsed = None
    if payload:
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except ValueError as error:
            raise ControllerProtocolError(
                "frame data is not valid JSON: {}".format(error)
            ) from error
    return command, parsed


@dataclass(frozen=True)
class NexBotTcpEndpoint:
    """Controller endpoint facts for the 6001 (motion) and 7000 (state) ports.

    Field-tested on the MOKA MR07S-930 (Inexbot C1102, RTL-22.07): the
    real-time command port answers on **6001**, not 6000 (6000 stays closed;
    verified 2026-08-22).
    """

    host: str
    port_motion: int = 6001
    port_state: int = 7000
    robot: int = 1
    channel: int = 1
    connect_timeout_s: float = 2.0
    io_timeout_s: float = 1.0
    keepalive: bool = True
    max_frame_bytes: int = 1024 * 1024
    external_axes: int = 0
    wait_for_finish: bool = True
    motion_finish_timeout_s: float = 60.0
    velocity_eps_rad_s: float = 0.02
    heartbeat_s: float = 0.0

    def __post_init__(self):
        host = str(self.host).strip()
        if not host:
            raise ValueError("NexBot TCP host is required")
        object.__setattr__(self, "host", host)
        for name in ("port_motion", "port_state"):
            value = int(getattr(self, name))
            if not 1 <= value <= 65535:
                raise ValueError("{} must be within 1..65535".format(name))
            object.__setattr__(self, name, value)
        robot = int(self.robot)
        if not 1 <= robot <= 4:
            raise ValueError("robot must be within 1..4")
        object.__setattr__(self, "robot", robot)
        channel = int(self.channel)
        if not 1 <= channel <= 9:
            raise ValueError("channel must be within 1..9")
        object.__setattr__(self, "channel", channel)
        for name in ("connect_timeout_s", "io_timeout_s", "motion_finish_timeout_s"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError("{} must be positive".format(name))
            object.__setattr__(self, name, value)
        external_axes = int(self.external_axes)
        if external_axes not in (0, 5):
            raise ValueError("external_axes must be 0 (7-slot body array) or 5 (12-slot array)")
        object.__setattr__(self, "external_axes", external_axes)
        frame_limit = int(self.max_frame_bytes)
        if not 8 <= frame_limit <= 16 * 1024 * 1024:
            raise ValueError("max_frame_bytes must be within 8..16777216")
        object.__setattr__(self, "max_frame_bytes", frame_limit)
        heartbeat = float(self.heartbeat_s)
        if heartbeat < 0.0:
            raise ValueError("heartbeat_s cannot be negative")
        object.__setattr__(self, "heartbeat_s", heartbeat)


class NexBotTcpTransport:
    """One blocking, thread-safe TCP transport for a single controller port."""

    def __init__(self, endpoint: NexBotTcpEndpoint, port: int, socket_factory=None):
        self.endpoint = endpoint
        self.port = int(port)
        self._socket_factory = socket_factory or socket.create_connection
        self._socket = None
        self._lock = threading.RLock()

    @property
    def connected(self):
        return self._socket is not None

    def connect(self):
        with self._lock:
            if self._socket is not None:
                return self
            try:
                sock = self._socket_factory(
                    (self.endpoint.host, self.port),
                    timeout=self.endpoint.connect_timeout_s,
                )
            except OSError as error:
                raise ControllerConnectionError(
                    "could not connect to {}:{}: {}".format(
                        self.endpoint.host, self.port, error
                    )
                ) from error
            sock.settimeout(self.endpoint.io_timeout_s)
            if self.endpoint.keepalive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self._socket = sock
            return self

    def send_frame(self, command: int, data: Optional[Dict[str, Any]] = None):
        with self._lock:
            self.connect()
            try:
                self._socket.sendall(build_frame(command, data))
            except OSError as error:
                self.close()
                raise ControllerConnectionError(
                    "send to {}:{} failed: {}".format(
                        self.endpoint.host, self.port, error
                    )
                ) from error

    def read_frame(self, timeout: Optional[float] = None):
        with self._lock:
            self.connect()
            previous = self._socket.gettimeout()
            try:
                if timeout is not None:
                    self._socket.settimeout(timeout)
                return read_frame(self._socket, self.endpoint.max_frame_bytes)
            except socket.timeout as error:
                raise ControllerTimeout(
                    "no frame from {}:{} within {:.3f}s".format(
                        self.endpoint.host, self.port,
                        self._socket.gettimeout() if timeout is not None else self.endpoint.io_timeout_s,
                    )
                ) from error
            except OSError as error:
                self.close()
                raise ControllerConnectionError(
                    "read from {}:{} failed: {}".format(
                        self.endpoint.host, self.port, error
                    )
                ) from error
            finally:
                try:
                    self._socket.settimeout(previous)
                except OSError:
                    pass

    def close(self):
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                except OSError:
                    pass
                self._socket = None


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


class NexBotTcpRobotController(RobotController):
    """Framework ``RobotController`` over the NexBot 6001/7000 ports.

    Motion commands go to port 6001 (MOVJ ``0x4501``, MOVL ``0x4502``,
    emergency stop ``0x2314``); state queries go to port 7000 (``0x9512``).
    Both connections are lazy and guarded by per-port locks, so one instance
    is safe to share across polling and execution threads.
    """

    def __init__(self, endpoint: NexBotTcpEndpoint, socket_factory=None):
        self.endpoint = endpoint
        self.motion = NexBotTcpTransport(endpoint, endpoint.port_motion, socket_factory)
        self.state = NexBotTcpTransport(endpoint, endpoint.port_state, socket_factory)
        self._closed = False
        self._heartbeat = None
        if endpoint.heartbeat_s > 0.0:
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop, name="nexbot-heartbeat", daemon=True
            )
            self._heartbeat.start()

    def _heartbeat_loop(self):
        while not self._closed:
            try:
                self.motion.send_frame(CMD_HEARTBEAT)
                try:
                    self.motion.read_frame(timeout=min(self.endpoint.io_timeout_s, 0.1))
                except (ControllerTimeout, ControllerProtocolError):
                    pass
            except (ControllerConnectionError, ControllerProtocolError):
                pass
            time.sleep(self.endpoint.heartbeat_s)

    def close(self):
        self._closed = True
        self.motion.close()
        self.state.close()

    # -- helpers -----------------------------------------------------------

    def _check_errors(self):
        """Surface any pending controller error/warning frame without blocking."""
        for _ in range(4):
            try:
                command, data = self.motion.read_frame(
                    timeout=min(self.endpoint.io_timeout_s, 0.05)
                )
            except (ControllerTimeout, ControllerConnectionError):
                return
            if command in CMD_ERRORS or command in CMD_WARNINGS:
                message = "controller error"
                if isinstance(data, dict):
                    message = (
                        str(data.get("message"))
                        or str(data.get("error"))
                        or json.dumps(data, ensure_ascii=False)
                    )
                raise ControllerProtocolError(
                    "controller frame 0x{:04X}: {}".format(command, message)
                )

    def _send_motion(self, command: int, point: InexbotPoint, velocity: int):
        if not isinstance(point, InexbotPoint):
            raise ValueError("NexBot motion requires InexbotPoint, got {!r}".format(type(point)))
        position = [float(value) for value in point.axes]
        if self.endpoint.external_axes:
            position = position + [0.0] * self.endpoint.external_axes
        payload = {
            "robot": self.endpoint.robot,
            "vel": velocity,
            "coord": int(point.coordinate_system),
            "pos": position,
        }
        self.motion.send_frame(command, payload)

    def _request_state(self, query_type: Sequence[str]):
        payload = {
            "channel": self.endpoint.channel,
            "robot": self.endpoint.robot,
            "mode": 0,
            "queryType": list(query_type),
        }
        deadline = time.monotonic() + max(2.0, self.endpoint.io_timeout_s * 4.0)
        self.state.send_frame(CMD_QUERY, payload)
        while time.monotonic() < deadline:
            command, data = self.state.read_frame()
            if command == CMD_QUERY_REPLY:
                return data if isinstance(data, dict) else {}
            if command in CMD_ERRORS or command in CMD_WARNINGS:
                message = "controller error"
                if isinstance(data, dict):
                    message = (
                        str(data.get("message"))
                        or str(data.get("error"))
                        or json.dumps(data, ensure_ascii=False)
                    )
                raise ControllerProtocolError(
                    "controller frame 0x{:04X}: {}".format(command, message)
                )
        raise ControllerTimeout("state query timed out after {:.1f}s".format(deadline))

    def _query_axis_vel(self):
        data = self._request_state(["axisVel"])
        reply = data.get("replyData") or {}
        values = reply.get("axisVel")
        if isinstance(values, list) and len(values) >= 6:
            return tuple(float(value) for value in values[:6])
        return None

    def _wait_motion_finish(self):
        deadline = time.monotonic() + self.endpoint.motion_finish_timeout_s
        quiet = 0
        while time.monotonic() < deadline:
            velocities = self._query_axis_vel()
            if velocities is not None and all(
                abs(value) < self.endpoint.velocity_eps_rad_s for value in velocities
            ):
                quiet += 1
                if quiet >= 2:
                    return
            else:
                quiet = 0
            time.sleep(0.05)
        raise ControllerTimeout(
            "motion did not finish within {:.1f}s".format(
                self.endpoint.motion_finish_timeout_s
            )
        )

    # -- RobotController interface ----------------------------------------

    def read_state(self, now_s=None) -> RobotState:
        # Field-tested on MOKA MR07S-930 / Inexbot C1102 (RTL-22.07): with the
        # calibrated 工具手1 active the state service reports the TCP pose in
        # "realPosPCS" (tool coordinate = TCP w.r.t. robot base), which is the
        # pose the hand-eye calibration samples need.  MCS is kept as fallback
        # (older builds / fake-server tests) and for cross-checking.
        data = self._request_state(["realPosPCS", "realPosMCS", "realPosACS"])
        reply = data.get("replyData") or {}
        pose = reply.get("realPosPCS") or reply.get("realPosMCS")
        if not isinstance(pose, list) or len(pose) < 6:
            raise ControllerProtocolError(
                "state reply is missing realPosPCS/realPosMCS: {}".format(json.dumps(data, ensure_ascii=False)[:200])
            )
        timestamp = reply.get("timestamp")
        timestamp_s = None
        if isinstance(timestamp, list) and len(timestamp) >= 2:
            timestamp_s = float(timestamp[0]) + float(timestamp[1]) / 1e9
        xyz_mm = np.asarray(pose[:3], dtype=np.float64)
        abc_rad = np.asarray(pose[3:6], dtype=np.float64)
        # IMPORTANT: NexBot A/B/C are intrinsic X'Y'Z' (== fixed ZYX), so the
        # rotation is Rx(A)Ry(B)Rz(C) -- NOT the fixed XYZ order that
        # transform_from_xyz_rpy uses.  Field-verified 2026-08-22: using the
        # wrong order broke the checkerboard hand-eye solve by ~190 mm.
        base_from_gripper = transform_from_inexbot_abc(
            xyz_mm / 1000.0, abc_rad
        )
        return RobotState(
            valid=True,
            base_from_gripper=base_from_gripper,
            timestamp_s=float(now_s if now_s is not None else (timestamp_s or time.time())),
            simulated=False,
            reason="",
        )

    def move_to(self, base_from_gripper, speed_scale=0.1):
        matrix = as_transform(base_from_gripper, "base_from_gripper")
        xyz_m, abc_rad = inexbot_abc_from_transform(matrix)
        point = InexbotPoint(
            name="P0001",
            coordinate_system=COORD_CARTESIAN,
            angle_unit=1,
            shape=1,
            tool_id=0,
            user_id=0,
            axes=[*(xyz_m * 1000.0), *abc_rad, 0.0],
        )
        velocity = _clamp(round(float(speed_scale) * 1000.0), 1, 1000)
        self.move_l([point], speed_mm_s=velocity)

    def move_j(self, points, speed_scale=0.1):
        points = tuple(points)
        if not points:
            raise ValueError("move_j requires at least one point")
        velocity = _clamp(round(float(speed_scale) * 100.0), 1, 100)
        for point in points:
            self._send_motion(CMD_MOVJ, point, velocity)
            self._check_errors()
            if self.endpoint.wait_for_finish:
                self._wait_motion_finish()

    def move_l(self, points, speed_mm_s=30.0):
        points = tuple(points)
        if not points:
            raise ValueError("move_l requires at least one point")
        velocity = _clamp(round(float(speed_mm_s)), 1, 1000)
        for point in points:
            self._send_motion(CMD_MOVL, point, velocity)
            self._check_errors()
            if self.endpoint.wait_for_finish:
                self._wait_motion_finish()

    def stop(self):
        self.motion.send_frame(CMD_EMERGENCY_STOP, {"robot": self.endpoint.robot})
        self._check_errors()


def nexbot_tcp_client_from_config(settings):
    """Build a ``NexBotTcpEndpoint`` from the ``controller.nexbot_tcp`` section."""
    controller = settings.get("controller", {}) if hasattr(settings, "get") else {}
    config = controller.get("nexbot_tcp", {}) if isinstance(controller, dict) else {}
    if not isinstance(config, dict):
        raise ValueError("controller.nexbot_tcp must be a mapping")
    host = str(config.get("host", "")).strip()
    if not host:
        raise ValueError(
            "controller.nexbot_tcp.host is required for the nexbot_tcp robot adapter"
        )
    return NexBotTcpEndpoint(
        host=host,
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
        heartbeat_s=float(config.get("heartbeat_s", 0.0)),
    )


__all__ = [
    "CMD_EMERGENCY_STOP",
    "CMD_MOVJ",
    "CMD_MOVL",
    "CMD_QUERY",
    "CMD_QUERY_REPLY",
    "ControllerConnectionError",
    "ControllerProtocolError",
    "ControllerTimeout",
    "NexBotTcpEndpoint",
    "NexBotTcpRobotController",
    "NexBotTcpTransport",
    "build_frame",
    "nexbot_tcp_client_from_config",
    "read_frame",
]
