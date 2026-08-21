"""Fail-closed Modbus fallback: write GP point data, then trigger local job.

The controller manual documents global GP points and Modbus remote mode, but
not the mapping between a GP point/program signal and Modbus addresses.  This
adapter implements the *mechanism* entirely from an official field mapping;
it contains no vendor address, program number, encoding or trigger polarity.

The pendant must contain a pre-reviewed local program which reads the selected
global GP point and performs MOVJ/MOVL.  The network side writes a full point,
commits a monotonically increasing sequence, then pulses the configured start
signal.  It is a fallback when a private motion TCP API cannot be commissioned.
"""

from dataclasses import replace
import math
import struct
import time

from .inexbot_modbus import InexbotPoint
from ..interfaces import RobotController
from ..shape_latch import ShapeLatch
from ..transforms import transform_from_xyz_rpy
from ..types import RobotState


class ModbusFallbackError(RuntimeError):
    pass


_POINT_FIELDS = (
    "coordinate_system", "angle_unit", "shape", "tool_id", "user_id",
    "reserved_1", "reserved_2",
    "axis_1", "axis_2", "axis_3", "axis_4", "axis_5", "axis_6", "axis_7",
)


def _register_words(value, encoding="u16", scale=1.0, offset=0.0):
    """Encode a physical value using an explicitly configured register codec."""
    encoding = str(encoding).lower()
    scale = float(scale)
    offset = float(offset)
    if scale == 0.0:
        raise ModbusFallbackError("register scale cannot be zero")
    raw_value = (float(value) - offset) / scale
    if encoding == "u16":
        integer = int(round(raw_value))
        if not 0 <= integer <= 0xFFFF:
            raise ModbusFallbackError("u16 value is out of range")
        return [integer]
    if encoding == "s16":
        integer = int(round(raw_value))
        if not -0x8000 <= integer <= 0x7FFF:
            raise ModbusFallbackError("s16 value is out of range")
        return list(struct.unpack(">H", struct.pack(">h", integer)))
    if encoding in ("u32", "u32_le_words"):
        integer = int(round(raw_value))
        if not 0 <= integer <= 0xFFFFFFFF:
            raise ModbusFallbackError("u32 value is out of range")
        words = list(struct.unpack(">HH", struct.pack(">I", integer)))
    elif encoding in ("s32", "s32_le_words"):
        integer = int(round(raw_value))
        if not -0x80000000 <= integer <= 0x7FFFFFFF:
            raise ModbusFallbackError("s32 value is out of range")
        words = list(struct.unpack(">HH", struct.pack(">i", integer)))
    elif encoding in ("f32_be", "f32_le_words"):
        if not math.isfinite(raw_value):
            raise ModbusFallbackError("f32 value must be finite")
        words = list(struct.unpack(">HH", struct.pack(">f", raw_value)))
    else:
        raise ModbusFallbackError("unsupported Modbus write encoding: {}".format(encoding))
    return words[::-1] if encoding.endswith("_le_words") else words


class ModbusGlobalPointRobotController(RobotController):
    """Use a validated local robot job as a motion protocol fallback."""

    def __init__(self, client, settings, *, state_provider=None, sleep=time.sleep,
                 clock=time.monotonic):
        data = settings.data if hasattr(settings, "data") else settings
        controller = data.get("controller", data)
        config = controller.get("modbus_global_point_fallback", {}) or {}
        if not bool(config.get("enabled", False)):
            raise ModbusFallbackError("modbus_global_point_fallback.enabled is false")
        if not bool(config.get("local_program_verified", False)):
            raise ModbusFallbackError(
                "local_program_verified must be true after pendant dry-run validation"
            )
        self.client = client
        self.config = config
        self.state_provider = state_provider
        self._sleep = sleep
        self._clock = clock
        self.timeout_s = float(config.get("timeout_s", 10.0))
        self.poll_interval_s = float(config.get("poll_interval_s", 0.05))
        if self.timeout_s <= 0.0 or self.poll_interval_s <= 0.0:
            raise ModbusFallbackError("fallback timeout and poll interval must be positive")
        self.field_map = dict(config.get("global_point_fields", {}) or {})
        missing = [field for field in _POINT_FIELDS if field not in self.field_map]
        if missing:
            raise ModbusFallbackError(
                "global_point_fields missing: {}".format(", ".join(missing))
            )
        self.command_fields = dict(config.get("command_fields", {}) or {})
        for field in ("motion_code", "sequence_id"):
            if field not in self.command_fields:
                raise ModbusFallbackError("command_fields.{} is required".format(field))
        self.start = dict(config.get("start_signal", {}) or {})
        if "address" not in self.start:
            raise ModbusFallbackError("start_signal.address is required")
        self.complete = dict(config.get("complete_signal", {}) or {})
        if "address" not in self.complete:
            raise ModbusFallbackError("complete_signal.address is required")
        self.stop_signal = dict(config.get("stop_signal", {}) or {})
        if "address" not in self.stop_signal:
            raise ModbusFallbackError(
                "stop_signal.address is required; start deassertion is not an emergency stop"
            )
        self.motion_codes = dict(config.get("motion_codes", {"MOVJ": 1, "MOVL": 2}))
        if "MOVJ" not in self.motion_codes or "MOVL" not in self.motion_codes:
            raise ModbusFallbackError("motion_codes must define MOVJ and MOVL")
        self.sequence = int(config.get("initial_sequence_id", 0)) & 0xFFFF
        self.shape_latch = ShapeLatch(config.get("initial_shape"))
        self.pose_convention = str(
            controller.get("state_pose_convention", "unverified")
        )
        self.last_reason = ""

    @staticmethod
    def _point_values(point):
        return {
            "coordinate_system": point.coordinate_system,
            "angle_unit": point.angle_unit,
            "shape": point.shape,
            "tool_id": point.tool_id,
            "user_id": point.user_id,
            "reserved_1": point.reserved[0],
            "reserved_2": point.reserved[1],
            **{"axis_{}".format(index + 1): point.axes[index] for index in range(7)},
        }

    def _write_register(self, spec, value):
        address = int(spec["address"])
        words = _register_words(
            value, spec.get("encoding", "u16"), spec.get("scale", 1.0),
            spec.get("offset", 0.0),
        )
        return self.client.write_multiple_registers(address, words)

    def _read_signal(self, spec):
        address = int(spec["address"])
        source = str(spec.get("source", "discrete")).lower()
        if source in ("coil", "coils"):
            value = self.client.read_coils(address, 1)[0]
        elif source in ("discrete", "discrete_input", "discrete_inputs"):
            value = self.client.read_discrete_inputs(address, 1)[0]
        elif source == "holding":
            value = self.client.read_holding_registers(address, 1)[0]
        elif source == "input":
            value = self.client.read_input_registers(address, 1)[0]
        else:
            raise ModbusFallbackError("unsupported signal source: {}".format(source))
        return value == spec.get("active_value", True)

    def _write_signal(self, spec, active):
        address = int(spec["address"])
        source = str(spec.get("source", "coil")).lower()
        active_value = spec.get("active_value", True)
        value = active_value if active else spec.get("inactive_value", False)
        if source in ("coil", "coils"):
            return self.client.write_single_coil(address, bool(value))
        if source == "holding":
            return self._write_register(spec, value)
        raise ModbusFallbackError("start/stop signal source must be coil or holding")

    def _next_sequence(self):
        self.sequence = (self.sequence + 1) & 0xFFFF
        return self.sequence

    def _state_is_safe(self):
        if self.state_provider is None:
            return False
        state = self.state_provider()
        initial_shape = getattr(state, "initial_shape", None) if state is not None else None
        observed_shape = getattr(state, "shape", None) if state is not None else None
        shape_matches = (
            initial_shape in range(1, 9)
            and observed_shape in range(1, 9)
            and int(initial_shape) == int(observed_shape)
        )
        return bool(
            state is not None and getattr(state, "connected", False)
            and getattr(state, "emergency_stop", None) is False
            and not getattr(getattr(state, "alarm", None), "active", True)
            and shape_matches
            and not getattr(state, "shape_changed", True)
        )

    def _latched_point(self, point):
        if not isinstance(point, InexbotPoint):
            raise ModbusFallbackError("fallback expects InexbotPoint values")
        state = self.state_provider() if self.state_provider is not None else None
        state_initial = getattr(state, "initial_shape", None) if state is not None else None
        if self.shape_latch.value is None and state_initial is not None:
            self.shape_latch.reset(state_initial)
        observed = getattr(state, "shape", None)
        if observed is None:
            observed = getattr(state, "initial_shape", None)
        self.shape_latch.observe(observed)
        latch = self.shape_latch.state
        if self.shape_latch.value is None:
            raise ModbusFallbackError("initial controller shape has not been read")
        if latch.changed:
            raise ModbusFallbackError("controller shape changed; refuse fallback command")
        return replace(point, shape=self.shape_latch.value)

    def write_global_point(self, point):
        """Write one complete configured GP point but do not start the job."""
        point = self._latched_point(point)
        values = self._point_values(point)
        for field in _POINT_FIELDS:
            self._write_register(self.field_map[field], values[field])
        return point

    def _wait_complete(self):
        deadline = self._clock() + self.timeout_s
        while self._clock() <= deadline:
            if self._read_signal(self.complete):
                return True
            self._sleep(self.poll_interval_s)
        raise ModbusFallbackError("local robot program completion timed out")

    def execute_point(self, point, motion_type):
        if not self._state_is_safe():
            raise ModbusFallbackError("controller state is unsafe or initial shape is unavailable")
        motion_type = str(motion_type).upper()
        if motion_type not in self.motion_codes:
            raise ModbusFallbackError("unsupported local-program motion type: {}".format(motion_type))
        self.last_reason = ""
        try:
            point = self.write_global_point(point)
            sequence = self._next_sequence()
            self._write_register(self.command_fields["motion_code"], self.motion_codes[motion_type])
            self._write_register(self.command_fields["sequence_id"], sequence)
            self._write_signal(self.start, True)
            try:
                self._wait_complete()
            finally:
                self._write_signal(self.start, False)
            return True
        except Exception as error:
            self.last_reason = str(error)
            self.stop()
            raise

    def move_j(self, points, speed_scale=0.1):
        float(speed_scale)  # Speed must be implemented in the verified local job.
        for point in tuple(points):
            self.execute_point(point, "MOVJ")
        return True

    def move_l(self, points, speed_mm_s=30.0):
        float(speed_mm_s)  # Speed must be implemented in the verified local job.
        for point in tuple(points):
            self.execute_point(point, "MOVL")
        return True

    def read_state(self, now_s=None):
        """Return the framework ``RobotState`` when a confirmed TCP map exists.

        The raw ``ControllerState`` remains available through
        ``controller_state_provider`` for shape/alarm checks.  A pose is only
        exposed to localization after the field team confirms the controller's
        RPY convention; otherwise the pipeline receives an invalid state and
        cannot silently use an unverified transform.
        """
        if self.state_provider is None:
            return None
        state = self.state_provider()
        if state is None or not getattr(state, "connected", False):
            return RobotState(
                False, None, time.time() if now_s is None else float(now_s),
                reason="controller state unavailable",
            )
        if self.pose_convention != "fixed_zyx_rpy_deg":
            return RobotState(
                False, None, getattr(state, "timestamp_s", time.time()),
                reason="controller TCP RPY convention is unverified",
            )
        xyz = getattr(state, "tcp_xyz_mm", None)
        rpy = getattr(state, "tcp_rpy_deg", None)
        if xyz is None or rpy is None:
            return RobotState(
                False, None, getattr(state, "timestamp_s", time.time()),
                reason="controller TCP pose fields are unavailable",
            )
        try:
            transform = transform_from_xyz_rpy(
                [float(value) / 1000.0 for value in xyz], rpy
            )
        except (TypeError, ValueError) as error:
            return RobotState(
                False, None, getattr(state, "timestamp_s", time.time()),
                reason="controller TCP pose is invalid: {}".format(error),
            )
        safe = (
            getattr(state, "emergency_stop", None) is False
            and not getattr(getattr(state, "alarm", None), "active", True)
        )
        return RobotState(
            bool(safe), transform if safe else None,
            getattr(state, "timestamp_s", time.time()),
            simulated=False,
            reason="controller TCP pose" if safe else "controller state unsafe",
        )

    def move_to(self, base_from_gripper, speed_scale=0.1):
        raise ModbusFallbackError("fallback accepts only validated InexbotPoint MOVJ/MOVL commands")

    def stop(self):
        self._write_signal(self.stop_signal, True)
        self._sleep(min(self.poll_interval_s, 0.05))
        self._write_signal(self.stop_signal, False)
        return True


__all__ = [
    "ModbusFallbackError", "ModbusGlobalPointRobotController", "_register_words",
]
