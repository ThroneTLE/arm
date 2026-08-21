"""Configuration-driven, read-only controller state decoder.

No register address is embedded here.  The real controller/IO sheet can be
entered under ``controller.state_registers`` and the UI will immediately show
decoded values without requiring a code change.
"""

import struct
import time
from typing import Any, Mapping

from .controller_state import ControllerAlarm, ControllerState


class StateCodecError(ValueError):
    pass


def _words_to_bytes(words):
    return b"".join(struct.pack(">H", int(word) & 0xFFFF) for word in words)


def decode_registers(words, encoding="u16", scale=1.0, offset=0.0):
    values = list(words)
    encoding = str(encoding).lower()
    if encoding == "u16":
        value = values[0]
    elif encoding == "s16":
        value = struct.unpack(">h", _words_to_bytes(values[:1]))[0]
    elif encoding in ("u32", "s32", "f32_be", "u32_le_words", "s32_le_words"):
        if len(values) < 2:
            raise StateCodecError("{} requires two registers".format(encoding))
        selected = values[:2][::-1] if encoding.endswith("_le_words") else values[:2]
        raw = _words_to_bytes(selected)
        if encoding in ("u32", "u32_le_words"):
            value = struct.unpack(">I", raw)[0]
        elif encoding in ("s32", "s32_le_words"):
            value = struct.unpack(">i", raw)[0]
        else:
            value = struct.unpack(">f", raw)[0]
    elif encoding == "f32_le_words":
        if len(values) < 2:
            raise StateCodecError("f32_le_words requires two registers")
        value = struct.unpack(">f", _words_to_bytes(values[:2][::-1]))[0]
    else:
        raise StateCodecError("unsupported register encoding: {}".format(encoding))
    return float(value) * float(scale) + float(offset)


class ControllerStateReader:
    """Decode only explicitly mapped values from a Modbus-like client."""

    def __init__(self, client, settings):
        self.client = client
        data = settings.data if hasattr(settings, "data") else settings
        controller = data.get("controller", data)
        self.mapping = dict(controller.get("state_registers", {}) or {})
        self.codec = dict(controller.get("state_codec", {}) or {})

    def _spec(self, name):
        spec = self.mapping.get(name)
        if spec is None:
            return None
        if not isinstance(spec, Mapping):
            raise StateCodecError("state_registers.{} must be a mapping".format(name))
        return spec

    def _read(self, name):
        spec = self._spec(name)
        if spec is None:
            return None
        address = int(spec["address"])
        source = str(spec.get("source", "holding")).lower()
        quantity = int(spec.get("quantity", 1))
        if source == "holding":
            raw = self.client.read_holding_registers(address, quantity)
            value = decode_registers(raw, spec.get("encoding", "u16"), spec.get("scale", 1.0), spec.get("offset", 0.0))
        elif source == "input":
            raw = self.client.read_input_registers(address, quantity)
            value = decode_registers(raw, spec.get("encoding", "u16"), spec.get("scale", 1.0), spec.get("offset", 0.0))
        elif source in ("coil", "coils"):
            value = bool(self.client.read_coils(address, 1)[0])
        elif source in ("discrete", "discrete_input", "discrete_inputs"):
            value = bool(self.client.read_discrete_inputs(address, 1)[0])
        else:
            raise StateCodecError("unsupported source for {}: {}".format(name, source))
        if "bit_mask" in spec:
            value = bool(int(value) & int(spec["bit_mask"], 0)) if isinstance(spec["bit_mask"], str) else bool(int(value) & int(spec["bit_mask"]))
        if bool(spec.get("invert", False)):
            value = not bool(value)
        return value

    def read(self):
        raw = {}
        for name in self.mapping:
            raw[name] = self._read(name)

        def vector(prefix, count):
            values = []
            for index in range(count):
                value = raw.get("{}_{}".format(prefix, index + 1))
                if value is None:
                    return None
                values.append(value)
            return tuple(values)

        alarm_code = raw.get("alarm_code")
        alarm_text = raw.get("alarm_text", "")
        if not alarm_text and alarm_code is not None:
            alarm_text = self.codec.get("alarm_texts", {}).get(
                str(int(alarm_code)), self.codec.get("alarm_texts", {}).get(int(alarm_code), "")
            )
        alarm_active = raw.get("alarm_active", bool(alarm_code))
        alarm = ControllerAlarm(
            code=None if alarm_code is None else int(alarm_code),
            text=str(alarm_text or ""),
            severity=str(self.codec.get("alarm_severity", "error" if alarm_active else "none")),
            active=bool(alarm_active),
        )
        return ControllerState(
            timestamp_s=time.time(), connected=True,
            servo_on=raw.get("servo_on"), emergency_stop=raw.get("emergency_stop"),
            moving=raw.get("moving"), joint_deg=vector("joint_deg", 6),
            point_name=str(self.codec.get("current_point_name", "")),
            coordinate_system=(None if raw.get("coordinate_system") is None else int(raw["coordinate_system"])),
            angle_unit=None if raw.get("angle_unit") is None else int(raw["angle_unit"]),
            reserved=vector("reserved", 2),
            axes=vector("axis", 7),
            tcp_xyz_mm=tuple(raw.get(key) for key in ("tcp_x_mm", "tcp_y_mm", "tcp_z_mm"))
            if all(raw.get(key) is not None for key in ("tcp_x_mm", "tcp_y_mm", "tcp_z_mm")) else None,
            tcp_rpy_deg=tuple(raw.get(key) for key in ("tcp_rx_deg", "tcp_ry_deg", "tcp_rz_deg"))
            if all(raw.get(key) is not None for key in ("tcp_rx_deg", "tcp_ry_deg", "tcp_rz_deg")) else None,
            tool_id=None if raw.get("tool_id") is None else int(raw["tool_id"]),
            user_id=None if raw.get("user_id") is None else int(raw["user_id"]),
            shape=None if raw.get("shape") is None else int(raw["shape"]),
            alarm=alarm, raw_registers=raw,
        )


__all__ = ["StateCodecError", "decode_registers", "ControllerStateReader"]
