"""Stable controller state and vision/control message contracts.

The vendor manual describes the fields shown on the pendant (joint pose,
Cartesian TCP pose, tool/user and the configuration/shape value), but does not
publish a complete Ethernet state packet.  These dataclasses deliberately keep
the wire transport separate from that contract: a field adapter can populate
``ControllerState`` from Modbus registers or an official SDK without changing
the vision pipeline.
"""

from dataclasses import asdict, dataclass, field
import json
import re
import time
from typing import Any, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "arm_vision.control.v1"


def _tuple(values, length, name, cast=float):
    if values is None:
        return None
    result = tuple(cast(value) for value in values)
    if len(result) != length:
        raise ValueError("{} must contain {} values".format(name, length))
    return result


@dataclass(frozen=True)
class ControllerAlarm:
    code: Optional[int] = None
    text: str = ""
    severity: str = "none"
    active: bool = False

    def __post_init__(self):
        if self.code is not None:
            object.__setattr__(self, "code", int(self.code))
        severity = str(self.severity).strip().lower() or "none"
        if severity not in ("none", "info", "warning", "error", "fatal"):
            raise ValueError("unsupported alarm severity: {}".format(severity))
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "text", str(self.text or ""))
        object.__setattr__(self, "active", bool(self.active))

    def is_suspected_singularity(self):
        return bool(re.search(
            r"singular|singularity|奇异|奇点|逆运动学|ik\b|configuration",
            self.text,
            re.IGNORECASE,
        ))


@dataclass(frozen=True)
class ControllerState:
    """Controller state in the units used by the competition contract.

    Joint and RPY angles are degrees; Cartesian XYZ is millimetres.  ``None``
    means that a register mapping has not been supplied or the value is stale.
    """

    timestamp_s: float = field(default_factory=time.time)
    connected: bool = False
    servo_on: Optional[bool] = None
    emergency_stop: Optional[bool] = None
    moving: Optional[bool] = None
    joint_deg: Optional[Tuple[float, ...]] = None
    point_name: str = ""
    coordinate_system: Optional[int] = None
    angle_unit: Optional[int] = None
    reserved: Optional[Tuple[float, float]] = None
    axes: Optional[Tuple[float, ...]] = None
    tcp_xyz_mm: Optional[Tuple[float, float, float]] = None
    tcp_rpy_deg: Optional[Tuple[float, float, float]] = None
    tool_id: Optional[int] = None
    user_id: Optional[int] = None
    shape: Optional[int] = None
    initial_shape: Optional[int] = None
    shape_changed: bool = False
    alarm: ControllerAlarm = field(default_factory=ControllerAlarm)
    raw_registers: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""

    def __post_init__(self):
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "connected", bool(self.connected))
        object.__setattr__(self, "joint_deg", _tuple(self.joint_deg, 6, "joint_deg"))
        object.__setattr__(self, "point_name", str(self.point_name or ""))
        object.__setattr__(self, "reserved", _tuple(self.reserved, 2, "reserved"))
        object.__setattr__(self, "axes", _tuple(self.axes, 7, "axes"))
        object.__setattr__(self, "tcp_xyz_mm", _tuple(self.tcp_xyz_mm, 3, "tcp_xyz_mm"))
        object.__setattr__(self, "tcp_rpy_deg", _tuple(self.tcp_rpy_deg, 3, "tcp_rpy_deg"))
        for name in ("servo_on", "emergency_stop", "moving"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, bool(value))
        for name in ("tool_id", "user_id", "shape", "initial_shape"):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if name in ("shape", "initial_shape") and not 1 <= value <= 8:
                    raise ValueError("{} must be within 1..8".format(name))
                object.__setattr__(self, name, value)
        object.__setattr__(self, "shape_changed", bool(self.shape_changed))
        if self.coordinate_system is not None:
            coordinate_system = int(self.coordinate_system)
            if coordinate_system not in (0, 1, 2, 3):
                raise ValueError("coordinate_system must be within 0..3")
            object.__setattr__(self, "coordinate_system", coordinate_system)
        if self.angle_unit is not None:
            angle_unit = int(self.angle_unit)
            if angle_unit not in (0, 1):
                raise ValueError("angle_unit must be 0 (degree) or 1 (radian)")
            object.__setattr__(self, "angle_unit", angle_unit)
        if not isinstance(self.alarm, ControllerAlarm):
            object.__setattr__(self, "alarm", ControllerAlarm(**dict(self.alarm)))
        object.__setattr__(self, "raw_registers", dict(self.raw_registers or {}))
        object.__setattr__(self, "error", str(self.error or ""))

    def to_dict(self):
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        return data

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        payload = dict(data)
        version = payload.pop("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError("unsupported controller schema: {}".format(version))
        payload["alarm"] = ControllerAlarm(**payload.get("alarm", {}))
        return cls(**payload)

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class VisualTaskCommand:
    """Versioned vision-to-controller command envelope.

    ``targets`` contains either one pose or a discrete trajectory.  The
    controller bridge may convert it to MOVJ/MOVL only after validating its
    own vendor protocol; this object is never serialized as a guessed packet.
    """

    command_id: str
    motion_type: str
    targets: Tuple[Mapping[str, Any], ...]
    frame_id: str = "base"
    xyz_unit: str = "mm"
    angle_unit: str = "deg"
    tool_id: Optional[int] = None
    user_id: Optional[int] = None
    shape: Optional[int] = None
    timestamp_s: float = field(default_factory=time.time)
    safe_to_execute: bool = False
    safety_state: str = "unvalidated"
    source: str = "vision"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        command_id = str(self.command_id).strip()
        if not command_id:
            raise ValueError("command_id is required")
        motion = str(self.motion_type).strip().upper()
        if motion not in ("MOVJ", "MOVL", "STOP", "RECOVER"):
            raise ValueError("motion_type must be MOVJ, MOVL, STOP, or RECOVER")
        targets = tuple(dict(item) for item in (self.targets or ()))
        if motion in ("MOVJ", "MOVL") and not targets:
            raise ValueError("motion commands require at least one target")
        for index, target in enumerate(targets):
            if motion == "MOVJ":
                joints = target.get("joint_deg")
                axes = target.get("axes")
                if (joints is None or len(joints) != 6) and (axes is None or len(axes) != 7):
                    raise ValueError(
                        "MOVJ target {} requires joint_deg[6] or axes[7]".format(index)
                    )
            elif motion == "MOVL":
                if len(target.get("xyz_mm", ())) != 3 or len(target.get("rpy_deg", ())) != 3:
                    raise ValueError(
                        "MOVL target {} requires xyz_mm[3] and rpy_deg[3]".format(index)
                    )
        if str(self.xyz_unit) != "mm" or str(self.angle_unit) != "deg":
            raise ValueError("control contract units are mm and deg")
        if self.shape is not None and not 1 <= int(self.shape) <= 8:
            raise ValueError("shape must be within 1..8")
        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(self, "motion_type", motion)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "safe_to_execute", bool(self.safe_to_execute))
        safety_state = str(self.safety_state).strip().lower()
        if safety_state not in ("unvalidated", "validated", "blocked", "emergency_stop"):
            raise ValueError("unsupported safety_state: {}".format(safety_state))
        if self.safe_to_execute and safety_state != "validated":
            raise ValueError("safe_to_execute requires safety_state=validated")
        object.__setattr__(self, "safety_state", safety_state)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        for name in ("tool_id", "user_id", "shape"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, int(value))

    def to_dict(self):
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        return data

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        payload = dict(data)
        version = payload.pop("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError("unsupported control schema: {}".format(version))
        return cls(**payload)

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))


__all__ = ["SCHEMA_VERSION", "ControllerAlarm", "ControllerState", "VisualTaskCommand"]
