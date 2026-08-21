"""Replaceable hardware and algorithm adapters."""

from .mock import MockPoseEstimator, MockRobotController, MockSegmenter
from .foundationpose import FoundationPoseEstimator, FoundationPoseRuntime
from .inexbot_modbus import (
    ConfiguredRemoteIo,
    ControllerConnectionError,
    ControllerProtocolError,
    ControllerTimeout,
    InexbotPoint,
    ModbusExceptionResponse,
    ModbusProtocolError,
    ModbusTcpClient,
    TcpEndpoint,
    TcpTransport,
    point_from_joint_degrees,
    shape_from_joint_degrees,
)
from .io_gripper import RemoteIoGripper, gripper_from_config

__all__ = [
    "MockPoseEstimator",
    "MockRobotController",
    "MockSegmenter",
    "FoundationPoseEstimator",
    "FoundationPoseRuntime",
    "ConfiguredRemoteIo",
    "ControllerConnectionError",
    "ControllerProtocolError",
    "ControllerTimeout",
    "InexbotPoint",
    "ModbusExceptionResponse",
    "ModbusProtocolError",
    "ModbusTcpClient",
    "TcpEndpoint",
    "TcpTransport",
    "point_from_joint_degrees",
    "shape_from_joint_degrees",
    "RemoteIoGripper",
    "gripper_from_config",
]
from .modbus_global_point import ModbusFallbackError, ModbusGlobalPointRobotController

__all__ += ["ModbusFallbackError", "ModbusGlobalPointRobotController"]
