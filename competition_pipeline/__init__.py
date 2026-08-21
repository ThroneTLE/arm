"""Hardware-independent competition calibration and localization pipeline."""

from .configuration import CompetitionConfig
from .checkerboard_target import CheckerboardObservation, CheckerboardTarget
from .control import MotionSafetyError, SafeRobotController
from .execution import ExecutionResult, GraspExecutor
from .planning import (
    AnyGraspFallbackPlanner,
    AnyGraspFallbackSettings,
    FallbackGraspPlanner,
    GraspPlan,
    GraspPlanningError,
    GraspTarget,
    TopDownGraspPlanner,
    planner_from_config,
)
from .hand_eye import HandEyeCalibrator, HandEyeResult
from .localization import HybridLocalizer, LocalizationResult
from .runtime import CompetitionRuntime
from .tag_map import TagMap
from .controller_tcp import (
    ControllerConnectionError,
    ControllerProtocolError,
    ControllerTimeout,
    ConfiguredRemoteIo,
    InexbotPoint,
    ModbusExceptionResponse,
    ModbusProtocolError,
    ModbusTcpClient,
    TcpEndpoint,
    TcpTransport,
    point_from_joint_degrees,
    shape_from_joint_degrees,
)

__all__ = [
    "CompetitionConfig",
    "CheckerboardObservation",
    "CheckerboardTarget",
    "CompetitionRuntime",
    "HandEyeCalibrator",
    "HandEyeResult",
    "HybridLocalizer",
    "LocalizationResult",
    "MotionSafetyError",
    "SafeRobotController",
    "ExecutionResult",
    "GraspExecutor",
    "AnyGraspFallbackPlanner",
    "AnyGraspFallbackSettings",
    "FallbackGraspPlanner",
    "GraspPlan",
    "GraspPlanningError",
    "GraspTarget",
    "TopDownGraspPlanner",
    "planner_from_config",
    "TagMap",
    "ControllerConnectionError",
    "ControllerProtocolError",
    "ControllerTimeout",
    "ConfiguredRemoteIo",
    "InexbotPoint",
    "ModbusExceptionResponse",
    "ModbusProtocolError",
    "ModbusTcpClient",
    "TcpEndpoint",
    "TcpTransport",
    "point_from_joint_degrees",
    "shape_from_joint_degrees",
]
