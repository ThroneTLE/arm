"""Competition-oriented robot vision framework."""

from .parameters import CalibrationStore, load_system_parameters
from .pipeline import CompetitionPipeline
from .motion_execution import (
    ObservationPlan, PickPlaceExecutor, PickPlacePlan,
    TwoViewPickPlaceCoordinator, cartesian_point, joint_point,
)
from .oak_depthai import OakDProProfile, profile_from_config
from .controller_state import ControllerAlarm, ControllerState, VisualTaskCommand
from .controller_state_reader import ControllerStateReader, StateCodecError
from .oak_imu import OakImuConfig, config_from_camera
from .safety_recovery import SafeRecoveryManager, SafeRecoveryState
from .shape_latch import ShapeLatch, ShapeLatchState
from .adapters.modbus_global_point import (
    ModbusFallbackError, ModbusGlobalPointRobotController,
)
from .object_ordering import sort_workspace_objects

__all__ = [
    "CalibrationStore", "CompetitionPipeline", "load_system_parameters",
    "ObservationPlan", "PickPlaceExecutor", "PickPlacePlan",
    "TwoViewPickPlaceCoordinator", "cartesian_point", "joint_point",
    "OakDProProfile", "profile_from_config",
    "ControllerAlarm", "ControllerState", "VisualTaskCommand",
    "ControllerStateReader", "StateCodecError",
    "OakImuConfig", "config_from_camera",
    "SafeRecoveryManager", "SafeRecoveryState",
    "ShapeLatch", "ShapeLatchState",
    "ModbusFallbackError", "ModbusGlobalPointRobotController",
    "sort_workspace_objects",
]
