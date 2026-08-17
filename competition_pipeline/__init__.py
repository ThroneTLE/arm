"""Hardware-independent competition calibration and localization pipeline."""

from .configuration import CompetitionConfig
from .control import MotionSafetyError, SafeRobotController
from .execution import ExecutionResult, GraspExecutor
from .planning import GraspPlan, GraspPlanningError, GraspTarget, TopDownGraspPlanner
from .hand_eye import HandEyeCalibrator, HandEyeResult
from .localization import HybridLocalizer, LocalizationResult
from .runtime import CompetitionRuntime
from .tag_map import TagMap

__all__ = [
    "CompetitionConfig",
    "CompetitionRuntime",
    "HandEyeCalibrator",
    "HandEyeResult",
    "HybridLocalizer",
    "LocalizationResult",
    "MotionSafetyError",
    "SafeRobotController",
    "ExecutionResult",
    "GraspExecutor",
    "GraspPlan",
    "GraspPlanningError",
    "GraspTarget",
    "TopDownGraspPlanner",
    "TagMap",
]
