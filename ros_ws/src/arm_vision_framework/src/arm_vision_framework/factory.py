"""Construct framework components from YAML parameters."""

from .adapters.foundationpose import FoundationPoseEstimator
from .adapters.mock import MockPoseEstimator, MockRobotController, MockSegmenter
from .adapters.topic_robot import TopicRobotController
from .adapters.yolo import YoloSegmenter
from .errors import ConfigurationError
from .localization import HybridCameraLocalizer
from .pipeline import CompetitionPipeline


def build_segmenter(settings):
    config = settings["segmentation"]
    backend = str(config.get("backend", "mock"))
    if backend == "mock":
        return MockSegmenter()
    if backend == "yolo":
        return YoloSegmenter(
            config.get("weights", ""),
            config.get("target_classes", []),
            config.get("confidence_threshold", 0.5),
        )
    raise ConfigurationError("unknown segmentation backend: {}".format(backend))


def build_pose_estimator(settings, foundationpose_runtime=None):
    config = settings["pose_estimation"]
    backend = str(config.get("backend", "mock"))
    if backend == "mock":
        return MockPoseEstimator()
    if backend in ("foundationpose", "foundationpose_plus_plus"):
        return FoundationPoseEstimator(
            config.get("mesh_path", ""),
            config.get("mesh_scale_to_meters", 1.0),
            runtime=foundationpose_runtime,
            require_aligned_depth=config.get("require_aligned_depth", True),
        )
    raise ConfigurationError("unknown pose backend: {}".format(backend))


def build_robot(settings):
    config = settings["robot"]
    backend = str(config.get("adapter", "mock"))
    if backend == "mock":
        return MockRobotController(
            allow_motion=bool(settings.get("safety", {}).get("allow_robot_motion", False))
        )
    if backend == "ros_pose":
        safety = settings.get("safety", {})
        return TopicRobotController(
            allow_motion=bool(safety.get("allow_robot_motion", False))
            and not bool(safety.get("dry_run", True))
        )
    raise ConfigurationError(
        "robot adapter {} is not implemented; add a vendor bridge after hardware assignment".format(backend)
    )


def build_pipeline(settings, calibration, foundationpose_runtime=None):
    robot = build_robot(settings)
    localizer = HybridCameraLocalizer(
        calibration,
        maximum_robot_pose_age_s=settings["runtime"].get(
            "maximum_robot_pose_age_s", 0.25
        ),
        use_robot_fallback=settings.get("localization", {}).get(
            "use_robot_fallback", True
        ),
    )
    return CompetitionPipeline(
        build_segmenter(settings),
        build_pose_estimator(settings, foundationpose_runtime),
        localizer,
        robot,
    )
