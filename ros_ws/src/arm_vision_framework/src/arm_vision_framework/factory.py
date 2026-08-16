"""Construct framework components from YAML parameters."""

from .adapters.foundationpose import FoundationPoseEstimator, FoundationPoseRuntime
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


def build_foundationpose_runtime(settings):
    """Construct the external CUDA runtime only for a FoundationPose backend."""
    config = settings["pose_estimation"]
    backend = str(config.get("backend", "mock"))
    if backend not in ("foundationpose", "foundationpose_plus_plus"):
        return None
    root = str(config.get("foundationpose_root", "")).strip()
    if not root:
        raise ConfigurationError(
            "pose_estimation.foundationpose_root is required for {}".format(backend)
        )
    return FoundationPoseRuntime(
        root,
        debug_dir=config.get("debug_dir", "/tmp/arm_foundationpose"),
        debug=config.get("debug", 0),
        est_refine_iter=config.get("est_refine_iter", 5),
        track_refine_iter=config.get("track_refine_iter", 2),
        device=config.get("device", "cuda:0"),
        use_mask_center_guidance=config.get("use_mask_center_guidance", True),
    )


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
    if foundationpose_runtime is None:
        foundationpose_runtime = build_foundationpose_runtime(settings)
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
