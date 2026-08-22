"""Construct framework components from YAML parameters."""

from dataclasses import replace

from .adapters.foundationpose import FoundationPoseEstimator, FoundationPoseRuntime
from .adapters.inexbot_modbus import modbus_client_from_config
from .adapters.mock import MockPoseEstimator, MockRobotController, MockSegmenter
from .adapters.modbus_global_point import (
    ModbusFallbackError, ModbusGlobalPointRobotController,
)
from .adapters.nexbot_tcp import (
    NexBotTcpRobotController, nexbot_tcp_client_from_config,
)
from .adapters.topic_robot import TopicRobotController
from .adapters.yolo import YoloSegmenter
from .controller_state_reader import ControllerStateReader
from .errors import ConfigurationError
from .localization import HybridCameraLocalizer
from .pipeline import CompetitionPipeline
from .shape_latch import ShapeLatch


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
            config.get("bbox_mask_fallback", True),
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
            mesh_paths=config.get("mesh_paths", {}),
            roi_padding_pixels=config.get("roi_padding_pixels", 12),
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


def _configured_controller_state_provider(client, settings):
    """Build a read-only state callback for the Modbus motion fallback.

    The callback is intentionally configuration-driven.  An empty state map
    is rejected instead of returning a fabricated ``shape`` or TCP pose.
    """
    reader = ControllerStateReader(client, settings)
    if not reader.mapping:
        raise ConfigurationError(
            "Modbus fallback requires controller.state_registers from the official map"
        )
    controller = settings.data.get("controller", {}) if hasattr(settings, "data") else settings.get("controller", {})
    latch = ShapeLatch(controller.get("initial_shape"))

    def read_state():
        state = reader.read()
        latched = latch.observe(state.shape)
        return replace(
            state,
            initial_shape=latched.initial_shape,
            shape_changed=latched.changed,
            raw_registers={
                **state.raw_registers,
                "observed_shape": latched.observed_shape,
                "initial_shape": latched.initial_shape,
                "shape_changed": latched.changed,
            },
        )

    return read_state


def build_robot(settings, controller_client=None, state_provider=None):
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
    if backend == "modbus_global_point":
        try:
            client = controller_client or modbus_client_from_config(settings)
            if client is None:
                raise ConfigurationError(
                    "Modbus fallback requires an enabled, configured controller"
                )
            provider = state_provider or _configured_controller_state_provider(
                client, settings
            )
            robot = ModbusGlobalPointRobotController(
                client, settings, state_provider=provider
            )
            # Execution assembly can reuse the same read-only callback for
            # shape locking and recovery checks without guessing a second map.
            robot.controller_state_provider = provider
            return robot
        except (ModbusFallbackError, ValueError, KeyError) as error:
            raise ConfigurationError(str(error)) from error
    if backend == "nexbot_tcp":
        try:
            endpoint = nexbot_tcp_client_from_config(settings)
            return NexBotTcpRobotController(endpoint)
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
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
        use_visual_tags=settings.get("localization", {}).get(
            "use_apriltag_runtime", False
        ),
    )
    return CompetitionPipeline(
        build_segmenter(settings),
        build_pose_estimator(settings, foundationpose_runtime),
        localizer,
        robot,
    )
