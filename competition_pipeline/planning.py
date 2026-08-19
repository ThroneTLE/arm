"""Backend-neutral grasp contracts and configurable grasp backends.

The deterministic top-down planner remains the competition default.  The
AnyGrasp adapter below is deliberately lazy and optional: it is only loaded
when the deterministic planner rejects an object and the fallback is enabled.
This keeps the validated mainline independent of the proprietary SDK, CUDA,
and its license files.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import as_transform


class GraspPlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraspTarget:
    object_id: str
    base_from_grasp: np.ndarray
    width_m: float
    score: float = 1.0
    source: str = "deterministic_top_down"

    def __post_init__(self):
        object.__setattr__(
            self, "base_from_grasp", as_transform(self.base_from_grasp, "base_from_grasp")
        )
        object.__setattr__(self, "object_id", str(self.object_id))
        object.__setattr__(self, "width_m", float(self.width_m))


@dataclass(frozen=True)
class GraspPlan:
    target: GraspTarget
    base_from_tcp_pregrasp: np.ndarray
    base_from_tcp_grasp: np.ndarray
    base_from_tcp_lift: np.ndarray
    base_from_tcp_preplace: np.ndarray
    base_from_tcp_place: np.ndarray
    base_from_tcp_retreat: np.ndarray
    place_position_base_m: np.ndarray

    def __post_init__(self):
        for name in (
            "base_from_tcp_pregrasp", "base_from_tcp_grasp",
            "base_from_tcp_lift", "base_from_tcp_preplace",
            "base_from_tcp_place", "base_from_tcp_retreat",
        ):
            object.__setattr__(self, name, as_transform(getattr(self, name), name))
        object.__setattr__(
            self,
            "place_position_base_m",
            np.asarray(self.place_position_base_m, dtype=np.float64).reshape(3),
        )

    @property
    def ordered_waypoints(self) -> Tuple[np.ndarray, ...]:
        return (
            self.base_from_tcp_pregrasp,
            self.base_from_tcp_grasp,
            self.base_from_tcp_lift,
            self.base_from_tcp_preplace,
            self.base_from_tcp_place,
            self.base_from_tcp_retreat,
        )


@dataclass(frozen=True)
class TopDownPlannerSettings:
    tcp_from_grasp: np.ndarray
    minimum_grasp_width_m: float = 0.005
    maximum_grasp_width_m: float = 0.075
    width_margin_m: float = 0.006
    pregrasp_clearance_m: float = 0.12
    lift_distance_m: float = 0.15
    place_clearance_m: float = 0.15

    def __post_init__(self):
        object.__setattr__(
            self, "tcp_from_grasp", as_transform(self.tcp_from_grasp, "tcp_from_grasp")
        )
        if self.minimum_grasp_width_m < 0.0:
            raise ValueError("minimum grasp width cannot be negative")
        if self.maximum_grasp_width_m <= self.minimum_grasp_width_m:
            raise ValueError("maximum grasp width must exceed minimum")
        for name in ("pregrasp_clearance_m", "lift_distance_m", "place_clearance_m"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError("{} must be positive".format(name))


class TopDownGraspPlanner:
    """Make a deterministic parallel-jaw plan from grounded object bounds.

    Canonical base is +X forward, +Y left and +Z up. The grasp frame's local
    +Z is the approach direction, therefore a top-down grasp has +Z toward
    base -Z. Local +Y is the jaw closing/opening axis.
    """

    def __init__(self, settings):
        self.settings = settings

    def target_from_object(self, object_cloud, object_id=None):
        if not bool(getattr(object_cloud, "valid", False)):
            raise GraspPlanningError(
                "object cloud is invalid: {}".format(getattr(object_cloud, "reason", ""))
            )
        minimum = np.asarray(object_cloud.bounds_min_base_m, dtype=np.float64).reshape(3)
        maximum = np.asarray(object_cloud.bounds_max_base_m, dtype=np.float64).reshape(3)
        center = np.asarray(object_cloud.center_base_m, dtype=np.float64).reshape(3)
        dimensions = maximum - minimum
        if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
            raise GraspPlanningError("object bounds are empty or non-finite")

        # Close across the smaller horizontal extent to maximize clearance.
        closing_axis = (
            np.asarray([1.0, 0.0, 0.0])
            if dimensions[0] <= dimensions[1]
            else np.asarray([0.0, 1.0, 0.0])
        )
        width = float(min(dimensions[0], dimensions[1]) + self.settings.width_margin_m)
        if width < self.settings.minimum_grasp_width_m:
            raise GraspPlanningError("object is narrower than the configured grasp range")
        if width > self.settings.maximum_grasp_width_m:
            raise GraspPlanningError(
                "required opening {:.1f} mm exceeds {:.1f} mm".format(
                    width * 1000.0, self.settings.maximum_grasp_width_m * 1000.0
                )
            )

        approach = np.asarray([0.0, 0.0, -1.0])
        lateral = np.cross(closing_axis, approach)
        base_from_grasp = np.eye(4, dtype=np.float64)
        base_from_grasp[:3, :3] = np.column_stack([lateral, closing_axis, approach])
        base_from_grasp[:3, 3] = center
        return GraspTarget(
            object_id=(
                object_id
                if object_id is not None
                else getattr(object_cloud, "class_name", "object") or "object"
            ),
            base_from_grasp=base_from_grasp,
            width_m=width,
        )

    def plan(self, target, place_position_base_m):
        target = target if isinstance(target, GraspTarget) else GraspTarget(**target)
        place = np.asarray(place_position_base_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(place)):
            raise GraspPlanningError("place position is non-finite")
        grasp_tcp = target.base_from_grasp @ np.linalg.inv(self.settings.tcp_from_grasp)

        def translated(pose, offset):
            result = pose.copy()
            result[:3, 3] += np.asarray(offset, dtype=np.float64).reshape(3)
            return result

        pregrasp = translated(grasp_tcp, [0.0, 0.0, self.settings.pregrasp_clearance_m])
        lift = translated(grasp_tcp, [0.0, 0.0, self.settings.lift_distance_m])
        place_grasp = target.base_from_grasp.copy()
        place_grasp[:3, 3] = place
        place_tcp = place_grasp @ np.linalg.inv(self.settings.tcp_from_grasp)
        preplace = translated(place_tcp, [0.0, 0.0, self.settings.place_clearance_m])
        retreat = preplace.copy()
        return GraspPlan(
            target, pregrasp, grasp_tcp, lift, preplace, place_tcp, retreat, place
        )


@dataclass(frozen=True)
class AnyGraspFallbackSettings:
    """Configuration for the optional AnyGrasp candidate adapter."""

    sdk_grasp_dir: str
    checkpoint_path: str
    minimum_score: float = 0.01
    maximum_gripper_width_m: float = 0.08
    gripper_height_m: float = 0.03
    top_k: int = 20
    maximum_approach_deviation_deg: float = 45.0
    collision_detection: bool = True
    dense_grasp: bool = False

    def __post_init__(self):
        if float(self.minimum_score) < 0.0:
            raise ValueError("AnyGrasp minimum_score cannot be negative")
        if not 0.0 < float(self.maximum_gripper_width_m) <= 0.1:
            raise ValueError("AnyGrasp maximum_gripper_width_m must be within (0, 0.1]")
        if float(self.gripper_height_m) <= 0.0:
            raise ValueError("AnyGrasp gripper_height_m must be positive")
        if int(self.top_k) < 1:
            raise ValueError("AnyGrasp top_k must be positive")
        if not 0.0 <= float(self.maximum_approach_deviation_deg) <= 179.0:
            raise ValueError("AnyGrasp approach deviation must be within 0..179 degrees")


class AnyGraspFallbackPlanner:
    """Convert the best AnyGrasp candidate into the competition grasp frame.

    ``object_cloud.points_base_m`` is transformed with ``base_from_camera``
    before it reaches AnyGrasp.  The adapter accepts an injected ``planner``
    in tests; production code imports the proprietary wrapper only on first
    use.
    """

    def __init__(self, settings, planner=None):
        self.settings = settings
        self._planner = planner

    def _backend(self):
        if self._planner is None:
            from tool.grasp_planning.anygrasp_planner import AnyGraspPlanner

            self._planner = AnyGraspPlanner(
                checkpoint_path=self.settings.checkpoint_path,
                sdk_grasp_dir=self.settings.sdk_grasp_dir,
                max_gripper_width=self.settings.maximum_gripper_width_m,
                gripper_height=self.settings.gripper_height_m,
            )
        return self._planner

    @staticmethod
    def _points_in_camera(object_cloud, base_from_camera):
        points_base = np.asarray(object_cloud.points_base_m, dtype=np.float32).reshape(-1, 3)
        if len(points_base) < 64:
            raise GraspPlanningError("AnyGrasp requires at least 64 object points")
        camera_from_base = np.linalg.inv(as_transform(base_from_camera, "base_from_camera"))
        return (
            camera_from_base[:3, :3].dot(points_base.T).T
            + camera_from_base[:3, 3]
        ), camera_from_base

    def target_from_object(
        self, object_cloud, object_id=None, *, base_from_camera=None
    ):
        if not bool(getattr(object_cloud, "valid", False)):
            raise GraspPlanningError(
                "object cloud is invalid: {}".format(getattr(object_cloud, "reason", ""))
            )
        if base_from_camera is None:
            raise GraspPlanningError(
                "AnyGrasp fallback requires base_from_camera for point-cloud conversion"
            )
        points_camera, camera_from_base = self._points_in_camera(
            object_cloud, base_from_camera
        )
        approach_camera = camera_from_base[:3, :3].dot(
            np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        )
        try:
            candidates = self._backend().plan(
                points_camera,
                approach_camera=approach_camera,
                approach_thresh=np.deg2rad(
                    self.settings.maximum_approach_deviation_deg
                ),
                dense_grasp=self.settings.dense_grasp,
                collision_detection=self.settings.collision_detection,
                top_k=self.settings.top_k,
            )
        except Exception as error:
            raise GraspPlanningError("AnyGrasp fallback unavailable: {}".format(error)) from error
        if not candidates:
            raise GraspPlanningError("AnyGrasp fallback returned no grasp candidates")

        base_from_camera = as_transform(base_from_camera, "base_from_camera")
        downward = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        cosine_limit = np.cos(np.deg2rad(self.settings.maximum_approach_deviation_deg))
        accepted = []
        for candidate in candidates:
            width = float(candidate.width)
            score = float(candidate.score)
            if not np.isfinite(width) or not np.isfinite(score) or width <= 0.0:
                continue
            if width > float(self.settings.maximum_gripper_width_m):
                continue
            if score < float(self.settings.minimum_score):
                continue
            rotation_camera = np.asarray(candidate.rotation, dtype=np.float64).reshape(3, 3)
            translation_camera = np.asarray(candidate.translation, dtype=np.float64).reshape(3)
            rotation_base = base_from_camera[:3, :3].dot(rotation_camera)
            translation_base = (
                base_from_camera[:3, :3].dot(translation_camera)
                + base_from_camera[:3, 3]
            )
            approach_base = rotation_base[:, 0]
            if float(np.dot(approach_base, downward)) < cosine_limit:
                continue
            accepted.append((score, width, rotation_base, translation_base))
        if not accepted:
            raise GraspPlanningError(
                "AnyGrasp candidates did not pass score, width, or top-down filters"
            )
        _, width, rotation_base, translation_base = max(accepted, key=lambda item: item[0])

        # AnyGrasp uses local X=approach and local Y=jaw axis.  The competition
        # contract uses local Z=approach and local Y=jaw axis.
        approach = rotation_base[:, 0]
        jaw_axis = rotation_base[:, 1]
        lateral = np.cross(jaw_axis, approach)
        base_from_grasp = np.column_stack([lateral, jaw_axis, approach])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = base_from_grasp
        transform[:3, 3] = translation_base
        return GraspTarget(
            object_id=(
                object_id
                if object_id is not None
                else getattr(object_cloud, "class_name", "object") or "object"
            ),
            base_from_grasp=transform,
            width_m=width,
            score=max(0.0, float(max(accepted, key=lambda item: item[0])[0])),
            source="anygrasp_fallback",
        )


class FallbackGraspPlanner:
    """Try the deterministic planner first, then an optional fallback."""

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback

    def target_from_object(self, object_cloud, object_id=None, **context):
        try:
            return self.primary.target_from_object(object_cloud, object_id)
        except GraspPlanningError as primary_error:
            if self.fallback is None:
                raise
            try:
                return self.fallback.target_from_object(
                    object_cloud, object_id, **context
                )
            except GraspPlanningError as fallback_error:
                raise GraspPlanningError(
                    "primary deterministic planner failed ({}); AnyGrasp fallback failed ({})".format(
                        primary_error, fallback_error
                    )
                ) from fallback_error

    def plan(self, target, place_position_base_m):
        return self.primary.plan(target, place_position_base_m)


def planner_settings_from_config(config):
    data = config.data if hasattr(config, "data") else config
    entry = data["grasp_planning"]
    return TopDownPlannerSettings(
        tcp_from_grasp=np.asarray(entry["tcp_from_grasp"]["matrix"], dtype=np.float64),
        minimum_grasp_width_m=float(entry["minimum_grasp_width_mm"]) / 1000.0,
        maximum_grasp_width_m=float(entry["maximum_grasp_width_mm"]) / 1000.0,
        width_margin_m=float(entry["width_margin_mm"]) / 1000.0,
        pregrasp_clearance_m=float(entry["pregrasp_clearance_mm"]) / 1000.0,
        lift_distance_m=float(entry["lift_distance_mm"]) / 1000.0,
        place_clearance_m=float(entry["place_clearance_mm"]) / 1000.0,
    )


def planner_from_config(config):
    """Build the default planner and its explicitly configured fallback."""
    data = config.data if hasattr(config, "data") else config
    settings = planner_settings_from_config(data)
    primary = TopDownGraspPlanner(settings)
    fallback_config = data.get("grasp_planning", {}).get("fallback", {})
    if not bool(fallback_config.get("enabled", False)):
        return FallbackGraspPlanner(primary)
    backend = str(fallback_config.get("backend", "anygrasp")).strip().lower()
    if backend != "anygrasp":
        raise ValueError("unsupported grasp fallback backend: {}".format(backend))
    fallback = AnyGraspFallbackPlanner(
        AnyGraspFallbackSettings(
            sdk_grasp_dir=str(fallback_config.get("sdk_grasp_dir", "")),
            checkpoint_path=str(fallback_config.get("checkpoint_path", "")),
            minimum_score=float(fallback_config.get("minimum_score", 0.01)),
            maximum_gripper_width_m=float(
                min(
                    fallback_config.get("maximum_gripper_width_m", 0.08),
                    settings.maximum_grasp_width_m,
                )
            ),
            gripper_height_m=float(fallback_config.get("gripper_height_m", 0.03)),
            top_k=int(fallback_config.get("top_k", 20)),
            maximum_approach_deviation_deg=float(
                fallback_config.get("maximum_approach_deviation_deg", 45.0)
            ),
            collision_detection=bool(fallback_config.get("collision_detection", True)),
            dense_grasp=bool(fallback_config.get("dense_grasp", False)),
        )
    )
    return FallbackGraspPlanner(primary, fallback)


__all__ = [
    "AnyGraspFallbackPlanner", "AnyGraspFallbackSettings", "FallbackGraspPlanner",
    "GraspPlan", "GraspPlanningError", "GraspTarget", "TopDownGraspPlanner",
    "TopDownPlannerSettings", "planner_from_config", "planner_settings_from_config",
]
