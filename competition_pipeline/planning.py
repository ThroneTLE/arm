"""Backend-neutral grasp contracts and a deterministic top-down fallback."""

from dataclasses import dataclass
from typing import Optional, Tuple

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


__all__ = [
    "GraspPlan", "GraspPlanningError", "GraspTarget", "TopDownGraspPlanner",
    "TopDownPlannerSettings", "planner_settings_from_config",
]
