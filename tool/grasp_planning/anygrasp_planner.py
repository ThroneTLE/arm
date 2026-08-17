"""AnyGrasp SDK wrapper: point cloud -> grasp candidates, then to workspace frame.

The AnyGrasp SDK (``gsnet``) is a precompiled Python extension that requires
the license files and the detection checkpoint. Everything here loads lazily so
the rest of the tool remains usable while the license is pending.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class GraspCandidate:
    """One parallel-jaw grasp in the camera coordinate frame (meters)."""

    translation: np.ndarray  # (3,) grasp center
    rotation: np.ndarray  # (3, 3) graspnetAPI convention: X=approach, Y=open/close
    width: float  # required gripper opening
    score: float  # confidence, larger is better
    depth: float  # insertion depth
    height: float  # finger height

    @property
    def approach(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)[:, 0]

    @property
    def tip(self) -> np.ndarray:
        """Gripper tip position; translation is NOT the tip in graspnetAPI."""
        return np.asarray(self.translation, dtype=np.float64).reshape(3) + float(self.depth) * self.approach


@dataclass
class WorkspaceGrasp:
    """A grasp candidate transformed into the AprilTag workspace frame."""

    translation: np.ndarray
    rotation: np.ndarray
    tip: np.ndarray
    width: float
    score: float
    depth: float
    height: float

    @property
    def approach(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)[:, 0]


class AnyGraspUnavailableError(RuntimeError):
    """Raised when the SDK, license, or checkpoint cannot be loaded."""


class AnyGraspPlanner:
    """Lazy loader around the AnyGrasp SDK ``create_detector``/``get_grasp``."""

    def __init__(
        self,
        checkpoint_path: str,
        sdk_grasp_dir: str,
        max_gripper_width: float = 0.08,
        gripper_height: float = 0.03,
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.sdk_grasp_dir = Path(sdk_grasp_dir).expanduser()
        self.max_gripper_width = min(max(float(max_gripper_width), 0.001), 0.1)
        self.gripper_height = float(gripper_height)
        self._detector = None
        self.load_error: Optional[str] = None

    def _load(self):
        if self._detector is not None:
            return self._detector
        if not self.checkpoint_path.is_file():
            raise AnyGraspUnavailableError(
                "AnyGrasp checkpoint 不存在: {}（把邮件中的 checkpoint_detection.tar 放到该路径）".format(
                    self.checkpoint_path
                )
            )
        license_dir = self.sdk_grasp_dir / "license"
        if not license_dir.is_dir():
            raise AnyGraspUnavailableError(
                "AnyGrasp license 目录不存在: {}（把邮件中的 license 解压到 grasp_detection/license）".format(
                    license_dir
                )
            )
        import sys

        sdk_dir = str(self.sdk_grasp_dir.resolve())
        if sdk_dir not in sys.path:
            sys.path.insert(0, sdk_dir)
        # `gsnet` is a precompiled cp39 extension requiring torch + MinkowskiEngine.
        from gsnet import create_detector

        from argparse import Namespace

        config = Namespace(
            checkpoint_path=str(self.checkpoint_path.resolve()),
            max_gripper_width=self.max_gripper_width,
            gripper_height=self.gripper_height,
        )
        detector = create_detector(config)
        if detector is None:
            raise AnyGraspUnavailableError(
                "AnyGrasp create_detector 返回 None：license 校验失败或 checkpoint 不匹配"
            )
        self._detector = detector
        self.load_error = None
        return detector

    def ready(self) -> bool:
        try:
            self._load()
            return True
        except Exception as error:  # keep the node alive and retry later
            self.load_error = str(error)
            return False

    def plan(
        self,
        points_camera: np.ndarray,
        approach_camera: Optional[np.ndarray] = None,
        approach_thresh: float = np.pi,
        dense_grasp: bool = False,
        collision_detection: bool = True,
        region_steering: Optional[np.ndarray] = None,
        top_k: int = 20,
    ) -> List[GraspCandidate]:
        """Run AnyGrasp on a camera-frame point cloud (N,3) float32 meters.

        ``approach_camera`` steers the grasp X axis in the camera frame; the
        threshold is the maximum allowed angular deviation in radians.
        """
        detector = self._load()
        points = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
        if len(points) < 64:
            return []
        steering = None
        if region_steering is not None:
            steering = np.asarray(region_steering).reshape(-1)
            if len(steering) != len(points):
                raise ValueError(
                    "region_steering length {} does not match point count {}".format(
                        len(steering), len(points)
                    )
                )
        optional = {
            "dense_grasp": bool(dense_grasp),
            "collision_detection": bool(collision_detection),
            "region_steering": steering,
            "approach_steering": (
                None
                if approach_camera is None
                else np.asarray(approach_camera, dtype=np.float64).reshape(3)
            ),
            "approach_thresh": float(approach_thresh),
        }
        group = detector.get_grasp(points, optional)
        if group is None or len(group) == 0:
            return []
        if not dense_grasp:
            group = group.nms()
        group = group.sort_by_score()
        candidates = []
        for grasp in group[: max(1, int(top_k))]:
            candidates.append(
                GraspCandidate(
                    translation=np.asarray(grasp.translation, dtype=np.float64).reshape(3),
                    rotation=np.asarray(grasp.rotation_matrix, dtype=np.float64).reshape(3, 3),
                    width=float(grasp.width),
                    score=float(grasp.score),
                    depth=float(grasp.depth),
                    height=float(grasp.height),
                )
            )
        return candidates


def transform_to_workspace(
    candidates: List[GraspCandidate], workspace_from_camera: np.ndarray
) -> List[WorkspaceGrasp]:
    """Transform camera-frame candidates into the workspace frame."""
    transform = np.asarray(workspace_from_camera, dtype=np.float64).reshape(4, 4)
    rotation_ws = transform[:3, :3]
    translation_ws = transform[:3, 3]
    output = []
    for candidate in candidates:
        rotation = np.asarray(candidate.rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(candidate.translation, dtype=np.float64).reshape(3)
        tip = np.asarray(candidate.tip, dtype=np.float64).reshape(3)
        output.append(
            WorkspaceGrasp(
                translation=rotation_ws @ translation + translation_ws,
                rotation=rotation_ws @ rotation,
                tip=rotation_ws @ tip + translation_ws,
                width=float(candidate.width),
                score=float(candidate.score),
                depth=float(candidate.depth),
                height=float(candidate.height),
            )
        )
    return output


def filter_top_down(
    candidates: List[WorkspaceGrasp],
    workspace_up: np.ndarray = np.asarray([0.0, 0.0, -1.0]),
    max_deviation_deg: float = 45.0,
) -> List[WorkspaceGrasp]:
    """Keep grasps whose approach direction points downward (against workspace up)."""
    up = np.asarray(workspace_up, dtype=np.float64).reshape(3)
    up = up / np.linalg.norm(up)
    limit = float(np.cos(np.deg2rad(max_deviation_deg)))
    kept = []
    for grasp in candidates:
        if float(np.dot(grasp.approach, up)) <= -limit:
            kept.append(grasp)
    return kept


def filter_by_width(
    candidates: List[WorkspaceGrasp],
    minimum_width: float = 0.0,
    maximum_width: float = 0.1,
) -> List[WorkspaceGrasp]:
    return [
        grasp
        for grasp in candidates
        if float(minimum_width) <= grasp.width <= float(maximum_width)
    ]


def filter_by_score(
    candidates: List[WorkspaceGrasp], minimum_score: float = 0.0
) -> List[WorkspaceGrasp]:
    return [grasp for grasp in candidates if grasp.score >= float(minimum_score)]
