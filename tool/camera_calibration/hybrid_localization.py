#!/usr/bin/env python3
"""Tag-first camera localization with a robot-kinematics fallback."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from tool.camera_calibration.calib_common import require_coordinate_convention, tag_world_corners


SOURCE_TAG_VISUAL = "tag_visual"
SOURCE_ROBOT_FALLBACK = "robot_fallback"
SOURCE_SIMULATED_ROBOT = "simulated_robot"
SOURCE_UNAVAILABLE = "unavailable"


def as_transform(matrix: np.ndarray, name: str = "transform") -> np.ndarray:
    """Return a validated 4x4 rigid transform."""
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(transform)):
        raise ValueError("{} contains non-finite values".format(name))
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("{} has an invalid homogeneous row".format(name))
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("{} rotation is not orthonormal".format(name))
    if np.linalg.det(rotation) < 0.999 or np.linalg.det(rotation) > 1.001:
        raise ValueError("{} rotation determinant is not +1".format(name))
    return transform.copy()


def transform_from_xyz_rpy(
    xyz_m: Sequence[float], rpy_deg: Sequence[float]
) -> np.ndarray:
    """Build a transform using fixed-frame ZYX yaw-pitch-roll convention."""
    roll, pitch, yaw = np.radians(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    rotation_y = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rotation_z = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    return transform


def xyz_rpy_from_transform(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return translation in meters and ZYX roll-pitch-yaw in degrees."""
    transform = as_transform(matrix)
    rotation = transform[:3, :3]
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return transform[:3, 3].copy(), np.degrees([roll, pitch, yaw])


def matrix_from_config(entry: dict, name: str) -> np.ndarray:
    if not isinstance(entry, dict) or "matrix" not in entry:
        raise ValueError("{} must contain a matrix".format(name))
    return as_transform(entry["matrix"], name)


def load_hybrid_config(path: str) -> dict:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported hybrid localization schema")
    fallback = data.get("fallback", {})
    matrix_from_config(fallback.get("workspace_from_base", {}), "workspace_from_base")
    matrix_from_config(fallback.get("gripper_from_camera", {}), "gripper_from_camera")
    simulation = data.get("simulation", {})
    if len(simulation.get("base_from_gripper_xyz_mm", [])) != 3:
        raise ValueError("simulation base_from_gripper_xyz_mm must contain three values")
    if len(simulation.get("base_from_gripper_rpy_deg", [])) != 3:
        raise ValueError("simulation base_from_gripper_rpy_deg must contain three values")
    return data


@dataclass
class RobotPoseSample:
    base_from_gripper: np.ndarray
    timestamp_s: float
    simulated: bool = False

    def __post_init__(self) -> None:
        self.base_from_gripper = as_transform(
            self.base_from_gripper, "base_from_gripper"
        )


class RobotPoseProvider:
    """Adapter boundary for a future ROS, SDK, or fieldbus robot driver."""

    def latest_pose(self, now_s: Optional[float] = None) -> Optional[RobotPoseSample]:
        raise NotImplementedError


class ManualRobotPoseProvider(RobotPoseProvider):
    """In-memory provider used by tests and the no-hardware UI simulator."""

    def __init__(self, base_from_gripper: np.ndarray, simulated: bool = True):
        self._base_from_gripper = as_transform(
            base_from_gripper, "base_from_gripper"
        )
        self._simulated = bool(simulated)
        self._available = True
        self._timestamp_s = time.monotonic()

    def set_pose(self, base_from_gripper: np.ndarray) -> None:
        self._base_from_gripper = as_transform(
            base_from_gripper, "base_from_gripper"
        )
        self._timestamp_s = time.monotonic()

    def set_available(self, available: bool) -> None:
        self._available = bool(available)

    def latest_pose(self, now_s: Optional[float] = None) -> Optional[RobotPoseSample]:
        if not self._available:
            return None
        now_s = time.monotonic() if now_s is None else float(now_s)
        # The simulator represents a continuously reported static robot pose.
        timestamp_s = now_s if self._simulated else self._timestamp_s
        return RobotPoseSample(
            self._base_from_gripper.copy(), timestamp_s, self._simulated
        )


@dataclass
class VisualPoseEstimate:
    valid: bool
    workspace_from_camera: Optional[np.ndarray]
    camera_from_workspace: Optional[np.ndarray]
    visible_tag_ids: Tuple[int, ...]
    rms_reprojection_error_px: Optional[float]
    max_reprojection_error_px: Optional[float]
    reason: str


class TagMapPoseEstimator:
    """Estimate the camera pose from any visible subset of a fixed Tag map."""

    def __init__(
        self,
        layout: dict,
        minimum_tags: int = 1,
        max_rms_reprojection_error_px: float = 2.5,
    ):
        require_coordinate_convention(layout, "Tag map")
        self.layout = layout
        self.minimum_tags = max(1, int(minimum_tags))
        self.max_rms_reprojection_error_px = float(
            max_rms_reprojection_error_px
        )
        self.tag_size_m = float(layout["tag_size_mm"]) / 1000.0
        self.tag_entries = {
            int(tag_id): entry
            for tag_id, entry in layout.get("calibration_tags", {}).items()
        }
        if not self.tag_entries:
            raise ValueError("Tag map contains no calibration tags")

    def _correspondences(
        self, detections: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
        visible = tuple(sorted(set(detections).intersection(self.tag_entries)))
        object_points = []
        image_points = []
        for tag_id in visible:
            entry = self.tag_entries[tag_id]
            origin_m = np.asarray(entry["origin_mm"], dtype=np.float64) / 1000.0
            object_points.extend(
                tag_world_corners(
                    origin_m,
                    float(entry.get("yaw_deg", 0.0)),
                    self.tag_size_m,
                )
            )
            image_points.extend(
                np.asarray(detections[tag_id], dtype=np.float64).reshape(4, 2)
            )
        return (
            np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
            np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
            visible,
        )

    def _single_tag_fallback(
        self,
        detections: Dict[int, np.ndarray],
        visible: Tuple[int, ...],
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> Optional[VisualPoseEstimate]:
        """Retry one marker when a second marker is geometrically inconsistent."""
        if self.minimum_tags != 1 or len(visible) <= 1:
            return None
        candidates = []
        for tag_id in visible:
            candidate = self.estimate(
                {int(tag_id): detections[int(tag_id)]},
                camera_matrix,
                distortion,
            )
            if candidate.valid:
                candidates.append(candidate)
        if not candidates:
            return None
        best = min(
            candidates,
            key=lambda item: float(
                item.rms_reprojection_error_px
                if item.rms_reprojection_error_px is not None
                else float("inf")
            ),
        )
        return VisualPoseEstimate(
            True,
            best.workspace_from_camera,
            best.camera_from_workspace,
            best.visible_tag_ids,
            best.rms_reprojection_error_px,
            best.max_reprojection_error_px,
            "single Tag fallback (joint Tag pose rejected): {}".format(best.reason),
        )

    def estimate(
        self,
        detections: Dict[int, np.ndarray],
        camera_matrix: Optional[np.ndarray],
        distortion: Optional[np.ndarray],
    ) -> VisualPoseEstimate:
        object_points, image_points, visible = self._correspondences(detections)
        if len(visible) < self.minimum_tags:
            return VisualPoseEstimate(
                False, None, None, visible, None, None,
                "visible Tag count is below the configured minimum",
            )
        if camera_matrix is None or distortion is None:
            return VisualPoseEstimate(
                False, None, None, visible, None, None,
                "camera intrinsics are unavailable",
            )
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
        planar = float(np.ptp(object_points[:, 2])) < 1e-9
        flag = cv2.SOLVEPNP_IPPE if planar and hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=flag,
        )
        if not ok:
            fallback = self._single_tag_fallback(
                detections, visible, camera_matrix, distortion
            )
            if fallback is not None:
                return fallback
            return VisualPoseEstimate(
                False, None, None, visible, None, None, "cv2.solvePnP failed"
            )
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - image_points, axis=1
        )
        rms = float(np.sqrt(np.mean(errors ** 2)))
        maximum = float(errors.max())
        if rms > self.max_rms_reprojection_error_px:
            fallback = self._single_tag_fallback(
                detections, visible, camera_matrix, distortion
            )
            if fallback is not None:
                return fallback
            return VisualPoseEstimate(
                False,
                None,
                None,
                visible,
                rms,
                maximum,
                "Tag reprojection RMS exceeds the configured limit",
            )
        rotation, _ = cv2.Rodrigues(rvec)
        camera_from_workspace = np.eye(4, dtype=np.float64)
        camera_from_workspace[:3, :3] = rotation
        camera_from_workspace[:3, 3] = np.asarray(tvec).reshape(3)
        workspace_from_camera = np.linalg.inv(camera_from_workspace)
        return VisualPoseEstimate(
            True,
            workspace_from_camera,
            camera_from_workspace,
            visible,
            rms,
            maximum,
            "Tag pose accepted",
        )


@dataclass
class LocalizationEstimate:
    valid: bool
    workspace_from_camera: Optional[np.ndarray]
    source: str
    timestamp_s: float
    visible_tag_ids: Tuple[int, ...]
    rms_reprojection_error_px: Optional[float]
    simulated: bool
    reason: str


class HybridCameraLocalizer:
    """Prefer absolute Tag localization and fall back to robot kinematics."""

    def __init__(
        self,
        tag_estimator: TagMapPoseEstimator,
        robot_pose_provider: Optional[RobotPoseProvider],
        workspace_from_base: np.ndarray,
        gripper_from_camera: np.ndarray,
        hand_eye_calibrated: bool,
        maximum_robot_pose_age_s: float = 0.25,
    ):
        self.tag_estimator = tag_estimator
        self.robot_pose_provider = robot_pose_provider
        self.workspace_from_base = as_transform(
            workspace_from_base, "workspace_from_base"
        )
        self.gripper_from_camera = as_transform(
            gripper_from_camera, "gripper_from_camera"
        )
        self.hand_eye_calibrated = bool(hand_eye_calibrated)
        self.maximum_robot_pose_age_s = float(maximum_robot_pose_age_s)

    def update(
        self,
        detections: Dict[int, np.ndarray],
        camera_matrix: Optional[np.ndarray],
        distortion: Optional[np.ndarray],
        timestamp_s: Optional[float] = None,
    ) -> LocalizationEstimate:
        timestamp_s = time.monotonic() if timestamp_s is None else float(timestamp_s)
        visual = self.tag_estimator.estimate(
            detections, camera_matrix, distortion
        )
        if visual.valid:
            return LocalizationEstimate(
                True,
                visual.workspace_from_camera,
                SOURCE_TAG_VISUAL,
                timestamp_s,
                visual.visible_tag_ids,
                visual.rms_reprojection_error_px,
                False,
                visual.reason,
            )

        if self.robot_pose_provider is not None:
            robot_pose = self.robot_pose_provider.latest_pose(timestamp_s)
            if robot_pose is not None:
                age_s = timestamp_s - float(robot_pose.timestamp_s)
                if abs(age_s) <= self.maximum_robot_pose_age_s:
                    if self.hand_eye_calibrated or robot_pose.simulated:
                        workspace_from_camera = (
                            self.workspace_from_base
                            @ robot_pose.base_from_gripper
                            @ self.gripper_from_camera
                        )
                        source = (
                            SOURCE_SIMULATED_ROBOT
                            if robot_pose.simulated
                            else SOURCE_ROBOT_FALLBACK
                        )
                        reason = (
                            "simulation only; hand-eye transform is not validated"
                            if robot_pose.simulated and not self.hand_eye_calibrated
                            else "Tag pose unavailable; robot kinematics fallback active"
                        )
                        return LocalizationEstimate(
                            True,
                            workspace_from_camera,
                            source,
                            timestamp_s,
                            visual.visible_tag_ids,
                            visual.rms_reprojection_error_px,
                            robot_pose.simulated,
                            reason,
                        )
                    reason = "hand-eye transform has not been calibrated"
                else:
                    reason = "robot pose is stale"
            else:
                reason = "robot pose provider has no current sample"
        else:
            reason = "robot pose provider is not configured"
        return LocalizationEstimate(
            False,
            None,
            SOURCE_UNAVAILABLE,
            timestamp_s,
            visual.visible_tag_ids,
            visual.rms_reprojection_error_px,
            False,
            "{}; {}".format(visual.reason, reason),
        )
