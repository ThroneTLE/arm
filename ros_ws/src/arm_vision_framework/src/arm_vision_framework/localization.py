"""TCP + eye-in-hand runtime localization, with calibration-only AprilTags."""

import math
import time

import cv2
import numpy as np

from .types import CameraLocalization, FrameData, RobotState


SOURCE_TAG_VISUAL = "tag_visual"
SOURCE_ROBOT_FALLBACK = "robot_tcp_hand_eye"
SOURCE_SIMULATED_ROBOT = "simulated_robot"
SOURCE_UNAVAILABLE = "unavailable"


def _tag_world_corners(origin_m, yaw_deg, tag_size_m):
    """Return TL, TR, BR, BL from the black-frame top-left origin."""
    origin = np.asarray(origin_m, dtype=np.float64).reshape(3)
    local = np.asarray(
        [[0, 0, 0], [tag_size_m, 0, 0], [tag_size_m, tag_size_m, 0], [0, tag_size_m, 0]],
        dtype=np.float64,
    )
    yaw = math.radians(float(yaw_deg))
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return (rotation @ local.T).T + origin


class AprilTagWorkspaceLocalizer:
    def __init__(self, calibration):
        tag_map = calibration.tag_map
        dictionary_name = str(tag_map.get("dictionary", "DICT_APRILTAG_36h11"))
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError("unsupported AprilTag dictionary: {}".format(dictionary_name))
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary_name)
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.tag_size_m = float(tag_map["tag_size_mm"]) / 1000.0
        self.minimum_visible_tags = max(1, int(tag_map.get("minimum_visible_tags", 1)))
        self.max_rms_px = float(tag_map.get("max_rms_reprojection_error_px", 2.5))
        self.tags = {int(tag_id): entry for tag_id, entry in tag_map["tags"].items()}

    def detect(self, color_bgr):
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {
            int(tag_id): np.asarray(points, dtype=np.float64).reshape(4, 2)
            for points, tag_id in zip(corners, ids.reshape(-1))
        }

    def estimate(self, frame):
        detections = self.detect(frame.color_bgr)
        visible = tuple(sorted(set(detections).intersection(self.tags)))
        if len(visible) < self.minimum_visible_tags:
            return CameraLocalization(
                False, None, SOURCE_UNAVAILABLE, frame.timestamp_s,
                visible_tag_ids=visible,
                reason="not enough workspace Tags are visible",
            )
        object_points = []
        image_points = []
        for tag_id in visible:
            entry = self.tags[tag_id]
            origin_m = np.asarray(entry["origin_mm"], dtype=np.float64) / 1000.0
            object_points.extend(
                _tag_world_corners(
                    origin_m, entry.get("yaw_deg", 0.0), self.tag_size_m
                )
            )
            image_points.extend(detections[tag_id])
        object_points = np.asarray(object_points, dtype=np.float64)
        image_points = np.asarray(image_points, dtype=np.float64)
        planar = float(np.ptp(object_points[:, 2])) < 1e-9
        flag = cv2.SOLVEPNP_IPPE if planar and hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            frame.camera_matrix,
            frame.distortion,
            flags=flag,
        )
        if not ok:
            return CameraLocalization(
                False, None, SOURCE_UNAVAILABLE, frame.timestamp_s,
                visible_tag_ids=visible,
                reason="cv2.solvePnP failed",
            )
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                frame.camera_matrix,
                frame.distortion,
                rvec,
                tvec,
            )
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, frame.camera_matrix, frame.distortion
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        rms = float(np.sqrt(np.mean(errors ** 2)))
        if rms > self.max_rms_px:
            return CameraLocalization(
                False, None, SOURCE_UNAVAILABLE, frame.timestamp_s,
                visible_tag_ids=visible,
                rms_reprojection_error_px=rms,
                reason="Tag reprojection RMS exceeds {:.3f} px".format(self.max_rms_px),
            )
        camera_from_workspace = np.eye(4, dtype=np.float64)
        camera_from_workspace[:3, :3] = cv2.Rodrigues(rvec)[0]
        camera_from_workspace[:3, 3] = np.asarray(tvec).reshape(3)
        return CameraLocalization(
            True,
            np.linalg.inv(camera_from_workspace),
            SOURCE_TAG_VISUAL,
            frame.timestamp_s,
            visible_tag_ids=visible,
            rms_reprojection_error_px=rms,
            reason="workspace Tags accepted",
        )


class HybridCameraLocalizer:
    """Compute camera pose from the controller TCP and hand-eye transform.

    AprilTags stay available in :class:`AprilTagWorkspaceLocalizer` for hand-
    eye calibration and acceptance tests.  They are disabled in the runtime
    path by default: the industrial controller's TCP feedback is the primary
    metric pose source, not a visual fallback.
    """

    def __init__(
        self, calibration, maximum_robot_pose_age_s=0.25,
        use_robot_fallback=True, use_visual_tags=False,
    ):
        self.calibration = calibration
        self.use_visual_tags = bool(use_visual_tags)
        self.visual = (
            AprilTagWorkspaceLocalizer(calibration) if self.use_visual_tags else None
        )
        self.maximum_robot_pose_age_s = float(maximum_robot_pose_age_s)
        self.use_robot_fallback = bool(use_robot_fallback)

    def localize(self, frame: FrameData, robot_state: RobotState) -> CameraLocalization:
        if self.use_visual_tags:
            visual = self.visual.estimate(frame)
            if visual.valid:
                return visual
        else:
            visual = CameraLocalization(
                False, None, SOURCE_UNAVAILABLE, frame.timestamp_s,
                reason="runtime AprilTag localization is disabled by configuration",
            )
        if not self.use_robot_fallback:
            return visual
        if robot_state is None or not robot_state.valid or robot_state.base_from_gripper is None:
            visual.reason += "; robot state is unavailable"
            return visual
        if abs(frame.timestamp_s - robot_state.timestamp_s) > self.maximum_robot_pose_age_s:
            visual.reason += "; robot state is stale"
            return visual
        same_base_frame = (
            self.calibration.data.get("frames", {}).get("workspace")
            == self.calibration.data.get("frames", {}).get("base")
        )
        base_transform_ready = (
            same_base_frame or self.calibration.transform_valid("workspace_from_base")
        )
        hand_eye_ready = self.calibration.transform_valid("gripper_from_camera")
        if (not base_transform_ready or not hand_eye_ready) and not robot_state.simulated:
            visual.reason += "; TCP/hand-eye transforms are not calibrated"
            return visual
        workspace_from_base = (
            np.eye(4, dtype=np.float64)
            if same_base_frame
            else self.calibration.transform(
                "workspace_from_base", require_valid=not robot_state.simulated
            )
        )
        gripper_from_camera = self.calibration.transform(
            "gripper_from_camera", require_valid=not robot_state.simulated
        )
        workspace_from_camera = (
            workspace_from_base
            @ robot_state.base_from_gripper
            @ gripper_from_camera
        )
        return CameraLocalization(
            True,
            workspace_from_camera,
            SOURCE_SIMULATED_ROBOT if robot_state.simulated else SOURCE_ROBOT_FALLBACK,
            frame.timestamp_s,
            visible_tag_ids=visual.visible_tag_ids,
            rms_reprojection_error_px=visual.rms_reprojection_error_px,
            simulated=robot_state.simulated,
            reason=(
                "simulation fallback; not safe for robot execution"
                if robot_state.simulated
                else "current controller TCP pose composed with calibrated hand-eye transform"
            ),
        )
