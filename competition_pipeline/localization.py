"""AprilTag-first camera localization with a robot TCP fallback."""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import as_transform
from .tag_map import TagMap


SOURCE_TAG_VISUAL = "tag_visual"
SOURCE_TAG_VISUAL_HELD = "tag_visual_held"
SOURCE_TCP_FALLBACK = "tcp_hand_eye"
SOURCE_UNAVAILABLE = "unavailable"


@dataclass
class LocalizationResult:
    valid: bool
    base_from_camera: object
    source: str
    timestamp_s: float
    visible_tag_ids: tuple = ()
    used_tag_ids: tuple = ()
    rms_reprojection_error_px: object = None
    max_reprojection_error_px: object = None
    reason: str = ""


class AprilTagLocalizer:
    def __init__(self, config):
        self.config = config
        self.tag_map = TagMap(config)
        dictionary_name = self.tag_map.dictionary
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError("unsupported AprilTag dictionary: {}".format(dictionary_name))
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, image):
        image = np.asarray(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {
            int(tag_id): np.asarray(points, dtype=np.float64).reshape(4, 2)
            for points, tag_id in zip(corners, ids.reshape(-1))
        }

    def estimate_detections(self, detections, camera_matrix, distortion, timestamp_s=None):
        timestamp_s = time.monotonic() if timestamp_s is None else float(timestamp_s)
        visible = tuple(sorted(int(tag_id) for tag_id in detections))
        usable = tuple(sorted(set(visible).intersection(self.tag_map.ids)))
        settings = self.config.data["localization"]
        minimum = max(1, int(settings.get("minimum_visible_tags", 1)))
        if len(usable) < minimum:
            return LocalizationResult(
                False, None, SOURCE_UNAVAILABLE, timestamp_s, visible, usable,
                reason="mapped Tag count {} is below minimum {}".format(len(usable), minimum),
            )
        object_points = np.concatenate(
            [self.tag_map.corners_base_m(tag_id) for tag_id in usable], axis=0
        ).astype(np.float64)
        image_points = np.concatenate(
            [np.asarray(detections[tag_id], dtype=np.float64).reshape(4, 2) for tag_id in usable],
            axis=0,
        )
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return LocalizationResult(
                False, None, SOURCE_UNAVAILABLE, timestamp_s, visible, usable,
                reason="cv2.solvePnP failed",
            )
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, image_points, camera_matrix, distortion, rvec, tvec
            )
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        rms = float(np.sqrt(np.mean(errors ** 2)))
        maximum = float(np.max(errors))
        camera_from_base = np.eye(4, dtype=np.float64)
        camera_from_base[:3, :3] = cv2.Rodrigues(rvec)[0]
        camera_from_base[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        camera_depths = (camera_from_base[:3, :3] @ object_points.T).T[:, 2] + camera_from_base[2, 3]
        if np.any(camera_depths <= 0.0):
            return LocalizationResult(
                False, None, SOURCE_UNAVAILABLE, timestamp_s, visible, usable, rms, maximum,
                "PnP solution places mapped corners behind the camera",
            )
        max_rms = float(settings.get("max_rms_reprojection_error_px", 2.5))
        max_corner = float(settings.get("max_corner_reprojection_error_px", 6.0))
        if rms > max_rms or maximum > max_corner:
            return LocalizationResult(
                False, None, SOURCE_UNAVAILABLE, timestamp_s, visible, usable, rms, maximum,
                "Tag reprojection error exceeds limits (RMS {:.3f}/{:.3f}, max {:.3f}/{:.3f} px)".format(
                    rms, max_rms, maximum, max_corner
                ),
            )
        return LocalizationResult(
            True, np.linalg.inv(camera_from_base), SOURCE_TAG_VISUAL, timestamp_s,
            visible, usable, rms, maximum, "mapped Tag pose accepted",
        )

    def estimate(self, image, camera_matrix, distortion, timestamp_s=None):
        return self.estimate_detections(
            self.detect(image), camera_matrix, distortion, timestamp_s=timestamp_s
        )


class HybridLocalizer:
    """Prefer visual pose, with short multi-Tag hysteresis for live stability."""

    def __init__(self, config):
        self.config = config
        self.visual = AprilTagLocalizer(config)
        self._last_visual = None
        self._last_multi_tag_visual = None

    @staticmethod
    def _copy_with_pose(result, pose, source, timestamp_s, reason, used_tag_ids):
        return LocalizationResult(
            True,
            np.asarray(pose, dtype=np.float64).copy(),
            source,
            timestamp_s,
            visible_tag_ids=result.visible_tag_ids,
            used_tag_ids=tuple(used_tag_ids),
            rms_reprojection_error_px=result.rms_reprojection_error_px,
            max_reprojection_error_px=result.max_reprojection_error_px,
            reason=reason,
        )

    def _hold_recent_visual(self, result, timestamp_s, maximum_age_s, label):
        previous = self._last_visual
        if previous is None:
            return result
        age = timestamp_s - previous.timestamp_s
        if age < 0.0 or age > float(maximum_age_s):
            return result
        return self._copy_with_pose(
            result,
            previous.base_from_camera,
            SOURCE_TAG_VISUAL_HELD,
            timestamp_s,
            "{}；保持 {:.0f} ms 前的最近有效视觉位姿".format(
                label, age * 1000.0
            ),
            previous.used_tag_ids,
        )

    def localize(
        self,
        image,
        camera_matrix,
        distortion,
        base_from_tcp=None,
        image_timestamp_s=None,
        robot_timestamp_s=None,
        detections_override=None,
    ):
        timestamp_s = time.monotonic() if image_timestamp_s is None else float(image_timestamp_s)
        settings = self.config.data["localization"]
        use_apriltag_runtime = bool(settings.get("use_apriltag_runtime", False))
        if use_apriltag_runtime:
            if detections_override is None:
                visual = self.visual.estimate(
                    image, camera_matrix, distortion, timestamp_s=timestamp_s
                )
            else:
                visual = self.visual.estimate_detections(
                    detections_override, camera_matrix, distortion, timestamp_s=timestamp_s
                )
        else:
            visual = LocalizationResult(
                False, None, SOURCE_UNAVAILABLE, timestamp_s,
                reason="runtime AprilTag localization is disabled; TCP + hand-eye is required",
            )
        allow_visual_hold = use_apriltag_runtime and detections_override is None
        if visual.valid:
            single_hold_s = float(
                settings.get("single_tag_after_multi_hold_s", 0.8)
            )
            previous_multi = self._last_multi_tag_visual
            if (
                allow_visual_hold
                and len(visual.used_tag_ids) == 1
                and previous_multi is not None
                and 0.0 <= timestamp_s - previous_multi.timestamp_s <= single_hold_s
            ):
                return self._copy_with_pose(
                    visual,
                    previous_multi.base_from_camera,
                    SOURCE_TAG_VISUAL_HELD,
                    timestamp_s,
                    "当前仅 1 个 Tag；短时保持最近双 Tag 位姿，避免解算跳变",
                    previous_multi.used_tag_ids,
                )
            self._last_visual = visual
            if len(visual.used_tag_ids) >= 2:
                self._last_multi_tag_visual = visual
            return visual
        if not bool(settings.get("use_tcp_fallback", True)):
            visual.reason += "; TCP fallback disabled"
            return self._hold_recent_visual(
                visual, timestamp_s,
                settings.get("invalid_visual_hold_s", 0.6),
                "当前视觉定位无效",
            ) if allow_visual_hold else visual
        if not self.config.hand_eye_valid:
            visual.reason += "; hand-eye result is not valid"
            return self._hold_recent_visual(
                visual, timestamp_s,
                settings.get("invalid_visual_hold_s", 0.6),
                "当前视觉定位无效",
            ) if allow_visual_hold else visual
        if base_from_tcp is None or robot_timestamp_s is None:
            visual.reason += "; current TCP pose is unavailable"
            return self._hold_recent_visual(
                visual, timestamp_s,
                settings.get("invalid_visual_hold_s", 0.6),
                "当前视觉定位无效",
            ) if allow_visual_hold else visual
        maximum_age = float(settings.get("maximum_robot_pose_age_s", 0.25))
        age = abs(timestamp_s - float(robot_timestamp_s))
        if age > maximum_age:
            visual.reason += "; TCP pose is stale ({:.3f} s)".format(age)
            return self._hold_recent_visual(
                visual, timestamp_s,
                settings.get("invalid_visual_hold_s", 0.6),
                "当前视觉定位无效",
            ) if allow_visual_hold else visual
        base_from_camera = as_transform(base_from_tcp, "base_from_tcp") @ self.config.tcp_from_color_camera
        return LocalizationResult(
            True,
            base_from_camera,
            SOURCE_TCP_FALLBACK,
            timestamp_s,
            visible_tag_ids=visual.visible_tag_ids,
            used_tag_ids=visual.used_tag_ids,
            rms_reprojection_error_px=visual.rms_reprojection_error_px,
            max_reprojection_error_px=visual.max_reprojection_error_px,
            reason="current controller TCP pose composed with calibrated hand-eye transform",
        )
