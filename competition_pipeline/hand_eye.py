"""Eye-in-hand calibration from mapped Tags and synchronized TCP poses."""

from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np

from .checkerboard_target import CHECKERBOARD_TARGET, CheckerboardTarget
from .geometry import as_transform, average_transforms, rotation_angle_deg
from .localization import AprilTagLocalizer


APRILTAG_MAP_TARGET = "apriltag_map"


@dataclass
class HandEyeSample:
    base_from_tcp: np.ndarray
    base_from_camera: object = None
    visible_tag_ids: tuple = ()
    rms_reprojection_error_px: float = 0.0
    target_type: str = APRILTAG_MAP_TARGET
    camera_from_target: object = None
    target_corner_count: int = 0

    @property
    def tcp_from_camera(self):
        if self.base_from_camera is None:
            raise ValueError("absolute camera pose is unavailable for this target")
        return np.linalg.inv(self.base_from_tcp) @ self.base_from_camera

    @property
    def target_label(self):
        if self.target_type == CHECKERBOARD_TARGET:
            return "棋盘格 {} 角点".format(self.target_corner_count)
        return "Tag {}".format(",".join(map(str, self.visible_tag_ids)))


@dataclass
class HandEyeResult:
    tcp_from_camera: np.ndarray
    total_samples: int
    inlier_indices: tuple
    translation_rms_mm: float
    rotation_rms_deg: float
    translation_max_mm: float
    rotation_max_deg: float
    target_type: str = APRILTAG_MAP_TARGET


class HandEyeCalibrator:
    """Solve T_tcp_camera independently per capture, then robustly aggregate.

    This absolute-target formulation uses the robot-base Tag map. It is less
    error-prone at a competition than exchanging target/gripper directions in
    cv2.calibrateHandEye, and supports a different visible Tag subset per view.
    """

    def __init__(self, config):
        self.config = config
        self.localizer = AprilTagLocalizer(config)
        target = config.data.get("hand_eye", {}).get("calibration_target", {})
        self.target_type = str(target.get("type", APRILTAG_MAP_TARGET))
        # A deliberately unconfigured checkerboard must not disable the
        # independent AprilTag workflow. It is validated only when selected.
        self.checkerboard = (
            CheckerboardTarget(target.get("checkerboard", {}))
            if self.target_type == CHECKERBOARD_TARGET else None
        )
        self.samples = []

    def add_sample(self, base_from_tcp, base_from_camera, visible_tag_ids=(), rms_px=0.0):
        sample = HandEyeSample(
            as_transform(base_from_tcp, "base_from_tcp"),
            as_transform(base_from_camera, "base_from_camera"),
            tuple(int(value) for value in visible_tag_ids),
            float(rms_px),
            APRILTAG_MAP_TARGET,
        )
        self.samples.append(sample)
        return sample

    def add_checkerboard_sample(
        self, base_from_tcp, camera_from_target, rms_px=0.0, corner_count=None,
    ):
        sample = HandEyeSample(
            base_from_tcp=as_transform(base_from_tcp, "base_from_tcp"),
            rms_reprojection_error_px=float(rms_px),
            target_type=CHECKERBOARD_TARGET,
            camera_from_target=as_transform(
                camera_from_target, "camera_from_checkerboard"
            ),
            target_corner_count=(
                self.checkerboard.corner_count
                if corner_count is None else int(corner_count)
            ),
        )
        self.samples.append(sample)
        return sample

    def add_image_sample(self, image, base_from_tcp, camera_matrix, distortion):
        if self.target_type == CHECKERBOARD_TARGET:
            observation = self.checkerboard.estimate(
                image, camera_matrix, distortion
            )
            if not observation.valid:
                raise ValueError(
                    "visual sample rejected: {}".format(observation.reason)
                )
            return self.add_checkerboard_sample(
                base_from_tcp,
                observation.camera_from_board,
                observation.rms_reprojection_error_px,
                self.checkerboard.corner_count,
            )
        estimate = self.localizer.estimate(image, camera_matrix, distortion)
        if not estimate.valid:
            raise ValueError("visual sample rejected: {}".format(estimate.reason))
        return self.add_sample(
            base_from_tcp,
            estimate.base_from_camera,
            estimate.used_tag_ids,
            estimate.rms_reprojection_error_px,
        )

    def add_stored_sample(self, entry):
        target_type = str(entry.get("target_type", APRILTAG_MAP_TARGET))
        if target_type == CHECKERBOARD_TARGET:
            return self.add_checkerboard_sample(
                entry["base_from_tcp"], entry["camera_from_target"],
                entry.get("rms_reprojection_error_px", 0.0),
                entry.get("target_corner_count", self.checkerboard.corner_count),
            )
        if target_type != APRILTAG_MAP_TARGET:
            raise ValueError("unsupported hand-eye sample target: {}".format(target_type))
        return self.add_sample(
            entry["base_from_tcp"], entry["base_from_camera"],
            entry.get("visible_tag_ids", ()),
            entry.get("rms_reprojection_error_px", 0.0),
        )

    @staticmethod
    def _hand_eye_from_samples(samples, indices):
        selected = [samples[index] for index in indices]
        rotations_gripper_to_base = [item.base_from_tcp[:3, :3] for item in selected]
        translations_gripper_to_base = [item.base_from_tcp[:3, 3] for item in selected]
        rotations_target_to_camera = [item.camera_from_target[:3, :3] for item in selected]
        translations_target_to_camera = [item.camera_from_target[:3, 3] for item in selected]
        rotation, translation = cv2.calibrateHandEye(
            rotations_gripper_to_base,
            translations_gripper_to_base,
            rotations_target_to_camera,
            translations_target_to_camera,
            method=cv2.CALIB_HAND_EYE_PARK,
        )
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return as_transform(result, "tcp_from_camera")

    @staticmethod
    def _checkerboard_residuals(samples, tcp_from_camera):
        base_from_targets = [
            sample.base_from_tcp @ tcp_from_camera @ sample.camera_from_target
            for sample in samples
        ]
        center = average_transforms(base_from_targets)
        translation_errors = np.asarray([
            np.linalg.norm(item[:3, 3] - center[:3, 3]) * 1000.0
            for item in base_from_targets
        ])
        rotation_errors = np.asarray([
            rotation_angle_deg(center, item) for item in base_from_targets
        ])
        return translation_errors, rotation_errors

    def _solve_checkerboard(self, minimum, max_translation, max_rotation):
        if any(
            sample.target_type != CHECKERBOARD_TARGET
            or sample.camera_from_target is None
            for sample in self.samples
        ):
            raise ValueError("checkerboard session contains incompatible Tag samples")
        settings = self.config.data["hand_eye"]["calibration_target"]["checkerboard"]
        translation_span = max(
            np.linalg.norm(a.base_from_tcp[:3, 3] - b.base_from_tcp[:3, 3]) * 1000.0
            for index, a in enumerate(self.samples)
            for b in self.samples[index + 1:]
        )
        rotation_span = max(
            rotation_angle_deg(a.base_from_tcp, b.base_from_tcp)
            for index, a in enumerate(self.samples)
            for b in self.samples[index + 1:]
        )
        minimum_translation_span = float(
            settings.get("minimum_tcp_translation_span_mm", 30.0)
        )
        minimum_rotation_span = float(
            settings.get("minimum_tcp_rotation_span_deg", 15.0)
        )
        if translation_span < minimum_translation_span or rotation_span < minimum_rotation_span:
            raise ValueError(
                "checkerboard hand-eye poses lack diversity: span {:.1f} mm / {:.1f} deg; "
                "need at least {:.1f} mm / {:.1f} deg".format(
                    translation_span, rotation_span,
                    minimum_translation_span, minimum_rotation_span,
                )
            )
        all_indices = tuple(range(len(self.samples)))
        candidate_subsets = [all_indices]
        if len(self.samples) - 1 >= minimum:
            candidate_subsets.extend(
                tuple(index for index in all_indices if index != excluded)
                for excluded in all_indices
            )
        candidates = []
        for subset in candidate_subsets:
            try:
                estimate = self._hand_eye_from_samples(self.samples, subset)
                translation_errors, rotation_errors = self._checkerboard_residuals(
                    self.samples, estimate
                )
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                continue
            inliers = tuple(
                index for index in all_indices
                if translation_errors[index] <= max_translation
                and rotation_errors[index] <= max_rotation
            )
            cost = float(np.sum(
                translation_errors / max_translation
                + rotation_errors / max_rotation
            ))
            candidates.append((len(inliers), -cost, inliers, estimate))
        if not candidates:
            raise ValueError("OpenCV checkerboard hand-eye solver produced no valid candidate")
        _, _, inliers, estimate = max(candidates, key=lambda item: (item[0], item[1]))
        if len(inliers) < minimum:
            raise ValueError(
                "only {} checkerboard samples pass hand-eye limits; need {}".format(
                    len(inliers), minimum
                )
            )
        for _ in range(3):
            estimate = self._hand_eye_from_samples(self.samples, inliers)
            translation_errors, rotation_errors = self._checkerboard_residuals(
                self.samples, estimate
            )
            updated = tuple(
                index for index in all_indices
                if translation_errors[index] <= max_translation
                and rotation_errors[index] <= max_rotation
            )
            if len(updated) < minimum:
                raise ValueError(
                    "only {} checkerboard samples remain after refinement; need {}".format(
                        len(updated), minimum
                    )
                )
            if updated == inliers:
                break
            inliers = updated
        estimate = self._hand_eye_from_samples(self.samples, inliers)
        translation_errors, rotation_errors = self._checkerboard_residuals(
            [self.samples[index] for index in inliers], estimate
        )
        return HandEyeResult(
            estimate,
            len(self.samples),
            inliers,
            float(np.sqrt(np.mean(translation_errors ** 2))),
            float(np.sqrt(np.mean(rotation_errors ** 2))),
            float(np.max(translation_errors)),
            float(np.max(rotation_errors)),
            CHECKERBOARD_TARGET,
        )

    def solve(self):
        settings = self.config.data.get("hand_eye", {})
        minimum = max(3, int(settings.get("minimum_samples", 8)))
        if len(self.samples) < minimum:
            raise ValueError("need at least {} hand-eye samples; got {}".format(minimum, len(self.samples)))
        max_translation = float(settings.get("max_sample_translation_error_mm", 8.0))
        max_rotation = float(settings.get("max_sample_rotation_error_deg", 3.0))
        if self.target_type == CHECKERBOARD_TARGET:
            return self._solve_checkerboard(
                minimum, max_translation, max_rotation
            )
        if any(sample.target_type != APRILTAG_MAP_TARGET for sample in self.samples):
            raise ValueError("AprilTag session contains incompatible checkerboard samples")
        transforms = [sample.tcp_from_camera for sample in self.samples]
        candidate_sets = []
        for candidate in transforms:
            translation_errors = np.asarray(
                [np.linalg.norm(item[:3, 3] - candidate[:3, 3]) * 1000.0 for item in transforms]
            )
            rotation_errors = np.asarray(
                [rotation_angle_deg(candidate, item) for item in transforms]
            )
            indices = tuple(
                index for index in range(len(transforms))
                if translation_errors[index] <= max_translation and rotation_errors[index] <= max_rotation
            )
            normalized_cost = float(np.sum(translation_errors / max_translation + rotation_errors / max_rotation))
            candidate_sets.append((len(indices), -normalized_cost, indices))
        _, _, inliers = max(candidate_sets)
        if len(inliers) < minimum:
            raise ValueError(
                "no consistent hand-eye sample set has {} members within {} mm / {} deg".format(
                    minimum, max_translation, max_rotation
                )
            )
        estimate = average_transforms([transforms[index] for index in inliers])
        for _ in range(3):
            translation_errors = np.asarray(
                [np.linalg.norm(item[:3, 3] - estimate[:3, 3]) * 1000.0 for item in transforms]
            )
            rotation_errors = np.asarray(
                [rotation_angle_deg(estimate, item) for item in transforms]
            )
            inliers = tuple(
                index for index in range(len(transforms))
                if translation_errors[index] <= max_translation and rotation_errors[index] <= max_rotation
            )
            if len(inliers) < minimum:
                raise ValueError(
                    "only {} samples pass hand-eye limits; need {} (translation <= {} mm, rotation <= {} deg)".format(
                        len(inliers), minimum, max_translation, max_rotation
                    )
                )
            updated = average_transforms([transforms[index] for index in inliers])
            if np.allclose(updated, estimate, atol=1e-12):
                estimate = updated
                break
            estimate = updated
        translation_errors = np.asarray(
            [np.linalg.norm(transforms[index][:3, 3] - estimate[:3, 3]) * 1000.0 for index in inliers]
        )
        rotation_errors = np.asarray(
            [rotation_angle_deg(estimate, transforms[index]) for index in inliers]
        )
        return HandEyeResult(
            estimate,
            len(transforms),
            inliers,
            float(np.sqrt(np.mean(translation_errors ** 2))),
            float(np.sqrt(np.mean(rotation_errors ** 2))),
            float(np.max(translation_errors)),
            float(np.max(rotation_errors)),
            APRILTAG_MAP_TARGET,
        )

    def promote(self, result):
        entry = self.config.data["hand_eye"]["tcp_from_color_camera"]
        entry.update(
            {
                "valid": True,
                "matrix": as_transform(result.tcp_from_camera).tolist(),
                "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "quality": {
                    "total_samples": int(result.total_samples),
                    "inlier_samples": int(len(result.inlier_indices)),
                    "translation_rms_mm": float(result.translation_rms_mm),
                    "rotation_rms_deg": float(result.rotation_rms_deg),
                    "translation_max_mm": float(result.translation_max_mm),
                    "rotation_max_deg": float(result.rotation_max_deg),
                    "calibration_target": str(result.target_type),
                },
            }
        )
        self.config.save()
