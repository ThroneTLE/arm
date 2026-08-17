"""Eye-in-hand calibration from mapped Tags and synchronized TCP poses."""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .geometry import as_transform, average_transforms, rotation_angle_deg
from .localization import AprilTagLocalizer


@dataclass
class HandEyeSample:
    base_from_tcp: np.ndarray
    base_from_camera: np.ndarray
    visible_tag_ids: tuple
    rms_reprojection_error_px: float

    @property
    def tcp_from_camera(self):
        return np.linalg.inv(self.base_from_tcp) @ self.base_from_camera


@dataclass
class HandEyeResult:
    tcp_from_camera: np.ndarray
    total_samples: int
    inlier_indices: tuple
    translation_rms_mm: float
    rotation_rms_deg: float
    translation_max_mm: float
    rotation_max_deg: float


class HandEyeCalibrator:
    """Solve T_tcp_camera independently per capture, then robustly aggregate.

    This absolute-target formulation uses the robot-base Tag map. It is less
    error-prone at a competition than exchanging target/gripper directions in
    cv2.calibrateHandEye, and supports a different visible Tag subset per view.
    """

    def __init__(self, config):
        self.config = config
        self.localizer = AprilTagLocalizer(config)
        self.samples = []

    def add_sample(self, base_from_tcp, base_from_camera, visible_tag_ids=(), rms_px=0.0):
        sample = HandEyeSample(
            as_transform(base_from_tcp, "base_from_tcp"),
            as_transform(base_from_camera, "base_from_camera"),
            tuple(int(value) for value in visible_tag_ids),
            float(rms_px),
        )
        self.samples.append(sample)
        return sample

    def add_image_sample(self, image, base_from_tcp, camera_matrix, distortion):
        estimate = self.localizer.estimate(image, camera_matrix, distortion)
        if not estimate.valid:
            raise ValueError("visual sample rejected: {}".format(estimate.reason))
        return self.add_sample(
            base_from_tcp,
            estimate.base_from_camera,
            estimate.used_tag_ids,
            estimate.rms_reprojection_error_px,
        )

    def solve(self):
        settings = self.config.data.get("hand_eye", {})
        minimum = max(3, int(settings.get("minimum_samples", 8)))
        if len(self.samples) < minimum:
            raise ValueError("need at least {} hand-eye samples; got {}".format(minimum, len(self.samples)))
        transforms = [sample.tcp_from_camera for sample in self.samples]
        max_translation = float(settings.get("max_sample_translation_error_mm", 8.0))
        max_rotation = float(settings.get("max_sample_rotation_error_deg", 3.0))
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
                },
            }
        )
        self.config.save()
