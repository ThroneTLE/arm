#!/usr/bin/env python3
"""AprilTag-based camera pose provider for object scanning sessions."""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from tool.camera_calibration.calib_common import april_detector, load_layout
from tool.camera_calibration.hybrid_localization import (
    TagMapPoseEstimator,
    VisualPoseEstimate,
)


class TagPoseProvider:
    def __init__(
        self,
        layout_path: str,
        minimum_tags: int = 1,
        maximum_rms_px: float = 2.5,
    ):
        self.layout_path = Path(layout_path).expanduser().resolve()
        self.layout = load_layout(str(self.layout_path))
        self.detector = april_detector(self.layout["dictionary"])
        self.estimator = TagMapPoseEstimator(
            self.layout,
            minimum_tags=minimum_tags,
            max_rms_reprojection_error_px=maximum_rms_px,
        )

    def detect(self, color_bgr: np.ndarray) -> Dict[int, np.ndarray]:
        gray = cv2.cvtColor(np.asarray(color_bgr), cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {
            int(tag_id): np.asarray(corner, dtype=np.float64).reshape(4, 2)
            for tag_id, corner in zip(ids.reshape(-1), corners)
        }

    def estimate(
        self,
        color_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> Tuple[VisualPoseEstimate, Dict[int, np.ndarray]]:
        detections = self.detect(color_bgr)
        estimate = self.estimator.estimate(detections, camera_matrix, distortion)
        return estimate, detections

    @staticmethod
    def draw_status(
        color_bgr: np.ndarray,
        detections: Dict[int, np.ndarray],
        estimate: VisualPoseEstimate,
    ) -> np.ndarray:
        output = np.asarray(color_bgr).copy()
        for tag_id, corners in detections.items():
            polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [polygon], True, (46, 204, 113), 2, cv2.LINE_AA)
            origin = tuple(np.rint(corners[0]).astype(int))
            cv2.putText(
                output,
                "ID {}".format(tag_id),
                (origin[0] + 5, origin[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (46, 204, 113),
                2,
                cv2.LINE_AA,
            )
        status = "TAG POSE: INVALID"
        color = (40, 40, 220)
        if estimate.valid:
            status = "TAG POSE: OK  RMS {:.2f}px".format(
                estimate.rms_reprojection_error_px
            )
            color = (46, 204, 113)
        cv2.rectangle(
            output,
            (8, 8),
            (min(output.shape[1] - 8, 630), 68),
            (25, 30, 34),
            -1,
        )
        cv2.putText(
            output,
            status,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        if not estimate.valid:
            reason = str(estimate.reason or "no valid Tag pose")
            if len(reason) > 78:
                reason = reason[:75] + "..."
            cv2.putText(
                output,
                reason,
                (18, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
        return output
