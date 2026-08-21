import unittest
from pathlib import Path

import cv2
import numpy as np
from tool.camera_calibration.calib_common import load_layout, tag_world_corners
from tool.camera_calibration.hybrid_localization import TagMapPoseEstimator


class GeminiTagLayoutTests(unittest.TestCase):
    def test_pair_layout_accepts_two_tags_and_recovers_pose(self):
        layout_path = Path(__file__).resolve().parents[1] / "config" / "gemini_tag_layout.yaml"
        layout = load_layout(str(layout_path))
        self.assertEqual(float(layout["tag_size_mm"]), 75.0)
        self.assertEqual(int(layout["minimum_calibration_tags"]), 2)
        left = np.asarray(layout["calibration_tags"][0]["origin_mm"], dtype=float)
        right = np.asarray(layout["calibration_tags"][1]["origin_mm"], dtype=float)
        # Equal tag size/yaw makes top-left and bottom-right spacing identical.
        self.assertAlmostEqual(float(np.linalg.norm((right - left)[:2])), 150.0)

        camera_matrix = np.asarray(
            [[720.0, 0.0, 320.0], [0.0, 720.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros((5, 1), dtype=np.float64)
        rvec = np.asarray([[2.88], [0.14], [-0.09]], dtype=np.float64)
        tvec = np.asarray([[0.04], [-0.03], [0.80]], dtype=np.float64)
        detections = {}
        for tag_id, entry in layout["calibration_tags"].items():
            points = tag_world_corners(
                np.asarray(entry["origin_mm"], dtype=np.float64) / 1000.0,
                entry.get("yaw_deg", 0.0),
                0.075,
            )
            pixels, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion)
            detections[int(tag_id)] = pixels.reshape(4, 2)
        estimate = TagMapPoseEstimator(layout, minimum_tags=2).estimate(
            detections, camera_matrix, distortion
        )
        self.assertTrue(estimate.valid, estimate.reason)
        self.assertEqual(estimate.visible_tag_ids, (0, 1))
        self.assertLess(float(estimate.rms_reprojection_error_px), 1e-5)


if __name__ == "__main__":
    unittest.main()
