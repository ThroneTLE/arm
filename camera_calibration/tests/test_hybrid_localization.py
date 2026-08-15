#!/usr/bin/env python3

import unittest
import time

import cv2
import numpy as np

from calib_common import COORDINATE_CONVENTION, tag_world_corners
from hybrid_localization import (
    HybridCameraLocalizer,
    ManualRobotPoseProvider,
    SOURCE_SIMULATED_ROBOT,
    SOURCE_TAG_VISUAL,
    SOURCE_UNAVAILABLE,
    TagMapPoseEstimator,
    transform_from_xyz_rpy,
    xyz_rpy_from_transform,
)


class HybridLocalizationTest(unittest.TestCase):
    def setUp(self):
        self.layout = {
            "schema_version": 2,
            "coordinate_convention": dict(COORDINATE_CONVENTION),
            "tag_size_mm": 70.0,
            "calibration_tags": {
                100: {"origin_mm": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
                101: {"origin_mm": [150.0, 0.0, 0.0], "yaw_deg": 0.0},
                102: {"origin_mm": [0.0, 130.0, 0.0], "yaw_deg": 0.0},
            },
        }
        self.camera_matrix = np.asarray(
            [[900.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.distortion = np.zeros((5, 1), dtype=np.float64)
        self.rvec = np.asarray([[2.85], [0.18], [-0.12]], dtype=np.float64)
        self.tvec = np.asarray([[0.02], [-0.03], [0.72]], dtype=np.float64)
        self.detections = {}
        for tag_id, entry in self.layout["calibration_tags"].items():
            corners = tag_world_corners(
                np.asarray(entry["origin_mm"]) / 1000.0,
                entry["yaw_deg"],
                0.07,
            )
            pixels, _ = cv2.projectPoints(
                corners, self.rvec, self.tvec, self.camera_matrix, self.distortion
            )
            self.detections[tag_id] = pixels.reshape(4, 2)

    def test_xyz_rpy_round_trip(self):
        transform = transform_from_xyz_rpy([0.12, -0.4, 0.8], [15.0, -20.0, 35.0])
        xyz, rpy = xyz_rpy_from_transform(transform)
        np.testing.assert_allclose(xyz, [0.12, -0.4, 0.8], atol=1e-12)
        np.testing.assert_allclose(rpy, [15.0, -20.0, 35.0], atol=1e-10)

    def test_tag_pose_recovers_workspace_from_camera(self):
        estimator = TagMapPoseEstimator(self.layout, minimum_tags=1)
        result = estimator.estimate(
            self.detections, self.camera_matrix, self.distortion
        )
        self.assertTrue(result.valid, result.reason)
        expected_camera_from_workspace = np.eye(4)
        expected_camera_from_workspace[:3, :3] = cv2.Rodrigues(self.rvec)[0]
        expected_camera_from_workspace[:3, 3] = self.tvec.reshape(3)
        np.testing.assert_allclose(
            result.workspace_from_camera,
            np.linalg.inv(expected_camera_from_workspace),
            atol=1e-7,
        )
        self.assertLess(result.rms_reprojection_error_px, 1e-6)

    def test_one_visible_tag_can_localize(self):
        estimator = TagMapPoseEstimator(self.layout, minimum_tags=1)
        result = estimator.estimate(
            {100: self.detections[100]}, self.camera_matrix, self.distortion
        )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.visible_tag_ids, (100,))
        self.assertLess(result.rms_reprojection_error_px, 1e-5)

    def test_visual_pose_has_priority_over_simulated_robot(self):
        provider = ManualRobotPoseProvider(np.eye(4), simulated=True)
        localizer = HybridCameraLocalizer(
            TagMapPoseEstimator(self.layout),
            provider,
            np.eye(4),
            np.eye(4),
            hand_eye_calibrated=False,
        )
        result = localizer.update(
            self.detections, self.camera_matrix, self.distortion, timestamp_s=10.0
        )
        self.assertEqual(result.source, SOURCE_TAG_VISUAL)
        self.assertFalse(result.simulated)

    def test_robot_fallback_composes_all_transforms(self):
        workspace_from_base = transform_from_xyz_rpy([0.1, 0.2, 0.0], [0, 0, 10])
        base_from_gripper = transform_from_xyz_rpy([0.3, -0.1, 0.5], [5, 20, 0])
        gripper_from_camera = transform_from_xyz_rpy([0.02, 0.0, 0.08], [0, 0, 90])
        provider = ManualRobotPoseProvider(base_from_gripper, simulated=True)
        localizer = HybridCameraLocalizer(
            TagMapPoseEstimator(self.layout),
            provider,
            workspace_from_base,
            gripper_from_camera,
            hand_eye_calibrated=False,
        )
        result = localizer.update(
            {}, self.camera_matrix, self.distortion, timestamp_s=10.0
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.source, SOURCE_SIMULATED_ROBOT)
        self.assertTrue(result.simulated)
        np.testing.assert_allclose(
            result.workspace_from_camera,
            workspace_from_base @ base_from_gripper @ gripper_from_camera,
            atol=1e-12,
        )

    def test_no_tag_and_no_robot_is_unavailable(self):
        localizer = HybridCameraLocalizer(
            TagMapPoseEstimator(self.layout),
            None,
            np.eye(4),
            np.eye(4),
            hand_eye_calibrated=False,
        )
        result = localizer.update(
            {}, self.camera_matrix, self.distortion, timestamp_s=10.0
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.source, SOURCE_UNAVAILABLE)
        self.assertIsNone(result.workspace_from_camera)

    def test_real_robot_fallback_requires_calibrated_hand_eye(self):
        provider = ManualRobotPoseProvider(np.eye(4), simulated=False)
        localizer = HybridCameraLocalizer(
            TagMapPoseEstimator(self.layout),
            provider,
            np.eye(4),
            np.eye(4),
            hand_eye_calibrated=False,
        )
        result = localizer.update(
            {}, self.camera_matrix, self.distortion, timestamp_s=time.monotonic()
        )
        self.assertFalse(result.valid)
        self.assertIn("hand-eye", result.reason)


if __name__ == "__main__":
    unittest.main()
