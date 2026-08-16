#!/usr/bin/env python3

import unittest

import numpy as np

from tool.bottle_localization.estimator import (
    BottlePositionEstimator,
    BottlePositionSettings,
    fit_fixed_radius_circle,
)
from tool.object_model_builder.rgbd_geometry import CameraIntrinsics


class BottlePositionEstimatorTests(unittest.TestCase):
    @staticmethod
    def intrinsics():
        return CameraIntrinsics(
            width=9,
            height=9,
            matrix=np.asarray(
                [[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]
            ),
            distortion=np.zeros(5),
        )

    @staticmethod
    def depth_and_mask():
        depth = np.zeros((9, 9), dtype=np.float32)
        mask = np.zeros((9, 9), dtype=np.uint8)
        depth[2:7, 2:7] = 1.0
        mask[2:7, 2:7] = 1
        return depth, mask

    def test_visible_surface_position_and_table_base(self):
        depth, mask = self.depth_and_mask()
        estimator = BottlePositionEstimator(
            BottlePositionSettings(
                workspace_up=(0.0, 0.0, 1.0),
                minimum_object_height_m=0.1,
                maximum_object_height_m=2.0,
                minimum_valid_points=10,
                mask_erosion_pixels=0,
                smoothing_alpha=1.0,
            )
        )
        result = estimator.estimate(
            depth, mask, self.intrinsics(), np.eye(4, dtype=np.float64)
        )
        self.assertTrue(result.valid, result.reason)
        np.testing.assert_allclose(result.base_center_workspace_m, [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(result.center_workspace_m, [0.0, 0.0, 1.0], atol=1e-9)
        self.assertEqual(result.method, "visible_surface_median")
        self.assertAlmostEqual(result.depth_coverage, 1.0)

    def test_measured_height_defines_geometric_center_in_real_workspace_convention(self):
        depth, mask = self.depth_and_mask()
        workspace_from_camera = np.eye(4, dtype=np.float64)
        workspace_from_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
        workspace_from_camera[:3, 3] = [0.2, 0.1, 0.8]
        estimator = BottlePositionEstimator(
            BottlePositionSettings(
                workspace_up=(0.0, 0.0, -1.0),
                minimum_object_height_m=0.05,
                maximum_object_height_m=0.5,
                minimum_valid_points=10,
                mask_erosion_pixels=0,
                nominal_bottle_height_m=0.2,
                smoothing_alpha=1.0,
            )
        )
        result = estimator.estimate(
            depth, mask, self.intrinsics(), workspace_from_camera
        )
        self.assertTrue(result.valid, result.reason)
        np.testing.assert_allclose(result.base_center_workspace_m, [0.2, 0.1, 0.0], atol=1e-9)
        np.testing.assert_allclose(result.center_workspace_m, [0.2, 0.1, -0.1], atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(result.workspace_from_bottle[:3, :3]), 1.0)

    def test_known_radius_circle_fit_recovers_partial_visible_arc(self):
        center = np.asarray([0.12, -0.08])
        radius = 0.034
        angles = np.linspace(-0.9, 0.9, 160)
        points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        points = np.vstack([points, [[0.4, 0.4], [-0.3, 0.1]]])
        fitted, rms = fit_fixed_radius_circle(
            points,
            radius,
            initial_center_xy=center + np.asarray([0.004, -0.003]),
        )
        np.testing.assert_allclose(fitted, center, atol=1.5e-3)
        self.assertLess(rms, 0.04)

    def test_low_depth_coverage_is_rejected(self):
        depth, mask = self.depth_and_mask()
        depth[2:7, 2:6] = 0.0
        estimator = BottlePositionEstimator(
            BottlePositionSettings(
                workspace_up=(0.0, 0.0, 1.0),
                minimum_object_height_m=0.1,
                maximum_object_height_m=2.0,
                minimum_depth_coverage=0.5,
                minimum_valid_points=3,
                mask_erosion_pixels=0,
                smoothing_alpha=1.0,
            )
        )
        result = estimator.estimate(
            depth, mask, self.intrinsics(), np.eye(4, dtype=np.float64)
        )
        self.assertFalse(result.valid)
        self.assertIn("coverage", result.reason)

    def test_measured_radius_removes_far_mask_edge_depth(self):
        intrinsics = CameraIntrinsics(
            width=41,
            height=9,
            matrix=np.asarray(
                [[100.0, 0.0, 20.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]
            ),
            distortion=np.zeros(5),
        )
        depth = np.zeros((9, 41), dtype=np.float32)
        mask = np.zeros((9, 41), dtype=np.uint8)
        depth[2:7, 18:23] = 1.0
        mask[2:7, 18:23] = 1
        depth[4, 40] = 1.0
        mask[4, 40] = 1
        estimator = BottlePositionEstimator(
            BottlePositionSettings(
                workspace_up=(0.0, 0.0, 1.0),
                minimum_object_height_m=0.1,
                maximum_object_height_m=2.0,
                minimum_valid_points=10,
                mask_erosion_pixels=0,
                keep_largest_depth_component=False,
                nominal_bottle_diameter_m=0.08,
                maximum_radial_excess_m=0.01,
                smoothing_alpha=1.0,
            )
        )
        result = estimator.estimate(
            depth, mask, intrinsics, np.eye(4, dtype=np.float64)
        )
        self.assertTrue(result.valid, result.reason)
        radial = np.linalg.norm(result.cloud_workspace_m[:, :2], axis=1)
        self.assertLessEqual(float(radial.max()), 0.05 + 1e-9)
        self.assertEqual(result.valid_point_count, 25)

    def test_largest_depth_component_removes_isolated_edge_island(self):
        intrinsics = CameraIntrinsics(
            width=41,
            height=9,
            matrix=np.asarray(
                [[100.0, 0.0, 20.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]
            ),
            distortion=np.zeros(5),
        )
        depth = np.zeros((9, 41), dtype=np.float32)
        mask = np.zeros((9, 41), dtype=np.uint8)
        depth[2:7, 18:23] = 1.0
        mask[2:7, 18:23] = 1
        depth[3:5, 38:40] = 1.0
        mask[3:5, 38:40] = 1
        estimator = BottlePositionEstimator(
            BottlePositionSettings(
                workspace_up=(0.0, 0.0, 1.0),
                minimum_object_height_m=0.1,
                maximum_object_height_m=2.0,
                minimum_depth_coverage=0.5,
                minimum_valid_points=10,
                mask_erosion_pixels=0,
                smoothing_alpha=1.0,
            )
        )
        result = estimator.estimate(
            depth, mask, intrinsics, np.eye(4, dtype=np.float64)
        )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.valid_point_count, 25)


if __name__ == "__main__":
    unittest.main()
