#!/usr/bin/env python3

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from tool.camera_calibration.calib_common import (
    COORDINATE_CONVENTION,
    draw_tag_coordinate_axes,
    load_layout,
    pixel_to_workspace_plane,
    quad_center,
    require_coordinate_convention,
    tag_world_corners,
)
from tool.camera_calibration.calibrate_intrinsics import calibrate
from tool.camera_calibration.calibrate_workspace import (
    build_workspace_output,
    calibration_correspondences,
    solve_workspace_pose,
)
from tool.camera_calibration.hybrid_localization import TagMapPoseEstimator
from tool.camera_calibration.validate_workspace import (
    estimate_dynamic_validation,
    estimate_tag_corners_mm,
    estimate_tag_origin_mm,
    summarize_estimates,
    tag_yaw_deg_from_corners,
)


class GeometryTest(unittest.TestCase):
    def test_intrinsic_calibration_with_synthetic_views(self):
        camera_matrix = np.asarray(
            [[600.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        distortion = np.zeros((5, 1), dtype=float)
        points = np.asarray(
            [[x * 0.024, y * 0.024, 0.0] for y in range(9) for x in range(6)],
            dtype=np.float32,
        ).reshape(-1, 1, 3)
        object_views = []
        image_views = []
        for index in range(15):
            rvec = np.asarray(
                [[-0.25 + index * 0.035], [0.18 * np.sin(index)], [0.05 * np.cos(index)]],
                dtype=float,
            )
            tvec = np.asarray(
                [[-0.06 + index * 0.008], [-0.04 + (index % 4) * 0.02], [0.65 + (index % 3) * 0.08]],
                dtype=float,
            )
            pixels, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion)
            object_views.append(points.copy())
            image_views.append(pixels.astype(np.float32))
        result = calibrate(object_views, image_views, (640, 480))
        self.assertLess(result["rms"], 1e-3)
        np.testing.assert_allclose(result["camera_matrix"], camera_matrix, atol=0.1)

    def test_quad_center_uses_diagonal_intersection(self):
        quad = np.asarray([[100, 80], [260, 100], [230, 220], [80, 190]], dtype=float)
        center = quad_center(quad)
        self.assertTrue(np.all(np.isfinite(center)))
        self.assertGreater(center[0], 100)
        self.assertLess(center[0], 230)

    def test_tag_corner_order_and_yaw(self):
        corners = tag_world_corners([1.0, 2.0, 0.0], 0.0, 0.2)
        np.testing.assert_allclose(corners[0], [1.0, 2.0, 0.0])
        np.testing.assert_allclose(corners[1], [1.2, 2.0, 0.0])
        np.testing.assert_allclose(corners[2], [1.2, 2.2, 0.0])
        np.testing.assert_allclose(corners[3], [1.0, 2.2, 0.0])

    def test_pose_and_plane_intersection(self):
        camera_matrix = np.asarray([[600.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]])
        distortion = np.zeros((5, 1), dtype=float)
        object_points = []
        for origin in ([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.2, 0.0]):
            object_points.extend(tag_world_corners(origin, 0.0, 0.07))
        object_points = np.asarray(object_points)
        rvec_true = np.asarray([[2.9], [0.15], [-0.2]], dtype=float)
        tvec_true = np.asarray([[0.02], [-0.03], [0.75]], dtype=float)
        image_points, _ = cv2.projectPoints(
            object_points, rvec_true, tvec_true, camera_matrix, distortion
        )
        camera_from_workspace, _, errors = solve_workspace_pose(
            object_points, image_points.reshape(-1, 2), camera_matrix, distortion
        )
        self.assertLess(float(errors.max()), 1e-5)

        validation = np.asarray([0.3, 0.2, 0.0], dtype=float)
        projected, _ = cv2.projectPoints(
            validation.reshape(1, 3), rvec_true, tvec_true, camera_matrix, distortion
        )
        recovered = pixel_to_workspace_plane(
            projected.reshape(2), camera_matrix, distortion, camera_from_workspace, 0.0
        )
        np.testing.assert_allclose(recovered, validation, atol=1e-6)

        output = build_workspace_output(
            {"calibration_tags": {100: {}, 101: {}, 102: {}}, "tag_size_mm": 70.0},
            camera_from_workspace,
            np.linalg.inv(camera_from_workspace),
            errors,
            np.repeat([100, 101, 102], 4),
            60,
            "intrinsics.yaml",
            "layout.yaml",
        )
        self.assertLess(output["quality"]["rms_reprojection_error_px"], 1e-5)
        self.assertEqual(output["quality"]["frames_used"], 60)

    def test_correspondences_and_validation_use_top_left_origin(self):
        layout = {
            "tag_size_mm": 70.0,
            "calibration_tags": {
                100: {"origin_mm": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
                101: {"origin_mm": [150.0, 0.0, 0.0], "yaw_deg": 0.0},
                102: {"origin_mm": [0.0, 130.0, 0.0], "yaw_deg": 0.0},
            },
        }
        camera_matrix = np.asarray(
            [[700.0, 0.0, 320.0], [0.0, 705.0, 240.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros((5, 1))
        rvec = np.asarray([[2.8], [0.12], [-0.08]])
        tvec = np.asarray([[0.01], [-0.02], [0.7]])
        median_corners = {}
        for tag_id, entry in layout["calibration_tags"].items():
            world = tag_world_corners(
                np.asarray(entry["origin_mm"]) / 1000.0, 0.0, 0.07
            )
            pixels, _ = cv2.projectPoints(world, rvec, tvec, camera_matrix, distortion)
            median_corners[tag_id] = pixels.reshape(4, 2)
        object_points, _, _ = calibration_correspondences(layout, median_corners)
        np.testing.assert_allclose(object_points[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(object_points[4], [0.15, 0.0, 0.0])

        camera_from_workspace = np.eye(4)
        camera_from_workspace[:3, :3] = cv2.Rodrigues(rvec)[0]
        camera_from_workspace[:3, 3] = tvec.reshape(3)
        estimated_mm = estimate_tag_origin_mm(
            median_corners[102],
            camera_matrix,
            distortion,
            camera_from_workspace,
            0.0,
        )
        np.testing.assert_allclose(estimated_mm, [0.0, 130.0, 0.0], atol=1e-5)

    def test_validation_summary(self):
        estimates = np.asarray(
            [[300.0, 200.0, 0.0], [302.0, 198.0, 0.0], [301.0, 199.0, 0.0]]
        )
        summary = summarize_estimates(estimates, [300.0, 200.0, 0.0], 103)
        self.assertEqual(summary["validation_tag_id"], 103)
        self.assertAlmostEqual(summary["median_error_xy_mm"], np.sqrt(2.0))
        self.assertEqual(summary["expected_origin_mm"], [300.0, 200.0, 0.0])

    def test_dynamic_validation_is_invariant_to_camera_motion(self):
        layout = {
            "schema_version": 2,
            "coordinate_convention": dict(COORDINATE_CONVENTION),
            "tag_size_mm": 70.0,
            "calibration_tags": {
                100: {"origin_mm": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
                101: {"origin_mm": [180.0, 0.0, 0.0], "yaw_deg": 0.0},
                102: {"origin_mm": [0.0, 160.0, 0.0], "yaw_deg": 0.0},
            },
            "validation_tag": {
                "id": 103,
                "origin_mm": [230.0, 120.0, 0.0],
                "yaw_deg": 32.0,
            },
        }
        camera_matrix = np.asarray(
            [[900.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros((5, 1))
        estimator = TagMapPoseEstimator(layout, minimum_tags=3, max_rms_reprojection_error_px=2.0)
        camera_poses = (
            (
                np.asarray([[2.85], [0.18], [-0.12]]),
                np.asarray([[0.02], [-0.03], [0.72]]),
            ),
            (
                np.asarray([[2.65], [-0.28], [0.22]]),
                np.asarray([[-0.12], [0.06], [0.88]]),
            ),
        )
        recovered = []
        recovered_yaws = []
        camera_positions = []
        all_entries = dict(layout["calibration_tags"])
        all_entries[103] = layout["validation_tag"]
        for rvec, tvec in camera_poses:
            detections = {}
            for tag_id, entry in all_entries.items():
                corners = tag_world_corners(
                    np.asarray(entry["origin_mm"]) / 1000.0,
                    entry["yaw_deg"],
                    0.07,
                )
                pixels, _ = cv2.projectPoints(
                    corners, rvec, tvec, camera_matrix, distortion
                )
                detections[tag_id] = pixels.reshape(4, 2)
            visual_pose, estimate_mm = estimate_dynamic_validation(
                detections,
                estimator,
                103,
                camera_matrix,
                distortion,
                0.0,
            )
            self.assertTrue(visual_pose.valid, visual_pose.reason)
            recovered.append(estimate_mm)
            measured_corners_mm = estimate_tag_corners_mm(
                detections[103],
                camera_matrix,
                distortion,
                visual_pose.camera_from_workspace,
                0.0,
            )
            recovered_yaws.append(tag_yaw_deg_from_corners(measured_corners_mm))
            camera_positions.append(visual_pose.workspace_from_camera[:3, 3])
        np.testing.assert_allclose(
            recovered,
            [[230.0, 120.0, 0.0], [230.0, 120.0, 0.0]],
            atol=1e-4,
        )
        np.testing.assert_allclose(recovered_yaws, [32.0, 32.0], atol=1e-5)
        self.assertGreater(
            float(np.linalg.norm(camera_positions[0] - camera_positions[1])),
            0.1,
        )

    def test_tag_overlay_draws_top_left_origin_axes_and_labels(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = {
            100: np.asarray([[180, 140], [320, 150], [310, 290], [170, 275]], dtype=float)
        }
        camera_matrix = np.asarray(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        )
        draw_tag_coordinate_axes(
            image,
            detections,
            {100: np.asarray([0.0, 0.0, 0.0])},
            camera_matrix,
            np.zeros((5, 1)),
            70.0,
        )
        self.assertGreater(int(np.count_nonzero(image)), 100)
        self.assertGreater(int(image[140, 180].sum()), 0)

    def test_legacy_center_layout_and_extrinsics_are_rejected(self):
        legacy_layout = {
            "dictionary": "DICT_APRILTAG_36h11",
            "tag_size_mm": 70.0,
            "calibration_tags": {
                100: {"center_mm": [0, 0, 0]},
                101: {"center_mm": [100, 0, 0]},
                102: {"center_mm": [0, 100, 0]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.yaml"
            path.write_text(yaml.safe_dump(legacy_layout), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "左上角约定不兼容"):
                load_layout(str(path))
        with self.assertRaisesRegex(ValueError, "左上角约定不兼容"):
            require_coordinate_convention({"schema_version": 1}, "workspace extrinsics")

    def test_current_coordinate_convention_is_accepted(self):
        self.assertEqual(
            require_coordinate_convention(
                {"coordinate_convention": dict(COORDINATE_CONVENTION)}
            )["origin"],
            "black_border_top_left",
        )


if __name__ == "__main__":
    unittest.main()
