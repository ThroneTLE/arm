#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tool.object_model_builder.capture_session import CaptureSession
from tool.object_model_builder.mesh_fusion import (
    object_frame_from_workspace,
    validate_capture_quality,
)
from tool.object_model_builder.rgbd_calibration import (
    CalibrationTarget,
    calibration_target_from_mapping,
    calibration_view_signature,
    charuco_view_change,
    charuco_view_signature,
    detect_calibration_target,
    detect_charuco,
    infrared_to_uint8,
)
from tool.camera_calibration.calib_common import charuco_board
from tool.object_model_builder.rgbd_geometry import (
    CameraIntrinsics,
    DepthToColorAligner,
    RgbdCalibration,
    align_depth_to_color,
    depth_coverage,
    masked_depth_centroid,
    load_runtime_calibration,
)


def intrinsics(width, height, fx, fy, cx, cy):
    return CameraIntrinsics(
        width,
        height,
        np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        np.zeros(5),
    )


class RgbdGeometryTests(unittest.TestCase):
    def test_hardware_aligned_oak_runtime_uses_identity_geometry(self):
        calibration = load_runtime_calibration(
            str(
                Path(__file__).resolve().parents[3]
                / "ros_ws/src/arm_vision_framework/config/calibration_parameters.yaml"
            )
        )
        calibration.require_valid()
        self.assertEqual(
            (calibration.color.width, calibration.color.height), (1920, 1080)
        )
        self.assertEqual(calibration.color.distortion_model, "rational_polynomial")
        np.testing.assert_allclose(calibration.color_from_depth, np.eye(4))

    def test_different_resolutions_use_3d_projection(self):
        depth_camera = intrinsics(4, 3, 2.0, 2.0, 0.0, 0.0)
        color_camera = intrinsics(8, 6, 4.0, 4.0, 0.0, 0.0)
        calibration = RgbdCalibration(
            color_camera, depth_camera, np.eye(4), valid=True
        )
        raw_depth = np.ones((3, 4), dtype=np.float32)
        aligned = align_depth_to_color(raw_depth, calibration)
        expected_pixels = [(row * 2, column * 2) for row in range(3) for column in range(4)]
        self.assertGreater(np.count_nonzero(aligned), len(expected_pixels))
        for row, column in expected_pixels:
            self.assertAlmostEqual(float(aligned[row, column]), 1.0)

    def test_color_from_depth_translation_changes_projection(self):
        camera = intrinsics(5, 3, 2.0, 2.0, 2.0, 1.0)
        transform = np.eye(4)
        transform[0, 3] = 0.5
        calibration = RgbdCalibration(camera, camera, transform, valid=True)
        raw_depth = np.zeros((3, 5), dtype=np.float32)
        raw_depth[1, 2] = 1.0
        aligned = align_depth_to_color(raw_depth, calibration)
        self.assertAlmostEqual(float(aligned[1, 3]), 1.0)
        self.assertEqual(float(aligned[1, 2]), 0.0)

    def test_z_buffer_keeps_nearest_surface(self):
        depth_camera = intrinsics(2, 1, 100.0, 100.0, 0.0, 0.0)
        color_camera = intrinsics(1, 1, 1.0, 1.0, 0.0, 0.0)
        calibration = RgbdCalibration(
            color_camera, depth_camera, np.eye(4), valid=True
        )
        aligned = align_depth_to_color(
            np.asarray([[0.7, 1.1]], dtype=np.float32), calibration
        )
        self.assertAlmostEqual(float(aligned[0, 0]), 0.7, places=6)

    def test_invalid_calibration_is_rejected(self):
        camera = intrinsics(2, 2, 100.0, 100.0, 0.5, 0.5)
        calibration = RgbdCalibration(camera, camera, np.eye(4), valid=False)
        with self.assertRaisesRegex(ValueError, "RGB-D calibration is invalid"):
            align_depth_to_color(np.ones((2, 2), dtype=np.float32), calibration)

    def test_cached_aligner_reuses_geometry_without_changing_projection(self):
        depth_camera = intrinsics(6, 4, 4.5, 4.0, 2.5, 1.5)
        color_camera = intrinsics(8, 6, 6.0, 5.5, 3.5, 2.5)
        transform = np.eye(4)
        transform[:3, 3] = [0.08, -0.01, 0.02]
        calibration = RgbdCalibration(
            color_camera, depth_camera, transform, valid=True
        )
        aligner = DepthToColorAligner(calibration)
        first = np.linspace(0.4, 1.2, 24, dtype=np.float32).reshape(4, 6)
        first[0, 0] = 0.0
        second = first + 0.15
        second[1, 2] = np.nan
        np.testing.assert_array_equal(
            aligner.align(first), align_depth_to_color(first, calibration)
        )
        np.testing.assert_array_equal(
            aligner.align(second), align_depth_to_color(second, calibration)
        )

    def test_cached_aligner_rejects_wrong_depth_shape(self):
        camera = intrinsics(3, 2, 100.0, 100.0, 1.0, 0.5)
        aligner = DepthToColorAligner(
            RgbdCalibration(camera, camera, np.eye(4), valid=True)
        )
        with self.assertRaisesRegex(ValueError, "depth shape"):
            aligner.align(np.ones((3, 2), dtype=np.float32))

    def test_depth_coverage_is_measured_inside_mask(self):
        depth = np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        mask = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
        self.assertAlmostEqual(depth_coverage(depth, mask), 0.5)

    def test_masked_depth_centroid_uses_metric_camera_geometry(self):
        camera = intrinsics(5, 5, 2.0, 2.0, 2.0, 2.0)
        depth = np.zeros((5, 5), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=np.uint8)
        depth[1:4, 1:4] = 1.2
        mask[1:4, 1:4] = 1
        centroid = masked_depth_centroid(
            depth, mask, camera, minimum_valid_points=5
        )
        np.testing.assert_allclose(centroid, [0.0, 0.0, 1.2], atol=1e-7)

    def test_masked_depth_centroid_rejects_sparse_depth(self):
        camera = intrinsics(3, 3, 100.0, 100.0, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "valid depth points"):
            masked_depth_centroid(
                np.ones((3, 3), dtype=np.float32),
                np.eye(3, dtype=np.uint8),
                camera,
                minimum_valid_points=4,
            )


class InfraredCalibrationImageTests(unittest.TestCase):
    def test_uint16_ir_is_converted_to_full_range_uint8(self):
        infrared = np.linspace(30, 1000, 64, dtype=np.uint16).reshape(8, 8)
        converted = infrared_to_uint8(infrared)
        self.assertEqual(converted.dtype, np.uint8)
        self.assertEqual(converted.shape, infrared.shape)
        self.assertEqual(int(converted.min()), 0)
        self.assertEqual(int(converted.max()), 255)

    def test_charuco_detector_accepts_uint16_ir(self):
        corners, ids = detect_charuco(np.zeros((480, 640), dtype=np.uint16))
        self.assertEqual(corners.shape, (0, 2))
        self.assertEqual(ids.shape, (0,))

    def test_view_signature_detects_motion_independent_of_visible_ids(self):
        board_points = np.asarray(
            charuco_board().getChessboardCorners(), dtype=np.float32
        )[:, :2]
        ids = np.arange(len(board_points), dtype=np.int32)
        first_points = board_points * 1800.0 + np.asarray([180.0, 140.0])
        moved_points = first_points + np.asarray([64.0, 0.0])
        first = charuco_view_signature(first_points, ids, (1024, 1280))
        moved = charuco_view_signature(moved_points, ids, (1024, 1280))
        subset = charuco_view_signature(
            first_points[::2], ids[::2], (1024, 1280)
        )
        self.assertLess(charuco_view_change(first, subset), 1e-5)
        self.assertGreater(charuco_view_change(first, moved), 0.04)

    def test_view_signature_rejects_too_few_corners(self):
        self.assertIsNone(
            charuco_view_signature(
                np.zeros((3, 2), dtype=np.float32),
                np.arange(3, dtype=np.int32),
                (1024, 1280),
            )
        )

    def test_hardware_checkerboard_target_detection_and_scale(self):
        target = calibration_target_from_mapping(
            {
                "type": "checkerboard",
                "model": "SINE IMAGE YE0102-A540",
                "pattern_columns": 10,
                "pattern_rows": 5,
                "square_size_m": 0.029,
            }
        )
        image = np.zeros((420, 620), dtype=np.uint8)
        for row in range(6):
            for column in range(11):
                if (row + column) % 2 == 0:
                    cv2.rectangle(
                        image,
                        (70 + column * 42, 45 + row * 42),
                        (70 + (column + 1) * 42, 45 + (row + 1) * 42),
                        255,
                        -1,
                    )
        corners, ids = detect_calibration_target(image, target)
        self.assertEqual(corners.shape, (50, 2))
        self.assertEqual(ids.tolist(), list(range(50)))
        np.testing.assert_allclose(
            target.object_points()[1] - target.object_points()[0],
            np.asarray([0.029, 0.0, 0.0], dtype=np.float32),
            atol=1e-7,
        )
        self.assertIsNotNone(calibration_view_signature(corners, ids, image.shape, target))


class CaptureSessionTests(unittest.TestCase):
    def test_capture_round_trip_preserves_pose_and_depth(self):
        color_camera = intrinsics(4, 3, 200.0, 200.0, 1.5, 1.0)
        depth_camera = intrinsics(2, 2, 100.0, 100.0, 0.5, 0.5)
        with tempfile.TemporaryDirectory() as directory:
            session = CaptureSession.create(
                directory,
                color_camera,
                depth_camera,
                np.eye(4),
                "calibration.yaml",
                "tag_layout.yaml",
                "best.pt",
                "bottle",
            )
            color = np.full((3, 4, 3), 127, dtype=np.uint8)
            aligned = np.full((3, 4), 0.75, dtype=np.float32)
            mask = np.ones((3, 4), dtype=np.uint8)
            workspace_from_color = np.eye(4)
            workspace_from_color[0, 3] = 0.12
            session.add_view(
                color,
                aligned,
                mask,
                workspace_from_color,
                timestamp_s=12.5,
                depth_raw_m=np.ones((2, 2), dtype=np.float32),
            )
            self.assertEqual(len(session), 1)
            view = next(session.iter_views())
            np.testing.assert_allclose(view.depth_aligned_m, aligned, atol=0.0005)
            np.testing.assert_allclose(view.workspace_from_color, workspace_from_color)
            self.assertEqual(view.color_bgr.shape, color.shape)

    def test_capture_quality_rejects_low_depth_coverage(self):
        camera = intrinsics(10, 10, 100.0, 100.0, 4.5, 4.5)
        with tempfile.TemporaryDirectory() as directory:
            session = CaptureSession.create(
                directory,
                camera,
                camera,
                np.eye(4),
                "calibration.yaml",
                "tag_layout.yaml",
                "best.pt",
                "bottle",
            )
            depth = np.zeros((10, 10), dtype=np.float32)
            depth[:5] = 1.0
            session.add_view(
                np.zeros((10, 10, 3), dtype=np.uint8),
                depth,
                np.ones((10, 10), dtype=np.uint8),
                np.eye(4),
                timestamp_s=1.0,
            )
            with self.assertRaisesRegex(ValueError, "transparent"):
                validate_capture_quality(
                    session, minimum_mask_depth_coverage=0.70
                )

    def test_capture_quality_rejects_object_motion_in_tag_workspace(self):
        camera = intrinsics(10, 10, 100.0, 100.0, 4.5, 4.5)
        with tempfile.TemporaryDirectory() as directory:
            session = CaptureSession.create(
                directory,
                camera,
                camera,
                np.eye(4),
                "calibration.yaml",
                "tag_layout.yaml",
                "best.pt",
                "bottle",
            )
            depth = np.ones((10, 10), dtype=np.float32)
            color = np.zeros((10, 10, 3), dtype=np.uint8)
            mask = np.ones((10, 10), dtype=np.uint8)
            session.add_view(color, depth, mask, np.eye(4), timestamp_s=1.0)
            moved_pose = np.eye(4)
            moved_pose[0, 3] = 0.10
            session.add_view(color, depth, mask, moved_pose, timestamp_s=2.0)
            with self.assertRaisesRegex(ValueError, "object centroid moved"):
                validate_capture_quality(
                    session,
                    minimum_mask_depth_coverage=0.70,
                    maximum_object_centroid_shift_m=0.05,
                )


class ObjectFrameTests(unittest.TestCase):
    def test_bottom_center_frame_uses_configured_workspace_up(self):
        vertices = np.asarray(
            [
                [-0.05, -0.03, 0.0],
                [0.05, -0.03, 0.0],
                [-0.05, 0.03, 0.0],
                [0.05, 0.03, 0.0],
                [-0.05, -0.03, -0.20],
                [0.05, 0.03, -0.20],
            ]
        )
        workspace_from_object = object_frame_from_workspace(vertices, workspace_up=(0, 0, -1))
        object_from_workspace = np.linalg.inv(workspace_from_object)
        converted = vertices @ object_from_workspace[:3, :3].T + object_from_workspace[:3, 3]
        self.assertAlmostEqual(float(converted[:, 2].min()), 0.0, places=9)
        self.assertAlmostEqual(float(converted[:, 2].max()), 0.20, places=9)
        self.assertAlmostEqual(float(workspace_from_object[2, 3]), 0.0, places=9)
        self.assertAlmostEqual(float(np.linalg.det(workspace_from_object[:3, :3])), 1.0)


if __name__ == "__main__":
    unittest.main()
