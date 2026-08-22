#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from competition_pipeline.checkerboard_target import CheckerboardTarget
from competition_pipeline.configuration import CompetitionConfig
from competition_pipeline.geometry import transform_from_xyz_rpy
from competition_pipeline.hand_eye import HandEyeCalibrator
from competition_pipeline.sample_store import HandEyeSampleStore


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "competition_pipeline" / "config" / "competition.yaml"


class CheckerboardHandEyeTest(unittest.TestCase):
    def setUp(self):
        self.config = CompetitionConfig(CONFIG_PATH)
        self.config.data["hand_eye"]["calibration_target"]["type"] = "checkerboard"
        self.config.data["hand_eye"]["calibration_target"]["checkerboard"].update({
            "configured": True,
            "board_width_mm": 300.0,
            "board_height_mm": 225.0,
            "square_size_mm": 25.0,
            "squares_x": 12,
            "squares_y": 9,
            "inner_corners": [11, 8],
        })
        self.config.data["hand_eye"]["minimum_samples"] = 8

    def test_active_default_checkerboard_is_12x9_25mm(self):
        config = CompetitionConfig(CONFIG_PATH)
        target = config.data["hand_eye"]["calibration_target"]
        checkerboard = target["checkerboard"]
        self.assertEqual(target["type"], "checkerboard")
        self.assertTrue(checkerboard["configured"])
        self.assertEqual((checkerboard["squares_x"], checkerboard["squares_y"]), (12, 9))
        self.assertEqual(checkerboard["square_size_mm"], 25.0)
        self.assertEqual(checkerboard["inner_corners"], [11, 8])
        self.assertEqual(
            CheckerboardTarget(checkerboard).pattern_size, (11, 8)
        )

    def test_explicit_12x9_grid_means_11x8_inner_corners(self):
        target = CheckerboardTarget(
            self.config.data["hand_eye"]["calibration_target"]["checkerboard"]
        )
        self.assertEqual((target.squares_x, target.squares_y), (12, 9))
        self.assertEqual(target.pattern_size, (11, 8))
        self.assertEqual(target.corner_count, 88)
        self.assertEqual(target.object_points_m.shape, (88, 3))
        np.testing.assert_allclose(target.object_points_m[-1], [0.25, 0.175, 0.0])

    def test_outer_board_dimensions_do_not_guess_square_count(self):
        with self.assertRaisesRegex(ValueError, "square counts are not configured"):
            CheckerboardTarget({
                "board_width_mm": 60,
                "board_height_mm": 45,
                "square_size_mm": 5,
            })

    def test_invalid_square_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inner_corners"):
            CheckerboardTarget({
                "board_width_mm": 60,
                "board_height_mm": 45,
                "square_size_mm": 5,
                "squares_x": 11,
                "squares_y": 8,
                "inner_corners": [12, 9],
            })

    def test_rendered_checkerboard_is_detected_and_pnp_is_metric(self):
        square_pixels = 50
        margin = 50
        image = np.full((550, 700, 3), 255, dtype=np.uint8)
        for row in range(9):
            for column in range(12):
                value = 0 if (row + column) % 2 == 0 else 255
                image[
                    margin + row * square_pixels:margin + (row + 1) * square_pixels,
                    margin + column * square_pixels:margin + (column + 1) * square_pixels,
                ] = value
        camera_matrix = np.asarray(
            [[800.0, 0.0, 350.0], [0.0, 800.0, 275.0], [0.0, 0.0, 1.0]]
        )
        target = CheckerboardTarget(
            self.config.data["hand_eye"]["calibration_target"]["checkerboard"]
        )
        observation = target.estimate(image, camera_matrix, np.zeros(5))
        self.assertTrue(observation.valid, observation.reason)
        self.assertEqual(len(observation.corners), 88)
        # 25 mm projects to 50 px at fx=800, so the fronto-parallel board is
        # approximately 800 * 0.025 / 50 = 0.40 m from the camera.
        self.assertAlmostEqual(observation.camera_from_board[2, 3], 0.40, places=3)

    def test_fixed_checkerboard_solves_tcp_from_camera_without_base_board_pose(self):
        expected_tcp_from_camera = transform_from_xyz_rpy(
            [0.04, -0.03, 0.13], [5.0, -3.0, 8.0]
        )
        fixed_base_from_board = transform_from_xyz_rpy(
            [0.60, 0.10, 0.30], [0.0, 20.0, 0.0]
        )
        calibrator = HandEyeCalibrator(self.config)
        for index in range(9):
            base_from_tcp = transform_from_xyz_rpy(
                [0.20 + index * 0.03, -0.20 + index * 0.02, 0.45 + index * 0.01],
                [12.0 * index, -7.0 * index, 9.0 * index],
            )
            camera_from_board = (
                np.linalg.inv(expected_tcp_from_camera)
                @ np.linalg.inv(base_from_tcp)
                @ fixed_base_from_board
            )
            calibrator.add_checkerboard_sample(base_from_tcp, camera_from_board, 0.1)
        result = calibrator.solve()
        self.assertEqual(result.target_type, "checkerboard")
        self.assertEqual(len(result.inlier_indices), 9)
        np.testing.assert_allclose(
            result.tcp_from_camera, expected_tcp_from_camera, atol=1e-8
        )

    def test_checkerboard_sample_store_rejects_changed_board_definition(self):
        calibrator = HandEyeCalibrator(self.config)
        sample = calibrator.add_checkerboard_sample(np.eye(4), np.eye(4), 0.2)
        with tempfile.TemporaryDirectory() as directory:
            store = HandEyeSampleStore(Path(directory) / "samples.yaml", self.config)
            self.assertEqual(store.append(sample, "board.png"), 1)
            self.config.data["hand_eye"]["calibration_target"]["checkerboard"][
                "square_size_mm"
            ] = 4.0
            self.config.data["hand_eye"]["calibration_target"]["checkerboard"][
                "inner_corners"
            ] = [14, 10]
            with self.assertRaisesRegex(ValueError, "target definition changed"):
                store.load()


if __name__ == "__main__":
    unittest.main()
