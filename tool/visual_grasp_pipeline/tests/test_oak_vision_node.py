#!/usr/bin/env python3

import unittest
from pathlib import Path

import numpy as np

from tool.object_model_builder.rgbd_geometry import CameraIntrinsics
from tool.visual_grasp_pipeline.oak_vision_node import (
    LegacyArmClient,
    OakSnapshot,
    draw_pose_axes,
    find_sequence_target,
    load_oak_settings,
    prepare_foundationpose_input,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class OakVisionNodeTests(unittest.TestCase):
    def test_active_camera_config_is_pinned_to_current_oak(self):
        settings, maximum_sync = load_oak_settings(
            PROJECT_ROOT / "tool/object_model_builder/config/object_model_builder.yaml"
        )
        self.assertEqual(settings["mxid"], "14442C10D141C5D600")
        self.assertEqual(
            (settings["color_width"], settings["color_height"]), (1920, 1080)
        )
        self.assertLessEqual(maximum_sync, 0.03)

    def test_sequence_target_uses_stable_instance_id(self):
        objects = [
            {"name": "can", "id": 3},
            {"name": "can", "id": 7},
            {"name": "banana", "id": 8},
        ]
        self.assertIs(find_sequence_target(objects, "can", 7), objects[1])
        self.assertIs(find_sequence_target(objects, "banana", None), objects[2])
        self.assertIsNone(find_sequence_target(objects, "can", 9))

    def test_default_arm_client_is_dry_run(self):
        client = LegacyArmClient()
        self.assertFalse(client.enabled)
        self.assertEqual(client.execute(np.eye(4))["status"], "dry_run")
        client.close()

    def test_pose_axes_overlay_preserves_image_shape(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, 0.6]
        matrix = np.asarray(
            [[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]
        )
        overlay = draw_pose_axes(image, pose, matrix)
        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(np.count_nonzero(overlay)), 0)

    def test_foundationpose_input_is_cropped_and_intrinsics_are_shifted(self):
        matrix = np.asarray(
            [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
        )
        snapshot = OakSnapshot(
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            np.ones((1080, 1920), dtype=np.float32),
            CameraIntrinsics(1920, 1080, matrix, np.zeros(5)),
            1.0,
            0.001,
        )
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        mask[300:900, 500:1500] = 1
        prepared = prepare_foundationpose_input(
            snapshot,
            {"xyxy": (500, 300, 1500, 900)},
            mask,
            padding_pixels=20,
            maximum_size=640,
        )
        self.assertLessEqual(max(prepared.color_bgr.shape[:2]), 640)
        self.assertEqual(prepared.depth_m.shape, prepared.mask.shape)
        self.assertGreater(int(prepared.mask.sum()), 0)
        self.assertLess(prepared.camera_matrix[0, 0], matrix[0, 0])
        self.assertGreater(prepared.camera_matrix[0, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
