#!/usr/bin/env python3

import unittest

import numpy as np

from tool.object_model_builder.rgbd_odometry import RgbdOdometryTracker


class RgbdOdometryTests(unittest.TestCase):
    def test_first_frame_defines_camera_origin_without_tags(self):
        tracker = RgbdOdometryTracker()
        color = np.zeros((8, 10, 3), dtype=np.uint8)
        depth = np.ones((8, 10), dtype=np.float32)
        K = np.asarray(
            [[100.0, 0.0, 5.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]
        )
        result = tracker.update(color, depth, K)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.world_from_camera, np.eye(4))
        self.assertTrue(tracker.has_reference)

    def test_rejects_mismatched_rgb_depth_shapes(self):
        tracker = RgbdOdometryTracker()
        color = np.zeros((8, 10, 3), dtype=np.uint8)
        depth = np.ones((7, 10), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shapes do not match"):
            tracker.update(color, depth, np.eye(3))


if __name__ == "__main__":
    unittest.main()
