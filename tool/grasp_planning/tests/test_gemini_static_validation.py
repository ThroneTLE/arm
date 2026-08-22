"""Pure geometry tests for the Gemini static AnyGrasp validation entry."""

import unittest

import numpy as np

from tool.grasp_planning.gemini_static_validation import (
    build_steered_cloud,
    choose_target,
    clean_target_mask,
)


class GeminiStaticValidationTest(unittest.TestCase):
    def test_choose_target_uses_confidence_order_within_label(self):
        objects = [
            {"name": "can", "cls": 5, "conf": 0.7},
            {"name": "apple", "cls": 2, "conf": 0.99},
            {"name": "can", "cls": 5, "conf": 0.9},
        ]
        self.assertEqual(choose_target(objects, "can", 0)["conf"], 0.9)
        self.assertEqual(choose_target(objects, "can", 1)["conf"], 0.7)

    def test_clean_target_mask_rejects_far_depth_leak(self):
        depth = np.full((20, 20), 0.5, dtype=np.float32)
        mask = np.ones((20, 20), dtype=np.uint8)
        depth[8:12, 8:12] = 1.5
        cleaned = clean_target_mask(mask, depth, erosion_pixels=0)
        self.assertFalse(np.any(cleaned[8:12, 8:12]))
        self.assertGreater(np.count_nonzero(cleaned), 300)

    def test_cloud_projection_and_region_steering(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        depth = np.ones((10, 10), dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        mask[:8, :8] = True
        k = np.asarray([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
        points, colors, steering = build_steered_cloud(
            rgb, depth, k, mask, maximum_points=100
        )
        self.assertEqual(points.shape, (100, 3))
        self.assertEqual(colors.shape, (100, 3))
        self.assertEqual(np.count_nonzero(steering), 64)
        np.testing.assert_allclose(points[:, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
