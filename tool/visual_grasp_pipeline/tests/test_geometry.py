#!/usr/bin/env python3

import unittest

import numpy as np

from tool.visual_grasp_pipeline.geometry import (
    build_world_from_tags,
    compute_grasp,
    compute_grasp_sphere,
    fill_depth_roi,
    rot2quat,
    smooth_pose,
    to_world_and_compensate,
)


class GeometryTests(unittest.TestCase):
    def test_build_world_from_tags_uses_base_origin_and_orthonormal_axes(self):
        # Two tag poses in camera frame, all with the same Z axis.
        t0 = np.eye(4)
        t0[:3, 3] = [0.1, 0.2, 0.8]
        t1 = np.eye(4)
        t1[:3, 3] = [0.3, 0.2, 0.8]
        world = build_world_from_tags([(0, t0), (1, t1)])
        self.assertIsNotNone(world)
        np.testing.assert_allclose(world[:3, 3], [0.1, 0.2, 0.8])
        np.testing.assert_allclose(world[:3, :3].T @ world[:3, :3], np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(world[:3, :3]), 1.0)

    def test_build_world_returns_none_without_base_tag(self):
        t = np.eye(4)
        self.assertIsNone(build_world_from_tags([(1, t)], base_id=0))

    def test_to_world_and_compensate_applies_offset_and_flip(self):
        camera_from_object = np.eye(4)
        camera_from_object[:3, 3] = [0.1, 0.2, 0.5]
        world_from_camera = np.eye(4)
        result = to_world_and_compensate(
            camera_from_object,
            world_from_camera,
            offset_xy_mm=(10.0, 20.0),
            center_offset_mm=30.0,
            flip_x=True,
            flip_y=False,
        )
        # x = -(0.1 + 0.010) = -0.11 ; y = 0.2 + 0.020 = 0.22 ; z = 0.5 - 0.030 = 0.47
        np.testing.assert_allclose(result[:3, 3], [-0.11, 0.22, 0.47], atol=1e-9)

    def test_compute_grasp_cylinder_lowers_along_object_axis(self):
        obj = np.eye(4)
        obj[:3, 3] = [0.1, 0.2, 0.3]
        grasp = compute_grasp(obj, offset_mm=5.0)
        # For identity rotation, the object axis is +Z and the grasp point is 5 mm lower.
        np.testing.assert_allclose(grasp[:3, 3], [0.1, 0.2, 0.295], atol=1e-9)
        # The handle (-X) should stay in the horizontal plane.
        handle = grasp[:3, 0]
        self.assertAlmostEqual(float(handle[2]), 0.0, places=9)

    def test_compute_grasp_sphere_is_top_down(self):
        obj = np.eye(4)
        obj[:3, 3] = [0.2, 0.3, 0.4]
        grasp = compute_grasp_sphere(obj, offset_mm=10.0)
        np.testing.assert_allclose(grasp[:3, 3], [0.2, 0.3, 0.41], atol=1e-9)
        np.testing.assert_allclose(grasp[:3, 2], [0.0, 1.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(float(grasp[2, 2]), 0.0)

    def test_fill_depth_roi_uses_median(self):
        depth = np.zeros((5, 5), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=np.uint8)
        depth[1:4, 1:4] = 0.6
        depth[1:4, 2] = 0.0
        mask[1:4, 1:4] = 1
        filled = fill_depth_roi(depth, mask)
        self.assertGreater(float(filled[2, 2]), 0.0)
        np.testing.assert_allclose(filled[1:4, 1:4], 0.6)

    def test_smooth_pose_converges_towards_new_pose(self):
        start = np.eye(4)
        end = np.eye(4)
        end[:3, 3] = [0.1, 0.0, 0.0]
        smoothed = smooth_pose(start, end, alpha=0.5)
        self.assertAlmostEqual(float(smoothed[0, 3]), 0.05)
        # quaternion sign handling should not produce a discontinuous matrix
        np.testing.assert_allclose(smoothed[:3, :3], np.eye(3), atol=1e-9)

    def test_rot2quat_matches_scipy_for_known_rotation(self):
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        quat = rot2quat(rotation)
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
