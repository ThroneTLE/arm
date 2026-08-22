#!/usr/bin/env python3

import unittest

import numpy as np

from competition_pipeline.visual_grasp_bridge import (
    build_pick_place_plan,
    validate_plan,
)


class VisualGraspBridgeTest(unittest.TestCase):
    def test_composes_eye_in_hand_chain_and_places_at_user_y_minus_50(self):
        camera_from_object = np.eye(4)
        camera_from_object[:3, 3] = [0.01, 0.02, 0.04]
        user1_from_tcp = np.eye(4)
        user1_from_tcp[:3, 3] = [0.10, -0.02, 0.30]
        tcp_from_camera = np.eye(4)
        tcp_from_camera[:3, 3] = [-0.10, 0.0, -0.30]

        plan = build_pick_place_plan(
            camera_from_object,
            user1_from_tcp,
            tcp_from_camera,
            np.eye(4),
            grasp_type="elongated",
            place_offset_user_mm=(0.0, -50.0, 0.0),
        )

        self.assertTrue(
            np.allclose(plan["object"]["xyz_mm"], [10.0, 0.0, 40.0])
        )
        grasp = np.asarray(plan["grasp_tcp"]["xyz_mm"])
        place = np.asarray(plan["place_tcp"]["xyz_mm"])
        self.assertTrue(np.allclose(place - grasp, [0.0, -50.0, 0.0]))
        grasp_frame = np.asarray(plan["grasp_frame"]["matrix"])
        self.assertTrue(np.allclose(grasp_frame[:3, 2], [0.0, 0.0, -1.0]))

    def test_tcp_from_grasp_is_applied_to_both_targets(self):
        camera_from_object = np.eye(4)
        camera_from_object[:3, 3] = [0.0, 0.0, 0.05]
        tcp_from_grasp = np.eye(4)
        tcp_from_grasp[:3, 3] = [0.0, 0.0, 0.11]
        plan = build_pick_place_plan(
            camera_from_object,
            np.eye(4),
            np.eye(4),
            tcp_from_grasp,
        )
        grasp_frame = np.asarray(plan["grasp_frame"]["matrix"])
        grasp_tcp = np.asarray(plan["grasp_tcp"]["matrix"])
        self.assertTrue(np.allclose(grasp_tcp @ tcp_from_grasp, grasp_frame))

    def test_configured_tool_transform_keeps_tcp_above_lemon(self):
        camera_from_object = np.eye(4)
        camera_from_object[:3, 3] = [0.0, 0.0, 0.035]
        tcp_from_grasp = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.11],
            [0.0, 0.0, 0.0, 1.0],
        ])
        plan = build_pick_place_plan(
            camera_from_object,
            np.eye(4),
            np.eye(4),
            tcp_from_grasp,
            grasp_type="elongated",
        )
        self.assertAlmostEqual(plan["object"]["xyz_mm"][2], 35.0)
        self.assertAlmostEqual(plan["grasp_tcp"]["xyz_mm"][2], 145.0)

    def test_origin_quality_gate_is_fail_closed(self):
        camera_from_object = np.eye(4)
        camera_from_object[:3, 3] = [0.08, 0.0, 0.05]
        plan = build_pick_place_plan(
            camera_from_object, np.eye(4), np.eye(4), np.eye(4)
        )
        reasons = validate_plan(
            plan,
            origin_xy_tolerance_mm=50.0,
            workspace_min_mm=(-1000.0, -1000.0, -100.0),
            workspace_max_mm=(2000.0, 2000.0, 2000.0),
            lift_mm=80.0,
            confidence=0.95,
            minimum_confidence=0.85,
            depth_coverage=0.8,
            minimum_depth_coverage=0.15,
            depth_center_delta_mm=30.0,
            maximum_depth_center_delta_mm=80.0,
        )
        self.assertTrue(any("origin XY error" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
