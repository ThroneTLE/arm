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

    def test_origin_gate_only_fires_when_explicitly_requested(self):
        """单物体 demo 的遗留闸门：默认必须关闭。

        它要求物体摆在用户系原点 50mm 内。真实赛题是 49.3cm 桌面散放，启用它会把
        所有目标判死。只有标定/复现场景才显式传值。
        """
        camera_from_object = np.eye(4)
        camera_from_object[:3, 3] = [0.08, 0.0, 0.05]
        plan = build_pick_place_plan(
            camera_from_object, np.eye(4), np.eye(4), np.eye(4),
            mesh_bounds_m=_can_bounds(66.0, 100.0),
        )
        common = dict(
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
        default_reasons = validate_plan(plan, **common)
        self.assertFalse(
            any("原点" in reason for reason in default_reasons),
            "默认不该启用单物体原点闸门：{}".format(default_reasons),
        )
        opted_in = validate_plan(plan, origin_xy_tolerance_mm=50.0, **common)
        self.assertTrue(any("原点" in reason for reason in opted_in))


def _can_bounds(diameter_mm, height_mm):
    """直立圆柱的米制包围盒，长轴 Z，原点=几何中心。"""
    half = np.asarray(
        [diameter_mm / 2000.0, diameter_mm / 2000.0, height_mm / 2000.0]
    )
    return np.asarray([-half, half])


class GraspHeightIntegrationTest(unittest.TestCase):
    """把 grasp_geometry 接进 build_pick_place_plan 之后的端到端行为。"""

    def _plan(self, height_mm, diameter_mm=66.0, grasp_type="cylinder", **kwargs):
        # 物体立在桌面上：包围盒中心在 height/2
        camera_from_object = np.eye(4)
        camera_from_object[:3, 3] = [0.0, 0.0, height_mm / 2000.0]
        return build_pick_place_plan(
            camera_from_object, np.eye(4), np.eye(4), np.eye(4),
            grasp_type=grasp_type,
            grasp_offset_mm=0.0,
            mesh_bounds_m=_can_bounds(diameter_mm, height_mm),
            **kwargs,
        )

    def test_a_can_is_grasped_at_three_quarter_height(self):
        plan = self._plan(115.0)
        self.assertAlmostEqual(plan["grasp_height"]["z_mm"], 86.25, places=3)
        self.assertAlmostEqual(plan["grasp_height"]["engage_mm"], 28.75, places=3)
        self.assertFalse(plan["grasp_height"]["clamped"])
        self.assertAlmostEqual(plan["grasp_tcp"]["xyz_mm"][2], 86.25, places=3)

    def test_a_245mm_cola_bottle_no_longer_demands_more_than_the_cavity(self):
        """旧行为(对准中心)会要求 122.5mm 伸入 > 80mm 腔体 -> 压爆。"""
        plan = self._plan(245.0, diameter_mm=54.9)
        self.assertAlmostEqual(plan["grasp_height"]["engage_mm"], 61.25, places=3)
        self.assertFalse(plan["grasp_height"]["clamped"])
        old_engage = (plan["object_extent"]["z_top_mm"]
                      - plan["object_extent"]["z_center_mm"])
        self.assertAlmostEqual(old_engage, 122.5, places=3)

    def test_without_mesh_bounds_the_plan_is_refused_not_attempted(self):
        """没有物体高度就无法保证不压爆，必须拒绝而不是"尽力而为"。"""
        camera_from_object = np.eye(4)
        camera_from_object[:3, 3] = [0.0, 0.0, 0.05]
        plan = build_pick_place_plan(
            camera_from_object, np.eye(4), np.eye(4), np.eye(4)
        )
        self.assertNotIn("object_extent", plan)
        reasons = validate_plan(
            plan,
            workspace_min_mm=(-1000.0, -1000.0, -100.0),
            workspace_max_mm=(2000.0, 2000.0, 2000.0),
            lift_mm=80.0, confidence=0.95, minimum_confidence=0.85,
            depth_coverage=0.8, minimum_depth_coverage=0.15,
            depth_center_delta_mm=30.0, maximum_depth_center_delta_mm=80.0,
        )
        self.assertTrue(any("缺少物体尺寸" in reason for reason in reasons))

    def test_place_target_returns_the_object_bottom_to_the_table(self):
        """任务书要求"放置时需要直立"：松爪后底面应坐回桌面。"""
        plan = self._plan(115.0, place_user_xy_mm=(0.0, -160.0))
        place = plan["place_tcp"]["xyz_mm"]
        self.assertAlmostEqual(place[0], 0.0, places=3)
        self.assertAlmostEqual(place[1], -160.0, places=3)
        # 指尖在 z_top - engage 处抓住，物体底面在该点下方 (z_top-engage)-z_bottom
        engage = plan["grasp_height"]["engage_mm"]
        below = (115.0 - engage) - 0.0
        self.assertAlmostEqual(place[2] - below, 2.0, places=3)  # place_clearance

    def test_an_object_off_the_table_is_refused(self):
        plan = self._plan(115.0)
        # 把物体挪到桌面外
        plan["object_extent"]["center_xy_mm"] = [300.0, 0.0]
        reasons = validate_plan(
            plan,
            workspace_min_mm=(-1000.0, -1000.0, -100.0),
            workspace_max_mm=(2000.0, 2000.0, 2000.0),
            lift_mm=80.0, confidence=0.95, minimum_confidence=0.85,
            depth_coverage=0.8, minimum_depth_coverage=0.15,
            depth_center_delta_mm=30.0, maximum_depth_center_delta_mm=80.0,
            table_half_size_mm=246.5,
        )
        self.assertTrue(any("超出桌面范围" in reason for reason in reasons))

    def test_an_object_wider_than_the_jaw_is_refused_as_mechanical(self):
        plan = self._plan(115.0, diameter_mm=95.0)
        reasons = validate_plan(
            plan,
            workspace_min_mm=(-1000.0, -1000.0, -100.0),
            workspace_max_mm=(2000.0, 2000.0, 2000.0),
            lift_mm=80.0, confidence=0.95, minimum_confidence=0.85,
            depth_coverage=0.8, minimum_depth_coverage=0.15,
            depth_center_delta_mm=30.0, maximum_depth_center_delta_mm=80.0,
            jaw_max_open_mm=85.0, width_margin_mm=6.0,
        )
        self.assertTrue(any("机械限制不是软件问题" in reason for reason in reasons))

    def test_a_66mm_can_fits_the_85mm_jaw(self):
        plan = self._plan(115.0, diameter_mm=66.0)
        reasons = validate_plan(
            plan,
            workspace_min_mm=(-1000.0, -1000.0, -100.0),
            workspace_max_mm=(2000.0, 2000.0, 2000.0),
            lift_mm=80.0, confidence=0.95, minimum_confidence=0.85,
            depth_coverage=0.8, minimum_depth_coverage=0.15,
            depth_center_delta_mm=30.0, maximum_depth_center_delta_mm=80.0,
            table_half_size_mm=246.5, jaw_max_open_mm=85.0,
        )
        self.assertEqual(reasons, [])

    def test_fruit_is_grasped_at_its_centre(self):
        plan = self._plan(75.0, diameter_mm=75.0, grasp_type="sphere")
        self.assertEqual(plan["grasp_height"]["rule"], "对准中心")
        self.assertAlmostEqual(plan["grasp_height"]["z_mm"], 37.5, places=3)


if __name__ == "__main__":
    unittest.main()
