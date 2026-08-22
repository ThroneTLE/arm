#!/usr/bin/env python3
"""oak_vision_node 抓取高度后处理的回归测试。

这是明天真正会跑的那条路径：节点算出 ``user1_grasp`` 之后，必须把抓取点从
"网格原点"换成"不会压爆、也不会怼进桌面"的位置。

``compute_grasp`` 的 docstring 原话是 "grip the middle" —— 对 245mm 的可乐瓶
就是要求伸进夹爪 122.5mm，而腔体只有 80mm。隔壁组的瓶子就是这样爆的。
"""

import unittest

import numpy as np

from tool.visual_grasp_pipeline.oak_vision_node import apply_grasp_height_rule


GRIPPER = {
    "jaw_cavity_depth_mm": 80.0,
    "safety_clearance_mm": 15.0,
    "jaw_max_open_mm": 85.0,
    "width_margin_mm": 6.0,
}


def _bounds(diameter_mm, height_mm, origin_offset_mm=0.0, long_axis=2):
    """米制包围盒。``origin_offset_mm`` 是包围盒中心相对网格原点的偏移。"""
    half = np.zeros(3)
    for axis in range(3):
        half[axis] = (height_mm if axis == long_axis else diameter_mm) / 2000.0
    center = np.zeros(3)
    center[long_axis] = origin_offset_mm / 1000.0
    return np.asarray([center - half, center + half])


def _pose(origin_z_mm, rotation=None):
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    pose[:3, 3] = [0.0, 0.0, origin_z_mm / 1000.0]
    return pose


def _grasp_at(origin_z_mm):
    """模拟 compute_grasp 的输出：抓取点就放在网格原点上。"""
    grasp = np.eye(4)
    grasp[:3, :3] = np.asarray([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0]])       # 任意非单位姿态
    grasp[:3, 3] = [0.0, 0.0, origin_z_mm / 1000.0]
    return grasp


class GraspHeightRuleTest(unittest.TestCase):
    def test_a_245mm_cola_bottle_is_no_longer_grasped_at_its_middle(self):
        """原始行为要求伸进 122.5mm > 80mm 腔体 -> 压爆。"""
        bounds = _bounds(54.9, 245.0)                   # 原点=几何中心
        pose = _pose(122.5)                             # 立在桌面上
        corrected, info = apply_grasp_height_rule(
            _grasp_at(122.5), pose, bounds, "cylinder", GRIPPER
        )
        self.assertTrue(info["available"])
        self.assertEqual(info["reasons"], [])
        self.assertAlmostEqual(info["engage_mm"], 61.25, places=3)
        self.assertFalse(info["clamped"])
        # 抓取点抬到 3/4 高度处
        self.assertAlmostEqual(corrected[2, 3] * 1000.0, 183.75, places=3)
        # 原始(网格原点)是 122.5mm，会要求伸进 122.5mm
        self.assertGreater(corrected[2, 3] * 1000.0, 122.5)

    def test_the_orientation_is_left_untouched(self):
        """只动位置。姿态是 compute_grasp 定的，后处理不许改。"""
        bounds = _bounds(66.0, 115.0)
        grasp = _grasp_at(57.5)
        corrected, _info = apply_grasp_height_rule(
            grasp, _pose(57.5), bounds, "cylinder", GRIPPER
        )
        np.testing.assert_allclose(corrected[:3, :3], grasp[:3, :3], atol=1e-15)

    def test_a_can_lands_at_three_quarter_height(self):
        bounds = _bounds(66.0, 115.0)
        corrected, info = apply_grasp_height_rule(
            _grasp_at(57.5), _pose(57.5), bounds, "cylinder", GRIPPER
        )
        self.assertAlmostEqual(corrected[2, 3] * 1000.0, 86.25, places=3)
        self.assertAlmostEqual(info["engage_mm"], 28.75, places=3)

    def test_a_mesh_origin_near_the_bottle_base_no_longer_grasps_the_table(self):
        """nescafe 实测包围盒 y ∈ [-10, 159]mm：瓶底在网格原点**下方** 10mm。

        瓶底贴桌面时原点只在 z=+10mm 处，按原点抓等于贴着桌面合爪 —— 抓空，
        运气差还会把夹爪压在桌上。
        """
        # 60 x 169mm, 长轴 Y, 包围盒中心相对原点偏 +74.5mm
        bounds = _bounds(60.0, 169.0, origin_offset_mm=74.5, long_axis=1)
        rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                               [0.0, 1.0, 0.0]])        # 物体 +Y -> 用户 +Z
        origin_z = 10.0                                 # 瓶底落在 z=0
        corrected, info = apply_grasp_height_rule(
            _grasp_at(origin_z), _pose(origin_z, rotation), bounds,
            "cylinder", GRIPPER,
        )
        self.assertAlmostEqual(info["object_top_mm"], 169.0, places=3)
        self.assertAlmostEqual(info["object_height_mm"], 169.0, places=3)
        # 网格原点只有 10mm 高；修正后应落在 3/4 高度处
        self.assertAlmostEqual(corrected[2, 3] * 1000.0, 126.75, places=3)
        self.assertAlmostEqual(info["engage_mm"], 42.25, places=3)

    def test_an_apple_is_grasped_at_its_equator_not_at_the_mesh_origin(self):
        """apple 网格原点在果底下方 5.1mm，偏包围盒中心 49.1mm。"""
        bounds = _bounds(75.0, 88.0, origin_offset_mm=49.1, long_axis=1)
        rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                               [0.0, 1.0, 0.0]])
        corrected, info = apply_grasp_height_rule(
            _grasp_at(-5.1), _pose(-5.1, rotation), bounds, "sphere", GRIPPER,
        )
        self.assertEqual(info["rule"], "对准中心")
        self.assertAlmostEqual(corrected[2, 3] * 1000.0, 44.0, places=3)
        self.assertAlmostEqual(info["grasp_width_mm"], 75.0, places=3)

    def test_grasp_xy_comes_from_the_bounding_box_not_the_mesh_origin(self):
        """XY 也要用包围盒中心：原点在水平方向偏移的网格同样会抓偏。"""
        bounds = np.asarray([
            [0.020, -0.030, 0.0],       # X 方向整体偏 +20..80mm
            [0.080, 0.030, 0.115],
        ])
        pose = np.eye(4)
        corrected, _info = apply_grasp_height_rule(
            _grasp_at(0.0), pose, bounds, "cylinder", GRIPPER
        )
        self.assertAlmostEqual(corrected[0, 3] * 1000.0, 50.0, places=3)
        self.assertAlmostEqual(corrected[1, 3] * 1000.0, 0.0, places=3)

    def test_missing_bounds_is_reported_and_not_silently_accepted(self):
        grasp = _grasp_at(57.5)
        corrected, info = apply_grasp_height_rule(
            grasp, _pose(57.5), None, "cylinder", GRIPPER
        )
        self.assertFalse(info["available"])
        self.assertTrue(info["reasons"])
        np.testing.assert_allclose(corrected, grasp)

    def test_an_object_wider_than_the_jaw_is_reported_as_mechanical(self):
        bounds = _bounds(95.0, 115.0)
        _corrected, info = apply_grasp_height_rule(
            _grasp_at(57.5), _pose(57.5), bounds, "cylinder", GRIPPER
        )
        self.assertTrue(
            any("机械限制不是软件问题" in reason for reason in info["reasons"])
        )

    def test_an_absurdly_tall_object_is_clamped_not_crushed(self):
        bounds = _bounds(60.0, 400.0)
        _corrected, info = apply_grasp_height_rule(
            _grasp_at(200.0), _pose(200.0), bounds, "cylinder", GRIPPER
        )
        self.assertTrue(info["clamped"])
        self.assertAlmostEqual(info["engage_mm"], 65.0, places=3)
        self.assertAlmostEqual(info["requested_engage_mm"], 100.0, places=3)


class OriginInvarianceTest(unittest.TestCase):
    """抓取点与**网格原点在哪**完全无关 —— 所以不需要去"修复"任何模型的原点。

    可乐/雪碧的原点是对的（已用尺子验证），苹果/雀巢的原点偏了几十毫米。
    本方法用包围盒 8 角点的均值当中心，而刚体变换下角点均值恒等于包围盒中心，
    与原点位置无关。下面把这条性质钉死：同一个物体、同一个摆放，只把网格原点
    人为挪走，抓取点必须一模一样。
    """

    def _grasp_point(self, origin_offset_mm):
        """物体固定摆在桌面上，只改变网格原点的定义位置。"""
        height, diameter = 115.0, 66.0
        bounds = _bounds(diameter, height,
                         origin_offset_mm=origin_offset_mm, long_axis=2)
        # 让包围盒底面始终落在 z=0：原点位置随偏移反向移动
        origin_z = height / 2.0 - origin_offset_mm
        corrected, info = apply_grasp_height_rule(
            _grasp_at(origin_z), _pose(origin_z), bounds, "cylinder", GRIPPER
        )
        return corrected[:3, 3] * 1000.0, info

    def test_moving_the_mesh_origin_does_not_move_the_grasp_point(self):
        reference, reference_info = self._grasp_point(0.0)
        for offset in (-74.5, -49.1, 0.0, 30.0, 200.0):
            xyz, info = self._grasp_point(offset)
            np.testing.assert_allclose(
                xyz, reference, atol=1e-9,
                err_msg="原点偏移 {}mm 改变了抓取点".format(offset),
            )
            self.assertAlmostEqual(
                info["object_top_mm"], reference_info["object_top_mm"], places=9
            )
            self.assertAlmostEqual(
                info["engage_mm"], reference_info["engage_mm"], places=9
            )

    def test_a_correct_origin_gives_exactly_the_same_answer_as_before(self):
        """对可乐/雪碧这类原点已经正确的模型，本方法与"直接用原点"结果一致。

        也就是说这次改动**没有动**已经验证过的那两个模型的行为。
        """
        xyz, _info = self._grasp_point(0.0)
        # 原点在几何中心时，包围盒中心 == 原点 -> XY 与原点一致
        self.assertAlmostEqual(xyz[0], 0.0, places=9)
        self.assertAlmostEqual(xyz[1], 0.0, places=9)


class CloudCrossCheckTest(unittest.TestCase):
    """网格缩放只有可乐/雪碧用尺子对过，其余是按模型单位推的。

    缩放错 -> 物体高度错 -> 抓取高度错，而那正是压爆的成因。点云是唯一不依赖
    CAD 的第二来源。
    """

    def _cloud(self, top_mm, count=200, xy=(0.0, 0.0)):
        points = np.zeros((count, 3))
        points[:, 0] = xy[0]
        points[:, 1] = xy[1]
        points[:, 2] = top_mm
        return points

    def test_agreeing_cloud_passes_and_is_reported(self):
        bounds = _bounds(66.0, 115.0)
        _corrected, info = apply_grasp_height_rule(
            _grasp_at(57.5), _pose(57.5), bounds, "cylinder", GRIPPER,
            cloud_user1_mm=self._cloud(115.0),
        )
        self.assertEqual(info["reasons"], [])
        self.assertAlmostEqual(info["cloud_top_mm"], 115.0, places=3)
        self.assertEqual(info["cloud_points_used"], 200)

    def test_a_wrong_mesh_scale_is_caught_by_the_cloud(self):
        """模拟缩放错一倍：CAD 说 230mm 高，点云说顶面只有 115mm。"""
        bounds = _bounds(66.0, 230.0)
        _corrected, info = apply_grasp_height_rule(
            _grasp_at(115.0), _pose(115.0), bounds, "cylinder", GRIPPER,
            cloud_user1_mm=self._cloud(115.0),
        )
        self.assertTrue(
            any("object_model_scales" in reason for reason in info["reasons"]),
            info["reasons"],
        )

    def test_no_cloud_is_reported_as_unverified_not_as_failure(self):
        bounds = _bounds(66.0, 115.0)
        _corrected, info = apply_grasp_height_rule(
            _grasp_at(57.5), _pose(57.5), bounds, "cylinder", GRIPPER,
            cloud_user1_mm=None,
        )
        self.assertEqual(info["reasons"], [])
        self.assertIsNone(info["cloud_top_mm"])

    def test_a_taller_neighbour_outside_the_object_radius_is_ignored(self):
        """矩形掩膜里混着邻近物体；按物体半径取点才不会被更高的邻居污染。"""
        bounds = _bounds(66.0, 115.0)
        cloud = np.vstack([
            self._cloud(115.0, count=150),
            self._cloud(300.0, count=150, xy=(200.0, 0.0)),   # 远处更高的物体
        ])
        _corrected, info = apply_grasp_height_rule(
            _grasp_at(57.5), _pose(57.5), bounds, "cylinder", GRIPPER,
            cloud_user1_mm=cloud,
        )
        self.assertEqual(info["reasons"], [])
        self.assertAlmostEqual(info["cloud_top_mm"], 115.0, places=3)


class GripperGeometryConfigTest(unittest.TestCase):
    def test_competition_yaml_carries_the_measured_cavity_depth(self):
        from pathlib import Path

        from tool.visual_grasp_pipeline.oak_vision_node import (
            load_gripper_geometry,
        )

        config = (
            Path(__file__).resolve().parents[3]
            / "competition_pipeline" / "config" / "competition.yaml"
        )
        geometry = load_gripper_geometry(config)
        self.assertAlmostEqual(geometry["jaw_cavity_depth_mm"], 80.0)
        self.assertGreater(geometry["safety_clearance_mm"], 0.0)
        self.assertIsNotNone(geometry["jaw_max_open_mm"])


if __name__ == "__main__":
    unittest.main()
