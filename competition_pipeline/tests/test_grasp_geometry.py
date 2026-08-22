#!/usr/bin/env python3
"""抓取高度后处理的边界用例。

这些数字直接对应明天要抓的东西。任何一条挂掉都意味着现场会压爆物体或把夹爪
怼进桌面，所以断言写得比较死。
"""

import unittest

import numpy as np

from competition_pipeline.grasp_geometry import (
    JAW_CAVITY_DEPTH_MM,
    SAFETY_CLEARANCE_MM,
    check_graspable,
    cloud_top_consistency,
    grasp_height_mm,
    object_extent_user1,
    place_height_mm,
)


def _upright_bounds(diameter_mm, height_mm, origin_offset_mm=0.0, long_axis=2):
    """构造一个直立物体的米制包围盒。

    ``origin_offset_mm``: 网格原点相对包围盒中心沿长轴的偏移（apple/雀巢咖啡就是
    这种网格）。``long_axis``: 长轴是物体系的第几根轴（罐子=Z, 苹果/新瓶=Y, 柠檬=X）。
    """
    half = np.zeros(3)
    for axis in range(3):
        half[axis] = (height_mm if axis == long_axis else diameter_mm) / 2000.0
    center = np.zeros(3)
    center[long_axis] = origin_offset_mm / 1000.0
    return np.asarray([center - half, center + half])


def _pose(xyz_mm=(0.0, 0.0, 0.0), rotation=None):
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(xyz_mm, dtype=float) / 1000.0
    return pose


def _rot_x90():
    """物体系 +Y 转到用户系 +Z —— apple / 可乐 / 雀巢那类 Y 长轴网格的直立姿态。"""
    return np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _rot_z(degrees):
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.asarray([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


class ExtentTest(unittest.TestCase):
    def test_upright_can_reports_height_diameter_and_top(self):
        """旺仔罐 66x66x92mm，原点在几何中心，立在桌面上。"""
        bounds = _upright_bounds(66.0, 92.0)
        extent = object_extent_user1(_pose((0.0, 0.0, 46.0)), bounds, "cylinder")
        self.assertAlmostEqual(extent.height_mm, 92.0, places=6)
        self.assertAlmostEqual(extent.grasp_width_mm, 66.0, places=6)
        self.assertAlmostEqual(extent.z_bottom_mm, 0.0, places=6)
        self.assertAlmostEqual(extent.z_top_mm, 92.0, places=6)
        self.assertAlmostEqual(extent.z_center_mm, 46.0, places=6)

    def test_yaw_does_not_inflate_the_grasp_width(self):
        """绕 Z 偏航 45° 时，轴对齐包围盒会把直径放大 41%，宽度闸门会误伤。"""
        bounds = _upright_bounds(66.0, 92.0)
        for yaw in (0.0, 30.0, 45.0, 137.0):
            extent = object_extent_user1(
                _pose((0.0, 0.0, 46.0), _rot_z(yaw)), bounds, "cylinder"
            )
            self.assertAlmostEqual(extent.grasp_width_mm, 66.0, places=6,
                                   msg="yaw={}".format(yaw))
            self.assertAlmostEqual(extent.height_mm, 92.0, places=6)

    def test_a_mesh_whose_origin_is_at_its_base_still_reports_the_true_centre(self):
        """apple 的网格原点在果底下方，直接用位姿平移当中心会低 44mm。

        这里把苹果摆成"底面贴桌面"：原点在 z=5.1mm 处（果底再往下 5.1mm 就是原点）。
        """
        # apple: 直径 75mm, 长轴 Y 88mm, 包围盒中心相对原点偏 +49.1mm。
        # 摆成底面贴桌面 -> 网格原点要落在桌面**下方** 5.1mm。
        bounds = _upright_bounds(75.0, 88.0, origin_offset_mm=49.1, long_axis=1)
        origin_z = -5.1
        extent = object_extent_user1(
            _pose((0.0, 0.0, origin_z), _rot_x90()), bounds, "sphere"
        )
        self.assertAlmostEqual(extent.z_bottom_mm, 0.0, places=4)
        self.assertAlmostEqual(extent.z_top_mm, 88.0, places=4)
        self.assertAlmostEqual(extent.z_center_mm, 44.0, places=4)
        self.assertAlmostEqual(extent.grasp_width_mm, 75.0, places=4)
        # 关键：真实中心 44mm 与位姿平移 -5.1mm 相差 49.1mm。旧代码用位姿平移
        # 当抓取点，就是往桌面下方 5mm 处伸 —— 直接怼进桌子。
        self.assertAlmostEqual(extent.z_center_mm - origin_z, 49.1, places=4)

    def test_a_lying_lemon_uses_the_short_horizontal_axis_as_grasp_width(self):
        """横躺柠檬 89.9(长) x 68.1 x 69.7：顶抓跨短轴合拢，宽度不该取 89.9。"""
        bounds = np.asarray([
            [-0.04495, -0.03405, -0.03485],
            [0.04495, 0.03405, 0.03485],
        ])
        extent = object_extent_user1(_pose((0.0, 0.0, 34.85)), bounds, "elongated")
        # 竖直轴是 Z(69.7)，水平两轴是 89.9 与 68.1 -> 取小的
        self.assertAlmostEqual(extent.grasp_width_mm, 68.1, places=4)


class GraspHeightTest(unittest.TestCase):
    def _can(self, height_mm, diameter_mm=66.0):
        bounds = _upright_bounds(diameter_mm, height_mm)
        return object_extent_user1(
            _pose((0.0, 0.0, height_mm / 2.0)), bounds, "cylinder"
        )

    def test_115mm_can_engages_28_75mm(self):
        """标准 330ml 易拉罐。中点规则 = 高度/4。"""
        grasp = grasp_height_mm(self._can(115.0), "cylinder")
        self.assertAlmostEqual(grasp.engage_mm, 28.75, places=6)
        self.assertAlmostEqual(grasp.z_mm, 86.25, places=6)
        self.assertFalse(grasp.clamped)

    def test_245mm_cola_bottle_stays_within_the_cavity(self):
        """trans/可口可乐.glb 实测 245mm 高。旧的对准中心会要求 122.5mm > 80mm。"""
        extent = self._can(245.0, diameter_mm=54.9)
        grasp = grasp_height_mm(extent, "cylinder")
        self.assertAlmostEqual(grasp.engage_mm, 61.25, places=6)
        self.assertFalse(grasp.clamped)
        # 对照：旧规则(对准中心)所需的伸入深度会超过腔体深度 -> 压爆
        old_engage = extent.z_top_mm - extent.z_center_mm
        self.assertAlmostEqual(old_engage, 122.5, places=6)
        self.assertGreater(old_engage, JAW_CAVITY_DEPTH_MM)

    def test_169mm_nescafe_bottle(self):
        grasp = grasp_height_mm(self._can(169.0, diameter_mm=60.0), "cylinder")
        self.assertAlmostEqual(grasp.engage_mm, 42.25, places=6)
        self.assertFalse(grasp.clamped)

    def test_the_old_rule_crushes_anything_taller_than_160mm(self):
        """把"为什么必须改"钉成断言。

        旧规则(对准中心)的无碰上限是 ``2 x 腔体深度`` = 160mm。
        新规则(中点)不触发钳位的上限是 ``4 x (腔体深度 - 余量)`` = 260mm。
        260mm 以上会触发钳位 —— 那是**安全方向**(抓得更浅)，不是失败。
        """
        limit = JAW_CAVITY_DEPTH_MM - SAFETY_CLEARANCE_MM
        for height in (161.0, 200.0, 245.0, 259.0):
            extent = self._can(height)
            old_engage = extent.z_top_mm - extent.z_center_mm
            new = grasp_height_mm(extent, "cylinder")
            self.assertGreater(old_engage, JAW_CAVITY_DEPTH_MM,
                               "旧规则本应超限 height={}".format(height))
            self.assertFalse(new.clamped,
                             "新规则不该触发钳位 height={}".format(height))
            self.assertLessEqual(new.engage_mm, limit)

    def test_the_no_clamp_boundary_is_four_times_the_usable_cavity(self):
        """边界写死成断言，免得有人调了余量却不知道上限跟着变。"""
        limit = JAW_CAVITY_DEPTH_MM - SAFETY_CLEARANCE_MM     # 65mm
        boundary = 4.0 * limit                                # 260mm
        self.assertFalse(grasp_height_mm(self._can(boundary), "cylinder").clamped)
        self.assertTrue(grasp_height_mm(self._can(boundary + 1.0), "cylinder").clamped)

    def test_clamping_always_grasps_shallower_never_deeper(self):
        """钳位只会让伸入变浅。任何高度都不可能压爆，这是整套方案的兜底保证。"""
        for height in (100.0, 260.0, 400.0, 1000.0):
            grasp = grasp_height_mm(self._can(height), "cylinder")
            self.assertLessEqual(grasp.engage_mm, grasp.requested_engage_mm + 1e-9)
            self.assertLessEqual(
                grasp.engage_mm, JAW_CAVITY_DEPTH_MM - SAFETY_CLEARANCE_MM + 1e-9,
                "height={} 的伸入深度超过腔体可用深度".format(height),
            )

    def test_an_absurdly_tall_object_is_clamped_to_the_cavity_limit(self):
        grasp = grasp_height_mm(self._can(400.0), "cylinder")
        self.assertTrue(grasp.clamped)
        self.assertAlmostEqual(grasp.engage_mm, 65.0, places=6)   # 80 - 15
        self.assertAlmostEqual(grasp.requested_engage_mm, 100.0, places=6)
        self.assertAlmostEqual(grasp.z_mm, 400.0 - 65.0, places=6)

    def test_fruit_aims_at_the_centre(self):
        bounds = _upright_bounds(75.0, 75.0)
        extent = object_extent_user1(_pose((0.0, 0.0, 37.5)), bounds, "sphere")
        grasp = grasp_height_mm(extent, "sphere")
        self.assertAlmostEqual(grasp.z_mm, extent.z_center_mm, places=6)
        self.assertAlmostEqual(grasp.z_mm, 37.5, places=6)
        self.assertFalse(grasp.clamped)
        self.assertEqual(grasp.rule, "对准中心")

    def test_grasp_point_is_always_above_the_object_centre_for_cylinders(self):
        """中点规则的意义：抓取点必须落在 3/4 高度处，才能高于下部遮挡。"""
        extent = self._can(115.0)
        grasp = grasp_height_mm(extent, "cylinder")
        fraction = (grasp.z_mm - extent.z_bottom_mm) / extent.height_mm
        self.assertAlmostEqual(fraction, 0.75, places=6)


class GateTest(unittest.TestCase):
    def _extent(self, **overrides):
        bounds = _upright_bounds(
            overrides.pop("diameter_mm", 66.0), overrides.pop("height_mm", 115.0)
        )
        z = overrides.pop("origin_z_mm", 57.5)
        return object_extent_user1(_pose((0.0, 0.0, z)), bounds, "cylinder")

    def test_too_wide_is_refused_and_says_it_is_mechanical(self):
        extent = self._extent(diameter_mm=95.0)
        grasp = grasp_height_mm(extent, "cylinder")
        reasons = check_graspable(extent, grasp, jaw_max_open_mm=80.0)
        self.assertEqual(len(reasons), 1)
        self.assertIn("机械限制不是软件问题", reasons[0])

    def test_a_can_that_fits_is_accepted(self):
        extent = self._extent(diameter_mm=66.0)
        grasp = grasp_height_mm(extent, "cylinder")
        self.assertEqual(check_graspable(extent, grasp, jaw_max_open_mm=80.0), [])

    def test_an_implausible_top_height_is_refused(self):
        extent = self._extent(height_mm=115.0, origin_z_mm=600.0)
        grasp = grasp_height_mm(extent, "cylinder")
        reasons = check_graspable(extent, grasp, jaw_max_open_mm=80.0)
        self.assertTrue(any("不在合理区间" in reason for reason in reasons))

    def test_a_grasp_point_below_the_table_is_refused(self):
        """z=0 是桌面。抓取点跑到桌面下方一定是错的。"""
        bounds = _upright_bounds(66.0, 115.0)
        extent = object_extent_user1(_pose((0.0, 0.0, -200.0)), bounds, "cylinder")
        grasp = grasp_height_mm(extent, "cylinder")
        reasons = check_graspable(extent, grasp, jaw_max_open_mm=80.0)
        self.assertTrue(any("桌面下方" in reason for reason in reasons))


class PlaceHeightTest(unittest.TestCase):
    def test_placing_returns_the_object_bottom_to_the_table(self):
        bounds = _upright_bounds(66.0, 115.0)
        extent = object_extent_user1(_pose((0.0, 0.0, 57.5)), bounds, "cylinder")
        grasp = grasp_height_mm(extent, "cylinder")
        place_z = place_height_mm(extent, grasp, clearance_mm=2.0)
        # 松爪后物体底面应落在 z≈0（余量 2mm）
        object_bottom_at_place = place_z - (
            (extent.z_top_mm - grasp.engage_mm) - extent.z_bottom_mm
        )
        self.assertAlmostEqual(object_bottom_at_place, 2.0, places=6)
        self.assertAlmostEqual(place_z, 115.0 - 28.75 + 2.0, places=6)


class CloudConsistencyTest(unittest.TestCase):
    """矩形掩膜里混着桌面和邻近物体，校验必须按半径取点、按分位数取顶。"""

    def _cloud(self, points):
        return np.asarray(points, dtype=np.float64)

    def test_agreeing_cloud_passes(self):
        points = [[0.0, 0.0, 115.0]] * 50 + [[200.0, 200.0, 300.0]] * 50
        ok, top, used = cloud_top_consistency(
            115.0, self._cloud(points), (0.0, 0.0), radius_mm=40.0
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(top, 115.0, places=6)
        self.assertEqual(used, 50)

    def test_a_taller_neighbour_outside_the_radius_is_ignored(self):
        """旁边更高的物体在框内但不在半径内，不能污染顶面判断。"""
        points = [[0.0, 0.0, 115.0]] * 40 + [[120.0, 0.0, 250.0]] * 40
        ok, top, _used = cloud_top_consistency(
            115.0, self._cloud(points), (0.0, 0.0), radius_mm=40.0
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(top, 115.0, places=6)

    def test_a_real_disagreement_is_reported(self):
        points = [[0.0, 0.0, 180.0]] * 50
        ok, top, _used = cloud_top_consistency(
            115.0, self._cloud(points), (0.0, 0.0), radius_mm=40.0
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(top, 180.0, places=6)

    def test_too_few_points_does_not_block(self):
        ok, top, used = cloud_top_consistency(
            115.0, self._cloud([[0.0, 0.0, 900.0]]), (0.0, 0.0), radius_mm=40.0
        )
        self.assertTrue(ok)
        self.assertIsNone(top)
        self.assertEqual(used, 1)


if __name__ == "__main__":
    unittest.main()
