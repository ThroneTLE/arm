#!/usr/bin/env python3
"""用户坐标系1 与工作台约定的契约测试。

这些数字散落在 ``competition.yaml`` 和 ``ucs_grasp.py`` 两处，一旦对不上，
后果是抓取/放置整体偏移或跑出桌面 —— 而且不会报任何错。这里把它们钉在一起。

用户坐标系1（2026-08-22 现场标定）::

    原点 = 49.3cm 方桌的桌面中心
    +X = 前方   +Y = 左方   +Z = 上方
    z = 0 就是桌面
"""

import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPETITION = REPO_ROOT / "competition_pipeline" / "config" / "competition.yaml"


def _config():
    return yaml.safe_load(COMPETITION.read_text(encoding="utf-8")) or {}


class WorkspaceContractTest(unittest.TestCase):
    def setUp(self):
        self.data = _config()
        self.workspace = self.data.get("workspace", {}) or {}

    def test_table_half_size_matches_the_493mm_table(self):
        self.assertAlmostEqual(
            float(self.workspace["table_size_mm"]), 493.0, places=3
        )
        self.assertAlmostEqual(
            float(self.workspace["table_half_size_mm"]),
            float(self.workspace["table_size_mm"]) / 2.0,
            places=3,
        )

    def test_place_point_sits_on_the_table(self):
        half = float(self.workspace["table_half_size_mm"])
        place = [float(value) for value in self.workspace["place_user_xy_mm"]]
        self.assertEqual(len(place), 2)
        for axis, value in zip("XY", place):
            self.assertLess(
                abs(value), half,
                "放置点 {}={} 超出桌面 ±{}mm".format(axis, value, half),
            )

    def test_place_point_is_on_the_x_negative_half(self):
        """现场决定放在 X 负半边（桌子近端），避开中央的杂物堆放区。"""
        place = [float(value) for value in self.workspace["place_user_xy_mm"]]
        self.assertLess(place[0], 0.0)

    def test_place_point_agrees_with_the_ucs_grasp_defaults(self):
        """两处写死了同一个点。对不上就会出现"UI 显示一个位置、实际放到另一个"。"""
        from tool.visual_grasp_pipeline.ucs_grasp import (
            UCS_PLACE_X_MM, UCS_PLACE_Y_MM,
        )

        place = [float(value) for value in self.workspace["place_user_xy_mm"]]
        self.assertAlmostEqual(place[0], float(UCS_PLACE_X_MM), places=3)
        self.assertAlmostEqual(place[1], float(UCS_PLACE_Y_MM), places=3)

    def test_the_ucs_grasp_xy_limit_does_not_allow_leaving_the_table(self):
        from tool.visual_grasp_pipeline.ucs_grasp import SAFE_XY_MM

        self.assertLessEqual(
            float(SAFE_XY_MM), float(self.workspace["table_half_size_mm"]) + 1e-6,
            "XY 硬限比桌子还大，目标点可以跑到桌外",
        )


class GripperGeometryContractTest(unittest.TestCase):
    def setUp(self):
        self.gripper = _config().get("gripper_geometry", {}) or {}

    def test_measured_cavity_depth_is_recorded(self):
        """腔体深度 80mm 是 2026-08-22 实测值，抓取高度全靠它。"""
        self.assertAlmostEqual(
            float(self.gripper["jaw_cavity_depth_mm"]), 80.0, places=3
        )

    def test_safety_clearance_leaves_a_usable_cavity(self):
        cavity = float(self.gripper["jaw_cavity_depth_mm"])
        clearance = float(self.gripper["safety_clearance_mm"])
        self.assertGreater(clearance, 0.0)
        self.assertLess(clearance, cavity)
        # 不触发钳位的物体高度上限 = 4 x 可用深度；要覆盖最高的可乐瓶 245mm。
        self.assertGreaterEqual(4.0 * (cavity - clearance), 245.0)

    def test_the_jaw_opening_covers_the_widest_object_on_the_table(self):
        """苹果实测 75mm 是全场最宽的。张开减余量必须容得下它。"""
        planning = _config().get("grasp_planning", {}) or {}
        usable = (float(self.gripper["jaw_max_open_mm"])
                  - float(planning.get("width_margin_mm", 6.0)))
        self.assertGreaterEqual(
            usable, 75.0,
            "可用张开 {:.1f}mm 夹不住 75mm 的苹果；量准后改 jaw_max_open_mm".format(
                usable),
        )


if __name__ == "__main__":
    unittest.main()
