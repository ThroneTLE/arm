#!/usr/bin/env python3
"""多物体放置槽位与"已放置区域"排除的测试。

连抓多个物体时 抓 -> 放 -> 回复位 -> 重新拍照 -> 继续抓，会连锁出三个问题：
堆叠、二次识别、遮挡。这里把三条的处理都钉住。
"""

import unittest
from pathlib import Path

import yaml

from competition_pipeline.place_layout import (
    PlaceLayout,
    is_in_placed_region,
    layout_from_config,
)

COMPETITION = (
    Path(__file__).resolve().parents[2]
    / "competition_pipeline" / "config" / "competition.yaml"
)


class SlotGeometryTest(unittest.TestCase):
    def setUp(self):
        self.layout = PlaceLayout(
            origin_xy_mm=(-170.0, 170.0), direction=(0.0, -1.0),
            pitch_mm=100.0, count=4, table_half_mm=246.5,
            exclusion_radius_mm=45.0,
        )

    def test_slots_march_along_the_configured_direction(self):
        self.assertEqual(
            [(round(x), round(y)) for x, y in self.layout.all_slots_mm()],
            [(-170, 170), (-170, 70), (-170, -30), (-170, -130)],
        )

    def test_direction_is_normalised_so_pitch_is_a_real_distance(self):
        """方向给非单位向量时，间距不能被向量长度放大。"""
        layout = PlaceLayout(
            origin_xy_mm=(0.0, 0.0), direction=(0.0, -3.0), pitch_mm=100.0,
            count=2,
        )
        self.assertAlmostEqual(layout.slot_xy_mm(1)[1], -100.0, places=6)

    def test_a_zero_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            PlaceLayout(direction=(0.0, 0.0))

    def test_running_out_of_slots_says_so_instead_of_wrapping(self):
        """用满了必须显式报错 —— 悄悄回到 0 号会把新物体放到旧物体上。"""
        with self.assertRaises(IndexError) as ctx:
            self.layout.slot_xy_mm(4)
        self.assertIn("放置区已经用满", str(ctx.exception))


class LayoutValidationTest(unittest.TestCase):
    def test_a_good_layout_passes(self):
        layout = PlaceLayout(
            origin_xy_mm=(-170.0, 170.0), pitch_mm=100.0, count=4,
            exclusion_radius_mm=45.0,
        )
        self.assertEqual(layout.validate(largest_object_diameter_mm=75.0), [])

    def test_slots_off_the_table_are_reported(self):
        layout = PlaceLayout(
            origin_xy_mm=(-170.0, 170.0), pitch_mm=100.0, count=8,
        )
        reasons = layout.validate()
        self.assertTrue(any("超出桌面" in reason for reason in reasons))

    def test_a_pitch_smaller_than_the_object_is_reported(self):
        """间距小于物体直径 -> 相邻两个会碰到 -> 倒 -> 违反"放置时需要直立"。"""
        layout = PlaceLayout(pitch_mm=60.0, count=3, exclusion_radius_mm=25.0)
        reasons = layout.validate(largest_object_diameter_mm=75.0)
        self.assertTrue(any("会碰到" in reason for reason in reasons))

    def test_an_exclusion_radius_wider_than_half_the_pitch_is_reported(self):
        """排除半径过大会把相邻槽位也当成"已放置"，导致目标被误剔除。"""
        layout = PlaceLayout(pitch_mm=100.0, exclusion_radius_mm=60.0, count=2)
        reasons = layout.validate()
        self.assertTrue(any("相邻槽位" in reason for reason in reasons))


class PlacedRegionExclusionTest(unittest.TestCase):
    def setUp(self):
        self.layout = PlaceLayout(
            origin_xy_mm=(-170.0, 170.0), direction=(0.0, -1.0),
            pitch_mm=100.0, count=4, exclusion_radius_mm=45.0,
        )

    def test_nothing_is_excluded_before_anything_is_placed(self):
        self.assertFalse(is_in_placed_region((-170.0, 170.0), self.layout, []))

    def test_an_object_sitting_on_an_occupied_slot_is_excluded(self):
        """重新拍照时刚放好的物体会被再检测一遍，不能再当成抓取目标。"""
        self.assertTrue(is_in_placed_region((-170.0, 170.0), self.layout, [0]))
        self.assertTrue(is_in_placed_region((-160.0, 155.0), self.layout, [0]))

    def test_an_object_still_in_the_pile_is_not_excluded(self):
        self.assertFalse(is_in_placed_region((0.0, 0.0), self.layout, [0, 1]))
        self.assertFalse(is_in_placed_region((120.0, -80.0), self.layout, [0]))

    def test_an_unoccupied_slot_does_not_exclude(self):
        """还没放东西的槽位不该排除任何检测结果。"""
        self.assertFalse(is_in_placed_region((-170.0, 70.0), self.layout, [0]))
        self.assertTrue(is_in_placed_region((-170.0, 70.0), self.layout, [0, 1]))

    def test_the_boundary_is_inclusive(self):
        self.assertTrue(is_in_placed_region((-170.0, 125.0), self.layout, [0]))
        self.assertFalse(is_in_placed_region((-170.0, 124.0), self.layout, [0]))


class ConfigContractTest(unittest.TestCase):
    def setUp(self):
        self.data = yaml.safe_load(COMPETITION.read_text(encoding="utf-8")) or {}
        self.layout = layout_from_config(self.data.get("workspace"))

    def test_the_shipped_layout_is_self_consistent(self):
        self.assertEqual(
            self.layout.validate(largest_object_diameter_mm=75.0), [],
        )

    def test_slot_zero_is_the_configured_single_target_place_point(self):
        """单目标和多目标必须放到同一个地方，否则两条路径行为不一致。"""
        configured = self.data["workspace"]["place_user_xy_mm"]
        slot_zero = self.layout.slot_xy_mm(0)
        self.assertAlmostEqual(slot_zero[0], float(configured[0]), places=6)
        self.assertAlmostEqual(slot_zero[1], float(configured[1]), places=6)

    def test_it_agrees_with_the_ucs_grasp_defaults(self):
        from tool.visual_grasp_pipeline.ucs_grasp import (
            UCS_PLACE_X_MM, UCS_PLACE_Y_MM,
        )

        slot_zero = self.layout.slot_xy_mm(0)
        self.assertAlmostEqual(slot_zero[0], float(UCS_PLACE_X_MM), places=6)
        self.assertAlmostEqual(slot_zero[1], float(UCS_PLACE_Y_MM), places=6)

    def test_every_slot_is_inside_the_ucs_grasp_safety_box(self):
        """槽位若超出 ucs_grasp 的 XY 硬限，放置会在执行时被拒。"""
        from tool.visual_grasp_pipeline.ucs_grasp import SAFE_XY_MM

        for index, (x, y) in enumerate(self.layout.all_slots_mm()):
            with self.subTest(slot=index):
                self.assertLessEqual(abs(x), float(SAFE_XY_MM))
                self.assertLessEqual(abs(y), float(SAFE_XY_MM))


if __name__ == "__main__":
    unittest.main()
