#!/usr/bin/env python3
"""点画面选物的坐标映射与选中判定。

两处算错都会表现为"点了 A 却选中 B"，而且是**系统性偏移**，现场很难当场看出来：

1. ``_show_image`` 会把画面等比缩小到 920x690 以内；
2. ``tk.Label`` 又会把图片在自己的区域里**居中**。

少减一次居中留白、或少除一次缩放比，点击就会整体偏。
"""

import unittest

from tool.visual_grasp_pipeline.oak_vision_node import (
    display_to_image_pixel,
    pick_object_at_pixel,
    sort_objects_for_picking,
)


class DisplayToImagePixelTest(unittest.TestCase):
    def test_no_scaling_no_letterbox_is_identity(self):
        self.assertEqual(
            display_to_image_pixel((10, 20), (640, 480), (640, 480), (640, 480)),
            (10, 20),
        )

    def test_scaling_is_undone(self):
        """1280x960 的图缩到 640x480 贴出：点 (320,240) 对应原图 (640,480)。"""
        self.assertEqual(
            display_to_image_pixel((320, 240), (640, 480), (640, 480), (1280, 960)),
            (640, 480),
        )

    def test_letterbox_offset_is_removed(self):
        """Label 比图片大 -> 图片居中。左上角留白必须先减掉。"""
        # 图 640x480 贴在 900x700 的 Label 里 -> 留白 (130, 110)
        self.assertEqual(
            display_to_image_pixel((130, 110), (900, 700), (640, 480), (640, 480)),
            (0, 0),
        )
        self.assertEqual(
            display_to_image_pixel((130 + 64, 110 + 48), (900, 700),
                                   (640, 480), (640, 480)),
            (64, 48),
        )

    def test_scaling_and_letterbox_together(self):
        # 1920x1080 缩到 920x517 贴在 1000x700 的 Label 里
        shown = (920, 517)
        pixel = display_to_image_pixel(
            (40 + 460, 91 + 258), (1000, 700), shown, (1920, 1080)
        )
        self.assertIsNotNone(pixel)
        self.assertAlmostEqual(pixel[0], 960, delta=3)
        self.assertAlmostEqual(pixel[1], 540, delta=3)

    def test_clicking_the_letterbox_margin_returns_none(self):
        """点在黑边上不该选中任何东西，更不该当成 (0,0) 附近的物体。"""
        self.assertIsNone(
            display_to_image_pixel((10, 10), (900, 700), (640, 480), (640, 480))
        )

    def test_degenerate_sizes_return_none_instead_of_dividing_by_zero(self):
        """还没拍照就点画面 -> 尺寸是 (0,0)，必须安全返回而不是崩。"""
        self.assertIsNone(
            display_to_image_pixel((10, 10), (900, 700), (0, 0), (0, 0))
        )


class PickObjectAtPixelTest(unittest.TestCase):
    def setUp(self):
        self.objects = [
            {"name": "cola", "id": 1, "xyxy": (100, 50, 200, 400)},   # 高瓶子
            {"name": "apple", "id": 2, "xyxy": (120, 300, 190, 380)}, # 压在瓶子前
            {"name": "can", "id": 3, "xyxy": (400, 200, 470, 340)},
        ]

    def test_a_click_outside_every_box_returns_none(self):
        self.assertIsNone(pick_object_at_pixel(self.objects, 10, 10))

    def test_a_click_inside_one_box_selects_it(self):
        self.assertEqual(pick_object_at_pixel(self.objects, 430, 270), 2)

    def test_overlapping_boxes_pick_the_smaller_one(self):
        """遮挡场景：点在水果上应选水果，尽管瓶子的框也包含这个点。"""
        self.assertEqual(pick_object_at_pixel(self.objects, 150, 340), 1)

    def test_the_bottle_top_above_the_occluder_selects_the_bottle(self):
        """第二档的关键操作：水果挡住瓶子下部，点瓶子露出的上半部分要选到瓶子。"""
        self.assertEqual(pick_object_at_pixel(self.objects, 150, 100), 0)

    def test_box_edges_are_inclusive(self):
        self.assertEqual(pick_object_at_pixel(self.objects, 400, 200), 2)
        self.assertEqual(pick_object_at_pixel(self.objects, 470, 340), 2)

    def test_empty_and_malformed_input_does_not_raise(self):
        self.assertIsNone(pick_object_at_pixel([], 10, 10))
        self.assertIsNone(pick_object_at_pixel(None, 10, 10))
        self.assertIsNone(pick_object_at_pixel([{"name": "x"}], 10, 10))


class SortObjectsForPickingTest(unittest.TestCase):
    def test_highest_confidence_comes_first_and_original_indices_are_kept(self):
        objects = [
            {"name": "a", "conf": 0.61},
            {"name": "b", "conf": 0.97},
            {"name": "c", "conf": 0.80},
        ]
        ordered, original = sort_objects_for_picking(objects)
        self.assertEqual([item["name"] for item in ordered], ["b", "c", "a"])
        # 下拉框不重排，只用 original[0] 定位"置信度最高的那一项"，
        # 这样点图选中时的下标映射依然是直的。
        self.assertEqual(original, [1, 2, 0])
        self.assertIs(objects[original[0]], ordered[0])

    def test_missing_confidence_is_treated_as_zero(self):
        ordered, _original = sort_objects_for_picking(
            [{"name": "a"}, {"name": "b", "conf": 0.5}]
        )
        self.assertEqual(ordered[0]["name"], "b")

    def test_empty_input(self):
        self.assertEqual(sort_objects_for_picking([]), ([], []))
        self.assertEqual(sort_objects_for_picking(None), ([], []))


if __name__ == "__main__":
    unittest.main()
