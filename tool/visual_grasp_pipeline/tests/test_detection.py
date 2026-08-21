#!/usr/bin/env python3

import unittest

from tool.visual_grasp_pipeline.detection import select_target


class DetectionTests(unittest.TestCase):
    def test_select_target_by_name(self):
        objects = [
            {"name": "can", "cls": 5, "xyxy": (0, 0, 10, 10), "conf": 0.9},
            {"name": "apple", "cls": 2, "xyxy": (20, 20, 30, 30), "conf": 0.8},
        ]
        self.assertIs(select_target(objects, "apple"), objects[1])

    def test_select_target_by_class_id(self):
        objects = [
            {"name": "can", "cls": 5, "xyxy": (0, 0, 10, 10), "conf": 0.9},
            {"name": "apple", "cls": 2, "xyxy": (20, 20, 30, 30), "conf": 0.8},
        ]
        self.assertIs(select_target(objects, 5), objects[0])

    def test_select_largest_when_no_label(self):
        objects = [
            {"name": "small", "cls": 0, "xyxy": (0, 0, 10, 10), "conf": 0.9},
            {"name": "large", "cls": 1, "xyxy": (0, 0, 100, 100), "conf": 0.8},
        ]
        self.assertIs(select_target(objects, None), objects[1])

    def test_select_missing_label_returns_none(self):
        objects = [{"name": "can", "cls": 5, "xyxy": (0, 0, 10, 10), "conf": 0.9}]
        self.assertIsNone(select_target(objects, "banana"))


if __name__ == "__main__":
    unittest.main()
