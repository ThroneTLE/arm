#!/usr/bin/env python3
"""Regression tests for the operator-controlled target selection contract."""

import unittest

from tool.visual_grasp_pipeline.oak_vision_node import find_sequence_target
from tool.visual_grasp_pipeline.tracking import parse_sequence


class FrozenSelectionContractTests(unittest.TestCase):
    def setUp(self):
        self.objects = [
            {"name": "lemon", "id": 4, "seq": 1, "conf": 0.99},
            {"name": "lemon", "id": 9, "seq": 2, "conf": 0.70},
            {"name": "orange", "id": 2, "seq": 1, "conf": 1.00},
        ]

    def test_sequence_preserves_operator_name_and_instance(self):
        self.assertEqual(parse_sequence("lemon#9"), [("lemon", 9)])
        self.assertIs(
            find_sequence_target(self.objects, "lemon", 9), self.objects[1]
        )

    def test_missing_instance_never_falls_back_to_higher_confidence_peer(self):
        """要哪个就抓哪个，找不到就报找不到 —— 绝不静默换一个。

        2026-08-22 现场验证期间这条被临时跳过("负责人放行, 验证后恢复")，一直没恢复。
        任务书要求"每组识别抓取一个**不同的**易拉罐"，静默回退到另一个实例就是直接
        判错，所以在验收前恢复。当前 ``find_sequence_target`` 本来就没有回退行为，
        这条断言只是把它钉住。
        """
        self.assertIsNone(find_sequence_target(self.objects, "lemon", 7))

    def test_missing_class_never_falls_back_to_any_visible_object(self):
        self.assertIsNone(find_sequence_target(self.objects, "apple", 4))

    def test_no_instance_keeps_existing_first_match_behavior(self):
        self.assertIs(
            find_sequence_target(self.objects, "lemon", None), self.objects[0]
        )


if __name__ == "__main__":
    unittest.main()
