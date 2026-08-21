#!/usr/bin/env python3

import unittest

from tool.visual_grasp_pipeline.tracking import StableTracker, add_seq, box_iou, parse_sequence


class TrackingTests(unittest.TestCase):
    def test_box_iou(self):
        a = (0, 0, 10, 10)
        b = (5, 0, 15, 10)
        self.assertAlmostEqual(box_iou(a, b), 50.0 / 150.0)

    def test_stable_tracker_keeps_id_and_assigns_new_id(self):
        tracker = StableTracker(max_miss=2)
        objs = [
            {"name": "can", "xyxy": (0, 0, 10, 10)},
            {"name": "apple", "xyxy": (20, 20, 30, 30)},
        ]
        tracker.update(objs)
        self.assertEqual([o["id"] for o in objs], [1, 2])

        next_objs = [
            {"name": "can", "xyxy": (1, 1, 11, 11)},
            {"name": "apple", "xyxy": (20, 20, 30, 30)},
        ]
        tracker.update(next_objs)
        self.assertEqual([o["id"] for o in next_objs], [1, 2])

    def test_stable_tracker_keeps_missing_id_until_max_miss(self):
        tracker = StableTracker(max_miss=1)
        first = [{"name": "can", "xyxy": (0, 0, 10, 10)}]
        tracker.update(first)
        second = [{"name": "apple", "xyxy": (20, 20, 30, 30)}]
        tracker.update(second)
        self.assertEqual(second[0]["id"], 2)
        # One miss keeps id 1, but a second miss removes it.
        third = [{"name": "apple", "xyxy": (20, 20, 30, 30)}]
        tracker.update(third)
        self.assertEqual(third[0]["id"], 2)

    def test_add_seq_numbers_per_class(self):
        objs = [{"name": "can", "xyxy": (0, 0, 1, 1)}, {"name": "can", "xyxy": (2, 2, 3, 3)}]
        add_seq(objs)
        self.assertEqual([o["seq"] for o in objs], [1, 2])

    def test_parse_sequence(self):
        self.assertEqual(
            parse_sequence("can#2, red_apple，apple"),
            [("can", 2), ("red_apple", None), ("apple", None)],
        )


if __name__ == "__main__":
    unittest.main()
