#!/usr/bin/env python3

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from arm_vision_framework.adapters.yolo import YoloSegmenter
from arm_vision_framework.types import FrameData


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return FakeTensor(self.value[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBox:
    def __init__(self, xyxy, class_id, confidence):
        self.xyxy = FakeTensor([xyxy])
        self.cls = np.asarray([class_id])
        self.conf = np.asarray([confidence])


class FakeModel:
    def __init__(self, result):
        self.result = result

    def predict(self, **_kwargs):
        return [self.result]


def frame():
    return FrameData(
        color_bgr=np.zeros((20, 30, 3), dtype=np.uint8),
        camera_matrix=np.eye(3), distortion=np.zeros(5),
        timestamp_s=1.0, frame_id="camera",
    )


class YoloAdapterTest(unittest.TestCase):
    def _segmenter(self, result, bbox_mask_fallback=True):
        module = types.ModuleType("ultralytics")
        module.YOLO = lambda _weights: FakeModel(result)
        context = patch.dict(sys.modules, {"ultralytics": module})
        context.start()
        self.addCleanup(context.stop)
        return YoloSegmenter(
            "/tmp/fake.pt", confidence_threshold=0.5,
            bbox_mask_fallback=bbox_mask_fallback,
        )

    def test_multiple_bbox_and_classes_are_preserved(self):
        result = types.SimpleNamespace(
            boxes=[
                FakeBox((1, 2, 8, 10), 0, 0.7),
                FakeBox((10, 4, 25, 18), 1, 0.9),
            ],
            masks=None,
            names={0: "can", 1: "banana"},
        )
        segmented = self._segmenter(result).segment(frame())
        self.assertTrue(segmented.valid, segmented.reason)
        self.assertEqual(len(segmented.detections), 2)
        self.assertEqual(
            [item.class_name for item in segmented.detections],
            ["banana", "can"],
        )
        self.assertEqual(segmented.bbox_xyxy, (10, 4, 25, 18))

    def test_bbox_fallback_produces_rectangular_mask(self):
        result = types.SimpleNamespace(
            boxes=[FakeBox((3, 5, 9, 12), 0, 0.8)], masks=None,
            names={0: "can"},
        )
        segmented = self._segmenter(result).segment(frame())
        self.assertEqual(int(segmented.mask.sum()), 6 * 7)
        self.assertTrue(np.all(segmented.mask[5:12, 3:9] == 1))

    def test_missing_mask_is_rejected_when_fallback_disabled(self):
        result = types.SimpleNamespace(
            boxes=[FakeBox((3, 5, 9, 12), 0, 0.8)], masks=None,
            names={0: "can"},
        )
        segmented = self._segmenter(
            result, bbox_mask_fallback=False
        ).segment(frame())
        self.assertFalse(segmented.valid)
        self.assertEqual(len(segmented.detections), 1)


if __name__ == "__main__":
    unittest.main()
