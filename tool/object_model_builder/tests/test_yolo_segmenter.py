import unittest
from types import SimpleNamespace

import numpy as np

from tool.object_model_builder.yolo_segmenter import (
    MaskResult,
    YoloMaskProvider,
    deduplicate_mask_results,
    mask_overlap,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def item(self):
        return self.value.item()

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, index):
        return FakeTensor(self.value[index])


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.options = None

    def predict(self, **options):
        self.options = options
        return [self.result]


class YoloMaskProviderTests(unittest.TestCase):
    def provider(self):
        boxes = [
            SimpleNamespace(
                cls=FakeTensor(0), conf=FakeTensor(0.7),
                xyxy=FakeTensor([[1, 1, 5, 5]]),
            ),
            SimpleNamespace(
                cls=FakeTensor(1), conf=FakeTensor(0.9),
                xyxy=FakeTensor([[4, 4, 9, 9]]),
            ),
        ]
        masks = SimpleNamespace(
            data=[
                FakeTensor(np.ones((4, 4), dtype=np.float32)),
                FakeTensor(np.eye(4, dtype=np.float32)),
            ]
        )
        result = SimpleNamespace(
            boxes=boxes, masks=masks, names={0: "apple", 1: "banana"}
        )
        provider = object.__new__(YoloMaskProvider)
        provider.model = FakeModel(result)
        provider.target_classes = set()
        provider.confidence_threshold = 0.25
        provider.device = "cpu"
        provider.iou_threshold = 0.45
        provider.image_size = 640
        provider.agnostic_nms = True
        provider.deduplicate_instances = False
        provider.duplicate_mask_iou_threshold = 0.5
        provider.duplicate_mask_containment_threshold = 0.8
        provider.duplicate_center_distance_ratio = 0.35
        provider.duplicate_confidence_tie_margin = 0.05
        provider.maximum_detections = 50
        provider.last_reason = ""
        provider.last_model_instance_count = 0
        provider.last_suppressed_instance_count = 0
        return provider

    def test_predict_all_preserves_all_instances_and_predict_keeps_best(self):
        provider = self.provider()
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        instances = provider.predict_all(image)
        self.assertEqual([item.class_name for item in instances], ["banana", "apple"])
        self.assertEqual(instances[0].mask.shape, (10, 10))
        self.assertEqual(provider.model.options["iou"], 0.45)
        self.assertEqual(provider.model.options["imgsz"], 640)
        self.assertTrue(provider.model.options["agnostic_nms"])
        self.assertEqual(provider.model.options["max_det"], 50)
        best = provider.predict(image)
        self.assertEqual(best.class_name, "banana")

    def test_mask_overlap_reports_iou_containment_and_center_distance(self):
        full = np.zeros((20, 20), dtype=np.uint8)
        full[2:18, 4:16] = 1
        partial = np.zeros_like(full)
        partial[3:10, 6:14] = 1
        metrics = mask_overlap(full, partial)
        self.assertAlmostEqual(metrics.containment, 1.0)
        self.assertGreater(metrics.iou, 0.25)
        self.assertLess(metrics.center_distance_ratio, 0.35)

    def test_cross_class_mask_duplicates_keep_more_complete_near_tie(self):
        full = np.zeros((20, 20), dtype=np.uint8)
        full[2:18, 4:16] = 1
        partial = np.zeros_like(full)
        partial[3:10, 6:14] = 1
        results = [
            MaskResult(True, mask=partial, class_name="yellow_can", confidence=0.83),
            MaskResult(True, mask=full, class_name="green_apple", confidence=0.80),
        ]
        deduplicated = deduplicate_mask_results(results)
        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(deduplicated[0].class_name, "green_apple")

    def test_cross_class_mask_duplicates_keep_clearly_higher_confidence(self):
        full = np.zeros((20, 20), dtype=np.uint8)
        full[2:18, 4:16] = 1
        partial = np.zeros_like(full)
        partial[3:10, 6:14] = 1
        results = [
            MaskResult(True, mask=partial, class_name="yellow_can", confidence=0.93),
            MaskResult(True, mask=full, class_name="green_apple", confidence=0.72),
        ]
        deduplicated = deduplicate_mask_results(results)
        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(deduplicated[0].class_name, "yellow_can")

    def test_distinct_nearby_masks_are_not_merged_by_center_distance_alone(self):
        left = np.zeros((20, 20), dtype=np.uint8)
        right = np.zeros_like(left)
        left[5:15, 2:8] = 1
        right[5:15, 9:15] = 1
        results = [
            MaskResult(True, mask=left, class_name="can", confidence=0.9),
            MaskResult(True, mask=right, class_name="can", confidence=0.8),
        ]
        self.assertEqual(len(deduplicate_mask_results(results)), 2)


if __name__ == "__main__":
    unittest.main()
