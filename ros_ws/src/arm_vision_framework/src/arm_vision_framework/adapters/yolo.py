"""Ultralytics YOLO segmentation adapter with lazy dependency loading."""

import cv2
import numpy as np

from ..errors import BackendUnavailable
from ..interfaces import Segmenter
from ..types import SegmentationResult


class YoloSegmenter(Segmenter):
    def __init__(self, weights, target_classes=None, confidence_threshold=0.5):
        if not weights:
            raise BackendUnavailable("YOLO weights path is empty")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise BackendUnavailable(
                "ultralytics is not installed in the active Python environment"
            ) from error
        self.model = YOLO(weights)
        self.target_classes = set(str(name) for name in (target_classes or []))
        self.confidence_threshold = float(confidence_threshold)

    def segment(self, frame):
        results = self.model.predict(
            source=frame.color_bgr,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not results:
            return SegmentationResult(False, reason="YOLO returned no result")
        result = results[0]
        if result.boxes is None or result.masks is None or len(result.boxes) == 0:
            return SegmentationResult(False, reason="YOLO found no segmentation mask")
        names = result.names
        candidates = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            class_name = str(names.get(class_id, class_id))
            confidence = float(box.conf.item())
            if self.target_classes and class_name not in self.target_classes:
                continue
            candidates.append((confidence, index, class_id, class_name, box))
        if not candidates:
            return SegmentationResult(False, reason="YOLO target class is absent")
        confidence, index, class_id, class_name, box = max(candidates)
        raw_mask = result.masks.data[index].detach().cpu().numpy()
        height, width = frame.color_bgr.shape[:2]
        mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
        x1, y1, x2, y2 = np.rint(box.xyxy[0].detach().cpu().numpy()).astype(int)
        return SegmentationResult(
            True,
            mask=mask.astype(np.uint8),
            bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            reason="YOLO segmentation accepted",
        )
