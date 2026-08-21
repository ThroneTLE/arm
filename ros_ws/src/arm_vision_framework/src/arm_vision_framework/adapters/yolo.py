"""Ultralytics YOLO segmentation adapter with lazy dependency loading."""

import cv2
import numpy as np

from ..errors import BackendUnavailable
from ..interfaces import Segmenter
from ..types import DetectionResult, SegmentationResult


class YoloSegmenter(Segmenter):
    def __init__(self, weights, target_classes=None, confidence_threshold=0.5,
                 bbox_mask_fallback=True):
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
        self.bbox_mask_fallback = bool(bbox_mask_fallback)

    def segment(self, frame):
        results = self.model.predict(
            source=frame.color_bgr,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not results:
            return SegmentationResult(False, reason="YOLO returned no result")
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return SegmentationResult(False, reason="YOLO found no object bounding box")
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
        height, width = frame.color_bgr.shape[:2]
        detections = []
        for confidence, index, class_id, class_name, box in sorted(
            candidates, key=lambda item: (-item[0], item[1])
        ):
            x1, y1, x2, y2 = np.rint(box.xyxy[0].detach().cpu().numpy()).astype(int)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            mask = None
            if result.masks is not None and index < len(result.masks.data):
                raw_mask = result.masks.data[index].detach().cpu().numpy()
                mask = cv2.resize(
                    raw_mask, (width, height), interpolation=cv2.INTER_NEAREST
                ) > 0.5
            elif self.bbox_mask_fallback:
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
            detections.append(DetectionResult(
                bbox_xyxy=(x1, y1, x2, y2), class_id=class_id,
                class_name=class_name, confidence=confidence,
                mask=None if mask is None else mask.astype(np.uint8),
            ))
        if not detections:
            return SegmentationResult(False, reason="YOLO bounding boxes were invalid")
        selected = detections[0]
        if selected.mask is None:
            return SegmentationResult(
                False, detections=tuple(detections),
                reason="YOLO detections have no mask and bbox fallback is disabled",
            )
        return SegmentationResult(
            True,
            mask=selected.mask,
            bbox_xyxy=selected.bbox_xyxy,
            class_id=selected.class_id,
            class_name=selected.class_name,
            confidence=selected.confidence,
            reason="YOLO detection/classification accepted: {} objects".format(len(detections)),
            detections=tuple(detections),
        )
