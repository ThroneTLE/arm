#!/usr/bin/env python3
"""Ultralytics instance-segmentation boundary used by the model builder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class MaskResult:
    valid: bool
    mask: Optional[np.ndarray] = None
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    class_id: Optional[int] = None
    class_name: str = ""
    confidence: float = 0.0
    reason: str = ""


class YoloMaskProvider:
    def __init__(
        self,
        weights: str,
        target_classes: Optional[Sequence[str]] = None,
        confidence_threshold: float = 0.5,
        device: str = "0",
    ):
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("YOLO weights do not exist: {}".format(path))
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("ultralytics is not installed in the active environment") from error
        self.weights = path
        self.model = YOLO(str(path))
        self.target_classes = {str(value) for value in (target_classes or []) if str(value)}
        self.confidence_threshold = float(confidence_threshold)
        self.device = str(device)

    def predict(self, color_bgr: np.ndarray) -> MaskResult:
        image = np.asarray(color_bgr)
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return MaskResult(False, reason="YOLO returned no result")
        result = results[0]
        if result.boxes is None or result.masks is None or len(result.boxes) == 0:
            return MaskResult(False, reason="YOLO found no instance mask")
        candidates = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            class_name = str(result.names.get(class_id, class_id))
            confidence = float(box.conf.item())
            if self.target_classes and class_name not in self.target_classes:
                continue
            candidates.append((confidence, index, class_id, class_name, box))
        if not candidates:
            return MaskResult(False, reason="configured YOLO target class is absent")
        confidence, index, class_id, class_name, box = max(candidates)
        raw_mask = result.masks.data[index].detach().cpu().numpy()
        height, width = image.shape[:2]
        mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
        x1, y1, x2, y2 = np.rint(box.xyxy[0].detach().cpu().numpy()).astype(int)
        return MaskResult(
            True,
            mask=mask,
            bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            reason="YOLO instance mask accepted",
        )

    @staticmethod
    def overlay(color_bgr: np.ndarray, result: MaskResult) -> np.ndarray:
        output = np.asarray(color_bgr).copy()
        if not result.valid or result.mask is None:
            return output
        tint = np.zeros_like(output)
        tint[result.mask.astype(bool)] = (30, 180, 255)
        output = cv2.addWeighted(output, 1.0, tint, 0.35, 0.0)
        if result.bbox_xyxy is not None:
            x1, y1, x2, y2 = result.bbox_xyxy
            cv2.rectangle(output, (x1, y1), (x2, y2), (30, 180, 255), 2)
            label = "{} {:.0f}%".format(result.class_name, result.confidence * 100.0)
            cv2.putText(
                output,
                label,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (30, 180, 255),
                2,
                cv2.LINE_AA,
            )
        return output
