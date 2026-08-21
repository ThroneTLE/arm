#!/usr/bin/env python3
"""Ultralytics instance-segmentation boundary used by the model builder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import time


@dataclass
class MaskResult:
    valid: bool
    mask: Optional[np.ndarray] = None
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    class_id: Optional[int] = None
    class_name: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class MaskOverlap:
    iou: float
    containment: float
    center_distance_ratio: float


def mask_overlap(first: np.ndarray, second: np.ndarray) -> MaskOverlap:
    """Measure overlap without assuming that both detections have one class.

    ``containment`` is intersection / smaller-mask area.  The center distance is
    normalized by the smaller mask's bounding-box diagonal so the same settings
    work at different camera distances.
    """
    first_mask = np.asarray(first).astype(bool)
    second_mask = np.asarray(second).astype(bool)
    if first_mask.shape != second_mask.shape:
        raise ValueError("instance masks must have identical dimensions")
    first_area = int(np.count_nonzero(first_mask))
    second_area = int(np.count_nonzero(second_mask))
    if first_area == 0 or second_area == 0:
        return MaskOverlap(0.0, 0.0, float("inf"))
    intersection = int(np.count_nonzero(first_mask & second_mask))
    union = first_area + second_area - intersection

    def center_and_diagonal(mask):
        rows, columns = np.nonzero(mask)
        center = np.asarray([np.mean(columns), np.mean(rows)], dtype=np.float64)
        width = float(np.max(columns) - np.min(columns) + 1)
        height = float(np.max(rows) - np.min(rows) + 1)
        return center, max(float(np.hypot(width, height)), 1.0)

    first_center, first_diagonal = center_and_diagonal(first_mask)
    second_center, second_diagonal = center_and_diagonal(second_mask)
    return MaskOverlap(
        iou=float(intersection / union) if union else 0.0,
        containment=float(intersection / min(first_area, second_area)),
        center_distance_ratio=float(
            np.linalg.norm(first_center - second_center)
            / min(first_diagonal, second_diagonal)
        ),
    )


def deduplicate_mask_results(
    results,
    mask_iou_threshold: float = 0.50,
    containment_threshold: float = 0.80,
    center_distance_ratio: float = 0.35,
    confidence_tie_margin: float = 0.05,
):
    """Merge spatially duplicate instances, including cross-class duplicates.

    Confidence wins unless two candidates are within ``confidence_tie_margin``;
    in that case the larger, more complete mask wins.  Distinct nearby objects
    are not merged from center distance alone: they must also have strong mask
    IoU, or one mask must be substantially contained by the other.
    """
    pending = [
        item for item in results
        if item.valid and item.mask is not None and np.any(item.mask)
    ]
    kept = []

    def prefer(candidate, current):
        confidence_delta = float(candidate.confidence) - float(current.confidence)
        if abs(confidence_delta) > float(confidence_tie_margin):
            return confidence_delta > 0.0
        candidate_area = int(np.count_nonzero(candidate.mask))
        current_area = int(np.count_nonzero(current.mask))
        if candidate_area != current_area:
            return candidate_area > current_area
        return confidence_delta > 0.0

    # High-confidence candidates are considered first. Replacement is still
    # allowed when a near-tied candidate has a more complete mask.
    pending.sort(key=lambda item: float(item.confidence), reverse=True)
    for candidate in pending:
        duplicate_indices = []
        for index, current in enumerate(kept):
            overlap = mask_overlap(candidate.mask, current.mask)
            duplicate = overlap.iou >= float(mask_iou_threshold) or (
                overlap.containment >= float(containment_threshold)
                and overlap.center_distance_ratio <= float(center_distance_ratio)
            )
            if duplicate:
                duplicate_indices.append(index)
        if not duplicate_indices:
            kept.append(candidate)
            continue
        winner = candidate
        for index in duplicate_indices:
            current = kept[index]
            if not prefer(winner, current):
                winner = current
        for index in reversed(duplicate_indices):
            kept.pop(index)
        kept.append(winner)
    return sorted(kept, key=lambda item: float(item.confidence), reverse=True)


class YoloMaskProvider:
    def __init__(
        self,
        weights: str,
        target_classes: Optional[Sequence[str]] = None,
        confidence_threshold: float = 0.5,
        device: str = "0",
        iou_threshold: Optional[float] = None,
        image_size: Optional[int] = None,
        agnostic_nms: bool = False,
        deduplicate_instances: bool = False,
        duplicate_mask_iou_threshold: float = 0.50,
        duplicate_mask_containment_threshold: float = 0.80,
        duplicate_center_distance_ratio: float = 0.35,
        duplicate_confidence_tie_margin: float = 0.05,
        maximum_detections: int = 100,
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
        if str(getattr(self.model, "task", "")).lower() != "segment":
            raise ValueError(
                "YOLO weights are not an instance-segmentation model: {}".format(path)
            )
        self.target_classes = {str(value) for value in (target_classes or []) if str(value)}
        self.confidence_threshold = float(confidence_threshold)
        if not 0.05 <= self.confidence_threshold <= 1.0:
            raise ValueError("YOLO confidence_threshold must be within 0.05..1")
        self.device = str(device)
        self.iou_threshold = None if iou_threshold is None else float(iou_threshold)
        self.image_size = None if image_size is None else int(image_size)
        self.agnostic_nms = bool(agnostic_nms)
        self.deduplicate_instances = bool(deduplicate_instances)
        self.duplicate_mask_iou_threshold = float(duplicate_mask_iou_threshold)
        self.duplicate_mask_containment_threshold = float(
            duplicate_mask_containment_threshold
        )
        self.duplicate_center_distance_ratio = float(duplicate_center_distance_ratio)
        self.duplicate_confidence_tie_margin = float(duplicate_confidence_tie_margin)
        if not 0.0 <= self.duplicate_confidence_tie_margin <= 0.25:
            raise ValueError(
                "duplicate_confidence_tie_margin must be within 0..0.25"
            )
        self.maximum_detections = int(maximum_detections)
        if not 1 <= self.maximum_detections <= 300:
            raise ValueError("maximum_detections must be within 1..300")
        self.last_reason = ""
        self.last_model_instance_count = 0
        self.last_suppressed_instance_count = 0
        self.last_inference_ms = 0.0

    def predict_all(self, color_bgr: np.ndarray):
        """Return every accepted instance, sorted by descending confidence."""
        image = np.asarray(color_bgr)
        self.last_model_instance_count = 0
        self.last_suppressed_instance_count = 0
        options = {
            "source": image,
            "conf": self.confidence_threshold,
            "device": self.device,
            "verbose": False,
            "agnostic_nms": self.agnostic_nms,
            "max_det": self.maximum_detections,
        }
        if self.iou_threshold is not None:
            options["iou"] = self.iou_threshold
        if self.image_size is not None:
            options["imgsz"] = self.image_size
        started = time.perf_counter()
        try:
            results = self.model.predict(**options)
        finally:
            self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        if not results:
            self.last_reason = "YOLO returned no result"
            return []
        result = results[0]
        if result.boxes is None or result.masks is None or len(result.boxes) == 0:
            self.last_reason = "YOLO found no instance mask"
            return []
        self.last_model_instance_count = int(len(result.boxes))
        candidates = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            class_name = str(result.names.get(class_id, class_id))
            confidence = float(box.conf.item())
            if self.target_classes and class_name not in self.target_classes:
                continue
            candidates.append((confidence, index, class_id, class_name, box))
        if not candidates:
            self.last_reason = "configured YOLO target class is absent"
            return []
        height, width = image.shape[:2]
        accepted = []
        for confidence, index, class_id, class_name, box in sorted(
            candidates, reverse=True
        ):
            raw_mask = result.masks.data[index].detach().cpu().numpy()
            mask = cv2.resize(
                raw_mask, (width, height), interpolation=cv2.INTER_NEAREST
            ) > 0.5
            x1, y1, x2, y2 = np.rint(
                box.xyxy[0].detach().cpu().numpy()
            ).astype(int)
            accepted.append(
                MaskResult(
                    True,
                    mask=mask,
                    bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    reason="YOLO instance mask accepted",
                )
            )
        if self.deduplicate_instances:
            deduplicated = deduplicate_mask_results(
                accepted,
                mask_iou_threshold=self.duplicate_mask_iou_threshold,
                containment_threshold=self.duplicate_mask_containment_threshold,
                center_distance_ratio=self.duplicate_center_distance_ratio,
                confidence_tie_margin=self.duplicate_confidence_tie_margin,
            )
        else:
            deduplicated = accepted
        self.last_suppressed_instance_count = len(accepted) - len(deduplicated)
        self.last_reason = "{} instance mask(s) accepted; {} duplicate(s) suppressed".format(
            len(deduplicated), self.last_suppressed_instance_count
        )
        return deduplicated

    def predict(self, color_bgr: np.ndarray) -> MaskResult:
        accepted = self.predict_all(color_bgr)
        if accepted:
            return accepted[0]
        return MaskResult(False, reason=self.last_reason or "YOLO found no instance mask")

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

    @classmethod
    def overlay_many(cls, color_bgr: np.ndarray, results) -> np.ndarray:
        output = np.asarray(color_bgr).copy()
        for result in reversed(list(results)):
            output = cls.overlay(output, result)
        return output
