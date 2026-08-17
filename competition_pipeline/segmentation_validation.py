"""Segmentation-model validation helpers shared by the competition UI."""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from tool.object_model_builder.yolo_segmenter import MaskResult, YoloMaskProvider


@dataclass
class SegmentationQuality:
    valid: bool
    mask_area_ratio: float = 0.0
    confidence: float = 0.0
    class_name: str = ""
    reason: str = ""


def evaluate_mask_result(
    result: MaskResult,
    image_shape,
    minimum_confidence: float = 0.5,
    minimum_mask_area_ratio: float = 0.005,
    maximum_mask_area_ratio: float = 0.8,
) -> SegmentationQuality:
    """Apply inexpensive, model-independent gates to one instance mask."""
    if not result.valid or result.mask is None:
        return SegmentationQuality(
            False,
            confidence=float(result.confidence),
            class_name=str(result.class_name),
            reason=result.reason or "模型没有返回有效实例 Mask",
        )
    mask = np.asarray(result.mask).astype(bool)
    expected = tuple(int(value) for value in image_shape[:2])
    if mask.shape != expected:
        return SegmentationQuality(
            False,
            confidence=float(result.confidence),
            class_name=str(result.class_name),
            reason="Mask 尺寸 {} 与 RGB {} 不一致".format(mask.shape, expected),
        )
    confidence = float(result.confidence)
    area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if confidence < float(minimum_confidence):
        reason = "置信度 {:.1f}% 低于 {:.1f}%".format(
            confidence * 100.0, float(minimum_confidence) * 100.0
        )
        return SegmentationQuality(False, area_ratio, confidence, result.class_name, reason)
    if area_ratio < float(minimum_mask_area_ratio):
        return SegmentationQuality(
            False, area_ratio, confidence, result.class_name,
            "Mask 面积 {:.2f}% 小于 {:.2f}%".format(
                area_ratio * 100.0, float(minimum_mask_area_ratio) * 100.0
            ),
        )
    if area_ratio > float(maximum_mask_area_ratio):
        return SegmentationQuality(
            False, area_ratio, confidence, result.class_name,
            "Mask 面积 {:.2f}% 大于 {:.2f}%，可能分割到背景".format(
                area_ratio * 100.0, float(maximum_mask_area_ratio) * 100.0
            ),
        )
    return SegmentationQuality(
        True, area_ratio, confidence, str(result.class_name),
        "Mask 质量门通过",
    )


def file_sha256(path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("分割权重不存在：{}".format(source))
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SegmentationModel:
    """Small adapter kept separate so the UI never imports Ultralytics directly."""

    def __init__(
        self,
        weights,
        target_classes: Optional[Sequence[str]] = None,
        confidence_threshold: float = 0.5,
        device: str = "0",
        iou_threshold: float = 0.45,
        image_size: int = 640,
        agnostic_nms: bool = True,
        deduplicate_instances: bool = True,
        duplicate_mask_iou_threshold: float = 0.50,
        duplicate_mask_containment_threshold: float = 0.80,
        duplicate_center_distance_ratio: float = 0.35,
        duplicate_confidence_tie_margin: float = 0.05,
        maximum_detections: int = 50,
    ):
        requested_device = str(device).strip().lower()
        if requested_device == "auto":
            try:
                import torch
                requested_device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                requested_device = "cpu"
        self.provider = YoloMaskProvider(
            weights,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            device=requested_device,
            iou_threshold=iou_threshold,
            image_size=image_size,
            agnostic_nms=agnostic_nms,
            deduplicate_instances=deduplicate_instances,
            duplicate_mask_iou_threshold=duplicate_mask_iou_threshold,
            duplicate_mask_containment_threshold=duplicate_mask_containment_threshold,
            duplicate_center_distance_ratio=duplicate_center_distance_ratio,
            duplicate_confidence_tie_margin=duplicate_confidence_tie_margin,
            maximum_detections=maximum_detections,
        )
        self.weights = str(self.provider.weights)
        self.device = requested_device

    def predict(self, color_bgr, gates):
        started = time.perf_counter()
        model_instances = self.provider.predict_all(color_bgr)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        evaluated = [
            (
                item,
                evaluate_mask_result(
                    item,
                    np.asarray(color_bgr).shape,
                    minimum_confidence=gates["minimum_confidence"],
                    minimum_mask_area_ratio=gates["minimum_mask_area_ratio"],
                    maximum_mask_area_ratio=gates["maximum_mask_area_ratio"],
                ),
            )
            for item in model_instances
        ]
        instances = [item for item, item_quality in evaluated if item_quality.valid]
        if instances:
            result = instances[0]
            quality = next(
                item_quality for item, item_quality in evaluated if item is result
            )
        else:
            reason = (
                evaluated[0][1].reason if evaluated else
                self.provider.last_reason or "YOLO found no instance mask"
            )
            result = MaskResult(False, reason=reason)
            quality = evaluate_mask_result(
                result,
                np.asarray(color_bgr).shape,
                minimum_confidence=gates["minimum_confidence"],
                minimum_mask_area_ratio=gates["minimum_mask_area_ratio"],
                maximum_mask_area_ratio=gates["maximum_mask_area_ratio"],
            )
        overlay = self.provider.overlay_many(color_bgr, instances)
        statistics = {
            "model_instances": int(self.provider.last_model_instance_count),
            "suppressed_duplicates": int(
                self.provider.last_suppressed_instance_count
            ),
            "quality_rejected": int(len(model_instances) - len(instances)),
        }
        return result, quality, overlay, elapsed_ms, instances, statistics
