"""Configuration for the migrated visual grasping pipeline.

The original release hard-coded many machine-specific paths at module import
time.  This module keeps the same knobs in a small dataclass/YAML form so the
tool can be pointed at a different machine without editing Python code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


def _expand(path: str) -> str:
    return os.path.expanduser(str(path))


@dataclass
class GraspRule:
    type: str = "cylinder"
    offset_mm: float = 5.0
    yaw_align: str = "y+"


@dataclass
class VisualGraspConfig:
    yolo_weights: str = ""
    foundationpose_root: str = ""
    object_models: Dict[str, str] = field(default_factory=dict)
    object_model_scales: Dict[str, float] = field(default_factory=dict)
    yolo_to_object: Dict[str, str] = field(default_factory=dict)
    grasp_rules: Dict[str, GraspRule] = field(default_factory=dict)
    default_object: str = "sprite"
    target_label: Optional[str] = "red_apple"
    force_object: Optional[str] = None
    tag_size_mm: float = 80.0
    yolo_conf: float = 0.85
    yolo_imgsz: int = 640
    offset_xy_mm: Tuple[float, float] = (15.0, 32.0)
    center_offset_mm: float = 30.0
    flip_x: bool = True
    flip_y: bool = True
    est_refine_iter: int = 5
    track_refine_iter: int = 2
    use_track: bool = True
    smoothing_alpha: float = 0.15
    tag_smoothing_alpha: float = 0.3
    static_frame_dir: str = "~/fp_capture"
    debug_dir: str = "~/fp_debug"
    pose_file: str = "/tmp/can_pose.npy"
    grasp_file: str = "/tmp/grasp_pose.npy"
    mesh_scale_to_meters: float = 1.0
    device: str = "cuda:0"
    use_mask_center_guidance: bool = True
    foundationpose_roi_padding_pixels: int = 24
    foundationpose_max_input_size: int = 640
    foundationpose_registration_hypotheses: int = 64

    @classmethod
    def from_yaml(cls, path) -> "VisualGraspConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        paths = raw.get("paths", {})
        pipeline = raw.get("pipeline", {})
        objects = raw.get("object_models", {})
        model_scales = raw.get("object_model_scales", {})
        mapping = raw.get("yolo_to_object", {})
        rules_raw = raw.get("grasp_rules", {})
        rules = {}
        for name, rule in rules_raw.items():
            if isinstance(rule, dict):
                rules[name] = GraspRule(**rule)
            else:
                rules[name] = GraspRule()
        offset = pipeline.get("offset_xy_mm", [15.0, 32.0])
        if len(offset) != 2:
            offset = [15.0, 32.0]
        return cls(
            yolo_weights=_expand(paths.get("yolo_weights", "")),
            foundationpose_root=_expand(paths.get("foundationpose_root", "")),
            object_models={
                k: _expand(v) for k, v in objects.items() if v
            },
            object_model_scales={
                str(key): float(value) for key, value in model_scales.items()
            },
            yolo_to_object=dict(mapping),
            grasp_rules=rules,
            default_object=pipeline.get("default_object", "sprite"),
            target_label=pipeline.get("target_label"),
            force_object=pipeline.get("force_object"),
            tag_size_mm=float(pipeline.get("tag_size_mm", 80.0)),
            yolo_conf=float(pipeline.get("yolo_conf", 0.85)),
            yolo_imgsz=int(pipeline.get("yolo_imgsz", 640)),
            offset_xy_mm=(float(offset[0]), float(offset[1])),
            center_offset_mm=float(pipeline.get("center_offset_mm", 30.0)),
            flip_x=bool(pipeline.get("flip_x", True)),
            flip_y=bool(pipeline.get("flip_y", True)),
            est_refine_iter=int(pipeline.get("est_refine_iter", 5)),
            track_refine_iter=int(pipeline.get("track_refine_iter", 2)),
            use_track=bool(pipeline.get("use_track", True)),
            smoothing_alpha=float(pipeline.get("smoothing_alpha", 0.15)),
            tag_smoothing_alpha=float(pipeline.get("tag_smoothing_alpha", 0.3)),
            static_frame_dir=_expand(paths.get("static_frame_dir", "~/fp_capture")),
            debug_dir=_expand(paths.get("debug_dir", "~/fp_debug")),
            pose_file=_expand(pipeline.get("pose_file", "/tmp/can_pose.npy")),
            grasp_file=_expand(pipeline.get("grasp_file", "/tmp/grasp_pose.npy")),
            mesh_scale_to_meters=float(
                pipeline.get("mesh_scale_to_meters", 1.0)
            ),
            device=str(pipeline.get("device", "cuda:0")),
            use_mask_center_guidance=bool(
                pipeline.get("use_mask_center_guidance", True)
            ),
            foundationpose_roi_padding_pixels=max(
                0, int(pipeline.get("foundationpose_roi_padding_pixels", 24))
            ),
            foundationpose_max_input_size=max(
                160, int(pipeline.get("foundationpose_max_input_size", 640))
            ),
            foundationpose_registration_hypotheses=max(
                1,
                int(pipeline.get("foundationpose_registration_hypotheses", 64)),
            ),
        )

    def resolve_object_key(self, detected_name, class_id=None) -> str:
        if self.force_object:
            return self.force_object
        return self.yolo_to_object.get(
            str(detected_name),
            self.yolo_to_object.get(str(class_id), self.default_object),
        )

    def mesh_for_object(self, object_key: str) -> str:
        return self.object_models.get(object_key, "")

    def mesh_scale_for_object(self, object_key: str) -> float:
        return float(
            self.object_model_scales.get(object_key, self.mesh_scale_to_meters)
        )

    def rule_for_object(self, object_key: str) -> GraspRule:
        return self.grasp_rules.get(object_key, self.grasp_rules.get(self.default_object, GraspRule()))

    def ensure_paths(self):
        missing = []
        for name, path in [
            ("yolo_weights", self.yolo_weights),
            ("foundationpose_root", self.foundationpose_root),
            ("static_frame_dir", self.static_frame_dir),
        ]:
            if not path or not os.path.exists(path):
                missing.append(name)
        return missing
