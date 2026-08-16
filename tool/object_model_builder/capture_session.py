#!/usr/bin/env python3
"""On-disk capture session format for repeatable RGB-D reconstruction."""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import cv2
import numpy as np
import yaml

from .rgbd_geometry import CameraIntrinsics, _as_transform


SESSION_SCHEMA_VERSION = 1


@dataclass
class CapturedView:
    index: int
    color_bgr: np.ndarray
    depth_aligned_m: np.ndarray
    mask: np.ndarray
    workspace_from_color: np.ndarray
    timestamp_s: float
    metadata: dict
    depth_raw_m: Optional[np.ndarray] = None


class CaptureSession:
    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.yaml"

    @classmethod
    def create(
        cls,
        root: str,
        color_intrinsics: CameraIntrinsics,
        depth_intrinsics: CameraIntrinsics,
        color_from_depth: np.ndarray,
        calibration_source: str,
        tag_layout_source: str,
        yolo_weights: str,
        target_class: str,
    ) -> "CaptureSession":
        session = cls(root)
        session.root.mkdir(parents=True, exist_ok=True)
        for name in ("color", "depth_raw", "depth_aligned", "mask", "pose"):
            (session.root / name).mkdir(exist_ok=True)
        manifest = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "frame_count": 0,
            "units": "meters",
            "frames": {
                "world": "ruler_workspace",
                "camera": "camera_color_optical_frame",
            },
            "calibration": {
                "source": str(calibration_source),
                "color": color_intrinsics.to_mapping(),
                "depth": depth_intrinsics.to_mapping(),
                "color_from_depth": _as_transform(color_from_depth).tolist(),
            },
            "tag_layout_source": str(tag_layout_source),
            "segmentation": {
                "backend": "ultralytics_yolo",
                "weights": str(yolo_weights),
                "target_class": str(target_class),
            },
            "views": [],
        }
        provenance = {}
        for key, source, filename in (
            ("calibration_snapshot", calibration_source, "calibration.yaml"),
            ("tag_layout_snapshot", tag_layout_source, "tag_layout.yaml"),
        ):
            source_path = Path(source).expanduser()
            if source_path.is_file():
                provenance_dir = session.root / "provenance"
                provenance_dir.mkdir(exist_ok=True)
                destination = provenance_dir / filename
                shutil.copy2(source_path, destination)
                provenance[key] = destination.relative_to(session.root).as_posix()
        if provenance:
            manifest["provenance"] = provenance
        session._write_manifest(manifest)
        return session

    @classmethod
    def open(cls, root: str) -> "CaptureSession":
        session = cls(root)
        session.load_manifest()
        return session

    def load_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            raise FileNotFoundError("capture manifest does not exist: {}".format(self.manifest_path))
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        if int(manifest.get("schema_version", 0)) != SESSION_SCHEMA_VERSION:
            raise ValueError("unsupported capture session schema")
        return manifest

    def _write_manifest(self, manifest: dict) -> None:
        temporary = self.manifest_path.with_suffix(".yaml.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)
        temporary.replace(self.manifest_path)

    def add_view(
        self,
        color_bgr: np.ndarray,
        depth_aligned_m: np.ndarray,
        mask: np.ndarray,
        workspace_from_color: np.ndarray,
        timestamp_s: float,
        depth_raw_m: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        manifest = self.load_manifest()
        index = int(manifest.get("frame_count", 0))
        stem = "{:06d}".format(index)
        color = np.asarray(color_bgr)
        aligned = np.asarray(depth_aligned_m, dtype=np.float32)
        binary = np.asarray(mask).astype(bool)
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("color image must have shape HxWx3")
        if aligned.shape != color.shape[:2] or binary.shape != color.shape[:2]:
            raise ValueError("color, aligned depth, and mask dimensions must match")
        if not np.any(binary):
            raise ValueError("cannot capture an empty segmentation mask")
        if not np.any((aligned > 0.0) & binary):
            raise ValueError("segmentation mask contains no aligned depth")
        transform = _as_transform(workspace_from_color, "workspace_from_color")
        depth_mm = np.rint(np.clip(aligned, 0.0, 65.535) * 1000.0).astype(np.uint16)
        if not cv2.imwrite(str(self.root / "color" / (stem + ".png")), color):
            raise IOError("failed to write color image")
        if not cv2.imwrite(str(self.root / "depth_aligned" / (stem + ".png")), depth_mm):
            raise IOError("failed to write aligned depth image")
        if not cv2.imwrite(
            str(self.root / "mask" / (stem + ".png")), binary.astype(np.uint8) * 255
        ):
            raise IOError("failed to write segmentation mask")
        if depth_raw_m is not None:
            raw_mm = np.rint(
                np.clip(np.asarray(depth_raw_m, dtype=np.float32), 0.0, 65.535) * 1000.0
            ).astype(np.uint16)
            if not cv2.imwrite(str(self.root / "depth_raw" / (stem + ".png")), raw_mm):
                raise IOError("failed to write raw depth image")
        np.savetxt(self.root / "pose" / (stem + ".txt"), transform, fmt="%.12g")
        entry = {
            "index": index,
            "timestamp_s": float(timestamp_s),
            "color": "color/{}.png".format(stem),
            "depth_aligned": "depth_aligned/{}.png".format(stem),
            "mask": "mask/{}.png".format(stem),
            "workspace_from_color": "pose/{}.txt".format(stem),
            "metadata": dict(metadata or {}),
        }
        if depth_raw_m is not None:
            entry["depth_raw"] = "depth_raw/{}.png".format(stem)
        manifest.setdefault("views", []).append(entry)
        manifest["frame_count"] = index + 1
        self._write_manifest(manifest)
        return index

    def color_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics.from_mapping(self.load_manifest()["calibration"]["color"])

    def __len__(self) -> int:
        return int(self.load_manifest().get("frame_count", 0))

    def iter_views(self) -> Iterator[CapturedView]:
        manifest = self.load_manifest()
        for entry in manifest.get("views", []):
            color = cv2.imread(str(self.root / entry["color"]), cv2.IMREAD_COLOR)
            depth_mm = cv2.imread(
                str(self.root / entry["depth_aligned"]), cv2.IMREAD_UNCHANGED
            )
            mask = cv2.imread(str(self.root / entry["mask"]), cv2.IMREAD_GRAYSCALE)
            if color is None or depth_mm is None or mask is None:
                raise IOError("capture view {} is incomplete".format(entry.get("index")))
            raw = None
            if entry.get("depth_raw"):
                raw_mm = cv2.imread(str(self.root / entry["depth_raw"]), cv2.IMREAD_UNCHANGED)
                if raw_mm is None:
                    raise IOError("raw depth is missing for view {}".format(entry.get("index")))
                raw = raw_mm.astype(np.float32) * 0.001
            yield CapturedView(
                index=int(entry["index"]),
                color_bgr=color,
                depth_aligned_m=depth_mm.astype(np.float32) * 0.001,
                mask=mask > 127,
                workspace_from_color=np.loadtxt(
                    self.root / entry["workspace_from_color"], dtype=np.float64
                ).reshape(4, 4),
                timestamp_s=float(entry["timestamp_s"]),
                metadata=dict(entry.get("metadata", {})),
                depth_raw_m=raw,
            )
