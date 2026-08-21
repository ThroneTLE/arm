#!/usr/bin/env python3
"""Dependency and data readiness checks shown by the model-builder UI."""

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_checks(
    foundationpose_root: str,
    calibration_path: str,
    yolo_weights: Optional[str] = None,
) -> List[CheckResult]:
    root = Path(foundationpose_root).expanduser().resolve()
    calibration = Path(calibration_path).expanduser().resolve()
    results = []
    for module, label, required in (
        ("cv2", "OpenCV", True),
        ("open3d", "Open3D", True),
        ("trimesh", "trimesh", True),
        ("ultralytics", "Ultralytics", True),
        ("torch", "PyTorch", True),
        ("pytorch3d", "PyTorch3D", True),
        ("nvdiffrast", "nvdiffrast", True),
        ("kaolin", "Kaolin (无模型 Neural Object Field)", False),
        ("PyQt5", "PyQt5 desktop UI", True),
        ("rospy", "ROS Python", True),
        ("depthai", "DepthAI (OAK-D Pro)", False),
    ):
        ok = importlib.util.find_spec(module) is not None
        results.append(CheckResult(label, ok, "available" if ok else "missing", required))
    source_ok = (root / "src" / "obj_pose_track.py").is_file()
    results.append(CheckResult("FoundationPose++ source", source_ok, str(root)))
    mycpp = list((root / "FoundationPose" / "mycpp" / "build").glob("mycpp*.so"))
    results.append(
        CheckResult("FoundationPose mycpp", bool(mycpp), str(mycpp[0]) if mycpp else "missing")
    )
    mycuda_dir = root / "FoundationPose" / "bundlesdf" / "mycuda"
    mycuda_common = list(mycuda_dir.glob("common*.so"))
    mycuda_grid = list(mycuda_dir.glob("gridencoder*.so"))
    mycuda_ok = bool(mycuda_common and mycuda_grid)
    results.append(
        CheckResult(
            "FoundationPose mycuda (Model-free)",
            mycuda_ok,
            "common={}, gridencoder={}".format(
                "available" if mycuda_common else "missing",
                "available" if mycuda_grid else "missing",
            ),
            required=False,
        )
    )
    refiner = root / "FoundationPose" / "weights" / "2023-10-28-18-33-37" / "model_best.pth"
    scorer = root / "FoundationPose" / "weights" / "2024-01-11-20-02-45" / "model_best.pth"
    results.append(CheckResult("FoundationPose refiner weights", refiner.is_file(), str(refiner)))
    results.append(CheckResult("FoundationPose scorer weights", scorer.is_file(), str(scorer)))
    results.append(CheckResult("runtime calibration YAML", calibration.is_file(), str(calibration)))
    if yolo_weights:
        path = Path(yolo_weights).expanduser().resolve()
        results.append(CheckResult("YOLO segmentation weights", path.is_file(), str(path)))
    else:
        results.append(CheckResult("YOLO segmentation weights", False, "not selected"))
    free_gb = shutil.disk_usage(root).free / (1024.0 ** 3)
    results.append(
        CheckResult(
            "free disk space",
            free_gb >= 2.0,
            "{:.1f} GB free; 2 GB minimum for capture and mesh output".format(free_gb),
        )
    )
    return results


def all_required_ready(results: List[CheckResult]) -> bool:
    return all(result.ok for result in results if result.required)
