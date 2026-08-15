#!/usr/bin/env python3
"""Validate parameters and run a no-hardware pipeline smoke test."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arm_vision_framework.factory import build_pipeline
from arm_vision_framework.parameters import CalibrationStore, load_system_parameters
from arm_vision_framework.transforms import xyz_rpy_from_transform
from arm_vision_framework.types import FrameData


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PACKAGE_ROOT / "config" / "system_parameters.yaml"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PACKAGE_ROOT / "config" / "calibration_parameters.yaml",
    )
    parser.add_argument("--check", action="store_true", help="validate configuration and dependencies")
    parser.add_argument("--smoke", action="store_true", help="run the deterministic mock pipeline")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    return parser.parse_args()


def dependency_state(name):
    return importlib.util.find_spec(name) is not None


def check_report(settings, calibration):
    return {
        "schema_valid": True,
        "camera": calibration.data["camera"]["name"],
        "image_size": list(calibration.image_size),
        "depth_aligned_to_color": calibration.depth_aligned_to_color,
        "workspace_from_base_valid": calibration.transform_valid("workspace_from_base"),
        "gripper_from_camera_valid": calibration.transform_valid("gripper_from_camera"),
        "segmentation_backend": settings["segmentation"]["backend"],
        "pose_backend": settings["pose_estimation"]["backend"],
        "robot_adapter": settings["robot"]["adapter"],
        "dry_run": bool(settings["safety"]["dry_run"]),
        "allow_robot_motion": bool(settings["safety"]["allow_robot_motion"]),
        "dependencies": {
            "rospy": dependency_state("rospy"),
            "cv2": dependency_state("cv2"),
            "ultralytics": dependency_state("ultralytics"),
            "torch": dependency_state("torch"),
            "depthai": dependency_state("depthai"),
        },
    }


def smoke_report(settings, calibration):
    smoke_settings = dict(settings)
    smoke_settings["segmentation"] = dict(settings["segmentation"], backend="mock")
    smoke_settings["pose_estimation"] = dict(settings["pose_estimation"], backend="mock")
    smoke_settings["robot"] = dict(settings["robot"], adapter="mock")
    pipeline = build_pipeline(smoke_settings, calibration)
    width, height = calibration.image_size
    frame = FrameData(
        color_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        camera_matrix=calibration.camera_matrix,
        distortion=calibration.distortion,
        timestamp_s=time.monotonic(),
        frame_id=calibration.data["frames"]["camera_color"],
    )
    result = pipeline.process(frame)
    xyz_m, rpy_deg = xyz_rpy_from_transform(result.workspace_from_object)
    return {
        "valid": result.valid,
        "simulated": result.simulated,
        "localization_source": result.camera_localization.source,
        "workspace_object_xyz_mm": (xyz_m * 1000.0).round(3).tolist(),
        "workspace_object_rpy_deg": rpy_deg.round(3).tolist(),
        "reason": result.reason,
    }


def main():
    args = parse_args()
    settings = load_system_parameters(args.config)
    calibration = CalibrationStore(args.calibration)
    run_check = args.check or not args.smoke
    output = {}
    if run_check:
        output["check"] = check_report(settings, calibration)
    if args.smoke:
        output["smoke"] = smoke_report(settings, calibration)
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        for section, values in output.items():
            print("[{}]".format(section))
            for key, value in values.items():
                print("{}: {}".format(key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
