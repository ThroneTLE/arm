#!/usr/bin/env python3
"""Manage the framework's single calibration parameter file."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
ARM_ROOT = PACKAGE_ROOT.parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arm_vision_framework.parameters import CalibrationStore, COORDINATE_CONVENTION_ID
from arm_vision_framework.transforms import as_transform


DEFAULT_PARAMETER = PACKAGE_ROOT / "config" / "calibration_parameters.yaml"
DEFAULT_CAMERA_RUNTIME = (
    ARM_ROOT
    / "camera_calibration"
    / "calibration_snapshots"
    / "latest"
    / "runtime_calibration.yaml"
)


def read_yaml(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping: {}".format(path))
    return data


def timestamp_text():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def require_current_coordinate_convention(data, source_name):
    convention = data.get("coordinate_convention", {}) if isinstance(data, dict) else {}
    if convention.get("id") != COORDINATE_CONVENTION_ID:
        raise ValueError(
            "{} uses the legacy Tag-center convention; recalibrate with top-left origins".format(
                source_name
            )
        )
    return convention


def atomic_save(path, data, create_backup=True):
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if create_backup and destination.exists():
        backup_dir = destination.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = backup_dir / "{}_{}{}".format(
            destination.stem, stamp, destination.suffix
        )
        shutil.copy2(str(destination), str(backup))
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(destination.parent),
        prefix=".{}-".format(destination.name),
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    CalibrationStore(destination)
    return destination, backup


def matrix_from_hand_eye_file(data):
    if "camera_from_gripper" in data:
        raise ValueError(
            "input contains camera_from_gripper; provide an explicit gripper_from_camera matrix"
        )
    candidates = []
    direct = data.get("gripper_from_camera")
    if direct is not None:
        candidates.append(direct.get("matrix") if isinstance(direct, dict) else direct)
    nested = data.get("transforms", {}).get("gripper_from_camera")
    if nested is not None:
        candidates.append(nested.get("matrix") if isinstance(nested, dict) else nested)
    candidates = [candidate for candidate in candidates if candidate is not None]
    if len(candidates) != 1:
        raise ValueError("hand-eye YAML must contain exactly one gripper_from_camera matrix")
    return as_transform(candidates[0], "gripper_from_camera")


def sync_camera(parameter_path, runtime_path, tag_layout_path=None):
    parameters = read_yaml(parameter_path)
    runtime_path = Path(runtime_path).expanduser().resolve()
    runtime = read_yaml(runtime_path)
    intrinsics = runtime["rgb_intrinsics"]
    validity = runtime.get("validity", {})
    color = parameters["camera"]["color"]
    color.update(
        {
            "valid": True,
            "image_width": int(validity["image_width"]),
            "image_height": int(validity["image_height"]),
            "pixel_format": str(validity.get("pixel_format", "")),
            "fps": int(validity.get("fps", color.get("fps", 0))),
            "distortion_model": str(intrinsics.get("distortion_model", "plumb_bob")),
            "camera_matrix": np.asarray(
                intrinsics["camera_matrix"], dtype=np.float64
            ).reshape(3, 3).tolist(),
            "distortion_coefficients": np.asarray(
                intrinsics["distortion_coefficients"], dtype=np.float64
            ).reshape(-1).tolist(),
        }
    )
    parameters["camera"]["name"] = str(
        validity.get("camera_serial_name", parameters["camera"].get("name", "camera"))
    )
    runtime_convention = runtime.get("coordinate_convention", {})
    runtime_compatible = runtime_convention.get("id") == COORDINATE_CONVENTION_ID
    fixed_reference = parameters.setdefault("fixed_camera_validation_reference", {})
    fixed_reference.update(
        {
            "valid": runtime_compatible,
            "runtime_allowed_for_eye_in_hand": False,
            "coordinate_convention": (
                dict(runtime_convention) if runtime_compatible else "legacy_tag_center"
            ),
            "reason": (
                "Current top-left-origin fixed-camera validation reference."
                if runtime_compatible
                else "Preserved legacy reference; incompatible with the top-left-origin convention."
            ),
            "workspace_from_camera": as_transform(
                runtime["transforms"]["workspace_from_camera"],
                "workspace_from_camera",
            ).tolist(),
        }
    )
    parameters["quality"] = dict(runtime.get("quality", {}))
    parameters["quality"]["workspace_coordinate_convention_compatible"] = runtime_compatible
    if tag_layout_path is None:
        live_layout = ARM_ROOT / "camera_calibration" / "config" / "tag_layout.yaml"
        snapshot_layout = runtime_path.parent / "tag_layout.yaml"
        tag_layout_path = (
            live_layout
            if live_layout.is_file()
            else snapshot_layout if snapshot_layout.is_file() else None
        )
    if tag_layout_path is not None:
        layout = read_yaml(tag_layout_path)
        convention = require_current_coordinate_convention(layout, "tag layout")
        parameters["tag_map"].update(
            {
                "coordinate_convention": dict(convention),
                "dictionary": layout["dictionary"],
                "tag_size_mm": float(layout["tag_size_mm"]),
                "tags": {
                    int(tag_id): {
                        "origin_mm": [float(value) for value in entry["origin_mm"]],
                        "yaw_deg": float(entry.get("yaw_deg", 0.0)),
                    }
                    for tag_id, entry in layout["calibration_tags"].items()
                },
                "validation_tag": layout["validation_tag"],
            }
        )
    parameters["metadata"].update(
        {"updated_at": timestamp_text(), "source": str(runtime_path)}
    )
    return atomic_save(parameter_path, parameters)


def import_hand_eye(parameter_path, hand_eye_path):
    parameters = read_yaml(parameter_path)
    hand_eye_path = Path(hand_eye_path).expanduser().resolve()
    matrix = matrix_from_hand_eye_file(read_yaml(hand_eye_path))
    entry = parameters["transforms"]["gripper_from_camera"]
    entry.update(
        {
            "valid": True,
            "matrix": matrix.tolist(),
            "source": str(hand_eye_path),
            "updated_at": timestamp_text(),
        }
    )
    parameters["metadata"]["updated_at"] = timestamp_text()
    return atomic_save(parameter_path, parameters)


def set_transform(parameter_path, name, matrix_values, valid=True):
    parameters = read_yaml(parameter_path)
    if name not in parameters.get("transforms", {}):
        raise ValueError("unknown transform: {}".format(name))
    matrix = as_transform(np.asarray(matrix_values, dtype=np.float64), name)
    parameters["transforms"][name].update(
        {"valid": bool(valid), "matrix": matrix.tolist(), "updated_at": timestamp_text()}
    )
    parameters["metadata"]["updated_at"] = timestamp_text()
    return atomic_save(parameter_path, parameters)


def invalidate_transform(parameter_path, name):
    parameters = read_yaml(parameter_path)
    if name not in parameters.get("transforms", {}):
        raise ValueError("unknown transform: {}".format(name))
    parameters["transforms"][name]["valid"] = False
    parameters["transforms"][name]["updated_at"] = timestamp_text()
    parameters["metadata"]["updated_at"] = timestamp_text()
    return atomic_save(parameter_path, parameters)


def summary(parameter_path):
    store = CalibrationStore(parameter_path)
    return {
        "parameter": str(store.path),
        "profile": store.data["metadata"].get("profile"),
        "camera": store.data["camera"]["name"],
        "image_size": list(store.image_size),
        "depth_aligned_to_color": store.depth_aligned_to_color,
        "coordinate_convention": store.tag_map["coordinate_convention"]["id"],
        "tag_ids": sorted(int(tag_id) for tag_id in store.tag_map["tags"]),
        "transforms": {
            name: bool(entry.get("valid", False))
            for name, entry in store.data["transforms"].items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", type=Path, default=DEFAULT_PARAMETER)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show")
    subparsers.add_parser("validate")
    camera = subparsers.add_parser("sync-camera")
    camera.add_argument("--runtime", type=Path, default=DEFAULT_CAMERA_RUNTIME)
    camera.add_argument("--tag-layout", type=Path)
    hand_eye = subparsers.add_parser("import-hand-eye")
    hand_eye.add_argument("--input", type=Path, required=True)
    transform = subparsers.add_parser("set-transform")
    transform.add_argument(
        "--name", choices=("workspace_from_base", "gripper_from_camera", "color_from_depth"), required=True
    )
    transform.add_argument("--matrix", nargs=16, type=float, required=True)
    invalidate = subparsers.add_parser("invalidate")
    invalidate.add_argument(
        "--name", choices=("workspace_from_base", "gripper_from_camera", "color_from_depth"), required=True
    )
    subparsers.add_parser("camera-ui")
    return parser.parse_args()


def print_summary(values):
    for key, value in values.items():
        print("{}: {}".format(key, value))


def main():
    args = parse_args()
    if args.command in ("show", "validate"):
        print_summary(summary(args.parameter))
        print("validation: PASS")
        return 0
    if args.command == "sync-camera":
        destination, backup = sync_camera(args.parameter, args.runtime, args.tag_layout)
    elif args.command == "import-hand-eye":
        destination, backup = import_hand_eye(args.parameter, args.input)
    elif args.command == "set-transform":
        destination, backup = set_transform(args.parameter, args.name, args.matrix)
    elif args.command == "invalidate":
        destination, backup = invalidate_transform(args.parameter, args.name)
    elif args.command == "camera-ui":
        script = ARM_ROOT / "camera_calibration" / "run_ui.sh"
        return subprocess.call([str(script)])
    else:
        raise RuntimeError("unhandled command")
    print("updated: {}".format(destination))
    print("backup: {}".format(backup or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
