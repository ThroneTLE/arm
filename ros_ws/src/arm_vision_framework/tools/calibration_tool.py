#!/usr/bin/env python3
"""Manage the framework's single calibration parameter file."""

import argparse
import copy
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

from arm_vision_framework.parameters import (
    CalibrationStore, COORDINATE_CONVENTION_ID, load_system_parameters,
)
from arm_vision_framework.transforms import as_transform
from arm_vision_framework.oak_calibration_import import (
    export_oak_device_eeprom, inspect_oak_eeprom,
)


DEFAULT_PARAMETER = PACKAGE_ROOT / "config" / "calibration_parameters.yaml"
DEFAULT_CAMERA_RUNTIME = (
    ARM_ROOT
    / "tool"
    / "camera_calibration"
    / "calibration_snapshots"
    / "latest"
    / "runtime_calibration.yaml"
)
DEFAULT_SYSTEM_PARAMETER = PACKAGE_ROOT / "config" / "system_parameters.yaml"


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


def atomic_save_system(path, data):
    """Atomically write runtime parameters without treating them as calibration.

    Controller endpoint/state maps and recovery points live in system
    parameters, so they cannot go through :class:`CalibrationStore`'s
    calibration-only validator.
    """
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        backup_dir = destination.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / "{}_{}{}".format(
            destination.stem, datetime.now().strftime("%Y%m%d_%H%M%S_%f"), destination.suffix
        )
        shutil.copy2(str(destination), str(backup))
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(destination.parent),
        prefix=".{}-".format(destination.name), suffix=".tmp", delete=False,
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
    load_system_parameters(destination)
    return destination, backup


def import_competition_controller(system_path, competition_path):
    """Copy the UI-tested controller map and safe MOVJ points into ROS config."""
    system = read_yaml(system_path)
    competition = read_yaml(competition_path)
    controller = competition.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("competition YAML has no controller mapping")
    recovery = competition.get("safety", {}).get("recovery", {})
    if not isinstance(recovery, dict):
        raise ValueError("competition safety.recovery must be a mapping")
    system["controller"] = copy.deepcopy(controller)
    system.setdefault("safety", {})["recovery"] = copy.deepcopy(recovery)
    return atomic_save_system(system_path, system)


def matrix_from_hand_eye_file(data):
    if "camera_from_gripper" in data:
        raise ValueError(
            "input contains camera_from_gripper; provide an explicit gripper_from_camera matrix"
        )
    def candidate_matrix(entry, label):
        if isinstance(entry, dict):
            if entry.get("valid") is False:
                raise ValueError("{} input is marked invalid".format(label))
            return entry.get("matrix")
        return entry

    candidates = []
    direct = data.get("gripper_from_camera")
    if direct is not None:
        candidates.append(candidate_matrix(direct, "gripper_from_camera"))
    nested = data.get("transforms", {}).get("gripper_from_camera")
    if nested is not None:
        candidates.append(candidate_matrix(nested, "transforms.gripper_from_camera"))
    # Native competition_pipeline output names the same rigid transform
    # T_tcp_color_camera.  In the formal ROS package, the production TCP is
    # represented by the gripper frame, so this is an explicit alias rather
    # than an inferred inverse.
    competition = data.get("hand_eye", {}).get("tcp_from_color_camera")
    if competition is not None:
        candidates.append(candidate_matrix(competition, "competition hand-eye"))
    direct_competition = data.get("tcp_from_color_camera")
    if direct_competition is not None:
        candidates.append(candidate_matrix(direct_competition, "TCP hand-eye"))
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
        live_layout = ARM_ROOT / "tool" / "camera_calibration" / "config" / "tag_layout.yaml"
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


def _copy_file_atomic(source, destination):
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise ValueError("source file does not exist: {}".format(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        return destination
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=str(destination.parent),
        prefix=".{}-".format(destination.name), suffix=".tmp", delete=False,
    )
    temporary = Path(handle.name)
    try:
        with source.open("rb") as source_handle, handle:
            shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def import_oak_eeprom(parameter_path, source_json, color_width=1920, color_height=1080,
                       depth_width=1280, depth_height=800, factory_output=None,
                       color_fps=10.0, device_mxid=None, device_usb_speed=None):
    """Import an official OAK EEPROM JSON into the ROS calibration file.

    This is intentionally an offline import.  It does not claim to flash an
    OAK device; flashing must be performed by the official DepthAI/Luxonis
    calibration tool when the camera is physically connected.
    """

    if float(color_fps) <= 0.0:
        raise ValueError("OAK RGB FPS must be positive")
    parameters = read_yaml(parameter_path)
    info = inspect_oak_eeprom(
        source_json,
        color_width=color_width,
        color_height=color_height,
        depth_width=depth_width,
        depth_height=depth_height,
    )
    source = Path(source_json).expanduser().resolve()
    factory_output = (
        Path(factory_output).expanduser().resolve()
        if factory_output is not None
        else Path(parameter_path).expanduser().resolve().parent / "oak_factory_calibration.json"
    )
    saved_factory = _copy_file_atomic(source, factory_output)

    camera = parameters.setdefault("camera", {})
    color = camera.setdefault("color", {})
    color.update(
        {
            "valid": True,
            "image_width": int(info["color"]["image_width"]),
            "image_height": int(info["color"]["image_height"]),
            "pixel_format": "BGR",
            "fps": float(color_fps),
            "distortion_model": info["color"]["distortion_model"],
            "camera_matrix": np.asarray(info["color"]["camera_matrix"], dtype=np.float64).tolist(),
            "distortion_coefficients": np.asarray(
                info["color"]["distortion_coefficients"], dtype=np.float64
            ).tolist(),
            "source": str(saved_factory),
        }
    )
    camera["name"] = info["product_name"] or info["device_name"] or "OAK-D Pro"
    depth = camera.setdefault("depth", {})
    depth.update(
        {
            "valid": True,
            # The published depth is aligned and resampled into the RGB pixel
            # grid, so its output geometry is the RGB geometry.  Preserve the
            # native CAM_C calibration separately below instead of attaching
            # mono intrinsics to aligned RGB pixels.
            "image_width": int(info["color"]["image_width"]),
            "image_height": int(info["color"]["image_height"]),
            "unit": "millimeters_uint16",
            "aligned_to_color": True,
            "note": "DepthAI depth is aligned and resampled into the RGB pixel grid.",
            "distortion_model": info["color"]["distortion_model"],
            "camera_matrix": np.asarray(info["color"]["camera_matrix"], dtype=np.float64).tolist(),
            "distortion_coefficients": np.asarray(
                info["color"]["distortion_coefficients"], dtype=np.float64
            ).tolist(),
            "intrinsics_valid": True,
            "source": str(saved_factory),
            "native_cam_c": {
                "image_width": int(info["depth"]["image_width"]),
                "image_height": int(info["depth"]["image_height"]),
                "distortion_model": info["depth"]["distortion_model"],
                "camera_matrix": np.asarray(
                    info["depth"]["camera_matrix"], dtype=np.float64
                ).tolist(),
                "distortion_coefficients": np.asarray(
                    info["depth"]["distortion_coefficients"], dtype=np.float64
                ).tolist(),
            },
        }
    )
    # The OAK runtime will publish depth aligned to RGB. Do not retain an
    # Astra transform under the new camera profile unless its direction has
    # been explicitly verified from the official SDK output.
    color_from_depth = parameters.setdefault("transforms", {}).setdefault(
        "color_from_depth", {}
    )
    color_from_depth.update(
        {
            "valid": False,
            "source": str(saved_factory),
            "reason": "OAK runtime alignment is used; verify an explicit color_from_depth transform before enabling it",
        }
    )
    frames = parameters.setdefault("frames", {})
    if frames.get("camera_color"):
        frames["camera_depth"] = frames["camera_color"]
    hand_eye = parameters.setdefault("transforms", {}).setdefault(
        "gripper_from_camera", {}
    )
    hand_eye.update(
        {
            "valid": False,
            "reason": "camera profile changed to OAK; redo eye-in-hand calibration",
            "updated_at": timestamp_text(),
        }
    )
    metadata = parameters.setdefault("metadata", {})
    metadata.update(
        {
            "profile": "oak_d_pro_competition_{}x{}".format(
                info["color"]["image_width"], info["color"]["image_height"]
            ),
            "updated_at": timestamp_text(),
            "source": str(saved_factory),
            "note": "OAK-D Pro EEPROM imported for the active competition RGB-D profile.",
            "camera_calibration_source": "official_depthai_eeprom_json",
            "camera_calibration_eeprom_version": int(info["eeprom_version"]),
            "camera_calibration_baseline_mm": info["baseline_mm"],
        }
    )
    if device_mxid:
        metadata["camera_mxid"] = str(device_mxid)
    else:
        metadata.pop("camera_mxid", None)
    if device_usb_speed:
        metadata["camera_usb_speed_at_import"] = str(device_usb_speed)
    else:
        metadata.pop("camera_usb_speed_at_import", None)
    # Every existing quality number was measured with the previous camera;
    # retaining it after a profile switch would make the report misleading.
    parameters["quality"] = {
        "camera_intrinsics_source": "official_depthai_eeprom_json",
    }
    fixed_reference = parameters.setdefault("fixed_camera_validation_reference", {})
    fixed_reference.update(
        {
            "valid": False,
            "runtime_allowed_for_eye_in_hand": False,
            "reason": "camera profile changed to OAK; redo eye-in-hand validation",
        }
    )
    return atomic_save(parameter_path, parameters)


def import_oak_device(parameter_path, mxid=None, color_width=1920, color_height=1080,
                      depth_width=1280, depth_height=800, factory_output=None,
                      color_fps=10.0):
    """Read one connected OAK device and import its EEPROM calibration."""

    with tempfile.TemporaryDirectory(prefix="arm-oak-eeprom-") as directory:
        exported = Path(directory) / "device_eeprom.json"
        device = export_oak_device_eeprom(exported, mxid=mxid)
        return import_oak_eeprom(
            parameter_path,
            exported,
            color_width=color_width,
            color_height=color_height,
            depth_width=depth_width,
            depth_height=depth_height,
            factory_output=factory_output,
            color_fps=color_fps,
            device_mxid=device["mxid"],
            device_usb_speed=device["usb_speed"],
        )


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
    hand_eye_competition = subparsers.add_parser(
        "import-competition-hand-eye",
        help="import competition_pipeline hand_eye.tcp_from_color_camera",
    )
    hand_eye_competition.add_argument("--input", type=Path, required=True)
    controller_competition = subparsers.add_parser(
        "import-competition-controller",
        help="import controller/state/recovery settings from competition_pipeline",
    )
    controller_competition.add_argument("--input", type=Path, required=True)
    controller_competition.add_argument(
        "--system", type=Path, default=DEFAULT_SYSTEM_PARAMETER
    )
    oak = subparsers.add_parser(
        "import-oak-eeprom",
        help="offline-import official DepthAI/Luxonis EEPROM JSON",
    )
    oak.add_argument("--input", type=Path, required=True)
    oak.add_argument("--color-width", type=int, default=1920)
    oak.add_argument("--color-height", type=int, default=1080)
    oak.add_argument("--depth-width", type=int, default=1280)
    oak.add_argument("--depth-height", type=int, default=800)
    oak.add_argument("--factory-output", type=Path)
    oak.add_argument("--fps", type=float, default=10.0)
    oak_device = subparsers.add_parser(
        "import-oak-device",
        help="read and import EEPROM calibration from a connected OAK device",
    )
    oak_device.add_argument("--mxid")
    oak_device.add_argument("--color-width", type=int, default=1920)
    oak_device.add_argument("--color-height", type=int, default=1080)
    oak_device.add_argument("--depth-width", type=int, default=1280)
    oak_device.add_argument("--depth-height", type=int, default=800)
    oak_device.add_argument("--factory-output", type=Path)
    oak_device.add_argument("--fps", type=float, default=10.0)
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
    elif args.command in ("import-hand-eye", "import-competition-hand-eye"):
        destination, backup = import_hand_eye(args.parameter, args.input)
    elif args.command == "import-competition-controller":
        destination, backup = import_competition_controller(args.system, args.input)
    elif args.command == "import-oak-eeprom":
        destination, backup = import_oak_eeprom(
            args.parameter,
            args.input,
            color_width=args.color_width,
            color_height=args.color_height,
            depth_width=args.depth_width,
            depth_height=args.depth_height,
            factory_output=args.factory_output,
            color_fps=args.fps,
        )
    elif args.command == "import-oak-device":
        destination, backup = import_oak_device(
            args.parameter,
            mxid=args.mxid,
            color_width=args.color_width,
            color_height=args.color_height,
            depth_width=args.depth_width,
            depth_height=args.depth_height,
            factory_output=args.factory_output,
            color_fps=args.fps,
        )
    elif args.command == "set-transform":
        destination, backup = set_transform(args.parameter, args.name, args.matrix)
    elif args.command == "invalidate":
        destination, backup = invalidate_transform(args.parameter, args.name)
    elif args.command == "camera-ui":
        script = ARM_ROOT / "tool" / "camera_calibration" / "run_ui.sh"
        return subprocess.call([str(script)])
    else:
        raise RuntimeError("unhandled command")
    print("updated: {}".format(destination))
    print("backup: {}".format(backup or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
