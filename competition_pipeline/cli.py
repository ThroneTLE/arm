#!/usr/bin/env python3
"""Field CLI for RGB-D calibration, Tag mapping, hand-eye, and localization."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .configuration import CompetitionConfig, load_camera_intrinsics
from .checkerboard_target import CHECKERBOARD_TARGET
from .geometry import transform_from_xyz_rpy_mm, xyz_rpy_from_transform
from .hand_eye import APRILTAG_MAP_TARGET, HandEyeCalibrator
from .localization import HybridLocalizer
from .sample_store import HandEyeSampleStore
from .tag_map import TagMap
from tool.object_model_builder.camera_source import OakDProSource


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "competition.yaml"
DEFAULT_SAMPLES = None


def _configured_samples(config, override=None):
    return (
        Path(override).expanduser().resolve()
        if override
        else config.resolve_path(config.camera["hand_eye_samples_file"])
    )


def _append_sample(path, config, sample, image_source):
    return HandEyeSampleStore(path, config).append(sample, image_source)


def _configured_intrinsics(config, override=None):
    path = (
        Path(override).expanduser().resolve()
        if override
        else config.resolve_path(config.camera["color_intrinsics_file"])
    )
    return load_camera_intrinsics(path)


def _matrix_text(matrix):
    return yaml.safe_dump(np.asarray(matrix).tolist(), sort_keys=False).strip()


def _open_camera(device, width, height, fps):
    capture = cv2.VideoCapture(int(device) if str(device).isdigit() else str(device), cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError("cannot open camera {}".format(device))
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def _hand_eye_live(args, config):
    matrix, distortion, image_size = _configured_intrinsics(config, args.intrinsics)
    capture_width, capture_height = image_size
    camera = config.camera
    oak_source = None
    capture = None
    if camera.get("backend") == "oak_depthai":
        oak_source = OakDProSource(
            color_width=camera.get("color_width", 1920),
            color_height=camera.get("color_height", 1080),
            fps=args.fps or camera.get("color_fps", 10),
            mxid=camera.get("mxid", ""),
            dot_projector_mA=camera.get("dot_projector_mA", 800),
            floodlight_mA=camera.get("floodlight_mA", 0),
            mono_resolution=camera.get("mono_resolution", "800p"),
            extended_disparity=camera.get("extended_disparity", True),
            subpixel=camera.get("subpixel", False),
            left_right_check=camera.get("left_right_check", True),
            focus_mode=camera.get("focus_mode", "device_default"),
            manual_focus=camera.get("manual_focus"),
        )
        oak_source.start()
    else:
        capture = _open_camera(
            args.camera_device, capture_width, capture_height, args.fps or 30.0
        )
    calibrator = HandEyeCalibrator(config)
    print("SPACE freeze and enter TCP X Y Z R P Y | Q quit")
    try:
        while True:
            if oak_source is not None:
                bundle = oak_source.latest()
                if bundle is None:
                    time.sleep(0.01)
                    continue
                frame = bundle.color_bgr
            else:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("camera read failed")
            if (frame.shape[1], frame.shape[0]) != image_size:
                raise RuntimeError(
                    "camera eye size {} does not match intrinsics {}".format(
                        (frame.shape[1], frame.shape[0]), image_size
                    )
                )
            preview = frame.copy()
            if calibrator.target_type == CHECKERBOARD_TARGET:
                observation = calibrator.checkerboard.estimate(frame, matrix, distortion)
                preview = calibrator.checkerboard.draw(frame, observation)
                target_text = "checkerboard={}/{}".format(
                    calibrator.checkerboard.corner_count if observation.corners is not None else 0,
                    calibrator.checkerboard.corner_count,
                )
            else:
                detections = calibrator.localizer.detect(frame)
                if detections:
                    ids = np.asarray(sorted(detections), dtype=np.int32).reshape(-1, 1)
                    corners = [np.asarray(detections[int(tag_id)], dtype=np.float32).reshape(1, 4, 2) for tag_id in ids.reshape(-1)]
                    cv2.aruco.drawDetectedMarkers(preview, corners, ids)
                target_text = "mapped={}".format(
                    sorted(set(detections).intersection(TagMap(config).ids))
                )
            cv2.putText(preview, "{} saved={}".format(target_text, _sample_count(args.samples, config)),
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
            cv2.imshow("competition hand-eye capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == 32:
                text = input("TCP X Y Z R P Y (mm, deg): ").strip().split()
                if len(text) != 6:
                    print("rejected: enter exactly six values")
                    continue
                values = [float(value) for value in text]
                base_from_tcp = transform_from_xyz_rpy_mm(values[:3], values[3:])
                sample = calibrator.add_image_sample(frame, base_from_tcp, matrix, distortion)
                image_source = (
                    "oak:{}".format(camera.get("mxid", ""))
                    if oak_source is not None
                    else "live:{}".format(args.camera_device)
                )
                count = _append_sample(args.samples, config, sample, image_source)
                print("saved sample {}: {} RMS={:.3f}px".format(count, sample.target_label, sample.rms_reprojection_error_px))
    finally:
        if capture is not None:
            capture.release()
        if oak_source is not None:
            oak_source.stop()
        cv2.destroyAllWindows()


def _sample_count(path, config):
    try:
        return len(HandEyeSampleStore(path, config).entries())
    except ValueError:
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("tag-list")
    target = commands.add_parser("hand-eye-target")
    target.add_argument(
        "--type", choices=(APRILTAG_MAP_TARGET, CHECKERBOARD_TARGET), required=True
    )
    target.add_argument("--board-width-mm", type=float)
    target.add_argument("--board-height-mm", type=float)
    target.add_argument("--square-size-mm", type=float)
    target.add_argument(
        "--squares-x", type=int,
        help="printed black+white square count along the board long side",
    )
    target.add_argument(
        "--squares-y", type=int,
        help="printed black+white square count along the board short side",
    )
    tag_set = commands.add_parser("tag-set")
    tag_set.add_argument("--id", type=int, required=True)
    tag_set.add_argument("--bottom-right-mm", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    tag_set.add_argument("--rpy-deg", nargs=3, type=float, metavar=("ROLL", "PITCH", "YAW"))
    tag_remove = commands.add_parser("tag-remove")
    tag_remove.add_argument("--id", type=int, required=True)
    tag_default = commands.add_parser("tag-default-rpy")
    tag_default.add_argument("--rpy-deg", nargs=3, type=float, required=True)
    add = commands.add_parser("hand-eye-add")
    add.add_argument("--image", type=Path, required=True)
    add.add_argument("--tcp-xyz-mm", nargs=3, type=float, required=True)
    add.add_argument("--tcp-rpy-deg", nargs=3, type=float, required=True)
    add.add_argument("--intrinsics", type=Path)
    add.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    hand_live = commands.add_parser("hand-eye-live")
    hand_live.add_argument("--camera-device", default="0")
    hand_live.add_argument("--fps", type=float)
    hand_live.add_argument("--intrinsics", type=Path)
    hand_live.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    solve = commands.add_parser("hand-eye-solve")
    solve.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    localize = commands.add_parser("localize-image")
    localize.add_argument("--image", type=Path, required=True)
    localize.add_argument("--intrinsics", type=Path)
    localize.add_argument("--tcp-xyz-mm", nargs=3, type=float)
    localize.add_argument("--tcp-rpy-deg", nargs=3, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    config = CompetitionConfig(args.config)
    if hasattr(args, "samples"):
        args.samples = _configured_samples(config, args.samples)
    tag_map = TagMap(config)
    if args.command == "check":
        intrinsics = config.camera["color_intrinsics_file"]
        depth_calibration = config.camera.get(
            "rgbd_calibration_file", config.camera.get("factory_calibration_file")
        )
        print(
            "config: PASS\ncamera profile: {}\nTag IDs: {}\nhand-eye target: {}\nhand-eye valid: {}\n"
            "color intrinsics: {}\ndepth calibration: {}\nsegmentation valid: {}\n"
            "planning valid: {}\ngrasp execution valid: {}".format(
                config.active_camera_profile, list(tag_map.ids),
                config.data["hand_eye"]["calibration_target"]["type"],
                config.hand_eye_valid, intrinsics, depth_calibration,
                config.segmentation_valid,
                bool(config.data.get("planning_validation", {}).get("valid", False)),
                bool(config.data.get("grasp_execution_validation", {}).get("valid", False)),
            )
        )
        return 0
    if args.command == "tag-list":
        for tag_id in tag_map.ids:
            entry = tag_map.entry(tag_id)
            print("ID {}: BR={} mm, RPY={} deg".format(tag_id, entry["bottom_right_xyz_mm"], entry.get("base_from_tag_rpy_deg", tag_map.default_rpy_deg.tolist())))
        return 0
    if args.command == "hand-eye-target":
        sample_path = _configured_samples(config)
        target = config.data["hand_eye"]["calibration_target"]
        target["type"] = args.type
        checkerboard = target["checkerboard"]
        for argument, field in (
            (args.board_width_mm, "board_width_mm"),
            (args.board_height_mm, "board_height_mm"),
            (args.square_size_mm, "square_size_mm"),
            (args.squares_x, "squares_x"),
            (args.squares_y, "squares_y"),
        ):
            if argument is not None:
                checkerboard[field] = int(argument) if field.startswith("squares_") else float(argument)
        configured = (
            checkerboard.get("squares_x") is not None
            and checkerboard.get("squares_y") is not None
        )
        checkerboard["configured"] = configured
        checkerboard["inner_corners"] = (
            [int(checkerboard["squares_x"]) - 1, int(checkerboard["squares_y"]) - 1]
            if configured else None
        )
        if args.type == CHECKERBOARD_TARGET and not configured:
            raise ValueError(
                "checkerboard requires --squares-x and --squares-y; count all black+white cells, not black cells only"
            )
        config.data["hand_eye"]["tcp_from_color_camera"]["valid"] = False
        config.save()
        backup = HandEyeSampleStore(sample_path, config).reset()
        print(
            "hand-eye target={} checkerboard={}x{} mm, square={} mm, squares={}x{}, inner corners={}; "
            "previous session archived={} and reset".format(
                args.type, checkerboard["board_width_mm"],
                checkerboard["board_height_mm"], checkerboard["square_size_mm"],
                checkerboard.get("squares_x"), checkerboard.get("squares_y"),
                checkerboard["inner_corners"], backup or "none",
            )
        )
        return 0
    if args.command == "tag-set":
        tag_map.set_tag(args.id, args.bottom_right_mm, args.rpy_deg)
        print("updated Tag {}; previous hand-eye result invalidated".format(args.id))
        return 0
    if args.command == "tag-remove":
        tag_map.remove_tag(args.id)
        print("removed Tag {}; previous hand-eye result invalidated".format(args.id))
        return 0
    if args.command == "tag-default-rpy":
        tag_map.set_default_rpy(args.rpy_deg)
        print("updated default Tag orientation; previous hand-eye result invalidated")
        return 0
    if args.command == "hand-eye-live":
        return _hand_eye_live(args, config)
    if args.command == "hand-eye-add":
        image = cv2.imread(str(args.image.expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("failed to read image {}".format(args.image))
        matrix, distortion, expected_size = _configured_intrinsics(config, args.intrinsics)
        if (image.shape[1], image.shape[0]) != expected_size:
            raise ValueError("image size does not match intrinsics: {} != {}".format((image.shape[1], image.shape[0]), expected_size))
        calibrator = HandEyeCalibrator(config)
        base_from_tcp = transform_from_xyz_rpy_mm(args.tcp_xyz_mm, args.tcp_rpy_deg)
        sample = calibrator.add_image_sample(image, base_from_tcp, matrix, distortion)
        count = _append_sample(args.samples, config, sample, args.image.resolve())
        print("saved sample {}: {} RMS={:.3f}px".format(count, sample.target_label, sample.rms_reprojection_error_px))
        return 0
    if args.command == "hand-eye-solve":
        data = HandEyeSampleStore(args.samples, config).load()
        calibrator = HandEyeCalibrator(config)
        for entry in data["samples"]:
            calibrator.add_stored_sample(entry)
        result = calibrator.solve()
        calibrator.promote(result)
        print("T_tcp_color_camera:\n{}".format(_matrix_text(result.tcp_from_camera)))
        print("inliers: {}/{} | translation RMS/max: {:.3f}/{:.3f} mm | rotation RMS/max: {:.3f}/{:.3f} deg".format(len(result.inlier_indices), result.total_samples, result.translation_rms_mm, result.translation_max_mm, result.rotation_rms_deg, result.rotation_max_deg))
        return 0
    if args.command == "localize-image":
        image = cv2.imread(str(args.image.expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("failed to read image {}".format(args.image))
        matrix, distortion, expected_size = _configured_intrinsics(config, args.intrinsics)
        if (image.shape[1], image.shape[0]) != expected_size:
            raise ValueError("image size does not match intrinsics")
        tcp = None
        robot_time = None
        if args.tcp_xyz_mm is not None or args.tcp_rpy_deg is not None:
            if args.tcp_xyz_mm is None or args.tcp_rpy_deg is None:
                raise ValueError("TCP XYZ and RPY must be supplied together")
            tcp = transform_from_xyz_rpy_mm(args.tcp_xyz_mm, args.tcp_rpy_deg)
            robot_time = time.monotonic()
        result = HybridLocalizer(config).localize(image, matrix, distortion, tcp, robot_timestamp_s=robot_time)
        print(json.dumps({"valid": result.valid, "source": result.source, "visible_tag_ids": result.visible_tag_ids, "used_tag_ids": result.used_tag_ids, "rms_px": result.rms_reprojection_error_px, "reason": result.reason}, ensure_ascii=False, indent=2))
        if result.valid:
            xyz_m, rpy_deg = xyz_rpy_from_transform(result.base_from_camera)
            print("T_base_color_camera:\n{}".format(_matrix_text(result.base_from_camera)))
            print("XYZ mm: {}\nRPY deg: {}".format((xyz_m * 1000.0).round(3).tolist(), rpy_deg.round(4).tolist()))
        return 0 if result.valid else 2
    raise RuntimeError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
