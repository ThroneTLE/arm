"""Offline verification CLI for the migrated visual grasping pipeline.

This uses the static RGB/depth/K files from the release (``static_frame``) to
exercise YOLO, FoundationPose, coordinate conversion and grasp generation
without a live camera or ROS bridge.

Example::

    python -m tool.visual_grasp_pipeline.offline \
        --config tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml \
        --label can
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import VisualGraspConfig
from .detection import detect_all_objects, detect_tags, select_target
from .foundationpose import FoundationPosePoseEstimator
from .geometry import (
    build_world_from_tags,
    compute_grasp,
    compute_grasp_sphere,
    fill_depth_roi,
    to_world_and_compensate,
)


def load_static_frame(static_dir: str):
    static_dir = Path(static_dir).expanduser()
    rgb_path = static_dir / "rgb.png"
    depth_path = static_dir / "depth.png"
    k_path = static_dir / "cam_K.txt"
    if not rgb_path.is_file():
        raise FileNotFoundError(f"missing static RGB: {rgb_path}")
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise RuntimeError(f"failed to read RGB image: {rgb_path}")
    height, width = rgb.shape[:2]

    if depth_path.is_file():
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"failed to read depth image: {depth_path}")
        if depth.shape[:2] != (height, width):
            aligned = np.zeros((height, width), dtype=np.uint16)
            aligned[: depth.shape[0], :] = depth
            depth = aligned
    else:
        depth = np.zeros((height, width), dtype=np.uint16)

    if k_path.is_file():
        k = np.loadtxt(k_path).reshape(3, 3)
    else:
        # Fallback matching the release's default Gemini intrinsics.
        k = np.array(
            [[451.9166, 0.0, 326.2534], [0.0, 451.9166, 245.3197], [0.0, 0.0, 1.0]]
        )
    return rgb, depth.astype(np.float32) / 1000.0, k


def run_offline(config: VisualGraspConfig, label=None, save_dir=None):
    missing = config.ensure_paths()
    if missing:
        raise FileNotFoundError(
            "missing configured paths: {}".format(", ".join(missing))
        )
    if not config.object_models:
        raise RuntimeError("no object models configured")

    from ultralytics import YOLO

    model = YOLO(config.yolo_weights)
    rgb, depth_m, k = load_static_frame(config.static_frame_dir)
    objects = detect_all_objects(rgb, model, conf=config.yolo_conf, imgsz=config.yolo_imgsz)
    if not objects:
        return {
            "status": "no_objects",
            "classes": list(model.names.values()),
            "detections": [],
        }

    target = select_target(objects, label if label is not None else config.target_label)
    if target is None:
        return {
            "status": "target_not_found",
            "target_label": label if label is not None else config.target_label,
            "detections": [
                {
                    "name": obj["name"],
                    "cls": obj["cls"],
                    "conf": obj["conf"],
                    "xyxy": list(obj["xyxy"]),
                }
                for obj in objects
            ],
        }

    object_key = config.resolve_object_key(target["name"], target["cls"])
    mesh_path = config.mesh_for_object(object_key)
    if not mesh_path or not os.path.exists(mesh_path):
        raise FileNotFoundError(
            f"mesh for object '{object_key}' is not configured or does not exist: {mesh_path}"
        )

    height, width = rgb.shape[:2]
    # 掩膜: 直接用 YOLO 识别框(矩形)作为掩膜, 不使用实例分割输出
    mask = np.zeros((height, width), np.uint8)
    bx1, by1, bx2, by2 = (int(round(v)) for v in target["xyxy"])
    mask[max(0, by1):min(height, by2), max(0, bx1):min(width, bx2)] = 255

    estimator = FoundationPosePoseEstimator(
        foundationpose_root=config.foundationpose_root,
        mesh_path=mesh_path,
        mesh_scale_to_meters=config.mesh_scale_for_object(object_key),
        debug_dir=config.debug_dir,
        est_refine_iter=config.est_refine_iter,
        track_refine_iter=config.track_refine_iter,
        device=config.device,
        use_mask_center_guidance=config.use_mask_center_guidance,
        registration_max_hypotheses=(
            config.foundationpose_registration_hypotheses
        ),
    )
    try:
        camera_from_object = estimator.register(
            rgb, fill_depth_roi(depth_m, mask), mask, k
        )
    finally:
        estimator.close()

    # Optional: use AprilTag if the static frame contains a known tag.
    tags = []
    try:
        tags = detect_tags(rgb, config.tag_size_mm, k)
    except Exception:
        tags = []
    world_from_camera = None
    if tags:
        tag_world = build_world_from_tags(tags)
        if tag_world is not None:
            world_from_camera = np.linalg.inv(tag_world)

    if world_from_camera is not None:
        world_from_object = to_world_and_compensate(
            camera_from_object,
            world_from_camera,
            offset_xy_mm=config.offset_xy_mm,
            center_offset_mm=config.center_offset_mm,
            flip_x=config.flip_x,
            flip_y=config.flip_y,
        )
    else:
        # Without tags, still report the camera-frame pose and a default identity
        # workspace transform so the callers can inspect the raw FoundationPose result.
        world_from_object = camera_from_object.copy()

    rule = config.rule_for_object(object_key)
    if rule.type == "sphere":
        grasp = compute_grasp_sphere(world_from_object, rule.offset_mm)
    else:
        grasp = compute_grasp(world_from_object, rule.offset_mm)

    result = {
        "status": "ok",
        "target": {
            "name": target["name"],
            "cls": target["cls"],
            "conf": target["conf"],
            "xyxy": list(target["xyxy"]),
        },
        "object_key": object_key,
        "mesh_path": mesh_path,
        "tag_count": len(tags),
        "camera_from_object": camera_from_object.tolist(),
        "world_from_object": world_from_object.tolist(),
        "grasp": grasp.tolist(),
        "camera_matrix": k.tolist(),
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "camera_from_object.npy", camera_from_object)
        np.save(save_dir / "world_from_object.npy", world_from_object)
        np.save(save_dir / "grasp.npy", grasp)
        (save_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml",
        help="path to visual_grasp_pipeline.yaml",
    )
    parser.add_argument("--label", default=None, help="override target class name/id")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="optional output directory for npy/json verification artifacts",
    )
    args = parser.parse_args(argv)

    config = VisualGraspConfig.from_yaml(args.config)
    # FoundationPose's vendor code prints diagnostics directly to file
    # descriptor 1 (not only through Python sys.stdout).  Redirect fd 1 to
    # stderr while running so the only stdout output is the final JSON object.
    saved_stdout_fd = os.dup(1)
    original_stdout = sys.stdout
    try:
        sys.stdout.flush()
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        result = run_offline(config, label=args.label, save_dir=args.save_dir)
    finally:
        sys.stdout.flush()
        try:
            # Flush C/C++ stdio buffers while fd 1 is still pointed at stderr.
            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass
        sys.stdout = original_stdout
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
