#!/usr/bin/env python3
"""One-shot lemon vision plan for the competition UI.

The process is deliberately split from the Qt workbench: the workbench owns
the OAK camera and NexBot controller, while this module runs YOLO and
FoundationPose in the ``foundationpose`` Conda environment using one frozen
RGB-D snapshot and one frozen UCS1 TCP pose.

Coordinate contract::

    T_user1_object = T_user1_tcp @ T_tcp_camera @ T_camera_object

The output is a preview only.  It contains TCP grasp/place poses, but never
opens a controller connection and never sends motion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from competition_pipeline.geometry import (
    as_transform,
    inexbot_abc_from_transform,
    transform_from_inexbot_abc,
)
from tool.object_model_builder.rgbd_geometry import CameraIntrinsics
from tool.visual_grasp_pipeline.config import VisualGraspConfig
from tool.visual_grasp_pipeline.geometry import fill_depth_roi


def _matrix(entry, name):
    if isinstance(entry, dict):
        if entry.get("valid") is False:
            raise ValueError("{} is marked invalid".format(name))
        entry = entry.get("matrix")
    return as_transform(np.asarray(entry, dtype=np.float64), name)


def _pose_mapping(transform):
    xyz_m, abc_rad = inexbot_abc_from_transform(transform)
    return {
        "xyz_mm": [round(float(value) * 1000.0, 3) for value in xyz_m],
        "abc_rad": [round(float(value), 7) for value in abc_rad],
        "abc_deg": [round(float(np.degrees(value)), 4) for value in abc_rad],
        "matrix": np.asarray(transform, dtype=np.float64).tolist(),
    }


def top_down_grasp_frame(user1_from_object, grasp_type="elongated", offset_mm=0.0):
    """Return the canonical top-down grasp frame used by ``planning.py``.

    Local ``+Z`` is the approach direction and therefore points along user
    ``-Z``.  For the lemon mesh the local X axis is the 90 mm long axis; the
    closing axis is chosen perpendicular to its horizontal projection so the
    gripper closes across the smaller diameter.
    """

    source = as_transform(user1_from_object, "user1_from_object")
    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if str(grasp_type) == "elongated":
        long_axis = np.asarray(source[:3, 0], dtype=np.float64).copy()
        long_axis[2] = 0.0
        norm = float(np.linalg.norm(long_axis))
        if norm > 1e-6:
            long_axis /= norm
            closing = np.array([-long_axis[1], long_axis[0], 0.0])
        else:
            closing = np.array([0.0, 1.0, 0.0])
    else:
        closing = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(closing, approach)
    grasp = np.eye(4, dtype=np.float64)
    grasp[:3, :3] = np.column_stack([lateral, closing, approach])
    grasp[:3, 3] = source[:3, 3]
    grasp[:3, 3] += approach * (float(offset_mm) / 1000.0)
    return as_transform(grasp, "user1_from_grasp")


def build_pick_place_plan(
    camera_from_object,
    user1_from_tcp,
    tcp_from_camera,
    tcp_from_grasp,
    *,
    grasp_type="elongated",
    grasp_offset_mm=0.0,
    place_offset_user_mm=(0.0, -50.0, 0.0),
):
    """Compose user-frame object/grasp poses and TCP pick/place targets."""

    camera_from_object = as_transform(camera_from_object, "camera_from_object")
    user1_from_tcp = as_transform(user1_from_tcp, "user1_from_tcp")
    tcp_from_camera = as_transform(tcp_from_camera, "tcp_from_camera")
    tcp_from_grasp = as_transform(tcp_from_grasp, "tcp_from_grasp")
    offset = np.asarray(place_offset_user_mm, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(offset)):
        raise ValueError("place offset contains non-finite values")

    user1_from_object = user1_from_tcp @ tcp_from_camera @ camera_from_object
    user1_from_grasp = top_down_grasp_frame(
        user1_from_object, grasp_type, float(grasp_offset_mm)
    )
    user1_from_tcp_grasp = user1_from_grasp @ np.linalg.inv(tcp_from_grasp)

    user1_from_place_grasp = user1_from_grasp.copy()
    user1_from_place_grasp[:3, 3] += offset / 1000.0
    user1_from_tcp_place = user1_from_place_grasp @ np.linalg.inv(tcp_from_grasp)

    object_xyz_mm = user1_from_object[:3, 3] * 1000.0
    return {
        "object": _pose_mapping(user1_from_object),
        "grasp_frame": _pose_mapping(user1_from_grasp),
        "grasp_tcp": _pose_mapping(user1_from_tcp_grasp),
        "place_tcp": _pose_mapping(user1_from_tcp_place),
        "place_offset_user_mm": [float(value) for value in offset],
        "origin_xy_error_mm": round(float(np.linalg.norm(object_xyz_mm[:2])), 3),
    }


def validate_plan(
    plan,
    *,
    origin_xy_tolerance_mm,
    workspace_min_mm,
    workspace_max_mm,
    lift_mm,
    confidence,
    minimum_confidence,
    depth_coverage,
    minimum_depth_coverage,
    depth_center_delta_mm,
    maximum_depth_center_delta_mm,
):
    """Return fail-closed reasons for a preview plan."""

    reasons = []
    if float(confidence) < float(minimum_confidence):
        reasons.append(
            "YOLO confidence {:.3f} < {:.3f}".format(
                confidence, minimum_confidence
            )
        )
    if float(depth_coverage) < float(minimum_depth_coverage):
        reasons.append(
            "mask depth coverage {:.1%} < {:.1%}".format(
                depth_coverage, minimum_depth_coverage
            )
        )
    if float(depth_center_delta_mm) > float(maximum_depth_center_delta_mm):
        reasons.append(
            "FoundationPose/depth delta {:.1f} mm > {:.1f} mm".format(
                depth_center_delta_mm, maximum_depth_center_delta_mm
            )
        )
    if float(plan["origin_xy_error_mm"]) > float(origin_xy_tolerance_mm):
        reasons.append(
            "lemon origin XY error {:.1f} mm > {:.1f} mm".format(
                plan["origin_xy_error_mm"], origin_xy_tolerance_mm
            )
        )

    grasp_matrix = np.asarray(plan["grasp_frame"]["matrix"], dtype=np.float64)
    approach_alignment = float(
        np.dot(grasp_matrix[:3, 2], np.array([0.0, 0.0, -1.0]))
    )
    if approach_alignment < float(np.cos(np.deg2rad(5.0))):
        reasons.append("grasp approach is not top-down in user frame")

    minimum = np.asarray(workspace_min_mm, dtype=np.float64).reshape(3)
    maximum = np.asarray(workspace_max_mm, dtype=np.float64).reshape(3)
    lift = np.array([0.0, 0.0, float(lift_mm)], dtype=np.float64)
    for name in ("grasp_tcp", "place_tcp"):
        xyz = np.asarray(plan[name]["xyz_mm"], dtype=np.float64)
        for label, point in ((name, xyz), (name + "_above", xyz + lift)):
            if np.any(point < minimum) or np.any(point > maximum):
                reasons.append(
                    "{} XYZ {} outside workspace {}..{} mm".format(
                        label,
                        np.round(point, 2).tolist(),
                        minimum.tolist(),
                        maximum.tolist(),
                    )
                )
    return reasons


def _load_snapshot(path):
    with np.load(str(path), allow_pickle=False) as data:
        color = np.asarray(data["color_bgr"], dtype=np.uint8)
        depth = np.asarray(data["depth_m"], dtype=np.float32)
        matrix = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("snapshot color must be HxWx3 BGR")
    if depth.shape != color.shape[:2]:
        raise ValueError("snapshot depth and color dimensions differ")
    return color, depth, matrix


def _select_target(objects, label):
    candidates = [item for item in objects if str(item.get("name")) == str(label)]
    if not candidates:
        names = sorted(set(str(item.get("name")) for item in objects))
        raise RuntimeError(
            "target {!r} not detected; visible classes={}".format(label, names)
        )
    return max(
        candidates,
        key=lambda item: (
            float(item.get("conf", 0.0)),
            (item["xyxy"][2] - item["xyxy"][0])
            * (item["xyxy"][3] - item["xyxy"][1]),
        ),
    )


def run(args):
    visual = VisualGraspConfig.from_yaml(args.visual_config)
    competition = yaml.safe_load(
        Path(args.competition_config).read_text(encoding="utf-8")
    ) or {}
    color, depth, camera_matrix = _load_snapshot(args.snapshot)
    tcp_data = json.loads(Path(args.tcp_pose).read_text(encoding="utf-8"))
    user1_from_tcp = transform_from_inexbot_abc(
        np.asarray(tcp_data["xyz_mm"], dtype=np.float64) / 1000.0,
        np.asarray(tcp_data["abc_rad"], dtype=np.float64),
    )

    from ultralytics import YOLO
    from tool.visual_grasp_pipeline.detection import detect_all_objects, draw_boxes
    from tool.visual_grasp_pipeline.foundationpose import FoundationPosePoseEstimator
    from tool.visual_grasp_pipeline.oak_vision_node import (
        draw_pose_axes,
        prepare_foundationpose_input,
        OakSnapshot,
    )

    model = YOLO(visual.yolo_weights)
    objects = detect_all_objects(
        color, model, conf=visual.yolo_conf, imgsz=visual.yolo_imgsz
    )
    target = _select_target(objects, args.target_label)
    mask = target.get("mask")
    if mask is None or not np.any(mask):
        raise RuntimeError("lemon detector did not provide an instance mask")

    valid = (mask > 0) & np.isfinite(depth) & (depth > 0.05)
    mask_pixels = int(np.count_nonzero(mask))
    valid_points = int(np.count_nonzero(valid))
    coverage = float(valid_points) / float(max(mask_pixels, 1))
    if valid_points < 30:
        raise RuntimeError("lemon mask has only {} valid depth points".format(valid_points))

    intrinsics = CameraIntrinsics(
        color.shape[1], color.shape[0], camera_matrix, np.zeros(5)
    )
    snapshot = OakSnapshot(
        color_bgr=color,
        depth_m=depth,
        intrinsics=intrinsics,
        timestamp_s=float(tcp_data.get("image_timestamp_s", 0.0)),
        sync_delta_s=float(tcp_data.get("sync_delta_s", 0.0)),
    )
    fp_input = prepare_foundationpose_input(
        snapshot,
        target,
        mask,
        padding_pixels=visual.foundationpose_roi_padding_pixels,
        maximum_size=visual.foundationpose_max_input_size,
    )
    try:
        model.to("cpu")
        model.predictor = None
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    object_key = visual.resolve_object_key(target["name"], target["cls"])
    mesh_path = visual.mesh_for_object(object_key)
    if not mesh_path or not Path(mesh_path).is_file():
        raise RuntimeError("CAD mesh is missing for {}: {}".format(object_key, mesh_path))
    estimator = FoundationPosePoseEstimator(
        foundationpose_root=visual.foundationpose_root,
        mesh_path=mesh_path,
        mesh_scale_to_meters=visual.mesh_scale_for_object(object_key),
        debug_dir=visual.debug_dir,
        est_refine_iter=visual.est_refine_iter,
        track_refine_iter=visual.track_refine_iter,
        device=visual.device,
        use_mask_center_guidance=visual.use_mask_center_guidance,
        registration_max_hypotheses=visual.foundationpose_registration_hypotheses,
    )
    mesh_bounds = estimator.mesh_bounds
    try:
        camera_from_object = estimator.register(
            fp_input.color_bgr,
            fill_depth_roi(fp_input.depth_m, fp_input.mask),
            fp_input.mask,
            fp_input.camera_matrix,
        )
    finally:
        estimator.close()

    camera_from_object = as_transform(camera_from_object, "camera_from_object")
    if not 0.10 < float(camera_from_object[2, 3]) < 3.0:
        raise RuntimeError(
            "FoundationPose camera distance is invalid: {:.3f} m".format(
                camera_from_object[2, 3]
            )
        )
    depth_median = float(np.median(depth[valid]))
    depth_delta_mm = abs(float(camera_from_object[2, 3]) - depth_median) * 1000.0
    hand_eye = _matrix(
        competition["hand_eye"]["tcp_from_color_camera"],
        "hand_eye.tcp_from_color_camera",
    )
    tcp_from_grasp = _matrix(
        competition["grasp_planning"]["tcp_from_grasp"],
        "grasp_planning.tcp_from_grasp",
    )
    rule = visual.rule_for_object(object_key)
    place_offset = [float(value) for value in args.place_offset_user_mm]
    plan = build_pick_place_plan(
        camera_from_object,
        user1_from_tcp,
        hand_eye,
        tcp_from_grasp,
        grasp_type=rule.type,
        grasp_offset_mm=rule.offset_mm,
        place_offset_user_mm=place_offset,
    )
    reasons = validate_plan(
        plan,
        origin_xy_tolerance_mm=args.origin_xy_tolerance_mm,
        workspace_min_mm=competition["safety"]["workspace_min_mm"],
        workspace_max_mm=competition["safety"]["workspace_max_mm"],
        lift_mm=args.lift_mm,
        confidence=target["conf"],
        minimum_confidence=visual.yolo_conf,
        depth_coverage=coverage,
        minimum_depth_coverage=args.minimum_depth_coverage,
        depth_center_delta_mm=depth_delta_mm,
        maximum_depth_center_delta_mm=args.maximum_depth_center_delta_mm,
    )
    estimated_grasp_width_mm = None
    if mesh_bounds is None:
        reasons.append("lemon CAD bounds are unavailable")
    else:
        extents_mm = np.sort(
            (np.asarray(mesh_bounds[1]) - np.asarray(mesh_bounds[0])) * 1000.0
        )
        # Lemon local X is the long axis.  The larger of the two minor CAD
        # dimensions is the conservative diameter for the binary gripper.
        estimated_grasp_width_mm = float(extents_mm[1])
        minimum_width = float(
            competition["grasp_planning"]["minimum_grasp_width_mm"]
        )
        maximum_width = float(
            competition["grasp_planning"]["maximum_grasp_width_mm"]
        )
        if not minimum_width <= estimated_grasp_width_mm <= maximum_width:
            reasons.append(
                "estimated lemon width {:.1f} mm outside gripper {:.1f}..{:.1f} mm".format(
                    estimated_grasp_width_mm, minimum_width, maximum_width
                )
            )

    overlay = draw_boxes(color, objects)
    contours, _ = cv2.findContours(
        (np.asarray(mask) > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
    overlay = draw_pose_axes(overlay, camera_from_object, camera_matrix)
    Path(args.overlay).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.overlay), overlay):
        raise RuntimeError("failed to save overlay: {}".format(args.overlay))

    result = {
        "schema": 1,
        "target": {
            "name": str(target["name"]),
            "class_id": int(target["cls"]),
            "confidence": round(float(target["conf"]), 5),
            "bbox_xyxy": [int(value) for value in target["xyxy"]],
            "object_key": object_key,
        },
        "quality": {
            "mask_pixels": mask_pixels,
            "valid_depth_points": valid_points,
            "depth_coverage": round(coverage, 5),
            "mask_depth_median_mm": round(depth_median * 1000.0, 2),
            "foundationpose_depth_delta_mm": round(depth_delta_mm, 2),
            "estimated_grasp_width_mm": (
                None
                if estimated_grasp_width_mm is None
                else round(estimated_grasp_width_mm, 2)
            ),
        },
        "camera_from_object": _pose_mapping(camera_from_object),
        "user1_from_tcp_at_capture": _pose_mapping(user1_from_tcp),
        "plan": plan,
        "safe_to_execute": not reasons,
        "blocked_reasons": reasons,
        "overlay": str(Path(args.overlay).resolve()),
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--tcp-pose", required=True)
    parser.add_argument("--visual-config", required=True)
    parser.add_argument("--competition-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--target-label", default="lemon")
    parser.add_argument(
        "--place-offset-user-mm", nargs=3, type=float, default=(0.0, -50.0, 0.0)
    )
    parser.add_argument("--origin-xy-tolerance-mm", type=float, default=50.0)
    parser.add_argument("--lift-mm", type=float, default=80.0)
    parser.add_argument("--minimum-depth-coverage", type=float, default=0.15)
    parser.add_argument("--maximum-depth-center-delta-mm", type=float, default=80.0)
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return run(parse_args(argv))
    except Exception as error:
        print("visual grasp planning failed: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
