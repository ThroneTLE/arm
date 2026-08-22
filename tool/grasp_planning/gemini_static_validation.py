#!/usr/bin/env python3
"""Offline Gemini static-frame validation: YOLO segmentation -> AnyGrasp.

The defaults intentionally reuse the released Gemini ``static_frame`` and
YOLO paths from ``tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml``.
No ROS master, live camera, FoundationPose mesh, or hand-eye calibration is
required.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import yaml

from tool.grasp_planning.anygrasp_planner import (
    AnyGraspPlanner,
    GraspCandidate,
    filter_by_score,
    filter_by_width,
)
from tool.visual_grasp_pipeline.config import VisualGraspConfig
from tool.visual_grasp_pipeline.detection import detect_all_objects
from tool.visual_grasp_pipeline.offline import load_static_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISUAL_CONFIG = (
    PROJECT_ROOT / "tool" / "visual_grasp_pipeline" / "config" / "visual_grasp_pipeline.yaml"
)
DEFAULT_GRASP_CONFIG = Path(__file__).resolve().parent / "config" / "grasp_planning.yaml"
DEFAULT_OUTPUT_DIR = Path(
    "/home/throne/workspaces/arm_data/anygrasp_gemini_static_validation"
)


def choose_target(objects, label: Optional[str], instance: int = 0):
    """Choose one instance, with indices ordered by model confidence."""
    candidates = list(objects)
    if label:
        candidates = [
            item
            for item in candidates
            if str(item.get("name")) == str(label) or str(item.get("cls")) == str(label)
        ]
    candidates.sort(key=lambda item: float(item.get("conf", 0.0)), reverse=True)
    if not candidates:
        return None
    index = int(instance)
    if index < 0 or index >= len(candidates):
        raise IndexError(
            "target instance {} is outside the {} matching detections".format(
                index, len(candidates)
            )
        )
    return candidates[index]


def clean_target_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    erosion_pixels: int = 2,
    iqr_multiplier: float = 2.0,
) -> np.ndarray:
    """Remove mask-edge leakage and implausible depth outliers."""
    selected = np.asarray(mask).astype(bool)
    if selected.shape != depth_m.shape:
        raise ValueError("YOLO mask and aligned Gemini depth must have identical dimensions")
    radius = max(0, int(erosion_pixels))
    if radius:
        size = radius * 2 + 1
        selected = cv2.erode(
            selected.astype(np.uint8), np.ones((size, size), dtype=np.uint8)
        ).astype(bool)
    valid_values = np.asarray(depth_m, dtype=np.float32)[selected & (depth_m > 0.0)]
    if len(valid_values) < 64:
        return np.zeros_like(selected, dtype=bool)
    first_quartile, third_quartile = np.percentile(valid_values, [25.0, 75.0])
    spread = max(float(third_quartile - first_quartile), 0.005)
    lower = max(0.0, float(first_quartile) - float(iqr_multiplier) * spread)
    upper = float(third_quartile) + float(iqr_multiplier) * spread
    return selected & (depth_m >= lower) & (depth_m <= upper)


def _sample_indices(indices: np.ndarray, limit: int, generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(indices) <= int(limit):
        return indices
    selected = generator.choice(len(indices), size=int(limit), replace=False)
    return indices[np.sort(selected)]


def build_steered_cloud(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    target_mask: np.ndarray,
    minimum_depth_m: float = 0.15,
    maximum_depth_m: float = 2.0,
    maximum_points: int = 40000,
    seed: int = 7,
):
    """Project registered Gemini depth and retain a YOLO steering vector."""
    depth = np.asarray(depth_m, dtype=np.float32)
    rgb = np.asarray(rgb_bgr)
    target = np.asarray(target_mask).astype(bool)
    if depth.shape != rgb.shape[:2] or target.shape != depth.shape:
        raise ValueError("RGB, aligned depth, and target mask dimensions must match")
    intrinsics = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    valid = (
        np.isfinite(depth)
        & (depth >= float(minimum_depth_m))
        & (depth <= float(maximum_depth_m))
    )
    target_indices = np.flatnonzero(valid & target)
    context_indices = np.flatnonzero(valid & ~target)
    if len(target_indices) < 64:
        raise ValueError("YOLO mask contains fewer than 64 valid Gemini depth pixels")

    point_limit = max(64, int(maximum_points))
    generator = np.random.default_rng(int(seed))
    target_limit = min(len(target_indices), max(64, point_limit // 2))
    target_indices = _sample_indices(target_indices, target_limit, generator)
    context_limit = max(0, point_limit - len(target_indices))
    context_indices = _sample_indices(context_indices, context_limit, generator)
    flat_indices = np.concatenate([target_indices, context_indices])
    steering = np.zeros(len(flat_indices), dtype=bool)
    steering[: len(target_indices)] = True

    rows, columns = np.unravel_index(flat_indices, depth.shape)
    z = depth[rows, columns].astype(np.float32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (columns.astype(np.float32) - cx) * z / fx
    y = (rows.astype(np.float32) - cy) * z / fy
    points = np.column_stack([x, y, z]).astype(np.float32)
    colors = rgb[rows, columns, ::-1].astype(np.float32) / 255.0
    return points, colors, steering


def _project(point: np.ndarray, camera_matrix: np.ndarray):
    xyz = np.asarray(point, dtype=np.float64).reshape(3)
    if xyz[2] <= 1e-6:
        return None
    uvw = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3) @ xyz
    return tuple(np.rint(uvw[:2] / uvw[2]).astype(int))


def draw_grasp_overlay(
    image: np.ndarray,
    grasps: Sequence[GraspCandidate],
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Project grasp centers, approach axes, and jaw axes onto RGB."""
    output = np.asarray(image).copy()
    shown = list(grasps[:1])
    for grasp in shown:
        center = np.asarray(grasp.translation, dtype=np.float64)
        approach = np.asarray(grasp.rotation, dtype=np.float64)[:, 0]
        jaw_axis = np.asarray(grasp.rotation, dtype=np.float64)[:, 1]
        half_width = max(0.5 * float(grasp.width), 0.012)
        palm_center = center - 0.055 * approach
        finger_center = center + max(float(grasp.depth), 0.02) * approach
        palm_left = _project(palm_center + half_width * jaw_axis, camera_matrix)
        palm_right = _project(palm_center - half_width * jaw_axis, camera_matrix)
        tip_left = _project(finger_center + half_width * jaw_axis, camera_matrix)
        tip_right = _project(finger_center - half_width * jaw_axis, camera_matrix)
        center_uv = _project(center, camera_matrix)
        color = (40, 230, 40)
        if None not in (palm_left, palm_right, tip_left, tip_right):
            cv2.line(output, palm_left, palm_right, color, 5, cv2.LINE_AA)
            cv2.line(output, palm_left, tip_left, color, 5, cv2.LINE_AA)
            cv2.line(output, palm_right, tip_right, color, 5, cv2.LINE_AA)
            cv2.circle(output, tip_left, 5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(output, tip_right, 5, (255, 255, 255), 2, cv2.LINE_AA)
        if center_uv is not None:
            cv2.circle(output, center_uv, 6, color, -1, cv2.LINE_AA)
    if shown:
        summary = "Best AnyGrasp: {:.3f}  width: {:.1f} mm".format(
            float(shown[0].score), float(shown[0].width) * 1000.0
        )
        y = output.shape[0] - 14
        cv2.rectangle(output, (5, y - 20), (390, y + 6), (0, 0, 0), -1)
        cv2.putText(
            output, summary, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 255), 1, cv2.LINE_AA
        )
    return output


def draw_detection_view(image: np.ndarray, detections, target, mask: np.ndarray) -> np.ndarray:
    """Draw readable class/confidence labels and highlight the selected mask."""
    output = np.asarray(image).copy()
    selected_color = np.zeros_like(output)
    selected_color[:, :, 2] = 255
    alpha = np.asarray(mask).astype(np.float32)[..., None] * 0.32
    output = np.clip(
        output.astype(np.float32) * (1.0 - alpha)
        + selected_color.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    for item in detections:
        x1, y1, x2, y2 = [int(value) for value in item["xyxy"]]
        selected = item is target
        color = (20, 220, 20) if selected else (255, 80, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3 if selected else 2)
        label = "{} {:.3f}{}".format(
            item["name"], float(item["conf"]), " target" if selected else ""
        )
        cv2.putText(
            output,
            label,
            (x1, max(16, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def _gripper_lines(grasp: GraspCandidate):
    center = np.asarray(grasp.translation, dtype=np.float64)
    rotation = np.asarray(grasp.rotation, dtype=np.float64).reshape(3, 3)
    approach, jaw_axis = rotation[:, 0], rotation[:, 1]
    back = center - 0.06 * approach
    half_width = 0.5 * float(grasp.width)
    points = np.asarray(
        [
            back,
            back + half_width * jaw_axis,
            back - half_width * jaw_axis,
            center + half_width * jaw_axis,
            center - half_width * jaw_axis,
        ]
    )
    return points, np.asarray([[0, 1], [0, 2], [1, 3], [2, 4]], dtype=np.int32)


def visualize_open3d(points, colors, steering, grasps) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points))
    display_colors = np.asarray(colors).copy()
    display_colors[np.asarray(steering).astype(bool)] = [1.0, 0.15, 0.05]
    cloud.colors = o3d.utility.Vector3dVector(display_colors)
    geometries = [cloud]
    for index, grasp in enumerate(grasps):
        line_points, lines = _gripper_lines(grasp)
        gripper = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        color = [0.1, 1.0, 0.1] if index == 0 else [1.0, 0.65, 0.0]
        gripper.colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(lines), 1))
        )
        geometries.append(gripper)
    o3d.visualization.draw_geometries(
        geometries, window_name="Gemini static frame: YOLO + AnyGrasp"
    )


def run(arguments) -> dict:
    visual = VisualGraspConfig.from_yaml(arguments.visual_config)
    weights = str(Path(arguments.weights or visual.yolo_weights).expanduser().resolve())
    static_dir = str(
        Path(arguments.static_frame or visual.static_frame_dir).expanduser().resolve()
    )
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(arguments.grasp_config).expanduser().open("r", encoding="utf-8") as handle:
        grasp_data = yaml.safe_load(handle) or {}
    planner_settings = grasp_data.get("anygrasp", {})

    from ultralytics import YOLO

    rgb, depth_m, camera_matrix = load_static_frame(static_dir)
    model = YOLO(weights)
    confidence = visual.yolo_conf if arguments.confidence is None else arguments.confidence
    detections = detect_all_objects(
        rgb, model, conf=float(confidence), imgsz=int(visual.yolo_imgsz)
    )
    target = choose_target(detections, arguments.label, arguments.instance)
    if target is None:
        raise RuntimeError(
            "YOLO found no target matching {!r}; detections={}".format(
                arguments.label, [item["name"] for item in detections]
            )
        )
    if target.get("mask") is None or not np.any(target["mask"]):
        raise RuntimeError("selected YOLO result has no instance-segmentation mask")

    selected_mask = clean_target_mask(
        target["mask"], depth_m, erosion_pixels=arguments.mask_erosion_pixels
    )
    points, colors, steering = build_steered_cloud(
        rgb,
        depth_m,
        camera_matrix,
        selected_mask,
        minimum_depth_m=arguments.minimum_depth,
        maximum_depth_m=arguments.maximum_depth,
        maximum_points=arguments.maximum_points,
    )
    planner = AnyGraspPlanner(
        checkpoint_path=planner_settings.get("checkpoint_path", ""),
        sdk_grasp_dir=planner_settings.get("sdk_grasp_dir", ""),
        max_gripper_width=planner_settings.get("max_gripper_width_m", 0.08),
        gripper_height=planner_settings.get("gripper_height_m", 0.03),
    )
    started = time.perf_counter()
    candidates = planner.plan(
        points,
        approach_camera=np.asarray([0.0, 0.0, 1.0]),
        approach_thresh=np.deg2rad(float(arguments.approach_cone_deg)),
        dense_grasp=bool(arguments.dense_grasp),
        collision_detection=not bool(arguments.no_collision_detection),
        region_steering=steering,
        top_k=int(arguments.top_k),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    candidates = filter_by_width(
        candidates,
        minimum_width=float(arguments.minimum_width),
        maximum_width=float(arguments.maximum_width),
    )
    candidates = filter_by_score(candidates, float(arguments.minimum_score))

    detection_view = draw_detection_view(rgb, detections, target, selected_mask)
    overlay = draw_grasp_overlay(
        detection_view, candidates[: max(1, int(arguments.overlay_grasps))], camera_matrix
    )
    cv2.imwrite(str(output_dir / "yolo_detections.png"), detection_view)
    cv2.imwrite(str(output_dir / "anygrasp_overlay.png"), overlay)
    cv2.imwrite(
        str(output_dir / "selected_mask.png"), selected_mask.astype(np.uint8) * 255
    )
    np.savez_compressed(
        output_dir / "steered_cloud.npz",
        points=points,
        colors=colors,
        region_steering=steering,
        camera_matrix=camera_matrix,
    )

    result = {
        "status": "ok" if candidates else "no_grasps",
        "weights": weights,
        "static_frame": static_dir,
        "target": {
            "name": str(target["name"]),
            "class_id": int(target["cls"]),
            "confidence": float(target["conf"]),
            "bbox_xyxy": [int(value) for value in target["xyxy"]],
            "valid_depth_pixels": int(np.count_nonzero(selected_mask)),
        },
        "detections": [
            {
                "name": str(item["name"]),
                "class_id": int(item["cls"]),
                "confidence": float(item["conf"]),
                "bbox_xyxy": [int(value) for value in item["xyxy"]],
            }
            for item in detections
        ],
        "cloud_points": int(len(points)),
        "steered_points": int(np.count_nonzero(steering)),
        "elapsed_ms": round(float(elapsed_ms), 1),
        "grasp_count": int(len(candidates)),
        "grasps": [
            {
                "score": float(item.score),
                "width_m": float(item.width),
                "depth_m": float(item.depth),
                "translation_camera_m": np.asarray(item.translation).tolist(),
                "rotation_camera": np.asarray(item.rotation).tolist(),
            }
            for item in candidates
        ],
        "outputs": {
            "overlay": str(output_dir / "anygrasp_overlay.png"),
            "detections": str(output_dir / "yolo_detections.png"),
            "mask": str(output_dir / "selected_mask.png"),
            "cloud": str(output_dir / "steered_cloud.npz"),
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if arguments.vis:
        visualize_open3d(points, colors, steering, candidates)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-config", default=str(DEFAULT_VISUAL_CONFIG))
    parser.add_argument("--grasp-config", default=str(DEFAULT_GRASP_CONFIG))
    parser.add_argument("--weights", help="override YOLO segmentation weights")
    parser.add_argument("--static-frame", help="override Gemini static_frame directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label", default="can", help="YOLO class name/id")
    parser.add_argument("--instance", type=int, default=0, help="matching instance by confidence")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--mask-erosion-pixels", type=int, default=2)
    parser.add_argument("--minimum-depth", type=float, default=0.15)
    parser.add_argument("--maximum-depth", type=float, default=2.0)
    parser.add_argument("--maximum-points", type=int, default=40000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--overlay-grasps", type=int, default=1)
    parser.add_argument("--minimum-score", type=float, default=0.01)
    parser.add_argument("--minimum-width", type=float, default=0.02)
    parser.add_argument("--maximum-width", type=float, default=0.08)
    parser.add_argument("--approach-cone-deg", type=float, default=60.0)
    parser.add_argument("--dense-grasp", action="store_true")
    parser.add_argument("--no-collision-detection", action="store_true")
    parser.add_argument("--vis", action="store_true", help="open an interactive Open3D window")
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = run(arguments)
    except Exception as error:
        print(json.dumps({"status": "error", "reason": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
