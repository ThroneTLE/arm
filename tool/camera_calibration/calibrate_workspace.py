#!/usr/bin/env python3
"""Calibrate a fixed camera against ruler-measured workspace AprilTags."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from tool.camera_calibration.calib_common import (
    COORDINATE_CONVENTION,
    add_source_arguments,
    camera_matrix_from_yaml,
    detect_apriltags,
    draw_tag_detections,
    draw_tag_coordinate_axes,
    draw_workspace_coordinate_axes,
    layout_origins_mm,
    load_layout,
    open_frame_source,
    pixel_to_workspace_plane,
    tag_world_corners,
    transform_from_pose,
)


ROOT = Path(__file__).resolve().parent


def calibration_correspondences(layout: dict, median_corners):
    tag_size_m = float(layout["tag_size_mm"]) / 1000.0
    object_points = []
    image_points = []
    point_tag_ids = []
    for raw_id, entry in layout["calibration_tags"].items():
        tag_id = int(raw_id)
        origin_m = np.asarray(entry["origin_mm"], dtype=np.float64) / 1000.0
        corners_m = tag_world_corners(origin_m, float(entry.get("yaw_deg", 0.0)), tag_size_m)
        object_points.extend(corners_m)
        image_points.extend(median_corners[tag_id])
        point_tag_ids.extend([tag_id] * 4)
    return (
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        np.asarray(point_tag_ids, dtype=np.int32),
    )


def solve_workspace_pose(object_points, image_points, camera_matrix, distortion):
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed")
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points, image_points, camera_matrix, distortion, rvec, tvec
        )
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    camera_from_workspace, workspace_from_camera = transform_from_pose(rvec, tvec)
    return camera_from_workspace, workspace_from_camera, errors


def matrix_list(matrix):
    return np.asarray(matrix, dtype=np.float64).tolist()


def build_workspace_output(
    layout,
    camera_from_workspace,
    workspace_from_camera,
    errors,
    point_tag_ids,
    frames_used,
    intrinsics_path,
    layout_path,
):
    required_ids = sorted(int(tag_id) for tag_id in layout["calibration_tags"])
    rms = float(np.sqrt(np.mean(errors ** 2)))
    max_error = float(errors.max())
    per_tag_error = {
        int(tag_id): float(errors[point_tag_ids == tag_id].mean()) for tag_id in required_ids
    }
    return {
        "schema_version": 2,
        "coordinate_convention": dict(COORDINATE_CONVENTION),
        "workspace_frame": layout.get("workspace_frame", "ruler_workspace"),
        "camera_frame": layout.get("camera_frame", "camera_color_optical_frame"),
        "units": "meters",
        "camera_from_workspace": {
            "description": "p_camera = R * p_workspace + t",
            "rotation": matrix_list(camera_from_workspace[:3, :3]),
            "translation": camera_from_workspace[:3, 3].tolist(),
            "matrix": matrix_list(camera_from_workspace),
        },
        "workspace_from_camera": {
            "description": "p_workspace = R * p_camera + t",
            "rotation": matrix_list(workspace_from_camera[:3, :3]),
            "translation": workspace_from_camera[:3, 3].tolist(),
            "matrix": matrix_list(workspace_from_camera),
        },
        "quality": {
            "rms_reprojection_error_px": rms,
            "max_reprojection_error_px": max_error,
            "mean_error_by_tag_px": per_tag_error,
            "frames_used": int(frames_used),
            "calibration_tag_ids": required_ids,
        },
        "source": {
            "intrinsics": str(Path(intrinsics_path).expanduser().resolve()),
            "layout": str(Path(layout_path).expanduser().resolve()),
            "tag_size_mm": float(layout["tag_size_mm"]),
            "workspace_plane_z_mm": float(layout.get("workspace_plane_z_mm", 0.0)),
        },
    }


def save_workspace_output(path, output_data):
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        yaml.safe_dump(output_data, handle, sort_keys=False, allow_unicode=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument(
        "--intrinsics", default=ROOT / "output" / "astra_pro_rgb_1280x720.yaml"
    )
    parser.add_argument("--layout", default=ROOT / "config" / "tag_layout.yaml")
    parser.add_argument(
        "--output", default=ROOT / "output" / "workspace_extrinsics_1280x720.yaml"
    )
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--max-rms-px", type=float, default=2.0)
    args = parser.parse_args()

    camera_matrix, distortion, expected_size, _ = camera_matrix_from_yaml(args.intrinsics)
    layout = load_layout(args.layout)
    required_ids = sorted(int(tag_id) for tag_id in layout["calibration_tags"])
    configured_origins = layout_origins_mm(layout)
    samples = {tag_id: [] for tag_id in required_ids}
    valid_frames = 0
    capturing = False
    last_frame = None
    source = open_frame_source(args, "astra_pro_workspace_calibration")
    print("Place IDs {} flat and aligned. SPACE: start | R: reset | Q: quit".format(required_ids))
    try:
        while valid_frames < args.samples:
            frame = source.read()
            last_frame = frame.copy()
            if (frame.shape[1], frame.shape[0]) != expected_size:
                raise RuntimeError(
                    "image size {}x{} does not match intrinsics {}x{}".format(
                        frame.shape[1], frame.shape[0], expected_size[0], expected_size[1]
                    )
                )
            detections = detect_apriltags(frame)
            visible = [tag_id for tag_id in required_ids if tag_id in detections]
            all_visible = len(visible) == len(required_ids)
            if capturing and all_visible:
                for tag_id in required_ids:
                    samples[tag_id].append(detections[tag_id].copy())
                valid_frames += 1
            preview = frame.copy()
            draw_tag_detections(preview, detections)
            draw_tag_coordinate_axes(
                preview,
                detections,
                configured_origins,
                camera_matrix,
                distortion,
                float(layout["tag_size_mm"]),
            )
            text = "samples: {}/{} | visible: {}".format(valid_frames, args.samples, visible)
            color = (0, 255, 0) if all_visible else (0, 0, 255)
            cv2.putText(preview, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
            cv2.imshow("Workspace calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 1
            if key == ord("r"):
                samples = {tag_id: [] for tag_id in required_ids}
                valid_frames = 0
                capturing = False
            if key == 32:
                capturing = True
                print("Capture started")
    finally:
        source.close()
        cv2.destroyAllWindows()

    median_corners = {
        tag_id: np.median(np.asarray(tag_samples), axis=0) for tag_id, tag_samples in samples.items()
    }
    object_points, image_points, point_tag_ids = calibration_correspondences(layout, median_corners)
    camera_from_workspace, workspace_from_camera, errors = solve_workspace_pose(
        object_points, image_points, camera_matrix, distortion
    )
    rms = float(np.sqrt(np.mean(errors ** 2)))
    max_error = float(errors.max())
    if rms > args.max_rms_px:
        print("RMS {:.3f} px exceeds limit {:.3f} px".format(rms, args.max_rms_px))
        print("Check printed size, top-left origin coordinates, tag yaw, and paper flatness")
        return 2

    output_data = build_workspace_output(
        layout,
        camera_from_workspace,
        workspace_from_camera,
        errors,
        point_tag_ids,
        valid_frames,
        args.intrinsics,
        args.layout,
    )
    destination = save_workspace_output(args.output, output_data)

    debug = last_frame.copy()
    draw_tag_detections(debug, median_corners)
    draw_tag_coordinate_axes(
        debug,
        median_corners,
        configured_origins,
        camera_matrix,
        distortion,
        float(layout["tag_size_mm"]),
    )
    draw_workspace_coordinate_axes(
        debug, camera_from_workspace, camera_matrix, distortion, length_m=0.1
    )
    cv2.imwrite(str(destination.with_suffix(".png")), debug)

    plane_z_m = float(layout.get("workspace_plane_z_mm", 0.0)) / 1000.0
    print("RMS reprojection error: {:.4f} px (max {:.4f} px)".format(rms, max_error))
    print("Camera origin in workspace [mm]: {}".format(
        np.round(workspace_from_camera[:3, 3] * 1000.0, 3).tolist()
    ))
    for tag_id in required_ids:
        estimated = pixel_to_workspace_plane(
            median_corners[tag_id][0],
            camera_matrix,
            distortion,
            camera_from_workspace,
            plane_z_m,
        )
        expected = np.asarray(layout["calibration_tags"][tag_id]["origin_mm"], dtype=float)
        print("Tag {} top-left fit check [mm]: estimate={} expected={}".format(
            tag_id, np.round(estimated * 1000.0, 2).tolist(), expected.tolist()
        ))
    print("Saved: {}".format(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
