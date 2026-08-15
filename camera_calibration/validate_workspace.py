#!/usr/bin/env python3
"""Validate ID 103 while localizing a moving camera from fixed IDs 100-102."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from calib_common import (
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
    require_coordinate_convention,
)
from hybrid_localization import TagMapPoseEstimator


def summarize_estimates(estimates_mm, expected_mm, tag_id):
    estimates = np.asarray(estimates_mm, dtype=np.float64)
    expected = np.asarray(expected_mm, dtype=np.float64)
    if estimates.ndim != 2 or estimates.shape[0] == 0 or estimates.shape[1] != 3:
        raise ValueError("validation estimates must have shape Nx3")
    median = np.median(estimates, axis=0)
    mean = np.mean(estimates, axis=0)
    std = np.std(estimates, axis=0)
    error = median - expected
    return {
        "schema_version": 2,
        "coordinate_convention": dict(COORDINATE_CONVENTION),
        "validation_tag_id": int(tag_id),
        "samples": int(len(estimates)),
        "expected_origin_mm": expected.tolist(),
        "median_estimate_mm": median.tolist(),
        "mean_estimate_mm": mean.tolist(),
        "standard_deviation_mm": std.tolist(),
        "median_error_xyz_mm": error.tolist(),
        "median_error_xy_mm": float(np.linalg.norm(error[:2])),
        "median_error_3d_mm": float(np.linalg.norm(error)),
    }


def estimate_tag_origin_mm(
    corners,
    camera_matrix,
    distortion,
    camera_from_workspace,
    plane_z_m,
):
    """Intersect the detected TL corner ray with the configured workspace plane."""
    return estimate_tag_corners_mm(
        corners,
        camera_matrix,
        distortion,
        camera_from_workspace,
        plane_z_m,
    )[0]


def estimate_tag_corners_mm(
    corners,
    camera_matrix,
    distortion,
    camera_from_workspace,
    plane_z_m,
):
    """Project detected TL, TR, BR, BL pixels onto the workspace plane."""
    pixels = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return np.asarray(
        [
            pixel_to_workspace_plane(
                pixel,
                camera_matrix,
                distortion,
                camera_from_workspace,
                plane_z_m,
            )
            * 1000.0
            for pixel in pixels
        ],
        dtype=np.float64,
    )


def tag_yaw_deg_from_corners(corners_mm):
    """Return in-plane yaw, where positive rotation maps +X toward +Y."""
    points = np.asarray(corners_mm, dtype=np.float64).reshape(4, 3)
    x_edge = points[1, :2] - points[0, :2]
    return float(np.degrees(np.arctan2(x_edge[1], x_edge[0])))


def estimate_dynamic_validation(
    detections,
    pose_estimator,
    validation_tag_id,
    camera_matrix,
    distortion,
    plane_z_m,
):
    """Localize the camera from reference Tags, then estimate validation TL."""
    visual_pose = pose_estimator.estimate(detections, camera_matrix, distortion)
    if not visual_pose.valid or validation_tag_id not in detections:
        return visual_pose, None
    estimate_mm = estimate_tag_origin_mm(
        detections[validation_tag_id],
        camera_matrix,
        distortion,
        visual_pose.camera_from_workspace,
        plane_z_m,
    )
    return visual_pose, estimate_mm


def save_validation_summary(path, summary):
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--intrinsics", default="output/astra_pro_rgb_1280x720.yaml")
    parser.add_argument("--extrinsics", default="output/workspace_extrinsics_1280x720.yaml")
    parser.add_argument("--layout", default="config/tag_layout.yaml")
    parser.add_argument("--output", default="output/validation_tag_103_1280x720.yaml")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    camera_matrix, distortion, expected_size, _ = camera_matrix_from_yaml(args.intrinsics)
    layout = load_layout(args.layout)
    with open(args.extrinsics, "r", encoding="utf-8") as handle:
        extrinsics = yaml.safe_load(handle)
    try:
        require_coordinate_convention(extrinsics, "workspace extrinsics")
    except ValueError as error:
        print(str(error))
        return 2
    validation = layout["validation_tag"]
    tag_id = int(validation["id"])
    expected_mm = np.asarray(validation["origin_mm"], dtype=np.float64)
    plane_z_m = float(expected_mm[2]) / 1000.0
    reference_ids = tuple(sorted(int(value) for value in layout["calibration_tags"]))
    pose_estimator = TagMapPoseEstimator(
        layout,
        minimum_tags=len(reference_ids),
        max_rms_reprojection_error_px=2.0,
    )
    estimates_mm = []
    reference_rms_px = []
    source = open_frame_source(args, "astra_pro_workspace_validation")
    try:
        while len(estimates_mm) < args.samples:
            frame = source.read()
            if (frame.shape[1], frame.shape[0]) != expected_size:
                raise RuntimeError("image resolution does not match intrinsics")
            detections = detect_apriltags(frame)
            display_origins = layout_origins_mm(layout, include_validation=False)
            visual_pose, estimate_mm = estimate_dynamic_validation(
                detections,
                pose_estimator,
                tag_id,
                camera_matrix,
                distortion,
                plane_z_m,
            )
            if estimate_mm is not None:
                estimates_mm.append(estimate_mm)
                reference_rms_px.append(visual_pose.rms_reprojection_error_px)
                display_origins[tag_id] = estimate_mm
                error_xy = np.linalg.norm(estimate_mm[:2] - expected_mm[:2])
                cv2.putText(
                    frame,
                    "XYZ mm: {:.1f}, {:.1f}, {:.1f} | XY err: {:.1f} | Ref RMS: {:.2f}px".format(
                        estimate_mm[0], estimate_mm[1], estimate_mm[2], error_xy,
                        visual_pose.rms_reprojection_error_px,
                    ),
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.53,
                    (0, 255, 0),
                    2,
                )
                draw_workspace_coordinate_axes(
                    frame,
                    visual_pose.camera_from_workspace,
                    camera_matrix,
                    distortion,
                    length_m=0.1,
                )
            else:
                visible_references = [
                    tag_value for tag_value in reference_ids if tag_value in detections
                ]
                cv2.putText(
                    frame,
                    "Waiting refs {}/{} and ID {}".format(
                        len(visible_references), len(reference_ids), tag_id
                    ),
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.53,
                    (0, 0, 255),
                    2,
                )
            draw_tag_detections(frame, detections)
            draw_tag_coordinate_axes(
                frame,
                detections,
                display_origins,
                camera_matrix,
                distortion,
                float(layout["tag_size_mm"]),
            )
            if not args.headless:
                cv2.imshow("Workspace validation - tag {}".format(tag_id), frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 1
    finally:
        source.close()
        cv2.destroyAllWindows()

    summary = summarize_estimates(estimates_mm, expected_mm, tag_id)
    summary["localization_mode"] = "dynamic_camera_pose_from_reference_tags"
    summary["reference_tag_ids"] = list(reference_ids)
    summary["reference_pose_rms_mean_px"] = float(np.mean(reference_rms_px))
    summary["reference_pose_rms_max_px"] = float(np.max(reference_rms_px))
    destination = save_validation_summary(args.output, summary)
    print(yaml.safe_dump(summary, sort_keys=False))
    print("Saved: {}".format(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
