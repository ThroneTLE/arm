#!/usr/bin/env python3
"""Interactive ChArUco calibration for the Astra Pro UVC color camera."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from calib_common import (
    CHARUCO_MARKER_LENGTH_M,
    CHARUCO_SQUARE_LENGTH_M,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    add_source_arguments,
    charuco_board,
    open_frame_source,
    save_camera_yaml,
)


def calibrate(object_points, image_points, image_size):
    result = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    rms, camera_matrix, distortion, rvecs, tvecs, std_intrinsics, _, per_view = result
    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "std_intrinsics": std_intrinsics,
        "per_view_errors": per_view.reshape(-1),
    }


def save_report(path: Path, result, view_count: int, coverage) -> None:
    report = {
        "method": "ChArUco with cv2.calibrateCameraExtended",
        "views": view_count,
        "rms_reprojection_error_px": result["rms"],
        "per_view_errors_px": result["per_view_errors"].tolist(),
        "image_coverage": {"x_fraction": coverage[0], "y_fraction": coverage[1]},
        "board": {
            "squares_x": CHARUCO_SQUARES_X,
            "squares_y": CHARUCO_SQUARES_Y,
            "square_length_mm": CHARUCO_SQUARE_LENGTH_M * 1000.0,
            "marker_length_mm": CHARUCO_MARKER_LENGTH_M * 1000.0,
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--output", default="output/astra_pro_rgb_1280x720.yaml")
    parser.add_argument("--camera-name", default="astra_pro_rgb")
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--min-views", type=int, default=20)
    parser.add_argument("--max-rms-px", type=float, default=0.8)
    args = parser.parse_args()

    board = charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    object_points = []
    image_points = []
    all_pixels = []
    image_size = None
    source = open_frame_source(args, "astra_pro_intrinsic_calibration")

    print("SPACE: capture view | C: calibrate and save | R: reset | Q: quit")
    try:
        while True:
            frame = source.read()
            height, width = frame.shape[:2]
            current_size = (width, height)
            if image_size is None:
                image_size = current_size
            elif current_size != image_size:
                raise RuntimeError("image resolution changed during calibration")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            valid_count = 0 if charuco_ids is None else len(charuco_ids)
            preview = frame.copy()
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)
            if charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids)
            cv2.putText(
                preview,
                "views: {} | corners: {}".format(len(object_points), valid_count),
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if valid_count >= args.min_corners else (0, 0, 255),
                2,
            )
            cv2.imshow("Astra Pro intrinsic calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 1
            if key == ord("r"):
                object_points.clear()
                image_points.clear()
                all_pixels.clear()
                print("Captured views cleared")
            if key == 32:
                if valid_count < args.min_corners:
                    print("Rejected: only {} ChArUco corners".format(valid_count))
                    continue
                ids = charuco_ids.reshape(-1).astype(np.int32)
                object_points.append(board_corners[ids].reshape(-1, 1, 3).copy())
                pixels = charuco_corners.astype(np.float32).reshape(-1, 1, 2).copy()
                image_points.append(pixels)
                all_pixels.append(pixels.reshape(-1, 2))
                print("Captured view {} with {} corners".format(len(object_points), valid_count))
            if key == ord("c"):
                if len(object_points) < args.min_views:
                    print("Need at least {} views".format(args.min_views))
                    continue
                break
    finally:
        source.close()
        cv2.destroyAllWindows()

    result = calibrate(object_points, image_points, image_size)
    pixels = np.concatenate(all_pixels, axis=0)
    coverage = (
        float((pixels[:, 0].max() - pixels[:, 0].min()) / image_size[0]),
        float((pixels[:, 1].max() - pixels[:, 1].min()) / image_size[1]),
    )
    output = Path(args.output).expanduser().resolve()
    save_camera_yaml(
        str(output), result["camera_matrix"], result["distortion"], image_size, args.camera_name
    )
    save_report(output.with_name(output.stem + "_report.yaml"), result, len(object_points), coverage)
    print("RMS reprojection error: {:.4f} px".format(result["rms"]))
    print("Image coverage: x={:.1%}, y={:.1%}".format(*coverage))
    print("Saved: {}".format(output))
    if result["rms"] > args.max_rms_px:
        print("WARNING: RMS exceeds {:.3f} px; recapture more diverse views".format(args.max_rms_px))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
