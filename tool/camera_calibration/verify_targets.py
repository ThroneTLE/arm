#!/usr/bin/env python3
"""Render and verify the physical dimensions and IDs in the target PDF."""

import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tool.camera_calibration.calib_common import (
    CHARUCO_SQUARE_LENGTH_M,
    april_dictionary,
    charuco_board,
)


DPI = 600
EXPECTED_A4_POINTS = (595.276, 841.89)
EXPECTED_TAG_MM = 70.0
EXPECTED_SQUARE_MM = CHARUCO_SQUARE_LENGTH_M * 1000.0


def pdf_page_size(pdf: Path):
    result = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, text=True, capture_output=True
    )
    match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", result.stdout)
    if not match:
        raise RuntimeError("could not read PDF page size")
    return float(match.group(1)), float(match.group(2))


def adjacent_charuco_distances(corners, ids, board):
    pixels = {int(tag_id): point.reshape(2) for point, tag_id in zip(corners, ids.reshape(-1))}
    objects = np.asarray(board.getChessboardCorners())
    distances = []
    for first in range(len(objects)):
        for second in range(first + 1, len(objects)):
            object_distance = float(np.linalg.norm(objects[first] - objects[second]))
            if (
                abs(object_distance - CHARUCO_SQUARE_LENGTH_M) < 1e-5
                and first in pixels
                and second in pixels
            ):
                distances.append(float(np.linalg.norm(pixels[first] - pixels[second])))
    return distances


def main() -> int:
    root = Path(__file__).resolve().parent
    pdf = root / "targets" / "calibration_targets_A4_2pages.pdf"
    page_points = pdf_page_size(pdf)
    if not np.allclose(page_points, EXPECTED_A4_POINTS, atol=0.02):
        raise RuntimeError("PDF is not ISO A4: {}".format(page_points))

    with tempfile.TemporaryDirectory(prefix="astra_target_check_") as temporary:
        prefix = Path(temporary) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "1",
                "-l",
                "2",
                "-r",
                str(DPI),
                "-png",
                str(pdf),
                str(prefix),
            ],
            check=True,
        )
        page1 = cv2.imread(str(prefix) + "-1.png", cv2.IMREAD_GRAYSCALE)
        page2 = cv2.imread(str(prefix) + "-2.png", cv2.IMREAD_GRAYSCALE)
        if page1 is None or page2 is None:
            raise RuntimeError("failed to render both PDF pages")

        expected_pixels = (round(210.0 / 25.4 * DPI), round(297.0 / 25.4 * DPI))
        actual_pixels = (page1.shape[1], page1.shape[0])
        if actual_pixels != expected_pixels:
            raise RuntimeError("unexpected rendered A4 size: {}".format(actual_pixels))

        board = charuco_board()
        corners, ids, marker_corners, marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(page1)
        if ids is None or len(ids) < 20:
            raise RuntimeError("ChArUco page detection failed")
        square_pixels = adjacent_charuco_distances(corners, ids, board)
        square_mm = float(np.median(square_pixels)) / DPI * 25.4
        if abs(square_mm - EXPECTED_SQUARE_MM) > 0.05:
            raise RuntimeError("ChArUco square size mismatch: {:.4f} mm".format(square_mm))

        detector = cv2.aruco.ArucoDetector(
            april_dictionary(), cv2.aruco.DetectorParameters()
        )
        tag_corners, tag_ids, _ = detector.detectMarkers(page2)
        detected = {}
        if tag_ids is not None:
            for corners_for_tag, tag_id in zip(tag_corners, tag_ids.reshape(-1)):
                points = corners_for_tag.reshape(4, 2)
                edges = [
                    float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
                    for index in range(4)
                ]
                detected[int(tag_id)] = float(np.mean(edges)) / DPI * 25.4
        if set(detected) != {100, 101, 102, 103}:
            raise RuntimeError("unexpected AprilTag IDs: {}".format(sorted(detected)))
        for tag_id, measured_mm in detected.items():
            if abs(measured_mm - EXPECTED_TAG_MM) > 0.05:
                raise RuntimeError(
                    "Tag {} size mismatch: {:.4f} mm".format(tag_id, measured_mm)
                )

    print("PASS: ISO A4 page {:.3f} x {:.3f} pt".format(*page_points))
    print("PASS: ChArUco square {:.4f} mm".format(square_mm))
    for tag_id in sorted(detected):
        print("PASS: AprilTag {} black edge {:.4f} mm".format(tag_id, detected[tag_id]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
