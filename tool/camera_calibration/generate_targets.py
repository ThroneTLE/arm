#!/usr/bin/env python3
"""Generate exact-size A4 ChArUco and AprilTag calibration targets."""

import argparse
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from tool.camera_calibration.calib_common import (
    CHARUCO_MARKER_LENGTH_M,
    CHARUCO_SQUARE_LENGTH_M,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    PRINTED_TAG_SIZE_MM,
    april_dictionary,
    charuco_board,
)


DPI = 600
TAG_IDS = (100, 101, 102, 103)
TAG_SIZE_MM = PRINTED_TAG_SIZE_MM
ROOT = Path(__file__).resolve().parent


def raster_pixels(length_mm: float) -> int:
    return int(round(length_mm * DPI / 25.4))


def generate_rasters(output_dir: Path):
    board_width_mm = CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_M * 1000.0
    board_height_mm = CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_M * 1000.0
    board_image = charuco_board().generateImage(
        (raster_pixels(board_width_mm), raster_pixels(board_height_mm)),
        marginSize=0,
        borderBits=1,
    )
    board_path = output_dir / "charuco_5x7_large_preview_600dpi.png"
    cv2.imwrite(str(board_path), board_image)

    tag_paths = {}
    for tag_id in TAG_IDS:
        tag_image = cv2.aruco.generateImageMarker(
            april_dictionary(), tag_id, raster_pixels(TAG_SIZE_MM), borderBits=1
        )
        tag_path = output_dir / "apriltag36h11_id{}_600dpi.png".format(tag_id)
        cv2.imwrite(str(tag_path), tag_image)
        tag_paths[tag_id] = tag_path
    return board_path, tag_paths


def draw_header(pdf, title: str, subtitle: str) -> None:
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawCentredString(A4[0] / 2.0, A4[1] - 11 * mm, title)
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(A4[0] / 2.0, A4[1] - 16 * mm, subtitle)


def draw_scale_check(pdf, y_mm: float = 14.0) -> None:
    start_x = 55 * mm
    end_x = start_x + 100 * mm
    y = y_mm * mm
    pdf.setLineWidth(0.5)
    pdf.line(start_x, y, end_x, y)
    pdf.line(start_x, y - 2 * mm, start_x, y + 2 * mm)
    pdf.line(end_x, y - 2 * mm, end_x, y + 2 * mm)
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString((start_x + end_x) / 2.0, y + 2.5 * mm, "100.0 mm SCALE CHECK")


def draw_charuco_page(pdf, board_path: Path) -> None:
    draw_header(
        pdf,
        "Astra Pro RGB Intrinsic Calibration - ChArUco",
        "PRINT AT 100% / ACTUAL SIZE - disable Fit, Shrink, and Scale to page",
    )
    board_width_mm = CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_M * 1000.0
    board_height_mm = CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_M * 1000.0
    x = (A4[0] - board_width_mm * mm) / 2.0
    y = (A4[1] - board_height_mm * mm) / 2.0
    pdf.drawImage(
        str(board_path),
        x,
        y,
        width=board_width_mm * mm,
        height=board_height_mm * mm,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.setFont("Helvetica", 7)
    pdf.drawString(
        8 * mm,
        20 * mm,
        "5 x 7 squares | square 36.0 mm | marker 27.0 mm | AprilTag 36h11",
    )
    draw_scale_check(pdf)


def draw_tag_page(pdf, tag_paths) -> None:
    draw_header(
        pdf,
        "Workspace Calibration Tags - AprilTag 36h11",
        "IDs 100-102: calibration | ID 103: validation only | black edge: 70.0 mm",
    )
    tile_width_mm = 90.0
    tile_height_mm = 100.0
    tile_positions_mm = ((10, 185), (110, 185), (10, 75), (110, 75))
    for tag_id, (x_mm, y_mm) in zip(TAG_IDS, tile_positions_mm):
        x = x_mm * mm
        y = y_mm * mm
        pdf.setDash(3, 2)
        pdf.setLineWidth(0.4)
        pdf.rect(x, y, tile_width_mm * mm, tile_height_mm * mm)
        pdf.setDash()
        marker_x = x + 10 * mm
        marker_y = y + 15 * mm
        pdf.drawImage(
            str(tag_paths[tag_id]),
            marker_x,
            marker_y,
            width=TAG_SIZE_MM * mm,
            height=TAG_SIZE_MM * mm,
            preserveAspectRatio=True,
            mask="auto",
        )
        pdf.setFont("Helvetica-Bold", 7)
        pdf.drawCentredString(
            x + tile_width_mm * mm / 2.0,
            y + 89 * mm,
            "ORIGIN: TOP LEFT | +X RIGHT | +Y DOWN",
        )
        role = "VALIDATION ONLY" if tag_id == 103 else "CALIBRATION"
        pdf.drawCentredString(
            x + tile_width_mm * mm / 2.0,
            y + 5 * mm,
            "ID {} | {}".format(tag_id, role),
        )
    draw_scale_check(pdf, y_mm=38.0)
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(A4[0] / 2.0, 29 * mm, "Cut on dashed lines; keep the white quiet zone intact")


def write_pdf(path: Path, pages) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
    for index, page in enumerate(pages):
        page(pdf)
        if index != len(pages) - 1:
            pdf.showPage()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=ROOT / "targets")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    board_path, tag_paths = generate_rasters(output_dir)
    charuco_page = lambda pdf: draw_charuco_page(pdf, board_path)
    tag_page = lambda pdf: draw_tag_page(pdf, tag_paths)
    write_pdf(output_dir / "charuco_intrinsics_A4.pdf", [charuco_page])
    write_pdf(output_dir / "apriltags_100_103_A4.pdf", [tag_page])
    write_pdf(output_dir / "calibration_targets_A4_2pages.pdf", [charuco_page, tag_page])
    print("Generated targets in {}".format(output_dir))


if __name__ == "__main__":
    main()
