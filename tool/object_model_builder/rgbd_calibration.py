#!/usr/bin/env python3
"""RGB/IR stereo calibration used to derive T_color_depth."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from tool.camera_calibration.calib_common import charuco_board

from .rgbd_geometry import CameraIntrinsics, RgbdCalibration, _as_transform


@dataclass
class StereoCalibrationResult:
    calibration: RgbdCalibration
    pairs_used: int
    common_corner_counts: Tuple[int, ...]


@dataclass(frozen=True)
class CalibrationTarget:
    """Physical target model used by both RGB and IR detectors."""

    target_type: str = "charuco"
    model: str = "ChArUco 5x7"
    pattern_columns: int = 0
    pattern_rows: int = 0
    square_size_m: float = 0.0

    @property
    def point_count(self) -> int:
        if self.target_type == "checkerboard":
            return int(self.pattern_columns * self.pattern_rows)
        return int(len(charuco_board().getChessboardCorners()))

    @property
    def display_name(self) -> str:
        if self.target_type == "checkerboard":
            return "棋盘格 {}×{}（{:.1f} mm）".format(
                self.pattern_columns, self.pattern_rows, self.square_size_m * 1000.0
            )
        return "ChArUco 5×7（36 mm）"

    def object_points(self) -> np.ndarray:
        if self.target_type == "checkerboard":
            return np.asarray(
                [
                    [column * self.square_size_m, row * self.square_size_m, 0.0]
                    for row in range(self.pattern_rows)
                    for column in range(self.pattern_columns)
                ],
                dtype=np.float32,
            )
        return np.asarray(charuco_board().getChessboardCorners(), dtype=np.float32)


def calibration_target_from_mapping(mapping=None) -> CalibrationTarget:
    """Parse the target section while keeping ChArUco as the legacy default."""
    data = dict(mapping or {})
    target_type = str(data.get("type", data.get("target_type", "charuco"))).strip().lower()
    if target_type in ("chessboard", "checker", "checker_board"):
        target_type = "checkerboard"
    if target_type not in ("charuco", "checkerboard"):
        raise ValueError("unsupported RGB-D calibration target type: {}".format(target_type))
    if target_type == "charuco":
        return CalibrationTarget(target_type="charuco", model=str(data.get("model", "ChArUco 5x7")))
    columns = int(data.get("pattern_columns", data.get("columns", 0)))
    rows = int(data.get("pattern_rows", data.get("rows", 0)))
    square_size_m = float(data.get("square_size_m", 0.0))
    if columns < 2 or rows < 2 or square_size_m <= 0.0:
        raise ValueError("checkerboard target needs pattern_columns, pattern_rows and square_size_m")
    return CalibrationTarget(
        target_type="checkerboard",
        model=str(data.get("model", "checkerboard")),
        pattern_columns=columns,
        pattern_rows=rows,
        square_size_m=square_size_m,
    )


def infrared_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert Astra Y10/mono16 data to contrast-stretched 8-bit grayscale."""
    infrared = np.asarray(image)
    if infrared.ndim == 3:
        infrared = cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY)
    if infrared.dtype == np.uint8:
        return infrared.copy()
    values = infrared.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    valid = values[finite]
    low, high = np.percentile(valid, (0.5, 99.5))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - float(low)) * (255.0 / float(high - low))
    normalized[~finite] = 0.0
    return np.clip(normalized, 0.0, 255.0).astype(np.uint8)


def detect_charuco(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = infrared_to_uint8(gray)
    board = charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or corners is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1).astype(np.int32)


def _as_gray_uint8(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = infrared_to_uint8(gray)
    return gray


def _canonicalize_checkerboard(corners: np.ndarray, target: CalibrationTarget) -> np.ndarray:
    """Use a deterministic image-space corner orientation for RGB/IR pairing."""
    grid = np.asarray(corners, dtype=np.float32).reshape(
        target.pattern_rows, target.pattern_columns, 2
    )
    candidates = (
        grid,
        grid[:, ::-1, :],
        grid[::-1, :, :],
        grid[::-1, ::-1, :],
    )

    def score(candidate):
        outer = np.asarray(
            [candidate[0, 0], candidate[0, -1], candidate[-1, 0], candidate[-1, -1]],
            dtype=np.float32,
        )
        first = outer[int(np.argmin(outer[:, 0] + outer[:, 1]))]
        right = candidate[0, -1]
        return (float(first[0] + first[1]), float(first[1]), float(first[0]), float(right[1]), float(right[0]))

    return min(candidates, key=score).reshape(-1, 2).copy()


def _checkerboard_detection_images(gray: np.ndarray):
    """Yield fast-to-robust IR variants while preserving pixel coordinates."""
    height, width = gray.shape[:2]
    scale = min(1.0, 720.0 / float(max(height, width)))
    if scale < 1.0:
        detection = cv2.resize(
            gray,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection = gray
    blurred = cv2.GaussianBlur(detection, (5, 5), 0.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yield clahe.apply(blurred), scale
    yield detection, scale
    yield blurred, scale


def _checkerboard_candidate_score(
    corners: np.ndarray, target: CalibrationTarget, image_shape
) -> float:
    """Reject false grids caused by the composite chart's interrupted center."""
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    height, width = int(image_shape[0]), int(image_shape[1])
    if (
        len(points) != target.point_count
        or np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] >= height)
    ):
        return float("inf")
    object_xy = target.object_points()[:, :2]
    homography, _ = cv2.findHomography(object_xy, points, method=0)
    if homography is None:
        return float("inf")
    projected = cv2.perspectiveTransform(
        object_xy.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=1))))
    grid = points.reshape(target.pattern_rows, target.pattern_columns, 2)
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2).reshape(-1)
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2).reshape(-1)
    spacing = np.concatenate((horizontal, vertical))
    median_spacing = float(np.median(spacing)) if spacing.size else 0.0
    if median_spacing < 3.0:
        return float("inf")
    return rms / median_spacing


def detect_calibration_target(
    image: np.ndarray, target: Optional[CalibrationTarget] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect a configured ChArUco or ordinary checkerboard target."""
    target = target or CalibrationTarget()
    if target.target_type == "charuco":
        return detect_charuco(image)
    gray = _as_gray_uint8(image)
    flags = int(getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0))
    flags |= int(getattr(cv2, "CALIB_CB_ACCURACY", 0))
    flags |= int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
    pattern = (int(target.pattern_columns), int(target.pattern_rows))
    detector = getattr(cv2, "findChessboardCornersSB", None)
    found = False
    corners = None
    best_corners = None
    best_score = float("inf")
    for detection_image, scale in _checkerboard_detection_images(gray):
        if detector is not None:
            found, corners = detector(detection_image, pattern, flags=flags)
        else:  # OpenCV versions before findChessboardCornersSB
            legacy_flags = int(getattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 0))
            legacy_flags |= int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
            found, corners = cv2.findChessboardCorners(
                detection_image, pattern, flags=legacy_flags
            )
        if found and corners is not None:
            if scale != 1.0:
                corners = np.asarray(corners, dtype=np.float32) / float(scale)
            score = _checkerboard_candidate_score(corners, target, gray.shape)
            if score < best_score:
                best_score = score
                best_corners = np.asarray(corners, dtype=np.float32).copy()
            if score <= 0.20:
                break
    if best_corners is None or best_score > 0.35:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    corners = best_corners
    refinement = cv2.GaussianBlur(gray, (5, 5), 0.0)
    corners = cv2.cornerSubPix(
        refinement,
        np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
    )
    normalized = _canonicalize_checkerboard(corners, target)
    identifiers = np.arange(normalized.shape[0], dtype=np.int32)
    return normalized, identifiers


def charuco_view_signature(
    corners: np.ndarray,
    ids: np.ndarray,
    image_shape,
):
    """Return normalized board bounds estimated from the detected inner corners."""
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    identifiers = np.asarray(ids, dtype=np.int32).reshape(-1)
    height, width = int(image_shape[0]), int(image_shape[1])
    board_points = np.asarray(charuco_board().getChessboardCorners(), dtype=np.float32)
    valid = (
        identifiers.size == points.shape[0]
        and identifiers.size >= 4
        and width > 0
        and height > 0
        and np.all((identifiers >= 0) & (identifiers < len(board_points)))
    )
    if not valid:
        return None
    homography, _ = cv2.findHomography(board_points[identifiers, :2], points, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        return None
    minimum = board_points[:, :2].min(axis=0)
    maximum = board_points[:, :2].max(axis=0)
    bounds = np.asarray(
        [
            [minimum[0], minimum[1]],
            [maximum[0], minimum[1]],
            [maximum[0], maximum[1]],
            [minimum[0], maximum[1]],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(bounds, homography).reshape(-1, 2)
    projected[:, 0] /= float(width)
    projected[:, 1] /= float(height)
    if not np.all(np.isfinite(projected)):
        return None
    return projected.astype(np.float32)


def calibration_view_signature(
    corners: np.ndarray,
    ids: np.ndarray,
    image_shape,
    target: Optional[CalibrationTarget] = None,
):
    """Return normalized full-target bounds for either supported target."""
    target = target or CalibrationTarget()
    if target.target_type == "charuco":
        return charuco_view_signature(corners, ids, image_shape)
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    identifiers = np.asarray(ids, dtype=np.int32).reshape(-1)
    board_points = target.object_points()
    height, width = int(image_shape[0]), int(image_shape[1])
    valid = (
        identifiers.size == points.shape[0]
        and identifiers.size >= 4
        and width > 0
        and height > 0
        and np.all((identifiers >= 0) & (identifiers < len(board_points)))
    )
    if not valid:
        return None
    homography, _ = cv2.findHomography(board_points[identifiers, :2], points, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        return None
    minimum = board_points[:, :2].min(axis=0)
    maximum = board_points[:, :2].max(axis=0)
    bounds = np.asarray(
        [[minimum[0], minimum[1]], [maximum[0], minimum[1]], [maximum[0], maximum[1]], [minimum[0], maximum[1]]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(bounds, homography).reshape(-1, 2)
    projected[:, 0] /= float(width)
    projected[:, 1] /= float(height)
    if not np.all(np.isfinite(projected)):
        return None
    return projected.astype(np.float32)


def charuco_view_change(first, second) -> float:
    if first is None or second is None:
        return float("inf")
    delta = np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)
    if delta.shape != (4, 2):
        return float("inf")
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def calibration_view_change(first, second) -> float:
    """Compare normalized target footprints for automatic de-duplication."""
    return charuco_view_change(first, second)


def _common_observations(
    color_image: np.ndarray,
    ir_image: np.ndarray,
    target: Optional[CalibrationTarget] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = target or CalibrationTarget()
    color_corners, color_ids = detect_calibration_target(color_image, target)
    ir_corners, ir_ids = detect_calibration_target(ir_image, target)
    common_ids = sorted(set(color_ids.tolist()).intersection(ir_ids.tolist()))
    if not common_ids:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
    board_points = target.object_points()
    color_lookup = {int(tag_id): corner for tag_id, corner in zip(color_ids, color_corners)}
    ir_lookup = {int(tag_id): corner for tag_id, corner in zip(ir_ids, ir_corners)}
    object_points = np.asarray([board_points[tag_id] for tag_id in common_ids], dtype=np.float32)
    color_points = np.asarray([color_lookup[tag_id] for tag_id in common_ids], dtype=np.float32)
    ir_points = np.asarray([ir_lookup[tag_id] for tag_id in common_ids], dtype=np.float32)
    return object_points, color_points, ir_points


def calibrate_color_from_depth(
    image_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    color_intrinsics: CameraIntrinsics,
    minimum_common_corners: int = 10,
    minimum_pairs: int = 10,
    target: Optional[CalibrationTarget] = None,
) -> StereoCalibrationResult:
    """Calibrate IR as camera 1 and color as camera 2; output T_color_depth."""
    target = target or CalibrationTarget()
    object_points: List[np.ndarray] = []
    ir_points: List[np.ndarray] = []
    color_points: List[np.ndarray] = []
    counts = []
    ir_size = None
    for color_image, ir_image in image_pairs:
        color = np.asarray(color_image)
        infrared = np.asarray(ir_image)
        if color.shape[:2] != (color_intrinsics.height, color_intrinsics.width):
            raise ValueError("color calibration image dimensions do not match intrinsics")
        current_ir_size = (infrared.shape[1], infrared.shape[0])
        if ir_size is None:
            ir_size = current_ir_size
        elif ir_size != current_ir_size:
            raise ValueError("all IR calibration images must use one resolution")
        objects, colors, infrared_points = _common_observations(color, infrared, target)
        if len(objects) < int(minimum_common_corners):
            continue
        object_points.append(objects)
        color_points.append(colors)
        ir_points.append(infrared_points)
        counts.append(len(objects))
    if len(object_points) < int(minimum_pairs):
        raise ValueError(
            "only {} valid RGB/IR pairs; at least {} are required".format(
                len(object_points), minimum_pairs
            )
        )
    calibration_flags = 0
    ir_rms, ir_matrix, ir_distortion, _, _ = cv2.calibrateCamera(
        object_points,
        ir_points,
        ir_size,
        None,
        None,
        flags=calibration_flags,
    )
    if not np.isfinite(ir_rms):
        raise RuntimeError("IR intrinsic calibration failed")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    stereo_result = cv2.stereoCalibrate(
        object_points,
        ir_points,
        color_points,
        ir_matrix,
        ir_distortion,
        color_intrinsics.matrix.copy(),
        color_intrinsics.distortion.copy(),
        ir_size,
        flags=stereo_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8),
    )
    stereo_rms, _, _, _, _, rotation, translation, _, _ = stereo_result
    color_from_depth = np.eye(4, dtype=np.float64)
    color_from_depth[:3, :3] = rotation
    color_from_depth[:3, 3] = np.asarray(translation).reshape(3)
    depth_intrinsics = CameraIntrinsics(
        width=ir_size[0],
        height=ir_size[1],
        matrix=ir_matrix,
        distortion=ir_distortion,
    )
    calibration = RgbdCalibration(
        color=color_intrinsics,
        depth=depth_intrinsics,
        color_from_depth=_as_transform(color_from_depth),
        valid=True,
        source="RGB/IR {} stereo calibration".format(target.display_name),
        rms_reprojection_error_px=float(stereo_rms),
    )
    return StereoCalibrationResult(calibration, len(object_points), tuple(counts))


def load_image_pairs(directory: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    root = Path(directory).expanduser().resolve()
    color_dir = root / "color"
    ir_dir = root / "ir"
    pairs = []
    for color_path in sorted(color_dir.glob("*.png")):
        ir_path = ir_dir / color_path.name
        if not ir_path.is_file():
            continue
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        infrared = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
        if color is not None and infrared is not None:
            pairs.append((color, infrared))
    return pairs


def update_runtime_calibration(
    runtime_path: str,
    calibration: RgbdCalibration,
    backup_suffix: str,
) -> Path:
    """Atomically update only depth intrinsics and T_color_depth in central YAML."""
    path = Path(runtime_path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    backup = path.with_name(path.name + ".backup_" + str(backup_suffix))
    with open(backup, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    depth = data.setdefault("camera", {}).setdefault("depth", {})
    depth.update(calibration.depth.to_mapping())
    depth["valid"] = True
    depth["intrinsics_valid"] = True
    depth["aligned_to_color"] = False
    transform = data.setdefault("transforms", {}).setdefault("color_from_depth", {})
    transform.update(
        {
            "valid": True,
            "description": "p_color = T_color_depth * p_depth",
            "matrix": calibration.color_from_depth.tolist(),
        }
    )
    data.setdefault("quality", {})["rgbd_stereo_rms_px"] = (
        calibration.rms_reprojection_error_px
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    temporary.replace(path)
    return backup
