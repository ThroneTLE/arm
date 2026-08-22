"""Zhang-style checkerboard target used by eye-in-hand calibration."""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import as_transform


CHECKERBOARD_TARGET = "checkerboard"


@dataclass(frozen=True)
class CheckerboardObservation:
    valid: bool
    camera_from_board: object = None
    corners: object = None
    rms_reprojection_error_px: float = None
    reason: str = ""


class CheckerboardTarget:
    """Detect a metric checkerboard and estimate ``T_camera_board``.

    OpenCV's pattern size is the number of *inner corners*, not the number of
    black/white squares.  The physical outer-board dimensions are deliberately
    separate from the printed-grid square count: common calibration boards have
    a white border, so a 300 x 225 mm board with 25 mm cells can, for example,
    contain a 12 x 9 printed grid (11 x 8 inner corners).
    """

    def __init__(self, settings):
        settings = dict(settings or {})
        self.board_width_mm = float(settings.get("board_width_mm", 300.0))
        self.board_height_mm = float(settings.get("board_height_mm", 225.0))
        self.square_size_mm = float(settings.get("square_size_mm", 25.0))
        self.maximum_rms_px = float(settings.get("maximum_rms_px", 1.5))
        self.prefer_sb = bool(settings.get("prefer_find_chessboard_sb", True))
        if min(
            self.board_width_mm, self.board_height_mm,
            self.square_size_mm, self.maximum_rms_px,
        ) <= 0.0:
            raise ValueError("checkerboard dimensions and maximum RMS must be positive")
        # ``squares_x/y`` are the printed black/white cell counts. They must
        # be measured from the real board; outer dimensions alone are not a
        # reliable source because they commonly include a white border.
        if settings.get("squares_x") is None or settings.get("squares_y") is None:
            raise ValueError(
                "checkerboard square counts are not configured; fill long/short total squares"
            )
        self.squares_x = int(settings["squares_x"])
        self.squares_y = int(settings["squares_y"])
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("checkerboard must contain at least 3 x 3 squares")
        if (
            self.squares_x * self.square_size_mm > self.board_width_mm + 1e-9
            or self.squares_y * self.square_size_mm > self.board_height_mm + 1e-9
        ):
            raise ValueError(
                "printed checkerboard grid cannot be larger than the physical board"
            )
        self.pattern_size = (self.squares_x - 1, self.squares_y - 1)
        configured_pattern = settings.get("inner_corners")
        if configured_pattern is not None and tuple(
            int(value) for value in configured_pattern
        ) != self.pattern_size:
            raise ValueError(
                "checkerboard.inner_corners must be {}; got {}".format(
                    list(self.pattern_size), list(configured_pattern)
                )
            )

    @property
    def corner_count(self):
        return int(self.pattern_size[0] * self.pattern_size[1])

    @property
    def object_points_m(self):
        columns, rows = self.pattern_size
        points = np.zeros((columns * rows, 3), dtype=np.float32)
        points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        points[:, :2] *= self.square_size_mm / 1000.0
        return points

    def detect_corners(self, image):
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        corners = None
        found = False
        if self.prefer_sb and hasattr(cv2, "findChessboardCornersSB"):
            flags = 0
            for name in ("CALIB_CB_NORMALIZE_IMAGE", "CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY"):
                flags |= int(getattr(cv2, name, 0))
            found, corners = cv2.findChessboardCornersSB(
                gray, self.pattern_size, flags=flags
            )
        if not found:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(
                gray, self.pattern_size, flags=flags
            )
            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    40, 1e-4,
                )
                corners = cv2.cornerSubPix(
                    gray, np.asarray(corners, dtype=np.float32),
                    (5, 5), (-1, -1), criteria,
                )
        if not found or corners is None:
            return None
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if len(corners) != self.corner_count:
            return None
        return corners

    def estimate(self, image, camera_matrix, distortion):
        corners = self.detect_corners(image)
        if corners is None:
            return CheckerboardObservation(
                False,
                reason="checkerboard {}x{} inner corners not found".format(
                    *self.pattern_size
                ),
            )
        matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
        ok, rotation_vector, translation = cv2.solvePnP(
            self.object_points_m,
            corners.reshape(-1, 1, 2),
            matrix,
            coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return CheckerboardObservation(False, corners=corners, reason="checkerboard solvePnP failed")
        if hasattr(cv2, "solvePnPRefineLM"):
            rotation_vector, translation = cv2.solvePnPRefineLM(
                self.object_points_m,
                corners.reshape(-1, 1, 2),
                matrix,
                coefficients,
                rotation_vector,
                translation,
            )
        projected, _ = cv2.projectPoints(
            self.object_points_m, rotation_vector, translation, matrix, coefficients
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)
        rms = float(np.sqrt(np.mean(errors ** 2)))
        if not np.isfinite(rms) or rms > self.maximum_rms_px:
            return CheckerboardObservation(
                False, corners=corners, rms_reprojection_error_px=rms,
                reason="checkerboard RMS {:.3f} px exceeds {:.3f} px".format(
                    rms, self.maximum_rms_px
                ),
            )
        camera_from_board = np.eye(4, dtype=np.float64)
        camera_from_board[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        camera_from_board[:3, 3] = np.asarray(translation).reshape(3)
        return CheckerboardObservation(
            True,
            as_transform(camera_from_board, "camera_from_checkerboard"),
            corners,
            rms,
            "checkerboard accepted",
        )

    def draw(self, image, observation):
        preview = np.asarray(image).copy()
        corners = getattr(observation, "corners", None)
        if corners is not None:
            cv2.drawChessboardCorners(
                preview, self.pattern_size,
                np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
                bool(observation.valid),
            )
        return preview

    def metadata(self):
        return {
            "board_width_mm": self.board_width_mm,
            "board_height_mm": self.board_height_mm,
            "square_size_mm": self.square_size_mm,
            "squares": [self.squares_x, self.squares_y],
            "inner_corners": list(self.pattern_size),
            "corner_count": self.corner_count,
            "maximum_rms_px": self.maximum_rms_px,
        }


__all__ = [
    "CHECKERBOARD_TARGET", "CheckerboardObservation", "CheckerboardTarget",
]
