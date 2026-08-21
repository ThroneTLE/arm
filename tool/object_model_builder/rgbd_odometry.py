"""Small RGB-D odometry tracker used for marker-free model capture.

The first accepted RGB-D view defines the reconstruction/world frame. Each
subsequent view is registered against the previous accepted view with Open3D
RGB-D odometry. This is deliberately capture-only: FoundationPose live output
still reports ``camera_from_object`` in the current camera optical frame.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class OdometryResult:
    world_from_camera: np.ndarray
    success: bool
    fitness: float
    rmse: float
    translation_m: float
    rotation_deg: float


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for marker-free RGB-D odometry") from error
    return o3d


class RgbdOdometryTracker:
    """Estimate camera motion while the object and scene remain stationary."""

    def __init__(
        self,
        minimum_fitness: float = 0.08,
        maximum_rmse: float = 0.03,
        maximum_depth_m: float = 2.0,
    ):
        self.minimum_fitness = float(minimum_fitness)
        self.maximum_rmse = float(maximum_rmse)
        self.maximum_depth_m = float(maximum_depth_m)
        self.reset()

    def reset(self) -> None:
        self._previous_color = None
        self._previous_depth = None
        self._world_from_camera = np.eye(4, dtype=np.float64)

    @property
    def has_reference(self) -> bool:
        return self._previous_color is not None

    @staticmethod
    def _rgbd(o3d, color_bgr, depth_m):
        color = np.ascontiguousarray(np.asarray(color_bgr)[..., ::-1])
        depth = np.asarray(depth_m, dtype=np.float32)
        depth = np.where(
            np.isfinite(depth) & (depth >= 0.001), depth, 0.0
        )
        depth = np.where(depth <= 2.0, depth, 0.0)
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

    def update(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> OdometryResult:
        color = np.asarray(color_bgr)
        depth = np.asarray(depth_m, dtype=np.float32)
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("RGB odometry color must have shape HxWx3")
        if depth.shape != color.shape[:2]:
            raise ValueError("RGB-D odometry color/depth shapes do not match")
        if not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            raise ValueError("RGB-D odometry camera matrix is invalid")
        current_color = color.copy()
        current_depth = depth.copy()
        if self._previous_color is None:
            self._previous_color = current_color
            self._previous_depth = current_depth
            return OdometryResult(
                self._world_from_camera.copy(), True, 1.0, 0.0, 0.0, 0.0
            )

        o3d = _open3d()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(color.shape[1]), int(color.shape[0]),
            float(K[0, 0]), float(K[1, 1]),
            float(K[0, 2]), float(K[1, 2]),
        )
        source = self._rgbd(o3d, self._previous_color, self._previous_depth)
        target = self._rgbd(o3d, current_color, current_depth)
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        success, current_from_previous, information = (
            o3d.pipelines.odometry.compute_rgbd_odometry(
                source, target, intrinsic, np.eye(4), jacobian
            )
        )
        information = np.asarray(information, dtype=np.float64)
        fitness = float(np.trace(information) / max(float(information.shape[0]), 1.0))
        # Open3D's information matrix scale is not a universal probability;
        # use the correspondence count proxy exposed by the matrix only for
        # diagnostics and accept successful finite transforms by default.
        transform = np.asarray(current_from_previous, dtype=np.float64).reshape(4, 4)
        valid_transform = bool(
            success
            and np.isfinite(transform).all()
            and np.allclose(transform[3], [0, 0, 0, 1], atol=1e-4)
        )
        if not valid_transform:
            self._previous_color = current_color
            self._previous_depth = current_depth
            return OdometryResult(
                self._world_from_camera.copy(), False, fitness, float("inf"), 0.0, 0.0
            )
        # current_from_previous maps points in the previous camera frame into
        # the current camera frame. Invert the accumulated transform to obtain
        # the current camera pose in the first-camera/world frame.
        camera_from_world = transform @ np.linalg.inv(self._world_from_camera)
        world_from_camera = np.linalg.inv(camera_from_world)
        relative = np.linalg.inv(self._world_from_camera) @ world_from_camera
        translation = float(np.linalg.norm(relative[:3, 3]))
        cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        rotation_deg = float(np.degrees(np.arccos(cosine)))
        self._world_from_camera = world_from_camera
        self._previous_color = current_color
        self._previous_depth = current_depth
        return OdometryResult(
            world_from_camera.copy(), True, fitness, 0.0,
            translation, rotation_deg,
        )
