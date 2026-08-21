"""FoundationPose runtime wrapper for the migrated visual grasping pipeline.

This uses the existing verified ``FoundationPoseRuntime`` adapter in
``ros_ws`` instead of duplicating the vendor import/fallback logic from the
original release.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARM_FRAMEWORK_SRC = (
    PROJECT_ROOT
    / "ros_ws"
    / "src"
    / "arm_vision_framework"
    / "src"
)


class FoundationPosePoseEstimator:
    """Thin synchronous wrapper around :class:`FoundationPoseRuntime`."""

    def __init__(
        self,
        foundationpose_root: str,
        mesh_path: str,
        mesh_scale_to_meters: float = 1.0,
        debug_dir: str = "/tmp/visual_grasp_pipeline",
        debug: int = 0,
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        device: str = "cuda:0",
        use_mask_center_guidance: bool = True,
    ):
        self.mesh_path = Path(mesh_path).expanduser().resolve()
        self.mesh_scale_to_meters = float(mesh_scale_to_meters)
        self.est_refine_iter = int(est_refine_iter)
        self.track_refine_iter = int(track_refine_iter)
        self.device = device
        self.use_mask_center_guidance = bool(use_mask_center_guidance)
        self.debug_dir = str(Path(debug_dir).expanduser().resolve())
        self._runtime = None
        self._runtime_config = (
            str(Path(foundationpose_root).expanduser().resolve()),
            self.debug_dir,
            int(debug),
            self.est_refine_iter,
            self.track_refine_iter,
            self.device,
            self.use_mask_center_guidance,
        )

    def _ensure_runtime(self):
        if self._runtime is not None:
            return self._runtime
        source = str(ARM_FRAMEWORK_SRC)
        if source not in sys.path:
            sys.path.insert(0, source)
        from arm_vision_framework.adapters.foundationpose import (
            FoundationPoseRuntime,
        )

        (
            root,
            debug_dir,
            debug,
            est_iter,
            track_iter,
            device,
            mask_guidance,
        ) = self._runtime_config
        self._runtime = FoundationPoseRuntime(
            foundationpose_root=root,
            debug_dir=debug_dir,
            debug=debug,
            est_refine_iter=est_iter,
            track_refine_iter=track_iter,
            device=device,
            use_mask_center_guidance=mask_guidance,
        )
        return self._runtime

    def register(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> np.ndarray:
        runtime = self._ensure_runtime()
        return runtime.register_frame(
            rgb=rgb,
            depth_m=np.asarray(depth_m, dtype=np.float32),
            mask=(np.asarray(mask) > 0).astype(np.uint8),
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            mesh_path=str(self.mesh_path),
            mesh_scale_to_meters=self.mesh_scale_to_meters,
        )

    def track(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> np.ndarray:
        runtime = self._ensure_runtime()
        return runtime.track_frame(
            rgb=rgb,
            depth_m=np.asarray(depth_m, dtype=np.float32),
            mask=(np.asarray(mask) > 0).astype(np.uint8),
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            mesh_path=str(self.mesh_path),
            mesh_scale_to_meters=self.mesh_scale_to_meters,
        )

    def reset(self) -> None:
        if self._runtime is not None:
            self._runtime.reset()

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    @property
    def mesh_bounds(self) -> Optional[np.ndarray]:
        """Return mesh bounds in metres, or ``None`` if the mesh cannot be read."""
        try:
            import trimesh

            mesh = trimesh.load(str(self.mesh_path), process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            if mesh is None:
                return None
            return np.asarray(mesh.bounds, dtype=np.float64) * self.mesh_scale_to_meters
        except Exception:
            return None
