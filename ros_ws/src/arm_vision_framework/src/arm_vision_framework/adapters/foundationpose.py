"""FoundationPose/ FoundationPose++ runtime boundary."""

from pathlib import Path

import numpy as np

from ..errors import BackendUnavailable
from ..interfaces import PoseEstimator
from ..transforms import as_transform
from ..types import ObjectPoseEstimate


class FoundationPoseEstimator(PoseEstimator):
    """Adapter around a runtime exposing register_frame() and track_frame()."""

    def __init__(
        self,
        mesh_path,
        mesh_scale_to_meters=1.0,
        runtime=None,
        require_aligned_depth=True,
    ):
        self.mesh_path = Path(mesh_path).expanduser() if mesh_path else None
        self.mesh_scale_to_meters = float(mesh_scale_to_meters)
        self.runtime = runtime
        self.require_aligned_depth = bool(require_aligned_depth)
        self.registered = False

    def reset(self):
        self.registered = False
        if self.runtime is not None and hasattr(self.runtime, "reset"):
            self.runtime.reset()

    def estimate(self, frame, segmentation):
        if self.runtime is None:
            raise BackendUnavailable(
                "FoundationPose runtime is not attached; install the selected source and provide a runtime wrapper"
            )
        if self.mesh_path is None or not self.mesh_path.is_file():
            raise BackendUnavailable("object mesh file is not configured")
        if frame.depth_m is None:
            return ObjectPoseEstimate(False, reason="FoundationPose requires depth")
        if self.require_aligned_depth and not frame.depth_aligned_to_color:
            return ObjectPoseEstimate(False, reason="depth is not aligned to the color image")
        if frame.depth_m.shape != frame.color_bgr.shape[:2]:
            return ObjectPoseEstimate(False, reason="RGB and depth dimensions do not match")
        if not segmentation.valid or segmentation.mask is None:
            return ObjectPoseEstimate(False, reason="FoundationPose requires a valid object mask")
        if segmentation.mask.shape != frame.color_bgr.shape[:2]:
            return ObjectPoseEstimate(False, reason="mask and RGB dimensions do not match")
        arguments = dict(
            rgb=frame.color_bgr,
            depth_m=np.asarray(frame.depth_m, dtype=np.float32),
            mask=np.asarray(segmentation.mask, dtype=np.uint8),
            camera_matrix=frame.camera_matrix,
            mesh_path=str(self.mesh_path),
            mesh_scale_to_meters=self.mesh_scale_to_meters,
        )
        if not self.registered:
            matrix = self.runtime.register_frame(**arguments)
            self.registered = True
            tracking = False
        else:
            matrix = self.runtime.track_frame(**arguments)
            tracking = True
        return ObjectPoseEstimate(
            True,
            as_transform(matrix, "camera_from_object"),
            score=1.0,
            tracking=tracking,
            reason="FoundationPose pose accepted",
        )
