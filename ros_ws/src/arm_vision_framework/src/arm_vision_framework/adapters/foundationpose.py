"""FoundationPose/ FoundationPose++ runtime boundary.

The vendor project stays outside this ROS package.  The runtime below imports
it from the configured checkout and exposes the small frame-based contract used
by :class:`FoundationPoseEstimator`.
"""

import gc
import sys
from pathlib import Path

import cv2
import numpy as np

from ..errors import BackendUnavailable
from ..interfaces import PoseEstimator
from ..transforms import as_transform
from ..types import DetectionResult, ObjectPoseEstimate


def _fallback_bilateral_filter_depth(depth, radius=2, **_kwargs):
    """OpenCV fallback for the optional Warp depth filter."""
    tensor = hasattr(depth, "detach") and hasattr(depth, "device")
    if tensor:
        import torch
        import torch.nn.functional as F

        source = depth.float()
        valid = torch.isfinite(source) & (source >= 0.001)
        size = max(3, int(radius) * 2 + 1)
        area = float(size * size)
        filtered = F.avg_pool2d(
            (source * valid).unsqueeze(0).unsqueeze(0),
            size,
            stride=1,
            padding=int(radius),
        )[0, 0]
        count = F.avg_pool2d(
            valid.float().unsqueeze(0).unsqueeze(0),
            size,
            stride=1,
            padding=int(radius),
        )[0, 0]
        filtered = filtered / torch.clamp(count, min=1.0 / area)
        return torch.where(valid, filtered, torch.zeros_like(filtered)).to(depth.dtype)
    source = (
        np.asarray(depth, dtype=np.float32)
    )
    valid = np.isfinite(source) & (source >= 0.001)
    filtered = cv2.bilateralFilter(source, d=max(3, int(radius) * 2 + 1), sigmaColor=0.02, sigmaSpace=float(radius))
    filtered[~valid] = 0.0
    return filtered


def _fallback_erode_depth(depth, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, **_kwargs):
    """Keep depth pixels with a locally consistent neighborhood."""
    tensor = hasattr(depth, "detach") and hasattr(depth, "device")
    if tensor:
        import torch
        import torch.nn.functional as F

        source = depth.float()
        valid = torch.isfinite(source) & (source >= 0.001)
        size = max(3, int(radius) * 2 + 1)
        area = float(size * size)
        valid_count = F.avg_pool2d(
            valid.float().unsqueeze(0).unsqueeze(0),
            size,
            stride=1,
            padding=int(radius),
        )[0, 0] * area
        mean = F.avg_pool2d(
            (source * valid).unsqueeze(0).unsqueeze(0),
            size,
            stride=1,
            padding=int(radius),
        )[0, 0] * area / torch.clamp(valid_count, min=1.0)
        consistent = valid & (valid_count >= area * (1.0 - float(ratio_thres)))
        consistent &= torch.abs(source - mean) <= float(depth_diff_thres)
        return torch.where(consistent, source, torch.zeros_like(source)).to(depth.dtype)
    source = (
        np.asarray(depth, dtype=np.float32)
    )
    valid = np.isfinite(source) & (source >= 0.001)
    kernel_size = max(3, int(radius) * 2 + 1)
    median = cv2.medianBlur(np.where(valid, source, 0.0), kernel_size)
    valid_count = cv2.boxFilter(
        valid.astype(np.float32), -1, (kernel_size, kernel_size), normalize=False
    )
    consistent = valid & (valid_count >= (kernel_size * kernel_size) * (1.0 - float(ratio_thres)))
    consistent &= np.abs(source - median) <= float(depth_diff_thres)
    output = np.where(consistent, source, 0.0).astype(np.float32)
    return output


class FoundationPoseRuntime:
    """Own one GPU FoundationPose estimator for one metric mesh."""

    def __init__(
        self,
        foundationpose_root,
        debug_dir="/tmp/arm_foundationpose",
        debug=0,
        est_refine_iter=5,
        track_refine_iter=2,
        device="cuda:0",
        use_mask_center_guidance=True,
        registration_max_hypotheses=0,
    ):
        self.root = Path(foundationpose_root).expanduser().resolve()
        self.debug_dir = Path(debug_dir).expanduser().resolve()
        self.debug = int(debug)
        self.est_refine_iter = int(est_refine_iter)
        self.track_refine_iter = int(track_refine_iter)
        self.device = str(device)
        self.use_mask_center_guidance = bool(use_mask_center_guidance)
        self.registration_max_hypotheses = max(
            0, int(registration_max_hypotheses)
        )
        self._foundation_pose = None
        self._score_predictor = None
        self._refine_predictor = None
        self._trimesh = None
        self._estimator = None
        self._mesh_key = None
        self._glctx = None
        self._load_vendor_modules()

    def _load_vendor_modules(self):
        if not self.root.is_dir():
            raise BackendUnavailable(
                "FoundationPose root does not exist: {}".format(self.root)
            )
        try:
            import torch
        except ImportError as error:
            raise BackendUnavailable("PyTorch is required for FoundationPose") from error
        if not torch.cuda.is_available():
            raise BackendUnavailable(
                "FoundationPose requires a CUDA-enabled PyTorch environment"
            )
        if not self.device.startswith("cuda"):
            raise BackendUnavailable(
                "FoundationPose vendor code currently requires CUDA, got {}".format(
                    self.device
                )
            )
        foundationpose_dir = self.root / "FoundationPose"
        if not (foundationpose_dir / "estimater.py").is_file():
            raise BackendUnavailable(
                "FoundationPose estimater.py is missing under {}".format(foundationpose_dir)
            )
        # estimater.py imports Utils as a top-level module, while callers use
        # FoundationPose.estimater. Both paths are therefore required.
        for path in (str(self.root), str(foundationpose_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            import trimesh
            import FoundationPose.estimater as estimater_module
            from FoundationPose.estimater import (
                FoundationPose,
                PoseRefinePredictor,
                ScorePredictor,
                dr,
            )
        except Exception as error:
            raise BackendUnavailable(
                "failed to import FoundationPose++ from {}: {}".format(self.root, error)
            ) from error
        try:
            self._glctx = dr.RasterizeCudaContext(self.device)
        except TypeError:
            self._glctx = dr.RasterizeCudaContext()
        except Exception as error:
            raise BackendUnavailable(
                "failed to create nvdiffrast CUDA context: {}".format(error)
            ) from error
        # Warp is optional in the environment.  The vendor module only
        # defines these functions when Warp imported successfully, although
        # register()/track_one() call them unconditionally.
        if not hasattr(estimater_module, "erode_depth"):
            estimater_module.erode_depth = _fallback_erode_depth
        if not hasattr(estimater_module, "bilateral_filter_depth"):
            estimater_module.bilateral_filter_depth = _fallback_bilateral_filter_depth
        self._foundation_pose = FoundationPose
        self._score_predictor = ScorePredictor
        self._refine_predictor = PoseRefinePredictor
        self._trimesh = trimesh

    @staticmethod
    def _load_mesh(trimesh_module, mesh_path, scale_to_meters):
        try:
            mesh = trimesh_module.load(str(mesh_path), process=False)
        except Exception as error:
            raise BackendUnavailable(
                "failed to load FoundationPose mesh: {}".format(error)
            ) from error
        if isinstance(mesh, trimesh_module.Scene):
            mesh = mesh.dump(concatenate=True)
        if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise BackendUnavailable("FoundationPose mesh is not a triangle mesh")
        scale = float(scale_to_meters)
        if not np.isfinite(scale) or scale <= 0.0:
            raise BackendUnavailable("mesh_scale_to_meters must be positive")
        mesh = mesh.copy()
        mesh.apply_scale(scale)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) < 4 or len(faces) < 4 or not np.isfinite(vertices).all():
            raise BackendUnavailable(
                "FoundationPose mesh has invalid or insufficient geometry"
            )
        # make_mesh_tensors expects visual data for a colorless OBJ/STL.
        colors = getattr(mesh.visual, "vertex_colors", None)
        if colors is None or len(colors) != len(vertices):
            mesh.visual.vertex_colors = np.tile(
                np.asarray([150, 150, 150, 255], dtype=np.uint8),
                (len(vertices), 1),
            )
        return mesh

    def _ensure_estimator(self, mesh_path, mesh_scale_to_meters):
        mesh_path = Path(mesh_path).expanduser().resolve()
        if not mesh_path.is_file():
            raise BackendUnavailable(
                "object mesh file does not exist: {}".format(mesh_path)
            )
        key = (str(mesh_path), float(mesh_scale_to_meters), mesh_path.stat().st_mtime_ns)
        if self._estimator is not None and self._mesh_key == key:
            return self._estimator
        mesh = self._load_mesh(self._trimesh, mesh_path, mesh_scale_to_meters)
        try:
            estimator = self._foundation_pose(
                model_pts=np.asarray(mesh.vertices),
                model_normals=np.asarray(mesh.vertex_normals),
                mesh=mesh,
                scorer=self._score_predictor(),
                refiner=self._refine_predictor(),
                glctx=self._glctx,
                debug=self.debug,
                debug_dir=str(self.debug_dir),
            )
        except Exception as error:
            raise BackendUnavailable(
                "FoundationPose estimator initialization failed: {}".format(error)
            ) from error
        if (
            self.registration_max_hypotheses > 0
            and len(estimator.rot_grid) > self.registration_max_hypotheses
        ):
            import torch

            indices = torch.linspace(
                0,
                len(estimator.rot_grid) - 1,
                steps=self.registration_max_hypotheses,
                device=estimator.rot_grid.device,
            ).round().long()
            estimator.rot_grid = estimator.rot_grid.index_select(0, indices)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._estimator = estimator
        self._mesh_key = key
        return estimator

    @staticmethod
    def _prepare_inputs(rgb_bgr, depth_m, mask, camera_matrix):
        rgb_bgr = np.asarray(rgb_bgr)
        if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
            raise BackendUnavailable("RGB input must have shape HxWx3")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = np.asarray(depth_m, dtype=np.float32)
        depth = np.where(np.isfinite(depth) & (depth >= 0.001), depth, 0.0)
        object_mask = (np.asarray(mask) > 0).astype(np.uint8)
        if depth.shape != rgb.shape[:2] or object_mask.shape != rgb.shape[:2]:
            raise BackendUnavailable(
                "FoundationPose RGB, depth and mask shapes must match"
            )
        # The reference implementation keeps intrinsics in float64 (the
        # same dtype produced by its dataset readers); retaining that dtype
        # avoids a matmul mismatch after its CUDA default-tensor conversion.
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            raise BackendUnavailable("camera intrinsics are invalid")
        return rgb, depth, object_mask, K

    @staticmethod
    def _validate_pose(pose):
        matrix = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(matrix).all() or not np.allclose(
            matrix[3], [0, 0, 0, 1], atol=1e-4
        ):
            raise BackendUnavailable("FoundationPose returned an invalid 4x4 pose")
        return matrix

    def register_frame(
        self, rgb, depth_m, mask, camera_matrix, mesh_path, mesh_scale_to_meters=1.0
    ):
        estimator = self._ensure_estimator(mesh_path, mesh_scale_to_meters)
        rgb, depth, object_mask, K = self._prepare_inputs(
            rgb, depth_m, mask, camera_matrix
        )
        try:
            import torch

            torch.cuda.empty_cache()
            pose = estimator.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=object_mask,
                iteration=self.est_refine_iter,
            )
        except Exception as error:
            raise BackendUnavailable(
                "FoundationPose register failed: {}".format(error)
            ) from error
        return self._validate_pose(pose)

    def track_frame(
        self, rgb, depth_m, mask, camera_matrix, mesh_path, mesh_scale_to_meters=1.0
    ):
        estimator = self._ensure_estimator(mesh_path, mesh_scale_to_meters)
        if getattr(estimator, "pose_last", None) is None:
            raise BackendUnavailable("FoundationPose tracking requested before registration")
        rgb, depth, object_mask, K = self._prepare_inputs(
            rgb, depth_m, mask, camera_matrix
        )
        if self.use_mask_center_guidance:
            self._guide_pose_xy(estimator, object_mask, K)
        try:
            pose = estimator.track_one(
                rgb=rgb,
                depth=depth,
                K=K,
                iteration=self.track_refine_iter,
            )
        except Exception as error:
            raise BackendUnavailable(
                "FoundationPose tracking failed: {}".format(error)
            ) from error
        return self._validate_pose(pose)

    @staticmethod
    def _guide_pose_xy(estimator, object_mask, camera_matrix):
        """Use the current segmentation center as the Plus-Plus 2D cue."""
        ys, xs = np.where(object_mask > 0)
        if len(xs) == 0:
            return
        pose = getattr(estimator, "pose_last", None)
        if pose is None or not hasattr(pose, "device"):
            return
        import torch

        K = torch.as_tensor(camera_matrix, device=pose.device, dtype=pose.dtype)
        center_u = float(xs.min() + xs.max()) * 0.5
        center_v = float(ys.min() + ys.max()) * 0.5
        corrected = pose.clone()
        # FoundationPose versions differ: some keep pose_last as [4, 4],
        # while newer register() paths retain a singleton batch dimension
        # [1, 4, 4]. Apply the same XY center guidance to either layout.
        if corrected.ndim == 3:
            if corrected.shape[0] != 1:
                return
            depth = corrected[0, 2, 3]
            corrected[0, 0, 3] = (center_u - K[0, 2]) * depth / K[0, 0]
            corrected[0, 1, 3] = (center_v - K[1, 2]) * depth / K[1, 1]
        else:
            depth = corrected[2, 3]
            corrected[0, 3] = (center_u - K[0, 2]) * depth / K[0, 0]
            corrected[1, 3] = (center_v - K[1, 2]) * depth / K[1, 1]
        estimator.pose_last = corrected

    def reset(self):
        if self._estimator is not None:
            self._estimator.pose_last = None

    def close(self):
        self.reset()
        self._estimator = None
        self._mesh_key = None
        self._glctx = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


class FoundationPoseEstimator(PoseEstimator):
    """Adapter around a runtime exposing register_frame() and track_frame()."""

    def __init__(
        self,
        mesh_path,
        mesh_scale_to_meters=1.0,
        runtime=None,
        require_aligned_depth=True,
        mesh_paths=None,
        roi_padding_pixels=12,
    ):
        self.mesh_path = Path(mesh_path).expanduser() if mesh_path else None
        self.mesh_scale_to_meters = float(mesh_scale_to_meters)
        self.runtime = runtime
        self.require_aligned_depth = bool(require_aligned_depth)
        self.mesh_paths = {
            str(name): Path(path).expanduser() for name, path in (mesh_paths or {}).items()
        }
        self.roi_padding_pixels = max(0, int(roi_padding_pixels))
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

    def _mesh_for_detection(self, detection):
        mesh = self.mesh_paths.get(str(detection.class_name), self.mesh_path)
        if mesh is None or not Path(mesh).is_file():
            raise BackendUnavailable(
                "no FoundationPose mesh configured for class {}".format(
                    detection.class_name
                )
            )
        return mesh

    def estimate_detection(self, frame, detection):
        """Estimate one object from its YOLO ROI and adjusted intrinsics."""
        if frame.depth_m is None:
            return ObjectPoseEstimate(False, reason="FoundationPose requires depth")
        if self.require_aligned_depth and not frame.depth_aligned_to_color:
            return ObjectPoseEstimate(False, reason="depth is not aligned to the color image")
        x1, y1, x2, y2 = map(int, detection.bbox_xyxy)
        height, width = frame.color_bgr.shape[:2]
        padding = self.roi_padding_pixels
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            return ObjectPoseEstimate(False, reason="FoundationPose ROI is empty")
        mask = detection.mask
        if mask is None:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
        roi_mask = np.asarray(mask[y1:y2, x1:x2], dtype=np.uint8)
        camera_matrix = np.asarray(frame.camera_matrix, dtype=np.float64).reshape(3, 3).copy()
        camera_matrix[0, 2] -= x1
        camera_matrix[1, 2] -= y1
        mesh_path = self._mesh_for_detection(detection)
        if self.runtime is None:
            raise BackendUnavailable("FoundationPose runtime is not attached")
        pose = self.runtime.register_frame(
            rgb=frame.color_bgr[y1:y2, x1:x2],
            depth_m=np.asarray(frame.depth_m[y1:y2, x1:x2], dtype=np.float32),
            mask=roi_mask,
            camera_matrix=camera_matrix,
            mesh_path=str(mesh_path),
            mesh_scale_to_meters=self.mesh_scale_to_meters,
        )
        pose = as_transform(pose, "camera_from_object_roi")
        # The pose translation is expressed in the cropped camera frame. Move
        # it back to the original RGB optical frame with the same pixel shift.
        # This is equivalent to using the adjusted principal point above.
        return ObjectPoseEstimate(
            True, pose, score=float(detection.confidence), tracking=False,
            reason="FoundationPose ROI accepted",
            bbox_xyxy=detection.bbox_xyxy, class_id=detection.class_id,
            class_name=detection.class_name, confidence=detection.confidence,
        )

    def estimate_all(self, frame, segmentation):
        detections = tuple(segmentation.detections or ())
        if not detections:
            detections = (DetectionResult(
                segmentation.bbox_xyxy, segmentation.class_id,
                segmentation.class_name, segmentation.confidence, segmentation.mask,
            ),)
        estimates = []
        for detection in detections:
            estimate = self.estimate_detection(frame, detection)
            if estimate.valid:
                estimates.append(estimate)
        return tuple(estimates)
