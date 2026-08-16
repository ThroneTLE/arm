#!/usr/bin/env python3
"""Non-blocking FoundationPose registration/tracking for the desktop UI."""

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS_SOURCE_ROOT = (
    PROJECT_ROOT
    / "ros_ws"
    / "src"
    / "arm_vision_framework"
    / "src"
)


@dataclass(frozen=True)
class FoundationPoseLiveConfig:
    foundationpose_root: str
    mesh_path: str
    mesh_scale_to_meters: float = 1.0
    debug_dir: str = "/tmp/arm_foundationpose_live"
    debug: int = 0
    est_refine_iter: int = 5
    track_refine_iter: int = 2
    device: str = "cuda:0"
    use_mask_center_guidance: bool = True


@dataclass
class FoundationPoseLiveFrame:
    frame_id: object
    timestamp_s: float
    color_bgr: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    camera_matrix: np.ndarray


@dataclass
class FoundationPoseLiveResult:
    generation: int
    status: str
    frame_id: object = None
    timestamp_s: Optional[float] = None
    camera_from_object: Optional[np.ndarray] = None
    mesh_bounds_m: Optional[np.ndarray] = None
    inference_ms: float = 0.0
    error: Optional[Exception] = None


def create_foundationpose_runtime(config: FoundationPoseLiveConfig):
    """Load the existing ROS runtime adapter without copying vendor logic."""
    source = str(ROS_SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    from arm_vision_framework.adapters.foundationpose import FoundationPoseRuntime

    return FoundationPoseRuntime(
        foundationpose_root=config.foundationpose_root,
        debug_dir=config.debug_dir,
        debug=config.debug,
        est_refine_iter=config.est_refine_iter,
        track_refine_iter=config.track_refine_iter,
        device=config.device,
        use_mask_center_guidance=config.use_mask_center_guidance,
    )


def load_mesh_bounds_m(mesh_path: str, scale_to_meters: float) -> np.ndarray:
    """Return the scaled mesh bounds in the same object frame used by runtime."""
    import trimesh

    path = Path(mesh_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("FoundationPose mesh does not exist: {}".format(path))
    scale = float(scale_to_meters)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("mesh scale to meters must be positive")
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if mesh is None or not hasattr(mesh, "bounds"):
        raise ValueError("FoundationPose mesh is not a triangle mesh")
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(2, 3) * scale
    if not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("FoundationPose mesh bounds are invalid")
    return bounds


def validate_live_frame(frame: FoundationPoseLiveFrame) -> None:
    color = np.asarray(frame.color_bgr)
    depth = np.asarray(frame.depth_m)
    mask = np.asarray(frame.mask)
    matrix = np.asarray(frame.camera_matrix)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("FoundationPose RGB must have shape HxWx3")
    if depth.shape != color.shape[:2] or mask.shape != color.shape[:2]:
        raise ValueError("FoundationPose RGB, aligned depth and Mask dimensions must match")
    if not np.any(mask > 0):
        raise ValueError("FoundationPose requires a non-empty Mask")
    if not np.any(np.isfinite(depth) & (depth >= 0.001) & (mask > 0)):
        raise ValueError("FoundationPose Mask contains no valid aligned depth")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("FoundationPose camera matrix is invalid")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("FoundationPose focal length must be positive")


class FoundationPoseLiveWorker:
    """Run one estimator on one thread while retaining only the newest frame."""

    def __init__(
        self,
        runtime_factory: Callable[[FoundationPoseLiveConfig], object] = create_foundationpose_runtime,
        mesh_bounds_loader: Callable[[str, float], np.ndarray] = load_mesh_bounds_m,
    ):
        self._runtime_factory = runtime_factory
        self._mesh_bounds_loader = mesh_bounds_loader
        self._condition = threading.Condition()
        self._control_pending = deque()
        self._frame_pending = None
        self._results = deque(maxlen=1)
        self._worker_active = False
        self._shutdown = False
        self._enabled = False
        self._ready = False
        self._paused_on_error = False
        self._registered = False
        self._generation = 0
        self._config = None
        self._runtime = None
        self._mesh_bounds_m = None

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled

    @property
    def ready(self) -> bool:
        with self._condition:
            return self._ready

    def configure(self, config: FoundationPoseLiveConfig) -> int:
        path = Path(config.mesh_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("FoundationPose mesh does not exist: {}".format(path))
        scale = float(config.mesh_scale_to_meters)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("mesh scale to meters must be positive")
        normalized = FoundationPoseLiveConfig(
            foundationpose_root=str(Path(config.foundationpose_root).expanduser().resolve()),
            mesh_path=str(path),
            mesh_scale_to_meters=scale,
            debug_dir=str(Path(config.debug_dir).expanduser().resolve()),
            debug=int(config.debug),
            est_refine_iter=int(config.est_refine_iter),
            track_refine_iter=int(config.track_refine_iter),
            device=str(config.device),
            use_mask_center_guidance=bool(config.use_mask_center_guidance),
        )
        with self._condition:
            if self._shutdown:
                raise RuntimeError("FoundationPose live worker is closed")
            self._generation += 1
            generation = self._generation
            self._enabled = True
            self._ready = False
            self._paused_on_error = False
            self._registered = False
            self._config = normalized
            self._frame_pending = None
            self._results.clear()
            self._control_pending.clear()
            self._control_pending.append(("configure", generation))
            self._ensure_thread_locked()
            return generation

    def reset(self) -> bool:
        with self._condition:
            if (
                self._shutdown
                or not self._enabled
                or not self._ready
                or self._runtime is None
            ):
                return False
            self._generation += 1
            generation = self._generation
            self._ready = self._runtime is not None
            self._paused_on_error = False
            self._registered = False
            self._frame_pending = None
            self._results.clear()
            self._control_pending.append(("reset", generation))
            self._ensure_thread_locked()
            return True

    def stop(self) -> None:
        with self._condition:
            if self._shutdown:
                return
            self._generation += 1
            generation = self._generation
            self._enabled = False
            self._ready = False
            self._paused_on_error = False
            self._registered = False
            self._frame_pending = None
            self._results.clear()
            self._control_pending.clear()
            self._control_pending.append(("stop", generation))
            self._ensure_thread_locked()

    def submit(self, frame: FoundationPoseLiveFrame) -> bool:
        validate_live_frame(frame)
        copied = FoundationPoseLiveFrame(
            frame_id=frame.frame_id,
            timestamp_s=float(frame.timestamp_s),
            color_bgr=np.asarray(frame.color_bgr).copy(),
            depth_m=np.asarray(frame.depth_m, dtype=np.float32).copy(),
            mask=(np.asarray(frame.mask) > 0).astype(np.uint8),
            camera_matrix=np.asarray(frame.camera_matrix, dtype=np.float64).copy(),
        )
        with self._condition:
            if (
                self._shutdown
                or not self._enabled
                or self._paused_on_error
                or self._config is None
            ):
                return False
            self._frame_pending = (self._generation, copied)
            self._ensure_thread_locked()
            return True

    def poll(self) -> Optional[FoundationPoseLiveResult]:
        with self._condition:
            if not self._results:
                return None
            result = self._results.pop()
            self._results.clear()
            if result.generation != self._generation:
                return None
            return result

    def wait_until_idle(self, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while self._worker_active or self._control_pending or self._frame_pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self) -> None:
        with self._condition:
            self._shutdown = True
            self._enabled = False
            self._generation += 1
            self._control_pending.clear()
            self._frame_pending = None
            self._results.clear()
            self._condition.notify_all()
        self.wait_until_idle(timeout_s=5.0)

    def _ensure_thread_locked(self) -> None:
        if self._worker_active:
            self._condition.notify_all()
            return
        self._worker_active = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            with self._condition:
                if self._shutdown:
                    self._close_runtime()
                    self._worker_active = False
                    self._condition.notify_all()
                    return
                if self._control_pending:
                    task = self._control_pending.popleft()
                    frame_task = None
                else:
                    task = None
                    frame_task = self._frame_pending
                    self._frame_pending = None
                if task is None and frame_task is None:
                    self._worker_active = False
                    self._condition.notify_all()
                    return
            if task is not None:
                self._run_control(*task)
            else:
                self._run_frame(*frame_task)

    def _run_control(self, action: str, generation: int) -> None:
        if action == "configure":
            try:
                self._close_runtime()
                config = self._config
                bounds = self._mesh_bounds_loader(
                    config.mesh_path, config.mesh_scale_to_meters
                )
                runtime = self._runtime_factory(config)
                with self._condition:
                    if generation != self._generation or not self._enabled:
                        if hasattr(runtime, "close"):
                            runtime.close()
                        return
                    self._runtime = runtime
                    self._mesh_bounds_m = np.asarray(bounds, dtype=np.float64)
                    self._ready = True
                    self._registered = False
                self._publish(FoundationPoseLiveResult(generation, "ready"))
            except Exception as error:
                self._fail(generation, error)
        elif action == "reset":
            try:
                runtime = self._runtime
                if runtime is not None and hasattr(runtime, "reset"):
                    runtime.reset()
                self._publish(FoundationPoseLiveResult(generation, "reset"))
            except Exception as error:
                self._fail(generation, error)
        elif action == "stop":
            self._close_runtime()
            self._publish(FoundationPoseLiveResult(generation, "stopped"))

    def _run_frame(self, generation: int, frame: FoundationPoseLiveFrame) -> None:
        with self._condition:
            if (
                generation != self._generation
                or not self._enabled
                or not self._ready
                or self._paused_on_error
            ):
                return
            runtime = self._runtime
            config = self._config
            registered = self._registered
            bounds = None if self._mesh_bounds_m is None else self._mesh_bounds_m.copy()
        arguments = {
            "rgb": frame.color_bgr,
            "depth_m": frame.depth_m,
            "mask": frame.mask,
            "camera_matrix": frame.camera_matrix,
            "mesh_path": config.mesh_path,
            "mesh_scale_to_meters": config.mesh_scale_to_meters,
        }
        started = time.perf_counter()
        try:
            if registered:
                pose = runtime.track_frame(**arguments)
                status = "tracking"
            else:
                pose = runtime.register_frame(**arguments)
                status = "registered"
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
            if not np.isfinite(pose).all():
                raise ValueError("FoundationPose returned a non-finite pose")
            with self._condition:
                if generation != self._generation:
                    return
                self._registered = True
            self._publish(
                FoundationPoseLiveResult(
                    generation=generation,
                    status=status,
                    frame_id=frame.frame_id,
                    timestamp_s=frame.timestamp_s,
                    camera_from_object=pose,
                    mesh_bounds_m=bounds,
                    inference_ms=elapsed_ms,
                )
            )
        except Exception as error:
            self._fail(generation, error)

    def _fail(self, generation: int, error: Exception) -> None:
        with self._condition:
            if generation != self._generation:
                return
            self._ready = self._runtime is not None
            self._registered = False
            self._paused_on_error = True
            runtime = self._runtime
        if runtime is not None and hasattr(runtime, "reset"):
            try:
                runtime.reset()
            except Exception:
                pass
        self._publish(
            FoundationPoseLiveResult(
                generation=generation,
                status="error",
                error=error,
            )
        )

    def _publish(self, result: FoundationPoseLiveResult) -> None:
        with self._condition:
            if result.generation == self._generation:
                self._results.clear()
                self._results.append(result)
            self._condition.notify_all()

    def _close_runtime(self) -> None:
        runtime = self._runtime
        self._runtime = None
        self._mesh_bounds_m = None
        self._ready = False
        self._registered = False
        if runtime is not None and hasattr(runtime, "close"):
            try:
                runtime.close()
            except Exception:
                pass


def box_corners(bounds_m: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds_m, dtype=np.float64).reshape(2, 3)
    low, high = bounds
    return np.asarray(
        [
            [low[0], low[1], low[2]],
            [high[0], low[1], low[2]],
            [high[0], high[1], low[2]],
            [low[0], high[1], low[2]],
            [low[0], low[1], high[2]],
            [high[0], low[1], high[2]],
            [high[0], high[1], high[2]],
            [low[0], high[1], high[2]],
        ],
        dtype=np.float64,
    )


def project_object_points(
    points_object: np.ndarray,
    camera_from_object: np.ndarray,
    camera_matrix: np.ndarray,
):
    points = np.asarray(points_object, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(camera_from_object, dtype=np.float64).reshape(4, 4)
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    camera = (pose @ homogeneous.T).T[:, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.001)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected = (matrix @ camera[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def draw_pose_overlay(
    color_bgr: np.ndarray,
    camera_from_object: np.ndarray,
    camera_matrix: np.ndarray,
    mesh_bounds_m: np.ndarray,
    mode: str = "TRACK",
    inference_ms: Optional[float] = None,
) -> np.ndarray:
    """Draw a metric 3-D box and object-frame axes on an RGB preview."""
    image = np.asarray(color_bgr).copy()
    bounds = np.asarray(mesh_bounds_m, dtype=np.float64).reshape(2, 3)
    corners = box_corners(bounds)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    axis_length = max(0.01, diagonal * 0.35)
    axes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    points = np.vstack([corners, axes])
    pixels, valid = project_object_points(points, camera_from_object, camera_matrix)
    rounded = np.rint(pixels).astype(np.int32, casting="unsafe", copy=False)
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for first, second in edges:
        if valid[first] and valid[second]:
            cv2.line(
                image,
                tuple(rounded[first]),
                tuple(rounded[second]),
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
    axis_colors = ((0, 0, 255), (0, 220, 0), (255, 80, 0))
    axis_labels = ("X", "Y", "Z")
    origin_index = 8
    for offset, (color, label) in enumerate(zip(axis_colors, axis_labels), start=1):
        target_index = origin_index + offset
        if valid[origin_index] and valid[target_index]:
            cv2.arrowedLine(
                image,
                tuple(rounded[origin_index]),
                tuple(rounded[target_index]),
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.16,
            )
            cv2.putText(
                image,
                label,
                tuple(rounded[target_index]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    label = "FoundationPose {}".format(str(mode).upper())
    if inference_ms is not None:
        label += "  {:.0f} ms".format(float(inference_ms))
    cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 330), 31), (20, 24, 27), -1)
    cv2.putText(
        image,
        label,
        (9, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (90, 235, 210),
        1,
        cv2.LINE_AA,
    )
    return image
