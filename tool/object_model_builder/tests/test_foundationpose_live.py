#!/usr/bin/env python3

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from tool.object_model_builder.foundationpose_live import (
    FoundationPoseLiveConfig,
    FoundationPoseLiveFrame,
    FoundationPoseLiveWorker,
    draw_pose_overlay,
)


class FakeRuntime:
    def __init__(self, calls, block_first=False):
        self.calls = calls
        self.block_first = block_first
        self.started = threading.Event()
        self.release = threading.Event()

    def register_frame(self, **kwargs):
        self.calls.append(("register", kwargs["rgb"][0, 0, 0]))
        if self.block_first:
            self.started.set()
            self.release.wait(1.0)
        return np.eye(4)

    def track_frame(self, **kwargs):
        self.calls.append(("track", kwargs["rgb"][0, 0, 0]))
        return np.eye(4)

    def reset(self):
        return None

    def close(self):
        return None


class FoundationPoseLiveTests(unittest.TestCase):
    def _frame(self, value, frame_id=None):
        return FoundationPoseLiveFrame(
            frame_id=frame_id if frame_id is not None else value,
            timestamp_s=float(value),
            color_bgr=np.full((8, 10, 3), value, dtype=np.uint8),
            depth_m=np.ones((8, 10), dtype=np.float32),
            mask=np.ones((8, 10), dtype=np.uint8),
            camera_matrix=np.asarray(
                [[100.0, 0.0, 5.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]
            ),
        )

    def test_register_then_track_and_reset(self):
        calls = []
        runtime = FakeRuntime(calls)
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "object.obj"
            mesh.write_text("placeholder", encoding="utf-8")
            worker = FoundationPoseLiveWorker(
                runtime_factory=lambda _config: runtime,
                mesh_bounds_loader=lambda _path, _scale: np.asarray(
                    [[-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]]
                ),
            )
            worker.configure(
                FoundationPoseLiveConfig("/vendor", str(mesh))
            )
            self.assertTrue(worker.wait_until_idle())
            self.assertEqual(worker.poll().status, "ready")
            worker.submit(self._frame(1))
            self.assertTrue(worker.wait_until_idle())
            self.assertEqual(worker.poll().status, "registered")
            worker.submit(self._frame(2))
            self.assertTrue(worker.wait_until_idle())
            self.assertEqual(worker.poll().status, "tracking")
            self.assertEqual([item[0] for item in calls], ["register", "track"])
            worker.reset()
            self.assertTrue(worker.wait_until_idle())
            self.assertEqual(worker.poll().status, "reset")
            worker.submit(self._frame(3))
            self.assertTrue(worker.wait_until_idle())
            self.assertEqual(worker.poll().status, "registered")
            worker.close()

    def test_only_latest_pending_frame_is_processed(self):
        calls = []
        runtime = FakeRuntime(calls, block_first=True)
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "object.obj"
            mesh.write_text("placeholder", encoding="utf-8")
            worker = FoundationPoseLiveWorker(
                runtime_factory=lambda _config: runtime,
                mesh_bounds_loader=lambda _path, _scale: np.asarray(
                    [[-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]]
                ),
            )
            worker.configure(FoundationPoseLiveConfig("/vendor", str(mesh)))
            self.assertTrue(worker.wait_until_idle())
            worker.poll()
            worker.submit(self._frame(1))
            self.assertTrue(runtime.started.wait(1.0))
            worker.submit(self._frame(2))
            worker.submit(self._frame(3))
            runtime.release.set()
            self.assertTrue(worker.wait_until_idle())
            result = worker.poll()
            self.assertEqual(result.frame_id, 3)
            self.assertEqual(calls, [("register", 1), ("track", 3)])
            worker.close()

    def test_invalid_frame_is_rejected(self):
        frame = self._frame(1)
        frame.depth_m[:] = 0.0
        with self.assertRaises(ValueError):
            FoundationPoseLiveWorker().submit(frame)

        frame = self._frame(1)
        frame.mask[:] = 0
        with self.assertRaises(ValueError):
            FoundationPoseLiveWorker().submit(frame)

    def test_missing_mesh_is_rejected_before_runtime_load(self):
        worker = FoundationPoseLiveWorker()
        with self.assertRaises(FileNotFoundError):
            worker.configure(
                FoundationPoseLiveConfig("/vendor", "/missing/object.obj")
            )

    def test_overlay_draws_box_and_axes(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        output = draw_pose_overlay(
            image,
            np.eye(4),
            np.asarray([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]),
            np.asarray([[-0.2, -0.2, 0.8], [0.2, 0.2, 1.2]]),
            mode="TRACK",
            inference_ms=12.5,
        )
        self.assertEqual(output.shape, image.shape)
        self.assertGreater(int(np.count_nonzero(output)), 0)
        self.assertFalse(np.array_equal(output, image))


if __name__ == "__main__":
    unittest.main()
