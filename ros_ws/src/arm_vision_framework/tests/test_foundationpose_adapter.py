#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arm_vision_framework.adapters.foundationpose import (  # noqa: E402
    FoundationPoseEstimator,
    FoundationPoseRuntime,
)
from arm_vision_framework.types import (  # noqa: E402
    DetectionResult, FrameData, SegmentationResult,
)


class FakeRuntime:
    def __init__(self):
        self.register_calls = 0
        self.track_calls = 0
        self.last_rgb = None

    def register_frame(self, **kwargs):
        self.register_calls += 1
        self.last_rgb = kwargs["rgb"].copy()
        return np.eye(4)

    def track_frame(self, **kwargs):
        self.track_calls += 1
        return np.eye(4)

    def reset(self):
        pass


class FoundationPoseAdapterTest(unittest.TestCase):
    def _frame(self, depth_aligned=True):
        color = np.zeros((4, 5, 3), dtype=np.uint8)
        color[0, 0] = [1, 2, 3]  # BGR; the vendor runtime converts it to RGB.
        return FrameData(
            color_bgr=color,
            camera_matrix=np.asarray([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]),
            distortion=np.zeros(5),
            timestamp_s=1.0,
            frame_id="camera_color",
            depth_m=np.ones((4, 5), dtype=np.float32),
            depth_aligned_to_color=depth_aligned,
        )

    def _segmentation(self):
        return SegmentationResult(True, mask=np.ones((4, 5), dtype=np.uint8))

    def test_runtime_input_conversion_is_explicit(self):
        rgb, depth, mask, K = FoundationPoseRuntime._prepare_inputs(
            self._frame().color_bgr,
            self._frame().depth_m,
            self._segmentation().mask,
            self._frame().camera_matrix,
        )
        np.testing.assert_array_equal(rgb[0, 0], [3, 2, 1])
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(mask.dtype, np.uint8)
        np.testing.assert_allclose(K[0, 0], 100.0)

    def test_adapter_registers_then_tracks(self):
        runtime = FakeRuntime()
        estimator = FoundationPoseEstimator("/tmp/object.obj", runtime=runtime)
        estimator.mesh_path = Path(__file__)  # satisfy the file gate
        first = estimator.estimate(self._frame(), self._segmentation())
        second = estimator.estimate(self._frame(), self._segmentation())
        self.assertTrue(first.valid)
        self.assertFalse(first.tracking)
        self.assertTrue(second.tracking)
        self.assertEqual(runtime.register_calls, 1)
        self.assertEqual(runtime.track_calls, 1)
        np.testing.assert_array_equal(runtime.last_rgb[0, 0], [1, 2, 3])

    def test_unaligned_depth_is_rejected_before_backend(self):
        runtime = FakeRuntime()
        estimator = FoundationPoseEstimator(__file__, runtime=runtime)
        result = estimator.estimate(self._frame(depth_aligned=False), self._segmentation())
        self.assertFalse(result.valid)
        self.assertIn("aligned", result.reason)
        self.assertEqual(runtime.register_calls, 0)

    def test_detection_roi_crops_all_inputs_and_offsets_intrinsics(self):
        runtime = FakeRuntime()
        runtime.last_arguments = None
        original_register = runtime.register_frame

        def capture(**kwargs):
            runtime.last_arguments = kwargs
            return original_register(**kwargs)

        runtime.register_frame = capture
        estimator = FoundationPoseEstimator(
            "", runtime=runtime, mesh_paths={"can": __file__},
            roi_padding_pixels=2,
        )
        color = np.zeros((20, 30, 3), dtype=np.uint8)
        frame = FrameData(
            color_bgr=color,
            camera_matrix=np.asarray(
                [[100.0, 0.0, 15.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]
            ),
            distortion=np.zeros(5), timestamp_s=1.0, frame_id="camera_color",
            depth_m=np.ones((20, 30), dtype=np.float32),
            depth_aligned_to_color=True,
        )
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:15, 10:20] = 1
        detection = DetectionResult((10, 5, 20, 15), 0, "can", 0.9, mask)
        result = estimator.estimate_detection(frame, detection)
        self.assertTrue(result.valid, result.reason)
        arguments = runtime.last_arguments
        self.assertEqual(arguments["rgb"].shape[:2], (14, 14))
        self.assertEqual(arguments["depth_m"].shape, (14, 14))
        self.assertEqual(arguments["mask"].shape, (14, 14))
        self.assertAlmostEqual(arguments["camera_matrix"][0, 2], 7.0)
        self.assertAlmostEqual(arguments["camera_matrix"][1, 2], 7.0)


if __name__ == "__main__":
    unittest.main()
