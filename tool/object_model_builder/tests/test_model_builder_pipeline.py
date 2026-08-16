#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import numpy as np

from tool.object_model_builder.camera_source import FrameBundle
from tool.object_model_builder.model_builder_ui import ModelBuilderApp
from tool.object_model_builder.rgbd_geometry import CameraIntrinsics
from tool.object_model_builder.yolo_segmenter import MaskResult


def camera_intrinsics(width=4, height=3):
    return CameraIntrinsics(
        width,
        height,
        np.asarray(
            [[100.0, 0.0, 1.5], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        np.zeros(5),
    )


class FakeTagProvider:
    def __init__(self):
        self.calls = 0

    def estimate(self, _color, _matrix, _distortion):
        self.calls += 1
        return SimpleNamespace(valid=True), {100: np.zeros((4, 2))}


class FakeYoloProvider:
    def __init__(self):
        self.calls = 0

    def predict(self, color):
        self.calls += 1
        return MaskResult(True, mask=np.ones(color.shape[:2], dtype=bool))


class FakeDepthAligner:
    def __init__(self):
        self.calls = 0

    def align(self, depth):
        self.calls += 1
        return np.full((3, 4), float(np.mean(depth)), dtype=np.float32)


class FakeValue:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = str(value)


class CaptureAnalysisPipelineTests(unittest.TestCase):
    def test_reuses_alignment_for_repeated_depth_while_rgb_analysis_continues(self):
        app = ModelBuilderApp.__new__(ModelBuilderApp)
        app._analysis_cache_generation = None
        app._analysis_cached_depth_timestamp_s = None
        app._analysis_cached_aligned_depth = None
        app._analysis_cached_depth_preview_bgr = None
        app._analysis_cached_aligned_preview_bgr = None
        app._analysis_cached_device_calibration = None
        app._analysis_cached_mask_result = MaskResult(
            False, reason="not analyzed"
        )
        app._analysis_last_yolo_monotonic_s = 0.0
        app._yolo_preview_interval_s = 60.0
        app.tag_provider = FakeTagProvider()
        app.yolo_provider = FakeYoloProvider()
        app.depth_aligner = FakeDepthAligner()
        app.mask_result = MaskResult(False, reason="not analyzed")
        app.color_intrinsics = camera_intrinsics()
        depth = np.ones((2, 2), dtype=np.float32)
        first = FrameBundle(
            color_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
            color_timestamp_s=1.0,
            depth_m=depth,
            depth_timestamp_s=0.98,
            depth_intrinsics=None,
            ir_image=None,
            ir_timestamp_s=None,
        )
        second = FrameBundle(
            color_bgr=np.full((3, 4, 3), 20, dtype=np.uint8),
            color_timestamp_s=1.03,
            depth_m=depth,
            depth_timestamp_s=0.98,
            depth_intrinsics=None,
            ir_image=None,
            ir_timestamp_s=None,
        )
        intrinsics = camera_intrinsics()
        first_result = app._analyze_capture_frame(
            3, first, first.color_bgr, intrinsics
        )
        second_result = app._analyze_capture_frame(
            3, second, second.color_bgr, intrinsics
        )
        self.assertEqual(app.tag_provider.calls, 2)
        self.assertEqual(app.yolo_provider.calls, 1)
        self.assertEqual(app.depth_aligner.calls, 1)
        self.assertIs(first_result.aligned_depth, second_result.aligned_depth)
        self.assertTrue(second_result.mask_result.valid)

    def test_capture_button_waits_for_a_later_valid_pair_without_popup(self):
        app = ModelBuilderApp.__new__(ModelBuilderApp)
        app.capture_session = object()
        app.capture_config = {"capture_request_timeout_s": 5.0}
        app._capture_request_pending = False
        app._capture_request_deadline_s = 0.0
        app._capture_request_last_reason = ""
        app._capture_feedback_text = ""
        app._capture_feedback_until_s = 0.0
        app._last_captured_view_index = None
        app.capture_status_text = FakeValue()
        app.status_text = FakeValue()
        attempts = []

        def save():
            attempts.append(True)
            if len(attempts) == 1:
                raise ValueError("等待下一组同步深度（当前 160 ms）")
            return 0, 0.82, 0.03

        app._save_current_capture_view = save
        app._capture_view()
        self.assertTrue(app._capture_request_pending)
        self.assertIn("等待下一组同步深度", app.capture_status_text.value)
        app._try_pending_capture()
        self.assertFalse(app._capture_request_pending)
        self.assertEqual(app._last_captured_view_index, 0)
        self.assertIn("已拍摄第 1 张", app.capture_status_text.value)
        self.assertEqual(len(attempts), 2)


if __name__ == "__main__":
    unittest.main()
