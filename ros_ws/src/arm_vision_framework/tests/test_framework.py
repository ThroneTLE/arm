#!/usr/bin/env python3

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arm_vision_framework.adapters.mock import MockRobotController
from arm_vision_framework.adapters.topic_robot import TopicRobotController
from arm_vision_framework.errors import ConfigurationError, SafetyInterlockError
from arm_vision_framework.factory import build_pipeline
from arm_vision_framework.localization import (
    HybridCameraLocalizer,
    SOURCE_TAG_VISUAL,
    _tag_world_corners,
)
from arm_vision_framework.parameters import CalibrationStore, load_system_parameters
from arm_vision_framework.transforms import (
    quaternion_from_transform,
    transform_from_quaternion,
    transform_from_xyz_rpy,
)
from arm_vision_framework.types import FrameData


CALIBRATION_PATH = PACKAGE_ROOT / "config" / "calibration_parameters.yaml"
SYSTEM_PATH = PACKAGE_ROOT / "config" / "system_parameters.yaml"
LEGACY_RUNTIME_PATH = (
    PACKAGE_ROOT.parents[2]
    / "tool"
    / "camera_calibration"
    / "calibration_snapshots"
    / "latest"
    / "runtime_calibration.yaml"
)


def blank_frame(calibration, timestamp=None):
    width, height = calibration.image_size
    return FrameData(
        np.zeros((height, width, 3), dtype=np.uint8),
        calibration.camera_matrix,
        calibration.distortion,
        time.monotonic() if timestamp is None else float(timestamp),
        calibration.data["frames"]["camera_color"],
    )


class FrameworkTest(unittest.TestCase):
    def setUp(self):
        self.calibration = CalibrationStore(CALIBRATION_PATH)
        self.settings = load_system_parameters(SYSTEM_PATH)

    def test_central_calibration_is_safe_by_default(self):
        self.assertEqual(self.calibration.image_size, (1280, 720))
        self.assertFalse(self.calibration.transform_valid("workspace_from_base"))
        self.assertFalse(self.calibration.transform_valid("gripper_from_camera"))
        self.assertFalse(self.calibration.depth_aligned_to_color)
        self.assertEqual(
            self.calibration.tag_map["coordinate_convention"]["id"],
            "tag_top_left_x_right_y_down_v1",
        )
        self.assertFalse(self.calibration.data["fixed_camera_validation_reference"]["valid"])

    def test_quaternion_transform_round_trip(self):
        original = transform_from_xyz_rpy([0.2, -0.1, 0.8], [15.0, -25.0, 40.0])
        quaternion = quaternion_from_transform(original)
        recovered = transform_from_quaternion(original[:3, 3], quaternion)
        np.testing.assert_allclose(recovered, original, atol=1e-10)

    def test_mock_pipeline_composes_workspace_object_pose(self):
        pipeline = build_pipeline(self.settings, self.calibration)
        result = pipeline.process(blank_frame(self.calibration))
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.simulated)
        self.assertEqual(result.camera_localization.source, "simulated_robot")
        self.assertAlmostEqual(result.workspace_from_object[2, 3], 0.2, places=9)

    def test_visual_tag_pose_has_priority(self):
        localizer = HybridCameraLocalizer(self.calibration)
        rvec = np.asarray([[2.85], [0.18], [-0.12]], dtype=np.float64)
        tvec = np.asarray([[0.02], [-0.03], [0.72]], dtype=np.float64)
        detections = {}
        tag_map = self.calibration.tag_map
        for tag_id, entry in tag_map["tags"].items():
            corners = _tag_world_corners(
                np.asarray(entry["origin_mm"], dtype=np.float64) / 1000.0,
                entry["yaw_deg"],
                float(tag_map["tag_size_mm"]) / 1000.0,
            )
            pixels, _ = cv2.projectPoints(
                corners,
                rvec,
                tvec,
                self.calibration.camera_matrix,
                self.calibration.distortion,
            )
            detections[int(tag_id)] = pixels.reshape(4, 2)
        localizer.visual.detect = lambda image: detections
        frame = blank_frame(self.calibration)
        result = localizer.localize(frame, MockRobotController().read_state(frame.timestamp_s))
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.source, SOURCE_TAG_VISUAL)
        self.assertFalse(result.simulated)
        self.assertLess(result.rms_reprojection_error_px, 1e-6)

    def test_real_robot_fallback_rejected_without_hand_eye(self):
        frame = blank_frame(self.calibration)
        robot = TopicRobotController()
        robot.update_state(np.eye(4), frame.timestamp_s)
        result = HybridCameraLocalizer(self.calibration).localize(
            frame, robot.read_state()
        )
        self.assertFalse(result.valid)
        self.assertIn("not calibrated", result.reason)

    def test_robot_motion_interlock_is_closed(self):
        robot = MockRobotController(allow_motion=False)
        with self.assertRaises(SafetyInterlockError):
            robot.move_to(np.eye(4))

    def test_hand_eye_parameter_can_be_promoted_in_copy(self):
        sys.path.insert(0, str(PACKAGE_ROOT / "tools"))
        import calibration_tool

        with tempfile.TemporaryDirectory() as directory:
            parameter_copy = Path(directory) / "calibration.yaml"
            hand_eye_file = Path(directory) / "eye_in_hand.yaml"
            shutil.copy2(CALIBRATION_PATH, parameter_copy)
            hand_eye_file.write_text(
                yaml.safe_dump({"gripper_from_camera": {"matrix": np.eye(4).tolist()}}),
                encoding="utf-8",
            )
            destination, backup = calibration_tool.import_hand_eye(
                parameter_copy, hand_eye_file
            )
            updated = CalibrationStore(destination)
            self.assertTrue(updated.transform_valid("gripper_from_camera"))
            self.assertTrue(backup.is_file())

    def test_legacy_center_tag_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parameter_copy = Path(directory) / "legacy_calibration.yaml"
            data = yaml.safe_load(CALIBRATION_PATH.read_text(encoding="utf-8"))
            data["tag_map"].pop("coordinate_convention")
            for entry in data["tag_map"]["tags"].values():
                entry["center_mm"] = entry.pop("origin_mm")
            parameter_copy.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                CalibrationStore(parameter_copy)

    def test_sync_camera_imports_top_left_tag_origins_only(self):
        sys.path.insert(0, str(PACKAGE_ROOT / "tools"))
        import calibration_tool

        with tempfile.TemporaryDirectory() as directory:
            parameter_copy = Path(directory) / "calibration.yaml"
            shutil.copy2(CALIBRATION_PATH, parameter_copy)
            destination, _ = calibration_tool.sync_camera(
                parameter_copy,
                LEGACY_RUNTIME_PATH,
            )
            updated = CalibrationStore(destination)
            self.assertIn("origin_mm", updated.tag_map["tags"][100])
            self.assertNotIn("center_mm", updated.tag_map["tags"][100])
            self.assertFalse(updated.data["fixed_camera_validation_reference"]["valid"])


if __name__ == "__main__":
    unittest.main()
