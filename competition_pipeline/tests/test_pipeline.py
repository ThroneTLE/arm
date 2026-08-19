import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import yaml

from competition_pipeline.configuration import CompetitionConfig, load_camera_intrinsics
from competition_pipeline.control import MotionSafetyError, SafeRobotController
from competition_pipeline.geometry import transform_from_xyz_rpy
from competition_pipeline.hand_eye import HandEyeCalibrator
from competition_pipeline.localization import (
    HybridLocalizer, LocalizationResult, SOURCE_TAG_VISUAL,
    SOURCE_TAG_VISUAL_HELD, SOURCE_TCP_FALLBACK,
)
from competition_pipeline.oak_calibration import import_oak_calibration
from competition_pipeline.object_localization import (
    ObjectCloudSettings, localize_segmented_instance,
)
from competition_pipeline.interfaces import RobotPoseSample
from competition_pipeline.planning import FallbackGraspPlanner, planner_from_config
from competition_pipeline.sample_store import HandEyeSampleStore
from competition_pipeline.segmentation_validation import (
    SegmentationModel, evaluate_mask_result,
)
from competition_pipeline.rgbd_calibration import (
    calibration_for_depth_frame, load_rgbd_result,
)
from competition_pipeline.tag_map import TagMap
from tool.object_model_builder.yolo_segmenter import MaskResult
from tool.object_model_builder.rgbd_geometry import CameraIntrinsics, RgbdCalibration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "competition.yaml"


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temporary.name) / "competition.yaml"
        data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        data["tag_map"]["tags"] = {
            7: {"bottom_right_xyz_mm": [500.0, 100.0, 0.0]}
        }
        self.config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        self.config = CompetitionConfig(self.config_path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_grasp_planner_keeps_anygrasp_lazy_as_fallback(self):
        planner = planner_from_config(self.config)
        self.assertIsInstance(planner, FallbackGraspPlanner)
        self.assertIsNotNone(planner.fallback)
        self.assertIsNone(planner.fallback._planner)

    def test_bottom_right_corner_convention(self):
        corners = TagMap(self.config).corners_base_m(7)
        size = float(self.config.tag_map["tag_size_mm"]) / 1000.0
        expected = np.asarray(
            [
                [0.50 + size, 0.10 + size, 0.0],
                [0.50 + size, 0.10, 0.0],
                [0.50, 0.10, 0.0],
                [0.50, 0.10 + size, 0.0],
            ]
        )
        np.testing.assert_allclose(corners, expected, atol=1e-12)
        tag_x = corners[1] - corners[2]  # BR -> TR
        tag_y = corners[3] - corners[2]  # BR -> BL
        np.testing.assert_allclose(np.cross(tag_x, tag_y), [0.0, 0.0, size ** 2])

    def test_camera_profile_switch_invalidates_hand_eye(self):
        self.assertEqual(self.config.active_camera_profile, "astra_validation")
        self.assertEqual(self.config.camera["backend"], "astra_ros")
        self.config.data["hand_eye"]["tcp_from_color_camera"]["valid"] = True
        self.config.save()
        self.assertTrue(self.config.set_active_camera_profile("oak_competition"))
        loaded = CompetitionConfig(self.config_path)
        self.assertEqual(loaded.active_camera_profile, "oak_competition")
        self.assertEqual(loaded.camera["backend"], "oak_depthai")
        self.assertFalse(loaded.hand_eye_valid)

    def test_camera_profile_switch_invalidates_segmentation_validation(self):
        settings = self.config.data["segmentation_validation"]
        settings["validation"] = {
            "valid": True,
            "weights_sha256": "abc",
            "weights_file": settings["weights_file"],
            "camera_profile": "astra_validation",
            "confirmed_at": "test",
        }
        self.assertTrue(self.config.segmentation_valid)
        self.config.set_active_camera_profile("oak_competition")
        self.assertFalse(self.config.segmentation_valid)

    def test_astra_depth_mode_is_applied_to_runtime_driver(self):
        active_name = self.config.camera["depth_mode"]
        active = self.config.camera["depth_modes"][active_name]
        arguments = self.config.runtime_camera()["ros_driver"]["arguments"]
        self.assertEqual(
            (arguments["depth_width"], arguments["depth_height"], arguments["depth_fps"]),
            (active["depth_width"], active["depth_height"], active["depth_fps"]),
        )
        next_name = (
            "smooth_640x480"
            if active_name != "smooth_640x480" else "detail_1280x1024"
        )
        self.assertTrue(
            self.config.set_active_depth_mode(next_name)
        )
        runtime = CompetitionConfig(self.config_path).runtime_camera()
        selected = self.config.camera["depth_modes"][next_name]
        arguments = runtime["ros_driver"]["arguments"]
        self.assertEqual(
            (arguments["depth_width"], arguments["depth_height"], arguments["depth_fps"]),
            (selected["depth_width"], selected["depth_height"], selected["depth_fps"]),
        )

    def test_legacy_astra_stream_arguments_are_migrated_to_mode(self):
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        camera = data["camera"]["profiles"]["astra_validation"]
        camera.pop("depth_mode")
        camera.pop("depth_modes")
        camera["ros_driver"]["arguments"].update({
            "depth_width": 1280,
            "depth_height": 1024,
            "depth_fps": 7,
            "ir_width": 1280,
            "ir_height": 1024,
            "ir_fps": 30,
        })
        self.config_path.write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        migrated = CompetitionConfig(self.config_path)
        self.assertEqual(
            migrated.camera["depth_mode"], "detail_1280x1024"
        )

    def test_runtime_depth_mode_uses_matching_live_camera_info(self):
        color = CameraIntrinsics(
            1280, 720,
            np.asarray([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
        )
        calibrated_depth = CameraIntrinsics(
            1280, 1024,
            np.asarray([[1000.0, 0.0, 640.0], [0.0, 1000.0, 512.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
        )
        live_depth = CameraIntrinsics(
            640, 480,
            np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
        )
        calibration = RgbdCalibration(
            color=color,
            depth=calibrated_depth,
            color_from_depth=np.eye(4),
            valid=True,
        )
        runtime = calibration_for_depth_frame(
            calibration, live_depth, (480, 640)
        )
        self.assertIs(runtime.depth, live_depth)
        np.testing.assert_allclose(
            runtime.color_from_depth, calibration.color_from_depth
        )
        with self.assertRaisesRegex(ValueError, "CameraInfo"):
            calibration_for_depth_frame(calibration, None, (480, 640))

    def test_unsafe_segmentation_candidate_settings_are_rejected(self):
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        data["segmentation_validation"]["confidence_threshold"] = 0.0
        self.config_path.write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "confidence_threshold"):
            CompetitionConfig(self.config_path)
        data["segmentation_validation"]["confidence_threshold"] = 0.25
        data["segmentation_validation"]["duplicate_confidence_tie_margin"] = 1.0
        self.config_path.write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "tie_margin"):
            CompetitionConfig(self.config_path)

    def test_segmentation_mask_quality_gate(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:7, 3:8] = 1
        result = MaskResult(
            True, mask=mask, class_name="test_object", confidence=0.82,
            reason="accepted",
        )
        quality = evaluate_mask_result(
            result, (10, 10, 3), minimum_confidence=0.5,
            minimum_mask_area_ratio=0.05, maximum_mask_area_ratio=0.8,
        )
        self.assertTrue(quality.valid, quality.reason)
        self.assertAlmostEqual(quality.mask_area_ratio, 0.25)
        low_confidence = evaluate_mask_result(
            MaskResult(True, mask=mask, class_name="test_object", confidence=0.2),
            (10, 10, 3), minimum_confidence=0.5,
            minimum_mask_area_ratio=0.05, maximum_mask_area_ratio=0.8,
        )
        self.assertFalse(low_confidence.valid)

    def test_segmentation_rejects_background_sized_mask(self):
        quality = evaluate_mask_result(
            MaskResult(
                True, mask=np.ones((10, 10), dtype=np.uint8),
                class_name="background", confidence=0.9,
            ),
            (10, 10, 3), minimum_confidence=0.5,
            minimum_mask_area_ratio=0.01, maximum_mask_area_ratio=0.8,
        )
        self.assertFalse(quality.valid)
        self.assertIn("背景", quality.reason)

    def test_segmentation_overlay_excludes_instances_rejected_by_quality_gate(self):
        valid = np.zeros((10, 10), dtype=np.uint8)
        valid[2:6, 3:7] = 1
        background = np.ones((10, 10), dtype=np.uint8)
        low_confidence = valid.copy()

        class FakeProvider:
            last_model_instance_count = 3
            last_suppressed_instance_count = 0
            last_reason = ""

            def predict_all(self, image):
                return [
                    MaskResult(True, mask=background, class_name="bad", confidence=0.9),
                    MaskResult(True, mask=valid, class_name="can", confidence=0.8),
                    MaskResult(True, mask=low_confidence, class_name="weak", confidence=0.1),
                ]

            def overlay_many(self, image, instances):
                self.overlaid = list(instances)
                return np.asarray(image).copy()

        model = object.__new__(SegmentationModel)
        model.provider = FakeProvider()
        result = model.predict(
            np.zeros((10, 10, 3), dtype=np.uint8),
            {
                "minimum_confidence": 0.25,
                "minimum_mask_area_ratio": 0.01,
                "maximum_mask_area_ratio": 0.8,
            },
        )
        self.assertEqual([item.class_name for item in result[4]], ["can"])
        self.assertEqual([item.class_name for item in model.provider.overlaid], ["can"])
        self.assertEqual(result[5]["quality_rejected"], 2)

    def test_segmented_depth_cloud_is_transformed_to_robot_base(self):
        depth = np.zeros((10, 10), dtype=np.float32)
        depth[2:8, 3:7] = 1.0
        mask = depth > 0.0
        intrinsics = CameraIntrinsics(
            10, 10,
            np.asarray([[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
        )
        base_from_camera = np.eye(4, dtype=np.float64)
        base_from_camera[:3, 3] = [0.4, -0.2, 0.1]
        result = localize_segmented_instance(
            depth,
            MaskResult(True, mask=mask, class_name="yellow_can", confidence=0.9),
            intrinsics,
            base_from_camera,
            ObjectCloudSettings(
                minimum_valid_points=10,
                minimum_depth_coverage=0.8,
                mask_erosion_pixels=0,
                workspace_min_m=(-2.0, -2.0, -2.0),
                workspace_max_m=(2.0, 2.0, 2.0),
            ),
        )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.class_name, "yellow_can")
        self.assertAlmostEqual(result.center_base_m[0], 0.395, places=6)
        self.assertAlmostEqual(result.center_base_m[1], -0.205, places=6)
        self.assertAlmostEqual(result.center_base_m[2], 1.1, places=6)

    def test_segmented_depth_cloud_rejects_sparse_depth(self):
        depth = np.zeros((10, 10), dtype=np.float32)
        depth[5, 5] = 1.0
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        intrinsics = CameraIntrinsics(10, 10, np.eye(3), np.zeros(5))
        result = localize_segmented_instance(
            depth,
            MaskResult(True, mask=mask, class_name="object", confidence=0.8),
            intrinsics,
            np.eye(4),
            ObjectCloudSettings(
                minimum_valid_points=2,
                minimum_depth_coverage=0.5,
                mask_erosion_pixels=0,
            ),
        )
        self.assertFalse(result.valid)
        self.assertIn("coverage", result.reason)

    def test_supported_object_uses_grounded_bounds_without_moving_raw_points(self):
        depth = np.zeros((10, 10), dtype=np.float32)
        depth[2:8, 3:7] = 1.0
        mask = depth > 0.0
        intrinsics = CameraIntrinsics(
            10, 10,
            np.asarray([[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
        )
        result = localize_segmented_instance(
            depth,
            MaskResult(True, mask=mask, class_name="bottle", confidence=0.9),
            intrinsics,
            np.eye(4),
            ObjectCloudSettings(
                minimum_valid_points=10,
                minimum_depth_coverage=0.8,
                mask_erosion_pixels=0,
                assume_supported_objects=True,
                support_plane_z_m=0.9,
                maximum_support_gap_m=0.15,
            ),
        )
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.support_constrained)
        self.assertAlmostEqual(result.bounds_min_base_m[2], 0.9)
        self.assertAlmostEqual(result.center_base_m[2], 0.95)
        # Support constraints must never translate or invent measured points.
        self.assertTrue(np.allclose(result.points_base_m[:, 2], 1.0))
        self.assertAlmostEqual(result.observed_bounds_min_base_m[2], 1.0)

    def test_oak_profile_starts_without_fake_calibration(self):
        self.config.set_active_camera_profile("oak_competition")
        self.assertFalse(
            self.config.resolve_path(self.config.camera["color_intrinsics_file"]).exists()
        )
        self.assertFalse(
            self.config.resolve_path(self.config.camera["factory_calibration_file"]).exists()
        )

    def test_import_depthai_json_writes_canonical_rgb_yaml(self):
        try:
            import depthai as dai
        except ImportError:
            self.skipTest("depthai is not installed")
        source = Path(self.temporary.name) / "depthai_calibration.json"
        output_json = Path(self.temporary.name) / "saved.json"
        output_yaml = Path(self.temporary.name) / "oak_color.yaml"
        calibration = dai.CalibrationHandler()
        identity = np.eye(3).tolist()
        matrix = [[910.0, 0.0, 640.0], [0.0, 912.0, 360.0], [0.0, 0.0, 1.0]]
        for socket in (
            dai.CameraBoardSocket.CAM_A,
            dai.CameraBoardSocket.CAM_B,
            dai.CameraBoardSocket.CAM_C,
        ):
            calibration.setCameraIntrinsics(socket, matrix, 1280, 720)
            calibration.setDistortionCoefficients(socket, [0.1, -0.2, 0.01, 0.02, 0.03])
        calibration.setBoardInfo("OAK-D-PRO", "R3M1E3")
        calibration.setCameraExtrinsics(
            dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C,
            identity, [-7.5, 0.0, 0.0], [-7.5, 0.0, 0.0],
        )
        calibration.setCameraExtrinsics(
            dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A,
            identity, [3.75, 0.0, 0.0], [3.75, 0.0, 0.0],
        )
        calibration.setStereoLeft(dai.CameraBoardSocket.CAM_B, identity)
        calibration.setStereoRight(dai.CameraBoardSocket.CAM_C, identity)
        calibration.eepromToJsonFile(str(source))
        info = import_oak_calibration(
            source, output_json, output_yaml, width=1280, height=720
        )
        imported, distortion, size = load_camera_intrinsics(output_yaml)
        self.assertTrue(output_json.is_file())
        self.assertEqual(size, (1280, 720))
        self.assertAlmostEqual(info["baseline_mm"], 75.0)
        np.testing.assert_allclose(imported, matrix)
        self.assertEqual(distortion.shape, (5, 1))

    def _project_detection(self, base_from_camera):
        matrix = np.asarray([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
        distortion = np.zeros((5, 1), dtype=np.float64)
        camera_from_base = np.linalg.inv(base_from_camera)
        rvec = cv2.Rodrigues(camera_from_base[:3, :3])[0]
        pixels, _ = cv2.projectPoints(
            TagMap(self.config).corners_base_m(7), rvec, camera_from_base[:3, 3], matrix, distortion
        )
        return {7: pixels.reshape(4, 2)}, matrix, distortion

    def test_visual_pose_uses_tag_before_tcp(self):
        expected = transform_from_xyz_rpy([0.46, 0.13, 0.8], [180.0, 0.0, 0.0])
        detections, matrix, distortion = self._project_detection(expected)
        localizer = HybridLocalizer(self.config)
        localizer.visual.detect = lambda image: detections
        result = localizer.localize(
            np.zeros((720, 1280, 3), dtype=np.uint8), matrix, distortion,
            base_from_tcp=np.eye(4), image_timestamp_s=5.0, robot_timestamp_s=5.0,
        )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.source, SOURCE_TAG_VISUAL)
        np.testing.assert_allclose(result.base_from_camera, expected, atol=1e-6)

    def test_recent_multi_tag_pose_prevents_single_tag_jump_and_one_bad_frame(self):
        localizer = HybridLocalizer(self.config)
        stable = np.eye(4)
        stable[0, 3] = 0.40
        jumped = np.eye(4)
        jumped[0, 3] = 0.44
        results = iter((
            LocalizationResult(
                True, stable, SOURCE_TAG_VISUAL, 1.0,
                visible_tag_ids=(100, 102), used_tag_ids=(100, 102),
                rms_reprojection_error_px=2.1,
            ),
            LocalizationResult(
                True, jumped, SOURCE_TAG_VISUAL, 1.2,
                visible_tag_ids=(100,), used_tag_ids=(100,),
                rms_reprojection_error_px=0.3,
            ),
            LocalizationResult(
                False, None, "unavailable", 1.4,
                visible_tag_ids=(100, 102), used_tag_ids=(100, 102),
                rms_reprojection_error_px=3.5,
                reason="quality rejected",
            ),
        ))
        localizer.visual.estimate = lambda *args, **kwargs: next(results)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        first = localizer.localize(image, np.eye(3), np.zeros(5), image_timestamp_s=1.0)
        second = localizer.localize(image, np.eye(3), np.zeros(5), image_timestamp_s=1.2)
        third = localizer.localize(image, np.eye(3), np.zeros(5), image_timestamp_s=1.4)
        self.assertEqual(first.source, SOURCE_TAG_VISUAL)
        self.assertEqual(second.source, SOURCE_TAG_VISUAL_HELD)
        self.assertEqual(third.source, SOURCE_TAG_VISUAL_HELD)
        np.testing.assert_allclose(second.base_from_camera, stable)
        np.testing.assert_allclose(third.base_from_camera, stable)

    def test_tcp_fallback_requires_valid_hand_eye(self):
        self.config.data["hand_eye"]["tcp_from_color_camera"].update(
            {"valid": True, "matrix": transform_from_xyz_rpy([0.0, 0.0, 0.2], [0, 0, 0]).tolist()}
        )
        self.config.save()
        base_from_tcp = transform_from_xyz_rpy([0.4, 0.1, 0.6], [0, 0, 0])
        localizer = HybridLocalizer(self.config)
        localizer.visual.detect = lambda image: {}
        result = localizer.localize(
            np.zeros((10, 10, 3), dtype=np.uint8), np.eye(3), np.zeros(5),
            base_from_tcp, image_timestamp_s=8.0, robot_timestamp_s=8.0,
        )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.source, SOURCE_TCP_FALLBACK)
        np.testing.assert_allclose(result.base_from_camera[:3, 3], [0.4, 0.1, 0.8])

    def test_hand_eye_aggregation_rejects_one_outlier(self):
        self.config.data["hand_eye"]["minimum_samples"] = 5
        expected = transform_from_xyz_rpy([0.04, -0.02, 0.16], [3.0, -2.0, 5.0])
        calibrator = HandEyeCalibrator(self.config)
        for index in range(5):
            base_from_tcp = transform_from_xyz_rpy(
                [0.3 + index * 0.03, -0.1 + index * 0.02, 0.5 + index * 0.01],
                [index * 4.0, -index * 2.0, index * 6.0],
            )
            calibrator.add_sample(base_from_tcp, base_from_tcp @ expected, [7], 0.1)
        outlier_tcp = transform_from_xyz_rpy([0.2, 0.1, 0.4], [0, 0, 0])
        outlier = transform_from_xyz_rpy([0.12, -0.02, 0.16], [15.0, 0.0, 0.0])
        calibrator.add_sample(outlier_tcp, outlier_tcp @ outlier, [7], 0.1)
        result = calibrator.solve()
        self.assertEqual(len(result.inlier_indices), 5)
        np.testing.assert_allclose(result.tcp_from_camera, expected, atol=1e-9)

    def test_tag_update_invalidates_hand_eye(self):
        self.config.data["hand_eye"]["tcp_from_color_camera"]["valid"] = True
        self.config.save()
        TagMap(self.config).set_tag(8, [600.0, 100.0, 0.0])
        self.assertFalse(CompetitionConfig(self.config_path).hand_eye_valid)

    def test_load_pipeline_color_intrinsics(self):
        calibration = Path(self.temporary.name) / "color.yaml"
        calibration.write_text(
            yaml.safe_dump(
                {
                    "cameras": {
                        "color": {
                            "image_width": 1280,
                            "image_height": 720,
                            "camera_matrix": np.eye(3).tolist(),
                            "distortion_coefficients": [0, 0, 0, 0, 0],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        matrix, distortion, size = load_camera_intrinsics(calibration)
        np.testing.assert_allclose(matrix, np.eye(3))
        self.assertEqual(distortion.shape, (5, 1))
        self.assertEqual(size, (1280, 720))

    def test_invalid_legacy_rgbd_result_is_rejected(self):
        calibration = ROOT.parent / "tool" / "object_model_builder" / "output" / "rgbd_calibration.yaml"
        if not calibration.exists():
            self.skipTest("legacy RGB-D output is not present")
        with self.assertRaises(ValueError):
            load_rgbd_result(calibration, maximum_rms_px=2.0)

    def test_hand_eye_sample_store_rejects_changed_tag_map(self):
        path = Path(self.temporary.name) / "samples.yaml"
        store = HandEyeSampleStore(path, self.config)
        sample = SimpleNamespace(
            base_from_tcp=np.eye(4),
            base_from_camera=np.eye(4),
            visible_tag_ids=(7,),
            rms_reprojection_error_px=0.2,
        )
        self.assertEqual(store.append(sample, "test.png"), 1)
        self.config.data["tag_map"]["tags"][8] = {
            "bottom_right_xyz_mm": [700.0, 100.0, 0.0]
        }
        with self.assertRaisesRegex(ValueError, "Tag map changed"):
            store.load()

    def test_hand_eye_sample_store_rejects_other_camera_profile(self):
        path = Path(self.temporary.name) / "samples.yaml"
        store = HandEyeSampleStore(path, self.config)
        sample = SimpleNamespace(
            base_from_tcp=np.eye(4),
            base_from_camera=np.eye(4),
            visible_tag_ids=(7,),
            rms_reprojection_error_px=0.2,
        )
        store.append(sample, "astra.png")
        self.config.set_active_camera_profile("oak_competition")
        with self.assertRaisesRegex(ValueError, "Camera profile changed"):
            HandEyeSampleStore(path, self.config).load()

    def test_robot_control_is_fail_closed(self):
        class Adapter:
            moved = False

            def latest_pose(self):
                return RobotPoseSample(np.eye(4), time.monotonic())

            def move_tcp(self, target, speed_scale):
                self.moved = True

            def stop(self):
                return "stopped"

        adapter = Adapter()
        controller = SafeRobotController(self.config, adapter)
        with self.assertRaises(MotionSafetyError):
            controller.move_tcp(transform_from_xyz_rpy([0.01, 0, 0], [0, 0, 0]), 0.1)
        self.assertFalse(adapter.moved)
        self.assertEqual(controller.stop(), "stopped")

    def test_robot_control_checks_jump_after_enable(self):
        class Adapter:
            def latest_pose(self):
                return RobotPoseSample(np.eye(4), time.monotonic())

            def move_tcp(self, target, speed_scale):
                raise AssertionError("unsafe target reached adapter")

            def stop(self):
                pass

        self.config.data["safety"]["dry_run"] = False
        self.config.data["safety"]["allow_robot_motion"] = True
        controller = SafeRobotController(self.config, Adapter())
        with self.assertRaisesRegex(MotionSafetyError, "jump"):
            controller.move_tcp(transform_from_xyz_rpy([0.2, 0, 0], [0, 0, 0]), 0.1)


if __name__ == "__main__":
    unittest.main()
