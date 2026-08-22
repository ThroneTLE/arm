#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tool.object_model_builder.camera_source import (
    AstraRosSource, OakDProSource,
    OrbbecRosSource,
    native_ros_environment,
    nearest_timestamped_frame,
    ros_image_to_numpy,
)


def image_message(array, encoding, step=None, is_bigendian=False):
    array = np.asarray(array)
    return SimpleNamespace(
        encoding=encoding,
        is_bigendian=is_bigendian,
        height=array.shape[0],
        width=array.shape[1],
        step=step if step is not None else array.strides[0],
        data=array.tobytes(),
    )


class RosImageDecoderTests(unittest.TestCase):
    def test_decodes_little_endian_depth(self):
        source = np.asarray([[0, 1000], [1500, 2000]], dtype="<u2")
        decoded = ros_image_to_numpy(image_message(source, "16UC1"))
        np.testing.assert_array_equal(decoded, source)
        self.assertEqual(decoded.dtype, np.uint16)

    def test_decodes_big_endian_ir(self):
        source = np.asarray([[12, 345], [678, 901]], dtype=">u2")
        decoded = ros_image_to_numpy(
            image_message(source, "mono16", is_bigendian=True)
        )
        np.testing.assert_array_equal(decoded, source.astype(np.uint16))
        self.assertEqual(decoded.dtype, np.uint16)

    def test_ignores_row_padding(self):
        rows = np.asarray(
            [[1, 2, 99], [3, 4, 99]], dtype=np.uint16
        )
        message = SimpleNamespace(
            encoding="16UC1",
            is_bigendian=False,
            height=2,
            width=2,
            step=6,
            data=rows.tobytes(),
        )
        decoded = ros_image_to_numpy(message)
        np.testing.assert_array_equal(decoded, [[1, 2], [3, 4]])

    def test_rejects_unknown_encoding(self):
        message = image_message(np.zeros((2, 2), dtype=np.uint16), "12UC1")
        with self.assertRaisesRegex(ValueError, "不支持"):
            ros_image_to_numpy(message)


class RosDriverEnvironmentTests(unittest.TestCase):
    def test_removes_conda_libraries_but_keeps_ros_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conda_prefix = root / "conda" / "envs" / "foundationpose"
            conda_library = conda_prefix / "lib"
            stale_conda_library = root / "anaconda3" / "envs" / "old" / "lib"
            ros_library = root / "orbbec_ws" / "devel" / "lib"
            (conda_prefix / "conda-meta").mkdir(parents=True)
            conda_library.mkdir()
            ros_library.mkdir(parents=True)
            environment = native_ros_environment(
                {
                    "LD_LIBRARY_PATH": "{}:{}:{}".format(
                        conda_library, stale_conda_library, ros_library
                    ),
                    "CONDA_PREFIX": str(conda_prefix),
                }
            )
            self.assertEqual(environment["LD_LIBRARY_PATH"], str(ros_library))


class TimestampPairingTests(unittest.TestCase):
    def test_oak_defaults_match_the_active_camera_profile(self):
        source = OakDProSource()
        self.assertEqual((source.color_width, source.color_height), (1920, 1080))
        self.assertEqual(source.fps, 10)

    def test_selects_nearest_timestamped_frame(self):
        timestamp, frame = nearest_timestamped_frame(
            ((1.0, "old"), (1.1, "near"), (1.3, "new")), 1.16
        )
        self.assertEqual(timestamp, 1.1)
        self.assertEqual(frame, "near")

    def test_depth_anchor_pairs_the_nearest_rgb_frame(self):
        source = AstraRosSource("/dev/null", start_ros_driver=False)
        color_old = np.full((2, 2, 3), 10, dtype=np.uint8)
        color_near = np.full((2, 2, 3), 20, dtype=np.uint8)
        depth = np.full((2, 2), 1.0, dtype=np.float32)
        infrared = np.full((2, 2), 80, dtype=np.uint8)
        source._color_history.extend(((1.00, color_old), (1.18, color_near)))
        source._depth_history.append((1.20, depth))
        source._ir_history.append((1.19, infrared))
        bundle = source.latest(anchor="depth")
        self.assertIsNotNone(bundle)
        self.assertAlmostEqual(bundle.color_timestamp_s, 1.18)
        self.assertAlmostEqual(bundle.depth_timestamp_s, 1.20)
        self.assertAlmostEqual(bundle.sync_delta_s, 0.02)
        self.assertEqual(int(bundle.color_bgr[0, 0, 0]), 20)

    def test_orbbec_source_waits_for_live_camera_info(self):
        source = OrbbecRosSource(start_ros_driver=False)
        source._color_history.append((1.0, np.zeros((3, 4, 3), dtype=np.uint8)))
        self.assertIsNone(source.latest())
        from tool.object_model_builder.rgbd_geometry import CameraIntrinsics

        source._color_intrinsics = CameraIntrinsics(
            4,
            3,
            np.asarray(
                [[100.0, 0.0, 1.5], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
            ),
            np.zeros(5),
        )
        source._depth_history.append((1.01, np.ones((3, 4), dtype=np.float32)))
        bundle = source.latest()
        self.assertIsNotNone(bundle)
        self.assertTrue(bundle.depth_aligned_to_color)
        self.assertEqual(
            (bundle.color_intrinsics.width, bundle.color_intrinsics.height),
            (4, 3),
        )


if __name__ == "__main__":
    unittest.main()
