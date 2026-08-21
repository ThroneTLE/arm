import tempfile
import unittest
from pathlib import Path

import numpy as np

from tool.object_model_builder.model_free import _read_reference, run_model_free_reconstruction
from tool.object_model_builder.session_archive import export_foundationpose_reference_directory
from tool.object_model_builder.rgbd_geometry import CameraIntrinsics
from tool.object_model_builder.capture_session import CaptureSession


class ModelFreeReferenceTests(unittest.TestCase):
    def _session(self, root):
        intrinsics = CameraIntrinsics(
            8, 6,
            np.asarray([[250.0, 0.0, 3.5], [0.0, 250.0, 2.5], [0.0, 0.0, 1.0]]),
            np.zeros(5),
        )
        calibration = root / "calibration.yaml"
        tags = root / "tags.yaml"
        calibration.write_text("schema_version: 1\n", encoding="utf-8")
        tags.write_text("schema_version: 1\n", encoding="utf-8")
        session = CaptureSession.create(
            str(root / "scan"), intrinsics, intrinsics, np.eye(4),
            str(calibration), str(tags), "/tmp/object.pt", "object",
        )
        session.add_view(
            np.full((6, 8, 3), 90, dtype=np.uint8),
            np.full((6, 8), 0.6, dtype=np.float32),
            np.ones((6, 8), dtype=np.uint8),
            np.eye(4), timestamp_s=1.0,
        )
        return session

    def test_export_directory_matches_upstream_layout_and_loads_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self._session(root)
            reference = export_foundationpose_reference_directory(
                str(session.root), str(root / "reference"), object_id=3,
            )
            self.assertEqual(reference.name, "ob_0000003")
            rgbs, depths, masks, poses, K = _read_reference(str(reference))
            self.assertEqual(rgbs.shape, (1, 6, 8, 3))
            self.assertEqual(depths.shape, (1, 6, 8))
            self.assertEqual(masks.dtype, np.bool_)
            self.assertTrue(np.allclose(poses[0], np.eye(4)))
            self.assertAlmostEqual(float(K[0, 0]), 250.0)

    def test_model_free_rejects_non_empty_output_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self._session(root)
            reference = export_foundationpose_reference_directory(
                str(session.root), str(root / "reference"), object_id=1,
            )
            output = root / "output"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                run_model_free_reconstruction(
                    str(reference), "/missing/foundationpose", str(output)
                )


if __name__ == "__main__":
    unittest.main()
