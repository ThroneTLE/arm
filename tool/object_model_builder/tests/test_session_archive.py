#!/usr/bin/env python3

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import yaml

from tool.object_model_builder.capture_session import CaptureSession
from tool.object_model_builder.rgbd_geometry import CameraIntrinsics
from tool.object_model_builder.session_archive import (
    CAPTURE_ARCHIVE_MANIFEST,
    FOUNDATIONPOSE_REFERENCE_ARCHIVE_MANIFEST,
    create_capture_archive,
    create_foundationpose_reference_archive,
    create_result_archive,
    extract_capture_archive,
    validate_capture_session,
)


def intrinsics():
    return CameraIntrinsics(
        4,
        3,
        np.asarray([[200.0, 0.0, 1.5], [0.0, 200.0, 1.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
    )


def create_session(root: Path) -> CaptureSession:
    calibration = root.parent / "calibration.yaml"
    tag_layout = root.parent / "tag_layout.yaml"
    calibration.write_text("schema_version: 1\n", encoding="utf-8")
    tag_layout.write_text("schema_version: 2\n", encoding="utf-8")
    session = CaptureSession.create(
        str(root),
        intrinsics(),
        intrinsics(),
        np.eye(4),
        str(calibration),
        str(tag_layout),
        "/models/bottle.pt",
        "bottle",
    )
    color = np.full((3, 4, 3), 127, dtype=np.uint8)
    depth = np.full((3, 4), 0.75, dtype=np.float32)
    mask = np.ones((3, 4), dtype=np.uint8)
    session.add_view(
        color,
        depth,
        mask,
        np.eye(4),
        timestamp_s=12.5,
        depth_raw_m=np.full((2, 2), 0.75, dtype=np.float32),
    )
    return session


class CaptureArchiveTests(unittest.TestCase):
    def test_capture_archive_round_trip_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = create_session(root / "scan")
            archive = create_capture_archive(str(session.root), str(root / "scan.zip"))
            with zipfile.ZipFile(archive, "r") as handle:
                names = set(handle.namelist())
                self.assertIn(CAPTURE_ARCHIVE_MANIFEST, names)
                self.assertIn("provenance/calibration.yaml", names)
                self.assertIn("provenance/tag_layout.yaml", names)
                archive_manifest = yaml.safe_load(handle.read(CAPTURE_ARCHIVE_MANIFEST))
                self.assertFalse(archive_manifest["server_requires_camera"])
                self.assertFalse(archive_manifest["server_requires_yolo"])
            imported = extract_capture_archive(str(archive), str(root / "imports"))
            self.assertEqual(imported.frame_count, 1)
            validation = validate_capture_session(str(imported.session_path))
            self.assertEqual(validation.frame_count, 1)
            view = next(CaptureSession.open(str(imported.session_path)).iter_views())
            self.assertEqual(view.color_bgr.shape, (3, 4, 3))
            self.assertAlmostEqual(float(view.depth_aligned_m[0, 0]), 0.75, places=6)

    def test_modified_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = create_capture_archive(
                str(create_session(root / "scan").root), str(root / "scan.zip")
            )
            modified = root / "modified.zip"
            with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(
                modified, "w", compression=zipfile.ZIP_DEFLATED
            ) as destination:
                for info in source.infolist():
                    data = source.read(info.filename)
                    if info.filename == "color/000000.png":
                        data += b"modified"
                    destination.writestr(info.filename, data)
            with self.assertRaisesRegex(ValueError, "size mismatch|checksum mismatch"):
                extract_capture_archive(str(modified), str(root / "imports"))

    def test_foundationpose_reference_archive_has_upstream_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = create_session(root / "scan")
            archive = create_foundationpose_reference_archive(
                str(session.root),
                str(root / "reference.zip"),
                object_id=7,
                object_name="test_bottle",
            )
            prefix = "ob_0000007/"
            with zipfile.ZipFile(archive, "r") as handle:
                names = set(handle.namelist())
                self.assertIn(FOUNDATIONPOSE_REFERENCE_ARCHIVE_MANIFEST, names)
                self.assertIn("如何使用.txt", names)
                self.assertIn(prefix + "K.txt", names)
                self.assertIn(prefix + "select_frames.yml", names)
                self.assertIn(prefix + "rgb/0000000.png", names)
                self.assertIn(prefix + "depth_enhanced/0000000.png", names)
                self.assertIn(prefix + "mask/0000000.png", names)
                self.assertIn(prefix + "cam_in_ob/0000000.txt", names)
                archive_manifest = yaml.safe_load(
                    handle.read(FOUNDATIONPOSE_REFERENCE_ARCHIVE_MANIFEST)
                )
                self.assertEqual(
                    archive_manifest["archive_type"],
                    "foundationpose_model_free_reference",
                )
                self.assertEqual(archive_manifest["frame_count"], 1)
                self.assertFalse(archive_manifest["server_requires_camera"])
                guide = handle.read("如何使用.txt").decode("utf-8")
                self.assertIn("model.obj", guide)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            manifest = {
                "schema_version": 1,
                "archive_type": "object_model_builder_capture",
                "frame_count": 0,
                "files": {"../outside": {"size": 1, "sha256": "0" * 64}},
            }
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(CAPTURE_ARCHIVE_MANIFEST, yaml.safe_dump(manifest))
                handle.writestr("../outside", b"x")
            with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                extract_capture_archive(str(archive), str(root / "imports"))
            self.assertFalse((root / "outside").exists())

    def test_result_archive_contains_model_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "bottle"
            model.mkdir()
            (model / "bottle.obj").write_text("o bottle\n", encoding="utf-8")
            (model / "model_metadata.yaml").write_text(
                "schema_version: 1\n", encoding="utf-8"
            )
            archive = create_result_archive(str(model), str(root / "result.zip"))
            with zipfile.ZipFile(archive, "r") as handle:
                self.assertEqual(
                    set(handle.namelist()),
                    {"result_manifest.yaml", "bottle.obj", "model_metadata.yaml"},
                )


if __name__ == "__main__":
    unittest.main()
