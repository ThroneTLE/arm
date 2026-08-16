#!/usr/bin/env python3
"""Portable, checksummed archives for offline RGB-D reconstruction."""

import hashlib
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import yaml

from .capture_session import CaptureSession
from .rgbd_geometry import _as_transform


CAPTURE_ARCHIVE_SCHEMA_VERSION = 1
RESULT_ARCHIVE_SCHEMA_VERSION = 1
FOUNDATIONPOSE_REFERENCE_ARCHIVE_SCHEMA_VERSION = 1
CAPTURE_ARCHIVE_MANIFEST = "archive_manifest.yaml"
RESULT_ARCHIVE_MANIFEST = "result_manifest.yaml"
FOUNDATIONPOSE_REFERENCE_ARCHIVE_MANIFEST = (
    "foundationpose_reference_manifest.yaml"
)
DEFAULT_MAX_UNCOMPRESSED_BYTES = 100 * 1024**3
DEFAULT_MAX_FILE_COUNT = 10000


@dataclass(frozen=True)
class SessionValidation:
    frame_count: int
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class ImportedSession:
    session_path: Path
    frame_count: int
    archive_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> PurePosixPath:
    normalized = str(value).replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe archive path: {}".format(value))
    if any(part in ("", ".") for part in path.parts):
        raise ValueError("invalid archive path: {}".format(value))
    return path


def _session_payload_paths(session: CaptureSession) -> Tuple[Path, ...]:
    manifest = session.load_manifest()
    relative_paths = {PurePosixPath("manifest.yaml")}
    for entry in manifest.get("views", []):
        for key in (
            "color",
            "depth_raw",
            "depth_aligned",
            "mask",
            "workspace_from_color",
        ):
            if entry.get(key):
                relative_paths.add(_safe_relative_path(str(entry[key])))
    for value in (manifest.get("provenance") or {}).values():
        if value:
            relative_paths.add(_safe_relative_path(str(value)))
    paths = []
    for relative in sorted(relative_paths, key=str):
        path = session.root.joinpath(*relative.parts)
        if path.is_symlink():
            raise ValueError("capture session cannot contain symlinks: {}".format(relative))
        if not path.is_file():
            raise FileNotFoundError("capture session file is missing: {}".format(relative))
        paths.append(path)
    return tuple(paths)


def validate_capture_session(session_path: str) -> SessionValidation:
    session = CaptureSession.open(session_path)
    manifest = session.load_manifest()
    views = list(manifest.get("views", []))
    frame_count = int(manifest.get("frame_count", 0))
    if frame_count != len(views):
        raise ValueError(
            "capture manifest frame_count {} does not match {} view entries".format(
                frame_count, len(views)
            )
        )
    if frame_count <= 0:
        raise ValueError("capture session contains no captured views")
    indices = [int(entry.get("index", -1)) for entry in views]
    if indices != list(range(frame_count)):
        raise ValueError("capture view indices must be consecutive and start at zero")
    loaded = 0
    for view in session.iter_views():
        _as_transform(view.workspace_from_color, "workspace_from_color")
        if not np.isfinite(view.depth_aligned_m).all():
            raise ValueError("view {} contains non-finite aligned depth".format(view.index))
        if view.mask.shape != view.color_bgr.shape[:2]:
            raise ValueError("view {} mask dimensions do not match RGB".format(view.index))
        if view.depth_aligned_m.shape != view.color_bgr.shape[:2]:
            raise ValueError("view {} aligned depth dimensions do not match RGB".format(view.index))
        loaded += 1
    if loaded != frame_count:
        raise ValueError("capture session did not yield every manifest view")
    files = _session_payload_paths(session)
    return SessionValidation(
        frame_count=frame_count,
        file_count=len(files),
        total_bytes=sum(path.stat().st_size for path in files),
    )


def _write_payload_archive(
    files: Iterable[Tuple[str, Path]],
    destination: Path,
    manifest_name: str,
    metadata: dict,
) -> Path:
    destination = destination.expanduser().resolve()
    if destination.suffix.lower() != ".zip":
        destination = destination.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    records: Dict[str, dict] = {}
    normalized_files = []
    for archive_name, source in files:
        relative = str(_safe_relative_path(archive_name))
        source = source.expanduser().resolve()
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError("archive source is not a regular file: {}".format(source))
        if relative in records:
            raise ValueError("duplicate archive payload path: {}".format(relative))
        records[relative] = {
            "size": int(source.stat().st_size),
            "sha256": sha256_file(source),
        }
        normalized_files.append((relative, source))
    archive_manifest = dict(metadata)
    archive_manifest["files"] = records
    manifest_bytes = yaml.safe_dump(
        archive_manifest, sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
        ) as archive:
            archive.writestr(manifest_name, manifest_bytes)
            for archive_name, source in normalized_files:
                archive.write(source, arcname=archive_name)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def create_capture_archive(session_path: str, destination_zip: str) -> Path:
    session = CaptureSession.open(session_path)
    validation = validate_capture_session(str(session.root))
    files = []
    for path in _session_payload_paths(session):
        files.append((path.relative_to(session.root).as_posix(), path))
    metadata = {
        "schema_version": CAPTURE_ARCHIVE_SCHEMA_VERSION,
        "archive_type": "object_model_builder_capture",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session_name": session.root.name,
        "frame_count": validation.frame_count,
        "units": "meters",
        "contains_masks": True,
        "server_requires_camera": False,
        "server_requires_yolo": False,
    }
    return _write_payload_archive(
        files,
        Path(destination_zip),
        CAPTURE_ARCHIVE_MANIFEST,
        metadata,
    )


def create_foundationpose_reference_archive(
    session_path: str,
    destination_zip: str,
    object_id: int = 1,
    object_name: str = "object",
) -> Path:
    """Export a capture session in FoundationPose model-free reference format."""
    session = CaptureSession.open(session_path)
    validation = validate_capture_session(str(session.root))
    identifier = int(object_id)
    if identifier <= 0:
        raise ValueError("FoundationPose object_id must be positive")
    safe_name = "".join(
        character
        for character in str(object_name)
        if character.isalnum() or character in "_-"
    ) or "object"
    object_directory = "ob_{:07d}".format(identifier)
    intrinsics = session.color_intrinsics()
    with tempfile.TemporaryDirectory(prefix="foundationpose_reference_") as directory:
        root = Path(directory)
        reference_root = root / object_directory
        for name in ("rgb", "depth_enhanced", "mask", "cam_in_ob"):
            (reference_root / name).mkdir(parents=True, exist_ok=True)
        np.savetxt(reference_root / "K.txt", intrinsics.matrix, fmt="%.12g")
        selected_frames = []
        for view in session.iter_views():
            stem = "{:07d}".format(view.index)
            selected_frames.append(view.index)
            if not cv2.imwrite(
                str(reference_root / "rgb" / (stem + ".png")), view.color_bgr
            ):
                raise IOError("failed to write FoundationPose RGB reference")
            depth_mm = np.rint(
                np.clip(view.depth_aligned_m, 0.0, 65.535) * 1000.0
            ).astype(np.uint16)
            if not cv2.imwrite(
                str(reference_root / "depth_enhanced" / (stem + ".png")),
                depth_mm,
            ):
                raise IOError("failed to write FoundationPose depth reference")
            if not cv2.imwrite(
                str(reference_root / "mask" / (stem + ".png")),
                view.mask.astype(np.uint8) * 255,
            ):
                raise IOError("failed to write FoundationPose mask reference")
            np.savetxt(
                reference_root / "cam_in_ob" / (stem + ".txt"),
                _as_transform(view.workspace_from_color, "workspace_from_color"),
                fmt="%.12g",
            )
        with open(reference_root / "select_frames.yml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "selected_frames": selected_frames,
                    "frame_count": validation.frame_count,
                },
                handle,
                sort_keys=False,
            )
        metadata = {
            "schema_version": FOUNDATIONPOSE_REFERENCE_ARCHIVE_SCHEMA_VERSION,
            "object_id": identifier,
            "object_name": safe_name,
            "frame_count": validation.frame_count,
            "units": "meters",
            "depth_storage_units": "millimeters_uint16",
            "camera_convention": "opencv_color_optical_frame",
            "pose_convention": "cam_in_ob maps camera points into fixed object/workspace frame",
            "object_frame_at_capture": "ruler_workspace",
            "foundationpose_layout": object_directory,
            "requires_model_free_reconstruction": True,
        }
        with open(
            reference_root / "reference_metadata.yaml", "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)
        (root / "如何使用.txt").write_text(
            "FoundationPose 无模型参考照片\n"
            "\n"
            "1. ob_XXXXXXX/rgb 是校正后的 RGB。\n"
            "2. depth_enhanced 是对齐 RGB 的 uint16 毫米深度。\n"
            "3. mask 是目标二值 Mask，cam_in_ob 是相机到固定物体/工作坐标的位姿。\n"
            "4. 该 ZIP 不是可直接跟踪的 CAD 模型。请先在安装 Kaolin/BundleSDF 的环境中"
            "运行 Neural Object Field 重建，得到 model.obj。\n"
            "5. 回到物体三维模型工作台，选择 model.obj，米制缩放填 1.0，再加载实时测试。\n",
            encoding="utf-8",
        )
        shutil.copy2(session.manifest_path, reference_root / "source_manifest.yaml")
        provenance = session.root / "provenance"
        if provenance.is_dir():
            shutil.copytree(provenance, reference_root / "provenance")
        files = [
            (path.relative_to(root).as_posix(), path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ]
        archive_metadata = dict(metadata)
        archive_metadata.update(
            {
                "archive_type": "foundationpose_model_free_reference",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "source_session": session.root.name,
                "server_requires_camera": False,
                "server_requires_yolo": False,
                "server_requires_tag_detection": False,
            }
        )
        return _write_payload_archive(
            files,
            Path(destination_zip),
            FOUNDATIONPOSE_REFERENCE_ARCHIVE_MANIFEST,
            archive_metadata,
        )


def _validated_archive_entries(
    archive: zipfile.ZipFile,
    manifest_name: str,
    expected_type: str,
    schema_version: int,
    max_uncompressed_bytes: int,
    max_file_count: int,
):
    infos = archive.infolist()
    if len(infos) > int(max_file_count) + 1:
        raise ValueError("archive contains too many files")
    names = []
    info_by_name = {}
    for info in infos:
        name = str(_safe_relative_path(info.filename))
        if name in info_by_name:
            raise ValueError("archive contains a duplicate entry: {}".format(name))
        if info.is_dir():
            raise ValueError("archive must not contain explicit directory entries")
        unix_mode = (info.external_attr >> 16) & 0o170000
        if unix_mode == 0o120000:
            raise ValueError("archive must not contain symbolic links")
        names.append(name)
        info_by_name[name] = info
    if manifest_name not in info_by_name:
        raise ValueError("archive manifest is missing: {}".format(manifest_name))
    manifest_info = info_by_name[manifest_name]
    if manifest_info.file_size > 10 * 1024 * 1024:
        raise ValueError("archive manifest is unexpectedly large")
    manifest = yaml.safe_load(archive.read(manifest_info)) or {}
    if int(manifest.get("schema_version", 0)) != int(schema_version):
        raise ValueError("unsupported archive schema version")
    if manifest.get("archive_type") != expected_type:
        raise ValueError("unexpected archive type: {}".format(manifest.get("archive_type")))
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("archive contains no payload file records")
    normalized_records = {}
    for raw_name, record in records.items():
        name = str(_safe_relative_path(raw_name))
        if name in normalized_records:
            raise ValueError("archive manifest contains a duplicate payload path")
        if not isinstance(record, dict):
            raise ValueError("archive manifest contains an invalid file record")
        normalized_records[name] = record
    declared = set(normalized_records)
    actual = set(names) - {manifest_name}
    if declared != actual:
        missing = sorted(declared - actual)
        extra = sorted(actual - declared)
        raise ValueError(
            "archive payload does not match manifest; missing={}, extra={}".format(
                missing, extra
            )
        )
    total = 0
    for name in declared:
        record = normalized_records[name]
        info = info_by_name[name]
        expected_size = int(record.get("size", -1))
        if expected_size < 0 or expected_size != info.file_size:
            raise ValueError("archive file size mismatch: {}".format(name))
        total += expected_size
        if total > int(max_uncompressed_bytes):
            raise ValueError("archive uncompressed payload is too large")
    return manifest, normalized_records, info_by_name


def extract_capture_archive(
    archive_path: str,
    destination_root: str,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
    max_file_count: int = DEFAULT_MAX_FILE_COUNT,
) -> ImportedSession:
    source = Path(archive_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("capture ZIP does not exist: {}".format(source))
    destination = Path(destination_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_digest = sha256_file(source)
    temporary = Path(tempfile.mkdtemp(prefix=".capture_import_", dir=str(destination)))
    try:
        with zipfile.ZipFile(source, "r") as archive:
            manifest, records, info_by_name = _validated_archive_entries(
                archive,
                CAPTURE_ARCHIVE_MANIFEST,
                "object_model_builder_capture",
                CAPTURE_ARCHIVE_SCHEMA_VERSION,
                max_uncompressed_bytes,
                max_file_count,
            )
            total_written = 0
            for name in sorted(records):
                relative = _safe_relative_path(name)
                output = temporary.joinpath(*relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                with archive.open(info_by_name[name], "r") as reader, open(
                    output, "wb"
                ) as writer:
                    while True:
                        block = reader.read(1024 * 1024)
                        if not block:
                            break
                        writer.write(block)
                        digest.update(block)
                        written += len(block)
                        total_written += len(block)
                        if written > int(records[name]["size"]):
                            raise ValueError("archive file exceeds declared size: {}".format(name))
                        if total_written > int(max_uncompressed_bytes):
                            raise ValueError("archive extraction exceeds the size limit")
                record = records[name]
                if written != int(record["size"]) or digest.hexdigest() != str(
                    record["sha256"]
                ):
                    raise ValueError("archive checksum mismatch: {}".format(name))
        validation = validate_capture_session(str(temporary))
        expected_frames = int(manifest.get("frame_count", -1))
        if expected_frames != validation.frame_count:
            raise ValueError("archive frame count does not match capture manifest")
        safe_name = "".join(
            character
            for character in str(manifest.get("session_name", "capture"))
            if character.isalnum() or character in "_-"
        ) or "capture"
        suffix = time.strftime("%Y%m%d_%H%M%S")
        final = destination / (safe_name + "_" + suffix)
        counter = 1
        while final.exists():
            final = destination / (safe_name + "_" + suffix + "_{}".format(counter))
            counter += 1
        temporary.replace(final)
        return ImportedSession(final, validation.frame_count, archive_digest)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def create_result_archive(model_directory: str, destination_zip: str) -> Path:
    root = Path(model_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("model output directory does not exist: {}".format(root))
    files = [
        (path.relative_to(root).as_posix(), path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    if not files:
        raise ValueError("model output directory is empty")
    metadata = {
        "schema_version": RESULT_ARCHIVE_SCHEMA_VERSION,
        "archive_type": "foundationpose_model_result",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_directory": root.name,
    }
    return _write_payload_archive(
        files,
        Path(destination_zip),
        RESULT_ARCHIVE_MANIFEST,
        metadata,
    )
