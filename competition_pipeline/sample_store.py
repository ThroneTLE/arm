"""Persistent hand-eye sample sessions tied to an exact Tag map."""

import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

from .configuration import atomic_write_yaml


def tag_map_signature(config):
    payload = yaml.safe_dump(config.tag_map, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class HandEyeSampleStore:
    def __init__(self, path, config):
        self.path = Path(path).expanduser().resolve()
        self.config = config

    def load(self, create=False):
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        elif create:
            data = self._empty_data()
        else:
            raise ValueError("sample file does not exist: {}".format(self.path))
        if int(data.get("schema_version", 0)) != 1:
            raise ValueError("unsupported hand-eye sample schema")
        if data.get("tag_map_sha256") != tag_map_signature(self.config):
            raise ValueError("Tag map changed; archive old samples and collect again")
        sample_profile = data.get("camera_profile")
        if sample_profile is not None and sample_profile != self.config.active_camera_profile:
            raise ValueError(
                "Camera profile changed; use the matching hand-eye sample session"
            )
        return data

    def entries(self):
        return self.load(create=True)["samples"]

    def append(self, sample, image_source):
        data = self.load(create=True)
        data["samples"].append(
            {
                "captured_at_unix_s": time.time(),
                "image_source": str(image_source),
                "base_from_tcp": sample.base_from_tcp.tolist(),
                "base_from_camera": sample.base_from_camera.tolist(),
                "visible_tag_ids": list(sample.visible_tag_ids),
                "rms_reprojection_error_px": float(sample.rms_reprojection_error_px),
            }
        )
        atomic_write_yaml(self.path, data)
        return len(data["samples"])

    def reset(self):
        backup = None
        if self.path.is_file():
            backup_dir = self.path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = backup_dir / "{}_{}{}".format(
                self.path.stem, stamp, self.path.suffix
            )
            shutil.copy2(str(self.path), str(backup))
        atomic_write_yaml(self.path, self._empty_data())
        return backup

    def _empty_data(self):
        return {
            "schema_version": 1,
            "tag_map_sha256": tag_map_signature(self.config),
            "camera_profile": self.config.active_camera_profile,
            "samples": [],
        }
