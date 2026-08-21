"""Persistent hand-eye sessions tied to their exact calibration target."""

import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

from .configuration import atomic_write_yaml
from .checkerboard_target import CHECKERBOARD_TARGET
from .hand_eye import APRILTAG_MAP_TARGET


def tag_map_signature(config):
    payload = yaml.safe_dump(config.tag_map, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hand_eye_target_signature(config):
    """Hash only the target definition relevant to the active solver."""
    target = config.data.get("hand_eye", {}).get("calibration_target", {})
    target_type = str(target.get("type", APRILTAG_MAP_TARGET))
    if target_type == APRILTAG_MAP_TARGET:
        payload = {"type": target_type, "tag_map": config.tag_map}
    elif target_type == CHECKERBOARD_TARGET:
        payload = {
            "type": target_type,
            "checkerboard": target.get("checkerboard", {}),
        }
    else:
        payload = {"type": target_type}
    encoded = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def active_hand_eye_target_type(config):
    return str(
        config.data.get("hand_eye", {}).get("calibration_target", {}).get(
            "type", APRILTAG_MAP_TARGET
        )
    )


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
        target_type = active_hand_eye_target_type(self.config)
        stored_type = str(data.get("calibration_target_type", APRILTAG_MAP_TARGET))
        if stored_type != target_type:
            raise ValueError(
                "Hand-eye calibration target changed ({} -> {}); archive old samples and collect again".format(
                    stored_type, target_type
                )
            )
        stored_signature = data.get("calibration_target_sha256")
        if stored_signature is not None:
            if stored_signature != hand_eye_target_signature(self.config):
                if target_type == APRILTAG_MAP_TARGET:
                    raise ValueError("Tag map changed; archive old samples and collect again")
                raise ValueError(
                    "Hand-eye calibration target definition changed; archive old samples and collect again"
                )
        elif target_type == APRILTAG_MAP_TARGET:
            # Schema-v1 AprilTag sessions stored only this legacy key.  Keep
            # them loadable if and only if their exact Tag map is unchanged.
            if data.get("tag_map_sha256") != tag_map_signature(self.config):
                raise ValueError("Tag map changed; archive old samples and collect again")
        else:
            raise ValueError(
                "legacy hand-eye samples cannot be used with checkerboard calibration"
            )
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
        entry = {
            "captured_at_unix_s": time.time(),
            "image_source": str(image_source),
            "base_from_tcp": sample.base_from_tcp.tolist(),
            "target_type": str(getattr(sample, "target_type", APRILTAG_MAP_TARGET)),
            "rms_reprojection_error_px": float(sample.rms_reprojection_error_px),
        }
        if entry["target_type"] == CHECKERBOARD_TARGET:
            entry.update({
                "camera_from_target": sample.camera_from_target.tolist(),
                "target_corner_count": int(getattr(sample, "target_corner_count", 0)),
            })
        else:
            entry.update({
                "base_from_camera": sample.base_from_camera.tolist(),
                "visible_tag_ids": list(sample.visible_tag_ids),
            })
        data["samples"].append(entry)
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
            "calibration_target_type": active_hand_eye_target_type(self.config),
            "calibration_target_sha256": hand_eye_target_signature(self.config),
            "camera_profile": self.config.active_camera_profile,
            "samples": [],
        }
