"""Load and validate system and calibration parameter files."""

from pathlib import Path

import numpy as np
import yaml

from .errors import ConfigurationError
from .transforms import as_transform


COORDINATE_CONVENTION_ID = "tag_top_left_x_right_y_down_v1"


def load_yaml(path):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError("parameter file does not exist: {}".format(source))
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError("parameter root must be a mapping: {}".format(source))
    return data


def load_system_parameters(path):
    data = load_yaml(path)
    if int(data.get("schema_version", 0)) != 1:
        raise ConfigurationError("unsupported system parameter schema")
    return data


class CalibrationStore:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.data = load_yaml(self.path)
        self.validate()

    def validate(self):
        if int(self.data.get("schema_version", 0)) != 1:
            raise ConfigurationError("unsupported calibration parameter schema")
        color = self.data.get("camera", {}).get("color", {})
        matrix = np.asarray(color.get("camera_matrix", []), dtype=np.float64)
        if matrix.size != 9:
            raise ConfigurationError("camera.color.camera_matrix must be 3x3")
        matrix = matrix.reshape(3, 3)
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.isclose(matrix[2, 2], 1.0):
            raise ConfigurationError("camera color intrinsics are invalid")
        distortion = np.asarray(color.get("distortion_coefficients", []), dtype=np.float64)
        if distortion.size < 4:
            raise ConfigurationError("camera color distortion coefficients are incomplete")
        tag_map = self.data.get("tag_map", {})
        convention = tag_map.get("coordinate_convention", {})
        if convention.get("id") != COORDINATE_CONVENTION_ID:
            raise ConfigurationError(
                "tag_map uses an incompatible legacy center-based coordinate convention"
            )
        if not tag_map.get("tags"):
            raise ConfigurationError("tag_map.tags is empty")
        if float(tag_map.get("tag_size_mm", 0.0)) <= 0:
            raise ConfigurationError("tag_map.tag_size_mm must be positive")
        for tag_id, entry in tag_map["tags"].items():
            origin = np.asarray(entry.get("origin_mm", []), dtype=np.float64)
            if origin.size != 3 or not np.all(np.isfinite(origin)):
                raise ConfigurationError(
                    "tag_map tag {} origin_mm must contain three finite values".format(tag_id)
                )
        for name, entry in self.data.get("transforms", {}).items():
            if "matrix" not in entry:
                raise ConfigurationError("transform {} has no matrix".format(name))
            try:
                as_transform(entry["matrix"], name)
            except ValueError as error:
                raise ConfigurationError(str(error))
        return True

    @property
    def camera_matrix(self):
        return np.asarray(
            self.data["camera"]["color"]["camera_matrix"], dtype=np.float64
        ).reshape(3, 3)

    @property
    def distortion(self):
        return np.asarray(
            self.data["camera"]["color"]["distortion_coefficients"],
            dtype=np.float64,
        ).reshape(-1, 1)

    @property
    def image_size(self):
        color = self.data["camera"]["color"]
        return int(color["image_width"]), int(color["image_height"])

    @property
    def tag_map(self):
        return self.data["tag_map"]

    def transform(self, name, require_valid=True):
        entry = self.data.get("transforms", {}).get(name)
        if entry is None:
            raise ConfigurationError("transform is not configured: {}".format(name))
        if require_valid and not bool(entry.get("valid", False)):
            raise ConfigurationError("transform is not calibrated: {}".format(name))
        return as_transform(entry["matrix"], name)

    def transform_valid(self, name):
        return bool(self.data.get("transforms", {}).get(name, {}).get("valid", False))

    @property
    def depth_aligned_to_color(self):
        return bool(self.data.get("camera", {}).get("depth", {}).get("aligned_to_color", False))
