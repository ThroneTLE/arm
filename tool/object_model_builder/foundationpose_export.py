#!/usr/bin/env python3
"""Export and validate meshes for model-based FoundationPose inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

from .rgbd_geometry import _as_transform


@dataclass
class MeshValidation:
    valid: bool
    vertex_count: int
    triangle_count: int
    dimensions_m: Tuple[float, float, float]
    watertight: Optional[bool]
    warnings: Tuple[str, ...]
    reason: str


def _dimension_limit_vector(value, default, name) -> np.ndarray:
    if value is None:
        value = default
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), 3)
    array = array.reshape(-1)
    if array.size != 3 or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError("{} must contain three positive metric dimensions".format(name))
    return array


def validate_mesh(
    mesh,
    minimum_dimensions_m=None,
    maximum_dimensions_m=None,
    require_watertight: bool = False,
) -> MeshValidation:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    warnings = []
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 100:
        return MeshValidation(False, len(vertices), len(triangles), (0.0, 0.0, 0.0), None, tuple(), "mesh has too few vertices")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,) or len(triangles) < 100:
        return MeshValidation(False, len(vertices), len(triangles), (0.0, 0.0, 0.0), None, tuple(), "mesh has too few triangles")
    if not np.isfinite(vertices).all():
        return MeshValidation(False, len(vertices), len(triangles), (0.0, 0.0, 0.0), None, tuple(), "mesh contains non-finite vertices")
    dimensions = vertices.max(axis=0) - vertices.min(axis=0)
    minimum_dimensions = _dimension_limit_vector(
        minimum_dimensions_m, (0.005, 0.005, 0.005), "minimum_dimensions_m"
    )
    maximum_dimensions = _dimension_limit_vector(
        maximum_dimensions_m, (1.0, 1.0, 1.0), "maximum_dimensions_m"
    )
    if np.any(minimum_dimensions >= maximum_dimensions):
        raise ValueError("minimum mesh dimensions must be below maximum dimensions")
    dimension_tuple = tuple(float(value) for value in dimensions)
    if np.any(dimensions < minimum_dimensions):
        return MeshValidation(
            False,
            len(vertices),
            len(triangles),
            dimension_tuple,
            None,
            tuple(),
            "mesh dimensions {} m are below the configured minimum {} m".format(
                [round(float(value), 4) for value in dimensions],
                [round(float(value), 4) for value in minimum_dimensions],
            ),
        )
    if np.any(dimensions > maximum_dimensions):
        return MeshValidation(
            False,
            len(vertices),
            len(triangles),
            dimension_tuple,
            None,
            tuple(),
            "mesh dimensions {} m exceed the configured maximum {} m".format(
                [round(float(value), 4) for value in dimensions],
                [round(float(value), 4) for value in maximum_dimensions],
            ),
        )
    watertight = None
    try:
        watertight = bool(mesh.is_watertight())
    except Exception:
        pass
    if require_watertight and watertight is not True:
        return MeshValidation(
            False,
            len(vertices),
            len(triangles),
            dimension_tuple,
            watertight,
            tuple(),
            (
                "mesh is not watertight"
                if watertight is False
                else "mesh watertightness could not be verified"
            ),
        )
    if watertight is False:
        warnings.append("mesh is not watertight; FoundationPose can run but edge quality may degrade")
    return MeshValidation(
        True,
        len(vertices),
        len(triangles),
        dimension_tuple,
        watertight,
        tuple(warnings),
        "mesh geometry is compatible with FoundationPose",
    )


def export_foundationpose_model(
    mesh,
    destination_root: str,
    model_name: str,
    workspace_from_object: np.ndarray,
    source_session: str,
    quality_config: Optional[dict] = None,
) -> Tuple[Path, MeshValidation]:
    quality = dict(quality_config or {})
    validation = validate_mesh(
        mesh,
        minimum_dimensions_m=quality.get("minimum_dimensions_m"),
        maximum_dimensions_m=quality.get("maximum_dimensions_m"),
        require_watertight=bool(quality.get("require_watertight", False)),
    )
    if not validation.valid:
        raise ValueError(validation.reason)
    safe_name = "".join(character for character in str(model_name) if character.isalnum() or character in "_-")
    if not safe_name:
        raise ValueError("model name must contain letters, digits, '-' or '_'")
    output = Path(destination_root).expanduser().resolve() / safe_name
    output.mkdir(parents=True, exist_ok=True)
    obj_path = output / (safe_name + ".obj")
    ply_path = output / (safe_name + ".ply")
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required to export the mesh") from error
    if not o3d.io.write_triangle_mesh(str(obj_path), mesh, write_vertex_normals=True):
        raise IOError("failed to export OBJ mesh")
    if not o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_normals=True):
        raise IOError("failed to export PLY mesh")
    metadata = {
        "schema_version": 1,
        "model_name": safe_name,
        "units": "meters",
        "mesh_frame": {
            "origin": "object_bottom_center",
            "x_axis": "workspace_x",
            "z_axis": "workspace_up",
            "handedness": "right_handed",
        },
        "mesh": {
            "obj": obj_path.name,
            "ply": ply_path.name,
            "vertex_count": validation.vertex_count,
            "triangle_count": validation.triangle_count,
            "dimensions_m": list(validation.dimensions_m),
            "mesh_scale_to_meters": 1.0,
        },
        "workspace_from_object_at_scan": _as_transform(
            workspace_from_object, "workspace_from_object"
        ).tolist(),
        "source_session": str(Path(source_session).expanduser().resolve()),
        "foundationpose": {
            "mesh_path": str(obj_path),
            "mesh_scale_to_meters": 1.0,
            "requires_aligned_depth": True,
        },
        "quality_gate": {
            "minimum_dimensions_m": quality.get("minimum_dimensions_m"),
            "maximum_dimensions_m": quality.get("maximum_dimensions_m"),
            "require_watertight": bool(quality.get("require_watertight", False)),
            "passed": True,
        },
        "warnings": list(validation.warnings),
    }
    with open(output / "model_metadata.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)
    return obj_path, validation
