#!/usr/bin/env python3
"""Masked multi-view TSDF fusion and mesh-frame normalization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .capture_session import CaptureSession
from .rgbd_geometry import (
    _as_transform,
    depth_coverage,
    mask_aligned_depth,
    masked_depth_centroid,
    transform_points,
)


@dataclass
class MeshFusionResult:
    mesh: object
    workspace_from_object: np.ndarray
    views_integrated: int
    vertex_count: int
    triangle_count: int
    dimensions_m: Tuple[float, float, float]


@dataclass(frozen=True)
class CaptureQualityReport:
    minimum_depth_coverage: float
    maximum_object_centroid_shift_m: float


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for TSDF mesh fusion") from error
    return o3d


def _largest_triangle_component(mesh):
    if len(mesh.triangles) == 0:
        return mesh
    clusters, counts, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(clusters)
    counts_array = np.asarray(counts)
    if counts_array.size == 0:
        return mesh
    keep = int(np.argmax(counts_array))
    mesh.remove_triangles_by_mask(cluster_ids != keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def object_frame_from_workspace(
    vertices_workspace: np.ndarray,
    workspace_up: Sequence[float] = (0.0, 0.0, -1.0),
    workspace_x_hint: Sequence[float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    vertices = np.asarray(vertices_workspace, dtype=np.float64).reshape(-1, 3)
    if len(vertices) < 3:
        raise ValueError("mesh contains too few vertices")
    z_axis = np.asarray(workspace_up, dtype=np.float64).reshape(3)
    z_axis /= np.linalg.norm(z_axis)
    x_hint = np.asarray(workspace_x_hint, dtype=np.float64).reshape(3)
    x_axis = x_hint - z_axis * np.dot(x_hint, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        raise ValueError("workspace_x_hint is parallel to workspace_up")
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    coordinates = vertices @ rotation
    bottom = float(coordinates[:, 2].min())
    center_x = float((coordinates[:, 0].min() + coordinates[:, 0].max()) * 0.5)
    center_y = float((coordinates[:, 1].min() + coordinates[:, 1].max()) * 0.5)
    origin_workspace = rotation @ np.asarray([center_x, center_y, bottom])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = origin_workspace
    return _as_transform(transform, "workspace_from_object")


def validate_capture_quality(
    session: CaptureSession,
    minimum_mask_depth_coverage: float = 0.0,
    maximum_object_centroid_shift_m: Optional[float] = None,
) -> CaptureQualityReport:
    """Reject RGB-D sessions that cannot support stable rigid-object fusion."""
    minimum_required = float(minimum_mask_depth_coverage)
    maximum_shift_allowed = (
        None
        if maximum_object_centroid_shift_m is None
        else float(maximum_object_centroid_shift_m)
    )
    intrinsics = session.color_intrinsics()
    minimum_observed = 1.0
    maximum_observed_shift = 0.0
    centroids_workspace = []
    view_count = 0
    for view in session.iter_views():
        view_count += 1
        coverage = depth_coverage(view.depth_aligned_m, view.mask)
        minimum_observed = min(minimum_observed, coverage)
        if coverage < minimum_required:
            raise ValueError(
                "view {} mask depth coverage {:.1%} is below {:.1%}; "
                "the object may be transparent, reflective, or poorly segmented".format(
                    view.index, coverage, minimum_required
                )
            )
        if maximum_shift_allowed is not None and maximum_shift_allowed > 0.0:
            centroid_camera = masked_depth_centroid(
                view.depth_aligned_m,
                view.mask,
                intrinsics,
            )
            centroid_workspace = transform_points(
                centroid_camera.reshape(1, 3), view.workspace_from_color
            )[0]
            for previous in centroids_workspace:
                shift = float(np.linalg.norm(centroid_workspace - previous))
                maximum_observed_shift = max(maximum_observed_shift, shift)
            if maximum_observed_shift > maximum_shift_allowed:
                raise ValueError(
                    "object centroid moved {:.1f} mm in the Tag workspace; "
                    "maximum allowed shift is {:.1f} mm. Keep the object and Tags "
                    "fixed and move only the camera".format(
                        maximum_observed_shift * 1000.0,
                        maximum_shift_allowed * 1000.0,
                    )
                )
            centroids_workspace.append(centroid_workspace)
    if view_count == 0:
        minimum_observed = 0.0
    return CaptureQualityReport(
        minimum_depth_coverage=float(minimum_observed),
        maximum_object_centroid_shift_m=float(maximum_observed_shift),
    )


def fuse_session(
    session_path: str,
    voxel_length_m: float = 0.002,
    sdf_trunc_m: float = 0.008,
    maximum_depth_m: float = 2.0,
    minimum_views: int = 8,
    mask_erosion_pixels: int = 1,
    simplify_triangles: Optional[int] = 80000,
    workspace_up: Sequence[float] = (0.0, 0.0, -1.0),
    minimum_mask_depth_coverage: float = 0.0,
    maximum_object_centroid_shift_m: Optional[float] = None,
) -> MeshFusionResult:
    session = CaptureSession.open(session_path)
    if len(session) < int(minimum_views):
        raise ValueError(
            "capture session has {} views; at least {} are required".format(
                len(session), minimum_views
            )
        )
    if voxel_length_m <= 0.0 or sdf_trunc_m <= voxel_length_m:
        raise ValueError("TSDF truncation must be larger than the positive voxel size")
    validate_capture_quality(
        session,
        minimum_mask_depth_coverage=minimum_mask_depth_coverage,
        maximum_object_centroid_shift_m=maximum_object_centroid_shift_m,
    )
    o3d = _open3d()
    intrinsics = session.color_intrinsics()
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width,
        intrinsics.height,
        intrinsics.matrix[0, 0],
        intrinsics.matrix[1, 1],
        intrinsics.matrix[0, 2],
        intrinsics.matrix[1, 2],
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length_m),
        sdf_trunc=float(sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    views_integrated = 0
    for view in session.iter_views():
        masked_depth = mask_aligned_depth(
            view.depth_aligned_m,
            view.mask,
            erosion_pixels=mask_erosion_pixels,
        )
        if np.count_nonzero(masked_depth) < 50:
            continue
        color_rgb = np.ascontiguousarray(view.color_bgr[..., ::-1])
        depth_mm = np.rint(masked_depth * 1000.0).astype(np.uint16)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(depth_mm),
            depth_scale=1000.0,
            depth_trunc=float(maximum_depth_m),
            convert_rgb_to_intensity=False,
        )
        camera_from_workspace = np.linalg.inv(
            _as_transform(view.workspace_from_color, "workspace_from_color")
        )
        volume.integrate(rgbd, pinhole, camera_from_workspace)
        views_integrated += 1
    if views_integrated < int(minimum_views):
        raise ValueError("too few captured views contain usable masked depth")
    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh = _largest_triangle_component(mesh)
    if len(mesh.vertices) < 100 or len(mesh.triangles) < 100:
        raise RuntimeError("TSDF output is empty or too sparse")
    if simplify_triangles and len(mesh.triangles) > int(simplify_triangles):
        mesh = mesh.simplify_quadric_decimation(int(simplify_triangles))
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    vertices_workspace = np.asarray(mesh.vertices).copy()
    workspace_from_object = object_frame_from_workspace(
        vertices_workspace,
        workspace_up=workspace_up,
    )
    object_from_workspace = np.linalg.inv(workspace_from_object)
    vertices_object = (
        vertices_workspace @ object_from_workspace[:3, :3].T
        + object_from_workspace[:3, 3]
    )
    mesh.vertices = o3d.utility.Vector3dVector(vertices_object)
    mesh.compute_vertex_normals()
    extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    return MeshFusionResult(
        mesh=mesh,
        workspace_from_object=workspace_from_object,
        views_integrated=views_integrated,
        vertex_count=len(mesh.vertices),
        triangle_count=len(mesh.triangles),
        dimensions_m=tuple(float(value) for value in extent),
    )
