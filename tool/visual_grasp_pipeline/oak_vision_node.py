#!/usr/bin/env python3
"""Original vision-node workflow rebuilt for the active OAK-D-PRO-FF.

This is a single-process replacement for the legacy ``fp_bridge.py`` plus
``vision_node.py`` camera path. It connects to DepthAI directly, reads live
EEPROM intrinsics, consumes hardware-aligned depth, runs YOLO and
FoundationPose, and keeps the original target-sequence UI. Motion is dry-run
by default. The optional ZMQ endpoint is intended only for the legacy
simulated ``arm_node.py`` service.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import yaml

from tool.object_model_builder.camera_source import OakDProSource
from tool.object_model_builder.tag_pose_provider import TagPoseProvider
from tool.object_model_builder.rgbd_geometry import (
    CameraIntrinsics,
    depth_coverage,
    rectified_intrinsics,
    rectify_aligned_depth_image,
    rectify_color_image,
)
from tool.visual_grasp_pipeline.config import VisualGraspConfig
from tool.visual_grasp_pipeline.detection import (
    detect_all_track,
    detect_tags,
    draw_boxes,
)
from tool.visual_grasp_pipeline.foundationpose import FoundationPosePoseEstimator
from tool.visual_grasp_pipeline.geometry import (
    build_world_from_tags,
    compute_grasp,
    compute_grasp_sphere,
    fill_depth_roi,
    to_world_and_compensate,
)
from tool.visual_grasp_pipeline.tracking import StableTracker, parse_sequence
from tool.visual_grasp_pipeline.ucs_grasp import (
    UCS_PLACE_X_MM,
    UCS_PLACE_Y_MM,
    UcsGraspExecutorError,
    UcsGraspRunner,
    UcsGraspSafetyError,
    build_jog,
)

try:
    from PIL import Image, ImageTk
except ImportError as error:  # pragma: no cover - checked by the launcher
    raise RuntimeError("Pillow is required for the OAK vision-node UI") from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISUAL_CONFIG = (
    PROJECT_ROOT / "tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml"
)
DEFAULT_CAMERA_CONFIG = (
    PROJECT_ROOT / "tool/object_model_builder/config/object_model_builder.yaml"
)

@dataclass(frozen=True)
class OakSnapshot:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_s: float
    sync_delta_s: float


@dataclass(frozen=True)
class FoundationPoseInput:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    camera_matrix: np.ndarray
    roi_xyxy: tuple


def prepare_foundationpose_input(
    snapshot: OakSnapshot,
    target,
    mask,
    padding_pixels=24,
    maximum_size=640,
) -> FoundationPoseInput:
    """Crop a YOLO ROI and scale its intrinsics to bound CUDA memory."""

    height, width = snapshot.color_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, target["xyxy"])
    padding = max(0, int(padding_pixels))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("FoundationPose ROI is empty")
    color = np.ascontiguousarray(snapshot.color_bgr[y1:y2, x1:x2])
    depth = np.ascontiguousarray(snapshot.depth_m[y1:y2, x1:x2], dtype=np.float32)
    roi_mask = np.ascontiguousarray(np.asarray(mask[y1:y2, x1:x2]) > 0, dtype=np.uint8)
    matrix = np.asarray(snapshot.intrinsics.matrix, dtype=np.float64).copy()
    matrix[0, 2] -= x1
    matrix[1, 2] -= y1

    limit = max(160, int(maximum_size))
    scale = min(1.0, float(limit) / max(color.shape[:2]))
    if scale < 1.0:
        new_width = max(1, int(round(color.shape[1] * scale)))
        new_height = max(1, int(round(color.shape[0] * scale)))
        color = cv2.resize(color, (new_width, new_height), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        roi_mask = cv2.resize(
            roi_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        matrix[0, 0] *= scale
        matrix[0, 2] *= scale
        matrix[1, 1] *= scale
        matrix[1, 2] *= scale
    return FoundationPoseInput(
        color_bgr=color,
        depth_m=depth,
        mask=roi_mask,
        camera_matrix=matrix,
        roi_xyxy=(x1, y1, x2, y2),
    )


def load_oak_settings(path) -> tuple[dict, float]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    camera = data.get("camera", {})
    if camera.get("backend") != "oak_depthai":
        raise ValueError("camera.backend must be oak_depthai for oak_vision_node")
    oak = dict(camera.get("oak", {}))
    required = ("mxid", "color_width", "color_height", "fps")
    missing = [name for name in required if not str(oak.get(name, "")).strip()]
    if missing:
        raise ValueError("missing OAK settings: {}".format(", ".join(missing)))
    maximum_sync = float(camera.get("maximum_sync_delta_s", 0.03))
    if maximum_sync <= 0.0:
        raise ValueError("camera.maximum_sync_delta_s must be positive")
    return oak, maximum_sync


def build_oak_source(settings: dict) -> OakDProSource:
    return OakDProSource(
        color_width=settings.get("color_width", 1920),
        color_height=settings.get("color_height", 1080),
        fps=settings.get("fps", 10),
        mxid=settings.get("mxid", ""),
        dot_projector_mA=settings.get("dot_projector_mA", 800),
        floodlight_mA=settings.get("floodlight_mA", 0),
        mono_resolution=settings.get("mono_resolution", "800p"),
        extended_disparity=settings.get("extended_disparity", True),
        subpixel=settings.get("subpixel", False),
        left_right_check=settings.get("left_right_check", True),
        focus_mode=settings.get("focus_mode", "device_default"),
        manual_focus=settings.get("manual_focus"),
    )


def build_tag_provider(camera_config_path) -> TagPoseProvider:
    source = Path(camera_config_path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    paths = data.get("paths", {})
    settings = data.get("tag_pose", {})
    layout = paths.get("tag_layout")
    if not layout:
        raise ValueError("paths.tag_layout is required for workspace localization")
    return TagPoseProvider(
        layout,
        minimum_tags=settings.get("minimum_tags", 1),
        maximum_rms_px=settings.get("maximum_rms_px", 2.5),
    )


def find_sequence_target(objects, name: str, instance: Optional[int]):
    matching = [item for item in objects if item.get("name") == name]
    if instance is None:
        return matching[0] if matching else None
    return next(
        (
            item for item in matching
            if int(item.get("id", item.get("seq", -1))) == int(instance)
        ),
        None,
    )


def draw_pose_axes(image, camera_from_object, camera_matrix, length_m=0.06):
    output = np.asarray(image).copy()
    pose = np.asarray(camera_from_object, dtype=np.float64).reshape(4, 4)
    axes = np.asarray(
        [[0.0, 0.0, 0.0], [length_m, 0.0, 0.0],
         [0.0, length_m, 0.0], [0.0, 0.0, length_m]],
        dtype=np.float64,
    )
    points = axes @ pose[:3, :3].T + pose[:3, 3]
    if np.any(points[:, 2] <= 0.0):
        return output
    projected = points @ np.asarray(camera_matrix, dtype=np.float64).T
    pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
    origin = tuple(pixels[0])
    for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(output, origin, tuple(endpoint), color, 3, cv2.LINE_AA)
    return output


def draw_mask_overlay(image, mask, alpha=0.25):
    """旧版 draw_2d 同款: 目标掩膜绿色半透明填充 + 绿色轮廓。"""
    output = np.asarray(image).copy()
    mask = np.asarray(mask) > 0
    if not np.any(mask):
        return output
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    overlay = output.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
    output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


def draw_pose_box_2d(image, camera_from_object, mesh_bounds, camera_matrix):
    """旧版 draw_posed_3d_box 同款: 物体包围盒按位姿投影, 画黄色 3D 姿态框。"""
    output = np.asarray(image).copy()
    pose = np.asarray(camera_from_object, dtype=np.float64).reshape(4, 4)
    if mesh_bounds is None:
        return output
    mn, mx = (np.asarray(mesh_bounds[0], dtype=np.float64),
              np.asarray(mesh_bounds[1], dtype=np.float64))
    corners = np.array([[x, y, z]
                        for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1])
                        for z in (mn[2], mx[2])])
    points = corners @ pose[:3, :3].T + pose[:3, 3]
    if np.any(points[:, 2] <= 0.02):
        return output
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    px = points[:, 0] * K[0, 0] / points[:, 2] + K[0, 2]
    py = points[:, 1] * K[1, 1] / points[:, 2] + K[1, 2]
    pix = np.rint(np.column_stack([px, py])).astype(int)
    edges = [(0, 1), (0, 2), (0, 4), (7, 6), (7, 5), (7, 3),
             (1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]
    for a, b in edges:
        cv2.line(output, tuple(pix[a]), tuple(pix[b]), (0, 255, 255), 2, cv2.LINE_AA)
    return output


def draw_mesh_contour_2d(image, camera_from_object, mesh_path, camera_matrix,
                         scale=1.0):
    """旧版 draw_2d 同款: 模型在估计位姿下的投影轮廓(绿色)。"""
    output = np.asarray(image).copy()
    try:
        import trimesh
        mesh = trimesh.load(str(mesh_path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    except Exception:
        return output
    pose = np.asarray(camera_from_object, dtype=np.float64).reshape(4, 4)
    points = vertices @ pose[:3, :3].T + pose[:3, 3]
    z = points[:, 2]
    if np.all(z <= 0.02):
        return output
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = points[:, 0] * K[0, 0] / np.where(z > 0.02, z, 1e-9) + K[0, 2]
        py = points[:, 1] * K[1, 1] / np.where(z > 0.02, z, 1e-9) + K[1, 2]
    good = np.all(z[faces] > 0.02, axis=1)
    polygons = [np.column_stack([px[f], py[f]]).astype(np.int32)
                for f in faces[good]]
    silhouette = np.zeros(output.shape[:2], dtype=np.uint8)
    if polygons:
        cv2.fillPoly(silhouette, polygons, 255)
    if not silhouette.any():
        return output
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


def load_hand_eye_tcp_from_camera(competition_yaml: Path) -> np.ndarray:
    """从 competition.yaml 读取现场手眼标定 T_tcp_color_camera(15/16 内点)。"""
    data = yaml.safe_load(competition_yaml.read_text(encoding="utf-8"))
    entry = data["hand_eye"]["tcp_from_color_camera"]
    if entry.get("valid") is not True:
        raise RuntimeError("hand_eye.tcp_from_color_camera.valid != true")
    from competition_pipeline.geometry import as_transform

    return as_transform(
        np.asarray(entry["matrix"], dtype=np.float64),
        "tcp_from_color_camera",
    )


def load_gripper_geometry(competition_yaml: Path) -> dict:
    """读 competition.yaml 的 ``gripper_geometry``；缺省用 grasp_geometry 的默认值。"""
    from competition_pipeline.grasp_geometry import (
        JAW_CAVITY_DEPTH_MM, SAFETY_CLEARANCE_MM,
    )
    data = yaml.safe_load(competition_yaml.read_text(encoding="utf-8")) or {}
    section = data.get("gripper_geometry", {}) or {}
    return {
        "jaw_cavity_depth_mm": float(
            section.get("jaw_cavity_depth_mm", JAW_CAVITY_DEPTH_MM)
        ),
        "safety_clearance_mm": float(
            section.get("safety_clearance_mm", SAFETY_CLEARANCE_MM)
        ),
        "jaw_max_open_mm": section.get("jaw_max_open_mm"),
        "width_margin_mm": float(
            (data.get("grasp_planning", {}) or {}).get("width_margin_mm", 6.0)
        ),
    }


def apply_grasp_height_rule(user1_grasp, user1_from_object, mesh_bounds_m,
                            grasp_type, gripper_geometry):
    """把抓取点的 XYZ 换成"不会压爆、也不会怼进桌面"的位置。

    返回 ``(修正后的 4x4, 说明 dict 或 None)``。姿态原样保留 —— 这里只动位置。

    规则(见 :mod:`competition_pipeline.grasp_geometry`)::

        瓶/罐 (cylinder)   z = (顶点 + 中心)/2   -> 伸入深度 = 高度/4
        水果   (其余)       z = 中心
        统一钳位 伸入 <= 腔体深度 - 安全余量

    XY 取**包围盒中心**而不是网格原点。``mesh_bounds_m`` 为 None 时原样返回并给出
    说明，由调用方决定是否放行 —— 没有物体尺寸就无法保证不压爆。
    """
    from competition_pipeline.grasp_geometry import (
        check_graspable, grasp_height_mm, object_extent_user1,
    )

    if mesh_bounds_m is None:
        return user1_grasp, {
            "available": False,
            "reasons": ["缺少 CAD 包围盒，无法判断伸进夹爪的深度是否超过腔体"],
        }
    extent = object_extent_user1(user1_from_object, mesh_bounds_m, grasp_type)
    height = grasp_height_mm(
        extent, grasp_type,
        jaw_cavity_depth_mm=gripper_geometry["jaw_cavity_depth_mm"],
        safety_clearance_mm=gripper_geometry["safety_clearance_mm"],
    )
    corrected = np.asarray(user1_grasp, dtype=np.float64).copy()
    corrected[:3, 3] = np.asarray(
        [extent.center_xy_mm[0], extent.center_xy_mm[1], height.z_mm],
        dtype=np.float64,
    ) / 1000.0
    reasons = []
    if gripper_geometry.get("jaw_max_open_mm") is not None:
        reasons = check_graspable(
            extent, height,
            jaw_max_open_mm=float(gripper_geometry["jaw_max_open_mm"]),
            width_margin_mm=float(gripper_geometry["width_margin_mm"]),
        )
    return corrected, {
        "available": True,
        "rule": height.rule,
        "z_mm": height.z_mm,
        "engage_mm": height.engage_mm,
        "requested_engage_mm": height.requested_engage_mm,
        "clamped": height.clamped,
        "object_height_mm": extent.height_mm,
        "object_top_mm": extent.z_top_mm,
        "grasp_width_mm": extent.grasp_width_mm,
        "reasons": reasons,
    }


def compose_user1_object(user1_from_tcp, tcp_from_camera, camera_from_object):
    """Map one camera-frame object pose into controller UCS1."""
    from competition_pipeline.geometry import as_transform

    return as_transform(user1_from_tcp, "user1_from_tcp") @ as_transform(
        tcp_from_camera, "tcp_from_camera"
    ) @ as_transform(camera_from_object, "camera_from_object")


def user1_pose_values(user1_from_object):
    """Return UI-ready ``(XYZ mm, ABC deg)`` under NexBot convention."""
    from competition_pipeline.geometry import inexbot_abc_from_transform

    xyz_m, abc_rad = inexbot_abc_from_transform(user1_from_object)
    return np.asarray(xyz_m) * 1000.0, np.degrees(np.asarray(abc_rad))


class LiveTcpPoseReader:
    """Best-effort live read of ``T_user1_tcp`` from the NexBot controller.

    Uses the same verified path as the competition UI
    (``competition_pipeline.tcp_pose`` -> official 7000-port ``0x9512`` state
    service, ``pose_frame=UCS``), so the object pose can be mapped into
    用户坐标系1 with the arm pose that is actually current at capture time
    instead of a hand-typed constant.

    ``read()`` is a one-shot state query (read-only, never sends motion).
    """

    def __init__(self, competition_yaml: Path, host: str = ""):
        from competition_pipeline.tcp_pose import (
            NexBotTcpPoseSource,
            pose_endpoint_from_config,
        )

        data = yaml.safe_load(competition_yaml.read_text(encoding="utf-8"))
        settings = json.loads(json.dumps(data.get("controller", {}) or {}))
        if host:
            tcp_section = settings.setdefault("nexbot_tcp", {})
            tcp_section["host"] = str(host)
        # Field-verified: background 0x7266 can interleave with MOVL 0x4502
        # and make this controller reject both commands.
        settings.setdefault("nexbot_tcp", {})["heartbeat_s"] = 0.0
        if not settings.get("nexbot_tcp"):
            raise RuntimeError("competition.yaml 缺少 controller.nexbot_tcp 配置")
        self._endpoint = pose_endpoint_from_config(settings)
        if str(self._endpoint.pose_frame).upper() != "UCS":
            raise RuntimeError(
                "用户系映射要求 controller.nexbot_tcp.pose_frame=UCS"
            )
        self._source = None
        self._closed = False

    def _ensure_source(self):
        from competition_pipeline.tcp_pose import NexBotTcpPoseSource

        if self._source is None:
            self._source = NexBotTcpPoseSource(self._endpoint).connect()
        return self._source

    def read(self) -> np.ndarray:
        """Return ``T_user1_tcp`` (4x4, frame=用户坐标系1).

        On failure the connection is dropped so the next call reconnects
        (the single-client state port may be held by the competition UI or
        the controller comms service may be restarting).
        """
        from competition_pipeline.geometry import transform_from_inexbot_abc

        try:
            xyz_mm, abc_deg = self._ensure_source().read()
        except Exception:
            self._drop_source()
            raise
        return transform_from_inexbot_abc(
            np.asarray(xyz_mm, dtype=np.float64) / 1000.0,
            np.radians(np.asarray(abc_deg, dtype=np.float64)),
        )

    def _drop_source(self):
        if self._source is not None:
            try:
                self._source.close()
            except Exception:
                pass
            self._source = None

    def detach_controller(self):
        """Transfer the persistent controller without closing 6001/7000.

        The controller accepts one client per port.  Closing this controller
        and immediately constructing another one races the controller's slot
        release and produces ``Connection refused``.  Execution therefore
        takes ownership of the exact controller already used for visual TCP
        reads.
        """
        source = self._ensure_source()
        controller = source.controller
        if controller is None:
            raise RuntimeError("视觉 TCP reader 尚未建立 controller")
        # NexBotTcpPoseSource owns this reference; detach it so source.close()
        # cannot close the transferred persistent sockets.
        source._controller = None
        self._source = None
        self._closed = True
        return controller

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._drop_source()


def render_workspace_3d(workspace_from_object, grasp, mesh_bounds, pose_frame,
                        debug_dir, out_name="workspace_3d.png"):
    """旧版 vision_node 同款 3D 工作台视图: 工作系原点 + 物体位姿框 + 抓取三叉戟."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib import font_manager

    for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
        try:
            font_manager.fontManager.addfont(_fp)
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei",
                                       "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # 工作系原点三叉戟(0.12 m, 同旧版 draw_3d)
    length = 0.12
    for i, (color, label) in enumerate(zip(["r", "g", "b"], ["X", "Y", "Z"])):
        v = np.zeros(3)
        v[i] = length
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=color, lw=2)
        ax.text(v[0] * 1.1, v[1] * 1.1, v[2] * 1.1, label, color=color)

    # 物体位姿框(mesh bounds 米制, 由工作系位姿旋转平移)
    if mesh_bounds is not None:
        mn, mx = (np.asarray(mesh_bounds[0], dtype=np.float64),
                  np.asarray(mesh_bounds[1], dtype=np.float64))
        corners = np.array([[x, y, z]
                            for x in (mn[0], mx[0])
                            for y in (mn[1], mx[1])
                            for z in (mn[2], mx[2])])
        pts = corners @ workspace_from_object[:3, :3].T + workspace_from_object[:3, 3]
        edges = [(0, 1), (0, 2), (0, 4), (7, 6), (7, 5), (7, 3),
                 (1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]
        for a, b in edges:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    [pts[a, 2], pts[b, 2]], color="y", lw=2)
    # 物体自身坐标轴
    origin = workspace_from_object[:3, 3]
    for i, color in enumerate(["r", "g", "b"]):
        d = workspace_from_object[:3, i] * length * 0.5
        ax.plot([origin[0], origin[0] + d[0]], [origin[1], origin[1] + d[1]],
                [origin[2], origin[2] + d[2]], color=color, lw=1.5)
    # 抓取三叉戟 + 点(加粗加大, 突出显示)
    g0 = grasp[:3, 3]
    for i, color in enumerate(["m", "c", "k"]):
        d = grasp[:3, i] * 0.15
        ax.plot([g0[0], g0[0] + d[0]], [g0[1], g0[1] + d[1]],
                [g0[2], g0[2] + d[2]], color=color, lw=3, linestyle="--")
    ax.scatter([g0[0]], [g0[1]], [g0[2]], color="magenta", s=60, edgecolors="k")
    ax.text(g0[0], g0[1], g0[2] + 0.02, "  grasp", color="magenta", fontsize=9)

    span = 0.35
    ax.set_xlim(origin[0] - span, origin[0] + span)
    ax.set_ylim(origin[1] - span, origin[1] + span)
    ax.set_zlim(origin[2] - span, origin[2] + span)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    g_mm = g0 * 1000.0
    ax.set_title("{} | grasp t=({:.0f}, {:.0f}, {:.0f}) mm".format(
        pose_frame, g_mm[0], g_mm[1], g_mm[2]))
    fig.tight_layout()
    out_dir = Path(debug_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / out_name)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


class LegacyArmClient:
    """Optional client for the legacy simulated arm_node.py only."""

    def __init__(self, endpoint=""):
        self.endpoint = str(endpoint or "").strip()
        self.context = None
        self.socket = None
        if not self.endpoint:
            return
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 3600 * 1000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.socket.connect(self.endpoint)

    @property
    def enabled(self):
        return self.socket is not None

    def execute(self, grasp):
        if not self.enabled:
            return {"status": "dry_run"}
        self.socket.send_pyobj(np.asarray(grasp, dtype=np.float64))
        return self.socket.recv_pyobj()

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None


class OakVisionNode:
    def __init__(
        self,
        root,
        visual_config: VisualGraspConfig,
        oak_settings: dict,
        maximum_sync_delta_s: float,
        tag_provider: TagPoseProvider,
        arm_service="",
        tcp_xyz_mm=None,
        tcp_rpy_deg=None,
        controller_host="",
        no_tcp_read=False,
        place_x_mm=UCS_PLACE_X_MM,
        place_y_mm=UCS_PLACE_Y_MM,
        enable_robot_motion=False,
    ):
        self.root = root
        self.config = visual_config
        self.maximum_sync_delta_s = float(maximum_sync_delta_s)
        self.tag_provider = tag_provider
        self.source = build_oak_source(oak_settings)
        self.arm = LegacyArmClient(arm_service)
        # 用户坐标系链: T_user1_object = T_user1_tcp @ T_tcp_color_camera @ T_camera_from_object
        # T_user1_tcp 优先控制器实时回读(coord=3 用户系1), 也可用
        # --tcp-xyz-mm/--tcp-rpy-deg 静态指定; 两者都没有时用户1映射标记为不可用
        # (绝不静默退化为"用户1原点"单位阵, 那会产生错误坐标)。
        competition_yaml = (
            Path(__file__).resolve().parents[2]
            / "competition_pipeline" / "config" / "competition.yaml"
        )
        self._competition_yaml = competition_yaml
        self._controller_host = str(controller_host or "")
        self.place_x_mm = float(place_x_mm)
        self.place_y_mm = float(place_y_mm)
        self.enable_robot_motion = bool(enable_robot_motion)
        self._jog = None
        self._last_grasp_xyz_mm = None
        self._last_place_xyz_mm = None
        #: 最近一次抓取高度后处理的说明（见 apply_grasp_height_rule）
        self._last_grasp_height_info = None
        self._gripper_geometry = load_gripper_geometry(competition_yaml)
        self._hand_eye = load_hand_eye_tcp_from_camera(competition_yaml)
        self._live_reader = None
        self._tcp_user1 = None
        self._tcp_timestamp = None
        self._tcp_source = ""
        if tcp_xyz_mm is not None and tcp_rpy_deg is not None:
            # 控制器约定 A/B/C(RxRyRz, 与手眼样本修正一致)
            from competition_pipeline.geometry import transform_from_inexbot_abc
            self._tcp_user1 = transform_from_inexbot_abc(
                np.asarray(tcp_xyz_mm, dtype=np.float64) / 1000.0,
                np.radians(np.asarray(tcp_rpy_deg, dtype=np.float64)),
            )
            self._tcp_source = "手动指定 --tcp-xyz-mm/--tcp-rpy-deg (静态)"
        elif no_tcp_read:
            self._tcp_source = "--no-tcp-read: 未读取机械臂位姿"
        else:
            try:
                self._live_reader = LiveTcpPoseReader(
                    competition_yaml, host=controller_host
                )
                self._tcp_user1 = self._live_reader.read()
                self._tcp_timestamp = time.strftime("%H:%M:%S")
                self._tcp_source = "控制器实时回读 (用户系1 coord=3)"
            except Exception as error:
                self._tcp_user1 = None
                self._tcp_source = "控制器回读失败: {} —— 用户1映射不可用".format(
                    error
                )
        self.model = None
        self.tracker = StableTracker(max_miss=10)
        self.active_estimator = None
        self.active_estimator_key = None
        self.objects = []
        self.sequence_items = []
        self.latest_snapshot = None
        self.busy = False
        self.closed = False
        self.ui_queue = queue.Queue()

        root.title("OAK 视觉抓取节点（原版兼容 · 默认 Dry-run）")
        root.geometry("1420x840")
        root.minsize(1080, 680)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        root.after(50, self._update_ui)
        self._set_busy(True, "正在连接 OAK-D-PRO-FF……")
        threading.Thread(target=self._connect_camera, daemon=True).start()

    def _refresh_tcp_live(self):
        """Refresh ``T_user1_tcp`` and fail closed on every read error."""
        if self._jog is not None:
            # After the first execution, vision and motion intentionally share
            # the same persistent controller instead of reopening 7000/6001.
            try:
                xyz_mm, abc_deg = self._jog.current_pose()
                from competition_pipeline.geometry import transform_from_inexbot_abc

                self._tcp_user1 = transform_from_inexbot_abc(
                    np.asarray(xyz_mm, dtype=np.float64) / 1000.0,
                    np.radians(np.asarray(abc_deg, dtype=np.float64)),
                )
                self._tcp_timestamp = time.strftime("%H:%M:%S")
                self._tcp_source = "视觉/执行共享 controller (用户系1 coord=3)"
            except Exception as error:
                self._tcp_user1 = None
                self._tcp_timestamp = None
                self._tcp_source = "共享 controller 回读失败: {}".format(error)
            return self._tcp_user1
        if self._live_reader is None:
            # 执行抓取时已释放回读连接(7000 单客户端, 交给执行器独占);
            # 下次视觉计算前按需重建。
            try:
                self._live_reader = LiveTcpPoseReader(
                    self._competition_yaml, host=self._controller_host
                )
            except Exception as error:
                self._tcp_user1 = None
                self._tcp_timestamp = None
                self._tcp_source = "控制器回读失败: {} —— 用户1映射不可用".format(
                    error
                )
                return None
        try:
            self._tcp_user1 = self._live_reader.read()
            self._tcp_timestamp = time.strftime("%H:%M:%S")
            self._tcp_source = "控制器实时回读 (用户系1 coord=3)"
        except Exception as error:
            # Eye-in-hand mapping must never combine a fresh image with a
            # stale robot pose.  Keep the error text, but invalidate the pose.
            self._tcp_user1 = None
            self._tcp_timestamp = None
            self._tcp_source = "控制器回读失败: {} —— 用户1映射不可用".format(
                error
            )
        return self._tcp_user1

    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill="x")
        self.capture_button = ttk.Button(
            toolbar, text="拍照识别", command=self.on_capture
        )
        self.capture_button.pack(side="left")
        self.object_combo = ttk.Combobox(toolbar, state="readonly", width=28)
        self.object_combo.pack(side="left", padx=8)
        ttk.Button(toolbar, text="＋加入序列", command=self.on_add).pack(side="left")
        ttk.Button(toolbar, text="清空序列", command=self.on_clear).pack(
            side="left", padx=8
        )

        sequence_row = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        sequence_row.pack(fill="x")
        ttk.Label(sequence_row, text="目标序列：").pack(side="left")
        self.sequence_entry = ttk.Entry(sequence_row, width=56)
        self.sequence_entry.pack(side="left", fill="x", expand=True, padx=6)
        button_text = (
            "开始序列（旧模拟机械臂）" if self.arm.enabled
            else "开始算法序列（Dry-run，不运动）"
        )
        self.start_button = ttk.Button(
            sequence_row, text=button_text, command=self.on_start
        )
        self.start_button.pack(side="left")

        grasp_row = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        grasp_row.pack(fill="x")
        ttk.Label(grasp_row, text="一键抓取(用户1)：").pack(side="left")
        self.grasp_button = ttk.Button(
            grasp_row, text="执行抓取", command=self.on_grasp_execute,
            state="disabled",
        )
        self.grasp_button.pack(side="left", padx=6)
        self.estop_button = ttk.Button(
            grasp_row, text="急停", command=self.on_grasp_estop,
            state="disabled",
        )
        self.estop_button.pack(side="left")
        self.grasp_info = ttk.Label(grasp_row, text="先运行目标序列计算坐标")
        self.grasp_info.pack(side="left", padx=8)

        content = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        content.pack(fill="both", expand=True)
        self.image_label = tk.Label(content, bg="#11181c")
        self.image_label.pack(side="left", fill="both", expand=True)
        self.image3d_label = tk.Label(content, bg="#14171a")
        self.image3d_label.pack(side="left", fill="both", expand=True, padx=(10, 0))
        result_frame = ttk.Frame(content, width=410)
        result_frame.pack(side="right", fill="y", padx=(10, 0))
        ttk.Label(result_frame, text="算法结果").pack(anchor="w")
        self.result_text = tk.Text(
            result_frame, width=48, height=32, wrap="word",
            background="#20272b", foreground="#e4ecef",
        )
        self.result_text.pack(fill="both", expand=True, pady=(5, 0))
        self.result_text.insert(
            "end",
            "相机：OAK-D-PRO-FF\n"
            "流程：YOLO → FoundationPose → 手眼矩阵 → 用户坐标系1(UCS1)\n"
            "位姿输出：物体/抓取点均已映射到用户坐标系1；相机系仅作参考\n"
            "安全：默认 Dry-run，不会发送真实机械臂运动。\n",
        )

        self.status = ttk.Label(
            self.root, text="准备启动", anchor="w", padding=(10, 5, 10, 10)
        )
        self.status.pack(fill="x")

    def _connect_camera(self):
        try:
            self.source.start()
            snapshot = self._grab_snapshot(timeout_s=12.0)
            self.latest_snapshot = snapshot
            self.ui_queue.put(("image", snapshot.color_bgr))
            self.ui_queue.put(("status", "OAK 已连接：1920×1080 RGB-D，点击“拍照识别”"))
        except Exception as error:
            self.ui_queue.put(("error", ("OAK 启动失败", str(error))))
        finally:
            self.ui_queue.put(("busy", False))

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.config.yolo_weights)
        return self.model

    def _grab_snapshot(self, timeout_s=5.0):
        deadline = time.monotonic() + float(timeout_s)
        bundle = None
        while time.monotonic() < deadline and not self.closed:
            bundle = self.source.latest()
            if bundle is not None and bundle.depth_m is not None:
                if bundle.sync_delta_s is not None and (
                    bundle.sync_delta_s <= self.maximum_sync_delta_s
                ):
                    break
            time.sleep(0.01)
        if bundle is None or bundle.depth_m is None:
            raise RuntimeError("等待 OAK RGB-D 帧超时")
        if bundle.sync_delta_s is None or bundle.sync_delta_s > self.maximum_sync_delta_s:
            raise RuntimeError(
                "OAK RGB-D 时间差 {:.1f} ms 超过 {:.1f} ms".format(
                    -1.0 if bundle.sync_delta_s is None else bundle.sync_delta_s * 1000.0,
                    self.maximum_sync_delta_s * 1000.0,
                )
            )
        intrinsics = bundle.color_intrinsics
        if intrinsics is None:
            raise RuntimeError("OAK EEPROM RGB 内参不可用")
        color = rectify_color_image(bundle.color_bgr, intrinsics)
        depth = rectify_aligned_depth_image(bundle.depth_m, intrinsics)
        return OakSnapshot(
            color_bgr=color,
            depth_m=depth,
            intrinsics=rectified_intrinsics(intrinsics),
            timestamp_s=float(bundle.color_timestamp_s),
            sync_delta_s=float(bundle.sync_delta_s),
        )

    def _detect(self, snapshot):
        return detect_all_track(
            snapshot.color_bgr,
            self._load_model(),
            self.tracker,
            conf=self.config.yolo_conf,
            imgsz=self.config.yolo_imgsz,
        )

    def on_capture(self):
        if self.busy:
            return
        self._set_busy(True, "正在拍照并运行 YOLO……")
        threading.Thread(target=self._capture_worker, daemon=True).start()

    def _capture_worker(self):
        try:
            snapshot = self._grab_snapshot()
            objects = self._detect(snapshot)
            self.latest_snapshot = snapshot
            self.objects = objects
            self.ui_queue.put(("objects", objects))
            self.ui_queue.put(("image", draw_boxes(snapshot.color_bgr, objects)))
            self.ui_queue.put((
                "status",
                "识别到 {} 个物体；RGB-D 时间差 {:.2f} ms".format(
                    len(objects), snapshot.sync_delta_s * 1000.0
                ),
            ))
        except Exception as error:
            self.ui_queue.put(("error", ("拍照识别失败", str(error))))
        finally:
            self.ui_queue.put(("busy", False))

    def on_add(self):
        index = self.object_combo.current()
        if index < 0 or index >= len(self.objects):
            self.status.configure(text="请先拍照识别并选择物体")
            return
        item = self.objects[index]
        self.sequence_items.append("{}#{}".format(item["name"], item.get("id")))
        self.sequence_entry.delete(0, tk.END)
        self.sequence_entry.insert(0, ", ".join(self.sequence_items))

    def on_clear(self):
        self.sequence_items = []
        self.sequence_entry.delete(0, tk.END)

    def on_start(self):
        if self.busy:
            return
        try:
            sequence = parse_sequence(self.sequence_entry.get())
        except Exception as error:
            messagebox.showerror("序列格式错误", str(error), parent=self.root)
            return
        if not sequence:
            messagebox.showerror(
                "序列为空", "示例：can#1, green_apple#2", parent=self.root
            )
            return
        self._set_busy(True, "开始运行目标序列……")
        threading.Thread(
            target=self._sequence_worker, args=(sequence,), daemon=True
        ).start()

    def _estimator(self, object_key, mesh_path):
        scale = self.config.mesh_scale_for_object(object_key)
        key = (str(mesh_path), float(scale))
        if self.active_estimator is None or self.active_estimator_key != key:
            if self.active_estimator is not None:
                self.active_estimator.close()
            self.active_estimator = FoundationPosePoseEstimator(
                foundationpose_root=self.config.foundationpose_root,
                mesh_path=mesh_path,
                mesh_scale_to_meters=scale,
                debug_dir=self.config.debug_dir,
                est_refine_iter=self.config.est_refine_iter,
                track_refine_iter=self.config.track_refine_iter,
                device=self.config.device,
                use_mask_center_guidance=self.config.use_mask_center_guidance,
                registration_max_hypotheses=(
                    self.config.foundationpose_registration_hypotheses
                ),
            )
            self.active_estimator_key = key
        return self.active_estimator

    def _release_yolo_cuda(self):
        """Keep YOLO available on CPU while reserving CUDA for FoundationPose."""

        if self.model is None:
            return
        try:
            self.model.to("cpu")
            # Ultralytics caches an AutoBackend on the predictor. Rebuilding
            # it later avoids retaining CUDA buffers after moving the model.
            self.model.predictor = None
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _process_target(self, name, instance):
        snapshot = self._grab_snapshot()
        # Freeze the TCP immediately after the RGB-D snapshot. FoundationPose
        # may take seconds; reading TCP afterwards would pair an old image with
        # a newer arm pose and produce a false user-frame coordinate.
        self._refresh_tcp_live()
        tcp_user1_at_capture = (
            None if self._tcp_user1 is None else self._tcp_user1.copy()
        )
        tcp_source_at_capture = str(self._tcp_source)
        tcp_timestamp_at_capture = self._tcp_timestamp
        objects = self._detect(snapshot)
        target = find_sequence_target(objects, name, instance)
        if target is None:
            raise RuntimeError("目标 {}#{} 不在当前画面".format(name, instance or "*"))
        object_key = self.config.resolve_object_key(target["name"], target["cls"])
        mesh_path = self.config.mesh_for_object(object_key)
        if not mesh_path or not Path(mesh_path).is_file():
            raise RuntimeError("物体 {} 没有可用 CAD 网格".format(object_key))
        # 掩膜: 直接用 YOLO 识别框(矩形)作为掩膜, 不使用实例分割输出
        mask = np.zeros(snapshot.depth_m.shape, dtype=np.uint8)
        bx1, by1, bx2, by2 = (int(round(v)) for v in target["xyxy"])
        mask[max(0, by1):min(mask.shape[0], by2),
             max(0, bx1):min(mask.shape[1], bx2)] = 255
        coverage = depth_coverage(snapshot.depth_m, mask)
        valid_depth = int(np.count_nonzero((snapshot.depth_m > 0.0) & (mask > 0)))
        if valid_depth < 30:
            raise RuntimeError("目标 Mask 内有效深度不足：{} 点".format(valid_depth))

        fp_input = prepare_foundationpose_input(
            snapshot,
            target,
            mask,
            padding_pixels=self.config.foundationpose_roi_padding_pixels,
            maximum_size=self.config.foundationpose_max_input_size,
        )
        self._release_yolo_cuda()
        estimator = self._estimator(object_key, mesh_path)
        try:
            camera_from_object = estimator.register(
                fp_input.color_bgr,
                fill_depth_roi(fp_input.depth_m, fp_input.mask),
                fp_input.mask,
                fp_input.camera_matrix,
            )
        except Exception as error:
            if "out of memory" in str(error).lower():
                estimator.close()
                self.active_estimator = None
                self.active_estimator_key = None
                raise RuntimeError(
                    "FoundationPose 显存不足；已释放运行时，请关闭其他 CUDA 程序后重试"
                ) from error
            raise
        distance = float(camera_from_object[2, 3])
        if not 0.10 < distance < 3.0:
            raise RuntimeError("FoundationPose 距离异常：{:.3f} m".format(distance))

        tag_estimate, mapped_detections = self.tag_provider.estimate(
            snapshot.color_bgr,
            snapshot.intrinsics.matrix,
            snapshot.intrinsics.distortion,
        )
        overlay_base = self.tag_provider.draw_status(
            snapshot.color_bgr, mapped_detections, tag_estimate
        )
        tag_ids = sorted(int(tag_id) for tag_id in mapped_detections)
        workspace_valid = bool(tag_estimate.valid)
        if workspace_valid:
            workspace_from_object = (
                np.asarray(tag_estimate.workspace_from_camera, dtype=np.float64)
                @ camera_from_object
            )
            pose_frame = str(
                self.tag_provider.layout.get("workspace_frame", "AprilTag 工作台")
            )
        else:
            # Compatibility fallback for the original single tag0 workspace.
            legacy_tags = detect_tags(
                snapshot.color_bgr,
                self.config.tag_size_mm,
                snapshot.intrinsics.matrix,
                snapshot.intrinsics.distortion.reshape(-1, 1),
            )
            tag_world = build_world_from_tags(legacy_tags)
            if tag_world is not None:
                workspace_valid = True
                workspace_from_object = to_world_and_compensate(
                    camera_from_object,
                    np.linalg.inv(tag_world),
                    offset_xy_mm=self.config.offset_xy_mm,
                    center_offset_mm=self.config.center_offset_mm,
                    flip_x=self.config.flip_x,
                    flip_y=self.config.flip_y,
                )
                pose_frame = "legacy tag0 工作台"
                tag_ids = [int(tag_id) for tag_id, _ in legacy_tags]
            else:
            # Useful for camera/algorithm verification, but never eligible for
            # an arm request because it is not expressed in a robot/workspace frame.
                workspace_from_object = camera_from_object.copy()
                pose_frame = "相机光学系（无有效 Tag 地图，禁止执行）"

        rule = self.config.rule_for_object(object_key)
        grasp = (
            compute_grasp_sphere(workspace_from_object, rule.offset_mm)
            if rule.type == "sphere"
            else compute_grasp(workspace_from_object, rule.offset_mm)
        )
        Path(self.config.pose_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.grasp_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        np.save(self.config.pose_file, workspace_from_object)
        np.save(self.config.grasp_file, grasp)

        # 3D 工作台视图: 用户坐标系(UCS1) 基准
        # T_user1_object = T_user1_tcp @ T_tcp_color_camera @ T_camera_from_object
        camera_grasp = None
        user1_from_object = None
        user1_grasp = None
        user1_grasp_abc_deg = None
        viz3d = None
        user1_object_abc_deg = None
        if tcp_user1_at_capture is not None:
            try:
                camera_grasp = (
                    compute_grasp_sphere(camera_from_object, rule.offset_mm)
                    if rule.type == "sphere"
                    else compute_grasp(camera_from_object, rule.offset_mm)
                )
                user1_from_object = compose_user1_object(
                    tcp_user1_at_capture, self._hand_eye, camera_from_object
                )
                user1_grasp = (
                    compute_grasp_sphere(user1_from_object, rule.offset_mm)
                    if rule.type == "sphere"
                    else compute_grasp(user1_from_object, rule.offset_mm)
                )
                # ⚠️ compute_grasp/compute_grasp_sphere 把抓取点放在**网格原点**上
                # (docstring 原话 "grip the middle")。这有两个会毁掉物体的后果:
                #   - 高物体: 伸进夹爪的深度 = 高度/2。腔体只有 80mm，
                #     245mm 的可乐瓶要求 122.5mm -> 掌根压爆瓶口(隔壁组已发生);
                #   - 网格原点未必是几何中心: apple 偏 49.1mm、nescafe 偏 74.5mm，
                #     按原点抓会往桌面下方伸 -> 把夹爪怼进桌子。
                # 这里做纯后处理，只改抓取点的位置，不动姿态、不动 FoundationPose、
                # 不改任何坐标约定。规则见 competition_pipeline.grasp_geometry。
                user1_grasp, grasp_height_info = apply_grasp_height_rule(
                    user1_grasp,
                    user1_from_object,
                    estimator.mesh_bounds,
                    rule.type,
                    self._gripper_geometry,
                )
                if grasp_height_info is not None:
                    self._last_grasp_height_info = grasp_height_info
                _, user1_object_abc_deg = user1_pose_values(user1_from_object)
                _, user1_grasp_abc_deg = user1_pose_values(user1_grasp)
                viz3d = render_workspace_3d(
                    user1_from_object,
                    user1_grasp,
                    estimator.mesh_bounds,
                    "用户坐标系 (UCS1)",
                    self.config.debug_dir,
                )
                user1_out = Path(self.config.grasp_file).with_name(
                    "grasp_user1.npy"
                )
                object_user1_out = Path(self.config.pose_file).with_name(
                    "object_pose_user1.npy"
                )
                np.save(user1_out, user1_grasp)
                np.save(object_user1_out, user1_from_object)
                print("[用户系] tcp:", tcp_source_at_capture)
                print("[用户系] grasp XYZ(mm):",
                      np.round(user1_grasp[:3, 3] * 1000.0, 2).tolist(),
                      "ABC(deg):", np.round(user1_grasp_abc_deg, 2).tolist())
            except Exception as error:
                print("[3d] 3D 视图渲染失败(不影响算法结果):", error)
                viz3d = None
        else:
            print("[用户系] 无机械臂实时位姿: 跳过用户1映射 ({}); 相机系结果不受影响".format(
                tcp_source_at_capture))

        overlay = draw_boxes(overlay_base, objects)
        # 旧版视觉: 绿框=模型投影轮廓+掩膜轮廓, 黄框=姿态框, 末尾坐标轴
        overlay = draw_mask_overlay(overlay, mask)
        overlay = draw_mesh_contour_2d(
            overlay, camera_from_object, mesh_path, snapshot.intrinsics.matrix,
            self.config.mesh_scale_for_object(object_key),
        )
        overlay = draw_pose_box_2d(
            overlay, camera_from_object, estimator.mesh_bounds,
            snapshot.intrinsics.matrix,
        )
        overlay = draw_pose_axes(
            overlay, camera_from_object, snapshot.intrinsics.matrix
        )
        return {
            "target": target,
            "object_key": object_key,
            "camera_from_object": camera_from_object,
            "workspace_from_object": workspace_from_object,
            "grasp": grasp,
            "camera_grasp": camera_grasp,
            "user1_from_object": user1_from_object,
            "user1_object_abc_deg": user1_object_abc_deg,
            "user1_grasp": user1_grasp,
            "user1_grasp_abc_deg": user1_grasp_abc_deg,
            "tcp_source_at_capture": tcp_source_at_capture,
            "tcp_timestamp_at_capture": tcp_timestamp_at_capture,
            "viz3d": viz3d,
            "pose_frame": pose_frame,
            "workspace_valid": workspace_valid,
            "tag_ids": tag_ids,
            "depth_coverage": coverage,
            "valid_depth": valid_depth,
            "overlay": overlay,
        }

    def _sequence_worker(self, sequence):
        try:
            for name, instance in sequence:
                self.ui_queue.put((
                    "status", "计算目标 {}#{}……".format(name, instance or "*")
                ))
                result = self._process_target(name, instance)
                self.ui_queue.put(("image", result["overlay"]))
                if result.get("viz3d"):
                    self.ui_queue.put(("img3d", result["viz3d"]))
                camera_xyz = result["camera_from_object"][:3, 3] * 1000.0
                pose_xyz = result["workspace_from_object"][:3, 3] * 1000.0
                grasp_xyz = result["grasp"][:3, 3] * 1000.0
                camera_grasp_xyz = (
                    result["camera_grasp"][:3, 3] * 1000.0
                    if result["camera_grasp"] is not None
                    else np.zeros(3)
                )
                tcp_source = result["tcp_source_at_capture"]
                tcp_timestamp = result.get("tcp_timestamp_at_capture") or "--"
                if result["user1_from_object"] is None:
                    user1_text = "user1 object XYZ mm: 未映射（{}）\n".format(
                        tcp_source
                    )
                    user1grasp_text = "user1 grasp XYZ mm: 未映射\n"
                    user1objectabc_text = ""
                    user1abc_text = ""
                else:
                    user1_xyz = result["user1_from_object"][:3, 3] * 1000.0
                    user1_grasp_xyz = result["user1_grasp"][:3, 3] * 1000.0
                    user1_abc = result.get("user1_grasp_abc_deg")
                    object_abc = result.get("user1_object_abc_deg")
                    user1_text = "user1 object XYZ mm: {}\n".format(
                        np.round(user1_xyz, 2).tolist()
                    )
                    user1grasp_text = "user1 grasp XYZ mm: {}\n".format(
                        np.round(user1_grasp_xyz, 2).tolist()
                    )
                    user1objectabc_text = (
                        "user1 object ABC deg: {}\n".format(
                            np.round(object_abc, 2).tolist()
                        )
                        if object_abc is not None
                        else ""
                    )
                    user1abc_text = (
                        "user1 grasp ABC deg: {}\n".format(
                            np.round(user1_abc, 2).tolist()
                        )
                        if user1_abc is not None
                        else ""
                    )
                text = (
                    "target: {name} #{instance}\n"
                    "object_key: {object_key}\n"
                    "pose_frame: {pose_frame}\n"
                    "tags: {tags}\n"
                    "depth coverage: {coverage:.1%} ({points} points)\n"
                    "=== 用户坐标系1 (UCS1) ===\n"
                    "tcp 来源: {tcp_source} @ {tcp_timestamp}\n"
                    "{user1}{user1objectabc}{user1grasp}{user1abc}"
                    "--- 相机系 / 工作台系(参考) ---\n"
                    "camera object XYZ mm: {camera}\n"
                    "camera grasp XYZ mm: {cameragrasp}\n"
                    "pose XYZ mm: {pose}\n"
                    "grasp XYZ mm: {grasp}\n"
                ).format(
                    name=result["target"]["name"],
                    instance=result["target"].get("id"),
                    object_key=result["object_key"],
                    pose_frame=result["pose_frame"],
                    tags=result["tag_ids"],
                    coverage=result["depth_coverage"],
                    points=result["valid_depth"],
                    tcp_source=tcp_source,
                    tcp_timestamp=tcp_timestamp,
                    user1=user1_text,
                    user1objectabc=user1objectabc_text,
                    user1grasp=user1grasp_text,
                    user1abc=user1abc_text,
                    camera=np.round(camera_xyz, 2).tolist(),
                    cameragrasp=np.round(camera_grasp_xyz, 2).tolist(),
                    pose=np.round(pose_xyz, 2).tolist(),
                    grasp=np.round(grasp_xyz, 2).tolist(),
                )
                if self.arm.enabled:
                    if not result["workspace_valid"]:
                        text += "arm: BLOCKED（缺少工作台坐标）\n"
                    else:
                        response = self.arm.execute(result["grasp"])
                        text += "arm simulator: {}\n".format(response)
                else:
                    text += "arm: DRY-RUN（未发送运动）\n"
                if result.get("user1_grasp") is not None:
                    grasp = np.round(
                        result["user1_grasp"][:3, 3] * 1000.0, 2
                    ).tolist()
                    info = self._last_grasp_height_info or {}
                    blocked = list(info.get("reasons") or [])
                    if not info.get("available", False):
                        # 没有物体尺寸就无法保证伸入深度不超过腔体 —— 那正是压爆
                        # 的成因，宁可不放行也不要"尽力而为"。
                        blocked = blocked or ["缺少物体尺寸，无法判断是否会压爆"]
                    if blocked:
                        self._last_grasp_xyz_mm = None
                        self._last_place_xyz_mm = None
                        text += "⛔ 抓取被拒绝：{}\n".format("；".join(blocked))
                        self.ui_queue.put(("grasp_ready", None))
                        self.ui_queue.put(("result", text))
                        self.ui_queue.put(("status", "目标序列算法运行完成"))
                        continue
                    self._last_grasp_xyz_mm = grasp
                    self._last_place_xyz_mm = [
                        self.place_x_mm, self.place_y_mm, float(grasp[2])
                    ]
                    text += ("✋ 一键抓取已就绪：抓取 {} → 放置 {} （姿态=复位位置初始姿态）\n".format(
                        grasp, np.round(self._last_place_xyz_mm, 2).tolist()))
                    text += (
                        "   抓取高度规则：{rule}｜物体高 {h:.1f}mm 顶面 {top:.1f}mm"
                        "｜伸进夹爪 {e:.1f}mm（腔体 {c:.0f}mm）{clamp}\n"
                        "   夹持宽度 {w:.1f}mm\n".format(
                            rule=info.get("rule", "?"),
                            h=info.get("object_height_mm", float("nan")),
                            top=info.get("object_top_mm", float("nan")),
                            e=info.get("engage_mm", float("nan")),
                            c=self._gripper_geometry["jaw_cavity_depth_mm"],
                            clamp=("｜⚠️ 已按腔体深度抬高（原需 {:.1f}mm）".format(
                                info.get("requested_engage_mm", float("nan")))
                                if info.get("clamped") else ""),
                            w=info.get("grasp_width_mm", float("nan")),
                        )
                    )
                    self.ui_queue.put(("grasp_ready", (
                        list(grasp), list(self._last_place_xyz_mm)
                    )))
                else:
                    self._last_grasp_xyz_mm = None
                    self._last_place_xyz_mm = None
                    text += "✋ 一键抓取不可用（无用户1坐标）\n"
                    self.ui_queue.put(("grasp_ready", None))
                self.ui_queue.put(("result", text))
            self.ui_queue.put(("status", "目标序列算法运行完成"))
        except Exception as error:
            self.ui_queue.put(("error", ("序列运行失败", str(error))))
        finally:
            self.ui_queue.put(("busy", False))

    def _set_busy(self, busy, status=None):
        self.busy = bool(busy)
        state = "disabled" if self.busy else "normal"
        self.capture_button.configure(state=state)
        self.start_button.configure(state=state)
        # Emergency stop must remain clickable during real motion/busy work.
        self.estop_button.configure(state="normal")
        self.grasp_button.configure(state=(
            "disabled"
            if (self.busy or self._last_grasp_xyz_mm is None)
            else "normal"
        ))
        if status is not None:
            self.status.configure(text=str(status))

    # -- 一键抓取(用户坐标系1): 计算 -> 坐标确认 -> 执行 -------------------

    def on_grasp_execute(self):
        if self.busy:
            return
        if self._last_grasp_xyz_mm is None:
            messagebox.showwarning(
                "无可执行抓取",
                "请先运行目标序列，得到用户1抓取坐标。",
                parent=self.root,
            )
            return
        grasp = list(self._last_grasp_xyz_mm)
        place = [self.place_x_mm, self.place_y_mm, float(grasp[2])]
        dry_run = not self.enable_robot_motion
        if dry_run:
            proceed = messagebox.askokcancel(
                "Dry-run 验证（默认）",
                "未加 --enable-robot-motion，不会发送真实运动。\n\n"
                "抓取 XYZ(mm): {}\n"
                "放置 XYZ(mm): {}\n"
                "姿态: 复位位置初始姿态(只动 XYZ)\n\n"
                "确认生成执行计划？".format(
                    np.round(grasp, 2).tolist(), np.round(place, 2).tolist()
                ),
                parent=self.root,
            )
            if not proceed:
                return
        else:
            proceed = messagebox.askokcancel(
                "确认执行抓取（真实运动）",
                "⚠ 机械臂将实际运动！\n\n"
                "流程: 回复位 -> 抓取 {} -> 夹爪合 -> 放置 {} -> 夹爪开 -> 回复位\n"
                "姿态: 按复位位置初始姿态\n\n"
                "确认开始？".format(
                    np.round(grasp, 2).tolist(), np.round(place, 2).tolist()
                ),
                parent=self.root,
            )
            if not proceed:
                return
        self._set_busy(True, "执行抓取（{}）……".format(
            "Dry-run" if dry_run else "真实运动"
        ))
        threading.Thread(
            target=self._grasp_worker, args=(grasp, dry_run), daemon=True
        ).start()

    def _grasp_worker(self, grasp_mm, dry_run):
        try:
            if self._jog is None:
                # 7000/6001 均为单客户端: 不关闭后重连，直接把视觉
                # 回读的持久 controller 移交给执行器。
                shared_controller = None
                if self._live_reader is not None:
                    try:
                        shared_controller = self._live_reader.detach_controller()
                    except Exception as error:
                        raise UcsGraspExecutorError(
                            "无法复用视觉 controller: {}".format(error)
                        ) from error
                    self._live_reader = None
                self._jog = build_jog(
                    self._competition_yaml,
                    host=self._controller_host,
                    controller=shared_controller,
                )
            runner = UcsGraspRunner(
                self._jog,
                place_x_mm=self.place_x_mm,
                place_y_mm=self.place_y_mm,
                on_event=lambda message: self.ui_queue.put(("status", message)),
            )
            result = runner.execute(grasp_mm, dry_run=dry_run)
            if result["status"] == "ok":
                self.ui_queue.put((
                    "status",
                    "✅ 抓取-放置完成，已回到复位位置；放置实际 XYZ(mm)={}".format(
                        np.round(result["place_xyz_mm"], 2).tolist()
                    ),
                ))
            else:
                self.ui_queue.put(("status", "Dry-run 计划生成（未发送运动）"))
        except UcsGraspSafetyError as error:
            self.ui_queue.put(("error", ("安全拦截", str(error))))
        except UcsGraspExecutorError as error:
            self.ui_queue.put(("error", ("执行抓取失败", self._grasp_error_with_hint(error))))
        except Exception as error:
            self.ui_queue.put(("error", ("执行抓取失败", self._grasp_error_with_hint(error))))
        finally:
            self.ui_queue.put(("busy", False))

    def _grasp_error_with_hint(self, error) -> str:
        message = str(error)
        if "6001" in message and "connect" in message.lower():
            message += (
                "\n\n控制器实时命令口(6001)不可用。请依次确认："
                "① 示教器切换到远程/运行模式(或拔出示教器)；"
                "② 关闭其他占用 6001 的程序(比赛UI等，单客户端)；"
                "③ 控制器已上电且伺服就绪。"
            )
        return message

    def on_grasp_estop(self):
        def _stop():
            if self._jog is None:
                self.ui_queue.put(("status", "急停: 尚未建立机械臂连接"))
                return
            try:
                self._jog.emergency_stop()
                self.ui_queue.put(("status", "急停已发送"))
            except Exception as error:
                self.ui_queue.put(("status", "急停异常: {}".format(error)))

        threading.Thread(target=_stop, daemon=True).start()

    def _show_image(self, frame):
        rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_width, max_height = 920, 690
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        if scale < 1.0:
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)), Image.LANCZOS
            )
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def _show_image_3d(self, frame):
        """显示 3D 工作台视图(用户1坐标系), 同旧版 vision_node 的右侧 3D 画面。"""
        rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_width, max_height = 560, 560
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        if scale < 1.0:
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)), Image.LANCZOS
            )
        photo = ImageTk.PhotoImage(image)
        self.image3d_label.configure(image=photo)
        self.image3d_label.image = photo

    def _update_ui(self):
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "image":
                    self._show_image(value)
                elif kind == "img3d":
                    frame = cv2.imread(str(value))
                    if frame is not None:
                        self._show_image_3d(frame)
                elif kind == "objects":
                    self.objects = list(value)
                    labels = [
                        "{} #{}  {:.3f}".format(
                            item["name"], item.get("id"), item["conf"]
                        )
                        for item in self.objects
                    ]
                    self.object_combo["values"] = labels
                    if labels:
                        self.object_combo.current(0)
                elif kind == "result":
                    self.result_text.delete("1.0", tk.END)
                    self.result_text.insert("1.0", value)
                elif kind == "status":
                    self.status.configure(text=str(value))
                elif kind == "busy":
                    self._set_busy(bool(value))
                elif kind == "grasp_ready":
                    if value:
                        grasp, place = value
                        self.grasp_info.configure(text=(
                            "就绪 抓取{} → 放置{}".format(
                                np.round(grasp, 2).tolist(),
                                np.round(place, 2).tolist(),
                            )
                        ))
                    else:
                        self.grasp_info.configure(text="未计算（无用户1坐标）")
                    self.grasp_button.configure(state=(
                        "disabled"
                        if (self.busy or value is None)
                        else "normal"
                    ))
                elif kind == "error":
                    title, message = value
                    self.status.configure(text=message)
                    messagebox.showerror(title, message, parent=self.root)
        except queue.Empty:
            pass
        if not self.closed:
            self.root.after(50, self._update_ui)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.source.stop()
        finally:
            if self.active_estimator is not None:
                try:
                    self.active_estimator.close()
                except Exception:
                    pass
                self.active_estimator = None
            if self._live_reader is not None:
                try:
                    self._live_reader.close()
                except Exception:
                    pass
                self._live_reader = None
            if self._jog is not None:
                try:
                    self._jog.close()
                except Exception:
                    pass
                self._jog = None
            self.arm.close()
            self.root.destroy()


def camera_check(visual_config, oak_settings, maximum_sync_delta_s):
    source = build_oak_source(oak_settings)
    try:
        source.start()
        deadline = time.monotonic() + 12.0
        bundle = None
        while time.monotonic() < deadline:
            bundle = source.latest()
            if bundle is not None and bundle.depth_m is not None:
                break
            time.sleep(0.01)
        if bundle is None or bundle.depth_m is None:
            raise RuntimeError("OAK RGB-D frame timeout")
        if bundle.sync_delta_s > maximum_sync_delta_s:
            raise RuntimeError("OAK RGB-D synchronization failed")
        from ultralytics import YOLO

        objects = detect_all_track(
            bundle.color_bgr,
            YOLO(visual_config.yolo_weights),
            StableTracker(max_miss=0),
            conf=visual_config.yolo_conf,
            imgsz=visual_config.yolo_imgsz,
        )
        return {
            "status": "ok",
            "mxid": oak_settings["mxid"],
            "rgb_shape": list(bundle.color_bgr.shape),
            "depth_shape": list(bundle.depth_m.shape),
            "sync_delta_ms": round(bundle.sync_delta_s * 1000.0, 3),
            "camera_matrix": bundle.color_intrinsics.matrix.tolist(),
            "detections": [
                {
                    "name": item["name"],
                    "confidence": item["conf"],
                    "bbox": list(item["xyxy"]),
                }
                for item in objects
            ],
        }
    finally:
        source.stop()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_VISUAL_CONFIG))
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument(
        "--legacy-arm-service",
        default="",
        help="optional legacy simulated arm_node ZMQ endpoint, e.g. tcp://127.0.0.1:5556",
    )
    parser.add_argument(
        "--camera-check",
        action="store_true",
        help="capture one OAK frame, run YOLO, print JSON and exit without a UI",
    )
    parser.add_argument(
        "--tcp-xyz-mm", nargs=3, type=float, default=None,
        help="手动指定 T_user1_tcp XYZ(mm), 用户坐标系; 与 --tcp-rpy-deg 同时给出",
    )
    parser.add_argument(
        "--tcp-rpy-deg", nargs=3, type=float, default=None,
        help="手动指定 A/B/C(deg), 与 --tcp-xyz-mm 同时给出",
    )
    parser.add_argument(
        "--controller-host", default="",
        help="控制器 IP 覆盖(默认用 competition.yaml 的 controller.nexbot_tcp.host)",
    )
    parser.add_argument(
        "--no-tcp-read", action="store_true",
        help="不连接控制器读取机械臂位姿(仅相机系输出, 用户1映射标记为不可用)",
    )
    parser.add_argument(
        "--place-x-mm", type=float, default=UCS_PLACE_X_MM,
        help="放置点 X(mm, 用户坐标系1, 默认 -100)",
    )
    parser.add_argument(
        "--place-y-mm", type=float, default=UCS_PLACE_Y_MM,
        help="放置点 Y(mm, 用户坐标系1, 默认 100)",
    )
    parser.add_argument(
        "--enable-robot-motion", action="store_true",
        help="允许真实机械臂运动(默认仅 Dry-run; 需在 UI 二次确认)",
    )
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    if (arguments.tcp_xyz_mm is None) != (arguments.tcp_rpy_deg is None):
        raise SystemExit(
            "--tcp-xyz-mm 与 --tcp-rpy-deg 必须同时提供"
        )
    visual_config = VisualGraspConfig.from_yaml(arguments.config)
    oak_settings, maximum_sync = load_oak_settings(arguments.camera_config)
    tag_provider = build_tag_provider(arguments.camera_config)
    if arguments.camera_check:
        print(json.dumps(
            camera_check(visual_config, oak_settings, maximum_sync),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    root = tk.Tk()
    node = OakVisionNode(
        root,
        visual_config,
        oak_settings,
        maximum_sync,
        tag_provider,
        arm_service=arguments.legacy_arm_service,
        tcp_xyz_mm=arguments.tcp_xyz_mm,
        tcp_rpy_deg=arguments.tcp_rpy_deg,
        controller_host=arguments.controller_host,
        no_tcp_read=arguments.no_tcp_read,
        place_x_mm=arguments.place_x_mm,
        place_y_mm=arguments.place_y_mm,
        enable_robot_motion=arguments.enable_robot_motion,
    )
    signal.signal(signal.SIGINT, lambda *_args: root.after(0, node.close))
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
