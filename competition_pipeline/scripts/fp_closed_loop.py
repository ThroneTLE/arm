#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FoundationPose 定位闭环验证（用户坐标系1）

链路（全闭环）:
    相机RGB-D(眼在手) --YOLO分割--> 掩膜 --FoundationPose(can2 CAD)-->
    T_cam_obj --手眼标定 T_tcp_cam + 控制器回读 T_user1_tcp(UCS,用户1)-->
    T_user1_obj   （直接可用于用户1系控制面板/抓取）

闭环验证手段:
  1) 单视角内置自检: FP网格投影与YOLO掩膜 IoU; FP平移深度 vs 掩膜深度中值;
     （可选）画面里有AprilTag时, 用tag地图独立解算 T_user1_cam, 与
     "TCP回读+手眼" 链路的相机位姿互检;
  2) 多视角一致性: 机械臂换2个以上不同姿态各拍一次(必须停稳!),
     compare 命令比较不同视角下 T_user1_obj 的位置/罐轴方向——若一致
     说明 YOLO+FP+手眼+用户1回读整条链路是通的(闭环成立)。

用法:
  # 现场(相机已连OAK, 机械臂停稳, 罐子在画面里):
  python competition_pipeline/scripts/fp_closed_loop.py run --view v1 \
      --out competition_pipeline/output/fp_closed_loop

  # 换个机械臂姿态再拍一次(看另一个角度):
  python competition_pipeline/scripts/fp_closed_loop.py run --view v2 ...

  # 多视角闭环判定:
  python competition_pipeline/scripts/fp_closed_loop.py compare \
      --out competition_pipeline/output/fp_closed_loop

  # 无相机冒烟测试(用保存的照片, 只验证视觉链, 不读机械臂):
  python competition_pipeline/scripts/fp_closed_loop.py run --view smoke \
      --offline static_frame/rgb.png static_frame/depth.png static_frame/cam_K.txt \
      --no-robot --out /tmp/fp_smoke

说明:
  - 相机: 当前 competition.yaml 的 active profile(oak_competition,
    OAK-D-PRO-FF 1920x1080, 深度已对齐RGB);
  - 网格: fp_release_20260821_155930/models/can2/mesh/can.obj(罐轴=局部Z),
    --mesh fixed 切换 fixed/can.obj(罐轴=局部Y);
  - 本脚本只读, 不会发送任何机械臂运动指令。
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from competition_pipeline.configuration import (  # noqa: E402
    CompetitionConfig,
    load_camera_intrinsics,
)
from competition_pipeline.geometry import (  # noqa: E402
    as_transform,
    inexbot_abc_from_transform,
    rotation_angle_deg,
    transform_from_inexbot_abc,
)
from competition_pipeline.tcp_pose import (  # noqa: E402
    NexBotTcpPoseSource,
    pose_endpoint_from_config,
)

CONFIG_PATH = REPO / "competition_pipeline" / "config" / "competition.yaml"
YOLO_PATH = REPO / "model" / "yolo_model.pt"
FP_DIR = Path(os.path.expanduser("~/FoundationPose"))
RELEASE_DIR = Path("/home/throne/workspaces/fp_release_20260821_155930")

MESH_CHOICES = {
    "mesh": RELEASE_DIR / "models" / "can2" / "mesh" / "can.obj",
    "fixed": RELEASE_DIR / "models" / "can2" / "fixed" / "can.obj",
}

LABEL = "can"             # yolo_model.pt 类别 {0:orange..5:can}
YOLO_CONF = 0.85
YOLO_IMGSZ = 640
FP_WIDTH_DEFAULT = 960    # FP 运行分辨率(宽), K 同步缩放; 1080p 下省显存
FP_ITER = 5
FP_MAX_HYP = 64

# 闭环判定阈值
POS_TOL_MM_DEFAULT = 25.0
AXIS_TOL_DEG_DEFAULT = 8.0


# --------------------------------------------------------------------------
def err(message):
    print("[FAIL] " + message)
    return 1


def warn(message):
    print("[WARN] " + message)


def log(message):
    print("[FP-verify] " + message)


# --------------------------------------------------------------------------
def load_frame_live(config):
    """OAK 相机取一帧 (color_bgr, depth_m, live_K)。"""
    from tool.object_model_builder.camera_source import OakDProSource

    profile = config.camera
    source = OakDProSource(
        color_width=int(profile["color_width"]),
        color_height=int(profile["color_height"]),
        fps=int(profile["color_fps"]),
        mxid=str(profile.get("mxid", "")),
        dot_projector_mA=int(profile.get("dot_projector_mA", 800)),
        floodlight_mA=int(profile.get("floodlight_mA", 0)),
        mono_resolution=str(profile.get("mono_resolution", "800p")),
        extended_disparity=bool(profile.get("extended_disparity", True)),
        subpixel=bool(profile.get("subpixel", False)),
        left_right_check=bool(profile.get("left_right_check", True)),
        focus_mode=str(profile.get("focus_mode", "device_default")),
        manual_focus=profile.get("manual_focus"),
    )
    source.start()
    try:
        deadline = time.monotonic() + 30.0
        bundle = None
        while bundle is None and time.monotonic() < deadline:
            bundle = source.latest(anchor="depth", copy_frames=True)
            if bundle is None:
                time.sleep(0.2)
        if bundle is None:
            raise RuntimeError("OAK 相机 30s 内没有出帧（相机已连接？被别的程序占用？）")
        live_K = None
        if bundle.color_intrinsics is not None:
            live_K = np.asarray(bundle.color_intrinsics.matrix, dtype=np.float64)
        return bundle.color_bgr, bundle.depth_m, live_K
    finally:
        source.stop()


def load_frame_offline(rgb_path, depth_path, k_path):
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise RuntimeError("无法读取 RGB: {}".format(rgb_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("无法读取深度(16bit mm): {}".format(depth_path))
    H, W = rgb.shape[:2]
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.shape[:2] != (H, W):
        warn("深度尺寸 {} 与 RGB {} 不一致, 顶部对齐填充".format(depth.shape, rgb.shape))
        aligned = np.zeros((H, W), np.uint16)
        aligned[: min(depth.shape[0], H), : min(depth.shape[1], W)] = \
            depth[: min(depth.shape[0], H), : min(depth.shape[1], W)]
        depth = aligned
    K = np.eye(3)
    if k_path and Path(k_path).exists():
        K = np.loadtxt(k_path).reshape(3, 3).astype(np.float64)
    else:
        warn("无 K 文件, 使用默认 640x480 K")
    depth_m = depth.astype(np.float32) / 1000.0
    return rgb, depth_m, K


# --------------------------------------------------------------------------
def detect_can_mask(yolo, rgb, label=LABEL, conf=YOLO_CONF):
    t0 = time.time()
    res = yolo.predict(rgb, conf=conf, imgsz=YOLO_IMGSZ, verbose=False)[0]
    ms = time.time() - t0
    H, W = rgb.shape[:2]
    if res.boxes is None or len(res.boxes) == 0:
        return None, None, ms
    masks = res.masks.data.cpu().numpy() if res.masks is not None else None
    best = None
    for i in range(len(res.boxes)):
        name = str(yolo.names[int(res.boxes.cls[i])])
        if name != label:
            continue
        if masks is None or masks.ndim != 3 or i >= len(masks):
            continue
        m = (masks[i] > 0.5).astype(np.uint8) * 255
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        area = int((m > 0).sum())
        if best is None or area > best[0]:
            best = (area, m, float(res.boxes.conf[i]),
                    tuple(int(v) for v in res.boxes.xyxy[i].cpu().numpy()),
                    name, int(res.boxes.cls[i]))
    if best is None:
        return None, None, ms
    return best, ms


# --------------------------------------------------------------------------
def fill_depth_roi(depth_m, mask):
    """掩膜内深度空洞(反光0值)用区域内有效中位数填充。"""
    roi = depth_m.copy()
    vals = roi[mask > 0]
    valid = vals[vals > 0.05]
    if len(valid) == 0:
        return roi
    med = float(np.median(valid))
    hole = (mask > 0) & (roi <= 0.05)
    roi[hole] = med
    return roi


# --------------------------------------------------------------------------
def make_fp_estimator(mesh_path):
    """加载 CAD(OBJ按mm→m), 构造 FoundationPose(注册候选限64省显存)。"""
    import torch
    import trimesh

    if not mesh_path.exists():
        raise RuntimeError("CAD 模型不存在: {}".format(mesh_path))
    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.apply_scale(0.001)  # OBJ 按 mm 建模 -> 米
    _, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    sys.path.insert(0, str(FP_DIR))
    os.chdir(FP_DIR)
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # noqa: E402
    from Utils import set_logging_format, set_seed  # noqa: E402

    set_logging_format()
    set_seed(0)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = _make_glctx()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir="/tmp/fp_closed_loop_debug",
        debug=0,
        glctx=glctx,
    )
    if FP_MAX_HYP > 0 and len(est.rot_grid) > FP_MAX_HYP:
        ids = torch.linspace(0, len(est.rot_grid) - 1, steps=FP_MAX_HYP,
                             device=est.rot_grid.device).round().long()
        est.rot_grid = est.rot_grid.index_select(0, ids)
    return est, mesh, bbox


def _make_glctx():
    import nvdiffrast.torch as dr
    return dr.RasterizeCudaContext()


def fp_register(est, K, rgb, depth_m, mask):
    t0 = time.time()
    M = est.register(K, rgb, depth_m, mask, iteration=FP_ITER)
    return M, (time.time() - t0) * 1000.0


# --------------------------------------------------------------------------
def silhouette_mask(mesh, K, M_cam_obj, height, width):
    mask = np.zeros((height, width), np.uint8)
    V = mesh.vertices
    P = (M_cam_obj[:3, :3] @ V.T).T + M_cam_obj[:3, 3]
    z = P[:, 2]
    if np.any(z <= 0.01):
        return None
    u = P[:, 0] * K[0, 0] / z + K[0, 2]
    v = P[:, 1] * K[1, 1] / z + K[1, 2]
    pts = np.column_stack([u, v])
    for face in mesh.faces:
        f = face[(face < len(pts))]
        if len(f) < 3:
            continue
        cv2.fillConvexPoly(mask, pts[f].astype(np.int32), 255)
    return mask


# --------------------------------------------------------------------------
def read_robot_tcp(config):
    ep = pose_endpoint_from_config(config.data["controller"])
    src = NexBotTcpPoseSource(ep)
    try:
        xyz_mm, abc_deg = src.read()
        T = transform_from_inexbot_abc(
            np.asarray(xyz_mm, dtype=np.float64) / 1000.0,
            np.radians(np.asarray(abc_deg, dtype=np.float64)),
        )
        return T, tuple(float(v) for v in xyz_mm), tuple(float(v) for v in abc_deg)
    finally:
        src.close()


# --------------------------------------------------------------------------
def tag_cross_check(config, rgb, K, dist, T_user1_cam_he):
    """如果画面里看得见 tag 地图(用户1系), 用 PnP 独立求相机位姿并互检。"""
    import copy

    import yaml

    from competition_pipeline.localization import AprilTagLocalizer

    tag_map_path = REPO / "competition_pipeline" / "config" / "tag_map_user1.yaml"
    if not tag_map_path.exists():
        return None
    override = copy.deepcopy(config.data["tag_map"])
    try:
        user1_map = yaml.safe_load(tag_map_path.read_text(encoding="utf-8"))
        for tag_id, entry in user1_map.get("tags", {}).items():
            override.setdefault("tags", {})[str(tag_id)] = {
                "bottom_right_xyz_mm": list(entry["bottom_right_xyz_mm"]),
                "base_from_tag_rpy_deg": list(entry["base_from_tag_rpy_deg"]),
            }
    except Exception as e:
        warn("tag_map_user1.yaml 解析失败: {}".format(e))
        return None

    class _Cfg:
        pass

    shell = _Cfg()
    shell.tag_map = override
    shell.data = {"localization": {"minimum_visible_tags": 1}}
    localizer = AprilTagLocalizer(shell)
    detections = localizer.detect(rgb)
    if not detections:
        return {"visible_tag_ids": [], "used_tag_ids": [], "note": "画面无tag"}
    result = localizer.estimate_detections(detections, K, dist)
    if not result.valid:
        return {"visible_tag_ids": list(result.visible_tag_ids),
                "used_tag_ids": list(result.used_tag_ids),
                "note": result.reason}
    # result.base_from_camera 即 T_user1_cam(tag地图已迁到用户1系)
    with_tcp = T_user1_cam_he
    with_tag = np.asarray(result.base_from_camera, dtype=np.float64)
    dpos = np.linalg.norm(with_tag[:3, 3] - with_tcp[:3, 3]) * 1000.0
    drot = rotation_angle_deg(with_tag, with_tcp)
    return {
        "visible_tag_ids": list(result.visible_tag_ids),
        "used_tag_ids": list(result.used_tag_ids),
        "rms_reprojection_error_px": result.rms_reprojection_error_px,
        "max_reprojection_error_px": result.max_reprojection_error_px,
        "camera_position_delta_mm": round(float(dpos), 2),
        "camera_rotation_delta_deg": round(float(drot), 2),
    }


# --------------------------------------------------------------------------
def run_command(args):
    config = CompetitionConfig(str(CONFIG_PATH))
    if not config.hand_eye_valid:
        return err("competition.yaml hand_eye.tcp_from_color_camera.valid=false")

    # ---- 相机帧 ----
    if args.offline:
        rgb, depth_m, offline_K = load_frame_offline(*args.offline)
        live_K = None
        mode = "offline"
    else:
        try:
            rgb, depth_m, live_K = load_frame_live(config)
        except Exception as e:
            return err("取相机帧失败: {}".format(e))
        offline_K = None
        mode = "live"
    H, W = rgb.shape[:2]
    log("帧: {} {}x{} mode={}".format("live" if mode == "live" else "offline", W, H, mode))

    # ---- 内参选择: 现场用手眼标定同款 yaml; 离线用照片自带 K ----
    yaml_matrix, yaml_dist, (yaml_w, yaml_h) = load_camera_intrinsics(
        str(config.resolve_path(config.camera["color_intrinsics_file"])), "color")
    if mode == "offline":
        if offline_K is None:
            return err("离线模式缺少相机内参文件")
        use_K = offline_K
        k_source = "offline_file"
        yaml_dist_use = np.zeros((4, 1))
    else:
        if args.k not in ("yaml", "live"):
            return err("--k 只能是 yaml 或 live")
        use_K = yaml_matrix if args.k == "yaml" else live_K
        if use_K is None:
            return err("没有 live 内参（设备返回为空）")
        k_source = args.k
        log("K 对比: yaml fx={:.2f} cy={:.2f} | live fx={:.2f} cy={:.2f} | 使用 {}".
            format(yaml_matrix[0, 0], yaml_matrix[1, 2],
                   live_K[0, 0], live_K[1, 2], args.k))
        yaml_dist_use = yaml_dist

    # ---- YOLO 分割 ----
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_PATH))
    best, ms = detect_can_mask(yolo, rgb)
    if best is None:
        return err("YOLO 没有检测到类别 '{}' (conf>={})".format(LABEL, YOLO_CONF))
    area, mask, conf, bbox, name, cls = best
    log("YOLO: {} conf={:.2f} bbox={} 掩膜像素={} 用时={:.0f}ms".
        format(name, conf, bbox, area, ms * 1000.0))

    # ---- FoundationPose (降采样运行, K 同步缩放) ----
    mesh_path = MESH_CHOICES[args.mesh]
    log("FP CAD: {}".format(mesh_path))
    est, fp_mesh, fp_bbox = make_fp_estimator(mesh_path)
    # 罐轴在网格局部坐标系的方向: mesh=局部Z, fixed=局部Y
    axis_local = np.array([0.0, 0.0, 1.0] if args.mesh == "mesh" else [0.0, 1.0, 0.0])

    fp_width = min(int(args.fp_width), W)
    scale = float(fp_width) / W
    fp_w, fp_h = int(round(W * scale)), int(round(H * scale))
    rgb_fp = cv2.resize(rgb, (fp_w, fp_h), interpolation=cv2.INTER_AREA)
    depth_fp = cv2.resize(depth_m, (fp_w, fp_h), interpolation=cv2.INTER_NEAREST)
    mask_fp = cv2.resize(mask, (fp_w, fp_h), interpolation=cv2.INTER_NEAREST)
    K_fp = use_K.copy()
    K_fp[:2, :3] *= scale
    depth_fp = fill_depth_roi(depth_fp, mask_fp)

    M_cam_obj, reg_ms = fp_register(est, K_fp, rgb_fp, depth_fp, mask_fp)
    t_cam = M_cam_obj[:3, 3] * 1000.0
    z = M_cam_obj[:3, 3][2]
    log("FP 注册: 用时={:.0f}ms 相机系位置(mm)=({:.1f},{:.1f},{:.1f}) 距离={:.0f}mm".
        format(reg_ms, t_cam[0], t_cam[1], t_cam[2], z * 1000.0))

    # ---- 单视角自检: 投影 IoU + 深度一致性 ----
    sil = silhouette_mask(fp_mesh, K_fp, M_cam_obj, fp_h, fp_w)
    iou = None
    if sil is not None:
        a = sil > 0
        b = mask_fp > 0
        both = a & b
        union = a | b
        iou = float(both.sum()) / float(union.sum()) if union.sum() > 0 else None
        log("FP投影与YOLO掩膜 IoU = {:.3f}".format(iou) if iou is not None else
            "FP投影与YOLO掩膜 IoU = 无法计算")
    valid = (mask_fp > 0) & (depth_fp >= 0.05)
    med_depth = float(np.median(depth_fp[valid])) if valid.sum() > 0 else None
    depth_delta_mm = None
    if med_depth is not None:
        depth_delta_mm = abs(t_cam[2] - med_depth * 1000.0)
        log("掩膜深度中值={:.0f}mm vs FP z={:.0f}mm (差 {:.0f}mm)".
            format(med_depth * 1000.0, t_cam[2], depth_delta_mm))

    # ---- 机械臂回读(用户1) + 手眼 ----
    robot = None
    if args.no_robot:
        warn("--no-robot: 跳过 TCP 回读, 只输出相机系结果")
    else:
        try:
            T_user1_tcp, xyz_mm, abc_deg = read_robot_tcp(config)
            robot = {"xyz_mm": [round(float(v), 2) for v in xyz_mm],
                     "abc_deg": [round(float(v), 3) for v in abc_deg]}
            log("TCP 回读(用户1 UCS): XYZ={} ABC={}".format(xyz_mm, abc_deg))
        except Exception as e:
            return err("控制器回读失败: {}".format(e))

    report = {
        "schema": 1,
        "view": args.view,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "camera": {
            "profile": config.active_camera_profile,
            "size": [W, H],
            "fp_size": [fp_w, fp_h],
            "k_source": k_source,
            "k_yaml": yaml_matrix.tolist(),
            "k_live": live_K.tolist() if live_K is not None else None,
        },
        "yolo": {"label": name, "class_id": cls, "conf": round(float(conf), 4),
                 "bbox": [int(v) for v in bbox], "mask_pixels": area,
                 "inference_ms": round(ms * 1000.0, 1)},
        "fp": {
            "mesh": args.mesh,
            "mesh_file": str(mesh_path),
            "register_ms": round(reg_ms, 1),
            "position_cam_mm": [round(float(v), 2) for v in t_cam],
            "distance_mm": round(float(z * 1000.0), 1),
            "iou_vs_mask": None if iou is None else round(iou, 4),
            "mask_depth_median_mm": None if med_depth is None else round(med_depth * 1000.0, 1),
            "depth_delta_mm": None if depth_delta_mm is None else round(depth_delta_mm, 1),
        },
        "hand_eye": {
            "valid": config.hand_eye_valid,
            "description": config.data["hand_eye"]["tcp_from_color_camera"].get("description", ""),
            "matrix": np.asarray(config.tcp_from_color_camera).tolist(),
        },
    }

    if robot is not None:
        report["robot"] = robot
        heat_T = as_transform(config.tcp_from_color_camera)
        T_user1_cam = as_transform(T_user1_tcp) @ heat_T
        T_user1_obj = T_user1_cam @ M_cam_obj

        def xyz_abc(T):
            xyz, abc = inexbot_abc_from_transform(T)
            return ([round(float(v) * 1000.0, 2) for v in xyz],
                    [round(float(np.degrees(v)), 3) for v in abc])

        obj_xyz, obj_abc = xyz_abc(T_user1_obj)
        cam_xyz, cam_abc = xyz_abc(T_user1_cam)
        axis_user1 = T_user1_obj[:3, :3] @ axis_local
        report["user1"] = {
            "object_xyz_mm": obj_xyz,
            "object_abc_deg": obj_abc,
            "object_axis_xyz": [round(float(v), 4) for v in axis_user1],
            "camera_xyz_mm": cam_xyz,
            "camera_abc_deg": cam_abc,
        }
        report["object_matrix_user1"] = np.asarray(T_user1_obj).tolist()

        # tag 互检(有 tag 才算); 在原始分辨率上用 yaml 内参(与手眼标定一致)
        try:
            tc = tag_cross_check(config, rgb, use_K, yaml_dist_use, T_user1_cam)
        except Exception as e:
            warn("tag 互检跳过: {}".format(e))
            tc = None
        if tc:
            report["tag_crosscheck"] = tc
            if tc.get("used_tag_ids"):
                log("tag互检: 用 tag {} 解出的相机位姿与 TCP+手眼 相差 " 
                    "{:.1f}mm / {:.2f}deg (rms {:.2f}px)".
                    format(tc["used_tag_ids"], tc["camera_position_delta_mm"],
                           tc["camera_rotation_delta_deg"],
                           tc.get("rms_reprojection_error_px", -1)))

        log("T_user1_obj: 位置(mm)={} / A-B-C(deg)={} / 罐轴(user1)={}".
            format(obj_xyz, obj_abc, np.round(axis_user1, 3)))
        log("判定参考: 罐轴应基本竖直(±z), 位置应与罐子在用户1系的实际位置一致")

    # ---- 保存 ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    view_dir = out / args.view
    view_dir.mkdir(parents=True, exist_ok=True)

    # 可视化: 原图 + 掩膜轮廓 + FP投影轮廓 + 轴
    viz = rgb.copy()
    cv2.rectangle(viz, bbox[:2], bbox[2:], (255, 0, 0), 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(viz, contours, -1, (0, 255, 0), 2)
    sil_full = None
    if iou is not None:
        sil_full = silhouette_mask(fp_mesh, K_fp, M_cam_obj, fp_h, fp_w)
        if sil_full is not None:
            sil_full = cv2.resize(sil_full, (W, H), interpolation=cv2.INTER_NEAREST)
            sc, _ = cv2.findContours(sil_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz, sc, -1, (0, 255, 255), 2)
    # 罐轴投影(相机系): 从物体中心沿局部轴画线
    center = M_cam_obj[:3, 3]
    axis = M_cam_obj[:3, :3] @ axis_local
    half = 0.06
    p1 = center - axis * half
    p2 = center + axis * half
    for p, color in ((p1, (0, 0, 255)), (p2, (255, 0, 255))):
        if p[2] > 0.01:
            pt = (int(round(p[0] * K_fp[0, 0] / p[2] + K_fp[0, 2])),
                  int(round(p[1] * K_fp[1, 1] / p[2] + K_fp[1, 2])))
            cv2.circle(viz, pt, 6, color, -1)
    txt = "{} conf={:.2f} IoU={} | cam z={:.0f}mm".format(
        name, conf, "-" if iou is None else "{:.2f}".format(iou), t_cam[2])
    cv2.putText(viz, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if robot is not None:
        cv2.putText(viz, "user1 xyz={} abc={}".format(
            report["user1"]["object_xyz_mm"], report["user1"]["object_abc_deg"]),
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
    viz_path = view_dir / "overlay.png"
    cv2.imwrite(str(viz_path), viz)

    report_path = view_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("报告: {}".format(report_path))
    log("可视化: {}".format(viz_path))
    return 0


# --------------------------------------------------------------------------
def compare_command(args):
    out = Path(args.out)
    reports = []
    if out.exists():
        for p in sorted(out.glob("*/report.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if d.get("user1") is not None:
                    reports.append(d)
            except Exception as e:
                warn("跳过 {}: {}".format(p, e))
    if not reports:
        return err("没有找到可比较的视角报告(需要 run --view vN 至少2个): {}".format(out))

    def pos(r):
        return np.asarray(r["user1"]["object_xyz_mm"], dtype=np.float64)

    def axis(r):
        v = np.asarray(r["user1"]["object_axis_xyz"], dtype=np.float64)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def norm(v):
        return float(np.linalg.norm(v))

    print("=" * 78)
    print("多视角闭环一致性（全部以第一个视角 {} 为参考）".format(reports[0]["view"]))
    print("=" * 78)
    header = "{:<10} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "视角", "位置差mm", "罐轴差deg", "FP距离mm", "IoU", "结论")
    print(header)
    ok_all = True
    ref_pos, ref_axis = pos(reports[0]), axis(reports[0])
    for r in reports:
        dpos = norm(pos(r) - ref_pos)
        dax = math.degrees(math.acos(min(1.0, float(np.clip(np.dot(axis(r), ref_axis), -1, 1)))))
        dist = r["fp"].get("distance_mm")
        iou = r["fp"].get("iou_vs_mask")
        ok = dpos <= args.pos_tol_mm and dax <= args.axis_tol_deg
        ok_all = ok_all and ok
        print("{:<10} {:>10.2f} {:>10.2f} {:>10} {:>10} {:>12}".format(
            r["view"], dpos, dax,
            "-" if dist is None else "{:.0f}".format(dist),
            "-" if iou is None else "{:.2f}".format(iou),
            "PASS" if ok else "FAIL"))
    print("-" * 78)
    print("判别: 位置差 <= {:.0f}mm 且 罐轴方向差 <= {:.0f}deg（机械臂静止、"
          "同一罐子、来自不同视角）".format(args.pos_tol_mm, args.axis_tol_deg))
    print("结论: " + ("闭环一致 ✔ (YOLO+FP+手眼+用户1回读链路可用)"
                      if ok_all else "闭环不一致 ✘ (检查手眼/FP/网格/用户1)"))
    return 0 if ok_all else 1


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="拍一帧并计算用户1系物体位姿")
    p_run.add_argument("--view", default=None, help="视角名(默认时间戳)")
    p_run.add_argument("--out", default=str(REPO / "competition_pipeline" / "output" /
                                            "fp_closed_loop"))
    p_run.add_argument("--mesh", choices=sorted(MESH_CHOICES), default="mesh",
                       help="can2 网格: mesh=罐轴局部Z(默认), fixed=罐轴局部Y")
    p_run.add_argument("--fp-width", type=int, default=FP_WIDTH_DEFAULT,
                       help="FP 运行宽度(默认960) K同步缩放")
    p_run.add_argument("--k", choices=["yaml", "live"], default="yaml",
                       help="FP用哪套内参: yaml=手眼标定同款(默认), live=设备实时读取")
    p_run.add_argument("--offline", nargs=3, metavar=("RGB", "DEPTH16MM", "K"),
                       help="用保存的照片运行(不连相机)")
    p_run.add_argument("--no-robot", action="store_true", help="不读机械臂,只输出相机系")
    p_run.set_defaults(func=run_command)

    p_cmp = sub.add_parser("compare", help="多视角闭环一致性判定")
    p_cmp.add_argument("--out", default=str(REPO / "competition_pipeline" / "output" /
                                            "fp_closed_loop"))
    p_cmp.add_argument("--pos-tol-mm", type=float, default=POS_TOL_MM_DEFAULT)
    p_cmp.add_argument("--axis-tol-deg", type=float, default=AXIS_TOL_DEG_DEFAULT)
    p_cmp.set_defaults(func=compare_command)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    if args.command == "run" and args.view is None:
        args.view = datetime.now().strftime("v%Y%m%d_%H%M%S")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
