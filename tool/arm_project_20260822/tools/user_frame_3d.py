#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户坐标系(用户1)3D 可视化: 静态图 -> 物体位姿 -> 手眼/基座链换算 -> 3D 窗口。

坐标链(全米制 4x4):
  T_user1_object = inv(T_base_user1) @ T_base_tcp @ T_tcp_color_camera @ T_camera_from_object

数据来源:
  - 手眼矩阵 tcp_from_color_camera: competition_pipeline/config/competition.yaml (现场标定 15/16 内点)
  - T_base_user1: 现场实测(工具1停在用户1原点: PCS 578.3,-79.3,302.3mm / ABC 174.64,-4.47,-174.43deg)
  - T_base_tcp: 默认取 T_base_user1(工具停在用户1原点,即标定姿态); 可 --tcp-* 覆盖
  - 物体位姿: 静态图(rgb+depth+K)走 YOLO + FoundationPose(如旧版 vision_node)

用法(图形窗口):
  conda activate foundationpose
  python ~/tools/user_frame_3d.py
  python ~/tools/user_frame_3d.py --label can --tcp-xyz-mm 500 0 300 --tcp-rpy-deg 0 0 0
无窗口保存 PNG:
  python ~/tools/user_frame_3d.py --save /tmp/user_frame_3d.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = "/home/huyk/arm"
sys.path.insert(0, REPO)

from competition_pipeline.geometry import (  # noqa: E402
    transform_from_inexbot_abc,
    transform_from_xyz_rpy,
    xyz_rpy_from_transform,
)
from tool.visual_grasp_pipeline.offline import load_static_frame  # noqa: E402
from tool.visual_grasp_pipeline.config import VisualGraspConfig  # noqa: E402
from tool.visual_grasp_pipeline.detection import (  # noqa: E402
    detect_all_track,
)
from tool.visual_grasp_pipeline.foundationpose import (  # noqa: E402
    FoundationPosePoseEstimator,
)
from tool.visual_grasp_pipeline.tracking import StableTracker  # noqa: E402
from tool.visual_grasp_pipeline.geometry import compute_grasp  # noqa: E402

# 现场实测: 用户1零点在基座系(PCS)的位姿(工具1, MOKA MR07S-930 / Inexbot C1102)
USER1_ORIGIN_BASE_XYZ_MM = (578.3, -79.3, 302.3)
USER1_ORIGIN_BASE_ABC_DEG = (174.64, -4.47, -174.43)


def base_from_user1() -> np.ndarray:
    return transform_from_inexbot_abc(
        np.asarray(USER1_ORIGIN_BASE_XYZ_MM) / 1000.0,
        np.radians(USER1_ORIGIN_BASE_ABC_DEG),
    )


def load_hand_eye(competition_yaml: Path) -> np.ndarray:
    import yaml
    data = yaml.safe_load(competition_yaml.read_text())
    m = data["hand_eye"]["tcp_from_color_camera"]["matrix"]
    return np.asarray(m, dtype=np.float64).reshape(4, 4)


def draw_frames(ax):
    for i, (c, name) in enumerate(zip(["r", "g", "b"], ["X", "Y", "Z"])):
        v = np.zeros(3)
        v[i] = 0.15
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=c, lw=2.5)
        # 轴端点小箭头示意
        ax.text(v[0] * 1.1, v[1] * 1.1, v[2] * 1.1, name, color=c, fontsize=11)
    ax.text(0, 0, 0, " 用户1原点", color="k", fontsize=10)


def draw_pose_box(ax, T, bounds_m, color="y", alpha=0.35):
    """把物体包围盒(米制 bounds)按位姿画到用户1坐标系。"""
    mn, mx = bounds_m[0], bounds_m[1]
    corners = np.array([[x, y, z]
                        for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1])
                        for z in (mn[2], mx[2])])
    pts = corners @ T[:3, :3].T + T[:3, 3]
    edges = [(0, 1), (0, 2), (0, 4), (7, 6), (7, 5), (7, 3),
             (1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]
    for a, b in edges:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                [pts[a, 2], pts[b, 2]], color=color, lw=2)
    # 物体自身坐标轴(短三叉戟)
    origin = T[:3, 3]
    L = 0.06
    for i, c in enumerate(["r", "g", "b"]):
        d = T[:3, i] * L
        ax.plot([origin[0], origin[0] + d[0]],
                [origin[1], origin[1] + d[1]],
                [origin[2], origin[2] + d[2]], color=c, lw=1.5)
    return pts


def draw_grasp(ax, T_grasp):
    origin = T_grasp[:3, 3]
    L = 0.08
    for i, c in enumerate(["m", "c", "k"]):
        d = T_grasp[:3, i] * L
        ax.plot([origin[0], origin[0] + d[0]],
                [origin[1], origin[1] + d[1]],
                [origin[2], origin[2] + d[2]], color=c, lw=2,
                linestyle="--")
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color="magenta", s=40)
    return origin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="can", help="目标类别(需在 yaml 的 yolo_to_object 中)")
    ap.add_argument("--static-dir", default="", help="静态帧目录(rgb.png/depth.png/cam_K.txt); 默认取流水线配置")
    ap.add_argument("--tcp-xyz-mm", nargs=3, type=float, default=None,
                    help="工具TCP在基座系 XYZ(mm); 默认=用户1原点位姿")
    ap.add_argument("--tcp-rpy-deg", nargs=3, type=float, default=None,
                    help="工具TCP在基座系 RPY(deg)")
    ap.add_argument("--save", default="", help="保存 PNG 而不是开窗口 (Agg)")
    args = ap.parse_args()

    import yaml
    from pathlib import Path as P
    from ultralytics import YOLO

    vcfg = VisualGraspConfig.from_yaml(P(f"{REPO}/tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml"))
    static_dir = args.static_dir or vcfg.static_frame_dir
    rgb, depth_m, K = load_static_frame(static_dir)
    print(f"[1] 静态帧: {static_dir}  rgb={rgb.shape}  K[0,0]={K[0,0]:.1f}")

    obj_key = vcfg.yolo_to_object.get(args.label, args.label)
    mesh_path = vcfg.object_models.get(obj_key, "")
    assert mesh_path and P(mesh_path).exists(), f"模型未配置: {obj_key} -> {mesh_path}"
    print(f"[2] 目标: {args.label} -> {obj_key} -> {mesh_path}")

    print("[3] YOLO + FoundationPose 位姿估计(约 1-2 分钟)...")
    yolo = YOLO(vcfg.yolo_weights)
    objs = detect_all_track(rgb, yolo, StableTracker(max_miss=0),
                            conf=vcfg.yolo_conf, imgsz=vcfg.yolo_imgsz)
    target = next((o for o in objs if o["name"] == args.label), objs[0] if objs else None)
    assert target is not None, f"未检测到 {args.label} (检测到: {[o['name'] for o in objs]})"
    mask = target["mask"] if target["mask"] is not None else np.zeros(rgb.shape[:2], np.uint8)
    if mask.sum() == 0:
        x1, y1, x2, y2 = map(int, target["xyxy"])
        mask[y1:y2, x1:x2] = 255
    est = FoundationPosePoseEstimator(
        foundationpose_root=vcfg.foundationpose_root,
        mesh_path=mesh_path,
        mesh_scale_to_meters=vcfg.object_model_scales.get(obj_key, 1.0),
        device=vcfg.device,
    )
    T_cam_obj = est.register(rgb, depth_m, mask, K)
    est.close()
    t_mm = T_cam_obj[:3, 3] * 1000.0
    print(f"    camera_from_object t(mm) = ({t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f})")

    # 坐标链
    T_tcp_cam = load_hand_eye(P(f"{REPO}/competition_pipeline/config/competition.yaml"))
    T_base_user1 = base_from_user1()
    if args.tcp_xyz_mm is not None or args.tcp_rpy_deg is not None:
        assert args.tcp_xyz_mm is not None and args.tcp_rpy_deg is not None
        T_base_tcp = transform_from_xyz_rpy(
            np.asarray(args.tcp_xyz_mm) / 1000.0, args.tcp_rpy_deg)
    else:
        T_base_tcp = T_base_user1  # 工具停在用户1原点(标定姿态)
    T_user1_obj = np.linalg.inv(T_base_user1) @ T_base_tcp @ T_tcp_cam @ T_cam_obj
    T_user1_grasp = compute_grasp(T_user1_obj)

    def show(name, T, unit=1000.0):
        xyz, rpy = xyz_rpy_from_transform(T)
        print(f"    {name}: XYZ(mm)=({xyz[0]*unit:.1f}, {xyz[1]*unit:.1f}, {xyz[2]*unit:.1f}) "
              f"RPY(deg)=({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f})")

    print("[4] 坐标换算:")
    show("tcp_from_camera(手眼)", T_tcp_cam)
    show("base_from_user1", T_base_user1)
    show("base_tcp(工具)", T_base_tcp)
    show("user1_from_object(物体)", T_user1_obj)
    show("user1_grasp(抓取)", T_user1_grasp)

    # 3D 显示(用户1坐标系)
    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    else:
        import matplotlib
        try:
            # 先 import cv2 会加载 opencv 自带 Qt 插件, 与 PyQt5 冲突;
            # 用 TkAgg(Tk)打开窗口, 与旧版 vision_node 一致且稳定
            matplotlib.use("TkAgg")
        except Exception:
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
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_frames(ax)
    bounds_m = est.mesh_bounds if est.mesh_bounds is not None else np.array([[-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]])
    draw_pose_box(ax, T_user1_obj, bounds_m, color="y")
    g = draw_grasp(ax, T_user1_grasp)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"用户1坐标系  | 目标 {args.label} | grasp t=({g[0]*1000:.0f}, {g[1]*1000:.0f}, {g[2]*1000:.0f}) mm")
    ax.set_box_aspect((1, 1, 1))
    try:
        ax.set_xlim(-0.1, 0.7); ax.set_ylim(-0.4, 0.4); ax.set_zlim(-0.1, 0.5)
    except Exception:
        pass
    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"[5] 3D 视图已保存: {args.save}")
    else:
        print("[5] 打开 3D 窗口(关闭窗口退出)...")
        try:
            plt.show()
        except Exception as exc:  # 无可用显示时降级为 PNG
            import matplotlib
            matplotlib.use("Agg")
            fig.savefig("/tmp/user_frame_3d.png", dpi=110)
            print(f"    窗口打开失败({exc}), 已保存: /tmp/user_frame_3d.png")


if __name__ == "__main__":
    main()
