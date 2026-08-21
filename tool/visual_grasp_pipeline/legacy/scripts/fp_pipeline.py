#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整理版抓取流水线：YOLO 识别全部物体 -> 指定目标 -> FoundationPose 姿态
-> AprilTag(tag0) 工作台定位 -> 抓取位姿。

考核用法：
  1. 相机启动: bash ~/start_camera.sh
  2. 桥接:      /usr/bin/python3 ~/fp_bridge.py
  3. 本程序:    conda activate foundationpose && python ~/fp_pipeline.py

指定要抓的物体：改下方配置区 TARGET_LABEL（如 "can" 或类别 ID 0）；
后续做 UI 时只需在运行时修改这个变量。

依赖文件:
  ~/yolo_model.pt       你的 YOLO 模型（检测/分割）
  ~/FoundationPose/demo_data/can/mesh/can.obj   wznn CAD 模型
  ~/fp_debug/tag_layout.npy   可选：tag 布局（--calibrate 生成）
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import trimesh
import zmq
from scipy.spatial.transform import Rotation as R_util

# ==================== 配置区（考核时改这里） ====================
TARGET_LABEL = "red_apple"  # 要抓的物体标签：类别名或 ID；None=自动取最大目标
TAG_SIZE_MM = 80           # AprilTag 边长
MODEL_PATH = os.path.expanduser("~/yolo_model.pt")
YOLO_CONF = 0.85
YOLO_IMGSZ = 640

# 工作台系校准（之前实测的补偿）
OFFSET_X_MM = 15.0
OFFSET_Y_MM = 32.0
CENTER_OFFSET_MM = 30.0   # 原点从罐顶移到中心
FLIP_X = True             # X 轴极性
FLIP_Y = True             # Y 轴极性

# FoundationPose
REG_ITER = 5              # 注册迭代（第一帧/重新注册）
TRACK_ITER = 2            # 跟踪迭代
USE_TRACK = True          # True=跟踪(快)，False=每帧注册(稳但慢)

# 平滑
ALPHA = 0.15              # 输出位姿平滑（新值权重）
TAG_ALPHA = 0.3           # tag 工作台系平滑

# 抓取
GRASP_OFFSET_MM = 5.0     # 抓取点沿罐轴比中心低多少
GRASP_FORK_MM = 60.0      # 三叉戟叉长（夹爪手指长）
GRASP_SPACING_MM = 60.0   # 叉间距（已放大 3 倍，所有规则生效）
GRASP_HANDLE_MM = 80.0    # 柄长

# 文件/路径
FP_DIR = os.path.expanduser("~/FoundationPose")
MESH_FILE = os.path.join(FP_DIR, "demo_data/can2/mesh/can.obj")  # 默认雪碧罐

# ============ 物体种类 -> CAD 模型 映射表（按需添加/修改） ============
OBJECT_MODELS = {
    "wangzai": os.path.join(FP_DIR, "demo_data/can/mesh/can.obj"),
    "sprite": os.path.join(FP_DIR, "demo_data/can2/mesh/can.obj"),
    "orange": "",        # 待补充: 橙子模型
    "banana": os.path.join(FP_DIR, "demo_data/banana/mesh/banana.glb"),
    "red_apple": os.path.join(FP_DIR, "demo_data/apple/mesh/apple.glb"),
    "green_apple": os.path.join(FP_DIR, "demo_data/apple/mesh/apple.glb"),
    "bottle": "",        # 待补充: 瓶子模型
}

# YOLO 类别名 -> 物体种类（按实际 yolo_model.pt 标签）
YOLO_TO_OBJECT = {
    "orange": "orange",
    "banana": "banana",
    "red_apple": "red_apple",
    "green_apple": "green_apple",
    "bottle": "bottle",
    "can": "sprite",     # can 标签统一先用雪碧罐模型
}
DEFAULT_OBJECT = "sprite"   # 找不到映射时的兜底

# 强制指定物体种类：YOLO 区分不了旺仔/雪碧时，手动指定当前抓哪个
# 例如 FORCE_OBJECT = "wangzai" 或 "sprite"；None=按 YOLO 映射
FORCE_OBJECT = None

# ============ 物体种类 -> 抓取规则 映射表 ============
# type: cylinder=抓圆柱中部(垂直轴线) / sphere=对称物体(任意yaw) / elongated=长条形
# yaw_align: y+=三叉戟柄朝向与 Y+ 夹角最小（确定唯一姿态）
GRASP_RULES = {
    "wangzai": {"type": "cylinder", "offset_mm": 5.0, "yaw_align": "y+"},
    "sprite": {"type": "cylinder", "offset_mm": 5.0, "yaw_align": "y+"},
    "orange": {"type": "sphere", "offset_mm": 0.0, "yaw_align": "any"},
    "banana": {"type": "elongated", "offset_mm": 0.0, "yaw_align": "y+"},
    "red_apple": {"type": "sphere", "offset_mm": 0.0, "yaw_align": "any"},
    "green_apple": {"type": "sphere", "offset_mm": 0.0, "yaw_align": "any"},
    "bottle": {"type": "cylinder", "offset_mm": 5.0, "yaw_align": "y+"},
}
POSE_FILE = "/tmp/can_pose.npy"
GRASP_FILE = "/tmp/grasp_pose.npy"
VIZ2D_FILE = "/mnt/c/Users/Administrator/Desktop/live_pipeline.png"
VIZ3D_FILE = "/mnt/c/Users/Administrator/Desktop/live_pipeline_3d.png"
DEBUG_DIR = os.path.expanduser("~/fp_debug")
LAYOUT_FILE = os.path.join(DEBUG_DIR, "tag_layout.npy")
ZMQ_ADDR = "tcp://127.0.0.1:5555"

sys.path.insert(0, FP_DIR)
os.chdir(FP_DIR)

from estimater import *  # noqa: E402
from Utils import (  # noqa: E402
    draw_posed_3d_box,
    set_logging_format,
    set_seed,
)


# ==================== 1. YOLO 检测模块 ====================
def detect_all_objects(rgb, model):
    """识别桌面上所有物体，返回列表 [dict(xyxy, cls, conf, mask)]。"""
    H, W = rgb.shape[:2]
    res = model.predict(rgb, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)[0]
    objs = []
    masks = res.masks.data.cpu().numpy() if res.masks is not None else None
    if res.boxes is None or len(res.boxes) == 0:
        return objs
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    for i, (b, c, cl) in enumerate(zip(xyxy, confs, cls)):
        x1, y1, x2, y2 = b.astype(int)
        mask = None
        if masks is not None and i < len(masks):
            m = (masks[i] > 0.5).astype(np.uint8) * 255
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = m
        objs.append({
            "xyxy": (max(0, x1), max(0, y1), min(W, x2), min(H, y2)),
            "cls": cl,
            "name": model.names[cl],
            "conf": float(c),
            "mask": mask,
        })
    return objs


def select_target(objs, label):
    """指定要抓的目标：按标签名/ID 过滤；None 或未匹配则取面积最大。"""
    if not objs:
        return None
    if label is not None:
        for o in objs:
            if o["name"] == label or o["cls"] == label:
                return o
        print(f"[目标] 没找到标签 {label}，本帧跳过")
        return None
    return max(objs, key=lambda o: (o["xyxy"][2] - o["xyxy"][0])
               * (o["xyxy"][3] - o["xyxy"][1]))


def fill_depth_roi(depth_m, mask):
    """填充 mask 区域内的深度空洞（反光导致 0 值）：
    用区域内有效深度的中位数填充，避免 FoundationPose 腐蚀后 valid 为空。"""
    roi = depth_m.copy()
    vals = roi[mask > 0]
    valid = vals[vals > 0.05]
    if len(valid) == 0:
        return roi
    med = float(np.median(valid))
    hole = (mask > 0) & (roi <= 0.05)
    roi[hole] = med
    return roi


# ==================== 2. AprilTag 工作台系模块 ====================
TAG_DICTS = [
    (cv2.aruco.DICT_APRILTAG_25h9, "APRILTAG_25h9"),
    (cv2.aruco.DICT_APRILTAG_16h5, "APRILTAG_16h5"),
    (cv2.aruco.DICT_APRILTAG_36h11, "APRILTAG_36h11"),
]


def detect_tags(rgb, size_mm):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    s = size_mm / 1000.0
    obj_pts = np.array([
        [-s / 2, -s / 2, 0], [s / 2, -s / 2, 0],
        [s / 2, s / 2, 0], [-s / 2, s / 2, 0],
    ], dtype=np.float32)
    K = np.loadtxt(os.path.join(DEBUG_DIR, "../fp_capture/cam_K.txt")).astype(np.float64)
    dist = np.zeros((4, 1))
    for dict_id, name in TAG_DICTS:
        try:
            aruco_dict = cv2.aruco.Dictionary_get(dict_id)
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        except AttributeError:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        found = []
        for i, tag_id in enumerate(ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(obj_pts, corners[i][0], K, dist)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            if R[1, 2] > 0:
                R = R @ np.diag([1.0, -1.0, -1.0])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            found.append((int(tag_id), T))
        if found:
            return found
    return []


def build_world(tags, base_id=0):
    """tag0 定原点，多 tag 平面拟合定方向。返回 T_cam_world。"""
    t0 = [t for t in tags if t[0] == base_id]
    if not t0:
        return None
    T0 = t0[0][1]
    origin = T0[:3, 3]
    Ts = [T for _, T in tags]
    if len(Ts) >= 2:
        pts = np.array([T[:3, 3] for T in Ts])
        center = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - center)
        z = vh[-1]
        if z[1] > 0:
            z = -z
        z /= np.linalg.norm(z)
        if np.dot(z, T0[:3, 2]) < 0.7:
            z = T0[:3, 2]
    else:
        z = T0[:3, 2]
    if z[1] > 0:
        z = -z
    z /= np.linalg.norm(z)
    x = T0[:3, 0] - np.dot(T0[:3, 0], z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = origin
    return T


# ==================== 3. FoundationPose 模块 ====================
class PoseEstimator:
    def __init__(self, mesh_file=None):
        self.mesh = trimesh.load(mesh_file if mesh_file else MESH_FILE,
                                 force="mesh")
        # OBJ/STL/PLY 按 mm 建模 -> 转米；GLB/glTF/FBX 通常是米，不再缩放
        if str(mesh_file).lower().endswith((".obj", ".stl", ".ply")):
            self.mesh.apply_scale(0.001)
        # GLB/glTF 的 PBR 材质 -> FoundationPose 需要的 SimpleMaterial(image)
        try:
            from trimesh.visual.material import PBRMaterial, SimpleMaterial
            if isinstance(self.mesh.visual.material, PBRMaterial):
                img = self.mesh.visual.material.baseColorTexture
                if img is not None:
                    self.mesh.visual = trimesh.visual.TextureVisuals(
                        uv=self.mesh.visual.uv,
                        material=SimpleMaterial(image=img))
                    print("[模型] GLB 纹理已转换为 FoundationPose 格式")
        except Exception as e:
            print(f"[模型] 材质转换跳过: {e}")
        _, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=DEBUG_DIR,
            debug=1,
            glctx=glctx,
        )
        self.has_pose = False

    def register(self, K, rgb, depth_m, mask):
        t0 = time.time()
        res = self.est.register(K, rgb, depth_m, mask, iteration=REG_ITER)
        self.has_pose = True
        print(f"[FP] 注册完成 {time.time() - t0:.1f}s")
        return res[:3, :3], res[:3, 3]

    def track(self, K, rgb, depth_m):
        t0 = time.time()
        res = self.est.track_one(rgb, depth_m, K, iteration=TRACK_ITER)
        dt = time.time() - t0
        if dt > 0:
            print(f"[FP] 跟踪 {1.0 / dt:.1f} Hz")
        return res[:3, :3], res[:3, 3]


# ==================== 4. 坐标转换 + 抓取位姿 ====================
def to_world_and_compensate(M_cam_obj, T_world_cam):
    M_world = T_world_cam @ M_cam_obj
    # 补偿（先加后翻转）
    M_world[:3, 3] += np.array([
        OFFSET_X_MM / 1000.0, OFFSET_Y_MM / 1000.0, -CENTER_OFFSET_MM / 1000.0,
    ])
    F = np.eye(3)
    if FLIP_X:
        F[0, 0] = -1.0
    if FLIP_Y:
        F[1, 1] = -1.0
    if FLIP_X or FLIP_Y:
        M_world[:3, :3] = F @ M_world[:3, :3] @ F
        M_world[:3, 3] = F @ M_world[:3, 3]
    return M_world


def compute_grasp(M_world):
    """圆柱抓取规则：抓圆柱体中部、夹持方向垂直于中轴。
    姿态跟随罐子（z 沿中轴），绕中轴的 yaw 由约束唯一确定：
    三叉戟戟柄（局部 -x）水平投影与工作台 Y+ 夹角最小。"""
    import math
    R = M_world[:3, :3]
    t = M_world[:3, 3].copy() - R[:, 2] * (GRASP_OFFSET_MM / 1000.0)
    # 绕局部 z 轴旋转，使戟柄方向 -x 的水平投影对齐工作台 Y+
    A, B = R[0, 0], R[0, 1]
    n = math.hypot(A, B)
    if n > 1e-6:
        ca, sa = B / n, -A / n
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)
        Rg = R @ Rz
    else:
        Rg = R
    T = np.eye(4)
    T[:3, :3] = Rg
    T[:3, 3] = t
    return T


def compute_grasp_sphere(M_world, offset_mm=0.0):
    """对称物体(苹果)抓取规则：从上往下抓。
    夹爪竖直（叉尖朝工作台 -z 方向），三叉戟叉面法线沿工作台 +Y，
    即叉面垂直于 Y 轴。"""
    t = M_world[:3, 3].copy()
    t[2] += offset_mm / 1000.0  # 从上方接近，略高于物体中心
    d = np.array([0.0, 0.0, -1.0])   # 叉伸出方向：竖直向下
    n = np.array([0.0, 1.0, 0.0])    # 叉面法线：工作台 Y
    R = np.column_stack([d, np.cross(n, d), n])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ==================== 5. 可视化模块 ====================
def draw_2d(rgb, mesh, R, t, K, bbox, target, yboxes):
    out = rgb.copy()
    # 所有 YOLO 框（蓝色）
    for o in yboxes:
        x1, y1, x2, y2 = o
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # 目标掩膜（绿色半透明）
    if target is not None and target["mask"] is not None:
        mask = target["mask"]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = out.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    # 目标姿态框（黄色）
    P = mesh.vertices @ R.T + t
    z = P[:, 2]
    pts = np.column_stack([
        P[:, 0] * K[0, 0] / z + K[0, 2],
        P[:, 1] * K[1, 1] / z + K[1, 2],
    ])
    mm = np.zeros(rgb.shape[:2], np.uint8)
    for f in mesh.faces:
        if np.any(z[f] < 0.02):
            continue
        cv2.fillConvexPoly(mm, pts[f].astype(np.int32), 255)
    contours, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    ob_in_cam = np.eye(4)
    ob_in_cam[:3, :3] = R
    ob_in_cam[:3, 3] = t
    out = draw_posed_3d_box(K, out, ob_in_cam, bbox,
                            line_color=(0, 255, 255), linewidth=2)
    if target is not None:
        cv2.putText(out, f"TARGET: {target['name']} {target['conf']:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def draw_3d(ax, fig, plt, M_world, T_grasp, save_path, mn, mx):
    ax.clear()
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.5)
    ax.set_zlim(0.0, 0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    L = 0.12
    for i, c in enumerate(["r", "g", "b"]):
        v = np.zeros(3)
        v[i] = L
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=c, lw=2)
    ax.text(0.13, 0, 0, "X", color="r")
    ax.text(0, 0.13, 0, "Y", color="g")
    ax.text(0, 0, 0.13, "Z", color="b")

    corners = np.array([[x, y, z]
                        for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1])
                        for z in (mn[2], mx[2])])
    R = M_world[:3, :3]
    t = M_world[:3, 3]
    P = corners @ R.T + t
    edges = [(0, 1), (0, 2), (0, 4), (7, 6), (7, 5), (7, 3),
             (1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]
    for a, b in edges:
        ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]],
                [P[a, 2], P[b, 2]], color="orange", lw=2)
    c = M_world[:3, 3]
    ax.scatter([c[0]], [c[1]], [c[2]], color="red", s=40)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()
    for name, p in {"XY": (c[0], c[1], z0),
                    "XZ": (c[0], y1, c[2]),
                    "YZ": (x0, c[1], c[2])}.items():
        ax.scatter([p[0]], [p[1]], [p[2]], color="yellow", s=50,
                   edgecolors="black", linewidths=1.2)
        ax.plot([c[0], p[0]], [c[1], p[1]], [c[2], p[2]],
                color="gray", ls="--", lw=1.5)
        ax.text(p[0], p[1], p[2] + 0.005, name, color="black",
                fontsize=8, ha="center")
    ax.text(c[0], c[1], c[2] + 0.01,
            f"center ({c[0]*1000:.0f}, {c[1]*1000:.0f}, {c[2]*1000:.0f}) mm",
            color="red", fontsize=9)

    if T_grasp is not None:
        Rg = T_grasp[:3, :3]
        tg = T_grasp[:3, 3]

        def w3(local):
            return Rg @ local + tg

        fork_m = GRASP_FORK_MM / 1000.0
        sp_m = GRASP_SPACING_MM / 1000.0
        for sy in (-sp_m, 0.0, sp_m):
            base = w3(np.array([-fork_m, sy, 0]))
            tip = w3(np.array([0.0, sy, 0]))
            ax.plot([base[0], tip[0]], [base[1], tip[1]],
                    [base[2], tip[2]], color="cyan", lw=2)
        a = w3(np.array([-fork_m, 0, 0]))
        b = w3(np.array([-fork_m - GRASP_HANDLE_MM / 1000.0, 0, 0]))
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="cyan", lw=3)
        ax.text(*(w3(np.array([0, 0, 0.01]))), "GRASP", color="cyan")

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=30, azim=-60)
    try:
        fig.savefig(save_path, dpi=90)
    except Exception as e:
        print(f"[3D] 保存到桌面失败(文件可能被占用): {e}")
    try:
        fig.savefig(os.path.join(DEBUG_DIR, "live_pipeline_3d.png"), dpi=90)
    except Exception:
        pass
    try:
        plt.draw()
        plt.pause(0.01)
    except Exception:
        pass


# ==================== 6. 主循环 ====================
def recv_latest(sub):
    parts = sub.recv_multipart()
    while True:
        try:
            parts = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    return parts


def main():
    global MESH_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="标定 tag 布局后退出")
    ap.add_argument("--no-3d", action="store_true", help="关闭 3D 窗口")
    ap.add_argument("--mesh", default=MESH_FILE, help="CAD 模型 OBJ 路径")
    args = ap.parse_args()

    MESH_FILE = args.mesh
    print("CAD 模型:", MESH_FILE)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    set_logging_format()
    set_seed(0)

    # 3D 窗口
    _fig3d = _ax3d = _plt3d = None
    if not args.no_3d:
        import matplotlib
        try:
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            _fig3d = plt.figure(figsize=(6, 6))
            _ax3d = _fig3d.add_subplot(111, projection="3d")
            plt.ion()
            _plt3d = plt
            print("[3D] 实时窗口已启用")
        except Exception:
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            _fig3d = plt.figure(figsize=(6, 6))
            _ax3d = _fig3d.add_subplot(111, projection="3d")
            _plt3d = plt

    # 模型
    from ultralytics import YOLO
    yolo = YOLO(MODEL_PATH)
    print(f"YOLO 模型: {MODEL_PATH}")
    try:
        print("类别:", yolo.names)
    except Exception:
        pass
    print(f"目标标签: {TARGET_LABEL}")

    fp = PoseEstimator(MESH_FILE)
    current_obj = None

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_ADDR)
    sub.setsockopt_string(zmq.SUBSCRIBE, "FRAME")
    print("等待 fp_bridge 画面...")

    frame_id = 0
    smooth_q = smooth_t = None
    tag_q = tag_t = None
    track_count = 0
    first = True
    yboxes_all = []

    while True:
        try:
            parts = recv_latest(sub)
        except KeyboardInterrupt:
            break
        frame_id += 1
        rgb = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
        depth = np.frombuffer(parts[2], np.uint16).reshape(400, 640)
        K = np.frombuffer(parts[3], np.float64).reshape(3, 3)
        H, W = rgb.shape[:2]
        if depth.shape[:2] != (H, W):
            aligned = np.zeros((H, W), np.uint16)
            aligned[: depth.shape[0], :] = depth
            depth = aligned
        depth_m = depth.astype(np.float32) / 1000.0

        # --- 2. AprilTag 工作台系 ---
        tags = detect_tags(rgb, TAG_SIZE_MM)
        T_world = build_world(tags) if tags else None
        if T_world is None:
            cv2.putText(rgb, "NO tag0", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if frame_id % 5 == 0:
                cv2.imwrite(VIZ2D_FILE, rgb)
            continue
        if TAG_ALPHA > 0:
            q = R_util.from_matrix(T_world[:3, :3]).as_quat()
            tv = T_world[:3, 3].copy()
            if tag_q is None:
                tag_q, tag_t = q, tv
            else:
                if np.dot(tag_q, q) < 0:
                    q = -q
                tag_q = tag_q + TAG_ALPHA * (q - tag_q)
                tag_q /= np.linalg.norm(tag_q)
                tag_t = tag_t + TAG_ALPHA * (tv - tag_t)
            T_world = np.eye(4)
            T_world[:3, :3] = R_util.from_quat(tag_q).as_matrix()
            T_world[:3, 3] = tag_t
        T_world_cam = np.linalg.inv(T_world)

        # --- 1. YOLO 检测所有物体 + 选目标 ---
        objs = detect_all_objects(rgb, yolo)
        target = select_target(objs, TARGET_LABEL)
        yboxes_all = [o["xyxy"] for o in objs]
        if target is None:
            print("[目标] 没有检测到物体")
            continue

        # 物体种类 -> 模型 / 抓取规则
        obj_key = FORCE_OBJECT or YOLO_TO_OBJECT.get(
            str(target["name"]), YOLO_TO_OBJECT.get(
                str(target["cls"]), DEFAULT_OBJECT))
        mesh_file = OBJECT_MODELS.get(obj_key, "")
        if not mesh_file or not os.path.exists(mesh_file):
            print(f"[目标] 物体 {obj_key} 的 CAD 模型未配置，跳过（"
                  f"请在 OBJECT_MODELS 里补充）")
            continue
        rule = GRASP_RULES.get(obj_key, GRASP_RULES[DEFAULT_OBJECT])
        if obj_key != current_obj:
            print(f"[目标] 物体={obj_key}，加载模型: {mesh_file}")
            fp = PoseEstimator(mesh_file)
            current_obj = obj_key
            first = True          # 新模型需要重新注册
            track_count = 0
            smooth_q = smooth_t = None
        global GRASP_OFFSET_MM
        GRASP_OFFSET_MM = float(rule.get("offset_mm", 5.0))

        mask = target["mask"] if target["mask"] is not None else np.zeros((H, W), np.uint8)
        if mask.sum() == 0:
            x1, y1, x2, y2 = target["xyxy"]
            mask[y1:y2, x1:x2] = 255

        # --- 3. FoundationPose 位姿 ---
        if first or not USE_TRACK:
            depth_filled = fill_depth_roi(depth_m, mask)
            R, t = fp.register(K, rgb, depth_filled, mask)
            first = False
            track_count = 0
            smooth_q = smooth_t = None
        else:
            R, t = fp.track(K, rgb, depth_m)
            track_count += 1
        if not (0.3 < t[2] < 1.5):
            print(f"[FP] 距离异常 z={t[2]*1000:.0f}mm，重新注册")
            first = True
            continue

        # --- 4. 坐标转换（相机→工作台 + 补偿/flip） + 平滑 ---
        M_cam = np.eye(4)
        M_cam[:3, :3] = R
        M_cam[:3, 3] = t
        M_world = to_world_and_compensate(M_cam, T_world_cam)
        if ALPHA > 0:
            q = R_util.from_matrix(M_world[:3, :3]).as_quat()
            tv = M_world[:3, 3].copy()
            if smooth_q is None:
                smooth_q, smooth_t = q, tv
            else:
                if np.dot(smooth_q, q) < 0:
                    q = -q
                smooth_q = smooth_q + ALPHA * (q - smooth_q)
                smooth_q /= np.linalg.norm(smooth_q)
                smooth_t = smooth_t + ALPHA * (tv - smooth_t)
            M_world = np.eye(4)
            M_world[:3, :3] = R_util.from_quat(smooth_q).as_matrix()
            M_world[:3, 3] = smooth_t
        np.save(POSE_FILE, M_world)
        w = M_world[:3, 3] * 1000.0
        print(f"world (mm): x={w[0]:.1f} y={w[1]:.1f} z={w[2]:.1f}")

        # --- 5. 抓取位姿 ---
        rtype = rule.get("type", "cylinder")
        if rtype == "sphere":
            T_grasp = compute_grasp_sphere(M_world, rule.get("offset_mm", 0.0))
        else:
            T_grasp = compute_grasp(M_world)
        np.save(GRASP_FILE, T_grasp)
        g = T_grasp[:3, 3] * 1000.0
        print(f"grasp(mm): x={g[0]:.1f} y={g[1]:.1f} z={g[2]:.1f}")

        # --- 可视化 ---
        R_cam = R          # fp 返回的 R/t 就是相机系位姿，直接用于 2D 画框
        t_cam = t
        viz = draw_2d(rgb, fp.mesh, R_cam, t_cam, K, fp.bbox,
                      target, yboxes_all)
        if frame_id % 3 == 0:
            cv2.imwrite(VIZ2D_FILE, viz)
            cv2.imwrite(os.path.join(DEBUG_DIR, "live_pipeline.png"), viz)
        if _ax3d is not None:
            mn = fp.mesh.bounds[0]
            mx = fp.mesh.bounds[1]
            draw_3d(_ax3d, _fig3d, _plt3d, M_world, T_grasp, VIZ3D_FILE, mn, mx)
        try:
            cv2.imshow("pipeline (q: quit)", viz)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception:
            pass

    print("退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
