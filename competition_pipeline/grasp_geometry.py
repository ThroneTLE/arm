"""抓取高度后处理：把 FoundationPose 给出的物体位姿变成一个不会压爆的抓取点。

这个模块**不改变任何坐标约定**，也不参与位姿估计。输入是 FoundationPose 已经算好的
``user1_from_object``（用户坐标系1 下的物体位姿）和网格包围盒，输出只有一个 Z。

为什么需要它
------------
夹爪的**腔体深度**（物体能伸进夹爪的长度）实测 **80mm**。原来的做法是把抓取点直接放在
物体位姿的平移上（``visual_grasp_bridge.top_down_grasp_frame`` 里的
``grasp[:3,3] = source[:3,3]``），也就是对准物体中心。对一个高 H 的直立物体，这要求

    H / 2 <= 80mm      即   H <= 160mm

高于 160mm 的瓶子必然让掌根压在瓶口上。2026-08-22 现场隔壁组就是这样把瓶子压爆的。

改用"顶点与中心的中点"后（本模块 :func:`grasp_height_mm` 的 cylinder 规则）伸入深度
变成 ``H/4``，配合 ``腔体深度 - 安全余量 = 65mm`` 的钳位：

    H / 4 <= 65mm      即   H <= 260mm 不触发钳位

实际要抓的：易拉罐 115mm -> 28.75mm，雀巢咖啡 169mm -> 42.25mm，
可口可乐 245mm -> 61.25mm，都在 65mm 内。

**超过 260mm 也不会压爆** —— 钳位把伸入深度压回 65mm，即抓得更浅。钳位永远是安全
方向，不是失败；触发时只在预览里提示一句。

抓取点落在 3/4 高度处还有个附带好处：**在遮挡之上**，直接解决"水果挡住瓶子下部"那一档。

两个把人坑过的细节
------------------
1. **"中心"必须是包围盒中心，不是位姿平移。** 各网格的原点约定并不统一：罐体/橙子/
   柠檬/可乐的原点在几何中心，但 **apple 的原点在果底下方 5mm（偏 49.1mm）**、
   **雀巢咖啡的原点在瓶底下方 10mm（偏 74.5mm）**。直接拿位姿平移当中心，抓苹果会
   低 44mm —— 把夹爪怼进桌面。

2. **长轴方向各网格不同**（罐子 Z、苹果 Y、柠檬 X、两个新瓶 Y），所以高度/直径不能按
   固定轴取。这里改为看旋转矩阵哪一行的 Z 分量最大，即"哪根物体轴当前最竖直"。

用户坐标系1 的 z=0 就是桌面（``docs/竞赛视觉引导方案-20260822.md``），所以这里所有的
Z 都可以直接读成"离桌面多高"。
"""

import math
from dataclasses import dataclass

import numpy as np

from .geometry import as_transform

#: 夹爪腔体深度（物体能伸进夹爪的长度），2026-08-22 实测保守值。
JAW_CAVITY_DEPTH_MM = 80.0

#: 腔体深度上留的安全余量：视觉高度估计误差 + 罐口凸起 + 到位误差。
SAFETY_CLEARANCE_MM = 15.0

#: 物体顶面高度的合理区间。超出多半是位姿估错了，宁可拒绝也不要往下压。
MIN_PLAUSIBLE_TOP_MM = 10.0
MAX_PLAUSIBLE_TOP_MM = 350.0

#: 采用"顶点与中心的中点"规则的形状类。其余一律对准中心。
MIDPOINT_SHAPES = ("cylinder",)


@dataclass(frozen=True)
class ObjectExtent:
    """物体在**用户坐标系1** 下的尺寸，全部毫米。"""

    z_top_mm: float
    z_bottom_mm: float
    z_center_mm: float
    height_mm: float
    #: 夹爪需要跨越的宽度：柱/球取水平最大跨度；横躺的长条取水平**最小**跨度，
    #: 因为顶抓是跨短轴合拢的（见 ``top_down_grasp_frame`` 的 elongated 分支）。
    grasp_width_mm: float
    center_xy_mm: tuple


def _bounds_corners(bounds_m):
    """包围盒 (2,3) -> 8 个角点 (8,3)。"""
    bounds = np.asarray(bounds_m, dtype=np.float64).reshape(2, 3)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("mesh bounds 含非有限值：{!r}".format(bounds_m))
    if np.any(bounds[1] < bounds[0]):
        raise ValueError("mesh bounds 的 max 小于 min：{!r}".format(bounds_m))
    lower, upper = bounds
    return np.asarray(
        [
            [lower[0] if not i & 1 else upper[0],
             lower[1] if not i & 2 else upper[1],
             lower[2] if not i & 4 else upper[2]]
            for i in range(8)
        ],
        dtype=np.float64,
    )


def object_extent_user1(user1_from_object, mesh_bounds_m, grasp_type="cylinder"):
    """把网格包围盒变换到用户系，返回 :class:`ObjectExtent`（毫米）。

    ``mesh_bounds_m`` 是**米制、已应用 object_model_scales 缩放**的包围盒，正是
    ``tool.visual_grasp_pipeline.foundationpose.FoundationPosePoseEstimator.mesh_bounds()``
    的返回值 —— 直接复用，不要在这里再乘一次缩放。

    高度与直径取自**物体系**的边长（按"哪根轴最竖直"分配），而不是变换后的轴对齐
    包围盒：后者会随偏航角把直径放大最多 41%（D·√2），让宽度闸门误伤本可抓取的罐子。
    Z 相关的值则取自变换后的角点，那是精确的。
    """
    pose = as_transform(user1_from_object, "user1_from_object")
    corners_m = _bounds_corners(mesh_bounds_m)
    world = (pose[:3, :3] @ corners_m.T).T + pose[:3, 3]
    world_mm = world * 1000.0

    z_top = float(world_mm[:, 2].max())
    z_bottom = float(world_mm[:, 2].min())
    center = world_mm.mean(axis=0)      # 8 角点均值 == 包围盒中心，随刚体变换保持

    # 哪根物体轴当前最竖直 -> 它的边长是高度，另外两根是水平尺寸。
    extents_mm = (np.asarray(mesh_bounds_m, dtype=np.float64).reshape(2, 3)[1]
                  - np.asarray(mesh_bounds_m, dtype=np.float64).reshape(2, 3)[0]) * 1000.0
    verticality = np.abs(pose[:3, :3][2, :])       # 各物体轴在用户 Z 上的分量
    vertical_axis = int(np.argmax(verticality))
    horizontal = [extents_mm[i] for i in range(3) if i != vertical_axis]

    if str(grasp_type) == "elongated":
        # 横躺长条：顶抓跨**短轴**合拢，宽度取小的那个。
        grasp_width = float(min(horizontal))
    else:
        grasp_width = float(max(horizontal))

    return ObjectExtent(
        z_top_mm=z_top,
        z_bottom_mm=z_bottom,
        z_center_mm=float(0.5 * (z_top + z_bottom)),
        height_mm=float(z_top - z_bottom),
        grasp_width_mm=grasp_width,
        center_xy_mm=(float(center[0]), float(center[1])),
    )


@dataclass(frozen=True)
class GraspHeight:
    """抓取高度决策结果，全部毫米。"""

    z_mm: float
    #: 指尖到物体顶面的距离，也就是物体伸进夹爪腔体的深度。
    engage_mm: float
    #: 是否被腔体深度钳位过（正常物体不会触发）。
    clamped: bool
    #: 钳位前按规则算出的原始值，便于在预览里说清楚发生了什么。
    requested_engage_mm: float
    rule: str


def grasp_height_mm(extent, grasp_type="cylinder",
                    jaw_cavity_depth_mm=JAW_CAVITY_DEPTH_MM,
                    safety_clearance_mm=SAFETY_CLEARANCE_MM):
    """决定抓取点的 Z（用户系，毫米）。

    规则（现场确定）::

        瓶/罐 (cylinder)          z = (z_顶点 + z_中心) / 2      -> engage = 高度/4
        水果等 (sphere/elongated) z = z_中心                      -> engage = 高度/2

    然后统一钳位 ``engage <= 腔体深度 - 安全余量``。钳位是兜底：中点规则本身对
    高度 <= 320mm 的物体就不会触发，它只防"超高物体"和"视觉把高度估大了"。
    """
    if not isinstance(extent, ObjectExtent):
        raise TypeError("expected ObjectExtent, got {!r}".format(type(extent)))
    cavity = float(jaw_cavity_depth_mm)
    clearance = float(safety_clearance_mm)
    if cavity <= 0.0:
        raise ValueError("jaw_cavity_depth_mm 必须为正")
    if not 0.0 <= clearance < cavity:
        raise ValueError("safety_clearance_mm 必须在 [0, 腔体深度) 内")

    if str(grasp_type) in MIDPOINT_SHAPES:
        z = 0.5 * (extent.z_top_mm + extent.z_center_mm)
        rule = "顶点与中心的中点"
    else:
        z = extent.z_center_mm
        rule = "对准中心"

    requested = float(extent.z_top_mm - z)
    limit = cavity - clearance
    if requested > limit:
        z = extent.z_top_mm - limit
        return GraspHeight(float(z), float(limit), True, requested, rule)
    return GraspHeight(float(z), requested, False, requested, rule)


def check_graspable(extent, grasp_height, jaw_max_open_mm, width_margin_mm=6.0):
    """返回拒绝原因列表（空列表 = 可抓）。

    宽度超限是**机械限制**，报文要说清楚，免得明天现场把它当成软件 bug 去调。
    """
    reasons = []
    usable = float(jaw_max_open_mm) - float(width_margin_mm)
    if extent.grasp_width_mm > usable:
        reasons.append(
            "物体夹持宽度 {:.1f}mm 超过夹爪可用张开 {:.1f}mm"
            "（最大张开 {:.1f} − 余量 {:.1f}）。这是**机械限制不是软件问题**，"
            "请换目标物体或换夹爪。".format(
                extent.grasp_width_mm, usable,
                float(jaw_max_open_mm), float(width_margin_mm),
            )
        )
    if not (MIN_PLAUSIBLE_TOP_MM <= extent.z_top_mm <= MAX_PLAUSIBLE_TOP_MM):
        reasons.append(
            "物体顶面高度 {:.1f}mm 不在合理区间 [{:.0f}, {:.0f}]mm，"
            "位姿多半估错了；高度估错的后果正是压爆，因此拒绝执行。".format(
                extent.z_top_mm, MIN_PLAUSIBLE_TOP_MM, MAX_PLAUSIBLE_TOP_MM,
            )
        )
    if extent.height_mm <= 0.0 or not math.isfinite(extent.height_mm):
        reasons.append("物体高度 {:.1f}mm 非法".format(extent.height_mm))
    if grasp_height.z_mm <= 0.0:
        reasons.append(
            "抓取点 Z={:.1f}mm 在桌面下方，会把夹爪怼进桌子".format(grasp_height.z_mm)
        )
    return reasons


def place_height_mm(extent, grasp_height, clearance_mm=2.0):
    """放置时指尖应落在的 Z，使物体底面正好坐回桌面（用户系 z=0）并保持直立。

    抓取时指尖在 ``z_top - engage``，此时物体底面在 ``z_bottom``。夹爪与物体的相对
    关系在搬运途中不变，所以要让底面回到 0，指尖就应落在

        engage 之上的那段物体长度 = (z_top - engage) - z_bottom

    再加一点余量，松爪后物体自己坐稳。任务书要求"放置时需要直立"，顶抓+垂直下放
    全程不改变物体姿态，这一条自动满足。
    """
    above_bottom = (extent.z_top_mm - grasp_height.engage_mm) - extent.z_bottom_mm
    return float(above_bottom + float(clearance_mm))


def cloud_top_consistency(z_top_mm, cloud_points_user1_mm, center_xy_mm,
                          radius_mm, tolerance_mm=20.0):
    """用掩膜点云的上沿交叉校验 ``z_top``；返回 (是否一致, 点云顶面, 用到的点数)。

    只做**校验**，不改变 z_top 的取值来源。高度估错的后果是压爆，所以值得用第二个
    独立来源确认一次。

    注意掩膜是 YOLO 的**矩形框**（现场决定不用实例分割），框内混着桌面和邻近物体，
    所以不能直接取 max Z —— 那可能是旁边更高的物体。这里只取物体 XY 中心 ``radius_mm``
    半径内的点，并用 95 分位数而不是最大值，避开深度噪点。
    """
    points = np.asarray(cloud_points_user1_mm, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return True, None, 0            # 没有点云可比 -> 不阻塞
    center = np.asarray(center_xy_mm, dtype=np.float64).reshape(2)
    radial = np.linalg.norm(points[:, :2] - center, axis=1)
    near = points[radial <= float(radius_mm)]
    if near.shape[0] < 20:
        return True, None, int(near.shape[0])
    cloud_top = float(np.percentile(near[:, 2], 95.0))
    return (
        abs(cloud_top - float(z_top_mm)) <= float(tolerance_mm),
        cloud_top,
        int(near.shape[0]),
    )


__all__ = [
    "JAW_CAVITY_DEPTH_MM",
    "SAFETY_CLEARANCE_MM",
    "MIN_PLAUSIBLE_TOP_MM",
    "MAX_PLAUSIBLE_TOP_MM",
    "GraspHeight",
    "ObjectExtent",
    "check_graspable",
    "cloud_top_consistency",
    "grasp_height_mm",
    "object_extent_user1",
    "place_height_mm",
]
