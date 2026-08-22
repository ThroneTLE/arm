"""多物体放置的槽位排布与"已放置区域"排除。

要解决三个连锁问题（2026-08-22 现场推演）
------------------------------------------
连续抓多个物体时，流程是 抓 -> 放 -> 回复位 -> 重新拍照 -> 继续抓。于是：

1. **堆叠**：每次都放同一个点，第二个物体落在第一个上面必然倒。任务书明确要求
   "放置时需要直立"，堆叠直接判负。-> 用**槽位**，每放一个前进一格。
2. **二次识别**：重新拍照时，刚放好的物体还在桌上，会被再检测一遍；如果按名字
   指定目标，很可能把已经放好的又抓回来。-> 用**排除区**，落在已占用槽位附近的
   检测结果不参与抓取。
3. **遮挡**：放置区若离物件堆太近，放好的物体会挡住还没抓的目标。
   -> 放置区默认放在 X 负半轴**外侧**，远离桌面中央的杂物区。

坐标系
------
全部是用户坐标系1（原点=49.3cm 方桌中心，+X 前方，+Y 左方，+Z 上方，z=0 即桌面）。
本模块只管 XY 布局；放置高度由 :mod:`competition_pipeline.grasp_geometry` 按物体
几何算，保证松爪后底面正好坐回桌面。
"""

import math
from dataclasses import dataclass

import numpy as np

#: 默认放置区起点（用户系 mm）。选在 X 负半轴外侧的近端左角，远离桌面中央杂物区。
DEFAULT_PLACE_ORIGIN_MM = (-170.0, 170.0)

#: 默认排布方向：沿 -Y（从左往右）排开一列。
DEFAULT_PLACE_DIRECTION = (0.0, -1.0)

#: 默认槽位间距（mm）。必须大于最大物体直径 + 余量，否则相邻两个会碰到。
#: 场上最宽的是苹果 75mm，90mm 留 15mm 余量。
DEFAULT_PLACE_PITCH_MM = 90.0

#: 默认槽位数。三档任务最多也就抓几个。
DEFAULT_PLACE_SLOT_COUNT = 4

#: 判定"这个检测结果是我刚放的"时用的半径（mm）。
DEFAULT_EXCLUSION_RADIUS_MM = 60.0


@dataclass(frozen=True)
class PlaceLayout:
    """一列放置槽位。全部单位 mm，用户坐标系1。"""

    origin_xy_mm: tuple = DEFAULT_PLACE_ORIGIN_MM
    direction: tuple = DEFAULT_PLACE_DIRECTION
    pitch_mm: float = DEFAULT_PLACE_PITCH_MM
    count: int = DEFAULT_PLACE_SLOT_COUNT
    table_half_mm: float = 246.5
    exclusion_radius_mm: float = DEFAULT_EXCLUSION_RADIUS_MM

    def __post_init__(self):
        direction = np.asarray(self.direction, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-9:
            raise ValueError("放置方向不能是零向量")
        object.__setattr__(self, "direction", tuple(direction / norm))
        if int(self.count) < 1:
            raise ValueError("槽位数至少为 1")
        if float(self.pitch_mm) <= 0.0:
            raise ValueError("槽位间距必须为正")

    def slot_xy_mm(self, index):
        """第 ``index`` 个槽位的 XY（从 0 开始）。"""
        index = int(index)
        if not 0 <= index < int(self.count):
            raise IndexError(
                "槽位 {} 超出范围 [0, {})，放置区已经用满 —— "
                "先把已放置的物体清走，或调大 place_slots.count".format(
                    index, int(self.count)
                )
            )
        origin = np.asarray(self.origin_xy_mm, dtype=np.float64).reshape(2)
        direction = np.asarray(self.direction, dtype=np.float64).reshape(2)
        point = origin + direction * (float(self.pitch_mm) * index)
        return (float(point[0]), float(point[1]))

    def all_slots_mm(self):
        return [self.slot_xy_mm(index) for index in range(int(self.count))]

    def validate(self, largest_object_diameter_mm=75.0):
        """返回问题列表（空 = 布局可用）。

        这些都是"配置写错了但不会报错、只会在现场表现为物体互相碰倒"的情况。
        """
        reasons = []
        half = float(self.table_half_mm)
        for index, (x, y) in enumerate(self.all_slots_mm()):
            if abs(x) > half or abs(y) > half:
                reasons.append(
                    "槽位 {} 在 ({:.0f}, {:.0f})，超出桌面 ±{:.1f}mm".format(
                        index, x, y, half
                    )
                )
        needed = float(largest_object_diameter_mm) + 15.0
        if float(self.pitch_mm) < needed:
            reasons.append(
                "槽位间距 {:.0f}mm 小于最大物体直径 {:.0f}mm + 15mm 余量，"
                "相邻两个物体会碰到".format(
                    float(self.pitch_mm), float(largest_object_diameter_mm)
                )
            )
        if float(self.exclusion_radius_mm) > float(self.pitch_mm) / 2.0:
            reasons.append(
                "排除半径 {:.0f}mm 超过槽位间距的一半 {:.0f}mm，"
                "会把相邻槽位也当成『已放置』".format(
                    float(self.exclusion_radius_mm), float(self.pitch_mm) / 2.0
                )
            )
        return reasons


def is_in_placed_region(xy_mm, layout, occupied_slots, radius_mm=None):
    """这个检测结果是不是"我刚放到那儿的"。

    重新拍照时刚放好的物体还在桌上，会被再检测一遍。按名字指定目标时很容易把
    已经放好的又抓回来 —— 这个判定就是用来把它们从候选里剔除的。
    """
    if not occupied_slots:
        return False
    radius = float(radius_mm if radius_mm is not None
                   else layout.exclusion_radius_mm)
    point = np.asarray(xy_mm, dtype=np.float64).reshape(2)
    for index in occupied_slots:
        slot = np.asarray(layout.slot_xy_mm(index), dtype=np.float64)
        if float(np.linalg.norm(point - slot)) <= radius:
            return True
    return False


def layout_from_config(workspace):
    """从 ``competition.yaml`` 的 ``workspace`` 段构建；缺省用默认值。

    ``place_user_xy_mm`` 同时是"单目标放置点"和"第 0 号槽位"，两者保持一致，
    免得单目标和多目标放到不同地方。
    """
    workspace = dict(workspace or {})
    slots = dict(workspace.get("place_slots", {}) or {})
    origin = workspace.get("place_user_xy_mm") or DEFAULT_PLACE_ORIGIN_MM
    return PlaceLayout(
        origin_xy_mm=(float(origin[0]), float(origin[1])),
        direction=tuple(
            float(value) for value in
            (slots.get("direction") or DEFAULT_PLACE_DIRECTION)
        ),
        pitch_mm=float(slots.get("pitch_mm", DEFAULT_PLACE_PITCH_MM)),
        count=int(slots.get("count", DEFAULT_PLACE_SLOT_COUNT)),
        table_half_mm=float(workspace.get("table_half_size_mm", 246.5)),
        exclusion_radius_mm=float(
            slots.get("exclusion_radius_mm", DEFAULT_EXCLUSION_RADIUS_MM)
        ),
    )


__all__ = [
    "DEFAULT_EXCLUSION_RADIUS_MM",
    "DEFAULT_PLACE_DIRECTION",
    "DEFAULT_PLACE_ORIGIN_MM",
    "DEFAULT_PLACE_PITCH_MM",
    "DEFAULT_PLACE_SLOT_COUNT",
    "PlaceLayout",
    "is_in_placed_region",
    "layout_from_config",
]
