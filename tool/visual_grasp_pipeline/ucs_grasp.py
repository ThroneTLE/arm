"""用户坐标系1 一键"计算->确认->抓取放置"执行器（复用比赛控制栈）。

现场结论（2026-08-22，日志+官方手册）：控制器"复位点设置-安全使能"开启后，
任何新一轮运动/程序启动前，机器人必须位于复位点允许范围；上一轮若结束在
放置位上方（非复位点），下一轮会被控制器拒绝并报六轴偏差
（实测 [-1.3, 39.3, -35.5, -2.2, -22.9, 1.7]°）。

因此每轮固定流程（姿态全程 = 复位点姿态，只动 XYZ）：

    回复位(0x3007) -> 读初始姿态 ->
    抓取位**上方** -> 垂直下降到抓取位 -> 夹爪合(DOUT15/16) ->
    垂直抬升 -> 高位横移到放置位上方 -> 垂直下降到放置位 -> 夹爪开 ->
    垂直抬升 -> 回复位

为什么必须有"上方/抬升"这几段
------------------------------
旧版是 ``抓取位 -> 直接横移到放置位``，两点 Z 相同，等于**夹着物体在桌面上方
八十几毫米处横扫一两百毫米**，会把路径上的其它物品全部撞倒。赛题第二档是
"水果挡在瓶子前面"、第三档是"水果在另一个水果后面"，横移必然扫到。
旧版也没有接近段：从复位点到抓取位是一条斜线，会斜着插进物件堆里。

现在改成"垂直进、垂直出、高位横移"，与 ``competition_pipeline.grasp_demo``
里已验证的序列一致。

失败/取消：立即急停，绝不把机器人留在放置位上方。

复用 ``competition_pipeline.nexbot_jog.NexBotTcpJog``（6001 运动口，
MOVL coord=3 用户系1，与比赛 UI 同一套已验证协议栈）。运动统一走
``jog.move_to_ucs``，从而继承单位闸门、姿态闸门、使能前置与到位校验；
**不要**再直调 ``jog.controller.move_to`` —— 那会绕过事务锁和全部闸门。
所有守卫默认关闭真实运动：需显式 ``--enable-robot-motion`` 并在 UI 二次确认。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

import yaml

#: 放置点默认值（用户坐标系1, mm）；Z 始终取"抓取点的 Z"
UCS_PLACE_X_MM = -100.0
UCS_PLACE_Y_MM = 100.0

#: 安全硬限（用户坐标系1, mm）：原点在桌面上，Z=0 即桌面
SAFE_XY_MM = 300.0
SAFE_Z_MIN_MM = 10.0
SAFE_Z_MAX_MM = 350.0
MAX_SINGLE_LEG_MM = 600.0

#: 速度比例（0.05 -> 50 mm/s，与比赛 UI 步进一致，安全）
SPEED_SCALE = 0.05

#: 接近/抬升高度（mm）。抓取位上方这么高开始垂直下降，夹住后垂直抬这么高再横移。
#: 必须高过场上最高的干扰物，否则横移仍会撞。赛题物件最高是可乐瓶 245mm，
#: 抓取点在 3/4 高度处(约 184mm)，抬 120mm 后指尖到约 304mm，高过全部物件。
APPROACH_CLEARANCE_MM = 120.0


class UcsGraspSafetyError(RuntimeError):
    """目标点超出安全硬限，禁止执行。"""


class UcsGraspExecutorError(RuntimeError):
    """执行过程中控制器报错/拒绝（已尝试急停）。"""


@dataclass(frozen=True)
class UcsGraspPlan:
    grasp_xyz_mm: np.ndarray
    place_xyz_mm: np.ndarray
    steps: tuple
    description: str


def validate_targets(grasp_xyz_mm, place_xyz_mm) -> None:
    """纯函数守卫：工作区间硬限 + 单段距离硬限，不满足抛
    :class:`UcsGraspSafetyError`。"""
    grasp = np.asarray(grasp_xyz_mm, dtype=np.float64).reshape(3)
    place = np.asarray(place_xyz_mm, dtype=np.float64).reshape(3)
    for name, point in (("抓取", grasp), ("放置", place)):
        if float(np.max(np.abs(point[:2]))) > SAFE_XY_MM:
            raise UcsGraspSafetyError(
                "{}点 XY({:.1f}, {:.1f}) 超出 ±{:.0f}mm 安全区间".format(
                    name, point[0], point[1], SAFE_XY_MM
                )
            )
        if not SAFE_Z_MIN_MM <= float(point[2]) <= SAFE_Z_MAX_MM:
            raise UcsGraspSafetyError(
                "{}点 Z={:.1f}mm 超出 [{:.0f}, {:.0f}]mm 安全高度".format(
                    name, point[2], SAFE_Z_MIN_MM, SAFE_Z_MAX_MM
                )
            )
    leg_mm = float(np.linalg.norm(place - grasp))
    if leg_mm > MAX_SINGLE_LEG_MM:
        raise UcsGraspSafetyError(
            "抓取->放置距离 {:.0f}mm 超过 {:.0f}mm 单段上限".format(
                leg_mm, MAX_SINGLE_LEG_MM
            )
        )


def build_ucs_grasp_plan(
    grasp_xyz_mm,
    rotation: Optional[np.ndarray],
    place_x_mm: float = UCS_PLACE_X_MM,
    place_y_mm: float = UCS_PLACE_Y_MM,
    place_z_mm: Optional[float] = None,
    approach_clearance_mm: float = APPROACH_CLEARANCE_MM,
) -> UcsGraspPlan:
    """纯函数：构建整轮计划。不产生任何 I/O。

    序列（姿态恒为 ``rotation``，只动 XYZ）::

        复位 -> 抓取位上方 -> ↓抓取位 -> 夹爪合 -> ↑抬升
             -> 放置位上方(高位横移) -> ↓放置位 -> 夹爪开 -> ↑抬升 -> 复位

    ``rotation`` 为初始姿态 3x3（None 表示 dry-run 阶段未知，执行时取复位位置姿态）。

    ``place_z_mm`` 缺省时取抓取点的 Z。这在物体原本就坐在桌面上时是对的：夹爪与
    物体的相对关系全程不变，回到同一高度松爪，物体底面自然坐回桌面（任务书要求
    "放置时需要直立"）。物体原本**不在**桌面上时应显式传值。

    ``approach_clearance_mm`` 是垂直接近/抬升的高度。它决定横移时能越过多高的
    障碍物 —— 这是第二、三档不撞倒其它物品的关键。
    """
    grasp = np.asarray(grasp_xyz_mm, dtype=np.float64).reshape(3)
    place_z = float(grasp[2]) if place_z_mm is None else float(place_z_mm)
    place = np.array([float(place_x_mm), float(place_y_mm), place_z])
    validate_targets(grasp, place)

    clearance = float(approach_clearance_mm)
    if clearance <= 0.0:
        raise ValueError("approach_clearance_mm 必须为正")
    # 抬升后的高度也要落在安全区内；顶到上限就贴着上限走，不要越界。
    grasp_above = grasp + np.array([0.0, 0.0, clearance])
    place_above = place + np.array([0.0, 0.0, clearance])
    ceiling = float(SAFE_Z_MAX_MM)
    grasp_above[2] = min(grasp_above[2], ceiling)
    place_above[2] = min(place_above[2], ceiling)
    if grasp_above[2] <= grasp[2] or place_above[2] <= place[2]:
        raise UcsGraspSafetyError(
            "抓取/放置点已接近安全高度上限 {:.0f}mm，无法留出垂直接近段；"
            "低位横移会撞倒其它物品，拒绝执行。".format(ceiling)
        )

    def pose_with(translation_mm, rotate):
        matrix = np.eye(4, dtype=np.float64)
        if rotate is not None:
            matrix[:3, :3] = np.asarray(rotate, dtype=np.float64)
        matrix[:3, 3] = translation_mm / 1000.0
        return matrix

    def move(point, detail):
        return {"kind": "move", "target": pose_with(point, rotation),
                "xyz_mm": np.asarray(point, dtype=np.float64).copy(),
                "detail": "{} XYZ(mm)={}".format(
                    detail, np.round(point, 2).tolist())}

    steps = (
        {"kind": "reset", "detail": "回到复位位置(安全使能前置)"},
        move(grasp_above, "抓取位上方"),
        move(grasp, "↓ 垂直下降到抓取位"),
        {"kind": "gripper", "open": False, "detail": "夹爪闭合"},
        move(grasp_above, "↑ 垂直抬升(离开物件堆再横移)"),
        move(place_above, "放置位上方(高位横移)"),
        move(place, "↓ 垂直下降到放置位"),
        {"kind": "gripper", "open": True, "detail": "夹爪张开"},
        move(place_above, "↑ 垂直抬升"),
        {"kind": "reset", "detail": "回到复位位置(本轮结束)"},
    )
    description = (
        "复位 -> 上方({h:.0f}mm) -> ↓抓取({g}) -> 夹爪合 -> ↑抬升 -> "
        "高位横移 -> ↓放置({p}) -> 夹爪开 -> ↑抬升 -> 复位".format(
            h=clearance,
            g=np.round(grasp, 2).tolist(),
            p=np.round(place, 2).tolist(),
        )
    )
    return UcsGraspPlan(
        grasp_xyz_mm=grasp,
        place_xyz_mm=place,
        steps=steps,
        description=description,
    )


class UcsGraspRunner:
    """执行器：只用 XYZ 平移，姿态固定为"复位位置的姿态"。

    ``jog`` 为 :class:`competition_pipeline.nexbot_jog.NexBotTcpJog`
    （用户坐标系1）。默认 dry-run，只验证/打印计划，不连接、不运动。
    """

    def __init__(
        self,
        jog,
        place_x_mm: float = UCS_PLACE_X_MM,
        place_y_mm: float = UCS_PLACE_Y_MM,
        speed_scale: float = SPEED_SCALE,
        on_event: Optional[Callable[[str], None]] = None,
    ):
        self.jog = jog
        self.place_x_mm = float(place_x_mm)
        self.place_y_mm = float(place_y_mm)
        self.speed_scale = float(speed_scale)
        self._on_event = on_event or (lambda message: None)

    def _emit(self, message: str):
        print("[ucs-grasp] {}".format(message), flush=True)
        self._on_event(message)

    def _with_retry(self, fn, label, attempts=3, delay_s=0.8):
        """Retry controller calls after transient connection failures.

        只对幂等步骤重试(状态查询/回复位): 断线或服务暂态拒绝时, 适配器已
        关闭连接, 下一次调用自动重连。运动/夹爪指令绝不自动重发(可能已执行)。
        """
        from competition_pipeline.nexbot_tcp import ControllerConnectionError

        last_error = None
        for attempt in range(int(attempts)):
            try:
                return fn()
            except ControllerConnectionError as error:
                last_error = error
                if attempt + 1 < int(attempts):
                    self._emit(
                        "{} 连接失败({}): {}; {:.1f}s 后重试({}/{})……".format(
                            label, type(error).__name__, error,
                            delay_s, attempt + 1, int(attempts),
                        )
                    )
                    time.sleep(delay_s)
        raise ControllerConnectionError(
            "{} 重试 {} 次仍失败: {}".format(label, int(attempts), last_error)
        ) from last_error

    def _ensure_servo_running(self, stage):
        servo = self._with_retry(
            self.jog.controller.servo_status, "{} 伺服状态".format(stage)
        )
        self._emit("{} 伺服 status={}".format(stage, servo))
        if int(servo) != 3:
            self._emit("{} 发送 0x2311 上使能……".format(stage))
            self._with_retry(
                self.jog.controller.enable_servo,
                "{} 伺服上使能".format(stage),
            )

    def execute(self, grasp_xyz_mm, dry_run: bool = True) -> dict:
        """执行整轮抓取-放置。返回结果 dict；失败抛出 UcsGraspExecutorError
        （已尝试急停，机器人不会留在放置位上方）。"""
        grasp = np.asarray(grasp_xyz_mm, dtype=np.float64).reshape(3)
        plan = build_ucs_grasp_plan(
            grasp, rotation=None, place_x_mm=self.place_x_mm,
            place_y_mm=self.place_y_mm,
        )
        if dry_run:
            self._emit("DRY-RUN，不发送运动: {}".format(plan.description))
            for step in plan.steps:
                if step["kind"] == "move":
                    target = step["target"]
                    target_mm = np.round(target[:3, 3] * 1000.0, 2).tolist()
                    self._emit("  -> move XYZ(mm)={}（姿态=复位位置姿态）".format(
                        target_mm))
                else:
                    self._emit("  -> {}".format(step["detail"]))
            return {
                "status": "dry_run",
                "grasp_xyz_mm": plan.grasp_xyz_mm.tolist(),
                "place_xyz_mm": plan.place_xyz_mm.tolist(),
            }

        try:
            self._emit("检查伺服状态……")
            self._ensure_servo_running("回复位前")
            self._emit("回复位位置……")
            self._with_retry(self.jog.go_reset_position, "回复位")

            # Field behaviour: GO_RESET_POSITION finishes with servo
            # 3 -> 0 -> 1.  Its call may return before that transition has
            # settled; enabling immediately was observed to return status=0.
            self._emit("等待回复位状态稳定 1.5s……")
            time.sleep(1.5)
            self._ensure_servo_running("回复位后")

            self._emit("读取初始姿态（复位位置姿态）……")
            state = self._with_retry(
                self.jog.controller.read_state, "读取初始姿态"
            )
            matrix = np.asarray(state.base_from_gripper, dtype=np.float64)
            rotation = matrix[:3, :3].copy()
            initial_xyz_mm = matrix[:3, 3] * 1000.0

            plan = build_ucs_grasp_plan(
                grasp, rotation=rotation, place_x_mm=self.place_x_mm,
                place_y_mm=self.place_y_mm,
            )
            try:
                # 姿态恒为复位点姿态，一次算出弧度，全程复用。
                from competition_pipeline.geometry import (
                    inexbot_abc_from_transform,
                )
                _xyz_m, abc_rad = inexbot_abc_from_transform(
                    plan.steps[1]["target"]
                )
                gripper_unverified = []
                for step in plan.steps[1:-1]:
                    if step["kind"] == "gripper":
                        self._emit(step["detail"])
                        result = self.jog.gripper(bool(step["open"]))
                        # jog.gripper 回读 DOUT：读到相反值会抛错；读不到则返回
                        # ("未回读: …")，累积起来写进结论，不冒充"已确认"。
                        if isinstance(result, tuple) and result[1]:
                            gripper_unverified.append(
                                "{}（{}）".format(step["detail"], result[1])
                            )
                        continue
                    self._emit(step["detail"])
                    # 统一走 move_to_ucs：继承单位闸门/姿态闸门/使能前置/到位校验。
                    # 直调 controller.move_to 会绕过事务锁和全部闸门。
                    self.jog.move_to_ucs(
                        step["xyz_mm"], abc_rad,
                        vel_mm_s=self.speed_scale * 1000.0,
                    )
            except Exception as move_error:
                self._emit("运动失败，紧急停止: {}".format(move_error))
                try:
                    self.jog.emergency_stop()
                except Exception:
                    pass
                raise UcsGraspExecutorError(
                    "抓取执行失败（已急停）: {}".format(move_error)
                ) from move_error

            self._emit("回到复位位置（本轮结束）……")
            self._with_retry(self.jog.go_reset_position, "回复位(收尾)")
        except UcsGraspSafetyError:
            raise
        except UcsGraspExecutorError:
            raise
        except Exception as error:
            # Preparation failures (servo/reset/readback) happen before MOVL;
            # sending emergency stop here latches the controller and makes the
            # next enable attempt fail.  Motion failures are stopped inside
            # the inner move block above.
            self._emit("执行前准备失败: {}".format(error))
            raise UcsGraspExecutorError(
                "抓取执行前准备失败: {}".format(error)
            ) from error

        result = {
            "status": "ok",
            "grasp_xyz_mm": plan.grasp_xyz_mm.tolist(),
            "place_xyz_mm": plan.place_xyz_mm.tolist(),
            "initial_tcp_xyz_mm": [float(v) for v in initial_xyz_mm],
        }
        if gripper_unverified:
            # 运动有 0x3D03 + 到位校验背书，夹爪没有。差别要说出来。
            result["gripper_unverified"] = gripper_unverified
            self._emit("⚠️ 夹爪状态未能回读确认：{}；请目视确认".format(
                "；".join(gripper_unverified)))
        return result


def build_jog(competition_yaml: Path, host: str = "", controller=None):
    """从 competition.yaml 构建比赛 UI 同款 NexBotTcpJog（6001 运动口）。

    复用控制器配置：port_motion=6001, pose_frame=UCS, motion_coord=3。
    """
    from competition_pipeline.nexbot_jog import NexBotTcpJog
    from competition_pipeline.tcp_pose import pose_endpoint_from_config

    data = yaml.safe_load(competition_yaml.read_text(encoding="utf-8"))
    settings = json.loads(json.dumps(data.get("controller", {}) or {}))
    if host:
        settings.setdefault("nexbot_tcp", {})["host"] = str(host)
    # Do not let the adapter inject 0x7266 while 0x4502 is executing.
    settings.setdefault("nexbot_tcp", {})["heartbeat_s"] = 0.0
    if not settings.get("nexbot_tcp"):
        raise RuntimeError("competition.yaml 缺少 controller.nexbot_tcp 配置")
    endpoint = pose_endpoint_from_config(settings)
    jog = NexBotTcpJog(endpoint)
    if controller is not None:
        controller_endpoint = getattr(controller, "endpoint", None)
        if controller_endpoint is not None and (
            str(controller_endpoint.host) != str(endpoint.host)
            or int(controller_endpoint.port_motion) != int(endpoint.port_motion)
            or int(controller_endpoint.port_state) != int(endpoint.port_state)
        ):
            raise RuntimeError("visual controller endpoint differs from grasp endpoint")
        jog._controller = controller
    return jog


__all__ = [
    "UCS_PLACE_X_MM",
    "UCS_PLACE_Y_MM",
    "SAFE_XY_MM",
    "SAFE_Z_MIN_MM",
    "SAFE_Z_MAX_MM",
    "MAX_SINGLE_LEG_MM",
    "SPEED_SCALE",
    "UcsGraspSafetyError",
    "UcsGraspExecutorError",
    "UcsGraspPlan",
    "validate_targets",
    "build_ucs_grasp_plan",
    "UcsGraspRunner",
    "build_jog",
]
