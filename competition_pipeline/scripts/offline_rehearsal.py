#!/usr/bin/env python3
"""无硬件全栈演练：不接机械臂、不接相机，把整条抓取链路跑一遍。

这个脚本**完全独立**：只读配置、不写任何文件、不改任何运行时代码。跑完删掉也行，
对正常程序零影响。

它假冒的只有**最底层的控制器**，上面跑的是真货：

    真 UcsGraspRunner  ->  真 NexBotTcpJog  ->  [假控制器]
        真计划序列          真三道闸门/使能前置/到位校验/夹爪回读

所以下面这些都会被真正执行到，而不是被 mock 掉：

- 顶点与中心的中点 / 腔体深度钳位（competition_pipeline.grasp_geometry）
- 垂直进 → 高位横移 → 垂直出 的十步序列
- move_to_ucs 的单位闸门、姿态闸门、位移闸门
- 每段运动前的伺服使能前置
- 夹爪 DOUT 回读校验
- 放置槽位推进与"已放置区域"排除

用法::

    # 只演练运动链路（**不需要 GPU / 相机 / 机械臂**，几秒钟）
    python -m competition_pipeline.scripts.offline_rehearsal

    # 指定物体与位置
    python -m competition_pipeline.scripts.offline_rehearsal \\
        --object sprite --at 120 -80

    # 连抓 3 个，看槽位怎么推进
    python -m competition_pipeline.scripts.offline_rehearsal --rounds 3

    # 故意注入故障，**提前看清现场会报什么错**
    python -m competition_pipeline.scripts.offline_rehearsal --fault servo-refuse
    python -m competition_pipeline.scripts.offline_rehearsal --fault gripper-stuck
    python -m competition_pipeline.scripts.offline_rehearsal --fault motion-rejected

    # 附带跑一遍视觉（需要 GPU + FoundationPose，用仓库自带的 static_frame 样本）
    python -m competition_pipeline.scripts.offline_rehearsal --with-vision
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPETITION = REPO_ROOT / "competition_pipeline" / "config" / "competition.yaml"
DEFAULT_VISUAL = (
    REPO_ROOT / "tool" / "visual_grasp_pipeline" / "config" / "visual_grasp_pipeline.yaml"
)


class SimulatedController:
    """假冒 ``NexBotTcpRobotController``，行为按现场实测的真控制器来。

    重点是**像真的一样会拒绝**：真控制器在远程模式下会拒绝运动并顺手把伺服下电，
    所以这里的 ``servo_refuse`` 故障复现的是那个行为，而不是简单抛个异常。
    """

    def __init__(self, start_pose, faults=()):
        self.pose = np.asarray(start_pose, dtype=np.float64).copy()
        self.faults = set(faults or ())
        self.calls = []
        self.servo = 3
        self.dout = [0] * 14 + [1, 0]        # (15,16)=(1,0) 夹爪闭合
        self.motion = SimpleNamespace(close=lambda: None)

    # -- 伺服 ----------------------------------------------------------
    def servo_status(self):
        self.calls.append("servo_status")
        return self.servo

    def enable_servo(self):
        self.calls.append("enable_servo")
        if "servo-refuse" in self.faults:
            # 复位点安全闸门拒绝 -> 控制器下电 -> 伺服回到"就绪"
            self.servo = 1
            raise RuntimeError(
                "0x2311 上使能后伺服仍为 status=1 (需要 3=运行)。"
                "现场实测主因: 复位点安全闸门在远程模式下拒绝每条指令并下电。"
            )
        self.servo = 3
        return 3

    # -- 位姿 ----------------------------------------------------------
    def read_state(self):
        self.calls.append("read_state")
        return SimpleNamespace(base_from_gripper=self.pose.copy())

    def move_to(self, target, speed_scale=0.1):
        self.calls.append(("move_to", np.asarray(target)[:3, 3] * 1000.0))
        if "motion-rejected" in self.faults:
            self.servo = 1
            raise RuntimeError(
                "运动指令被控制器拒绝并已下电(伺服 status=1 != 3)。"
                "现场实测主因: 复位点安全闸门 deviation=null。"
            )
        self.pose = np.asarray(target, dtype=np.float64).copy()

    def go_reset_position(self):
        self.calls.append("go_reset_position")
        if "reset-refused" in self.faults:
            self.servo = 1
            raise RuntimeError("回复位被安全闸门拒绝并已下电")
        # 真控制器实测: GO_RESET_POSITION 结束时伺服会落到 1(就绪)
        self.servo = 1

    def go_home(self):
        self.calls.append("go_home")

    # -- 夹爪 ----------------------------------------------------------
    def set_digital_output(self, port, value):
        self.calls.append(("dout", int(port), int(value)))
        if "gripper-stuck" in self.faults:
            return                       # 线圈不动作：写了但状态不变
        if 1 <= int(port) <= len(self.dout):
            self.dout[int(port) - 1] = int(value)

    def digital_output_states(self):
        self.calls.append("dout_query")
        if "gripper-unreadable" in self.faults:
            raise RuntimeError("DOUT 查询超时")
        return list(self.dout)

    def stop(self):
        self.calls.append("stop")
        self.servo = 0

    def close(self):
        pass


def _pose(xyz_mm, rotation=None):
    matrix = np.eye(4, dtype=np.float64)
    if rotation is not None:
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(xyz_mm, dtype=np.float64) / 1000.0
    return matrix


def _upright_pose(bounds, xy_mm):
    """把物体**立起来**摆到桌面上：最长轴转到用户系 +Z，包围盒底面落在 z=0。

    各网格的长轴方向并不统一（罐子 Z、苹果 Y、柠檬 X、两个新瓶 Y），用单位旋转
    摆会把可乐瓶横躺放着 —— 那样演练出来的高度/直径是反的。真实链路里姿态由
    FoundationPose 给出，这里只是把"直立"这个前提补上。
    """
    lower, upper = np.asarray(bounds, dtype=np.float64)
    long_axis = int(np.argmax(upper - lower))
    if long_axis == 0:                      # X -> Z
        rotation = np.asarray([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    elif long_axis == 1:                    # Y -> Z
        rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    else:
        rotation = np.eye(3)
    corners = np.asarray([[x, y, z]
                          for x in (lower[0], upper[0])
                          for y in (lower[1], upper[1])
                          for z in (lower[2], upper[2])])
    rotated = corners @ rotation.T
    centre = (rotated.min(axis=0) + rotated.max(axis=0)) / 2.0
    translation_mm = np.asarray([
        float(xy_mm[0]) - centre[0] * 1000.0,
        float(xy_mm[1]) - centre[1] * 1000.0,
        -rotated[:, 2].min() * 1000.0,
    ])
    return _pose(translation_mm, rotation)


def _object_bounds(visual_config, object_key):
    """从视觉配置读该物体的米制包围盒（已应用缩放）。"""
    import trimesh

    raw = yaml.safe_load(Path(visual_config).read_text(encoding="utf-8")) or {}
    models = raw.get("object_models", {}) or {}
    scales = raw.get("object_model_scales", {}) or {}
    rules = raw.get("grasp_rules", {}) or {}
    path = models.get(object_key)
    if not path or not Path(path).is_file():
        raise SystemExit(
            "物体 {!r} 没有可用 CAD（object_models 里是 {!r}）。"
            "可选: {}".format(object_key, path, sorted(models))
        )
    scale = float(scales.get(object_key, raw.get("pipeline", {}).get(
        "mesh_scale_to_meters", 1.0)))
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    bounds = np.asarray(mesh.bounds, dtype=np.float64) * scale
    grasp_type = str((rules.get(object_key) or {}).get("type", "cylinder"))
    return bounds, grasp_type


def rehearse_motion(args):
    from competition_pipeline.grasp_geometry import (
        grasp_height_mm, object_extent_user1,
    )
    from competition_pipeline.nexbot_jog import NexBotTcpJog
    from competition_pipeline.place_layout import (
        is_in_placed_region, layout_from_config,
    )
    from tool.visual_grasp_pipeline.ucs_grasp import UcsGraspRunner

    competition = yaml.safe_load(
        Path(args.competition_config).read_text(encoding="utf-8")
    ) or {}
    workspace = competition.get("workspace", {}) or {}
    gripper = competition.get("gripper_geometry", {}) or {}
    layout = layout_from_config(workspace)

    bounds, grasp_type = _object_bounds(args.visual_config, args.object)
    extents_mm = (bounds[1] - bounds[0]) * 1000.0
    height_mm = float(np.max(extents_mm))

    print("=" * 72)
    print("无硬件全栈演练  |  物体 {}  形状类 {}".format(args.object, grasp_type))
    print("=" * 72)
    print("CAD 尺寸(缩放后): {} mm".format(np.round(extents_mm, 1).tolist()))
    if args.fault:
        print("⚠️ 注入故障: {}".format(args.fault))
    print()

    occupied = []
    ok_rounds = 0
    for round_index in range(int(args.rounds)):
        # 物体**立在**桌面上：最长轴朝 +Z，包围盒底面 z=0（见 _upright_pose）
        object_pose = _upright_pose(bounds, args.at)

        extent = object_extent_user1(object_pose, bounds, grasp_type)
        height = grasp_height_mm(
            extent, grasp_type,
            jaw_cavity_depth_mm=float(gripper.get("jaw_cavity_depth_mm", 80.0)),
            safety_clearance_mm=float(gripper.get("safety_clearance_mm", 15.0)),
        )
        grasp_xyz = [extent.center_xy_mm[0], extent.center_xy_mm[1], height.z_mm]

        print("── 第 {} 轮 ─────────────────────────────".format(round_index + 1))
        print("物体顶面 {:.1f}mm  高 {:.1f}mm  夹持宽 {:.1f}mm".format(
            extent.z_top_mm, extent.height_mm, extent.grasp_width_mm))
        print("抓取规则 {}  ->  抓取点 Z={:.1f}mm  伸进夹爪 {:.1f}mm{}".format(
            height.rule, height.z_mm, height.engage_mm,
            "  ⚠️已按腔体钳位(原需 {:.1f})".format(height.requested_engage_mm)
            if height.clamped else ""))

        if is_in_placed_region(grasp_xyz[:2], layout, occupied):
            print("⛔ 该位置在已放置区域内，跳过（这正是防止把刚放好的又抓回来）")
            break
        try:
            place_xy = layout.slot_xy_mm(len(occupied))
        except IndexError as error:
            print("⛔ {}".format(error))
            break
        print("放置槽位 {} -> ({:.0f}, {:.0f})".format(
            len(occupied), place_xy[0], place_xy[1]))

        controller = SimulatedController(
            _pose([0.0, 0.0, 300.0]), faults=[args.fault] if args.fault else []
        )
        patcher = patch(
            "competition_pipeline.nexbot_jog.NexBotTcpRobotController",
            lambda _endpoint: controller,
        )
        patcher.start()
        try:
            jog = NexBotTcpJog(object())
            jog.RETRY_WAIT_S = 0.0
            # 不传 on_event：runner 自己已经会打印 [ucs-grasp] 行，
            # 再挂一个回调会把每条消息印两遍。
            runner = UcsGraspRunner(
                jog, place_x_mm=place_xy[0], place_y_mm=place_xy[1],
            )
            result = runner.execute(np.asarray(grasp_xyz), dry_run=False)
        except Exception as error:
            print("\n❌ 执行失败（{}）：{}".format(type(error).__name__, error))
            print("\n   —— 这就是现场会看到的报错原文 ——")
            break
        finally:
            try:
                jog.close()
            except Exception:
                pass
            patcher.stop()

        moves = [call for call in controller.calls if isinstance(call, tuple)
                 and call[0] == "move_to"]
        print("   实际下发 {} 段 MOVL：".format(len(moves)))
        for step, (_kind, xyz) in enumerate(moves, 1):
            print("     {}. ({:7.1f}, {:7.1f}, {:7.1f}) mm".format(step, *xyz))
        if result.get("gripper_unverified"):
            print("   ⚠️ 夹爪未确认: {}".format(result["gripper_unverified"]))
        print("   ✅ 本轮完成，放置 {}".format(
            np.round(result["place_xyz_mm"], 1).tolist()))
        occupied.append(len(occupied))
        ok_rounds += 1
        print()

    print("=" * 72)
    print("完成 {}/{} 轮；占用槽位 {}".format(ok_rounds, args.rounds, occupied))
    return 0 if ok_rounds or args.fault else 1


def rehearse_vision(args):
    """用仓库自带的 static_frame 样本跑一遍识别（需要 GPU + FoundationPose）。"""
    import cv2

    raw = yaml.safe_load(Path(args.visual_config).read_text(encoding="utf-8")) or {}
    static_dir = Path((raw.get("paths", {}) or {}).get("static_frame_dir", ""))
    rgb_path = static_dir / "rgb.png"
    if not rgb_path.is_file():
        print("找不到样本帧 {} —— 跳过视觉演练".format(rgb_path), file=sys.stderr)
        return 1
    print("\n" + "=" * 72)
    print("视觉演练  |  样本帧 {}".format(static_dir))
    print("=" * 72)
    rgb = cv2.imread(str(rgb_path))
    depth = cv2.imread(str(static_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    print("RGB {}  深度 {} ({}~{} mm)".format(
        rgb.shape, None if depth is None else depth.shape,
        None if depth is None else int(depth.min()),
        None if depth is None else int(depth.max())))
    try:
        from ultralytics import YOLO

        from tool.visual_grasp_pipeline.config import VisualGraspConfig
        from tool.visual_grasp_pipeline.detection import detect_all_objects

        config = VisualGraspConfig.from_yaml(args.visual_config)
        model = YOLO(config.yolo_weights)
        objects = detect_all_objects(
            rgb, model, conf=config.yolo_conf, imgsz=config.yolo_imgsz
        )
    except Exception as error:                       # noqa: BLE001 - 现场工具
        print("视觉链路不可用（{}）：{}".format(type(error).__name__, error))
        print("这不影响上面的运动演练 —— 运动链路不需要 GPU。")
        return 1
    print("检测到 {} 个物体：".format(len(objects)))
    for item in objects:
        print("  {:<22} conf={:.3f}  框={}  掩膜={}".format(
            "{} #{}".format(item["name"], item.get("id")),
            item["conf"], item["xyxy"], item.get("mask") is not None))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--competition-config", default=str(DEFAULT_COMPETITION))
    parser.add_argument("--visual-config", default=str(DEFAULT_VISUAL))
    parser.add_argument("--object", default="sprite",
                        help="用哪个物体的 CAD（默认 sprite = 雪碧罐，已尺子核对）")
    parser.add_argument("--at", nargs=2, type=float, default=(120.0, -80.0),
                        metavar=("X", "Y"), help="物体在用户系的 XY(mm)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="连抓几轮（看放置槽位怎么推进）")
    parser.add_argument(
        "--fault",
        choices=("servo-refuse", "motion-rejected", "reset-refused",
                 "gripper-stuck", "gripper-unreadable"),
        help="注入故障，提前看清现场会报什么错",
    )
    parser.add_argument("--with-vision", action="store_true",
                        help="附带跑一遍识别（需要 GPU + FoundationPose）")
    parser.add_argument("--vision-only", action="store_true")
    args = parser.parse_args(argv)

    status = 0
    if not args.vision_only:
        status |= rehearse_motion(args)
    if args.with_vision or args.vision_only:
        rehearse_vision(args)
    return status


if __name__ == "__main__":
    sys.exit(main())
