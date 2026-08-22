#!/usr/bin/env python3
"""无硬件跑真 UI：假相机 + 假控制器，界面和交互全是真的。

用来在**没有机械臂、没有 OAK 相机**的情况下，把整个界面点一遍：
拍照识别 → 点画面里的物体 → 计算选中目标 → 看坐标 → 执行抓取 → 看槽位推进。

设计原则：**一个字都不改 ``oak_vision_node.py``**
------------------------------------------------
这是明天要跑竞赛的代码，不能为了能演练而在里面塞 ``if simulate:`` 分支 ——
那种分支会一直留在真实运行路径上。这里的做法是从**外面**替换两个构造入口，
再调用节点原本的 ``main()``：

    build_oak_source(...)                      -> 返回读样本帧的假相机
    nexbot_jog.NexBotTcpRobotController(...)   -> 返回假控制器

被替换掉的只有"硬件从哪来"。界面、检测、位姿估计、抓取几何、闸门、序列、
放置槽位——全部是真的在跑。

TCP 位姿不用假冒：节点本来就支持 ``--tcp-xyz-mm/--tcp-rpy-deg`` 静态指定。

用法::

    ./tool/visual_grasp_pipeline/run_sim_ui.sh              # 走完整执行链路（控制器是假的）
    ./tool/visual_grasp_pipeline/run_sim_ui.sh --dry-run    # 只算坐标不发运动

这里点"执行抓取"**不会**动任何真实硬件（控制器是假的），只是让完整的十步序列跑起来。

⚠️ 用户坐标系的数字在这里是**没有意义**的
--------------------------------------
``user1_from_object = T_user1_tcp @ 手眼 @ T_camera_object``。样本帧是以前用别的
相机拍的静态图，当时机械臂在哪儿没有记录；这里的 ``--tcp-xyz-mm 0 0 300`` 是
**编出来**的。于是 user1 的 XYZ 全部锚在一个虚构的位置上，Z 经常落到桌面以下
而被安全闸门拦掉 —— 那是闸门在正常工作，不是定位算法出错。
本模拟只用来验证**界面/交互/闸门/序列**；坐标准不准只能上实物看。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_static_frame(static_dir, color_size=None):
    """读仓库自带的样本帧，拼成节点期待的 ``FrameBundle``。

    样本是 640x480 RGB + 640x400 uint16 深度(mm)。深度比彩色矮，按 offline.py
    的老做法上对齐补零；单位换成米。
    """
    from tool.object_model_builder.rgbd_geometry import CameraIntrinsics

    static_dir = Path(static_dir).expanduser()
    rgb_path = static_dir / "rgb.png"
    if not rgb_path.is_file():
        raise SystemExit(
            "找不到样本帧 {}。\n"
            "它应该随 fp_release 一起提供；用 --static-frame 指定别的目录。".format(
                rgb_path
            )
        )
    color = cv2.imread(str(rgb_path))
    if color is None:
        raise SystemExit("样本 RGB 读取失败: {}".format(rgb_path))
    height, width = color.shape[:2]

    depth_path = static_dir / "depth.png"
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.is_file() else None
    depth_m = np.zeros((height, width), dtype=np.float32)
    if depth_raw is not None:
        rows = min(depth_raw.shape[0], height)
        columns = min(depth_raw.shape[1], width)
        # uint16 毫米 -> float32 米
        depth_m[:rows, :columns] = (
            depth_raw[:rows, :columns].astype(np.float32) / 1000.0
        )

    k_path = static_dir / "cam_K.txt"
    if k_path.is_file():
        matrix = np.loadtxt(str(k_path)).reshape(3, 3)
    else:
        raise SystemExit("找不到样本内参 {}".format(k_path))
    intrinsics = CameraIntrinsics(width, height, matrix, np.zeros(5))
    return color, depth_m, intrinsics


class FakeOakSource:
    """只回放一帧样本的假相机。接口对齐 ``OakDProSource`` 里节点用到的部分。"""

    def __init__(self, color, depth_m, intrinsics):
        self._color = color
        self._depth_m = depth_m
        self._intrinsics = intrinsics
        self.started = False

    def start(self):
        self.started = True

    def latest(self):
        from tool.object_model_builder.camera_source import FrameBundle

        now = time.time()
        return FrameBundle(
            color_bgr=self._color.copy(),
            color_timestamp_s=now,
            depth_m=self._depth_m.copy(),
            # 同一时刻 -> sync_delta_s = 0，稳过节点的时间差闸门
            depth_timestamp_s=now,
            depth_intrinsics=self._intrinsics,
            ir_image=None,
            ir_timestamp_s=None,
            color_intrinsics=self._intrinsics,
            depth_aligned_to_color=True,
            color_is_rectified=False,
        )

    def close(self):
        self.started = False

    stop = close


def _install_fakes(static_dir, faults):
    """把两个硬件构造入口换成假的。返回一个撤销函数（虽然通常不需要撤销）。"""
    from competition_pipeline.scripts.offline_rehearsal import SimulatedController
    from competition_pipeline import nexbot_jog
    from tool.visual_grasp_pipeline import oak_vision_node

    color, depth_m, intrinsics = _load_static_frame(static_dir)

    original_source = oak_vision_node.build_oak_source
    original_controller = nexbot_jog.NexBotTcpRobotController

    def fake_build_oak_source(_settings):
        print("[sim] 相机 -> 样本帧 {}x{}".format(intrinsics.width, intrinsics.height))
        return FakeOakSource(color, depth_m, intrinsics)

    # 复位/拍摄点的姿态：让机器人"停在桌面上方 300mm、姿态为俯视"。
    # 抓取序列会读它当作全程姿态（本流程只动 XYZ）。
    reset_pose = np.eye(4, dtype=np.float64)
    reset_pose[:3, 3] = [0.0, 0.0, 0.300]

    def fake_controller(_endpoint):
        print("[sim] 控制器 -> 假控制器 faults={}".format(sorted(faults) or "无"))
        return SimulatedController(reset_pose, faults=faults)

    oak_vision_node.build_oak_source = fake_build_oak_source
    nexbot_jog.NexBotTcpRobotController = fake_controller

    def restore():
        oak_vision_node.build_oak_source = original_source
        nexbot_jog.NexBotTcpRobotController = original_controller

    return restore


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--static-frame", default="",
        help="样本帧目录（默认读 visual_grasp_pipeline.yaml 的 paths.static_frame_dir）",
    )
    parser.add_argument(
        "--fault", action="append", default=[],
        choices=("servo-refuse", "motion-rejected", "reset-refused",
                 "gripper-stuck", "gripper-unreadable"),
        help="给假控制器注入故障，可重复。用来预演现场报错长什么样。",
    )
    known, passthrough = parser.parse_known_args(argv)

    static_dir = known.static_frame
    if not static_dir:
        import yaml

        config = REPO_ROOT / "tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml"
        raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        static_dir = (raw.get("paths", {}) or {}).get("static_frame_dir", "")
        if not static_dir:
            raise SystemExit("配置里没有 paths.static_frame_dir，请用 --static-frame 指定")

    _install_fakes(static_dir, set(known.fault))

    from tool.visual_grasp_pipeline import oak_vision_node

    # 节点本来就支持静态 TCP 位姿，不需要假冒位姿读取通道。
    # 给一个"桌面上方 300mm、俯视"的位姿，与假控制器的复位姿态一致。
    arguments = list(passthrough)
    if not any(item.startswith("--tcp-xyz-mm") for item in arguments):
        arguments += ["--tcp-xyz-mm", "0", "0", "300"]
    if not any(item.startswith("--tcp-rpy-deg") for item in arguments):
        arguments += ["--tcp-rpy-deg", "180", "0", "0"]

    print("[sim] 无硬件模式：相机与控制器都是假的，界面/检测/位姿/抓取几何全是真的")
    print("[sim] 传给节点的参数: {}".format(" ".join(arguments) or "(无)"))
    return oak_vision_node.main(arguments)


if __name__ == "__main__":
    sys.exit(main())
