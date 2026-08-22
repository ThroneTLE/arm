#!/usr/bin/env python3
"""切换工具坐标系（法兰 -> 工具手1）后，补偿手眼矩阵与 ``tcp_from_grasp``。

为什么切工具坐标系会破坏定位
----------------------------
视觉链路是::

    user1_from_object = user1_from_tcp @ tcp_from_camera @ camera_from_object
                        └── 控制器回读 ──┘  └── 手眼标定 ──┘

手眼矩阵 ``tcp_from_color_camera`` 是在**默认法兰**下标定的，它描述的是
"相机相对**法兰**"。一旦把工具坐标系标到夹爪尖端，控制器回读的
``user1_from_tcp`` 参考点就移到了尖端，而手眼矩阵还是按法兰标的 ——
两者不再匹配，**所有物体坐标会整体平移一个"法兰->尖端"的量**（本机约 110mm）。

补偿方法
--------
记法兰系 ``F``、新工具系 ``T``、相机 ``C``。已有 ``F_from_C``（旧手眼），
需要 ``T_from_C``::

    user1_from_T = user1_from_F @ F_from_T
    T_from_C     = inv(F_from_T) @ F_from_C

``F_from_T`` 不用查手册 —— **机器人不动**，切换工具前后各回读一次即可::

    F_from_T = inv(user1_from_F) @ user1_from_T

同时 ``grasp_planning.tcp_from_grasp`` 的平移（当前 110mm，含义是"TCP 在抓取点
上方 110mm"）在 TCP 变成尖端后应改为 **0**。

现场步骤（机器人全程**不要移动**）
----------------------------------
1. 保持当前姿态，工具仍是默认法兰，跑::

       python -m competition_pipeline.scripts.retarget_tool_frame --capture-before

2. 在示教器上完成工具手1 标定并**切换为工具1**。**不要移动机器人。**
3. 跑::

       python -m competition_pipeline.scripts.retarget_tool_frame --capture-after

4. 检查打印出的 ``F_from_T`` 是否合理（平移量应≈夹爪长度），然后::

       python -m competition_pipeline.scripts.retarget_tool_frame --apply

   它会更新 ``hand_eye.tcp_from_color_camera`` 与
   ``grasp_planning.tcp_from_grasp``，并把原值备份到 ``config/backups/``。

**验证（必做）**：把一个已知物体放在用户系原点，识别后应得到 X≈0 Y≈0
Z≈物体高度。偏了就说明补偿没做对，**立刻回滚**（备份文件在 config/backups/）。

不想切工具坐标系？那就什么都别做 —— 保持法兰、保持现有手眼与 110mm，
今天验证过的定位精度原样可用。
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "competition_pipeline" / "config" / "competition.yaml"
STATE_PATH = Path("/tmp/nexbot_tool_retarget.json")


def _read_pose(competition_yaml, host=""):
    """回读一次用户坐标系1 的 TCP 位姿（4x4）。只读，不发运动。"""
    import yaml

    from competition_pipeline.geometry import transform_from_inexbot_abc
    from competition_pipeline.tcp_pose import pose_endpoint_from_config
    from competition_pipeline.nexbot_tcp import NexBotTcpRobotController

    data = yaml.safe_load(Path(competition_yaml).read_text(encoding="utf-8"))
    settings = data.get("controller", {}) or {}
    if host:
        settings.setdefault("nexbot_tcp", {})["host"] = str(host)
    controller = NexBotTcpRobotController(pose_endpoint_from_config(settings))
    try:
        state = controller.read_state()
    finally:
        controller.close()
    return np.asarray(state.base_from_gripper, dtype=np.float64)


def _load_state():
    if not STATE_PATH.is_file():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _flange_from_tool(state):
    before = np.asarray(state["before"], dtype=np.float64)
    after = np.asarray(state["after"], dtype=np.float64)
    return np.linalg.inv(before) @ after


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--host", default="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture-before", action="store_true",
                       help="切换工具坐标系**前**回读（法兰）")
    group.add_argument("--capture-after", action="store_true",
                       help="切换工具坐标系**后**回读（工具1）。机器人不能移动过")
    group.add_argument("--apply", action="store_true",
                       help="用两次读数补偿手眼矩阵与 tcp_from_grasp")
    group.add_argument("--show", action="store_true", help="只打印当前推算结果")
    args = parser.parse_args(argv)

    state = _load_state()

    if args.capture_before or args.capture_after:
        key = "before" if args.capture_before else "after"
        try:
            pose = _read_pose(args.competition_config, args.host)
        except Exception as error:                      # noqa: BLE001 - 现场工具
            print("回读失败：{}".format(error), file=sys.stderr)
            return 1
        state[key] = pose.tolist()
        state[key + "_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_state(state)
        print("已记录 {}：XYZ = {} mm".format(
            key, np.round(pose[:3, 3] * 1000.0, 3).tolist()))
        if key == "before":
            print("\n现在去示教器完成工具手1 标定并切换为工具1。"
                  "\n⚠️ 期间**不要移动机器人**，否则这次标定作废。"
                  "\n完成后跑：--capture-after")
        else:
            print("\n接着跑 --show 检查，再跑 --apply 落盘。")
        return 0

    if "before" not in state or "after" not in state:
        print("缺少读数：先跑 --capture-before，切换工具后再跑 --capture-after。",
              file=sys.stderr)
        return 1

    flange_from_tool = _flange_from_tool(state)
    offset_mm = flange_from_tool[:3, 3] * 1000.0
    rotation_changed = not np.allclose(
        flange_from_tool[:3, :3], np.eye(3), atol=1e-6
    )
    print("法兰 -> 工具1 的变换 F_from_T：")
    print("  平移 XYZ = {} mm  (模长 {:.2f} mm)".format(
        np.round(offset_mm, 3).tolist(), float(np.linalg.norm(offset_mm))))
    print("  姿态是否改变: {}".format("是" if rotation_changed else "否（纯平移）"))
    if float(np.linalg.norm(offset_mm)) < 1.0:
        print("\n⚠️ 平移量几乎为 0 —— 两次读数一样，说明工具坐标系没真正切换，"
              "或者切换后机器人被移动过又回到了原位。", file=sys.stderr)
    if float(np.linalg.norm(offset_mm)) > 400.0:
        print("\n⚠️ 平移量 > 400mm，不像夹爪长度 —— 多半是两次读数之间机器人动过了。",
              file=sys.stderr)

    if args.show:
        return 0

    # --apply
    import yaml

    path = Path(args.competition_config)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    hand_eye = data.get("hand_eye", {}).get("tcp_from_color_camera", {})
    matrix = hand_eye.get("matrix") if isinstance(hand_eye, dict) else hand_eye
    if matrix is None:
        print("配置里找不到 hand_eye.tcp_from_color_camera.matrix", file=sys.stderr)
        return 1
    old_hand_eye = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    new_hand_eye = np.linalg.inv(flange_from_tool) @ old_hand_eye

    backups = path.parent / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = backups / "competition_before_tool_retarget_{}.yaml".format(stamp)
    shutil.copy2(path, backup)

    hand_eye["matrix"] = new_hand_eye.tolist()
    if isinstance(hand_eye, dict):
        hand_eye["note"] = (
            "{} 由 retarget_tool_frame.py 从法兰系补偿到工具手1；"
            "F_from_T 平移 {} mm".format(stamp, np.round(offset_mm, 2).tolist())
        )
    planning = data.setdefault("grasp_planning", {})
    tcp_from_grasp = planning.get("tcp_from_grasp", {})
    grasp_matrix = np.asarray(
        tcp_from_grasp["matrix"], dtype=np.float64
    ).reshape(4, 4)
    old_z_mm = float(grasp_matrix[2, 3] * 1000.0)
    grasp_matrix[:3, 3] = 0.0        # TCP 现在就在工具尖端，不再有额外偏移
    tcp_from_grasp["matrix"] = grasp_matrix.tolist()
    tcp_from_grasp["description"] = (
        "p_tcp = T_tcp_grasp * p_grasp；TCP 已标定到工具尖端(工具手1)，"
        "故平移为 0（切换前是 {:.0f}mm，对应 TCP 在法兰）".format(old_z_mm)
    )
    gripper = data.setdefault("gripper_geometry", {})
    gripper["tcp_to_fingertip_mm"] = 0.0

    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print("\n已更新 {}".format(path))
    print("  hand_eye.tcp_from_color_camera 平移: {} -> {} mm".format(
        np.round(old_hand_eye[:3, 3] * 1000.0, 2).tolist(),
        np.round(new_hand_eye[:3, 3] * 1000.0, 2).tolist()))
    print("  grasp_planning.tcp_from_grasp 平移 Z: {:.0f} -> 0 mm".format(old_z_mm))
    print("  gripper_geometry.tcp_to_fingertip_mm -> 0.0")
    print("  备份: {}".format(backup))
    print("\n⚠️ 必做验证：把已知物体放在用户系原点识别一次，"
          "应得 X≈0 Y≈0 Z≈物体高度。偏了就用备份回滚。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
