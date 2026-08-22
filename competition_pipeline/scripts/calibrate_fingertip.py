#!/usr/bin/env python3
"""指尖高度标定：判定 TCP 到底在不在工具尖端，并推出正确的 ``tcp_from_grasp``。

要解决的问题
------------
``competition.yaml`` 的 ``grasp_planning.tcp_from_grasp`` 平移是 ``(0, 0, 0.11)``。
代码里 ``user1_from_tcp_grasp = user1_from_grasp @ inv(tcp_from_grasp)``，
所以这 110mm 的含义是 **TCP 位于抓取点上方 110mm**。这个数只有在"工具坐标系没标定、
TCP 还在法兰上"时才成立。

**如果你把工具坐标系标定到了夹爪尖端（TCP = 工具尖端），这 110mm 就必须改成 0**，
否则指尖会悬在物体上方 110mm 处合爪 —— 抓空。

判定原理（一次读数同时回答两个问题）
------------------------------------
用户坐标系1 的 **z=0 就是桌面**（``docs/竞赛视觉引导方案-20260822.md``：
"柠檬放用户1原点 -> 期望 Z≈柠檬高度"）。把指尖降到**刚接触桌面**再回读 TCP 的 Z：

    读到 Z ≈ 0     -> TCP 就在工具尖端 -> tcp_from_grasp 的 Z 应为 **0**
    读到 Z ≈ 110   -> TCP 在指尖上方 110mm -> 当前配置正确，保持 0.11
    读到别的值 h   -> TCP 在指尖上方 h mm -> tcp_from_grasp 的 Z 应为 **h/1000**

也就是说：**读数 = tcp_from_grasp 的 Z（毫米）**。这条等式就是标定结果。

现场步骤（1 分钟，零风险，全程手动点动）
----------------------------------------
1. 示教器切**示教模式**（远程模式下复位点安全闸门会拒绝一切运动）。
2. 先完成工具坐标系标定（若打算把 TCP 标到尖端）。
3. 手动点动，把夹爪**张开**，让两根手指的**尖端**刚好碰到桌面 —— 以"能感觉到
   接触但不压弯"为准，纸片能勉强抽动即可。
4. 跑本脚本（只读，不发任何运动指令）。
5. 按提示更新 ``gripper_geometry.tcp_to_fingertip_mm`` 与
   ``grasp_planning.tcp_from_grasp`` 的 Z。两者必须一致，否则抓取会系统性偏移。

用法::

    python -m competition_pipeline.scripts.calibrate_fingertip
    python -m competition_pipeline.scripts.calibrate_fingertip --write   # 直接写回两处

本脚本**只读位姿**，不会发送任何运动指令。
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "competition_pipeline" / "config" / "competition.yaml"


def read_tcp_z_mm(competition_yaml, host=""):
    """回读一次用户坐标系1 的 TCP 位姿，返回 (xyz_mm, abc_deg)。只读。"""
    import numpy as np
    import yaml

    from competition_pipeline.geometry import inexbot_abc_from_transform
    from competition_pipeline.tcp_pose import pose_endpoint_from_config
    from competition_pipeline.nexbot_tcp import NexBotTcpRobotController

    data = yaml.safe_load(Path(competition_yaml).read_text(encoding="utf-8"))
    settings = data.get("controller", {}) or {}
    if host:
        settings.setdefault("nexbot_tcp", {})["host"] = str(host)
    endpoint = pose_endpoint_from_config(settings)
    controller = NexBotTcpRobotController(endpoint)
    try:
        state = controller.read_state()
    finally:
        controller.close()
    xyz_m, abc_rad = inexbot_abc_from_transform(state.base_from_gripper)
    return tuple(xyz_m * 1000.0), tuple(np.degrees(abc_rad))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--host", default="", help="覆盖控制器 IP")
    parser.add_argument(
        "--write", action="store_true",
        help="把读到的 Z 写回 gripper_geometry.tcp_to_fingertip_mm",
    )
    args = parser.parse_args(argv)

    print(__doc__.split("现场步骤")[1].split("用法")[0].strip())
    print()
    print("确认指尖已经接触桌面后按回车回读（Ctrl-C 取消）…", end="")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消")
        return 1

    try:
        xyz_mm, abc_deg = read_tcp_z_mm(args.competition_config, args.host)
    except Exception as error:                          # noqa: BLE001 - 现场工具
        print("回读失败：{}".format(error), file=sys.stderr)
        print("检查：控制器是否上电、7000 端口是否被别的程序占用（单客户端）。",
              file=sys.stderr)
        return 1

    z_mm = float(xyz_mm[2])
    print()
    print("用户坐标系1 TCP: X={:.2f} Y={:.2f} Z={:.2f} mm".format(*xyz_mm))
    print("               A={:.3f} B={:.3f} C={:.3f} °".format(*abc_deg))
    print()
    print("==> tcp_to_fingertip_mm = {:.2f}".format(z_mm))

    # ⚠️ 读到 ≈0 是**工具手1 已标定到尖端**时的正确结果，不是故障。
    # 只有明显为负(指尖跑到桌面下方)或大得离谱才是真出错了。
    if z_mm < -5.0:
        print("\n❌ Z={:.1f}mm，指尖不可能在桌面下方 5mm 还没压弯。"
              "确认用户坐标系1 的原点确实设在**桌面**上，且回读的是用户系"
              "(pose_frame=UCS)而不是基座系。".format(z_mm), file=sys.stderr)
        return 1
    if z_mm > 300.0:
        print("\n⚠️ Z={:.1f}mm 远大于任何夹爪长度 —— 指尖多半没真的碰到桌面，"
              "或者碰到的是夹爪其它部位。".format(z_mm), file=sys.stderr)

    if abs(z_mm) <= 5.0:
        print("\n判定：**TCP 就在工具尖端**（工具手1 已标定）。")
        print("     tcp_from_grasp 的平移应为 0。")
        print("     注意 visual_grasp_bridge 那条路径当前带 110mm 偏移，"
              "不改的话它会把 TCP 放到物体上方 110mm —— 抓空。")
        print("     （oak_vision_node -> ucs_grasp 那条路径本来就把 TCP 直接送到"
              "抓取点，无偏移，与 TCP=尖端 自洽。）")
    else:
        print("\n判定：TCP 在指尖上方 {:.1f}mm（多半还是默认法兰，工具手1 未生效）。"
              .format(z_mm))
        print("     tcp_from_grasp 的 Z 应为 {:.4f} m。".format(z_mm / 1000.0))
        print("     ⚠️ 但 oak_vision_node -> ucs_grasp 把 TCP 直接送到抓取点，"
              "不套这个偏移 —— 那条路径会**低 {:.0f}mm**。".format(z_mm))
        print("     两条路径要一致，请确认工具手1 是否真的切过去了。")

    if args.write:
        written = _write_back(Path(args.competition_config), z_mm)
        if written is None:
            return 1
        print("\n已写回 {}".format(args.competition_config))
        for line in written:
            print("  {}".format(line))
        print("\n跑一次 check_vision_assets 确认一致：")
        print("  python -m competition_pipeline.scripts.check_vision_assets")
    else:
        print("\n加 --write 自动写回 gripper_geometry.tcp_to_fingertip_mm 与 "
              "grasp_planning.tcp_from_grasp 的 Z（两者必须一致）。")
    return 0


def _write_back(path, z_mm):
    """同时更新 ``tcp_to_fingertip_mm`` 与 ``tcp_from_grasp`` 的 Z。

    两者描述的是**同一个距离**：TCP 到抓取参考点(指尖平面)的偏移。分开填就迟早
    对不上，而对不上的后果是抓取系统性偏移那个差值 —— 所以这里一次写两处。
    ``tcp_from_grasp`` 的**旋转部分原样保留**：那是抓取系与 TCP 系的轴向约定，
    与本次标定无关。
    """
    import numpy as np
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    gripper = data.setdefault("gripper_geometry", {})
    gripper["tcp_to_fingertip_mm"] = round(z_mm, 2)

    planning = data.setdefault("grasp_planning", {})
    entry = planning.get("tcp_from_grasp")
    if not isinstance(entry, dict) or "matrix" not in entry:
        print("配置里找不到 grasp_planning.tcp_from_grasp.matrix", file=sys.stderr)
        return None
    matrix = np.asarray(entry["matrix"], dtype=float).reshape(4, 4)
    old_z_mm = float(matrix[2, 3] * 1000.0)
    matrix[0, 3] = 0.0
    matrix[1, 3] = 0.0
    matrix[2, 3] = z_mm / 1000.0
    entry["matrix"] = matrix.tolist()
    entry["description"] = (
        "p_tcp = T_tcp_grasp * p_grasp；平移 = TCP 到指尖平面的距离，"
        "由 calibrate_fingertip.py 实测得到（指尖触桌读用户系 Z，因为 z=0 即桌面）"
    )

    backups = path.parent / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    backup = backups / "competition_before_fingertip_{}.yaml".format(
        time.strftime("%Y%m%d_%H%M%S")
    )
    shutil.copy2(path, backup)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return [
        "gripper_geometry.tcp_to_fingertip_mm = {:.2f}".format(z_mm),
        "grasp_planning.tcp_from_grasp 平移 Z: {:.1f} -> {:.1f} mm".format(
            old_z_mm, z_mm),
        "备份: {}".format(backup),
    ]


if __name__ == "__main__":
    sys.exit(main())
