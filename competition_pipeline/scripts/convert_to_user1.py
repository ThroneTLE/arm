#!/usr/bin/env python3
"""Convert base-frame poses to user-frame-1 poses for the competition pipeline.

Field constants (measured 2026-08-22, MOKA MR07S-930 / Inexbot C1102):
the TCP(工具1) stopped at the user-frame-1 origin reads, in the ROBOT BASE frame,
``realPosPCS = (578.3, -79.3, 302.3) mm, A/B/C = (174.64, -4.47, -174.43) deg``.
That pose IS ``T_base_user1`` (the tool frame coincides with the user frame at
the calibrated origin), so:

    p_user1   = inv(T_base_user1) @ p_base
    T_user1   = inv(T_base_user1) @ T_base

Anything frame-independent (camera intrinsics, RGBD calibration, the hand-eye
matrix ``tcp_from_color_camera``) does NOT need conversion -- rigid relative
geometry is the same in every reference frame.

Usage:
    python3 scripts/convert_to_user1.py            # convert config/tag_map (dry run print)
    python3 scripts/convert_to_user1.py --write    # also write config/tag_map_user1.yaml
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from competition_pipeline.configuration import load_camera_intrinsics  # noqa: E402
from competition_pipeline.geometry import (  # noqa: E402
    transform_from_inexbot_abc,
    transform_from_xyz_rpy,
    xyz_rpy_from_transform,
)

# (mm, deg) measured at the user-1 origin in the robot BASE frame (PCS, tool 1)
USER1_ORIGIN_BASE_XYZ_MM = (578.3, -79.3, 302.3)
USER1_ORIGIN_BASE_ABC_DEG = (174.64, -4.47, -174.43)


def base_from_user1():
    # The measured A/B/C are the CONTROLLER convention (intrinsic X'Y'Z' ->
    # R = Rx(A) Ry(B) Rz(C)); use the inexbot helper for the base transform.
    return transform_from_inexbot_abc(
        np.asarray(USER1_ORIGIN_BASE_XYZ_MM) / 1000.0,
        np.radians(USER1_ORIGIN_BASE_ABC_DEG),
    )


def transform_to_user1(base):
    """Return ``T_user1_child = inv(T_base_user1) @ child``."""
    user1_from_base = np.linalg.inv(base_from_user1())
    return user1_from_base @ np.asarray(base, dtype=np.float64).reshape(4, 4)


def convert_pose_to_user1(xyz_mm, rpy_deg):
    child = transform_from_xyz_rpy(
        np.asarray(xyz_mm, dtype=np.float64) / 1000.0, rpy_deg
    )
    converted = transform_to_user1(child)
    xyz_m, rpy_d = xyz_rpy_from_transform(converted)
    return np.round(xyz_m * 1000.0, 3), np.round(rpy_d, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write config/tag_map_user1.yaml")
    args = parser.parse_args()

    config_path = Path("competition_pipeline/config/competition.yaml")
    data = yaml.safe_load(open(config_path, encoding="utf-8"))
    tag_map = data.get("tag_map", {})
    tags = tag_map.get("tags", {})
    if not tags:
        print("tag_map.tags is empty; nothing to convert (add real tag poses first).")
        return 1

    converted = {"tags": {}}
    for tag_id, entry in tags.items():
        xyz, rpy = convert_pose_to_user1(
            entry.get("bottom_right_xyz_mm", [0.0, 0.0, 0.0]),
            entry.get("base_from_tag_rpy_deg", [0.0, 0.0, 0.0]),
        )
        converted["tags"][tag_id] = {
            "bottom_right_xyz_mm": xyz.tolist(),
            "base_from_tag_rpy_deg": rpy.tolist(),
        }
        print(
            "tag {}: base {} -> user1 {} / rpy {}".format(
                tag_id,
                entry.get("bottom_right_xyz_mm"),
                xyz.tolist(),
                rpy.tolist(),
            )
        )

    if args.write:
        out = Path("competition_pipeline/config/tag_map_user1.yaml")
        header = (
            "# 坐标系迁移结果：tag 世界坐标从【机器人基座系】转换到【用户坐标系1】\n"
            "# 转换基准 T_base_user1 = PCS 读数 (578.3, -79.3, 302.3)mm / "
            "ABC (174.64, -4.47, -174.43)deg（2026-08-22 现场实测，TCP/工具1 停在用户1零点）\n"
            "# 注意：仅在原坐标确实是【基座系实测】时使用；竞赛期间不要改动用户1坐标系/工具1，\n"
            "# 否则全局锚点全部偏移，需要重测。\n"
        )
        out.write_text(header + yaml.safe_dump(converted, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
