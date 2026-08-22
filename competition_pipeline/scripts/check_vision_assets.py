#!/usr/bin/env python3
"""赛前自检：权重类别、网格文件、缩放、抓取规则是否配齐且自洽。

为什么需要它
------------
2026-08-22 换 ``merged_best.pt`` 时发现类别名悄悄变了：

    旧 yolo_model.pt : 3 red_can  4 green_can  6 black_cola_bottle
    新 merged_best.pt: 3 green_can 4 red_can   6 black_red_cola_bottle

``config.resolve_object_key`` 按**名字**匹配，所以红绿罐索引互换是安全的；但 6 号改名
后 ``yolo_to_object`` 里没有对应项，会**静默**落到 ``default_object``(sprite)，
把可乐瓶当雪碧罐去估位姿。这种错不会报异常，只会让抓取莫名其妙地偏。

本脚本把这类"静默降级"变成显式失败。跑一次 5 秒，明天开场值得跑。

用法::

    python -m competition_pipeline.scripts.check_vision_assets \\
        [--visual-config tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml]
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml"
)
DEFAULT_COMPETITION = REPO_ROOT / "competition_pipeline/config/competition.yaml"

#: 顶抓可用的形状类。缺失或写错会让抓取高度规则退化成"对准中心"。
KNOWN_GRASP_TYPES = ("cylinder", "sphere", "elongated")


def _load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _weight_class_names(weights_path):
    """返回 {index: name}；读不出来时返回 None（不阻塞其余检查）。"""
    try:
        import torch
    except ImportError:
        return None
    try:
        checkpoint = torch.load(str(weights_path), map_location="cpu",
                                weights_only=False)
    except Exception as error:                      # noqa: BLE001 - 诊断脚本
        print("  ⚠️ 无法读取权重: {}".format(error))
        return None
    model = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    names = getattr(model, "names", None)
    if names is None and isinstance(checkpoint, dict):
        names = checkpoint.get("names")
    return dict(names) if names else None


def check(config_path, competition_config):
    problems = []
    notes = []
    raw = _load_yaml(config_path)
    paths = raw.get("paths", {}) or {}
    models = raw.get("object_models", {}) or {}
    scales = raw.get("object_model_scales", {}) or {}
    mapping = raw.get("yolo_to_object", {}) or {}
    rules = raw.get("grasp_rules", {}) or {}
    pipeline = raw.get("pipeline", {}) or {}
    default_object = pipeline.get("default_object")

    print("配置: {}".format(config_path))

    # 1) 权重存在且每个类别名都有映射
    weights = paths.get("yolo_weights", "")
    print("\n[1] YOLO 权重")
    if not weights or not Path(weights).is_file():
        problems.append("yolo_weights 不存在: {!r}".format(weights))
        print("  ❌ 不存在: {}".format(weights))
    else:
        print("  ✓ {}".format(weights))
        names = _weight_class_names(weights)
        if names is None:
            problems.append(
                "未能读出权重类别名 —— 映射校验被跳过了。这道检查专防"
                "『类别改名后静默落到 default_object』(merged_best.pt 就改过名)。"
                "请确认用的是 foundationpose 环境(torch 可用)。"
            )
            print("  ❌ 未能读出类别名(torch 不可用或格式不符) —— 映射无法校验")
        else:
            print("  类别: {}".format(names))
            for index, name in sorted(names.items()):
                if str(name) not in mapping:
                    problems.append(
                        "权重类别 {} '{}' 在 yolo_to_object 里没有映射，"
                        "会**静默**落到 default_object={!r}".format(
                            index, name, default_object)
                    )
                    print("  ❌ {} '{}' 未映射 -> 会静默当成 {}".format(
                        index, name, default_object))
                else:
                    print("  ✓ {} '{}' -> {}".format(index, name, mapping[str(name)]))

    # 2) 映射目标都有网格文件
    print("\n[2] 映射目标的 CAD 网格")
    for label, object_key in sorted(mapping.items()):
        mesh = models.get(object_key, "")
        if not mesh:
            problems.append(
                "{} -> {} 没有配 object_models 条目（无网格就拿不到物体高度，"
                "抓取高度后处理对它失效）".format(label, object_key))
            print("  ❌ {} -> {}: 无网格条目".format(label, object_key))
        elif not Path(mesh).is_file():
            problems.append("{} -> {} 的网格文件不存在: {}".format(
                label, object_key, mesh))
            print("  ❌ {} -> {}: 文件不存在 {}".format(label, object_key, mesh))
        else:
            print("  ✓ {} -> {}".format(label, object_key))

    # 3) 网格实际尺寸（缩放是否合理）
    print("\n[3] 网格实际尺寸（缩放后，毫米）")
    try:
        import numpy as np
        import trimesh
    except ImportError:
        problems.append(
            "缺少 trimesh/numpy —— 网格尺寸与缩放检查被跳过了。"
            "**跳过的检查会给假信心**，而缩放错会直接算错抓取高度。"
            "请改用 foundationpose 环境: "
            "PYTHONPATH=. /home/throne/miniconda3/envs/foundationpose/bin/python "
            "-m competition_pipeline.scripts.check_vision_assets"
        )
        print("  ❌ trimesh 不可用 —— 尺寸检查无法进行（见下方问题列表）")
    else:
        for object_key in sorted(set(mapping.values())):
            mesh_path = models.get(object_key, "")
            if not mesh_path or not Path(mesh_path).is_file():
                continue
            scale = float(scales.get(object_key,
                                     pipeline.get("mesh_scale_to_meters", 1.0)))
            try:
                mesh = trimesh.load(mesh_path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                bounds = np.asarray(mesh.bounds, dtype=float) * scale
            except Exception as error:              # noqa: BLE001 - 诊断脚本
                problems.append("{} 网格读取失败: {}".format(object_key, error))
                print("  ❌ {}: {}".format(object_key, error))
                continue
            extent_mm = (bounds[1] - bounds[0]) * 1000.0
            offset_mm = bounds.mean(axis=0) * 1000.0
            largest = float(extent_mm.max())
            flag = "✓"
            if largest < 20.0 or largest > 400.0:
                problems.append(
                    "{} 缩放后最大尺寸 {:.1f}mm 不像真实物体，"
                    "object_model_scales 多半错了".format(object_key, largest))
                flag = "❌"
            print("  {} {:<10} {} mm  原点偏移 {} mm".format(
                flag, object_key, np.round(extent_mm, 1), np.round(offset_mm, 1)))
            if float(np.abs(offset_mm).max()) > 5.0:
                print("      ↑ 原点不在包围盒中心，抓取高度必须用包围盒中心"
                      "（grasp_geometry 已处理）")

    # 4) 抓取规则齐备
    print("\n[4] 抓取规则")
    for object_key in sorted(set(mapping.values())):
        rule = rules.get(object_key)
        if not rule:
            problems.append(
                "{} 没有 grasp_rules 条目，抓取高度会退化成"
                "'对准中心'（高瓶子有压爆风险）".format(object_key))
            print("  ❌ {}: 无规则".format(object_key))
            continue
        grasp_type = str(rule.get("type", ""))
        if grasp_type not in KNOWN_GRASP_TYPES:
            problems.append("{} 的 grasp_rules.type={!r} 不是 {}".format(
                object_key, grasp_type, KNOWN_GRASP_TYPES))
            print("  ❌ {}: type={!r}".format(object_key, grasp_type))
        else:
            print("  ✓ {:<10} type={}".format(object_key, grasp_type))

    # 4.5) 缩放核对（尺子实测 vs CAD）
    print("\n[4.5] CAD 缩放 vs 尺子实测")
    measured = raw.get("measured_object_mm", {}) or {}
    used_keys = sorted(set(mapping.values()))
    unmeasured = [key for key in used_keys if key not in measured]
    try:
        import numpy as np
        import trimesh
    except ImportError:
        print("  ❌ trimesh 不可用 —— 缩放核对无法进行")
    else:
        for object_key in used_keys:
            expected = measured.get(object_key)
            if not expected:
                continue
            mesh_path = models.get(object_key, "")
            if not mesh_path or not Path(mesh_path).is_file():
                continue
            scale = float(scales.get(object_key,
                                     pipeline.get("mesh_scale_to_meters", 1.0)))
            mesh = trimesh.load(mesh_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            extent_mm = np.sort(
                (np.asarray(mesh.bounds, dtype=float)[1]
                 - np.asarray(mesh.bounds, dtype=float)[0]) * scale * 1000.0
            )
            cad_height = float(extent_mm[2])          # 最长边 = 高度(直立物体)
            cad_diameter = float(extent_mm[1])        # 次长边 = 直径
            want_height = float(expected["height"])
            want_diameter = float(expected["diameter"])
            bad = []
            if abs(cad_height - want_height) > max(3.0, want_height * 0.05):
                bad.append("高 CAD {:.1f} vs 实测 {:.1f}".format(
                    cad_height, want_height))
            if abs(cad_diameter - want_diameter) > max(3.0, want_diameter * 0.05):
                bad.append("直径 CAD {:.1f} vs 实测 {:.1f}".format(
                    cad_diameter, want_diameter))
            if bad:
                problems.append(
                    "{} 的 CAD 缩放与尺子实测不符（{}）。抓取高度正比于物体高度，"
                    "缩放错就会压爆或抓空 —— 改 object_model_scales。".format(
                        object_key, "；".join(bad))
                )
                print("  ❌ {:<10} {}".format(object_key, "；".join(bad)))
            else:
                print("  ✓ {:<10} 高 {:.1f}mm 直径 {:.1f}mm 与实测相符".format(
                    object_key, cad_height, cad_diameter))
    if unmeasured:
        notes.append(
            "以下物体的 CAD 缩放**未经尺子核对**：{}。"
            "缩放错会直接算错抓取高度；量完填进 measured_object_mm 即可自动校验。"
            .format("、".join(unmeasured))
        )
        print("  ⚠️ 未核对: {}".format("、".join(unmeasured)))
        print("     （运行时还有一道点云交叉校验兜底，但尺子最可靠）")

    # 5) 工具坐标系一致性
    print("\n[5] 工具坐标系 / 夹爪几何一致性")
    competition_path = Path(competition_config)
    if not competition_path.is_file():
        notes.append("找不到 {}，跳过工具坐标系检查".format(competition_path))
        print("  ⚠️ 跳过：{} 不存在".format(competition_path))
    else:
        competition = _load_yaml(competition_path)
        gripper = competition.get("gripper_geometry", {}) or {}
        planning = competition.get("grasp_planning", {}) or {}
        entry = planning.get("tcp_from_grasp", {}) or {}
        matrix = entry.get("matrix") if isinstance(entry, dict) else entry
        fingertip = gripper.get("tcp_to_fingertip_mm")
        if matrix is None:
            problems.append("grasp_planning.tcp_from_grasp.matrix 缺失")
            print("  ❌ tcp_from_grasp 缺失")
        elif fingertip is None:
            problems.append(
                "gripper_geometry.tcp_to_fingertip_mm 尚未标定（值为 null）。"
                "跑 scripts/calibrate_fingertip.py：指尖触桌读到的用户系 Z 就是它。"
            )
            print("  ❌ tcp_to_fingertip_mm 未标定")
        else:
            try:
                import numpy as np
                grasp_z_mm = float(
                    np.asarray(matrix, dtype=float).reshape(4, 4)[2, 3] * 1000.0
                )
            except Exception as error:                  # noqa: BLE001 - 诊断脚本
                problems.append("tcp_from_grasp.matrix 解析失败: {}".format(error))
                grasp_z_mm = None
            if grasp_z_mm is not None:
                delta = abs(grasp_z_mm - float(fingertip))
                if delta > 2.0:
                    problems.append(
                        "tcp_from_grasp 的 Z={:.1f}mm 与 tcp_to_fingertip_mm="
                        "{:.1f}mm 不一致（差 {:.1f}mm）。抓取会**系统性偏移这个量**。"
                        "若已把工具坐标系标到尖端(工具手1)，两者都应为 0，"
                        "并且手眼矩阵必须一起补偿 —— 跑 scripts/retarget_tool_frame.py。"
                        .format(grasp_z_mm, float(fingertip), delta)
                    )
                    print("  ❌ tcp_from_grasp.Z={:.1f} vs 指尖 {:.1f} 差 {:.1f}mm"
                          .format(grasp_z_mm, float(fingertip), delta))
                else:
                    print("  ✓ tcp_from_grasp.Z={:.1f}mm 与指尖标定一致".format(
                        grasp_z_mm))
        cavity = gripper.get("jaw_cavity_depth_mm")
        clearance = gripper.get("safety_clearance_mm")
        if cavity and clearance is not None:
            usable = float(cavity) - float(clearance)
            print("  ✓ 腔体可用深度 {:.0f}mm -> 不触发钳位的物体高度上限 {:.0f}mm"
                  .format(usable, 4.0 * usable))

    print("\n" + "=" * 60)
    for note in notes:
        print("⚠️ {}".format(note))
    if problems:
        print("❌ {} 个问题：".format(len(problems)))
        for problem in problems:
            print("   - {}".format(problem))
        return 1
    print("✅ 视觉资产自检通过")
    return 0


def record_measurement(config_path, object_key, diameter_mm, height_mm):
    """把尺子实测尺寸写进 ``measured_object_mm``，避免现场手改 YAML 缩进出错。"""
    import yaml

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    measured = data.setdefault("measured_object_mm", {})
    measured[str(object_key)] = {
        "diameter": round(float(diameter_mm), 2),
        "height": round(float(height_mm), 2),
    }
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print("已记录 {}: 直径 {:.1f}mm 高 {:.1f}mm -> {}\n".format(
        object_key, float(diameter_mm), float(height_mm), path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--competition-config", default=str(DEFAULT_COMPETITION))
    parser.add_argument(
        "--measure", nargs=3, metavar=("物体", "直径mm", "高mm"), action="append",
        help="记录尺子实测尺寸后再自检，可重复。"
             "例: --measure nescafe 60 169 --measure apple 75 88",
    )
    args = parser.parse_args(argv)
    for object_key, diameter, height in (args.measure or []):
        record_measurement(
            Path(args.visual_config), object_key, float(diameter), float(height)
        )
    return check(Path(args.visual_config), Path(args.competition_config))


if __name__ == "__main__":
    sys.exit(main())
