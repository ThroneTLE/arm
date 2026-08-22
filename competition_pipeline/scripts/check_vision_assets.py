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


def check(config_path):
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
            notes.append("未能读出权重类别名，跳过映射校验")
            print("  ⚠️ 未能读出类别名（torch 不可用或格式不符），跳过映射校验")
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
        notes.append("trimesh/numpy 不可用，跳过尺寸检查")
        print("  ⚠️ trimesh 不可用，跳过")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    return check(Path(args.visual_config))


if __name__ == "__main__":
    sys.exit(main())
