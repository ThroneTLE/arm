#!/usr/bin/env python3
"""GLB/glTF 材质归一化的回归测试。

FoundationPose 的 ``Utils.make_mesh_tensors`` 只认两种形态::

    isinstance(mesh.visual, TextureVisuals) -> 读 mesh.visual.material.image
    否则                                     -> 读 mesh.visual.vertex_colors

OBJ 走第一条没问题。但 **GLB 加载后是 PBRMaterial，``.image`` 恒为 None**，
纹理挂在 ``baseColorTexture`` 上 —— 第一条分支直接
``AttributeError: 'PBRMaterial' object has no attribute 'image'``。

2026-08-22 实测：``可口可乐.glb`` 与 ``雀巢咖啡.glb`` 都会这样崩；所有 OBJ 模型
不受影响（``lemon``/``orange`` 本来就是 ColorVisuals）。

下面的 ``_fp_branch`` 逐行照抄 FoundationPose 的分支逻辑，用来断言"归一化之后
FoundationPose 一定不会崩"，而不是断言我们自己的内部表示。
"""

import unittest

import numpy as np

try:
    import trimesh
except ImportError:                                   # pragma: no cover
    trimesh = None

from arm_vision_framework.adapters.foundationpose import FoundationPoseRuntime


def _fp_branch(mesh):
    """照抄 FoundationPose/Utils.py:104-123。崩就是崩，不做任何保护。"""
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        image = np.array(mesh.visual.material.image.convert("RGB"))[..., :3]
        uv = np.asarray(mesh.visual.uv, dtype=np.float32)
        if uv.shape[0] != len(mesh.vertices):
            raise AssertionError("uv 数量与顶点不符")
        return ("texture", image.shape)
    colors = mesh.visual.vertex_colors
    if colors is None or len(colors) != len(mesh.vertices):
        raise AssertionError("vertex_colors 长度与顶点不符")
    return ("vertex_color", np.asarray(colors)[0][:3])


def _box():
    return trimesh.creation.box(extents=(0.05, 0.05, 0.15))


@unittest.skipIf(trimesh is None, "trimesh 不可用")
class NormalizeVisualTest(unittest.TestCase):
    def test_a_pbr_material_with_a_texture_becomes_usable(self):
        """可乐那种：纹理挂在 baseColorTexture 上，to_simple() 能救回来。"""
        from PIL import Image

        mesh = _box()
        image = Image.new("RGB", (8, 8), (200, 30, 30))
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=image)
        uv = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        self.assertIsNone(getattr(mesh.visual.material, "image", None))
        with self.assertRaises(AttributeError):
            _fp_branch(mesh)                          # 未归一化 -> 崩

        normalized = FoundationPoseRuntime._normalize_visual(trimesh, mesh)
        kind, _detail = _fp_branch(normalized)
        self.assertEqual(kind, "texture")

    def test_a_pbr_material_without_a_texture_falls_back_to_vertex_colours(self):
        """雀巢那种：只有 baseColorFactor 纯色，必须改走顶点色分支。"""
        mesh = _box()
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[37, 14, 5, 255]
        )
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        with self.assertRaises(AttributeError):
            _fp_branch(mesh)

        normalized = FoundationPoseRuntime._normalize_visual(trimesh, mesh)
        kind, colour = _fp_branch(normalized)
        self.assertEqual(kind, "vertex_color")
        # 纯色要铺满所有顶点，而且保留原色而不是变成灰
        np.testing.assert_array_equal(colour, [37, 14, 5])
        self.assertEqual(
            len(normalized.visual.vertex_colors), len(normalized.vertices)
        )

    def test_replacing_the_visual_is_required_not_just_setting_vertex_colors(self):
        """只给 TextureVisuals 赋 vertex_colors 不改变 isinstance -> 照样崩。

        这正是旧 ``_load_mesh`` 里那段"补顶点色"没能挡住 GLB 的原因。
        """
        mesh = _box()
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[37, 14, 5, 255]
        )
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        mesh.visual.vertex_colors = np.tile(
            np.asarray([150, 150, 150, 255], dtype=np.uint8), (len(mesh.vertices), 1)
        )
        self.assertIsInstance(mesh.visual, trimesh.visual.texture.TextureVisuals)
        with self.assertRaises(AttributeError):
            _fp_branch(mesh)

    def test_an_obj_style_texture_is_left_untouched(self):
        """已验证的 OBJ 路径必须一个字节都不变。"""
        from PIL import Image

        mesh = _box()
        material = trimesh.visual.material.SimpleMaterial(
            image=Image.new("RGB", (4, 4), (10, 20, 30))
        )
        uv = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        before = mesh.visual
        normalized = FoundationPoseRuntime._normalize_visual(trimesh, mesh)
        self.assertIs(normalized.visual, before)
        self.assertEqual(_fp_branch(normalized)[0], "texture")

    def test_plain_colour_visuals_are_left_untouched(self):
        """lemon/orange 那种本来就是 ColorVisuals，不该被动。"""
        mesh = _box()
        before = mesh.visual
        normalized = FoundationPoseRuntime._normalize_visual(trimesh, mesh)
        self.assertIs(normalized.visual, before)


@unittest.skipIf(trimesh is None, "trimesh 不可用")
class RealModelTest(unittest.TestCase):
    """对仓库里真实存在的模型跑一遍（模型是 gitignore 的大文件，缺失就跳过）。"""

    MODELS = {
        "cola.glb": ("cola/mesh/cola.glb", 1.0),
        "nescafe.glb": ("nescafe/mesh/nescafe.glb", 0.001),
        "can2.obj": ("can2/mesh/can.obj", 0.001),
        "lemon.obj": ("lemon/mesh/lemon.obj", 0.0343),
    }

    @classmethod
    def _root(cls):
        from pathlib import Path
        return (
            Path(__file__).resolve().parents[2]
            / "tool/arm_project_20260822/fp_release_20260821_155930/models"
        )

    def test_every_available_model_survives_the_foundationpose_branch(self):
        root = self._root()
        checked = 0
        for name, (relative, scale) in sorted(self.MODELS.items()):
            path = root / relative
            if not path.is_file():
                continue
            with self.subTest(model=name):
                mesh = FoundationPoseRuntime._load_mesh(trimesh, path, scale)
                kind, _detail = _fp_branch(mesh)
                self.assertIn(kind, ("texture", "vertex_color"))
                checked += 1
        if checked == 0:
            self.skipTest("本机没有模型文件")


if __name__ == "__main__":
    unittest.main()
