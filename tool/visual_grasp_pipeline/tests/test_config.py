#!/usr/bin/env python3

import os
import tempfile
import unittest
from pathlib import Path

from tool.visual_grasp_pipeline.config import VisualGraspConfig


class ConfigTests(unittest.TestCase):
    def test_from_yaml_resolves_paths_and_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                """
paths:
  yolo_weights: ~/model.pt
  foundationpose_root: /opt/fp
  static_frame_dir: ~/frames
object_models:
  can: /opt/meshes/can.obj
object_model_scales:
  can: 0.001
yolo_to_object:
  can: sprite
grasp_rules:
  sprite: {type: cylinder, offset_mm: 7.0}
pipeline:
  target_label: can
  offset_xy_mm: [1.0, 2.0]
  flip_x: false
""",
                encoding="utf-8",
            )
            config = VisualGraspConfig.from_yaml(config_path)
            self.assertEqual(config.yolo_weights, os.path.expanduser("~/model.pt"))
            self.assertEqual(config.offset_xy_mm, (1.0, 2.0))
            self.assertFalse(config.flip_x)
            self.assertEqual(config.rule_for_object("sprite").offset_mm, 7.0)
            self.assertEqual(config.resolve_object_key("can", 5), "sprite")
            self.assertEqual(config.mesh_scale_for_object("can"), 0.001)
            self.assertEqual(config.mesh_scale_for_object("sprite"), 1.0)
            self.assertEqual(config.foundationpose_max_input_size, 640)
            self.assertEqual(config.foundationpose_registration_hypotheses, 64)

    def test_missing_mesh_is_empty_string(self):
        config = VisualGraspConfig()
        self.assertEqual(config.mesh_for_object("unknown"), "")

    def test_active_config_maps_new_lemon_model_with_metric_scale(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "config" / "visual_grasp_pipeline.yaml"
        )
        config = VisualGraspConfig.from_yaml(path)
        self.assertEqual(config.target_label, "lemon")
        self.assertEqual(config.resolve_object_key("lemon", 2), "lemon")
        self.assertAlmostEqual(config.mesh_scale_for_object("lemon"), 0.0343)
        self.assertTrue(Path(config.mesh_for_object("lemon")).is_file())


if __name__ == "__main__":
    unittest.main()
