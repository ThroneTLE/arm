#!/usr/bin/env python3
"""无硬件演练脚本的冒烟测试。

这个脚本是"明天之前唯一能端到端验证抓取链路"的手段，所以它自己不能坏。
它跑的是**真** UcsGraspRunner + **真** NexBotTcpJog，只假冒最底层控制器 ——
因此这组测试顺带就是整条链路的集成测试。
"""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from competition_pipeline.scripts.offline_rehearsal import (
    SimulatedController,
    _upright_pose,
    main,
)


def _run(argv):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        status = main(argv)
    return status, buffer.getvalue()


class UprightPoseTest(unittest.TestCase):
    """各网格长轴方向不一，用单位旋转摆会把可乐瓶横躺放着。"""

    def _bounds(self, extents_mm, long_axis):
        half = np.full(3, 30.0)
        half[long_axis] = float(extents_mm) / 2.0
        return np.asarray([-half, half]) / 1000.0

    def test_every_long_axis_ends_up_vertical_and_on_the_table(self):
        for long_axis in (0, 1, 2):
            with self.subTest(long_axis=long_axis):
                bounds = self._bounds(245.0, long_axis)
                pose = _upright_pose(bounds, (120.0, -80.0))
                corners = np.asarray([[x, y, z]
                                      for x in (bounds[0][0], bounds[1][0])
                                      for y in (bounds[0][1], bounds[1][1])
                                      for z in (bounds[0][2], bounds[1][2])])
                world = (pose[:3, :3] @ corners.T).T + pose[:3, 3]
                world_mm = world * 1000.0
                self.assertAlmostEqual(world_mm[:, 2].min(), 0.0, places=6)
                self.assertAlmostEqual(world_mm[:, 2].max(), 245.0, places=6)
                centre = (world_mm.min(axis=0) + world_mm.max(axis=0)) / 2.0
                self.assertAlmostEqual(centre[0], 120.0, places=6)
                self.assertAlmostEqual(centre[1], -80.0, places=6)

    def test_the_rotation_is_a_proper_rotation(self):
        for long_axis in (0, 1, 2):
            pose = _upright_pose(self._bounds(200.0, long_axis), (0.0, 0.0))
            rotation = pose[:3, :3]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)


class SimulatedControllerTest(unittest.TestCase):
    def test_dout_readback_reflects_writes(self):
        controller = SimulatedController(np.eye(4))
        controller.set_digital_output(15, 0)
        controller.set_digital_output(16, 1)
        self.assertEqual(tuple(controller.digital_output_states()[14:16]), (0, 1))

    def test_a_stuck_coil_ignores_writes(self):
        controller = SimulatedController(np.eye(4), faults=["gripper-stuck"])
        controller.set_digital_output(15, 0)
        controller.set_digital_output(16, 1)
        self.assertEqual(tuple(controller.digital_output_states()[14:16]), (1, 0))

    def test_reset_leaves_the_servo_in_ready_like_the_real_controller(self):
        """实测: GO_RESET_POSITION 结束时伺服落到 1(就绪)，不是 3。"""
        controller = SimulatedController(np.eye(4))
        controller.go_reset_position()
        self.assertEqual(controller.servo_status(), 1)


class RehearsalRunTest(unittest.TestCase):
    def test_the_happy_path_executes_the_full_vertical_sequence(self):
        status, output = _run(["--object", "sprite"])
        self.assertEqual(status, 0)
        self.assertIn("实际下发 6 段 MOVL", output)
        self.assertIn("本轮完成", output)
        # 雪碧罐 146.5mm -> 3/4 高度 109.9mm，伸入 36.6mm
        self.assertIn("109.9", output)
        self.assertIn("36.6", output)

    def test_a_tall_bottle_stays_within_the_cavity(self):
        _status, output = _run(["--object", "cola"])
        self.assertIn("245.0mm", output)
        self.assertIn("61.3", output)          # 245/4，仍在 65mm 内
        self.assertNotIn("已按腔体钳位", output)

    def test_slots_advance_and_then_report_being_full(self):
        _status, output = _run(["--rounds", "5"])
        for slot in range(4):
            self.assertIn("放置槽位 {}".format(slot), output)
        self.assertIn("放置区已经用满", output)
        self.assertIn("完成 4/5 轮", output)

    def test_a_refused_servo_surfaces_the_field_diagnosis(self):
        _status, output = _run(["--fault", "servo-refuse"])
        self.assertIn("执行失败", output)
        self.assertIn("复位点安全闸门", output)
        self.assertIn("完成 0/1 轮", output)

    def test_a_stuck_gripper_aborts_instead_of_reporting_success(self):
        _status, output = _run(["--fault", "gripper-stuck"])
        self.assertIn("DOUT 回读不符", output)
        self.assertNotIn("本轮完成", output)

    def test_an_unreadable_gripper_completes_but_says_it_is_unverified(self):
        """0x3603 未经现场验证，读不到不该判死；但必须如实标注。"""
        _status, output = _run(["--fault", "gripper-unreadable"])
        self.assertIn("本轮完成", output)
        self.assertIn("未回读", output)

    def test_a_rejected_motion_reports_the_controller_reason(self):
        _status, output = _run(["--fault", "motion-rejected"])
        self.assertIn("执行失败", output)
        self.assertIn("已下电", output)


if __name__ == "__main__":
    unittest.main()
