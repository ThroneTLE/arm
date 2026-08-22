#!/usr/bin/env python3
"""Offline tests for the one-touch grasp demo panel/worker.

FakeJog 必须与 ``NexBotTcpJog`` 的真实签名保持一致（含 ``current_pose_rad`` 与
``move_to_ucs`` 的 ``rotation_tolerance_deg``）。签名对不上时 worker 的兜底
``except Exception`` 会把 TypeError 吞成一条 "抓取失败" 消息，序列静静地停在
半路 —— 这正是我们要杜绝的那类"看起来跑完了"的假象，所以这里每个用例都断言
``done`` 信号的成败，而不只看命令流水。
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from competition_pipeline.grasp_demo import GraspDemoPanel, GraspDemoWorker


class FakeJog:
    """Mirror of the NexBotTcpJog surface the grasp worker actually uses."""

    def __init__(self, servo_running=True, gripper_detail=""):
        self.commands = []
        self.xyz = [0.0, 0.0, 0.0]
        self.abc = [0.0, 0.0, 0.0]
        self.servo_running = bool(servo_running)
        #: 非空时模拟"DOUT 回读失败"，真实实现会返回 (True, "未回读: …")
        self.gripper_detail = str(gripper_detail)

    # -- motion ---------------------------------------------------------
    def gripper(self, open_, verify=True):
        self.commands.append(("gripper", bool(open_)))
        return (True, self.gripper_detail)

    def move_to_ucs(self, xyz, abc, vel_mm_s=50.0, tolerance_mm=1.0,
                    rotation_tolerance_deg=3.0, **kwargs):
        if not self.servo_running:
            raise RuntimeError("伺服未能进入运行态(status=1)")
        self.commands.append(("move", tuple(xyz), tuple(abc)))
        self.xyz = list(xyz)
        self.abc = list(abc)
        return 0.0

    def go_reset_position(self):
        if not self.servo_running:
            raise RuntimeError("伺服未能进入运行态(status=1)")
        self.commands.append(("reset",))

    # -- readback -------------------------------------------------------
    def current_pose(self):
        """(xyz_mm, abc_deg) —— 角度制，和真实实现一致。"""
        return tuple(self.xyz), tuple(np.degrees(self.abc))

    def current_pose_rad(self):
        """(xyz_mm, abc_rad) —— 弧度，运动路径应该用这个。"""
        return tuple(self.xyz), tuple(self.abc)


def _run_worker(worker):
    """Run the worker body synchronously and capture its ``done`` payload."""
    captured = []
    worker.done.connect(lambda ok, message: captured.append((ok, message)))
    worker.run()
    return captured[-1] if captured else (None, "worker 没有发出 done 信号")


class GraspDemoWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _worker(self, jog):
        return GraspDemoWorker(
            lambda: jog,
            grasp=([0.0, 0.0, 40.0], [0.0, 0.0, 0.0]),
            place=([0.0, -50.0, 40.0], [0.0, 0.0, 0.0]),
            lift_mm=80.0,
        )

    def test_default_zero_targets_cannot_execute(self):
        panel = GraspDemoPanel(lambda: FakeJog())
        self.assertFalse(panel.btn_go.isEnabled())

    def test_sequence_opens_before_approach_and_places_after_grasp(self):
        jog = FakeJog()
        ok, message = _run_worker(self._worker(jog))
        self.assertTrue(ok, message)
        self.assertEqual(jog.commands[0], ("reset",))
        self.assertEqual(jog.commands[1], ("gripper", True))
        self.assertIn(("gripper", False), jog.commands)
        self.assertEqual(jog.commands[-3], ("gripper", True))
        self.assertEqual(jog.commands[-2][0], "move")
        self.assertEqual(jog.commands[-1], ("reset",))

    def test_worker_uses_the_radian_readback_not_the_degree_one(self):
        """姿态校验必须走 current_pose_rad；用 current_pose 会引入单位转换。"""
        jog = FakeJog()
        calls = []
        jog.current_pose = lambda: calls.append("deg") or (tuple(jog.xyz), (0.0, 0.0, 0.0))
        ok, message = _run_worker(self._worker(jog))
        self.assertTrue(ok, message)
        self.assertEqual(calls, [], "抓取序列不应再调用角度制的 current_pose()")

    def test_a_refused_reset_stops_the_sequence_instead_of_grasping_air(self):
        """复位被安全闸门拒绝(伺服下电)时，绝不能继续往下开合夹爪。"""
        jog = FakeJog(servo_running=False)
        ok, message = _run_worker(self._worker(jog))
        self.assertFalse(ok)
        self.assertIn("伺服", message)
        self.assertEqual(jog.commands, [], "被拒后不应再发任何后续动作")

    def test_an_unverified_gripper_is_not_reported_as_a_clean_success(self):
        """夹爪 DOUT 读不到时，结论里必须留下"未确认"，不能只剩 ✅。

        运动有 0x3D03 + 到位校验背书，夹爪没有；把两者混成同一个 ✅ 正是
        2026-08-22 "夹爪在开合、位置没动却报完成" 的那类假象。
        """
        jog = FakeJog(gripper_detail="未回读: DOUT 查询超时")
        ok, message = _run_worker(self._worker(jog))
        self.assertTrue(ok, message)
        self.assertIn("未能回读确认", message)
        self.assertIn("DOUT 查询超时", message)

    def test_a_verified_gripper_reports_a_clean_success(self):
        jog = FakeJog()
        ok, message = _run_worker(self._worker(jog))
        self.assertTrue(ok, message)
        self.assertNotIn("未能回读确认", message)
        self.assertTrue(message.startswith("✅"))

    def test_a_contradicted_gripper_readback_fails_the_sequence(self):
        """回读到与指令相反的 DOUT -> jog.gripper 抛错 -> 整个序列判失败。"""
        jog = FakeJog()

        def _refuse(open_, verify=True):
            raise RuntimeError("夹爪合指令已下发但 DOUT 回读不符")

        jog.gripper = _refuse
        ok, message = _run_worker(self._worker(jog))
        self.assertFalse(ok)
        self.assertIn("DOUT 回读不符", message)

    def test_abort_between_steps_stops_the_sequence(self):
        jog = FakeJog()
        worker = self._worker(jog)
        worker.abort.set()
        ok, message = _run_worker(worker)
        self.assertFalse(ok)
        self.assertIn("急停", message)


if __name__ == "__main__":
    unittest.main()
