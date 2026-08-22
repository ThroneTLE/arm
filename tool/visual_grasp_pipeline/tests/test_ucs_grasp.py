#!/usr/bin/env python3
"""Tests for the user-frame one-click grasp executor (no robot, no network)."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tool.visual_grasp_pipeline.ucs_grasp import (
    MAX_SINGLE_LEG_MM,
    SAFE_Z_MAX_MM,
    SAFE_Z_MIN_MM,
    UCS_PLACE_X_MM,
    UCS_PLACE_Y_MM,
    UcsGraspExecutorError,
    UcsGraspRunner,
    UcsGraspSafetyError,
    build_ucs_grasp_plan,
    build_jog,
    validate_targets,
)


def reset_pose(rotation=None, xyz_mm=(0.0, 0.0, 200.0)):
    """A user-frame reset pose (拍摄点) with an optional non-trivial rotation."""
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(xyz_mm, dtype=np.float64) / 1000.0
    return matrix


class FakeController:
    def __init__(self, state_matrix):
        self.state_matrix = state_matrix
        self.calls = []
        self.servo_state = 3

    def servo_status(self):
        self.calls.append("servo_status")
        return self.servo_state

    def enable_servo(self):
        self.calls.append("enable_servo")
        self.servo_state = 3
        return 3

    def read_state(self):
        self.calls.append("read_state")
        return SimpleNamespace(base_from_gripper=self.state_matrix.copy())

    def move_to(self, target, speed_scale=0.1):
        self.calls.append(("move_to", np.asarray(target, dtype=np.float64).copy(),
                           float(speed_scale)))


class FakeJog:
    def __init__(self, state_matrix):
        self.controller = FakeController(state_matrix)
        self.calls = []

    def go_reset_position(self):
        self.calls.append("go_reset_position")
        # Field controller ends GO_RESET_POSITION in ready(status=1).
        self.controller.servo_state = 1

    def gripper(self, open_, verify=True):
        self.calls.append(("gripper", bool(open_)))
        return (True, "")            # (ok, detail) —— 与真实 jog.gripper 一致

    def move_to_ucs(self, xyz_mm, abc_rad, vel_mm_s=50.0, **kwargs):
        """运动统一走这里：真实实现带单位/姿态/位移闸门 + 使能前置 + 到位校验。

        以前 ucs_grasp 直调 ``jog.controller.move_to``，绕过事务锁和全部闸门。
        """
        self.calls.append(("move_to_ucs", tuple(np.round(xyz_mm, 3))))
        self.controller.state_matrix = np.asarray(
            self.controller.state_matrix, dtype=np.float64
        ).copy()
        self.controller.state_matrix[:3, 3] = np.asarray(
            xyz_mm, dtype=np.float64
        ) / 1000.0
        return 0.0

    def emergency_stop(self):
        self.calls.append("emergency_stop")


class UcsGraspPlanTests(unittest.TestCase):
    def test_plan_approaches_vertically_and_traverses_high(self):
        """序列必须是"垂直进、垂直出、高位横移"。

        旧版是 ``抓取位 -> 直接横移到放置位``(同一 Z)，等于夹着物体在桌面上方
        八十几毫米处横扫，会把路上的其它物品撞倒。赛题第二/三档就是有遮挡物的
        场景，低位横移必然扫到。
        """
        from tool.visual_grasp_pipeline.ucs_grasp import APPROACH_CLEARANCE_MM

        grasp = np.array([5.0, 12.0, 60.0])
        plan = build_ucs_grasp_plan(grasp, rotation=np.eye(3))
        kinds = [step["kind"] for step in plan.steps]
        self.assertEqual(
            kinds,
            ["reset", "move", "move", "gripper", "move",
             "move", "move", "gripper", "move", "reset"],
        )
        high = 60.0 + APPROACH_CLEARANCE_MM
        # 抓取位上方 -> 垂直下降 (XY 不变)
        np.testing.assert_allclose(plan.steps[1]["xyz_mm"], [5.0, 12.0, high])
        np.testing.assert_allclose(plan.steps[2]["xyz_mm"], grasp)
        self.assertIs(plan.steps[3]["open"], False)
        # 抬升 -> 高位横移 -> 垂直下降
        np.testing.assert_allclose(plan.steps[4]["xyz_mm"], [5.0, 12.0, high])
        np.testing.assert_allclose(
            plan.steps[5]["xyz_mm"], [UCS_PLACE_X_MM, UCS_PLACE_Y_MM, high]
        )
        np.testing.assert_allclose(
            plan.steps[6]["xyz_mm"], [UCS_PLACE_X_MM, UCS_PLACE_Y_MM, 60.0]
        )
        self.assertIs(plan.steps[7]["open"], True)
        self.assertEqual(plan.place_xyz_mm.tolist(),
                         [UCS_PLACE_X_MM, UCS_PLACE_Y_MM, 60.0])

    def test_the_traverse_never_happens_at_grasp_height(self):
        """把"禁止低位横移"钉成断言：任何 XY 发生变化的一段，Z 都必须在高位。"""
        from tool.visual_grasp_pipeline.ucs_grasp import APPROACH_CLEARANCE_MM

        grasp = np.array([5.0, 12.0, 60.0])
        plan = build_ucs_grasp_plan(grasp, rotation=np.eye(3))
        moves = [step["xyz_mm"] for step in plan.steps if step["kind"] == "move"]
        high = 60.0 + APPROACH_CLEARANCE_MM
        for before, after in zip(moves, moves[1:]):
            xy_moved = not np.allclose(before[:2], after[:2])
            if xy_moved:
                self.assertAlmostEqual(before[2], high, places=6)
                self.assertAlmostEqual(after[2], high, places=6)

    def test_a_grasp_too_close_to_the_ceiling_is_refused(self):
        """留不出垂直接近段就必须拒绝，而不是退化成低位横移。"""
        from tool.visual_grasp_pipeline.ucs_grasp import (
            SAFE_Z_MAX_MM, UcsGraspSafetyError,
        )

        with self.assertRaises(UcsGraspSafetyError) as ctx:
            build_ucs_grasp_plan(
                np.array([0.0, 0.0, SAFE_Z_MAX_MM]), rotation=np.eye(3)
            )
        self.assertIn("低位横移", str(ctx.exception))

    def test_place_override(self):
        plan = build_ucs_grasp_plan(
            np.array([0.0, 0.0, 80.0]), rotation=None,
            place_x_mm=-50.0, place_y_mm=200.0,
        )
        self.assertEqual(plan.place_xyz_mm.tolist(), [-50.0, 200.0, 80.0])

    def test_z_too_low_rejected(self):
        with self.assertRaises(UcsGraspSafetyError):
            validate_targets(np.array([0.0, 0.0, SAFE_Z_MIN_MM - 1.0]),
                             np.array([-100.0, 100.0, SAFE_Z_MIN_MM - 1.0]))

    def test_z_too_high_rejected(self):
        with self.assertRaises(UcsGraspSafetyError):
            validate_targets(np.array([0.0, 0.0, SAFE_Z_MAX_MM + 1.0]),
                             np.array([-100.0, 100.0, SAFE_Z_MAX_MM + 1.0]))

    def test_xy_out_of_box_rejected(self):
        with self.assertRaises(UcsGraspSafetyError):
            validate_targets(np.array([0.0, 0.0, 60.0]),
                             np.array([-500.0, 100.0, 60.0]))

    def test_long_leg_rejected(self):
        with self.assertRaises(UcsGraspSafetyError):
            validate_targets(np.array([0.0, 0.0, 60.0]),
                             np.array([-MAX_SINGLE_LEG_MM, 100.0, 60.0]))


class UcsGraspRunnerTests(unittest.TestCase):
    def test_build_jog_reuses_injected_persistent_controller(self):
        config = (
            Path(__file__).resolve().parents[3]
            / "competition_pipeline" / "config" / "competition.yaml"
        )
        controller = object()
        jog = build_jog(config, controller=controller)
        self.assertIs(jog.controller, controller)

    def test_build_jog_forces_protocol_heartbeat_off(self):
        config = (
            Path(__file__).resolve().parents[3]
            / "competition_pipeline" / "config" / "competition.yaml"
        )
        jog = build_jog(config)
        self.assertEqual(jog.endpoint.heartbeat_s, 0.0)

    def test_real_execution_sequence_and_fixed_orientation(self):
        # 非单位旋转, 验证姿态全程保持"复位位置姿态"
        rotation = np.asarray([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        jog = FakeJog(reset_pose(rotation=rotation, xyz_mm=(10.0, 20.0, 200.0)))
        runner = UcsGraspRunner(jog, place_x_mm=-100.0, place_y_mm=100.0)
        result = runner.execute(np.array([3.0, 4.0, 65.0]), dry_run=False)

        from tool.visual_grasp_pipeline.ucs_grasp import APPROACH_CLEARANCE_MM

        high = 65.0 + APPROACH_CLEARANCE_MM
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["place_xyz_mm"], [-100.0, 100.0, 65.0])
        # 垂直进、垂直出、高位横移；运动全部走 move_to_ucs(带闸门)，
        # 不再直调 controller.move_to(绕过事务锁与全部闸门)。
        self.assertEqual(
            jog.calls,
            [
                "go_reset_position",
                ("move_to_ucs", (3.0, 4.0, high)),      # 抓取位上方
                ("move_to_ucs", (3.0, 4.0, 65.0)),      # ↓ 抓取位
                ("gripper", False),
                ("move_to_ucs", (3.0, 4.0, high)),      # ↑ 抬升
                ("move_to_ucs", (-100.0, 100.0, high)), # 高位横移
                ("move_to_ucs", (-100.0, 100.0, 65.0)), # ↓ 放置位
                ("gripper", True),
                ("move_to_ucs", (-100.0, 100.0, high)), # ↑ 抬升
                "go_reset_position",
            ],
        )
        controller_calls = jog.controller.calls
        self.assertEqual(
            controller_calls[:4],
            ["servo_status", "servo_status", "enable_servo", "read_state"],
        )
        # 姿态全程 = 复位点姿态：move_to_ucs 收的是从该姿态解出的弧度。
        self.assertNotIn("emergency_stop", jog.calls)

    def test_dry_run_touches_nothing(self):
        jog = FakeJog(reset_pose())
        runner = UcsGraspRunner(jog)
        result = runner.execute(np.array([0.0, 0.0, 60.0]), dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(jog.calls, [])
        self.assertEqual(jog.controller.calls, [])

    def test_failure_triggers_emergency_stop_and_is_reported(self):
        jog = FakeJog(reset_pose())
        jog.move_to_ucs = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("controller rejected motion")
        )
        runner = UcsGraspRunner(jog)
        with self.assertRaises(UcsGraspExecutorError):
            runner.execute(np.array([0.0, 0.0, 60.0]), dry_run=False)
        self.assertEqual(jog.calls[-1], "emergency_stop")

    def test_the_orientation_stays_at_the_reset_pose_for_every_leg(self):
        """姿态全程不变是这条路径的安全前提：只动 XYZ，绝不换姿态。

        move_to_ucs 自带 20° 姿态闸门；把复位姿态一次解成弧度全程复用，
        保证每一段的姿态增量都是 0。
        """
        from competition_pipeline.geometry import inexbot_abc_from_transform

        rotation = np.asarray([
            [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        ])
        pose = reset_pose(rotation=rotation, xyz_mm=(10.0, 20.0, 200.0))
        jog = FakeJog(pose)
        seen = []
        original = jog.move_to_ucs

        def _record(xyz_mm, abc_rad, **kwargs):
            seen.append(np.asarray(abc_rad, dtype=float).copy())
            return original(xyz_mm, abc_rad, **kwargs)

        jog.move_to_ucs = _record
        UcsGraspRunner(jog).execute(np.array([3.0, 4.0, 65.0]), dry_run=False)
        _xyz, expected = inexbot_abc_from_transform(pose)
        self.assertEqual(len(seen), 6)
        for abc in seen:
            np.testing.assert_allclose(abc, expected, atol=1e-12)

    def test_safety_error_blocks_execution(self):
        jog = FakeJog(reset_pose())
        runner = UcsGraspRunner(jog)
        with self.assertRaises(UcsGraspSafetyError):
            runner.execute(np.array([0.0, 0.0, 5.0]), dry_run=False)
        self.assertEqual(jog.calls, [])

    def test_transient_drop_retried_only_for_idempotent_steps(self):
        from competition_pipeline.nexbot_tcp import ControllerConnectionError

        jog = FakeJog(reset_pose())
        drops = {"go_reset_position": 1}

        original_reset = jog.go_reset_position

        def flaky_reset():
            if drops["go_reset_position"] > 0:
                drops["go_reset_position"] -= 1
                jog.calls.append("go_reset_position(dropped)")
                raise ControllerConnectionError("simulated drop")
            return original_reset()

        jog.go_reset_position = flaky_reset
        runner = UcsGraspRunner(jog)
        result = runner.execute(np.array([0.0, 0.0, 60.0]), dry_run=False)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(jog.calls.count("go_reset_position(dropped)"), 1)
        self.assertEqual(jog.calls.count("go_reset_position"), 2)


if __name__ == "__main__":
    unittest.main()
