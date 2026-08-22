#!/usr/bin/env python3

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arm_vision_framework.adapters.io_gripper import RemoteIoGripper
from arm_vision_framework.motion_execution import (
    ObservationPlan, PickPlaceExecutor, PickPlacePlan,
    TwoViewPickPlaceCoordinator, cartesian_point, joint_point,
)
from arm_vision_framework.oak_depthai import OakDProProfile, profile_from_config


class FakeRobot:
    def __init__(self):
        self.commands = []
        self.stopped = False

    def move_j(self, points, speed_scale):
        self.commands.append(("move_j", tuple(points), speed_scale))
        return True

    def move_l(self, points, speed_mm_s):
        self.commands.append(("move_l", tuple(points), speed_mm_s))
        return True

    def stop(self):
        self.stopped = True
        return True


class FakeGripper:
    def __init__(self):
        self.commands = []
        self.stopped = False

    def open(self):
        self.commands.append("open")
        return True

    def close(self, width_mm, maximum_effort=None):
        self.commands.append(("close", width_mm, maximum_effort))
        return True

    def stop(self):
        self.stopped = True
        return True


class FakeIo:
    def __init__(self):
        self.outputs = []
        self.done = False

    def set_output(self, name, value):
        self.outputs.append((name, value))
        return True

    def read_input(self, name):
        self.last_input = name
        return self.done


def _plan():
    first = joint_point("P0001", [0, 0, 0, 0, 0, 0], tool_id=1, user_id=2)
    second = joint_point("P0002", [10, 0, 0, 0, 0, 0], tool_id=1, user_id=2)
    line = cartesian_point(
        "P0003", [400, 0, 200], [180, 0, 0], shape=8, tool_id=1, user_id=2
    )
    return PickPlacePlan(
        movej_pregrasp=(first,), movel_grasp=(line,), movel_lift=(line,),
        movej_preplace=(second,), movel_place=(line,), movej_retreat=(first,),
        grasp_width_mm=45.0,
    )


class CompetitionExecutionTest(unittest.TestCase):
    def test_all_motion_points_use_latched_initial_shape(self):
        robot, gripper = FakeRobot(), FakeGripper()
        state = type("State", (), {"shape": 8, "initial_shape": 8})()
        plan = _plan()
        # Deliberately make every Cartesian point carry stale metadata.  The
        # executor must replace it with the first controller shape rather than
        # recomputing shape from each waypoint.
        stale_line = replace(plan.movel_grasp[0], shape=1)
        plan = replace(
            plan, movel_grasp=(stale_line,), movel_lift=(stale_line,),
            movel_place=(stale_line,),
        )
        executor = PickPlaceExecutor(
            robot, gripper, dry_run=False, allow_motion=True,
            shape_provider=lambda: state,
        )
        result = executor.execute(plan)
        self.assertTrue(result.success, result.reason)
        sent = [point for _, points, _ in robot.commands for point in points]
        self.assertTrue(sent)
        self.assertTrue(all(point.shape == 8 for point in sent))

    def test_shape_change_blocks_remaining_motion(self):
        robot, gripper = FakeRobot(), FakeGripper()
        states = iter([
            type("State", (), {"shape": 8, "initial_shape": 8})(),
            type("State", (), {"shape": 7, "initial_shape": 8})(),
        ])
        executor = PickPlaceExecutor(
            robot, gripper, dry_run=False, allow_motion=True,
            shape_provider=lambda: next(states),
        )
        result = executor.execute(_plan())
        self.assertFalse(result.success)
        self.assertIn("shape changed", result.reason)
        self.assertTrue(robot.stopped)

    def test_dry_run_refusal_does_not_de_energise_the_arm(self):
        """联锁在运动前拒绝时不能发 stop()。

        在这台 C1102 上 robot.stop() 是 0x2314 -> Deadan_End -> PowerOff，
        即**直接下电**。机器人根本没动过就把伺服打掉，是纯粹的自伤；
        对伸展着的手臂更是坠落风险（2026-08-22 摔臂就是 PowerOff 造成的）。
        """
        for dry_run, allow_motion in ((True, True), (False, False)):
            robot, gripper = FakeRobot(), FakeGripper()
            executor = PickPlaceExecutor(
                robot, gripper, dry_run=dry_run, allow_motion=allow_motion,
            )
            result = executor.execute(_plan())
            self.assertFalse(result.success)
            self.assertFalse(
                robot.stopped,
                "dry_run={} allow_motion={} 时不应下电".format(dry_run, allow_motion),
            )
            self.assertEqual(robot.commands, [], "不应下发任何运动")

    def test_static_oak_profile_matches_manual_constraints(self):
        profile = OakDProProfile()
        self.assertEqual(profile.image_size, (1920, 1080))
        self.assertEqual(profile.mono_resolution, "800p")
        self.assertTrue(profile.extended_disparity)
        self.assertFalse(profile.subpixel)
        with self.assertRaisesRegex(ValueError, "cannot"):
            OakDProProfile(extended_disparity=True, subpixel=True)
        from_config = profile_from_config({"oak_d_pro": profile.metadata()})
        self.assertEqual(from_config.image_size, (1920, 1080))

    def test_two_view_observation_happens_before_pick_place(self):
        robot, gripper = FakeRobot(), FakeGripper()
        executor = PickPlaceExecutor(robot, gripper, dry_run=False, allow_motion=True)
        captures = []
        coordinator = TwoViewPickPlaceCoordinator(
            executor,
            capture_snapshot=lambda: captures.append("frame") or captures[-1],
            plan_from_snapshots=lambda first, second: _plan(),
            sleep=lambda seconds: None,
        )
        observation = ObservationPlan(
            movej_first_view=(joint_point("P0010", [0, 0, 0, 0, 0, 0]),),
            movej_second_view=(joint_point("P0011", [5, 0, 0, 0, 0, 0]),),
            settle_time_s=0.1,
        )
        result = coordinator.execute(observation)
        self.assertTrue(result.success, result.reason)
        self.assertEqual(captures, ["frame", "frame"])
        self.assertEqual(
            [command[0] for command in robot.commands],
            ["move_j", "move_j", "move_j", "move_l", "move_l", "move_j", "move_l", "move_j"],
        )
        self.assertEqual(gripper.commands, ["open", ("close", 45.0, None), "open"])
        self.assertLess(
            [event.state for event in result.events].index("capturing_second_view"),
            [event.state for event in result.events].index("movej_pregrasp"),
        )

    def test_gripper_remote_io_uses_only_configured_names(self):
        io = FakeIo()
        gripper = RemoteIoGripper(
            io,
            {"gripper_open": True, "gripper_close": False},
            {"gripper_open": False, "gripper_close": True},
        )
        gripper.open()
        gripper.close(42.0)
        self.assertEqual(
            io.outputs,
            [("gripper_open", True), ("gripper_close", False),
             ("gripper_open", False), ("gripper_close", True)],
        )

    def test_failed_snapshot_stops_both_devices(self):
        robot, gripper = FakeRobot(), FakeGripper()
        coordinator = TwoViewPickPlaceCoordinator(
            PickPlaceExecutor(robot, gripper, dry_run=False, allow_motion=True),
            capture_snapshot=lambda: (_ for _ in ()).throw(RuntimeError("camera unavailable")),
            plan_from_snapshots=lambda first, second: _plan(),
            sleep=lambda seconds: None,
        )
        observation = ObservationPlan(
            movej_first_view=(joint_point("P0010", [0, 0, 0, 0, 0, 0]),),
            movej_second_view=(joint_point("P0011", [5, 0, 0, 0, 0, 0]),),
        )
        result = coordinator.execute(observation)
        self.assertFalse(result.success)
        self.assertTrue(robot.stopped)
        self.assertTrue(gripper.stopped)


if __name__ == "__main__":
    unittest.main()
