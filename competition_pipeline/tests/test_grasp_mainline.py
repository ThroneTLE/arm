import time
import unittest
from types import SimpleNamespace

import numpy as np

from competition_pipeline.control import SafeRobotController
from competition_pipeline.execution import GraspExecutor
from competition_pipeline.geometry import interpolate_transforms
from competition_pipeline.interfaces import ObjectPoseSample, RobotPoseSample
from competition_pipeline.object_localization import SegmentedObjectCloud
from competition_pipeline.planning import (
    GraspPlanningError, TopDownGraspPlanner, TopDownPlannerSettings,
)


def _cloud(dimensions=(0.04, 0.06, 0.10)):
    center = np.asarray([0.45, 0.10, 0.10])
    half = np.asarray(dimensions) * 0.5
    return SegmentedObjectCloud(
        True,
        class_name="cube",
        center_base_m=center,
        bounds_min_base_m=center - half,
        bounds_max_base_m=center + half,
        points_base_m=np.asarray([center]),
        valid_point_count=1,
    )


def _planner(maximum_width=0.075):
    return TopDownGraspPlanner(
        TopDownPlannerSettings(
            tcp_from_grasp=np.eye(4),
            maximum_grasp_width_m=maximum_width,
            pregrasp_clearance_m=0.10,
            lift_distance_m=0.10,
            place_clearance_m=0.10,
        )
    )


class _RobotAdapter:
    def __init__(self, pose):
        self.pose = pose.copy()
        self.stopped = False

    def latest_pose(self):
        return RobotPoseSample(self.pose, time.monotonic())

    def move_tcp(self, target, speed_scale):
        self.pose = np.asarray(target).copy()
        return True

    def stop(self):
        self.stopped = True


class _Gripper:
    def __init__(self):
        self.commands = []
        self.stopped = False

    def open(self):
        self.commands.append("open")
        return True

    def close(self, width_m, maximum_effort=None):
        self.commands.append(("close", width_m))
        return True

    def stop(self):
        self.stopped = True


class _ObjectProvider:
    def __init__(self, positions):
        self.positions = list(positions)

    def latest_object_pose(self, object_id):
        position = self.positions.pop(0) if len(self.positions) > 1 else self.positions[0]
        pose = np.eye(4)
        pose[:3, 3] = position
        return ObjectPoseSample(object_id, pose, time.monotonic())


def _safe_controller(adapter):
    return SafeRobotController(
        SimpleNamespace(data={
            "safety": {
                "dry_run": False,
                "allow_robot_motion": True,
                "workspace_min_mm": [-1000, -1000, -1000],
                "workspace_max_mm": [2000, 2000, 2000],
                "maximum_single_step_mm": 50,
                "maximum_single_rotation_deg": 10,
                "maximum_speed_scale": 0.2,
                "maximum_robot_pose_age_s": 0.25,
            }
        }),
        adapter,
    )


class GraspMainlineTest(unittest.TestCase):
    def test_top_down_uses_negative_base_z(self):
        target = _planner().target_from_object(_cloud())
        np.testing.assert_allclose(target.base_from_grasp[:3, 2], [0, 0, -1])
        self.assertAlmostEqual(target.width_m, 0.046)

    def test_too_wide_object_is_rejected(self):
        with self.assertRaisesRegex(GraspPlanningError, "exceeds"):
            _planner(maximum_width=0.05).target_from_object(_cloud((0.06, 0.08, 0.1)))

    def test_invalid_object_is_rejected(self):
        invalid = _cloud()
        invalid.valid = False
        invalid.reason = "outside workspace"
        with self.assertRaisesRegex(GraspPlanningError, "outside workspace"):
            _planner().target_from_object(invalid)

    def test_interpolation_obeys_translation_and_rotation_limits(self):
        start = np.eye(4)
        target = np.eye(4)
        target[:3, 3] = [0.12, 0, 0]
        target[:3, :3] = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        poses = interpolate_transforms(start, target, 0.04, 30.0)
        previous = start
        for pose in poses:
            self.assertLessEqual(np.linalg.norm(pose[:3, 3] - previous[:3, 3]), 0.040001)
            previous = pose
        np.testing.assert_allclose(poses[-1], target, atol=1e-9)

    def test_failed_lift_stops_before_place(self):
        plan = _planner().plan(
            _planner().target_from_object(_cloud()), [0.55, -0.10, 0.10]
        )
        adapter = _RobotAdapter(plan.base_from_tcp_pregrasp)
        gripper = _Gripper()
        provider = _ObjectProvider([[0.45, 0.1, 0.1], [0.45, 0.1, 0.11]])
        result = GraspExecutor(
            _safe_controller(adapter), gripper, provider,
            minimum_verified_lift_mm=30,
        ).execute(plan)
        self.assertFalse(result.success)
        self.assertEqual(result.state, "verifying_lift")
        self.assertTrue(adapter.stopped)
        self.assertTrue(gripper.stopped)

    def test_stale_object_pose_fails_before_motion(self):
        planner = _planner()
        plan = planner.plan(planner.target_from_object(_cloud()), [0.55, -0.10, 0.10])
        adapter = _RobotAdapter(plan.base_from_tcp_pregrasp)
        gripper = _Gripper()

        class StaleProvider:
            def latest_object_pose(self, object_id):
                pose = np.eye(4)
                pose[:3, 3] = [0.45, 0.10, 0.10]
                return ObjectPoseSample(object_id, pose, time.monotonic() - 2.0)

        result = GraspExecutor(
            _safe_controller(adapter), gripper, StaleProvider(),
            maximum_object_pose_age_s=0.1,
        ).execute(plan)
        self.assertFalse(result.success)
        self.assertIn("stale", result.reason)
        self.assertEqual(gripper.commands, [])

    def test_workspace_rejection_stops_execution(self):
        planner = _planner()
        plan = planner.plan(planner.target_from_object(_cloud()), [3.0, 0.0, 0.1])
        adapter = _RobotAdapter(plan.base_from_tcp_pregrasp)
        gripper = _Gripper()
        result = GraspExecutor(
            _safe_controller(adapter), gripper,
            _ObjectProvider([[0.45, 0.10, 0.10], [0.45, 0.10, 0.20]]),
        ).execute(plan)
        self.assertFalse(result.success)
        self.assertIn("outside workspace", result.reason)
        self.assertTrue(adapter.stopped)

    def test_successful_pick_place_has_closed_loop_events(self):
        place = np.asarray([0.55, -0.10, 0.10])
        planner = _planner()
        plan = planner.plan(planner.target_from_object(_cloud()), place)
        adapter = _RobotAdapter(plan.base_from_tcp_pregrasp)
        gripper = _Gripper()
        provider = _ObjectProvider([
            [0.45, 0.10, 0.10],
            [0.45, 0.10, 0.20],
            place,
        ])
        result = GraspExecutor(
            _safe_controller(adapter), gripper, provider,
            minimum_verified_lift_mm=30,
        ).execute(plan)
        self.assertTrue(result.success, result.reason)
        self.assertEqual(result.state, "complete")
        self.assertIn("verifying_lift", [event.state for event in result.events])
        self.assertIn("verifying_place", [event.state for event in result.events])

    def test_cancel_stops_robot_and_gripper(self):
        planner = _planner()
        plan = planner.plan(planner.target_from_object(_cloud()), [0.55, -0.10, 0.10])
        adapter = _RobotAdapter(plan.base_from_tcp_pregrasp)
        gripper = _Gripper()
        provider = _ObjectProvider([[0.45, 0.10, 0.10]])
        result = GraspExecutor(
            _safe_controller(adapter), gripper, provider,
            cancel_requested=lambda: True,
        ).execute(plan)
        self.assertFalse(result.success)
        self.assertTrue(adapter.stopped)
        self.assertTrue(gripper.stopped)


if __name__ == "__main__":
    unittest.main()
