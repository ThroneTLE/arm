#!/usr/bin/env python3
"""Plan or execute one deterministic cube pick using the thin ROS adapters."""

import argparse
import copy
from pathlib import Path
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SRC = ROOT / "sim" / "ws" / "src" / "competition_sim_bridge" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BRIDGE_SRC))

from competition_pipeline.configuration import CompetitionConfig
from competition_pipeline.control import SafeRobotController
from competition_pipeline.execution import executor_from_config
from competition_pipeline.object_localization import SegmentedObjectCloud
from competition_pipeline.planning import TopDownGraspPlanner, planner_settings_from_config
from competition_sim_bridge import (
    GazeboObjectPoseProvider, MoveItRobotController, RobotiqGripperController,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute", action="store_true",
        help="send motion to the simulated robot; default is plan-only",
    )
    args = parser.parse_args()

    import rospy

    rospy.init_node("competition_demo_runner", anonymous=True)
    sim_path = ROOT / "sim" / "config" / "competition_sim.yaml"
    sim = yaml.safe_load(sim_path.read_text(encoding="utf-8"))
    config = CompetitionConfig(ROOT / "competition_pipeline" / "config" / "competition.yaml")
    # Override in memory only. The real competition file remains dry-run.
    config.data["safety"] = copy.deepcopy(sim["safety"])
    config.data["grasp_execution"].update(copy.deepcopy(sim["grasp_execution"]))

    object_id = str(sim["object_id"])
    objects = GazeboObjectPoseProvider({object_id: sim["object_link"]})
    deadline = time.monotonic() + 10.0
    sample = None
    while sample is None and time.monotonic() < deadline and not rospy.is_shutdown():
        sample = objects.latest_object_pose(object_id)
        rospy.sleep(0.05)
    if sample is None:
        raise RuntimeError("competition object truth is unavailable")

    dimensions = np.asarray(sim["object_dimensions_m"], dtype=np.float64)
    center = sample.base_from_object[:3, 3]
    cloud = SegmentedObjectCloud(
        True,
        class_name=object_id,
        center_base_m=center,
        bounds_min_base_m=center - dimensions * 0.5,
        bounds_max_base_m=center + dimensions * 0.5,
        points_base_m=np.asarray([center]),
        valid_point_count=1,
        reason="Gazebo ground truth deterministic fallback",
    )
    planner = TopDownGraspPlanner(planner_settings_from_config(config))
    target = planner.target_from_object(cloud, object_id=object_id)
    plan = planner.plan(target, sim["place_position_base_m"])
    print(
        "planned object={} width={:.1f}mm approach={} waypoints={}".format(
            object_id,
            target.width_m * 1000.0,
            target.base_from_grasp[:3, 2].tolist(),
            len(plan.ordered_waypoints),
        )
    )
    if not args.execute:
        print("plan-only complete; pass --execute to move the simulated robot")
        return 0

    robot_adapter = MoveItRobotController()
    robot = SafeRobotController(config, robot_adapter)
    gripper = RobotiqGripperController()
    result = executor_from_config(config, robot, gripper, objects).execute(plan)
    print("execution success={} state={} reason={}".format(
        result.success, result.state, result.reason
    ))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
