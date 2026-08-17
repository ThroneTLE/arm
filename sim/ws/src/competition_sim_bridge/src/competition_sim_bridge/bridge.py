"""Direct adapters around the upstream ROS interfaces; no robot model is duplicated."""

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class RobotPoseSample:
    base_from_tcp: np.ndarray
    timestamp_s: float


@dataclass
class ObjectPoseSample:
    object_id: str
    base_from_object: np.ndarray
    timestamp_s: float


def _matrix_from_pose(pose):
    from tf.transformations import quaternion_matrix

    matrix = quaternion_matrix(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def _pose_from_matrix(matrix):
    from geometry_msgs.msg import Pose
    from tf.transformations import quaternion_from_matrix

    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    quaternion = quaternion_from_matrix(matrix)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = matrix[:3, 3]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion
    return pose


class MoveItRobotController:
    """RobotController semantics in base_link, backed by MoveIt move_group."""

    def __init__(
        self, group_name="manipulator", base_frame="base_link", tcp_link="tool0",
        planning_time_s=20.0, planning_attempts=8,
    ):
        import moveit_commander
        import rospy
        import tf2_ros

        moveit_commander.roscpp_initialize(sys.argv)
        self._rospy = rospy
        self._tf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self._listener = tf2_ros.TransformListener(self._tf)
        self._group = moveit_commander.MoveGroupCommander(group_name)
        self.base_frame = str(base_frame)
        self.tcp_link = str(tcp_link)
        self._group.set_pose_reference_frame(self.base_frame)
        self._group.set_end_effector_link(self.tcp_link)
        self._group.set_planning_time(float(planning_time_s))
        self._group.set_num_planning_attempts(int(planning_attempts))
        self._group.set_goal_position_tolerance(0.005)
        self._group.set_goal_orientation_tolerance(0.05)

    def latest_pose(self):
        try:
            transform = self._tf.lookup_transform(
                self.base_frame, self.tcp_link, self._rospy.Time(0), self._rospy.Duration(0.25)
            ).transform
        except Exception:
            return None
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
        from tf.transformations import quaternion_matrix

        matrix[:3, :3] = quaternion_matrix([
            transform.rotation.x, transform.rotation.y,
            transform.rotation.z, transform.rotation.w,
        ])[:3, :3]
        return RobotPoseSample(matrix, time.monotonic())

    def move_tcp(self, base_from_tcp, speed_scale):
        scale = float(speed_scale)
        self._group.set_max_velocity_scaling_factor(scale)
        self._group.set_max_acceleration_scaling_factor(scale)
        self._group.set_start_state_to_current_state()
        try:
            # This upstream config uses a 5 ms KDL timeout. Pose-goal sampling
            # often produces zero goal states, while the same pose resolves
            # reliably when converted to a joint goal with approximate IK.
            self._group.set_joint_value_target(
                _pose_from_matrix(base_from_tcp), self.tcp_link, True
            )
        except Exception:
            self._group.clear_pose_targets()
            return False
        try:
            return bool(self._group.go(wait=True))
        finally:
            self._group.stop()
            self._group.clear_pose_targets()

    def stop(self):
        self._group.stop()
        self._group.clear_pose_targets()
        return True


class RobotiqGripperController:
    """Robotiq 2F-85 adapter for the upstream GripperCommand controller."""

    def __init__(
        self,
        action_name="/robotiq_2f_85_gripper_controller/gripper_cmd",
        open_joint_position=0.0,
        closed_joint_position=0.3,
        timeout_s=25.0,
    ):
        import actionlib
        import rospy
        from control_msgs.msg import GripperCommandAction

        self._rospy = rospy
        self._client = actionlib.SimpleActionClient(action_name, GripperCommandAction)
        self.open_joint_position = float(open_joint_position)
        self.closed_joint_position = float(closed_joint_position)
        self.timeout_s = float(timeout_s)
        if not self._client.wait_for_server(rospy.Duration(self.timeout_s)):
            raise RuntimeError("Robotiq gripper action is unavailable")

    def _command(self, position, effort, accept_stall=False):
        from control_msgs.msg import GripperCommandGoal

        goal = GripperCommandGoal()
        goal.command.position = float(position)
        goal.command.max_effort = float(effort)
        self._client.send_goal(goal)
        if not self._client.wait_for_result(self._rospy.Duration(self.timeout_s)):
            # The upstream controller has no stall_timeout/goal_tolerance
            # parameters. Closing on an object therefore remains ACTIVE even
            # though contact is the desired outcome. The executor's mandatory
            # object-lift check is the authoritative success criterion.
            if accept_stall:
                return True
            self._client.cancel_goal()
            return False
        return self._client.get_state() == 3

    def open(self):
        return self._command(self.open_joint_position, 20.0)

    def close(self, width_m, maximum_effort=None):
        # Width feasibility is checked by the generic planner. The upstream
        # controller exposes the Robotiq knuckle joint, not metres of opening.
        return self._command(
            self.closed_joint_position,
            60.0 if maximum_effort is None else float(maximum_effort),
            accept_stall=True,
        )

    def stop(self):
        self._client.cancel_all_goals()
        return True


class GazeboObjectPoseProvider:
    """Read object link truth and express it in the robot base frame."""

    def __init__(
        self,
        object_links=None,
        topic="/gazebo/link_states",
        base_frame="base_link",
        world_frame="world",
        gazebo_base_link="robot::base_link",
    ):
        import rospy
        import tf2_ros
        from gazebo_msgs.msg import LinkStates

        self._rospy = rospy
        self.base_frame = str(base_frame)
        self.world_frame = str(world_frame)
        self.gazebo_base_link = str(gazebo_base_link)
        self.object_links = dict(object_links or {"competition_object": "competition_object::link"})
        self._latest = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._tf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self._listener = tf2_ros.TransformListener(self._tf)
        self._subscriber = rospy.Subscriber(topic, LinkStates, self._callback, queue_size=1)

    def _callback(self, message):
        with self._lock:
            self._latest = message
            self._stamp = time.monotonic()

    def latest_object_pose(self, object_id):
        link_name = self.object_links.get(str(object_id), str(object_id))
        with self._lock:
            message, stamp = self._latest, self._stamp
        if message is None or link_name not in message.name:
            return None
        if self.gazebo_base_link not in message.name:
            return None
        world_from_object = _matrix_from_pose(message.pose[message.name.index(link_name)])
        # The upstream robot is spawned at world Z=1.02, but its MoveIt virtual
        # joint publishes world->base_link as identity. Gazebo link truth is the
        # authoritative transform for converting object world poses to the
        # RobotController's base_link contract.
        world_from_base = _matrix_from_pose(
            message.pose[message.name.index(self.gazebo_base_link)]
        )
        base_from_object = np.linalg.inv(world_from_base) @ world_from_object
        return ObjectPoseSample(str(object_id), base_from_object, stamp)
