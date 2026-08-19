#!/usr/bin/env python3
"""ROS node: AnyGrasp grasp planning on the bottle_localization object cloud.

Inputs (published by tool/bottle_localization and tool/camera_calibration):
    /bottle_localization/object_cloud   PointCloud2, workspace frame
    /camera_calibration/camera_pose     PoseStamped = workspace_from_camera

Outputs (all in the workspace frame of the input cloud):
    /grasp_planning/grasps              MarkerArray of gripper candidates
    /grasp_planning/best_grasp          PoseStamped of the best candidate
    /grasp_planning/status              JSON status string

The AnyGrasp SDK is loaded lazily; while the license/checkpoint is missing the
node stays alive, publishes a readable status, and retries periodically.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

from tool.grasp_planning.anygrasp_planner import (
    AnyGraspPlanner,
    filter_by_score,
    filter_by_width,
    filter_top_down,
    transform_to_workspace,
)
from tool.grasp_planning.grasp_markers import (
    build_best_grasp_label,
    build_gripper_markers,
    delete_all_markers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "grasp_planning.yaml"


def load_config(path: str) -> dict:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict) or int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported grasp planning config: {}".format(source))
    return data


def pose_to_matrix(pose_message) -> np.ndarray:
    """PoseStamped -> 4x4 transform whose frame is the message header frame."""
    from tf.transformations import quaternion_matrix

    quaternion = pose_message.pose.orientation
    matrix = np.asarray(
        quaternion_matrix([quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
        dtype=np.float64,
    ).reshape(4, 4)
    matrix[0, 3] = float(pose_message.pose.position.x)
    matrix[1, 3] = float(pose_message.pose.position.y)
    matrix[2, 3] = float(pose_message.pose.position.z)
    return matrix


class AnyGraspNode:
    def __init__(self, arguments):
        self.arguments = arguments
        self.config = load_config(arguments.config)
        self.config_path = Path(arguments.config).expanduser().resolve()

        planner = self.config.get("anygrasp", {})
        self.planner = AnyGraspPlanner(
            checkpoint_path=planner.get("checkpoint_path", ""),
            sdk_grasp_dir=planner.get("sdk_grasp_dir", ""),
            max_gripper_width=planner.get("max_gripper_width_m", 0.08),
            gripper_height=planner.get("gripper_height_m", 0.03),
        )
        self.planner_settings = planner
        self.runtime = self.config.get("runtime", {})
        self.topics = self.config.get("topics", {})
        self.visualization = self.config.get("visualization", {})

        self._latest_cloud = None
        self._latest_pose = None
        self._last_load_attempt_s = 0.0
        self._rviz_process = None
        self._publishers = {}

    # ------------------------------------------------------------------ ROS
    def _init_ros(self):
        import rospy
        import sensor_msgs.point_cloud2 as point_cloud2
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray

        if not rospy.core.is_initialized():
            rospy.init_node("anygrasp_grasp_planning", anonymous=True, disable_signals=True)
        self.rospy = rospy
        self.point_cloud2 = point_cloud2
        cloud_topic = self.topics.get("cloud", "/bottle_localization/object_cloud")
        pose_topic = self.topics.get("camera_pose", "/camera_calibration/camera_pose")
        rospy.Subscriber(cloud_topic, PointCloud2, self._on_cloud, queue_size=1)
        rospy.Subscriber(pose_topic, PoseStamped, self._on_pose, queue_size=1)
        self._publishers = {
            "grasps": rospy.Publisher(
                self.topics.get("grasps", "/grasp_planning/grasps"), MarkerArray, queue_size=1
            ),
            "best": rospy.Publisher(
                self.topics.get("best_grasp", "/grasp_planning/best_grasp"), PoseStamped, queue_size=1
            ),
            "status": rospy.Publisher(
                self.topics.get("status", "/grasp_planning/status"), String, queue_size=5
            ),
        }

    def _on_cloud(self, message):
        self._latest_cloud = message

    def _on_pose(self, message):
        self._latest_pose = message

    # --------------------------------------------------------------- logic
    def _workspace_frame(self) -> Optional[str]:
        if self._latest_cloud is None:
            return None
        return self._latest_cloud.header.frame_id

    def _process_once(self):
        rospy = self.rospy
        stamp = rospy.Time.now()
        now = time.monotonic()
        maximum_age = float(self.runtime.get("maximum_input_age_s", 0.6))

        if not self.planner.ready():
            retry_interval = float(self.planner_settings.get("retry_interval_s", 10.0))
            if now - self._last_load_attempt_s >= retry_interval:
                self._last_load_attempt_s = now
                rospy.logwarn("AnyGrasp 不可用：%s", self.planner.load_error)
            self._publish_status({"planner_ready": False, "load_error": self.planner.load_error})
            self._publish_cleared(stamp)
            return

        cloud = self._latest_cloud
        pose = self._latest_pose
        if cloud is None or pose is None:
            self._publish_status({"planner_ready": True, "reason": "等待 object_cloud 与 camera_pose 话题"})
            self._publish_cleared(stamp)
            return
        cloud_age = now - (cloud.header.stamp.to_sec() if cloud.header.stamp.to_sec() > 0 else now)
        pose_age = now - (pose.header.stamp.to_sec() if pose.header.stamp.to_sec() > 0 else now)
        if cloud_age > maximum_age or pose_age > maximum_age:
            self._publish_status(
                {
                    "planner_ready": True,
                    "reason": "输入数据过期 cloud={:.2f}s pose={:.2f}s".format(cloud_age, pose_age),
                }
            )
            self._publish_cleared(stamp)
            return
        if pose.header.frame_id != cloud.header.frame_id:
            self._publish_status(
                {
                    "planner_ready": True,
                    "reason": "camera_pose 帧 {} 与 object_cloud 帧 {} 不一致".format(
                        pose.header.frame_id, cloud.header.frame_id
                    ),
                }
            )
            self._publish_cleared(stamp)
            return

        started = time.monotonic()
        workspace_from_camera = pose_to_matrix(pose)
        camera_from_workspace = np.linalg.inv(workspace_from_camera)
        points_workspace = self._read_cloud(cloud)
        maximum_points = int(self.runtime.get("maximum_cloud_points", 6000))
        if len(points_workspace) > maximum_points:
            stride = int(np.ceil(len(points_workspace) / float(maximum_points)))
            points_workspace = points_workspace[::stride]
        if len(points_workspace) < 64:
            self._publish_status(
                {"planner_ready": True, "reason": "object_cloud 点数不足 ({})".format(len(points_workspace))}
            )
            self._publish_cleared(stamp)
            return

        rotation = camera_from_workspace[:3, :3]
        translation = camera_from_workspace[:3, 3]
        points_camera = (rotation @ points_workspace.T).T + translation

        # Top-down approach: workspace +Z points into the table, so a
        # downward approach direction is workspace +Z expressed in the camera.
        approach_camera = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        cone_deg = float(self.planner_settings.get("approach_cone_deg", 45.0))
        approach_thresh = float(np.deg2rad(min(max(cone_deg, 0.0), 179.0)))

        candidates = self.planner.plan(
            points_camera,
            approach_camera=approach_camera,
            approach_thresh=approach_thresh,
            dense_grasp=bool(self.planner_settings.get("dense_grasp", False)),
            collision_detection=bool(self.planner_settings.get("collision_detection", True)),
            top_k=int(self.planner_settings.get("top_k", 20)),
        )
        workspace_grasps = transform_to_workspace(candidates, workspace_from_camera)
        if bool(self.planner_settings.get("top_down_only", True)):
            workspace_grasps = filter_top_down(workspace_grasps, max_deviation_deg=cone_deg)
        workspace_grasps = filter_by_width(
            workspace_grasps,
            minimum_width=float(self.planner_settings.get("minimum_width_m", 0.0)),
            maximum_width=float(self.planner_settings.get("maximum_width_m", 0.1)),
        )
        workspace_grasps = filter_by_score(
            workspace_grasps, minimum_score=float(self.planner_settings.get("minimum_score", 0.0))
        )
        self._publish_grasps(workspace_grasps, stamp)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        self._publish_status(
            {
                "planner_ready": True,
                "cloud_points": int(len(points_workspace)),
                "input_age_s": float(min(cloud_age, pose_age)),
                "grasp_count": len(workspace_grasps),
                "best_score": None if not workspace_grasps else float(workspace_grasps[0].score),
                "top_scores": [round(float(g.score), 3) for g in workspace_grasps[:5]],
                "elapsed_ms": round(elapsed_ms, 1),
            }
        )

    def _read_cloud(self, message) -> np.ndarray:
        points = []
        for record in self.point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        ):
            points.append(record)
        if not points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def _publish_grasps(self, grasps, stamp):
        frame_id = self._workspace_frame() or "ruler_workspace"
        import visualization_msgs.msg as visualization_msgs
        from geometry_msgs.msg import PoseStamped
        from tf.transformations import quaternion_from_matrix

        if not grasps:
            self._publishers["grasps"].publish(
                visualization_msgs.MarkerArray(
                    markers=[delete_all_markers(frame_id, stamp, namespace="grasps")]
                )
            )
            return
        markers = build_gripper_markers(
            grasps,
            frame_id,
            stamp,
            max_markers=int(self.planner_settings.get("top_k", 20)),
            gripper_height=float(self.planner_settings.get("gripper_height_m", 0.03)),
        )
        markers.append(build_best_grasp_label(grasps[0], frame_id, stamp))
        self._publishers["grasps"].publish(visualization_msgs.MarkerArray(markers=markers))

        best = grasps[0]
        best_pose = PoseStamped()
        best_pose.header.stamp = stamp
        best_pose.header.frame_id = frame_id
        best_pose.pose.position.x = float(best.translation[0])
        best_pose.pose.position.y = float(best.translation[1])
        best_pose.pose.position.z = float(best.translation[2])
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = best.rotation
        quaternion = quaternion_from_matrix(matrix)
        best_pose.pose.orientation.x = float(quaternion[0])
        best_pose.pose.orientation.y = float(quaternion[1])
        best_pose.pose.orientation.z = float(quaternion[2])
        best_pose.pose.orientation.w = float(quaternion[3])
        self._publishers["best"].publish(best_pose)

    def _publish_cleared(self, stamp):
        if "grasps" not in self._publishers:
            return
        import visualization_msgs.msg as visualization_msgs

        frame_id = self._workspace_frame() or "ruler_workspace"
        self._publishers["grasps"].publish(
            visualization_msgs.MarkerArray(
                markers=[
                    delete_all_markers(frame_id, stamp, namespace="grasps"),
                    delete_all_markers(frame_id, stamp, namespace="best_grasp_label"),
                ]
            )
        )

    def _publish_status(self, status: dict):
        if "status" not in self._publishers:
            return
        from std_msgs.msg import String

        payload = {"timestamp_s": time.monotonic()}
        payload.update(status)
        self._publishers["status"].publish(String(data=json.dumps(payload, ensure_ascii=False)))

    # ------------------------------------------------------------- runtime
    def run(self) -> None:
        self._init_ros()
        if self.arguments.rviz:
            self._launch_rviz()
        self.rospy.loginfo("AnyGrasp 抓取规划节点就绪（等待 bottle_localization 输入）")
        rate = self.rospy.Rate(float(self.runtime.get("rate_hz", 2.0)))
        while not self.rospy.is_shutdown():
            try:
                self._process_once()
            except Exception as error:
                self.rospy.logerr("抓取规划循环异常：%s", error)
            rate.sleep()

    def _launch_rviz(self) -> None:
        rviz_config = Path(
            self.visualization.get("rviz_config", Path(__file__).resolve().parent / "config" / "grasp_planning.rviz")
        ).expanduser()
        if not rviz_config.is_absolute():
            rviz_config = self.config_path.parent / rviz_config
        if shutil.which("rviz") is None:
            raise RuntimeError("rviz is not installed or is not on PATH")
        # The inference node needs Conda's libraries, but RViz is a system Qt
        # application.  Inheriting Conda's libffi makes system p11-kit fail to
        # resolve `ffi_type_pointer` on this machine.
        rviz_environment = os.environ.copy()
        conda_prefix = rviz_environment.get("CONDA_PREFIX", "")
        if conda_prefix:
            library_paths = rviz_environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            system_paths = [
                path
                for path in library_paths
                if path and path != conda_prefix and not path.startswith(conda_prefix + os.sep)
            ]
            if system_paths:
                rviz_environment["LD_LIBRARY_PATH"] = os.pathsep.join(system_paths)
            else:
                rviz_environment.pop("LD_LIBRARY_PATH", None)
        self._rviz_process = subprocess.Popen(
            ["rviz", "-d", str(rviz_config.resolve())], env=rviz_environment
        )

    def shutdown(self) -> None:
        if self._rviz_process is not None and self._rviz_process.poll() is None:
            self._rviz_process.terminate()
        self._rviz_process = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AnyGrasp grasp planning on the bottle_localization object cloud"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--rviz", action="store_true", help="launch RViz with the supplied view")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    node = AnyGraspNode(arguments)
    try:
        node.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print("AnyGrasp 抓取规划启动失败：{}".format(error))
        return 1
    finally:
        node.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
