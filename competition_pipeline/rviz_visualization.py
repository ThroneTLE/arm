"""ROS/RViz publishing boundary for segmented objects and future grasp plans."""

import json
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from tool.object_model_builder.camera_source import native_ros_environment

from .tag_map import TagMap


POINT_COLORS = (
    (255, 92, 92),
    (72, 220, 130),
    (70, 170, 255),
    (255, 190, 55),
    (190, 105, 255),
    (55, 220, 220),
)


def ros_master_available() -> bool:
    try:
        import rosgraph

        rosgraph.Master("/competition_pipeline_ui").getPid()
        return True
    except Exception:
        return False


def ensure_ros_master(timeout_s: float = 5.0):
    """Return a roscore process only when this UI had to start one."""
    if ros_master_available():
        return None
    if shutil.which("roscore") is None:
        raise RuntimeError("未找到 roscore；请安装或 source ROS Noetic")
    process = subprocess.Popen(
        ["roscore"],
        env=native_ros_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("roscore 启动后立即退出")
        if ros_master_available():
            return process
        time.sleep(0.1)
    process.terminate()
    raise RuntimeError("等待 roscore 超时")


def launch_rviz(config_path):
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("RViz 配置不存在：{}".format(path))
    if shutil.which("rviz") is None:
        raise RuntimeError("未找到 rviz；请安装或 source ROS Noetic")
    return subprocess.Popen(
        ["rviz", "-d", str(path)],
        env=native_ros_environment(),
        start_new_session=True,
    )


def stop_process(process, timeout_s=3.0):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        process.kill()


class CompetitionRvizVisualizer:
    def __init__(self, config):
        import rospy
        import sensor_msgs.point_cloud2 as point_cloud2
        import tf2_ros
        from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, TransformStamped
        from nav_msgs.msg import Path
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
        from std_msgs.msg import Header, String
        from tf.transformations import quaternion_from_matrix
        from visualization_msgs.msg import Marker, MarkerArray

        if not rospy.core.is_initialized():
            rospy.init_node(
                "competition_pipeline_rviz",
                anonymous=True,
                disable_signals=True,
            )
        self.rospy = rospy
        self.point_cloud2 = point_cloud2
        self.Point = Point
        self.Pose = Pose
        self.PoseArray = PoseArray
        self.PoseStamped = PoseStamped
        self.TransformStamped = TransformStamped
        self.Path = Path
        self.PointField = PointField
        self.CameraInfo = CameraInfo
        self.Image = Image
        self.Header = Header
        self.String = String
        self.Marker = Marker
        self.MarkerArray = MarkerArray
        self.quaternion_from_matrix = quaternion_from_matrix
        self.base_frame = str(config.data.get("frames", {}).get("base", "robot_base"))
        visualization = config.data.get("planning_validation", {}).get(
            "visualization", {}
        )
        # Do not reuse the frame name published by the physical camera driver.
        # A TF child may only have one parent; Astra commonly publishes
        # camera_color_optical_frame below camera_link, while this visualizer
        # needs the localized optical frame below robot_base.
        self.camera_frame = str(
            visualization.get(
                "localized_camera_frame",
                "competition_camera_color_optical_frame",
            )
        )
        self.topic_prefix = str(
            visualization.get("topic_prefix", "/competition_pipeline")
        ).rstrip("/")
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self._camera_transform_lock = threading.Lock()
        self._last_base_from_camera = None
        tf_rate_hz = max(
            1.0, float(visualization.get("tf_publish_rate_hz", 10.0))
        )
        self._tf_timer = rospy.Timer(
            rospy.Duration(1.0 / tf_rate_hz), self._republish_camera_transform
        )
        self.tag_publisher = rospy.Publisher(
            self.topic_prefix + "/tag_markers", MarkerArray, queue_size=1, latch=True
        )
        self.object_marker_publisher = rospy.Publisher(
            self.topic_prefix + "/object_markers", MarkerArray, queue_size=1
        )
        self.cloud_publisher = rospy.Publisher(
            self.topic_prefix + "/object_cloud", PointCloud2, queue_size=1
        )
        self.camera_pose_publisher = rospy.Publisher(
            self.topic_prefix + "/camera_pose", PoseStamped, queue_size=1
        )
        self.camera_image_publisher = rospy.Publisher(
            self.topic_prefix + "/camera/image", Image, queue_size=1
        )
        self.camera_info_publisher = rospy.Publisher(
            self.topic_prefix + "/camera/camera_info", CameraInfo, queue_size=1
        )
        self.object_pose_publisher = rospy.Publisher(
            self.topic_prefix + "/object_poses", PoseArray, queue_size=1
        )
        self.grasp_pose_publisher = rospy.Publisher(
            self.topic_prefix + "/grasp_candidates", PoseArray, queue_size=1
        )
        self.path_publisher = rospy.Publisher(
            self.topic_prefix + "/planned_path", Path, queue_size=1, latch=True
        )
        self.status_publisher = rospy.Publisher(
            self.topic_prefix + "/status", String, queue_size=5
        )
        self.publish_tag_scene(TagMap(config))

    def close(self):
        """Stop the TF keepalive timer when RViz visualization is closed."""
        timer = self._tf_timer
        self._tf_timer = None
        if timer is not None:
            timer.shutdown()

    def _publish_camera_transform(self, transform, stamp):
        quaternion = self.quaternion_from_matrix(transform)
        tf_message = self.TransformStamped()
        tf_message.header.stamp = stamp
        tf_message.header.frame_id = self.base_frame
        tf_message.child_frame_id = self.camera_frame
        tf_message.transform.translation.x = float(transform[0, 3])
        tf_message.transform.translation.y = float(transform[1, 3])
        tf_message.transform.translation.z = float(transform[2, 3])
        tf_message.transform.rotation.x = float(quaternion[0])
        tf_message.transform.rotation.y = float(quaternion[1])
        tf_message.transform.rotation.z = float(quaternion[2])
        tf_message.transform.rotation.w = float(quaternion[3])
        self.tf_broadcaster.sendTransform(tf_message)

    def _republish_camera_transform(self, _event):
        """Keep the last valid camera TF alive between slow inference frames."""
        with self._camera_transform_lock:
            transform = (
                None
                if self._last_base_from_camera is None
                else self._last_base_from_camera.copy()
            )
        if transform is not None:
            self._publish_camera_transform(transform, self.rospy.Time.now())

    def _marker(self, namespace, marker_id, marker_type, stamp):
        marker = self.Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.base_frame
        marker.ns = str(namespace)
        marker.id = int(marker_id)
        marker.type = int(marker_type)
        marker.action = self.Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _pose(self, transform):
        transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        quaternion = self.quaternion_from_matrix(transform)
        pose = self.Pose()
        pose.position.x = float(transform[0, 3])
        pose.position.y = float(transform[1, 3])
        pose.position.z = float(transform[2, 3])
        pose.orientation.x = float(quaternion[0])
        pose.orientation.y = float(quaternion[1])
        pose.orientation.z = float(quaternion[2])
        pose.orientation.w = float(quaternion[3])
        return pose

    def publish_tag_scene(self, tag_map):
        stamp = self.rospy.Time.now()
        markers = []
        for tag_id in tag_map.ids:
            transform = tag_map.base_from_tag(tag_id)
            center_local = np.asarray(
                [tag_map.tag_size_m / 2.0, tag_map.tag_size_m / 2.0, 0.0, 1.0]
            )
            center = (transform @ center_local)[:3]
            quaternion = self.quaternion_from_matrix(transform)
            plate = self._marker("tags", tag_id, self.Marker.CUBE, stamp)
            plate.pose.position.x = float(center[0])
            plate.pose.position.y = float(center[1])
            plate.pose.position.z = float(center[2])
            plate.pose.orientation.x = float(quaternion[0])
            plate.pose.orientation.y = float(quaternion[1])
            plate.pose.orientation.z = float(quaternion[2])
            plate.pose.orientation.w = float(quaternion[3])
            plate.scale.x = tag_map.tag_size_m
            plate.scale.y = tag_map.tag_size_m
            plate.scale.z = 0.002
            plate.color.r = 0.10
            plate.color.g = 0.72
            plate.color.b = 0.56
            plate.color.a = 0.78
            markers.append(plate)

            label = self._marker(
                "tag_labels", 1000 + tag_id, self.Marker.TEXT_VIEW_FACING, stamp
            )
            label.pose.position.x = float(center[0])
            label.pose.position.y = float(center[1])
            label.pose.position.z = float(center[2] + 0.025)
            label.scale.z = 0.022
            label.color.r = label.color.g = label.color.b = 0.95
            label.color.a = 1.0
            label.text = "Tag {} · BR origin".format(tag_id)
            markers.append(label)
        origin = self._marker("base_origin", 2000, self.Marker.SPHERE, stamp)
        origin.scale.x = origin.scale.y = origin.scale.z = 0.018
        origin.color.r = 1.0
        origin.color.g = 0.8
        origin.color.a = 1.0
        markers.append(origin)
        self.tag_publisher.publish(self.MarkerArray(markers=markers))

    def publish_observation(
        self,
        base_from_camera,
        objects,
        diagnostics=None,
        annotated_bgr=None,
        camera_intrinsics=None,
    ):
        stamp = self.rospy.Time.now()
        self.publish_camera_pose(base_from_camera, stamp=stamp)
        if annotated_bgr is not None and camera_intrinsics is not None:
            self._publish_camera_image(
                annotated_bgr, camera_intrinsics, stamp
            )

        valid_objects = [item for item in objects if item.valid]
        poses = self.PoseArray()
        poses.header.stamp = stamp
        poses.header.frame_id = self.base_frame
        markers = []
        markers.append(self._camera_frustum(stamp))
        cloud_rows = []
        for index, item in enumerate(valid_objects):
            color = POINT_COLORS[index % len(POINT_COLORS)]
            center = np.asarray(item.center_base_m, dtype=np.float64)
            minimum = np.asarray(item.bounds_min_base_m, dtype=np.float64)
            maximum = np.asarray(item.bounds_max_base_m, dtype=np.float64)
            object_transform = np.eye(4, dtype=np.float64)
            object_transform[:3, 3] = center
            poses.poses.append(self._pose(object_transform))

            bounds = self._marker("object_bounds", index, self.Marker.CUBE, stamp)
            bounds_center = (minimum + maximum) * 0.5
            bounds.pose.position.x = float(bounds_center[0])
            bounds.pose.position.y = float(bounds_center[1])
            bounds.pose.position.z = float(bounds_center[2])
            scale = np.maximum(maximum - minimum, 0.005)
            bounds.scale.x, bounds.scale.y, bounds.scale.z = map(float, scale)
            bounds.color.r = color[0] / 255.0
            bounds.color.g = color[1] / 255.0
            bounds.color.b = color[2] / 255.0
            bounds.color.a = 0.22
            markers.append(bounds)

            if item.support_constrained:
                # A separate, opaque-enough footprint makes the distinction
                # clear in RViz: colored points are measured Depth, while the
                # footprint/bounds encode the known table-support constraint.
                footprint = self._marker(
                    "object_support_footprints", 3000 + index,
                    self.Marker.CUBE, stamp,
                )
                footprint.pose.position.x = float(bounds_center[0])
                footprint.pose.position.y = float(bounds_center[1])
                footprint.pose.position.z = float(item.support_plane_z_m + 0.0015)
                footprint.scale.x = float(max(scale[0], 0.012))
                footprint.scale.y = float(max(scale[1], 0.012))
                footprint.scale.z = 0.003
                footprint.color.r = color[0] / 255.0
                footprint.color.g = color[1] / 255.0
                footprint.color.b = color[2] / 255.0
                footprint.color.a = 0.72
                markers.append(footprint)

            center_marker = self._marker(
                "object_centers", 1000 + index, self.Marker.SPHERE, stamp
            )
            center_marker.pose.position.x = float(center[0])
            center_marker.pose.position.y = float(center[1])
            center_marker.pose.position.z = float(center[2])
            center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = 0.018
            center_marker.color.r = color[0] / 255.0
            center_marker.color.g = color[1] / 255.0
            center_marker.color.b = color[2] / 255.0
            center_marker.color.a = 1.0
            markers.append(center_marker)

            label = self._marker(
                "object_labels", 2000 + index, self.Marker.TEXT_VIEW_FACING, stamp
            )
            label.pose.position.x = float(center[0])
            label.pose.position.y = float(center[1])
            label.pose.position.z = float(maximum[2] + 0.035)
            label.scale.z = 0.026
            label.color.r = label.color.g = label.color.b = 0.98
            label.color.a = 1.0
            support_label = " · support Z {:.0f} mm".format(
                item.support_plane_z_m * 1000.0
            ) if item.support_constrained else ""
            label.text = "{} {:.0f}%\nXYZ ({:.1f}, {:.1f}, {:.1f}) mm · depth {:.0f}%{}".format(
                item.class_name,
                item.confidence * 100.0,
                *(center * 1000.0),
                item.depth_coverage * 100.0,
                support_label,
            )
            markers.append(label)
            packed = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
            rgb_float = struct.unpack("f", struct.pack("I", packed))[0]
            for point in np.asarray(item.points_base_m).reshape(-1, 3):
                cloud_rows.append((float(point[0]), float(point[1]), float(point[2]), rgb_float))

        delete_all = self._marker("objects", 0, self.Marker.CUBE, stamp)
        delete_all.action = self.Marker.DELETEALL
        self.object_marker_publisher.publish(
            self.MarkerArray(markers=[delete_all] + markers)
        )
        self.object_pose_publisher.publish(poses)
        self._publish_cloud(cloud_rows, stamp)
        status = dict(diagnostics or {})
        status.update({
            "valid_object_count": len(valid_objects),
            "objects": [
                {
                    "class": item.class_name,
                    "confidence": float(item.confidence),
                    "center_base_mm": (
                        np.asarray(item.center_base_m) * 1000.0
                    ).tolist(),
                    "depth_coverage": float(item.depth_coverage),
                    "point_count": int(item.valid_point_count),
                    "support_constrained": bool(item.support_constrained),
                    "observed_z_range_mm": (
                        [
                            float(item.observed_bounds_min_base_m[2] * 1000.0),
                            float(item.observed_bounds_max_base_m[2] * 1000.0),
                        ]
                        if item.observed_bounds_min_base_m is not None else None
                    ),
                }
                for item in valid_objects
            ],
            "rejected_objects": [item.reason for item in objects if not item.valid],
        })
        self.status_publisher.publish(
            self.String(data=json.dumps(status, ensure_ascii=False))
        )

    def publish_camera_pose(self, base_from_camera, stamp=None):
        """Publish lightweight localized camera state independently of clouds."""
        stamp = self.rospy.Time.now() if stamp is None else stamp
        transform = np.asarray(base_from_camera, dtype=np.float64).reshape(4, 4)
        with self._camera_transform_lock:
            self._last_base_from_camera = transform.copy()
        self._publish_camera_transform(transform, stamp)
        camera_pose = self.PoseStamped()
        camera_pose.header.stamp = stamp
        camera_pose.header.frame_id = self.base_frame
        camera_pose.pose = self._pose(transform)
        self.camera_pose_publisher.publish(camera_pose)

    def publish_camera_image(self, annotated_bgr, camera_intrinsics, stamp=None):
        """Publish the RGB/YOLO view without waiting for point-cloud work."""
        stamp = self.rospy.Time.now() if stamp is None else stamp
        self._publish_camera_image(annotated_bgr, camera_intrinsics, stamp)

    def _camera_frustum(self, stamp):
        marker = self._marker("camera_frustum", 9000, self.Marker.LINE_LIST, stamp)
        marker.header.frame_id = self.camera_frame
        marker.scale.x = 0.004
        marker.color.r = 0.18
        marker.color.g = 0.65
        marker.color.b = 1.0
        marker.color.a = 1.0
        origin = (0.0, 0.0, 0.0)
        corners = (
            (-0.12, -0.07, 0.18),
            (0.12, -0.07, 0.18),
            (0.12, 0.07, 0.18),
            (-0.12, 0.07, 0.18),
        )
        segments = [(origin, corner) for corner in corners]
        segments.extend(
            (corners[index], corners[(index + 1) % 4]) for index in range(4)
        )
        for start, end in segments:
            marker.points.append(self.Point(*start))
            marker.points.append(self.Point(*end))
        return marker

    def _publish_camera_image(self, annotated_bgr, intrinsics, stamp):
        image = np.ascontiguousarray(annotated_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RViz camera image must be BGR8")
        message = self.Image()
        message.header.stamp = stamp
        message.header.frame_id = self.camera_frame
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = int(image.shape[1] * 3)
        message.data = image.tobytes()
        self.camera_image_publisher.publish(message)

        matrix = np.asarray(intrinsics.matrix, dtype=np.float64).reshape(3, 3)
        info = self.CameraInfo()
        info.header = message.header
        info.height = int(intrinsics.height)
        info.width = int(intrinsics.width)
        info.distortion_model = "plumb_bob"
        info.D = np.asarray(intrinsics.distortion, dtype=np.float64).reshape(-1).tolist()
        info.K = matrix.reshape(-1).tolist()
        info.R = np.eye(3, dtype=np.float64).reshape(-1).tolist()
        projection = np.zeros((3, 4), dtype=np.float64)
        projection[:3, :3] = matrix
        info.P = projection.reshape(-1).tolist()
        self.camera_info_publisher.publish(info)

    def _publish_cloud(self, cloud_rows, stamp):
        fields = [
            self.PointField("x", 0, self.PointField.FLOAT32, 1),
            self.PointField("y", 4, self.PointField.FLOAT32, 1),
            self.PointField("z", 8, self.PointField.FLOAT32, 1),
            self.PointField("rgb", 12, self.PointField.FLOAT32, 1),
        ]
        header = self.Header(stamp=stamp, frame_id=self.base_frame)
        self.cloud_publisher.publish(
            self.point_cloud2.create_cloud(header, fields, cloud_rows)
        )

    def publish_invalid(self, reason, diagnostics=None):
        stamp = self.rospy.Time.now()
        delete_all = self._marker("objects", 0, self.Marker.CUBE, stamp)
        delete_all.action = self.Marker.DELETEALL
        self.object_marker_publisher.publish(self.MarkerArray(markers=[delete_all]))
        self._publish_cloud([], stamp)
        poses = self.PoseArray()
        poses.header.stamp = stamp
        poses.header.frame_id = self.base_frame
        self.object_pose_publisher.publish(poses)
        status = dict(diagnostics or {})
        status.update({"valid_object_count": 0, "reason": str(reason)})
        self.status_publisher.publish(
            self.String(data=json.dumps(status, ensure_ascii=False))
        )

    def publish_grasp_candidates(self, base_from_grasps):
        """Stable future-planner boundary: sequence of 4x4 base-frame poses."""
        stamp = self.rospy.Time.now()
        poses = self.PoseArray()
        poses.header.stamp = stamp
        poses.header.frame_id = self.base_frame
        poses.poses = [self._pose(transform) for transform in base_from_grasps]
        self.grasp_pose_publisher.publish(poses)

    def publish_planned_path(self, base_from_tcp_waypoints):
        """Stable future-planner boundary: ordered Cartesian TCP waypoints."""
        stamp = self.rospy.Time.now()
        path = self.Path()
        path.header.stamp = stamp
        path.header.frame_id = self.base_frame
        for transform in base_from_tcp_waypoints:
            pose = self.PoseStamped()
            pose.header = path.header
            pose.pose = self._pose(transform)
            path.poses.append(pose)
        self.path_publisher.publish(path)


__all__ = [
    "CompetitionRvizVisualizer",
    "ensure_ros_master",
    "launch_rviz",
    "ros_master_available",
    "stop_process",
]
