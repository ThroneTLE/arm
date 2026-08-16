#!/usr/bin/env python3
"""ROS publishers used by the bottle localization RViz view."""

import json
from typing import Optional

import numpy as np

from .estimator import BottleEstimate


class BottleRosVisualizer:
    def __init__(
        self,
        workspace_frame: str,
        bottle_frame: str = "bottle_estimated",
        topic_prefix: str = "/bottle_localization",
    ):
        import rospy
        import sensor_msgs.point_cloud2 as point_cloud2
        import tf2_ros
        from geometry_msgs.msg import Point, PoseStamped, TransformStamped
        from sensor_msgs.msg import Image, PointCloud2
        from std_msgs.msg import Header, String
        from tf.transformations import quaternion_from_matrix
        from visualization_msgs.msg import Marker, MarkerArray

        self.rospy = rospy
        self.point_cloud2 = point_cloud2
        self.Point = Point
        self.PoseStamped = PoseStamped
        self.TransformStamped = TransformStamped
        self.Image = Image
        self.PointCloud2 = PointCloud2
        self.Header = Header
        self.String = String
        self.Marker = Marker
        self.MarkerArray = MarkerArray
        self.quaternion_from_matrix = quaternion_from_matrix
        self.workspace_frame = str(workspace_frame)
        self.bottle_frame = str(bottle_frame)
        self.topic_prefix = str(topic_prefix).rstrip("/")
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pose_publisher = rospy.Publisher(
            self.topic_prefix + "/bottle_pose", PoseStamped, queue_size=1
        )
        self.base_pose_publisher = rospy.Publisher(
            self.topic_prefix + "/bottle_base_pose", PoseStamped, queue_size=1
        )
        self.cloud_publisher = rospy.Publisher(
            self.topic_prefix + "/object_cloud", PointCloud2, queue_size=1
        )
        self.marker_publisher = rospy.Publisher(
            self.topic_prefix + "/markers", MarkerArray, queue_size=1
        )
        self.annotated_publisher = rospy.Publisher(
            self.topic_prefix + "/annotated_image", Image, queue_size=1
        )
        self.mask_publisher = rospy.Publisher(
            self.topic_prefix + "/mask", Image, queue_size=1
        )
        self.depth_publisher = rospy.Publisher(
            self.topic_prefix + "/aligned_depth_preview", Image, queue_size=1
        )
        self.status_publisher = rospy.Publisher(
            self.topic_prefix + "/status", String, queue_size=5
        )

    def publish(
        self,
        estimate: BottleEstimate,
        annotated_bgr: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        depth_preview_bgr: Optional[np.ndarray] = None,
        diagnostics: Optional[dict] = None,
    ) -> None:
        stamp = self.rospy.Time.now()
        if annotated_bgr is not None:
            self.annotated_publisher.publish(
                self._image_message(annotated_bgr, "bgr8", stamp)
            )
        if mask is not None:
            mask_u8 = (np.asarray(mask).astype(bool).astype(np.uint8) * 255)
            self.mask_publisher.publish(self._image_message(mask_u8, "mono8", stamp))
        if depth_preview_bgr is not None:
            self.depth_publisher.publish(
                self._image_message(depth_preview_bgr, "bgr8", stamp)
            )

        status = dict(diagnostics or {})
        status.update(
            {
                "valid": bool(estimate.valid),
                "reason": estimate.reason,
                "method": estimate.method,
                "depth_coverage": float(estimate.depth_coverage),
                "valid_point_count": int(estimate.valid_point_count),
            }
        )
        if estimate.valid:
            status["center_workspace_mm"] = (
                np.asarray(estimate.center_workspace_m) * 1000.0
            ).tolist()
            status["base_center_workspace_mm"] = (
                np.asarray(estimate.base_center_workspace_m) * 1000.0
            ).tolist()
            status["observed_height_mm"] = float(estimate.observed_height_m * 1000.0)
            status["observed_diameter_mm"] = float(
                estimate.observed_diameter_m * 1000.0
            )
            status["nominal_height_mm"] = float(estimate.nominal_height_m * 1000.0)
            status["nominal_diameter_mm"] = float(
                estimate.nominal_diameter_m * 1000.0
            )
            status["circle_fit_rms_mm"] = (
                None
                if estimate.circle_fit_rms_m is None
                else float(estimate.circle_fit_rms_m * 1000.0)
            )
        self.status_publisher.publish(
            self.String(data=json.dumps(status, ensure_ascii=False))
        )

        if not estimate.valid:
            self._delete_markers(stamp)
            return
        self._publish_pose_and_tf(estimate, stamp)
        self._publish_cloud(estimate, stamp)
        self._publish_markers(estimate, stamp)

    def _publish_pose_and_tf(self, estimate: BottleEstimate, stamp) -> None:
        pose = self._pose_message(estimate.workspace_from_bottle, stamp)
        base_pose = self._pose_message(estimate.workspace_from_bottle_base, stamp)
        self.pose_publisher.publish(pose)
        self.base_pose_publisher.publish(base_pose)
        transform = np.asarray(estimate.workspace_from_bottle, dtype=np.float64)
        quaternion = self.quaternion_from_matrix(transform)
        message = self.TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.workspace_frame
        message.child_frame_id = self.bottle_frame
        message.transform.translation.x = float(transform[0, 3])
        message.transform.translation.y = float(transform[1, 3])
        message.transform.translation.z = float(transform[2, 3])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])
        self.tf_broadcaster.sendTransform(message)

    def _pose_message(self, transform: np.ndarray, stamp):
        transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        quaternion = self.quaternion_from_matrix(transform)
        pose = self.PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.workspace_frame
        pose.pose.position.x = float(transform[0, 3])
        pose.pose.position.y = float(transform[1, 3])
        pose.pose.position.z = float(transform[2, 3])
        pose.pose.orientation.x = float(quaternion[0])
        pose.pose.orientation.y = float(quaternion[1])
        pose.pose.orientation.z = float(quaternion[2])
        pose.pose.orientation.w = float(quaternion[3])
        return pose

    def _publish_cloud(self, estimate: BottleEstimate, stamp) -> None:
        points = np.asarray(estimate.cloud_workspace_m, dtype=np.float32).reshape(-1, 3)
        header = self.Header(stamp=stamp, frame_id=self.workspace_frame)
        cloud = self.point_cloud2.create_cloud_xyz32(header, points.tolist())
        self.cloud_publisher.publish(cloud)

    def _publish_markers(self, estimate: BottleEstimate, stamp) -> None:
        transform = np.asarray(estimate.workspace_from_bottle, dtype=np.float64)
        quaternion = self.quaternion_from_matrix(transform)
        height = (
            float(estimate.nominal_height_m)
            if float(estimate.nominal_height_m) > 0.0
            else max(float(estimate.observed_height_m), 0.04)
        )
        diameter = (
            float(estimate.nominal_diameter_m)
            if float(estimate.nominal_diameter_m) > 0.0
            else max(float(estimate.observed_diameter_m), 0.025)
        )
        center = np.asarray(estimate.center_workspace_m, dtype=np.float64)
        base = np.asarray(estimate.base_center_workspace_m, dtype=np.float64)
        up = transform[:3, 2]

        cylinder = self._marker("bottle", 0, self.Marker.CYLINDER, stamp)
        cylinder.pose.position.x = float(center[0])
        cylinder.pose.position.y = float(center[1])
        cylinder.pose.position.z = float(center[2])
        cylinder.pose.orientation.x = float(quaternion[0])
        cylinder.pose.orientation.y = float(quaternion[1])
        cylinder.pose.orientation.z = float(quaternion[2])
        cylinder.pose.orientation.w = float(quaternion[3])
        cylinder.scale.x = diameter
        cylinder.scale.y = diameter
        cylinder.scale.z = height
        cylinder.color.r = 0.12
        cylinder.color.g = 0.86
        cylinder.color.b = 0.42
        cylinder.color.a = 0.36

        base_marker = self._marker("bottle_base", 1, self.Marker.SPHERE, stamp)
        base_marker.pose.position.x = float(base[0])
        base_marker.pose.position.y = float(base[1])
        base_marker.pose.position.z = float(base[2])
        base_marker.scale.x = base_marker.scale.y = base_marker.scale.z = 0.018
        base_marker.color.r = 1.0
        base_marker.color.g = 0.72
        base_marker.color.b = 0.08
        base_marker.color.a = 1.0

        axis = self._marker("bottle_axis", 2, self.Marker.LINE_LIST, stamp)
        axis.scale.x = 0.006
        axis.color.r = 0.2
        axis.color.g = 0.95
        axis.color.b = 1.0
        axis.color.a = 1.0
        top = base + up * max(height, float(np.linalg.norm(center - base)) * 2.0)
        axis.points = [self.Point(*base.tolist()), self.Point(*top.tolist())]

        label = self._marker("bottle_label", 3, self.Marker.TEXT_VIEW_FACING, stamp)
        label_position = top + up * 0.035
        label.pose.position.x = float(label_position[0])
        label.pose.position.y = float(label_position[1])
        label.pose.position.z = float(label_position[2])
        label.scale.z = 0.026
        label.color.r = 0.95
        label.color.g = 0.98
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = (
            "BOTTLE base=({:.1f}, {:.1f}, {:.1f}) mm\n"
            "center=({:.1f}, {:.1f}, {:.1f}) mm | depth {:.0f}% | {}"
        ).format(
            *(base * 1000.0),
            *(center * 1000.0),
            estimate.depth_coverage * 100.0,
            estimate.method,
        )
        self.marker_publisher.publish(
            self.MarkerArray(markers=[cylinder, base_marker, axis, label])
        )

    def _marker(self, namespace: str, marker_id: int, marker_type: int, stamp):
        marker = self.Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.workspace_frame
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = int(marker_type)
        marker.action = self.Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _delete_markers(self, stamp) -> None:
        marker = self._marker("bottle", 0, self.Marker.CUBE, stamp)
        marker.action = self.Marker.DELETEALL
        self.marker_publisher.publish(self.MarkerArray(markers=[marker]))

    def _image_message(self, image: np.ndarray, encoding: str, stamp):
        array = np.ascontiguousarray(image)
        if encoding == "bgr8":
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("bgr8 image must have three channels")
            step = int(array.shape[1] * 3)
        elif encoding == "mono8":
            if array.ndim != 2:
                raise ValueError("mono8 image must have one channel")
            step = int(array.shape[1])
        else:
            raise ValueError("unsupported visualization image encoding: {}".format(encoding))
        message = self.Image()
        message.header.stamp = stamp
        message.header.frame_id = self.workspace_frame
        message.height = int(array.shape[0])
        message.width = int(array.shape[1])
        message.encoding = encoding
        message.is_bigendian = 0
        message.step = step
        message.data = array.tobytes()
        return message
