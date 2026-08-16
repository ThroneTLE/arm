#!/usr/bin/env python3
"""ROS TF, pose, path, and Marker publishers for calibration RViz views."""

import math
import time

import numpy as np


class RosPoseVisualizer:
    """Publish a fixed Tag map and a dynamic workspace-from-camera pose."""

    def __init__(self, workspace_frame, camera_frame):
        import rospy
        import tf2_ros
        from geometry_msgs.msg import Point, PoseStamped, TransformStamped
        from nav_msgs.msg import Path
        from tf.transformations import quaternion_from_matrix
        from visualization_msgs.msg import Marker, MarkerArray

        if not rospy.core.is_initialized():
            rospy.init_node(
                "astra_pro_calibration_rviz",
                anonymous=True,
                disable_signals=True,
            )
        self.rospy = rospy
        self.Point = Point
        self.PoseStamped = PoseStamped
        self.TransformStamped = TransformStamped
        self.Path = Path
        self.Marker = Marker
        self.MarkerArray = MarkerArray
        self.quaternion_from_matrix = quaternion_from_matrix
        self.workspace_frame = str(workspace_frame)
        self.camera_frame = str(camera_frame)
        self.view_frame = "{}_rviz".format(self.workspace_frame)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.marker_publisher = rospy.Publisher(
            "/camera_calibration/markers", MarkerArray, queue_size=1, latch=True
        )
        self.pose_publisher = rospy.Publisher(
            "/camera_calibration/camera_pose", PoseStamped, queue_size=1
        )
        self.path_publisher = rospy.Publisher(
            "/camera_calibration/camera_path", Path, queue_size=1, latch=True
        )
        self.path = Path()
        self.path.header.frame_id = self.workspace_frame
        self._last_path_time = 0.0
        self._scene_markers = []
        self._validation_entry = None
        self._tag_size_m = 0.0
        self._publish_z_up_view_transform()

    def publish_scene(self, layout):
        markers = []
        tag_size_m = float(layout["tag_size_mm"]) / 1000.0
        self._tag_size_m = tag_size_m
        entries = {
            int(tag_id): entry
            for tag_id, entry in layout.get("calibration_tags", {}).items()
        }
        validation = layout.get("validation_tag")
        if validation:
            self._validation_entry = dict(validation)
            entries[int(validation["id"])] = validation

        for tag_id, entry in sorted(entries.items()):
            origin = np.asarray(entry["origin_mm"], dtype=np.float64) / 1000.0
            yaw = math.radians(float(entry.get("yaw_deg", 0.0)))
            rotation = np.asarray(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
                dtype=np.float64,
            )
            center_xy = origin[:2] + rotation @ np.asarray(
                [tag_size_m / 2.0, tag_size_m / 2.0]
            )
            color = (0.12, 0.72, 0.55) if tag_id != 103 else (0.95, 0.55, 0.12)

            plate = self._marker("tags", tag_id, self.Marker.CUBE, self.workspace_frame)
            plate.pose.position.x = float(center_xy[0])
            plate.pose.position.y = float(center_xy[1])
            plate.pose.position.z = float(origin[2])
            plate.pose.orientation.z = math.sin(yaw / 2.0)
            plate.pose.orientation.w = math.cos(yaw / 2.0)
            plate.scale.x = tag_size_m
            plate.scale.y = tag_size_m
            plate.scale.z = 0.002
            plate.color.r, plate.color.g, plate.color.b = color
            plate.color.a = 0.38 if tag_id == 103 else 0.78
            markers.append(plate)

            label = self._marker("tag_labels", 1000 + tag_id, self.Marker.TEXT_VIEW_FACING, self.workspace_frame)
            label.pose.position.x = float(center_xy[0])
            label.pose.position.y = float(center_xy[1])
            label.pose.position.z = float(origin[2] - 0.025)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.022
            label.color.r = label.color.g = label.color.b = 0.95
            label.color.a = 1.0
            role = "EXPECTED" if tag_id == 103 else "FIXED"
            label.text = "ID {} {}  O=({:.1f}, {:.1f}, {:.1f}) mm".format(
                tag_id, role, *(origin * 1000.0)
            )
            markers.append(label)

        workspace_origin = self._marker(
            "workspace_origin", 2000, self.Marker.SPHERE, self.workspace_frame
        )
        workspace_origin.pose.orientation.w = 1.0
        workspace_origin.scale.x = workspace_origin.scale.y = workspace_origin.scale.z = 0.018
        workspace_origin.color.r = 1.0
        workspace_origin.color.g = 0.82
        workspace_origin.color.a = 1.0
        markers.append(workspace_origin)

        frustum = self._camera_frustum_marker()
        markers.append(frustum)
        self._scene_markers = markers
        self.marker_publisher.publish(self.MarkerArray(markers=markers))

    def publish_pose(
        self,
        workspace_from_camera,
        validation_origin_mm=None,
        expected_origin_mm=None,
        validation_corners_mm=None,
    ):
        transform = np.asarray(workspace_from_camera, dtype=np.float64).reshape(4, 4)
        quaternion = self.quaternion_from_matrix(transform)
        stamp = self.rospy.Time.now()

        tf_message = self.TransformStamped()
        tf_message.header.stamp = stamp
        tf_message.header.frame_id = self.workspace_frame
        tf_message.child_frame_id = self.camera_frame
        tf_message.transform.translation.x = float(transform[0, 3])
        tf_message.transform.translation.y = float(transform[1, 3])
        tf_message.transform.translation.z = float(transform[2, 3])
        tf_message.transform.rotation.x = float(quaternion[0])
        tf_message.transform.rotation.y = float(quaternion[1])
        tf_message.transform.rotation.z = float(quaternion[2])
        tf_message.transform.rotation.w = float(quaternion[3])
        self.tf_broadcaster.sendTransform(tf_message)

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
        self.pose_publisher.publish(pose)

        now = time.monotonic()
        if now - self._last_path_time >= 0.1:
            self.path.header.stamp = stamp
            self.path.poses.append(pose)
            self.path.poses = self.path.poses[-1000:]
            self.path_publisher.publish(self.path)
            self._last_path_time = now

        camera_label = self._marker(
            "camera_label", 3000, self.Marker.TEXT_VIEW_FACING, self.workspace_frame
        )
        camera_label.header.stamp = stamp
        camera_label.pose.position.x = float(transform[0, 3])
        camera_label.pose.position.y = float(transform[1, 3])
        camera_label.pose.position.z = float(transform[2, 3] - 0.06)
        camera_label.pose.orientation.w = 1.0
        camera_label.scale.z = 0.03
        camera_label.color.r = 0.2
        camera_label.color.g = 0.65
        camera_label.color.b = 1.0
        camera_label.color.a = 1.0
        camera_label.text = "CAMERA [{:.1f}, {:.1f}, {:.1f}] mm".format(
            *(transform[:3, 3] * 1000.0)
        )
        validation_markers = self._validation_measurement_markers(
            validation_origin_mm, expected_origin_mm, validation_corners_mm
        )
        self.marker_publisher.publish(
            self.MarkerArray(
                markers=self._scene_markers + [camera_label] + validation_markers
            )
        )

    def hide_validation_measurement(self):
        self.marker_publisher.publish(
            self.MarkerArray(
                markers=self._scene_markers
                + self._validation_measurement_markers(None, None, None)
            )
        )

    def clear_path(self):
        self.path = self.Path()
        self.path.header.frame_id = self.workspace_frame
        self._last_path_time = 0.0
        self.path_publisher.publish(self.path)

    def _publish_z_up_view_transform(self):
        """Publish a ROS-friendly Z-up view without changing calibration math."""
        transform = self.TransformStamped()
        transform.header.stamp = self.rospy.Time.now()
        transform.header.frame_id = self.view_frame
        transform.child_frame_id = self.workspace_frame
        # Rx(pi): X stays right, while page-down Y and into-page Z are flipped.
        transform.transform.rotation.x = 1.0
        transform.transform.rotation.w = 0.0
        self.static_tf_broadcaster.sendTransform(transform)

    def _marker(self, namespace, marker_id, marker_type, frame_id):
        marker = self.Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.rospy.Time.now()
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = marker_type
        marker.action = self.Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _validation_measurement_markers(
        self,
        validation_origin_mm,
        expected_origin_mm,
        validation_corners_mm,
    ):
        marker_specs = (
            ("validation_measured", 3100, self.Marker.CUBE),
            ("validation_measured", 3101, self.Marker.TEXT_VIEW_FACING),
            ("validation_error", 3102, self.Marker.LINE_LIST),
        )
        if validation_origin_mm is None:
            deleted = []
            for namespace, marker_id, marker_type in marker_specs:
                marker = self._marker(
                    namespace, marker_id, marker_type, self.workspace_frame
                )
                marker.action = self.Marker.DELETE
                deleted.append(marker)
            return deleted

        measured = np.asarray(validation_origin_mm, dtype=np.float64).reshape(3) / 1000.0
        if expected_origin_mm is None:
            expected_origin_mm = self._validation_entry["origin_mm"]
        expected = np.asarray(expected_origin_mm, dtype=np.float64).reshape(3) / 1000.0
        if validation_corners_mm is not None:
            measured_corners = (
                np.asarray(validation_corners_mm, dtype=np.float64).reshape(4, 3)
                / 1000.0
            )
            measured = measured_corners[0]
            center = measured_corners.mean(axis=0)
            x_edge = measured_corners[1, :2] - measured_corners[0, :2]
            y_edge = measured_corners[3, :2] - measured_corners[0, :2]
            yaw = math.atan2(float(x_edge[1]), float(x_edge[0]))
            measured_size_x = float(np.linalg.norm(x_edge))
            measured_size_y = float(np.linalg.norm(y_edge))
        else:
            yaw = math.radians(
                float((self._validation_entry or {}).get("yaw_deg", 0.0))
            )
            rotation = np.asarray(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
                dtype=np.float64,
            )
            center_xy = measured[:2] + rotation @ np.asarray(
                [self._tag_size_m / 2.0, self._tag_size_m / 2.0]
            )
            center = np.asarray([center_xy[0], center_xy[1], measured[2]])
            measured_size_x = self._tag_size_m
            measured_size_y = self._tag_size_m

        plate = self._marker(
            "validation_measured", 3100, self.Marker.CUBE, self.workspace_frame
        )
        plate.pose.position.x = float(center[0])
        plate.pose.position.y = float(center[1])
        plate.pose.position.z = float(center[2] - 0.003)
        plate.pose.orientation.z = math.sin(yaw / 2.0)
        plate.pose.orientation.w = math.cos(yaw / 2.0)
        plate.scale.x = measured_size_x
        plate.scale.y = measured_size_y
        plate.scale.z = 0.003
        plate.color.r = 0.95
        plate.color.g = 0.15
        plate.color.b = 0.85
        plate.color.a = 0.82

        error_xy_mm = float(np.linalg.norm((measured - expected)[:2]) * 1000.0)
        label = self._marker(
            "validation_measured",
            3101,
            self.Marker.TEXT_VIEW_FACING,
            self.workspace_frame,
        )
        label.pose.position.x = float(center[0])
        label.pose.position.y = float(center[1])
        label.pose.position.z = float(measured[2] - 0.055)
        label.scale.z = 0.024
        label.color.r = 1.0
        label.color.g = 0.25
        label.color.b = 0.9
        label.color.a = 1.0
        label.text = (
            "ID 103 MEASURED O=({:.1f}, {:.1f}, {:.1f}) mm | "
            "Yaw {:.1f} deg | XY err {:.2f} mm"
        ).format(*(measured * 1000.0), math.degrees(yaw), error_xy_mm)

        error_line = self._marker(
            "validation_error", 3102, self.Marker.LINE_LIST, self.workspace_frame
        )
        error_line.scale.x = 0.004
        error_line.color.r = 1.0
        error_line.color.g = 0.15
        error_line.color.b = 0.15
        error_line.color.a = 1.0
        error_line.points = [
            self.Point(float(expected[0]), float(expected[1]), float(expected[2] - 0.008)),
            self.Point(float(measured[0]), float(measured[1]), float(measured[2] - 0.008)),
        ]
        return [plate, label, error_line]

    def _camera_frustum_marker(self):
        marker = self._marker(
            "camera_frustum", 2500, self.Marker.LINE_LIST, self.camera_frame
        )
        marker.scale.x = 0.004
        marker.color.r = 0.2
        marker.color.g = 0.65
        marker.color.b = 1.0
        marker.color.a = 1.0
        origin = (0.0, 0.0, 0.0)
        corners = (
            (-0.09, -0.052, 0.14),
            (0.09, -0.052, 0.14),
            (0.09, 0.052, 0.14),
            (-0.09, 0.052, 0.14),
        )
        segments = []
        for corner in corners:
            segments.append((origin, corner))
        for index in range(4):
            segments.append((corners[index], corners[(index + 1) % 4]))
        for start, end in segments:
            marker.points.append(self.Point(*start))
            marker.points.append(self.Point(*end))
        return marker
