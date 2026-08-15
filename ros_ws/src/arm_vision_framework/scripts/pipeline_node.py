#!/usr/bin/env python3
"""ROS1 transport node for the competition vision pipeline."""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from arm_vision_framework.adapters.topic_robot import TopicRobotController
from arm_vision_framework.factory import build_pipeline
from arm_vision_framework.parameters import CalibrationStore, load_system_parameters
from arm_vision_framework.transforms import (
    quaternion_from_transform,
    transform_from_quaternion,
)
from arm_vision_framework.types import FrameData


class PipelineNode:
    def __init__(self):
        default_config = PACKAGE_ROOT / "config" / "system_parameters.yaml"
        default_calibration = PACKAGE_ROOT / "config" / "calibration_parameters.yaml"
        config_path = Path(rospy.get_param("~config", str(default_config)))
        calibration_path = Path(
            rospy.get_param("~calibration", str(default_calibration))
        )
        self.settings = load_system_parameters(config_path)
        self.calibration = CalibrationStore(calibration_path)
        self.pipeline = build_pipeline(self.settings, self.calibration)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.latest_depth = None
        self.latest_depth_stamp = 0.0
        self.last_processed = 0.0
        self.camera_info_seen = False

        outputs = self.settings["outputs"]
        camera = self.settings["camera"]
        self.pose_publisher = rospy.Publisher(
            outputs["object_pose_topic"], PoseStamped, queue_size=1
        )
        self.mask_publisher = rospy.Publisher(
            outputs["segmentation_mask_topic"], Image, queue_size=1
        )
        self.status_publisher = rospy.Publisher(
            outputs["status_topic"], String, queue_size=5
        )
        self.color_subscriber = rospy.Subscriber(
            camera["color_topic"], Image, self.on_color, queue_size=1,
            buff_size=8 * 1024 * 1024,
        )
        self.depth_subscriber = rospy.Subscriber(
            camera["aligned_depth_topic"], Image, self.on_depth, queue_size=1,
            buff_size=8 * 1024 * 1024,
        )
        self.info_subscriber = rospy.Subscriber(
            camera["camera_info_topic"], CameraInfo, self.on_camera_info, queue_size=1
        )
        self._configure_robot_topics()
        rospy.Service("~stop", Trigger, self.on_stop)
        rospy.logwarn(
            "arm_vision_framework started: dry_run=%s, robot_motion=%s",
            self.settings["safety"]["dry_run"],
            self.settings["safety"]["allow_robot_motion"],
        )

    def _configure_robot_topics(self):
        robot = self.pipeline.robot_controller
        if not isinstance(robot, TopicRobotController):
            return
        config = self.settings["robot"]
        self.robot_target_publisher = rospy.Publisher(
            config["target_pose_topic"], PoseStamped, queue_size=1
        )
        self.robot_state_subscriber = rospy.Subscriber(
            config["state_pose_topic"], PoseStamped, self.on_robot_pose, queue_size=1
        )
        robot.set_command_sinks(self.publish_robot_target, self.call_robot_stop)

    def on_camera_info(self, message):
        self.camera_info_seen = True
        configured = self.calibration.camera_matrix
        received = np.asarray(message.K, dtype=np.float64).reshape(3, 3)
        if not np.allclose(configured, received, rtol=0.0, atol=1e-3):
            rospy.logwarn_throttle(
                5.0,
                "CameraInfo differs from calibration_parameters.yaml; central calibration remains authoritative",
            )

    def on_depth(self, message):
        try:
            raw = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            scale = float(self.settings["camera"].get("depth_scale_to_meters", 0.001))
            depth_m = np.asarray(raw, dtype=np.float32) * scale
            stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
            with self.lock:
                self.latest_depth = depth_m
                self.latest_depth_stamp = stamp
        except Exception as error:
            rospy.logerr_throttle(2.0, "depth conversion failed: %s", error)

    def on_robot_pose(self, message):
        position = message.pose.position
        orientation = message.pose.orientation
        matrix = transform_from_quaternion(
            [position.x, position.y, position.z],
            [orientation.x, orientation.y, orientation.z, orientation.w],
        )
        stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
        self.pipeline.robot_controller.update_state(matrix, stamp)

    def on_color(self, message):
        rate_hz = max(float(self.settings["runtime"].get("rate_hz", 10.0)), 0.1)
        stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
        if stamp - self.last_processed < 1.0 / rate_hz:
            return
        if not self.processing_lock.acquire(False):
            return
        self.last_processed = stamp
        try:
            color = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            expected_width, expected_height = self.calibration.image_size
            if color.shape[:2] != (expected_height, expected_width):
                self.publish_error(
                    stamp,
                    "RGB size {}x{} does not match calibration {}x{}".format(
                        color.shape[1], color.shape[0], expected_width, expected_height
                    ),
                )
                return
            with self.lock:
                depth = None if self.latest_depth is None else self.latest_depth.copy()
                depth_stamp = self.latest_depth_stamp
            if depth is not None and abs(stamp - depth_stamp) > 0.1:
                depth = None
            frame = FrameData(
                color_bgr=color,
                depth_m=depth,
                depth_aligned_to_color=self.calibration.depth_aligned_to_color,
                camera_matrix=self.calibration.camera_matrix,
                distortion=self.calibration.distortion,
                timestamp_s=stamp,
                frame_id=message.header.frame_id
                or self.calibration.data["frames"]["camera_color"],
            )
            result = self.pipeline.process(frame)
            self.publish_result(message, result)
        except Exception as error:
            self.publish_error(stamp, str(error))
            rospy.logerr_throttle(2.0, "pipeline error: %s", error)
        finally:
            self.processing_lock.release()

    def publish_result(self, source_message, result):
        if result.segmentation is not None and result.segmentation.valid:
            mask = (result.segmentation.mask.astype(np.uint8) * 255)
            message = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            message.header = source_message.header
            self.mask_publisher.publish(message)
        if result.valid:
            self.pose_publisher.publish(
                self.matrix_to_pose_message(result.workspace_from_object, result.timestamp_s)
            )
        localization = result.camera_localization
        payload = {
            "valid": result.valid,
            "simulated": result.simulated,
            "reason": result.reason,
            "localization_source": None if localization is None else localization.source,
            "visible_tag_ids": [] if localization is None else list(localization.visible_tag_ids),
            "tag_rms_px": None if localization is None else localization.rms_reprojection_error_px,
            "elapsed_ms": result.diagnostics.get("elapsed_ms"),
        }
        self.status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def publish_error(self, stamp, reason):
        payload = {"valid": False, "simulated": False, "stamp": stamp, "reason": reason}
        self.status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def matrix_to_pose_message(self, matrix, stamp_s):
        message = PoseStamped()
        message.header.stamp = rospy.Time.from_sec(float(stamp_s))
        message.header.frame_id = self.calibration.data["frames"]["workspace"]
        message.pose.position.x = float(matrix[0, 3])
        message.pose.position.y = float(matrix[1, 3])
        message.pose.position.z = float(matrix[2, 3])
        quaternion = quaternion_from_transform(matrix)
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        return message

    def publish_robot_target(self, matrix, speed_scale):
        message = self.matrix_to_pose_message(matrix, rospy.Time.now().to_sec())
        message.header.frame_id = self.calibration.data["frames"]["base"]
        self.robot_target_publisher.publish(message)
        return True

    def call_robot_stop(self):
        service_name = self.settings["robot"]["stop_service"]
        rospy.wait_for_service(service_name, timeout=1.0)
        proxy = rospy.ServiceProxy(service_name, Trigger)
        return bool(proxy().success)

    def on_stop(self, request):
        self.pipeline.pose_estimator.reset()
        stopped = self.pipeline.robot_controller.stop()
        return TriggerResponse(success=bool(stopped), message="pipeline reset and stop requested")


def main():
    rospy.init_node("arm_vision_pipeline")
    PipelineNode()
    rospy.spin()


if __name__ == "__main__":
    main()
