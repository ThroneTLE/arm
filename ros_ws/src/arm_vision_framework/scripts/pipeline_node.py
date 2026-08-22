#!/usr/bin/env python3
"""ROS1 transport node for the competition vision pipeline."""

import json
import logging
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import rospy
import message_filters
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
from arm_vision_framework.types import DetectionResult
from arm_vision_framework.controller_state import VisualTaskCommand
from arm_vision_framework.controller_state import ControllerState
from arm_vision_framework.transforms import xyz_rpy_from_transform
from arm_vision_framework.transforms import transform_from_xyz_rpy
from arm_vision_framework.shape_latch import ShapeLatch


class PipelineNode:
    def __init__(self):
        default_config = PACKAGE_ROOT / "config" / "system_parameters.yaml"
        default_calibration = PACKAGE_ROOT / "config" / "calibration_parameters.yaml"
        config_path = Path(rospy.get_param("~config", str(default_config)))
        calibration_path = Path(
            rospy.get_param("~calibration", str(default_calibration))
        )
        self.settings = load_system_parameters(config_path)
        # Keep ROS console/rosout verbosity in the formal parameter file so a
        # competition run can switch to DEBUG without editing launch files.
        level_name = str(
            self.settings.get("runtime", {}).get("log_level", "INFO")
        ).upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.getLogger().setLevel(level)
        # ``rospy.core._base_logger`` is a logging function in ROS Noetic,
        # not a ``logging.Logger`` instance. Configure the named loggers
        # through Python's public logging API so startup works across rospy
        # patch releases.
        logging.getLogger("rospy").setLevel(level)
        logging.getLogger("rosout").setLevel(level)
        self.calibration = CalibrationStore(calibration_path)
        self.pipeline = build_pipeline(self.settings, self.calibration)
        self.bridge = CvBridge()
        self.processing_lock = threading.Lock()
        self.last_processed = 0.0
        self.camera_info_seen = False
        self.latest_controller_state = None
        self.shape_latch = ShapeLatch(
            self.settings.get("controller", {}).get("initial_shape")
        )

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
        self.task_command_publisher = rospy.Publisher(
            outputs.get("task_command_topic", "/arm_vision/task_command"),
            String, queue_size=5,
        )
        self.diagnostics_publisher = rospy.Publisher(
            outputs.get("diagnostics_topic", "/arm_vision/diagnostics"),
            String, queue_size=10,
        )
        self.color_subscriber = message_filters.Subscriber(
            camera["color_topic"], Image, queue_size=2,
            buff_size=16 * 1024 * 1024,
        )
        self.depth_subscriber = message_filters.Subscriber(
            camera["aligned_depth_topic"], Image, queue_size=2,
            buff_size=16 * 1024 * 1024,
        )
        self.rgbd_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber],
            queue_size=3,
            slop=float(camera.get("maximum_sync_delta_s", 0.03)),
            allow_headerless=False,
        )
        self.rgbd_synchronizer.registerCallback(self.on_rgbd)
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
        if bool(self.settings["runtime"].get("debug_logging", False)):
            rospy.loginfo(
                "debug logging enabled: segmentation=bbox+class+ROI, "
                "FoundationPose=per-object, ordering=near_to_far_then_high_to_low"
            )

    def _configure_robot_topics(self):
        robot = self.pipeline.robot_controller
        if not isinstance(robot, TopicRobotController):
            return
        config = self.settings["robot"]
        self.robot_target_publisher = rospy.Publisher(
            config["target_pose_topic"], PoseStamped, queue_size=1
        )
        controller = self.settings.get("controller", {})
        if bool(controller.get("use_state_topic", False)):
            convention = str(controller.get("state_pose_convention", "unverified"))
            if convention != "fixed_zyx_rpy_deg":
                raise RuntimeError(
                    "controller.use_state_topic requires state_pose_convention="
                    "fixed_zyx_rpy_deg confirmed from the official controller manual"
                )
            self.controller_state_subscriber = rospy.Subscriber(
                self.settings["outputs"].get("controller_state_topic", "/arm_vision/controller/state"),
                String, self.on_controller_state, queue_size=2,
            )
        else:
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

    def on_rgbd(self, color_message, depth_message):
        self.on_color(color_message, depth_message)

    def _decode_depth(self, message):
        try:
            raw = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            scale = float(self.settings["camera"].get("depth_scale_to_meters", 0.001))
            depth_m = np.asarray(raw, dtype=np.float32) * scale
            stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
            return depth_m, stamp
        except Exception as error:
            raise ValueError("depth conversion failed: {}".format(error)) from error

    def on_robot_pose(self, message):
        position = message.pose.position
        orientation = message.pose.orientation
        matrix = transform_from_quaternion(
            [position.x, position.y, position.z],
            [orientation.x, orientation.y, orientation.z, orientation.w],
        )
        stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
        self.pipeline.robot_controller.update_state(matrix, stamp)

    def on_controller_state(self, message):
        """Use controller TCP only after its RPY convention is field-confirmed."""
        try:
            state = ControllerState.from_json(message.data)
            latch = self.shape_latch.observe(
                state.initial_shape if state.initial_shape is not None else state.shape
            )
            if latch.changed:
                rospy.logerr_throttle(
                    1.0,
                    "controller shape changed from %s to %s; motion remains blocked until replanned",
                    latch.initial_shape, latch.observed_shape,
                )
            self.latest_controller_state = state
            if not state.connected or state.tcp_xyz_mm is None or state.tcp_rpy_deg is None:
                return
            if state.emergency_stop is True or state.alarm.active:
                rospy.logwarn_throttle(2.0, "controller state unsafe; TCP pose is not accepted")
                return
            matrix = transform_from_xyz_rpy(
                np.asarray(state.tcp_xyz_mm, dtype=np.float64) / 1000.0,
                state.tcp_rpy_deg,
            )
            self.pipeline.robot_controller.update_state(matrix, state.timestamp_s)
        except Exception as error:
            rospy.logwarn_throttle(2.0, "controller state JSON rejected: %s", error)

    def on_color(self, message, depth_message=None):
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
            depth, depth_stamp = (
                (None, stamp)
                if depth_message is None else self._decode_depth(depth_message)
            )
            maximum_delta = float(
                self.settings["camera"].get("maximum_sync_delta_s", 0.03)
            )
            if depth is not None and abs(stamp - depth_stamp) > maximum_delta:
                self.publish_error(
                    stamp,
                    "RGB-D timestamp delta {:.1f} ms exceeds {:.1f} ms".format(
                        abs(stamp - depth_stamp) * 1000.0, maximum_delta * 1000.0
                    ),
                )
                return
            if depth is not None and depth.shape != color.shape[:2]:
                self.publish_error(
                    stamp,
                    "aligned depth size {} does not match RGB {}".format(
                        depth.shape, color.shape[:2]
                    ),
                )
                return
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
            # This is a planning hand-off only.  ``safe_to_execute`` remains
            # false until the operator/MoveIt bridge validates the route and
            # adds the controller's tool/user/shape metadata.
            xyz_m, rpy_deg = xyz_rpy_from_transform(result.workspace_from_object)
            controller_state = self.latest_controller_state
            maximum_state_age = float(
                self.settings["runtime"].get("maximum_robot_pose_age_s", 0.25)
            )
            controller_current = bool(
                controller_state is not None
                and controller_state.connected
                and controller_state.emergency_stop is False
                and not controller_state.alarm.active
                and not self.shape_latch.state.changed
                and abs(float(result.timestamp_s) - float(controller_state.timestamp_s))
                <= maximum_state_age
            )
            command = VisualTaskCommand(
                command_id="vision-{}".format(uuid.uuid4().hex[:12]),
                motion_type="MOVL",
                frame_id=self.calibration.data["frames"]["workspace"],
                targets=({
                    "xyz_mm": (np.asarray(xyz_m) * 1000.0).tolist(),
                    "rpy_deg": np.asarray(rpy_deg).tolist(),
                },),
                tool_id=controller_state.tool_id if controller_current else None,
                user_id=controller_state.user_id if controller_current else None,
                shape=(
                    self.shape_latch.value if controller_current else None
                ),
                safe_to_execute=False,
                metadata={
                    "source_status": "object_pose",
                    "simulated": bool(result.simulated),
                    "controller_state_current": controller_current,
                },
            )
            self.task_command_publisher.publish(String(data=command.to_json()))
        detections = []
        if result.segmentation is not None:
            detections = [
                {
                    "bbox_xyxy": list(detection.bbox_xyxy),
                    "class_id": int(detection.class_id),
                    "class_name": detection.class_name,
                    "confidence": float(detection.confidence),
                }
                for detection in result.segmentation.detections
            ]
            if not detections and result.segmentation.bbox_xyxy is not None:
                detections = [{
                    "bbox_xyxy": list(result.segmentation.bbox_xyxy),
                    "class_id": result.segmentation.class_id,
                    "class_name": result.segmentation.class_name,
                    "confidence": result.segmentation.confidence,
                }]
        object_poses = []
        for index, pose in enumerate(result.object_poses or (() if result.object_pose is None else (result.object_pose,))):
            object_poses.append({
                "index": index,
                "bbox_xyxy": None if pose.bbox_xyxy is None else list(pose.bbox_xyxy),
                "class_id": pose.class_id,
                "class_name": pose.class_name,
                "confidence": float(pose.confidence),
                "tracking": bool(pose.tracking),
            })
        localization = result.camera_localization
        payload = {
            "valid": result.valid,
            "timestamp_s": float(result.timestamp_s),
            "simulated": result.simulated,
            "reason": result.reason,
            "localization_source": None if localization is None else localization.source,
            "visible_tag_ids": [] if localization is None else list(localization.visible_tag_ids),
            "tag_rms_px": None if localization is None else localization.rms_reprojection_error_px,
            "elapsed_ms": result.diagnostics.get("elapsed_ms"),
            "schema_version": "arm_vision.status.v1",
            "detections": detections,
            "objects": object_poses,
            "ordering": result.diagnostics.get("ordering"),
        }
        encoded = json.dumps(payload, ensure_ascii=False)
        self.status_publisher.publish(String(data=encoded))
        self.diagnostics_publisher.publish(String(data=json.dumps({
            "schema_version": "arm_vision.diagnostics.v1",
            "timestamp_s": float(result.timestamp_s),
            "valid": bool(result.valid),
            "reason": result.reason,
            "object_count": len(object_poses),
            "detection_count": len(detections),
            "diagnostics": result.diagnostics,
        }, ensure_ascii=False)))
        if bool(self.settings["runtime"].get("debug_logging", False)):
            rospy.loginfo(
                "vision frame %.3f: detections=%d poses=%d valid=%s reason=%s",
                float(result.timestamp_s), len(detections), len(object_poses),
                result.valid, result.reason,
            )

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
