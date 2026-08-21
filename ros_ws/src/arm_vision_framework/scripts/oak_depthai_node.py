#!/usr/bin/env python3
"""On-demand OAK-D Pro RGB-D publisher for the eye-in-hand competition flow.

The node keeps the DepthAI pipeline running at a modest 10 FPS, but publishes
only when the explicit capture service is called.  This prevents the vision
stack from treating a moving-arm frame as a grasp observation.
"""

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_srvs.srv import Trigger, TriggerResponse

from arm_vision_framework.oak_depthai import profile_from_config
from arm_vision_framework.oak_imu import config_from_camera
from arm_vision_framework.parameters import load_system_parameters


class OakDepthAiNode:
    def __init__(self):
        try:
            import depthai as dai
        except ImportError as error:
            raise RuntimeError("DepthAI 2.x is required for OAK-D Pro") from error
        self.dai = dai
        config_path = Path(rospy.get_param(
            "~config", str(PACKAGE_ROOT / "config" / "system_parameters.yaml")
        ))
        self.settings = load_system_parameters(config_path)
        camera = self.settings["camera"]
        if str(camera.get("adapter", "")).lower() != "oak_depthai":
            raise RuntimeError("camera.adapter must be oak_depthai for this node")
        self.profile = profile_from_config(camera)
        self.imu_config = config_from_camera(camera)
        self.capture_timeout_s = float(camera.get("capture_timeout_s", 2.0))
        self.maximum_sync_delta_s = float(camera.get("maximum_sync_delta_s", 0.03))
        if self.capture_timeout_s <= 0.0 or self.maximum_sync_delta_s <= 0.0:
            raise ValueError("OAK capture timeout and sync delta must be positive")
        self.frame_id = str(camera.get("color_frame_id", "camera_color_optical_frame"))
        self.depth_frame_id = str(camera.get("depth_frame_id", self.frame_id))
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.color_packets = deque(maxlen=4)
        self.depth_packets = deque(maxlen=4)
        self.device = None
        self.queues = {}

        self.color_publisher = rospy.Publisher(
            camera["color_topic"], Image, queue_size=1
        )
        self.depth_publisher = rospy.Publisher(
            camera["aligned_depth_topic"], Image, queue_size=1
        )
        self.info_publisher = rospy.Publisher(
            camera["camera_info_topic"], CameraInfo, queue_size=1, latch=True
        )
        self.imu_publisher = rospy.Publisher(
            self.imu_config.topic, Imu, queue_size=10
        ) if self.imu_config.enabled else None
        self.imu_timer = None
        service_name = str(camera.get("capture_service", "~capture"))
        self.capture_service = rospy.Service(service_name, Trigger, self.on_capture)
        self._start_device()
        if self.imu_publisher is not None:
            self.imu_timer = rospy.Timer(
                rospy.Duration(1.0 / self.imu_config.report_rate_hz),
                self._publish_imu,
            )
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "OAK-D Pro ready: %dx%d @ %.1f FPS, on-demand capture service %s",
            self.profile.color_width, self.profile.color_height,
            self.profile.fps, rospy.resolve_name(service_name),
        )

    @staticmethod
    def _packet_timestamp(packet):
        stamp = packet.getTimestampDevice()
        if hasattr(stamp, "total_seconds"):
            return float(stamp.total_seconds())
        return float(stamp)

    @staticmethod
    def _imu_timestamp(sample):
        """Prefer the device timestamp; fall back only when SDK lacks it."""
        for attribute in ("acceleroMeter", "gyroscope"):
            source = getattr(sample, attribute, None)
            if source is None:
                continue
            stamp = getattr(source, "timestamp", None)
            if stamp is not None:
                if hasattr(stamp, "total_seconds"):
                    return float(stamp.total_seconds())
                return float(stamp)
        return None

    def _output(self, pipeline, name):
        output = pipeline.create(self.dai.node.XLinkOut)
        output.setStreamName(name)
        return output

    def _start_device(self):
        dai = self.dai
        pipeline = dai.Pipeline()
        color = pipeline.create(dai.node.ColorCamera)
        left = pipeline.create(dai.node.MonoCamera)
        right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        color.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setVideoSize(self.profile.color_width, self.profile.color_height)
        color.setFps(float(self.profile.fps))
        color.setInterleaved(False)
        color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        if self.profile.focus_mode == "continuous_auto":
            color.initialControl.setAutoFocusMode(
                dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE
            )
        elif self.profile.focus_mode == "manual":
            color.initialControl.setManualFocus(self.profile.manual_focus)
        mono = dai.MonoCameraProperties.SensorResolution.THE_800_P
        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        left.setResolution(mono)
        right.setResolution(mono)
        left.setFps(float(self.profile.fps))
        right.setFps(float(self.profile.fps))
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(bool(self.profile.left_right_check))
        stereo.setExtendedDisparity(bool(self.profile.extended_disparity))
        stereo.setSubpixel(bool(self.profile.subpixel))
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(self.profile.color_width, self.profile.color_height)
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        color_output = self._output(pipeline, "color")
        depth_output = self._output(pipeline, "depth")
        color.video.link(color_output.input)
        stereo.depth.link(depth_output.input)
        if self.imu_config.enabled:
            imu = pipeline.create(dai.node.IMU)
            imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, int(self.imu_config.report_rate_hz))
            imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, int(self.imu_config.report_rate_hz))
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)
            imu_output = self._output(pipeline, "imu")
            imu.out.link(imu_output.input)
        self.device = dai.Device(pipeline)
        self.device.setIrLaserDotProjectorBrightness(self.profile.dot_projector_mA)
        self.device.setIrFloodLightBrightness(self.profile.floodlight_mA)
        self.queues = {
            "color": self.device.getOutputQueue("color", maxSize=4, blocking=False),
            "depth": self.device.getOutputQueue("depth", maxSize=4, blocking=False),
        }
        if self.imu_config.enabled:
            self.queues["imu"] = self.device.getOutputQueue("imu", maxSize=10, blocking=False)
        calibration = self.device.readCalibration()
        matrix = np.asarray(calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            self.profile.color_width, self.profile.color_height,
        ), dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(calibration.getDistortionCoefficients(
            dai.CameraBoardSocket.CAM_A
        ), dtype=np.float64).reshape(-1)
        info = CameraInfo()
        info.width, info.height = self.profile.image_size
        info.distortion_model = "plumb_bob"
        info.K = matrix.reshape(-1).tolist()
        info.D = distortion.tolist()
        info.R = np.eye(3, dtype=np.float64).reshape(-1).tolist()
        info.P = np.asarray([
            matrix[0, 0], 0.0, matrix[0, 2], 0.0,
            0.0, matrix[1, 1], matrix[1, 2], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ], dtype=np.float64).tolist()
        self.camera_info = info

    @staticmethod
    def _vector(sample, names):
        value = sample
        for name in names:
            value = getattr(value, name, None)
            if value is None:
                return None
        return [float(value.x), float(value.y), float(value.z)]

    def _publish_imu(self, _event):
        if self.imu_publisher is None or "imu" not in self.queues:
            return
        while True:
            packet = self.queues["imu"].tryGet()
            if packet is None:
                return
            for sample in getattr(packet, "packets", []) or []:
                message = Imu()
                # Device time has no ROS epoch without a measured clock
                # offset, so receipt time remains the ROS stamp.  Preserve
                # device time as a diagnostic parameter in the log instead
                # of fabricating an absolute ROS time.
                device_stamp = self._imu_timestamp(sample)
                message.header.stamp = rospy.Time.now()
                message.header.frame_id = self.imu_config.imu_frame
                accel = getattr(sample, "acceleroMeter", None)
                gyro = getattr(sample, "gyroscope", None)
                if accel is not None:
                    vector = self._vector(accel, ())
                    if vector is not None:
                        message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = vector
                if gyro is not None:
                    vector = self._vector(gyro, ())
                    if vector is not None:
                        message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = vector
                # The axis convention is intentionally not converted here.
                # Set camera.imu.camera_from_imu only after measuring the
                # physical mounting and publish a TF in the field launch.
                message.orientation_covariance[0] = -1.0
                self.imu_publisher.publish(message)
                if device_stamp is not None:
                    rospy.logdebug_throttle(
                        5.0, "OAK IMU device timestamp %.6f s (raw axis frame)", device_stamp
                    )

    def _drain(self):
        for name, target in (("color", self.color_packets), ("depth", self.depth_packets)):
            while True:
                packet = self.queues[name].tryGet()
                if packet is None:
                    break
                target.append((self._packet_timestamp(packet), packet))

    def _next_pair(self):
        self._drain()
        if not self.color_packets or not self.depth_packets:
            return None
        best = min(
            (
                (abs(color_time - depth_time), color_index, depth_index)
                for color_index, (color_time, _) in enumerate(self.color_packets)
                for depth_index, (depth_time, _) in enumerate(self.depth_packets)
            ),
            key=lambda item: item[0],
        )
        delta, color_index, depth_index = best
        if delta > self.maximum_sync_delta_s:
            # Discard the oldest unmatched packet; it can never form a newer
            # synchronized observation and must not leak into the next shot.
            if self.color_packets[0][0] <= self.depth_packets[0][0]:
                self.color_packets.popleft()
            else:
                self.depth_packets.popleft()
            return None
        color = self.color_packets[color_index][1]
        depth = self.depth_packets[depth_index][1]
        del self.color_packets[color_index]
        del self.depth_packets[depth_index]
        return color, depth, delta

    def capture(self):
        deadline = time.monotonic() + self.capture_timeout_s
        with self.lock:
            while time.monotonic() < deadline and not rospy.is_shutdown():
                pair = self._next_pair()
                if pair is not None:
                    return pair
                time.sleep(0.002)
        raise RuntimeError("timed out waiting for synchronized OAK RGB-D frame")

    def on_capture(self, _request):
        try:
            color_packet, depth_packet, delta = self.capture()
            color = color_packet.getCvFrame()
            depth = depth_packet.getFrame()
            expected = (self.profile.color_height, self.profile.color_width)
            if color.shape[:2] != expected or depth.shape[:2] != expected:
                raise RuntimeError(
                    "DepthAI output size mismatch: RGB {}, depth {}, expected {}".format(
                        color.shape[:2], depth.shape[:2], expected
                    )
                )
            stamp = rospy.Time.now()
            color_message = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
            depth_message = self.bridge.cv2_to_imgmsg(
                depth.astype(np.uint16, copy=False), encoding="16UC1"
            )
            color_message.header.stamp = stamp
            color_message.header.frame_id = self.frame_id
            depth_message.header.stamp = stamp
            depth_message.header.frame_id = self.depth_frame_id
            info = self.camera_info
            info.header.stamp = stamp
            info.header.frame_id = self.frame_id
            self.info_publisher.publish(info)
            self.depth_publisher.publish(depth_message)
            self.color_publisher.publish(color_message)
            return TriggerResponse(
                success=True,
                message="captured synchronized OAK RGB-D frame (delta {:.1f} ms)".format(delta * 1000.0),
            )
        except Exception as error:
            rospy.logerr("OAK capture failed: %s", error)
            return TriggerResponse(success=False, message=str(error))

    def shutdown(self):
        if self.imu_timer is not None:
            self.imu_timer.shutdown()
            self.imu_timer = None
        device, self.device = self.device, None
        if device is not None:
            try:
                device.setIrLaserDotProjectorBrightness(0)
                device.setIrFloodLightBrightness(0)
            except Exception:
                pass
            try:
                device.close()
            except Exception:
                pass


def main():
    rospy.init_node("oak_depthai_camera")
    OakDepthAiNode()
    rospy.spin()


if __name__ == "__main__":
    main()
