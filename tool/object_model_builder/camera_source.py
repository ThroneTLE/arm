#!/usr/bin/env python3
"""Astra Pro source: independent UVC color plus ROS depth/IR streams."""

import os
import signal
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .rgbd_geometry import CameraIntrinsics


_ROS_IMAGE_DTYPES = {
    "8UC1": (np.uint8, 1),
    "mono8": (np.uint8, 1),
    "16UC1": (np.uint16, 1),
    "mono16": (np.uint16, 1),
    "16SC1": (np.int16, 1),
    "32FC1": (np.float32, 1),
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
}


def nearest_timestamped_frame(history, timestamp):
    """Return the buffered frame whose host-arrival timestamp is closest."""
    if not history:
        return None, None
    target = float(timestamp)
    selected_timestamp, selected_frame = min(
        history, key=lambda item: abs(float(item[0]) - target)
    )
    return float(selected_timestamp), selected_frame


def ros_image_to_numpy(message) -> np.ndarray:
    """Decode common sensor_msgs/Image encodings without the cv_bridge ABI."""
    encoding = str(message.encoding)
    if encoding not in _ROS_IMAGE_DTYPES:
        raise ValueError("不支持的 ROS 图像编码：{}".format(encoding))
    base_dtype, channels = _ROS_IMAGE_DTYPES[encoding]
    native_dtype = np.dtype(base_dtype)
    byte_order = ">" if bool(message.is_bigendian) else "<"
    wire_dtype = native_dtype.newbyteorder(byte_order)
    height = int(message.height)
    width = int(message.width)
    row_elements = int(message.step) // wire_dtype.itemsize
    required_elements = height * row_elements
    flat = np.frombuffer(message.data, dtype=wire_dtype, count=required_elements)
    if flat.size != required_elements or row_elements < width * channels:
        raise ValueError(
            "ROS 图像数据长度无效：{}x{} {} step={}".format(
                width, height, encoding, message.step
            )
        )
    image = flat.reshape(height, row_elements)[:, : width * channels]
    if channels > 1:
        image = image.reshape(height, width, channels)
    else:
        image = image.reshape(height, width)
    return image.astype(native_dtype, copy=False).copy()


def native_ros_environment(environment=None) -> dict:
    """Remove Conda library directories from native ROS subprocesses."""
    result = dict(os.environ if environment is None else environment)
    library_paths = result.get("LD_LIBRARY_PATH", "").split(":")
    native_paths = []
    for entry in library_paths:
        if not entry:
            continue
        path = Path(entry).expanduser().resolve(strict=False)
        candidates = (path,) + tuple(path.parents)
        conda_path = any(
            "conda" in part.lower() or "mamba" in part.lower()
            for part in path.parts
        )
        if conda_path or any(
            (candidate / "conda-meta").is_dir() for candidate in candidates
        ):
            continue
        native_paths.append(entry)
    result["LD_LIBRARY_PATH"] = ":".join(native_paths)
    return result


@dataclass
class FrameBundle:
    color_bgr: np.ndarray
    color_timestamp_s: float
    depth_m: Optional[np.ndarray]
    depth_timestamp_s: Optional[float]
    depth_intrinsics: Optional[CameraIntrinsics]
    ir_image: Optional[np.ndarray]
    ir_timestamp_s: Optional[float]
    color_intrinsics: Optional[CameraIntrinsics] = None
    depth_aligned_to_color: bool = False
    color_is_rectified: bool = False

    @property
    def sync_delta_s(self) -> Optional[float]:
        if self.depth_timestamp_s is None:
            return None
        return abs(float(self.color_timestamp_s) - float(self.depth_timestamp_s))


class AstraRosSource:
    def __init__(
        self,
        color_device: str,
        color_width: int = 1280,
        color_height: int = 720,
        color_fps: int = 30,
        color_fourcc: str = "MJPG",
        depth_topic: str = "/camera/depth/image_raw",
        depth_info_topic: str = "/camera/depth/camera_info",
        ir_topic: str = "/camera/ir/image_raw",
        start_ros_driver: bool = True,
        driver_log_path: Optional[str] = None,
        driver_package: str = "astra_camera",
        driver_launch_file: str = "astra.launch",
        driver_arguments: Optional[dict] = None,
        driver_startup_timeout_s: float = 2.0,
        laser_service: str = "/camera/set_laser",
        ros_node_name: str = "object_model_builder",
    ):
        self.color_device = str(Path(color_device).expanduser())
        self.color_width = int(color_width)
        self.color_height = int(color_height)
        self.color_fps = int(color_fps)
        self.color_fourcc = str(color_fourcc)
        self.depth_topic = str(depth_topic)
        self.depth_info_topic = str(depth_info_topic)
        self.ir_topic = str(ir_topic)
        self.start_ros_driver = bool(start_ros_driver)
        self.driver_log_path = Path(driver_log_path).expanduser() if driver_log_path else None
        self.driver_package = str(driver_package)
        self.driver_launch_file = str(driver_launch_file)
        if driver_arguments is None:
            driver_arguments = {
                "enable_color": False,
                "enable_depth": True,
                "enable_ir": True,
                "enable_point_cloud": False,
                "enable_point_cloud_xyzrgb": False,
                "depth_width": 1280,
                "depth_height": 1024,
                "depth_fps": 7,
                "ir_width": 1280,
                "ir_height": 1024,
                "ir_fps": 30,
            }
        self.driver_arguments = dict(driver_arguments)
        self.driver_startup_timeout_s = float(driver_startup_timeout_s)
        self.laser_service = str(laser_service)
        self.ros_node_name = str(ros_node_name)
        self._laser_enabled = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture = None
        self._capture_thread = None
        self._driver_process = None
        self._driver_log = None
        self._subscribers = []
        self._color = None
        self._color_timestamp = None
        self._color_history = deque(maxlen=12)
        self._depth = None
        self._depth_timestamp = None
        self._depth_history = deque(maxlen=4)
        self._depth_intrinsics = None
        self._ir = None
        self._ir_timestamp = None
        self._ir_history = deque(maxlen=8)

    def start(self) -> None:
        if self._capture is not None:
            return
        if self.start_ros_driver:
            self._start_driver()
        self._start_ros_subscribers()
        capture = cv2.VideoCapture(self.color_device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.color_fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.color_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.color_height)
        capture.set(cv2.CAP_PROP_FPS, self.color_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            self.stop()
            raise RuntimeError("failed to open Astra UVC color device: {}".format(self.color_device))
        actual = (
            int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        if actual != (self.color_width, self.color_height):
            capture.release()
            self.stop()
            raise RuntimeError(
                "Astra color returned {}x{} instead of {}x{}".format(
                    actual[0], actual[1], self.color_width, self.color_height
                )
            )
        self._capture = capture
        self._capture_thread = threading.Thread(target=self._color_loop, daemon=True)
        self._capture_thread.start()

    @property
    def laser_enabled(self) -> Optional[bool]:
        return self._laser_enabled

    def set_laser_enabled(self, enabled: bool, timeout_s: float = 3.0) -> None:
        try:
            import rospy
            from std_srvs.srv import SetBool
        except ImportError as error:
            raise RuntimeError("ROS 激光控制服务不可用") from error
        if not rospy.core.is_initialized():
            raise RuntimeError("ROS 节点尚未初始化，无法控制红外投影器")
        try:
            rospy.wait_for_service(self.laser_service, timeout=float(timeout_s))
            service = rospy.ServiceProxy(self.laser_service, SetBool, persistent=False)
            service(bool(enabled))
        except Exception as error:
            raise RuntimeError(
                "无法{} Astra 红外投影器（{}）：{}".format(
                    "打开" if enabled else "关闭", self.laser_service, error
                )
            ) from error
        self._laser_enabled = bool(enabled)
        with self._lock:
            self._ir = None
            self._ir_timestamp = None
            self._ir_history.clear()
            self._depth = None
            self._depth_timestamp = None
            self._depth_history.clear()

    def _start_driver(self) -> None:
        driver_environment = native_ros_environment()
        self._validate_driver(driver_environment)
        command = ["roslaunch", self.driver_package, self.driver_launch_file]
        command.extend(
            "{}:={}".format(name, self._roslaunch_value(value))
            for name, value in self.driver_arguments.items()
        )
        if self.driver_log_path:
            self.driver_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._driver_log = open(self.driver_log_path, "w", encoding="utf-8")
        self._driver_process = subprocess.Popen(
            command,
            stdout=self._driver_log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=driver_environment,
        )
        time.sleep(self.driver_startup_timeout_s)
        if self._driver_process.poll() is not None:
            detail = self._driver_log_tail()
            message = "相机 ROS 驱动启动后异常退出"
            if detail:
                message += "\n\n驱动日志末尾：\n{}".format(detail)
            raise RuntimeError(message)

    def _validate_driver(self, driver_environment=None) -> None:
        if shutil.which("roslaunch") is None or shutil.which("rospack") is None:
            raise RuntimeError(
                "当前环境找不到 ROS 命令。请使用 run_ui.sh 启动，或先加载 ROS Noetic 环境。"
            )
        result = subprocess.run(
            ["rospack", "find", self.driver_package],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=driver_environment or native_ros_environment(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ROS 找不到软件包 '{}'. 请确认已加载相机 ROS 工作区；"
                "Astra Pro 默认路径为 /home/throne/astra_ws/devel/setup.bash。\n{}".format(
                    self.driver_package, result.stdout.strip()
                )
            )
        package_root = Path(result.stdout.strip().splitlines()[-1])
        launch_path = package_root / "launch" / self.driver_launch_file
        if not launch_path.is_file():
            raise RuntimeError("未找到相机驱动启动文件：{}".format(launch_path))

    @staticmethod
    def _roslaunch_value(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _driver_log_tail(self, maximum_lines: int = 18) -> str:
        if self._driver_log is not None:
            self._driver_log.flush()
        if self.driver_log_path is None or not self.driver_log_path.is_file():
            return ""
        try:
            lines = self.driver_log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return ""
        keywords = (
            "error",
            "exception",
            "failed",
            "died",
            "not found",
            "undefined symbol",
        )
        diagnostics = [
            line for line in lines if any(word in line.lower() for word in keywords)
        ][-8:]
        excerpt = []
        for line in diagnostics + lines[-maximum_lines:]:
            if line not in excerpt:
                excerpt.append(line)
        return "\n".join(excerpt).strip()

    def _start_ros_subscribers(self) -> None:
        try:
            import rospy
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as error:
            raise RuntimeError("ROS Noetic Python 消息模块不可用") from error
        if not rospy.core.is_initialized():
            rospy.init_node(
                self.ros_node_name,
                anonymous=True,
                disable_signals=True,
            )

        def depth_callback(message):
            depth = ros_image_to_numpy(message)
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) * 0.001
            else:
                depth = depth.astype(np.float32)
            with self._lock:
                self._depth = depth.copy()
                self._depth_timestamp = time.monotonic()
                self._depth_history.append((self._depth_timestamp, self._depth))

        def ir_callback(message):
            image = ros_image_to_numpy(message)
            with self._lock:
                self._ir = image
                self._ir_timestamp = time.monotonic()
                self._ir_history.append((self._ir_timestamp, self._ir))

        def info_callback(message):
            try:
                intrinsics = CameraIntrinsics(
                    width=int(message.width),
                    height=int(message.height),
                    matrix=np.asarray(message.K, dtype=np.float64).reshape(3, 3),
                    distortion=np.asarray(message.D, dtype=np.float64),
                )
            except ValueError:
                return
            with self._lock:
                self._depth_intrinsics = intrinsics

        self._subscribers = [
            rospy.Subscriber(self.depth_topic, Image, depth_callback, queue_size=1),
            rospy.Subscriber(self.ir_topic, Image, ir_callback, queue_size=1),
            rospy.Subscriber(self.depth_info_topic, CameraInfo, info_callback, queue_size=1),
        ]

    def _color_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._color = frame
                self._color_timestamp = time.monotonic()
                self._color_history.append((self._color_timestamp, self._color))

    def latest(
        self, anchor: str = "depth", copy_frames: bool = True,
    ) -> Optional[FrameBundle]:
        with self._lock:
            if not self._color_history:
                return None
            if anchor == "ir" and self._ir_history:
                ir_timestamp, infrared = self._ir_history[-1]
                color_timestamp, color = nearest_timestamped_frame(
                    self._color_history, ir_timestamp
                )
                depth_timestamp, depth = nearest_timestamped_frame(
                    self._depth_history, color_timestamp
                )
            elif anchor == "color":
                color_timestamp, color = self._color_history[-1]
                depth_timestamp, depth = nearest_timestamped_frame(
                    self._depth_history, color_timestamp
                )
                ir_timestamp, infrared = nearest_timestamped_frame(
                    self._ir_history, color_timestamp
                )
            elif self._depth_history:
                depth_timestamp, depth = self._depth_history[-1]
                color_timestamp, color = nearest_timestamped_frame(
                    self._color_history, depth_timestamp
                )
                ir_timestamp, infrared = nearest_timestamped_frame(
                    self._ir_history, color_timestamp
                )
            else:
                color_timestamp, color = self._color_history[-1]
                depth_timestamp, depth = None, None
                ir_timestamp, infrared = nearest_timestamped_frame(
                    self._ir_history, color_timestamp
                )
            clone = (lambda value: value.copy()) if copy_frames else (lambda value: value)
            return FrameBundle(
                color_bgr=clone(color),
                color_timestamp_s=float(color_timestamp),
                depth_m=None if depth is None else clone(depth),
                depth_timestamp_s=depth_timestamp,
                depth_intrinsics=self._depth_intrinsics,
                ir_image=None if infrared is None else clone(infrared),
                ir_timestamp_s=ir_timestamp,
                color_intrinsics=None,
                depth_aligned_to_color=False,
                color_is_rectified=False,
            )

    def stop(self) -> None:
        if self._laser_enabled is False:
            try:
                self.set_laser_enabled(True, timeout_s=1.0)
            except Exception:
                pass
        self._stop_event.set()
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass
        self._subscribers = []
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if self._driver_process is not None:
            if self._driver_process.poll() is None:
                try:
                    self._driver_process.send_signal(signal.SIGINT)
                    self._driver_process.wait(timeout=5.0)
                except Exception:
                    self._driver_process.terminate()
            self._driver_process = None
        if self._driver_log is not None:
            self._driver_log.close()
            self._driver_log = None


class OrbbecRosSource:
    """Orbbec SDK ROS source using live factory ``CameraInfo`` values.

    Unlike :class:`AstraRosSource`, Gemini color is owned by the Orbbec SDK
    driver as well.  Subscribing to all three ROS streams keeps one process in
    control of the USB device and, importantly, ties every RGB frame to the
    intrinsics returned by that same connected camera.
    """

    def __init__(
        self,
        color_topic="/camera/color/image_raw",
        color_info_topic="/camera/color/camera_info",
        depth_topic="/camera/depth/image_raw",
        depth_info_topic="/camera/depth/camera_info",
        ir_topic="/camera/ir/image_raw",
        start_ros_driver=True,
        driver_log_path=None,
        driver_package="orbbec_camera",
        driver_launch_file="gemini.launch",
        driver_arguments=None,
        driver_startup_timeout_s=2.0,
        laser_service="/camera/set_laser",
        depth_aligned_to_color=True,
        expected_serial=None,
        ros_node_name="orbbec_rgbd_source",
    ):
        self.color_topic = str(color_topic)
        self.color_info_topic = str(color_info_topic)
        self.depth_topic = str(depth_topic)
        self.depth_info_topic = str(depth_info_topic)
        self.ir_topic = str(ir_topic)
        self.start_ros_driver = bool(start_ros_driver)
        self.driver_log_path = (
            Path(driver_log_path).expanduser() if driver_log_path else None
        )
        self.driver_package = str(driver_package)
        self.driver_launch_file = str(driver_launch_file)
        self.driver_arguments = dict(driver_arguments or {})
        self.driver_startup_timeout_s = float(driver_startup_timeout_s)
        self.laser_service = str(laser_service)
        self.depth_aligned_to_color = bool(depth_aligned_to_color)
        self.expected_serial = str(expected_serial or "").strip()
        self.ros_node_name = str(ros_node_name)
        self._laser_enabled = None
        self._lock = threading.Lock()
        self._driver_process = None
        self._driver_log = None
        self._subscribers = []
        self._color_history = deque(maxlen=12)
        self._depth_history = deque(maxlen=8)
        self._ir_history = deque(maxlen=8)
        self._color_intrinsics = None
        self._depth_intrinsics = None

    def start(self) -> None:
        if self._subscribers:
            return
        try:
            if self.start_ros_driver:
                self._start_driver()
            self._start_ros_subscribers()
            self._verify_device_serial()
        except Exception:
            self.stop()
            raise

    @property
    def laser_enabled(self) -> Optional[bool]:
        return self._laser_enabled

    def set_laser_enabled(self, enabled: bool, timeout_s: float = 3.0) -> None:
        try:
            import rospy
            from std_srvs.srv import SetBool
        except ImportError as error:
            raise RuntimeError("ROS 激光控制服务不可用") from error
        try:
            rospy.wait_for_service(self.laser_service, timeout=float(timeout_s))
            rospy.ServiceProxy(self.laser_service, SetBool, persistent=False)(
                bool(enabled)
            )
        except Exception as error:
            raise RuntimeError(
                "无法{} Orbbec 红外投影器（{}）：{}".format(
                    "打开" if enabled else "关闭", self.laser_service, error
                )
            ) from error
        self._laser_enabled = bool(enabled)

    def _start_driver(self) -> None:
        environment = native_ros_environment()
        self._validate_driver(environment)
        command = ["roslaunch", self.driver_package, self.driver_launch_file]
        command.extend(
            "{}:={}".format(name, self._roslaunch_value(value))
            for name, value in self.driver_arguments.items()
        )
        if self.driver_log_path:
            self.driver_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._driver_log = open(self.driver_log_path, "w", encoding="utf-8")
        self._driver_process = subprocess.Popen(
            command,
            stdout=self._driver_log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        time.sleep(self.driver_startup_timeout_s)
        if self._driver_process.poll() is not None:
            raise RuntimeError("Orbbec ROS 驱动启动后异常退出")

    def _validate_driver(self, environment) -> None:
        if shutil.which("roslaunch") is None or shutil.which("rospack") is None:
            raise RuntimeError("当前环境找不到 ROS 命令，请使用工作台启动脚本")
        result = subprocess.run(
            ["rospack", "find", self.driver_package],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ROS 找不到软件包 '{}'. 请加载 /home/throne/orbbec_ws/devel/setup.bash\n{}".format(
                    self.driver_package, result.stdout.strip()
                )
            )
        launch_path = (
            Path(result.stdout.strip().splitlines()[-1])
            / "launch"
            / self.driver_launch_file
        )
        if not launch_path.is_file():
            raise RuntimeError("未找到 Orbbec 启动文件：{}".format(launch_path))

    @staticmethod
    def _roslaunch_value(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _intrinsics_from_info(message) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=int(message.width),
            height=int(message.height),
            matrix=np.asarray(message.K, dtype=np.float64).reshape(3, 3),
            distortion=np.asarray(message.D, dtype=np.float64),
        )

    def _start_ros_subscribers(self) -> None:
        try:
            import rospy
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as error:
            raise RuntimeError("ROS Noetic Python 消息模块不可用") from error
        if not rospy.core.is_initialized():
            rospy.init_node(self.ros_node_name, anonymous=True, disable_signals=True)

        def color_callback(message):
            image = ros_image_to_numpy(message)
            if str(message.encoding) == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            timestamp = time.monotonic()
            with self._lock:
                self._color_history.append((timestamp, image))

        def depth_callback(message):
            depth = ros_image_to_numpy(message)
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) * 0.001
            else:
                depth = depth.astype(np.float32)
            timestamp = time.monotonic()
            with self._lock:
                self._depth_history.append((timestamp, depth))

        def ir_callback(message):
            image = ros_image_to_numpy(message)
            timestamp = time.monotonic()
            with self._lock:
                self._ir_history.append((timestamp, image))

        def color_info_callback(message):
            try:
                intrinsics = self._intrinsics_from_info(message)
            except ValueError:
                return
            with self._lock:
                self._color_intrinsics = intrinsics

        def depth_info_callback(message):
            try:
                intrinsics = self._intrinsics_from_info(message)
            except ValueError:
                return
            with self._lock:
                self._depth_intrinsics = intrinsics

        self._subscribers = [
            rospy.Subscriber(self.color_topic, Image, color_callback, queue_size=1),
            rospy.Subscriber(self.depth_topic, Image, depth_callback, queue_size=1),
            rospy.Subscriber(self.ir_topic, Image, ir_callback, queue_size=1),
            rospy.Subscriber(
                self.color_info_topic, CameraInfo, color_info_callback, queue_size=1
            ),
            rospy.Subscriber(
                self.depth_info_topic, CameraInfo, depth_info_callback, queue_size=1
            ),
        ]

    def _verify_device_serial(self) -> None:
        if not self.expected_serial:
            return
        try:
            import rospy
            from orbbec_camera.srv import GetString

            rospy.wait_for_service("/camera/get_serial", timeout=5.0)
            response = rospy.ServiceProxy(
                "/camera/get_serial", GetString, persistent=False
            )()
            actual = str(response.data).strip()
        except Exception as error:
            raise RuntimeError("无法读取 Orbbec 设备序列号：{}".format(error)) from error
        if actual != self.expected_serial:
            raise RuntimeError(
                "连接的 Orbbec 序列号为 {}，配置要求 {}".format(
                    actual or "<empty>", self.expected_serial
                )
            )

    def latest(
        self, anchor: str = "color", copy_frames: bool = True,
    ) -> Optional[FrameBundle]:
        with self._lock:
            # Never expose an Orbbec RGB frame without the live factory
            # intrinsics read from the same driver/device.
            if not self._color_history or self._color_intrinsics is None:
                return None
            if anchor == "ir" and self._ir_history:
                ir_timestamp, infrared = self._ir_history[-1]
                color_timestamp, color = nearest_timestamped_frame(
                    self._color_history, ir_timestamp
                )
                depth_timestamp, depth = nearest_timestamped_frame(
                    self._depth_history, color_timestamp
                )
            elif anchor == "depth" and self._depth_history:
                depth_timestamp, depth = self._depth_history[-1]
                color_timestamp, color = nearest_timestamped_frame(
                    self._color_history, depth_timestamp
                )
                ir_timestamp, infrared = nearest_timestamped_frame(
                    self._ir_history, color_timestamp
                )
            else:
                color_timestamp, color = self._color_history[-1]
                depth_timestamp, depth = nearest_timestamped_frame(
                    self._depth_history, color_timestamp
                )
                ir_timestamp, infrared = nearest_timestamped_frame(
                    self._ir_history, color_timestamp
                )
            clone = (lambda value: value.copy()) if copy_frames else (lambda value: value)
            return FrameBundle(
                color_bgr=clone(color),
                color_timestamp_s=float(color_timestamp),
                depth_m=None if depth is None else clone(depth),
                depth_timestamp_s=depth_timestamp,
                depth_intrinsics=self._depth_intrinsics,
                ir_image=None if infrared is None else clone(infrared),
                ir_timestamp_s=ir_timestamp,
                color_intrinsics=self._color_intrinsics,
                depth_aligned_to_color=self.depth_aligned_to_color,
                color_is_rectified=False,
            )

    def stop(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass
        self._subscribers = []
        if self._driver_process is not None:
            if self._driver_process.poll() is None:
                try:
                    self._driver_process.send_signal(signal.SIGINT)
                    self._driver_process.wait(timeout=5.0)
                except Exception:
                    self._driver_process.terminate()
            self._driver_process = None
        if self._driver_log is not None:
            self._driver_log.close()
            self._driver_log = None


class OakDProSource:
    """DepthAI source using device calibration and depth aligned to RGB."""

    def __init__(
        self, color_width=1920, color_height=1080, fps=10, mxid="",
        dot_projector_mA=800, floodlight_mA=0, mono_resolution="800p",
        extended_disparity=True, subpixel=False, left_right_check=True,
        focus_mode="device_default", manual_focus=None,
    ):
        self.color_width = int(color_width)
        self.color_height = int(color_height)
        self.fps = int(fps)
        self.mxid = str(mxid or "").strip()
        self.dot_projector_mA = int(dot_projector_mA)
        self.floodlight_mA = int(floodlight_mA)
        self.mono_resolution = str(mono_resolution).lower()
        self.extended_disparity = bool(extended_disparity)
        self.subpixel = bool(subpixel)
        self.left_right_check = bool(left_right_check)
        self.focus_mode = str(focus_mode).strip().lower()
        self.manual_focus = manual_focus
        if self.color_width <= 0 or self.color_height <= 0 or self.fps <= 0:
            raise ValueError("OAK color size and FPS must be positive")
        if self.mono_resolution not in ("400p", "800p"):
            raise ValueError("OAK mono resolution must be 400p or 800p")
        if not 0 <= self.dot_projector_mA <= 1200:
            raise ValueError("OAK dot projector current must be within 0..1200 mA")
        if not 0 <= self.floodlight_mA <= 1500:
            raise ValueError("OAK floodlight current must be within 0..1500 mA")
        if self.extended_disparity and self.subpixel:
            raise ValueError("OAK extended disparity and subpixel cannot both be enabled")
        if self.focus_mode not in ("device_default", "continuous_auto", "manual"):
            raise ValueError("OAK focus mode is invalid")
        if self.focus_mode == "manual" and (
            manual_focus is None or not 0 <= int(manual_focus) <= 255
        ):
            raise ValueError("OAK manual focus must be within 0..255")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._device = None
        self._thread = None
        self._queues = {}
        self._pending_packets = {"color": {}, "depth": {}, "ir": {}}
        self._color = None
        self._color_timestamp = None
        self._depth = None
        self._depth_timestamp = None
        self._ir = None
        self._ir_timestamp = None
        self._color_intrinsics = None

    def start(self) -> None:
        try:
            import depthai as dai
        except ImportError as error:
            raise RuntimeError("depthai 2.x is not installed in the active environment") from error
        if self._device is not None:
            return
        self._stop_event.clear()
        pipeline = dai.Pipeline()
        color = pipeline.create(dai.node.ColorCamera)
        left = pipeline.create(dai.node.MonoCamera)
        right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        color.setBoardSocket(dai.CameraBoardSocket.RGB)
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setVideoSize(self.color_width, self.color_height)
        color.setFps(self.fps)
        color.setInterleaved(False)
        if self.focus_mode == "continuous_auto":
            color.initialControl.setAutoFocusMode(
                dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE
            )
        elif self.focus_mode == "manual":
            color.initialControl.setManualFocus(int(self.manual_focus))
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_mode = (
            dai.MonoCameraProperties.SensorResolution.THE_800_P
            if self.mono_resolution == "800p"
            else dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        left.setResolution(mono_mode)
        right.setResolution(mono_mode)
        left.setFps(self.fps)
        right.setFps(self.fps)
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )
        stereo.setLeftRightCheck(self.left_right_check)
        stereo.setExtendedDisparity(self.extended_disparity)
        stereo.setSubpixel(self.subpixel)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(self.color_width, self.color_height)
        left.out.link(stereo.left)
        right.out.link(stereo.right)

        def output(name):
            node = pipeline.create(dai.node.XLinkOut)
            node.setStreamName(name)
            return node

        color_output = output("color")
        depth_output = output("depth")
        ir_output = output("ir")
        color.video.link(color_output.input)
        stereo.depth.link(depth_output.input)
        left.out.link(ir_output.input)
        devices = list(dai.Device.getAllAvailableDevices())
        if self.mxid:
            deadline = time.monotonic() + 5.0
            devices = []
            while not devices and time.monotonic() < deadline:
                devices = [
                    info for info in dai.Device.getAllAvailableDevices()
                    if str(info.getMxId()).strip() == self.mxid
                ]
                if not devices:
                    time.sleep(0.25)
            if not devices:
                raise RuntimeError(
                    "configured OAK MXID {} is not connected".format(self.mxid)
                )
        elif len(devices) > 1:
            raise RuntimeError("multiple OAK devices connected; configure camera.oak.mxid")
        self._device = (
            dai.Device(pipeline, devices[0]) if devices else dai.Device(pipeline)
        )
        self._device.setIrLaserDotProjectorBrightness(self.dot_projector_mA)
        self._device.setIrFloodLightBrightness(self.floodlight_mA)
        self._queues = {
            "color": self._device.getOutputQueue("color", maxSize=2, blocking=False),
            "depth": self._device.getOutputQueue("depth", maxSize=2, blocking=False),
            "ir": self._device.getOutputQueue("ir", maxSize=2, blocking=False),
        }
        calibration = self._device.readCalibration()
        matrix = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB,
            self.color_width,
            self.color_height,
        )
        distortion = calibration.getDistortionCoefficients(dai.CameraBoardSocket.RGB)
        self._color_intrinsics = CameraIntrinsics(
            self.color_width,
            self.color_height,
            np.asarray(matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        )
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def set_laser_enabled(self, enabled: bool) -> None:
        if self._device is None:
            raise RuntimeError("OAK device is not started")
        current = self.dot_projector_mA if enabled else 0
        self._device.setIrLaserDotProjectorBrightness(current)

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            received = False
            for name, queue in self._queues.items():
                while True:
                    packet = queue.tryGet()
                    if packet is None:
                        break
                    self._pending_packets[name][int(packet.getSequenceNum())] = packet
                    received = True
            common = set(self._pending_packets["color"]).intersection(
                self._pending_packets["depth"]
            )
            if not common:
                # Bound unmatched packets while waiting for the corresponding
                # RGB/depth sequence number.
                for pending in self._pending_packets.values():
                    for sequence in sorted(pending)[:-6]:
                        pending.pop(sequence, None)
                if not received:
                    time.sleep(0.003)
                continue
            sequence = max(common)
            color = self._pending_packets["color"].pop(sequence)
            depth = self._pending_packets["depth"].pop(sequence)
            infrared = self._pending_packets["ir"].pop(sequence, None)
            color_frame = color.getCvFrame()
            depth_frame = depth.getFrame().astype(np.float32) * 0.001
            ir_frame = None if infrared is None else infrared.getFrame()
            with self._lock:
                self._color = color_frame
                self._color_timestamp = self._packet_timestamp(color)
                self._depth = depth_frame
                self._depth_timestamp = self._packet_timestamp(depth)
                if ir_frame is not None:
                    self._ir = ir_frame
                    self._ir_timestamp = self._packet_timestamp(infrared)
            for pending in self._pending_packets.values():
                for old_sequence in [value for value in pending if value <= sequence]:
                    pending.pop(old_sequence, None)
                time.sleep(0.003)

    @staticmethod
    def _packet_timestamp(packet) -> float:
        stamp = packet.getTimestampDevice()
        if hasattr(stamp, "total_seconds"):
            return float(stamp.total_seconds())
        if hasattr(stamp, "get"):
            stamp = stamp.get()
            if hasattr(stamp, "total_seconds"):
                return float(stamp.total_seconds())
        return float(stamp)

    def latest(
        self, anchor: str = "depth", copy_frames: bool = True,
    ) -> Optional[FrameBundle]:
        with self._lock:
            if self._color is None:
                return None
            clone = (lambda value: value.copy()) if copy_frames else (lambda value: value)
            return FrameBundle(
                color_bgr=clone(self._color),
                color_timestamp_s=float(self._color_timestamp),
                depth_m=None if self._depth is None else clone(self._depth),
                depth_timestamp_s=self._depth_timestamp,
                depth_intrinsics=self._color_intrinsics,
                ir_image=None if self._ir is None else clone(self._ir),
                ir_timestamp_s=self._ir_timestamp,
                color_intrinsics=self._color_intrinsics,
                depth_aligned_to_color=True,
                color_is_rectified=False,
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._queues = {}
        self._pending_packets = {"color": {}, "depth": {}, "ir": {}}
        device, self._device = self._device, None
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
