#!/usr/bin/env python3
"""Shared geometry, camera I/O, and YAML helpers for Astra Pro calibration."""

import math
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml


APRILTAG_DICTIONARY_NAME = "DICT_APRILTAG_36h11"
APRILTAG_DICTIONARY_ID = cv2.aruco.DICT_APRILTAG_36h11

COORDINATE_CONVENTION_ID = "tag_top_left_x_right_y_down_v1"
COORDINATE_CONVENTION = {
    "id": COORDINATE_CONVENTION_ID,
    "origin": "black_border_top_left",
    "x_axis": "top_left_to_top_right",
    "y_axis": "top_left_to_bottom_left",
    "z_axis": "x_cross_y_into_plane",
}
INCOMPATIBLE_CONVENTION_MESSAGE = (
    "旧中心基准外参与当前左上角约定不兼容，请重新采集"
)

CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
CHARUCO_SQUARE_LENGTH_M = 0.036
CHARUCO_MARKER_LENGTH_M = 0.027
PRINTED_TAG_SIZE_MM = 70.0
# Relative black-frame top-left origins on the generated A4 page.
PRINTED_TAG_LAYOUT_MM = {
    100: (0.0, 0.0, 0.0),
    101: (100.0, 0.0, 0.0),
    102: (0.0, 110.0, 0.0),
    103: (100.0, 110.0, 0.0),
}


def april_dictionary():
    return cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY_ID)


def charuco_board():
    return cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH_M,
        CHARUCO_MARKER_LENGTH_M,
        april_dictionary(),
    )


def april_detector():
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(april_dictionary(), parameters)


def detect_apriltags(image: np.ndarray) -> Dict[int, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = april_detector().detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(tag_id): np.asarray(tag_corners, dtype=np.float64).reshape(4, 2)
        for tag_corners, tag_id in zip(corners, ids.reshape(-1))
    }


def draw_tag_detections(image: np.ndarray, detections: Dict[int, np.ndarray]) -> None:
    for points in detections.values():
        polygon = np.rint(np.asarray(points).reshape(4, 2)).astype(np.int32)
        cv2.polylines(image, [polygon], True, (80, 230, 80), 2, cv2.LINE_AA)


def draw_tag_coordinate_axes(
    image: np.ndarray,
    detections: Dict[int, np.ndarray],
    origins_mm: Optional[Dict[int, np.ndarray]] = None,
    camera_matrix: Optional[np.ndarray] = None,
    distortion: Optional[np.ndarray] = None,
    tag_size_mm: float = PRINTED_TAG_SIZE_MM,
    workspace_origin_tag_id: int = 100,
) -> None:
    """Draw every Tag's top-left origin and local right-handed axes.

    Detected corners are ordered TL, TR, BR, BL. +X follows TL->TR, +Y
    follows TL->BL, and +Z=+X cross +Y points into the printed plane.
    """
    origins_mm = origins_mm or {}
    tag_boxes = [_corner_box(points) for points in detections.values()]
    occupied_label_boxes = []
    for tag_id in sorted(detections):
        raw_corners = detections[tag_id]
        corners = np.asarray(raw_corners, dtype=np.float32).reshape(4, 2)
        tl, tr, _, bl = corners
        axis_scale = 0.72
        origin = np.rint(tl).astype(int)
        x_end = np.rint(tl + axis_scale * (tr - tl)).astype(int)
        y_end = np.rint(tl + axis_scale * (bl - tl)).astype(int)
        thickness = max(2, int(round(min(image.shape[:2]) / 360.0)))
        cv2.circle(image, tuple(origin), thickness + 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(origin), thickness + 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            image, tuple(origin), tuple(x_end), (0, 0, 255), thickness, cv2.LINE_AA, tipLength=0.22
        )
        cv2.arrowedLine(
            image, tuple(origin), tuple(y_end), (0, 200, 0), thickness, cv2.LINE_AA, tipLength=0.22
        )
        for label, endpoint, color in (("+X", x_end, (0, 0, 255)), ("+Y", y_end, (0, 200, 0))):
            position = (int(endpoint[0] + 4), int(endpoint[1] - 4))
            cv2.putText(
                image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), thickness + 2, cv2.LINE_AA
            )
            cv2.putText(
                image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, thickness, cv2.LINE_AA
            )

        z_end = None
        if camera_matrix is not None and distortion is not None:
            local_corners_m = tag_world_corners(
                [0.0, 0.0, 0.0], 0.0, float(tag_size_mm) / 1000.0
            )
            ok, rvec, tvec = cv2.solvePnP(
                local_corners_m,
                corners.astype(np.float64),
                np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
                np.asarray(distortion, dtype=np.float64).reshape(-1, 1),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                z_axis = np.asarray(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, float(tag_size_mm) * 0.00072]],
                    dtype=np.float64,
                )
                projected, _ = cv2.projectPoints(
                    z_axis, rvec, tvec, camera_matrix, distortion
                )
                z_points = np.rint(projected.reshape(-1, 2)).astype(int)
                z_end = z_points[1]
                if np.linalg.norm(z_end - z_points[0]) >= 6.0:
                    cv2.arrowedLine(
                        image,
                        tuple(z_points[0]),
                        tuple(z_end),
                        (255, 80, 0),
                        thickness,
                        cv2.LINE_AA,
                        tipLength=0.22,
                    )
        if z_end is None or np.linalg.norm(z_end - origin) < 6.0:
            z_end = origin + np.asarray([-18, -18])
        _outlined_text(image, "+Z", tuple(z_end + np.asarray([4, -4])), (255, 80, 0), thickness)
        _outlined_text(image, "O", tuple(origin + np.asarray([-16, 18])), (0, 255, 255), thickness)

        coordinate = origins_mm.get(int(tag_id))
        labels = ["ID {}".format(tag_id)]
        if coordinate is not None:
            xyz = np.asarray(coordinate, dtype=np.float64).reshape(3)
            labels.append("O=({:.1f}, {:.1f}, {:.1f}) mm".format(*xyz))
        if int(tag_id) == int(workspace_origin_tag_id):
            labels.append("WORKSPACE ORIGIN")
        occupied_label_boxes.append(
            _draw_tag_label(image, corners, labels, occupied_label_boxes, tag_boxes)
        )


def _outlined_text(image, text, position, color, thickness=2, scale=0.48) -> None:
    cv2.putText(
        image,
        text,
        tuple(map(int, position)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        tuple(map(int, position)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _corner_box(corners):
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return (
        int(np.floor(points[:, 0].min())),
        int(np.floor(points[:, 1].min())),
        int(np.ceil(points[:, 0].max())),
        int(np.ceil(points[:, 1].max())),
    )


def _boxes_intersect(first, second, padding=3):
    return not (
        first[2] + padding < second[0]
        or second[2] + padding < first[0]
        or first[3] + padding < second[1]
        or second[3] + padding < first[1]
    )


def _draw_tag_label(image, corners, lines, occupied_boxes, tag_boxes) -> tuple:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1
    line_height = 18
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    box_width = max(widths) + 12
    box_height = line_height * len(lines) + 8
    minimum_x, minimum_y, maximum_x, maximum_y = _corner_box(corners)
    candidates = (
        (minimum_x, minimum_y - box_height - 5),
        (maximum_x + 5, minimum_y),
        (minimum_x - box_width - 5, minimum_y),
        (minimum_x, maximum_y + 5),
        ((minimum_x + maximum_x - box_width) // 2, minimum_y - box_height - 5),
        ((minimum_x + maximum_x - box_width) // 2, maximum_y + 5),
    )
    maximum_label_x = max(0, image.shape[1] - box_width - 1)
    maximum_label_y = max(0, image.shape[0] - box_height - 1)
    selected = None
    own_box = (minimum_x, minimum_y, maximum_x, maximum_y)
    for candidate_x, candidate_y in candidates:
        x = int(np.clip(candidate_x, 0, maximum_label_x))
        y = int(np.clip(candidate_y, 0, maximum_label_y))
        box = (x, y, x + box_width, y + box_height)
        overlaps_label = any(_boxes_intersect(box, other) for other in occupied_boxes)
        overlaps_other_tag = any(
            other != own_box and _boxes_intersect(box, other, padding=1)
            for other in tag_boxes
        )
        if not overlaps_label and not overlaps_other_tag:
            selected = box
            break
    if selected is None:
        x = int(np.clip(candidates[-1][0], 0, maximum_label_x))
        y = int(np.clip(candidates[-1][1], 0, maximum_label_y))
        selected = (x, y, x + box_width, y + box_height)
    x, y = selected[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (28, 34, 38), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0, image)
    for index, line in enumerate(lines):
        color = (0, 255, 255) if line == "WORKSPACE ORIGIN" else (255, 255, 255)
        cv2.putText(
            image,
            line,
            (x + 6, y + 16 + index * line_height),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return selected


def draw_workspace_coordinate_axes(
    image: np.ndarray,
    camera_from_workspace: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    length_m: float = 0.1,
) -> None:
    """Project the workspace axes and explicitly mark their shared origin."""
    axes = np.asarray(
        [[0, 0, 0], [length_m, 0, 0], [0, length_m, 0], [0, 0, length_m]],
        dtype=np.float64,
    )
    transform = np.asarray(camera_from_workspace, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    projected, _ = cv2.projectPoints(
        axes, rvec, transform[:3, 3], camera_matrix, distortion
    )
    points = np.rint(projected.reshape(-1, 2)).astype(int)
    origin = points[0]
    thickness = max(2, int(round(min(image.shape[:2]) / 300.0)))
    for label, endpoint, color in (
        ("+X", points[1], (0, 0, 255)),
        ("+Y", points[2], (0, 200, 0)),
        ("+Z", points[3], (255, 80, 0)),
    ):
        cv2.arrowedLine(
            image, tuple(origin), tuple(endpoint), color, thickness, cv2.LINE_AA, tipLength=0.15
        )
        _outlined_text(image, label, tuple(endpoint + np.asarray([5, -5])), color, thickness)
    cv2.drawMarker(
        image, tuple(origin), (0, 255, 255), cv2.MARKER_CROSS, 20, thickness + 1, cv2.LINE_AA
    )
    _outlined_text(
        image,
        "WORKSPACE ORIGIN O=(0, 0, 0) mm",
        tuple(origin + np.asarray([8, 22])),
        (0, 255, 255),
        thickness,
        scale=0.52,
    )


def quad_center(corners: np.ndarray) -> np.ndarray:
    """Return the projective center as the intersection of both diagonals."""
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    p0, p1, p2, p3 = points
    a = np.column_stack((p2 - p0, -(p3 - p1)))
    b = p1 - p0
    if abs(np.linalg.det(a)) < 1e-12:
        return points.mean(axis=0)
    scale = np.linalg.solve(a, b)[0]
    return p0 + scale * (p2 - p0)


def tag_world_corners(origin_m, yaw_deg: float, tag_size_m: float) -> np.ndarray:
    """Return TL, TR, BR, BL from the black-frame top-left origin.

    At yaw=0, +X follows TL->TR, +Y follows TL->BL, and +Z points into
    the printed plane. Positive yaw rotates +X toward +Y.
    """
    origin = np.asarray(origin_m, dtype=np.float64).reshape(3)
    local = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [tag_size_m, 0.0, 0.0],
            [tag_size_m, tag_size_m, 0.0],
            [0.0, tag_size_m, 0.0],
        ],
        dtype=np.float64,
    )
    yaw = math.radians(yaw_deg)
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (rotation @ local.T).T + origin


def camera_matrix_from_yaml(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    matrix_entry = data.get("camera_matrix", data.get("K"))
    distortion_entry = data.get("distortion_coefficients", data.get("D"))
    matrix_data = matrix_entry["data"] if isinstance(matrix_entry, dict) else matrix_entry
    distortion_data = (
        distortion_entry["data"] if isinstance(distortion_entry, dict) else distortion_entry
    )
    camera_matrix = np.asarray(matrix_data, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(distortion_data, dtype=np.float64).reshape(-1, 1)
    image_size = (int(data["image_width"]), int(data["image_height"]))
    return camera_matrix, distortion, image_size, data


def save_camera_yaml(
    path: str,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: Tuple[int, int],
    camera_name: str,
) -> None:
    width, height = image_size
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    d = np.asarray(distortion, dtype=np.float64).reshape(-1)
    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:, :3] = k
    output = {
        "image_width": int(width),
        "image_height": int(height),
        "camera_name": camera_name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": k.reshape(-1).tolist()},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(d.size),
            "data": d.tolist(),
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": np.eye(3, dtype=np.float64).reshape(-1).tolist(),
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": projection.reshape(-1).tolist(),
        },
    }
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False, allow_unicode=True)


def load_layout(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        layout = yaml.safe_load(handle)
    if layout.get("dictionary") != APRILTAG_DICTIONARY_NAME:
        raise ValueError("tag layout must use DICT_APRILTAG_36h11")
    require_coordinate_convention(layout, "tag layout")
    calibration_tags = layout.get("calibration_tags", {})
    if len(calibration_tags) < 3:
        raise ValueError("at least three calibration tags are required")
    origins = [np.asarray(entry["origin_mm"], dtype=np.float64) for entry in calibration_tags.values()]
    xy = np.asarray([origin[:2] for origin in origins])
    if np.linalg.matrix_rank(xy[1:] - xy[0]) < 2:
        raise ValueError("calibration tag origins must not be collinear")
    return layout


def require_coordinate_convention(data: dict, source_name: str = "calibration data") -> dict:
    convention = data.get("coordinate_convention", {}) if isinstance(data, dict) else {}
    if convention.get("id") != COORDINATE_CONVENTION_ID:
        raise ValueError("{}: {}".format(source_name, INCOMPATIBLE_CONVENTION_MESSAGE))
    return convention


def layout_origins_mm(layout: dict, include_validation: bool = False) -> Dict[int, np.ndarray]:
    origins = {
        int(tag_id): np.asarray(entry["origin_mm"], dtype=np.float64)
        for tag_id, entry in layout.get("calibration_tags", {}).items()
    }
    if include_validation and layout.get("validation_tag"):
        entry = layout["validation_tag"]
        origins[int(entry["id"])] = np.asarray(entry["origin_mm"], dtype=np.float64)
    return origins


def transform_from_pose(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera_from_workspace = np.eye(4, dtype=np.float64)
    camera_from_workspace[:3, :3] = rotation
    camera_from_workspace[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    workspace_from_camera = np.linalg.inv(camera_from_workspace)
    return camera_from_workspace, workspace_from_camera


def pixel_to_workspace_plane(
    pixel,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    camera_from_workspace: np.ndarray,
    plane_z_m: float,
) -> np.ndarray:
    pixel_array = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
    normalized = cv2.undistortPoints(pixel_array, camera_matrix, distortion).reshape(2)
    ray_camera = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
    workspace_from_camera = np.linalg.inv(camera_from_workspace)
    origin_workspace = workspace_from_camera[:3, 3]
    ray_workspace = workspace_from_camera[:3, :3] @ ray_camera
    if abs(ray_workspace[2]) < 1e-10:
        raise ValueError("camera ray is parallel to the workspace plane")
    distance = (plane_z_m - origin_workspace[2]) / ray_workspace[2]
    if distance <= 0:
        raise ValueError("workspace plane lies behind the camera")
    return origin_workspace + distance * ray_workspace


class RosFrameSource:
    def __init__(self, topic: str, node_name: str):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
        self._rospy = rospy
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._subscriber = rospy.Subscriber(topic, Image, self._callback, queue_size=1)

    def _callback(self, message) -> None:
        frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        with self._lock:
            self._frame = frame
        self._event.set()

    def read(self, timeout_s: float = 3.0) -> np.ndarray:
        if not self._event.wait(timeout_s):
            raise TimeoutError("timed out waiting for a ROS image")
        with self._lock:
            frame = self._frame.copy()
            self._event.clear()
        return frame

    def close(self) -> None:
        self._subscriber.unregister()


class V4L2FrameSource:
    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        fourcc: str = "YUYV",
    ):
        if len(fourcc) != 4:
            raise ValueError("V4L2 FOURCC must contain exactly four characters")
        self._capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._capture.isOpened():
            raise RuntimeError("failed to open V4L2 device: {}".format(device))
        actual_size = (
            int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        if actual_size != (width, height):
            self._capture.release()
            raise RuntimeError(
                "camera returned {}x{} instead of requested {}x{}".format(
                    actual_size[0], actual_size[1], width, height
                )
            )

    def read(self, timeout_s: float = 3.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ok, frame = self._capture.read()
            if ok:
                return frame
        raise TimeoutError("timed out waiting for a V4L2 image")

    def close(self) -> None:
        self._capture.release()


def add_source_arguments(parser) -> None:
    parser.add_argument("--input", choices=("ros", "v4l2"), default="ros")
    parser.add_argument("--topic", default="/usb_cam/image_raw")
    parser.add_argument(
        "--device",
        default=(
            "/dev/v4l/by-id/"
            "usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0"
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")


def open_frame_source(args, node_name: str):
    if args.input == "ros":
        return RosFrameSource(args.topic, node_name)
    return V4L2FrameSource(
        args.device, args.width, args.height, args.fps, args.fourcc
    )
