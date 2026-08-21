"""YOLO detection and AprilTag helpers migrated from the released pipeline."""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np

from .tracking import StableTracker, add_seq

# Common AprilTag families tried in order.
TAG_DICTS = [
    (cv2.aruco.DICT_APRILTAG_25h9, "APRILTAG_25h9"),
    (cv2.aruco.DICT_APRILTAG_16h5, "APRILTAG_16h5"),
    (cv2.aruco.DICT_APRILTAG_36h11, "APRILTAG_36h11"),
]


def detect_all_objects(rgb, model, conf: float = 0.85, imgsz: int = 640) -> List[dict]:
    """Run YOLO and return a list of object dicts with boxes, masks and names."""
    height, width = rgb.shape[:2]
    result = model.predict(rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    objects = []
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    if result.boxes is None or len(result.boxes) == 0:
        return objects

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for index, (box, confidence, cls) in enumerate(zip(boxes, confidences, classes)):
        x1, y1, x2, y2 = box.astype(int)
        mask = None
        if masks is not None and index < len(masks):
            candidate = (masks[index] > 0.5).astype(np.uint8) * 255
            if candidate.shape[:2] != (height, width):
                candidate = cv2.resize(
                    candidate, (width, height), interpolation=cv2.INTER_NEAREST
                )
            mask = candidate
        objects.append(
            {
                "xyxy": (
                    max(0, int(x1)),
                    max(0, int(y1)),
                    min(width, int(x2)),
                    min(height, int(y2)),
                ),
                "cls": int(cls),
                "name": model.names[cls],
                "conf": float(confidence),
                "mask": mask,
            }
        )
    return objects


def detect_all_track(rgb, model, tracker: StableTracker, conf: float = 0.85, imgsz: int = 640):
    """Detect objects and assign stable IDs using ``StableTracker``."""
    objects = detect_all_objects(rgb, model, conf=conf, imgsz=imgsz)
    if not objects:
        tracker.update([])
        return objects
    # The original function also copied built-in YOLO track IDs when available;
    # this migration keeps the simpler IOU tracker as the single source of IDs.
    objects = tracker.update(objects)
    return add_seq(objects)


def select_target(objects: List[dict], label: Optional[str]) -> Optional[dict]:
    """Select a target object by class name or class id; otherwise largest box."""
    if not objects:
        return None
    if label is not None:
        for obj in objects:
            if obj["name"] == label or obj["cls"] == label:
                return obj
        return None
    return max(
        objects,
        key=lambda obj: (obj["xyxy"][2] - obj["xyxy"][0])
        * (obj["xyxy"][3] - obj["xyxy"][1]),
    )


def detect_tags(rgb, size_mm: float, camera_matrix: np.ndarray, dist_coeffs=None):
    """Detect AprilTags and return ``[(tag_id, T_cam_tag), ...]``.

    ``camera_matrix`` is a 3x3 intrinsics matrix.  Distortion defaults to zero,
    matching the original release behaviour.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    size = float(size_mm) / 1000.0
    object_points = np.array(
        [
            [-size / 2, -size / 2, 0],
            [size / 2, -size / 2, 0],
            [size / 2, size / 2, 0],
            [-size / 2, size / 2, 0],
        ],
        dtype=np.float32,
    )
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    for dict_id, _name in TAG_DICTS:
        try:
            # OpenCV <= 4.7
            aruco_dict = cv2.aruco.Dictionary_get(dict_id)
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        except AttributeError:
            # OpenCV >= 4.7
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue
        found = []
        for index, tag_id in enumerate(ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(
                object_points, corners[index][0], camera_matrix, dist_coeffs
            )
            if not ok:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            if rotation[1, 2] > 0:
                rotation = rotation @ np.diag([1.0, -1.0, -1.0])
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = tvec.flatten()
            found.append((int(tag_id), transform))
        if found:
            return found
    return []


def draw_boxes(rgb, objects, color=(255, 0, 0)) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR image copy."""
    out = rgb.copy()
    for obj in objects:
        x1, y1, x2, y2 = obj["xyxy"]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        tag = obj.get("id") if obj.get("id") is not None else obj.get("seq")
        cv2.putText(
            out,
            f"{obj['name']} #{tag}",
            (int(x1), max(0, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return out
