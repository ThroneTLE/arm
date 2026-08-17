"""RViz MarkerArray visualization for parallel-jaw grasp candidates."""

from typing import List

import numpy as np

from .anygrasp_planner import WorkspaceGrasp


def _score_color(score: float, best: float) -> tuple:
    """Green for the best score, fading through yellow to red for lower scores."""
    fraction = 0.0 if best <= 1e-9 else max(0.0, min(1.0, score / best))
    red = 1.0 - 0.6 * fraction
    green = 0.25 + 0.75 * fraction
    return float(red), float(green), 0.12


def build_gripper_markers(
    grasps: List[WorkspaceGrasp],
    frame_id: str,
    stamp,
    max_markers: int = 20,
    namespace: str = "grasps",
    gripper_height: float = 0.03,
) -> List[object]:
    """Build one palm + two finger box markers per grasp.

    The grasp frame follows graspnetAPI: X is the approach direction (toward
    the object), Y is the open/close direction, Z completes the right-handed
    frame. The palm sits behind the origin, the fingers extend along +X.
    """
    import visualization_msgs.msg as visualization_msgs
    from tf.transformations import quaternion_from_matrix

    Marker = visualization_msgs.Marker
    markers = []
    best = max([float(grasp.score) for grasp in grasps] + [1e-9])
    count = min(max(0, int(max_markers)), len(grasps))
    for index, grasp in enumerate(grasps[:count]):
        rotation = np.asarray(grasp.rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(grasp.translation, dtype=np.float64).reshape(3)
        pose_matrix = np.eye(4, dtype=np.float64)
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = translation
        quaternion = quaternion_from_matrix(pose_matrix)
        color = _score_color(float(grasp.score), best)
        # RViz DELETEALL 按 ns 精确匹配：所有抓取共用 ns=namespace，
        # id 用 index*4+part 保持唯一。
        marker_base_id = index * 4

        width = max(float(grasp.width), 0.01)
        finger_height = max(float(grasp.height), float(gripper_height), 0.012)

        palm = Marker()
        palm.header.frame_id = frame_id
        palm.header.stamp = stamp
        palm.ns = namespace
        palm.id = marker_base_id + 0
        palm.type = Marker.CUBE
        palm.action = Marker.ADD
        palm.pose.position.x = -0.012
        palm.pose.position.y = 0.0
        palm.pose.position.z = 0.0
        palm.pose.orientation.x = quaternion[0]
        palm.pose.orientation.y = quaternion[1]
        palm.pose.orientation.z = quaternion[2]
        palm.pose.orientation.w = quaternion[3]
        palm.scale.x = 0.028
        palm.scale.y = min(width + 0.03, 0.12)
        palm.scale.z = 0.018
        palm.color.r, palm.color.g, palm.color.b = color
        palm.color.a = 0.85
        markers.append(palm)

        for side, sign, part in (("left", -1.0, 1), ("right", 1.0, 2)):
            finger = Marker()
            finger.header.frame_id = frame_id
            finger.header.stamp = stamp
            finger.ns = namespace
            finger.id = marker_base_id + part
            finger.type = Marker.CUBE
            finger.action = Marker.ADD
            finger.pose.position.x = 0.012
            finger.pose.position.y = sign * (width / 2.0 + 0.006)
            finger.pose.position.z = 0.0
            finger.pose.orientation.x = quaternion[0]
            finger.pose.orientation.y = quaternion[1]
            finger.pose.orientation.z = quaternion[2]
            finger.pose.orientation.w = quaternion[3]
            finger.scale.x = 0.045
            finger.scale.y = 0.012
            finger.scale.z = finger_height
            finger.color.r, finger.color.g, finger.color.b = color
            finger.color.a = 0.95
            markers.append(finger)

        approach_line = Marker()
        approach_line.header.frame_id = frame_id
        approach_line.header.stamp = stamp
        approach_line.ns = namespace
        approach_line.id = marker_base_id + 3
        approach_line.type = Marker.LINE_LIST
        approach_line.action = Marker.ADD
        approach_line.pose.orientation.w = 1.0
        approach_line.scale.x = 0.004
        approach_line.color.r, approach_line.color.g, approach_line.color.b = color
        approach_line.color.a = 0.9
        approach_line.points = [
            _point(translation),
            _point(np.asarray(grasp.tip, dtype=np.float64).reshape(3)),
        ]
        markers.append(approach_line)

    return markers


def build_best_grasp_label(
    grasp: WorkspaceGrasp, frame_id: str, stamp, namespace: str = "best_grasp_label"
) -> object:
    """Text label above the best grasp."""
    import visualization_msgs.msg as visualization_msgs

    Marker = visualization_msgs.Marker
    label = Marker()
    label.header.frame_id = frame_id
    label.header.stamp = stamp
    label.ns = namespace
    label.id = 0
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    tip = np.asarray(grasp.tip, dtype=np.float64).reshape(3)
    label.pose.position.x = float(tip[0])
    label.pose.position.y = float(tip[1])
    label.pose.position.z = float(tip[2] + 0.05)
    label.pose.orientation.w = 1.0
    label.scale.z = 0.02
    label.color.r = 1.0
    label.color.g = 0.95
    label.color.b = 0.4
    label.color.a = 1.0
    label.text = "BEST grasp score={:.2f} width={:.1f} mm\ntip=({:.1f}, {:.1f}, {:.1f}) mm".format(
        grasp.score,
        grasp.width * 1000.0,
        *(tip * 1000.0),
    )
    return label


def delete_all_markers(frame_id: str, stamp, namespace: str = "grasps") -> object:
    import visualization_msgs.msg as visualization_msgs

    Marker = visualization_msgs.Marker
    delete = Marker()
    delete.header.frame_id = frame_id
    delete.header.stamp = stamp
    delete.ns = namespace
    delete.id = 0
    delete.action = Marker.DELETEALL
    return delete


def _point(values: np.ndarray):
    from geometry_msgs.msg import Point

    point = Point()
    point.x, point.y, point.z = (float(value) for value in np.asarray(values).reshape(3))
    return point
