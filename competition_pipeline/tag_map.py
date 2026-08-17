"""AprilTag map in the robot base frame."""

import numpy as np

from .geometry import transform_from_xyz_rpy_mm


class TagMap:
    """Map black-border Tag corners using a bottom-right local origin.

    OpenCV detection order is TL, TR, BR, BL. With BR as O, local +X follows
    the right edge from BR to TR, and local +Y follows the bottom edge from
    BR to BL. Both positive axes point inward, so the printed square occupies
    positive local X and positive local Y.
    """

    def __init__(self, config):
        self.config = config
        data = config.tag_map
        self.dictionary = str(data.get("dictionary", "DICT_APRILTAG_36h11"))
        self.tag_size_m = float(data["tag_size_mm"]) / 1000.0
        self.default_rpy_deg = np.asarray(
            data["default_base_from_tag_rpy_deg"], dtype=np.float64
        )

    @property
    def ids(self):
        return tuple(sorted(int(tag_id) for tag_id in self.config.tag_map.get("tags", {})))

    def entry(self, tag_id):
        entries = self.config.tag_map.get("tags", {})
        if int(tag_id) in entries:
            return entries[int(tag_id)]
        if str(int(tag_id)) in entries:
            return entries[str(int(tag_id))]
        raise KeyError("Tag ID {} is not in the map".format(tag_id))

    def base_from_tag(self, tag_id):
        entry = self.entry(tag_id)
        rpy = entry.get("base_from_tag_rpy_deg", self.default_rpy_deg)
        return transform_from_xyz_rpy_mm(entry["bottom_right_xyz_mm"], rpy)

    def corners_base_m(self, tag_id):
        size = self.tag_size_m
        local = np.asarray(
            [[size, size, 0.0], [size, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, size, 0.0]],
            dtype=np.float64,
        )
        transform = self.base_from_tag(tag_id)
        homogeneous = np.column_stack([local, np.ones(4, dtype=np.float64)])
        return (transform @ homogeneous.T).T[:, :3]

    def set_tag(self, tag_id, bottom_right_xyz_mm, rpy_deg=None):
        entry = {
            "bottom_right_xyz_mm": [float(value) for value in bottom_right_xyz_mm],
        }
        if rpy_deg is not None:
            entry["base_from_tag_rpy_deg"] = [float(value) for value in rpy_deg]
        self.config.tag_map.setdefault("tags", {})[int(tag_id)] = entry
        self._invalidate_hand_eye()
        self.config.save()

    def remove_tag(self, tag_id):
        entries = self.config.tag_map.setdefault("tags", {})
        removed = entries.pop(int(tag_id), None)
        if removed is None:
            removed = entries.pop(str(int(tag_id)), None)
        if removed is None:
            raise KeyError("Tag ID {} is not in the map".format(tag_id))
        self._invalidate_hand_eye()
        self.config.save()

    def set_default_rpy(self, rpy_deg):
        self.config.tag_map["default_base_from_tag_rpy_deg"] = [
            float(value) for value in rpy_deg
        ]
        self._invalidate_hand_eye()
        self.config.save()

    def _invalidate_hand_eye(self):
        hand_eye = self.config.data["hand_eye"]["tcp_from_color_camera"]
        hand_eye["valid"] = False
        hand_eye.pop("quality", None)
