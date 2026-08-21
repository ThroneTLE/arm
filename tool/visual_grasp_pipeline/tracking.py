"""Small, hardware-free tracking helpers from the released grasp UI."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def box_iou(a, b) -> float:
    """Intersection-over-union for two ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class StableTracker:
    """Frame-to-frame IOU tracker that keeps IDs across short misses.

    The original release used this in the Tk UI to let the user select a
    specific instance (``can #2``) even when the detector does not output
    built-in track IDs.
    """

    def __init__(self, max_miss: int = 15, match_iou: float = 0.1):
        self.tracks: Dict[int, Tuple[Tuple[float, float, float, float], str, int]] = {}
        self.next_id = 1
        self.max_miss = int(max_miss)
        self.match_iou = float(match_iou)

    def update(self, objs: List[dict]) -> List[dict]:
        unmatched = list(range(len(objs)))
        new_tracks = {}

        for tid in list(self.tracks):
            txy, tname, miss = self.tracks[tid]
            best_i = -1
            best_v = self.match_iou
            for i in unmatched:
                obj = objs[i]
                if obj["name"] != tname:
                    continue
                value = box_iou(txy, obj["xyxy"])
                if value > best_v:
                    best_v = value
                    best_i = i
            if best_i >= 0:
                objs[best_i]["id"] = tid
                new_tracks[tid] = (objs[best_i]["xyxy"], tname, 0)
                unmatched.remove(best_i)
            else:
                new_tracks[tid] = (txy, tname, miss + 1)

        for i in unmatched:
            obj = objs[i]
            obj["id"] = self.next_id
            new_tracks[self.next_id] = (obj["xyxy"], obj["name"], 0)
            self.next_id += 1

        self.tracks = {
            tid: value
            for tid, value in new_tracks.items()
            if value[2] <= self.max_miss
        }
        return objs


def add_seq(objs: List[dict]) -> List[dict]:
    """Add per-class sequence numbers ``seq`` (1-based) as a display fallback."""
    counter = {}
    for obj in objs:
        counter[obj["name"]] = counter.get(obj["name"], 0) + 1
        obj["seq"] = counter[obj["name"]]
    return objs


def parse_sequence(text: str) -> List[Tuple[str, Optional[int]]]:
    """Parse ``can#2, red_apple`` into ``[('can', 2), ('red_apple', None)]``."""
    sequence = []
    for item in str(text).replace("，", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "#" in item:
            name, instance = item.split("#", 1)
            sequence.append((name.strip(), int(instance.strip())))
        else:
            sequence.append((item, None))
    return sequence
