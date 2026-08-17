"""Fail-closed safety gate around a vendor robot controller."""

import time

import numpy as np

from .geometry import as_transform, rotation_angle_deg


class MotionSafetyError(RuntimeError):
    pass


class SafeRobotController:
    def __init__(self, config, adapter):
        self.config = config
        self.adapter = adapter

    def move_tcp(self, base_from_tcp, speed_scale):
        safety = self.config.data.get("safety", {})
        if bool(safety.get("dry_run", True)):
            raise MotionSafetyError("dry_run is enabled")
        if not bool(safety.get("allow_robot_motion", False)):
            raise MotionSafetyError("robot motion is not enabled")
        target = as_transform(base_from_tcp, "target base_from_tcp")
        speed_scale = float(speed_scale)
        maximum_speed = float(safety.get("maximum_speed_scale", 0.2))
        if speed_scale <= 0.0 or speed_scale > maximum_speed:
            raise MotionSafetyError(
                "speed scale {:.3f} is outside (0, {:.3f}]".format(speed_scale, maximum_speed)
            )
        xyz_mm = target[:3, 3] * 1000.0
        minimum = np.asarray(safety["workspace_min_mm"], dtype=np.float64).reshape(3)
        maximum = np.asarray(safety["workspace_max_mm"], dtype=np.float64).reshape(3)
        if np.any(xyz_mm < minimum) or np.any(xyz_mm > maximum):
            raise MotionSafetyError(
                "target TCP {} mm is outside workspace {}..{} mm".format(
                    np.round(xyz_mm, 1).tolist(), minimum.tolist(), maximum.tolist()
                )
            )
        current = self.adapter.latest_pose()
        if current is None:
            raise MotionSafetyError("current TCP pose is unavailable")
        maximum_age = float(safety.get("maximum_robot_pose_age_s", 0.25))
        if abs(time.monotonic() - current.timestamp_s) > maximum_age:
            raise MotionSafetyError("current TCP pose is stale")
        jump_mm = np.linalg.norm(target[:3, 3] - current.base_from_tcp[:3, 3]) * 1000.0
        maximum_jump = float(safety.get("maximum_single_step_mm", 50.0))
        if jump_mm > maximum_jump:
            raise MotionSafetyError(
                "target jump {:.1f} mm exceeds {:.1f} mm".format(jump_mm, maximum_jump)
            )
        rotation_jump = rotation_angle_deg(current.base_from_tcp, target)
        maximum_rotation = float(safety.get("maximum_single_rotation_deg", 10.0))
        if rotation_jump > maximum_rotation:
            raise MotionSafetyError(
                "target rotation jump {:.1f} deg exceeds {:.1f} deg".format(
                    rotation_jump, maximum_rotation
                )
            )
        return self.adapter.move_tcp(target, speed_scale)

    def latest_pose(self):
        return self.adapter.latest_pose()

    def stop(self):
        # Emergency stop must remain callable even when normal motion is disabled.
        return self.adapter.stop()
