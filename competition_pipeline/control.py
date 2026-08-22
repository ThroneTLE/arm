"""Fail-closed safety gate around a vendor robot controller.

``speed_scale`` 是**无量纲比例**，这条路径最终变成 MOVL 的 mm/s：

    speed_scale ×1000 = mm/s        (NexBotTcpRobotController.move_to)

所以 ``safety.maximum_speed_scale=0.2`` 的实际含义是 **200 mm/s 上限**。
不要拿 ``move_j`` 的 ``speed_scale``（那是 ×100 的百分比）来类比 —— 同名
不同义，混用会把 50 mm/s 的意图变成 500 mm/s。详见 ``nexbot_move`` 模块头。
"""

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
        """``speed_scale`` 无量纲，×1000 = mm/s（见模块 docstring）。"""
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
        # ⚠️ 在 C1102 上这条是 0x2314 -> Deadan_End -> PowerOff，即**下电**：
        # 伸展着的手臂会失力下坠，且之后必须重新 0x2311 使能才能再动。
        # 真正的安全急停是示教器上的物理按钮，不是这里。
        # 运动尚未下发时不要"保险起见"调它 —— 那是纯粹的自伤。
        return self.adapter.stop()
