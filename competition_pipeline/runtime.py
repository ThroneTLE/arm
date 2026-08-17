"""Small runtime composition root for field camera and robot adapters."""

from .configuration import load_camera_intrinsics
from .localization import HybridLocalizer


class CompetitionRuntime:
    def __init__(self, config, robot_pose_provider=None, intrinsics_path=None):
        self.config = config
        if intrinsics_path is None:
            intrinsics_path = config.resolve_path(
                config.camera["color_intrinsics_file"]
            )
        self.camera_matrix, self.distortion, self.image_size = load_camera_intrinsics(
            intrinsics_path
        )
        self.robot_pose_provider = robot_pose_provider
        self.localizer = HybridLocalizer(config)

    def localize(self, color_bgr, timestamp_s):
        """Localize from the RGB stream, with the robot pose as fallback."""
        if (color_bgr.shape[1], color_bgr.shape[0]) != self.image_size:
            raise ValueError("RGB image size does not match calibrated intrinsics")
        robot = None if self.robot_pose_provider is None else self.robot_pose_provider.latest_pose()
        return self.localizer.localize(
            color_bgr,
            self.camera_matrix,
            self.distortion,
            base_from_tcp=None if robot is None else robot.base_from_tcp,
            image_timestamp_s=timestamp_s,
            robot_timestamp_s=None if robot is None else robot.timestamp_s,
        )
