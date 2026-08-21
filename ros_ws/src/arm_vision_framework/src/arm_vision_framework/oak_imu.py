"""OAK-D Pro IMU configuration without assumptions about sensor axes."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OakImuConfig:
    enabled: bool = False
    topic: str = "/camera/imu"
    frame_id: str = "oak_imu"
    imu_frame: str = "oak_imu"
    camera_from_imu: object = None
    report_rate_hz: float = 100.0

    def __post_init__(self):
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "topic", str(self.topic))
        object.__setattr__(self, "frame_id", str(self.frame_id))
        object.__setattr__(self, "imu_frame", str(self.imu_frame))
        rate = float(self.report_rate_hz)
        if rate <= 0.0 or rate > 500.0:
            raise ValueError("IMU report_rate_hz must be within (0, 500]")
        object.__setattr__(self, "report_rate_hz", rate)
        if self.camera_from_imu is not None:
            matrix = np.asarray(self.camera_from_imu, dtype=np.float64)
            if matrix.size != 16:
                raise ValueError("camera_from_imu must be a 4x4 matrix")
            matrix = matrix.reshape(4, 4)
            if not np.all(np.isfinite(matrix)) or not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise ValueError("camera_from_imu must be a finite homogeneous transform")
            object.__setattr__(self, "camera_from_imu", matrix.tolist())

    @property
    def axis_transform_known(self):
        return self.camera_from_imu is not None


def config_from_camera(camera):
    oak = camera.get("oak_d_pro", camera)
    entry = dict(oak.get("imu", {}) or {})
    return OakImuConfig(
        enabled=entry.get("enabled", False), topic=entry.get("topic", "/camera/imu"),
        frame_id=entry.get("frame_id", entry.get("imu_frame", "oak_imu")),
        imu_frame=entry.get("imu_frame", "oak_imu"),
        camera_from_imu=entry.get("camera_from_imu"),
        report_rate_hz=entry.get("report_rate_hz", 100.0),
    )


__all__ = ["OakImuConfig", "config_from_camera"]
