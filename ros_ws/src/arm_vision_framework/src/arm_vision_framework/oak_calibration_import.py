"""Offline reader for official Luxonis/DepthAI EEPROM JSON calibration files."""

from pathlib import Path

import numpy as np


def _depthai():
    try:
        import depthai as dai
    except ImportError as error:
        raise RuntimeError(
            "未安装 DepthAI；请先安装官方 DepthAI Python 包后再导入 OAK EEPROM JSON"
        ) from error
    return dai


def _text(value):
    return "" if value is None else str(value).strip()


def _intrinsics(calibration, socket, width, height, label):
    try:
        matrix = np.asarray(
            calibration.getCameraIntrinsics(socket, int(width), int(height)),
            dtype=np.float64,
        ).reshape(3, 3)
        distortion = np.asarray(
            calibration.getDistortionCoefficients(socket), dtype=np.float64
        ).reshape(-1)
    except Exception as error:
        raise ValueError("OAK EEPROM 缺少 {} 相机内参：{}".format(label, error)) from error
    if (
        not np.all(np.isfinite(matrix))
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or distortion.size < 4
        or not np.all(np.isfinite(distortion))
    ):
        raise ValueError("OAK EEPROM 的 {} 相机内参或畸变参数无效".format(label))
    return matrix, distortion


def inspect_oak_eeprom(source_json, color_width=1920, color_height=1080,
                       depth_width=1280, depth_height=800):
    """Read official EEPROM JSON without requiring an OAK device.

    The product PDF specifies DepthAI support but does not define the EEPROM
    JSON schema or a flash command.  DepthAI's official ``CalibrationHandler``
    is therefore the source of truth for this offline import.
    """

    source = Path(source_json).expanduser().resolve()
    if not source.is_file():
        raise ValueError("OAK EEPROM JSON 不存在：{}".format(source))
    color_width, color_height = int(color_width), int(color_height)
    depth_width, depth_height = int(depth_width), int(depth_height)
    if color_width <= 0 or color_height <= 0:
        raise ValueError("OAK RGB 输出分辨率必须为正数")
    if depth_width <= 0 or depth_height <= 0:
        raise ValueError("OAK 深度输出分辨率必须为正数")
    dai = _depthai()
    try:
        calibration = dai.CalibrationHandler(str(source))
    except Exception as error:
        raise ValueError("无法读取 OAK EEPROM JSON：{}".format(error)) from error

    rgb_socket = dai.CameraBoardSocket.CAM_A
    right_socket = dai.CameraBoardSocket.CAM_C
    color_matrix, color_distortion = _intrinsics(
        calibration, rgb_socket, color_width, color_height, "RGB CAM_A"
    )
    depth_matrix, depth_distortion = _intrinsics(
        calibration, right_socket, depth_width, depth_height, "深度 CAM_C"
    )

    baseline_mm = None
    try:
        baseline_cm = float(
            calibration.getBaselineDistance(
                dai.CameraBoardSocket.CAM_B,
                dai.CameraBoardSocket.CAM_C,
                False,
            )
        )
        if baseline_cm > 0.0:
            baseline_mm = baseline_cm * 10.0
    except Exception:
        pass

    eeprom = calibration.getEepromData()
    return {
        "source_file": str(source),
        "product_name": _text(getattr(eeprom, "productName", "")),
        "board_name": _text(getattr(eeprom, "boardName", "")),
        "board_revision": _text(getattr(eeprom, "boardRev", "")),
        "device_name": _text(getattr(eeprom, "deviceName", "")),
        "eeprom_version": int(getattr(eeprom, "version", 0)),
        "color": {
            "image_width": color_width,
            "image_height": color_height,
            "camera_matrix": color_matrix,
            "distortion_coefficients": color_distortion,
        },
        "depth": {
            "image_width": depth_width,
            "image_height": depth_height,
            "camera_matrix": depth_matrix,
            "distortion_coefficients": depth_distortion,
        },
        "baseline_mm": baseline_mm,
    }


__all__ = ["inspect_oak_eeprom"]

