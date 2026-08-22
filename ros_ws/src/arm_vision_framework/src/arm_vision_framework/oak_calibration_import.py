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


def normalize_distortion(coefficients):
    """Return a ROS-compatible model name and the meaningful coefficients.

    DepthAI commonly returns OpenCV's full 14-element vector even when the
    last coefficients are zero.  The OAK-D Pro EEPROM uses the rational terms
    k4..k6, so publishing it as ``plumb_bob`` (five coefficients) is wrong.
    """

    values = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if values.size < 4 or not np.all(np.isfinite(values)):
        raise ValueError("OAK 相机畸变参数无效")
    rational = values.size >= 8 and np.any(np.abs(values[5:8]) > 1.0e-12)
    if rational:
        # ROS rational_polynomial defines the first eight OpenCV terms. Keep
        # additional thin-prism/tilt terms only when the EEPROM really uses
        # them; DepthAI usually pads these positions with zeros.
        length = values.size if np.any(np.abs(values[8:]) > 1.0e-12) else 8
        return "rational_polynomial", values[:length].copy()
    length = 5 if values.size >= 5 else values.size
    return "plumb_bob", values[:length].copy()


def _intrinsics(calibration, socket, width, height, label):
    try:
        matrix = np.asarray(
            calibration.getCameraIntrinsics(socket, int(width), int(height)),
            dtype=np.float64,
        ).reshape(3, 3)
        raw_distortion = np.asarray(
            calibration.getDistortionCoefficients(socket), dtype=np.float64
        ).reshape(-1)
    except Exception as error:
        raise ValueError("OAK EEPROM 缺少 {} 相机内参：{}".format(label, error)) from error
    if (
        not np.all(np.isfinite(matrix))
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or raw_distortion.size < 4
        or not np.all(np.isfinite(raw_distortion))
    ):
        raise ValueError("OAK EEPROM 的 {} 相机内参或畸变参数无效".format(label))
    distortion_model, distortion = normalize_distortion(raw_distortion)
    return matrix, distortion_model, distortion


def export_oak_device_eeprom(destination, mxid=None):
    """Read calibration from one connected OAK device without flashing it."""

    dai = _depthai()
    devices = list(dai.Device.getAllAvailableDevices())
    requested_mxid = _text(mxid)
    if requested_mxid:
        devices = [info for info in devices if _text(info.getMxId()) == requested_mxid]
        if not devices:
            raise ValueError("未找到 MXID={} 的 OAK 相机".format(requested_mxid))
    elif not devices:
        raise ValueError("未检测到已连接的 OAK 相机")
    elif len(devices) > 1:
        ids = ", ".join(_text(info.getMxId()) or "unknown" for info in devices)
        raise ValueError("检测到多个 OAK 相机，请用 --mxid 指定：{}".format(ids))

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    info = devices[0]
    device = dai.Device(info)
    try:
        calibration = device.readCalibration()
        calibration.eepromToJsonFile(str(output))
        selected_mxid = _text(device.getMxId()) or _text(info.getMxId())
        usb_speed = _text(device.getUsbSpeed()) if hasattr(device, "getUsbSpeed") else ""
    except Exception as error:
        raise RuntimeError("读取 OAK 设备 EEPROM 失败：{}".format(error)) from error
    finally:
        device.close()
    return {
        "source_file": str(output),
        "mxid": selected_mxid,
        "usb_speed": usb_speed,
    }


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
    color_matrix, color_model, color_distortion = _intrinsics(
        calibration, rgb_socket, color_width, color_height, "RGB CAM_A"
    )
    depth_matrix, depth_model, depth_distortion = _intrinsics(
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
            "distortion_model": color_model,
            "camera_matrix": color_matrix,
            "distortion_coefficients": color_distortion,
        },
        "depth": {
            "image_width": depth_width,
            "image_height": depth_height,
            "distortion_model": depth_model,
            "camera_matrix": depth_matrix,
            "distortion_coefficients": depth_distortion,
        },
        "baseline_mm": baseline_mm,
    }


__all__ = [
    "export_oak_device_eeprom", "inspect_oak_eeprom", "normalize_distortion",
]
