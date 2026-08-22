"""Import and export Luxonis DepthAI EEPROM calibration data."""

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from .configuration import atomic_write_yaml


def _depthai():
    try:
        import depthai as dai
    except ImportError as error:
        raise RuntimeError(
            "当前 Python 环境未安装 depthai 2.x；请先运行 install_oak_support.sh"
        ) from error
    return dai


def _text(value):
    return "" if value is None else str(value).strip()


def _normalize_distortion(coefficients):
    values = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    rational = values.size >= 8 and np.any(np.abs(values[5:8]) > 1.0e-12)
    if rational:
        length = values.size if np.any(np.abs(values[8:]) > 1.0e-12) else 8
        return "rational_polynomial", values[:length]
    length = 5 if values.size >= 5 else values.size
    return "plumb_bob", values[:length]


def _eeprom_metadata(calibration):
    data = calibration.getEepromData()
    return {
        "product_name": _text(getattr(data, "productName", "")),
        "board_name": _text(getattr(data, "boardName", "")),
        "board_revision": _text(getattr(data, "boardRev", "")),
        "device_name": _text(getattr(data, "deviceName", "")),
        "version": int(getattr(data, "version", 0)),
    }


def _read_calibration(path):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("OAK 标定 JSON 不存在：{}".format(source))
    dai = _depthai()
    try:
        return source, dai, dai.CalibrationHandler(str(source))
    except Exception as error:
        raise ValueError("无法读取 DepthAI EEPROM JSON：{}".format(error)) from error


def inspect_oak_calibration(path, width=1920, height=1080):
    """Return validated RGB intrinsics and stereo metadata from an EEPROM JSON."""
    source, dai, calibration = _read_calibration(path)
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("OAK RGB 输出分辨率必须为正数")
    try:
        matrix = np.asarray(
            calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, width, height
            ),
            dtype=np.float64,
        ).reshape(3, 3)
        raw_distortion = np.asarray(
            calibration.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A),
            dtype=np.float64,
        ).reshape(-1)
    except Exception as error:
        raise ValueError("JSON 中没有可用的 OAK RGB(CAM_A) 内参：{}".format(error)) from error
    if (
        not np.all(np.isfinite(matrix))
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or raw_distortion.size < 4
        or not np.all(np.isfinite(raw_distortion))
    ):
        raise ValueError("OAK RGB 内参或畸变参数无效")
    baseline_cm = None
    stereo_extrinsics = None
    try:
        baseline_cm = float(
            calibration.getBaselineDistance(
                dai.CameraBoardSocket.CAM_B,
                dai.CameraBoardSocket.CAM_C,
                False,
            )
        )
        stereo_extrinsics = np.asarray(
            calibration.getCameraExtrinsics(
                dai.CameraBoardSocket.CAM_B,
                dai.CameraBoardSocket.CAM_C,
                False,
            ),
            dtype=np.float64,
        ).reshape(4, 4)
    except Exception:
        pass
    if (
        baseline_cm is None
        or baseline_cm <= 0.0
        or stereo_extrinsics is None
        or not np.all(np.isfinite(stereo_extrinsics))
    ):
        raise ValueError(
            "OAK JSON 缺少有效的 CAM_B/C 双目外参或基线；不能作为 RGB-D 工厂标定导入"
        )
    distortion_model, distortion = _normalize_distortion(raw_distortion)
    metadata = _eeprom_metadata(calibration)
    metadata.update(
        {
            "source_file": str(source),
            "image_width": width,
            "image_height": height,
            "camera_matrix": matrix,
            "distortion_coefficients": distortion,
            "distortion_model": distortion_model,
            "baseline_mm": None if baseline_cm is None else baseline_cm * 10.0,
            "cam_b_from_cam_c": stereo_extrinsics,
        }
    )
    return metadata


def _copy_atomic(source, destination):
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        return destination
    handle = tempfile.NamedTemporaryFile(
        dir=str(destination.parent), prefix=".{}-".format(destination.name),
        suffix=".tmp", delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        shutil.copy2(str(source), str(temporary))
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def import_oak_calibration(source_json, output_json, output_yaml, width=1920, height=1080):
    """Validate an official JSON, preserve it, and write canonical RGB YAML."""
    info = inspect_oak_calibration(source_json, width, height)
    saved_json = _copy_atomic(source_json, output_json)
    matrix = info["camera_matrix"]
    distortion = info["distortion_coefficients"]
    yaml_data = {
        "schema_version": 1,
        "source": {
            "type": "depthai_eeprom_json",
            "factory_calibration_file": str(saved_json),
            "product_name": info["product_name"],
            "board_name": info["board_name"],
            "board_revision": info["board_revision"],
            "device_name": info["device_name"],
            "eeprom_version": info["version"],
            "stereo_baseline_mm": info["baseline_mm"],
            "depth_aligned_to_color": True,
        },
        "cameras": {
            "color": {
                "camera_name": "oak_rgb_cam_a",
                "image_width": int(width),
                "image_height": int(height),
                "distortion_model": info["distortion_model"],
                "camera_matrix": matrix.tolist(),
                "distortion_coefficients": distortion.tolist(),
            }
        },
    }
    atomic_write_yaml(output_yaml, yaml_data)
    info["saved_json"] = str(saved_json)
    info["saved_yaml"] = str(Path(output_yaml).expanduser().resolve())
    return info


def export_connected_oak_eeprom(
    output_json, output_yaml, width=1920, height=1080, mxid="",
):
    """Read the connected device EEPROM, save official JSON, and import it."""
    dai = _depthai()
    output_json = Path(output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_name(".{}-device.tmp.json".format(output_json.stem))
    try:
        requested = str(mxid or "").strip()
        devices = list(dai.Device.getAllAvailableDevices())
        if requested:
            deadline = time.monotonic() + 5.0
            devices = []
            while not devices and time.monotonic() < deadline:
                devices = [
                    info for info in dai.Device.getAllAvailableDevices()
                    if str(info.getMxId()).strip() == requested
                ]
                if not devices:
                    time.sleep(0.25)
            if not devices:
                raise RuntimeError("未找到配置的 OAK MXID {}".format(requested))
        elif len(devices) > 1:
            raise RuntimeError("检测到多个 OAK，请先配置 camera profile 的 mxid")
        with (dai.Device(devices[0]) if devices else dai.Device()) as device:
            calibration = device.readCalibration()
            calibration.eepromToJsonFile(str(temporary))
        return import_oak_calibration(
            temporary, output_json, output_yaml, width=width, height=height
        )
    except Exception as error:
        raise RuntimeError("无法从已连接 OAK 导出 EEPROM：{}".format(error)) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def format_oak_summary(info):
    identity = info.get("product_name") or info.get("board_name") or "OAK"
    baseline = info.get("baseline_mm")
    matrix = np.asarray(info["camera_matrix"])
    return (
        "{} · {}x{} · fx/fy {:.2f}/{:.2f} · cx/cy {:.2f}/{:.2f} · 基线 {}"
    ).format(
        identity,
        info["image_width"], info["image_height"],
        matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2],
        "未知" if baseline is None else "{:.2f} mm".format(baseline),
    )
