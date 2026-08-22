#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OAK-D Pro 深度链路深度诊断:
 1. EEPROM 立体标定数据(基线/矩形化矩阵)
 2. 左右目原始图(核对是否正常)
 3. 逐帧采样几个固定像素, 看抖动的模式
 4. 对比: 对齐RGB / 原始LEFT / 关扩展视差 三种配置的统计
"""
import os
import time

import cv2
import numpy as np
import depthai as dai

OUT = "/tmp/oak_diag"
os.makedirs(OUT, exist_ok=True)


def stats(depth, name):
    v = depth[(depth > 0) & (depth < 4000)]
    if v.size == 0:
        print(f"  {name}: 无有效像素")
        return
    print(f"  {name}: 有效 {100 * (v.size / depth.size):.1f}%  "
          f"p50 {np.median(v):.0f}mm  范围 {v.min():.0f}-{v.max():.0f}mm")


def build(align, extended=True):
    p = dai.Pipeline()
    mono_l = p.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r = p.create(dai.node.MonoCamera)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    stereo = p.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(120)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(align)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    x = p.create(dai.node.XLinkOut)
    x.setStreamName("depth")
    stereo.depth.link(x.input)
    xl = p.create(dai.node.XLinkOut); xl.setStreamName("left")
    mono_l.out.link(xl.input)
    xr = p.create(dai.node.XLinkOut); xr.setStreamName("right")
    mono_r.out.link(xr.input)
    return p


with dai.Device(build(dai.CameraBoardSocket.CAM_A)) as device:
    device.setIrLaserDotProjectorBrightness(800)
    print("=== [1] EEPROM 立体标定数据 ===")
    calib = device.readCalibration()
    eeprom = calib.getEepromData()
    print("  产品:", eeprom.productName, "| 板:", eeprom.boardName, "| 版本:", eeprom.version)
    try:
        left_intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
        right_intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)
        print("  左目 K[0,0]:", left_intr[0][0], "| 右目 K[0,0]:", right_intr[0][0])
        rect = calib.getStereoRectificationData(dai.CameraBoardSocket.CAM_B)
        print("  基线(rectified):", rect.SpecTranslationTransform[0][3], "m")
        print("  左目 rect K[0,0]:", rect.LeftCameraIntrinsics[0][0],
              "| 右目 rect K[0,0]:", rect.RightCameraIntrinsics[0][0])
        print("  左右矩形化后主点差(px):",
              rect.RightCameraIntrinsics[0][2] - rect.LeftCameraIntrinsics[0][2])
        print("  旋转矩阵接近单位阵?",
              np.allclose(rect.Rotation, np.eye(3), atol=1e-3),
              "| R[1,0],R[0,1]:", rect.Rotation[1][0], rect.Rotation[0][1])
    except Exception as e:
        print("  (读取立体标定失败:", e, ")")

    print("\n=== [2] 检查原始左右目 ===")
    ql = device.getOutputQueue("left", 4, False)
    qr = device.getOutputQueue("right", 4, False)
    l = ql.get().getCvFrame()
    r = qr.get().getCvFrame()
    cv2.imwrite(f"{OUT}/left.png", l)
    cv2.imwrite(f"{OUT}/right.png", r)
    print("  左右目已保存:", l.shape, "| 亮度差异(mean):", float(np.abs(l - r).mean()))

with dai.Device(build(dai.CameraBoardSocket.CAM_A)) as device:
    device.setIrLaserDotProjectorBrightness(800)
    q = device.getOutputQueue("depth", 4, False)
    print("\n=== [3] 逐帧固定像素采样(对齐RGB, 扩展视差) ===")
    samples = {p: [] for p in [(540, 960), (540, 700), (400, 500)]}
    for i in range(20):
        d = q.get().getCvFrame()
        for (y, x) in samples:
            samples[(y, x)].append(int(d[y, x]))
    for k, v in samples.items():
        v = np.array(v)
        print(f"  像素{k}: {v[:8].tolist()} ... 约 {len(set(v.tolist()))} 种取值, std={v.std():.0f}mm")

with dai.Device(build(dai.CameraBoardSocket.CAM_A, extended=False)) as device:
    device.setIrLaserDotProjectorBrightness(800)
    q = device.getOutputQueue("depth", 4, False)
    time.sleep(1.5)
    print("\n=== [4] 关扩展视差 配置 ===")
    for i in range(3):
        d = q.get().getCvFrame()
        stats(d, f"帧{i+1}")

print("\n诊断图:", OUT)
