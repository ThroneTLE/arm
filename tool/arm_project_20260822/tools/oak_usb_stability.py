#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""USB 稳定性监测: 流传输时设备是否掉线重连(设备号变化 / 消失)。

同步做两件事:
 - 主线程: 开深度流连续取帧 10 秒
 - 子线程: 每 0.3s 记录 lsusb 的 03e7 设备号
"""
import subprocess
import threading
import time

import numpy as np
import depthai as dai


def build(light=False):
    p = dai.Pipeline()
    mono_l = p.create(dai.node.MonoCamera)
    mono_l.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P if light
        else dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_l.setFps(5)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r = p.create(dai.node.MonoCamera)
    mono_r.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P if light
        else dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_r.setFps(5)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    stereo = p.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)
    # 轻量模式: 深度对齐到左目(400p), 分辨率小, 更省带宽
    stereo.setDepthAlign(
        dai.CameraBoardSocket.CAM_B if light else dai.CameraBoardSocket.CAM_A)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    x = p.create(dai.node.XLinkOut)
    x.setStreamName("depth")
    stereo.depth.link(x.input)
    return p


def usb_watch(stop, timeline):
    while not stop.is_set():
        try:
            out = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=2).stdout
            devs = [ln.split()[3] for ln in out.splitlines() if "03e7" in ln]
        except Exception:
            devs = ["ERR"]
        timeline.append((time.monotonic(), tuple(devs)))
        time.sleep(0.3)


stop = threading.Event()
timeline = []
t = threading.Thread(target=usb_watch, args=(stop, timeline), daemon=True)
t.start()

print("开始 10 秒深度流 + USB 监控(轻量 400p 模式)...")
with dai.Device(build(light=True)) as device:
    device.setIrLaserDotProjectorBrightness(800)
    q = device.getOutputQueue("depth", 4, False)
    t0 = time.time()
    frames = 0
    samples = []
    while time.time() - t0 < 10:
        d = q.get().getCvFrame()
        frames += 1
        if frames % 3 == 0:
            samples.append((round((time.time() - t0), 1), int(d[d.shape[0]//2, d.shape[1]//2])))
stop.set()

print(f"帧数: {frames}")
print("帧时间/中心像素值(前 12 个):", samples[:12])

# 分析 USB 时间线: 找设备号变化
prev = None
changes = 0
for t_, devs in timeline:
    if prev is not None and devs != prev:
        changes += 1
        print(f"  ⚠️ USB 变化 @{t_ - timeline[0][0]:.1f}s: {prev} -> {devs}")
    prev = devs
print(f"USB 时间线采样 {len(timeline)} 次, 变化 {changes} 次")
