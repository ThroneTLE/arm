#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OAK-D Pro 深度数据分析: 平面拟合残差 / 时间抖动 / 分布统计。

在静态场景下对深度质量做客观评估:
  1. 桌面 ROI 平面拟合: 残差 RMS 越小, 说明立体匹配越一致(标定与视差算法正常)
  2. 帧间时间抖动: 每像素 10 帧 std 的 median(静态场景应 <2-3mm)
  3. 有效像素率 / 深度单位核验 / 直方图
  4. 保存分析可视化(残差图/抖动图)
"""
import os
import time

import cv2
import numpy as np
import depthai as dai


def build_pipeline():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(5)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setInterleaved(False)
    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_l.setFps(5)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_r.setFps(5)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    xout_rgb = pipeline.create(dai.node.XLinkOut); xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)
    xout_depth = pipeline.create(dai.node.XLinkOut); xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    return pipeline


N_FRAMES = 12


def main():
    out_dir = "/tmp/oak_depth_analysis"
    os.makedirs(out_dir, exist_ok=True)
    with dai.Device(build_pipeline()) as device:
        print("设备:", [c.name for c in device.getConnectedCameras()])
        device.setIrLaserDotProjectorBrightness(800)
        q_rgb = device.getOutputQueue("rgb", 4, False)
        q_depth = device.getOutputQueue("depth", 4, False)

        print(f"预热 10 秒(跳过开机重枚举/匹配收敛期)...")
        time.sleep(10)
        print(f"采集 {N_FRAMES} 帧静态画面(健康帧筛选: 有效深度占比 >=50%)...")
        qd = q_depth
        frames = []
        tried = 0
        while len(frames) < N_FRAMES and tried < 60:
            depth = qd.get().getCvFrame()
            tried += 1
            v = depth[(depth > 0) & (depth < 3000)]
            if v.size >= 0.5 * depth.size:
                frames.append(depth.astype(np.float32))
        if len(frames) < N_FRAMES:
            print(f"  (仅收集到 {len(frames)} 帧健康帧, 用这些分析)")
        rgb = q_rgb.get().getCvFrame()
    stack = np.stack(frames)  # (N,H,W) mm

    print("\n========== 深度数据分析 ==========")
    d_med = np.median(stack, axis=0)
    valid = (d_med > 0) & (d_med < 3000)
    print(f"[1] 深度单位: uint16 mm(DepthAlign RGB)")
    print(f"[2] 有效像素率: {100 * valid.mean():.1f}%")
    v = d_med[valid]
    print(f"[3] 有效深度: min {v.min():.0f}mm  p50 {np.median(v):.0f}mm  p95 {np.percentile(v,95):.0f}mm  max {v.max():.0f}mm")

    # 帧间抖动(静态场景, 每像素 12 帧 std)
    std_map = stack.std(axis=0)
    std_valid = std_map[valid]
    print(f"[4] 时间抖动: median {np.median(std_valid):.2f}mm  p95 {np.percentile(std_valid,95):.2f}mm")

    # 中央桌面 ROI 平面拟合: 取图中央 50%x60% 且有效区域
    h, w = d_med.shape
    y0, y1 = int(h * 0.25), int(h * 0.80)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    roi = d_med[y0:y1, x0:x1]
    roi_valid = (roi > 0) & (roi < 2000)
    ys, xs = np.nonzero(roi_valid)
    if len(xs) > 1000:
        X = xs.astype(np.float64)
        Y = ys.astype(np.float64)
        Z = roi[roi_valid].astype(np.float64)
        A = np.column_stack([X, Y, np.ones_like(X)])
        coef, *_ = np.linalg.lstsq(A, Z, rcond=None)
        fitted = A @ coef
        resid = Z - fitted
        tilt_deg = np.degrees(np.arctan(np.hypot(coef[0], coef[1])))
        print(f"[5] 桌面平面拟合(ROI): 平面倾角 {tilt_deg:.1f}°, "
              f"残差 RMS {np.sqrt((resid**2).mean()):.2f}mm, "
              f"残差 p95 {np.percentile(np.abs(resid),95):.2f}mm")
        print(f"    平面距离: 中心点深度 ≈ {np.median(Z):.0f}mm")
        resid_map = np.full(roi.shape, np.nan, dtype=np.float32)
        resid_map[roi_valid] = np.abs(resid)
        vis = np.clip(resid_map / 50.0 * 255, 0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(f"{out_dir}/plane_residual.png", vis)
    else:
        print("[5] 桌面 ROI 有效像素不足, 无法平面拟合")

    # 深度直方图
    hist, _ = np.histogram(v, bins=40, range=(0, 3000))
    top = hist.argsort()[::-1][:6]
    print("[6] 深度直方图峰值(mm):", [int(np.percentile(v, (i + 1) * 100 / 40)) for i in top[:6]])

    np.save(f"{out_dir}/depth_median.npy", d_med)
    np.save(f"{out_dir}/depth_std.npy", std_map)
    cv2.imwrite(f"{out_dir}/rgb.png", rgb)
    print(f"\n分析图已保存: {out_dir}/ (plane_residual.png, depth_median.npy 等)")


if __name__ == "__main__":
    main()
