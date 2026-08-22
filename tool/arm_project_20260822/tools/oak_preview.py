#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OAK-D Pro 实时预览: 两个窗口(RGB + 深度伪彩), 按 q 退出。

用法(在 WSL 终端, 需要有图形显示):
  conda activate foundationpose
  python ~/tools/oak_preview.py

无显示环境/自动退出:
  python ~/tools/oak_preview.py --seconds 3 --save-dir /tmp/oak_preview
"""
import argparse
import time

import cv2
import numpy as np
import depthai as dai


def build_pipeline(rgb_fps=5, mono_fps=5):
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(rgb_fps)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setInterleaved(False)

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_l.setFps(mono_fps)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_r.setFps(mono_fps)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    return pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=0, help="自动退出秒数(0=手动 q 退出)")
    parser.add_argument("--save-dir", default="", help="每帧另存 PNG 到此目录")
    parser.add_argument("--rgb-fps", type=float, default=5, help="RGB 帧率(USB2 环境建议 5)")
    parser.add_argument("--mono-fps", type=float, default=5, help="双目/深度帧率(USB2 环境建议 5)")
    args = parser.parse_args()

    with dai.Device(build_pipeline(args.rgb_fps, args.mono_fps)) as device:
        print("设备已启动:", [c.name for c in device.getConnectedCameras()])
        device.setIrLaserDotProjectorBrightness(800)  # 开点阵投影(白色表面匹配关键)
        device.setIrFloodLightBrightness(0)
        q_rgb = device.getOutputQueue("rgb", 4, False)
        q_depth = device.getOutputQueue("depth", 4, False)
        t0 = time.time()
        frames = 0
        while True:
            rgb = q_rgb.get().getCvFrame()
            depth = q_depth.get().getCvFrame()
            frames += 1

            # 深度可视化: 无效(0 或 >3m)置灰, 有效值按 0-2m 固定色标
            valid = (depth > 0) & (depth < 3000)
            disp = np.zeros(depth.shape, dtype=np.float32)
            disp[valid] = np.clip(depth[valid].astype(np.float32) / 2000.0, 0.0, 1.0)
            disp_color = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)
            disp_color[~valid] = (128, 128, 128)  # 无效 = 灰
            valid_pct = 100.0 * valid.mean()

            small_rgb = cv2.resize(rgb, (960, 540))
            small_disp = cv2.resize(disp_color, (960, 540))
            fps = frames / max(1e-6, (time.time() - t0))
            cv2.putText(small_rgb, f"OAK-D-PRO-FF | {fps:.1f} fps | q = quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(small_disp, f"effective {valid_pct:.0f}% | 0-2m | gray=invalid",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("OAK RGB", small_rgb)
            cv2.imshow("OAK Depth", small_disp)

            if args.save_dir:
                import os
                os.makedirs(args.save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_dir, "preview_rgb.png"), rgb)
                cv2.imwrite(os.path.join(args.save_dir, "preview_depth.png"), depth)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.seconds and (time.time() - t0) >= args.seconds:
                break
        cv2.destroyAllWindows()
        print("预览结束, 总帧数:", frames)


if __name__ == "__main__":
    main()
