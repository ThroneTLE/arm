环境已安装完成，并通过 Astra Pro 实机验证。

**安装内容**

- ROS1 驱动：`~/orbbec_ws/src/OrbbecSDK_ROS1`
  - Wrapper `1.5.19`
  - 内置 SDK `1.10.35`
- C/C++ SDK：`~/orbbec_sdk/OrbbecSDK`
- Python SDK：`~/orbbec_sdk/pyorbbecsdk`
- OrbbecViewer `1.10.27`：`~/Applications/OrbbecViewer-1.10.27`
- udev 规则：[99-obsensor-libusb.rules](/etc/udev/rules.d/99-obsensor-libusb.rules)
- 终端环境：[.bashrc](/home/throne/.bashrc)
- Viewer 桌面启动项：[orbbec-viewer.desktop](/home/throne/.local/share/applications/orbbec-viewer.desktop)

**实机结果**

- 设备：Astra Pro，固件 `RD108C-009`，USB 2.0
- 深度：`640×480 / 16UC1 / 约 29.7 Hz`
- 红外：`640×480 / mono16 / 约 29.6 Hz`
- 彩色：`640×480 / YUYV / 约 30.0 Hz`
- 彩色高清模式：`1280×720 / MJPEG / 约 30.0 Hz`（标定 UI 当前默认）
- Python SDK：成功识别 `1` 台设备，`pip check` 无依赖冲突
- Viewer：正常启动并识别相机
- 深度与彩色、红外与彩色均完成并发验证

Astra Pro 的彩色摄像头是独立 UVC 模组，不能使用当前 `astra.launch` 的 SDK 彩色通道，需要通过 `usb_cam` 发布。

深度启动：

```bash
roslaunch orbbec_camera astra.launch \
  enable_color:=false \
  enable_colored_point_cloud:=false \
  depth_fps:=30
```

另开终端启动彩色：

```bash
rosrun usb_cam usb_cam_node \
  _video_device:=/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0 \
  _image_width:=640 \
  _image_height:=480 \
  _framerate:=30 \
  _pixel_format:=yuyv \
  _color_format:=yuv422p \
  _camera_frame_id:=camera_color_optical_frame
```

红外模式：

```bash
roslaunch orbbec_camera astra.launch \
  enable_color:=false \
  enable_depth:=false \
  enable_ir:=true \
  enable_point_cloud:=false \
  enable_colored_point_cloud:=false \
  ir_fps:=30
```

Viewer 可从应用菜单打开“Orbbec Viewer”，或运行：

```bash
orbbec-viewer
```

Python 环境：

```bash
orbbec_python_env
python -c 'from pyorbbecsdk import Context; print(Context().query_devices().get_count())'
```

目前未配置 RGB 与深度的外参标定，因此原始图像采集可用，但做 RGB-D 对齐或点云融合前还需要单独标定。本次结果属于环境与硬件 smoke 验证，不是论文正式实验。测试进程均已关闭，当前项目仓库原有内容未被修改。

## 固定相机工作平面标定

标定程序、A4 打印素材和完整操作说明位于：

- `camera_calibration/README.md`
- `camera_calibration/targets/calibration_targets_A4_2pages.pdf`

图形界面可从应用菜单中的“Astra Pro 标定工作台”启动，或运行：

```bash
cd /home/throne/workspaces/arm/camera_calibration
./run_ui.sh
```

当前无机械臂阶段得到的是 `camera_color_optical_frame` 到尺子定义的
`ruler_workspace` 坐标变换。它用于先验证视觉平面坐标精度，不等同于最终机械臂
`base` 坐标系下的眼在外手眼标定。

RGB 是独立 UVC 模组，不能通过缩放像素直接与 `640×480` 红外/深度图对齐。
RGB-D 融合前还需分别标定 RGB/IR 内参和 `IR -> RGB` 外参，再通过深度反投影完成
三维重投影。当前标定 UI 尚不包含 RGB-IR 双目标定。
