# 相机标定入口

当前无参数启动会进入比赛流水线的 OAK-D-PRO-FF EEPROM 页面并自动连接当前设备：

```bash
./tool/camera_calibration/run_ui.sh
```

以下 Astra Pro 固定相机流程仅作历史回归；需要时显式运行
`./tool/camera_calibration/run_ui.sh --legacy-astra`。

这套工具完成四件事：

1. 使用整张 ChArUco A4 板标定 Astra Pro 独立 UVC 彩色相机内参。
2. 使用 ID 100、101、102 的三个 AprilTag 求固定相机到尺子工作平面的外参。
3. 使用未参与拟合的 ID 103 测量独立位置误差。
4. 使用 Tag 优先、机器人位姿回退的统一运行时定位框架。

当前没有机械臂，因此这里得到的是
`camera_color_optical_frame <-> ruler_workspace`，不是最终的经典手眼标定。
机械臂到位后，必须让 `ruler_workspace` 与机械臂 `base` 坐标系严格重合，
或者额外测得二者之间的刚体变换。

混合定位页中的机器人数据目前来自模拟适配器，只用于验证接口和矩阵方向，不能作为
真实机械臂的执行坐标。配置里的单位手眼矩阵也不会被标记为有效标定结果。

## 图形界面

现在只需要打开应用菜单中的“Astra Pro 标定工作台”，或只运行下面这一个脚本：

```bash
cd /home/throne/workspaces/arm
./tool/camera_calibration/run_ui.sh
```

该脚本会自动连接 Astra Pro，并以 `1280 x 720 MJPEG / 30 FPS` 进入 RGB 内参页。
左侧“RGB 模式”也可切换为 `640 x 480 YUYV / 30 FPS`。界面将打印、RGB 内参、
工作平面外参和 ID 103 验证集中在同一个窗口；完成内参后只需点击左侧下一阶段，
不要再单独启动目录里的其他 `.py` 文件。默认使用 `V4L2` 直连彩色相机；此模式下
必须先关闭占用相机的 `usb_cam` 和 OrbbecViewer。
需要从现有 ROS 图像话题读取时，在左侧切换到 `ROS` 并连接
`/usb_cam/image_raw`。

目录中的命令行程序只是 UI 调用的底层模块，保留用于调试，日常标定不需要手动运行。

需要直接打开混合定位页时运行：

```bash
./tool/camera_calibration/run_ui.sh --stage localization --auto-connect
```

## 混合定位框架

运行时只向瓶子定位和控制模块输出一个统一结果：

```text
T_workspace_camera
```

每一帧按以下顺序选择来源：

1. 视野中存在足够的 ID 100～102，且重投影 RMS 合格时，使用多 Tag PnP 直接求
   `T_workspace_camera`。
2. Tag 不可用时，读取机器人适配器的 `T_base_gripper`，按照下面的矩阵链反推：

```text
T_workspace_camera =
T_workspace_base × T_base_gripper × T_gripper_camera
```

3. 机器人位姿缺失、时间戳过期或真实手眼矩阵尚未标定时，返回“无有效位姿”。程序
   不会把最后一次视觉结果冒充当前位姿。

配置文件是 `config/hybrid_localization.yaml`。当前
`hand_eye_calibrated: false`，`T_workspace_base` 和 `T_gripper_camera` 都是单位矩阵
占位。完成真实手眼标定后才能写入结果并把该标志改为 `true`。

核心实现在 `hybrid_localization.py`。将来接入机械臂时，只需实现
`RobotPoseProvider.latest_pose()`，返回带单调时钟时间戳的 `RobotPoseSample`：

```python
class RosRobotPoseProvider(RobotPoseProvider):
    def latest_pose(self, now_s=None):
        return RobotPoseSample(
            base_from_gripper=self.base_from_gripper,
            timestamp_s=self.timestamp_s,
            simulated=False,
        )
```

上层不依赖 ROS、机械臂品牌或姿态消息格式。真实适配器必须先把欧拉角、四元数或厂商
位姿统一转换成以米为单位的 `4 x 4` 矩阵，并明确机器人数据使用的是法兰还是 TCP。

混合定位页的“模拟 Tag 丢失”可检查视觉到机器人回退的切换；“启用模拟机器人回退”
关闭后，Tag 也不可用时应显示“无有效位姿”。模拟结果始终带有模拟标志，禁止下发。

## 打印

推荐直接打印：

- `targets/calibration_targets_A4_2pages.pdf`
  - 第 1 页：5 x 7 ChArUco 大格内参板（36.0 mm 方格、27.0 mm 标签）
  - 第 2 页：四个可裁剪 AprilTag

也提供两页的独立 PDF。打印设置必须选择“实际大小”或 `100%`，关闭
“适合页面”“缩放到可打印区域”等选项。打印后先测量页面上的 `100.0 mm`
核对线，再测量 Tag 黑色方框外边缘。若不是 `70.0 mm`，把实测值写入
`config/tag_layout.yaml` 的 `tag_size_mm`。

生成后的 PDF 已通过 600 DPI 反向渲染验证。需要复查或重新生成素材时运行：

```bash
python3 -m tool.camera_calibration.generate_targets
python3 -m tool.camera_calibration.verify_targets
```

验证程序会检查 PDF 是否为 ISO A4、ChArUco 方格是否为 `36.0 mm`、四个 Tag
黑框是否为 `70.0 mm`，以及 ID 100～103 是否都能被检测。

裁剪时沿虚线剪，保留完整白色静区。三个标定 Tag 必须处于同一平面，纸张压平，
且所有 `ORIGIN: TOP LEFT` 朝向一致。建议把三个左上角基准点尽量分散，形成较大的
直角三角形，不能共线。ID 103 放在三角形内部和外部分别测一次更有参考价值。

## 1. 启动彩色相机

在一个终端运行：

```bash
roscore
```

另一个终端运行：

```bash
rosrun usb_cam usb_cam_node \
  _video_device:=/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0 \
  _image_width:=1280 _image_height:=720 _framerate:=30 \
  _pixel_format:=mjpeg \
  _camera_name:=astra_pro_rgb \
  _camera_frame_id:=camera_color_optical_frame
```

以下命令均从仓库根目录以 Python 模块方式执行：

```bash
cd /home/throne/workspaces/arm
```

也可以不启动 ROS，给各程序添加 `--input v4l2`，让程序直接打开相机。ROS 与
V4L2 不能同时占用同一个彩色设备。

## 2. RGB 内参

```bash
python3 -m tool.camera_calibration.calibrate_intrinsics
```

- 将完整 ChArUco 板保持平整。
- 让标定板覆盖画面中心、四角和边缘，并改变距离和三个方向的倾角。
- 每个明显不同的姿态按一次空格，至少采集 20 组，建议 30～40 组。
- 按 `C` 求解并保存，按 `R` 清空，按 `Q` 退出。
- 推荐 RMS 重投影误差不超过约 `0.5 px`；程序默认在超过 `0.8 px` 时告警。

输出：

```text
output/astra_pro_rgb_1280x720.yaml
output/astra_pro_rgb_1280x720_report.yaml
```

后续可给 `usb_cam` 增加：

```bash
_camera_info_url:=file:///home/throne/workspaces/arm/tool/camera_calibration/output/astra_pro_rgb_1280x720.yaml
```

内参只适用于标定时选择的分辨率和取流模式。UI 会把 `1280x720` 和 `640x480`
的内参、外参及验证结果分别保存，禁止混用。

## 3. 修改实测坐标

编辑 `config/tag_layout.yaml`：

- 三个 `origin_mm` 是 Tag 黑色方框左上角在尺子坐标系下的 `[X, Y, Z]`。
- 统一约定为：`O=TL`，`+X=TL→TR`，`+Y=TL→BL`，`+Z=+X×+Y` 指向纸面内部。
- `yaw_deg=0` 表示打印件上边沿 `+X`，左边沿 `+Y`；正 Y 是页面向下。
- 示例坐标只是占位，运行前必须替换。
- ID 103 的坐标仅用于验证，不会参与外参求解。

手工测量必须量黑框左上角之间的距离，不要从裁剪纸边测量。纸边不是视觉特征基准。
外参页可以输入 `100-101`、`100-102`、`101-102` 三条左上角基准点距离并自动计算坐标，
不要求徒手把三个 Tag 摆成严格直角。若直接填写 `(0,0)`、`(97,0)`、`(0,97)`，
则第三条基准点距离必须为 `sqrt(97^2 + 97^2) = 137.18 mm`。

如果不裁开 ID100～103，而是在原始 A4 第 2 页上遮住 ID103，应点击“使用原始
A4 排布”。源 PDF 的黑框左上角布局为：100→101 沿 `+X` 相距 `100 mm`，100→102
沿 `+Y` 相距 `110 mm`。程序会根据实测黑框边长相对设计值 `70 mm` 的比例同步
缩放基准点距离；例如实测边长 `69 mm` 时得到 `98.57 mm` 和 `108.43 mm`。

识别画面会在每个 Tag 的黑框左上角标出 `O`：红色 `+X` 沿上边，绿色 `+Y` 沿
左边，蓝色 `+Z` 指向纸内。每个 Tag 同时显示 ID 和左上角工作坐标，ID 100 额外标为
`WORKSPACE ORIGIN`。若打印件发生旋转，应修改对应 Yaw，不能只靠调整原点坐标补偿。

## 4. 工作平面外参

保证 ID 100～102 同时出现在画面中，然后运行：

```bash
python3 -m tool.camera_calibration.calibrate_workspace
```

确认三个 Tag 都被识别后按空格，程序自动采集 60 帧并取角点中值。输出：

```text
output/workspace_extrinsics_1280x720.yaml
output/workspace_extrinsics_1280x720.png
```

默认 RMS 上限是 `2.0 px`。超限时不要强行使用结果，应重新检查打印比例、Tag 左上角
坐标、边方向、纸面平整度和内参。缺少新坐标约定字段的旧中心基准外参会被保留，
但 UI 不会将其标为完成，也禁止用它执行 ID 103 验证。

## 5. ID 103 独立验证

把 ID 100～102 固定在桌面作为参考地图，把 ID 103 的黑框左上角放到配置文件记录的
已知位置。验证时必须让四个 Tag 同时可见，然后运行：

```bash
python3 -m tool.camera_calibration.validate_workspace
```

程序不会复用标定时的固定相机姿态。每一帧都只用 ID 100～102 的 12 个角点重新计算
当前 `camera_from_workspace`，再独立计算 ID 103 左上角坐标。因此四个 Tag 固定时可以
移动相机，ID 103 不参与相机位姿求解。参考 Tag 不全或参考重投影 RMS 超过 `2.0 px`
时，该帧不会进入验证统计。

程序采集 100 个有效帧并输出中值位置、抖动标准差、XY 平面误差和参考位姿 RMS：

```text
output/validation_tag_103_1280x720.yaml
```

建议至少更换 5 个未参与标定的位置重复验证。三角形内部的误差代表插值性能，
三角形外部的误差代表外推性能，通常后者更差。即使所有 Tag 静止，数值仍可能因角点
亚像素噪声、镜头畸变残差和手工尺量误差产生小幅变化；不应再随相机移动系统性漂移。

### RViz 检查相机位姿

在 UI 的“移动相机位置验证”页点击“打开 RViz 位姿”。程序会在需要时自动启动
`roscore`，并发布：

```text
TF:     ruler_workspace -> camera_color_optical_frame
Pose:   /camera_calibration/camera_pose
Path:   /camera_calibration/camera_path
Marker: /camera_calibration/markers
```

RViz 固定坐标系为 `ruler_workspace_rviz`，绿色平面是固定参考 ID 100～102，橙色半透明
平面是配置的 `ID103 EXPECTED`，洋红色平面是每帧计算的 `ID103 MEASURED`。红线连接
ID103 理想左上角与实测左上角；洋红色平面的中心、边长和 Yaw 均由四个实测角点计算，
文字同时显示实测坐标、实时 Yaw 和 XY 误差。蓝色视锥和坐标轴表示
相机光学坐标系。只有参考 ID 100～102 全部可见且参考 RMS 合格时，TF 和轨迹才更新；
参考或 ID103 丢失时实测标记会隐藏，不保留上一帧冒充实时值。右侧清空图标可以清除轨迹。

当前约定 `+Z=+X×+Y` 指向纸面内部，因此相机位于纸面上方时，其工作坐标 Z 通常为
负数。程序额外发布 `ruler_workspace_rviz -> ruler_workspace` 的绕 X 轴 180° 静态变换，
只用于把 RViz 转为常见的 Z-up 视图；因此相机会显示在桌面上方。标定数据、ID103 坐标
和机器人计算仍使用原始 `ruler_workspace`，没有偷偷翻转单独一根轴。
`camera_color_optical_frame` 使用光学轴：X 向图像右侧、Y 向图像下方、Z 朝镜头观察方向。

## 误差边界

- 这套方法使用每个 Tag 的四个角点求外参，但所有对外坐标均以左上角角点为基准。
- 手工尺量、纸张翘曲和 Tag 朝向误差会直接进入外参结果。
- 第四个 Tag 只能验证 RGB 平面定位，不能验证 Astra Pro 深度或 RGB-D 外参。
- 验证时 Z 是配置中给定的平面高度，不是由单目射线独立测得的深度精度。
- 将来机械臂的工具中心点还需要独立的 `base -> tool` 和工具偏置定义。

## RGB 与红外/深度像素对齐

Astra Pro 的 RGB 是独立 UVC 模组，RGB 可用 `1280x720`，红外和深度通常为
`640x480`。尺寸相同也不代表像素对齐，不能直接把 RGB 像素乘以 `0.5`，原因是两颗
相机的内参、畸变、视场角和光心位置均不同。

完整 RGB-D 对齐需要：

1. 单独标定 RGB 内参和 IR 内参。
2. 用两路都能看到的 ChArUco 姿态求 `IR -> RGB` 刚体外参。
3. 对每个深度像素先用 IR/深度内参反投影为三维点，再经 `IR -> RGB` 变换投影到
   RGB 图像。输出可以是 `1280x720` 的 aligned depth，空洞和遮挡属于正常现象。

当前 UI 完成的是 RGB 内参与 RGB 到工作平面的外参，不包含 RGB-IR 双目标定。
如果目标始终位于同一个工作平面，可用该平面的单应性做近似映射；目标离开平面后
必须使用深度和完整 RGB-IR 外参。
