# 机械臂视觉定位通用框架

该 ROS1 Noetic 包用于快速组合机械臂控制、YOLO 实例分割、FoundationPose 6D
位姿估计、AprilTag 绝对定位和眼在手上机器人位姿回退。当前阶段只建立稳定接口、
坐标链、参数治理和可运行的 Mock 链路，不代表真实机械臂抓取已经完成。

## 数据流

```text
ROS RGB / aligned depth / CameraInfo
                   |
       +-----------+-----------+
       |                       |
  AprilTag PnP             YOLO segmentation
       |                       |
       |                FoundationPose 6D
       |                       |
       +---- T_workspace_camera+
                               |
                  T_workspace_object
                               |
                  robot adapter / safety gate
```

Tag 可见时直接输出视觉绝对定位；Tag 不可见时才尝试：

```text
T_workspace_camera =
T_workspace_base * T_base_gripper * T_gripper_camera
```

瓶子最终位姿为：

```text
T_workspace_object = T_workspace_camera * T_camera_object
```

所有平移在程序内部使用米，角度只在配置输入和显示时使用度，运行时姿态统一为
`4 x 4` 齐次矩阵。

## 目录

```text
arm_vision_framework/
├── config/
│   ├── calibration_parameters.yaml  # 唯一标定数据入口
│   └── system_parameters.yaml       # 后端、话题和安全配置
├── launch/competition_pipeline.launch
├── scripts/
│   ├── pipeline_node.py             # ROS 主节点
│   └── run_pipeline.py              # 配置检查和无硬件 smoke
├── tools/calibration_tool.py        # 单文件标定数据工具
├── src/arm_vision_framework/
│   ├── adapters/                    # 算法和机械臂适配器
│   ├── localization.py              # Tag 优先/机器人回退
│   └── pipeline.py                  # 与 ROS 解耦的主流水线
├── tests/                           # 框架回归测试
└── validation/                      # 后续准确性验证小脚本
```

## 当前安全状态

| 项目 | 当前状态 |
|---|---|
| Astra Pro RGB 内参 | 有效，`1280 x 720 MJPG` |
| Tag 100～102 地图 | 有效，用于算法验证 |
| 眼在手上 `T_gripper_camera` | 无效，占位单位矩阵 |
| `T_workspace_base` | 无效，占位单位矩阵 |
| RGB-D 对齐 | 未标定，FoundationPose 实机输入尚不可用 |
| YOLO / PyTorch | 当前 Python 环境未安装 |
| FoundationPose 运行时 | 尚未接入 |
| 机械臂运动 | 强制关闭 |

旧的固定 Astra 外参保存在 `fixed_camera_validation_reference`，只用于追溯之前的桌面
验证，眼在手上运行时禁止使用。

## 构建与 Smoke

```bash
cd /home/throne/workspaces/arm/ros_ws
catkin_make
source devel/setup.bash

rosrun arm_vision_framework run_pipeline.py --check --smoke
```

默认后端全部是 Mock，smoke 输出必须同时满足：

```text
valid: true
simulated: true
localization_source: simulated_robot
```

该结果仅验证软件坐标链，不是算法准确率或抓取结果。

启动 ROS 节点：

```bash
roslaunch arm_vision_framework competition_pipeline.launch
```

## Canonical ROS 接口

所有厂商驱动都应桥接到下列稳定接口，上层不直接导入厂商 SDK：

| 方向 | 默认话题/服务 | 类型 |
|---|---|---|
| 输入 | `/camera/color/image_raw` | `sensor_msgs/Image` |
| 输入 | `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` |
| 输入 | `/camera/color/camera_info` | `sensor_msgs/CameraInfo` |
| 输入 | `/arm_vision/robot/state/tool_pose` | `geometry_msgs/PoseStamped` |
| 输出 | `/arm_vision/object_pose` | `geometry_msgs/PoseStamped` |
| 输出 | `/arm_vision/segmentation/mask` | `sensor_msgs/Image` |
| 输出 | `/arm_vision/status` | `std_msgs/String`，JSON |
| 预留输出 | `/arm_vision/robot/command/target_pose` | `geometry_msgs/PoseStamped` |
| 预留服务 | `/arm_vision/robot/stop` | `std_srvs/Trigger` |

框架不会自动把目标位姿发送给机器人。只有 `dry_run: false`、
`allow_robot_motion: true`、真实机械臂适配器和安全检查同时到位后，控制模块才允许实现
运动命令。

## 算法后端替换

YOLO 权重准备好后修改：

```yaml
segmentation:
  backend: yolo
  weights: /absolute/path/to/best.pt
  target_classes: [can]
```

FoundationPose 接入后修改：

```yaml
pose_estimation:
  backend: foundationpose_plus_plus
  mesh_path: /absolute/path/to/can_mesh.obj
  mesh_scale_to_meters: 0.001  # BOP 毫米模型
```

`FoundationPoseEstimator` 要求 RGB、Mask 和深度完全同尺寸且深度已经对齐到 RGB。
适配的运行时对象只需实现 `register_frame(**kwargs)`、`track_frame(**kwargs)` 和可选的
`reset()`。这样 FoundationPose 原版和 FoundationPose++ 可以共用上层接口。

比赛若改用 OAK，相机驱动只需发布上述 RGB-D/CameraInfo 话题，并使用标定工具替换中央
参数；Astra Pro 继续作为算法验证相机，不在业务代码中写死。

## 标定工具

查看和校验中央参数：

```bash
rosrun arm_vision_framework calibration_tool.py show
rosrun arm_vision_framework calibration_tool.py validate
```

打开现有标定 UI，然后同步最新 Astra RGB 标定结果：

```bash
rosrun arm_vision_framework calibration_tool.py camera-ui
rosrun arm_vision_framework calibration_tool.py sync-camera
```

未来完成眼在手上标定后，输入 YAML 必须明确包含 `gripper_from_camera.matrix`：

```bash
rosrun arm_vision_framework calibration_tool.py \
  import-hand-eye --input /absolute/path/to/eye_in_hand.yaml
```

每次替换前都会在 `config/backups/` 自动保存旧参数，并在原子替换后重新检查矩阵。
工具不会猜测 `camera_from_gripper` 的方向，方向不明确时会拒绝导入。

## 机械臂适配边界

驱动索引见 [docs/readme.txt](../../../docs/readme.txt)。现有资料表明：

- 埃夫特 ER 的笛卡尔点位是 `X/Y/Z/A/B/C`，`A/B/C` 对应绕 `Z/Y/X` 的欧拉角。
- 纳博特手册描述的远程模式主要是数字 IO 和 Modbus 启停已示教程序。
- 两份操作手册都不足以可靠实现 Python 笛卡尔在线控制，不能自行猜测端口或报文。

机械臂随机分配后，应取得对应官方 SDK/通信协议，再实现 `RobotController` 或一个桥接到
canonical topic 的独立节点。无论厂商如何变化，YOLO、FoundationPose、标定参数和
准确性验证脚本都不应修改。
