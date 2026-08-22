# 机械臂视觉定位通用框架

该 ROS1 Noetic 包用于快速组合机械臂控制、YOLO 实例分割、FoundationPose 6D
位姿估计、AprilTag 手眼标定和基于控制器 TCP 的眼在手上定位。当前阶段只建立稳定接口、
坐标链、参数治理和可运行的 Mock 链路，不代表真实机械臂抓取已经完成。

## 数据流

```text
OAK-D Pro 停稳后 RGB-D 快照 / CameraInfo
                   |
       +-----------+-----------+
       |                       |
TCP + 手眼外参          YOLO segmentation
       |                       |
       |                FoundationPose 6D
       |                       |
       +---- T_base_camera -----+
                               |
                      T_base_object
                               |
                  robot adapter / safety gate
```

比赛运行时不使用 AprilTag 连续定位。控制器回读的 Tool1 TCP 与已标定的
`T_tcp_color_camera` 直接组成相机位姿：

```text
T_workspace_camera =
T_workspace_base * T_base_gripper * T_gripper_camera
```

默认 `workspace == base`，即简化为 `T_base_tcp * T_tcp_color_camera`。AprilTag 只在手眼
采样、标定验收和故障诊断中使用；正式运行保持 `use_apriltag_runtime: false`。

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
│   ├── pipeline_node.py             # ROS 视觉主节点
│   ├── oak_depthai_node.py          # OAK-D Pro 按需同步 RGB-D 快照
│   └── run_pipeline.py              # 配置检查和无硬件 smoke
├── tools/calibration_tool.py        # 单文件标定数据工具
├── src/arm_vision_framework/
│   ├── adapters/                    # 算法和机械臂适配器
│   ├── localization.py              # TCP + 手眼运行时定位；Tag 仅标定
│   └── pipeline.py                  # 与 ROS 解耦的主流水线
├── tests/                           # 框架回归测试
└── validation/                      # 后续准确性验证小脚本
```

## 当前安全状态

| 项目 | 当前状态 |
|---|---|
| OAK-D-PRO-FF RGB 内参 | 已从 MXID `14442C10D141C5D600` EEPROM 导入，`1920 x 1080` |
| Tag 100～102 地图 | 有效，用于算法验证 |
| 眼在手上 `T_gripper_camera` | 无效，占位单位矩阵 |
| `T_workspace_base` | 无效，占位单位矩阵 |
| RGB-D 对齐 | DepthAI 硬件对齐到 RGB，实机同步抓帧已通过 |
| YOLO / PyTorch | 已安装并保留真实模型入口 |
| FoundationPose 运行时 | 已接入 CUDA 运行时；真实目标仍需匹配网格和 Mask |
| 机械臂运动 | 强制关闭 |

旧的固定 Astra 外参保存在 `fixed_camera_validation_reference`，只用于追溯之前的桌面
验证，眼在手上运行时禁止使用。

## OAK-D Pro 比赛采集档位

根据产品手册（IMX378 RGB 12MP，OV9282 双目原生 `1280×800`、基线 75 mm）及“机械臂停稳
后拍两张再抓取”的流程，正式档位固定为：

| 项目 | 值 | 原因 |
|---|---:|---|
| RGB / 对齐 Depth | `1920×1080` | 分割细节充足，RGB、Mask、Depth、内参同尺寸 |
| 管线帧率 | `10 FPS` | 只为缓冲同步快照，不把计算资源用于实时视频 |
| 双目 | `800p` | 使用 OV9282 全局快门的原生档位 |
| Extended disparity | 开启 | 手册中 800P 标准近距约 70 cm；扩展视差目标为约 35 cm，必须现场验收 |
| Subpixel | 关闭 | DepthAI 的扩展视差与 Subpixel 不能同时开启 |

OAK-D Pro 可能是 AF 或 FF RGB 版本。`focus_mode` 默认 `device_default`，到场确认镜头版本后，
AF 可选 `continuous_auto` 或已测得的 `manual`（0–255）；FF 保持默认，不能假设软件调焦存在。

`oak_depthai_node.py` 只在 `/arm_vision/camera/capture` 服务被调用时发布一组同步 RGB-D。
先 MoveJ 到观察位、等待停稳、再调用服务；抓取规划器不会接收运动过程中的图像。

节点同时可发布 OAK 原始 IMU 到 `/camera/imu`。ROS RGB 光学帧为 `+X` 向右、`+Y` 向下、
`+Z` 向前；机械臂 TCP 的 `+Z` 是工具伸出方向。配置中的 `camera_from_imu` 默认 `null`，
因此 IMU 数据不会被旋转进相机/TCP 坐标，也不能参与绝对定位或替代 TCP 回读。到场确认
具体 IMU 型号、EEPROM 外参和物理安装后，才可填写该 4×4 外参并发布对应 TF。

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
当前 `camera_profile_ready` 为 `true`；`field_runtime_ready` 仍为 `false`，因为更换相机后
必须重新标定真实 `T_gripper_camera`。不能把相机就绪误解成机械臂抓取链已经验收。

启动 ROS 节点：

```bash
roslaunch arm_vision_framework competition_pipeline.launch start_oak:=true
```

填完官方寄存器表并把 `controller.enabled` 设为 `true` 后，可额外启动只读状态桥：

```bash
roslaunch arm_vision_framework competition_pipeline.launch \
  start_oak:=true start_controller_state:=true
rostopic echo /arm_vision/controller/state
```

该节点没有 MOVJ/MOVL/IO 写接口，断线或解码失败会发布 `connected: false` 和错误文本。
若参数是在比赛 UI 第 09 页完成的，先一键导入，避免在两份 YAML 中重复手抄：

```bash
rosrun arm_vision_framework calibration_tool.py import-competition-controller \
  --input /home/throne/workspaces/arm/competition_pipeline/config/competition.yaml
```

## Canonical ROS 接口

所有厂商驱动都应桥接到下列稳定接口，上层不直接导入厂商 SDK：

| 方向 | 默认话题/服务 | 类型 |
|---|---|---|
| 输入 | `/camera/color/image_raw` | `sensor_msgs/Image` |
| 输入 | `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` |
| 输入 | `/camera/color/camera_info` | `sensor_msgs/CameraInfo` |
| 输入/诊断 | `/camera/imu` | `sensor_msgs/Imu`，原始 `oak_imu` 帧 |
| 输入 | `/arm_vision/robot/state/tool_pose` | `geometry_msgs/PoseStamped` |
| 输入/状态 | `/arm_vision/controller/state` | `std_msgs/String`，`arm_vision.control.v1` JSON |
| 输出 | `/arm_vision/object_pose` | `geometry_msgs/PoseStamped` |
| 输出 | `/arm_vision/segmentation/mask` | `sensor_msgs/Image` |
| 输出 | `/arm_vision/status` | `std_msgs/String`，JSON |
| 输出 | `/arm_vision/task_command` | `std_msgs/String`，`arm_vision.control.v1` JSON |
| 预留输出 | `/arm_vision/robot/command/target_pose` | `geometry_msgs/PoseStamped` |
| 预留服务 | `/arm_vision/robot/stop` | `std_srvs/Trigger` |
| OAK 按需采集 | `/arm_vision/camera/capture` | `std_srvs/Trigger` |

框架不会自动把目标位姿发送给机器人。只有 `dry_run: false`、
`allow_robot_motion: true`、真实机械臂适配器和安全检查同时到位后，控制模块才允许实现
运动命令。

`/arm_vision/task_command` 固定包含 command ID、时间戳、frame、XYZ `mm`、姿态 `deg`、
MOVJ/MOVL 类型、目标点/点列、Tool ID、User ID、shape 和 `safe_to_execute`。视觉节点发布的
物体目标始终是 `safe_to_execute: false`；MoveIt/控制器桥必须完成碰撞、IK、形态和安全门
后才能执行，不能把这个 JSON 直接当作厂商网络报文。

执行异常时 `SafeRecoveryManager` 先调用 STOP，并仅对包含 `singular/奇异/IK/
configuration` 等报警的失败考虑安全点回退。默认 `auto_recover: false`；没有预先确认的
MOVJ 安全点、急停异常、TCP 状态失效或控制器断线时保持锁定。现场可在
`competition_pipeline` 第 09 页保存停稳状态下的完整 Tool/User/shape/保留字段构型，然后
在验证无碰撞路径后才决定是否开启自动回退。

## 算法后端替换

YOLO 权重准备好后修改：

```yaml
segmentation:
  backend: yolo
  weights: /absolute/path/to/best.pt
  target_classes: [can]
```

FoundationPose 接入后修改（当前 ROS 节点已经提供运行时封装）：

```yaml
pose_estimation:
  backend: foundationpose_plus_plus
  foundationpose_root: /home/throne/workspaces/arm_data/third_party/FoundationPose-plus-plus
  mesh_path: /absolute/path/to/can_mesh.obj
  mesh_scale_to_meters: 1.0    # 当前建模工具导出的是米；BOP 毫米模型填 0.001
  device: cuda:0
  est_refine_iter: 5
  track_refine_iter: 2
  use_mask_center_guidance: true
```

`pipeline_node.py` 在选择该后端时会自动从 `foundationpose_root` 加载 CUDA 运行时、
refiner/scorer 权重和 nvdiffrast 上下文。首帧调用 FoundationPose `register`，后续帧调用
`track_one`；后续帧默认用 YOLO Mask 的包围盒中心修正上一帧的 XY 初值，这对应
FoundationPose++ 的 2D tracker 接口。这样不强制启用 Cutie/SAM/Qwen 组件，仍可用现有
分割模型验证核心 6D 算法；需要关闭该校正时把 `use_mask_center_guidance` 设为 `false`。

`FoundationPoseEstimator` 要求 RGB、Mask 和深度完全同尺寸且深度已经对齐到 RGB。运行时
会把 ROS 的 BGR 转换为算法需要的 RGB、将无效深度置零，并严格检查相机内参和 4x4 输出；
不会把不同分辨率的深度图直接缩放。

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

比赛流水线输出的 `hand_eye.tcp_from_color_camera` 也可以一键导入。这里的生产 TCP
与 ROS 的 `gripper` 帧是同一个末端工具坐标，程序不会求逆或改变方向：

```bash
rosrun arm_vision_framework calibration_tool.py \
  import-competition-hand-eye \
  --input /home/throne/workspaces/arm/competition_pipeline/config/competition.yaml
```

导入前会检查 `valid: true`、4x4 齐次矩阵和旋转正交性；替换前自动备份
`calibration_parameters.yaml`。

### OAK-D Pro 官方 EEPROM/Flash 导入

[`docs/OAK-D-Pro产品手册.pdf`](../../../docs/OAK-D-Pro产品手册.pdf) 是产品规格手册，明确
设备支持 DepthAI，但没有定义 EEPROM JSON 字段或 Flash 命令。当前采用官方 DepthAI
`CalibrationHandler` 读取 Luxonis 官方工具导出的 EEPROM JSON，因此无实物时可以先完成
离线解析和 ROS 参数生成：

```bash
rosrun arm_vision_framework calibration_tool.py \
  import-oak-eeprom \
  --input /absolute/path/to/oak_calibration.json \
  --color-width 1920 --color-height 1080 \
  --depth-width 1280 --depth-height 800
```

该命令会把 RGB CAM_A、原生深度 CAM_C 的内参/畸变、设备元数据和官方 JSON 备份写入
`calibration_parameters.yaml`；发布的对齐 Depth 使用 RGB 分辨率和 RGB 像素几何，原生 CAM_C
内参作为 `native_cam_c` 保存，避免把 800P 双目内参误套到 RGB 像素。它**不会**伪造
Flash 成功，也不会在没有设备时写硬件 EEPROM。到场后先用官方 Luxonis/DepthAI 标定工具
完成真正的设备 EEPROM 写入，再把设备导出的 JSON 通过同一命令导入；相机内参变化后必须
重新做眼在手上标定。

相机已连接时可以直接读取 EEPROM 并同步中央参数，不需要先手工导出 JSON：

```bash
rosrun arm_vision_framework calibration_tool.py import-oak-device
```

检测到多台设备时需增加 `--mxid <设备序列号>`。该命令只读设备 EEPROM，不会写入或
Flash 相机；原始 EEPROM 会保存为 `config/oak_factory_calibration.json`，旧中央参数自动备份。

## 机械臂适配边界

### 两视角抓取与放置执行

正式包的 `motion_execution.py` 已固化下列顺序：

1. MoveIt/现场规划器输出无碰撞关节轨迹，桥接成带 `shape/tool/user` 的 MOVJ 点列；
2. MoveJ 到第一观察位、停稳、请求 OAK 快照；MoveJ 到第二观察位、停稳、再请求一次；
3. 以每张图片时刻的 TCP 回读和 `T_tcp_color_camera` 求相机位姿，分割/深度得到物体基座坐标；
4. MoveJ 到预抓取与预放置位，近距离接近、抬升、放置使用低速 MOVL；
5. 夹爪由 `RemoteIoGripper` 写明确配置的命名 IO；配置 `done_input` 时必须等完成反馈；
6. 任意失败、取消、拒绝或超时都会 stop 机械臂和夹爪；默认仍是 `dry_run` 且禁止运动。

`ObservationPlan`、`PickPlacePlan`、`TwoViewPickPlaceCoordinator` 是稳定的正式接口。厂商
MOVJ/MOVL 报文现已有官方公开资料（见 `docs/纳博特通讯协议.md`），
`RobotController.move_j/move_l` 桥接已按 22.07 协议实现为
`adapters/nexbot_tcp.py`；不要改动标定、视觉、规划与状态机。

驱动索引见 [docs/readme.txt](../../../docs/readme.txt)。现有资料表明：

- 埃夫特 ER 的笛卡尔点位是 `X/Y/Z/A/B/C`，`A/B/C` 对应绕 `Z/Y/X` 的欧拉角。
- 纳博特手册描述的远程模式主要是数字 IO 和 Modbus 启停已示教程序。
- 两份操作手册都不足以可靠实现 Python 笛卡尔在线控制，不能自行猜测端口或报文；
  官方 JSON-over-TCP 协议（RTL-22.07，6000/7000 端口）见 `docs/纳博特通讯协议.md`。

机械臂随机分配后，应取得对应官方 SDK/通信协议，再实现 `RobotController` 或一个桥接到
canonical topic 的独立节点。当前正式 ROS 包已加入
`adapters/inexbot_modbus.py`：它实现标准 Modbus-TCP 的传输、MBAP 校验、常用 IO 功能码、
配置化 IO 和手册中的点位/形态数据模型；`adapters/modbus_global_point.py` 额外提供了
依赖现场已验证本地程序的 GP-MOVJ/MOVL 保底适配器，但默认关闭，不会连接或运动。
`adapters/nexbot_tcp.py` 实现官方 22.07 JSON-over-TCP 协议（帧 `0x4E66`+长度+命令字+
JSON+CRC32；6000 端口 MOVJ `0x4501`/MOVL `0x4502`/急停 `0x2314`，7000 端口状态查询
`0x9512`），通过 `robot.adapter: nexbot_tcp` + `controller.nexbot_tcp.host` 启用；
剩余现场确认项（固件版本、pos 数组长度、构型行为）见 `docs/纳博特通讯协议.md` §11。
`competition_pipeline/controller_tcp.py` 与 `competition_pipeline/nexbot_tcp.py` 仅是
离线测试加载器。无论厂商如何变化，YOLO、FoundationPose、标定参数和准确性验证脚本
都不应修改。
