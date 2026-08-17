# 机械臂视觉项目现状

更新时间：2026-08-15

## 1. 项目目标

本项目面向随机分配机械臂的视觉抓取任务，计划包含：

1. 相机内参、工具坐标和眼在手上手眼标定。
2. 机械臂状态读取、基础控制和安全停止。
3. YOLO 实例分割识别特定易拉罐。
4. FoundationPose / FoundationPose++ 六自由度位姿估计。
5. AprilTag 绝对定位与机械臂位姿回退相结合的定位链路。

当前处于“无机械臂实物的算法与接口验证阶段”。Astra Pro 只用于验证 RGB 标定、
AprilTag 定位和软件坐标链；它不是比赛最终相机。机械臂控制、真实眼在手上标定和
易拉罐抓取尚未完成。

## 2. 当前完成情况

| 模块 | 状态 | 说明 |
|---|---|---|
| Astra Pro 环境 | 已完成 | ROS1、Orbbec SDK、Python SDK、Viewer 和 UVC 彩色取流已验证 |
| RGB 内参标定 | 已完成 | 当前运行档位为 `1280 x 720 MJPG`，RMS 为 `0.5571 px`；SDK 的 `1280 x 800` 需要单独标定 |
| AprilTag 打印素材 | 已完成 | 提供经过尺寸校验的 A4 PDF，打印必须选择实际大小或 100% |
| 工作平面外参 | 已实现 | ID100～102 多 Tag PnP，当前新坐标约定下 RMS 为 `0.3148 px` |
| 移动相机绝对定位 | 已实现 | 每帧使用固定 ID100～102 重新计算相机位姿 |
| ID103 独立验证 | 程序已完成，结果未通过 | 最新结果约 `310.5 mm`，需要重新核对实物位置和配置后采集 |
| RViz 可视化 | 已完成 | 显示相机轨迹、参考 Tag、ID103 理想/实测姿态和误差 |
| ROS1 通用框架 | 骨架已完成 | 坐标链、参数入口、Mock 后端、适配器接口和安全门禁可运行 |
| 眼在手上手眼标定 | 未完成 | 缺少真实机械臂末端位姿和多姿态采样数据 |
| 工具坐标标定 | 未完成 | 需要确定法兰、TCP 和实际夹具几何关系 |
| RGB-D 对齐 | 工具已实现，Astra 标定待采集 | 当前尝试使用 UVC `1280x720` 彩色、ROS 深度/IR `1280x1024`（深度约 7 FPS）；USB2 带宽不足时两路一起退回 `640x480@30`，并重新标定 |
| YOLO 分割 | 接口和 UI 已完成 | 等待用户提供目标 `.pt/.pth`，当前没有真实目标 Mask 结果 |
| FoundationPose 6D | 核心运行时已接入，业务待真实数据验证 | ROS 适配器会按配置加载 FoundationPose++ CUDA、mycpp、PyTorch3D、nvdiffrast、refiner/scorer 权重；仍缺真实网格、RGB-D 标定和目标数据 |
| 物体三维建模工具 | 已完成骨架 | UI、Astra/OAK 采集后端、Tag 位姿、RGB-D 配准、TSDF 和 FoundationPose OBJ/PLY 导出已实现 |
| 真实机械臂控制 | 未接入 | 机械臂型号和官方通信接口未确定，运动输出强制关闭 |

## 3. 坐标约定与当前标定数据

AprilTag 对外坐标统一使用黑框左上角，不再使用中心点：

```text
O  = 黑框左上角 TL
+X = TL -> TR
+Y = TL -> BL
+Z = +X x +Y，指向纸面内部
```

当前标定 UI 配置采用实测黑框边长 `69.0 mm`，参考点为：

```text
ID100: [ 0.00,   0.00, 0.00] mm
ID101: [98.57,   0.00, 0.00] mm
ID102: [ 0.00, 108.43, 0.00] mm
ID103: [98.60, 108.40, 0.00] mm（验证点）
```

当前存在两个不能忽略的数据问题：

1. ROS 中央参数中的 ID103 仍为历史位置 `[230.0, 0.0, 0.0] mm`，与标定 UI 配置
   不一致。下一次正式验证前必须统一位置并执行 `calibration_tool.py sync-camera`。
2. 历史快照中的 `2.17 mm` 验证结果使用旧的 Tag 中心坐标约定，已经标记为不兼容；
   最新左上角坐标约定下的 ID103 结果约为 `310.5 mm`，因此当前没有可接受的验证精度
   结论。应把 ID103 左上角准确放到配置位置，保证四个 Tag 同时可见后重新采集。

标定数据唯一运行时入口为：

```text
/home/throne/workspaces/arm/ros_ws/src/arm_vision_framework/config/calibration_parameters.yaml
```

## 4. 软件入口

标定 UI：

```bash
cd /home/throne/workspaces/arm
./tool/camera_calibration/run_ui.sh
```

ROS 框架构建与无硬件检查：

```bash
cd /home/throne/workspaces/arm/ros_ws
catkin_make
source devel/setup.bash
rosrun arm_vision_framework run_pipeline.py --check --smoke
```

当前默认使用 Mock 分割、Mock 位姿估计和 Mock 机械臂。`dry_run: true`、
`allow_robot_motion: false`，框架不会向真实机械臂发送运动命令。Mock smoke 只说明接口和
矩阵链可以运行，不代表真实识别、定位精度或抓取成功率。

## 5. 文件结构规范（强制）

后续文件必须先按职责分类，再放入下面的固定位置。仓库根目录不是临时工作区；不在
本节中的新类别或新顶层目录，必须先说明用途并更新本规范，不能直接创建。

### 5.1 标准目录树

```text
/home/throne/workspaces/arm/
├── .gitignore                         # 全仓库忽略规则
├── camconfig.MD                       # Astra Pro 环境和实机取流记录
├── tool/                              # 独立于 ROS 业务包的通用工具
│   ├── __init__.py                    # Python 工具包入口
│   ├── camera_calibration/            # 相机/Tag 标定工具包
│       ├── README.md                  # 标定操作和坐标约定
│       ├── calibration_ui.py          # 唯一图形界面入口
│       ├── calib_common.py            # 标定、PnP 和几何公共实现
│       ├── calibrate_*.py             # 可复用命令行标定入口
│       ├── validate_workspace.py      # 工作平面独立验证入口
│       ├── hybrid_localization.py     # Tag 优先、机器人回退定位
│       ├── rviz_visualization.py      # 标定专用 ROS/RViz 发布
│       ├── config/                    # 标定工具输入配置
│       ├── targets/                   # 可打印 A4 PDF 和预览图
│       ├── output/                    # 当前正在使用的标定结果
│       ├── calibration_snapshots/     # 已冻结的正式标定快照
│       └── tests/                     # 标定几何回归测试
│   └── object_model_builder/           # RGB-D 物体网格生成工具
│       ├── README.md                   # 建模、Astra/OAK 和 FoundationPose 说明
│       ├── model_builder_ui.py         # 唯一图形界面入口
│       ├── camera_source.py             # Astra ROS/UVC 和 OAK DepthAI 后端
│       ├── rgbd_geometry.py             # 三维反投影、畸变校正和 Z-buffer 对齐
│       ├── rgbd_calibration.py          # RGB/IR ChArUco 双目标定
│       ├── tag_pose_provider.py         # AprilTag 工作空间相机位姿
│       ├── yolo_segmenter.py            # 用户 YOLO 实例分割接口
│       ├── capture_session.py           # 可复现的多视角数据集格式
│       ├── mesh_fusion.py               # Mask TSDF 融合和模型坐标系
│       ├── foundationpose_export.py    # OBJ/PLY 和元数据导出
│       ├── environment_check.py         # 依赖、权重和磁盘检查
│       ├── config/                      # Astra/OAK 和 TSDF 参数
│       ├── tests/                       # 无硬件 RGB-D/会话回归测试
│       └── run_ui.sh                    # 唯一启动入口
├── ros_ws/
│   ├── src/
│   │   ├── CMakeLists.txt
│   │   ├── arm_vision_framework/      # 本项目唯一上层业务 ROS 包
│   │   │   ├── README.md              # ROS 接口、构建和接入说明
│   │   │   ├── config/                # 中央标定参数和系统参数
│   │   │   ├── launch/                # 多节点组合启动文件
│   │   │   ├── scripts/               # 薄 ROS 节点/命令入口
│   │   │   ├── src/arm_vision_framework/
│   │   │   │   ├── adapters/          # 外部算法和机械臂接口适配
│   │   │   │   └── *.py               # 与 ROS 解耦的核心逻辑
│   │   │   ├── tools/                 # 参数导入、检查和维护工具
│   │   │   ├── tests/                 # 快速、确定性的单元/回归测试
│   │   │   └── validation/            # 实机精度和集成验证脚本
│   │   └── <vendor>_robot_driver/      # 未来厂商驱动独立 ROS 包
│   └── runs/                           # 测试输出，按任务和时间分目录
└── docs/                               # 总体说明、硬件手册和考核资料
```

构建产生的 `ros_ws/build/`、`devel/`、`install/`、Python 缓存和 `ros_ws/runs/`
不属于源码，已经加入忽略规则，禁止手工复制进其他目录保存。

### 5.2 各类文件的唯一位置

所有不依赖 ROS 节点生命周期、可独立运行的标定、数据转换、数据检查和可视化工具，统一
建立在 `tool/<tool_name>/`。每个工具必须有自己的 `README.md`、明确入口和测试；禁止把
零散 `.py` 文件直接堆在 `tool/` 根目录。必须通过 catkin 安装并由 `rosrun` 调用的包内
维护命令是唯一例外，仍放在对应 ROS 包的 `tools/` 中，但通用逻辑应复用 `tool/` 或核心
模块，不能复制实现。

| 文件类型 | 固定位置 | 规则 |
|---|---|---|
| 项目现状和开发顺序 | `docs/readme.txt` | 只描述全局状态，不复制模块操作手册 |
| 独立标定/转换/检查工具 | `tool/<tool_name>/` | 一个工具一个子目录，必须包含 README 和测试 |
| 标定使用说明 | `tool/camera_calibration/README.md` | 坐标约定、打印、采集和 RViz 说明放这里 |
| 物体网格生成工具 | `tool/object_model_builder/` | RGB-D 配准、Tag/YOLO 采集、TSDF 和 FoundationPose 导出统一放这里 |
| ROS 框架说明 | `ros_ws/src/arm_vision_framework/README.md` | 话题、适配器、构建和运行说明放这里 |
| 当前 Tag 布局 | `tool/camera_calibration/config/tag_layout.yaml` | UI 的实测输入，禁止在 Python 中硬编码 |
| 混合定位配置 | `tool/camera_calibration/config/hybrid_localization.yaml` | 只服务标定 UI 和独立定位工具 |
| 运行时标定参数 | `arm_vision_framework/config/calibration_parameters.yaml` | 业务程序唯一标定数据入口，只能用工具同步/导入 |
| 系统和安全参数 | `arm_vision_framework/config/system_parameters.yaml` | 后端、话题、阈值和运动门禁统一放这里 |
| ROS 启动文件 | `arm_vision_framework/launch/` | 只负责组合节点与参数，不写算法逻辑 |
| ROS 可执行入口 | `arm_vision_framework/scripts/` | 保持轻量，只做消息转换和调用核心模块 |
| 通用算法逻辑 | `arm_vision_framework/src/arm_vision_framework/` | 不直接依赖具体机械臂 SDK |
| YOLO/FoundationPose 适配 | `.../adapters/` | 每个外部后端一个有明确名称的适配器 |
| 厂商机械臂驱动 | `ros_ws/src/<vendor>_robot_driver/` | 独立 ROS 包，上层只依赖 canonical topic/service |
| ROS 包维护命令 | `arm_vision_framework/tools/` | 仅放必须由 catkin/rosrun 暴露的轻量入口 |
| 单元/回归测试 | 对应模块的 `tests/test_<module>.py` | 不需要相机或机械臂，结果必须确定 |
| 实机/精度验证脚本 | `arm_vision_framework/validation/validate_<capability>.py` | 一个脚本验证一个明确能力 |
| 运行结果 | `ros_ws/runs/<task>/<YYYYMMDD_HHMMSS>/` | 图片、CSV、日志和临时诊断统一放这里 |
| 正式标定快照 | `tool/camera_calibration/calibration_snapshots/<timestamp>_<profile>/` | 创建后只读，不覆盖、不手改 |
| 打印素材 | `tool/camera_calibration/targets/` | 只放生成器产生且尺寸验证通过的文件 |
| 硬件/考核资料 | `docs/` | PDF 不进入 Git，代码不得放在这里 |

表格中的 `arm_vision_framework/` 均指：

```text
/home/throne/workspaces/arm/ros_ws/src/arm_vision_framework/
```

### 5.3 大模型、数据集和第三方源码

YOLO 权重、FoundationPose 权重、BOP 数据集、物体网格和第三方仓库体积大、版本独立，
禁止放入当前 Git 仓库。统一放到仓库外：

```text
/home/throne/workspaces/arm_data/
├── models/yolo/                       # best.pt 等 YOLO 权重
├── models/foundationpose/             # FoundationPose 权重
├── meshes/                            # 易拉罐 OBJ/PLY 和纹理
├── model_sessions/                     # 物体多视角 RGB-D 采集
├── calibration_sessions/               # RGB/IR ChArUco 采集
├── datasets/bop/                      # BOP 数据集
└── third_party/                       # FoundationPose++ 等外部源码
```

业务配置只保存这些资源的绝对路径，不复制资源本体。外部仓库的修改应在其自己的 Git
仓库中管理，不能把其源码粘贴进 `arm_vision_framework`。

### 5.4 标定数据治理

标定数据存在三个阶段，不能混用：

```text
tool/camera_calibration/config/   实测布局和采集输入
             |
             v
tool/camera_calibration/output/   当前标定程序输出，可重新生成
             |
             v
calibration_snapshots/<timestamp> 正式冻结快照，只读
             |
             v
calibration_parameters.yaml       ROS 业务运行时唯一入口
```

规则如下：

1. 禁止业务节点直接读取 `output/` 或历史快照。
2. 禁止手工修改 `calibration_snapshots/` 内的 YAML、图片和 `SHA256SUMS`。
3. `calibration_snapshots/latest` 只由快照工具更新，不手工替换为普通目录。
4. 正式结果通过 `calibration_tool.py sync-camera` 或 `import-hand-eye` 进入中央参数。
5. 每次替换中央参数必须保留自动备份并运行 `calibration_tool.py validate`。
6. 不同相机、分辨率、像素格式和坐标约定必须使用不同 profile，禁止覆盖复用。

### 5.5 测试脚本与运行产物

后续用于验证框架准确性的小脚本按以下标准放置：

- 能验证纯 Python 逻辑且不依赖硬件：放 `tests/`，名称为 `test_<module>.py`。
- 需要相机、Tag、机械臂或 ROS 图像：放 `validation/`，名称为
  `validate_<capability>.py`。
- 临时排查问题但可能复用：放 `validation/diagnose_<problem>.py`，并在
  `validation/README.md` 写清输入、输出和删除条件。
- 生成的截图、CSV、YAML、视频和日志：只能写入 `ros_ws/runs/<task>/<timestamp>/`。
- 物体建模的大体积标定采集、RGB-D 会话和网格是例外，固定写入仓库外
  `/home/throne/workspaces/arm_data/{calibration_sessions,model_sessions,meshes}/`。
- 验证稳定后，公共逻辑必须移入核心模块；不能长期复制在多个验证脚本中。

`scripts/` 不是测试脚本目录，仓库根目录和 `docs/` 也禁止放测试脚本。

### 5.6 命名和新增文件规则

1. Python 文件和目录使用小写 `snake_case`，类名使用 `PascalCase`。
2. 测试使用 `test_*.py`，实机验证使用 `validate_*.py`，诊断使用 `diagnose_*.py`。
3. 配置名称必须表达职责，如 `calibration_parameters.yaml`，禁止使用
   `config2.yaml`、`new.yaml`、`final.yaml`。
4. 源码名称禁止出现 `copy`、`副本`、`test1`、`new`、`final_v2`、日期或 `(1)`。
5. 日期只允许出现在 `runs/` 和 `calibration_snapshots/` 的结果目录中。
6. 一个文件只承担一个清晰职责；公共矩阵、坐标和参数逻辑必须复用现有核心模块。
7. 禁止在代码中写死标定矩阵、模型路径、机械臂地址和 ROS 话题。
8. 新增后端优先实现现有接口，禁止为了一个设备复制整条 pipeline。
9. 新增文件前先搜索是否已有同类实现；不确定归属时先更新本节再创建。

### 5.7 每次改动的收尾要求

新增或移动文件后至少完成以下检查：

1. 更新对应模块 README 和本文件的现状表。
2. 为公共逻辑增加或更新回归测试。
3. 运行相关 `unittest`；修改 ROS 包时再运行 `catkin_make`。
4. 检查 `git status`，确认没有缓存、模型、数据集、构建目录或运行产物被暂存。
5. 提交信息说明修改的模块和验证结果，不把多个无关功能塞进一个 commit。

OAK-D-Pro 官方资料：

```text
https://docs.oakchina.cn/en/latest/
```

`docs/` 中保存了 OAK-D-Pro 产品手册、两类候选机械臂资料和考核说明。机械臂为随机
分配，确认实际型号并取得官方 SDK 或通信协议前，不应根据操作手册猜测控制端口和报文。

## 6. 下一阶段顺序

1. 统一 ID103 配置并重新完成动态验证，确认移动相机时误差不随视角系统性漂移。
2. 确认比赛机械臂型号，接入只读末端位姿并明确数据是法兰还是 TCP。
3. 完成工具坐标和 `T_gripper_camera` 眼在手上标定，再验证 Tag 不可见时的机器人回退。
4. 用 `tool/object_model_builder` 完成 Astra RGB/IR 外参采集，或确认 OAK-D Pro 工厂对齐输出。
5. 接入用户 YOLO 权重、目标类别和分割 Mask，采集瓶子多视角数据。
6. 使用已经实现的带 SHA-256 清单的采集 ZIP，在无 ROS 服务器上完成真实瓶子的 TSDF
   重建，检查米制 OBJ/PLY、重建报告和 FoundationPose 结果 ZIP。
7. 运行 FoundationPose++ 真实 RGB-D + Mask 的端到端定位测试。
8. 比较 Astra 与 OAK-D Pro 的深度覆盖率、位姿稳定性和运行速度；IMU 只作为预测/诊断辅助。
9. 在运动输出保持关闭的条件下完成定位验证，最后再接入机械臂安全控制。



## 7. OAK-D Pro 兼容边界

相机标定和物体建模不得绑定 Astra Pro。当前物体建模采集层已经分为 `astra_ros` 和
`oak_depthai` 后端；DepthAI `2.32.0.0` 已安装在 `foundationpose` 环境。切换 OAK 时修改：

```yaml
camera:
  backend: oak_depthai
```

OAK-D Pro 使用设备工厂内参、双目外参和 depth-to-RGB 对齐，可省去 Astra 独立 UVC RGB
与 IR 之间的手工外参步骤，但仍必须验证输出分辨率、深度单位、对齐像素和尺度。现有相机
标定 UI 可继续通过 ROS 图像话题完成 RGB/Tag 标定，业务层继续消费统一的 RGB、aligned
depth 和 CameraInfo，不直接依赖 DepthAI。

具体 OAK-D Pro 型号可能带 IMU。IMU 可用于短时姿态预测和时间同步诊断，不能替代 RGB-D
内外参、眼在手上 `T_gripper_camera` 或 AprilTag 工作空间绝对位姿。安装约束为：

```bash
python -m pip install -U --prefer-binary 'depthai>=2.17,<3'
```
