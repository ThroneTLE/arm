# 机械臂视觉项目现状

更新时间：2026-08-17

## 1. 项目目标

本项目面向随机分配机械臂的视觉抓取任务，围绕“比赛现场最薄主线”和“可复用视觉工具/
ROS 框架”两条路径展开，核心能力包括：

1. 相机内参、工具坐标和眼在手上手眼标定。
2. 机械臂状态读取、基础控制和安全停止。
3. YOLO 实例分割和 RGB-D 物体定位。
4. FoundationPose / FoundationPose++ 六自由度位姿估计与 AnyGrasp 候选抓取。
5. AprilTag 绝对定位与机械臂位姿回退相结合的定位链路。
6. MoveIt/Gazebo 抓取规划验证，以及真实厂商控制器适配。

当前仍处于“无机械臂实物的算法与接口验证阶段”。Astra Pro 用于软件链验证，比赛相机
配置已经支持 OAK-D Pro；UR5e + Robotiq 85 的上游 Gazebo/MoveIt 控制链和比赛闭环
抓取仿真已经跑通。真实机械臂控制、真实眼在手上标定和实物抓取仍需到场验收。

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
| 比赛主线 | 已完成软件骨架 | `competition_pipeline/` 固化 Tag/手眼/分割/定位/俯视抓取/闭环执行接口，确定性俯视为主线，AnyGrasp 为惰性加载备选，正式配置仍 fail-closed |
| UR5e + Robotiq 仿真 | 已验证 | 固定上游版本、MoveIt 控制链、GazeboGraspFix 和 `competition_sim_bridge` 已编译；完整木块抓取放置成功 |
| 真实机械臂控制 | 未接入 | 机械臂型号和官方通信接口未确定，运动输出强制关闭 |

## 3. 坐标约定与当前标定数据

历史 `tool/camera_calibration` 资料使用黑框左上角 TL；即将用于比赛的
`competition_pipeline` 使用用户确认的黑框右下角 BR。两套数据不能混用，切换坐标约定
必须重新检查 Tag 地图并使旧手眼结果失效。

比赛主线的 AprilTag 坐标约定为：

```text
O  = 黑框右下角 BR
+X = BR -> TR，沿右侧边向上
+Y = BR -> BL，沿下侧边向左
+Z = +X x +Y，垂直 Tag 平面
```

机器人基座统一为 `+X` 向前、`+Y` 向左、`+Z` 向上；所有 UI 输入使用毫米/度，内部
统一使用米制 `4x4` 齐次矩阵。比赛正式配置位于：

```text
/home/throne/workspaces/arm/competition_pipeline/config/competition.yaml
```

比赛配置中的 Tag 地图、RGB 内参、手眼矩阵和运行时安全门必须通过
`competition_pipeline` 工具维护，不能手工复制旧标定文件。

以下标定数据是历史 Astra 固定相机主线，仍保留供旧工具回归使用，不是比赛主线的
BR 坐标配置：

历史标定 UI 配置采用实测黑框边长 `69.0 mm`，参考点为：

```text
ID100: [ 0.00,   0.00, 0.00] mm
ID101: [98.57,   0.00, 0.00] mm
ID102: [ 0.00, 108.43, 0.00] mm
ID103: [98.60, 108.40, 0.00] mm（验证点）
```

历史 Astra 数据存在两个不能忽略的问题：

1. ROS 中央参数中的 ID103 仍为历史位置 `[230.0, 0.0, 0.0] mm`，与标定 UI 配置
   不一致。下一次正式验证前必须统一位置并执行 `calibration_tool.py sync-camera`。
2. 历史快照中的 `2.17 mm` 验证结果使用旧的 Tag 中心坐标约定，已经标记为不兼容；
   最新左上角坐标约定下的 ID103 结果约为 `310.5 mm`，因此当前没有可接受的验证精度
   结论。应把 ID103 左上角准确放到配置位置，保证四个 Tag 同时可见后重新采集。

历史 Astra ROS 业务的运行时入口为：

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

比赛流水线 UI 和 CLI：

```bash
cd /home/throne/workspaces/arm
./competition_pipeline/run_ui.sh
python3 -m competition_pipeline.cli check
```

UR5e + Robotiq 仿真：

```bash
cd /home/throne/workspaces/arm
./sim/build.sh                         # 首次或依赖变化后执行
./sim/run_competition_demo.sh          # 只规划，不允许运动
./sim/run_competition_demo.sh --execute # 无界面完整抓取
./sim/run_competition_demo.sh --execute --gui  # Gazebo + RViz
./sim/stop.sh                           # 清理托管仿真
```

仿真不自制机器人模型，使用固定版本的上游 UR5e + Robotiq 85、MoveIt、Gazebo world
和控制器；`sim/bootstrap.sh` 会在新机器上拉取上游源码并应用 Noetic 兼容补丁。仿真成功
只代表规划和控制链可用，不代表真实控制器的 MOVJ/MOVL 协议已经接入。

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
├── competition_pipeline/                # 比赛现场标定、定位和抓取主线
│   ├── config/                          # 比赛相机、Tag、手眼和安全参数
│   ├── planning.py                      # 抓取目标和预/抓/放置位姿
│   ├── execution.py                     # fail-closed 闭环执行状态机
│   ├── interfaces.py                    # 机械臂、夹爪、相机和物体接口
│   ├── tests/                           # 无硬件回归测试
│   └── run_ui.sh                        # 比赛流水线 UI 入口
├── sim/                                 # 固定上游 UR5e + Robotiq 仿真
│   ├── upstream.repos                   # 上游 commit 锁定
│   ├── bootstrap.sh / build.sh          # 拉取、补丁和 catkin 构建
│   └── ws/src/competition_sim_bridge/   # 薄 MoveIt/Gazebo 适配层
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
| 项目现状和开发顺序 | `docs/readme.md` | 只描述全局状态，不复制模块操作手册 |
| 独立标定/转换/检查工具 | `tool/<tool_name>/` | 一个工具一个子目录，必须包含 README 和测试 |
| 标定使用说明 | `tool/camera_calibration/README.md` | 坐标约定、打印、采集和 RViz 说明放这里 |
| 物体网格生成工具 | `tool/object_model_builder/` | RGB-D 配准、Tag/YOLO 采集、TSDF 和 FoundationPose 导出统一放这里 |
| 比赛现场主线 | `competition_pipeline/` | 比赛配置、Tag/手眼、定位、抓取规划和 fail-closed 执行统一放这里 |
| 机械臂仿真 | `sim/` | 只维护上游固定版本、补丁、薄 bridge 和启动脚本，不复制机器人模型 |
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

1. 在 Gazebo 增加桌面、横梁、侧墙等复杂障碍，并将同一几何同步到 MoveIt
   PlanningScene，验证可绕行、不可达拒绝和携物碰撞三类场景。
2. 将真实控制边界扩展为 MoveIt 关节轨迹 + 厂商 MOVJ/MOVL：全局避障使用关节轨迹，
   预抓取到抓取、抬升和撤退使用经过采样检查的低速 MOVL。
3. 确认比赛机械臂型号，接入只读关节/TCP 状态，明确数据是法兰还是 TCP；在完成厂商
   急停、限速和通信超时验收前保持 `dry_run: true`。
4. 完成工具坐标和 `T_tcp_color_camera` 眼在手上标定，再验证 Tag 不可见时的机器人回退。
5. 到场验证 OAK-D Pro 枚举、EEPROM、投影器、对齐深度和最终分辨率；Astra 仅作为备用
   软件验证相机。
6. 接入比赛正式 YOLO 权重、目标类别和分割 Mask，完成多实例、遮挡和复杂背景验证。
7. 根据物体和比赛规则，在确定性俯视抓取、AnyGrasp 与 FoundationPose 之间选择最薄的
   正式定位/抓取组合，不让尚未验收的模型阻塞主线。
8. 完成真实 RGB-D、定位、规划、抓取、抬升验证和放置验证的端到端 dry-run，最后才打开
   真实运动许可。



## 7. 22/23 现场最小闭环与控制器对齐

本节是无实物阶段必须提前固化的现场顺序。控制器手册为
[`系统操作手册`](c9c6716a-f022-46a1-b5c4-3f982a816a50.pdf)；它能确认示教器中的
TCP、用户坐标、点位和远程 IO 行为，但不包含 TCP 网络报文、端口、寄存器地址或夹爪
厂商协议。因此真实通信适配器必须等到现场拿到对应的通信协议/SDK 后再实现，不能根据
`MOVJ/MOVL` 名称猜测报文。

### 7.1 到场后的安全和状态采集

在任何自动运动前，先确认控制柜和示教器急停能够切断伺服，再以低速手动确认工作空间、
机械臂型号、控制器 IP/接口、工具编号、用户坐标编号和当前伺服状态。第一条只读通信应
同时记录：

- 六个关节角、控制器返回的 TCP 位姿及其坐标系（基座/用户/工具）；
- 当前 Tool ID、User ID、速度/加速度倍率和运动模式；
- 当前形态参数，以及示教器显示的活动工具和用户坐标；
- 远程模式、数字 IO/Modbus 是否已使能和夹爪反馈信号。

### 7.2 TCP/工具坐标标定

手册的七点标定要求固定一个尖锐参考点，按 `TC1`～`TC4` 取得四个不同姿态，`TC5`
回到与 `TC1` 相同的姿态，`TC6` 在 `TC5` 基础上沿笛卡尔 X 负方向移动任意距离，
`TC7` 再沿笛卡尔 Y 正方向移动任意距离，最后计算并用“运行至该点”复核。各姿态不能
只绕同一方向旋转，参考点在整个过程中必须固定。

关于“夹笔再减去笔尖到夹爪顶端距离”的做法：七点法计算的是实际接触参考点的 TCP，
所以夹着笔标定得到的是**笔尖 TCP**。如果笔尖和实际抓取 TCP 共线，且笔尖只沿工具坐标
系的 Z 轴突出，便可以直接用测得的有符号长度 `d` 修正：

```text
p_gripper_tcp = p_pen_tip - d * z_tool
```

这里 `z_tool` 是当前工具坐标系的单位 Z 轴，`d > 0` 表示笔尖位于 `+Z_tool` 方向；如果
现场工具坐标定义相反，符号随之相反。这个修正只改变平移，不改变 A/B/C 姿态。实际操作
可以把七点法结果写入临时 `Tool2`（笔尖），再将 `Z_tool` 减去笔尖到目标抓取 TCP 的
距离，保存为生产 `Tool1`，并用“运行至该点”复核 TCP 是否仍固定在预期夹爪位置。

只有在笔尖不与目标 TCP 共线、夹持有倾角、或目标 TCP 不是同一条 Z 轴上的点时，才需要
记录完整三维刚体变换，而不能只减一个长度。无论采用哪种方式，笔和夹爪必须在采样期间
保持刚性不动；拆笔、换夹具或修改 Tool 参数后，原手眼结果必须作废并重新采集。

手册中的直接参数方式也可用于已知工具尺寸：X/Y/Z 是相对法兰中心的毫米偏移，A/B/C
是相对法兰的旋转角。

### 7.3 用户坐标系

用户坐标应在生产 TCP 已确认后设置。按手册的三点法：

1. 用 TCP 到用户坐标原点，记录“标定原点”；
2. 沿期望的用户 X 正方向移动任意距离，记录“标定 X 轴”；
3. 沿期望的用户 Y 正方向移动任意距离，记录“标定 Y 轴”。

直接填写用户坐标参数时，X/Y/Z 是用户原点相对机器人基座的偏移，A/B/C 是绕基座轴的
旋转且单位为**弧度**。建议保存为控制器 `User1`，同时把 `T_base_user1` 读回并写入
现场记录。软件内部仍以 `base` 坐标规划；如果厂商命令使用 User1，在驱动适配层完成
`base ↔ User1` 转换，不在视觉和 MoveIt 中混用两个坐标系。

### 7.4 手眼标定和运行时定位策略

AprilTag 在比赛主线的主要用途是采集手眼样本和验收刚体安装关系：

```text
T_base_camera = T_base_tcp × T_tcp_camera
```

每个样本必须同时保存当前活动 Tool1 的 TCP 位姿、图像时间戳、Tag PnP 结果和相机 profile。
Tag 地图变化、Tool1 变化、相机分辨率/内参变化或刚性安装变化后，样本全部失效。

实物运行时优先使用控制器反馈的毫米级 `T_base_tcp`，不要求 AprilTag 持续在视野中；
AprilTag 视觉结果保留为开机验收、手眼质量检查和 TCP 反馈失效时的可选诊断回退。当前
`competition_pipeline/localization.py` 仍是“视觉优先、TCP 回退”的兼容实现，后续真实
控制器接入时应增加 `tcp_primary` 运行模式，且视觉回退默认不允许直接触发运动。

### 7.5 形态参数与 MoveIt 轨迹

手册定义六轴机器人的形态值为 1/3/5 轴所在区间的二进制编码：某轴角度在
`[-90°, +90°]` 内记为 1，否则为 0，按 1/3/5 轴组成二进制数后再加 1。例如 `110₂ + 1`
得到形态值 7。位置变量还绑定坐标系、角度/弧度标志、Tool ID 和 User ID；这些字段
不能省略。

“第一次读到的形态参数后一直复用”只有在整条轨迹保持同一构型分支时才成立。工程上应
从 MoveIt 的每个关节点计算并校验形态值；若 1/3/5 轴跨过区间边界，或者出现关节跳变、
接近奇异位形，就必须重新规划/拒绝执行，而不是强行发送固定形态值。

运动执行的最薄方案为：

- 远距离和绕障：MoveIt 生成带时间戳的关节轨迹，转换为控制器可接受的连续 MOVJ 点列；
- 预抓取到抓取、抬升和短距离撤退：先做 Cartesian 碰撞/IK/奇异性检查，再用低速 MOVL；
- 放置前重新规划 MOVJ，释放后用 MOVL 或安全 MOVJ 撤离；
- 每个点带完整坐标系、Tool/User ID 和形态字段，执行后读取实际关节/TCP 做误差确认。

当前仿真已验证 MoveIt/FollowJointTrajectory，但尚未验证厂商 MOVJ/MOVL 的点列、速度、
加速度、blend、停止和错误码语义；这些必须在通信适配器和现场低速测试中单独验收。

### 7.6 夹爪远程控制

手册确认远程模式支持数字 IO 和 Modbus 从站，Modbus 优先级高于数字 IO；两者同时使用时
可在 `modbusAddr.json` 中将 `coexistIOControl` 设为 `true`，示教器拔出会触发远程模式。
但当前手册没有夹爪具体输出位、寄存器地址、动作值、完成反馈或急停语义，因此现场必须
补齐以下表格后才能实现 `GripperController`：

```text
控制方式：数字 IO / Modbus
输出通道或寄存器：待现场确认
打开值、闭合值、保持值：待现场确认
动作完成反馈/夹持反馈：待现场确认
有效电平、超时、急停和断线行为：待现场确认
```

在没有反馈之前，不能把“命令发送成功”当成“夹住物体”；仿真中已经采用物体抬升和放置
位置验证，真实控制器也必须保留同等的 fail-closed 验证门。

### 7.7 明日（无实物准备日）执行清单

明天不做任何真实机械臂运动，也不填写猜测的控制器 IP、端口、寄存器或 TCP 数值。目标
是把到场后的一天半压缩成可执行的表单、接口和验收脚本。每项完成后必须留下可复用的
文件或日志，而不是只在终端手工试过。

| 时段 | 任务 | 产出 | 完成门槛 |
|---|---|---|---|
| 09:00–09:30 | 固定代码和环境基线 | `git diff`、依赖版本、仿真启动日志 | `sim/build.sh` 成功；仿真可启动、可停止，`sim/stop.sh` 后无残留 `gzserver/gzclient/roslaunch` |
| 09:30–10:30 | 冻结比赛配置 | `competition.yaml` 的 BR Tag、基座轴向、相机 profile、`dry_run` 和工具/用户编号占位 | `python3 -m competition_pipeline.cli check` 通过；运动许可仍为关闭 |
| 10:30–11:30 | 固化 TCP 七点法记录表 | 七个姿态记录模板、笔尖突出距离 `d`、Tool1/Tool2 关系说明 | 明确 `p_gripper_tcp = p_pen_tip - d*z_tool` 的符号和测量基准；没有实测值时留空，不用默认值 |
| 13:30–14:30 | 固化相机与手眼采集接口 | 相机内参/Tag 地图检查命令、手眼样本 JSON/CSV 字段、质量报告入口 | 能用仿真或 mock 数据跑通 `T_base_tcp × T_tcp_camera`，并拒绝缺字段样本 |
| 14:30–16:00 | 固化规划执行边界 | MoveIt 轨迹到 MOVJ 点列的适配接口、近距离 MOVL 接口、形态参数校验测试 | 每个点都带 Tool/User/shape；跨构型边界或奇异风险时拒绝执行；默认 `dry_run` |
| 16:00–16:45 | 整理控制器和夹爪待确认表 | `controller现场信息.md`（IP/端口、协议、Tool/User、IO/Modbus、反馈、急停） | 所有未知项明确标记“现场确认”，不根据 MOVJ/MOVL 名称猜报文 |
| 16:45–17:30 | 完整 dry-run 和交接包 | 一键命令、日志、问题清单、现场打印版检查表 | 从启动、定位、规划、夹爪 mock 到停止全流程可重复；形成当天唯一版本归档 |

明天结束时必须得到以下五项：

1. 一份不含猜测数值的 `competition.yaml`；
2. 一份可打印的 TCP 七点法和用户坐标三点法记录表；
3. 一份手眼采样字段和 Tag 地图检查脚本；
4. 一套 MoveJ 远距离、MoveL 近距离的仿真/mock 回归测试；
5. 一份现场只读采集、低速试运动、夹爪 IO 和急停确认的待办表。

到场后的第一小时只执行安全确认和只读状态采集；第二小时才开始夹笔七点法。只有
Tool1、User1、控制器返回的形态参数和 TCP 坐标全部读回并复核后，才允许打开低速运动；
真实抓取前仍需通过“抬升保持”和“放置到目标区域”两个验证门。

## 8. OAK-D Pro 兼容边界

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
