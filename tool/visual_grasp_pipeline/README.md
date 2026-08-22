# 视觉抓取流水线工具（从 fp_release 迁移）

本目录把外部复现包 `fp_release_20260821_155930` 中的“YOLO + FoundationPose +
AprilTag + 抓取位姿”逻辑迁移进本仓库的 `tool/`，并去掉原包里的机器相关硬编码和
FoundationPose 厂商代码重复导入。

## 与原包的关系

- 原包入口是散落在 `~/` 下的 `fp_pipeline.py`、`grasp_ui.py`、`vision_node.py`、
  `arm_node.py`、`fp_bridge.py`、`capture_frame.py` 等脚本。
- 迁移后的公共逻辑拆分到本目录：
  - `geometry.py`：Tag 工作台系、坐标补偿、圆柱/球体抓取位姿、深度空洞填充；
  - `tracking.py`：YOLO 多实例 IOU 稳定跟踪和序列解析；
  - `detection.py`：YOLO 检测、目标选择、AprilTag 检测；
  - `foundationpose.py`：复用 `ros_ws` 里已经验证过的 `FoundationPoseRuntime`；
  - `config.py` + `config/visual_grasp_pipeline.yaml`：路径和参数集中配置；
  - `pipeline.py`：集中导出公共 API 的兼容入口；
  - `offline.py`：无相机离线验证入口。
- 大文件（YOLO 权重、CAD 网格、FoundationPose 权重）不进入 Git，仍放在仓库外或原
  复现包路径，通过 YAML 配置引用。
- `legacy/` 保留原复现包的完整脚本快照，便于对照和排查；实际运行请使用本目录的
  重构模块和 `offline.py`。

## 图形界面（识别结果）

在 `foundationpose` conda 环境中运行：

```bash
cd /home/throne/workspaces/arm
./tool/visual_grasp_pipeline/run_ui.sh
```

界面功能：

- **静态帧识别**：加载 `static_frame` 中的 RGB，显示 YOLO 检测框和类别/置信度；
- **相机识别**：连接原 `fp_bridge` 的 ZMQ 帧源（`tcp://127.0.0.1:5555`），显示实时识别结果；
- **识别并计算位姿**：对下拉框选中的目标运行 FoundationPose + Tag + 抓取位姿计算，结果显示在右侧文本区。

没有相机时先点“静态帧识别”即可看到识别结果；点“相机识别”失败时会自动改用静态帧。
“识别并计算位姿”首次会加载 FoundationPose 权重和 CUDA 上下文，因此较慢；同一个模型的
后续计算会复用已加载的实例，速度会明显提升。

## OAK 原版 Vision Node 兼容入口

原版“拍照识别 → 选择/编排目标序列 → FoundationPose → Tag/相机系 → 抓取位姿”界面已
迁移为单进程 OAK 直连脚本，不再需要 `fp_bridge.py`：

```bash
cd /home/throne/workspaces/arm
./tool/visual_grasp_pipeline/run_oak_vision_node.sh
```

默认是算法 Dry-run，只保存和显示位姿，不发送机械臂运动。若需要配合旧版模拟
`arm_node.py`，先启动模拟节点，再显式运行：

```bash
./tool/visual_grasp_pipeline/run_oak_vision_node.sh \
  --legacy-arm-service tcp://127.0.0.1:5556
```

没有图形桌面时可执行相机与 YOLO 自检：

```bash
./tool/visual_grasp_pipeline/run_oak_vision_node.sh --camera-check
```

该脚本直接使用当前 OAK EEPROM 内参、1920×1080 硬件对齐 Depth 和配置中的 MXID；没有
检测到当前 ID100–102 地图或兼容 `tag0` 时，仍会输出相机系 FoundationPose 结果，但
机械臂调用会被强制阻止。
针对 RTX 4060 Laptop 的 8GB 显存，FoundationPose 会把 YOLO ROI 裁剪到最多 640 像素、
将注册候选限制为 64，并在切换模型时释放上一套 CUDA 运行时，避免完整 1080P × 252
候选造成显存峰值。

### 物体位姿映射到用户坐标系1（UCS1，默认开启）

识别结果不再只是“相对相机的位置”，而是按现场手眼标定链

```
T_user1_object = T_user1_tcp @ T_tcp_color_camera @ T_camera_from_object
```

映射到用户坐标系1（`competition_pipeline/config/competition.yaml` 的
`hand_eye.tcp_from_color_camera`，15/16 内点 3.02 mm / 0.72°）。其中
`T_user1_tcp` 默认从控制器实时回读（7000 端口状态服务，`pose_frame=UCS`）。
程序在冻结 RGB-D 快照后立即冻结 TCP，FoundationPose 即使计算数秒也不会把
“旧照片”与“新机械臂姿态”混用。算法结果区先显示用户系的物体/抓取点
XYZ 和 A/B/C，相机系与工作台系仅作参考；同时保存
`object_pose_user1.npy` 和 `grasp_user1.npy`。

可配置项：

- `--no-tcp-read`：完全不连接控制器（仅相机系输出，用户1映射显示为“未映射”，
  绝不静默用“用户1原点”单位阵冒充真实位姿）；
- `--controller-host <ip>`：覆盖 `controller.nexbot_tcp.host`；
- `--tcp-xyz-mm x y z --tcp-rpy-deg a b c`：不用回读，手动指定用户系下的
  当前 TCP 位姿（静态值，适合机械臂不可用时验证映射链）；
- 控制器暂时不可达/被其他程序占用（6001/7000 单客户端）时本次用户系
  映射直接标记为“未映射（原因）”，绝不沿用上次 TCP。

只读状态查询，不会向机械臂发送任何运动指令。

### 一键抓取：计算 → 出坐标 → 我确认 → 执行（用户坐标系1）

界面流程：目标序列计算完成后，结果区给出用户1抓取点 XYZ（含 ABC 参考值），
下方按钮 `执行抓取` 变为可用；点击后弹出确认框（显示抓取/放置坐标），确认后
整轮执行。**每轮开始与结束都回到复位位置**（控制器"复位点设置-安全使能"开启
后，新一轮运动必须从复位点启动；这是 2026-08-22 日志+官方手册确认的现场
结论，本轮结束位置=复位点）。

整轮计划（姿态固定为"复位位置的姿态"，只动 XYZ）：

```
回复位(0x3007) -> 抓取(视觉 user1 XYZ) -> 夹爪合(DOUT15/16)
-> 放置(X=-100, Y=100, Z=抓取Z) -> 夹爪开 -> 回复位
```

- 执行器：`tool/visual_grasp_pipeline/ucs_grasp.py`，复用比赛 UI 同款控制栈
  （`competition_pipeline` 的 NexBotTcpJog：6001 运动口、MOVL coord=3 用户系1、
  夹爪 DOUT15/16、0x3007 回复位、0x2314 急停）；
- 安全硬限（用户坐标系1）：XY 不超过 ±300 mm、Z 在 10–350 mm、
  单段位移 ≤ 600 mm，超限直接拒绝并弹窗；
- 失败/取消：立即急停，绝不把机器人留在放置位上方；
- 默认 **Dry-run**（只打印/展示计划，不连接不运动）；真实运动需
  `--enable-robot-motion` 启动并在确认框二次确认；
- CLI：`--place-x-mm` / `--place-y-mm`（默认 -100 / 100，放置 Z 恒等于抓取 Z）。

```bash
# 验证计划（默认，安全）：
./tool/visual_grasp_pipeline/run_oak_vision_node.sh
# 真实运动：
./tool/visual_grasp_pipeline/run_oak_vision_node.sh --enable-robot-motion
```

注意：6001/7000 为控制器单客户端端口，运行本节点时请勿同时开启比赛 UI，
避免抢占连接。

## 离线验证

在 `foundationpose` conda 环境中运行：

```bash
cd /home/throne/workspaces/arm
./tool/visual_grasp_pipeline/run_offline.sh \
  --config tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml \
  --label can
```

当前模型库单位不统一：两个罐体 OBJ 使用毫米，苹果/香蕉 GLB 使用米。配置通过
`object_model_scales` 逐模型指定比例（罐体 `0.001`、水果 `1.0`），禁止再用一个全局
比例覆盖所有类别。

也可以指定输出目录保存 `camera_from_object.npy`、`world_from_object.npy`、
`grasp.npy` 和 `result.json`。CLI 的 stdout 只输出 JSON，FoundationPose 的
原生诊断信息会转到 stderr：

```bash
./tool/visual_grasp_pipeline/run_offline.sh \
  --config tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml \
  --label can \
  --save-dir /tmp/visual_grasp_verify
```

脚本会加载 `static_frame/rgb.png`、`depth.png`、`cam_K.txt`，依次执行：

1. YOLO 检测并选择目标；
2. FoundationPose 注册得到相机系位姿；
3. 如果画面中有 AprilTag，建立工作台系并做原包同样的坐标补偿/翻转；
4. 按物体规则生成抓取位姿；
5. 输出 JSON 并保存 npy 验证文件。

## 单元测试

不需要相机、GPU、YOLO 或 FoundationPose，只验证迁移后的纯逻辑：

```bash
cd /home/throne/workspaces/arm
python3 -m unittest discover -s tool/visual_grasp_pipeline/tests -v
```

## 配置说明

`config/visual_grasp_pipeline.yaml` 中的默认路径指向本机用于验证的
`fp_release_20260821_155930` 和 `arm_data`。换机器后只需改：

- `paths.yolo_weights`
- `paths.foundationpose_root`
- `paths.static_frame_dir`
- `object_models` 下的 CAD 网格路径

其余参数（偏移、翻转、抓取规则、FoundationPose 迭代次数等）与原包配置区一一对应。
