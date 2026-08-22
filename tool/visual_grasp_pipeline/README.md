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
