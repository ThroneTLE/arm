# 物体三维模型工作台

这个工具把多视角 RGB-D 采集、AprilTag 相机位姿、YOLO 实例分割和 TSDF 融合串成一个可重复的工作流，输出 FoundationPose/ FoundationPose++ 可以读取的米制 `OBJ`/`PLY` 网格。

## 先说结论

高斯泼溅（Gaussian Splatting）不是当前的最终建模主链。它适合新视角渲染，但 FoundationPose 的 CAD 模式需要显式三角网格、法向和尺度；从高斯表示再转网格会增加误差和处理步骤。因此这里使用经过标定的多视角 RGB-D + Mask + TSDF。高斯表示以后可以作为可视化或补洞的实验分支，不能绕过 RGB-D 外参和尺度校验。

AprilTag 对建模有帮助，但条件是瓶子与工作空间中的 Tag 在整段采集期间保持刚性不动，移动的是相机。程序每帧用 Tag Map 计算 `T_workspace_color`，然后把每个深度点变换到统一工作坐标系。如果瓶子被拿起或相对 Tag 旋转，Tag 位姿不能再用于融合；需要给物体本身建立标记、使用转台标定，或另建物体运动轨迹。

## 环境和启动

FoundationPose++ 外部源码放在：

```text
/home/throne/workspaces/arm_data/third_party/FoundationPose-plus-plus
```

当前 `foundationpose` 环境已验证：Python 3.9、PyTorch 2.4.1 + CUDA 12.1、Open3D、PyTorch3D 0.7.8、nvdiffrast、`mycpp`、PyQt5、官方 refiner/scorer 权重和 GPU 光栅化初始化。桌面前端与相机标定工具统一使用 PyQt5 和同一套工作台样式。Qwen-VL、SAM-HQ、Cutie 不是本工具的必需依赖，因为首帧 Mask 由用户的 YOLO 分割模型提供。

ROS 运行时适配器位于 `ros_ws/src/arm_vision_framework/src/arm_vision_framework/adapters/foundationpose.py`。
它会将 BGR 转为 RGB、检查米制深度和尺寸，并复用 FoundationPose++ 的 refiner/scorer。
`warp` 是上游的可选深度滤波依赖；当前环境没有安装时，适配器使用 OpenCV 回退，不会在首帧因
`erode_depth` 缺失而崩溃。

检查环境：

```bash
cd /home/throne/workspaces/arm
./tool/object_model_builder/run_ui.sh --check
```

打开 UI：

```bash
cd /home/throne/workspaces/arm
./tool/object_model_builder/run_ui.sh
```

相机模式会依次加载 ROS Noetic 和默认 Orbbec 工作区
`/home/throne/orbbec_ws/devel/setup.bash`。如果驱动编译在其他目录，通过环境变量指定：

```bash
ORBBEC_ROS_SETUP=/path/to/orbbec_ws/devel/setup.bash \
  ./tool/object_model_builder/run_ui.sh
```

驱动包名、launch 文件和启动参数集中配置在 `camera.ros_driver`。若连接失败，UI 会显示
ROS 驱动日志末尾，完整日志保存在
`/home/throne/workspaces/arm_data/model_sessions/orbbec_driver.log`。
深度和红外消息由工具直接解析，不调用与 Conda 二进制库容易冲突的 ROS `cv_bridge`。
Astra 的 `roslaunch` 子进程会过滤 Conda 动态库目录，避免原生驱动误加载 Conda 的
`libffi`；FoundationPose 主程序仍保留自己的 Conda 运行环境。
Astra 在没有安装厂家 CameraInfo 文件时会发布 `NaN` 内参，工具会拒绝该无效值；完成
UI 中的 RGB-D 标定后，建模改用中央参数中的真实深度内参和 `T_color_depth`。

进入“RGB-D 标定”页时，工具会通过 `/camera/set_laser` 自动关闭 Astra 红外投影器，
避免结构光散斑破坏标定板；离开该页、断开相机或关闭程序时会重新打开投影器。
主预览区会同时显示归一化红外图、当前标定板角点数和投影器状态。每次保存前必须在
RGB 与 IR 中找到配置数量的共同角点，否则该图像对会被拒绝，不会混入标定数据。

UI 的默认配置在 `config/object_model_builder.yaml`。权重、采集数据和网格都在仓库外的 `arm_data`，不会进入 Git。

## 本机采集与服务器重建

采集和重建可以分开执行。本机连接相机，在“无模型拍照”页完成 Tag 位姿、YOLO Mask 和
RGB-D 数据采集，然后点击“导出 FoundationPose 参考照片 ZIP”。如果需要 TSDF 服务器
重建，仍可在重建页使用通用采集 ZIP；两种 ZIP 的用途不同。

```text
archive_manifest.yaml       # 格式版本、逐文件大小和 SHA-256
manifest.yaml               # 内参、RGB-D 外参、坐标系、分割模型信息
color/*.png                 # 校正后的 RGB
depth_raw/*.png             # 原始深度，单位 mm
depth_aligned/*.png         # 对齐到 RGB 的深度，单位 mm
mask/*.png                  # YOLO 二值 Mask
pose/*.txt                  # 每帧 T_workspace_color
provenance/*.yaml           # 标定参数和 Tag 排布快照
```

采集 ZIP 不复制 YOLO 权重，因为 Mask 已经逐帧保存；服务器不需要相机、ROS、Tag 检测或
YOLO 权重。服务器只需要本仓库、`foundationpose` Python 环境、Open3D 和足够的磁盘空间。
ZIP 解包时会校验路径、文件数量、总大小和 SHA-256，篡改、损坏或路径穿越归档会被拒绝。

本机也可以用命令行打包已有会话：

```bash
./tool/object_model_builder/run_ui.sh \
  --pack-session /home/throne/workspaces/arm_data/model_sessions/object_scan_xxx \
  --archive-output /home/throne/workspaces/arm_data/model_sessions/object_scan_xxx.zip
```

把 ZIP 上传到服务器后，无界面重建：

```bash
CONDA_ROOT=/server/miniconda3 \
./tool/object_model_builder/run_ui.sh \
  --reconstruct-zip /server/uploads/object_scan_xxx.zip \
  --model-name bottle \
  --output-root /server/results \
  --work-root /server/work
```

离线模式不启动 ROS 和相机。每个任务使用独立时间戳目录，最终生成：

```text
/server/results/bottle_<timestamp>.zip
/server/results/bottle_<timestamp>/bottle/
├── bottle.obj
├── bottle.ply
├── model_metadata.yaml
└── reconstruction_report.yaml
```

结果 ZIP 同样带 SHA-256 清单，可以直接从服务器下载。UI 的“重建与导出”页也支持选择
采集 ZIP 后一键完成相同流程。当前分工是本机执行轻量实时 YOLO/Tag/对齐和数据落盘，
服务器执行 TSDF 融合与 CAD 导出；如果以后连实时 YOLO 也需要卸载到服务器，应新增
“原始无 Mask 采集”协议，不能把没有 Mask 的 ZIP 冒充当前格式。

当前 Open3D Scalable TSDF 主要消耗服务器 CPU、内存和磁盘，不会充分利用 4060；本机
GPU 主要用于实时 YOLO，后续 FoundationPose 在线位姿估计才是明显的 GPU 工作负载。
因此 ZIP 方案的主要价值是把采集与批处理解耦，并为后续服务器 FoundationPose 部署保留
同一份标定数据和模型产物，而不是假设 TSDF 本身必须使用更大的显卡。

## FoundationPose 无模型拍照与实时测试

“无模型”指采集参考视图后再用上游 BundleSDF / Neural Object Field 重建网格，不代表
FoundationPose 能直接用裸 RGB-D 永久跳过网格。采集页提供以下工作流：

1. 点击“1 加载 YOLO 分割模型”；
2. 点击“2 新建参考拍照会话”，固定物体与 Tag，只移动相机；
3. 点击“3 拍摄参考图”。按钮会等待下一组同步 RGB-D，不会因某一帧时间差弹窗失败；
4. 点击“查看已拍照片（N 张）”检查 RGB、Mask、对齐深度和覆盖率；
5. 拍够 16 张后，可点击“照片快速预览三维”用 TSDF 检查现场拍摄效果；
6. 点击“4 导出 FoundationPose 参考照片 ZIP”保存可移植数据；
7. 点击“5 用 16 张参考图训练神经隐式模型”，执行官方 BundleSDF/Neural Object Field；
8. 训练完成后，Tool 会自动填入生成的 `model.obj`，点击“6 加载重建模型并实时测试”。

第 7 步是真正的 FoundationPose Model-free 路线：参考 RGB-D 被送入
`FoundationPose/bundlesdf/run_nerf.py::run_neural_object_field`，先训练神经隐式
表示，再从隐式场提取带纹理的网格。FoundationPose 的 REGISTER/TRACK 接口仍以这个
提取出的网格作为统一下游输入；它不是把 16 张 PNG 直接当作 CAD，也不会训练一个
新的 FoundationPose 网络。运行该步骤需要 CUDA、PyTorch3D、nvdiffrast 和 Kaolin；
缺少 Kaolin 时环境页仍允许拍照和导出 ZIP，但 Model-free 按钮会给出明确依赖错误。
每个任务目录会同时保留 `reference/ob_0000001/`、`result/nerf/`（神经隐式训练检查点）
和 `result/model/model.obj`，方便后续复查或在另一台机器继续处理。

也可以在无界面环境运行同一条入口（`--session` 会先导出参考目录）：

```bash
python -m tool.object_model_builder.model_free \
  --session /path/to/capture_session \
  --reference-dir /path/to/reference_root \
  --foundationpose-root /home/throne/workspaces/arm_data/third_party/FoundationPose-plus-plus \
  --output-dir /home/throne/workspaces/arm_data/model_free_jobs/object_001
```

无模型 ZIP 直接包含上游 `run_nerf.py` 使用的参考布局：

```text
foundationpose_reference_manifest.yaml    # 文件大小和 SHA-256
ob_0000001/
├── K.txt                                 # 校正 RGB 内参
├── select_frames.yml
├── reference_metadata.yaml
├── source_manifest.yaml
├── rgb/0000000.png
├── depth_enhanced/0000000.png             # uint16 mm
├── mask/0000000.png
├── cam_in_ob/0000000.txt                  # camera -> 固定物体/工作坐标
└── provenance/
```

当前机器的 `foundationpose` 环境可以运行已有网格的实时 FoundationPose，但没有安装
Kaolin，因此 UI 环境检查会把“Kaolin (无模型 Neural Object Field)”标记为缺失；这不会
阻止拍照、打包 ZIP 或使用已有网格实时测试，但点击 Model-free 建模前必须在该环境补齐
Kaolin。工具不会在运行时自动安装或改动环境，建议按上游版本安装与当前 PyTorch/CUDA
匹配的 Kaolin，再重启 UI。

采集门禁分为两类：彩深时间差使用主机到达时间戳，深度约 `7 FPS`，因此点击拍摄后会
等待最多 5 秒的下一组配对帧；Mask 内有效深度覆盖率、物体相对 Tag 是否移动和是否换了
新视角仍会拦截，以免 ZIP 看起来成功但重建必然失败。ZIP 根目录还包含“如何使用.txt”。

实时测试复用当前校正 RGB、YOLO Mask 和彩深对齐结果。首个有效帧执行 `REGISTER`，后续
帧执行 `TRACK`；初始化按钮会清除历史位姿并让下一帧重新注册。推理在独立单线程中运行，
待处理队列只保留最新帧，因此 GPU 推理变慢时不会堆积旧帧或阻塞 30 Hz RGB 预览。主预览
叠加米制网格的 3D 包围盒和 XYZ 轴，控制页显示推理模式、耗时及 4×4
`camera_from_object`。分析过期、彩深不同步、Mask/深度为空时会拒绝该帧；更换网格或
重新加载分割链路后必须重新注册。

### Gemini Max 无 CAD 调试 UI

Gemini Max 调试相机可直接使用独立入口：

```bash
./tool/object_model_builder/run_gemini_foundationpose_ui.sh
```

该 UI 从连接设备的 `/camera/color/camera_info` 保存出厂内参，并使用驱动的
Depth→RGB 硬件对齐；不会使用 Astra Pro 标定。选择任意 YOLO *分割* `.pt` 后，将物体
放在 ID0（左）和 ID1（右）两个 AprilTag 中间，只移动相机采集 16 个通过质量门的
RGB-D 视角。当前 Gemini Tag Map 使用 `DICT_APRILTAG_25h9`、75 mm Tag 边长，并把两个 Tag 黑框右下角间距
定义为 150 mm（Tag 左上角原点因此相距 150 mm）。然后可点击 Model-free 按钮执行
BundleSDF/Neural Object Field，也可先执行 TSDF 快速预览；生成的米制模型加载到
FoundationPose 实时测试后，即可在界面中看到 `camera_from_object` 的相对 XYZ 和相机到
物体原点距离。实时输出始终以相机光学坐标系为原点，且该入口不会执行机械臂动作。采集
时双 Tag 会优先联合求解，单个 Tag 也允许作为定位回退；物体与 Tag 仍需保持刚性不动，
相邻视角仍需有足够重叠。更详细说明见
[`GEMINI_FOUNDATIONPOSE_DEBUG.md`](GEMINI_FOUNDATIONPOSE_DEBUG.md)。
当前 `SV1301S_U3` 配置保持稳定的 RGB `640×480@30` 和对齐深度 `640×400@30`，
同时请求 IR `1280×800@30`；UI 图像栏会显示驱动实际返回的尺寸。不要在未重新查询设备
profile 前强行把 Depth 也改成 `1280×800`，此前该组合会使设备深度接口复位。

## Astra Pro 工作流

### 1. RGB-D 外参

Astra Pro 当前建模使用独立 UVC 彩色 `1280x720 MJPG`，以及匹配的 ROS 深度/IR
`1280x1024`（深度约 `7 FPS`、IR `30 FPS`）。深度和 IR 必须保持同一分辨率，不能只
提高 IR 而继续用 `640x480` 深度，否则 RGB-D 配准内参不匹配。设备通过 USB2 连接时，
高分辨率组合可能降低深度稳定性；如果 IR 或深度流启动失败，应把两路一起退回配置中的
`640x480@30` 模式。OrbbecSDK 还枚举到 SDK 彩色
`1280x800 UYVY/30Hz`，但这不是当前 UVC + ROS 默认组合；主机的 `/dev/video2` 没有
`1280x800` UVC 模式。如果切换到 SDK 彩色通道，必须确认两路能同时启动，并为新的实际
分辨率重新标定，不能直接复用 `1280x720` 内参。无论选择哪一档，都不能按宽高比例缩放
深度。先在 UI 的“RGB-D 标定”页启动相机，用同一张标定板在多个距离和角度采集至少
15 对 RGB/IR 图像，然后计算并检查 Stereo RMS。程序把 IR 视作深度光学相机，求得：

当前配置使用硬质 SINE IMAGE YE0102-A540 的连续棋盘格区域，而不是把整张综合测试卡
当作 ChArUco 或 AprilTag。程序使用中央楔形图上方完整横条，配置为 `19×3` 个内角点、
方格尺寸 `29 mm`，即物点为：

```yaml
rgbd_calibration_target:
  type: checkerboard
  model: SINE IMAGE YE0102-A540
  pattern_columns: 19
  pattern_rows: 3
  square_size_m: 0.029
```

这里的 `19×3` 是内角点数量，不是黑白方格数量。只使用中央分辨率楔形图上方的完整横条；
中央和下方区域被打断，不能混入同一个矩形棋盘模式。检测先缩放图像，再依次尝试高斯降噪、
CLAHE 局部对比度增强和原图，并用网格单应性残差拒绝跨越楔形图的假角点。29 mm 是徒手
测量值，正式使用前应使用卡尺在多个位置复测；尺寸误差会按比例影响平移尺度，但不改变
旋转求解。硬板应固定平整，并
避免红外反光。

当前普通棋盘格自动采集阈值为 RGB/IR 至少 `45` 个共同角点（完整检测为 `57`），
目标为 `20` 对。点击
“开始自动采集”后，程序仅在角点数量达标、时间同步合格且板面位置或角度相对上一张
发生足够变化时保存；板面静止时不会重复写入。达到目标对数后自动停止，再由用户点击
“计算 RGB-D 标定”。阈值和目标对数可在标定页直接调整。

```text
p_color = T_color_depth * p_depth
```

正式写入中央参数前，程序会生成带时间戳的备份。有效标定会写入：

```text
/home/throne/workspaces/arm/ros_ws/src/arm_vision_framework/config/calibration_parameters.yaml
```

### 2. RGB 校正和深度配准

建模时原始 RGB 先按内参畸变校正，Tag、YOLO、Mask 和 TSDF 都在校正后的 RGB 坐标中工作。深度则执行：

```text
深度像素 -> 深度相机反投影 -> T_color_depth -> 校正 RGB 投影 -> Z-buffer
```

低分辨率深度映射到高清 RGB 时，程序只对真实投影邻域做有限像素铺展并取最近 Z，不会把深度图拉伸成“看起来对齐”。若没有有效的深度内参或 `T_color_depth`，UI 会禁止建模采集。

深度像素的去畸变射线和外参投影系数会在标定加载时一次性缓存，后续帧只更新深度值和
Z-buffer。采集页以 RGB 时间戳约 `30 Hz` 刷新；Tag、YOLO 和彩深配准在单独工作线程中
执行，队列只保留最新帧，因此深度 `7 FPS` 或一次较慢的 YOLO 推理不会把旧任务堆积到 UI。

### 3. YOLO 和 Tag

在“无模型拍照”页选择用户提供的 `.pt`/`.pth` 权重，填写训练时的目标类别（例如 `bottle` 或 `can`），加载后确认 Mask 覆盖目标。Tag Map 使用现有 `tool/camera_calibration/config/tag_layout.yaml`，至少看到 ID100～102 中的一个即可计算绝对相机姿态，但建议每帧看到两个或三个。

采集门禁包括：RGB-D 标定有效、Tag 重投影 RMS 合格、YOLO Mask 有效、RGB/深度时间差合格、后台分析结果未过期、Mask 内有效深度覆盖率合格、物体相对 Tag 工作区未移动，以及相对上一帧有足够视角变化。默认深度覆盖率阈值为 `70%`；低于阈值会按透明、反光或分割错误风险拒绝。物体深度中心在工作区内的最大变化默认不得超过 `70 mm`。视角不变、深度为空、物体移动或 Tag 漂移都会拒绝保存。

### 4. TSDF 和导出

瓶子固定在 Tag 工作平面上，移动相机从正面、侧面、斜上方采集至少 12 个有效视角。透明/高反光瓶身会产生深度孔洞，正式建模前应先检查“对齐深度 / Mask”预览；必要时使用不改变几何的临时消光处理，并在最终颜色数据上重新确认 YOLO Mask。

TSDF 输出会清理退化面、重复面和小的孤立组件，然后把网格转换为：

```text
原点：物体底面中心
+Z：工作空间定义的向上方向（当前 Tag 纸面约定为 [0, 0, -1]）
单位：米
```

融合前会再次逐帧执行深度覆盖率和物体固定性检查，旧 ZIP 也不能绕过采集门禁。导出前按
`fusion.mesh_quality` 检查米制包围盒和封闭性；当前 `bottle` 配置要求尺寸位于
`[20, 20, 120] mm` 到 `[300, 300, 500] mm` 之间且网格 watertight。其他物体必须先按
真实尺寸修改这个配置，不能沿用瓶子阈值。

导出目录示例：

```text
/home/throne/workspaces/arm_data/meshes/bottle/
├── bottle.obj
├── bottle.ply
└── model_metadata.yaml
```

`model_metadata.yaml` 会保存扫描时的 `workspace_from_object`、网格尺寸、三角面数、单位和 FoundationPose 参数。FoundationPose 适配器使用 `mesh_scale_to_meters: 1.0`；BOP 或其他毫米模型则必须显式配置 `0.001`，不能让程序猜尺度。

## OAK-D Pro 兼容性

默认配置已绑定当前 `OAK-D-PRO-FF`（MXID `14442C10D141C5D600`）；Astra 后端只用于旧数据
回归。配置仍把采集后端抽象为 `astra_ros` 和 `oak_depthai`：

```yaml
camera:
  backend: oak_depthai
```

OAK-D Pro 的 DepthAI 后端使用设备工厂内参、双目外参和硬件对齐后的 depth-to-RGB 输出，随后只做与 RGB 一致的畸变校正。这样不需要重复做 Astra 的 RGB/IR 手工外参，但仍需检查设备输出的 `CameraInfo`、深度单位和对齐尺寸。需要安装：

```bash
source /home/throne/miniconda3/etc/profile.d/conda.sh
conda activate foundationpose
python -m pip install -U --prefer-binary 'depthai>=2.17,<3'
```

OAK-D Pro 是否包含 IMU 取决于具体型号和固件。IMU 可以提供短时角速度/姿态预测、时间同步诊断和运动质量检查，但不能替代：

- RGB/双目内参和设备内置双目外参；
- 相机与机械臂法兰的 `T_gripper_camera`；
- AprilTag 提供的工作空间绝对位姿。

后续接 OAK ROS 驱动时，上层只需提供现有的 RGB、aligned depth 和 CameraInfo 契约；不应把 DepthAI API 传入 FoundationPose 或机械臂业务层。

## 数据边界

```text
tool/object_model_builder/                 # 源码、配置、测试和 UI
/home/throne/workspaces/arm_data/
├── calibration_sessions/                  # RGB/IR 标定采集
├── model_sessions/                        # 物体多视角采集
├── meshes/                                # FoundationPose OBJ/PLY
└── third_party/FoundationPose-plus-plus/  # 外部算法源码和权重
```

没有有效 RGB-D 标定、Tag 位姿和 YOLO Mask 时，程序不允许导出正式模型。当前 FoundationPose++ 的 GPU/网络权重 smoke 已通过，但还没有真实瓶子和用户 YOLO 权重的端到端精度结论。
