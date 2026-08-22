# RGB-D 机械臂比赛标定流水线

> # ⛔ 本文含已被推翻的结论 —— 阅读前必看
>
> **状态**：部分作废
> **推翻日期**：2026-08-22（依据：控制器原始日志 + 出厂配置备份）
> **权威文档**：[docs/纳博特C1102-现场真相-必读.md](../docs/纳博特C1102-现场真相-必读.md)
> **原始证据**：[根因证据-控制器日志摘录.txt](../docs/现场备份-20260822/根因证据-控制器日志摘录.txt)
>
> 本文写于现场排障过程中，视觉/标定部分（第 01~07 步、坐标契约、手眼、分割、RViz）全部
> 仍然有效，但涉及**机械臂运动通道**与**控制器安全使能**的两处结论建立在错误前提上：
> 当时以为“回复位点就能满足安全使能条件”、“运动报文没有公开、MOVJ/MOVL 还没法实现”。
> 实际上安全闸门坏在出厂配置的 `deviation=null`，与机器人在不在复位点完全无关；
> 而 0x4501（MOVJ）/0x4502（MOVL）早已实现并在示教模式下实测通过。
> 保留这些原文是为了让后来的 agent/人知道**哪条路已经试过并且是错的**，
> 不要把被划掉的段落当作可执行方案。
>
> | 本文的旧结论 | 现已确证的事实 |
> |---|---|
> | 放置撤离后“自动回复位点”，可保证下一轮满足控制器安全使能起始条件（L291 附近） | `global.json` 的 `RemoteIO[0].posReset` 里 `safeEnable=true` 而逐轴 `deviation=null`，闸门判据坏掉，**远程模式下与位姿无关，恒判“不在安全位置附近”**；而且“回复位点”本身走的就是 0x3007，每被拒绝一次控制器就 stop→JobClear→使能=0→PowerOff，**掉一次伺服** |
> | 手册没有公开运动 socket 报文，MOVJ/MOVL 适配器必须等取得官方协议/SDK 后再实现（L300 附近） | MOVJ=0x4501 / MOVL=0x4502 **已实现并实测可用**，前提是报文必须带 `acc` 和 `dec`（22.07 公开文档漏了这两个字段），解析顺序 robot→vel→acc→dec→coord→pos[7]；补齐后实测 ±5mm/-3mm 精确到位 |
> | `inexbot_modbus.py` 是“正式协议代码”，运动只能靠 Modbus 作业文件 | 运动主线是 **6001** 端口的 JSON 实时指令通道（0x4501/0x4502/0x3003 等）；7000 是**只读状态口**（0x9512/0x9513），从不下发运动；Modbus 只用于 IO/状态旁路，不是运动正式通道 |

## ⚠️ 上机前必读

先读 [docs/纳博特C1102-现场真相-必读.md](../docs/纳博特C1102-现场真相-必读.md)，再碰机械臂。

### ✅ 唯一已实测的可用路径：全程留在示教模式

示教器**不要切远程模式**。0x3003 / 0x3007 / 0x3002 / 0x4501 / 0x4502 在控制器内部走
同一入口 `startRobotJobTask(..., safepos=1, call=moveToPos)`：远程模式下被坏掉的复位点
安全闸门拒绝，示教模式下正常执行。**实测远程模式 15 次全拒、示教模式 40+ 次全成功，
零例外。** 示教模式下 0x2311 上使能后 `status=3` 会一直保持，**不需要按住 deadman**。

闸门为什么坏：出厂配置 `RemoteIO[0].posReset` 里 `safeEnable=true` 但逐轴 `deviation=null`
（`deviationSync` 也是 `null`），判据失效 —— 远程模式下**无论机器人停在哪里
（逐轴只差 1e-5° 也一样）都恒判“机器人1不在安全位置附近”**，随即
stop → JobClear → 脉冲使能=0 → Deadan_End → PowerOff，每拒绝一次掉一次伺服。

### ⚠️ 未验证的候选修复：把 `deviation` 填成 `1.0°`

路径：示教器 → 复位点设置 → 逐轴偏差。

**这是从配置推出来的假说，现场从未验证过** —— 我们只确证了 `deviation=null` 是闸门
失效的原因，没有验证过填上数值就能让远程模式恢复可用（同一结构下另外三台未启用的
机器人配置里 `deviation` 是 `[1,1,1,1,1,1,1]`，这是数值来源，不是验证结果）。
真要试，先在**空载、低速、人手离开工作范围**的条件下单独验证闸门行为，
不要把它当成上机前的必做步骤。**比赛当天请直接用上面那条已实测的示教模式路径。**

其它上机纪律（已落地在代码里，改代码时不要退回去）：

- 每段运动前都要 0x2002 查状态，`!=3` 就补 0x2311 —— 被拒绝一次就掉一次电，不能只查一次。
- 运动后必须等 0x3D03 `status=2` 才算真的动了（`_await_motion_ack`）；“位姿没变”绝不能当成功。
- 报警帧的真实文本在 0x2B03 的 **`data`** 键，不是 `message`。
- 下发前三道闸门：位移 ≤400 mm、`|A/B/C| ≤7 rad`（挡“度当弧度”）、姿态变化 ≤20°。
- `realPosACS` 本来就是**度**，不要再乘 57.29578；`realPosUCS/PCS` 的 A/B/C 才是弧度。
  历史上 teleport 面板的“A/B/C 用弧度”复选框把回读的度当弧度发出，
  16:58:19 发出 (1.9406, 1.1961, 2.2181) rad —— 与真实姿态差 119.6°，6 秒后六轴 0F15
  故障、控制器 PowerOff、机械臂坠落。该复选框已永久删除，任何形式都不要恢复。
- `coord=3` 是**用户坐标系**（本机实测），整条流水线以用户坐标系 1 为准。
- 夹爪 0x3601 走另一条码路，不受使能和闸门影响 —— **“夹爪能动”推不出“机器人能动”**。

`competition_pipeline` 是比赛现场的独立入口。默认固定当前 OAK-D-PRO-FF，旧相机 profile
只作为显式回归选项保留：

- `astra_validation`：当前 Astra Pro 验证相机，RGB 走 UVC，IR/Depth 走 ROS。
- `oak_competition`：当前默认，绑定 MXID `14442C10D141C5D600`，RGB、双 OV9282 和对齐 Depth 走 DepthAI。

RGB 是手眼标定和物体定位的主相机；当前手眼默认使用张正友棋盘格，AprilTag 保留为兼容
流程和单独诊断，不参与比赛运行时相机定位。切换相机配置会自动使当前手眼矩阵失效，
并使用各自独立的手眼样本文件，避免把 Astra 的结果误用于 OAK-D Pro。

## 现场 UI

从干净 shell 启动，脚本会自动加载 `/opt/ros/noetic`、`/home/throne/astra_ws`
和 `/home/throne/orbbec_ws`（Astra / Gemini 两种相机后端都能直接使用）：

```bash
cd /home/throne/workspaces/arm
./competition_pipeline/run_ui.sh
```

UI 左侧“相机配置”可切换 Astra 与 OAK。相机已连接时切换会先安全断开。按九个阶段操作：

Astra 还可在侧栏切换 Depth/IR 模式：默认 `640×480 @ 30` 用于流畅验证，
`1280×1024 @ 7` 用于需要更多深度细节的静态检查。切换不会改变 1280×720 RGB，
也不会使 RGB 内参或手眼结果失效；正在采集的 RGB/IR 图像对会清空。

1. **RGB 内参**：打开 `tool/camera_calibration/targets/charuco_intrinsics_A4.pdf`，采集至少
   20 个不同距离和倾角的 RGB ChArUco 姿态，计算通过 RMS 质量门后写入当前 profile 的
   `color_intrinsics_file`。OAK 通常直接由官方 JSON 生成该文件，不必重复采集。
2. **Astra RGB-IR 标定**：以 IR 帧作为同步基准。室内默认开启 IR 投影器；太阳光下可用
   侧栏开关关闭点阵，但必须确认 IR 图像未饱和且 ChArUco 清晰。采集至少 12 对同步
   RGB/IR ChArUco 图像，求出 IR 内参和
   `T_color_depth`。RMS、焦距、主点、基线和相对旋转均有物理质量门。
   求解时会按跨相机重投影误差自动剔除离群图像对，最多剔除有效样本的 40%，且始终保留
   至少 12 对；若稳健求解后 RMS 仍超限，程序继续拒绝该结果。选择 OAK-D Pro 时，本阶段
   会替换成官方 EEPROM 标定页面，不会重复执行 Astra 的 RGB/IR 标定。
3. **Tag 地图**：录入任意数量的 AprilTag ID，以及每个黑框右下角 `BR` 在机器人基座的
   XYZ（毫米）。表格中的 R/P/Y 是 Tag 平面朝向，默认值可一次填充到所有行。
4. **眼在手上**：页面顶部先连接“TCP 通信验证”（NexBot 官方 7000 端口协议，直读
   控制器当前 `T_base_tcp`，不再手抄示教器；IP / 7000 端口 / Robot 号保存在
   `controller.nexbot_tcp`）。连接成功且读数与示教器一致后再采样：机械臂停稳，
   输入当前 `T_base_tcp`（X/Y/Z 毫米，**A/B/C 度**——控制器原生姿态，内旋 X'Y'Z'，
   `R=Rx(A)Ry(B)Rz(C)`，**不是 RPY**；可勾选“读取值自动填入”），采集
   至少 8 个姿态。当前默认使用“张正友棋盘格”：固定棋盘无需录入其基座坐标，程序使用
   多姿态 `T_base_tcp` 与每帧 `T_camera_board` 的 OpenCV `calibrateHandEye` 求
   `T_tcp_color_camera`。也可以切回“已建图 AprilTag”流程，由 RGB Tag PnP 求
   `T_base_color_camera`，再求并写回 `T_tcp_color_camera`。
5. **定位验证**：比赛主线只接受时间戳新鲜的控制器 TCP 回读，并计算
   `T_base_camera = T_base_tcp × T_tcp_color_camera`。AprilTag 只用于手眼采样和单独诊断；
   `localization.use_apriltag_runtime` 默认 `false`，不要在抓取时改成视觉优先。
6. **分割验证**：选择 Ultralytics 实例分割权重、目标类别、置信度、IOU、推理尺寸和设备。
   模型在独立线程中常驻，只处理最新 RGB 帧；UI 叠加全部实例 Mask，并显示类别、置信度、
   Mask 面积、推理耗时和连续合格帧。连续通过质量门后由操作员确认，程序保存模型 SHA256、
   相机 profile 和图像尺寸。换模型或换相机后必须重新验证。
7. **抓取规划**：`planning.py` 将合格物体点云的接地包围框转成确定性俯视抓取。抓取局部
   `+Z` 是接近方向，因此在机器人基座中必须指向 `-Z`；程序选择较短水平尺寸作为夹爪
   闭合轴，并检查开口宽度，再生成预抓取、抓取、抬升、预放置、放置和撤离 TCP 位姿。
   这条确定性策略仍是默认主线。`FallbackGraspPlanner` 只有在主线拒绝物体且
   `grasp_planning.fallback.enabled: true` 时才惰性加载 AnyGrasp；SDK、CUDA、许可证或
   checkpoint 不可用时，主线不会被启动阶段阻塞。
8. **抓取执行**：`execution.py` 已实现 open→pregrasp→approach→close→lift→place→release→
   retreat 状态机，支持取消/stop、结构化结果和运动分段。若有物体追踪器，抬升高度与放置
   XY 误差是强制闭环门；只完成机械臂轨迹不会被误报为抓取成功。真实适配器未验收前仍
   保持 fail-closed、dry-run 和禁止运动。
9. **控制器/TCP 测试**：可现场填写 IP、Port 和 Unit ID，保存后只读连接。寄存器映射为空
   时不会猜地址；按官方表配置后显示 Servo、急停、运动状态、Axis1..7、TCP XYZ/ABC、
   Tool ID、User ID、shape、两个保留字段、报警码/文本及原始值。只有机械臂明确停稳、
   急停关闭、无活动报警且这些字段全部有效时，才能把当前构型保存为 `P9000` 安全 MOVJ 点。

第 09 页验收后可把控制器映射和安全点一次导入正式 ROS 参数（自动备份旧文件）：

```bash
source ros_ws/devel/setup.bash
rosrun arm_vision_framework calibration_tool.py import-competition-controller \
  --input competition_pipeline/config/competition.yaml
```

没有连接相机时，UI 仍可编辑 Tag 地图、查看配置和离屏检查页面；点击“连接深度相机”才
会启动当前 profile 对应的后端。不要把 `tool/object_model_builder/output` 中的旧
RGB-D 结果直接复制过来，已知文件的 RMS 约 `77944 px`，质量门会拒绝它。

## IR 投影器开关

侧栏的“启用 IR 投影器 / 点阵”可在相机运行时切换 Astra 激光点阵或 OAK 点阵。室内获取
Depth 通常保持开启；太阳光含有很强的近红外，室外重新采集 RGB-IR ChArUco 标定图时可
关闭投影器，使用均匀的环境红外照明。避免镜头直对太阳，并检查 IR 画面没有大面积饱和；
几何标定只需要清晰 IR 角点，不依赖结构光点阵。重新做 Depth/点云验证时再打开投影器。

## OAK-D Pro 标定与驱动

OAK-D Pro 出厂 EEPROM 已包含 RGB/左右相机内参、畸变、双目外参和基线，DepthAI 后端会
直接输出对齐到 RGB 的 Depth。比赛现场有三种入口：

比赛推荐 `1920×1080 @ 10 FPS` RGB/对齐 Depth、OV9282 `800p`、扩展视差开启、Subpixel
关闭。我们是停稳后拍两张而不是实时跟踪，10 FPS 足以在服务调用时取得新同步帧；1080P
兼顾物体分割细节、USB/CPU 负载和 RGB/Depth 同尺寸。手册给出的 800P 标准近距约 70 cm，
扩展视差约 35 cm，最终接近抓取位不可继续依赖深度，必须在观察高度完成定位。

1. 点击“导入 OAK 官方标定 JSON”，选择 Luxonis `calibrate.py` 的输出；UI 会保留原始
   EEPROM JSON，并生成现有定位代码可直接读取的 `oak_color_intrinsics.yaml`。
2. 相机未被实时预览占用时，点击“从已连接 OAK 导出 EEPROM”，直接读取设备当前标定。
3. 需要重新标定时，核对标定板实际方格/Marker 尺寸后启动 Luxonis 官方工具。官方程序
   求解成功会写入设备用户 EEPROM，因此供电和尺寸必须可靠。

当前已安装：系统 Python 的 DepthAI 2.30、`/etc/udev/rules.d/80-movidius.rules`，以及独立
官方标定环境 `/home/throne/miniconda3/envs/oak-calibration`。官方工具位于仓库外：
`/home/throne/workspaces/arm_data/third_party/oak_calibration_tool/calibrate.py`。新机器可运行：

```bash
./competition_pipeline/install_oak_support.sh
```

当前 OAK-D-PRO-FF 已完成枚举、EEPROM 导入、投影器启动、1920×1080 RGB-D 同步和 IMU
发布验证。机械臂安装后仍需重新做手眼标定。正式 ROS 参数的一键导入入口是：

```bash
rosrun arm_vision_framework calibration_tool.py \
  import-oak-eeprom --input /absolute/path/to/oak_calibration.json \
  --color-width 1920 --color-height 1080
```

该入口只解析官方 EEPROM JSON 并生成 ROS 参数；真正 Flash 写入仍由官方 DepthAI/Luxonis
标定工具在连接实物后完成。

OAK-D Pro 的 IMU 由正式 ROS 节点发布到 `/camera/imu`。`oak_imu` 是独立原始传感器帧；
当前 `camera_from_imu: null` 表示轴向/外参未知，数据只允许用于时间同步、静止抖动和安装
诊断。ROS RGB 光学坐标固定为 `+X` 向图像右、`+Y` 向图像下、`+Z` 沿镜头向前；机械臂
工具 `+Z` 沿工具伸出方向。两者看起来可能近似同向，但实际关系始终由
`T_tcp_color_camera` 手眼标定给出，不能靠 IMU 或肉眼假设。

## 分割模型验证

比赛 UI 复用 `tool/object_model_builder/yolo_segmenter.py`，也就是
`tool/bottle_localization` 使用的实例 Mask 接口，不复制第二套推理实现。当前示例权重为：

```text
/home/throne/workspaces/Myolotrain/苹果香蕉_yolov8n-seg_best_20260816.pt
```

它对应 `Myolotrain/local_infer.py` 的本地参数：`conf=0.25`、`iou=0.45`、`imgsz=640`。
配置中的 `device: auto` 会在 CUDA 可用时选择 GPU 0，否则退回 CPU。实际比赛物品确定后，
在第 06 步选择新 `.pt`，填写需要验收的类别，再从不同距离、位置、遮挡和背景连续观察；
示例苹果/香蕉模型只用于验证工作流，不能作为正式比赛模型。

当前质量门只自动检查模型确实是实例分割模型、目标类别、置信度、Mask 尺寸和面积范围；
边界是否贴合、是否漏检多实例、是否把背景并入 Mask 仍需操作员观察画面后点击确认。

比赛配置默认打开两级实例去重：Ultralytics 的跨类别 NMS（`agnostic_nms`），以及 NMS
之后的 Mask 空间去重。第二级同时检查 Mask IoU、小 Mask 被包含率和按物体尺寸归一化的
中心距离；置信度差异明显时保留高置信度结果，置信度接近时保留面积更完整的 Mask。UI 会
显示“模型实例数、合并数量和最终实例数”，相邻但不重叠的物体不会仅因中心较近而被合并。
为防止零置信度导致候选爆炸，配置拒绝小于 `0.05` 的推理置信度，并用
`maximum_detections: 50` 限制单帧候选上限；置信度近似范围最多允许 `0.25`。Mask 面积或
置信度不通过质量门的实例在绘制前就会被剔除，不会再覆盖整张预览。

## RViz 分割定位与规划可视化

第 07 步的“打开 RViz 点云验证”不会再次打开相机。它直接消费比赛 UI 已同步的 RGB、Depth、
去重后 Mask，以及 TCP+手眼得到的 `T_base_camera`，将每个 Mask 内的有效
Depth 反投影到机器人基座坐标系。点云会经过边界腐蚀、最大深度连通域、深度覆盖率、工作
空间和最小点数质量门。

主要 ROS 话题：

- `/competition_pipeline/object_cloud`：按实例着色的分割物体点云。
- `/competition_pipeline/object_markers`、`object_poses`：物体边界、中心和坐标。
- `/competition_pipeline/camera_pose`、`tag_markers`：相机定位和固定 Tag 地图。
- `/competition_pipeline/camera/image`、`camera/camera_info`：分割相机画面和内参，供
  RViz Camera 视角使用；同时显示相机光学坐标轴和视锥。
- `/competition_pipeline/grasp_candidates`：预留的候选抓取 `PoseArray`。
- `/competition_pipeline/planned_path`：预留的 TCP 规划轨迹 `nav_msgs/Path`。

RViz 定位相机使用独立 TF 名 `competition_camera_color_optical_frame`，不要改回相机驱动的
`camera_color_optical_frame`。后者通常已经挂在 `camera_link` 下；复用同名子坐标系会形成
TF 双父节点，让坐标轴和 Camera 视图在正常/Error 之间闪烁。发布器会以 10 Hz 保活最后一次
有效定位 TF；定位失效时仍会清空物体点云并在状态话题报告错误，不会保留旧物体结果。
当 Astra 的 Depth 分辨率与 RGB-IR 标定文件不同时，运行时使用当前 ROS `CameraInfo` 的
Depth 内参，并继续使用刚性不变的 `T_color_depth` 外参完成三维重投影；不会把高分辨率内参
直接套到 640×480 图像上。正式比赛前仍建议在最终选定模式下重新检查一次 Depth→RGB 对齐。

抓取规划器可调用 `CompetitionRvizVisualizer.publish_grasp_candidates()` 和
`publish_planned_path()`，不需要修改 RViz 配置。UI 的 RViz 页面仍只做输入和候选可视化，
不会直接发送机械臂运动命令；正式执行由显式接入的安全控制器触发。Astra 的 RGB-D 文件若嵌入的 RGB 内参与当前 RGB 内参
不一致，发布器会按当前内参重投影并在 UI 显示警告，现场应重新执行第 02 步确认外参。

## 坐标和配置契约

- 机器人基座：`+X` 向前、`+Y` 向左、`+Z` 向上。
- Tag 原点：黑框右下角 `BR`；局部 `+X` 沿右侧边 `BR → TR` 向内，局部 `+Y`
  沿下侧边 `BR → BL` 向内，`+Z = +X × +Y` 并垂直 Tag 平面。标定板水平且正面朝上时，
  `+Z` 就是机器人基座的竖直向上。画面中红色箭头为 `+X`、绿色箭头为 `+Y`，蓝色圆点
  符号和 `+Z UP` 表示正法向。
- ChArUco/OpenCV 角点顺序：`TL, TR, BR, BL`。
- `T_base_tcp` 的输入单位为毫米/度，内部统一使用米制 `4x4` 齐次矩阵。
- 手眼字段固定为 `hand_eye.tcp_from_color_camera`，含义是
  `p_tcp = T_tcp_color_camera * p_color`。

### 张正友棋盘格手眼标定

`hand_eye.calibration_target` 同时兼容原有 `apriltag_map` 和 `checkerboard`。当前流水线默认
使用棋盘格，标定板为黑白格合计 **12 × 9**、单格 **25 mm**，因此印刷网格尺寸为
**300 × 225 mm**，OpenCV `findChessboardCorners` 的 `patternSize` 自动为
**`[11, 8]` 个内角点**（共 88 点）。若现场换成其他棋盘，请在“眼在手上”页填写长边/短边的
黑白格总数和单格尺寸；这里数的是该方向全部黑白格子，不是只数黑格，未填写时棋盘实时预览
会明确显示“待设置”，不会误按 AprilTag 或虚构的角点数采样。

棋盘格路径使用固定外部靶标的标准手眼方程，不需要输入 `T_base_board`：

```text
T_base_board = T_base_tcp × T_tcp_color_camera × T_camera_board
```

程序对所有采样同时估计未知的 `T_base_board` 与 `T_tcp_color_camera`，并用每帧反推出的
`T_base_board` 一致性剔除离群样本。采样要覆盖明显的姿态变化（默认 TCP 位移跨度至少
30 mm、旋转跨度至少 15°）；不要只在一个小范围平移相机。

纯黑白棋盘存在 180° 朝向二义性。比赛前请在棋盘一个角贴上非棋盘色的小标签/写上“TOP”，
全程保持同一朝向；采样画面中不要把棋盘旋转到难以分辨正反/上下的姿态。切换靶标、改变
板宽高或方格尺寸会自动使手眼矩阵失效，并归档旧样本，AprilTag 样本与棋盘样本绝不会混用。

所有现场参数集中在 [`config/competition.yaml`](config/competition.yaml)。修改 Tag 地图会
自动使旧手眼矩阵失效；样本文件还带 Tag 地图哈希，地图改变后不能误用旧样本。配置保存
前会在 `config/backups/` 留一份备份。

## AnyGrasp 备选抓取

比赛配置默认保持 `deterministic_top_down` 主线，并预留如下备选入口：

```yaml
grasp_planning:
  backend: deterministic_top_down
  fallback:
    enabled: true
    backend: anygrasp
```

代码调用 `planner_from_config(config)` 得到组合规划器。主线成功时不会导入 `gsnet`；主线
失败时，调用方需要把分割点云对应的 `base_from_camera` 传给
`planner.target_from_object(cloud, base_from_camera=...)`。AnyGrasp 候选会转换为比赛坐标
约定并重新检查分数、夹爪宽度和自上而下方向。许可证和权重仍放在仓库外，配置中的路径
按现场机器修改；不要把 license ZIP 或 checkpoint 提交到 Git。

## CLI 辅助命令

CLI 用于自动化和无 UI 的验收，不再提供左右双目命令：

```bash
python3 -m competition_pipeline.cli check
python3 -m competition_pipeline.cli tag-set --id 100 --bottom-right-mm 520 240 0
python3 -m competition_pipeline.cli tag-list
python3 -m competition_pipeline.cli hand-eye-add \
  --image data/hand_eye/001.png \
  --tcp-xyz-mm 430 120 680 --tcp-rpy-deg 178 -3 12
python3 -m competition_pipeline.cli hand-eye-solve
python3 -m competition_pipeline.cli localize-image --image data/check.png
```

无 UI 时切换到该棋盘格目标（命令会归档并重建当前相机 profile 的样本会话）：

```bash
python3 -m competition_pipeline.cli hand-eye-target \
  --type checkerboard \
  --board-width-mm 300 --board-height-mm 225 --square-size-mm 25 \
  --squares-x 12 --squares-y 9
```

`check` 只检查配置和路径字段，不会自动连接硬件。RGB 图像分辨率必须与
`color_intrinsics_file` 一致。机械臂适配器实现 `interfaces.RobotPoseProvider` 和
`interfaces.RobotController`，夹爪实现 `GripperController` 即可接入运行时；
`SafeRobotController` 默认 `dry_run: true`
且禁止运动，必须在现场完成独立安全联锁验收后再打开。

## 柠檬视觉抓取演示

“控制器/TCP 测试”页的视觉抓取面板使用 U 盘备份中的新 YOLO 权重和
柠檬 CAD，坐标只使用眼在手上链路：

```text
T_user1_object = T_user1_tcp × T_tcp_color_camera × T_camera_object
```

操作顺序：

1. 连接 OAK，确认实体急停有效，勾选运动授权；
2. 把柠檬放在用户坐标系原点，点“回复位点并识别柠檬”；
   （这一步的“回复位点”走 0x3007，是整条序列里最危险的一句：远程模式下必然被坏掉的
   复位点闸门拒绝并掉伺服。点它之前先确认已完成顶部“上机前必读”的两件事。）
3. 程序只计算并显示物体、抓取 TCP、放置 TCP 坐标，此时不执行抓取；
4. 原点误差、YOLO 置信度、Depth 覆盖率和工作空间校验全部通过后，
   点“确认执行视觉抓取”并在弹窗再次确认；
5. 机器人抓取后将抓取坐标沿用户系 `Y-` 平移 50 mm 作为放置点，
   放置撤离后自动回复位点。

> ~~放置撤离后自动回复位点，保证下一轮满足控制器“安全使能”的起始条件。~~
> ❌ **已推翻**：回复位点**不能**保证任何“安全使能”起始条件。出厂 `global.json` 的
> `RemoteIO[0].posReset` 中 `safeEnable=true` 而逐轴 `deviation=null`，闸门判据坏掉，
> 远程模式下与机器人实际位姿**完全无关**：实测机器人精确停在复位点、逐轴只差 1e-5°
> 也照样判“机器人1不在安全位置附近”。更糟的是“回复位点”本身走的就是 0x3007，
> 它是每条抓取序列的第一句，在远程模式下**必然掉使能**（拒绝后控制器执行
> stop → JobClear → 脉冲使能=0 → Deadan_End → PowerOff）。
> 正确做法见顶部“上机前必读”：**全程留在示教模式**（唯一已实测路径）；
> 使能靠“每段运动前 0x2002 查状态、`!=3` 就 0x2311”来保证，不靠回复位点。
> 证据：[根因证据-控制器日志摘录.txt](../docs/现场备份-20260822/根因证据-控制器日志摘录.txt)、
> `docs/现场备份-20260822/SystemBackup08-22-22-13/configFile-21.05.23-4.1.1-202608222213/global.json`。
> 另外注意 3x101=1（复位点判定）与 ioControl 的闸门是两条不同代码路径，
> 3x101=1 时闸门照样拒绝，**不能拿 3x101 当“能过闸门”的依据**。

视觉计划默认 60 s 过期，且坐标框被人工修改后必须重新识别。所有参数在
`config/competition.yaml` 的 `visual_grasp_demo` 中。

UR5e+Robotiq 的开源仿真、上游复现、比赛 bridge 和闭环执行说明见 `sim/README.md`。

真实控制器的基础通信边界见
[`controller_protocol.md`](controller_protocol.md)。

> ~~当前仅实现标准 Modbus-TCP 传输和配置化 IO 读写；手册没有公开运动 socket 报文，
> 因此 MOVJ/MOVL 适配器必须等现场取得官方协议/SDK 后再实现。正式协议代码在
> `ros_ws/src/arm_vision_framework/.../adapters/inexbot_modbus.py`，本目录只保留离线测试
> 入口。~~
> ❌ **已推翻**（两处都错）：
>
> 1. **MOVJ/MOVL 早已实现并实测可用**，不需要等任何官方 SDK。运动走 **6001** 端口的 JSON 实时指令通道
>    （7000 是只读状态口，从不下发运动）：`0x4501` = MOVJ（关节插补）、`0x4502` = MOVL（直线插补）。以前判定“不支持 / 一律参数错误”，
>    真实原因是**报文必须带 `acc` 和 `dec`** —— 22.07 版公开文档漏印了这两个字段。
>    正确解析顺序是 `robot → vel → acc → dec → coord → pos[7]`；补齐后实测
>    ±5 mm / -3 mm 精确到位。`coord=3` 是用户坐标系。
> 2. **`inexbot_modbus.py` 不是“正式协议代码”**，Modbus 也不是运动通道。Modbus 只承担
>    IO / 状态旁路读写；`0x3003 / 0x3007 / 0x3002 / 0x4501 / 0x4502` 在控制器内部走同一
>    入口 `startRobotJobTask(..., safepos=1, call=moveToPos)`，它们在远程模式下被坏掉的
>    复位点安全闸门拒绝、在示教模式下正常执行（实测远程 15 次全拒、示教 40+ 次全成功）。
>    “运动只能靠 Modbus 作业文件（45/19/29/61/71）”这条结论同样作废。
>
> 上机前置条件（留在示教模式；`deviation=1.0°` 只是未验证的候选修复）见本文顶部“上机前必读”，
> 完整说明见 [docs/纳博特C1102-现场真相-必读.md](../docs/纳博特C1102-现场真相-必读.md)。

本目录只保留离线测试入口，真机运行时的控制器适配器在
`ros_ws/src/arm_vision_framework/.../adapters/` 下。

## 运行测试

```bash
python3 -m unittest discover -s competition_pipeline/tests -v
python3 -m unittest discover -s tool/camera_calibration/tests -v
python3 -m unittest discover -s tool/object_model_builder/tests -v
```
