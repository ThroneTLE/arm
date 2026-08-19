# RGB-D 机械臂比赛标定流水线

`competition_pipeline` 是比赛现场的独立入口，包含两个互不覆盖的相机配置：

- `astra_validation`：当前 Astra Pro 验证相机，RGB 走 UVC，IR/Depth 走 ROS。
- `oak_competition`：比赛 OAK-D Pro，RGB、双 OV9282 和对齐 Depth 走 DepthAI。

RGB 是 AprilTag、手眼标定和最终定位的主相机。切换相机配置会自动使当前手眼矩阵失效，
并使用各自独立的手眼样本文件，避免把 Astra 的结果误用于 OAK-D Pro。

## 现场 UI

从干净 shell 启动，脚本会自动加载 `/opt/ros/noetic` 和 `/home/throne/astra_ws`：

```bash
cd /home/throne/workspaces/arm
./competition_pipeline/run_ui.sh
```

UI 左侧“相机配置”可切换 Astra 与 OAK。相机已连接时切换会先安全断开。按八个阶段操作：

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
4. **眼在手上**：机械臂停稳后输入当前 `T_base_tcp`（X/Y/Z 毫米，R/P/Y 度），采集
   至少 8 个姿态。程序由 RGB Tag PnP 求 `T_base_color_camera`，再求并写回
   `T_tcp_color_camera`。
5. **定位验证**：有合格的已登记 Tag 时使用视觉结果；Tag 不可见时，只有在手眼矩阵有效且
   TCP 时间戳新鲜时才使用 TCP 回退。画面会显示当前来源和质量。运行验证时采用多 Tag
   短时迟滞：双 Tag 偶发掉成单 Tag 时保持最近双 Tag 位姿 0.8 秒，RMS 偶发超限时最多保持
   最近有效位姿 0.6 秒，来源显示为 `tag_visual_held`。手眼采样不使用该保持逻辑。
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

目前没有 OAK-D Pro 实物，因此软件接口、JSON 导入和离屏 UI 已验证，但相机枚举、投影器
电流、实时帧率和实机 EEPROM 导出必须到场后再做一次硬件验收。

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
去重后 Mask，以及 AprilTag 优先/TCP 回退得到的 `T_base_camera`，将每个 Mask 内的有效
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

`check` 只检查配置和路径字段，不会自动连接硬件。RGB 图像分辨率必须与
`color_intrinsics_file` 一致。机械臂适配器实现 `interfaces.RobotPoseProvider` 和
`interfaces.RobotController`，夹爪实现 `GripperController` 即可接入运行时；
`SafeRobotController` 默认 `dry_run: true`
且禁止运动，必须在现场完成独立安全联锁验收后再打开。

UR5e+Robotiq 的开源仿真、上游复现、比赛 bridge 和闭环执行说明见 `sim/README.md`。

## 运行测试

```bash
python3 -m unittest discover -s competition_pipeline/tests -v
python3 -m unittest discover -s tool/camera_calibration/tests -v
python3 -m unittest discover -s tool/object_model_builder/tests -v
```
