# AnyGrasp 抓取规划（复用 bottle_localization 的 RViz 链路）

把 AnyGrasp SDK 的 6-DoF 抓取检测接到现有的瓶子定位流水线上：bottle_localization
输出 YOLO Mask 内的深度点云与相机位姿，本工具把点云变换到相机系后调用 AnyGrasp，
再把候选抓取位姿变换回工作系，以夹爪 Marker 的形式发布到 RViz。

## 数据流

```text
/bottle_localization/object_cloud (工作系点云)
/camera_calibration/camera_pose    (workspace_from_camera)
        │
        ▼
点云 -> 相机系 -> AnyGrasp get_grasp（自上而下逼近约束 + 碰撞过滤 + NMS）
        │
        ▼
候选 -> 工作系 -> 宽度/分数/朝向过滤
        │
        ▼
/grasp_planning/grasps       MarkerArray（夹爪三件套，按分数着色）
/grasp_planning/best_grasp   PoseStamped（最优抓取，坐标系 X=逼近、Y=开合）
/grasp_planning/status       JSON 状态
```

## 许可证与权重（一次性人工步骤）

AnyGrasp SDK 需要许可证才能运行，权重随许可证邮件发放：

1. 用本机生成机器码（当前官方 SDK 的 `get_feature_id` 是权威方式，无需 license）：
   ```bash
   source /home/throne/miniconda3/etc/profile.d/conda.sh && conda activate anygraspenv
   cd /home/throne/workspaces/arm_data/third_party/anygrasp_sdk/grasp_detection
   python -c "from gsnet import get_feature_id; print(get_feature_id())"
   ```
   本机输出为 **`N55983334442629014371`**（N + 20 位数字）。
   注意：不要用旧仓库 `anygrasp/`（2023 版）里的 `license_checker -f`，它输出的
   19 位 id 与当前官方 SDK 的算法/位数都不同，会浪费一次申请。
2. 在 https://forms.gle/XVV3Eip8njTYJEBo6 （需要梯子）填写机器码
   `N55983334442629014371`，1～2 天收到邮件：
   - `license/`（licenseCfg.json + 三件套）→ 解压到 `anygrasp_sdk/grasp_detection/license/`
   - `checkpoint_detection.tar` → 放到 `anygrasp_sdk/grasp_detection/log/`
   - `checkpoint_tracking.tar` → 放到 `anygrasp_sdk/grasp_tracking/log/`（抓取跟踪，暂未使用）
3. license 到位后验证许可证：
   ```bash
   source /home/throne/miniconda3/etc/profile.d/conda.sh && conda activate anygraspenv
   cd /home/throne/workspaces/arm_data/third_party/anygrasp_sdk/grasp_detection
   python -c "from gsnet import check_license; check_license('license')"
   ```

机器码绑定机器硬件：许可证必须用最终运行机器的机器码申请（比赛机器需用比赛机器重新生成）。

## 环境搭建

```bash
cd /home/throne/workspaces/arm
./tool/grasp_planning/setup_anygrasp_env.sh anygraspenv /home/throne/anygrasp_env_setup.log
```

脚本按以下流程（CUDA 11.8 + PyTorch 2.5.1 + Python 3.9，与 SDK 预编译 .so 匹配）：

1. conda 环境 `anygraspenv`（python 3.9）；
2. openblas-devel（conda-forge 源）；
3. pytorch 2.5.1 + pytorch-cuda 11.8（TUNA pytorch 镜像 + 官方 nvidia 频道）；
4. conda 的 nvcc 11.8 工具链（本机没有系统 CUDA，用环境内编译器）；
5. MinkowskiEngine 从源码编译（`--blas=openblas`）；
6. graspnetAPI 本地安装（setup.py 已按官方经验修改 numpy/transforms3d/sklearn 依赖）；
7. AnyGrasp SDK 内的 pointnet2 编译安装；
8. torch / MinkowskiEngine / graspnetAPI / pointnet2 导入校验。

## 运行（两个终端）

终端 1：瓶子定位（沿用现有工具）：

```bash
cd /home/throne/workspaces/arm
./tool/bottle_localization/run_rviz.sh \
  --weights /absolute/path/to/best.pt \
  --bottle-diameter-mm 68 --bottle-height-mm 235
```

终端 2：AnyGrasp 抓取规划 + RViz：

```bash
cd /home/throne/workspaces/arm
./tool/grasp_planning/run_rviz.sh
```

RViz 中新增两个显示：`AnyGrasp candidates`（绿=高分、红=低分的夹爪模型，带逼近方向
线段与最优抓取文字标签）和 `Best grasp pose`（最优抓取坐标系）。

## Gemini 静态帧离线验证

无需 ROS、实机相机、FoundationPose 或手眼标定。默认复用
`tool/visual_grasp_pipeline/config/visual_grasp_pipeline.yaml` 中的 Gemini
`static_frame` 与 `yolo_model.pt` 路径，选择置信度最高的 `can` 实例：

```bash
cd /home/throne/workspaces/arm
./tool/grasp_planning/run_gemini_static_validation.sh
```

结果写入
`/home/throne/workspaces/arm_data/anygrasp_gemini_static_validation/`：YOLO 检测图、
清理后的实例 Mask、抓取投影图、带 region-steering 标记的点云和完整 JSON。追加
`--vis` 可打开 Open3D 三维交互窗口；用 `--instance 1` / `--instance 2` 验证另外两个
罐子，用 `--label banana` 等切换类别。

## 调参

所有参数集中在 `config/grasp_planning.yaml`：

- `max_gripper_width_m`：按真实二指夹爪最大开度填写（SDK 上限 0.1 m）；
- `top_down_only` + `approach_cone_deg`：只保留自上而下的抓取（工作系 +Z 指向桌面内，
  因此自上而下的逼近方向是 +Z）；
- `minimum_width_m` / `maximum_width_m`：按瓶身直径过滤（直径 68 mm 的瓶子建议
  0.05～0.08 m）；
- `minimum_score`：低于该分数的候选丢弃；
- `rate_hz`：AnyGrasp 在 4060 上单帧约 100～300 ms，深度 7 FPS，默认 2 Hz 已足够。

## 排查

- `rostopic echo /grasp_planning/status` 会给出 `planner_ready` 与 `load_error`；
  license/checkpoint 未就位时节点保持运行并每 10 秒重试，不会崩。
- SDK 的 `gsnet` 是 cp39 预编译扩展，必须在 `anygraspenv`（torch 2.5.1 + MinkowskiEngine）
  里运行，不能在 foundationpose 环境运行。
- 若 `create_detector` 返回 None：检查 `grasp_detection/license/` 是否解压正确、
  checkpoint 路径是否正确、以及 `./license_checker -c` 是否通过。
- 单元测试（不需要 SDK/许可证）：
  ```bash
  source /opt/ros/noetic/setup.bash
  /home/throne/miniconda3/envs/foundationpose/bin/python -m unittest \
    discover -s tool/grasp_planning/tests -v
  ```
