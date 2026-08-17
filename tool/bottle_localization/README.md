# 无 CAD 瓶子定位工具

本工具复用两个已有工具的标定结果和实现：

- `tool/camera_calibration`：固定 AprilTag Map、RGB 内参、相机工作空间位姿和 RViz Tag 场景；
- `tool/object_model_builder`：相机输入、RGB-D 外参、原始深度到校正 RGB 的配准和 YOLO 实例分割。

它不需要 CAD。每个有效深度帧都会计算：

```text
固定 Tag Map -> T_workspace_camera
原始深度 -> 对齐到校正 RGB -> YOLO Mask 内三维点
三维点 -> 桌面上方筛选 -> 瓶子底面中心/几何中心
```

## 启动

先准备训练好的 Ultralytics 实例分割权重，然后运行：

```bash
cd /home/throne/workspaces/arm
./tool/bottle_localization/run_rviz.sh \
  --weights /absolute/path/to/best.pt \
  --target-class bottle
```

如果相机 ROS 驱动已经启动，加 `--no-driver`，避免重复启动：

```bash
./tool/bottle_localization/run_rviz.sh \
  --no-driver \
  --weights /absolute/path/to/best.pt
```

## 用尺子提供瓶子尺寸

不提供尺寸时，工具发布 YOLO Mask 可见表面的稳健中值位置，适合先确认数据链，但它不等于瓶子的真实轴心。

测量瓶身最大直径和整瓶高度后传入毫米值：

```bash
./tool/bottle_localization/run_rviz.sh \
  --weights /absolute/path/to/best.pt \
  --bottle-diameter-mm 68 \
  --bottle-height-mm 235
```

提供直径后，XY 使用已知半径的直立圆柱拟合；提供高度后，`bottle_pose` 的平移为底面中心上方一半瓶高。底面坐标始终单独发布，可直接与桌面尺量值比较。

已提供直径时，RViz 圆柱严格使用尺量直径，不再被 Mask 边缘散点放大。工具先保留 Mask 内最大的连续有效深度区域；圆柱拟合前再按相机方向构造物理种子，拟合后移除超出“半径 + 10 mm”以及“瓶高 + 15 mm”的深度点。容差可通过 `maximum_radial_excess_m` 和 `maximum_height_excess_m` 调整；边缘仍有明显漏点时，可继续增大 `mask_erosion_pixels`。

也可以把尺寸长期写入 `config/bottle_localization.yaml`：

```yaml
nominal_bottle_height_m: 0.235
nominal_bottle_diameter_m: 0.068
```

## RViz 内容与 ROS 输出

RViz 中显示固定 Tag、实时相机、YOLO Mask 深度点云、瓶子圆柱、底面黄点、瓶身轴和毫米坐标标签。主要话题：

| 话题 | 内容 |
|---|---|
| `/camera_calibration/camera_pose` | AprilTag 推算的相机工作空间位姿 |
| `/bottle_localization/bottle_pose` | 瓶子中心位姿 |
| `/bottle_localization/bottle_base_pose` | 桌面上的瓶子底面中心 |
| `/bottle_localization/object_cloud` | YOLO Mask 内且位于桌面上方的深度点云 |
| `/bottle_localization/annotated_image` | Tag、YOLO Mask 和位置投影叠加图 |
| `/bottle_localization/aligned_depth_preview` | 对齐深度，仅突出 Mask 区域 |
| `/bottle_localization/status` | JSON 状态、坐标、覆盖率和质量信息 |

查看数值：

```bash
rostopic echo /bottle_localization/status
rostopic echo /bottle_localization/bottle_base_pose
```

工作空间原点是 ID100 黑框左上角。数学坐标约定保持 `+X` 向右、`+Y` 向纸面下方、`+Z` 进入桌面；RViz 通过 `ruler_workspace_rviz` 只翻转显示视角，不改发布坐标。因此瓶子位于桌面上方时，原始工作空间 Z 通常为负值，底面 Z 为零。

## 有效结果门禁

只有以下条件全部满足才发布新的瓶子位置：

- RGB/深度时间差不超过建模工具配置的阈值；
- 至少看到配置数量的 ID100～102，且 Tag 重投影 RMS 合格；
- YOLO 返回目标类别实例 Mask；
- Mask 内有效深度覆盖率和桌面上方点数合格；
- RGB、Mask、对齐深度使用完全相同的校正像素坐标。

透明或高反光瓶身会让结构光深度产生孔洞。先查看 RViz 的对齐深度图和状态中的 `depth_coverage`；必要时使用不改变几何尺寸的消光处理，再用尺子进行坐标验收。

## 抓取规划（AnyGrasp）

本工具发布的 `/bottle_localization/object_cloud` 与 `/camera_calibration/camera_pose`
同时是 AnyGrasp 抓取规划节点的输入，见 `tool/grasp_planning/`。规划节点独立运行
（独立的 `anygraspenv` 环境），在 RViz 中叠加显示夹爪候选，不需要修改本工具。
