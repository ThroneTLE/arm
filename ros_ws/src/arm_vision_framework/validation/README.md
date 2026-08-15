# 验证脚本约定

后续的小型验证脚本放在本目录，不要直接修改运行时节点来做一次性实验。

每个脚本应满足：

- 默认 `dry_run`，不发送机械臂运动命令。
- 通过参数指定 `config/system_parameters.yaml` 和
  `config/calibration_parameters.yaml`，不要在脚本中复制矩阵。
- 输入、时间戳、坐标系、单位和算法版本必须写入输出结果。
- 输出保存到 `runs/<脚本名>/<时间戳>/`，不同测试轮次不得覆盖。
- 同时保存原始观测值和汇总指标，不能只保留截图。
- 模拟数据与实机数据必须使用不同的 `simulated` 标志。

建议的后续脚本：

```text
validate_tag_localization.py       Tag 多位置绝对定位误差
validate_hand_eye_consistency.py   T_base_target 跨姿态一致性
validate_rgb_depth_alignment.py    RGB-D 重投影误差
validate_foundationpose_pose.py    6D 位姿平移/旋转误差
validate_grasp_repeatability.py    重复抓取位置和成功率
```

框架级回归测试运行：

```bash
python3 -m unittest discover -s tests -v
```
