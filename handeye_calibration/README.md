# ER / iNeXBot 手眼标定

Ubuntu 环境已验证：`/home/user/.venvs/handeye-003/bin/python`、OpenCV 4.5.4、`cv2.calibrateHandEye`、ArUco 与 YAML 可用。

本工程不会发送机械臂运动指令。采集时由现场人员/已审核程序把机械臂移动到安全位姿；代码仅统一坐标、读取相机坐标文件并求解。

## 坐标约定

- `base_T_tool`：机器人基座到 TCP/法兰的齐次变换。
- `target_T_camera`：标定板到相机的齐次变换，来自相机标定、PnP 或外部视觉程序。
- `camera_on_tool` 输出 `tool_T_camera`；`camera_fixed` 输出 `base_T_camera`。
- 所有内部长度均为 **m**，旋转为右手系；四元数顺序为 **x,y,z,w**。

ER 默认输入 `x,y,z,rx,ry,rz`（mm、RPY 度）；iNeXBot 默认输入 `x,y,z,qx,qy,qz,qw`（m、XYWZ）。两份默认配置均为已验证的末端相机模式；固定相机需按实际标定板与 TCP 的刚性安装关系另行验证后才启用。如控制器定义不同，只改对应 YAML，不要共用两类机器人配置。

## 使用

```bash
cd /mnt/f/SEU/003/handeye_calibration
/home/user/.venvs/handeye-003/bin/python -m handeye.cli capture --robot er --robot-pose inputs/er_pose.json --camera-pose inputs/target_to_camera.json --samples data/er.jsonl
/home/user/.venvs/handeye-003/bin/python -m handeye.cli solve --config configs/er_camera_on_tool.yaml --samples data/er.jsonl --output results/er.yaml
/home/user/.venvs/handeye-003/bin/python -m handeye.cli test
```

每台机械臂至少采集 15 个姿态，包含大于 20 度的多方向旋转。相机/标定板必须在一次采集期间固定；不要将不同 TCP、相机或标定板安装方式的样本混合。

## 测试文件

`tests/fixtures/er_pose.json`、`tests/fixtures/inexbot_pose.json` 和 `tests/fixtures/target_to_camera.json` 是可直接用于接口验证的输入样例。执行 `handeye.cli test` 会验证两类机器人单位/坐标转换，并以 12 组多轴合成样本验证手眼求解器。
