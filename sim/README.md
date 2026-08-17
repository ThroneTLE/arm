# UR5e + Robotiq 比赛仿真

这里不维护自制机器人模型。机械臂、夹爪、MoveIt 配置、Gazebo world、控制器和
GazeboGraspFix 全部来自
[`blackcoffeerobotics/ur5e_robotiq_85_mtc`](https://github.com/blackcoffeerobotics/ur5e_robotiq_85_mtc)，
固定在 `371f8a51...`；MTC 固定在 ROS1 `0.1.3` 对应的 `4781ed56...`。固定版本见
`upstream.repos`，Noetic 兼容修改保存在 `patches/`，不会把第三方源码提交到主仓库。

## 已验证基线

环境为 Ubuntu 20.04、ROS Noetic、Gazebo Classic 11。上游原始演示已完成编译并报告
`Pick and Place Successful`，六轴 FollowJointTrajectory、Robotiq GripperCommand、
MoveIt move_group、joint_states 和 Gazebo model/link states 均在线。基线也暴露出一个问题：
上游啤酒罐最终没有移动到配置的 0.5 m 偏移，因此“轨迹执行成功”不能作为抓取成功判据。

比赛 bridge 使用同一个上游 world，额外从 OSRF Gazebo model database 生成现成的
`wood_cube_5cm`，没有制作新模型。完整比赛状态机实测成功，木块从基座约
`(0.500, 0.250, 0.019) m` 搬到 `(0.496, -0.103, 0.019) m`，目标为
`(0.500, -0.100, 0.025) m`。

## 依赖与构建

系统依赖（本机已安装）：

```bash
sudo apt install ros-noetic-moveit \
  ros-noetic-moveit-task-constructor-core \
  ros-noetic-moveit-task-constructor-msgs \
  ros-noetic-rosparam-shortcuts \
  ros-noetic-roboticsgroup-upatras-gazebo-plugins \
  ros-noetic-trac-ik-kinematics-plugin \
  ros-noetic-joint-trajectory-controller ros-noetic-ros-controllers
./sim/build.sh
```

`bootstrap.sh` 可重复拉取固定 commit、初始化 pybind11、应用兼容补丁并建立 catkin 所需
symlink。默认 `BUILD_JOBS=2`，避免 MTC 编译占满内存。

## 运行

复现上游演示：

```bash
./sim/run_upstream_demo.sh
```

比赛流程默认只规划，不运动：

```bash
./sim/run_competition_demo.sh
```

明确允许仿真机械臂运动并执行完整闭环：

```bash
./sim/run_competition_demo.sh --execute
```

观察完整比赛抓取（同时打开 Gazebo 和 RViz）：

```bash
./sim/run_competition_demo.sh --execute --gui
```

两个 wrapper 默认移除 `DISPLAY`，不会打开 Gazebo/RViz 窗口。它们以独立进程组启动，退出
时只清理自己创建的 ROS/Gazebo 进程。需要人工中止时运行 `./sim/stop.sh`。GUI 仅在确实
需要观察时用 `--gui` 显式开启；上游演示对应命令为
`./sim/run_upstream_demo.sh --gui`。

## bridge 边界

`ws/src/competition_sim_bridge` 很薄，只包含：

- `MoveItRobotController`：以 `base_link` 下的 `T_base_tcp` 实现通用 RobotController；
- `RobotiqGripperController`：复用上游 GripperCommand action；
- `GazeboObjectPoseProvider`：用 `robot::base_link` 的 Gazebo truth 转换物体位姿；
- competition world launch：复用上游 world 并生成 OSRF 5 cm 木块；
- TRAC-IK 仿真覆盖：旧配置的 5 ms KDL pose sampler 经常找不到有效 goal state。

真实比赛 YAML 始终保持 `dry_run: true`。`config/competition_sim.yaml` 的运动许可只在
`run_competition_demo.py` 内存中覆盖，不会保存回正式配置。仿真中 endpoint 间路径由
MoveIt 碰撞检查，因此允许长 endpoint；真实低级控制仍使用 50 mm/10° 安全分段。

夹爪有一个上游特性：GripperActionController 没有配置 stall timeout，夹到物体时 action
会保持 ACTIVE。bridge 在 close 超时后继续保持 goal，不把它当作最终成功；随后必须看到
Gazebo 物体实际抬升至少 30 mm，状态机才会继续。这样既能让 GazeboGraspFix 建立 attachment，
又不会因夹爪命令“已发送”而虚报成功。
