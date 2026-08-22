# 控制器基础 TCP/Modbus 通信层

## 手册边界

`docs/c9c6716a-f022-46a1-b5c4-3f982a816a50.pdf` 的“远程模式”章节确认：

- 远程模式支持数字 IO 和 Modbus 从站；
- Modbus 优先级高于数字 IO；
- Modbus 与数字 IO 可以同时使用，需在 `modbusAddr.json` 中打开
  `coexistIOControl`；
- 示教器拔下后可触发远程模式；
- 远程点到点速度和远程直线速度由控制器中的远程速度、指令速度共同决定，远程速度默认
  15%。

这本手册没有给出控制器 IP、端口、Modbus 单元号、线圈/寄存器地址、数据编码，也没有
给出通过 TCP 直接发送 MOVJ/MOVL 的厂商报文。因此代码不能根据指令名称猜测运动协议。
当前实现的是标准 Modbus **TCP** 的基础链路；现场仍必须确认控制器的物理接口确实是
Modbus TCP，而不是串口 Modbus 或厂商自定义 TCP。

## 协议知识来源（两份资料）

1. **`docs/c9c6716a-...pdf`（纳博特系统操作手册，73 页）**：示教界面、指令语义、坐标
   系与形态参数说明、远程模式功能描述（数字 IO + Modbus 从站、`modbusAddr.json` 的
   `coexistIOControl`、远程速度默认 15%）。**没有** IP/端口/寄存器地址/报文格式。
2. **`docs/机械臂协议问答.pdf` / `机械臂协议问答.md`（现场问答总结，2026-08-17）**：
   已给出**语义级协议事实**（比操作手册更接近报文，但仍不是字节级）：
   - 不同机械臂本体共用纳博特控制系统：编程指令、操作逻辑、**通讯协议完全通用**；
   - TCP Socket 通信可调用**读笛卡尔坐标指令**（指定坐标系、工具号、用户坐标号），
     返回 XYZABC（**弧度**），10~50 Hz 轮询即可满足抓取；
   - **完整点位结构体** = 坐标系类型、弧度标记、形态参数、工具号、用户坐标号、
     XYZABC，与示教器保存的点位格式一致；只发 6 个坐标会随机选逆解、跳构型；
   - **纳博特 TCP 协议默认小端序（Little-Endian）**，float/int 统一小端；
   - A/B/C 对应绕 X/Y/Z 轴的旋转量，单位为弧度；X'Y'Z' 动系旋转 = ZYX 固定角
     （与代码 `transform_from_xyz_rpy` 的固定轴 RzRyRx 一致）；
   - 远程模式下既可通过 TCP 指令**直接调用全局 GP 点**，也可触发本地程序；
     基准点推荐提前存为全局点位；
   - MOVJ/MOVL 使用场景（MOVJ 长距离空行程、MOVL 精确直线下降）、奇异报警处理
     （MOVJ 逃逸、MOVL 失败自动降级 MOVJ）等现场经验。

   该文档**仍未给出字节级报文**：读坐标指令、GP 调用指令、MOVJ/MOVL 指令的帧头、
   命令码、长度、校验和字段字节偏移都没有。因此"运动协议未实现"的结论不变，缺口
   已从"完全没有协议资料"缩小为"只缺字节级帧格式"——需要官方 SDK 或现场抓包。
3. **官方开放文档站（2026-08-21 已补全字节级协议）**：纳博特官方
   [`open.inexbot.com`](https://open.inexbot.com/zh/05.JSON-协议/)（开放机器人联盟
   SDK 开发文档）与 [`doc.inexbot.com`](https://doc.inexbot.com/) 知识库公开了
   22.07/24.03 版完整协议，已整理到 [`docs/纳博特通讯协议.md`](../docs/纳博特通讯协议.md)：
   - 帧格式：`0x4E66` 固定包头 + 2 字节长度 + 2 字节命令字 + JSON 数据 + CRC32
     （对 Length+Command+data 计算，大端发送；已用官方示例帧验证 `zlib.crc32`）；
   - 端口：5000 文件传输、**6000 实时命令口**（运动/状态/工艺）、7000 上位机状态
     查询服务（`0x9512/0x9513`）；
   - **MOVJ `0x4501` / MOVL `0x4502`**（`robot/vel/coord/pos[7]`）、MOVC `0x4503`、
     MOVS `0x4504`、GO_POSITION `0x3003`（完整 RobotPos）、GO_HOME `0x3002`、
     运行状态 `0x9102/0x9103`、伺服 `0x2001`、急停 `0x2314`、全局变量 `0x5602–0x560C`；
   - 24.03 版命令字不同（7000 端口 `0x1E00/0x1E01` + `targetMode/cfg/moveMode`），
     现场先确认固件版本。

   **结论**：字节级运动协议已获得（官方公开文档），"只缺帧格式"的缺口已补齐；
   剩余为现场确认项（固件版本、6000 端口直连与心跳、`pos` 数组长度/弧度单位、
   JSON 命令无 shape 字段时的构型行为），清单见 `docs/纳博特通讯协议.md` §11。

## 已实现的层次

正式实现位于 ROS 包的
`ros_ws/src/arm_vision_framework/src/arm_vision_framework/adapters/inexbot_modbus.py`。
`competition_pipeline/controller_tcp.py` 只是离线测试兼容加载器，确保现场测试不需要先
source catkin overlay，但不会产生第二套协议实现。

正式模块包含：

1. `TcpEndpoint` / `TcpTransport`：阻塞式 TCP 连接、精确收包、超时、断线清理和
   keepalive；不自动重试、不隐藏失败。
2. `ModbusTcpClient`：标准 MBAP 头（Transaction、Protocol、Length、Unit）和常用
   功能码：
   - `01/02` 读线圈/离散输入；
   - `03/04` 读保持/输入寄存器；
   - `05/06` 写单线圈/单寄存器；
   - `15/16` 写多线圈/多寄存器。
3. `ConfiguredRemoteIo`：从配置读取命名数字 IO 地址，提供 `set_output` 和
   `read_input`；不会内置任何夹爪地址。
4. `InexbotPoint`：按手册保存点位字段（坐标系、角度/弧度、形态、Tool、User、两个
   预留字段和 7 个轴值）。字段集与问答文档的"完整点位结构体"（坐标系类型、弧度标记、
   形态参数、工具号、用户坐标号、XYZABC）一致，代码已按该语义建模；但它只是点位
   数据模型，不是猜测的网络报文——字节级布局仍未知。
5. `shape_from_joint_degrees`：按手册对 1/3/5 轴的 `[-90°, +90°]` 区间进行二进制
   编码并加 1。
6. `ControllerStateReader`：只读、配置驱动的控制器状态解码，支持 `u16/s16/u32/s32/
   f32_be/f32_le_words`；地址、字序、倍率全部必须来自官方寄存器表。
7. `ControllerState` / `VisualTaskCommand`：版本化 `arm_vision.control.v1` JSON 契约，
   固定 XYZ 为毫米、姿态为度，并保留 Tool/User/shape、命令 ID、安全状态和原始寄存器。

所有读写 API 都使用 Modbus 规范的零基地址和 16 位寄存器值。收到异常响应、事务号/单元
号不匹配、长度错误或连接超时会抛出异常；不会自动重发可能改变状态的写操作。

## 配置

`config/competition.yaml` 中的控制器配置默认关闭：

```yaml
controller:
  enabled: false
  transport: modbus_tcp
  host: ''
  port: null
  unit_id: 1
  connect_timeout_s: 2.0
  io_timeout_s: 1.0
  remote_io:
    outputs: {}
    inputs: {}
  state_registers: {}
```

未拿到现场协议前保持 `enabled: false`。确认协议后，只填入现场提供的 IP、端口、单元
号和地址映射，例如：

```yaml
controller:
  enabled: true
  transport: modbus_tcp
  host: 192.168.1.20       # 示例占位，不能直接使用
  port: 502                # 仅在现场确认 Modbus TCP 后填写
  unit_id: 1
  remote_io:
    outputs:
      gripper_open: 100    # 待现场确认
      gripper_close: 101   # 待现场确认
    inputs:
      gripper_done: 200    # 待现场确认
```

配置校验会拒绝空的启用端点、非法单元号/地址，并始终拒绝
`controller.motion.enabled: true`。字节级运动协议现已在
[`docs/纳博特通讯协议.md`](../docs/纳博特通讯协议.md) 中给出（22.07：6000 端口
`0x4501/0x4502` 等），并已实现为 `ros_ws/.../adapters/nexbot_tcp.py`
（`robot.adapter: nexbot_tcp`，本地假服务器回环测试通过）；实现
`RobotController.move_j/move_l` 后按现场验收结果打开
`controller.motion.enabled`——在此之前不能通过本模块写寄存器来“试探” MOVJ/MOVL。

状态映射使用下列字段名；具体地址和编码不能照抄示例，必须逐项对照示教器：

```yaml
controller:
  state_registers:
    # joint_deg_1 ... joint_deg_6
    # tcp_x_mm, tcp_y_mm, tcp_z_mm
    # tcp_rx_deg, tcp_ry_deg, tcp_rz_deg
    # tool_id, user_id, shape, servo_on, emergency_stop, moving
    # reserved_1, reserved_2, axis_1 ... axis_7
    # alarm_code, alarm_active
    joint_deg_1:
      address: <官方零基地址>
      source: holding       # holding / input / coil / discrete
      encoding: s32         # 必须确认字序
      scale: <官方倍率>
```

第 09 页 UI 可填写 IP/Port/Unit ID 并保存，只会执行读操作。显示格式与手册点位元数据对齐：
坐标系、度/弧度标志、shape、Tool ID、User ID、两个保留字段和 Axis1..Axis7；当前状态另显
示 Servo、急停、运动和报警。读取到的 shape 应原样保留到后续控制命令，不用仅凭 XYZ/RPY
重新猜构型。

## Modbus-GP 运动保底（默认关闭）

私有运动 TCP 不可用时，可由示教器上的已验证本地程序执行运动：程序读取选定的全局点
`GP0001..GP9999`，根据动作码调用 MOVJ/MOVL，并置位完成信号。ROS 侧
`ModbusGlobalPointRobotController` 只负责写完整 GP 字段、动作码、递增序列号、启动脉冲，
再轮询完成信号；超时或异常会停止并拒绝继续。

示教器/官方表必须提供并验证：

- 每个 GP 字段对应的保持寄存器地址、字节序、编码和倍率；
- 动作码、序列号、启动/完成/停止信号及有效电平；
- 本地程序名称、读取哪个 GP 点以及 MOVJ/MOVL 的速度/停止语义。

这些信息在当前手册中不存在，问答文档也只给出字段语义（坐标系类型/弧度标记/形态/
Tool/User/轴值），未给出 Modbus 地址；因此配置保持空模板，代码不内置任何地址。启用还必须同时
满足 `local_program_verified: true`、完整 `global_point_fields`、`command_fields`、
`start_signal`、`complete_signal`、`stop_signal`，并能从状态回读确认在线、急停关闭、无报警和初始 shape
已锁存；否则 `factory.build_robot(... adapter: modbus_global_point)` 和适配器都会
fail-closed。私有 TCP 失败后才按现场验收结果显式切换，不会自动试探 Modbus 地址。

## 奇异点安全回退

正式执行器遇到控制器错误时先 STOP，原失败轨迹立即作废。报警码需要现场加入厂商映射；
在映射完成前也会检查中英文文本中的 `singular`、`singularity`、`奇异`、`奇点`、`IK`、
`逆运动学` 和 `configuration`。回退目标必须是现场预先保存并确认无碰撞的完整 MOVJ 点列。

默认 `auto_recover: false`。只有以下条件全部成立才允许显式/自动回退：控制器在线、急停
明确关闭、TCP 状态有效、存在安全点，且 STOP 成功。任何字段未知都保持锁定；不能在报警后
继续剩余 MOVL，也不能把“回到上一个任意轨迹点”当作安全点。

## 最小调用示例

```python
from competition_pipeline.controller_tcp import (
    ModbusTcpClient, TcpEndpoint,
)

client = ModbusTcpClient(TcpEndpoint('192.168.1.20', 502), unit_id=1)
try:
    # 地址和含义必须来自现场寄存器表。
    client.write_single_coil(100, True)
    done = client.read_coils(200, 1)[0]
finally:
    client.close()
```

仓库内 pipeline 回归测试使用本地假服务器验证 MBAP、分片收包、异常响应和事务号校验：

```bash
python3 -m unittest competition_pipeline.tests.test_controller_tcp -v
```

## 到场确认顺序

1. 先确认控制器型号、固件、网口/串口物理连接、远程模式开关和急停行为；
2. 向厂商或现场老师索取 Modbus TCP/RTU 寄存器表及运动控制 SDK，记录 IP、端口、Unit
   ID、字节序、线圈/寄存器地址、动作完成反馈和断线行为。协议语义事实（点位结构体、
   小端序、弧度、GP 点 TCP 调用、读坐标指令）见 `docs/机械臂协议问答.pdf`；**字节级
   报文**已由官方开放文档获得（`docs/纳博特通讯协议.md`，22.07：6000 端口
   `0x4501/0x4502`、7000 端口 `0x9512`），现场只需确认固件版本、6000 端口直连行为
   和坐标单位（见该文档 §11 清单）；
3. 启动 `./competition_pipeline/run_ui.sh --stage controller`，只填写 IP/Port/Unit ID 和
   `state_registers`。先不填夹爪输出、更不发送 MOVJ/MOVL；
4. 逐项让示教器显示已知数值，再对照 UI 的 Servo、急停、运动、Axis1..7、TCP、
   坐标系、度/弧度、shape、Tool/User、保留字段和报警。每个寄存器都确认零基地址、字序、
   倍率与有效电平；
5. 机械臂停稳、急停关闭且无报警时，用第 09 页保存 `P9000`。这一步保存的是完整 MOVJ
   构型，不能用仅 XYZ/RPY 的位姿代替；
6. 导入正式 ROS 参数：

   ```bash
   source ros_ws/devel/setup.bash
   rosrun arm_vision_framework calibration_tool.py import-competition-controller \
     --input competition_pipeline/config/competition.yaml
   ```

7. 仅在确认控制器 RPY 是固定 ZYX 度制，并让“示教器 TCP”和 ROS 位姿做过停稳对照后，才设
   `controller.use_state_topic: true`、
   `controller.state_pose_convention: fixed_zyx_rpy_deg`，并启动
   `start_controller_state:=true`；
8. 断开伺服或在安全低速条件下，单独验证一个无负载输出；再实现 `GripperController` 的地址
   映射和反馈超时；
9. MOVJ/MOVL 按 `docs/纳博特通讯协议.md` 实现（22.07：6000 端口 `0x4501` vel 百分比 /
   `0x4502` vel mm/s，帧为 `0x4E66+len+cmd+JSON+CRC32`；构型不确定时改用 `0x3003`
   GO_POSITION 完整点位），接入 `SafeRobotController` 的 fail-closed 门。实现时严格
   遵循问答文档：完整点位结构体（不裸发 XYZABC）、小端序（仅 Modbus 数据，JSON 帧
   为 UTF-8 无字节序问题）、A/B/C 弧度制、下发后轮询到位再动作（不用固定延时）、
   统一工具号/坐标系号。
