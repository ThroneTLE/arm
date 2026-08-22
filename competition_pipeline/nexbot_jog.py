"""Qt-free robot jog + gripper controls for the competition UI (NexBot).

Wraps ``NexBotTcpRobotController`` (official RTL-22.07 protocol) with the
teach-pendant style actions the verification page needs:

- 六方向步进（用户坐标系1，安全小步长）: 前进/后退/左/右/上/下；
- 夹爪开/关（DOUT16/DOUT15 双线圈，现场实测映射 (15,16)=(1,0)关 <-> (0,1)开）；
- 回零（0x3002 GO_HOME）与回复位点（0x3007 GO_RESET_POSITION，现场约定
  复位点=拍摄点）；
- 急停（0x2314）。

All poses are read/written in the active user coordinate frame (用户坐标系1),
so Y/X/Z values match the teach pendant's user-coordinate display.

单位约定（跨这条边界出过事，务必遵守）
--------------------------------------
- ``current_pose()``     -> (xyz_mm, abc_**deg**)   给界面显示用
- ``current_pose_rad()`` -> (xyz_mm, abc_**rad**)   给运动调用用
- ``move_to_ucs()``      收 abc_**rad**

把 ``current_pose()`` 的角度值直接喂给 ``move_to_ucs()`` 会摔机械臂
（2026-08-22 实际发生过）。``check_abc_is_radians`` 会拦下这种调用。

安全模型
--------
控制器的复位点安全闸门在**远程模式**下对每条运动指令必然判定"不在安全位置"，
并且每拒绝一次就把伺服下电一次。因此：

- 每个运动入口在下发前都要 ``_ensure_servo_running_locked()``，
  不能假设"刚才成功过所以现在还使能着"；
- 下发后由适配器 ``_await_motion_ack()`` 用 ``0x3D03 status=2`` 确认真的动了。

详见 ``docs/纳博特C1102-现场真相-必读.md``。
"""

import logging
import threading
import time

_TELEPORT_LOG = logging.getLogger("nexbot.teleport")
_TELEPORT_LOG.setLevel(logging.DEBUG)
try:
    import os
    _handler = logging.FileHandler(os.path.expanduser("/tmp/nexbot_teleport.log"), encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(threadName)s %(levelname)s %(message)s"))
    if not _TELEPORT_LOG.handlers:
        _TELEPORT_LOG.addHandler(_handler)
except Exception:
    pass

import numpy as np

from .geometry import (
    as_transform,
    inexbot_abc_from_transform,
    rotation_angle_deg,
)
from .nexbot_tcp import (
    NexBotTcpEndpoint,
    NexBotTcpRobotController,
    ControllerConnectionError,
)

#: 现场实测 (2026-08-22): DOUT15/16 = 双线圈气阀; (15,16)=(1,0) 关闭 <-> (0,1) 打开
GRIPPER_PORT_CLOSE = 15
GRIPPER_PORT_OPEN = 16

#: 步进默认速度（速度比例 -> move_to 内部 mm/s = speed_scale*1000）
JOG_SPEED_SCALE = 0.05  # 50 mm/s，安全小步进

#: A/B/C 合法弧度上界。控制器回读的 A/B/C 落在 [-pi, pi]；任何 |值| 超过这个门限
#: 都不可能是弧度，只可能是**有人把角度制的数当弧度传进来了**。
#: 2026-08-22 现场就是这样把机械臂摔了：回读得到 (177.87, 13.76, -179.99) 度，
#: 被当成弧度发出，折叠后变成 (111.2, 68.5, 127.1) 度，与真实姿态差 119.6°，
#: 而 XYZ 只走 30mm 直线 -> 六轴同时 0F15 故障 -> 控制器下电 -> 臂坠落。
#: 证据: docs/现场备份-20260822/根因证据-控制器日志摘录.txt 证据 C-1 / C-2。
MAX_ABC_RAD = 7.0  # ≈ 401°，比 2*pi 稍宽，留给 ±180° 附近的正常值

#: 单次绝对运动允许的最大姿态变化（度）。位置护栏（400mm）挡不住姿态突变，
#: 因为摔臂那条指令 XYZ 只动了 30mm。默认收紧到 20°，需要大幅换姿态时显式放宽。
MAX_ROTATION_STEP_DEG = 20.0

#: 单次绝对运动允许的最大位置变化（mm）。
MAX_TRANSLATION_STEP_MM = 400.0


def check_abc_is_radians(abc, where=""):
    """Reject an A/B/C triplet that is obviously in degrees.

    Raises ``ValueError`` describing the exact mistake instead of letting the
    value reach the controller.  ``move_to_ucs`` takes **radians**; the UI and
    ``current_pose()`` both work in **degrees**, so this boundary is crossed on
    every teleport and every grasp step.
    """
    values = np.asarray(abc, dtype=float).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError("A/B/C 必须是 3 个有限数，收到 {!r}".format(abc))
    worst = float(np.max(np.abs(values)))
    if worst > MAX_ABC_RAD:
        raise ValueError(
            "{}A/B/C 期望**弧度**，但收到 [{:.4f}, {:.4f}, {:.4f}]（最大 {:.2f} > {:.1f}），"
            "这是角度制的数值。若来自 current_pose() 请先 np.radians() 转换。"
            "（2026-08-22 现场正是这个错误导致六轴 0F15 故障、机械臂坠落）".format(
                "{}: ".format(where) if where else "", *values, worst, MAX_ABC_RAD
            )
        )
    return values


class NexBotTcpJog:
    """One controller instance used for jog/gripper/home/reset actions.

    6001 是单客户端端口：旧连接被控制器释放前，新连接会 [Errno 111] 拒绝。
    ``_run`` 对每个动作做 关闭旧连接 -> 等待 -> 重连重试（运动为绝对/幂等，
    夹爪重试重发 15/16 全序列，不会出现只写一半的半开阀状态）。
    """

    #: 重连尝试次数与间隔（6001 槽位释放延迟通常 < 2s）
    MAX_RETRIES = 3
    RETRY_WAIT_S = 1.2

    def __init__(self, endpoint: NexBotTcpEndpoint, keepalive_s=None):
        """keepalive_s: 非空时启动持久保活线程（6001 单客户端：占住槽位，
        让控制器保持"已连接"；掉线后由 `_run` 自动重连）。"""
        self.endpoint = endpoint
        self._controller = None
        self._controller_lock = threading.RLock()
        self._lock = threading.Lock()
        self._estop_lock = threading.Lock()
        #: 急停正在抢占 6001 时置位；``_run_locked`` 见到它就放弃重连重试，
        #: 免得工作线程和急停线程互抢单客户端槽位（见 :meth:`emergency_stop`）。
        self._estop_pending = threading.Event()
        self._closed = False
        self._keepalive_s = keepalive_s
        self._keepalive_stop = threading.Event()
        self._keepalive_thread = None
        #: 保活线程观测到的最近伺服状态（见 :meth:`health`）
        self.last_servo_status = None
        self.last_keepalive_error = None
        self.servo_dropped_count = 0
        #: 最近一次 :meth:`gripper` 的回读结论 ``(ok, detail)``
        self.last_gripper_verify = None
        if keepalive_s is not None:
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, daemon=True,
                name="nexbot-keepalive",
            )
            self._keepalive_thread.start()

    def _keepalive_loop(self):
        """保活 + **记录**伺服状态。

        以前这里是 ``try: servo_status() except Exception: drop``：返回值被丢掉、
        异常被全吞，于是控制器把伺服下电之后界面上什么都看不出来，操作者只会
        觉得"点了没反应"。现在把最近一次状态和错误留下来供 :meth:`health` 查询。
        """
        while not self._keepalive_stop.wait(self._keepalive_s):
            if not self._lock.acquire(blocking=False):
                continue
            try:
                status = int(self.controller.servo_status())
                self.last_servo_status = status
                self.last_keepalive_error = None
                if status != 3:
                    # 不主动重新使能：使能必须由某个明确的动作发起，
                    # 后台线程偷偷上电会让"机器人为什么突然带电"无法追溯。
                    self.servo_dropped_count += 1
            except ControllerConnectionError as error:
                self.last_servo_status = None
                self.last_keepalive_error = str(error)
                self._drop_controller()
            except Exception as error:            # 协议错误/报警帧等，连接仍可用
                self.last_servo_status = None
                self.last_keepalive_error = str(error)
            finally:
                self._lock.release()

    def health(self):
        """(servo_status, error, dropped_count) —— 保活线程看到的最近状态。

        ``servo_status`` 为 None 表示最近一次保活没读到状态。``dropped_count``
        统计"保活期间发现伺服不在运行态"的次数：在现场，这个数字每涨一次
        就意味着控制器刚刚拒绝过一条指令并把伺服下了电。
        """
        return (
            self.last_servo_status,
            self.last_keepalive_error,
            self.servo_dropped_count,
        )

    @property
    def controller(self):
        with self._controller_lock:
            if self._closed:
                raise ControllerConnectionError("NexBot jog controller is closed")
            if self._controller is None:
                self._controller = NexBotTcpRobotController(self.endpoint)
            return self._controller

    def _run(self, action, *args, **kwargs):
        """Run ``action(controller, ...)`` as one serialized transaction."""
        with self._lock:
            return self._run_locked(action, *args, **kwargs)

    def _run_locked(self, action, *args, **kwargs):
        """Reconnect-and-retry, but ONLY for actions that are safe to repeat.

        ``retry_on_disconnect=False`` marks an action whose re-execution could
        move the robot a second time.  A dropped 6001 connection is ambiguous:
        the controller may well have received and started the command before the
        socket died -- which is exactly what happens when the safety gate powers
        the servo off mid-transaction.  Blindly resending a motion in that state
        means the arm can lurch again the instant it is re-enabled, with nobody
        expecting it.  ``step()`` is the one motion that opts back in, because it
        recomputes an *absolute* target once and resending the identical target
        is idempotent (see ``test_step_retry_resends_the_same_absolute_target``).
        """
        retry_on_disconnect = bool(kwargs.pop("retry_on_disconnect", True))
        attempts = self.MAX_RETRIES if retry_on_disconnect else 1
        last_error = None
        for attempt in range(1, attempts + 1):
            if self._closed:
                raise ControllerConnectionError("NexBot jog controller is closed")
            try:
                return action(self.controller, *args, **kwargs)
            except ControllerConnectionError as error:
                last_error = error
                self._drop_controller()
                if self._estop_pending.is_set():
                    # 急停刚刚把这条连接掐了。重连重试会和急停线程抢 6001 的
                    # 单客户端槽位，最坏情况是把急停帧挤掉。放弃，向上报错。
                    raise ControllerConnectionError(
                        "已发出急停，放弃本次动作的重连重试：{}".format(error)
                    ) from error
                if attempt < attempts:
                    time.sleep(self.RETRY_WAIT_S * attempt)
        if not retry_on_disconnect:
            raise ControllerConnectionError(
                "连接在动作执行中断开，且该动作不可安全重发（可能已经在控制器上执行）。"
                "请先【回读当前】确认机器人实际位置，再决定下一步：{}".format(last_error)
            ) from last_error
        raise last_error

    def _drop_controller(self):
        """Discard a failed connection without stopping keepalive."""
        with self._controller_lock:
            controller = self._controller
            self._controller = None
            # The controller accepts one client per port.  Release the old
            # slot before controller() may construct its replacement.
            if controller is not None:
                controller.close()

    def close(self):
        """Permanently stop keepalive and release the shared connections."""
        with self._controller_lock:
            if self._closed:
                return
            self._closed = True
        self._keepalive_stop.set()
        self._drop_controller()

    def _ensure_servo_running_locked(self, stage=""):
        """Guarantee 伺服 status==3 before any motion.  Caller holds ``_lock``.

        ``0x2311`` is the only enable channel that works on this firmware, and
        the servo does NOT stay enabled by itself across a refused command: the
        reset-point safety gate powers it off on every refusal.  Every motion
        entry point therefore has to re-check, not just the first one.

        Before 2026-08-22 only ``step()`` did this, which is why the field
        symptom was "点一次点动之后传送/抓取才偶尔能动" -- everything else was
        silently riding on the enable that ``step()`` happened to leave behind.
        """
        status = self._run_locked(lambda controller: controller.servo_status())
        if int(status) == 3:
            return 3
        self._run_locked(lambda controller: controller.enable_servo())
        status = self._run_locked(lambda controller: controller.servo_status())
        if int(status) != 3:
            raise RuntimeError(
                "{}伺服未能进入运行态(status={})，已放弃下发运动。"
                "请检查示教器：是否在示教模式、是否有未清报警。".format(
                    "{} ".format(stage) if stage else "", status
                )
            )
        return int(status)

    def ensure_servo_running(self, stage=""):
        """Public wrapper: take the transaction lock, then ensure 伺服 status==3."""
        with self._lock:
            return self._ensure_servo_running_locked(stage)

    def current_pose(self):
        """(xyz_mm, abc_deg) 当前 TCP 位姿（用户坐标系1）。

        ⚠️ 返回的 A/B/C 是**角度制**。``move_to_ucs`` 收的是**弧度**。
        跨这个边界必须 ``np.radians()``，或者直接用 ``current_pose_rad()``。
        """
        xyz_mm, abc_rad = self.current_pose_rad()
        return xyz_mm, tuple(np.degrees(abc_rad))

    def current_pose_rad(self):
        """(xyz_mm, abc_rad) 当前 TCP 位姿（用户坐标系1），A/B/C 为**弧度**。

        Prefer this over ``current_pose()`` whenever the value is going to be
        fed back into a motion call -- it removes the unit conversion entirely.
        """
        state = self._run(lambda controller: controller.read_state())
        xyz_m, abc_rad = inexbot_abc_from_transform(state.base_from_gripper)
        return tuple(xyz_m * 1000.0), tuple(abc_rad)

    def step(self, axis: int, step_mm: float):
        """在用户坐标系1 中沿 axis(0=X,1=Y,2=Z) 平移 step_mm（可负）。

        绝对运动：当前位姿 + 轴偏移 -> MOVL(coord=用户坐标)。小步长专为
        坐标核对设计，速度按 JOG_SPEED_SCALE。自动确保伺服上电（0x2311）。
        6001 为单客户端端口，连接被抢占时重连重试一次（绝对运动幂等）。
        """
        axis = int(axis)
        if axis not in (0, 1, 2):
            raise ValueError("jog axis must be 0, 1 or 2")
        step_mm = float(step_mm)
        if not np.isfinite(step_mm) or step_mm == 0.0:
            raise ValueError("jog step must be a finite non-zero distance")

        # Build the absolute destination once.  An ambiguous disconnect after
        # MOVL therefore resends the same target rather than adding the step
        # again from an already-advanced pose.
        with self._lock:
            self._ensure_servo_running_locked("步进")
            state = self._run_locked(lambda controller: controller.read_state())
            matrix = as_transform(state.base_from_gripper, "world_from_gripper")
            delta = np.zeros(3, dtype=np.float64)
            delta[axis] = step_mm / 1000.0
            target = np.eye(4, dtype=np.float64)
            target[:3, :3] = matrix[:3, :3]
            target[:3, 3] = matrix[:3, 3] + delta
            self._run_locked(
                lambda controller: controller.move_to(
                    target, speed_scale=JOG_SPEED_SCALE
                )
            )

    def move_to_ucs(self, xyz_mm, abc_rad, vel_mm_s=50.0, tolerance_mm=1.0,
                    max_rotation_deg=MAX_ROTATION_STEP_DEG,
                    max_translation_mm=MAX_TRANSLATION_STEP_MM,
                    rotation_tolerance_deg=3.0):
        """用户坐标系1 绝对传送（0x2311 上使能 -> 0x4502 MOVL）并核验到位。

        ``abc_rad`` 是**弧度**。传角度制会被 :func:`check_abc_is_radians` 拦下。

        下发前的三道闸门（缺一不可，2026-08-22 摔臂就是因为只有第一道）：

        1. 位置：|目标 - 当前| <= ``max_translation_mm``
        2. 单位：|A/B/C| <= ``MAX_ABC_RAD``，挡住"度数当弧度"
        3. 姿态：目标姿态与当前姿态夹角 <= ``max_rotation_deg``

        闸门 3 是真正的兜底 —— 摔臂那条指令 XYZ 只动了 30mm（轻松通过闸门 1），
        姿态却要转 119.6°，腕部沿 30mm 直线摆 120° 导致六轴同时 0F15 故障下电。
        若目标姿态本身数值不大（比如真实姿态接近零位），闸门 2 也挡不住，只有闸门 3 能挡。

        返回到达位置偏差 mm；位置或姿态超差抛 RuntimeError。
        """
        from .geometry import transform_from_inexbot_abc

        def _read(c):
            return c.read_state()

        _TELEPORT_LOG.debug(
            "move_to_ucs start target=%s abc_rad=%s vel=%s", xyz_mm, abc_rad, vel_mm_s)

        # 闸门 2：单位。放在最前面 —— 连一次读取都不做就能否掉明显错误的入参。
        abc = check_abc_is_radians(abc_rad, where="move_to_ucs")
        target_xyz = np.asarray(xyz_mm, dtype=float).reshape(-1)
        if target_xyz.size != 3 or not np.all(np.isfinite(target_xyz)):
            raise ValueError("XYZ 必须是 3 个有限数，收到 {!r}".format(xyz_mm))

        start_state = self._run(_read)
        _TELEPORT_LOG.debug("read start pose ok")
        target = transform_from_inexbot_abc(target_xyz / 1000.0, abc)
        start_pose = np.asarray(start_state.base_from_gripper, dtype=np.float64)
        start_xyz = start_pose[:3, 3] * 1000.0

        # 闸门 1：位置
        distance = float(np.linalg.norm(target_xyz - start_xyz))
        if distance > float(max_translation_mm):
            raise RuntimeError(
                "目标距当前 {:.0f} mm > {:.0f}mm：疑似未确认输入（默认 0,0,0）或目标超工作范围；"
                "请先【回读当前】核对（当前 {:.2f},{:.2f},{:.2f}）".format(
                    distance, float(max_translation_mm), *start_xyz)
            )

        # 闸门 3：姿态
        rotation_step = rotation_angle_deg(start_pose, target)
        if rotation_step > float(max_rotation_deg):
            raise RuntimeError(
                "目标姿态与当前姿态相差 {:.1f}° > {:.1f}°，已拒绝下发。"
                "位移只有 {:.1f}mm 却要求大幅换姿态，腕部角速度会超限；"
                "2026-08-22 现场正是这种指令（119.6°/30mm）造成六轴 0F15 故障、"
                "控制器下电、机械臂坠落。若确需大幅换姿态，请显式放宽 "
                "max_rotation_deg 并分段执行。"
                "（当前 A/B/C={:.4f},{:.4f},{:.4f} rad，目标 {:.4f},{:.4f},{:.4f} rad）".format(
                    rotation_step, float(max_rotation_deg), distance,
                    *inexbot_abc_from_transform(start_pose)[1], *abc)
            )
        _TELEPORT_LOG.debug(
            "guards ok: dist=%.2fmm rot=%.2fdeg", distance, rotation_step)

        # 使能：每段运动都要重新确认。安全闸门每拒绝一次就把伺服下电一次，
        # 所以"上一步成功过"不能推出"这一步伺服还在使能"。
        self.ensure_servo_running("传送")

        def _go(c):
            c.move_to(target, speed_scale=vel_mm_s / 1000.0)

        # 绝对目标，但**不允许**断链自动重发：传送是大位移，断链时机器人可能
        # 已经在半路上，盲目重发会在操作者毫无预期时再动一次。
        # 报错让人先【回读当前】看清楚在哪。

        # 竞态修复：控制器收到指令后有百毫秒级延迟才开始动，"静止判停"可能
        # 在启动前误判完成。改为"等到达目标"：|当前-目标|<容差 连续 2 次
        # 采样（0.25s 间隔）才算到位，上限 20s。
        # 注：move_to -> move_l 内部已经用 0x3D03 status=2 确认过"真的动了"，
        # 这里的轮询只负责"确认停在了对的地方"。
        self._run(_go, retry_on_disconnect=False)
        _TELEPORT_LOG.debug("motion sent (move_to returned)")
        deadline = time.monotonic() + 20.0
        reached = 0
        final_state = None
        final_xyz = start_xyz
        while time.monotonic() < deadline:
            final_state = self._run(_read)
            final_xyz = final_state.base_from_gripper[:3, 3] * 1000.0
            dist = float(np.linalg.norm(target_xyz - final_xyz))
            reached = reached + 1 if dist <= tolerance_mm else 0
            if reached >= 2:
                break
            time.sleep(0.25)
        if reached < 2:
            _TELEPORT_LOG.error(
                "arrival timeout: final=%s target=%s", final_xyz, xyz_mm)
            raise RuntimeError(
                "未在 20s 内到达目标（当前 {:.2f},{:.2f},{:.2f}，目标 {:.2f},{:.2f},{:.2f}）".format(
                    *final_xyz, *target_xyz)
            )
        _TELEPORT_LOG.debug("arrived, reached=%s", reached)
        final_pose = np.asarray(final_state.base_from_gripper, dtype=np.float64)
        final_xyz = final_pose[:3, 3] * 1000.0
        deviation = float(np.linalg.norm(target_xyz - final_xyz))
        _TELEPORT_LOG.debug("deviation=%.3f mm", deviation)
        if deviation > tolerance_mm:
            raise RuntimeError(
                "到达偏差 {:.2f} mm 超出容差 {:.1f} mm（起点 {:.2f},{:.2f},{:.2f}）".format(
                    deviation, tolerance_mm, *start_xyz)
            )
        # 到位校验必须同时看姿态：只比 XYZ 会放过"位置对了但姿态错了"的情况。
        rotation_error = rotation_angle_deg(target, final_pose)
        if rotation_error > float(rotation_tolerance_deg):
            raise RuntimeError(
                "到达姿态偏差 {:.2f}° 超出容差 {:.1f}°（位置偏差 {:.2f} mm 正常）".format(
                    rotation_error, float(rotation_tolerance_deg), deviation)
            )
        return deviation

    def gripper(self, open_: bool, verify=True):
        """开/关夹爪：开=(15,16)=(0,1) 关=(15,16)=(1,0)，并回读确认。

        返回 ``(ok, detail)``：

        - ``(True,  "")``          回读结果与指令一致，确实切换了；
        - ``(True,  "未回读: …")`` 回读本身失败（超时/协议错），动作已下发但
          **无法确认**；
        - 不一致则抛 ``RuntimeError``。

        为什么必须回读
        --------------
        夹爪走 0x3601 DOUT，**不经过** ``startRobotJobTask``，因此不受复位点
        安全闸门管辖，也拿不到 ``0x3D03`` 那种"真的动了"的确认。2026-08-22 的
        现场表象正是"夹爪照常开合、机械臂纹丝不动"：夹爪那条码路一直是通的，
        于是界面报"✅ 一键抓取完成"，实际什么都没抓到。这里回读 DOUT 是这条
        路径上唯一能拿到的客观证据。

        为什么"回读失败"不算失败
        ------------------------
        ``digital_output_states()``(0x3603) 在本固件上**未经现场验证**。若它
        本身答不上来就把整个抓取判死，等于用一个没验证过的查询去否决一个验证
        过的动作 —— 那才是真正的自伤。所以只在"读到了、而且和指令相反"时才
        报错；读不到就如实标注"未确认"，由调用方写进结论。
        """
        close_value, open_value = (0, 1) if open_ else (1, 0)

        def _do_gripper(controller):
            controller.set_digital_output(GRIPPER_PORT_CLOSE, close_value)
            controller.set_digital_output(GRIPPER_PORT_OPEN, open_value)
            if not verify:
                return (True, "")
            try:
                states = controller.digital_output_states()
            except Exception as error:                # 含超时/协议/连接错误
                return (True, "未回读: {}".format(error))
            if len(states) < GRIPPER_PORT_OPEN:
                return (True, "未回读: DOUT 状态只有 {} 路".format(len(states)))
            actual = (
                int(states[GRIPPER_PORT_CLOSE - 1]),
                int(states[GRIPPER_PORT_OPEN - 1]),
            )
            if actual != (close_value, open_value):
                raise RuntimeError(
                    "夹爪{}指令已下发但 DOUT 回读不符：期望 (15,16)=({},{})，"
                    "实读 ({},{})。气阀线圈可能未动作或接线与约定相反，"
                    "**不要**据此认为已经夹住/松开。".format(
                        "开" if open_ else "合",
                        close_value, open_value, *actual,
                    )
                )
            return (True, "")

        result = self._run(_do_gripper)
        self.last_gripper_verify = result
        return result

    def gripper_state(self):
        """返回 (DOUT15, DOUT16) 当前状态。"""
        status = self._run(lambda controller: controller.digital_output_states())
        if len(status) < 16:
            raise ValueError(
                "DOUT status has {} entries; expected >= 16".format(len(status))
            )
        return status[GRIPPER_PORT_CLOSE - 1], status[GRIPPER_PORT_OPEN - 1]

    def go_home(self):
        """回零 (0x3002)。见 ``go_reset_position`` 的安全闸门警告。"""
        with self._lock:
            self._ensure_servo_running_locked("回零前")
            self._run_locked(lambda controller: controller.go_home())

    def go_reset_position(self):
        """回复位点/拍摄点 (0x3007)。

        ⚠️ 这条指令走控制器的 ``startRobotJobTask(safepos=1)`` 入口，受复位点
        安全闸门管辖。**在远程模式下它必然被拒，且每拒绝一次就把伺服下电一次**
        （出厂配置 RemoteIO[0].posReset.deviation=null 而 safeEnable=true）。

        2026-08-22 现场：``GraspDemoWorker`` 与 ``VisualPlanWorker`` 都把这条
        放在流程第一句，于是每次抓取一开始就把使能打掉，后续 MOVL 全部无效，
        而夹爪（0x3601 走另一条码路）照常开合 —— 这就是"夹爪在动、位置不动"的来源。

        现在：下发前先确保使能，下发后由适配器的 ``_await_motion_ack`` 校验
        ``0x3D03 status=2``，被拒会立刻抛异常而不是静默继续。
        """
        with self._lock:
            self._ensure_servo_running_locked("回复位前")
            self._run_locked(lambda controller: controller.go_reset_position())

    #: 急停等待正常事务让出 6001 的最长时间；超过就抢占。
    ESTOP_PREEMPT_WAIT_S = 0.5

    def emergency_stop(self):
        """下发 0x2314，必要时**抢占**正在进行的正常事务。

        ⚠️ 语义提醒：0x2314 在这台 C1102 上映射到 ``Deadan_End -> PowerOff``，
        是**下电**而不是受控停止，伸展着的手臂会失力下坠。真正的安全急停是
        示教器上的物理按钮；本方法只是"尽快让控制器停下"的软件补充。

        为什么不能简单地绕过 ``_lock`` 直发
        ------------------------------------
        6001 是单客户端端口，整个 :class:`NexBotTcpJog` 共用一条 socket。
        旧实现直接 ``self.controller.stop()``，与正在 ``_run`` 里收发帧的工作
        线程**并发写同一个 socket**：两个 ``sendall`` 交错，急停帧的字节就会
        插进另一帧中间，控制器按帧头/长度/CRC 解析，结果是**两帧都被丢弃** ——
        最需要它工作的时候，急停悄无声息地失效。

        现在的顺序：

        1. 先礼后兵：等 ``_lock`` 最多 ``ESTOP_PREEMPT_WAIT_S``。拿到就走干净
           路径，不必打断任何东西。
        2. 拿不到就抢占：置 ``_estop_pending`` 并 ``_drop_controller()`` 关掉
           socket。持锁的工作线程立刻在读写上拿到 ``ControllerConnectionError``，
           而 ``_run_locked`` 看到 ``_estop_pending`` 会直接放弃重连重试，
           不会回头和我们抢槽位。
        3. 独占地重连并发 0x2314。

        任何时刻只有一个线程在 6001 上发帧，帧不再交错。
        """
        with self._estop_lock:
            acquired = self._lock.acquire(timeout=self.ESTOP_PREEMPT_WAIT_S)
            self._estop_pending.set()
            try:
                if not acquired:
                    # 抢占：掐断工作线程正在用的连接，独占 6001。
                    self._drop_controller()
                last_error = None
                for attempt in range(1, self.MAX_RETRIES + 1):
                    try:
                        self.controller.stop()
                        return
                    except ControllerConnectionError as error:
                        last_error = error
                        self._drop_controller()
                        if attempt < self.MAX_RETRIES:
                            time.sleep(self.RETRY_WAIT_S * attempt)
                raise last_error
            finally:
                self._estop_pending.clear()
                if acquired:
                    self._lock.release()


__all__ = [
    "GRIPPER_PORT_CLOSE",
    "GRIPPER_PORT_OPEN",
    "JOG_SPEED_SCALE",
    "MAX_ABC_RAD",
    "MAX_ROTATION_STEP_DEG",
    "MAX_TRANSLATION_STEP_MM",
    "NexBotTcpJog",
    "check_abc_is_radians",
]
