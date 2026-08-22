#!/usr/bin/env python3
"""Offline tests for the NexBot jog/gripper controls (fake controller)."""

import json
import socket
import struct
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from competition_pipeline.geometry import transform_from_inexbot_abc
from competition_pipeline.nexbot_jog import (
    MAX_ABC_RAD,
    MAX_ROTATION_STEP_DEG,
    NexBotTcpJog,
    check_abc_is_radians,
)
from competition_pipeline.nexbot_tcp import (
    NexBotTcpEndpoint,
    build_frame,
    CMD_DOUT_SET,
    CMD_DOUT_QUERY,
    CMD_DOUT_QUERY_REPLY,
    CMD_GO_HOME,
    CMD_GO_RESET_POSITION,
    CMD_MOVJ,
    CMD_MOVL,
    CMD_PROGRAM_STATUS,
    CMD_SERVO_RESPOND,
    ControllerConnectionError,
)

#: 真控制器接受运动指令后会推 ``0x3D03 status=2``；假控制器必须照做。
MOTION_STARTED = build_frame(CMD_PROGRAM_STATUS, {"robot": 1, "status": 2})


def read_frame(conn, max_bytes, timeout=5.0):
    conn.settimeout(timeout)
    header = b""
    while len(header) < 6:
        chunk = conn.recv(6 - len(header))
        if not chunk:
            raise ConnectionError("connection closed")
        header += chunk
    if header[:2] != b"\x4e\x66":
        raise ValueError("bad sync bytes: {}".format(header.hex()))
    length = struct.unpack(">H", header[2:4])[0]
    if length > max_bytes:
        raise ValueError("frame too large: {}".format(length))
    command = struct.unpack(">H", header[4:6])[0]
    body = b""
    while len(body) < length + 4:
        chunk = conn.recv(length + 4 - len(body))
        if not chunk:
            raise ConnectionError("connection closed")
        body += chunk
    payload = body[:length] if length else b""
    return command, json.loads(payload.decode("utf-8")) if payload else {}


class FakeController:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(4)
        self.port = self.server.getsockname()[1]
        self.received = []
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self.replies = {}
        self.current_ucs = [700.0, 100.0, 400.0, 0.1, 0.2, 0.3]
        #: DOUT 线圈实况(1-based 端口 -> 值)。初始 (15,16)=(1,0) = 夹爪闭合。
        #: 写 0x3601 会改这里, 查 0x3602 从这里答 —— 假控制器必须像真的一样
        #: "写什么读到什么", 否则 gripper() 的回读校验测的是一个静态桩。
        self.dout = [0] * 14 + [1, 0]
        #: 置 True 模拟"线圈没动作/接线相反": 写入被忽略, 回读保持原样。
        self.dout_ignores_writes = False

    def _apply_dout(self, data):
        port = int(data.get("port", 0))
        if 1 <= port <= len(self.dout) and not self.dout_ignores_writes:
            self.dout[port - 1] = int(data.get("status", 0))

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        while not self._stop:
            try:
                command, data = read_frame(conn, 1 << 20)
            except Exception:
                try:
                    conn.close()
                except OSError:
                    pass
                return
            self.received.append((command, data))
            if command == CMD_DOUT_SET:
                self._apply_dout(data)
            if command == CMD_DOUT_QUERY and CMD_DOUT_QUERY not in self.replies:
                conn.sendall(build_frame(
                    CMD_DOUT_QUERY_REPLY,
                    {"status": list(self.dout), "robot": 1},
                ))
                continue
            for reply in self.replies.get(command, []):
                conn.sendall(reply)

    def close(self):
        self._stop = True
        try:
            self.server.close()
        except OSError:
            pass


def endpoint(server, **overrides):
    facts = dict(
        host="127.0.0.1",
        port_motion=server.port,
        port_state=server.port,
        io_timeout_s=0.5,
        connect_timeout_s=0.5,
        wait_for_finish=False,
        heartbeat_s=0.0,
        pose_frame="UCS",
        motion_coord=3,
    )
    facts.update(overrides)
    return NexBotTcpEndpoint(**facts)


class NexBotJogTest(unittest.TestCase):
    def setUp(self):
        self.server = FakeController()
        self.server.replies[0x9512] = [
            build_frame(
                0x9513,
                {
                    "channel": 1,
                    "robot": 1,
                    "replyData": {
                        "realPosUCS": list(self.server.current_ucs),
                        "realPosMCS": list(self.server.current_ucs),
                        "realPosACS": [0.0] * 6,
                    },
                    "robot": 1,
                },
            )
        ]
        self.server.replies[0x2002] = [
            build_frame(CMD_SERVO_RESPOND,
                        {"mode": 0, "robot": 1, "status": 3}),
        ]
        # 0x3602 由 FakeController 按 self.dout 实况作答(见 _apply_dout)。
        for command in (CMD_MOVJ, CMD_MOVL, CMD_GO_HOME, CMD_GO_RESET_POSITION):
            self.server.replies[command] = [MOTION_STARTED]

    def tearDown(self):
        self.server.close()

    def _jog(self, **overrides):
        return NexBotTcpJog(endpoint(self.server, **overrides))

    def test_gripper_open_sends_dout_15_16(self):
        jog = self._jog()
        jog.gripper(True)
        jog.close()
        time.sleep(0.2)  # allow the fake server thread to record frames
        dout = [item for item in self.server.received if item[0] == CMD_DOUT_SET]
        self.assertEqual(
            [item[1] for item in dout],
            [{"port": 15, "status": 0}, {"port": 16, "status": 1}],
        )

    def test_gripper_close_sends_dout_15_16(self):
        jog = self._jog()
        jog.gripper(False)
        jog.close()
        time.sleep(0.2)  # allow the fake server thread to record frames
        dout = [item for item in self.server.received if item[0] == CMD_DOUT_SET]
        self.assertEqual(
            [item[1] for item in dout],
            [{"port": 15, "status": 1}, {"port": 16, "status": 0}],
        )

    def test_gripper_state_reads_back(self):
        jog = self._jog()
        self.assertEqual(jog.gripper_state(), (1, 0))
        jog.close()

    def test_step_moves_along_user_axis(self):
        jog = self._jog()
        jog.step(0, 20.0)
        jog.close()
        time.sleep(0.2)
        movl = [item for item in self.server.received if item[0] == CMD_MOVL]
        self.assertEqual(len(movl), 1)
        payload = movl[0][1]
        self.assertEqual(payload["coord"], 3)
        self.assertEqual(payload["acc"], 10)
        self.assertEqual(payload["dec"], 10)
        self.assertEqual(payload["vel"], 50)
        self.assertAlmostEqual(payload["pos"][0], 720.0, places=4)
        self.assertAlmostEqual(payload["pos"][1], 100.0, places=4)

    def test_step_negative_axis(self):
        jog = self._jog()
        jog.step(2, -5.0)
        jog.close()
        time.sleep(0.2)
        movl = [item for item in self.server.received if item[0] == CMD_MOVL]
        self.assertAlmostEqual(movl[0][1]["pos"][2], 395.0, places=4)

    def test_go_home_command(self):
        jog = self._jog()
        jog.go_home()
        jog.close()
        time.sleep(0.2)
        frames = [item for item in self.server.received if item[0] == 0x3002]
        self.assertEqual(frames[0][1], {"robot": 1, "type": 0})

    def test_go_reset_position_command(self):
        jog = self._jog()
        jog.go_reset_position()
        jog.close()
        time.sleep(0.2)
        frames = [item for item in self.server.received if item[0] == 0x3007]
        self.assertEqual(frames[0][1], {"robot": 1})

    def test_emergency_stop_command(self):
        jog = self._jog()
        jog.emergency_stop()
        jog.close()
        time.sleep(0.2)
        frames = [item for item in self.server.received if item[0] == 0x2314]
        self.assertEqual(frames[0][1], {"robot": 1})

    def test_current_pose_matches_user_frame(self):
        jog = self._jog()
        xyz, abc = jog.current_pose()
        jog.close()
        self.assertTrue(np.allclose(xyz, [700.0, 100.0, 400.0], atol=1e-6))

    def test_reconnect_uses_a_new_controller_without_deadlock(self):
        instances = []

        class ReconnectingController:
            def __init__(self, _endpoint):
                self.index = len(instances)
                self.closed = False
                instances.append(self)

            def read_state(self):
                if self.index == 0:
                    raise ControllerConnectionError("old connection dropped")
                return SimpleNamespace(base_from_gripper=np.eye(4))

            def close(self):
                self.closed = True

        with patch(
            "competition_pipeline.nexbot_jog.NexBotTcpRobotController",
            ReconnectingController,
        ):
            jog = NexBotTcpJog(object())
            jog.RETRY_WAIT_S = 0.0
            xyz, _abc = jog.current_pose()
            jog.close()

        self.assertEqual(len(instances), 2)
        self.assertTrue(instances[0].closed)
        self.assertTrue(np.allclose(xyz, [0.0, 0.0, 0.0]))

    def test_step_retry_resends_the_same_absolute_target(self):
        instances = []
        targets = []

        class AmbiguousMoveController:
            def __init__(self, _endpoint):
                self.index = len(instances)
                instances.append(self)

            def servo_status(self):
                return 3

            def read_state(self):
                matrix = np.eye(4)
                matrix[0, 3] = 0.0 if self.index == 0 else 0.01
                return SimpleNamespace(base_from_gripper=matrix)

            def move_to(self, target, speed_scale):
                targets.append(np.asarray(target).copy())
                if self.index == 0:
                    raise ControllerConnectionError("ambiguous MOVL disconnect")

            def close(self):
                pass

        with patch(
            "competition_pipeline.nexbot_jog.NexBotTcpRobotController",
            AmbiguousMoveController,
        ):
            jog = NexBotTcpJog(object())
            jog.RETRY_WAIT_S = 0.0
            jog.step(0, 10.0)
            jog.close()

        self.assertEqual(len(targets), 2)
        self.assertAlmostEqual(targets[0][0, 3], 0.01)
        self.assertTrue(np.array_equal(targets[0], targets[1]))

    def test_emergency_stop_bypasses_an_active_motion(self):
        motion_started = threading.Event()
        release_motion = threading.Event()
        stop_sent = threading.Event()

        class BlockingMoveController:
            def __init__(self, _endpoint):
                self.motion = SimpleNamespace(close=lambda: None)

            def servo_status(self):
                return 3

            def read_state(self):
                return SimpleNamespace(base_from_gripper=np.eye(4))

            def move_to(self, _target, speed_scale):
                motion_started.set()
                release_motion.wait(2.0)

            def stop(self):
                stop_sent.set()

            def close(self):
                release_motion.set()

        with patch(
            "competition_pipeline.nexbot_jog.NexBotTcpRobotController",
            BlockingMoveController,
        ):
            jog = NexBotTcpJog(object())
            worker = threading.Thread(target=jog.step, args=(0, 10.0))
            worker.start()
            self.assertTrue(motion_started.wait(0.5))
            jog.emergency_stop()
            self.assertTrue(stop_sent.wait(0.2))
            release_motion.set()
            worker.join(1.0)
            jog.close()

        self.assertFalse(worker.is_alive())

    def test_keepalive_recovers_without_stopping_itself(self):
        instances = []

        class KeepaliveController:
            def __init__(self, _endpoint):
                self.index = len(instances)
                instances.append(self)

            def servo_status(self):
                if self.index == 0:
                    raise ControllerConnectionError("idle connection dropped")
                return 3

            def close(self):
                pass

        with patch(
            "competition_pipeline.nexbot_jog.NexBotTcpRobotController",
            KeepaliveController,
        ):
            jog = NexBotTcpJog(object(), keepalive_s=0.01)
            deadline = time.monotonic() + 1.0
            while len(instances) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(len(instances), 2)
            self.assertFalse(jog._keepalive_stop.is_set())
            jog.close()


class RecordingController:
    """A stand-in controller that logs the order of protocol-level calls."""

    def __init__(self, _endpoint=None, pose=None, servo_sequence=None):
        self.calls = []
        self.targets = []
        self.servo_sequence = list(servo_sequence or [3])
        matrix = np.eye(4)
        if pose is not None:
            matrix = np.asarray(pose, dtype=np.float64)
        self.pose = matrix
        self.motion = SimpleNamespace(close=lambda: None)

    def servo_status(self):
        self.calls.append("servo_status")
        value = self.servo_sequence[0]
        if len(self.servo_sequence) > 1:
            self.servo_sequence.pop(0)
        return value

    def enable_servo(self):
        self.calls.append("enable_servo")
        self.servo_sequence = [3]
        return 3

    def read_state(self):
        self.calls.append("read_state")
        return SimpleNamespace(base_from_gripper=self.pose.copy())

    def move_to(self, target, speed_scale=0.1):
        self.calls.append("move_to")
        self.targets.append(np.asarray(target, dtype=np.float64).copy())
        self.pose = np.asarray(target, dtype=np.float64).copy()

    def go_reset_position(self):
        self.calls.append("go_reset_position")

    def go_home(self):
        self.calls.append("go_home")

    def close(self):
        pass


def _pose_at(xyz_mm, abc_rad):
    return transform_from_inexbot_abc(
        np.asarray(xyz_mm, dtype=float) / 1000.0,
        np.asarray(abc_rad, dtype=float),
    )


class AbcUnitGuardTest(unittest.TestCase):
    """度数当弧度 —— 2026-08-22 摔臂的直接成因。"""

    def test_radian_values_pass(self):
        for abc in ([0.0, 0.0, 0.0], [3.1044, 0.2402, -3.1415], [-3.14, 1.5, 3.14]):
            check_abc_is_radians(abc)

    def test_the_exact_field_readback_that_crashed_the_arm_is_rejected(self):
        # 控制器日志 16:56:16 那条 MOVL 的姿态, 换成度数就是下面三个数。
        with self.assertRaises(ValueError) as ctx:
            check_abc_is_radians([177.8698, 13.7624, -179.9942])
        self.assertIn("角度制", str(ctx.exception))

    def test_only_one_axis_out_of_range_is_enough(self):
        with self.assertRaises(ValueError):
            check_abc_is_radians([0.1, 13.76, 0.2])

    def test_non_finite_and_wrong_length_are_rejected(self):
        with self.assertRaises(ValueError):
            check_abc_is_radians([0.0, 0.0, float("nan")])
        with self.assertRaises(ValueError):
            check_abc_is_radians([0.0, 0.0])

    def test_boundary_is_max_abc_rad(self):
        check_abc_is_radians([MAX_ABC_RAD, 0.0, 0.0])
        with self.assertRaises(ValueError):
            check_abc_is_radians([MAX_ABC_RAD + 1e-6, 0.0, 0.0])


class MoveToUcsGuardTest(unittest.TestCase):
    """move_to_ucs 的三道闸门 + 使能前置。"""

    def _jog(self, controller):
        # 连接是惰性建立的：patch 必须在整个测试期间保持生效, 不能只包住构造。
        patcher = patch(
            "competition_pipeline.nexbot_jog.NexBotTcpRobotController",
            lambda _endpoint: controller,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        jog = NexBotTcpJog(object())
        jog.RETRY_WAIT_S = 0.0
        self.addCleanup(jog.close)
        return jog

    def test_degree_valued_abc_never_reaches_the_controller(self):
        controller = RecordingController()
        jog = self._jog(controller)
        with self.assertRaises(ValueError):
            jog.move_to_ucs([0.0, 0.0, 60.0], [177.8698, 13.7624, -179.9942])
        jog.close()
        # 闸门在读位姿之前就拦下了，一次通信都没发生。
        self.assertEqual(controller.calls, [])
        self.assertEqual(controller.targets, [])

    def test_large_orientation_step_is_rejected(self):
        """摔臂那条指令的形状：XYZ 只走 30mm，姿态却要转 119.6°。"""
        start = _pose_at([5.0, 0.0, 60.0], [3.104412961019, 0.240200009301, -3.141491526695])
        controller = RecordingController(pose=start)
        jog = self._jog(controller)
        with self.assertRaises(RuntimeError) as ctx:
            jog.move_to_ucs(
                [35.0, 0.0, 60.0],
                [1.9405550572349954, 1.196135558646883, 2.2180637457374814],
            )
        jog.close()
        message = str(ctx.exception)
        self.assertIn("119", message)          # 实际夹角
        self.assertNotIn("move_to", controller.calls)

    def test_long_translation_is_rejected(self):
        start = _pose_at([0.0, 0.0, 60.0], [0.0, 0.0, 0.0])
        controller = RecordingController(pose=start)
        jog = self._jog(controller)
        with self.assertRaises(RuntimeError) as ctx:
            jog.move_to_ucs([900.0, 0.0, 60.0], [0.0, 0.0, 0.0])
        jog.close()
        self.assertIn("400mm", str(ctx.exception))
        self.assertNotIn("move_to", controller.calls)

    def test_servo_is_enabled_before_the_motion_is_sent(self):
        start = _pose_at([0.0, 0.0, 60.0], [0.0, 0.0, 0.0])
        controller = RecordingController(pose=start, servo_sequence=[1, 3])
        jog = self._jog(controller)
        jog.move_to_ucs([0.0, 0.0, 70.0], [0.0, 0.0, 0.0], tolerance_mm=1.0)
        jog.close()
        self.assertIn("enable_servo", controller.calls)
        self.assertLess(
            controller.calls.index("enable_servo"),
            controller.calls.index("move_to"),
            "0x2311 必须在运动指令之前",
        )

    def test_servo_that_refuses_to_run_aborts_before_moving(self):
        start = _pose_at([0.0, 0.0, 60.0], [0.0, 0.0, 0.0])
        controller = RecordingController(pose=start, servo_sequence=[1])
        controller.enable_servo = lambda: controller.calls.append("enable_servo")
        jog = self._jog(controller)
        with self.assertRaises(RuntimeError) as ctx:
            jog.move_to_ucs([0.0, 0.0, 70.0], [0.0, 0.0, 0.0])
        jog.close()
        self.assertIn("伺服未能进入运行态", str(ctx.exception))
        self.assertNotIn("move_to", controller.calls)

    def test_arrival_check_also_covers_orientation(self):
        """位置到了但姿态没到, 不能报成功。"""
        start = _pose_at([0.0, 0.0, 60.0], [0.0, 0.0, 0.0])
        controller = RecordingController(pose=start)

        def _lands_with_wrong_orientation(target, speed_scale=0.1):
            controller.calls.append("move_to")
            landed = np.asarray(target, dtype=np.float64).copy()
            landed[:3, :3] = _pose_at([0, 0, 0], [0.0, 0.0, 0.35])[:3, :3]
            controller.pose = landed

        controller.move_to = _lands_with_wrong_orientation
        jog = self._jog(controller)
        with self.assertRaises(RuntimeError) as ctx:
            jog.move_to_ucs([0.0, 0.0, 65.0], [0.0, 0.0, 0.0],
                            tolerance_mm=1.0, rotation_tolerance_deg=3.0)
        jog.close()
        self.assertIn("姿态偏差", str(ctx.exception))

    def test_a_dropped_connection_does_not_silently_resend_the_motion(self):
        """断链是模糊状态：控制器可能已经开始动了，重发会让臂再动一次。"""
        start = _pose_at([0.0, 0.0, 60.0], [0.0, 0.0, 0.0])
        controller = RecordingController(pose=start)
        sent = []

        def _always_drops(target, speed_scale=0.1):
            sent.append(np.asarray(target).copy())
            raise ControllerConnectionError("6001 dropped mid-MOVL")

        controller.move_to = _always_drops
        jog = self._jog(controller)
        with self.assertRaises(ControllerConnectionError) as ctx:
            jog.move_to_ucs([0.0, 0.0, 65.0], [0.0, 0.0, 0.0])
        jog.close()
        self.assertEqual(len(sent), 1, "传送不允许断链自动重发")
        self.assertIn("回读当前", str(ctx.exception))


class ResetAndHomeGuardTest(unittest.TestCase):
    def _jog(self, controller):
        # 连接是惰性建立的：patch 必须在整个测试期间保持生效, 不能只包住构造。
        patcher = patch(
            "competition_pipeline.nexbot_jog.NexBotTcpRobotController",
            lambda _endpoint: controller,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        jog = NexBotTcpJog(object())
        jog.RETRY_WAIT_S = 0.0
        self.addCleanup(jog.close)
        return jog

    def test_go_reset_position_enables_the_servo_first(self):
        controller = RecordingController(servo_sequence=[1, 3])
        jog = self._jog(controller)
        jog.go_reset_position()
        jog.close()
        self.assertEqual(
            controller.calls,
            ["servo_status", "enable_servo", "servo_status", "go_reset_position"],
        )

    def test_go_home_enables_the_servo_first(self):
        controller = RecordingController(servo_sequence=[1, 3])
        jog = self._jog(controller)
        jog.go_home()
        jog.close()
        self.assertLess(
            controller.calls.index("enable_servo"),
            controller.calls.index("go_home"),
        )

    def test_already_running_servo_is_not_re_enabled(self):
        controller = RecordingController(servo_sequence=[3])
        jog = self._jog(controller)
        jog.go_reset_position()
        jog.close()
        self.assertNotIn("enable_servo", controller.calls)


class GripperReadbackTest(unittest.TestCase):
    """夹爪走 0x3601 DOUT，拿不到 0x3D03 那种"真的动了"的确认。

    2026-08-22 现场"夹爪照常开合、机械臂纹丝不动"正是因为这条码路一直是通的，
    界面于是报"✅ 完成"。DOUT 回读是这条路径上唯一的客观证据。
    """

    def setUp(self):
        self.server = FakeController()
        self.server.replies[0x2002] = [
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 3}),
        ]

    def tearDown(self):
        self.server.close()

    def _jog(self):
        jog = NexBotTcpJog(endpoint(self.server))
        self.addCleanup(jog.close)
        return jog

    def test_a_matching_readback_reports_verified(self):
        jog = self._jog()
        self.assertEqual(jog.gripper(True), (True, ""))
        self.assertEqual(tuple(self.server.dout[14:16]), (0, 1))
        self.assertEqual(jog.gripper(False), (True, ""))
        self.assertEqual(tuple(self.server.dout[14:16]), (1, 0))

    def test_a_coil_that_never_moves_is_reported_not_silently_accepted(self):
        """线圈没动作 -> 抛错。绝不能让"没夹住"被当成"夹住了"。"""
        self.server.dout_ignores_writes = True
        jog = self._jog()
        with self.assertRaises(RuntimeError) as ctx:
            jog.gripper(True)
        message = str(ctx.exception)
        self.assertIn("回读不符", message)
        self.assertIn("(0,1)", message)

    def test_an_unreadable_dout_is_reported_as_unverified_not_as_failure(self):
        """0x3603 在本固件未经现场验证：读不到就标注"未回读"，不判死。

        用一个没验证过的查询去否决一个验证过的动作，才是真正的自伤。
        """
        self.server.replies[CMD_DOUT_QUERY] = []      # 查询永不作答
        jog = self._jog()
        ok, detail = jog.gripper(True)
        self.assertTrue(ok)
        self.assertIn("未回读", detail)
        self.assertEqual(jog.last_gripper_verify, (ok, detail))
        # 动作本身照发不误
        dout = [item for item in self.server.received if item[0] == CMD_DOUT_SET]
        self.assertEqual([item[1] for item in dout],
                         [{"port": 15, "status": 0}, {"port": 16, "status": 1}])

    def test_verification_can_be_skipped_explicitly(self):
        self.server.dout_ignores_writes = True
        jog = self._jog()
        self.assertEqual(jog.gripper(True, verify=False), (True, ""))


class EmergencyStopPreemptionTest(unittest.TestCase):
    """6001 是单客户端端口，整个 jog 共用一条 socket。

    旧实现绕过 ``_lock`` 直发 0x2314，与正在收发帧的工作线程并发写同一个
    socket：两个 sendall 交错 -> 急停帧插进另一帧中间 -> 控制器按长度/CRC
    解析后**两帧都丢弃**。最需要它工作的时候它悄无声息地失效。
    """

    def _jog(self, controller):
        patcher = patch(
            "competition_pipeline.nexbot_jog.NexBotTcpRobotController",
            lambda _endpoint: controller,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        jog = NexBotTcpJog(object())
        jog.RETRY_WAIT_S = 0.0
        jog.ESTOP_PREEMPT_WAIT_S = 0.05
        self.addCleanup(jog.close)
        return jog

    def test_estop_preempts_a_worker_that_is_holding_the_transaction_lock(self):
        controller = RecordingController()
        controller.stop = lambda: controller.calls.append("stop")
        jog = self._jog(controller)
        jog._lock.acquire()                       # 模拟工作线程正持锁收发
        try:
            jog.emergency_stop()                  # 必须抢占，不能干等
        finally:
            jog._lock.release()
        self.assertIn("stop", controller.calls)

    def test_estop_does_not_wait_forever_and_clears_its_flag(self):
        controller = RecordingController()
        controller.stop = lambda: controller.calls.append("stop")
        jog = self._jog(controller)
        jog.emergency_stop()
        self.assertIn("stop", controller.calls)
        self.assertFalse(jog._estop_pending.is_set(),
                         "急停结束后必须清标志，否则后续动作全部无法重连")

    def test_a_worker_stops_retrying_once_an_estop_is_in_flight(self):
        """急停已发出时，工作线程不能回头抢 6001 的单客户端槽位。"""
        controller = RecordingController()
        jog = self._jog(controller)
        jog._estop_pending.set()
        self.addCleanup(jog._estop_pending.clear)

        def _boom(_controller):
            raise ControllerConnectionError("connection closed mid-frame")

        with self.assertRaises(ControllerConnectionError) as ctx:
            jog._run(_boom)
        self.assertIn("急停", str(ctx.exception))


class PoseUnitApiTest(unittest.TestCase):
    def test_current_pose_is_degrees_and_current_pose_rad_is_radians(self):
        abc = [3.104412961019, 0.240200009301, -3.141491526695]
        controller = RecordingController(pose=_pose_at([5.0, 0.0, 60.0], abc))
        with patch("competition_pipeline.nexbot_jog.NexBotTcpRobotController",
                   lambda _endpoint: controller):
            jog = NexBotTcpJog(object())
            xyz_deg, abc_deg = jog.current_pose()
            xyz_rad, abc_rad = jog.current_pose_rad()
            jog.close()
        self.assertTrue(np.allclose(xyz_deg, xyz_rad))
        self.assertTrue(np.allclose(np.radians(abc_deg), abc_rad, atol=1e-9))
        self.assertTrue(np.allclose(abc_rad, abc, atol=1e-9))
        # 角度制的返回值直接喂给 move_to_ucs 必须被拦住。
        with self.assertRaises(ValueError):
            check_abc_is_radians(abc_deg)


class KeepaliveHealthTest(unittest.TestCase):
    def test_keepalive_records_a_servo_that_dropped_out_of_run_state(self):
        controller = RecordingController(servo_sequence=[1])
        with patch("competition_pipeline.nexbot_jog.NexBotTcpRobotController",
                   lambda _endpoint: controller):
            jog = NexBotTcpJog(object(), keepalive_s=0.01)
            deadline = time.monotonic() + 1.0
            while jog.servo_dropped_count == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            status, error, dropped = jog.health()
            jog.close()
        self.assertEqual(status, 1)
        self.assertIsNone(error)
        self.assertGreater(dropped, 0)

    def test_keepalive_records_a_connection_error(self):
        class Failing(RecordingController):
            def servo_status(self):
                raise ControllerConnectionError("6001 gone")

        controller = Failing()
        with patch("competition_pipeline.nexbot_jog.NexBotTcpRobotController",
                   lambda _endpoint: controller):
            jog = NexBotTcpJog(object(), keepalive_s=0.01)
            deadline = time.monotonic() + 1.0
            while jog.last_keepalive_error is None and time.monotonic() < deadline:
                time.sleep(0.01)
            status, error, _dropped = jog.health()
            jog.close()
        self.assertIsNone(status)
        self.assertIn("6001 gone", error or "")


if __name__ == "__main__":
    unittest.main()
