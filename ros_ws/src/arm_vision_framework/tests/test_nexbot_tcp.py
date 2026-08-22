#!/usr/bin/env python3
"""NexBot JSON-over-TCP protocol adapter tests against a fake controller."""

import json
import socket
import struct
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from arm_vision_framework.adapters.inexbot_modbus import (
    ControllerConnectionError,
    ControllerProtocolError,
    ControllerTimeout,
    InexbotPoint,
)
from arm_vision_framework.adapters.nexbot_tcp import (
    CMD_ALARM,
    CMD_EMERGENCY_STOP,
    CMD_ENABLE,
    CMD_GO_HOME,
    CMD_GO_RESET_POSITION,
    CMD_MOVJ,
    CMD_MOVL,
    CMD_PROGRAM_STATUS,
    CMD_QUERY,
    CMD_QUERY_REPLY,
    CMD_SERVO_INQUIRE,
    CMD_SERVO_RESPOND,
    NexBotTcpEndpoint,
    NexBotTcpRobotController,
    _frame_message,
    build_frame,
    read_frame,
)
from arm_vision_framework.transforms import transform_from_inexbot_abc


#: 真控制器接受一条运动指令后, 会在 6001 上推 ``0x3D03 status=2``(开始运动).
#: 假控制器必须照做, 否则测试跑的是一个现实中不存在的控制器。
MOTION_STARTED = build_frame(CMD_PROGRAM_STATUS, {"robot": 1, "status": 2})
MOTION_COMMANDS = (CMD_MOVJ, CMD_MOVL, CMD_GO_HOME, CMD_GO_RESET_POSITION)


class FakeController:
    """One TCP listener; every accepted connection gets a handler thread."""

    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(5)
        self.port = self.server.getsockname()[1]
        self.received = []
        # command -> list of already-encoded frames to send on receipt
        self.replies = {}
        for command in MOTION_COMMANDS:
            self.replies[command] = [MOTION_STARTED]
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        conn.settimeout(5.0)
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
    )
    facts.update(overrides)
    return NexBotTcpEndpoint(**facts)


class BuildFrameTest(unittest.TestCase):
    def test_matches_official_query_example(self):
        """The published query frame must be produced byte-for-byte."""
        frame = build_frame(
            0x1E00,
            {"channel": 1, "robot": 1, "mode": 0, "queryType": ["realPosACS"]},
        )
        expected = bytes.fromhex(
            "4e66003b1e007b226368616e6e656c223a312c22726f626f74223a312c226d6f6465223a30"
            "2c22717565727954797065223a5b227265616c506f73414353225d7d9090fa38"
        )
        self.assertEqual(frame, expected)

    def test_empty_data_frame(self):
        import zlib
        frame = build_frame(0x2314, {"robot": 1})
        self.assertEqual(frame[:4], b"\x4e\x66\x00\x0b")
        expected_crc = zlib.crc32(frame[2:-4]) & 0xFFFFFFFF
        self.assertEqual(frame[-4:], struct.pack(">I", expected_crc))


class NexBotControllerTest(unittest.TestCase):
    def setUp(self):
        self.server = FakeController()

    def tearDown(self):
        self.server.close()

    def _controller(self, **overrides):
        return NexBotTcpRobotController(endpoint(self.server, **overrides))

    def test_move_j_sends_0x4501(self):
        controller = self._controller()
        point = InexbotPoint(
            name="P0001", coordinate_system=0, angle_unit=0, shape=1,
            tool_id=0, user_id=0, axes=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0),
        )
        controller.move_j([point], speed_scale=0.5)
        controller.close()
        command, data = self.server.received[0]
        self.assertEqual(command, CMD_MOVJ)
        self.assertEqual(data["robot"], 1)
        self.assertEqual(data["vel"], 50)
        self.assertEqual(data["acc"], 10)
        self.assertEqual(data["dec"], 10)
        self.assertEqual(data["coord"], 0)
        self.assertEqual(data["pos"], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0])

    def test_move_l_sends_0x4502_with_mm_and_radians(self):
        controller = self._controller()
        point = InexbotPoint(
            name="P0001", coordinate_system=1, angle_unit=1, shape=1,
            tool_id=0, user_id=0,
            axes=(100.0, 200.0, 300.0, 0.1, -0.2, 0.3, 0.0),
        )
        controller.move_l([point], speed_mm_s=30.0)
        controller.close()
        command, data = self.server.received[0]
        self.assertEqual(command, CMD_MOVL)
        self.assertEqual(data["vel"], 30)
        self.assertEqual(data["acc"], 10)
        self.assertEqual(data["dec"], 10)
        self.assertEqual(data["coord"], 1)
        self.assertEqual(data["pos"], [100.0, 200.0, 300.0, 0.1, -0.2, 0.3, 0.0])

    def test_move_j_waits_until_axes_are_still(self):
        reply = build_frame(
            CMD_QUERY_REPLY,
            {"channel": 1, "robot": 1, "replyData": {
                "realPosUCS": [700.0, 100.0, 400.0, 0.1, 0.2, 0.3],
                "realPosMCS": [700.0, 100.0, 400.0, 0.1, 0.2, 0.3],
                "axisVel": [0.01, -0.01, 0.0, 0.01, 0.0, -0.01],
                "timestamp": [1759052356, 264138361],
            }, "robot": 1},
        )
        self.server.replies[CMD_QUERY] = [reply]
        controller = self._controller(wait_for_finish=True)
        point = InexbotPoint(
            name="P0001", coordinate_system=0, angle_unit=0, shape=1,
            tool_id=0, user_id=0, axes=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0),
        )
        controller.move_j([point], speed_scale=0.1)
        controller.close()
        self.assertEqual(self.server.received[0][0], CMD_MOVJ)
        queries = [item for item in self.server.received if item[0] == CMD_QUERY]
        self.assertGreaterEqual(len(queries), 2)

    def test_read_state_parses_real_pos_mcs(self):
        reply = build_frame(
            CMD_QUERY_REPLY,
            {"channel": 1, "robot": 1,
             "replyData": {
                 "realPosMCS": [863.7, -56.4, 922.96, 3.14159265359, 0.0, 0.065227250055],
                 "realPosACS": [-3.737, 0.0, 0.0, 0.0, 0.0, 0.0],
                 "timestamp": [1759052356, 264138361],
             }, "robot": 1},
        )
        self.server.replies[CMD_QUERY] = [reply]
        controller = self._controller()
        state = controller.read_state()
        controller.close()
        self.assertTrue(state.valid)
        # NexBot A/B/C are intrinsic X'Y'Z' (== fixed ZYX): R = Rx Ry Rz.
        # Field-verified 2026-08-22 on MOKA MR07S-930 / Inexbot C1102 (the
        # old fixed-XYZ order broke checkerboard hand-eye by ~190 mm).
        expected = transform_from_inexbot_abc(
            [0.8637, -0.0564, 0.92296], [3.14159265359, 0.0, 0.065227250055]
        )
        self.assertTrue(np.allclose(state.base_from_gripper, expected, atol=1e-9))
        self.assertAlmostEqual(state.timestamp_s, 1759052356.2641383, places=6)

    def test_stop_sends_emergency_stop(self):
        controller = self._controller()
        controller.stop()
        controller.close()
        command, data = self.server.received[0]
        self.assertEqual(command, CMD_EMERGENCY_STOP)
        self.assertEqual(data, {"robot": 1})

    def test_closed_controller_cannot_lazily_reconnect(self):
        controller = self._controller()
        controller.close()
        with self.assertRaises(ControllerConnectionError):
            controller.read_state()

    def test_crc_mismatch_raises_protocol_error(self):
        good = build_frame(
            CMD_QUERY_REPLY,
            {"channel": 1, "robot": 1, "replyData": {"realPosMCS": [1, 1, 1, 0, 0, 0]}, "robot": 1},
        )
        bad = bytearray(good)
        bad[-1] ^= 0xFF
        self.server.replies[CMD_QUERY] = [bytes(bad)]
        controller = self._controller()
        with self.assertRaises(ControllerProtocolError):
            controller.read_state()
        controller.close()

    def test_error_frame_raises_with_message(self):
        self.server.replies[CMD_QUERY] = [
            build_frame(0x6020, {"message": "joint velocity limit exceeded"})
        ]
        controller = self._controller()
        with self.assertRaises(ControllerProtocolError) as ctx:
            controller.read_state()
        controller.close()
        self.assertIn("joint velocity limit", str(ctx.exception))

    def test_missing_reply_raises_timeout(self):
        controller = self._controller(io_timeout_s=0.2)
        with self.assertRaises(ControllerTimeout):
            controller.read_state()
        controller.close()

    def test_endpoint_validation(self):
        with self.assertRaises(ValueError):
            NexBotTcpEndpoint(host="")
        with self.assertRaises(ValueError):
            NexBotTcpEndpoint(host="127.0.0.1", external_axes=3)
        with self.assertRaises(ValueError):
            NexBotTcpEndpoint(host="127.0.0.1", robot=9)
        with self.assertRaises(ValueError):
            NexBotTcpEndpoint(host="127.0.0.1", motion_ack_timeout_s=-1.0)


class MotionAckTest(unittest.TestCase):
    """指令被拒时必须报错, 绝不能报成功。

    2026-08-22 现场三种被拒签名全部覆盖 —— 旧代码对三种都返回"运动完成"。
    """

    def setUp(self):
        self.server = FakeController()

    def tearDown(self):
        self.server.close()

    def _controller(self, **overrides):
        return NexBotTcpRobotController(endpoint(self.server, **overrides))

    @staticmethod
    def _point():
        return InexbotPoint(
            name="P0001", coordinate_system=3, angle_unit=1, shape=1,
            tool_id=1, user_id=1, axes=(1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.0),
        )

    def test_silently_ignored_command_raises_instead_of_reporting_success(self):
        """签名 3：控制器什么都不回。旧代码判"位姿没变 -> 运动完成"。"""
        self.server.replies[CMD_MOVL] = []
        controller = self._controller(motion_ack_timeout_s=0.4)
        with self.assertRaises(ControllerTimeout) as ctx:
            controller.move_l([self._point()], speed_mm_s=30.0)
        controller.close()
        self.assertIn("0x3D03", str(ctx.exception))

    def test_parameter_error_alarm_surfaces_the_real_message(self):
        """签名 1：0x2B03，真实文本在 ``data`` 键而不是 ``message``。"""
        self.server.replies[CMD_MOVL] = [
            build_frame(CMD_ALARM, {
                "code": 25530, "data": "指令[0x4502]参数错误",
                "kind": 2, "param": [17666, 17666, 2], "robot": 0,
            })
        ]
        controller = self._controller(motion_ack_timeout_s=1.0)
        with self.assertRaises(ControllerProtocolError) as ctx:
            controller.move_l([self._point()], speed_mm_s=30.0)
        controller.close()
        self.assertIn("指令[0x4502]参数错误", str(ctx.exception))

    def test_safety_gate_refusal_is_reported_with_the_field_hint(self):
        """签名 2：复位点安全闸门拒绝后把伺服下电, 6001 收到 0x2003 status 1。"""
        self.server.replies[CMD_GO_RESET_POSITION] = [
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 0}),
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 1}),
        ]
        controller = self._controller(motion_ack_timeout_s=1.0)
        with self.assertRaises(ControllerProtocolError) as ctx:
            controller.go_reset_position()
        controller.close()
        message = str(ctx.exception)
        self.assertIn("拒绝", message)
        self.assertIn("deviation=null", message)

    def test_servo_status_3_push_does_not_count_as_motion_started(self):
        """伺服还在使能 != 运动开始了。只有 0x3D03 status=2 才算。"""
        self.server.replies[CMD_MOVJ] = [
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 3}),
        ]
        controller = self._controller(motion_ack_timeout_s=0.4)
        with self.assertRaises(ControllerTimeout):
            controller.move_j([self._point()], speed_scale=0.1)
        controller.close()

    def test_program_status_0_does_not_count_as_motion_started(self):
        self.server.replies[CMD_MOVL] = [
            build_frame(CMD_PROGRAM_STATUS, {"robot": 1, "status": 0}),
        ]
        controller = self._controller(motion_ack_timeout_s=0.4)
        with self.assertRaises(ControllerTimeout):
            controller.move_l([self._point()], speed_mm_s=30.0)
        controller.close()

    def test_enable_servo_sees_the_final_state_not_the_first_push(self):
        """0x2311 先推 3, 安全闸门随后把它打回 1。必须读到 1 并报错。

        这就是现场"刚才打开了运动伺服但是瞬间关闭了"的机器可读版本。
        """
        self.server.replies[CMD_ENABLE] = [
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 3}),
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 0}),
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 1}),
        ]
        self.server.replies[CMD_SERVO_INQUIRE] = [
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 1}),
        ]
        controller = self._controller()
        with self.assertRaises(ControllerProtocolError) as ctx:
            controller.enable_servo(settle_s=0.05)
        controller.close()
        self.assertIn("status=1", str(ctx.exception))

    def test_enable_servo_accepts_a_servo_that_stays_running(self):
        self.server.replies[CMD_ENABLE] = [
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 3}),
        ]
        self.server.replies[CMD_SERVO_INQUIRE] = [
            build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 3}),
        ]
        controller = self._controller()
        self.assertEqual(controller.enable_servo(settle_s=0.05), 3)
        controller.close()


class FrameMessageTest(unittest.TestCase):
    """``str(None)`` == ``"None"`` is truthy -- the old extractor was dead code."""

    def test_data_key_wins_because_that_is_what_the_controller_sends(self):
        self.assertEqual(
            _frame_message({"code": 25530, "data": "指令[0x4502]参数错误"}),
            "指令[0x4502]参数错误",
        )

    def test_message_and_error_keys_still_work(self):
        self.assertEqual(_frame_message({"message": "joint limit"}), "joint limit")
        self.assertEqual(_frame_message({"error": "singularity"}), "singularity")

    def test_missing_text_falls_back_to_json_not_the_string_none(self):
        message = _frame_message({"code": 7, "robot": 1})
        self.assertNotEqual(message, "None")
        self.assertIn("code", message)

    def test_non_dict_payloads(self):
        self.assertEqual(_frame_message(None), "controller error")
        self.assertEqual(_frame_message(""), "controller error")
        self.assertEqual(_frame_message("boom"), "boom")


if __name__ == "__main__":
    unittest.main()
