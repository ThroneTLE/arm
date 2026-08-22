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
from competition_pipeline.nexbot_jog import NexBotTcpJog
from competition_pipeline.nexbot_tcp import (
    NexBotTcpEndpoint,
    build_frame,
    CMD_DOUT_SET,
    CMD_DOUT_QUERY,
    CMD_DOUT_QUERY_REPLY,
    CMD_MOVL,
    CMD_SERVO_RESPOND,
    ControllerConnectionError,
)


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
        self.server.replies[0x3602] = [
            build_frame(
                CMD_DOUT_QUERY_REPLY,
                {"status": [0] * 14 + [1, 0], "robot": 1},
            )
        ]

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


if __name__ == "__main__":
    unittest.main()
