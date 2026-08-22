#!/usr/bin/env python3
"""Offline tests for the NexBot jog/gripper controls (fake controller)."""

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


if __name__ == "__main__":
    unittest.main()
