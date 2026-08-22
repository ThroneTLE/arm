#!/usr/bin/env python3
"""Pipeline NexBot TCP bridge tests (SafeRobotController -> wire protocol)."""

import socket
import struct
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from competition_pipeline.control import SafeRobotController
from competition_pipeline.geometry import transform_from_xyz_rpy_mm
from competition_pipeline.nexbot_move import NexBotTcpMoveController
from competition_pipeline.nexbot_tcp import (
    CMD_EMERGENCY_STOP,
    CMD_GO_HOME,
    CMD_GO_RESET_POSITION,
    CMD_MOVJ,
    CMD_MOVL,
    CMD_PROGRAM_STATUS,
    CMD_QUERY,
    CMD_QUERY_REPLY,
    CMD_SERVO_INQUIRE,
    CMD_SERVO_RESPOND,
    ControllerProtocolError,
    NexBotTcpEndpoint,
    NexBotTcpRobotController,
    build_frame,
    read_frame,
)

#: 真控制器收到运动指令后会在 6001 推 ``0x3D03 status=2``(开始运动)。
#: 假控制器必须照做, 否则测的是一个现实中不存在的、"发了就算成功"的控制器。
MOTION_STARTED = build_frame(CMD_PROGRAM_STATUS, {"robot": 1, "status": 2})
MOTION_COMMANDS = (CMD_MOVJ, CMD_MOVL, CMD_GO_HOME, CMD_GO_RESET_POSITION)

#: 适配器在每条运动前查一次 0x2002 伺服状态(``_ensure_servo_enabled``)。
#: 假控制器答 3=运行, 否则测的是一个"不问使能就发运动"的旧控制器。
SERVO_RUNNING = build_frame(CMD_SERVO_RESPOND, {"mode": 0, "robot": 1, "status": 3})


class FakeController:
    """One TCP listener; every accepted connection gets a handler thread."""

    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(5)
        self.port = self.server.getsockname()[1]
        self.received = []
        self.replies = {}
        for command in MOTION_COMMANDS:
            self.replies[command] = [MOTION_STARTED]
        self.replies[CMD_SERVO_INQUIRE] = [SERVO_RUNNING]
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


def state_reply(xyz_mm=(100.0, 100.0, 200.0), rpy_rad=(0.0, 0.0, 0.0)):
    now = time.time()
    return build_frame(
        CMD_QUERY_REPLY,
        {"channel": 1, "robot": 1,
         "replyData": {
             "realPosMCS": [*xyz_mm, *rpy_rad],
             "realPosACS": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             "timestamp": [int(now), int((now - int(now)) * 1e9)],
         }, "robot": 1},
    )


def safety_config():
    return SimpleNamespace(data={"safety": {
        "dry_run": False,
        "allow_robot_motion": True,
        "maximum_speed_scale": 0.2,
        "workspace_min_mm": [-1000.0, -1000.0, -100.0],
        "workspace_max_mm": [2000.0, 2000.0, 2000.0],
        "maximum_robot_pose_age_s": 0.5,
        "maximum_single_step_mm": 50.0,
        "maximum_single_rotation_deg": 10.0,
    }})


class NexBotMoveTest(unittest.TestCase):
    def setUp(self):
        self.server = FakeController()
        self.server.replies[CMD_QUERY] = [state_reply()]

    def tearDown(self):
        self.server.close()

    def _bridge(self):
        controller = NexBotTcpRobotController(endpoint(self.server))
        return controller, NexBotTcpMoveController(controller)

    def test_move_tcp_reaches_wire_as_movl(self):
        _, bridge = self._bridge()
        target = transform_from_xyz_rpy_mm([105.0, 100.0, 200.0], [0.0, 0.0, 0.0])
        bridge.move_tcp(target, speed_scale=0.1)
        bridge.stop()
        bridge.close()
        movl = [item for item in self.server.received if item[0] == CMD_MOVL]
        self.assertEqual(len(movl), 1)
        command, data = movl[0]
        self.assertEqual(command, CMD_MOVL)
        self.assertEqual(data["vel"], 100)
        self.assertEqual(data["coord"], 1)
        self.assertEqual(data["pos"][:3], [105.0, 100.0, 200.0])
        self.assertEqual(self.server.received[-1][0], CMD_EMERGENCY_STOP)

    def test_latest_pose_maps_to_base_from_tcp(self):
        _, bridge = self._bridge()
        pose = bridge.latest_pose()
        bridge.close()
        self.assertTrue(np.allclose(pose.base_from_tcp[:3, 3], [0.1, 0.1, 0.2], atol=1e-9))
        self.assertGreater(pose.timestamp_s, 0.0)

    def test_stop_sends_emergency_stop(self):
        _, bridge = self._bridge()
        bridge.stop()
        bridge.close()
        command, data = self.server.received[-1]
        self.assertEqual(command, CMD_EMERGENCY_STOP)
        self.assertEqual(data, {"robot": 1})

    def test_safe_controller_to_wire_end_to_end(self):
        """Full pipeline chain: safety gate -> move controller -> 0x4502."""
        controller, bridge = self._bridge()
        robot = SafeRobotController(safety_config(), bridge)
        target = transform_from_xyz_rpy_mm([108.0, 100.0, 200.0], [0.0, 0.0, 0.0])
        robot.move_tcp(target, speed_scale=0.1)
        robot.stop()
        controller.close()
        movl = [item for item in self.server.received if item[0] == CMD_MOVL]
        self.assertEqual(len(movl), 1)
        self.assertEqual(movl[0][1]["pos"][:3], [108.0, 100.0, 200.0])

    def test_safe_controller_stays_fail_closed_without_config(self):
        controller, bridge = self._bridge()
        robot = SafeRobotController(SimpleNamespace(data={"safety": {}}), bridge)
        target = transform_from_xyz_rpy_mm([108.0, 100.0, 200.0], [0.0, 0.0, 0.0])
        with self.assertRaises(RuntimeError):
            robot.move_tcp(target, speed_scale=0.1)
        controller.close()
        self.assertEqual(len(self.server.received), 0)


if __name__ == "__main__":
    unittest.main()
