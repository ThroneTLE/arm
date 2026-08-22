#!/usr/bin/env python3
"""Tests for the live TCP pose source used by the hand-eye page."""

import socket
import threading
import time
import unittest

import numpy as np

from competition_pipeline.nexbot_tcp import (
    CMD_QUERY,
    CMD_QUERY_REPLY,
    NexBotTcpEndpoint,
    build_frame,
    read_frame,
)
from competition_pipeline.tcp_pose import (
    NexBotTcpPoseSource,
    pose_endpoint_from_config,
)


class FakeController:
    """One TCP listener; replies to 0x9512 with a documented-shaped 0x9513."""

    def __init__(self, reply=None):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(5)
        self.port = self.server.getsockname()[1]
        self.reply = reply or self._default_reply()
        self.received = []
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @staticmethod
    def _default_reply():
        return build_frame(
            CMD_QUERY_REPLY,
            {"channel": 1, "robot": 1,
             "replyData": {
                 "realPosMCS": [863.7, -56.4, 922.96, 3.14159265359, 0.0, 0.065227250055],
                 "realPosACS": [-3.737, 0.0, 0.0, 0.0, 0.0, 0.0],
                 "timestamp": [1759052356, 264138361],
             }, "robot": 1},
        )

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
            conn.sendall(self.reply)

    def close(self):
        self._stop = True
        try:
            self.server.close()
        except OSError:
            pass


class PoseEndpointTest(unittest.TestCase):
    def test_defaults_from_config(self):
        endpoint = pose_endpoint_from_config({
            "nexbot_tcp": {"host": "192.168.1.20"},
        })
        self.assertEqual(endpoint.host, "192.168.1.20")
        self.assertEqual(endpoint.port_motion, 6000)
        self.assertEqual(endpoint.port_state, 7000)
        self.assertEqual(endpoint.robot, 1)

    def test_missing_section_is_rejected_without_host(self):
        # A missing section leaves host empty; the endpoint refuses it so the
        # UI surfaces the connection error instead of connecting nowhere.
        with self.assertRaises(ValueError):
            pose_endpoint_from_config({})

    def test_empty_host_is_rejected(self):
        with self.assertRaises(ValueError):
            pose_endpoint_from_config({})


class PoseSourceTest(unittest.TestCase):
    def setUp(self):
        self.server = FakeController()

    def tearDown(self):
        self.server.close()

    def _source(self):
        return NexBotTcpPoseSource(NexBotTcpEndpoint(
            host="127.0.0.1",
            port_motion=self.server.port,
            port_state=self.server.port,
            io_timeout_s=0.5,
            connect_timeout_s=0.5,
        ))

    def test_read_returns_mm_and_deg(self):
        from competition_pipeline.geometry import transform_from_inexbot_abc_mm
        source = self._source()
        xyz_mm, abc_deg = source.read()
        source.close()
        self.assertTrue(np.allclose(xyz_mm, [863.7, -56.4, 922.96], atol=1e-9))
        # The returned mm/deg re-builds the same 4x4 transform the wire pose
        # encoded, under the controller-native A/B/C convention (intrinsic
        # X'Y'Z' -> R = Rx(A) Ry(B) Rz(C)); field-verified 2026-08-22.
        rebuilt = transform_from_inexbot_abc_mm(xyz_mm, abc_deg)
        a, b, c = 3.14159265359, 0.0, 0.065227250055
        expected = transform_from_inexbot_abc_mm(
            [863.7, -56.4, 922.96], np.degrees([a, b, c])
        )
        self.assertTrue(np.allclose(rebuilt, expected, atol=1e-6))
        # the wire saw exactly one 0x9512 state query
        self.assertEqual(self.server.received[0][0], CMD_QUERY)

    def test_read_after_reconnect_still_works(self):
        source = self._source()
        source.connect()
        first = source.read()
        source.close()
        second_source = self._source()
        second = second_source.read()
        second_source.close()
        self.assertTrue(np.allclose(first[0], second[0], atol=1e-9))

    def test_shared_poller_skips_instead_of_queueing_behind_motion(self):
        class BusyJog:
            def __init__(self):
                self._lock = threading.Lock()

            def _run_locked(self, action):
                raise AssertionError("busy poll must not start a state query")

        jog = BusyJog()
        source = NexBotTcpPoseSource(object(), jog=jog)
        jog._lock.acquire()
        try:
            self.assertIsNone(source.try_read())
        finally:
            jog._lock.release()


if __name__ == "__main__":
    unittest.main()
