import socket
import struct
import threading
import unittest
from pathlib import Path

import yaml

from competition_pipeline.configuration import CompetitionConfig
from competition_pipeline.controller_tcp import (
    ConfiguredRemoteIo,
    ModbusExceptionResponse,
    ModbusProtocolError,
    ModbusTcpClient,
    TcpEndpoint,
    InexbotPoint,
    point_from_joint_degrees,
    shape_from_joint_degrees,
)


class _ModbusServer:
    def __init__(self, response_builder):
        self.response_builder = response_builder
        self.ready = threading.Event()
        self.error = None
        self.requests = []
        self._stop = threading.Event()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(0.2)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(2.0):
            raise RuntimeError("test server did not start")

    @staticmethod
    def _recv_exact(connection, size):
        result = bytearray()
        while len(result) < size:
            chunk = connection.recv(size - len(result))
            if not chunk:
                return None
            result.extend(chunk)
        return bytes(result)

    def _run(self):
        self.ready.set()
        connection = None
        try:
            while not self._stop.is_set():
                try:
                    connection, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if connection is None:
                return
            connection.settimeout(0.5)
            with connection:
                while not self._stop.is_set():
                    header = self._recv_exact(connection, 7)
                    if header is None:
                        return
                    transaction, protocol, length, unit = struct.unpack(">HHHB", header)
                    pdu = self._recv_exact(connection, length - 1)
                    if pdu is None:
                        return
                    self.requests.append((transaction, protocol, unit, pdu))
                    response = self.response_builder(transaction, unit, pdu)
                    if response is None:
                        return
                    # Fragment the response to prove that the client handles
                    # TCP stream boundaries rather than assuming one recv().
                    for index in range(0, len(response), 2):
                        try:
                            connection.sendall(response[index:index + 2])
                        except BrokenPipeError:
                            return
        except Exception as error:  # pragma: no cover - surfaced by tearDown
            self.error = error
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

    def close(self):
        self._stop.set()
        try:
            self.listener.close()
        except OSError:
            pass
        self.thread.join(2.0)
        if self.error is not None:
            raise self.error


def _frame(transaction, unit, pdu):
    return struct.pack(">HHHB", transaction, 0, len(pdu) + 1, unit) + pdu


class ControllerTcpTest(unittest.TestCase):
    def test_read_coils_and_write_registers_use_standard_mbap(self):
        def respond(transaction, unit, pdu):
            if pdu[0] == 1:
                self.assertEqual(pdu, b"\x01\x00\x10\x00\x0a")
                return _frame(transaction, unit, b"\x01\x02\x55\x01")
            if pdu[0] == 6:
                self.assertEqual(pdu, b"\x06\x00\x20\x12\x34")
                return _frame(transaction, unit, pdu)
            raise AssertionError("unexpected function {}".format(pdu[0]))

        server = _ModbusServer(respond)
        try:
            client = ModbusTcpClient(TcpEndpoint("127.0.0.1", server.port), unit_id=7)
            self.assertEqual(
                client.read_coils(0x10, 10),
                [True, False, True, False, True, False, True, False, True, False],
            )
            self.assertTrue(client.write_single_register(0x20, 0x1234))
            client.close()
            self.assertEqual([entry[2] for entry in server.requests], [7, 7])
        finally:
            server.close()

    def test_exception_response_is_exposed_without_guessing_retry(self):
        def respond(transaction, unit, pdu):
            return _frame(transaction, unit, bytes([pdu[0] | 0x80, 2]))

        server = _ModbusServer(respond)
        try:
            client = ModbusTcpClient(TcpEndpoint("127.0.0.1", server.port))
            with self.assertRaises(ModbusExceptionResponse) as context:
                client.read_holding_registers(0, 1)
            self.assertEqual(context.exception.function_code, 3)
            self.assertEqual(context.exception.exception_code, 2)
            client.close()
        finally:
            server.close()

    def test_transaction_mismatch_is_rejected(self):
        def respond(transaction, unit, pdu):
            return _frame((transaction + 1) & 0xFFFF, unit, bytes([pdu[0], 2, 0, 0]))

        server = _ModbusServer(respond)
        try:
            client = ModbusTcpClient(TcpEndpoint("127.0.0.1", server.port))
            with self.assertRaises(ModbusProtocolError):
                client.read_input_registers(0, 1)
            client.close()
        finally:
            server.close()

    def test_manual_point_shape_and_metadata_are_preserved(self):
        self.assertEqual(shape_from_joint_degrees([59, 69, 79, 89, 99, 109]), 7)
        self.assertEqual(shape_from_joint_degrees([-100, 0, 0, 0, 100, 0]), 3)
        self.assertEqual(shape_from_joint_degrees([-100, 0, -100, 0, 100, 0]), 1)
        # Axis 1/3/5 bits are 1/0/0, binary 100 = 4, then the manual adds 1.
        self.assertEqual(shape_from_joint_degrees([0, 0, 100, 0, 100, 0]), 5)
        self.assertEqual(shape_from_joint_degrees([0, 0, 0, 0, 0, 0]), 8)
        point = point_from_joint_degrees("P0002", [1, 2, 3, 4, 5, 6], tool_id=2, user_id=1)
        self.assertEqual(point.fields()[:7], ("P0002", 0, 0, point.shape, 2, 1, 0.0))
        self.assertEqual(len(point.fields()), 15)
        self.assertEqual(point.axes[-1], 0.0)
        with self.assertRaises(ValueError):
            InexbotPoint("P1", 1, 1, 1, 1, 1, [0, 0, 0])

    def test_disabled_controller_is_not_constructed(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "competition.yaml"
        config = CompetitionConfig(config_path)
        from competition_pipeline.controller_tcp import modbus_client_from_config

        self.assertIsNone(modbus_client_from_config(config))

    def test_configured_remote_io_does_not_hard_code_addresses(self):
        class FakeClient:
            def __init__(self):
                self.writes = []

            def write_single_coil(self, address, value):
                self.writes.append((address, value))
                return True

            def read_discrete_inputs(self, address, quantity):
                self.writes.append(("read", address, quantity))
                return [True]

        settings = {
            "controller": {
                "remote_io": {
                    "outputs": {"gripper_close": 321},
                    "inputs": {"gripper_done": 654},
                }
            }
        }
        fake = FakeClient()
        io = ConfiguredRemoteIo(fake, settings)
        self.assertTrue(io.set_output("gripper_close", True))
        self.assertTrue(io.read_input("gripper_done"))
        self.assertEqual(fake.writes, [(321, True), ("read", 654, 1)])
        with self.assertRaises(KeyError):
            io.set_output("unknown", True)


if __name__ == "__main__":
    unittest.main()
