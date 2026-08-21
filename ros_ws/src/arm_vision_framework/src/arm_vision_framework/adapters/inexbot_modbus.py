"""Safe, dependency-free Modbus-TCP foundation for the field controller.

The Inexbot system manual confirms that remote mode can use a Modbus slave and
digital IO, but it does not publish an Ethernet motion protocol, controller
port, or register map.  This module therefore implements only the standard
Modbus-TCP transport and leaves all controller-specific addresses in the field
configuration.  It must not be used to infer MOVJ/MOVL packets.
"""

from dataclasses import dataclass
import socket
import struct
import threading
from typing import Iterable, List, Optional, Sequence


class ControllerProtocolError(RuntimeError):
    """Base class for transport and protocol failures."""


class ControllerConnectionError(ControllerProtocolError):
    """The controller could not be reached or the connection was lost."""


class ControllerTimeout(ControllerConnectionError):
    """A connect, send, or receive operation exceeded its timeout."""


class ModbusProtocolError(ControllerProtocolError):
    """The peer sent an invalid Modbus-TCP frame."""


class ModbusExceptionResponse(ModbusProtocolError):
    """The controller returned a Modbus exception response."""

    EXCEPTION_NAMES = {
        1: "illegal function",
        2: "illegal data address",
        3: "illegal data value",
        4: "server device failure",
        5: "acknowledge",
        6: "server device busy",
        8: "memory parity error",
        10: "gateway path unavailable",
        11: "gateway target device failed to respond",
    }

    def __init__(self, function_code: int, exception_code: int):
        self.function_code = int(function_code)
        self.exception_code = int(exception_code)
        description = self.EXCEPTION_NAMES.get(
            self.exception_code, "unknown exception"
        )
        super().__init__(
            "Modbus exception function=0x{:02X} code={} ({})".format(
                self.function_code, self.exception_code, description
            )
        )


@dataclass(frozen=True)
class TcpEndpoint:
    """Controller endpoint settings.

    ``port`` is intentionally required by this low-level class.  The shipped
    competition configuration leaves it empty until the real controller and
    its official communication sheet are available.
    """

    host: str
    port: int
    connect_timeout_s: float = 2.0
    io_timeout_s: float = 1.0
    keepalive: bool = True
    max_frame_bytes: int = 260

    def __post_init__(self):
        host = str(self.host).strip()
        if not host:
            raise ValueError("controller host is required")
        object.__setattr__(self, "host", host)
        port = int(self.port)
        if not 1 <= port <= 65535:
            raise ValueError("controller port must be within 1..65535")
        object.__setattr__(self, "port", port)
        for name in ("connect_timeout_s", "io_timeout_s"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError("{} must be positive".format(name))
            object.__setattr__(self, name, value)
        frame_limit = int(self.max_frame_bytes)
        if not 8 <= frame_limit <= 260:
            raise ValueError("max_frame_bytes must be within 8..260")
        object.__setattr__(self, "max_frame_bytes", frame_limit)


class TcpTransport:
    """Small blocking TCP transport with exact reads and no hidden retries."""

    def __init__(self, endpoint: TcpEndpoint, socket_factory=None):
        self.endpoint = endpoint
        self._socket_factory = socket_factory or socket.create_connection
        self._socket = None
        self._lock = threading.RLock()

    @property
    def connected(self):
        return self._socket is not None

    def connect(self):
        with self._lock:
            if self._socket is not None:
                return self
            try:
                sock = self._socket_factory(
                    (self.endpoint.host, self.endpoint.port),
                    timeout=self.endpoint.connect_timeout_s,
                )
                sock.settimeout(self.endpoint.io_timeout_s)
                if self.endpoint.keepalive:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            except socket.timeout as error:
                raise ControllerTimeout(
                    "timeout connecting to {}:{}".format(
                        self.endpoint.host, self.endpoint.port
                    )
                ) from error
            except OSError as error:
                raise ControllerConnectionError(
                    "cannot connect to {}:{}: {}".format(
                        self.endpoint.host, self.endpoint.port, error
                    )
                ) from error
            self._socket = sock
            return self

    def close(self):
        with self._lock:
            sock, self._socket = self._socket, None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def send_all(self, payload: bytes):
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        with self._lock:
            self.connect()
            try:
                self._socket.sendall(payload)
            except socket.timeout as error:
                self.close()
                raise ControllerTimeout("timeout sending controller frame") from error
            except OSError as error:
                self.close()
                raise ControllerConnectionError(
                    "controller send failed: {}".format(error)
                ) from error

    def recv_exact(self, size: int) -> bytes:
        size = int(size)
        if size < 0:
            raise ValueError("receive size cannot be negative")
        with self._lock:
            self.connect()
            chunks = bytearray()
            try:
                while len(chunks) < size:
                    chunk = self._socket.recv(size - len(chunks))
                    if not chunk:
                        self.close()
                        raise ControllerConnectionError(
                            "controller closed the connection"
                        )
                    chunks.extend(chunk)
            except socket.timeout as error:
                self.close()
                raise ControllerTimeout("timeout receiving controller frame") from error
            except OSError as error:
                self.close()
                raise ControllerConnectionError(
                    "controller receive failed: {}".format(error)
                ) from error
            return bytes(chunks)


def _u16(value: int, field: str = "address") -> int:
    value = int(value)
    if not 0 <= value <= 0xFFFF:
        raise ValueError("{} must be within 0..65535".format(field))
    return value


def _quantity(value: int, minimum: int, maximum: int) -> int:
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError("quantity must be within {}..{}".format(minimum, maximum))
    return value


class ModbusTcpClient:
    """Modbus Application Protocol over a TCP stream.

    The MBAP header is ``transaction_id:u16, protocol_id:u16,
    length:u16, unit_id:u8``.  ``length`` counts the unit identifier and PDU;
    the PDU begins with a function code.  Addresses are the zero-based values
    on the wire, as required by the Modbus specification.
    """

    def __init__(self, endpoint: TcpEndpoint, unit_id: int = 1, transport=None):
        unit_id = int(unit_id)
        if not 1 <= unit_id <= 247:
            raise ValueError("Modbus TCP unit_id must be within 1..247")
        self.endpoint = endpoint
        self.unit_id = unit_id
        self.transport = transport or TcpTransport(endpoint)
        self._transaction_id = 0
        self._request_lock = threading.Lock()

    @property
    def connected(self):
        return self.transport.connected

    def connect(self):
        self.transport.connect()
        return self

    def close(self):
        self.transport.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _next_transaction(self):
        self._transaction_id = (self._transaction_id + 1) & 0xFFFF
        return self._transaction_id

    def _request(self, pdu: bytes) -> bytes:
        if not 1 <= len(pdu) <= 253:
            raise ValueError("Modbus PDU must contain 1..253 bytes")
        with self._request_lock:
            transaction = self._next_transaction()
            # MBAP length includes unit_id (one byte) and the PDU.
            frame = struct.pack(">HHHB", transaction, 0, len(pdu) + 1, self.unit_id) + pdu
            self.transport.send_all(frame)
            header = self.transport.recv_exact(7)
            response_transaction, protocol_id, length, unit_id = struct.unpack(
                ">HHHB", header
            )
            if response_transaction != transaction:
                raise ModbusProtocolError(
                    "transaction mismatch: expected {}, received {}".format(
                        transaction, response_transaction
                    )
                )
            if protocol_id != 0:
                raise ModbusProtocolError(
                    "unsupported Modbus protocol identifier {}".format(protocol_id)
                )
            if unit_id != self.unit_id:
                raise ModbusProtocolError(
                    "unit mismatch: expected {}, received {}".format(
                        self.unit_id, unit_id
                    )
                )
            # length includes the already-read unit identifier.
            pdu_length = length - 1
            if pdu_length < 2 or pdu_length > 253:
                self.close()
                raise ModbusProtocolError(
                    "invalid Modbus MBAP length {}".format(length)
                )
            if 6 + length > self.endpoint.max_frame_bytes:
                self.close()
                raise ModbusProtocolError(
                    "Modbus frame exceeds configured limit {}".format(
                        self.endpoint.max_frame_bytes
                    )
                )
            response = self.transport.recv_exact(pdu_length)
            function = response[0]
            if function & 0x80:
                if len(response) != 2:
                    raise ModbusProtocolError("malformed Modbus exception response")
                raise ModbusExceptionResponse(function & 0x7F, response[1])
            if function != pdu[0]:
                raise ModbusProtocolError(
                    "function mismatch: expected 0x{:02X}, received 0x{:02X}".format(
                        pdu[0], function
                    )
                )
            return response

    @staticmethod
    def _bits_from_response(response: bytes, quantity: int) -> List[bool]:
        if len(response) < 3 or response[1] != len(response) - 2:
            raise ModbusProtocolError("malformed bit response")
        byte_count = response[1]
        expected = (quantity + 7) // 8
        if byte_count != expected or len(response) != byte_count + 2:
            raise ModbusProtocolError("bit response byte count mismatch")
        return [bool(response[2 + index // 8] & (1 << (index % 8))) for index in range(quantity)]

    @staticmethod
    def _registers_from_response(response: bytes, quantity: int) -> List[int]:
        if len(response) < 3 or response[1] != len(response) - 2:
            raise ModbusProtocolError("malformed register response")
        byte_count = response[1]
        if byte_count != quantity * 2 or len(response) != byte_count + 2:
            raise ModbusProtocolError("register response byte count mismatch")
        return list(struct.unpack(">{}H".format(quantity), response[2:]))

    def _read_bits(self, function: int, address: int, quantity: int) -> List[bool]:
        address = _u16(address)
        quantity = _quantity(quantity, 1, 2000)
        response = self._request(struct.pack(">BHH", function, address, quantity))
        return self._bits_from_response(response, quantity)

    def _read_registers(self, function: int, address: int, quantity: int) -> List[int]:
        address = _u16(address)
        quantity = _quantity(quantity, 1, 125)
        response = self._request(struct.pack(">BHH", function, address, quantity))
        return self._registers_from_response(response, quantity)

    def read_coils(self, address: int, quantity: int) -> List[bool]:
        return self._read_bits(1, address, quantity)

    def read_discrete_inputs(self, address: int, quantity: int) -> List[bool]:
        return self._read_bits(2, address, quantity)

    def read_holding_registers(self, address: int, quantity: int) -> List[int]:
        return self._read_registers(3, address, quantity)

    def read_input_registers(self, address: int, quantity: int) -> List[int]:
        return self._read_registers(4, address, quantity)

    def write_single_coil(self, address: int, value: bool) -> bool:
        address = _u16(address)
        response = self._request(
            struct.pack(">BHH", 5, address, 0xFF00 if bool(value) else 0x0000)
        )
        if len(response) != 5 or response[1:] != struct.pack(">HH", address, 0xFF00 if value else 0):
            raise ModbusProtocolError("write single coil response mismatch")
        return True

    def write_multiple_coils(self, address: int, values: Iterable[bool]) -> bool:
        address = _u16(address)
        values = [bool(value) for value in values]
        quantity = _quantity(len(values), 1, 1968)
        byte_count = (quantity + 7) // 8
        payload = bytearray(byte_count)
        for index, value in enumerate(values):
            if value:
                payload[index // 8] |= 1 << (index % 8)
        response = self._request(
            struct.pack(">BHHB", 15, address, quantity, byte_count) + bytes(payload)
        )
        if len(response) != 5 or response[1:] != struct.pack(">HH", address, quantity):
            raise ModbusProtocolError("write multiple coils response mismatch")
        return True

    def write_single_register(self, address: int, value: int) -> bool:
        address = _u16(address)
        value = _u16(value, "register value")
        response = self._request(struct.pack(">BHH", 6, address, value))
        if len(response) != 5 or response[1:] != struct.pack(">HH", address, value):
            raise ModbusProtocolError("write single register response mismatch")
        return True

    def write_multiple_registers(self, address: int, values: Sequence[int]) -> bool:
        address = _u16(address)
        values = [_u16(value, "register value") for value in values]
        quantity = _quantity(len(values), 1, 123)
        response = self._request(
            struct.pack(">BHHB", 16, address, quantity, quantity * 2)
            + struct.pack(">{}H".format(quantity), *values)
        )
        if len(response) != 5 or response[1:] != struct.pack(">HH", address, quantity):
            raise ModbusProtocolError("write multiple registers response mismatch")
        return True


class ConfiguredRemoteIo:
    """Named digital IO facade with no hard-coded controller addresses.

    Output names are resolved from ``controller.remote_io.outputs`` and are
    written as Modbus coils.  Input names are resolved from
    ``controller.remote_io.inputs`` and read as discrete inputs.  The exact
    address and the meaning of each bit still have to come from the official
    controller/IO map.
    """

    def __init__(self, client: ModbusTcpClient, settings):
        self.client = client
        data = settings.data if hasattr(settings, "data") else settings
        controller = data.get("controller", data)
        remote_io = controller.get("remote_io", {})
        self.outputs = {
            str(name): _u16(address, "output address")
            for name, address in remote_io.get("outputs", {}).items()
        }
        self.inputs = {
            str(name): _u16(address, "input address")
            for name, address in remote_io.get("inputs", {}).items()
        }

    def _address(self, mapping, name, kind):
        key = str(name)
        if key not in mapping:
            raise KeyError("unknown configured {} IO: {}".format(kind, key))
        return mapping[key]

    def set_output(self, name: str, value: bool) -> bool:
        return self.client.write_single_coil(
            self._address(self.outputs, name, "output"), bool(value)
        )

    def read_input(self, name: str) -> bool:
        return self.client.read_discrete_inputs(
            self._address(self.inputs, name, "input"), 1
        )[0]


@dataclass(frozen=True)
class InexbotPoint:
    """The point-variable fields shown in the system manual.

    This is a data model, not a guessed network packet.  The manual lists the
    fields as ``P-name, coordinate system, degree/radian flag, shape,
    tool, user, two reserved fields, axis1..axis7``.
    """

    name: str
    coordinate_system: int
    angle_unit: int
    shape: int
    tool_id: int
    user_id: int
    axes: Sequence[float]
    reserved: Sequence[float] = (0.0, 0.0)

    def __post_init__(self):
        name = str(self.name).strip()
        if (
            len(name) != 5
            or name[0] != "P"
            or not name[1:].isdigit()
            or not 1 <= int(name[1:]) <= 9999
        ):
            raise ValueError("point name must be within P0001..P9999")
        if int(self.coordinate_system) not in (0, 1, 2, 3):
            raise ValueError("coordinate_system must be 0 (joint), 1 (rect), 2 (tool), or 3 (user)")
        if int(self.angle_unit) not in (0, 1):
            raise ValueError("angle_unit must be 0 (degree) or 1 (radian)")
        if not 1 <= int(self.shape) <= 8:
            raise ValueError("shape must be within 1..8")
        if int(self.tool_id) < 0 or int(self.user_id) < 0:
            raise ValueError("tool_id and user_id cannot be negative")
        axes = tuple(self.axes)
        reserved = tuple(self.reserved)
        if len(axes) != 7:
            raise ValueError("point axes must contain exactly seven values")
        if len(reserved) != 2:
            raise ValueError("point reserved fields must contain two values")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coordinate_system", int(self.coordinate_system))
        object.__setattr__(self, "angle_unit", int(self.angle_unit))
        object.__setattr__(self, "shape", int(self.shape))
        object.__setattr__(self, "tool_id", int(self.tool_id))
        object.__setattr__(self, "user_id", int(self.user_id))
        object.__setattr__(self, "axes", tuple(float(value) for value in axes))
        object.__setattr__(self, "reserved", tuple(float(value) for value in reserved))

    def fields(self):
        return (
            self.name,
            self.coordinate_system,
            self.angle_unit,
            self.shape,
            self.tool_id,
            self.user_id,
            self.reserved[0],
            self.reserved[1],
            *self.axes,
        )

    def csv(self):
        return ",".join(str(value) for value in self.fields())


def shape_from_joint_degrees(joints: Sequence[float]) -> int:
    """Compute the manual's 1/3/5-axis shape value for six joint angles."""

    joints = tuple(joints)
    if len(joints) < 5:
        raise ValueError("at least five joint angles are required")
    bits = []
    for index in (0, 2, 4):
        bits.append(1 if -90.0 <= float(joints[index]) <= 90.0 else 0)
    return ((bits[0] << 2) | (bits[1] << 1) | bits[2]) + 1


def point_from_joint_degrees(name: str, joints: Sequence[float], tool_id: int = 0, user_id: int = 0) -> InexbotPoint:
    """Build a manual-compatible joint point while retaining all metadata."""

    joints = tuple(float(value) for value in joints)
    if len(joints) != 6:
        raise ValueError("six joint angles are required")
    return InexbotPoint(
        name=name,
        coordinate_system=0,
        angle_unit=0,
        shape=shape_from_joint_degrees(joints),
        tool_id=tool_id,
        user_id=user_id,
        axes=joints + (0.0,),
    )


def modbus_client_from_config(config, *, transport=None) -> Optional[ModbusTcpClient]:
    """Create a client only when explicitly enabled and fully configured."""

    data = config.data if hasattr(config, "data") else config
    settings = data.get("controller", {})
    if not bool(settings.get("enabled", False)):
        return None
    if str(settings.get("transport", "")).strip().lower() != "modbus_tcp":
        raise ValueError("controller.transport must be modbus_tcp")
    host = str(settings.get("host", "")).strip()
    port = settings.get("port")
    unit_id = settings.get("unit_id")
    if not host or port is None or unit_id is None:
        raise ValueError("enabled controller requires host, port, and unit_id")
    endpoint = TcpEndpoint(
        host,
        port,
        connect_timeout_s=settings.get("connect_timeout_s", 2.0),
        io_timeout_s=settings.get("io_timeout_s", 1.0),
        keepalive=settings.get("keepalive", True),
        max_frame_bytes=settings.get("max_frame_bytes", 260),
    )
    return ModbusTcpClient(endpoint, unit_id=unit_id, transport=transport)


__all__ = [
    "ControllerProtocolError", "ControllerConnectionError", "ControllerTimeout",
    "ModbusProtocolError", "ModbusExceptionResponse", "TcpEndpoint", "TcpTransport",
    "ModbusTcpClient", "InexbotPoint", "shape_from_joint_degrees",
    "point_from_joint_degrees", "ConfiguredRemoteIo", "modbus_client_from_config",
]
