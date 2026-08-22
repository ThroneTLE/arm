"""NexBot (纳博特) JSON-over-TCP protocol adapter (RTL-22.07).

Official protocol: open.inexbot.com SDK docs (RTL-22.07) and doc.inexbot.com
knowledge base, summarized in ``docs/纳博特通讯协议.md``.  Wire frame::

    0x4E66 | Length u16 BE | Command u16 BE | JSON data | CRC32 u32 BE

CRC32 covers Length+Command+data and is sent big-endian; verified against the
official example (``zlib.crc32`` of the documented query frame equals
``0x9090FA38``).  Port 6001 is the real-time command channel (field-tested on
the MOKA MR07S-930 / Inexbot C1102: 6000 stays closed, 6001 answers; MOVJ
``0x4501``, MOVL ``0x4502``, MOVC ``0x4503``, MOVS ``0x4504``, GO_HOME
``0x3002``, emergency stop ``0x2314``, DOUT set ``0x3601``); port 7000 is the
host-computer state service (``0x9512`` query / ``0x9513`` reply).

This adapter implements the framework ``RobotController`` boundary only.
Every motion safety gate (dry-run, workspace, segmentation, pose freshness)
stays with the caller; this class never moves the robot on its own.

Field-verification items (documented, not guessed): the exact ``pos`` array
length (official text says 7 body slots + 5 external-axis slots while the
example shows 7; ``external_axes`` selects), joint angle units (``joint_unit``),
and whether ``0x4501/0x4502`` accept Cartesian targets without a shape field.

帧级抓取(默认关闭)
------------------
``NEXBOT_FRAME_LOG=/tmp/frames.log`` 会把每一条收发帧(时间戳/方向/端口/指令字/
JSON/原始字节 hex)写进该文件, 用 ``competition_pipeline/scripts/decode_frames.py``
解码。不设这个环境变量时**零副作用零开销**: 见 ``NexBotTcpTransport.__init__``
的注释 —— 不设就不绑包装方法, 收发路径的字节码与加日志之前完全一致。
"""

import json
import os
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import RobotController
from ..transforms import (
    as_transform,
    inexbot_abc_from_transform,
    transform_from_inexbot_abc,
    transform_from_xyz_rpy,
    xyz_rpy_from_transform,
)
from ..types import RobotState
from .inexbot_modbus import (
    ControllerConnectionError,
    ControllerProtocolError,
    ControllerTimeout,
    InexbotPoint,
)

SYNC = b"\x4E\x66"
CMD_HEARTBEAT = 0x7266
CMD_HEARTBEAT_REPLY = 0x7267
CMD_MOVJ = 0x4501
CMD_MOVL = 0x4502
CMD_ENABLE = 0x2311
CMD_SERVO_INQUIRE = 0x2002
CMD_SERVO_RESPOND = 0x2003
CMD_EMERGENCY_STOP = 0x2314
CMD_QUERY = 0x9512
CMD_QUERY_REPLY = 0x9513
CMD_DOUT_SET = 0x3601
CMD_DOUT_QUERY = 0x3602
CMD_DOUT_QUERY_REPLY = 0x3603
CMD_GO_HOME = 0x3002
CMD_GO_RESET_POSITION = 0x3007
#: 控制器报警推送帧.  现场实测(controlLogs08-22-22-16)真实内容在 ``data`` 键:
#: ``{"code":25530,"data":"指令[0x4502]参数错误","kind":2,"param":[...],"robot":0}``
CMD_ALARM = 0x2B03
#: 作业运行状态推送帧: ``{"robot":1,"status":n}``, 0=停止 / 2=运行中.
#: **这是"运动真的开始了"的唯一权威信号** —— 见 ``_await_motion_ack``.
CMD_PROGRAM_STATUS = 0x3D03
CMD_ERRORS = frozenset({0x6010, 0x6020, 0x6030, 0x6040})
CMD_WARNINGS = frozenset({0x6110, 0x6210})

#: 坐标系编号: 0 关节 / 1 直角 / 2 工具 / 3 用户.
#: 【实测 + 手册双证】现场用 coord=3 下发 MOVL, 机器人按用户坐标系1 运动(2026-08-22);
#: 《iNexBot 系统操作手册 2207》"坐标系说明与切换"一节给出的示教器切换顺序也是
#: 关节 -> 直角 -> 工具 -> 用户, 与本编号一致。
#: (曾有一个 open.inexbot.com 的 JSON 文档页写 2=用户/3=工具 —— 那页是错的。)
COORD_JOINT = 0
COORD_CARTESIAN = 1
COORD_TOOL = 2
COORD_USER = 3


def build_frame(command: int, data: Optional[Dict[str, Any]] = None) -> bytes:
    """Build one wire frame (0x4E66 + length + command + JSON + CRC32)."""
    payload = (
        b""
        if data is None
        else json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    if len(payload) > 0xFFFF:
        raise ControllerProtocolError("frame data exceeds 0xFFFF bytes")
    length = struct.pack(">H", len(payload))
    command_bytes = struct.pack(">H", int(command))
    crc = zlib.crc32(length + command_bytes + payload) & 0xFFFFFFFF
    return SYNC + length + command_bytes + payload + struct.pack(">I", crc)


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ControllerConnectionError("connection closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _RecvTap:
    """把 ``read_frame`` 实际读走的字节顺手抄一份到 ``sink``。

    为什么要 tap, 而不是让 ``read_frame`` 多返回一个原始字节串: 帧日志最值钱的
    时刻恰恰是 ``read_frame`` **抛异常**的那一刻(同步字错位 / 长度越界 / CRC 不
    匹配 / 对端半路断开)。那时候没有返回值可用, 只有"已经读进来的这几段字节"
    能说明问题。装在 socket 上就能让异常路径照样落盘。
    """

    __slots__ = ("_sock", "_sink")

    def __init__(self, sock, sink):
        self._sock = sock
        self._sink = sink

    def recv(self, count):
        chunk = self._sock.recv(count)
        self._sink.append(chunk)
        return chunk


def read_frame(sock: socket.socket, max_frame_bytes: int, _raw_sink=None) -> Tuple[int, Any]:
    """Read and validate one frame; returns ``(command, parsed_data)``.

    ``_raw_sink`` 只服务于帧日志(见 ``NEXBOT_FRAME_LOG``)。默认 ``None`` 时这里
    只多一次 ``is not None`` 判断, 之后执行的是与加日志之前**逐字节相同**的代码
    路径 —— 不新建对象、不做字符串操作、不碰 logging。
    """
    if _raw_sink is not None:
        sock = _RecvTap(sock, _raw_sink)
    sync = _recv_exact(sock, 2)
    if sync != SYNC:
        raise ControllerProtocolError(
            "frame sync mismatch: expected 0x4E66, got 0x{}".format(sync.hex())
        )
    length = struct.unpack(">H", _recv_exact(sock, 2))[0]
    if length > max_frame_bytes:
        raise ControllerProtocolError(
            "frame data length {} exceeds limit {}".format(length, max_frame_bytes)
        )
    command = struct.unpack(">H", _recv_exact(sock, 2))[0]
    payload = _recv_exact(sock, length)
    crc = struct.unpack(">I", _recv_exact(sock, 4))[0]
    expected = (
        zlib.crc32(struct.pack(">H", length) + struct.pack(">H", command) + payload)
        & 0xFFFFFFFF
    )
    if crc != expected:
        raise ControllerProtocolError(
            "frame CRC mismatch: got 0x{:08X}, expected 0x{:08X}".format(crc, expected)
        )
    parsed = None
    if payload:
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except ValueError as error:
            raise ControllerProtocolError(
                "frame data is not valid JSON: {}".format(error)
            ) from error
    return command, parsed


#: 掉使能的头号现场原因, 附在异常里让调用方一眼看懂该去示教器改什么.
#: 证据: ``docs/现场备份-20260822/根因证据-控制器日志摘录.txt`` 证据 A-1 / A-2.
SAFETY_GATE_HINT = (
    "现场实测主因: 控制器出厂配置 global.json 的 RemoteIO[0].posReset 里 "
    "safeEnable=true 但逐轴容差 deviation=null, 于是在**远程模式**下"
    "复位点安全闸门对每条运动指令必然判定“机器人1不在安全位置附近”"
    "(实测机器人精确停在复位点、逐轴只差 1e-5° 也照样被拒), "
    "拒绝后控制器会执行 stop -> JobClear -> 脉冲使能=0 -> Deadan_End -> PowerOff, "
    "即每拒绝一次就把伺服下电一次。"
    "对策: (1) 示教器切回**示教模式**(实测示教模式下 40+ 条 MOVL 全部成功); "
    "或 (2) 示教器【复位点设置】把逐轴偏差 deviation 填成 1.0°; 或 (3) 关闭安全使能。"
)


def _frame_message(data: Any) -> str:
    """Extract the human-readable text out of an alarm/error frame.

    IMPORTANT: the real controller puts the text in ``data``, not ``message``.
    The previous implementation was ``str(d.get("message")) or str(d.get("error"))``
    which is doubly broken: it never looked at ``data``, and ``str(None)`` is the
    non-empty string ``"None"`` -- always truthy -- so the fallback branches were
    dead and every alarm surfaced as the literal text ``None``.
    """
    if isinstance(data, dict):
        for key in ("data", "message", "error", "msg", "desc"):
            value = data.get(key)
            if value not in (None, ""):
                return str(value)
        return json.dumps(data, ensure_ascii=False)
    if data in (None, ""):
        return "controller error"
    return str(data)


@dataclass(frozen=True)
class NexBotTcpEndpoint:
    """Controller endpoint facts for the 6001 (motion) and 7000 (state) ports.

    Field-tested on the MOKA MR07S-930 (Inexbot C1102, RTL-22.07): the
    real-time command port answers on **6001**, not 6000 (6000 stays closed;
    verified 2026-08-22).
    """

    host: str
    port_motion: int = 6001
    port_state: int = 7000
    robot: int = 1
    channel: int = 1
    connect_timeout_s: float = 2.0
    io_timeout_s: float = 1.0
    keepalive: bool = True
    max_frame_bytes: int = 1024 * 1024
    external_axes: int = 0
    wait_for_finish: bool = True
    motion_finish_timeout_s: float = 60.0
    #: Reference frame of the state readback (realPos*): PCS (tool coord =
    #: TCP w.r.t. robot base), MCS (Cartesian w.r.t. base) or UCS (user
    #: coordinate frame; with user 1 active this is 用户坐标系1).
    pose_frame: str = "PCS"
    #: Coordinate system for motion commands (COORD_* above).  With the whole
    #: pipeline on 用户坐标系1 set this to COORD_USER; the controller then
    #: consumes targets directly in the user frame (no conversion needed).
    motion_coord: int = COORD_CARTESIAN
    #: Active tool/user id on the controller (for documentation and checks).
    tool_id: int = 1
    user_id: int = 1
    velocity_eps_rad_s: float = 0.02
    heartbeat_s: float = 0.0
    #: 发出运动指令后, 等待控制器推送 ``0x3D03 status=2``(真的开始动了) 的秒数.
    #: 设为 0 关闭该确认 —— **不要在实机上关掉**: 关掉就回到"指令被拒也报成功"
    #: 的旧行为(见 ``_await_motion_ack`` 的 docstring)。
    motion_ack_timeout_s: float = 3.0
    #: 每条运动指令下发前先确认伺服 status==3, 不满足就发 ``0x2311`` 上使能.
    #: 见 :meth:`NexBotTcpRobotController._ensure_servo_enabled` —— 这条默认开启,
    #: 因为在这台控制器上"上一条指令成功过"根本推不出"现在还使能着".
    #: 设为 False 只应出现在协议级单元测试里。
    ensure_servo_before_motion: bool = True

    def __post_init__(self):
        host = str(self.host).strip()
        if not host:
            raise ValueError("NexBot TCP host is required")
        object.__setattr__(self, "host", host)
        for name in ("port_motion", "port_state"):
            value = int(getattr(self, name))
            if not 1 <= value <= 65535:
                raise ValueError("{} must be within 1..65535".format(name))
            object.__setattr__(self, name, value)
        robot = int(self.robot)
        if not 1 <= robot <= 4:
            raise ValueError("robot must be within 1..4")
        object.__setattr__(self, "robot", robot)
        channel = int(self.channel)
        if not 1 <= channel <= 9:
            raise ValueError("channel must be within 1..9")
        object.__setattr__(self, "channel", channel)
        for name in ("connect_timeout_s", "io_timeout_s", "motion_finish_timeout_s"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError("{} must be positive".format(name))
            object.__setattr__(self, name, value)
        external_axes = int(self.external_axes)
        if external_axes not in (0, 5):
            raise ValueError("external_axes must be 0 (7-slot body array) or 5 (12-slot array)")
        object.__setattr__(self, "external_axes", external_axes)
        frame_limit = int(self.max_frame_bytes)
        if not 8 <= frame_limit <= 16 * 1024 * 1024:
            raise ValueError("max_frame_bytes must be within 8..16777216")
        object.__setattr__(self, "max_frame_bytes", frame_limit)
        heartbeat = float(self.heartbeat_s)
        if heartbeat < 0.0:
            raise ValueError("heartbeat_s cannot be negative")
        object.__setattr__(self, "heartbeat_s", heartbeat)
        motion_ack = float(self.motion_ack_timeout_s)
        if motion_ack < 0.0:
            raise ValueError("motion_ack_timeout_s cannot be negative")
        object.__setattr__(self, "motion_ack_timeout_s", motion_ack)
        object.__setattr__(
            self, "ensure_servo_before_motion", bool(self.ensure_servo_before_motion)
        )


#: 帧级抓取开关: 设成一个文件路径就把每一条收发帧写进去, 不设就完全不存在。
#: 用途是定位"控制器像是没收到"这类没见过的通信问题 —— 有了原始字节, CRC 能
#: 立刻区分"我们发错了"和"控制器不认"。
FRAME_LOG_ENV = "NEXBOT_FRAME_LOG"
#: 每帧 hex 的字节上限。状态查询应答几百字节是常态, 512 够放整帧(还能校 CRC),
#: 又不会让一次演练把磁盘写满。真碰上更大的帧再用这个环境变量抬高。
FRAME_LOG_HEX_BYTES_ENV = "NEXBOT_FRAME_LOG_HEX_BYTES"
_FRAME_LOG_HEX_BYTES_DEFAULT = 512


class _FrameLog:
    """行 JSON 格式的帧级日志。**只在 ``NEXBOT_FRAME_LOG`` 指向某个路径时才存在。**

    为什么不用 ``logging``: 那会往进程里塞一套全局的 handler/formatter/propagate
    状态, 别人在任何地方调一次 ``logging.basicConfig`` 就可能把这些行冲进 stderr,
    或者被别人的 filter 吃掉。明天要跑竞赛, 诊断开关不允许有这种远程副作用。

    为什么写行 JSON 而不是给人看的对齐格式: 这个文件是给 ``decode_frames.py``
    吃的, JSON 里带引号带逗号的报警文本用空格分列必然切错。要给人看就跑
    decode_frames, 它会渲染成人读的样子。

    落盘用行缓冲: 现场最需要这个日志的场景就是进程卡死/被杀, 那时候攒在缓冲区里
    没写出去的恰恰是最后几帧 —— 也就是最关键的几帧。
    """

    _by_path = {}
    _by_path_lock = threading.Lock()

    @classmethod
    def for_path(cls, path, hex_limit):
        """同一路径共享同一个实例。

        6001(运动)和 7000(状态)是两条 transport, 但通常写同一个文件, 必须共用
        同一个句柄和同一把锁, 否则两条线程的行会互相插进对方中间。
        """
        with cls._by_path_lock:
            log = cls._by_path.get(path)
            if log is None:
                log = cls(path, hex_limit)
                cls._by_path[path] = log
            return log

    def __init__(self, path, hex_limit):
        self.path = path
        self.hex_limit = int(hex_limit)
        self._lock = threading.Lock()
        try:
            self._handle = open(path, "a", encoding="utf-8", buffering=1)
        except OSError:
            # 诊断开关打不开文件, 绝不能连累正常通信 —— 静默退化成"没开日志"。
            self._handle = None

    def write(self, direction, port, command, data, raw, error=None):
        handle = self._handle
        if handle is None:
            return
        now = time.time()
        record = {
            # 两种时间都留: "t" 给人对现场日志(控制器时钟 ≈ PC 时钟 - 10 分钟),
            # "ts" 给脚本做差值。
            "t": "{}.{:03d}".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                int((now % 1.0) * 1000.0),
            ),
            "ts": round(now, 3),
            "dir": direction,
            "port": port,
        }
        if command is not None:
            record["cmd"] = "0x{:04X}".format(int(command))
        record["len"] = len(raw)
        if data is not None:
            record["json"] = data
        if len(raw) > self.hex_limit:
            record["hex"] = raw[: self.hex_limit].hex()
            record["trunc"] = len(raw)
        else:
            record["hex"] = raw.hex()
        if error is not None:
            record["err"] = "{}: {}".format(type(error).__name__, error)
        try:
            # default=repr: 竞赛日不接受"记日志本身把程序搞崩"这种剧本。
            line = json.dumps(record, ensure_ascii=False, default=repr)
        except (TypeError, ValueError) as failure:
            line = json.dumps(
                {"t": record["t"], "dir": direction, "port": port,
                 "err": "frame log serialise failed: {}".format(failure)},
                ensure_ascii=False,
            )
        with self._lock:
            try:
                handle.write(line + "\n")
            except (OSError, ValueError):
                pass


def _open_frame_log():
    """按环境变量决定要不要开帧日志; 没设就返回 None。

    只在 transport 构造时读一次环境变量(一个 controller 两次), 不是每帧读一次。
    """
    path = os.environ.get(FRAME_LOG_ENV, "").strip()
    if not path:
        return None
    try:
        hex_limit = int(os.environ.get(FRAME_LOG_HEX_BYTES_ENV, "") or
                        _FRAME_LOG_HEX_BYTES_DEFAULT)
    except ValueError:
        hex_limit = _FRAME_LOG_HEX_BYTES_DEFAULT
    return _FrameLog.for_path(path, max(0, hex_limit))


class NexBotTcpTransport:
    """One blocking, thread-safe TCP transport for a single controller port.

    这里是整个工程收发帧的**唯一收敛点**: 二十多处 ``self.motion.send_frame`` /
    ``self.state.read_frame`` 全都落到下面这两个方法上。帧日志因此只挂在这里,
    不在调用点上散着加。
    """

    def __init__(self, endpoint: NexBotTcpEndpoint, port: int, socket_factory=None):
        self.endpoint = endpoint
        self.port = int(port)
        self._socket_factory = socket_factory or socket.create_connection
        self._socket = None
        self._lock = threading.RLock()
        # 帧日志是**诊断开关**, 不是功能。没设环境变量时下面这个分支不执行,
        # send_frame/read_frame 就还是类上那两个原封不动的方法 —— 热路径上没有
        # 多出任何一次判断、任何一个包装对象、任何一个 logging handler。
        # 开了日志才用实例属性遮蔽类方法, 代价只由开了开关的那个进程付。
        self._frame_log = _open_frame_log()
        if self._frame_log is not None:
            self.send_frame = self._send_frame_logged
            self.read_frame = self._read_frame_logged

    @property
    def connected(self):
        return self._socket is not None

    def connect(self):
        with self._lock:
            if self._socket is not None:
                return self
            try:
                sock = self._socket_factory(
                    (self.endpoint.host, self.port),
                    timeout=self.endpoint.connect_timeout_s,
                )
            except OSError as error:
                raise ControllerConnectionError(
                    "could not connect to {}:{}: {}".format(
                        self.endpoint.host, self.port, error
                    )
                ) from error
            sock.settimeout(self.endpoint.io_timeout_s)
            if self.endpoint.keepalive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self._socket = sock
            return self

    def send_frame(self, command: int, data: Optional[Dict[str, Any]] = None):
        with self._lock:
            self.connect()
            try:
                self._socket.sendall(build_frame(command, data))
            except OSError as error:
                self.close()
                raise ControllerConnectionError(
                    "send to {}:{} failed: {}".format(
                        self.endpoint.host, self.port, error
                    )
                ) from error

    def read_frame(self, timeout: Optional[float] = None, _raw_sink=None):
        with self._lock:
            if self._socket is None:
                # 对被关闭的旧连接读帧（并发陈旧引用）→ 明确连接错误而非 NoneType
                raise ControllerConnectionError("controller transport is closed")
            self.connect()
            previous = self._socket.gettimeout()
            try:
                if timeout is not None:
                    self._socket.settimeout(timeout)
                return read_frame(self._socket, self.endpoint.max_frame_bytes, _raw_sink)
            except socket.timeout as error:
                raise ControllerTimeout(
                    "no frame from {}:{} within {:.3f}s".format(
                        self.endpoint.host, self.port,
                        self._socket.gettimeout() if timeout is not None else self.endpoint.io_timeout_s,
                    )
                ) from error
            except OSError as error:
                self.close()
                raise ControllerConnectionError(
                    "read from {}:{} failed: {}".format(
                        self.endpoint.host, self.port, error
                    )
                ) from error
            finally:
                try:
                    self._socket.settimeout(previous)
                except OSError:
                    pass

    # -- 帧日志包装 (只有 NEXBOT_FRAME_LOG 设了才会被绑上去) ------------------
    #
    # 两个方法都显式走 ``NexBotTcpTransport.xxx(self, ...)``: 实例上的同名属性
    # 已经被遮蔽了, 写 ``self.send_frame`` 会无限递归。

    def _send_frame_logged(self, command: int, data: Optional[Dict[str, Any]] = None):
        try:
            NexBotTcpTransport.send_frame(self, command, data)
        except Exception as error:
            # 发失败也必须留痕: "控制器像是没收到"的另一半可能是我们压根没发出去。
            self._frame_log.write("tx", self.port, command, data, b"", error=error)
            raise
        # 重新 build 一次只为拿 hex。多一次序列化的代价只出现在开了日志的进程里,
        # 换来的是"日志里的字节 = 线上的字节"(build_frame 是纯函数, 必然一致)。
        self._frame_log.write("tx", self.port, command, data, build_frame(command, data))

    def _read_frame_logged(self, timeout: Optional[float] = None):
        sink = []
        try:
            command, data = NexBotTcpTransport.read_frame(
                self, timeout=timeout, _raw_sink=sink
            )
        except Exception as error:
            raw = b"".join(sink)
            # 规则: **线上有字节就一定记**(半截帧/CRC 坏帧/同步字错位全靠这条),
            # 一个字节都没读到的常规空转不记 —— 超时来自 _check_errors /
            # _drain_pushed_frames 每次调用的好几轮探测, 连接错误来自连不上或
            # socket 已关闭, 两者都不是"帧", 记下来只会把真正的帧淹掉。
            if raw or not isinstance(
                error, (ControllerTimeout, ControllerConnectionError)
            ):
                self._frame_log.write("rx", self.port, None, None, raw, error=error)
            raise
        self._frame_log.write("rx", self.port, command, data, b"".join(sink))
        return command, data

    def close(self):
        with self._lock:
            if self._socket is not None:
                try:
                    # Wake a thread blocked in recv immediately.  close()
                    # alone may leave the blocking syscall alive until its
                    # timeout on Linux, which prevents the Qt worker from
                    # terminating cleanly during UI shutdown.
                    self._socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    self._socket.close()
                except OSError:
                    pass
                self._socket = None


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


class NexBotTcpRobotController(RobotController):
    """Framework ``RobotController`` over the NexBot 6001/7000 ports.

    Motion commands go to port 6001 (MOVJ ``0x4501``, MOVL ``0x4502``,
    emergency stop ``0x2314``); state queries go to port 7000 (``0x9512``).
    Both connections are lazy and guarded by per-port locks, so one instance
    is safe to share across polling and execution threads.
    """

    def __init__(self, endpoint: NexBotTcpEndpoint, socket_factory=None):
        self.endpoint = endpoint
        self.motion = NexBotTcpTransport(endpoint, endpoint.port_motion, socket_factory)
        self.state = NexBotTcpTransport(endpoint, endpoint.port_state, socket_factory)
        self._closed = False
        self._heartbeat = None
        if endpoint.heartbeat_s > 0.0:
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop, name="nexbot-heartbeat", daemon=True
            )
            self._heartbeat.start()

    def _heartbeat_loop(self):
        while not self._closed:
            try:
                self.motion.send_frame(CMD_HEARTBEAT)
                try:
                    self.motion.read_frame(timeout=min(self.endpoint.io_timeout_s, 0.1))
                except (ControllerTimeout, ControllerProtocolError):
                    pass
            except (ControllerConnectionError, ControllerProtocolError):
                pass
            time.sleep(self.endpoint.heartbeat_s)

    def close(self):
        self._closed = True
        self.motion.close()
        self.state.close()

    # -- helpers -----------------------------------------------------------

    def _ensure_open(self):
        if self._closed:
            raise ControllerConnectionError("NexBot controller is closed")

    def _check_errors(self, first_timeout_s: float = 0.3):
        """Surface any pending controller error/warning/alarm frame.

        Field note (2026-08-22): the previous 50 ms budget was too short -- the
        C1102 needs ~100-250 ms to emit ``0x2B03`` after rejecting a command, so
        the real reason routinely arrived *after* this call had already returned
        and the caller reported success.  The first read now waits 300 ms.
        """
        self._ensure_open()
        timeout = min(self.endpoint.io_timeout_s, float(first_timeout_s))
        for _ in range(8):
            try:
                command, data = self.motion.read_frame(timeout=timeout)
            except (ControllerTimeout, ControllerConnectionError):
                return
            timeout = min(self.endpoint.io_timeout_s, 0.05)
            if command in CMD_ERRORS or command in CMD_WARNINGS or command == CMD_ALARM:
                raise ControllerProtocolError(
                    "controller frame 0x{:04X}: {}".format(
                        command, _frame_message(data)
                    )
                )

    def _drain_pushed_frames(self, budget_s: float = 0.3):
        """Discard queued unsolicited pushes, raising on alarms.

        6001 pushes ``0x2003``/``0x3D03`` without being asked.  A stale push
        sitting in the socket buffer is the classic way a query reads somebody
        else's answer -- e.g. ``0x2311`` pushes ``status:3``, the safety gate
        immediately knocks it back down and pushes ``status:0`` then ``1``, and
        a naive ``servo_status()`` reads the *first* frame and happily reports
        "enabled" for a servo that is already off.  Always drain before asking.
        """
        self._ensure_open()
        deadline = time.monotonic() + max(0.0, float(budget_s))
        while time.monotonic() < deadline:
            try:
                command, data = self.motion.read_frame(timeout=0.05)
            except (ControllerTimeout, ControllerConnectionError):
                return
            if command in CMD_ERRORS or command in CMD_WARNINGS or command == CMD_ALARM:
                raise ControllerProtocolError(
                    "controller frame 0x{:04X}: {}".format(
                        command, _frame_message(data)
                    )
                )

    def _await_motion_ack(self):
        """Block until the controller confirms the motion actually STARTED.

        This exists because "the robot did not move but the program said it
        did" was the single most expensive failure of the 2026-08-22 field
        session.  Three rejection signatures were captured on the wire
        (``docs/现场备份-20260822/根因证据-控制器日志摘录.txt``):

        1. malformed payload  -> ``0x2B03 {"data":"指令[0x4502]参数错误"}``
        2. safety-gate refusal -> 6001 receives ``0x2003 status:0`` then
           ``status:1`` (the controller powered the servo off), and the robot
           never moves.  ``0x3D03`` is never sent.
        3. silently ignored   -> nothing at all comes back.

        In all three cases the pose stays put, which is exactly what the old
        "wait until the pose stops changing" logic interprets as *finished*.
        The only positive evidence that a motion began is ``0x3D03 status=2``,
        so that is what we wait for.
        """
        timeout = float(self.endpoint.motion_ack_timeout_s)
        if timeout <= 0.0:
            return
        self._ensure_open()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise ControllerTimeout(
                    "控制器未在 {:.1f}s 内确认运动开始(没收到 0x3D03 status=2)。"
                    "{}".format(timeout, SAFETY_GATE_HINT)
                )
            try:
                command, data = self.motion.read_frame(
                    timeout=min(remaining, 0.2)
                )
            except ControllerTimeout:
                continue
            if command == CMD_PROGRAM_STATUS:
                if int((data or {}).get("status", 0)) == 2:
                    return
                continue
            if command == CMD_SERVO_RESPOND:
                status = int((data or {}).get("status", 0))
                if status != 3:
                    raise ControllerProtocolError(
                        "运动指令被控制器拒绝并已下电(伺服 status={} != 3)。{}".format(
                            status, SAFETY_GATE_HINT
                        )
                    )
                continue
            if command in CMD_ERRORS or command in CMD_WARNINGS or command == CMD_ALARM:
                raise ControllerProtocolError(
                    "运动指令被控制器拒绝 0x{:04X}: {}".format(
                        command, _frame_message(data)
                    )
                )

    def _ensure_servo_enabled(self, what: str = "运动"):
        """Guarantee 伺服 status==3 in the same breath as sending a motion.

        Why this lives in the adapter and not in each caller
        ----------------------------------------------------
        On this controller the servo does **not** stay enabled by itself.  The
        reset-point safety gate powers it off on every refusal (see
        ``SAFETY_GATE_HINT``), so "the previous command worked" says nothing
        about whether the next one will.  Before 2026-08-22 only
        ``NexBotTcpJog.step`` re-enabled, and every other entry point --
        ``NexBotTcpMoveController.move_tcp``, ``PickPlaceExecutor._move_j`` /
        ``_move_l``, ``SafeRecoveryManager.recover`` -- silently rode on the
        enable that a jog happened to leave behind.  That is the whole of the
        field symptom "得先卡个 bug 点一次点动, 之后传送/抓取才偶尔能动".

        Putting the check here means every present and future caller inherits
        it; a caller that forgets cannot reintroduce the bug.

        The queued-push drain matters: ``0x2311`` answers ``0x2003 status:3``
        immediately, and the gate then pushes ``0`` and ``1`` behind it.  Any
        stale frame left in the 6001 buffer would otherwise be read as the
        answer to *this* query.  It is deliberately called once per
        ``move_j``/``move_l`` **before** the point loop, never between points,
        so it can never swallow the ``0x3D03`` ack of a preceding point.
        """
        if not bool(self.endpoint.ensure_servo_before_motion):
            return None
        self._ensure_open()
        self._drain_pushed_frames(budget_s=0.1)
        status = self.servo_status()
        if int(status) == 3:
            return 3
        # enable_servo re-queries after settling and raises with the field hint
        # when the gate knocks the servo straight back down.
        return self.enable_servo()

    def _send_motion(self, command: int, point: InexbotPoint, velocity: int):
        self._ensure_open()
        if not isinstance(point, InexbotPoint):
            raise ValueError("NexBot motion requires InexbotPoint, got {!r}".format(type(point)))
        position = [float(value) for value in point.axes]
        if self.endpoint.external_axes:
            position = position + [0.0] * self.endpoint.external_axes
        # IMPORTANT (field-verified 2026-08-22 on MOKA MR07S-930 / C1102,
        # firmware 21.05.23): the motion parser reads
        #   robot -> vel -> acc -> dec -> coord -> pos[0..6]
        # The open.inexbot 22.07 doc omits ``acc``/``dec``; without them the
        # controller replies 指令[0x4501/0x4502]参数错误.  With them, MOVJ/MOVL
        # execute (0x3D03 status 2 -> 0), verified via +5mm/-3mm UCS moves.
        payload = {
            "robot": self.endpoint.robot,
            "vel": velocity,
            "acc": 10,
            "dec": 10,
            "coord": int(point.coordinate_system),
            "pos": position,
        }
        self.motion.send_frame(command, payload)

    def _request_state(self, query_type: Sequence[str]):
        self._ensure_open()
        payload = {
            "channel": self.endpoint.channel,
            "robot": self.endpoint.robot,
            "mode": 0,
            "queryType": list(query_type),
        }
        deadline = time.monotonic() + max(2.0, self.endpoint.io_timeout_s * 4.0)
        self.state.send_frame(CMD_QUERY, payload)
        while time.monotonic() < deadline:
            command, data = self.state.read_frame()
            if command == CMD_QUERY_REPLY:
                return data if isinstance(data, dict) else {}
            if command in CMD_ERRORS or command in CMD_WARNINGS or command == CMD_ALARM:
                raise ControllerProtocolError(
                    "controller frame 0x{:04X}: {}".format(
                        command, _frame_message(data)
                    )
                )
        raise ControllerTimeout("state query timed out after {:.1f}s".format(deadline))

    def _query_axis_vel(self):
        data = self._request_state(["axisVel"])
        reply = data.get("replyData") or {}
        values = reply.get("axisVel")
        if isinstance(values, list) and len(values) >= 6:
            return tuple(float(value) for value in values[:6])
        return None

    def _wait_motion_finish(self):
        """等运动结束：以位姿变化为准（对 axisVel 单位语义不敏感）。

        现场实测：axisVel 在本固件存在单位歧义（疑似 ×57.3 度/秒），
        21.05.23 上按 rad/s 阈值判断会永远"未停止"→ 卡死 60s/次。
        改为连续两个 0.5s 窗口的完整位姿无变化才视为静止。

        ⚠️ 这个函数**不能**单独用来判断"运动成功"：指令被拒时位姿本来就不变，
        它会立刻满足"静止"条件并返回，产出假成功。真正的成功判据是
        ``_await_motion_ack()`` 收到 ``0x3D03 status=2``；本函数只负责
        "已经确认动起来之后，等它停下来"。调用顺序必须是 ack 在前、本函数在后。
        """
        deadline = time.monotonic() + self.endpoint.motion_finish_timeout_s
        last_pose = None
        last_t = None
        quiet_windows = 0
        while time.monotonic() < deadline:
            state = self.read_state(now_s=time.time())
            pose = np.asarray(state.base_from_gripper, dtype=np.float64)
            now = time.monotonic()
            if last_pose is None:
                last_pose = pose.copy()
                last_t = now
            elif now - last_t >= 0.5:
                translation_delta = float(
                    np.linalg.norm(pose[:3, 3] - last_pose[:3, 3])
                )
                relative_rotation = last_pose[:3, :3].T @ pose[:3, :3]
                cosine = np.clip(
                    (np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0
                )
                rotation_delta_deg = float(np.degrees(np.arccos(cosine)))
                if translation_delta < 0.0003 and rotation_delta_deg < 0.1:
                    quiet_windows += 1
                else:
                    quiet_windows = 0
                if quiet_windows >= 4:  # >= ~2s 持续静止（覆盖指令延迟竞态）
                    return
                last_pose = pose.copy()
                last_t = now
            time.sleep(0.1)
        raise ControllerTimeout(
            "motion did not finish within {:.1f}s".format(
                self.endpoint.motion_finish_timeout_s
            )
        )

    # -- RobotController interface ----------------------------------------

    def read_state(self, now_s=None) -> RobotState:
        # Field-tested on MOKA MR07S-930 / Inexbot C1102 (RTL-22.07).  The
        # state service reports TCP poses in three frames; which one forms the
        # pipeline world frame is chosen by ``pose_frame``:
        #   PCS = tool coordinate (TCP w.r.t. robot base, hand-eye sampling),
        #   MCS = Cartesian (flange w.r.t. base),
        #   UCS = user coordinate (TCP w.r.t. the active user frame = user 1).
        # The full-migration workflow uses "UCS" so every pose is already in
        # 用户坐标系1; ACS (joints) stays useful for cross-checks.
        frame = str(self.endpoint.pose_frame).upper()
        if frame not in ("PCS", "MCS", "UCS"):
            raise ValueError("pose_frame must be PCS/MCS/UCS, got {!r}".format(frame))
        query_payload = ["realPos{}".format(frame)]
        if frame != "MCS":
            query_payload += ["realPosMCS"]
        query_payload += ["realPosACS"]
        data = self._request_state(query_payload)
        reply = data.get("replyData") or {}
        pose = reply.get("realPos{}".format(frame))
        if not isinstance(pose, list) or len(pose) < 6:
            pose = reply.get("realPosMCS")
            if not isinstance(pose, list) or len(pose) < 6:
                raise ControllerProtocolError(
                    "state reply is missing realPos{}: {}".format(
                        frame, json.dumps(data, ensure_ascii=False)[:200]
                    )
                )
        timestamp = reply.get("timestamp")
        timestamp_s = None
        if isinstance(timestamp, list) and len(timestamp) >= 2:
            timestamp_s = float(timestamp[0]) + float(timestamp[1]) / 1e9
        xyz_mm = np.asarray(pose[:3], dtype=np.float64)
        abc_rad = np.asarray(pose[3:6], dtype=np.float64)
        # IMPORTANT: NexBot A/B/C are intrinsic X'Y'Z' (== fixed ZYX), so the
        # rotation is Rx(A)Ry(B)Rz(C) -- NOT the fixed XYZ order that
        # transform_from_xyz_rpy uses.  Field-verified 2026-08-22: using the
        # wrong order broke the checkerboard hand-eye solve by ~190 mm.
        world_from_gripper = transform_from_inexbot_abc(
            xyz_mm / 1000.0, abc_rad
        )
        return RobotState(
            valid=True,
            # NOTE: with pose_frame="UCS" this is 用户坐标系1 下的 TCP 位姿,
            # i.e. "world" means the pipeline reference frame (user 1).
            base_from_gripper=world_from_gripper,
            timestamp_s=float(now_s if now_s is not None else (timestamp_s or time.time())),
            simulated=False,
            reason="",
        )

    def move_to(self, base_from_gripper, speed_scale=0.1):
        matrix = as_transform(base_from_gripper, "base_from_gripper")
        xyz_m, abc_rad = inexbot_abc_from_transform(matrix)
        point = InexbotPoint(
            name="P0001",
            coordinate_system=int(self.endpoint.motion_coord),
            angle_unit=1,
            shape=1,
            tool_id=int(self.endpoint.tool_id),
            user_id=int(self.endpoint.user_id),
            axes=[*(xyz_m * 1000.0), *abc_rad, 0.0],
        )
        velocity = _clamp(round(float(speed_scale) * 1000.0), 1, 1000)
        self.move_l([point], speed_mm_s=velocity)

    def move_j(self, points, speed_scale=0.1):
        points = tuple(points)
        if not points:
            raise ValueError("move_j requires at least one point")
        velocity = _clamp(round(float(speed_scale) * 100.0), 1, 100)
        # Once per call, before the loop -- see _ensure_servo_enabled for why it
        # must not run between points.
        self._ensure_servo_enabled("MOVJ")
        for point in points:
            self._send_motion(CMD_MOVJ, point, velocity)
            # Positive confirmation that the motion started -- never trust the
            # "pose stopped changing" heuristic on its own (see _await_motion_ack).
            self._await_motion_ack()
            if self.endpoint.wait_for_finish:
                self._wait_motion_finish()

    def move_l(self, points, speed_mm_s=30.0):
        points = tuple(points)
        if not points:
            raise ValueError("move_l requires at least one point")
        velocity = _clamp(round(float(speed_mm_s)), 1, 1000)
        self._ensure_servo_enabled("MOVL")
        for point in points:
            self._send_motion(CMD_MOVL, point, velocity)
            self._await_motion_ack()
            if self.endpoint.wait_for_finish:
                self._wait_motion_finish()

    def stop(self):
        """发 ``0x2314``。⚠️ 这**不是**受控急停, 是直接下电。

        实测 (2026-08-22 控制器日志, 5/5 次): ``0x2314`` 在这台 C1102 上映射到
        ``Deadan_End -> 设置脉冲使能为 0 -> PowerOff``。伺服失力, 伸展着的手臂会
        **靠自重坠落** —— 当天摔臂的最后一步正是控制器自己触发的 PowerOff。

        因此:

        - 真正的安全急停只能是示教器上的**物理急停按钮**;
        - 调用方在"运动还没开始"时不要为了保险发这条 (见
          ``PickPlaceExecutor.execute`` 的 SafetyInterlockError 分支);
        - 发过之后伺服是断电的, 下一条运动必须重新 ``0x2311`` 使能
          (``_ensure_servo_enabled`` 已自动处理)。
        """
        self._ensure_open()
        self.motion.send_frame(CMD_EMERGENCY_STOP, {"robot": self.endpoint.robot})
        self._check_errors()

    # -- teach-pendant style helpers --------------------------------------

    def servo_status(self) -> int:
        """伺服状态 (0x2002): 0 停止 / 1 就绪 / 2 错误 / 3 运行。

        Field-verified on C1102/21.05.23.  ``0x2301 deadman`` does NOT enable
        on this firmware; ``enable_servo`` (0x2311) is the working channel.
        """
        self._ensure_open()
        self.motion.send_frame(CMD_SERVO_INQUIRE, {"robot": self.endpoint.robot})
        deadline = time.monotonic() + max(2.0, self.endpoint.io_timeout_s * 4.0)
        while time.monotonic() < deadline:
            command, data = self.motion.read_frame()
            if command == CMD_SERVO_RESPOND:
                return int((data or {}).get("status", 0))
            if command in CMD_ERRORS or command in CMD_WARNINGS or command == CMD_ALARM:
                raise ControllerProtocolError(
                    "controller frame 0x{:04X}: {}".format(
                        command, _frame_message(data)
                    )
                )
        raise ControllerTimeout("servo status query timed out")

    def enable_servo(self, settle_s: float = 0.8) -> int:
        """上位机使能 (0x2311) 并返回伺服状态；状态 != 3 抛异常。

        The queued push from ``0x2311`` is drained before re-querying: the
        controller answers the enable with ``0x2003 status:3`` immediately, and
        if the safety gate then knocks the servo back down it pushes ``0`` and
        ``1`` right behind it.  Reading only the first frame reports "enabled"
        for a servo that is already off -- that is the "刚才打开了运动伺服但是
        瞬间关闭了" symptom from the field session.
        """
        self._ensure_open()
        self.motion.send_frame(CMD_ENABLE, {"robot": self.endpoint.robot})
        time.sleep(max(0.0, float(settle_s)))
        self._drain_pushed_frames()
        status = self.servo_status()
        if status != 3:
            raise ControllerProtocolError(
                "0x2311 上使能后伺服仍为 status={} (需要 3=运行)。{}".format(
                    status, SAFETY_GATE_HINT
                )
            )
        return status

    def go_home(self):
        """回零 (0x3002 GO_HOME, robot 0=机器人在回零/1=外部轴).

        WARNING: goes through the controller's ``startRobotJobTask(safepos=1)``
        entry point, i.e. it is subject to the reset-point safety gate.  In
        remote mode a refusal here powers the servo off -- see SAFETY_GATE_HINT.
        """
        self._ensure_open()
        self._ensure_servo_enabled("回零")
        self.motion.send_frame(CMD_GO_HOME, {"robot": self.endpoint.robot, "type": 0})
        self._await_motion_ack()
        if self.endpoint.wait_for_finish:
            self._wait_motion_finish()

    def go_reset_position(self):
        """回复位点 (0x3007 GO_RESET_POSITION); 现场约定复位点=拍摄点.

        WARNING: same ``startRobotJobTask(safepos=1)`` entry point as
        ``go_home``/MOVL.  This is the call that silently de-energised the arm
        at the start of every grasp attempt on 2026-08-22: the gate refused it,
        the controller powered off, and the following MOVLs went nowhere while
        the gripper (0x3601, a different code path) kept working.
        ``_await_motion_ack`` now turns that into a loud exception.
        """
        self._ensure_open()
        self._ensure_servo_enabled("回复位点")
        self.motion.send_frame(CMD_GO_RESET_POSITION, {"robot": self.endpoint.robot})
        self._await_motion_ack()
        if self.endpoint.wait_for_finish:
            self._wait_motion_finish()

    def set_digital_output(self, port: int, status: int):
        """GPIO_DOUT_SET 0x3601: port 从 1 开始; status 0 低/1 高."""
        self._ensure_open()
        self.motion.send_frame(
            CMD_DOUT_SET, {"port": int(port), "status": 1 if int(status) else 0}
        )

    def digital_output_states(self):
        """GPIO_DOUT_INQUIRE 0x3602 -> 0x3603 返回每个 DOUT 的状态数组[0/1]."""
        self._ensure_open()
        self.motion.send_frame(CMD_DOUT_QUERY, {})
        deadline = time.monotonic() + max(2.0, self.endpoint.io_timeout_s * 4.0)
        while time.monotonic() < deadline:
            command, data = self.motion.read_frame()
            if command == CMD_DOUT_QUERY_REPLY:
                status = (data or {}).get("status")
                if isinstance(status, list):
                    return [1 if int(value) == 1 else 0 for value in status]
                raise ControllerProtocolError(
                    "DOUT reply is missing status array: {}".format(
                        json.dumps(data, ensure_ascii=False)[:200]
                    )
                )
            if command in CMD_ERRORS or command in CMD_WARNINGS or command == CMD_ALARM:
                raise ControllerProtocolError(
                    "controller frame 0x{:04X}: {}".format(
                        command, _frame_message(data)
                    )
                )
        raise ControllerTimeout(
            "DOUT query timed out after {:.1f}s".format(deadline)
        )


def nexbot_tcp_client_from_config(settings):
    """Build a ``NexBotTcpEndpoint`` from the ``controller.nexbot_tcp`` section."""
    controller = settings.get("controller", {}) if hasattr(settings, "get") else {}
    config = controller.get("nexbot_tcp", {}) if isinstance(controller, dict) else {}
    if not isinstance(config, dict):
        raise ValueError("controller.nexbot_tcp must be a mapping")
    host = str(config.get("host", "")).strip()
    if not host:
        raise ValueError(
            "controller.nexbot_tcp.host is required for the nexbot_tcp robot adapter"
        )
    return NexBotTcpEndpoint(
        host=host,
        port_motion=int(config.get("port_motion", 6001)),
        port_state=int(config.get("port_state", 7000)),
        robot=int(config.get("robot", 1)),
        channel=int(config.get("channel", 1)),
        connect_timeout_s=float(config.get("connect_timeout_s", 2.0)),
        io_timeout_s=float(config.get("io_timeout_s", 1.0)),
        keepalive=bool(config.get("keepalive", True)),
        max_frame_bytes=int(config.get("max_frame_bytes", 1024 * 1024)),
        external_axes=int(config.get("external_axes", 0)),
        wait_for_finish=bool(config.get("wait_for_finish", True)),
        motion_finish_timeout_s=float(config.get("motion_finish_timeout_s", 60.0)),
        velocity_eps_rad_s=float(config.get("velocity_eps_rad_s", 0.02)),
        heartbeat_s=float(config.get("heartbeat_s", 0.0)),
        motion_ack_timeout_s=float(config.get("motion_ack_timeout_s", 3.0)),
        ensure_servo_before_motion=bool(
            config.get("ensure_servo_before_motion", True)
        ),
    )


__all__ = [
    "CMD_ALARM",
    "CMD_EMERGENCY_STOP",
    "CMD_GO_HOME",
    "CMD_GO_RESET_POSITION",
    "CMD_MOVJ",
    "CMD_MOVL",
    "CMD_ENABLE",
    "CMD_PROGRAM_STATUS",
    "CMD_SERVO_INQUIRE",
    "CMD_SERVO_RESPOND",
    "CMD_QUERY",
    "CMD_QUERY_REPLY",
    "CMD_DOUT_SET",
    "CMD_DOUT_QUERY",
    "CMD_DOUT_QUERY_REPLY",
    "FRAME_LOG_ENV",
    "FRAME_LOG_HEX_BYTES_ENV",
    "SAFETY_GATE_HINT",
    "ControllerConnectionError",
    "ControllerProtocolError",
    "ControllerTimeout",
    "NexBotTcpEndpoint",
    "NexBotTcpRobotController",
    "NexBotTcpTransport",
    "build_frame",
    "nexbot_tcp_client_from_config",
    "read_frame",
]
