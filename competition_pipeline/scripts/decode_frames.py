#!/usr/bin/env python3
"""把 NexBot 6001/7000 上的字节流翻译成人能读的帧, 并**独立复算 CRC**。

为什么要有这个东西
------------------
现场最难缠的一类故障是"控制器像是没收到"。光看我们这一端的日志永远分不清:

    (a) 我们发的帧本身就是坏的 —— CRC 对不上, 控制器丢弃且不吭声;
    (b) 帧完好无损地送到了, 是控制器**不认**这条指令(参数错 / 安全闸门拒绝)。

这两种情况现场处置完全相反: (a) 改我们的代码, (b) 去示教器改配置。CRC 一算就
分开了 —— 这也是本脚本存在的全部理由。

为什么**不 import 适配器**
--------------------------
CRC 的算法在这里是**重新写的一份**, 故意不复用 ``nexbot_tcp.build_frame``。用同
一份代码算 CRC 等于自己给自己判卷: 万一 build_frame 的 CRC 覆盖范围写错了(比如
漏了 length 字段), 拿它去校验只会一路"通过"。这里按协议文档独立实现, 两边对不上
才有意义。副作用是本脚本只依赖标准库, 系统自带的 python3 也能直接跑。

三种输入
--------
1. 帧日志(``NEXBOT_FRAME_LOG`` 产出的行 JSON)::

       python -m competition_pipeline.scripts.decode_frames /tmp/frames.log

2. 一段 hex(从别的日志里抠出来的原始字节, 空格/冒号/换行都能容忍)::

       python -m competition_pipeline.scripts.decode_frames --hex 4e66000b23147b22726f626f74223a317d...

3. tcpdump 存的 pcap(需要 root 抓包: ``sudo tcpdump -i eth0 -s 0 -w /tmp/c.pcap 'tcp port 6001 or tcp port 7000'``)::

       python -m competition_pipeline.scripts.decode_frames /tmp/c.pcap

   pcap 是自己按格式解的(本机没有 scapy/tshark): 24 字节全局头 + 每包 16 字节记
   录头, 链路层支持 Ethernet / Linux cooked(``-i any``) / raw IP, 网络层只认
   IPv4+TCP, 然后按 TCP 序号把每条流拼回字节流, 再按 0x4E66 同步字切帧。
   **不支持 pcapng**(``tcpdump -w`` 默认就是经典 pcap, 不用管; 只有 dumpcap /
   wireshark 存出来的才是 pcapng), 遇到会明确报错而不是瞎猜。

常用开关::

    --bad-only     只看 CRC 失败 / 解析失败的帧(现场排查先看这个)
    --only 0x4502,0x2B03   只看指定指令字
    --limit 50     只解前 N 帧
"""

import argparse
import binascii
import json
import re
import struct
import sys
import zlib

SYNC = b"\x4E\x66"
#: 同步字(2) + 长度(2) + 指令字(2) + CRC(4); 帧内 JSON 之外的固定开销。
FRAME_OVERHEAD = 10

#: 上位机侧端口。pcap 里靠它判方向: 目的端口是它 = 我们发的。
DEFAULT_CONTROLLER_PORTS = (6001, 7000)

#: 指令字含义。带"实测"字样的是 2026-08-22 现场日志里直接读出来的, 其余来自
#: RTL-22.07 协议文档。查不到的照实说"未知", 不编。
COMMAND_MEANINGS = {
    0x1E00: "状态查询(官方示例用的指令字)",
    0x2001: "伺服状态推送(实测出现在安全闸门拒绝链路: status=0 即已下电)",
    0x2002: "伺服状态查询",
    0x2003: "伺服状态应答 0停止/1就绪/2错误/3运行",
    0x2311: "上位机使能(伺服上电)",
    0x2314: "急停 —— 实测在 C1102 上等于直接 PowerOff, 伺服失力会坠臂",
    0x2401: "实测出现在安全闸门拒绝链路(stop 之后, 语义未在文档中确认)",
    0x2B03: "报警推送(真实文本在 JSON 的 data 键)",
    0x2F16: "实测: 控制器对 0x3007 之类指令的回复帧(语义未在文档中确认)",
    0x3002: "回零 GO_HOME",
    0x3007: "回复位点 GO_RESET_POSITION(现场约定复位点=拍摄点)",
    0x3601: "DOUT 置位 GPIO_DOUT_SET",
    0x3602: "DOUT 状态查询",
    0x3603: "DOUT 状态应答",
    0x3D03: "作业运行状态推送 0停止/2运行中 —— 运动真的开始了的唯一权威信号",
    0x4501: "MOVJ 关节插补运动",
    0x4502: "MOVL 直线插补运动",
    0x4503: "MOVC 圆弧插补运动",
    0x4504: "MOVS 样条插补运动",
    0x6010: "控制器错误",
    0x6020: "控制器错误",
    0x6030: "控制器错误",
    0x6040: "控制器错误",
    0x6110: "控制器警告",
    0x6210: "控制器警告",
    0x7266: "心跳",
    0x7267: "心跳应答",
    0x9512: "状态查询(7000 端口)",
    0x9513: "状态查询应答(7000 端口)",
}

#: 这些帧一出现就是坏消息, 渲染时单独提一句, 免得淹在正常帧里。
ALARM_COMMANDS = frozenset({0x2B03, 0x6010, 0x6020, 0x6030, 0x6040, 0x6110, 0x6210})


def command_meaning(command):
    return COMMAND_MEANINGS.get(command, "未知指令字(协议文档与现场日志里都没有)")


def alarm_text(data):
    """报警帧里的人话。真控制器把文本放在 ``data`` 键, 不是 ``message``。"""
    if isinstance(data, dict):
        for key in ("data", "message", "error", "msg", "desc"):
            value = data.get(key)
            if value not in (None, ""):
                return str(value)
    return None


class Frame(object):
    """一条解码结果。``ok`` 为假时 ``problem`` 说明是哪种坏法。"""

    __slots__ = ("time", "direction", "port", "command", "data", "raw",
                 "crc_found", "crc_expected", "problem", "source")

    def __init__(self, time=None, direction="?", port=None, command=None,
                 data=None, raw=b"", crc_found=None, crc_expected=None,
                 problem=None, source=""):
        self.time = time
        self.direction = direction
        self.port = port
        self.command = command
        self.data = data
        self.raw = raw
        self.crc_found = crc_found
        self.crc_expected = crc_expected
        self.problem = problem
        self.source = source

    @property
    def crc_ok(self):
        """None = 没法判(hex 被截断 / 根本没读到 CRC)。"""
        if self.crc_found is None or self.crc_expected is None:
            return None
        return self.crc_found == self.crc_expected

    @property
    def ok(self):
        return self.problem is None and self.crc_ok is not False


# -- 单帧解码 --------------------------------------------------------------


def decode_frame_bytes(blob, truncated_from=None):
    """解一段"应该正好是一帧"的字节。

    ``truncated_from`` 不是 None 表示这段 hex 在落盘时被截断过, 原长是它 ——
    这种情况下 CRC 不在手上, 只能明说"无法校验", 不能默认判通过。
    """
    frame = Frame(raw=blob)
    if len(blob) < 6:
        frame.problem = "半截帧: 只有 {} 字节, 连帧头(6 字节)都不完整".format(len(blob))
        return frame
    if blob[:2] != SYNC:
        frame.problem = "同步字不是 0x4E66, 实际 0x{}".format(
            binascii.hexlify(blob[:2]).decode("ascii").upper()
        )
        return frame
    length = struct.unpack(">H", blob[2:4])[0]
    frame.command = struct.unpack(">H", blob[4:6])[0]
    expected_total = FRAME_OVERHEAD + length
    payload = blob[6:6 + length]

    if truncated_from is not None:
        frame.problem = (
            "hex 在落盘时被截断({} -> {} 字节), CRC 无法校验; "
            "需要完整字节就调大 NEXBOT_FRAME_LOG_HEX_BYTES".format(
                truncated_from, len(blob)
            )
        )
    elif len(blob) < expected_total:
        frame.problem = (
            "半截帧: 帧头声明 {} 字节数据(整帧应为 {} 字节), 实际只有 {} 字节".format(
                length, expected_total, len(blob)
            )
        )
    else:
        frame.crc_found = struct.unpack(">I", blob[expected_total - 4:expected_total])[0]
        # 独立复算: CRC32 覆盖 长度+指令字+数据, 不含同步字, 不含 CRC 自身。
        frame.crc_expected = zlib.crc32(blob[2:expected_total - 4]) & 0xFFFFFFFF
        if len(blob) > expected_total:
            frame.problem = "帧尾多出 {} 字节(可能是粘包没切干净)".format(
                len(blob) - expected_total
            )

    # 只有拿到完整数据段才谈得上解 JSON。半截/被截断的数据段拿去解必然失败, 报一句
    # "不是合法 JSON"是误导 —— 真正的问题是字节没收全, 而不是控制器发了坏 JSON。
    complete = truncated_from is None and len(payload) == length
    if payload and complete:
        try:
            frame.data = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            if frame.problem is None:
                frame.problem = "数据段不是合法 JSON: {}".format(error)
            frame.data = payload.decode("utf-8", "replace")
    return frame


def iter_frames_from_stream(buffer, direction="?", port=None, source=""):
    """从一段拼好的字节流里按同步字切帧。

    切不动的时候的策略(重要): CRC 一旦不匹配, **长度字段本身就不可信**了, 顺着
    它往下切会把后面所有帧全部错开、变成一片假故障。所以 CRC 不过时先看帧尾是不是
    正好接上下一个同步字(或流末尾) —— 接上了就说明长度是对的、坏的只是内容, 照常
    往下走; 接不上才退回到"从下一个 0x4E66 重新找", 宁可多报一帧也不要一路错开。
    """
    view = bytes(buffer)
    index = 0
    total = len(view)
    while index < total:
        start = view.find(SYNC, index)
        if start < 0:
            if total - index > 0:
                yield Frame(direction=direction, port=port, source=source,
                            raw=view[index:],
                            problem="{} 字节找不到同步字 0x4E66 的垃圾数据".format(
                                total - index))
            return
        if start > index:
            yield Frame(direction=direction, port=port, source=source,
                        raw=view[index:start],
                        problem="同步字之前有 {} 字节游离数据".format(start - index))
        if total - start < 6:
            yield Frame(direction=direction, port=port, source=source,
                        raw=view[start:],
                        problem="半截帧: 流末尾只剩 {} 字节, 帧头都不够".format(
                            total - start))
            return
        length = struct.unpack(">H", view[start + 2:start + 4])[0]
        end = start + FRAME_OVERHEAD + length
        if end > total:
            frame = decode_frame_bytes(view[start:])
            frame.direction, frame.port, frame.source = direction, port, source
            yield frame
            return
        frame = decode_frame_bytes(view[start:end])
        frame.direction, frame.port, frame.source = direction, port, source
        yield frame
        if frame.crc_ok or end == total or view[end:end + 2] == SYNC:
            index = end
        else:
            index = start + 2


# -- 输入 1: 帧日志 --------------------------------------------------------


def parse_frame_log(text, source=""):
    """解 ``NEXBOT_FRAME_LOG`` 写出来的行 JSON。"""
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            yield Frame(source="{}:{}".format(source, number),
                        problem="不是合法的帧日志行(应为行 JSON): {}".format(line[:80]))
            continue
        raw = b""
        hex_text = record.get("hex") or ""
        if hex_text:
            try:
                raw = bytes.fromhex(hex_text)
            except ValueError:
                yield Frame(source="{}:{}".format(source, number),
                            problem="hex 字段不是合法十六进制")
                continue
        frame = decode_frame_bytes(raw, truncated_from=record.get("trunc"))
        frame.time = record.get("t")
        frame.direction = record.get("dir", "?")
        frame.port = record.get("port")
        frame.source = "{}:{}".format(source, number)
        if record.get("err") and frame.problem is None:
            # 适配器当时就报错了(超时/断连), 原样带出来 —— 那条错误信息往往比
            # 我们从字节里反推出的结论更准。
            frame.problem = "适配器报错: {}".format(record["err"])
        elif record.get("err"):
            frame.problem = "{} | 适配器报错: {}".format(frame.problem, record["err"])
        # 日志里已经存了解析好的 JSON; 字节被截断时它是唯一还看得到内容的地方。
        if frame.data is None and "json" in record:
            frame.data = record["json"]
        yield frame


# -- 输入 2: hex 串 --------------------------------------------------------

_HEX_JUNK = re.compile(r"(?:0x)|[^0-9a-fA-F]")


def parse_hex_text(text, source="hex"):
    """解一段 hex。空格/冒号/换行/``0x`` 前缀都当分隔符扔掉。"""
    cleaned = _HEX_JUNK.sub("", text)
    if not cleaned:
        yield Frame(source=source, problem="没有可解析的十六进制字符")
        return
    if len(cleaned) % 2:
        yield Frame(source=source,
                    problem="hex 字符数是奇数({}), 最后半个字节被丢弃".format(len(cleaned)))
        cleaned = cleaned[:-1]
    for frame in iter_frames_from_stream(bytes.fromhex(cleaned), source=source):
        yield frame


# -- 输入 3: pcap ----------------------------------------------------------

PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

LINKTYPE_NULL = 0
LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113
LINKTYPE_LINUX_SLL2 = 276


class PcapError(Exception):
    """pcap 结构本身有问题 —— 和"帧解坏了"是两回事, 不要混在一起报。"""


def _strip_link_layer(link_type, packet):
    """剥掉链路层, 返回 IPv4 报文; 不是 IPv4 就返回 None。"""
    if link_type == LINKTYPE_ETHERNET:
        if len(packet) < 14:
            return None
        ethertype = struct.unpack(">H", packet[12:14])[0]
        offset = 14
        # VLAN tag: 0x8100 / QinQ 0x88A8, 每层再吃掉 4 字节。
        while ethertype in (0x8100, 0x88A8) and len(packet) >= offset + 4:
            ethertype = struct.unpack(">H", packet[offset + 2:offset + 4])[0]
            offset += 4
        return packet[offset:] if ethertype == 0x0800 else None
    if link_type == LINKTYPE_LINUX_SLL:
        # tcpdump -i any 的格式: 16 字节头, 协议号在最后 2 字节。
        if len(packet) < 16:
            return None
        return packet[16:] if struct.unpack(">H", packet[14:16])[0] == 0x0800 else None
    if link_type == LINKTYPE_LINUX_SLL2:
        if len(packet) < 20:
            return None
        return packet[20:] if struct.unpack(">H", packet[0:2])[0] == 0x0800 else None
    if link_type == LINKTYPE_RAW:
        return packet
    if link_type == LINKTYPE_NULL:
        # BSD loopback: 4 字节地址族, 主机序; 2 = AF_INET。
        if len(packet) < 4:
            return None
        family = struct.unpack("<I", packet[:4])[0]
        if family > 0xFFFF:
            family = struct.unpack(">I", packet[:4])[0]
        return packet[4:] if family == 2 else None
    raise PcapError(
        "不支持的链路层类型 {}(只处理 Ethernet/Linux cooked/raw IP)".format(link_type)
    )


def _parse_ipv4_tcp(datagram):
    """返回 ``(src_ip, src_port, dst_ip, dst_port, seq, payload)``, 非 TCP 返回 None。"""
    if len(datagram) < 20 or (datagram[0] >> 4) != 4:
        return None
    header_length = (datagram[0] & 0x0F) * 4
    if header_length < 20 or len(datagram) < header_length:
        return None
    # 用 IP 总长裁剪: 以太网短包会补 0 到 60 字节, 不裁的话尾巴上的填充会被
    # 当成 TCP 载荷, 在流里凭空插进一段 0。
    total_length = struct.unpack(">H", datagram[2:4])[0]
    if 0 < total_length <= len(datagram):
        datagram = datagram[:total_length]
    if datagram[9] != 6:
        return None
    source_ip = ".".join(str(byte) for byte in datagram[12:16])
    dest_ip = ".".join(str(byte) for byte in datagram[16:20])
    segment = datagram[header_length:]
    if len(segment) < 20:
        return None
    source_port, dest_port = struct.unpack(">HH", segment[0:4])
    seq = struct.unpack(">I", segment[4:8])[0]
    data_offset = (segment[12] >> 4) * 4
    if data_offset < 20 or len(segment) < data_offset:
        return None
    return source_ip, source_port, dest_ip, dest_port, seq, segment[data_offset:]


class _TcpStream(object):
    """一个方向的 TCP 流, 按序号拼回连续字节。"""

    def __init__(self):
        self.buffer = bytearray()
        self.next_seq = None
        self.gaps = 0
        self.retransmits = 0
        self.first_time = None

    def add(self, seq, payload, timestamp):
        if not payload:
            return
        if self.first_time is None:
            self.first_time = timestamp
        if self.next_seq is None:
            self.next_seq = seq
        delta = (seq - self.next_seq) & 0xFFFFFFFF
        if delta == 0:
            self.buffer += payload
        elif delta > 0x7FFFFFFF:
            # 序号落在期望之前 = 重传或重叠。只补上超出已有数据的那一截。
            overlap = (self.next_seq - seq) & 0xFFFFFFFF
            self.retransmits += 1
            if overlap >= len(payload):
                return
            self.buffer += payload[overlap:]
        else:
            # 中间少了一段(内核丢包或抓包起点在连接中途)。照拼不误, 后面靠同步字
            # 重新对齐; 但要记一笔, 否则报出来的"坏帧"其实是抓包漏了。
            self.gaps += 1
            self.buffer += payload
        self.next_seq = (seq + len(payload)) & 0xFFFFFFFF


def parse_pcap(blob, controller_ports=DEFAULT_CONTROLLER_PORTS, source="pcap"):
    """解经典 pcap, 返回 ``(frames, notes)``。"""
    if blob[:4] == PCAPNG_MAGIC:
        raise PcapError(
            "这是 pcapng, 本脚本只解经典 pcap。请用 `tcpdump -w out.pcap` 重抓"
            "(tcpdump 默认就写经典 pcap; pcapng 是 dumpcap/wireshark 的默认格式)。"
        )
    if len(blob) < 24:
        raise PcapError("文件不足 24 字节, 连 pcap 全局头都不完整")
    endian_scale = PCAP_MAGICS.get(blob[:4])
    if endian_scale is None:
        raise PcapError(
            "pcap 魔数不认识: 0x{}".format(
                binascii.hexlify(blob[:4]).decode("ascii").upper()
            )
        )
    endian, time_scale = endian_scale
    link_type = struct.unpack(endian + "I", blob[20:24])[0]
    snap_length = struct.unpack(endian + "I", blob[16:20])[0]

    notes = []
    streams = {}
    offset = 24
    packet_count = 0
    truncated = 0
    non_ipv4 = 0
    while offset + 16 <= len(blob):
        ts_sec, ts_frac, incl_len, orig_len = struct.unpack(
            endian + "IIII", blob[offset:offset + 16]
        )
        offset += 16
        if incl_len > len(blob) - offset:
            notes.append(
                "文件在第 {} 包中间截断(记录头声明 {} 字节, 只剩 {} 字节)——"
                "抓包进程多半是被强杀的".format(packet_count + 1, incl_len, len(blob) - offset)
            )
            break
        packet = blob[offset:offset + incl_len]
        offset += incl_len
        packet_count += 1
        if incl_len < orig_len:
            truncated += 1
        datagram = _strip_link_layer(link_type, packet)
        if datagram is None:
            non_ipv4 += 1
            continue
        parsed = _parse_ipv4_tcp(datagram)
        if parsed is None:
            non_ipv4 += 1
            continue
        source_ip, source_port, dest_ip, dest_port, seq, payload = parsed
        key = (source_ip, source_port, dest_ip, dest_port)
        stream = streams.get(key)
        if stream is None:
            stream = streams[key] = _TcpStream()
        stream.add(seq, payload, ts_sec + ts_frac * time_scale)

    if truncated:
        notes.append(
            "有 {} 个包被 snaplen({}) 截断, 载荷不完整 —— 抓包时请加 `-s 0`, "
            "否则大帧必然报 CRC 失败(是抓包丢的, 不是线上坏的)".format(truncated, snap_length)
        )
    if non_ipv4:
        notes.append("跳过 {} 个非 IPv4/TCP 包".format(non_ipv4))
    notes.append("共 {} 个包, {} 条 TCP 流, 链路层类型 {}".format(
        packet_count, len(streams), link_type))

    frames = []
    ports = set(int(port) for port in controller_ports)
    for key in sorted(streams, key=lambda item: (streams[item].first_time or 0.0, item)):
        stream = streams[key]
        source_ip, source_port, dest_ip, dest_port = key
        if dest_port in ports:
            direction, port = "tx", dest_port
        elif source_port in ports:
            direction, port = "rx", source_port
        else:
            direction, port = "?", dest_port
        if stream.gaps:
            notes.append("流 {}:{} -> {}:{} 有 {} 处序号空洞(抓包漏了)".format(
                source_ip, source_port, dest_ip, dest_port, stream.gaps))
        if stream.retransmits:
            notes.append("流 {}:{} -> {}:{} 有 {} 个重传/重叠段(已按序号去重)".format(
                source_ip, source_port, dest_ip, dest_port, stream.retransmits))
        label = "{}  {}:{} -> {}:{}".format(source, source_ip, source_port,
                                            dest_ip, dest_port)
        for frame in iter_frames_from_stream(stream.buffer, direction=direction,
                                             port=port, source=label):
            frames.append(frame)
    return frames, notes


# -- 渲染 ------------------------------------------------------------------

ARROWS = {"tx": "-->", "rx": "<--", "?": "  ?"}
DIRECTION_WORDS = {"tx": "上位机->控制器", "rx": "控制器->上位机", "?": "方向未知"}


def render(frame, show_hex=False):
    lines = []
    head = [frame.time or "", ARROWS.get(frame.direction, "  ?")]
    if frame.port is not None:
        head.append("{:<5}".format(frame.port))
    if frame.command is None:
        # 连指令字都没解出来 = 这段字节根本不成帧。再去谈 CRC 只会误导, 直接说
        # 它是哪种碎片就够了。
        head.append("(不成帧的字节片段)")
        lines.append(" ".join(part for part in head if part != ""))
        lines.append("    !! {}".format(frame.problem or "无法解析"))
        if frame.source:
            lines.append("    ({})".format(frame.source))
        return "\n".join(lines)
    head.append("0x{:04X}".format(frame.command))
    head.append(command_meaning(frame.command))
    lines.append(" ".join(part for part in head if part != ""))

    crc_ok = frame.crc_ok
    if crc_ok is True:
        lines.append("    CRC 通过 0x{:08X}".format(frame.crc_found))
    elif crc_ok is False:
        lines.append(
            "    !! CRC 不匹配: 帧内 0x{:08X}, 按 长度+指令字+数据 复算应为 0x{:08X}"
            .format(frame.crc_found, frame.crc_expected)
        )
        lines.append(
            "       => 线上的字节被改过或错位了。这是**我们发/收的字节本身坏了**, "
            "不是控制器不认这条指令。"
        )
    else:
        lines.append("    CRC 无法校验(没拿到完整的 4 字节 CRC)")

    if frame.problem:
        lines.append("    !! {}".format(frame.problem))
    if frame.data is not None:
        lines.append("    {}".format(json.dumps(frame.data, ensure_ascii=False)))
    if frame.command in ALARM_COMMANDS:
        text = alarm_text(frame.data)
        if text:
            lines.append("    !! 控制器报警原文: {}".format(text))
    if show_hex and frame.raw:
        lines.append("    hex {}".format(
            binascii.hexlify(frame.raw).decode("ascii")))
    if frame.source:
        lines.append("    ({})".format(frame.source))
    return "\n".join(lines)


def summarise(frames):
    # 成帧的和不成帧的碎片分开数: 混在一起会让"CRC 无法判定"看着像一堆坏帧, 而
    # 碎片通常只是抓包起点在连接中途 / 上一帧的残骸。
    real = [frame for frame in frames if frame.command is not None]
    fragments = len(frames) - len(real)
    bad_crc = sum(1 for frame in real if frame.crc_ok is False)
    unknown_crc = sum(1 for frame in real if frame.crc_ok is None)
    problems = sum(1 for frame in real if frame.problem)
    by_command = {}
    for frame in real:
        by_command[frame.command] = by_command.get(frame.command, 0) + 1
    lines = ["", "== 汇总 =="]
    lines.append("帧数 {}  |  CRC 通过 {}  |  CRC 失败 {}  |  CRC 无法判定 {}  |  另有解析问题 {}  |  不成帧的碎片 {}".format(
        len(real), len(real) - bad_crc - unknown_crc, bad_crc, unknown_crc,
        problems, fragments))
    for command in sorted(by_command):
        lines.append("  0x{:04X} x{:<4} {}".format(
            command, by_command[command], command_meaning(command)))
    if not real:
        lines.append("结论: 一帧都没解出来 —— 先确认输入格式和端口选对了。")
        return "\n".join(lines)
    if bad_crc:
        lines.append(
            "结论: 有 {} 帧 CRC 对不上 —— 问题在字节层(发送端算错 / 链路丢改 / "
            "粘包切错), 先查我们这一端, 别去改控制器配置。".format(bad_crc))
    else:
        lines.append(
            "结论: 所有帧 CRC 都通过 —— 字节是好的。控制器若仍无反应, "
            "那是它**不认**这条指令(参数错 / 安全闸门拒绝), 看 0x2B03 与 0x2003。")
    return "\n".join(lines)


# -- 入口 ------------------------------------------------------------------


def detect_format(blob):
    if blob[:4] == PCAPNG_MAGIC or blob[:4] in PCAP_MAGICS:
        return "pcap"
    head = blob[:4096].lstrip()
    if head.startswith(b"{"):
        return "framelog"
    return "hex"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="decode_frames",
        description="解码 NexBot 0x4E66 帧并独立复算 CRC(帧日志 / hex / pcap)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?",
                        help="帧日志 / pcap 文件; 用 - 读标准输入")
    parser.add_argument("--hex", dest="hex_text",
                        help="直接解一段 hex(空格/冒号/0x 前缀都能容忍)")
    parser.add_argument("--format", choices=("auto", "framelog", "hex", "pcap"),
                        default="auto", help="强制输入格式, 默认按内容自动判断")
    parser.add_argument("--only", help="只看这些指令字, 逗号分隔, 如 0x4502,0x2B03")
    parser.add_argument("--bad-only", action="store_true",
                        help="只看 CRC 失败或解析出问题的帧")
    parser.add_argument("--limit", type=int, help="最多输出多少帧")
    parser.add_argument("--show-hex", action="store_true", help="连原始字节一起打印")
    parser.add_argument("--no-summary", action="store_true", help="不打印末尾汇总")
    parser.add_argument("--ports", default=",".join(str(p) for p in DEFAULT_CONTROLLER_PORTS),
                        help="控制器端口(pcap 靠它判方向), 默认 6001,7000")
    return parser


def load_frames(args):
    """返回 ``(frames, notes)``; 输入本身有问题就抛 ``PcapError``/``ValueError``。"""
    if args.hex_text is not None:
        return list(parse_hex_text(args.hex_text)), []
    if not args.path:
        raise ValueError("要么给一个文件路径, 要么用 --hex 给一段十六进制")
    if args.path == "-":
        blob = sys.stdin.buffer.read()
        source = "<stdin>"
    else:
        with open(args.path, "rb") as handle:
            blob = handle.read()
        source = args.path
    if not blob:
        return [], ["输入是空的"]

    kind = args.format if args.format != "auto" else detect_format(blob)
    if kind == "pcap":
        ports = [int(item, 0) for item in args.ports.split(",") if item.strip()]
        return parse_pcap(blob, ports, source=source)
    if kind == "framelog":
        return list(parse_frame_log(blob.decode("utf-8", "replace"), source=source)), []
    return list(parse_hex_text(blob.decode("utf-8", "replace"), source=source)), []


def _restore_default_sigpipe():
    """让 ``| head -30`` 正常收尾, 而不是吐一屏 BrokenPipeError。

    Python 默认把 SIGPIPE 忽略掉再翻译成异常, 于是每次 ``decode_frames ... | head``
    都会在真正的输出后面糊上一段 traceback —— 现场看日志的人第一反应会是"工具坏了"。
    恢复成系统默认行为(直接结束), 就跟 cat/grep 一样。
    """
    try:
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        # 非 POSIX 或不在主线程(被当模块调用): 忽略即可, 不影响解码。
        pass


def main(argv=None):
    _restore_default_sigpipe()
    args = build_parser().parse_args(argv)
    try:
        frames, notes = load_frames(args)
    except (PcapError, ValueError, OSError) as error:
        sys.stderr.write("解码失败: {}\n".format(error))
        return 2

    wanted = None
    if args.only:
        # 一律按十六进制解: 这个领域里指令字从来都是写成 4502/0x4502 的, 按十进制
        # 解会让 `--only 4502` 静默地筛出个不存在的 0x1196, 什么都匹配不到。
        try:
            wanted = set(int(item.strip(), 16)
                         for item in args.only.split(",") if item.strip())
        except ValueError:
            sys.stderr.write("--only 只接受十六进制指令字, 如 0x4502,2B03\n")
            return 2

    for note in notes:
        sys.stderr.write("# {}\n".format(note))

    shown = 0
    for frame in frames:
        if wanted is not None and frame.command not in wanted:
            continue
        if args.bad_only and frame.ok:
            continue
        if args.limit is not None and shown >= args.limit:
            break
        print(render(frame, show_hex=args.show_hex))
        shown += 1

    if not args.no_summary:
        print(summarise(frames))
    # CRC 失败用退出码 1 报出来, 好让脚本/CI 直接判。
    # 故意看**整个输入**而不是 --only/--limit 之后剩下的: 筛选是为了看得清楚,
    # 不该顺手把一帧坏帧藏进退出码里。
    return 1 if any(frame.crc_ok is False for frame in frames) else 0


if __name__ == "__main__":
    sys.exit(main())
