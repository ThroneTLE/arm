"""``decode_frames.py`` 与适配器帧日志(``NEXBOT_FRAME_LOG``)的测试。

重点不在"能不能解开一帧好帧" —— 那是最容易的部分。重点在**坏帧**: 现场真正需要
这套工具的时刻, 线上的字节一定是不正常的(CRC 被改、帧只到一半、粘包切错)。所以
下面每一类坏法都单独立一条用例。

另外必须钉死的一条: **不设 NEXBOT_FRAME_LOG 时一个文件都不许产生。**
明天是竞赛验收, 诊断开关关着的时候必须和没写过这段代码一样。
"""

import contextlib
import inspect
import io
import json
import os
import socket
import struct
import subprocess
import tempfile
import threading
import unittest
import zlib
from pathlib import Path

from competition_pipeline.scripts import decode_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_SRC = REPO_ROOT / "ros_ws" / "src" / "arm_vision_framework" / "src"

import sys

if str(FRAMEWORK_SRC) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_SRC))

from arm_vision_framework.adapters import nexbot_tcp  # noqa: E402
from arm_vision_framework.adapters.nexbot_tcp import (  # noqa: E402
    CMD_ENABLE,
    CMD_SERVO_RESPOND,
    NexBotTcpEndpoint,
    NexBotTcpTransport,
    build_frame,
)


def good_frame(command=0x4502, data=None):
    if data is None:
        data = {"robot": 1, "vel": 30, "acc": 10, "dec": 10, "coord": 3,
                "pos": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.0]}
    return build_frame(command, data)


def corrupt_crc(frame):
    """翻掉 CRC 的最后一个 bit —— 模拟"我们发出去的字节在线上被改了"。"""
    body = bytearray(frame)
    body[-1] ^= 0x01
    return bytes(body)


class DecodeSingleFrameTest(unittest.TestCase):
    def test_good_frame_decodes_with_crc_ok(self):
        frames = list(decode_frames.parse_hex_text(good_frame().hex()))
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame.command, 0x4502)
        self.assertTrue(frame.crc_ok)
        self.assertIsNone(frame.problem)
        self.assertEqual(frame.data["coord"], 3)
        self.assertIn("MOVL", decode_frames.command_meaning(frame.command))

    def test_hex_tolerates_separators_and_prefix(self):
        """从别人的日志里抠出来的 hex 什么分隔符都有, 不能因此解不开。"""
        raw = good_frame(0x2314, {"robot": 1}).hex()
        spaced = "0x" + " ".join(raw[i:i + 2] for i in range(0, len(raw), 2))
        frames = list(decode_frames.parse_hex_text(spaced.replace(" ", ":", 3)))
        self.assertEqual([frame.command for frame in frames], [0x2314])
        self.assertTrue(frames[0].crc_ok)

    def test_corrupt_crc_is_reported_not_swallowed(self):
        frame = list(decode_frames.parse_hex_text(corrupt_crc(good_frame()).hex()))[0]
        self.assertIs(frame.crc_ok, False)
        self.assertFalse(frame.ok)
        # 指令字和 JSON 仍要解出来: 现场需要知道"是哪条指令坏了"。
        self.assertEqual(frame.command, 0x4502)
        self.assertEqual(frame.data["vel"], 30)
        self.assertNotEqual(frame.crc_found, frame.crc_expected)
        text = decode_frames.render(frame)
        self.assertIn("CRC 不匹配", text)

    def test_crc_is_computed_independently_of_the_adapter(self):
        """复算的 CRC 必须等于协议定义(长度+指令字+数据), 不是照抄帧里那 4 字节。"""
        blob = good_frame(0x2002, {"robot": 1})
        frame = decode_frames.decode_frame_bytes(blob)
        self.assertEqual(frame.crc_expected, zlib.crc32(blob[2:-4]) & 0xFFFFFFFF)
        self.assertEqual(frame.crc_found, struct.unpack(">I", blob[-4:])[0])

    def test_half_frame_is_reported_as_half_frame(self):
        blob = good_frame()[:12]
        frame = decode_frames.decode_frame_bytes(blob)
        self.assertIsNone(frame.crc_ok)
        self.assertIn("半截帧", frame.problem)
        self.assertEqual(frame.command, 0x4502)

    def test_frame_shorter_than_header_is_reported(self):
        frame = decode_frames.decode_frame_bytes(b"\x4e\x66\x00")
        self.assertIn("半截帧", frame.problem)
        self.assertIsNone(frame.command)

    def test_wrong_sync_word_is_reported(self):
        frame = decode_frames.decode_frame_bytes(b"\xde\xad" + good_frame()[2:])
        self.assertIn("同步字", frame.problem)

    def test_invalid_json_payload_is_reported(self):
        payload = b"{not json"
        header = struct.pack(">H", len(payload)) + struct.pack(">H", 0x2B03)
        blob = (b"\x4e\x66" + header + payload
                + struct.pack(">I", zlib.crc32(header + payload) & 0xFFFFFFFF))
        frame = decode_frames.decode_frame_bytes(blob)
        self.assertTrue(frame.crc_ok)  # 字节没坏, 是内容不合法 —— 两件事要分开报
        self.assertIn("JSON", frame.problem)


class DecodeStreamTest(unittest.TestCase):
    def test_back_to_back_frames_are_split(self):
        blob = good_frame(0x2002, {"robot": 1}) + good_frame(0x2003, {"status": 3})
        frames = list(decode_frames.iter_frames_from_stream(blob))
        self.assertEqual([frame.command for frame in frames], [0x2002, 0x2003])
        self.assertTrue(all(frame.crc_ok for frame in frames))

    def test_bad_crc_does_not_derail_the_frames_behind_it(self):
        """坏帧之后的好帧必须照常解出来, 否则一个坏帧会伪造出一片假故障。"""
        blob = (corrupt_crc(good_frame(0x4501, {"robot": 1}))
                + good_frame(0x3D03, {"robot": 1, "status": 2}))
        frames = list(decode_frames.iter_frames_from_stream(blob))
        self.assertIs(frames[0].crc_ok, False)
        self.assertEqual(frames[0].command, 0x4501)
        recovered = [f for f in frames if f.command == 0x3D03 and f.crc_ok]
        self.assertEqual(len(recovered), 1)

    def test_trailing_half_frame_at_end_of_stream(self):
        blob = good_frame(0x2002, {"robot": 1}) + good_frame(0x4502)[:9]
        frames = list(decode_frames.iter_frames_from_stream(blob))
        self.assertTrue(frames[0].crc_ok)
        self.assertIn("半截帧", frames[-1].problem)

    def test_garbage_before_first_sync_word_is_flagged(self):
        frames = list(decode_frames.iter_frames_from_stream(b"\x00\x01\x02" + good_frame()))
        self.assertIn("游离数据", frames[0].problem)
        self.assertTrue(frames[1].crc_ok)


class FrameLogParsingTest(unittest.TestCase):
    def _line(self, **overrides):
        record = {"t": "2026-08-23 00:00:00.000", "ts": 1.0, "dir": "tx",
                  "port": 6001, "cmd": "0x4502", "len": 10,
                  "json": {"robot": 1}, "hex": good_frame(0x4502, {"robot": 1}).hex()}
        record.update(overrides)
        return json.dumps(record, ensure_ascii=False)

    def test_parses_direction_port_and_time(self):
        frame = list(decode_frames.parse_frame_log(self._line()))[0]
        self.assertEqual(frame.direction, "tx")
        self.assertEqual(frame.port, 6001)
        self.assertEqual(frame.time, "2026-08-23 00:00:00.000")
        self.assertTrue(frame.crc_ok)
        self.assertIn("-->", decode_frames.render(frame))

    def test_truncated_hex_refuses_to_claim_crc_ok(self):
        """hex 被截断时既不能报通过也不能报失败, 只能说"判不了"。"""
        blob = good_frame(0x9513, {"replyData": {"realPosMCS": [1, 2, 3, 4, 5, 6]}})
        line = self._line(cmd="0x9513", hex=blob[:16].hex(), trunc=len(blob))
        frame = list(decode_frames.parse_frame_log(line))[0]
        self.assertIsNone(frame.crc_ok)
        self.assertIn("截断", frame.problem)
        # 落盘时解析好的 JSON 还在, 截断也不至于什么都看不到。
        self.assertEqual(frame.data, {"robot": 1})

    def test_adapter_error_is_carried_through(self):
        line = self._line(hex="", cmd=None, err="ControllerTimeout: 没有帧")
        record = json.loads(line)
        record.pop("cmd", None)
        frame = list(decode_frames.parse_frame_log(json.dumps(record)))[0]
        self.assertIn("ControllerTimeout", frame.problem)

    def test_non_json_line_is_reported_not_crashed(self):
        frame = list(decode_frames.parse_frame_log("这不是 JSON\n"))[0]
        self.assertIn("行 JSON", frame.problem)


# -- pcap ------------------------------------------------------------------


def build_pcap(packets, link_type=1):
    """按 libpcap 经典格式手搓一个 pcap。

    手搓而不是录一个真的: 本机 tcpdump 抓包要 root(没有), 而**读** pcap 不要 —— 所以
    下面 ``PcapTest.test_file_is_accepted_by_real_tcpdump`` 会把这里生成的文件丢给
    真 tcpdump 去读一遍, 它认了才算这个格式是真的, 而不是我和我自己商量出来的。
    """
    blob = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, link_type)
    for index, packet in enumerate(packets):
        blob += struct.pack("<IIII", 1787000000 + index, index * 1000,
                            len(packet), len(packet))
        blob += packet
    return blob


def ipv4_checksum(header):
    total = 0
    for offset in range(0, len(header), 2):
        total += struct.unpack(">H", header[offset:offset + 2])[0]
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def tcp_packet(src_ip, src_port, dst_ip, dst_port, seq, payload):
    """一个 Ethernet + IPv4 + TCP 包(校验和算对, 好让真 tcpdump 认账)。"""
    tcp = struct.pack(">HHIIBBHHH", src_port, dst_port, seq, 0,
                      5 << 4, 0x18, 8192, 0, 0)
    tcp += payload
    src = bytes(int(part) for part in src_ip.split("."))
    dst = bytes(int(part) for part in dst_ip.split("."))
    total_length = 20 + len(tcp)
    header = struct.pack(">BBHHHBBH", 0x45, 0, total_length, 0, 0x4000, 64, 6, 0) + src + dst
    header = header[:10] + struct.pack(">H", ipv4_checksum(header)) + header[12:]
    ethernet = b"\x02\x00\x00\x00\x00\x02" + b"\x02\x00\x00\x00\x00\x01" + b"\x08\x00"
    return ethernet + header + tcp


class PcapTest(unittest.TestCase):
    PC = "192.168.1.100"
    CONTROLLER = "192.168.1.10"

    def _capture(self, link_type=1):
        outbound = good_frame(0x4502)
        inbound = good_frame(0x3D03, {"robot": 1, "status": 2})
        packets = [
            tcp_packet(self.PC, 51000, self.CONTROLLER, 6001, 1000, outbound),
            tcp_packet(self.CONTROLLER, 6001, self.PC, 51000, 2000, inbound),
        ]
        return build_pcap(packets, link_type=link_type)

    def test_direction_is_derived_from_the_controller_port(self):
        frames, _ = decode_frames.parse_pcap(self._capture())
        by_direction = dict((frame.direction, frame) for frame in frames)
        self.assertEqual(by_direction["tx"].command, 0x4502)
        self.assertEqual(by_direction["rx"].command, 0x3D03)
        self.assertTrue(all(frame.crc_ok for frame in frames))

    def test_frame_split_across_two_tcp_segments(self):
        """一帧被 TCP 切成两段是常态, 必须先拼流再切帧, 不能按包解。"""
        blob = good_frame(0x9513, {"replyData": {"realPosMCS": [1, 2, 3, 4, 5, 6]}})
        cut = 12
        packets = [
            tcp_packet(self.CONTROLLER, 6001, self.PC, 51000, 5000, blob[:cut]),
            tcp_packet(self.CONTROLLER, 6001, self.PC, 51000, 5000 + cut, blob[cut:]),
        ]
        frames, _ = decode_frames.parse_pcap(build_pcap(packets))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].command, 0x9513)
        self.assertTrue(frames[0].crc_ok)

    def test_retransmitted_segment_is_not_counted_twice(self):
        blob = good_frame(0x2311, {"robot": 1})
        packet = tcp_packet(self.PC, 51000, self.CONTROLLER, 6001, 7000, blob)
        frames, notes = decode_frames.parse_pcap(build_pcap([packet, packet]))
        self.assertEqual([frame.command for frame in frames], [0x2311])
        self.assertTrue(any("重传" in note for note in notes))

    def test_corrupt_crc_survives_the_pcap_path(self):
        packet = tcp_packet(self.PC, 51000, self.CONTROLLER, 6001, 9000,
                            corrupt_crc(good_frame(0x4501, {"robot": 1})))
        frames, _ = decode_frames.parse_pcap(build_pcap([packet]))
        self.assertIs(frames[0].crc_ok, False)

    def test_ethernet_padding_is_not_treated_as_payload(self):
        """短包会被网卡补 0 到 60 字节; 不按 IP 总长裁剪就会多解出一段垃圾。"""
        packet = tcp_packet(self.PC, 51000, self.CONTROLLER, 6001, 11000,
                            good_frame(0x2314, {"robot": 1}))
        frames, _ = decode_frames.parse_pcap(build_pcap([packet + b"\x00" * 12]))
        self.assertEqual([frame.command for frame in frames], [0x2314])
        self.assertIsNone(frames[0].problem)

    def test_linux_cooked_capture_is_supported(self):
        """``tcpdump -i any`` 存出来的是 SLL, 不是 Ethernet —— 现场很容易这么抓。"""
        inner = tcp_packet(self.PC, 51000, self.CONTROLLER, 6001, 13000,
                           good_frame(0x3007, {"robot": 1}))[14:]
        sll = struct.pack(">HHH", 0, 1, 6) + b"\x02\x00\x00\x00\x00\x01\x00\x00"
        sll += struct.pack(">H", 0x0800) + inner
        frames, _ = decode_frames.parse_pcap(build_pcap([sll], link_type=113))
        self.assertEqual([frame.command for frame in frames], [0x3007])

    def test_pcapng_is_refused_loudly(self):
        with self.assertRaises(decode_frames.PcapError) as caught:
            decode_frames.parse_pcap(b"\x0a\x0d\x0d\x0a" + b"\x00" * 32)
        self.assertIn("pcapng", str(caught.exception))

    def test_unknown_magic_is_refused(self):
        with self.assertRaises(decode_frames.PcapError):
            decode_frames.parse_pcap(b"\x00\x01\x02\x03" + b"\x00" * 32)

    def test_snaplen_truncation_is_called_out(self):
        """被 snaplen 砍掉的包一定 CRC 失败, 必须提醒是抓包的锅而不是线上的锅。"""
        packet = tcp_packet(self.PC, 51000, self.CONTROLLER, 6001, 15000, good_frame())
        blob = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 96, 1)
        blob += struct.pack("<IIII", 1787000000, 0, len(packet), len(packet) + 500)
        blob += packet
        _, notes = decode_frames.parse_pcap(blob)
        self.assertTrue(any("snaplen" in note for note in notes))

    @unittest.skipUnless(os.path.exists("/usr/sbin/tcpdump"), "本机没有 tcpdump")
    def test_file_is_accepted_by_real_tcpdump(self):
        """用真 tcpdump 读一遍自造的 pcap: 它认了才说明我们解的是真格式。"""
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as handle:
            handle.write(self._capture())
            path = handle.name
        try:
            result = subprocess.run(
                ["/usr/sbin/tcpdump", "-r", path, "-n"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        finally:
            os.unlink(path)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        text = result.stdout.decode("utf-8", "replace")
        self.assertIn("192.168.1.100.51000 > 192.168.1.10.6001", text)
        self.assertIn("192.168.1.10.6001 > 192.168.1.100.51000", text)


# -- 适配器侧: 开关关着必须什么都不发生 -------------------------------------


class _EchoController:
    """最小假控制器: 收到 0x2311 回一条 0x2003 status=3。"""

    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(2)
        self.port = self.server.getsockname()[1]
        self._stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        try:
            conn, _ = self.server.accept()
        except OSError:
            return
        conn.settimeout(3.0)
        try:
            while not self._stop:
                head = conn.recv(6)
                if len(head) < 6:
                    return
                length = struct.unpack(">H", head[2:4])[0]
                conn.recv(length + 4)
                conn.sendall(build_frame(CMD_SERVO_RESPOND,
                                         {"mode": 0, "robot": 1, "status": 3}))
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self):
        self._stop = True
        try:
            self.server.close()
        except OSError:
            pass


class FrameLogSwitchTest(unittest.TestCase):
    """``NEXBOT_FRAME_LOG`` 的开与关。"""

    def setUp(self):
        self.controller = _EchoController()
        self.addCleanup(self.controller.close)
        self.previous = os.environ.get("NEXBOT_FRAME_LOG")
        self.addCleanup(self._restore_env)
        self.addCleanup(self._close_frame_logs)
        self.directory = tempfile.mkdtemp(prefix="frame-log-")

    def _close_frame_logs(self):
        """帧日志句柄本来就是开一整个进程的(生产里正是要这样), 但测试会开好几个
        临时文件, 不收掉就一路 ResourceWarning。伸手进私有表关掉, 免得为了测试
        在生产代码上开一个没人用的 close() 接口。"""
        for log in list(nexbot_tcp._FrameLog._by_path.values()):
            if log._handle is not None:
                log._handle.close()
                log._handle = None
        nexbot_tcp._FrameLog._by_path.clear()

    def _restore_env(self):
        if self.previous is None:
            os.environ.pop("NEXBOT_FRAME_LOG", None)
        else:
            os.environ["NEXBOT_FRAME_LOG"] = self.previous

    def _endpoint(self):
        return NexBotTcpEndpoint(host="127.0.0.1", port_motion=self.controller.port,
                                 port_state=self.controller.port,
                                 io_timeout_s=1.0, connect_timeout_s=1.0)

    def _exchange(self):
        transport = NexBotTcpTransport(self._endpoint(), self.controller.port)
        try:
            transport.send_frame(CMD_ENABLE, {"robot": 1})
            return transport, transport.read_frame(timeout=2.0)
        finally:
            transport.close()

    def test_no_env_var_produces_no_file_at_all(self):
        """开关关着 = 目录里一个文件都不多出来, transport 上也不挂任何包装。"""
        os.environ.pop("NEXBOT_FRAME_LOG", None)
        transport, (command, data) = self._exchange()
        self.assertEqual(command, CMD_SERVO_RESPOND)
        self.assertEqual(data["status"], 3)
        self.assertEqual(os.listdir(self.directory), [])
        # send_frame/read_frame 必须还是类上那两个原样的方法: 没有实例属性遮蔽,
        # 就等于收发路径上没有多出任何一层。
        self.assertNotIn("send_frame", vars(transport))
        self.assertNotIn("read_frame", vars(transport))
        self.assertIsNone(transport._frame_log)

    def test_empty_env_var_counts_as_off(self):
        os.environ["NEXBOT_FRAME_LOG"] = "   "
        transport, _ = self._exchange()
        self.assertIsNone(transport._frame_log)
        self.assertEqual(os.listdir(self.directory), [])

    def test_env_var_records_both_directions(self):
        path = os.path.join(self.directory, "frames.log")
        os.environ["NEXBOT_FRAME_LOG"] = path
        transport, _ = self._exchange()
        self.assertIn("send_frame", vars(transport))
        with open(path, encoding="utf-8") as handle:
            frames = list(decode_frames.parse_frame_log(handle.read(), source=path))
        self.assertEqual([frame.direction for frame in frames], ["tx", "rx"])
        self.assertEqual([frame.command for frame in frames],
                         [CMD_ENABLE, CMD_SERVO_RESPOND])
        self.assertTrue(all(frame.crc_ok for frame in frames))
        self.assertEqual(frames[0].port, self.controller.port)

    def test_logged_bytes_are_the_bytes_that_went_on_the_wire(self):
        """日志里的 hex 必须逐字节等于 build_frame 的输出, 否则拿它校 CRC 毫无意义。"""
        path = os.path.join(self.directory, "frames.log")
        os.environ["NEXBOT_FRAME_LOG"] = path
        self._exchange()
        with open(path, encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        self.assertEqual(bytes.fromhex(record["hex"]),
                         build_frame(CMD_ENABLE, {"robot": 1}))

    def test_broken_log_path_does_not_break_communication(self):
        """诊断开关坏了(路径不可写)也绝不能拖垮通信 —— 这条比日志本身重要。"""
        os.environ["NEXBOT_FRAME_LOG"] = os.path.join(self.directory, "没有这个目录", "f.log")
        transport, (command, data) = self._exchange()
        self.assertEqual(command, CMD_SERVO_RESPOND)
        self.assertEqual(data["status"], 3)

    def test_log_survives_operator_removing_the_file_mid_run(self):
        """现场清场手法是 ``rm frames.log`` 而 UI 一直开着 —— 之后的帧不许丢。

        丢了的话现象是"文件根本没出现", 排查的人只会以为开关没生效, 而真相是帧
        全写进了那个已经被删掉的 inode 里。诊断工具在最需要它的时候哑掉, 比没有
        这个工具更坏。
        """
        path = os.path.join(self.directory, "frames.log")
        os.environ["NEXBOT_FRAME_LOG"] = path
        # 用同一条 transport 跑两轮: 假控制器只 accept 一次, 而且这样才贴近现场
        # ——— UI 一直开着不重连, 操作员只是在两轮之间把日志文件删了。
        transport = NexBotTcpTransport(self._endpoint(), self.controller.port)
        self.addCleanup(transport.close)
        transport.send_frame(CMD_ENABLE, {"robot": 1})
        transport.read_frame(timeout=2.0)
        os.remove(path)
        transport.send_frame(CMD_ENABLE, {"robot": 1})
        command, _ = transport.read_frame(timeout=2.0)
        self.assertEqual(command, CMD_SERVO_RESPOND)      # 通信本身照常
        self.assertTrue(os.path.exists(path), "rm 之后帧日志没有重新出现")
        with open(path, encoding="utf-8") as handle:
            frames = list(decode_frames.parse_frame_log(handle.read(), source=path))
        self.assertEqual([frame.direction for frame in frames], ["tx", "rx"])

    def test_truncating_the_log_keeps_the_same_handle(self):
        """``> frames.log`` 是同一个 inode, 不该触发重开(重开只针对被 rm 的情况)。"""
        path = os.path.join(self.directory, "frames.log")
        os.environ["NEXBOT_FRAME_LOG"] = path
        transport = NexBotTcpTransport(self._endpoint(), self.controller.port)
        self.addCleanup(transport.close)
        transport.send_frame(CMD_ENABLE, {"robot": 1})
        transport.read_frame(timeout=2.0)
        log = nexbot_tcp._FrameLog._by_path[path]
        handle_before = log._handle
        open(path, "w").close()
        transport.send_frame(CMD_ENABLE, {"robot": 1})
        transport.read_frame(timeout=2.0)
        self.assertIs(log._handle, handle_before)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(len(handle.read().splitlines()), 2)

    def test_logging_does_not_change_read_frame_signature(self):
        """开日志前后 ``read_frame`` 的调用约定必须一模一样。

        包装方法是用实例属性遮蔽类方法实现的, 一旦形参漏一个, 关日志时合法的
        ``read_frame(timeout, sink)`` 一开日志就 TypeError —— 诊断开关反过来改变
        调用约定, 是竞赛日最不该出现的惊喜。
        """
        expected = inspect.signature(NexBotTcpTransport.read_frame)
        expected = expected.replace(
            parameters=[p for name, p in expected.parameters.items() if name != "self"]
        )
        os.environ.pop("NEXBOT_FRAME_LOG", None)
        transport_off = NexBotTcpTransport(self._endpoint(), self.controller.port)
        self.addCleanup(transport_off.close)
        self.assertEqual(inspect.signature(transport_off.read_frame), expected)

        os.environ["NEXBOT_FRAME_LOG"] = os.path.join(self.directory, "frames.log")
        transport_on = NexBotTcpTransport(self._endpoint(), self.controller.port)
        self.addCleanup(transport_on.close)
        self.assertIn("read_frame", vars(transport_on))   # 确认真的被遮蔽了
        self.assertEqual(inspect.signature(transport_on.read_frame), expected)


class CliTest(unittest.TestCase):
    """跑 ``main()`` 的用例都把 stdout 收走: 不然解码结果会糊满测试输出, 真出错时
    反而看不见是哪条挂了。"""

    def _run(self, argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            return decode_frames.main(argv), buffer.getvalue()

    def test_main_returns_1_when_a_frame_fails_crc(self):
        """退出码要能直接判: 有坏帧就非 0, 好让脚本/CI 不用去读输出。"""
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(json.dumps({"t": "x", "dir": "tx", "port": 6001,
                                     "hex": corrupt_crc(good_frame()).hex()}) + "\n")
            path = handle.name
        try:
            code, output = self._run([path, "--no-summary"])
            self.assertEqual(code, 1)
            self.assertIn("CRC 不匹配", output)
        finally:
            os.unlink(path)

    def test_main_returns_0_on_a_clean_log(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(json.dumps({"t": "x", "dir": "tx", "port": 6001,
                                     "hex": good_frame().hex()}) + "\n")
            path = handle.name
        try:
            code, output = self._run([path])
            self.assertEqual(code, 0)
            self.assertIn("所有帧 CRC 都通过", output)
        finally:
            os.unlink(path)

    def test_hex_mode_needs_no_file(self):
        code, output = self._run(["--hex", good_frame().hex()])
        self.assertEqual(code, 0)
        self.assertIn("MOVL", output)

    def test_only_filter_accepts_both_hex_spellings(self):
        """``--only 4502`` 和 ``--only 0x4502`` 必须都能筛到 MOVL。"""
        for spelling in ("4502", "0x4502", "0x4502,2B03"):
            code, output = self._run(["--hex", good_frame().hex(),
                                      "--only", spelling, "--no-summary"])
            self.assertEqual(code, 0, spelling)
            self.assertIn("MOVL", output, spelling)
        code, output = self._run(["--hex", good_frame().hex(),
                                  "--only", "2B03", "--no-summary"])
        self.assertNotIn("MOVL", output)

    def test_only_filter_rejects_garbage(self):
        code, _ = self._run(["--hex", good_frame().hex(), "--only", "写错了"])
        self.assertEqual(code, 2)

    def test_format_is_auto_detected(self):
        self.assertEqual(decode_frames.detect_format(b'{"dir":"tx"}'), "framelog")
        self.assertEqual(decode_frames.detect_format(b"\xd4\xc3\xb2\xa1"), "pcap")
        self.assertEqual(decode_frames.detect_format(b"4e66000b"), "hex")


if __name__ == "__main__":
    unittest.main()
