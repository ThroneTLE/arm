#!/usr/bin/env python3
"""read_controller_logs 的测试。

主体用**合成的 gzip 日志**，不依赖归档大文件：明天带去现场的机器上不一定有那份
5MB 归档，但工具必须照样能验证。归档相关的断言单独放在
:class:`ArchiveTest`，归档不在就 skip。

跑法::

    cd /home/throne/workspaces/arm
    export PYTHONPATH=/home/throne/workspaces/arm:/home/throne/workspaces/arm/ros_ws/src/arm_vision_framework/src
    /home/throne/miniconda3/envs/foundationpose/bin/python -m unittest discover \
        -s competition_pipeline/tests -t . -p 'test_read_controller_logs.py'
"""

from __future__ import annotations

import contextlib
import gzip
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from competition_pipeline.scripts import read_controller_logs as R


# --------------------------------------------------------------------------
# 合成日志的辅助函数
# --------------------------------------------------------------------------

def log_line(level: str, module: str, ts: str, body: str) -> str:
    """按控制器真实的列宽拼一行（模块名补到 11 列，然后是 ``[时间戳] 正文``）。"""
    return "{}/{:<11}[{}] {}".format(level, module, ts, body)


def recv(ts: str, cmd: str, payload: str, port: str = "6001") -> str:
    """控制器收到指令。``port`` 传空字符串 = 示教器本机发的（日志里没有端口前缀）。"""
    return log_line("I", "teachBox", ts, "-->>{}收到指令[{}],${}$".format(port, cmd, payload))


def send(ts: str, cmd: str, payload: str, port: str = "") -> str:
    return log_line("I", "teachBox", ts, "{}发送指令-->>[{}],${}".format(port, cmd, payload))


def write_gz(path: Path, lines) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


class TempLogDir:
    """上下文管理器：造一个临时的 controlLogs 目录。"""

    def __init__(self):
        self.path = Path(tempfile.mkdtemp(prefix="ctrl-logs-"))

    def add(self, index: int, lines, compress: bool = True) -> Path:
        target = self.path / "logInfo.{}".format(index)
        if compress:
            write_gz(target, lines)
        else:
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


# --------------------------------------------------------------------------
# 解析 / 排序
# --------------------------------------------------------------------------

class ParseTest(unittest.TestCase):

    def test_gunzip_and_parse_fields(self):
        with TempLogDir() as d:
            d.add(0, [log_line("E", "ioControl", "2026-08-22 13:34:18.606",
                               "机器人1不在安全位置附近，不予与执行。(pos[1])")])
            records = R.parse_files(R.collect_log_files([str(d.path)]))
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.level, "E")
        self.assertEqual(rec.module, "ioControl")
        self.assertEqual(rec.ts, datetime(2026, 8, 22, 13, 34, 18, 606000))
        self.assertEqual(rec.body, "机器人1不在安全位置附近，不予与执行。(pos[1])")

    def test_plain_text_file_also_works(self):
        """有人手工 gunzip 过的样本也得能读——靠嗅探 gzip 魔数，不看后缀。"""
        with TempLogDir() as d:
            d.add(0, [log_line("I", "teachBox", "2026-08-22 10:00:00.000", "明文")],
                  compress=False)
            records = R.parse_files(R.collect_log_files([str(d.path)]))
        self.assertEqual([r.body for r in records], ["明文"])

    def test_file_index_descending_is_time_ascending(self):
        """logInfo.N 编号越大越旧，所以读入顺序必须是 N 降序。"""
        with TempLogDir() as d:
            d.add(2, [log_line("I", "a", "2026-08-22 09:00:00.000", "最旧")])
            d.add(1, [log_line("I", "a", "2026-08-22 10:00:00.000", "中间")])
            d.add(0, [log_line("I", "a", "2026-08-22 11:00:00.000", "最新")])
            files = R.collect_log_files([str(d.path)])
            self.assertEqual([f.name for f in files],
                             ["logInfo.2", "logInfo.1", "logInfo.0"])
            records = R.parse_files(files)
        self.assertEqual([r.body for r in records], ["最旧", "中间", "最新"])

    def test_timestamp_wins_over_file_index(self):
        """文件编号只是初始顺序，最终必须按行内时间戳排。"""
        with TempLogDir() as d:
            d.add(1, [log_line("I", "a", "2026-08-22 12:00:00.000", "晚")])
            d.add(0, [log_line("I", "a", "2026-08-22 08:00:00.000", "早")])
            records = R.parse_files(R.collect_log_files([str(d.path)]))
        self.assertEqual([r.body for r in records], ["早", "晚"])

    def test_same_millisecond_keeps_causal_order(self):
        """整条事故链挤在同一毫秒里，排序不稳定就会把因果颠倒。"""
        ts = "2026-08-22 13:34:18.746"
        with TempLogDir() as d:
            d.add(0, [
                log_line("I", "ioControl", ts, "远程IO控制：机器人1触发stop。"),
                log_line("I", "robotJob", ts, "机器人1执行JobClear命令"),
                log_line("I", "io", ts, "设置脉冲使能为 0"),
            ])
            records = R.parse_files(R.collect_log_files([str(d.path)]))
        self.assertEqual([r.module for r in records], ["ioControl", "robotJob", "io"])

    def test_continuation_lines_are_merged_not_dropped(self):
        """长 JSON 的收尾 ``$`` 单独一行；丢掉就等于把载荷截断。"""
        with TempLogDir() as d:
            d.add(0, [
                log_line("I", "robotJob", "2026-08-22 13:34:18.605",
                         'FileInstruction[1] ${"RobotPos":{"data":[0.0,'),
                "$",
                log_line("I", "teachBox", "2026-08-22 13:34:18.606", "下一条"),
            ])
            records = R.parse_files(R.collect_log_files([str(d.path)]))
        self.assertEqual(len(records), 2, "续行不应该变成独立记录")
        self.assertTrue(records[0].body.endswith("$"))
        self.assertIn("RobotPos", records[0].body)

    def test_config_dump_block_attaches_to_previous_record(self):
        with TempLogDir() as d:
            d.add(0, [
                log_line("I", "network", "2026-08-22 13:00:00.000", "网络配置:"),
                "auto lo",
                "iface lo inet loopback",
                "address 192.168.1.200",
            ])
            records = R.parse_files(R.collect_log_files([str(d.path)]))
        self.assertEqual(len(records), 1)
        self.assertIn("192.168.1.200", records[0].body)

    def test_debug_level_timestamp_variant(self):
        """``D/`` 级别的时间戳后面多两个空格，固定偏移会漏掉这 592 行。"""
        with TempLogDir() as d:
            d.add(0, ["D/socket     [2026-08-22 15:09:37.286  ] (getErrorInfo:139)接受的错误代码 0"])
            records = R.parse_files(R.collect_log_files([str(d.path)]))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].level, "D")
        self.assertEqual(records[0].ts, datetime(2026, 8, 22, 15, 9, 37, 286000))

    def test_single_file_target(self):
        with TempLogDir() as d:
            path = d.add(0, [log_line("I", "a", "2026-08-22 10:00:00.000", "单文件")])
            records = R.parse_files(R.collect_log_files([str(path)]))
        self.assertEqual([r.body for r in records], ["单文件"])

    def test_missing_path_is_a_clear_error(self):
        with self.assertRaises(SystemExit):
            R.collect_log_files(["/nonexistent/controlLogs-nope"])


# --------------------------------------------------------------------------
# 通信事件抽取
# --------------------------------------------------------------------------

class EventTest(unittest.TestCase):

    def _events(self, lines):
        with TempLogDir() as d:
            d.add(0, lines)
            records = R.parse_files(R.collect_log_files([str(d.path)]))
        events = R.extract_events(records)
        R.annotate_outcomes(records, events)
        return records, events

    def test_recv_send_alarm(self):
        _, events = self._events([
            recv("2026-08-22 13:34:18.604", "0x3007", '{"robot":1}'),
            recv("2026-08-22 13:34:18.605", "0x2301", '{"deadman":1}', port=""),
            send("2026-08-22 13:34:18.606", "0x2303", '{"deadman":1}'),
            send("2026-08-22 13:34:18.607", "0x2b03",
                 '{"code":25530,"data":"指令[0x4502]参数错误","kind":2,"robot":0}'),
        ])
        kinds = [(e.kind, e.command, e.port) for e in events]
        self.assertEqual(kinds, [
            ("recv", 0x3007, "6001"),
            ("recv", 0x2301, None),
            ("send", 0x2303, None),
            ("alarm", 0x2B03, None),
        ])
        self.assertEqual(events[0].payload, '{"robot":1}')
        self.assertIn("指令[0x4502]参数错误", events[3].alarm)

    def test_command_names_come_from_measured_table(self):
        self.assertEqual(R.cmd_label(0x4502), "MOVL 直线插补运动")
        self.assertIn("GO_RESET_POSITION", R.cmd_label(0x3007))
        self.assertIn("未定名", R.cmd_label(0xABCD))

    def test_mode_tracked_from_0x2101(self):
        """模式来自 0x2101，不是 startRobotJobTask 里那个恒为 0 的 mode=。"""
        _, events = self._events([
            recv("2026-08-22 13:20:46.040", "0x2101", '{"mode":1}', port=""),
            recv("2026-08-22 13:27:29.453", "0x3003", '{"robot":1}'),
            recv("2026-08-22 13:31:06.656", "0x2101", '{"mode":0}', port=""),
            recv("2026-08-22 13:31:09.616", "0x3007", '{"robot":1}'),
        ])
        by_cmd = {e.command: e.mode for e in events}
        self.assertEqual(by_cmd[0x3003], 1, "0x3003 应该被记为远程模式")
        self.assertEqual(by_cmd[0x3007], 0, "0x3007 应该被记为示教模式")
        self.assertEqual(R.MODE_NAMES[0], "示教")
        self.assertEqual(R.MODE_NAMES[1], "远程")

    def test_start_robot_job_task_mode_is_not_the_operating_mode(self):
        """回归防线：日志里 startRobotJobTask(mode=0) 恒为 0，绝不能拿它判模式。"""
        _, events = self._events([
            recv("2026-08-22 13:20:46.040", "0x2101", '{"mode":1}', port=""),
            recv("2026-08-22 13:27:29.453", "0x3003", '{"robot":1}'),
            log_line("I", "robotJob", "2026-08-22 13:27:29.454",
                     "执行startRobotJobTask命令(mode=0,num=1,nofile=1,safepos=1,"
                     "teach=-1,call=moveToPos,cb=0)"),
        ])
        motion = [e for e in events if e.command == 0x3003][0]
        self.assertEqual(motion.mode, 1)


# --------------------------------------------------------------------------
# 结局判定
# --------------------------------------------------------------------------

class OutcomeTest(EventTest):

    def test_safety_gate_rejection(self):
        _, events = self._events([
            recv("2026-08-22 13:20:46.040", "0x2101", '{"mode":1}', port=""),
            recv("2026-08-22 13:34:18.604", "0x3007", '{"robot":1}'),
            log_line("I", "robotJob", "2026-08-22 13:34:18.605",
                     "执行startRobotJobTask命令(mode=0,nofile=1,safepos=1,call=moveToPos)"),
            log_line("E", "ioControl", "2026-08-22 13:34:18.606",
                     "机器人1不在安全位置附近，不予与执行。(pos[1])"),
        ])
        motion = [e for e in events if e.command == 0x3007][0]
        self.assertEqual(motion.outcome, "被拒:安全闸门")
        self.assertEqual(motion.mode, 1)

    def test_alarm_rejection_uses_short_label(self):
        _, events = self._events([
            recv("2026-08-22 13:27:20.414", "0x4501", '{"robot":1,"vel":5}'),
            send("2026-08-22 13:27:20.414", "0x2b03",
                 '{"code":25530,"data":"指令[0x4501]参数错误","kind":2,"robot":0}'),
        ])
        motion = [e for e in events if e.command == 0x4501][0]
        self.assertEqual(motion.outcome, "被拒:参数错误")

    def test_background_alarm_is_not_attributed_to_the_command(self):
        """伺服映射错误会周期性刷屏，不能算成"这条指令被拒"，否则统计全是假阳性。"""
        _, events = self._events([
            recv("2026-08-22 13:27:20.414", "0x4502", '{"robot":1}'),
            send("2026-08-22 13:27:20.415", "0x2b03",
                 '{"code":8210,"data":"机器人1伺服映射错误","kind":2,"robot":1}'),
            send("2026-08-22 13:27:20.500", "0x3d03", '{"robot":1,"status":2}'),
        ])
        motion = [e for e in events if e.command == 0x4502][0]
        self.assertEqual(motion.outcome, "已启动")

    def test_motion_started(self):
        _, events = self._events([
            recv("2026-08-22 15:37:43.597", "0x4502", '{"robot":1,"vel":50}'),
            log_line("I", "robotJob", "2026-08-22 15:37:43.598", "机器人1开始单步运行作业文件"),
        ])
        self.assertEqual(events[0].outcome, "已启动")

    def test_accepted_but_never_moved(self):
        """指令被受理却既没被拒也没动——这是最值得单列的一类，混进"成功"会掩盖真问题。"""
        _, events = self._events([
            recv("2026-08-22 15:31:08.723", "0x3007", '{"robot":1}'),
            log_line("I", "robotJob", "2026-08-22 15:31:08.723",
                     "执行startRobotJobTask命令(mode=0,nofile=1,safepos=1,call=moveToPos)"),
            log_line("I", "startup", "2026-08-22 15:31:08.908",
                     "cycle time :curr= 999,min= 990,max=1010,avg= 999"),
        ])
        self.assertEqual(events[0].outcome, "已受理未动")

    def test_no_evidence_at_all(self):
        _, events = self._events([recv("2026-08-22 15:31:08.723", "0x3002", '{"robot":1}')])
        self.assertEqual(events[0].outcome, "未见结果")

    def test_window_truncated_at_next_motion_command(self):
        """连发时，后一条的拒绝不能算到前一条头上。"""
        _, events = self._events([
            recv("2026-08-22 13:37:11.000", "0x3003", '{"robot":1}'),
            log_line("I", "robotJob", "2026-08-22 13:37:11.001", "机器人1开始单步运行作业文件"),
            recv("2026-08-22 13:37:11.500", "0x3003", '{"robot":1}'),
            log_line("E", "ioControl", "2026-08-22 13:37:11.501",
                     "机器人1不在安全位置附近，不予与执行。(pos[1])"),
        ])
        motions = [e for e in events if e.command == 0x3003]
        self.assertEqual([m.outcome for m in motions], ["已启动", "被拒:安全闸门"])

    def test_query_commands_have_no_outcome(self):
        _, events = self._events([recv("2026-08-22 13:00:00.000", "0x2002", '{"robot":1}')])
        self.assertIsNone(events[0].outcome)


# --------------------------------------------------------------------------
# 事故链
# --------------------------------------------------------------------------

class ChainTest(unittest.TestCase):

    #: 现场实测的完整下电链路（证据 A-1/A-2）。
    FULL_CHAIN = [
        recv("2026-08-22 13:34:18.604", "0x3007", '{"robot":1}'),
        log_line("E", "ioControl", "2026-08-22 13:34:18.606",
                 "机器人1不在安全位置附近，不予与执行。(pos[1])"),
        log_line("I", "ioControl", "2026-08-22 13:34:18.746", "远程IO控制：机器人1触发stop。"),
        recv("2026-08-22 13:34:18.746", "0x2401", '{"robot":1,"type":1}', port=""),
        recv("2026-08-22 13:34:18.746", "0x2001", '{"robot":1,"status":0}', port=""),
        log_line("I", "robotJob", "2026-08-22 13:34:18.746", "机器人1执行JobClear命令"),
        log_line("I", "io", "2026-08-22 13:34:18.746", "设置脉冲使能为 0"),
        log_line("I", "servoOn", "2026-08-22 13:34:18.746",
                 "机器人1开始执行Deadan_End命令(Robot_Group_State=1)"),
        log_line("I", "servoOn", "2026-08-22 13:34:18.771", "1111机器人1执行PowerOff命令=0"),
    ]

    def _records(self, lines):
        with TempLogDir() as d:
            d.add(0, lines)
            return R.parse_files(R.collect_log_files([str(d.path)]))

    def test_full_chain_detected_in_causal_order(self):
        hits = R.chain_hits(self._records(self.FULL_CHAIN))
        self.assertEqual([label for label, _ in hits], [
            "安全闸门拒绝", "远程IO触发stop", "停止运行JobClear",
            "脉冲使能=0", "Deadan_End", "PowerOff",
        ])

    def test_power_off_follows_gate_within_200ms(self):
        records = self._records(self.FULL_CHAIN)
        per = {}
        for label, rec in R.chain_hits(records):
            per.setdefault(label, []).append(rec)
        self.assertEqual(
            R._followed_within(per["安全闸门拒绝"], per["PowerOff"], 200), 1)

    def test_deadan_end_start_only_not_the_end_marker(self):
        """``执行Deadan_End命令 结束`` 是收尾，不能和开始那条重复计数。"""
        records = self._records([
            log_line("I", "servoOn", "2026-08-22 13:34:18.746",
                     "机器人1开始执行Deadan_End命令(Robot_Group_State=1)"),
            log_line("I", "servoOn", "2026-08-22 13:34:18.801",
                     "机器人1执行Deadan_End命令 结束(Robot_Group_State=0)"),
        ])
        labels = [label for label, _ in R.chain_hits(records)]
        self.assertEqual(labels, ["Deadan_End"])

    def test_clean_log_has_no_chain(self):
        records = self._records([
            recv("2026-08-22 15:37:43.597", "0x4502", '{"robot":1,"vel":50}'),
            log_line("I", "robotJob", "2026-08-22 15:37:43.598", "机器人1开始单步运行作业文件"),
        ])
        self.assertEqual(R.chain_hits(records), [])

    def test_followed_within_ignores_events_before_the_anchor(self):
        records = self._records([
            log_line("I", "servoOn", "2026-08-22 13:34:18.000", "1111机器人1执行PowerOff命令=0"),
            log_line("E", "ioControl", "2026-08-22 13:34:18.606",
                     "机器人1不在安全位置附近，不予与执行。(pos[1])"),
        ])
        per = {}
        for label, rec in R.chain_hits(records):
            per.setdefault(label, []).append(rec)
        self.assertEqual(R._followed_within(per["安全闸门拒绝"], per["PowerOff"], 500), 0)


# --------------------------------------------------------------------------
# 时间参数与排版
# --------------------------------------------------------------------------

class TimeArgTest(unittest.TestCase):

    REF = datetime(2026, 8, 22, 17, 11, 26)

    def test_time_only_borrows_the_last_records_date(self):
        self.assertEqual(R.parse_time_arg("13:30", self.REF),
                         datetime(2026, 8, 22, 13, 30))
        self.assertEqual(R.parse_time_arg("13:30:05", self.REF),
                         datetime(2026, 8, 22, 13, 30, 5))

    def test_month_day_borrows_year(self):
        self.assertEqual(R.parse_time_arg("08-19 18:29", self.REF),
                         datetime(2026, 8, 19, 18, 29))

    def test_full_datetime(self):
        self.assertEqual(R.parse_time_arg("2026-08-22 13:34:18.606", self.REF),
                         datetime(2026, 8, 22, 13, 34, 18, 606000))

    def test_garbage_time_is_rejected_loudly(self):
        with self.assertRaises(SystemExit):
            R.parse_time_arg("昨天下午", self.REF)

    def test_cmd_args_are_hex(self):
        self.assertEqual(R.parse_cmd_args(["0x3007"]), {0x3007})
        self.assertEqual(R.parse_cmd_args(["0x3007,0x4502"]), {0x3007, 0x4502})
        self.assertEqual(R.parse_cmd_args(["3007"]), {0x3007},
                         "不带 0x 也按十六进制解析，避免静默指向别的指令字")
        with self.assertRaises(SystemExit):
            R.parse_cmd_args(["zzzz"])


class LayoutTest(unittest.TestCase):
    """输出要能直接粘进报告，中文占两列必须算对，否则表格全歪。"""

    def test_width_counts_cjk_as_two(self):
        self.assertEqual(R._width("abc"), 3)
        self.assertEqual(R._width("安全闸门"), 8)

    def test_pad_aligns_mixed_text(self):
        self.assertEqual(R._width(R._pad("被拒:安全闸门", 16)), 16)
        self.assertEqual(R._width(R._pad("已启动", 16)), 16)

    def test_clip_respects_display_width(self):
        clipped = R._clip("去指定点 GO_POSITION(完整 RobotPos)", 14)
        self.assertLessEqual(R._width(clipped), 14)
        self.assertTrue(clipped.endswith("…"))

    def test_clip_leaves_short_text_alone(self):
        self.assertEqual(R._clip("已启动", 14), "已启动")


# --------------------------------------------------------------------------
# CLI 端到端（合成日志）
# --------------------------------------------------------------------------

class CliTest(unittest.TestCase):

    LINES = [
        recv("2026-08-22 13:20:46.040", "0x2101", '{"mode":1}', port=""),
        recv("2026-08-22 13:27:29.453", "0x3003", '{"robot":1}'),
        log_line("E", "ioControl", "2026-08-22 13:27:29.455",
                 "机器人1不在安全位置附近，不予与执行。(pos[1])"),
        log_line("I", "ioControl", "2026-08-22 13:27:29.470", "远程IO控制：机器人1触发stop。"),
        log_line("I", "servoOn", "2026-08-22 13:27:29.500", "1111机器人1执行PowerOff命令=0"),
        recv("2026-08-22 13:31:06.656", "0x2101", '{"mode":0}', port=""),
        recv("2026-08-22 13:31:09.616", "0x3007", '{"robot":1}'),
        log_line("I", "robotJob", "2026-08-22 13:31:09.617", "机器人1开始单步运行作业文件"),
        recv("2026-08-22 13:31:20.000", "0x7266", '{"time":1787429843}', port=""),
        send("2026-08-22 13:31:20.001", "0x7267", '{"time":1787429843}'),
    ]

    def _run(self, *argv):
        with TempLogDir() as d:
            d.add(0, self.LINES)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = R.main([str(d.path)] + list(argv))
        self.assertEqual(code, 0)
        return buf.getvalue()

    @staticmethod
    def _rows(out: str):
        """只取数据行。表尾那句"已隐藏 N 条心跳 0x7266/0x7267"本身就含指令字，
        直接对整段输出做 assertNotIn 会自己骗自己。"""
        return [ln for ln in out.splitlines() if ln.startswith("2026-")]

    def test_default_event_table_hides_heartbeat(self):
        out = self._run()
        rows = self._rows(out)
        self.assertTrue(any("0x3003" in r for r in rows))
        self.assertFalse(any("0x7266" in r for r in rows))
        self.assertIn("已隐藏", out)

    def test_with_heartbeat_shows_them(self):
        rows = self._rows(self._run("--with-heartbeat"))
        self.assertTrue(any("0x7266" in r for r in rows))

    def test_cmd_filter(self):
        rows = self._rows(self._run("--cmd", "0x3007"))
        self.assertTrue(any("0x3007" in r for r in rows))
        self.assertFalse(any("0x3003" in r for r in rows))

    def test_cmd_filter_can_target_heartbeat_explicitly(self):
        rows = self._rows(self._run("--cmd", "0x7266"))
        self.assertTrue(any("0x7266" in r for r in rows))

    def test_time_window(self):
        rows = self._rows(self._run("--since", "13:27", "--until", "13:28"))
        self.assertTrue(any("0x3003" in r for r in rows))
        self.assertFalse(any("0x3007" in r for r in rows))

    def test_mode_survives_the_time_window(self):
        """窗口外的 0x2101 决定了窗口内指令的模式，切窗不能把它切没。"""
        out = self._run("--since", "13:27", "--until", "13:28")
        self.assertIn("远程", out)

    def test_errors_only(self):
        out = self._run("--errors-only")
        rows = self._rows(out)
        self.assertIn("被拒:安全闸门", out)
        self.assertFalse(any("0x7267" in r for r in rows))

    def test_tail(self):
        rows = self._rows(self._run("--tail", "1", "--with-heartbeat"))
        self.assertEqual(len(rows), 1)
        self.assertIn("0x7267", rows[0])

    def test_grep_with_context_shows_raw_lines(self):
        out = self._run("--grep", "不在安全位置", "--context", "1")
        self.assertIn("不在安全位置", out)
        self.assertIn("0x3003", out, "上下文里应该带出触发它的那条指令")

    def test_chain_view(self):
        out = self._run("--chain")
        self.assertIn("不在安全位置", out)
        self.assertIn("PowerOff", out)
        self.assertNotIn("0x7266", out)
        self.assertNotIn("0x3007", out, "--chain 只看事故链，普通指令不该混进来")

    def test_summary_headline(self):
        out = self._run("--summary")
        self.assertIn("一句话结论", out)
        self.assertIn("mode=1 远程", out)
        self.assertIn("该模式下没有任何一条运动指令跑起来", out)
        self.assertIn("回复位点 GO_RESET_POSITION", out, "指令字旁边要注明含义")

    def test_empty_time_window_reports_actual_span(self):
        with TempLogDir() as d:
            d.add(0, self.LINES)
            err = io.StringIO()
            with contextlib.redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = R.main([str(d.path), "--since", "2026-01-01", "--until", "2026-01-02"])
        self.assertEqual(code, 1)
        self.assertIn("2026-08-22", err.getvalue())


# --------------------------------------------------------------------------
# 归档（存在才跑）
# --------------------------------------------------------------------------

ARCHIVE = (Path(__file__).resolve().parents[2]
           / "docs" / "现场备份-20260822" / "controlLogs08-22-22-16")


@unittest.skipUnless(ARCHIVE.is_dir(), "归档 {} 不在，跳过".format(ARCHIVE))
class ArchiveTest(unittest.TestCase):
    """对着真实归档验证——合成日志只能证明代码自洽，证明不了假设成立。"""

    records = None
    events = None

    @classmethod
    def setUpClass(cls):
        # 全量 30 个文件 ~80 万条记录，解析约 2 秒，整个类共用一份。
        files = R.collect_log_files([str(ARCHIVE)])
        cls.records = R.parse_files(files)
        cls.events = R.extract_events(cls.records)
        R.annotate_outcomes(cls.records, cls.events)
        cls.chain = {}
        for label, rec in R.chain_hits(cls.records):
            cls.chain.setdefault(label, []).append(rec)

    def test_finds_the_safety_gate_rejection(self):
        gate = self.chain.get("安全闸门拒绝", [])
        self.assertTrue(gate, "归档里必须能找到'机器人1不在安全位置附近'")
        self.assertTrue(all("不在安全位置附近" in r.body for r in gate))

    def test_at_least_one_rejection_is_followed_by_poweroff_within_200ms(self):
        gate = self.chain["安全闸门拒绝"]
        power = self.chain["PowerOff"]
        paired = R._followed_within(gate, power, 200)
        self.assertGreaterEqual(
            paired, 1,
            "至少有一次'不在安全位置'之后 200ms 内出现 PowerOff（拒绝直接把臂下电）")

    def test_records_are_sorted_by_timestamp(self):
        ts = [r.ts for r in self.records]
        self.assertEqual(ts, sorted(ts))

    def test_every_remote_mode_motion_command_was_rejected(self):
        """核心结论：远程模式下没有任何一条运动指令跑起来。"""
        remote = [e for e in self.events
                  if e.kind == "recv" and e.command in R.MOTION_COMMANDS and e.mode == 1]
        self.assertTrue(remote)
        started = [e for e in remote if e.outcome == "已启动"]
        self.assertEqual(started, [], "远程模式下不该有任何一条运动指令启动")
        self.assertTrue(all((e.outcome or "").startswith("被拒") for e in remote))

    def test_teach_mode_movl_mostly_succeeded(self):
        """对照组：同一批 MOVL 在示教模式下是能跑的，所以问题不在指令本身。"""
        movl = [e for e in self.events
                if e.kind == "recv" and e.command == 0x4502 and e.mode == 0]
        started = [e for e in movl if e.outcome == "已启动"]
        self.assertGreaterEqual(len(started), 40,
                                "示教模式下 0x4502 应有 40 次以上成功启动")

    def test_gate_rejections_are_exclusively_remote_mode(self):
        rejected = [e for e in self.events
                    if e.kind == "recv" and e.outcome == "被拒:安全闸门"]
        self.assertTrue(rejected)
        self.assertEqual({e.mode for e in rejected}, {1},
                         "安全闸门拒绝应该只发生在远程模式(mode=1)")

    def test_summary_runs_end_to_end_on_the_archive(self):
        buf = io.StringIO()
        R.print_summary(R.collect_log_files([str(ARCHIVE)]),
                        self.records, self.events, out=buf)
        out = buf.getvalue()
        self.assertIn("一句话结论", out)
        self.assertIn("安全闸门拒绝", out)
        self.assertIn("MOVL 直线插补运动", out)
        # 模式停留时长必须是"有效时长"，跨月归档直接相减会得到 79 天这种废话。
        self.assertNotIn("days", out)

    # ----------------------------------------------------------------
    # 下面这几条钉的是**具体数字**，不是"大于零"。
    #
    # 起因：对抗性复核时发现原来的宽松断言（>=1、>=40）对四个真实缺陷全部放行，
    # 59 条测试一条没红。明天验收要报出去的是数字，所以数字必须被钉住——
    # 谁改了判据导致 18 变成 15 或 25，这里必须立刻红。
    # 每个数字都用 `zcat | grep` 独立核对过，核对命令写在各自的注释里。
    # ----------------------------------------------------------------

    def test_gate_rejection_count_is_exactly_18_on_the_incident_day(self):
        """事故当天(08-22)安全闸门拒绝 = 18 次，不是现场手记里写的 15 次。

        独立核对::

            zcat logInfo.* | grep -c "不在安全位置"        # 20（含 08-19 两条旧的）
            0x3003×13 + 0x3007×3 + 0x3002×2 = 18
        """
        gate = [e for e in self.events
                if e.kind == "recv" and e.outcome == "被拒:安全闸门"]
        self.assertEqual(len(gate), 18)
        by_cmd = {}
        for e in gate:
            by_cmd[e.command] = by_cmd.get(e.command, 0) + 1
        self.assertEqual(by_cmd, {0x3003: 13, 0x3007: 3, 0x3002: 2})
        # 归档里另有 2 条 08-19 的拒绝，来自 0x2501 启动作业文件而不是运动指令，
        # 所以链上是 20、能归到运动指令头上的是 18。两个数都得对。
        self.assertEqual(len(self.chain["安全闸门拒绝"]), 20)

    def test_exactly_two_rejections_actually_powered_the_arm_down(self):
        """"拒绝即下电"是错的：18 次全走 stop 链，只有 2 次真下电。

        差别在被拒瞬间伺服带不带电。独立核对::

            zcat logInfo.2 | grep -E "不在安全位置|开始执行Deadan_End|执行PowerOff"
            # 只有 13:34:18 和 13:36:15 两条的 Deadan_End 是 Robot_Group_State=1
        """
        gate = self.chain["安全闸门拒绝"]
        self.assertEqual(R._followed_within(gate, self.chain["远程IO触发stop"], 500), 18)
        self.assertEqual(R._followed_within(gate, self.chain["PowerOff"], 500), 2)

    def test_motion_completion_callback_is_not_counted_as_poweroff(self):
        """``333333333...执行PowerOff命令=1`` 是运动走完的回调，不是下电。

        归档里 396 条，条条紧跟 notice_completed_one_sequence。算进事故链就会得到
        "PowerOff 912 次"，其中 396 次其实是成功——把成功念成失电比不报还糟。

        独立核对::

            zcat logInfo.* | grep -c "执行PowerOff命令"                 # 912
            zcat logInfo.* | grep -c "333333333机器人1执行PowerOff"     # 396
        """
        power = self.chain["PowerOff"]
        self.assertEqual(len(power), 516)
        self.assertFalse([r for r in power if "333333333" in r.body])
        # 516 行 = 258 次真下电（每次打 1111/222 两条），
        # 而 Deadan_End 里真正带电的正好也是 246+12=258 次，对得上才说明没数错。
        armed = [r for r in self.records
                 if "开始执行Deadan_End命令" in r.body
                 and ("Robot_Group_State=1" in r.body or "Robot_Group_State=4" in r.body)]
        self.assertEqual(len(power) // 2, len(armed))

    def test_jog_computation_never_counts_as_motion_started(self):
        """点动的"计算任务"不能把一条没被受理的运动指令说成"已启动"。

        归档里点动 4053 次、点动计算日志 454 条，只要判据里留着裸的"计算任务"，
        撞上只是时间问题；报"已启动"而机械臂没动是这个工具能犯的最坏的错。
        """
        with TempLogDir() as d:
            d.add(0, [
                recv("2026-08-22 10:00:00.000", "0x2101", '{"mode":1}', port=""),
                recv("2026-08-22 10:00:01.000", "0x4502",
                     '{"robot":1,"vel":50,"coord":3,"pos":[1,2,3,0,0,0,0]}'),
                # 控制器对这条 MOVL 什么都没做；0.3s 后操作员点动了一下
                log_line("I", "robotCacl", "2026-08-22 10:00:01.300",
                         "机器人点动计算任务结果为0x20000000"),
            ])
            records = R.parse_files(R.collect_log_files([str(d.path)]))
        events = R.extract_events(records)
        R.annotate_outcomes(records, events)
        movl = [e for e in events if e.command == 0x4502]
        self.assertEqual([e.outcome for e in movl], ["未见结果"])

    def test_safe_pos_deviation_is_computed_not_just_printed(self):
        """闸门坏掉的证据必须由工具算出来，不能靠人比两行浮点数。

        13:34:18 那次逐轴偏差都在 1e-5 量级（最大 3.86e-05 度），比一个编码器脉冲
        还小，机器人就停在复位点上却被判"不在安全位置"。

        独立核对::

            zcat logInfo.2 | grep "13:34:18" | grep 0x2f16
            currentPos[0]=-1.254185607880  safePos[0]=-1.25420  -> 1.4e-05
        """
        margins = R.safe_pos_margins(self.events)
        # 0x2F16 与闸门拒绝在归档里严格 1:1，所以每次拒绝都能给出数字。
        self.assertEqual(len(margins), 20)
        by_ts = {R._fmt_ts(ev.ts): worst for ev, worst, _ in margins}
        self.assertLess(by_ts["2026-08-22 13:34:18.606"], 1e-4)
        self.assertLess(by_ts["2026-08-22 13:36:15.504"], 1e-4)
        # 反面同样重要：另外 13 次拒绝时机械臂真的偏了几十度，那些是**合理**拒绝。
        # 把 20 次一律当成"闸门坏了"的证据，验收时一查就穿帮。
        self.assertGreater(by_ts["2026-08-22 13:27:29.455"], 10.0)
        self.assertGreater(by_ts["2026-08-22 13:44:59.369"], 10.0)

    def test_bad_grep_regex_fails_with_a_readable_message(self):
        """--grep 编译不过要说人话，不能甩 re.error 栈回溯。

        日志里满屏 ``[0x3007]``，粘半截 ``[`` 是最容易发生的事；
        验收现场看到 traceback 会以为工具崩了，实际只是正则写法问题。
        """
        with self.assertRaises(SystemExit) as ctx:
            R._grep_re("[")
        self.assertIn("字符集", str(ctx.exception))

    def test_errors_only_event_view_does_not_pretend_to_show_the_chain(self):
        """事故链那些行一条都不是通信事件，事件表里按链过滤恒为空集。

        原实现留着这个筛选分支，会让人以为"没打出来 = 没发生"。
        """
        chain_ids = {id(rec) for _, rec in R.chain_hits(self.records)}
        event_ids = {id(e.record) for e in self.events}
        self.assertTrue(chain_ids)
        self.assertEqual(chain_ids & event_ids, set())


if __name__ == "__main__":
    unittest.main()
