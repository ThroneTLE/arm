#!/usr/bin/env python3
"""控制器日志考古工具：把示教器导出的一堆 gzip 日志变成一屏能看懂的通信时间线。

**为什么需要这个**：控制器是黑盒，但它自己用中文记录了收到的每一条指令、发出的
每一条回复、以及**它拒绝执行的理由**。现场出现"指令发出去了但机器人不动"这类
没见过的问题时，抓包只能看到我们发了什么，只有控制器日志能回答"它为什么不干"。
以前这个信息要靠 ``zcat | grep`` 人工考古两小时，现在跑一条命令。

**只读**：本脚本不导入任何运行时模块、不写任何文件、不连任何网络。删掉它对正常
程序零影响。

导出方式：示教器 -> 系统 -> 日志导出，得到 ``controlLogs<MM-DD-HH-MM>/`` 目录，
里面是 ``logInfo.0 .. logInfo.N``，**每个都是 gzip 压缩的文本**（没有 .gz 后缀，
所以 ``cat`` 会看到乱码，要用 ``zcat``）。**编号越大越旧**，logInfo.0 是最新的。

用法::

    P=/home/throne/miniconda3/envs/foundationpose/bin/python
    L=docs/现场备份-20260822/controlLogs08-22-22-16

    # 先看全局：指令次数、成功/被拒、模式分布、事故链
    $P -m competition_pipeline.scripts.read_controller_logs $L --summary

    # 只看最后 40 条通信事件（现场最常用：刚发了指令没反应，看它到底收到没）
    $P -m competition_pipeline.scripts.read_controller_logs $L --tail 40

    # 只看某个指令字（含义会打在旁边）
    $P -m competition_pipeline.scripts.read_controller_logs $L --cmd 0x3007

    # 只看错误/警告和事故链（下电、急停、安全闸门拒绝）
    $P -m competition_pipeline.scripts.read_controller_logs $L --errors-only

    # 卡时间窗（只给时分秒时按日志里最后一条记录的日期补全）
    $P -m competition_pipeline.scripts.read_controller_logs $L --since 13:30 --until 13:50

    # 出问题那一瞬间的原始上下文（前后各 12 行原始日志）
    $P -m competition_pipeline.scripts.read_controller_logs $L --grep 不在安全位置 --context 12

时钟提示（现场实测）：控制器时钟 ≈ PC 时钟 − 10 分钟；示教器时钟 ≈ 控制器 + 5h05m。
本工具打印的全部是**控制器时钟**，和日志原文一致，不做任何换算——换算过的时间戳
没法直接和原始日志对照，反而更难查。
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 指令字含义表
#
# 来源是**我们自己在这台 C1102 上的实测**（2026-08-22 现场日志 + 抓包），不是手册。
# 手册和这台机器的固件对不上的地方不止一处（例如 0x2314 手册叫"急停"，实测是直接
# 下电），所以这里只写实际观察到的行为，看不懂的指令字宁可留空也不猜。
# --------------------------------------------------------------------------
COMMAND_NAMES: Dict[int, str] = {
    0x2001: "伺服上下电请求(status 0=下电/1=上电)",
    0x2002: "伺服状态查询",
    0x2003: "伺服状态回复(status 3=已使能)",
    0x2101: "设置操作模式(mode 0=示教/1=远程/2=运行)",
    0x2102: "查询操作模式",
    0x2103: "操作模式回复",
    0x2201: "设置坐标系(coord 3=用户坐标)",
    0x2202: "查询坐标系",
    0x2203: "坐标系回复",
    0x2301: "deadman 按键(1=按下/0=松开)",
    0x2303: "deadman 状态回复",
    0x2311: "上使能",
    0x2314: "急停(本机实测=直接下电, 不是受控减速停)",
    0x2401: "停止运行",
    # 0x??01=设置 / 0x??02=查询 / 0x??03=回复 这个三元组在 22xx/26xx/36xx/3dxx
    # 上都成立（日志里成对出现），所以 0x2602 按同一规律定名。
    0x2601: "设置运行速度(%)",
    0x2602: "查询运行速度",
    0x2603: "运行速度回复",
    0x2901: "单轴点动开始",
    0x2902: "单轴点动停止",
    0x2A02: "查询当前位姿",
    0x2A03: "当前位姿回复(直角坐标)",
    0x2B03: "报警推送(真实文本在 data 键)",
    0x2B05: "提示推送(kind=0, 非报警)",
    0x2F07: "查询关节位置",
    0x2F08: "关节位置回复",
    0x2F16: "安全位置比对回复(currentPos vs safePos)",
    0x3002: "回原点 GO_HOME",
    0x3003: "去指定点 GO_POSITION(完整 RobotPos)",
    0x3007: "回复位点 GO_RESET_POSITION",
    0x3601: "DOUT 置位(夹爪走这条)",
    0x3602: "DOUT 查询",
    0x3603: "DOUT 状态回复",
    0x3D02: "程序运行状态查询",
    0x3D03: "程序运行状态推送(status 2=运行中/0=停止)",
    0x4501: "MOVJ 关节插补运动",
    0x4502: "MOVL 直线插补运动",
    0x4503: "MOVC 圆弧插补运动",
    0x4504: "MOVS 样条插补运动",
    0x7266: "心跳(示教器->控制器)",
    0x7267: "心跳回复",
    0x9512: "状态批量查询(7000 端口)",
    0x9513: "状态批量查询回复",
}

#: 心跳。占了全部通信事件的三分之二（归档里 11 万条），默认不进事件表——否则
#: ``--tail 20`` 永远只能看到心跳，把真正想看的那条运动指令挤出屏幕。
HEARTBEAT_COMMANDS = frozenset({0x7266, 0x7267})

#: 会真的让机器人动起来的指令字。只有这些需要判定"成功还是被拒"，
#: 查询类指令没有"被拒"这个概念，混在一起统计只会稀释信号。
MOTION_COMMANDS = frozenset({0x3002, 0x3003, 0x3007, 0x4501, 0x4502, 0x4503, 0x4504})

#: 操作模式。实测来源：``-->>收到指令[0x2101],${"mode":N}``，控制器同一毫秒回
#: 一条 ``设置操作模式为N%``。**注意**：``startRobotJobTask(mode=0,...)`` 里那个
#: mode 跟操作模式无关（全量日志里 99/99 都是 0），别拿它判模式。
MODE_NAMES = {0: "示教", 1: "远程", 2: "运行"}

# --------------------------------------------------------------------------
# 事故链
#
# 现场实测的固定链路（零例外）：安全闸门拒绝 -> 远程IO触发stop -> 0x2401 ->
# 0x2001 status=0 -> JobClear -> 设置脉冲使能为 0 -> Deadan_End -> PowerOff。
# 把这条链单独标出来，是因为它在原始日志里被上千条 EcMaster/debug 噪声淹没，
# 而它恰恰是"我发了指令，机械臂反而失电"的完整解释。
# --------------------------------------------------------------------------
CHAIN_RULES: Tuple[Tuple[str, str], ...] = (
    ("安全闸门拒绝", "不在安全位置附近"),
    ("远程IO触发stop", "触发stop"),
    ("停止运行JobClear", "执行JobClear命令"),
    ("脉冲使能=0", "设置脉冲使能为 0"),
    ("Deadan_End", "开始执行Deadan_End命令"),
    ("PowerOff", "执行PowerOff命令"),
)

#: 长得像事故链、其实不是的行。
#:
#: ``333333333机器人1执行PowerOff命令=1`` 字面上是 PowerOff，实际是**一段运动正常
#: 走完**的回调：归档里 396 条，396 条都紧跟 ``notice_completed_one_sequence``，零例外。
#: 不排掉的话事故链会报 "PowerOff 912 次"，其中 396 次恰恰是成功而不是事故——
#: 在验收现场把成功念成失电，比不报还糟。排掉后是 516 行 = 258 次真下电，
#: 和 ``Deadan_End(Robot_Group_State=1)`` 246 次 + ``(=4)`` 12 次严丝合缝。
CHAIN_EXCLUDE: Tuple[str, ...] = ("333333333机器人",)

#: 判定"运动真的启动了"的证据。0x3d03 status:2 是最通用的一条（示教/远程都推），
#: 后两条是作业文件解释器自己的日志，只在跑得更深时才出现。
#:
#: ``启动MOV`` 必须写这么长：控制器给**点动**也打 ``机器人点动计算任务结果为0x...``
#: （归档里 454 条），只匹配"计算任务"会把一条根本没被受理的 MOVL 判成"已启动"——
#: 而点动是操作员最常做的动作（归档里 4053 次 0x2901），撞上只是时间问题。
#: 报"已启动"却没动，是这个工具能犯的最坏的错，所以这里宁可漏判也不误判。
SUCCESS_MARKERS: Tuple[str, ...] = (
    "开始单步运行作业文件",
    "启动MOV",              # 机器人1启动MOVJ/MOVL计算任务
    "作业文件开始计算指令",
)

#: 这些报警码出现在运动指令之后 = 这条指令被吃掉了，不是背景噪声。
#: 背景噪声（伺服映射错误 8210、与从站通信失败 8211 等）会周期性刷屏，
#: 不能一律算作"这条指令被拒"，否则统计全是假阳性。
#: 值是**短标签**，只用来填结果列；报警原文在同一时刻的 0x2B03 行上一字不差地列着。
REFUSAL_ALARM_CODES = {
    4098: "作业运行中",
    4609: "伺服未连接",
    4611: "伺服数不符",
    4613: "伺服未OP",
    25530: "参数错误",
}

#: ``I/teachBox   [2026-08-22 13:34:18.604] 正文``。模块名补空格到固定列宽，但
#: 用 ``\s*`` 而不是 ``\s+``：万一某个模块名正好顶满列宽就没有空格了，
#: 那种行不该被当成续行丢掉。
_HEADER_RE = re.compile(r"^([A-Z])/(\S+?)\s*\[")
#: ``-->>6001收到指令[0x3007],${...}$``  端口前缀只在 6001 实时通道上出现，
#: 示教器自己发的没有前缀——这个差别是区分"我们发的"和"人在示教器上按的"的唯一依据。
_RECV_RE = re.compile(r"-->>(\d{4})?收到指令\[0x([0-9a-fA-F]{4})\],\$(.*)$", re.S)
_SEND_RE = re.compile(r"(\d{4})?发送指令-->>\[0x([0-9a-fA-F]{4})\],\$(.*)$", re.S)


class LogRecord:
    """一条日志记录。跨行的 JSON 载荷已经并进 ``body``。"""

    __slots__ = ("ts", "level", "module", "body", "source", "lineno")

    def __init__(self, ts, level, module, body, source, lineno):
        self.ts: datetime = ts
        self.level: str = level
        self.module: str = module
        self.body: str = body
        self.source: str = source
        self.lineno: int = lineno

    def raw(self) -> str:
        """还原成原始日志的样子，方便直接贴进报告和原文对照。"""
        return "{}/{:<10} [{}] {}".format(
            self.level, self.module, _fmt_ts(self.ts), self.body
        )


class CommEvent:
    """一次通信事件：收到的指令 / 发出的回复 / 报警。"""

    __slots__ = ("record", "kind", "command", "port", "payload", "mode", "outcome", "alarm")

    def __init__(self, record, kind, command, port, payload):
        self.record: LogRecord = record
        self.kind: str = kind          # recv / send / alarm
        self.command: int = command
        self.port: Optional[str] = port
        self.payload: str = payload
        self.mode: Optional[int] = None
        self.outcome: Optional[str] = None
        self.alarm: Optional[str] = None

    @property
    def ts(self) -> datetime:
        return self.record.ts


# --------------------------------------------------------------------------
# 读取与解析
# --------------------------------------------------------------------------

def _is_gzip(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def open_log(path: Path):
    """打开一个日志文件，自动识别是否 gzip。

    嗅探魔数而不是看后缀：示教器导出的文件叫 ``logInfo.3``（无后缀）却是 gzip，
    而有人手工 ``gunzip`` 过的样本又是明文，两种都得能读。
    """
    if _is_gzip(path):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def _file_sort_key(path: Path) -> Tuple[int, str]:
    """logInfo.N 的 N 越大越旧，所以按 N 降序 = 时间升序（作为排序的初值）。"""
    suffix = path.name.rsplit(".", 1)[-1]
    return (-int(suffix), path.name) if suffix.isdigit() else (1, path.name)


def collect_log_files(targets: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for target in targets:
        p = Path(target).expanduser()
        if p.is_dir():
            files.extend(sorted(p.glob("logInfo*"), key=_file_sort_key))
        elif p.exists():
            files.append(p)
        else:
            raise SystemExit("找不到路径: {}".format(p))
    if not files:
        raise SystemExit("目标里没有 logInfo* 文件: {}".format(", ".join(targets)))
    return files


def _parse_ts(text: str, cache: Dict[str, datetime]) -> Optional[datetime]:
    """把 ``2026-08-22 13:34:18.604`` 变成 datetime。

    不用 strptime：全量日志有 110 万行，strptime 单独就要十几秒，而现场排查
    最不能忍的就是等。按秒缓存 + 手工切片后整个解析只要 2 秒。
    """
    if len(text) < 19:
        return None
    key = text[:19]
    base = cache.get(key)
    if base is None:
        try:
            base = datetime(
                int(text[0:4]), int(text[5:7]), int(text[8:10]),
                int(text[11:13]), int(text[14:16]), int(text[17:19]),
            )
        except ValueError:
            return None
        cache[key] = base
    millis = text[20:23]
    if len(millis) == 3 and millis.isdigit():
        return base + timedelta(milliseconds=int(millis))
    return base


def parse_files(files: Sequence[Path]) -> List[LogRecord]:
    """读所有文件 -> 结构化记录 -> 按时间戳排序。

    两件容易踩的事：

    1. 无头部的行不是垃圾。控制器把长 JSON 的收尾 ``$`` 单独打一行（全量日志里
       有 31 万行是光秃秃的 ``$``），网络配置之类还会整段 dump。这些行属于**上一条**
       记录，丢掉就等于把 JSON 载荷截断了。
    2. ``D/`` 级别的时间戳后面多两个空格（``...286  ]``），固定偏移会漏掉它们，
       所以按 ``]`` 定位再 strip。
    """
    records: List[LogRecord] = []
    cache: Dict[str, datetime] = {}
    for path in files:
        name = path.name
        current: Optional[LogRecord] = None
        extra: List[str] = []
        with open_log(path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                head = _HEADER_RE.match(line)
                ts = None
                if head is not None:
                    lb = line.index("[", head.end() - 1)
                    rb = line.find("]", lb)
                    if rb > 0:
                        ts = _parse_ts(line[lb + 1:rb].strip(), cache)
                if ts is None:
                    if current is not None and line:
                        extra.append(line)
                    continue
                if current is not None and extra:
                    current.body += "\n".join([""] + extra)
                    extra = []
                current = LogRecord(
                    ts, head.group(1), head.group(2), line[rb + 1:].lstrip(" "), name, lineno
                )
                records.append(current)
        if current is not None and extra:
            current.body += "\n".join([""] + extra)

    # 文件顺序已经是"旧->新"，Python 的 sort 稳定，所以同一毫秒内的多条记录
    # 会保持原始先后 —— 事故链的因果顺序（拒绝 -> stop -> 下电）全在同一毫秒里，
    # 用不稳定排序会把因果打乱。
    records.sort(key=lambda r: r.ts)
    return records


# --------------------------------------------------------------------------
# 通信事件抽取
# --------------------------------------------------------------------------

def _clean_payload(raw: str) -> str:
    """去掉 ``${...}$`` 的外壳，并把跨行拼回来的换行压平成一行。"""
    return " ".join(raw.strip().rstrip("$").strip().split())


def _payload_json(payload: str) -> Optional[dict]:
    """尽力把载荷解析成 dict；日志会截断长 JSON，解析失败是正常的，不报错。"""
    if not payload.startswith("{"):
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_events(records: Sequence[LogRecord]) -> List[CommEvent]:
    """从记录流里挑出通信事件，并沿途跟踪当前操作模式。

    模式必须**沿时间轴推进**地跟踪：一条 0x3003 是在示教还是远程模式下被发出的，
    只能由它之前最近的一条 0x2101 决定。这正是"为什么同一条指令有时成有时败"的答案。
    """
    events: List[CommEvent] = []
    mode: Optional[int] = None
    for rec in records:
        body = rec.body
        if "收到指令[" in body:
            m = _RECV_RE.search(body)
            if m:
                payload = _clean_payload(m.group(3))
                ev = CommEvent(rec, "recv", int(m.group(2), 16), m.group(1), payload)
                if ev.command == 0x2101:
                    obj = _payload_json(payload)
                    if obj is not None and isinstance(obj.get("mode"), int):
                        mode = obj["mode"]
                ev.mode = mode
                events.append(ev)
                continue
        if "发送指令-->>[" in body:
            m = _SEND_RE.search(body)
            if m:
                command = int(m.group(2), 16)
                payload = _clean_payload(m.group(3))
                kind = "alarm" if command == 0x2B03 else "send"
                ev = CommEvent(rec, kind, command, m.group(1), payload)
                ev.mode = mode
                if kind == "alarm":
                    obj = _payload_json(payload)
                    if obj is not None:
                        ev.alarm = "{}({})".format(obj.get("data", "?"), obj.get("code", "?"))
                    else:
                        ev.alarm = payload[:60]
                events.append(ev)
    return events


def annotate_outcomes(
    records: Sequence[LogRecord],
    events: Sequence[CommEvent],
    window_sec: float = 2.0,
) -> None:
    """给每条运动指令标注结局：被拒(安全闸门) / 被拒(报警) / 已启动 / 已受理 / 未见结果。

    做法是往后扫一小段时间窗。窗口在**下一条运动指令**处提前截断，否则连发 MOVL
    时会把后一条的结果算到前一条头上。

    "已受理"和"已启动"要分开：现场有 0x3007 走到了 ``startRobotJobTask`` 却既没被
    拒绝也没推 0x3d03 status:2 —— 指令被吃了但机器人没动。把它归进"成功"会掩盖
    真问题，所以单列。
    """
    motion_ts = [e.ts for e in events if e.kind == "recv" and e.command in MOTION_COMMANDS]
    # 记录流按时间有序，用游标线性推进即可，不必对每条指令二分整表。
    n = len(records)
    idx_by_ts = 0
    motion_pos = 0
    for ev in events:
        if ev.kind != "recv" or ev.command not in MOTION_COMMANDS:
            continue
        while motion_pos < len(motion_ts) and motion_ts[motion_pos] <= ev.ts:
            motion_pos += 1
        deadline = ev.ts + timedelta(seconds=window_sec)
        if motion_pos < len(motion_ts) and motion_ts[motion_pos] < deadline:
            deadline = motion_ts[motion_pos]
        while idx_by_ts < n and records[idx_by_ts].ts < ev.ts:
            idx_by_ts += 1

        rejected = None
        alarm_hit = None
        started = False
        accepted = False
        hexname = "指令[0x{:04x}]".format(ev.command)
        j = idx_by_ts
        while j < n and records[j].ts <= deadline:
            body = records[j].body
            if "不在安全位置附近" in body:
                rejected = "被拒:安全闸门"
                break
            if "发送指令-->>[0x2b03]" in body.lower():
                obj = _payload_json(_clean_payload(body.split(",$", 1)[-1]))
                if obj is not None:
                    code = obj.get("code")
                    data = str(obj.get("data", ""))
                    if hexname in data or code in REFUSAL_ALARM_CODES:
                        alarm_hit = alarm_hit or "被拒:{}".format(
                            REFUSAL_ALARM_CODES.get(code) or data or code)
            if not started and any(mark in body for mark in SUCCESS_MARKERS):
                started = True
            if '[0x3d03],${"robot"' in body and '"status":2' in body:
                started = True
            if "执行startRobotJobTask命令" in body:
                accepted = True
            j += 1

        if rejected:
            ev.outcome = rejected
        elif alarm_hit:
            ev.outcome = alarm_hit
        elif started:
            ev.outcome = "已启动"
        elif accepted:
            ev.outcome = "已受理未动"
        else:
            ev.outcome = "未见结果"


def mode_dwell(
    records: Sequence[LogRecord], idle_gap: timedelta = timedelta(seconds=60)
) -> Dict[Optional[int], timedelta]:
    """按操作模式累计**有日志活动**的时长。

    不能简单用"下一次模式切换 − 本次模式切换"：一份归档能跨四个月，中间是整夜整周
    没开机的空档，那样算出来会得到"示教模式停留 79 天"这种没法写进报告的数字。
    所以按相邻两条日志记录的间隔累加，间隔超过 ``idle_gap`` 的当作停机不计。
    """
    dwell: Dict[Optional[int], timedelta] = {}
    mode: Optional[int] = None
    prev: Optional[datetime] = None
    for rec in records:
        if prev is not None:
            delta = rec.ts - prev
            if delta <= idle_gap:
                dwell[mode] = dwell.get(mode, timedelta()) + delta
        prev = rec.ts
        # 先记账再换模式：这一段时间属于**切换之前**的那个模式。
        if "[0x2101]" in rec.body and "收到指令" in rec.body:
            m = _RECV_RE.search(rec.body)
            if m and int(m.group(2), 16) == 0x2101:
                obj = _payload_json(_clean_payload(m.group(3)))
                if obj is not None and isinstance(obj.get("mode"), int):
                    mode = obj["mode"]
    return dwell


def safe_pos_margins(
    events: Sequence[CommEvent],
) -> List[Tuple[CommEvent, float, List[float]]]:
    """算出每次安全闸门拒绝时，机器人到底离复位点差多远（逐轴，单位=度）。

    **这是整个根因的落脚点**，所以必须由工具自己算，不能靠人盯着日志里两行浮点数
    比大小：控制器打的 ``currentPos`` 有 12 位小数、``safePos`` 只有 5 位，肉眼看过去
    满屏都是"不一样的数"，很容易顺着控制器的说法认下"确实不在安全位置"。真算一遍
    才看得到偏差只有 1e-5 度——比任何编码器的一个脉冲都小，也就是说**机器人就停在
    复位点上却被判为不在**，判据本身是坏的(global.json 里 posReset.deviation == null)。

    0x2F16 和闸门拒绝在归档里是严格 1:1（各 20 条），所以每次拒绝都能给出数字。
    """
    out: List[Tuple[CommEvent, float, List[float]]] = []
    for ev in events:
        if ev.command != 0x2F16:
            continue
        obj = _payload_json(ev.payload)
        if obj is None:
            continue
        cur, safe = obj.get("currentPos"), obj.get("safePos")
        if not isinstance(cur, list) or not isinstance(safe, list) or len(cur) != len(safe):
            continue
        try:
            delta = [abs(float(a) - float(b)) for a, b in zip(cur, safe)]
        except (TypeError, ValueError):
            continue
        if delta:
            out.append((ev, max(delta), delta))
    return out


def chain_hits(records: Sequence[LogRecord]) -> List[Tuple[str, LogRecord]]:
    """把事故链标记从记录流里挑出来（按时间顺序）。"""
    hits: List[Tuple[str, LogRecord]] = []
    for rec in records:
        if any(x in rec.body for x in CHAIN_EXCLUDE):
            continue
        for label, needle in CHAIN_RULES:
            if needle in rec.body:
                hits.append((label, rec))
                break
    return hits


# --------------------------------------------------------------------------
# 过滤
# --------------------------------------------------------------------------

def parse_time_arg(text: str, reference: Optional[datetime]) -> datetime:
    """接受 ``HH:MM`` / ``HH:MM:SS`` / ``MM-DD HH:MM:SS`` / 完整日期时间。

    只给时分秒时按**日志中最后一条记录的日期**补全：现场关心的永远是刚发生的那一段，
    补成最早的日期只会得到空结果。
    """
    text = text.strip()
    ref = reference or datetime.now()
    for fmt, borrow in (
        ("%Y-%m-%d %H:%M:%S.%f", None),
        ("%Y-%m-%d %H:%M:%S", None),
        ("%Y-%m-%d %H:%M", None),
        ("%Y-%m-%d", None),
        ("%m-%d %H:%M:%S", "year"),
        ("%m-%d %H:%M", "year"),
        ("%H:%M:%S.%f", "date"),
        ("%H:%M:%S", "date"),
        ("%H:%M", "date"),
    ):
        try:
            got = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if borrow == "year":
            return got.replace(year=ref.year)
        if borrow == "date":
            return got.replace(year=ref.year, month=ref.month, day=ref.day)
        return got
    raise SystemExit("无法解析时间 {!r}（可用: 13:30 / 13:30:05 / 08-22 13:30 / 2026-08-22 13:30:05）".format(text))


def _grep_re(pattern: str):
    """编译 --grep 的正则，编译不过就说人话。

    这不是吹毛求疵：日志正文里到处是 ``[0x3007]``，最自然的动作就是把它整段粘进
    --grep。而 ``[0x3007]`` 在正则里是**字符集**，会匹配任何含 0/x/3/7 的行
    （实测该时间窗内命中 53 行而不是 1 行），粘 ``[`` 半截则直接抛 re.error 栈回溯。
    一个静默错 53 倍、一个看起来像工具崩了，验收现场两种都输，所以这里把话讲明。
    """
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise SystemExit(
            "--grep 的正则编译不过: {}\n"
            "  提示: 日志里的 [ ] 在正则里是字符集。要找字面量请转义或去掉方括号，\n"
            "        例如把 --grep '[0x3007]' 写成 --grep '0x3007' 或 --grep '\\[0x3007\\]'"
            .format(exc))


def parse_cmd_args(values: Iterable[str]) -> Optional[set]:
    out = set()
    for value in values:
        for part in re.split(r"[,\s]+", value.strip()):
            if not part:
                continue
            try:
                # 一律按十六进制解析：日志里指令字只以 0xXXXX 出现，
                # 允许十进制只会让 "--cmd 3007" 静默指向 0x0BBF。
                out.add(int(part, 16))
            except ValueError:
                raise SystemExit("指令字要写成十六进制, 例如 0x3007（收到 {!r}）".format(part))
    return out or None


# --------------------------------------------------------------------------
# 输出（CJK 宽度对齐，方便直接粘进报告）
# --------------------------------------------------------------------------

def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def _clip(text: str, width: int) -> str:
    """按**显示宽度**截断（不是字符数）——中文占两列，按字符截会撑破表格。"""
    if _width(text) <= width:
        return text
    out = []
    used = 0
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + w > width - 1:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"


def _fmt_ts(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S.") + "{:03d}".format(ts.microsecond // 1000)


def cmd_label(command: int) -> str:
    return COMMAND_NAMES.get(command, "(未定名, 见示教器手册)")


_ARROWS = {"recv": "收到", "send": "发出", "alarm": "报警"}


def print_events(events: Sequence[CommEvent], payload_width: int, out=None) -> None:
    # 默认参数不能直接写 sys.stdout: 默认值在 import 时就绑定了，
    # 之后测试里的 redirect_stdout 换掉 sys.stdout 也拦不住，输出会漏到终端。
    out = out or sys.stdout
    cols = (("时间戳", 23), ("端口", 4), ("向", 4), ("指令字", 6),
            ("含义", 30), ("模式", 4), ("结果", 16))
    header = "  ".join(_pad(name, w) for name, w in cols) + "  载荷"
    widest = max([_width(header)] + [
        _width(header) - 4 + _width(_clip(e.alarm if e.kind == "alarm" else e.payload,
                                          payload_width))
        for e in events
    ]) if events else _width(header)
    out.write(header + "\n")
    out.write("-" * widest + "\n")
    for ev in events:
        row = [
            _pad(_fmt_ts(ev.ts), 23),
            _pad(ev.port or "-", 4),
            _pad(_ARROWS.get(ev.kind, ev.kind), 4),
            _pad("0x{:04x}".format(ev.command), 6),
            _pad(_clip(cmd_label(ev.command), 30), 30),
            _pad(MODE_NAMES.get(ev.mode, "?") if ev.mode is not None else "-", 4),
            # 报警行的结果列留空：报警原文马上就在载荷列里，重复一遍只会把表撑宽。
            _pad(_clip(ev.outcome or "", 16), 16),
        ]
        payload = ev.alarm if ev.kind == "alarm" else ev.payload
        out.write("  ".join(row) + "  " + _clip(payload, payload_width) + "\n")


def print_records(records: Sequence[LogRecord], out=None) -> None:
    out = out or sys.stdout
    for rec in records:
        for i, part in enumerate(rec.body.split("\n")):
            if i == 0:
                out.write("{}  {}/{}  {}\n".format(
                    _fmt_ts(rec.ts), rec.level, _pad(rec.module, 10), part))
            else:
                out.write(" " * 25 + "  |  " + part + "\n")


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------

def _followed_within(
    anchors: Sequence[LogRecord], followers: Sequence[LogRecord], millis: int
) -> int:
    """有多少条 ``anchors`` 在 ``millis`` 毫秒内跟着至少一条 ``followers``。

    两个列表都按时间有序，所以用双游标线性扫，不要写成 O(n·m) —— 归档里 PowerOff
    有近千条，平方复杂度在现场是要命的等待。
    """
    count = 0
    j = 0
    limit = timedelta(milliseconds=millis)
    for rec in anchors:
        while j < len(followers) and followers[j].ts < rec.ts:
            j += 1
        if j < len(followers) and followers[j].ts - rec.ts <= limit:
            count += 1
    return count


def print_summary(
    files: Sequence[Path],
    records: Sequence[LogRecord],
    events: Sequence[CommEvent],
    out=None,
) -> None:
    w = (out or sys.stdout).write
    w("=" * 100 + "\n")
    w("控制器日志摘要（时间戳全部是控制器时钟，未做任何换算）\n")
    w("=" * 100 + "\n\n")

    w("[数据源]\n")
    w("  文件      : {} 个（{}）\n".format(len(files), files[0].parent))
    w("  日志记录  : {} 条\n".format(len(records)))
    if records:
        w("  覆盖区间  : {}  ->  {}\n".format(_fmt_ts(records[0].ts), _fmt_ts(records[-1].ts)))
    levels: Dict[str, int] = {}
    for rec in records:
        levels[rec.level] = levels.get(rec.level, 0) + 1
    w("  级别分布  : {}\n\n".format(
        "  ".join("{}={}".format(k, levels[k]) for k in sorted(levels))))

    recv = [e for e in events if e.kind == "recv"]
    send = [e for e in events if e.kind == "send"]
    alarms = [e for e in events if e.kind == "alarm"]
    w("[通信事件]\n")
    w("  收到指令 {} 条 / 发出回复 {} 条 / 报警推送 {} 条\n\n".format(
        len(recv), len(send), len(alarms)))

    # --- 一句话结论 ---
    # 放在最前面，是因为现场看这份输出的人只有 30 秒：他要的就是"哪个模式下的
    # 运动指令被吃掉了"，别的都是佐证。
    w("[一句话结论] 运动指令(0x3002/0x3003/0x3007/0x450x) 按操作模式分\n")
    per_mode: Dict[Optional[int], Dict[str, int]] = {}
    for ev in recv:
        if ev.command not in MOTION_COMMANDS:
            continue
        bucket = per_mode.setdefault(ev.mode, {"总数": 0, "被拒": 0, "已启动": 0, "其它": 0})
        bucket["总数"] += 1
        outcome = ev.outcome or ""
        if outcome.startswith("被拒"):
            bucket["被拒"] += 1
        elif outcome == "已启动":
            bucket["已启动"] += 1
        else:
            bucket["其它"] += 1
    if not per_mode:
        w("  （范围内没有运动指令）\n")
    for mode in sorted(per_mode, key=lambda m: (m is None, m)):
        b = per_mode[mode]
        flag = ""
        if b["总数"] and b["已启动"] == 0:
            flag = "   <<< 该模式下没有任何一条运动指令跑起来"
        w("  {}  运动指令 {} 条  ->  被拒 {} / 已启动 {} / 其它 {}{}\n".format(
            _pad("mode={} {}".format(mode, MODE_NAMES.get(mode, "?")), 14),
            _pad(str(b["总数"]), 4), _pad(str(b["被拒"]), 4),
            _pad(str(b["已启动"]), 4), b["其它"], flag))
    w("\n")

    # --- 收到的指令字排行 ---
    w("[收到的指令字]（端口 6001 = 我们 PC 直连的实时通道；空 = 示教器本机发的）\n")
    tally: Dict[Tuple[int, Optional[str]], int] = {}
    for ev in recv:
        key = (ev.command, ev.port)
        tally[key] = tally.get(key, 0) + 1
    w("  {}  {}  {}  {}\n".format(_pad("指令字", 8), _pad("端口", 6), _pad("次数", 8), "含义"))
    for (command, port), count in sorted(tally.items(), key=lambda kv: -kv[1])[:25]:
        w("  {}  {}  {}  {}\n".format(
            _pad("0x{:04x}".format(command), 8),
            _pad(port or "-", 6),
            _pad(str(count), 8),
            cmd_label(command),
        ))
    w("\n")

    # --- 运动指令的结局，按模式拆开 ---
    w("[运动指令结局 × 操作模式]  <- 这张表是判断'指令为什么不执行'的核心\n")
    grid: Dict[Tuple[int, Optional[int], str], int] = {}
    for ev in recv:
        if ev.command in MOTION_COMMANDS:
            key = (ev.command, ev.mode, ev.outcome or "未判定")
            grid[key] = grid.get(key, 0) + 1
    if not grid:
        w("  （本次筛选范围内没有运动指令）\n")
    else:
        w("  {}  {}  {}  {}  {}\n".format(
            _pad("指令字", 8), _pad("模式", 6), _pad("结局", 16), _pad("次数", 6), "含义"))
        for (command, mode, outcome), count in sorted(
            grid.items(), key=lambda kv: (kv[0][0], kv[0][1] if kv[0][1] is not None else -1)
        ):
            w("  {}  {}  {}  {}  {}\n".format(
                _pad("0x{:04x}".format(command), 8),
                _pad(MODE_NAMES.get(mode, "?") if mode is not None else "-", 6),
                _pad(_clip(outcome, 16), 16),
                _pad(str(count), 6),
                cmd_label(command),
            ))
    w("\n")

    # --- 模式分布 ---
    w("[操作模式]（来源: 0x2101 ${\"mode\":N}，控制器同刻回 '设置操作模式为N%'）\n")
    switches = [e for e in recv if e.command == 0x2101]
    dwell = mode_dwell(records)
    if not switches:
        w("  （范围内没有模式切换记录）\n")
    for mode in sorted(dwell, key=lambda m: (m is None, m)):
        count = sum(1 for e in switches if e.mode == mode)
        label = ("(首次 0x2101 之前, 模式未知)" if mode is None
                 else "mode={} {}".format(mode, MODE_NAMES.get(mode, "?")))
        w("  {}  切换 {} 次  有效时长 {}（不含停机空档）\n".format(
            _pad(label, 26), _pad(str(count), 4), str(dwell[mode]).split(".")[0]))
    w("\n")

    # --- 事故链 ---
    w("[事故链]（安全闸门拒绝 -> 触发stop -> JobClear -> 脉冲使能=0 -> Deadan_End -> PowerOff）\n")
    hits = chain_hits(records)
    per_label: Dict[str, List[LogRecord]] = {}
    for label, rec in hits:
        per_label.setdefault(label, []).append(rec)
    if not hits:
        w("  （范围内没有事故链事件 —— 这是好消息）\n")
    for label, _ in CHAIN_RULES:
        recs = per_label.get(label)
        if not recs:
            continue
        note = ("  ※ 同一次下电打两条(1111/222), 即真下电 {} 次".format(len(recs) // 2)
                if label == "PowerOff" else "")
        w("  {}  {} 次   首 {}   末 {}{}\n".format(
            _pad(label, 18), _pad(str(len(recs)), 5),
            _fmt_ts(recs[0].ts), _fmt_ts(recs[-1].ts), note))
    gate = per_label.get("安全闸门拒绝", [])
    if gate:
        stopped = _followed_within(gate, per_label.get("远程IO触发stop", []), 500)
        powered = _followed_within(gate, per_label.get("PowerOff", []), 500)
        w("  -> 安全闸门拒绝 {} 次：{} 次 500ms 内触发 stop 链，{} 次 500ms 内真的 PowerOff。\n".format(
            len(gate), stopped, powered))
        # 差额不是漏检。stop 链每次都跑，但 Deadan_End 只在伺服当时带电
        # (Robot_Group_State=1) 时才走到 PowerOff；本来就没上电的那几次自然没有下电记录。
        w("     差额来自被拒时伺服本就未上电(Deadan_End 的 Robot_Group_State=0)，不是漏检。\n")
    w("\n")

    # --- 安全位置偏差 ---
    # 放在事故链后面：链回答"发生了什么"，这张表回答"凭什么判它不在安全位置"。
    margins = safe_pos_margins(events)
    if margins:
        w("[安全位置偏差 0x2F16]（逐轴 |currentPos - safePos| 的最大值，单位=度）\n")
        best = min(margins, key=lambda m: m[1])
        for ev, worst, delta in margins:
            w("  {}  最大偏差 {:.2e} 度   逐轴 {}\n".format(
                _fmt_ts(ev.ts), worst,
                " ".join("{:.1e}".format(d) for d in delta)))
        w("  -> 最接近的一次是 {}，最大偏差 {:.2e} 度，仍然被拒。\n".format(
            _fmt_ts(best[0].ts), best[1]))
        w("     这个量级远小于一个编码器脉冲，即机器人就停在复位点上而闸门说它不在，\n")
        w("     所以问题不是位置不对，是判据本身坏了(global.json RemoteIO[0].posReset.deviation == null)。\n\n")

    # --- 报警 ---
    w("[报警 0x2B03]（真实文本在 JSON 的 data 键）\n")
    atally: Dict[str, int] = {}
    for ev in alarms:
        atally[ev.alarm or ev.payload[:60]] = atally.get(ev.alarm or ev.payload[:60], 0) + 1
    if not atally:
        w("  （范围内没有报警）\n")
    for text, count in sorted(atally.items(), key=lambda kv: -kv[1])[:15]:
        w("  {}  {}\n".format(_pad(str(count), 6), text))
    w("\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="read_controller_logs",
        description="控制器 gzip 日志 -> 结构化通信时间线（只读，不改任何运行时行为）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  %(prog)s docs/现场备份-20260822/controlLogs08-22-22-16 --summary\n"
               "  %(prog)s <目录> --cmd 0x3007 --since 13:30 --until 13:50\n"
               "  %(prog)s <目录> --errors-only --tail 40\n"
               "  %(prog)s <目录> --grep 不在安全位置 --context 12\n",
    )
    p.add_argument("paths", nargs="+", help="controlLogs* 目录，或单个 logInfo.N 文件")
    p.add_argument("--since", help="起始时间 (13:30 / 08-22 13:30 / 2026-08-22 13:30:05)")
    p.add_argument("--until", help="结束时间，同上")
    p.add_argument("--cmd", action="append", default=[],
                   help="只看这些指令字，十六进制，可重复或逗号分隔: --cmd 0x3007,0x4502")
    p.add_argument("--errors-only", action="store_true",
                   help="只看 E/W 级别记录、报警和被拒的指令"
                        "（事故链是控制器内部日志、不是通信事件，要看请用 --chain）")
    p.add_argument("--tail", type=int, metavar="N", help="只看最后 N 条")
    p.add_argument("--summary", action="store_true", help="打印统计摘要")
    p.add_argument("--grep", metavar="RE",
                   help="正则过滤日志正文（作用在原始记录上）。"
                        "注意是正则不是字面量：[0x3007] 会被当成字符集，直接写 0x3007")
    p.add_argument("--context", type=int, default=0, metavar="N",
                   help="配合 --grep/--errors-only，额外打印命中行前后各 N 行原始日志")
    p.add_argument("--raw", action="store_true", help="打印原始记录而不是通信事件表")
    p.add_argument("--chain", action="store_true", help="只打印事故链事件（原始记录）")
    p.add_argument("--window", type=float, default=2.0, metavar="SEC",
                   help="判定运动指令结局时往后看的时间窗，默认 2.0 秒")
    p.add_argument("--payload-width", type=int, default=90, metavar="N",
                   help="载荷列宽度，默认 90")
    p.add_argument("--with-heartbeat", action="store_true",
                   help="事件表里也显示心跳 0x7266/0x7267（默认隐藏，它占了 2/3 的事件）")
    return p


def _slice_by_time(items, since, until):
    if since is not None:
        items = [x for x in items if x.ts >= since]
    if until is not None:
        items = [x for x in items if x.ts <= until]
    return items


def _with_context(records: Sequence[LogRecord], picked: Sequence[LogRecord],
                  context: int) -> List[LogRecord]:
    """把命中行前后各 N 条原始记录补回来。定位通信问题时上下文比命中行本身值钱。"""
    if context <= 0:
        return list(picked)
    index = {id(rec): i for i, rec in enumerate(records)}
    keep = set()
    for rec in picked:
        i = index.get(id(rec))
        if i is None:
            continue
        keep.update(range(max(0, i - context), min(len(records), i + context + 1)))
    return [records[i] for i in sorted(keep)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    files = collect_log_files(args.paths)
    records = parse_files(files)
    if not records:
        print("没有解析出任何记录（文件是空的？）", file=sys.stderr)
        return 1

    reference = records[-1].ts
    since = parse_time_arg(args.since, reference) if args.since else None
    until = parse_time_arg(args.until, reference) if args.until else None

    # 先在**全量**记录上抽事件、判结局，再切时间窗。顺序反过来的话，
    # "13:34 那条指令是在远程模式下发的"就查不出来了——决定模式的那条 0x2101
    # 在窗口之外（13:31），窗口里只剩一条孤零零的指令，模式列全是 "-"。
    # 同理，运动指令的结局证据可能落在窗口右边界之外。
    events = extract_events(records)
    annotate_outcomes(records, events, window_sec=args.window)

    span = (records[0].ts, records[-1].ts)
    records = _slice_by_time(records, since, until)
    events = _slice_by_time(events, since, until)
    if not records:
        print("时间窗内没有记录：日志实际覆盖 {} -> {}".format(
            _fmt_ts(span[0]), _fmt_ts(span[1])), file=sys.stderr)
        return 1

    wanted = parse_cmd_args(args.cmd) if args.cmd else None

    # --- 原始记录视图（--raw / --grep / --chain）---
    if args.raw or args.grep or args.chain:
        picked = records
        if args.chain:
            picked = [rec for _, rec in chain_hits(picked)]
        if args.grep:
            picked = [rec for rec in picked if _grep_re(args.grep).search(rec.body)]
        if args.errors_only:
            chain_ids = {id(rec) for _, rec in chain_hits(records)}
            picked = [r for r in picked if r.level in ("E", "W") or id(r) in chain_ids]
        if args.tail:
            picked = picked[-args.tail:]
        print_records(_with_context(records, picked, args.context))
        if args.summary:
            print_summary(files, records, events)
        return 0

    shown = events
    hidden_beats = 0
    # 显式 --cmd 0x7266 时不该被默认隐藏规则打脸，所以只在没点名心跳时才过滤。
    if not args.with_heartbeat and not (wanted and wanted & HEARTBEAT_COMMANDS):
        before = len(shown)
        shown = [e for e in shown if e.command not in HEARTBEAT_COMMANDS]
        hidden_beats = before - len(shown)
    if wanted:
        shown = [e for e in shown if e.command in wanted]
    if args.errors_only:
        # 这里**不能**再按事故链筛。事故链那些行(触发stop/JobClear/脉冲使能=0/
        # Deadan_End/PowerOff)全是控制器内部日志，一条都不是通信事件，所以在事件表
        # 里按 id 匹配恒为空集(实测 1752 条链记录 ∩ 33.8 万条事件记录 = 0)。
        # 留着只会让人以为"没打出来 = 没发生"。要看链请用 --chain / --raw。
        shown = [e for e in shown
                 if e.kind == "alarm"
                 or e.record.level in ("E", "W")
                 or (e.outcome or "").startswith("被拒")]
    if args.tail:
        shown = shown[-args.tail:]

    if args.summary:
        print_summary(files, records, events)
        # --summary 单独用时不再刷事件表；除非同时给了筛选条件，那说明是想看细节。
        if not (wanted or args.errors_only or args.tail):
            return 0

    print_events(shown, args.payload_width)
    if hidden_beats:
        print("（已隐藏 {} 条心跳 0x7266/0x7267，要看请加 --with-heartbeat）".format(hidden_beats))
    if args.errors_only and args.context:
        print("\n--- 命中行原始上下文 ---")
        print_records(_with_context(records, [e.record for e in shown], args.context))
    return 0


if __name__ == "__main__":
    sys.exit(main())
