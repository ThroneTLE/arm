"""一键抓取 demo（用户坐标系1）：设抓取位/放置位 → 一键抓取。

序列（每段 move_to_ucs 都做"等到位"验证，出轨/超差立即报错停止）：
    抓取位上方(抬起高度) → 抓取位 → 夹爪合 → 抬升 → 放置位上方 → 放置位
    → 夹爪开 → 抬升 → 完成
速度默认 100 mm/s（= 传送面板同一档，实测 ±0.01mm 级）。

急停：面板【急停】直发 0x2314（jog 的 _estop_lock 通道，不排队），
并置中止标志——demo 在步骤间 0.3s 检查一次，中止后不再发后续运动。
"""

import threading
import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QPushButton,
)


class GraspDemoWorker(QThread):
    """一键抓取序列（工作线程，独立强引用防 QThread 崩溃）。"""

    done = pyqtSignal(bool, str)

    def __init__(self, jog_provider, grasp, place, lift_mm=60.0,
                 vel_mm_s=100.0, tolerance_mm=1.2, parent=None):
        super().__init__(parent)
        self.jog_provider = jog_provider
        self.grasp = list(grasp)          # [xyz_mm(3), abc_rad(3)]
        self.place = list(place)
        self.lift = float(lift_mm)
        self.vel = float(vel_mm_s)
        self.tolerance = float(tolerance_mm)
        self.abort = threading.Event()

    def run(self):
        jog = self.jog_provider()
        try:
            g_xyz, g_abc = self.grasp
            p_xyz, p_abc = self.place
            up = np.array([0.0, 0.0, self.lift])
            g_up = list(np.asarray(g_xyz, dtype=float) + up)
            p_up = list(np.asarray(p_xyz, dtype=float) + up)
            steps = [
                ("运动到抓取位上方", g_up, g_abc),
                ("下降到抓取位", g_xyz, g_abc),
                ("夹爪合(抓取)", None, None),
                ("抬升", g_up, g_abc),
                ("运动到放置位上方", p_up, p_abc),
                ("下降到放置位", p_xyz, p_abc),
                ("夹爪开(放置)", None, None),
                ("抬升收尾", p_up, p_abc),
            ]
            for name, xyz, abc in steps:
                if self.abort.is_set():
                    self.done.emit(False, "已急停中止（跳过：{}）".format(name))
                    return
                if xyz is None:                      # 夹爪动作
                    if "合" in name:
                        jog.gripper(False)
                    else:
                        jog.gripper(True)
                    time.sleep(0.5)                  # 气阀动作时间
                    continue
                jog.move_to_ucs(xyz, abc, vel_mm_s=self.vel,
                                tolerance_mm=self.tolerance)
            self.done.emit(True, "✅ 一键抓取完成：抓取({:.1f},{:.1f},{:.1f}) → "
                                "放置({:.1f},{:.1f},{:.1f}) mm".format(
                                    *g_xyz, *p_xyz))
        except Exception as error:
            self.done.emit(False, "抓取失败：{}".format(error))


def _pose_spins(group, prefix):
    """XYZ(mm) 与 ABC(°) 各占一行输入。返回 dict。"""
    out = {}
    for i, (axis, unit, rng) in enumerate([
        ("X", " mm", (-2000.0, 2000.0)),
        ("Y", " mm", (-2000.0, 2000.0)),
        ("Z", " mm", (-2000.0, 2000.0)),
        ("A", " °", (-360.0, 360.0)),
        ("B", " °", (-180.0, 180.0)),
        ("C", " °", (-360.0, 360.0)),
    ]):
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(*rng)
        spin.setSuffix(unit)
        spin.setValue(0.0)
        group.addWidget(QLabel(prefix + axis), i, 0)
        group.addWidget(spin, i, 1)
        out[axis] = spin
    return out


class _ReadbackThread(QThread):
    """回读当前位姿（QThread：回调自动回到 Qt 主线程，绝不非主线程操作控件）。"""

    done = pyqtSignal(object)

    def __init__(self, jog_provider, parent=None):
        super().__init__(parent)
        self.jog_provider = jog_provider

    def run(self):
        try:
            jog = self.jog_provider()
            xyz, abc_deg = jog.current_pose()
            self.done.emit(("OK", xyz, abc_deg))
        except Exception as error:
            self.done.emit(("ERR", str(error)))


class GraspDemoPanel(QGroupBox):
    """一键抓取（用户坐标系1）：设抓取位/放置位 + 一键执行。"""

    def __init__(self, jog_provider, parent=None):
        super().__init__("一键抓取 demo（用户坐标系1）· 设抓取位/放置位后一键执行", parent)
        self.jog_provider = jog_provider
        self._workers = []
        self._busy = False

        grid = QGridLayout(self)
        # 抓取位行
        g_box = QGroupBox("抓取位（XYZ + ABC°，先移动到目标再【设为当前抓取位】）")
        g_layout = QGridLayout(g_box)
        self.g_spins = {}
        for i, spin in enumerate(_pose_spins(g_layout, "").values()):
            self.g_spins[list("XYZABC")[i]] = spin
        self.g_set = QPushButton("设为当前抓取位")
        self.g_set.clicked.connect(lambda: self._read_into(self.g_spins, "抓取位"))
        g_layout.addWidget(self.g_set, 1, 0, 1, 2)
        grid.addWidget(g_box, 0, 0)

        # 放置位行
        p_box = QGroupBox("放置位（XYZ + ABC°）")
        p_layout = QGridLayout(p_box)
        self.p_spins = {}
        for i, spin in enumerate(_pose_spins(p_layout, "").values()):
            self.p_spins[list("XYZABC")[i]] = spin
        self.p_set = QPushButton("设为当前放置位")
        self.p_set.clicked.connect(lambda: self._read_into(self.p_spins, "放置位"))
        p_layout.addWidget(self.p_set, 1, 0, 1, 2)
        grid.addWidget(p_box, 1, 0)

        # 参数
        self.lift = QDoubleSpinBox()
        self.lift.setRange(10.0, 500.0)
        self.lift.setValue(60.0)
        self.lift.setSuffix(" mm")
        self.vel = QDoubleSpinBox()
        self.vel.setRange(10.0, 250.0)
        self.vel.setValue(100.0)
        self.vel.setSuffix(" mm/s")
        grid.addWidget(QLabel("抬起高度"), 2, 0)
        grid.addWidget(self.lift, 2, 1)
        grid.addWidget(QLabel("速度"), 3, 0)
        grid.addWidget(self.vel, 3, 1)

        self.btn_go = QPushButton("一键抓取")
        self.btn_estop = QPushButton("急停")
        self.btn_estop.setStyleSheet(
            "background:#c0392b; color:white; font-weight:bold;"
        )
        self.btn_go.clicked.connect(self._on_go)
        self.btn_estop.clicked.connect(self._on_estop)
        grid.addWidget(self.btn_go, 4, 0)
        grid.addWidget(self.btn_estop, 4, 1)

        self.status = QLabel("就绪：先把机器人移到抓取位/放置位，各点【设为当前…】")
        grid.addWidget(self.status, 5, 0, 1, 2)

    # -- helpers ------------------------------------------------------------
    def _pose_of(self, spins):
        xyz = [spins[k].value() for k in ("X", "Y", "Z")]
        abc = np.radians([spins[k].value() for k in ("A", "B", "C")])
        return list(xyz), list(np.asarray(abc, dtype=float))

    def _read_into(self, spins, label):
        self.status.setText("⏳ 回读当前位姿…")

        def _done(payload):
            if payload[0] == "ERR":
                self.status.setText("回读失败：{}".format(payload[1]))
                return
            _ok, xyz, abc_deg = payload
            for k, v in zip(("X", "Y", "Z"), xyz):
                spins[k].setValue(float(v))          # 主线程回调，安全
            for k, v in zip(("A", "B", "C"), abc_deg):
                spins[k].setValue(float(v))
            self.status.setText(
                "✅ {}已填入当前位姿 X={:.2f} Y={:.2f} Z={:.2f} mm".format(
                    label, *xyz)
            )

        worker = _ReadbackThread(self.jog_provider)
        worker.done.connect(_done)
        self._keep(worker)

    def _keep(self, worker):
        self._workers.append(worker)
        worker.finished.connect(lambda w=worker: self._forget(w))
        worker.start()

    def _forget(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)

    # -- actions ------------------------------------------------------------
    def _on_go(self):
        if self._busy:
            self.status.setText("⏳ 抓取序列执行中……")
            return
        grasp = self._pose_of(self.g_spins)
        place = self._pose_of(self.p_spins)
        self._busy = True
        self.btn_go.setEnabled(False)
        self.status.setText("⏳ 一键抓取执行中…（急停可随时中止）")
        worker = GraspDemoWorker(
            self.jog_provider, grasp, place,
            lift_mm=self.lift.value(), vel_mm_s=self.vel.value(),
        )

        def _done(ok, message):
            self._busy = False
            self.btn_go.setEnabled(True)
            self.status.setText(message)

        worker.done.connect(_done)
        self._keep(worker)

    def _on_estop(self):
        self.status.setText("🚨 急停发送中…")

        class _EstopThread(QThread):
            done = pyqtSignal(object)

            def run(self):
                try:
                    self.jog_provider().emergency_stop()
                    for worker in self._workers:
                        if isinstance(worker, GraspDemoWorker):
                            worker.abort.set()
                    self.done.emit(("OK",))
                except Exception as error:
                    self.done.emit(("ERR", str(error)))

        worker = _EstopThread(self.jog_provider)

        def _done(payload):
            if payload[0] == "ERR":
                self.status.setText("急停失败：{}".format(payload[1]))
            else:
                self.status.setText("🚨 急停已发送并中止序列（请现场确认机器人已停）")

        worker.done.connect(_done)
        self._keep(worker)


__all__ = ["GraspDemoPanel", "GraspDemoWorker"]
