"""目标点传送面板（用户坐标系1）。

挂接（复用 UI 的共享持久连接，6001/7000 单客户端端口不新增连接）：
    from competition_pipeline.teleport_panel import TeleportPanel
    panel = TeleportPanel(jog_provider=lambda: self._get_robot_jog())
    layout.addWidget(panel)

面板内容：
  - X/Y/Z （QDoubleSpinBox，mm，±2000）
  - A/B/C （QDoubleSpinBox，默认角度制；勾选"弧度"切换）
  - 【传送】把输入点经 User1Mover 发到控制器并回读核对（工作线程，不卡 UI）
  - 【回读当前】填当前用户1系位姿
  - 【急停】0x2314 直发（不受"传送忙"限制）
  - 状态栏：距离/到达偏差（±0.01mm 级实测）/错误

链路（全部现场实测）：0x2311 上电 → 0x4501/0x4502(+acc=dec=10) coord=3 →
位姿变化判停 → 0x2A02 回读核对。
"""

import threading

import numpy as np
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QPushButton,
)

from competition_pipeline.geometry import inexbot_abc_from_transform


class TeleportWorker(QThread):
    """一次传送动作（复用共享持久 jog 连接）。"""

    done = pyqtSignal(bool, str)

    def __init__(self, jog_provider, xyz_mm, abc_rad, tolerance_mm=1.0,
                 vel_mm_s=100.0, parent=None):
        super().__init__(parent)
        self.jog_provider = jog_provider
        self.xyz = list(xyz_mm)
        self.abc = list(abc_rad)
        self.tolerance = float(tolerance_mm)
        self.vel = float(vel_mm_s)

    def run(self):
        try:
            jog = self.jog_provider()
            deviation = jog.move_to_ucs(
                self.xyz, self.abc, vel_mm_s=self.vel,
                tolerance_mm=self.tolerance,
            )
            self.done.emit(
                True,
                "✅ 传送完成：X={:.2f} Y={:.2f} Z={:.2f} mm → 偏差 {:.3f} mm".format(
                    *self.xyz, deviation
                ),
            )
        except Exception as error:
            self.done.emit(False, "传送失败：{}".format(error))


class ReadbackWorker(QThread):
    """回读当前位姿并回填输入框。"""

    done = pyqtSignal(bool, object, str)

    def __init__(self, jog_provider, parent=None):
        super().__init__(parent)
        self.jog_provider = jog_provider

    def run(self):
        try:
            jog = self.jog_provider()
            xyz, abc_deg = jog.current_pose()
            self.done.emit(True, (xyz, np.radians(abc_deg)), "")
        except Exception as error:
            self.done.emit(False, None, str(error))


class TeleportPanel(QGroupBox):
    """目标点传送（用户坐标系1）：输入 XYZ(mm) + ABC → 传送 → 到达偏差。"""

    def __init__(self, jog_provider, parent=None):
        super().__init__("目标点传送（用户坐标系1）· 输入点直接发到控制器并核对到位", parent)
        self.jog_provider = jog_provider
        self._workers = []          # 强引用防 QThread 被 GC（QThread 崩溃教训）
        self._busy = False
        self._requires_readback = True   # 未确认过当前位姿前禁止传送
        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.setInterval(35000)
        self._watchdog.timeout.connect(self._on_teleport_timeout)
        # 启动后自动回读填充（避免直接传默认 (0,0,0) 的坑）
        QTimer.singleShot(300, self._on_readback)

        grid = QGridLayout(self)
        self._spins = {}
        for i, (name, unit, rng) in enumerate([
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
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(spin, i, 1)
            self._spins[name] = spin
        self.rad_unit = QCheckBox("A/B/C 用弧度")
        grid.addWidget(self.rad_unit, 6, 0, 1, 2)

        self.btn_teleport = QPushButton("传送")
        self.btn_read = QPushButton("回读当前")
        self.btn_estop = QPushButton("急停")
        self.btn_estop.setStyleSheet(
            "background:#c0392b; color:white; font-weight:bold;"
        )
        self.btn_teleport.clicked.connect(self._on_teleport)
        self.btn_read.clicked.connect(self._on_readback)
        self.btn_estop.clicked.connect(self._on_estop)
        grid.addWidget(self.btn_teleport, 7, 0)
        grid.addWidget(self.btn_read, 7, 1)
        grid.addWidget(self.btn_estop, 7, 2)

        self.status = QLabel("就绪（先【回读当前】确认坐标在用户坐标系1）")
        grid.addWidget(self.status, 8, 0, 1, 3)

    # -- helpers ------------------------------------------------------------
    def _pending(self):
        xyz = [self._spins[k].value() for k in ("X", "Y", "Z")]
        abc_deg = [self._spins[k].value() for k in ("A", "B", "C")]
        if self.rad_unit.isChecked():
            abc = abc_deg
        else:
            abc = np.radians(abc_deg)
        return xyz, list(np.asarray(abc, dtype=float)), abc_deg

    def _keep(self, worker):
        self._workers.append(worker)
        worker.finished.connect(lambda w=worker: self._forget(w))
        worker.start()

    def _forget(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)

    # -- actions ------------------------------------------------------------
    def _on_teleport(self):
        if self._busy:
            self.status.setText("⏳ 传送中，请稍候……")
            return
        if self._requires_readback:
            self.status.setText("⚠️ 请先点【回读当前】确认起点（面板启动时会自动填充）")
            return
        xyz, abc, _deg = self._pending()
        self._busy = True
        self.btn_teleport.setEnabled(False)
        self.status.setText("⏳ 传送到 X={:.2f} Y={:.2f} Z={:.2f} mm …".format(*xyz))
        worker = TeleportWorker(self.jog_provider, xyz, abc)
        worker.done.connect(self._teleport_done)
        self._watchdog.start()
        self._keep(worker)

    def _teleport_done(self, ok, message):
        self._watchdog.stop()
        self._busy = False
        self.btn_teleport.setEnabled(True)
        self.status.setText(message)

    def _on_teleport_timeout(self):
        """看门狗：35s 未完成 -> 面板必定恢复（线程不强杀，引用放弃自然终止）。"""
        self._watchdog.stop()
        self._busy = False
        self.btn_teleport.setEnabled(True)
        self.status.setText("⚠️ 传送超过 35s 未完成（可能目标超出范围/干涉），请【回读当前】核对")

    def _on_readback(self):
        if self._busy:
            return
        self.status.setText("⏳ 回读当前位姿…")
        worker = ReadbackWorker(self.jog_provider)

        def _done(ok, payload, error):
            if ok:
                self._requires_readback = False
                self.btn_teleport.setEnabled(True)
                xyz, abc_rad = payload
                deg = np.degrees(abc_rad)
                for k, v in zip(("X", "Y", "Z"), xyz):
                    self._spins[k].setValue(float(v))
                for k, v in zip(("A", "B", "C"), deg):
                    self._spins[k].setValue(float(v))
                self.status.setText(
                    "✅ 当前：X={:.2f} Y={:.2f} Z={:.2f} mm · A={:.2f} B={:.2f} C={:.2f}°".format(
                        *xyz, *deg
                    )
                )
            else:
                self.status.setText("回读失败：{}".format(error))

        worker.done.connect(_done)
        self._keep(worker)

    def _on_estop(self):
        # 急停直发：0x2314（经 jog 的 _estop_lock 通道，不排队于普通动作）
        self.status.setText("🚨 急停发送中…")

        def _go():
            try:
                self.jog_provider().emergency_stop()
            except Exception as error:
                self.status.setText("急停失败：{}".format(error))
                return
            self.status.setText("🚨 急停已发送（请到现场确认机器人已停）")

        threading.Thread(target=_go, daemon=True).start()


__all__ = ["TeleportPanel", "TeleportWorker", "ReadbackWorker"]
