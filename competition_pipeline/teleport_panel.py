"""目标点传送面板（用户坐标系1）—— 等 pipeline UI 合并后一行挂接。

设计为独立模块：不修改 ``ui.py`` 既有逻辑，只导出：
    from competition_pipeline.teleport_panel import TeleportPanel
    panel = TeleportPanel(endpoint)      # endpoint = pose_endpoint_from_config(...)
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
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QPushButton,
)

from competition_pipeline.geometry import inexbot_abc_from_transform
from competition_pipeline.scripts.move_user1 import User1Mover


class TeleportWorker(QThread):
    """一次传送动作（独立线程；不碰 jog 面板的共享连接）。"""

    done = pyqtSignal(bool, str)

    def __init__(self, endpoint, xyz_mm, abc_rad, tolerance_mm=1.0, parent=None):
        super().__init__(parent)
        self.endpoint = endpoint
        self.xyz = list(xyz_mm)
        self.abc = list(abc_rad)
        self.tolerance = float(tolerance_mm)

    def run(self):
        mover = User1Mover(self.endpoint)
        try:
            (start_xyz, _), (_final, _), deviation = mover.move_to(
                self.xyz, self.abc, tolerance_mm=self.tolerance
            )
            self.done.emit(
                True,
                "✅ 传送完成：X={:.2f} Y={:.2f} Z={:.2f} mm → 偏差 {:.3f} mm".format(
                    *[_final[0], _final[1], _final[2]], deviation
                ),
            )
        except Exception as error:
            self.done.emit(False, "传送失败：{}".format(error))
        finally:
            mover.close()


class ReadbackWorker(QThread):
    """回读当前位姿并回填输入框。"""

    done = pyqtSignal(bool, object, str)

    def __init__(self, endpoint, parent=None):
        super().__init__(parent)
        self.endpoint = endpoint

    def run(self):
        mover = User1Mover(self.endpoint)
        try:
            xyz, abc_rad = mover.current_pose()
            self.done.emit(True, (xyz, abc_rad), "")
        except Exception as error:
            self.done.emit(False, None, str(error))
        finally:
            mover.close()


class TeleportPanel(QGroupBox):
    """目标点传送（用户坐标系1）：输入 XYZ(mm) + ABC → 传送 → 到达偏差。"""

    def __init__(self, endpoint, parent=None):
        super().__init__("目标点传送（用户坐标系1）· 输入点直接发到控制器并核对到位", parent)
        self.endpoint = endpoint
        self._workers = []          # 强引用防 QThread 被 GC（QThread 崩溃教训）
        self._busy = False

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
        xyz, abc, _deg = self._pending()
        self._busy = True
        self.btn_teleport.setEnabled(False)
        self.status.setText("⏳ 传送到 X={:.2f} Y={:.2f} Z={:.2f} mm …".format(*xyz))
        worker = TeleportWorker(self.endpoint, xyz, abc)

        def _done(ok, message):
            self._busy = False
            self.btn_teleport.setEnabled(True)
            self.status.setText(message)

        worker.done.connect(_done)
        self._keep(worker)

    def _on_readback(self):
        if self._busy:
            return
        self.status.setText("⏳ 回读当前位姿…")
        worker = ReadbackWorker(self.endpoint)

        def _done(ok, payload, error):
            if ok:
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
        # 急停直发：0x2314（万不得已人人可用；与传送互斥锁无关）
        from competition_pipeline.scripts.move_user1 import User1Mover
        mover = User1Mover(self.endpoint)

        def _go():
            try:
                mover.emergency_stop()
            finally:
                mover.close()

        threading.Thread(target=_go, daemon=True).start()
        self.status.setText("🚨 急停已发送")


__all__ = ["TeleportPanel", "TeleportWorker", "ReadbackWorker"]
