"""目标点传送面板（用户坐标系1）。

挂接（复用 UI 的共享持久连接，6001/7000 单客户端端口不新增连接）：
    from competition_pipeline.teleport_panel import TeleportPanel
    panel = TeleportPanel(
        jog_provider=lambda: self._get_robot_jog(),
        motion_authorized=lambda: self.robot_ctrl_enable.isChecked(),
    )
    layout.addWidget(panel)

面板内容：
  - X/Y/Z （QDoubleSpinBox，mm，±2000）
  - A/B/C （QDoubleSpinBox，**固定角度制**，见下方"单位"）
  - 【传送】把输入点发到控制器并回读核对（工作线程，不卡 UI）
  - 【回读当前】填当前用户1系位姿
  - 【急停】0x2314 直发（不受"传送忙"限制）
  - 状态栏：位移/姿态变化预览、到达偏差、错误

链路（全部现场实测）：0x2311 上电 → 0x4502(+acc=dec=10) coord=3 →
0x3D03 status=2 确认真的动了 → 回读核对位置与姿态。

单位：为什么没有"用弧度"复选框
------------------------------
曾经有过。``_on_readback()`` 永远把 ``np.degrees(abc_rad)`` 填进 A/B/C 框，
而 ``_pending()`` 在该复选框被勾选时**直接把框里的数当弧度用**。于是
"回读 → 只改 XYZ → 传送"这个完全正常的操作会把角度值当弧度发出去。

2026-08-22 现场实际后果（控制器日志 16:58:19，见
``docs/现场备份-20260822/根因证据-控制器日志摘录.txt`` 证据 C-1/C-2）：
回读得到 A/B/C = (177.8697, 13.7625, -179.9943) 度，被当成弧度发出，
折叠后成为 (111.19, 68.53, 127.09) 度 —— 与真实姿态相差 **119.6°**，
而 XYZ 只走 30mm 直线。6 秒后六个伺服同时报 0F15 故障，控制器 PowerOff，
**机械臂失电坠落**。

结论：这个复选框被永久删除。面板只用角度制，转换只在 ``_pending()`` 里
发生一次（``np.radians``），没有第二条路径。不要再把它加回来。
"""

import threading

import numpy as np
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QPushButton,
)

from competition_pipeline.geometry import (
    rotation_angle_deg, transform_from_inexbot_abc,
)
from competition_pipeline.nexbot_jog import MAX_ROTATION_STEP_DEG


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

    def __init__(self, jog_provider, motion_authorized=None, parent=None):
        super().__init__("目标点传送（用户坐标系1）· A/B/C 固定角度制 · 下发前校验位移与姿态", parent)
        self.jog_provider = jog_provider
        # 与主界面"已确认急停可用"复选框联动。缺省放行只是为了让面板能被
        # 单独实例化做离线测试；UI 挂接时必须把真实的授权回调传进来。
        self.motion_authorized = motion_authorized or (lambda: True)
        self._workers = []          # 强引用防 QThread 被 GC（QThread 崩溃教训）
        self._busy = False
        self._requires_readback = True   # 未确认过当前位姿前禁止传送
        self._readback_pose = None       # (xyz_mm, abc_deg) 最近一次回读，用于预览
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
            spin.valueChanged.connect(self._refresh_preview)
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(spin, i, 1)
            self._spins[name] = spin
        # NOTE: 这里**故意**没有"A/B/C 用弧度"复选框。它曾经存在，并且直接
        # 导致了 2026-08-22 的机械臂坠落 —— 详见模块 docstring。不要加回来。
        self.preview = QLabel("位移/姿态变化：待回读")
        self.preview.setWordWrap(True)
        grid.addWidget(self.preview, 6, 0, 1, 3)

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
        """(xyz_mm, abc_rad, abc_deg) —— 输入框恒为角度制，这里唯一一次转换。"""
        xyz = [self._spins[k].value() for k in ("X", "Y", "Z")]
        abc_deg = [self._spins[k].value() for k in ("A", "B", "C")]
        abc_rad = list(np.radians(np.asarray(abc_deg, dtype=float)))
        return xyz, abc_rad, abc_deg

    def _refresh_preview(self):
        """把"这一下会走多远、会转多少度"实时显示出来，下发前就能看见。"""
        if self._readback_pose is None:
            self.preview.setText("位移/姿态变化：待回读")
            return
        start_xyz, start_deg = self._readback_pose
        xyz, abc_rad, abc_deg = self._pending()
        distance = float(np.linalg.norm(
            np.asarray(xyz, dtype=float) - np.asarray(start_xyz, dtype=float)))
        start_pose = transform_from_inexbot_abc(
            np.asarray(start_xyz, dtype=float) / 1000.0,
            np.radians(np.asarray(start_deg, dtype=float)),
        )
        target_pose = transform_from_inexbot_abc(
            np.asarray(xyz, dtype=float) / 1000.0,
            np.asarray(abc_rad, dtype=float),
        )
        rotation = rotation_angle_deg(start_pose, target_pose)
        blocked = rotation > MAX_ROTATION_STEP_DEG
        self.preview.setText(
            "{}相对回读位姿：位移 {:.1f} mm · 姿态变化 {:.1f}°{}".format(
                "⛔ " if blocked else "",
                distance, rotation,
                "（超过 {:.0f}° 上限，传送会被拒绝）".format(MAX_ROTATION_STEP_DEG)
                if blocked else "",
            )
        )
        self.preview.setStyleSheet(
            "color:#c0392b; font-weight:bold;" if blocked else ""
        )

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
        if not bool(self.motion_authorized()):
            self.status.setText("⛔ 请先在现场确认急停有效并勾选主面板的运动授权")
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
                # 输入框恒为角度制 —— 这里填度数，_pending() 再转回弧度。
                # 只有这一条转换路径，不存在"填的单位和读的单位不一致"的可能。
                for k, v in zip(("X", "Y", "Z"), xyz):
                    self._spins[k].setValue(float(v))
                for k, v in zip(("A", "B", "C"), deg):
                    self._spins[k].setValue(float(v))
                self._readback_pose = (
                    [float(v) for v in xyz], [float(v) for v in deg]
                )
                self._refresh_preview()
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
