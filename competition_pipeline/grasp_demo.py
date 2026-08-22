"""一键抓取 demo（用户坐标系1）：设抓取位/放置位 → 一键抓取。

序列（每段 move_to_ucs 都做"等到位"验证，超差/出错立即停止）：
    回复位点 → 抓取位上方 → 抓取位 → 夹爪合 → 抬升 → 放置位上方 → 放置位
    → 夹爪开 → 抬升 → 回复位点 → 完成
速度默认 100 mm/s（= 传送面板同一档，实测 ±0.01mm 级）。

急停：面板【急停】直发 0x2314（jog 的 _estop_lock 通道，不排队），
并置中止标志——demo 序列在步骤间检查，中止后不再发后续运动。

线程纪律：所有 QThread 读取/执行，仅通过 pyqtSignal 回调到 Qt 主线程
更新控件（禁止非主线程 setValue/setText，会闪退）。
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QMessageBox, QPushButton,
)

from competition_pipeline.geometry import (
    rotation_angle_deg, transform_from_inexbot_abc,
)


class GraspDemoWorker(QThread):
    """一键抓取序列（工作线程，上层强引用防 QThread 被 GC）。"""

    done = pyqtSignal(bool, str)

    def __init__(self, jog_provider, grasp, place, lift_mm=60.0,
                 vel_mm_s=100.0, tolerance_mm=1.2,
                 rotation_tolerance_deg=3.0, parent=None):
        super().__init__(parent)
        self.jog_provider = jog_provider
        self.grasp = list(grasp)          # [xyz_mm(3), abc_rad(3)]
        self.place = list(place)
        self.lift = float(lift_mm)
        self.vel = float(vel_mm_s)
        self.tolerance = float(tolerance_mm)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)
        self.abort = threading.Event()

    def run(self):
        jog = self.jog_provider()
        try:
            # 回复位点 (0x3007) 走控制器的 startRobotJobTask(safepos=1) 入口。
            # 在远程模式下它必然被复位点安全闸门拒绝，而**每次拒绝都会把伺服下电**
            # （出厂配置 RemoteIO[0].posReset.deviation=null 但 safeEnable=true）。
            # 2026-08-22 现场，这一句就是"抓取一开始机械臂就不动了、但夹爪照常开合"
            # 的起点：使能被打掉后，后续 MOVL 全部无效，而夹爪 0x3601 走另一条码路。
            #
            # 现在 jog.go_reset_position() 会先确保使能，适配器再用 0x3D03 status=2
            # 校验运动真的开始，被拒会立刻抛异常而不是静默继续走完整个序列。
            jog.go_reset_position()
            if self.abort.is_set():
                self.done.emit(False, "已急停中止（跳过抓取）")
                return
            g_xyz, g_abc = self.grasp
            p_xyz, p_abc = self.place
            up = np.array([0.0, 0.0, self.lift])
            g_up = list(np.asarray(g_xyz, dtype=float) + up)
            p_up = list(np.asarray(p_xyz, dtype=float) + up)
            steps = [
                ("夹爪开(准备)", None, None),
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
                if xyz is None:
                    jog.gripper(False) if "合" in name else jog.gripper(True)
                    time.sleep(0.5)                  # 气阀动作时间
                    continue
                jog.move_to_ucs(
                    xyz, abc, vel_mm_s=self.vel,
                    tolerance_mm=self.tolerance,
                    rotation_tolerance_deg=self.rotation_tolerance_deg,
                )
                # current_pose_rad() 而不是 current_pose()：后者返回角度制，
                # 在这个边界上做单位转换正是 2026-08-22 摔臂的成因，能不转就不转。
                actual_xyz, actual_abc_rad = jog.current_pose_rad()
                expected_pose = transform_from_inexbot_abc(
                    np.asarray(xyz, dtype=np.float64) / 1000.0,
                    np.asarray(abc, dtype=np.float64),
                )
                actual_pose = transform_from_inexbot_abc(
                    np.asarray(actual_xyz, dtype=np.float64) / 1000.0,
                    np.asarray(actual_abc_rad, dtype=np.float64),
                )
                rotation_error = rotation_angle_deg(expected_pose, actual_pose)
                if rotation_error > self.rotation_tolerance_deg:
                    raise RuntimeError(
                        "{} 姿态偏差 {:.2f}° 超过 {:.1f}°".format(
                            name, rotation_error, self.rotation_tolerance_deg
                        )
                    )
            # Do not leave the arm at place-above.  With controller reset-point
            # safety enabled, that pose is not a legal start for the next run.
            jog.go_reset_position()
            self.done.emit(True, "✅ 一键抓取完成：抓取({:.1f},{:.1f},{:.1f}) → "
                                 "放置({:.1f},{:.1f},{:.1f}) mm → 已回复位点".format(
                                     *g_xyz, *p_xyz))
        except Exception as error:
            self.done.emit(False, "抓取失败：{}".format(error))


class VisualPlanWorker(QThread):
    """Move to the reset/photo point and compute one frozen lemon plan."""

    done = pyqtSignal(bool, object)

    def __init__(self, jog_provider, snapshot_provider, settings, parent=None):
        super().__init__(parent)
        self.jog_provider = jog_provider
        self.snapshot_provider = snapshot_provider
        self.settings = dict(settings or {})
        self.cancel = threading.Event()
        self.process = None

    def stop(self):
        self.cancel.set()
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    def run(self):
        try:
            jog = self.jog_provider()
            # Validate camera availability before commanding the reset move.
            snapshot = self.snapshot_provider()
            if bool(self.settings.get("reset_before_capture", True)):
                previous_stamp = float(snapshot.get("image_timestamp_s", 0.0))
                jog.go_reset_position()
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    candidate = self.snapshot_provider()
                    if float(candidate.get("image_timestamp_s", 0.0)) != previous_stamp:
                        snapshot = candidate
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError("机器人复位后 3s 内没有新 RGB-D 帧")
            if self.cancel.is_set():
                raise RuntimeError("视觉计算已取消")
            # 弧度直取，不经过角度制中转（见 nexbot_jog 的单位约定）。
            xyz_mm, abc_rad = jog.current_pose_rad()
            abc_rad = np.asarray(abc_rad, dtype=np.float64)

            with tempfile.TemporaryDirectory(prefix="lemon-visual-plan-") as directory:
                root = Path(directory)
                snapshot_path = root / "snapshot.npz"
                tcp_path = root / "tcp.json"
                result_path = root / "plan.json"
                overlay_path = root / "overlay.png"
                np.savez_compressed(
                    str(snapshot_path),
                    color_bgr=np.asarray(snapshot["color_bgr"], dtype=np.uint8),
                    depth_m=np.asarray(snapshot["depth_m"], dtype=np.float32),
                    camera_matrix=np.asarray(
                        snapshot["camera_matrix"], dtype=np.float64
                    ),
                )
                tcp_path.write_text(
                    json.dumps(
                        {
                            "xyz_mm": [float(value) for value in xyz_mm],
                            "abc_rad": [float(value) for value in abc_rad],
                            "image_timestamp_s": float(
                                snapshot.get("image_timestamp_s", 0.0)
                            ),
                            "sync_delta_s": float(snapshot.get("sync_delta_s", 0.0)),
                        }
                    ),
                    encoding="utf-8",
                )
                command = [
                    str(self.settings["foundationpose_python"]),
                    "-m", "competition_pipeline.visual_grasp_bridge",
                    "--snapshot", str(snapshot_path),
                    "--tcp-pose", str(tcp_path),
                    "--visual-config", str(self.settings["visual_config"]),
                    "--competition-config", str(self.settings["competition_config"]),
                    "--output", str(result_path),
                    "--overlay", str(overlay_path),
                    "--target-label", str(self.settings.get("target_label", "lemon")),
                    "--place-offset-user-mm",
                    *[str(value) for value in self.settings.get(
                        "place_offset_user_mm", [0.0, -50.0, 0.0]
                    )],
                    "--origin-xy-tolerance-mm",
                    str(self.settings.get("origin_xy_tolerance_mm", 50.0)),
                    "--lift-mm", str(self.settings.get("lift_mm", 80.0)),
                    "--minimum-depth-coverage",
                    str(self.settings.get("minimum_depth_coverage", 0.15)),
                    "--maximum-depth-center-delta-mm",
                    str(self.settings.get("maximum_depth_center_delta_mm", 80.0)),
                ]
                environment = os.environ.copy()
                environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                environment.setdefault("CUDA_MODULE_LOADING", "LAZY")
                if self.cancel.is_set():
                    raise RuntimeError("视觉计算已取消")
                self.process = subprocess.Popen(
                    command,
                    cwd=str(Path(__file__).resolve().parents[1]),
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    stdout, stderr = self.process.communicate(
                        timeout=float(self.settings.get("vision_timeout_s", 240.0))
                    )
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.communicate()
                    raise RuntimeError("FoundationPose 视觉计算超时")
                returncode = self.process.returncode
                self.process = None
                if self.cancel.is_set():
                    raise RuntimeError("视觉计算已取消")
                if returncode != 0:
                    detail = (stderr or stdout).strip()
                    raise RuntimeError(detail[-2000:] or "视觉子进程失败")
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                overlay = cv2.imread(str(overlay_path))
                if overlay is not None:
                    payload["overlay_bgr"] = overlay
                self.done.emit(True, payload)
        except Exception as error:
            self.done.emit(False, str(error))


class _ReadbackThread(QThread):
    """回读当前位姿（回调自动回到 Qt 主线程）。"""

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


class _EstopThread(QThread):
    """急停直发 + 中止序列标志（回调主线程更新状态）。"""

    done = pyqtSignal(object)

    def __init__(self, jog_provider, demo_worker, parent=None):
        super().__init__(parent)
        self.jog_provider = jog_provider
        self.demo_worker = demo_worker

    def run(self):
        try:
            self.jog_provider().emergency_stop()
            if self.demo_worker is not None:
                self.demo_worker.abort.set()
            self.done.emit(("OK",))
        except Exception as error:
            self.done.emit(("ERR", str(error)))


def _pose_rows(group):
    """紧凑布局：X/Y/Z 一行 + A/B/C 一行。返回 dict。"""
    out = {}

    def _add(axis, unit, rng, row, col):
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(*rng)
        spin.setSuffix(unit)
        spin.setValue(0.0)
        group.addWidget(QLabel(axis), row, col * 2)
        group.addWidget(spin, row, col * 2 + 1)
        out[axis] = spin

    for i, (axis, unit, rng) in enumerate([
        ("X", " mm", (-2000.0, 2000.0)),
        ("Y", " mm", (-2000.0, 2000.0)),
        ("Z", " mm", (-2000.0, 2000.0)),
    ]):
        _add(axis, unit, rng, 0, i)
    for i, (axis, unit, rng) in enumerate([
        ("A", " °", (-360.0, 360.0)),
        ("B", " °", (-180.0, 180.0)),
        ("C", " °", (-360.0, 360.0)),
    ]):
        _add(axis, unit, rng, 1, i)
    return out


class GraspDemoPanel(QGroupBox):
    """一键抓取（用户坐标系1）：移到目标后【设为当前…】，再一键执行。"""

    def __init__(self, jog_provider, snapshot_provider=None, settings=None,
                 motion_authorized=None, motion_revoke=None,
                 preview_callback=None, parent=None):
        super().__init__("视觉抓取（用户坐标系1）· 识别预览 → 人工确认 → 抓取放左侧50mm", parent)
        self.jog_provider = jog_provider
        self.snapshot_provider = snapshot_provider
        self.settings = dict(settings or {})
        self.motion_authorized = motion_authorized or (lambda: True)
        self.motion_revoke = motion_revoke or (lambda: None)
        self.preview_callback = preview_callback
        self._workers = []
        self._busy = False
        self._visual_plan = None
        self._visual_plan_created_s = None
        self._grasp_confirmed = False
        self._place_confirmed = False

        grid = QGridLayout(self)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.btn_visual = QPushButton("① 回复位点并识别柠檬（只计算，不抓取）")
        self.btn_visual.clicked.connect(self._on_visual_plan)
        grid.addWidget(self.btn_visual, 0, 0, 1, 4)

        self.visual_result = QLabel(
            "待识别：柠檬放在用户系原点，识别后会显示物体/抓取/放置坐标。"
        )
        self.visual_result.setWordWrap(True)
        self.visual_result.setStyleSheet(
            "background:#eef5f3; border:1px solid #c8ddd8; padding:7px;"
        )
        grid.addWidget(self.visual_result, 1, 0, 1, 4)

        g_box = QGroupBox("抓取位（先移到目标处再点【设为当前抓取位】）")
        g_layout = QGridLayout(g_box)
        self.g_spins = _pose_rows(g_layout)
        self.g_set = QPushButton("设为当前抓取位")
        self.g_set.clicked.connect(lambda: self._read_into(self.g_spins, "抓取位"))
        g_layout.addWidget(self.g_set, 2, 0, 1, 6)
        grid.addWidget(g_box, 2, 0, 1, 4)

        p_box = QGroupBox("放置位（先移到目标处再点【设为当前放置位】）")
        p_layout = QGridLayout(p_box)
        self.p_spins = _pose_rows(p_layout)
        self.p_set = QPushButton("设为当前放置位")
        self.p_set.clicked.connect(lambda: self._read_into(self.p_spins, "放置位"))
        p_layout.addWidget(self.p_set, 2, 0, 1, 6)
        grid.addWidget(p_box, 3, 0, 1, 4)

        self.lift = QDoubleSpinBox()
        self.lift.setRange(10.0, 500.0)
        self.lift.setValue(float(self.settings.get("lift_mm", 80.0)))
        self.lift.setSuffix(" mm")
        self.vel = QDoubleSpinBox()
        self.vel.setRange(10.0, 250.0)
        self.vel.setValue(float(self.settings.get("speed_mm_s", 50.0)))
        self.vel.setSuffix(" mm/s")
        grid.addWidget(QLabel("抬起"), 4, 0)
        grid.addWidget(self.lift, 4, 1)
        grid.addWidget(QLabel("速度"), 4, 2)
        grid.addWidget(self.vel, 4, 3)

        self.btn_go = QPushButton("一键抓取")
        self.btn_go.setEnabled(False)
        self.btn_estop = QPushButton("急停")
        self.btn_estop.setStyleSheet(
            "background:#c0392b; color:white; font-weight:bold;"
        )
        self.btn_go.clicked.connect(self._on_go)
        self.btn_estop.clicked.connect(self._on_estop)
        grid.addWidget(self.btn_go, 5, 0, 1, 2)
        grid.addWidget(self.btn_estop, 5, 2, 1, 2)

        self.status = QLabel("就绪：把机器人移到抓取位/放置位，各点一次【设为当前…】")
        grid.addWidget(self.status, 6, 0, 1, 4)

    # -- helpers ------------------------------------------------------------
    def _pose_of(self, spins):
        xyz = [spins[k].value() for k in ("X", "Y", "Z")]
        abc = np.radians([spins[k].value() for k in ("A", "B", "C")])
        return list(xyz), list(np.asarray(abc, dtype=float))

    def _read_into(self, spins, label):
        self._clear_visual_plan()
        self.status.setText("⏳ 回读当前位姿…")
        worker = _ReadbackThread(self.jog_provider)

        def _done(payload):
            if payload[0] == "ERR":
                self.status.setText("回读失败：{}".format(payload[1]))
                return
            _ok, xyz, abc_deg = payload
            for k, v in zip(("X", "Y", "Z"), xyz):
                spins[k].setValue(float(v))
            for k, v in zip(("A", "B", "C"), abc_deg):
                spins[k].setValue(float(v))
            if spins is self.g_spins:
                self._grasp_confirmed = True
            if spins is self.p_spins:
                self._place_confirmed = True
            self.btn_go.setEnabled(
                self._grasp_confirmed and self._place_confirmed and not self._busy
            )
            self.status.setText(
                "✅ {}已填入当前位姿 X={:.2f} Y={:.2f} Z={:.2f} mm".format(
                    label, *xyz)
            )

        worker.done.connect(_done)
        self._keep(worker)

    def _keep(self, worker):
        self._workers.append(worker)
        worker.finished.connect(lambda w=worker: self._forget(w))
        worker.start()

    def _forget(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)

    def stop_workers(self):
        """Request cancellation without blocking the Qt main thread."""
        for worker in list(self._workers):
            if isinstance(worker, VisualPlanWorker):
                worker.stop()
            elif isinstance(worker, GraspDemoWorker):
                worker.abort.set()

    def wait_workers(self, timeout_ms=5000):
        for worker in list(self._workers):
            worker.wait(int(timeout_ms))

    def _motion_ok(self):
        if not bool(self.motion_authorized()):
            self.status.setText("⛔ 请先确认现场急停有效并勾选运动授权")
            return False
        return True

    @staticmethod
    def _fill_pose(spins, mapping):
        for key, value in zip(("X", "Y", "Z"), mapping["xyz_mm"]):
            spins[key].setValue(float(value))
        for key, value in zip(("A", "B", "C"), mapping["abc_deg"]):
            spins[key].setValue(float(value))

    def _clear_visual_plan(self):
        self._visual_plan = None
        self._visual_plan_created_s = None
        self.btn_go.setText("一键抓取（手动位姿）")
        self.btn_go.setEnabled(
            self._grasp_confirmed and self._place_confirmed and not self._busy
        )

    def _on_visual_plan(self):
        if self._busy:
            self.status.setText("⏳ 当前任务尚未完成")
            return
        if self.snapshot_provider is None:
            self.status.setText("视觉快照接口未配置")
            return
        if not self._motion_ok():
            return
        self._clear_visual_plan()
        self._grasp_confirmed = False
        self._place_confirmed = False
        self._busy = True
        self.btn_visual.setEnabled(False)
        self.btn_go.setEnabled(False)
        self.status.setText("⏳ 正在回复位点、拍照并计算柠檬用户系坐标…")
        worker = VisualPlanWorker(
            self.jog_provider, self.snapshot_provider, self.settings
        )

        def _done(ok, payload):
            self._busy = False
            self.btn_visual.setEnabled(True)
            if not ok:
                self.btn_go.setEnabled(False)
                self.visual_result.setText("❌ 视觉计算失败：{}".format(payload))
                self.status.setText("识别失败，未生成任何运动目标")
                return
            if self.preview_callback is not None and payload.get("overlay_bgr") is not None:
                self.preview_callback(payload["overlay_bgr"])
            plan = payload["plan"]
            quality = payload["quality"]
            obj = plan["object"]["xyz_mm"]
            grasp = plan["grasp_tcp"]["xyz_mm"]
            place = plan["place_tcp"]["xyz_mm"]
            reasons = payload.get("blocked_reasons") or []
            text = (
                "目标 {name} conf={conf:.3f} · Depth={coverage:.1%} · 估计夹持宽={width}mm\n"
                "物体 UCS1 XYZ={obj} mm（原点XY误差 {origin:.1f}mm）\n"
                "TCP抓取={grasp} mm → TCP放置={place} mm（Y- 50mm）"
            ).format(
                name=payload["target"]["name"],
                conf=payload["target"]["confidence"],
                coverage=quality["depth_coverage"],
                width=quality.get("estimated_grasp_width_mm"),
                obj=[round(value, 2) for value in obj],
                origin=plan["origin_xy_error_mm"],
                grasp=[round(value, 2) for value in grasp],
                place=[round(value, 2) for value in place],
            )
            if reasons:
                self.visual_result.setText(text + "\n⛔ 禁止执行：" + "；".join(reasons))
                self.btn_go.setEnabled(False)
                self.status.setText("坐标已显示，但质量/原点校验未通过")
                return
            self._fill_pose(self.g_spins, plan["grasp_tcp"])
            self._fill_pose(self.p_spins, plan["place_tcp"])
            self._grasp_confirmed = True
            self._place_confirmed = True
            self._visual_plan = payload
            self._visual_plan_created_s = time.monotonic()
            self.btn_go.setText("② 确认执行视觉抓取（放左侧50mm）")
            self.btn_go.setEnabled(True)
            self.visual_result.setText(text + "\n✅ 校验通过，请核对后点击确认执行。")
            self.status.setText("视觉只计算完成；机器人尚未执行抓取")

        worker.done.connect(_done)
        self._keep(worker)

    # -- actions ------------------------------------------------------------
    def _on_go(self):
        if self._busy:
            self.status.setText("⏳ 抓取序列执行中……")
            return
        if not self._motion_ok():
            return
        if not (self._grasp_confirmed and self._place_confirmed):
            self.status.setText("⛔ 抓取位和放置位尚未经视觉或当前位姿确认")
            return
        grasp = self._pose_of(self.g_spins)
        place = self._pose_of(self.p_spins)
        if self._visual_plan is not None:
            maximum_age = float(self.settings.get("maximum_plan_age_s", 60.0))
            age = time.monotonic() - self._visual_plan_created_s
            expected_g = self._visual_plan["plan"]["grasp_tcp"]
            expected_p = self._visual_plan["plan"]["place_tcp"]
            current = np.asarray(grasp[0] + list(np.degrees(grasp[1]))
                                 + place[0] + list(np.degrees(place[1])))
            expected = np.asarray(
                expected_g["xyz_mm"] + expected_g["abc_deg"]
                + expected_p["xyz_mm"] + expected_p["abc_deg"]
            )
            if age > maximum_age:
                self.status.setText("⛔ 视觉计划已超过 {:.0f}s，请重新识别".format(maximum_age))
                return
            if not np.allclose(current, expected, atol=0.02, rtol=0.0):
                self.status.setText("⛔ 视觉坐标已被修改，请重新识别后执行")
                return
            answer = QMessageBox.question(
                self,
                "确认视觉抓取",
                "即将抓取柠檬：\n"
                "物体 UCS1 XYZ={} mm\n"
                "抓取 TCP XYZ={} mm\n"
                "放置 TCP XYZ={} mm（用户系 Y- 50mm）\n\n"
                "已确认路径无障碍且实体急停可用吗？".format(
                    self._visual_plan["plan"]["object"]["xyz_mm"],
                    expected_g["xyz_mm"], expected_p["xyz_mm"],
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self.status.setText("已取消，未发送抓取指令")
                return
        self._busy = True
        self.btn_go.setEnabled(False)
        self.status.setText("⏳ 一键抓取执行中…（急停可随时中止）")
        worker = GraspDemoWorker(
            self.jog_provider, grasp, place,
            lift_mm=self.lift.value(), vel_mm_s=self.vel.value(),
            tolerance_mm=float(self.settings.get("arrival_tolerance_mm", 2.0)),
            rotation_tolerance_deg=float(
                self.settings.get("arrival_rotation_tolerance_deg", 3.0)
            ),
        )

        def _done(ok, message):
            self._busy = False
            self.btn_go.setEnabled(True)
            self.status.setText(message)
            if self._visual_plan is not None:
                self._clear_visual_plan()
            self._grasp_confirmed = False
            self._place_confirmed = False
            self.btn_go.setEnabled(False)

        worker.done.connect(_done)
        self._keep(worker)

    def _on_estop(self):
        self.status.setText("🚨 急停发送中…")
        demo_worker = next(
            (w for w in self._workers if isinstance(w, GraspDemoWorker)), None
        )
        worker = _EstopThread(self.jog_provider, demo_worker)

        def _done(payload):
            if payload[0] == "ERR":
                self.status.setText("急停失败：{}".format(payload[1]))
            else:
                self.motion_revoke()
                self.status.setText("🚨 急停已发送并中止序列（现场确认机器人已停）")

        worker.done.connect(_done)
        self._keep(worker)


__all__ = ["GraspDemoPanel", "GraspDemoWorker"]
