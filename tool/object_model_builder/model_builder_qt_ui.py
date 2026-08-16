#!/usr/bin/env python3
"""PyQt5 frontend for the object model builder backend."""

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .capture_session import CaptureSession
from .model_builder_ui import ModelBuilderApp
from .offline_reconstruction import reconstruct_capture_archive
from .session_archive import (
    create_capture_archive,
    create_foundationpose_reference_archive,
    extract_capture_archive,
)


class WidgetValue:
    """Expose Qt text widgets through the small get/set API used by the backend."""

    def __init__(self, widget):
        self.widget = widget

    def get(self):
        return self.widget.text()

    def set(self, value):
        self.widget.setText(str(value))


class StatusValue:
    def __init__(self, status_bar):
        self.status_bar = status_bar

    def get(self):
        return self.status_bar.currentMessage()

    def set(self, value):
        self.status_bar.showMessage(str(value))


class ButtonAdapter:
    def __init__(self, button):
        self.widget = button

    def configure(self, **options):
        if "text" in options:
            self.widget.setText(str(options["text"]))


class PlainTextAdapter:
    def __init__(self, widget):
        self.widget = widget

    def configure(self, **_options):
        if _options.get("state") == "disabled":
            self.widget.moveCursor(QTextCursor.Start)
        return None

    def delete(self, *_args):
        self.widget.clear()

    def insert(self, _position, text):
        self.widget.insertPlainText(str(text))


class ImageCanvas(QWidget):
    def __init__(self, empty_text):
        super().__init__()
        self._image = QImage()
        self._empty_text = empty_text
        self.setMinimumSize(180, 130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, frame):
        rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self._image = QImage(
            rgb.data, width, height, channels * width, QImage.Format_RGB888
        ).copy()
        self.update()

    def configure(self, **options):
        if options.get("image", None) == "":
            self._image = QImage()
        if "text" in options:
            self._empty_text = str(options["text"])
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#14191d"))
        if self._image.isNull():
            painter.setPen(QColor("#aab3ba"))
            font = QFont()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, self._empty_text)
            return
        target = QSize(max(1, self.width() - 20), max(1, self.height() - 20))
        scaled = self._image.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) / 2.0
        y = (self.height() - scaled.height()) / 2.0
        painter.drawImage(QRectF(x, y, scaled.width(), scaled.height()), scaled)


def bgr_to_qimage(frame):
    rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(
        rgb.data, width, height, channels * width, QImage.Format_RGB888
    ).copy()


class CaptureGalleryDialog(QDialog):
    """Inspect saved RGB, Mask and aligned depth without leaving the UI."""

    def __init__(self, session_path, parent=None):
        super().__init__(parent)
        self.session = CaptureSession.open(session_path)
        manifest = self.session.load_manifest()
        self.entries = list(manifest.get("views", []))
        if not self.entries:
            raise ValueError("当前会话还没有拍摄照片")
        self.setWindowTitle("已拍参考照片")
        self.resize(1120, 720)
        root = QVBoxLayout(self)
        body = QHBoxLayout()
        body.setSpacing(12)
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(230)
        self.list_widget.setIconSize(QSize(128, 72))
        for entry in self.entries:
            color = cv2.imread(
                str(self.session.root / entry["color"]), cv2.IMREAD_COLOR
            )
            if color is None:
                continue
            thumbnail = bgr_to_qimage(color).scaled(
                128, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            coverage = float(
                (entry.get("metadata") or {}).get("mask_depth_coverage", 0.0)
            )
            item = QListWidgetItem(
                QIcon(QPixmap.fromImage(thumbnail)),
                "第 {} 张\n深度 {:.0%}".format(int(entry["index"]) + 1, coverage),
            )
            item.setSizeHint(QSize(205, 88))
            self.list_widget.addItem(item)
        body.addWidget(self.list_widget)

        preview = QVBoxLayout()
        self.detail_label = QLabel()
        self.detail_label.setObjectName("resultBanner")
        self.detail_label.setWordWrap(True)
        preview.addWidget(self.detail_label)
        color_title = QLabel("保存的 RGB + Mask")
        color_title.setObjectName("previewCaption")
        self.color_canvas = ImageCanvas("没有 RGB")
        preview.addWidget(color_title)
        preview.addWidget(self.color_canvas, 2)
        lower = QHBoxLayout()
        mask_group = QVBoxLayout()
        mask_group.addWidget(QLabel("二值 Mask"))
        self.mask_canvas = ImageCanvas("没有 Mask")
        mask_group.addWidget(self.mask_canvas)
        depth_group = QVBoxLayout()
        depth_group.addWidget(QLabel("对齐深度"))
        self.depth_canvas = ImageCanvas("没有深度")
        depth_group.addWidget(self.depth_canvas)
        lower.addLayout(mask_group, 1)
        lower.addLayout(depth_group, 1)
        preview.addLayout(lower, 1)
        body.addLayout(preview, 1)
        root.addLayout(body, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.list_widget.currentRowChanged.connect(self._show_entry)
        self.list_widget.setCurrentRow(len(self.entries) - 1)

    @staticmethod
    def _depth_preview(depth_mm):
        depth = np.asarray(depth_mm, dtype=np.float32)
        valid = depth > 0
        output = np.zeros((*depth.shape, 3), dtype=np.uint8)
        if not np.any(valid):
            return output
        low, high = np.percentile(depth[valid], [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        colored = cv2.applyColorMap(
            np.rint((1.0 - normalized) * 255.0).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        colored[~valid] = 0
        return colored

    def _show_entry(self, row):
        if row < 0 or row >= len(self.entries):
            return
        entry = self.entries[row]
        color = cv2.imread(str(self.session.root / entry["color"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(
            str(self.session.root / entry["mask"]), cv2.IMREAD_GRAYSCALE
        )
        depth = cv2.imread(
            str(self.session.root / entry["depth_aligned"]), cv2.IMREAD_UNCHANGED
        )
        if color is None or mask is None or depth is None:
            self.detail_label.setText("这张照片的数据不完整")
            return
        binary = mask > 127
        overlay = color.copy()
        overlay[~binary] = (overlay[~binary] * 0.28).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 220, 255), 2)
        self.color_canvas.set_frame(overlay)
        self.mask_canvas.set_frame(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        self.depth_canvas.set_frame(self._depth_preview(depth))
        metadata = entry.get("metadata") or {}
        coverage = float(metadata.get("mask_depth_coverage", 0.0))
        sync_ms = float(metadata.get("rgb_depth_sync_delta_s", 0.0)) * 1000.0
        self.detail_label.setText(
            "第 {} 张 · Mask 深度覆盖 {:.1%} · 彩深时间差 {:.0f} ms · {}".format(
                int(entry["index"]) + 1,
                coverage,
                sync_ms,
                metadata.get("yolo_class", "object"),
            )
        )


class UiDispatcher(QObject):
    requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.requested.connect(self._invoke, Qt.QueuedConnection)

    @staticmethod
    def _invoke(callback):
        callback()


class RootAdapter:
    def __init__(self, window):
        self.window = window
        self.dispatcher = UiDispatcher()

    def after(self, delay_ms, callback):
        if int(delay_ms) <= 0:
            self.dispatcher.requested.emit(callback)
        else:
            QTimer.singleShot(int(delay_ms), callback)

    def configure(self, **options):
        cursor = options.get("cursor")
        if cursor == "watch":
            QApplication.setOverrideCursor(Qt.WaitCursor)
        elif cursor == "" and QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    @staticmethod
    def update_idletasks():
        QApplication.processEvents()

    def destroy(self):
        self.window.close()


class MetricLabel(QWidget):
    def __init__(self, caption, value="--"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        caption_label = QLabel(caption)
        caption_label.setObjectName("metricCaption")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        layout.addWidget(caption_label)
        layout.addWidget(self.value_label)
        self.value = WidgetValue(self.value_label)


def standard_icon(widget, name):
    return widget.style().standardIcon(getattr(QStyle, name))


def action_button(parent, text, icon_name, primary=False):
    button = QPushButton(text, parent)
    button.setObjectName("primaryButton" if primary else "secondaryButton")
    button.setIcon(standard_icon(parent, icon_name))
    button.setMinimumHeight(36 if primary else 34)
    return button


class ModelBuilderWindow(QMainWindow):
    def __init__(self, config_path):
        super().__init__()
        self.controller = QtModelBuilderController(self, config_path)

    def closeEvent(self, event):
        self.controller._shutdown_capture_analysis()
        if self.controller.source is not None:
            self.controller.source.stop()
            self.controller.source = None
        event.accept()


class QtModelBuilderController(ModelBuilderApp):
    STAGES = (
        "01  环境与相机",
        "02  RGB-D 标定",
        "03  无模型拍照",
        "04  重建与导出",
    )

    def __init__(self, window, config_path):
        self.window = window
        self.root = RootAdapter(window)
        self._initialize_backend(config_path)
        window.setWindowTitle("物体三维模型工作台")
        window.resize(1460, 900)
        window.setMinimumSize(1180, 720)
        icon_path = (
            Path.home()
            / "Applications"
            / "OrbbecViewer-1.10.27"
            / "res"
            / "orbbec_icon.png"
        )
        if icon_path.exists():
            from PyQt5.QtGui import QIcon

            window.setWindowIcon(QIcon(str(icon_path)))
        self._build_qt_ui()
        self._apply_qt_style()
        self._reload_rgbd_calibration(show_dialog=False)
        self._run_environment_check()
        QTimer.singleShot(80, self._tick)

    def _build_qt_ui(self):
        root = QWidget()
        self.window.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_preview_area(), 1)
        body.addWidget(self._build_control_area())
        outer.addLayout(body, 1)

        self.window.statusBar().setSizeGripEnabled(False)
        self.status_text = StatusValue(self.window.statusBar())
        self.status_text.set("等待操作")

    def _build_header(self):
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(66)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 20, 0)
        title = QLabel("物体三维模型工作台")
        title.setObjectName("appTitle")
        subtitle = QLabel("Astra Pro · AprilTag · YOLO Mask · TSDF · FoundationPose")
        subtitle.setObjectName("appSubtitle")
        titles = QVBoxLayout()
        titles.setSpacing(1)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        layout.addLayout(titles)
        layout.addStretch()

        badge = QLabel("相机未连接")
        badge.setObjectName("connectionBadge")
        self.connection_text = WidgetValue(badge)
        connect = action_button(self.window, "连接相机", "SP_MediaPlay", primary=True)
        connect.clicked.connect(self._toggle_camera)
        self.connect_button = ButtonAdapter(connect)
        layout.addWidget(badge)
        layout.addSpacing(10)
        layout.addWidget(connect)
        return header

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(238)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 18)
        layout.setSpacing(12)
        label = QLabel("建模流程")
        label.setObjectName("sectionLabel")
        layout.addWidget(label)
        self.stage_list = QListWidget()
        self.stage_list.setObjectName("stageList")
        self.stage_list.setSpacing(4)
        self.stage_list.setFixedHeight(208)
        self.stage_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for text in self.STAGES:
            item = QListWidgetItem(text)
            item.setSizeHint(QSize(190, 44))
            self.stage_list.addItem(item)
        self.stage_list.currentRowChanged.connect(self._change_stage)
        layout.addWidget(self.stage_list)

        source_label = QLabel("图像源")
        source_label.setObjectName("sectionLabel")
        layout.addWidget(source_label)
        backend = self.camera_config.get("backend", "astra_ros")
        source = QLabel(
            "Astra Pro · ROS" if backend == "astra_ros" else "OAK-D Pro · DepthAI"
        )
        source.setObjectName("sourceBanner")
        layout.addWidget(source)
        layout.addStretch()
        note = QLabel("工作坐标：ruler_workspace\n输出：FoundationPose CAD 模型")
        note.setObjectName("sidebarNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        return sidebar

    def _build_preview_area(self):
        area = QWidget()
        area.setObjectName("previewArea")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        heading = QHBoxLayout()
        title = QLabel("RGB-D 实时预览")
        title.setObjectName("viewTitle")
        state = QLabel("待机")
        state.setObjectName("viewState")
        self.preview_state_text = WidgetValue(state)
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(state)
        layout.addLayout(heading)

        color_panel, self.color_preview = self._preview_panel(
            "RGB / Tag / 分割结果", "相机未连接"
        )
        layout.addWidget(color_panel, 5)
        lower = QHBoxLayout()
        lower.setSpacing(10)
        depth_panel, self.depth_preview = self._preview_panel("原始深度", "等待深度图像")
        ir_panel, self.ir_preview = self._preview_panel(
            "红外原图 / {}".format(self.calibration_target.display_name), "等待红外图像"
        )
        aligned_panel, self.aligned_preview = self._preview_panel(
            "对齐深度 / Mask", "等待彩深对齐"
        )
        lower.addWidget(depth_panel, 1)
        lower.addWidget(ir_panel, 1)
        lower.addWidget(aligned_panel, 1)
        layout.addLayout(lower, 3)

        metrics = QFrame()
        metrics.setObjectName("metricBar")
        metric_layout = QHBoxLayout(metrics)
        metric_layout.setContentsMargins(18, 9, 18, 9)
        frame_metric = MetricLabel("图像", "--")
        detection_metric = MetricLabel("检测", "待机")
        capture_metric = MetricLabel("采集", "0 帧")
        self.frame_metric_text = frame_metric.value
        self.detection_metric_text = detection_metric.value
        self.capture_metric_text = capture_metric.value
        metric_layout.addWidget(frame_metric)
        metric_layout.addWidget(self._vertical_line())
        metric_layout.addWidget(detection_metric)
        metric_layout.addWidget(self._vertical_line())
        metric_layout.addWidget(capture_metric)
        layout.addWidget(metrics)
        return area

    def _preview_panel(self, title, empty_text):
        panel = QFrame()
        panel.setObjectName("previewPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        caption = QLabel(title)
        caption.setObjectName("previewCaption")
        canvas = ImageCanvas(empty_text)
        layout.addWidget(caption)
        layout.addWidget(canvas, 1)
        return panel, canvas

    @staticmethod
    def _vertical_line():
        line = QFrame()
        line.setObjectName("metricDivider")
        line.setFrameShape(QFrame.VLine)
        return line

    def _build_control_area(self):
        area = QFrame()
        area.setObjectName("controlArea")
        area.setFixedWidth(440)
        layout = QVBoxLayout(area)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(0)
        self.control_stack = QStackedWidget()
        self.control_stack.addWidget(self._scroll_page(self._build_environment_page()))
        self.control_stack.addWidget(self._scroll_page(self._build_calibration_page()))
        self.control_stack.addWidget(self._scroll_page(self._build_capture_page()))
        self.control_stack.addWidget(self._scroll_page(self._build_reconstruction_page()))
        layout.addWidget(self.control_stack, 1)
        self.stage_list.setCurrentRow(0)
        return area

    @staticmethod
    def _scroll_page(page):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(page)
        return scroll

    def _page(self, title, subtitle):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("panelSubtitle")
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(4)
        return page, layout

    def _build_environment_page(self):
        page, layout = self._page("环境与相机", "运行依赖与设备状态")
        log = QPlainTextEdit()
        log.setReadOnly(True)
        log.setMaximumBlockCount(300)
        self.environment_text = PlainTextAdapter(log)
        layout.addWidget(log, 1)
        refresh = action_button(self.window, "重新检查", "SP_BrowserReload")
        refresh.clicked.connect(self._run_environment_check)
        layout.addWidget(refresh)
        return page

    def _build_calibration_page(self):
        page, layout = self._page(
            "RGB-D 标定", "{} 彩色与红外图像对".format(self.calibration_target.display_name)
        )
        explanation = QLabel(
            "RGB 与深度不可按分辨率缩放。使用同一 {} 求 T_color_depth。"
            "自动采集只在共同角点达到阈值且视角变化后记录。"
            "当前方格尺寸以配置中的 {:.1f} mm 为准。".format(
                self.calibration_target.display_name,
                self.calibration_target.square_size_m * 1000.0
                if self.calibration_target.target_type == "checkerboard"
                else 36.0,
            )
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("mutedText")
        layout.addWidget(explanation)
        status = QLabel("RGB-D 外参无效")
        status.setObjectName("resultBanner")
        status.setWordWrap(True)
        self.rgbd_status_text = WidgetValue(status)
        pair_status = QLabel("尚未采集 RGB/IR 图像对")
        pair_status.setObjectName("mutedText")
        pair_status.setWordWrap(True)
        self.calibration_pair_text = WidgetValue(pair_status)
        layout.addWidget(status)
        layout.addWidget(pair_status)
        projector_status = QLabel("红外投影器：连接相机后自动控制")
        projector_status.setObjectName("resultBanner")
        projector_status.setWordWrap(True)
        self.projector_status_text = WidgetValue(projector_status)
        layout.addWidget(projector_status)
        corner_edit = self._line_field(
            layout,
            "共同角点阈值",
            str(self._auto_calibration_minimum_common),
        )
        target_edit = self._line_field(
            layout,
            "目标图像对",
            str(self._auto_calibration_target_pairs),
        )
        self.auto_corner_threshold_var = WidgetValue(corner_edit)
        self.auto_target_pairs_var = WidgetValue(target_edit)
        auto_button = action_button(
            self.window,
            "开始自动采集",
            "SP_MediaPlay",
            True,
        )
        auto_button.clicked.connect(self._toggle_auto_calibration)
        self.auto_calibration_button = ButtonAdapter(auto_button)
        layout.addWidget(auto_button)
        for text, callback, icon, primary in (
            ("手动采集当前 RGB/IR 对", self._capture_calibration_pair, "SP_DialogSaveButton", False),
            ("计算 RGB-D 标定", self._solve_rgbd_calibration, "SP_DialogApplyButton", False),
            ("写入中央参数", self._write_rgbd_calibration, "SP_DriveHDIcon", True),
            ("重新载入中央参数", self._reload_rgbd_calibration, "SP_BrowserReload", False),
        ):
            button = action_button(self.window, text, icon, primary)
            button.clicked.connect(callback)
            layout.addWidget(button)
        log = QPlainTextEdit()
        log.setReadOnly(True)
        log.setMaximumBlockCount(200)
        self.calibration_log = PlainTextAdapter(log)
        layout.addWidget(log, 1)
        return page

    def _line_field(self, layout, label, value, browse=None):
        row = QHBoxLayout()
        row.setSpacing(8)
        caption = QLabel(label)
        caption.setFixedWidth(82)
        edit = QLineEdit(value)
        edit.setToolTip(value)
        edit.setCursorPosition(0)
        row.addWidget(caption)
        row.addWidget(edit, 1)
        if browse is not None:
            button = QToolButton()
            button.setIcon(standard_icon(self.window, "SP_DirOpenIcon"))
            button.setToolTip("选择{}".format(label))
            button.setFixedSize(34, 34)
            button.clicked.connect(lambda: browse(edit))
            row.addWidget(button)
        layout.addLayout(row)
        return edit

    def _build_capture_page(self):
        page, layout = self._page(
            "无模型拍照与实时测试",
            "RGB-D 参考视图 · AprilTag · YOLO Mask · FoundationPose",
        )

        def choose_weights(edit):
            path, _ = QFileDialog.getOpenFileName(
                self.window, "选择 YOLO 分割权重", "", "PyTorch weights (*.pt *.pth)"
            )
            if path:
                edit.setText(path)
                edit.setToolTip(path)

        def choose_session(edit):
            path = QFileDialog.getExistingDirectory(
                self.window, "选择采集会话", self.paths["capture_root"]
            )
            if path:
                try:
                    self._open_capture_session(path)
                    edit.setToolTip(path)
                except Exception as error:
                    self._show_error("无法打开参考会话", error)

        def choose_mesh(edit):
            path, _ = QFileDialog.getOpenFileName(
                self.window,
                "选择 FoundationPose 米制网格",
                self.paths["mesh_root"],
                "Triangle mesh (*.obj *.ply *.stl)",
            )
            if path:
                edit.setText(path)
                edit.setToolTip(path)
                edit.setCursorPosition(0)

        weights = self._line_field(
            layout, "YOLO 权重", self.paths.get("yolo_weights", ""), choose_weights
        )
        classes = self._line_field(
            layout,
            "目标类别",
            ",".join(self.segmentation_config.get("target_classes", [])),
        )
        session = self._line_field(layout, "采集会话", "", choose_session)
        object_id = self._line_field(layout, "参考物体 ID", "1")
        object_name = self._line_field(
            layout,
            "参考物体名",
            (self.segmentation_config.get("target_classes") or ["object"])[0],
        )
        self.yolo_weights_var = WidgetValue(weights)
        self.target_class_var = WidgetValue(classes)
        self.session_var = WidgetValue(session)
        self.reference_object_id_var = WidgetValue(object_id)
        self.reference_object_name_var = WidgetValue(object_name)
        for text, callback, icon in (
            ("1  加载 YOLO 分割模型", self._load_yolo, "SP_ComputerIcon"),
            ("2  开始本次拍照（只需一次）", self._new_capture_session, "SP_FileDialogNewFolder"),
        ):
            button = action_button(self.window, text, icon)
            button.clicked.connect(callback)
            layout.addWidget(button)
        capture_button = action_button(
            self.window, "3  拍摄参考图", "SP_DialogSaveButton", primary=True
        )
        capture_button.clicked.connect(self._capture_view)
        self.capture_button = ButtonAdapter(capture_button)
        layout.addWidget(capture_button)
        gallery_button = action_button(
            self.window, "查看已拍照片（0 张）", "SP_FileDialogContentsView"
        )
        gallery_button.clicked.connect(self._show_capture_gallery)
        self.capture_gallery_button = ButtonAdapter(gallery_button)
        layout.addWidget(gallery_button)
        preview_button = action_button(
            self.window,
            "照片快速预览三维（至少 {} 张）".format(
                int(self.fusion_config["minimum_views"])
            ),
            "SP_FileDialogContentsView",
        )
        preview_button.clicked.connect(self._preview_captured_model)
        layout.addWidget(preview_button)
        export_button = action_button(
            self.window,
            "4  导出 FoundationPose 参考照片 ZIP",
            "SP_DriveFDIcon",
        )
        export_button.clicked.connect(self._pack_foundationpose_reference_zip)
        layout.addWidget(export_button)
        capture_status = QLabel("先加载 YOLO，再新建参考拍照会话")
        capture_status.setObjectName("resultBanner")
        capture_status.setWordWrap(True)
        self.capture_status_text = WidgetValue(capture_status)
        layout.addWidget(capture_status)
        gates = QLabel("Tag · Mask · 对齐深度 · 新视角")
        gates.setObjectName("mutedText")
        layout.addWidget(gates)

        live_title = QLabel("FoundationPose 实时测试")
        live_title.setObjectName("sectionLabel")
        layout.addSpacing(8)
        layout.addWidget(live_title)
        live_mesh = self._line_field(
            layout,
            "重建网格",
            str(self.foundationpose_live_config.get("mesh_path", "")),
            choose_mesh,
        )
        live_scale = self._line_field(
            layout,
            "米制缩放",
            str(self.foundationpose_live_config.get("mesh_scale_to_meters", 1.0)),
        )
        self.foundationpose_mesh_var = WidgetValue(live_mesh)
        self.foundationpose_mesh_scale_var = WidgetValue(live_scale)
        for text, callback, icon, primary in (
            ("5  加载网格并实时测试", self._load_foundationpose_live, "SP_ComputerIcon", True),
            ("初始化 / 重新初始化", self._reset_foundationpose_live, "SP_BrowserReload", False),
            ("停止实时测试", self._stop_foundationpose_live, "SP_MediaStop", False),
        ):
            button = action_button(self.window, text, icon, primary)
            button.clicked.connect(callback)
            layout.addWidget(button)
        live_status = QLabel("需要先把参考照片 ZIP 重建为 OBJ/PLY/STL")
        live_status.setObjectName("resultBanner")
        live_status.setWordWrap(True)
        self.foundationpose_live_status_text = WidgetValue(live_status)
        layout.addWidget(live_status)
        pose_text = QPlainTextEdit()
        pose_text.setReadOnly(True)
        pose_text.setMaximumHeight(132)
        pose_text.setPlainText("camera_from_object\n--")
        self.foundationpose_pose_text = PlainTextAdapter(pose_text)
        layout.addWidget(pose_text)
        layout.addStretch()
        return page

    def _new_capture_session(self):
        if self.capture_session is not None:
            count = len(self.capture_session)
            if count == 0:
                text = "当前拍照会话已经开始，直接点击“3 拍摄参考图”"
                self._set_capture_feedback(text, 5.0)
                self._set_status(text)
                return
            answer = QMessageBox.question(
                self.window,
                "开始新的拍照会话",
                "当前会话已有 {} 张照片。\n\n"
                "继续当前物体请点“否”并直接拍摄；只有更换物体时才新建会话。".format(
                    count
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        super()._new_capture_session()

    def _build_reconstruction_page(self):
        page, layout = self._page("重建与导出", "TSDF 网格与 FoundationPose 模型")

        def choose_archive(edit):
            path, _ = QFileDialog.getOpenFileName(
                self.window, "选择采集 ZIP", "", "Capture archive (*.zip)"
            )
            if path:
                edit.setText(path)
                edit.setToolTip(path)
                edit.setCursorPosition(0)

        def choose_mesh_root(edit):
            path = QFileDialog.getExistingDirectory(
                self.window, "选择模型输出目录", edit.text()
            )
            if path:
                edit.setText(path)
                edit.setToolTip(path)

        archive = self._line_field(layout, "素材 ZIP", "", choose_archive)
        self.archive_var = WidgetValue(archive)
        import_button = action_button(
            self.window, "导入并校验 ZIP", "SP_DialogOpenButton"
        )
        import_button.clicked.connect(self._import_capture_zip)
        layout.addWidget(import_button)
        mesh_root = self._line_field(
            layout, "输出目录", self.paths["mesh_root"], choose_mesh_root
        )
        model_name = self._line_field(layout, "模型名称", "bottle")
        voxel = self._line_field(
            layout, "TSDF 体素", str(self.fusion_config["voxel_length_m"])
        )
        truncation = self._line_field(
            layout, "TSDF 截断", str(self.fusion_config["sdf_trunc_m"])
        )
        self.mesh_root_var = WidgetValue(mesh_root)
        self.model_name_var = WidgetValue(model_name)
        self.voxel_var = WidgetValue(voxel)
        self.trunc_var = WidgetValue(truncation)
        archive_status = QLabel("可以选择本机或服务器收到的采集 ZIP")
        archive_status.setObjectName("resultBanner")
        archive_status.setWordWrap(True)
        self.archive_status_text = WidgetValue(archive_status)
        layout.addWidget(archive_status)
        reconstruct = action_button(
            self.window,
            "从 ZIP 一键重建并导出",
            "SP_MediaPlay",
            primary=True,
        )
        reconstruct.clicked.connect(self._reconstruct_capture_zip)
        layout.addWidget(reconstruct)
        for text, callback, icon, primary in (
            ("融合采集会话", self._fuse_session, "SP_MediaPlay", False),
            ("查看三维网格", self._view_mesh, "SP_FileDialogContentsView", False),
            ("导出 FoundationPose 模型", self._export_model, "SP_DialogSaveButton", False),
        ):
            button = action_button(self.window, text, icon, primary)
            button.clicked.connect(callback)
            layout.addWidget(button)
        mesh_status = QLabel("尚未生成网格")
        mesh_status.setObjectName("resultBanner")
        mesh_status.setWordWrap(True)
        self.mesh_status_text = WidgetValue(mesh_status)
        layout.addWidget(mesh_status)
        layout.addStretch()
        return page

    def _run_archive_job(self, status, worker, finished, error_title):
        if self._busy:
            self._set_status("已有任务正在运行")
            return
        self._busy = True
        self.control_stack.setEnabled(False)
        self.stage_list.setEnabled(False)
        self._set_status(status)

        def run():
            try:
                result = worker()
                self.root.after(0, lambda: self._archive_job_finished(result, finished))
            except Exception as error:
                self.root.after(
                    0,
                    lambda current=error: self._archive_job_failed(error_title, current),
                )

        threading.Thread(target=run, daemon=True).start()

    def _show_capture_gallery(self):
        session_path = self.session_var.get().strip()
        if not session_path:
            self._show_error("无法查看", "请先新建或打开参考拍照会话")
            return
        try:
            dialog = CaptureGalleryDialog(session_path, self.window)
            dialog.exec_()
        except Exception as error:
            self._show_error("无法查看已拍照片", error)

    def _archive_job_finished(self, result, callback):
        self._busy = False
        self.control_stack.setEnabled(True)
        self.stage_list.setEnabled(True)
        callback(result)

    def _archive_job_failed(self, title, error):
        self._busy = False
        self.control_stack.setEnabled(True)
        self.stage_list.setEnabled(True)
        self._show_error(title, error)

    def _pack_capture_zip(self):
        session_path = self.session_var.get().strip()
        if not session_path:
            self._show_error("无法打包", "请先新建、采集或选择一个采集会话")
            return
        session = Path(session_path).expanduser()
        default_path = session.parent / (session.name + ".zip")
        destination, _ = QFileDialog.getSaveFileName(
            self.window,
            "保存采集 ZIP",
            str(default_path),
            "Capture archive (*.zip)",
        )
        if not destination:
            return

        def worker():
            return create_capture_archive(session_path, destination)

        def finished(path):
            self.archive_var.set(str(path))
            self.archive_status_text.set("采集 ZIP 已生成：{}".format(path))
            self._set_status("采集 ZIP 已生成：{}".format(path))

        self._run_archive_job("正在校验并打包采集数据...", worker, finished, "ZIP 打包失败")

    def _pack_foundationpose_reference_zip(self):
        session_path = self.session_var.get().strip()
        if not session_path:
            self._show_error("无法打包", "请先新建、拍摄或选择参考拍照会话")
            return
        try:
            object_id = int(self.reference_object_id_var.get())
            if object_id <= 0:
                raise ValueError
        except ValueError:
            self._show_error("无法打包", "参考物体 ID 必须是正整数")
            return
        object_name = self.reference_object_name_var.get().strip() or "object"
        session = Path(session_path).expanduser()
        default_path = session.parent / (
            session.name + "_foundationpose_reference.zip"
        )
        destination, _ = QFileDialog.getSaveFileName(
            self.window,
            "保存 FoundationPose 无模型参考 ZIP",
            str(default_path),
            "FoundationPose reference archive (*.zip)",
        )
        if not destination:
            return

        def worker():
            return create_foundationpose_reference_archive(
                session_path,
                destination,
                object_id=object_id,
                object_name=object_name,
            )

        def finished(path):
            capture_text = "参考照片 ZIP 已生成：{}".format(path)
            self.capture_status_text.set(capture_text)
            self.foundationpose_live_status_text.set(
                "下一步：用 BundleSDF 重建 model.obj，再在“重建网格”中选择它"
            )
            self._set_status(capture_text)
            QMessageBox.information(
                self.window,
                "参考照片 ZIP 已生成",
                "ZIP 已保存到：\n{}\n\n"
                "这个 ZIP 是重建素材，不是可直接跟踪的模型。"
                "请先用 BundleSDF / Neural Object Field 生成 model.obj，"
                "再回到本页选择该网格进行实时测试。\n\n"
                "ZIP 内也包含“如何使用.txt”。".format(path),
            )

        self._run_archive_job(
            "正在生成 FoundationPose 无模型参考 ZIP...",
            worker,
            finished,
            "无模型参考 ZIP 打包失败",
        )

    def _import_capture_zip(self):
        archive_path = self.archive_var.get().strip()
        if not archive_path:
            self._show_error("无法导入", "请先选择采集 ZIP")
            return
        destination = Path(self.paths["capture_root"]) / "offline_imports"

        def worker():
            return extract_capture_archive(archive_path, str(destination))

        def finished(imported):
            self._open_capture_session(str(imported.session_path))
            self.archive_status_text.set(
                "校验通过：{} 帧，SHA-256 {}...".format(
                    imported.frame_count, imported.archive_sha256[:12]
                )
            )
            self._set_status("ZIP 已导入：{}".format(imported.session_path))

        self._run_archive_job("正在校验并导入采集 ZIP...", worker, finished, "ZIP 导入失败")

    def _reconstruct_capture_zip(self):
        archive_path = self.archive_var.get().strip()
        if not archive_path:
            self._show_error("无法重建", "请先选择采集 ZIP")
            return

        def worker():
            return reconstruct_capture_archive(
                archive_path,
                self.model_name_var.get(),
                config_path=str(self.config_path),
                output_root=self.mesh_root_var.get(),
                work_root=str(Path(self.paths["capture_root"]) / "offline_imports"),
                keep_extracted=True,
                voxel_length_m=float(self.voxel_var.get()),
                sdf_trunc_m=float(self.trunc_var.get()),
            )

        def finished(result):
            self.fusion_result = result.fusion_result
            self.session_var.set(str(result.imported_session.session_path))
            dimensions = [value * 1000.0 for value in result.fusion_result.dimensions_m]
            text = (
                "ZIP 重建完成：{} 个视角，尺寸 {:.1f} × {:.1f} × {:.1f} mm；结果 {}"
            ).format(
                result.fusion_result.views_integrated,
                dimensions[0],
                dimensions[1],
                dimensions[2],
                result.result_zip,
            )
            self.mesh_status_text.set(text)
            self.archive_status_text.set("校验、融合和模型打包均已完成")
            self._set_status(text)

        self._run_archive_job(
            "服务器模式：正在校验 ZIP、融合网格并导出 CAD...",
            worker,
            finished,
            "ZIP 重建失败",
        )

    def _change_stage(self, index):
        if index >= 0 and hasattr(self, "control_stack"):
            if int(index) != 1 and self._auto_calibration_active:
                self._stop_auto_calibration("离开标定页，自动采集已停止")
            if int(index) != 2 and self.foundationpose_live_active:
                self._stop_foundationpose_live()
            index = int(index)
            if index != self._active_stage_index:
                self._invalidate_capture_analysis(clear_state=index != 2)
            self._active_stage_index = index
            self._last_processed_anchor_timestamp_s = None
            self.control_stack.setCurrentIndex(index)
            self._sync_calibration_projector()

    def _set_preview(self, widget, bgr, _maximum_size, _key):
        widget.set_frame(bgr)

    def _show_error(self, title, error):
        self._set_status(str(error))
        QMessageBox.critical(self.window, title, str(error))

    def _apply_qt_style(self):
        self.window.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6f7; color: #20282d; font-size: 13px; }
            QFrame#header { background: #20272b; border: 0; }
            QLabel#appTitle { color: #ffffff; background: transparent; font-size: 18px; font-weight: 650; }
            QLabel#appSubtitle { color: #aeb8bd; background: transparent; font-size: 11px; }
            QLabel#connectionBadge { color: #d7dde0; background: #323b40; border: 1px solid #465158;
                                     padding: 6px 11px; border-radius: 4px; }
            QFrame#sidebar { background: #edf0f1; border-right: 1px solid #d6dcdf; }
            QFrame#controlArea { background: #ffffff; border-left: 1px solid #dce1e3; }
            QWidget#previewArea { background: #f4f6f7; }
            QLabel#sectionLabel { color: #647078; font-size: 11px; font-weight: 650; }
            QLabel#sourceBanner { background: #ffffff; border: 1px solid #cfd6d9;
                                  border-radius: 4px; padding: 9px; color: #243139; }
            QLabel#viewTitle { font-size: 17px; font-weight: 650; }
            QLabel#viewState { color: #0f766e; font-weight: 650; }
            QLabel#panelTitle { font-size: 18px; font-weight: 650; }
            QLabel#panelSubtitle, QLabel#mutedText { color: #6b767c; }
            QLabel#metricCaption { color: #707b81; font-size: 10px; }
            QLabel#metricValue { font-size: 14px; font-weight: 650; }
            QLabel#sidebarNote { color: #66737a; background: #e2e7e9; padding: 10px; border-radius: 4px; }
            QLabel#resultBanner { background: #eef5f3; color: #285c55; border: 1px solid #c8ddd8;
                                  padding: 10px; border-radius: 5px; }
            QListWidget#stageList { background: transparent; border: 0; outline: 0; }
            QListWidget#stageList::item { padding: 8px 10px; border-radius: 5px; color: #435057; }
            QListWidget#stageList::item:selected { background: #ffffff; color: #0b645c; font-weight: 650;
                                                   border: 1px solid #d3ddda; }
            QPushButton#primaryButton { background: #176d64; color: #ffffff; border: 1px solid #176d64;
                                        border-radius: 5px; padding: 7px 14px; font-weight: 650; }
            QPushButton#primaryButton:hover { background: #115c54; }
            QPushButton#primaryButton:disabled { background: #b7c3c2; border-color: #b7c3c2; }
            QPushButton#secondaryButton { background: #ffffff; border: 1px solid #cbd3d6;
                                          border-radius: 5px; padding: 7px 14px; }
            QPushButton#secondaryButton:hover { background: #f0f3f4; }
            QToolButton { background: #ffffff; border: 1px solid #cbd3d6; border-radius: 5px; }
            QToolButton:hover { background: #f0f3f4; }
            QLineEdit { background: #ffffff; border: 1px solid #cbd3d6; border-radius: 4px;
                        padding: 6px; min-height: 23px; }
            QFrame#previewPanel { background: #14191d; border: 0; }
            QLabel#previewCaption { background: #20272b; color: #d7dee1; padding: 7px 10px;
                                    font-weight: 650; }
            QFrame#metricBar { background: #ffffff; border: 1px solid #dce1e3; border-radius: 6px; }
            QFrame#metricDivider { color: #dce1e3; }
            QPlainTextEdit { background: #20272b; color: #d7dee1; border: 0; border-radius: 5px;
                             font-family: monospace; font-size: 11px; padding: 7px; }
            QScrollArea { background: #ffffff; border: 0; }
            QScrollArea > QWidget > QWidget { background: #ffffff; }
            QStatusBar { background: #eef1f2; color: #526068; border-top: 1px solid #d7dde0; }
            """
        )


def run_qt_ui(config_path, screenshot=None, initial_stage=0):
    plugin_path = Path(QtCore.__file__).resolve().parent / "Qt5" / "plugins"
    if plugin_path.is_dir():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_path)
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Object Model Builder")
    window = ModelBuilderWindow(config_path)
    window.controller.stage_list.setCurrentRow(int(initial_stage))
    window.show()
    if screenshot:
        destination = Path(screenshot).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        def save_and_exit():
            window.grab().save(str(destination))
            app.quit()

        QTimer.singleShot(700, save_and_exit)
    return app.exec_()
