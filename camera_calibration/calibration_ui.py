#!/usr/bin/env python3
"""PyQt5 workbench for Astra Pro calibration and hybrid localization."""

import argparse
import math
import sys
from pathlib import Path

from PyQt5.QtCore import QProcess, QRectF, QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import cv2
import numpy as np
import yaml

from calib_common import (
    COORDINATE_CONVENTION,
    INCOMPATIBLE_CONVENTION_MESSAGE,
    PRINTED_TAG_LAYOUT_MM,
    PRINTED_TAG_SIZE_MM,
    RosFrameSource,
    V4L2FrameSource,
    camera_matrix_from_yaml,
    charuco_board,
    detect_apriltags,
    draw_tag_detections,
    draw_tag_coordinate_axes,
    draw_workspace_coordinate_axes,
    layout_origins_mm,
    load_layout,
    require_coordinate_convention,
    save_camera_yaml,
)
from calibrate_intrinsics import calibrate, save_report
from calibrate_workspace import (
    build_workspace_output,
    calibration_correspondences,
    save_workspace_output,
    solve_workspace_pose,
)
from validate_workspace import (
    estimate_dynamic_validation,
    estimate_tag_corners_mm,
    save_validation_summary,
    summarize_estimates,
    tag_yaw_deg_from_corners,
)
from hybrid_localization import (
    HybridCameraLocalizer,
    ManualRobotPoseProvider,
    SOURCE_ROBOT_FALLBACK,
    SOURCE_SIMULATED_ROBOT,
    SOURCE_TAG_VISUAL,
    TagMapPoseEstimator,
    load_hybrid_config,
    matrix_from_config,
    transform_from_xyz_rpy,
    xyz_rpy_from_transform,
)


ROOT = Path(__file__).resolve().parent
TARGET_PDF = ROOT / "targets" / "calibration_targets_A4_2pages.pdf"
TARGET_PREVIEW = ROOT / "targets" / "charuco_7x10_preview_600dpi.png"
LAYOUT_PATH = ROOT / "config" / "tag_layout.yaml"
HYBRID_CONFIG_PATH = ROOT / "config" / "hybrid_localization.yaml"
RVIZ_CONFIG_PATH = ROOT / "config" / "calibration_rviz.rviz"
V4L2_DEVICE = (
    "/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0"
)
V4L2_MODES = (
    ("1280 × 720 · 清晰", 1280, 720, 30, "MJPG"),
    ("640 × 480 · 兼容", 640, 480, 30, "YUYV"),
)


def calibration_output_paths(width: int, height: int):
    suffix = "{}x{}".format(width, height)
    output = ROOT / "output"
    return (
        output / "astra_pro_rgb_{}.yaml".format(suffix),
        output / "workspace_extrinsics_{}.yaml".format(suffix),
        output / "validation_tag_103_{}.yaml".format(suffix),
    )


class VideoCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self._image = QImage()
        self._empty_text = "相机未连接"
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self._image = QImage(
            rgb.data, width, height, channels * width, QImage.Format_RGB888
        ).copy()
        self.update()

    def set_image(self, path: Path) -> None:
        self._image = QImage(str(path))
        self.update()

    def clear(self, text: str = "相机未连接") -> None:
        self._image = QImage()
        self._empty_text = text
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#14191d"))
        if self._image.isNull():
            painter.setPen(QColor("#aab3ba"))
            font = QFont()
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, self._empty_text)
            return
        target = QSize(self.width() - 24, self.height() - 24)
        scaled = self._image.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) / 2.0
        y = (self.height() - scaled.height()) / 2.0
        painter.drawImage(QRectF(x, y, scaled.width(), scaled.height()), scaled)


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)
    connected = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        mode: str,
        endpoint: str,
        width: int,
        height: int,
        fps: int,
        fourcc: str,
    ):
        super().__init__()
        self.mode = mode
        self.endpoint = endpoint
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self._stopping = False

    def stop(self) -> None:
        self._stopping = True

    def run(self) -> None:
        source = None
        try:
            if self.mode == "v4l2":
                source = V4L2FrameSource(
                    self.endpoint,
                    self.width,
                    self.height,
                    self.fps,
                    self.fourcc,
                )
            else:
                source = RosFrameSource(self.endpoint, "astra_pro_calibration_ui")
            self.connected.emit()
            while not self._stopping:
                frame = source.read(timeout_s=1.0)
                if not self._stopping:
                    self.frame_ready.emit(frame)
        except Exception as error:
            if not self._stopping:
                self.failed.emit(str(error))
        finally:
            if source is not None:
                source.close()


class MetricLabel(QWidget):
    def __init__(self, caption: str, value: str = "--"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.caption = QLabel(caption)
        self.caption.setObjectName("metricCaption")
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        layout.addWidget(self.caption)
        layout.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


def standard_icon(widget: QWidget, name: str) -> QIcon:
    return widget.style().standardIcon(getattr(QStyle, name))


def primary_button(parent: QWidget, text: str, icon_name: str) -> QPushButton:
    button = QPushButton(text, parent)
    button.setObjectName("primaryButton")
    button.setIcon(standard_icon(parent, icon_name))
    button.setMinimumHeight(36)
    return button


def secondary_button(parent: QWidget, text: str, icon_name: str) -> QPushButton:
    button = QPushButton(text, parent)
    button.setObjectName("secondaryButton")
    button.setIcon(standard_icon(parent, icon_name))
    button.setMinimumHeight(34)
    return button


class CalibrationWindow(QMainWindow):
    def __init__(self, initial_stage: int = 0, auto_connect: bool = False):
        super().__init__()
        self.setWindowTitle("Astra Pro 标定工作台")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 720)
        icon_path = Path.home() / "Applications" / "OrbbecViewer-1.10.27" / "res" / "orbbec_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.camera_worker = None
        self.camera_connected = False
        self.current_frame = None
        self.current_stage = 0
        self.frame_counter = 0

        self.board = charuco_board()
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        self.board_corners = np.asarray(self.board.getChessboardCorners(), dtype=np.float32)
        self.current_charuco = None
        self.intrinsic_object_points = []
        self.intrinsic_image_points = []
        self.intrinsic_all_pixels = []
        self.intrinsic_image_size = None

        self.current_tag_detections = {}
        self.workspace_collecting = False
        self.workspace_samples = {100: [], 101: [], 102: []}
        self.workspace_valid_frames = 0
        self.validation_running = False
        self.validation_estimates = []
        self.validation_reference_rms = []
        self.validation_runtime = None
        self.rviz_visualizer = None
        self.rviz_process = None
        self.roscore_process = None
        self.rviz_master_attempts = 0
        self.rviz_stopping = False

        self.hybrid_config = load_hybrid_config(str(HYBRID_CONFIG_PATH))
        simulation = self.hybrid_config["simulation"]
        simulated_pose = transform_from_xyz_rpy(
            np.asarray(simulation["base_from_gripper_xyz_mm"], dtype=np.float64)
            / 1000.0,
            simulation["base_from_gripper_rpy_deg"],
        )
        self.robot_pose_provider = ManualRobotPoseProvider(
            simulated_pose, simulated=True
        )
        self.robot_pose_provider.set_available(bool(simulation.get("enabled", True)))
        self.hybrid_localizer = None
        self.hybrid_last_source = None

        _, width, height, _, _ = V4L2_MODES[0]
        (
            self.intrinsics_path,
            self.extrinsics_path,
            self.validation_path,
        ) = calibration_output_paths(width, height)

        self.layout_data = self._read_layout()
        self._build_ui()
        self._apply_style()
        self._load_layout_fields()
        self._restore_stage_status()
        self.stage_list.setCurrentRow(initial_stage)
        self._log("标定素材与配置已载入")
        if auto_connect:
            QTimer.singleShot(0, self.toggle_camera)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
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

        self.statusBar().showMessage("就绪")
        self.statusBar().setSizeGripEnabled(False)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(66)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 20, 0)
        title = QLabel("Astra Pro 标定工作台")
        title.setObjectName("appTitle")
        subtitle = QLabel("RGB 标定 · Tag 视觉定位 · 机器人回退")
        subtitle.setObjectName("appSubtitle")
        title_group = QVBoxLayout()
        title_group.setSpacing(1)
        title_group.addWidget(title)
        title_group.addWidget(subtitle)
        layout.addLayout(title_group)
        layout.addStretch()

        self.connection_badge = QLabel("未连接")
        self.connection_badge.setObjectName("connectionBadge")
        self.connection_button = primary_button(self, "连接相机", "SP_MediaPlay")
        self.connection_button.clicked.connect(self.toggle_camera)
        output_button = QToolButton()
        output_button.setIcon(standard_icon(self, "SP_DirOpenIcon"))
        output_button.setToolTip("打开输出目录")
        output_button.setFixedSize(36, 36)
        output_button.clicked.connect(self.open_output_folder)
        layout.addWidget(self.connection_badge)
        layout.addSpacing(10)
        layout.addWidget(self.connection_button)
        layout.addWidget(output_button)
        return header

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(238)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 18)
        layout.setSpacing(12)

        label = QLabel("标定流程")
        label.setObjectName("sectionLabel")
        layout.addWidget(label)
        self.stage_list = QListWidget()
        self.stage_list.setObjectName("stageList")
        self.stage_list.setSpacing(4)
        self.stage_list.setFixedHeight(250)
        self.stage_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for text in (
            "01  打印标定板",
            "02  RGB 相机内参",
            "03  工作平面外参",
            "04  ID 103 验证",
            "05  混合定位",
        ):
            item = QListWidgetItem(text)
            item.setSizeHint(QSize(190, 44))
            self.stage_list.addItem(item)
        self.stage_list.currentRowChanged.connect(self.change_stage)
        layout.addWidget(self.stage_list)

        source_label = QLabel("图像源")
        source_label.setObjectName("sectionLabel")
        layout.addWidget(source_label)
        segment = QHBoxLayout()
        segment.setSpacing(0)
        self.v4l2_button = QToolButton()
        self.v4l2_button.setText("V4L2")
        self.v4l2_button.setCheckable(True)
        self.v4l2_button.setChecked(True)
        self.v4l2_button.setObjectName("sourceSegmentLeft")
        self.ros_button = QToolButton()
        self.ros_button.setText("ROS")
        self.ros_button.setCheckable(True)
        self.ros_button.setObjectName("sourceSegmentRight")
        self.source_group = QButtonGroup(self)
        self.source_group.setExclusive(True)
        self.source_group.addButton(self.v4l2_button)
        self.source_group.addButton(self.ros_button)
        self.v4l2_button.clicked.connect(self.update_source_endpoint)
        self.ros_button.clicked.connect(self.update_source_endpoint)
        segment.addWidget(self.v4l2_button)
        segment.addWidget(self.ros_button)
        layout.addLayout(segment)

        self.endpoint_label = QLabel("设备")
        self.endpoint_label.setObjectName("fieldLabel")
        self.endpoint_edit = QComboBox()
        self.endpoint_edit.setEditable(True)
        self.endpoint_edit.addItem(V4L2_DEVICE)
        self.endpoint_edit.setToolTip(V4L2_DEVICE)
        layout.addWidget(self.endpoint_label)
        layout.addWidget(self.endpoint_edit)

        self.resolution_label = QLabel("RGB 模式")
        self.resolution_label.setObjectName("fieldLabel")
        self.resolution_combo = QComboBox()
        for label, width, height, fps, fourcc in V4L2_MODES:
            self.resolution_combo.addItem(label, (width, height, fps, fourcc))
        self.resolution_combo.currentIndexChanged.connect(self.resolution_changed)
        layout.addWidget(self.resolution_label)
        layout.addWidget(self.resolution_combo)

        source_meta = QGridLayout()
        source_meta.setHorizontalSpacing(10)
        source_meta.setVerticalSpacing(4)
        source_meta.addWidget(QLabel("分辨率"), 0, 0)
        self.resolution_value = QLabel("1280 × 720")
        source_meta.addWidget(self.resolution_value, 0, 1)
        source_meta.addWidget(QLabel("格式"), 1, 0)
        self.format_value = QLabel("MJPG · 30 Hz")
        source_meta.addWidget(self.format_value, 1, 1)
        layout.addLayout(source_meta)
        layout.addStretch()

        self.sidebar_note = QLabel("当前坐标系：ruler_workspace")
        self.sidebar_note.setObjectName("sidebarNote")
        self.sidebar_note.setWordWrap(True)
        layout.addWidget(self.sidebar_note)
        return sidebar

    def _build_preview_area(self) -> QWidget:
        area = QWidget()
        area.setObjectName("previewArea")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        heading_row = QHBoxLayout()
        self.preview_title = QLabel("打印标定板")
        self.preview_title.setObjectName("viewTitle")
        self.preview_state = QLabel("素材已生成")
        self.preview_state.setObjectName("viewState")
        heading_row.addWidget(self.preview_title)
        heading_row.addStretch()
        heading_row.addWidget(self.preview_state)
        layout.addLayout(heading_row)
        self.video_canvas = VideoCanvas()
        layout.addWidget(self.video_canvas, 1)

        metric_bar = QFrame()
        metric_bar.setObjectName("metricBar")
        metric_layout = QHBoxLayout(metric_bar)
        metric_layout.setContentsMargins(18, 10, 18, 10)
        self.frame_metric = MetricLabel("图像", "1280 × 720")
        self.detection_metric = MetricLabel("检测", "待机")
        self.task_metric = MetricLabel("进度", "素材就绪")
        metric_layout.addWidget(self.frame_metric)
        metric_layout.addWidget(self._vertical_line())
        metric_layout.addWidget(self.detection_metric)
        metric_layout.addWidget(self._vertical_line())
        metric_layout.addWidget(self.task_metric)
        layout.addWidget(metric_bar)
        return area

    def _build_control_area(self) -> QWidget:
        area = QFrame()
        area.setObjectName("controlArea")
        area.setFixedWidth(440)
        layout = QVBoxLayout(area)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(10)
        self.control_stack = QStackedWidget()
        self.control_stack.addWidget(self._scroll_page(self._build_print_page(), 400))
        self.control_stack.addWidget(self._scroll_page(self._build_intrinsic_page(), 500))
        self.control_stack.addWidget(self._scroll_page(self._build_workspace_page(), 560))
        self.control_stack.addWidget(self._scroll_page(self._build_validation_page(), 650))
        self.control_stack.addWidget(self._scroll_page(self._build_hybrid_page(), 620))
        layout.addWidget(self.control_stack, 1)
        log_label = QLabel("运行日志")
        log_label.setObjectName("sectionLabel")
        layout.addWidget(log_label)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(200)
        self.log_output.setFixedHeight(132)
        layout.addWidget(self.log_output)
        return area

    def _build_print_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._page_title("打印标定板", "ISO A4 · 2 页"))

        info = QGroupBox("文件")
        info_layout = QFormLayout(info)
        info_layout.setLabelAlignment(Qt.AlignLeft)
        info_layout.addRow("第 1 页", QLabel("ChArUco 7 × 10"))
        info_layout.addRow("第 2 页", QLabel("AprilTag 100–103"))
        info_layout.addRow("Tag 黑框", QLabel("70.0 mm"))
        info_layout.addRow("方格", QLabel("24.0 mm"))
        layout.addWidget(info)

        self.open_pdf_button = primary_button(self, "打开打印 PDF", "SP_DialogOpenButton")
        self.open_pdf_button.clicked.connect(self.open_target_pdf)
        self.verify_button = secondary_button(self, "校验 PDF 尺寸", "SP_DialogApplyButton")
        self.verify_button.clicked.connect(self.verify_target_pdf)
        layout.addWidget(self.open_pdf_button)
        layout.addWidget(self.verify_button)

        self.print_verify_status = QLabel("已生成 · 等待打印")
        self.print_verify_status.setObjectName("resultBanner")
        self.print_verify_status.setWordWrap(True)
        layout.addWidget(self.print_verify_status)
        layout.addStretch()
        return page

    def _build_intrinsic_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._page_title("RGB 相机内参", "ChArUco 多视角"))

        metrics = QGridLayout()
        self.intrinsic_corner_value = self._value_label("0")
        self.intrinsic_view_value = self._value_label("0 / 20")
        metrics.addWidget(QLabel("当前角点"), 0, 0)
        metrics.addWidget(self.intrinsic_corner_value, 0, 1)
        metrics.addWidget(QLabel("已采视角"), 1, 0)
        metrics.addWidget(self.intrinsic_view_value, 1, 1)
        layout.addLayout(metrics)

        layout.addWidget(QLabel("画面覆盖 X"))
        self.coverage_x = QProgressBar()
        self.coverage_x.setRange(0, 100)
        layout.addWidget(self.coverage_x)
        layout.addWidget(QLabel("画面覆盖 Y"))
        self.coverage_y = QProgressBar()
        self.coverage_y.setRange(0, 100)
        layout.addWidget(self.coverage_y)

        self.capture_intrinsic_button = primary_button(self, "采集当前视角", "SP_DialogSaveButton")
        self.capture_intrinsic_button.clicked.connect(self.capture_intrinsic_view)
        self.solve_intrinsic_button = primary_button(self, "计算并保存内参", "SP_DialogApplyButton")
        self.solve_intrinsic_button.clicked.connect(self.solve_intrinsics)
        self.solve_intrinsic_button.setEnabled(False)
        self.clear_intrinsic_button = secondary_button(self, "清空采样", "SP_BrowserReload")
        self.clear_intrinsic_button.clicked.connect(self.clear_intrinsic_views)
        layout.addWidget(self.capture_intrinsic_button)
        layout.addWidget(self.solve_intrinsic_button)
        layout.addWidget(self.clear_intrinsic_button)

        self.intrinsic_result = QLabel("尚未计算")
        self.intrinsic_result.setObjectName("resultBanner")
        self.intrinsic_result.setWordWrap(True)
        layout.addWidget(self.intrinsic_result)
        layout.addStretch()
        return page

    def _build_workspace_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._page_title("工作平面外参", "Tag 100 · 101 · 102"))

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Tag 黑框边长"))
        size_row.addStretch()
        self.tag_size_spin = self._coordinate_spin(0.1, 500.0, 70.0)
        self.tag_size_spin.setSuffix(" mm")
        size_row.addWidget(self.tag_size_spin)
        layout.addLayout(size_row)

        self.tag_table = QTableWidget(3, 5)
        self.tag_table.setHorizontalHeaderLabels(("ID", "X", "Y", "Z", "Yaw"))
        self.tag_table.verticalHeader().setVisible(False)
        self.tag_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tag_table.setFixedHeight(150)
        self.tag_spins = {}
        for row, tag_id in enumerate((100, 101, 102)):
            item = QTableWidgetItem(str(tag_id))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignCenter)
            self.tag_table.setItem(row, 0, item)
            self.tag_spins[tag_id] = []
            for column in range(1, 5):
                spin = self._coordinate_spin(-10000.0, 10000.0, 0.0)
                self.tag_table.setCellWidget(row, column, spin)
                self.tag_spins[tag_id].append(spin)
        layout.addWidget(self.tag_table)

        distance_group = QGroupBox("三个左上角基准点距离 / mm")
        distance_layout = QGridLayout(distance_group)
        self.tag_distance_spins = {}
        for column, pair in enumerate(((100, 101), (100, 102), (101, 102))):
            spin = self._coordinate_spin(0.1, 20000.0, 100.0)
            self.tag_distance_spins[pair] = spin
            distance_layout.addWidget(QLabel("{}–{}".format(*pair)), 0, column)
            distance_layout.addWidget(spin, 1, column)
        self.calculate_layout_button = secondary_button(
            self, "按三条基准点距离计算坐标", "SP_DialogApplyButton"
        )
        self.calculate_layout_button.clicked.connect(
            self.calculate_layout_from_distances
        )
        distance_layout.addWidget(self.calculate_layout_button, 2, 0, 1, 3)
        self.printed_layout_button = secondary_button(
            self, "使用原始 A4 排布", "SP_BrowserReload"
        )
        self.printed_layout_button.clicked.connect(self.apply_printed_a4_layout)
        distance_layout.addWidget(self.printed_layout_button, 3, 0, 1, 3)
        layout.addWidget(distance_group)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("采样帧数"))
        self.workspace_sample_target = QSpinBox()
        self.workspace_sample_target.setRange(20, 300)
        self.workspace_sample_target.setValue(60)
        sample_row.addStretch()
        sample_row.addWidget(self.workspace_sample_target)
        layout.addLayout(sample_row)
        self.workspace_progress = QProgressBar()
        self.workspace_progress.setRange(0, 60)
        layout.addWidget(self.workspace_progress)
        self.workspace_visible = QLabel("可见 ID：--")
        layout.addWidget(self.workspace_visible)

        self.save_layout_button = secondary_button(self, "保存坐标", "SP_DialogSaveButton")
        self.save_layout_button.clicked.connect(self.save_layout_fields)
        self.workspace_capture_button = primary_button(self, "开始采集外参", "SP_MediaPlay")
        self.workspace_capture_button.clicked.connect(self.toggle_workspace_capture)
        layout.addWidget(self.save_layout_button)
        layout.addWidget(self.workspace_capture_button)

        self.workspace_result = QLabel("等待内参与三个标定 Tag")
        self.workspace_result.setObjectName("resultBanner")
        self.workspace_result.setWordWrap(True)
        layout.addWidget(self.workspace_result)
        layout.addStretch()
        return page

    def _build_validation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(11)
        layout.addWidget(
            self._page_title("移动相机位置验证", "ID 100–102 实时定位 · ID 103 独立验证")
        )

        position_group = QGroupBox("已知左上角坐标 / mm")
        position_layout = QGridLayout(position_group)
        self.validation_spins = []
        for index, axis in enumerate(("X", "Y", "Z")):
            spin = self._coordinate_spin(-10000.0, 10000.0, 0.0)
            self.validation_spins.append(spin)
            position_layout.addWidget(QLabel(axis), 0, index)
            position_layout.addWidget(spin, 1, index)
        layout.addWidget(position_group)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("采样帧数"))
        self.validation_sample_target = QSpinBox()
        self.validation_sample_target.setRange(20, 500)
        self.validation_sample_target.setValue(100)
        sample_row.addStretch()
        sample_row.addWidget(self.validation_sample_target)
        layout.addLayout(sample_row)
        self.validation_progress = QProgressBar()
        self.validation_progress.setRange(0, 100)
        layout.addWidget(self.validation_progress)

        values = QGridLayout()
        self.validation_xyz_value = self._value_label("--")
        self.validation_error_value = self._value_label("--")
        self.validation_yaw_value = self._value_label("--")
        values.addWidget(QLabel("实时 XYZ / mm"), 0, 0)
        values.addWidget(self.validation_xyz_value, 0, 1)
        values.addWidget(QLabel("实时 XY 误差"), 1, 0)
        values.addWidget(self.validation_error_value, 1, 1)
        values.addWidget(QLabel("实时 Yaw / deg"), 2, 0)
        values.addWidget(self.validation_yaw_value, 2, 1)
        layout.addLayout(values)

        camera_pose_group = QGroupBox("当前相机位姿 / ruler_workspace")
        camera_pose_layout = QGridLayout(camera_pose_group)
        self.validation_camera_xyz_value = self._value_label("--")
        self.validation_camera_rpy_value = self._value_label("--")
        camera_pose_layout.addWidget(QLabel("XYZ / mm"), 0, 0)
        camera_pose_layout.addWidget(self.validation_camera_xyz_value, 0, 1)
        camera_pose_layout.addWidget(QLabel("RPY / deg"), 1, 0)
        camera_pose_layout.addWidget(self.validation_camera_rpy_value, 1, 1)
        pose_note = QLabel("标定坐标 Z<0 表示纸面上方；RViz 自动转换为 Z-up")
        pose_note.setObjectName("panelSubtitle")
        pose_note.setWordWrap(True)
        camera_pose_layout.addWidget(pose_note, 2, 0, 1, 2)
        layout.addWidget(camera_pose_group)

        rviz_row = QHBoxLayout()
        self.rviz_button = secondary_button(self, "打开 RViz 位姿", "SP_ComputerIcon")
        self.rviz_button.clicked.connect(self.toggle_rviz_view)
        self.rviz_clear_path_button = QToolButton()
        self.rviz_clear_path_button.setIcon(standard_icon(self, "SP_BrowserReload"))
        self.rviz_clear_path_button.setToolTip("清空 RViz 相机轨迹")
        self.rviz_clear_path_button.setFixedSize(36, 36)
        self.rviz_clear_path_button.setEnabled(False)
        self.rviz_clear_path_button.clicked.connect(self.clear_rviz_path)
        rviz_row.addWidget(self.rviz_button, 1)
        rviz_row.addWidget(self.rviz_clear_path_button)
        layout.addLayout(rviz_row)
        self.rviz_status = QLabel("RViz 未启动")
        self.rviz_status.setObjectName("panelSubtitle")
        self.rviz_status.setWordWrap(True)
        layout.addWidget(self.rviz_status)

        self.validation_button = primary_button(self, "开始验证", "SP_MediaPlay")
        self.validation_button.clicked.connect(self.toggle_validation)
        self.validation_reset_button = secondary_button(self, "清空结果", "SP_BrowserReload")
        self.validation_reset_button.clicked.connect(self.reset_validation)
        layout.addWidget(self.validation_button)
        layout.addWidget(self.validation_reset_button)
        self.validation_result = QLabel("尚未运行")
        self.validation_result.setObjectName("resultBanner")
        self.validation_result.setWordWrap(True)
        layout.addWidget(self.validation_result)
        layout.addStretch()
        return page

    def _build_hybrid_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(11)
        layout.addWidget(
            self._page_title("混合定位", "Tag 优先 · 机器人位姿回退")
        )

        state_group = QGroupBox("统一输出 T_workspace_camera")
        state_layout = QGridLayout(state_group)
        self.hybrid_source_value = self._value_label("无有效位姿")
        self.hybrid_visible_value = self._value_label("--")
        self.hybrid_quality_value = self._value_label("--")
        for value_label in (
            self.hybrid_source_value,
            self.hybrid_visible_value,
            self.hybrid_quality_value,
        ):
            value_label.setMinimumWidth(210)
        state_layout.setColumnStretch(1, 1)
        state_layout.addWidget(QLabel("当前来源"), 0, 0)
        state_layout.addWidget(self.hybrid_source_value, 0, 1)
        state_layout.addWidget(QLabel("可见 Tag"), 1, 0)
        state_layout.addWidget(self.hybrid_visible_value, 1, 1)
        state_layout.addWidget(QLabel("视觉质量"), 2, 0)
        state_layout.addWidget(self.hybrid_quality_value, 2, 1)
        layout.addWidget(state_group)

        pose_group = QGroupBox("相机在工作坐标系中的位姿")
        pose_layout = QGridLayout(pose_group)
        self.hybrid_xyz_value = self._value_label("--")
        self.hybrid_rpy_value = self._value_label("--")
        self.hybrid_xyz_value.setMinimumWidth(210)
        self.hybrid_rpy_value.setMinimumWidth(210)
        pose_layout.setColumnStretch(1, 1)
        pose_layout.addWidget(QLabel("XYZ / mm"), 0, 0)
        pose_layout.addWidget(self.hybrid_xyz_value, 0, 1)
        pose_layout.addWidget(QLabel("RPY / deg"), 1, 0)
        pose_layout.addWidget(self.hybrid_rpy_value, 1, 1)
        layout.addWidget(pose_group)

        simulation_group = QGroupBox("无机械臂模拟器")
        simulation_layout = QGridLayout(simulation_group)
        self.hybrid_simulation_check = QCheckBox("启用模拟机器人回退")
        self.hybrid_simulation_check.setChecked(
            bool(self.hybrid_config["simulation"].get("enabled", True))
        )
        self.hybrid_simulation_check.toggled.connect(
            self._hybrid_simulation_toggled
        )
        self.hybrid_hide_tags_check = QCheckBox("模拟 Tag 丢失")
        self.hybrid_hide_tags_check.toggled.connect(
            self._refresh_hybrid_without_frame
        )
        simulation_layout.addWidget(self.hybrid_simulation_check, 0, 0, 1, 3)
        simulation_layout.addWidget(self.hybrid_hide_tags_check, 1, 0, 1, 3)

        simulation = self.hybrid_config["simulation"]
        xyz_mm = simulation["base_from_gripper_xyz_mm"]
        rpy_deg = simulation["base_from_gripper_rpy_deg"]
        self.hybrid_pose_spins = {}
        for column, (name, value) in enumerate(zip(("X", "Y", "Z"), xyz_mm)):
            spin = self._coordinate_spin(-5000.0, 5000.0, float(value))
            spin.setDecimals(1)
            spin.setSuffix(" mm")
            self.hybrid_pose_spins[name] = spin
            simulation_layout.addWidget(QLabel(name), 2, column)
            simulation_layout.addWidget(spin, 3, column)
        for column, (name, value) in enumerate(zip(("R", "P", "Yaw"), rpy_deg)):
            spin = self._coordinate_spin(-180.0, 180.0, float(value))
            spin.setDecimals(1)
            spin.setSuffix(" deg")
            self.hybrid_pose_spins[name] = spin
            simulation_layout.addWidget(QLabel(name), 4, column)
            simulation_layout.addWidget(spin, 5, column)
        self.hybrid_apply_pose_button = secondary_button(
            self, "更新模拟末端位姿", "SP_BrowserReload"
        )
        self.hybrid_apply_pose_button.clicked.connect(
            self.apply_simulated_robot_pose
        )
        simulation_layout.addWidget(self.hybrid_apply_pose_button, 6, 0, 1, 3)
        layout.addWidget(simulation_group)

        self.hybrid_result = QLabel(
            "模拟回退仅验证接口和矩阵方向，不可用于真实机械臂执行"
        )
        self.hybrid_result.setObjectName("resultBanner")
        self.hybrid_result.setWordWrap(True)
        layout.addWidget(self.hybrid_result)
        layout.addStretch()
        return page

    def _page_title(self, title: str, subtitle: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("panelTitle")
        detail = QLabel(subtitle)
        detail.setObjectName("panelSubtitle")
        layout.addWidget(heading)
        layout.addWidget(detail)
        return widget

    def _scroll_page(self, page: QWidget, minimum_height: int) -> QScrollArea:
        page.setMinimumHeight(minimum_height)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _vertical_line(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setObjectName("metricDivider")
        return line

    def _value_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("strongValue")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return label

    def _coordinate_spin(self, minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        return spin

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6f7; color: #20282d; font-size: 13px; }
            QFrame#header { background: #20272b; border: 0; }
            QLabel#appTitle { color: #ffffff; font-size: 18px; font-weight: 650; }
            QLabel#appSubtitle { color: #aeb8bd; font-size: 11px; }
            QLabel#connectionBadge { color: #d7dde0; background: #323b40; border: 1px solid #465158;
                                     padding: 6px 11px; border-radius: 4px; }
            QFrame#sidebar { background: #edf0f1; border-right: 1px solid #d6dcdf; }
            QFrame#controlArea { background: #ffffff; border-left: 1px solid #dce1e3; }
            QWidget#previewArea { background: #f4f6f7; }
            QLabel#sectionLabel, QLabel#fieldLabel { color: #647078; font-size: 11px; font-weight: 650; }
            QLabel#viewTitle { font-size: 17px; font-weight: 650; }
            QLabel#viewState { color: #0f766e; font-weight: 650; }
            QLabel#panelTitle { font-size: 18px; font-weight: 650; }
            QLabel#panelSubtitle { color: #6b767c; }
            QLabel#metricCaption { color: #707b81; font-size: 10px; }
            QLabel#metricValue { font-size: 14px; font-weight: 650; }
            QLabel#strongValue { font-weight: 650; color: #243139; }
            QLabel#sidebarNote { color: #66737a; background: #e2e7e9; padding: 10px; border-radius: 4px; }
            QLabel#resultBanner { background: #eef5f3; color: #285c55; border: 1px solid #c8ddd8;
                                  padding: 10px; border-radius: 5px; }
            QListWidget#stageList { background: transparent; border: 0; outline: 0; }
            QListWidget#stageList::item { padding: 8px 10px; border-radius: 5px; color: #435057; }
            QListWidget#stageList::item:selected { background: #ffffff; color: #0b645c; font-weight: 650;
                                                   border: 1px solid #d3ddda; }
            QToolButton#sourceSegmentLeft, QToolButton#sourceSegmentRight {
                min-height: 32px; background: #ffffff; border: 1px solid #cfd6d9; padding: 0 18px;
            }
            QToolButton#sourceSegmentLeft { border-top-left-radius: 5px; border-bottom-left-radius: 5px; }
            QToolButton#sourceSegmentRight { border-top-right-radius: 5px; border-bottom-right-radius: 5px;
                                             border-left: 0; }
            QToolButton#sourceSegmentLeft:checked, QToolButton#sourceSegmentRight:checked {
                background: #2f5f5b; color: white; border-color: #2f5f5b;
            }
            QPushButton#primaryButton { background: #176d64; color: #ffffff; border: 1px solid #176d64;
                                        border-radius: 5px; padding: 7px 14px; font-weight: 650; }
            QPushButton#primaryButton:hover { background: #115c54; }
            QPushButton#primaryButton:disabled { background: #b7c3c2; border-color: #b7c3c2; }
            QPushButton#secondaryButton { background: #ffffff; border: 1px solid #cbd3d6;
                                          border-radius: 5px; padding: 7px 14px; }
            QPushButton#secondaryButton:hover { background: #f0f3f4; }
            QToolButton { background: #ffffff; border: 1px solid #cbd3d6; border-radius: 5px; }
            QToolButton:hover { background: #f0f3f4; }
            QComboBox, QSpinBox, QDoubleSpinBox { background: #ffffff; border: 1px solid #cbd3d6;
                                                 border-radius: 4px; padding: 6px; min-height: 23px; }
            QGroupBox { border: 1px solid #d9dfe1; border-radius: 6px; margin-top: 12px; padding: 12px 8px 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; color: #58656c; }
            QProgressBar { background: #e5e9eb; border: 0; border-radius: 3px; height: 8px; text-align: center; }
            QProgressBar::chunk { background: #238176; border-radius: 3px; }
            QTableWidget { background: #ffffff; border: 1px solid #d7dde0; gridline-color: #e5e9eb; }
            QHeaderView::section { background: #eef1f2; border: 0; border-bottom: 1px solid #d7dde0;
                                   padding: 5px; font-weight: 650; }
            QFrame#metricBar { background: #ffffff; border: 1px solid #dce1e3; border-radius: 6px; }
            QFrame#metricDivider { color: #dce1e3; }
            QPlainTextEdit { background: #20272b; color: #d7dee1; border: 0; border-radius: 5px;
                             font-family: monospace; font-size: 11px; padding: 7px; }
            QScrollArea { background: #ffffff; border: 0; }
            QScrollArea > QWidget > QWidget { background: #ffffff; }
            QStatusBar { background: #eef1f2; color: #526068; border-top: 1px solid #d7dde0; }
            """
        )

    def _read_layout(self) -> dict:
        return load_layout(str(LAYOUT_PATH))

    def _load_layout_fields(self) -> None:
        self.tag_size_spin.setValue(float(self.layout_data.get("tag_size_mm", 70.0)))
        for tag_id, spins in self.tag_spins.items():
            entry = self.layout_data["calibration_tags"][tag_id]
            values = list(entry["origin_mm"]) + [entry.get("yaw_deg", 0.0)]
            for spin, value in zip(spins, values):
                spin.setValue(float(value))
        validation = self.layout_data["validation_tag"]
        for spin, value in zip(self.validation_spins, validation["origin_mm"]):
            spin.setValue(float(value))
        origins = {
            int(tag_id): np.asarray(entry["origin_mm"], dtype=np.float64)
            for tag_id, entry in self.layout_data["calibration_tags"].items()
        }
        for pair, spin in self.tag_distance_spins.items():
            spin.setValue(float(np.linalg.norm(origins[pair[1]] - origins[pair[0]])))

    def calculate_layout_from_distances(self) -> None:
        d01 = self.tag_distance_spins[(100, 101)].value()
        d02 = self.tag_distance_spins[(100, 102)].value()
        d12 = self.tag_distance_spins[(101, 102)].value()
        if d01 + d02 <= d12 or d01 + d12 <= d02 or d02 + d12 <= d01:
            self._show_error("基准点距离无效", "三个基准点距离不能组成三角形")
            return
        x102 = (d01 * d01 + d02 * d02 - d12 * d12) / (2.0 * d01)
        y_squared = d02 * d02 - x102 * x102
        if y_squared <= 0.0:
            self._show_error("基准点距离无效", "计算得到的三角形面积为零")
            return
        y102 = math.sqrt(y_squared)
        coordinates = {
            100: (0.0, 0.0, 0.0),
            101: (d01, 0.0, 0.0),
            102: (x102, y102, 0.0),
        }
        for tag_id, values in coordinates.items():
            for spin, value in zip(self.tag_spins[tag_id][:3], values):
                spin.setValue(value)
        self._log(
            "已按左上角基准点距离计算：100=(0,0)，101=({:.2f},0)，102=({:.2f},{:.2f}) mm".format(
                d01, x102, y102
            )
        )
        self.statusBar().showMessage("左上角坐标已计算，请确认 +X 向右、+Y 向下后保存", 5000)

    def apply_printed_a4_layout(self) -> None:
        scale = self.tag_size_spin.value() / PRINTED_TAG_SIZE_MM
        coordinates = {
            tag_id: np.asarray(PRINTED_TAG_LAYOUT_MM[tag_id], dtype=np.float64) * scale
            for tag_id in (100, 101, 102)
        }
        for tag_id, values in coordinates.items():
            for spin, value in zip(self.tag_spins[tag_id][:3], values):
                spin.setValue(float(value))
            self.tag_spins[tag_id][3].setValue(0.0)
        for pair, spin in self.tag_distance_spins.items():
            spin.setValue(
                float(np.linalg.norm(coordinates[pair[1]] - coordinates[pair[0]]))
            )
        self._log(
            "已按原始 A4 排布和 {:.2f} mm 黑框缩放："
            "101=({:.2f},0)，102=(0,{:.2f}) mm".format(
                self.tag_size_spin.value(),
                coordinates[101][0],
                coordinates[102][1],
            )
        )
        self.statusBar().showMessage("已应用原始 A4 排布，请保存坐标", 5000)

    def _layout_from_fields(self) -> dict:
        layout = dict(self.layout_data)
        layout["schema_version"] = 2
        layout["coordinate_convention"] = dict(COORDINATE_CONVENTION)
        layout["tag_size_mm"] = float(self.tag_size_spin.value())
        calibration_tags = {}
        origins = []
        for tag_id, spins in self.tag_spins.items():
            origin = [spins[index].value() for index in range(3)]
            origins.append(origin[:2])
            calibration_tags[tag_id] = {
                "origin_mm": origin,
                "yaw_deg": spins[3].value(),
            }
        if np.linalg.matrix_rank(np.asarray(origins[1:]) - np.asarray(origins[0])) < 2:
            raise ValueError("三个标定 Tag 的左上角基准点不能共线")
        layout["calibration_tags"] = calibration_tags
        validation = dict(layout["validation_tag"])
        validation["origin_mm"] = [spin.value() for spin in self.validation_spins]
        layout["validation_tag"] = validation
        return layout

    def save_layout_fields(self, quiet: bool = False) -> bool:
        try:
            layout = self._layout_from_fields()
            with open(LAYOUT_PATH, "w", encoding="utf-8") as handle:
                yaml.safe_dump(layout, handle, sort_keys=False, allow_unicode=True)
            self.layout_data = layout
            self.hybrid_localizer = None
            if not quiet:
                self._log("Tag 左上角基准坐标已保存")
                self.statusBar().showMessage("已保存 tag_layout.yaml", 4000)
            return True
        except Exception as error:
            self._show_error("坐标配置无效", str(error))
            return False

    def _restore_stage_status(self) -> None:
        extrinsics_current = False
        validation_current = False
        self._mark_stage(0, True)
        self._mark_stage(1, self.intrinsics_path.exists())
        self.intrinsic_result.setText("尚未计算")
        self.workspace_result.setText("等待内参与三个标定 Tag")
        self.validation_result.setText("尚未运行")
        if self.intrinsics_path.exists():
            try:
                _, _, _, data = camera_matrix_from_yaml(str(self.intrinsics_path))
                self.intrinsic_result.setText("已有内参文件 · {}".format(data.get("camera_name", "camera")))
            except Exception:
                pass
        if self.extrinsics_path.exists():
            try:
                result = self._load_current_extrinsics()
                extrinsics_current = True
                rms = result.get("quality", {}).get("rms_reprojection_error_px")
                if rms is not None:
                    self.workspace_result.setText("已有左上角基准外参 · RMS {:.3f} px".format(rms))
            except Exception:
                self.workspace_result.setText(INCOMPATIBLE_CONVENTION_MESSAGE)
        if self.validation_path.exists():
            try:
                with open(self.validation_path, "r", encoding="utf-8") as handle:
                    result = yaml.safe_load(handle)
                require_coordinate_convention(result, "validation result")
                if result.get("localization_mode") != "dynamic_camera_pose_from_reference_tags":
                    self.validation_result.setText(
                        "旧固定相机姿态验证结果已保留，请按动态参考模式重新验证"
                    )
                else:
                    validation_current = extrinsics_current
                    error = result.get("median_error_xy_mm")
                    if error is not None:
                        self.validation_result.setText(
                            "已有动态验证 · XY 中值误差 {:.2f} mm".format(error)
                        )
            except Exception:
                self.validation_result.setText("旧中心基准验证结果已保留，但不可用于当前约定")
        self._mark_stage(2, extrinsics_current)
        self._mark_stage(3, validation_current)

    def _load_current_extrinsics(self) -> dict:
        with open(self.extrinsics_path, "r", encoding="utf-8") as handle:
            extrinsics = yaml.safe_load(handle)
        require_coordinate_convention(extrinsics, "workspace extrinsics")
        return extrinsics

    def _mark_stage(self, index: int, complete: bool) -> None:
        item = self.stage_list.item(index)
        item.setIcon(
            standard_icon(self, "SP_DialogApplyButton") if complete else QIcon()
        )

    def update_source_endpoint(self) -> None:
        if self.v4l2_button.isChecked():
            self.endpoint_label.setText("设备")
            endpoint = V4L2_DEVICE
            self.resolution_combo.setEnabled(True)
        else:
            self.endpoint_label.setText("话题")
            endpoint = "/usb_cam/image_raw"
            self.resolution_combo.setEnabled(False)
        self.endpoint_edit.clear()
        self.endpoint_edit.addItem(endpoint)
        self.endpoint_edit.setToolTip(endpoint)

    def resolution_changed(self) -> None:
        width, height, fps, fourcc = self.resolution_combo.currentData()
        self.resolution_value.setText("{} × {}".format(width, height))
        self.format_value.setText("{} · {} Hz".format(fourcc, fps))
        self.frame_metric.set_value("{} × {}".format(width, height))
        (
            self.intrinsics_path,
            self.extrinsics_path,
            self.validation_path,
        ) = calibration_output_paths(width, height)
        self.clear_intrinsic_views()
        self.hybrid_localizer = None
        self._restore_stage_status()
        self._log("RGB 模式切换为 {}x{} {}".format(width, height, fourcc))

    def toggle_camera(self) -> None:
        if self.camera_worker is not None:
            self.disconnect_camera()
            return
        mode = "v4l2" if self.v4l2_button.isChecked() else "ros"
        endpoint = self.endpoint_edit.currentText().strip()
        if not endpoint:
            self._show_error("图像源无效", "请输入设备路径或 ROS 图像话题")
            return
        self.connection_button.setEnabled(False)
        self.connection_badge.setText("连接中")
        self._log("正在连接 {}：{}".format(mode.upper(), endpoint))
        width, height, fps, fourcc = self.resolution_combo.currentData()
        self.camera_worker = CameraWorker(
            mode, endpoint, width, height, fps, fourcc
        )
        self.camera_worker.frame_ready.connect(self.handle_frame)
        self.camera_worker.connected.connect(self.camera_did_connect)
        self.camera_worker.failed.connect(self.camera_failed)
        self.camera_worker.finished.connect(self.camera_finished)
        self.camera_worker.start()

    def camera_did_connect(self) -> None:
        self.camera_connected = True
        self.connection_badge.setText("已连接")
        self.connection_button.setText("断开相机")
        self.connection_button.setIcon(standard_icon(self, "SP_MediaStop"))
        self.connection_button.setEnabled(True)
        self.v4l2_button.setEnabled(False)
        self.ros_button.setEnabled(False)
        self.endpoint_edit.setEnabled(False)
        self.resolution_combo.setEnabled(False)
        self.preview_state.setText("实时")
        self._log("相机连接成功")

    def camera_failed(self, message: str) -> None:
        self._log("相机连接失败：{}".format(message))
        self.connection_badge.setText("连接失败")
        self.connection_button.setEnabled(True)
        self._show_error(
            "无法连接相机",
            message + "\n\n若选择 V4L2，请先关闭正在占用彩色设备的 usb_cam 或 Viewer。",
        )

    def camera_finished(self) -> None:
        worker = self.camera_worker
        self.camera_worker = None
        if worker is not None:
            worker.deleteLater()
        self.camera_connected = False
        self.connection_badge.setText("未连接")
        self.connection_button.setText("连接相机")
        self.connection_button.setIcon(standard_icon(self, "SP_MediaPlay"))
        self.connection_button.setEnabled(True)
        self.v4l2_button.setEnabled(True)
        self.ros_button.setEnabled(True)
        self.endpoint_edit.setEnabled(True)
        self.resolution_combo.setEnabled(self.v4l2_button.isChecked())
        self.preview_state.setText("待机")
        if self.current_stage != 0:
            self.video_canvas.clear()

    def disconnect_camera(self) -> None:
        if self.camera_worker is None:
            return
        self._log("正在断开相机")
        self.connection_button.setEnabled(False)
        self.camera_worker.stop()

    def handle_frame(self, frame: np.ndarray) -> None:
        self.current_frame = frame
        self.frame_counter += 1
        self.frame_metric.set_value("{} × {}".format(frame.shape[1], frame.shape[0]))
        if self.current_stage == 0:
            return
        if self.current_stage == 1:
            preview = self._process_intrinsic_frame(frame)
        elif self.current_stage == 2:
            preview = self._process_workspace_frame(frame)
        elif self.current_stage == 3:
            preview = self._process_validation_frame(frame)
        else:
            preview = self._process_hybrid_frame(frame)
        self.video_canvas.set_frame(preview)

    def _process_intrinsic_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
        self.current_charuco = (charuco_corners, charuco_ids)
        preview = frame.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)
        if charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids)
        count = 0 if charuco_ids is None else len(charuco_ids)
        self.intrinsic_corner_value.setText(str(count))
        self.detection_metric.set_value("{} 个角点".format(count))
        return preview

    def capture_intrinsic_view(self) -> None:
        if not self._require_camera():
            return
        corners, ids = self.current_charuco if self.current_charuco is not None else (None, None)
        count = 0 if ids is None else len(ids)
        if count < 12:
            self._show_error("角点不足", "当前只检测到 {} 个角点，至少需要 12 个".format(count))
            return
        image_size = (self.current_frame.shape[1], self.current_frame.shape[0])
        if self.intrinsic_image_size is None:
            self.intrinsic_image_size = image_size
        elif image_size != self.intrinsic_image_size:
            self._show_error("分辨率变化", "采样期间相机分辨率不能改变")
            return
        id_values = ids.reshape(-1).astype(np.int32)
        self.intrinsic_object_points.append(
            self.board_corners[id_values].reshape(-1, 1, 3).copy()
        )
        pixels = corners.astype(np.float32).reshape(-1, 1, 2).copy()
        self.intrinsic_image_points.append(pixels)
        self.intrinsic_all_pixels.append(pixels.reshape(-1, 2))
        count_views = len(self.intrinsic_object_points)
        self.intrinsic_view_value.setText("{} / 20".format(count_views))
        self.solve_intrinsic_button.setEnabled(count_views >= 20)
        self._update_intrinsic_coverage()
        self.task_metric.set_value("内参 {}/20".format(count_views))
        self._log("已采集内参视角 {}，角点 {}".format(count_views, count))

    def _update_intrinsic_coverage(self) -> None:
        if not self.intrinsic_all_pixels or self.intrinsic_image_size is None:
            self.coverage_x.setValue(0)
            self.coverage_y.setValue(0)
            return
        pixels = np.concatenate(self.intrinsic_all_pixels, axis=0)
        width, height = self.intrinsic_image_size
        x_coverage = int(round((pixels[:, 0].max() - pixels[:, 0].min()) / width * 100))
        y_coverage = int(round((pixels[:, 1].max() - pixels[:, 1].min()) / height * 100))
        self.coverage_x.setValue(min(100, x_coverage))
        self.coverage_y.setValue(min(100, y_coverage))

    def clear_intrinsic_views(self) -> None:
        self.intrinsic_object_points.clear()
        self.intrinsic_image_points.clear()
        self.intrinsic_all_pixels.clear()
        self.intrinsic_image_size = None
        self.intrinsic_view_value.setText("0 / 20")
        self.solve_intrinsic_button.setEnabled(False)
        self._update_intrinsic_coverage()
        self._log("内参采样已清空")

    def solve_intrinsics(self) -> None:
        if len(self.intrinsic_object_points) < 20:
            self._show_error("采样不足", "至少采集 20 个不同视角")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = calibrate(
                self.intrinsic_object_points,
                self.intrinsic_image_points,
                self.intrinsic_image_size,
            )
            pixels = np.concatenate(self.intrinsic_all_pixels, axis=0)
            width, height = self.intrinsic_image_size
            coverage = (
                float((pixels[:, 0].max() - pixels[:, 0].min()) / width),
                float((pixels[:, 1].max() - pixels[:, 1].min()) / height),
            )
            save_camera_yaml(
                str(self.intrinsics_path),
                result["camera_matrix"],
                result["distortion"],
                self.intrinsic_image_size,
                "astra_pro_rgb",
            )
            save_report(
                self.intrinsics_path.with_name(
                    self.intrinsics_path.stem + "_report.yaml"
                ),
                result,
                len(self.intrinsic_object_points),
                coverage,
            )
            rms = result["rms"]
            self.intrinsic_result.setText(
                "RMS {:.4f} px · 覆盖 X {:.0%} / Y {:.0%}".format(rms, *coverage)
            )
            self._mark_stage(1, rms <= 0.8)
            self.task_metric.set_value("内参 RMS {:.3f}px".format(rms))
            self._log("内参已保存，RMS {:.4f} px".format(rms))
            if rms > 0.8:
                QMessageBox.warning(self, "内参质量告警", "RMS 超过 0.8 px，建议补充不同姿态后重算")
        except Exception as error:
            self._show_error("内参计算失败", str(error))
        finally:
            QApplication.restoreOverrideCursor()

    def _process_workspace_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = detect_apriltags(frame)
        self.current_tag_detections = detections
        preview = frame.copy()
        draw_tag_detections(preview, detections)
        origins_mm = {
            tag_id: np.asarray([spin.value() for spin in spins[:3]], dtype=np.float64)
            for tag_id, spins in self.tag_spins.items()
        }
        camera_matrix, distortion = self._hybrid_intrinsics()
        draw_tag_coordinate_axes(
            preview,
            detections,
            origins_mm,
            camera_matrix,
            distortion,
            self.tag_size_spin.value(),
        )
        visible = [tag_id for tag_id in (100, 101, 102) if tag_id in detections]
        self.workspace_visible.setText("可见 ID：{}".format(", ".join(map(str, visible)) or "--"))
        self.detection_metric.set_value("{}/3 个标定 Tag".format(len(visible)))
        if self.workspace_collecting and len(visible) == 3:
            for tag_id in visible:
                self.workspace_samples[tag_id].append(detections[tag_id].copy())
            self.workspace_valid_frames += 1
            self.workspace_progress.setValue(self.workspace_valid_frames)
            self.task_metric.set_value(
                "外参 {}/{}".format(self.workspace_valid_frames, self.workspace_sample_target.value())
            )
            if self.workspace_valid_frames >= self.workspace_sample_target.value():
                self.workspace_collecting = False
                self.workspace_capture_button.setEnabled(False)
                QTimer.singleShot(0, self.solve_workspace)
        return preview

    def toggle_workspace_capture(self) -> None:
        if self.workspace_collecting:
            self.workspace_collecting = False
            self.workspace_capture_button.setText("继续采集外参")
            self.workspace_capture_button.setIcon(standard_icon(self, "SP_MediaPlay"))
            self._log("外参采集已暂停")
            return
        if not self._require_camera() or not self.intrinsics_path.exists():
            if not self.intrinsics_path.exists():
                self._show_error("缺少内参", "请先完成 RGB 相机内参标定")
            return
        if not self.save_layout_fields(quiet=True):
            return
        target = self.workspace_sample_target.value()
        if self.workspace_valid_frames == 0 or self.workspace_valid_frames >= target:
            self.workspace_samples = {100: [], 101: [], 102: []}
            self.workspace_valid_frames = 0
        self.workspace_progress.setRange(0, target)
        self.workspace_progress.setValue(self.workspace_valid_frames)
        self.workspace_collecting = True
        self.workspace_capture_button.setText("暂停采集")
        self.workspace_capture_button.setIcon(standard_icon(self, "SP_MediaPause"))
        self._log("开始采集工作平面外参")

    def solve_workspace(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            camera_matrix, distortion, image_size, _ = camera_matrix_from_yaml(
                str(self.intrinsics_path)
            )
            if self.current_frame is None or (
                self.current_frame.shape[1], self.current_frame.shape[0]
            ) != image_size:
                raise ValueError("当前图像分辨率与内参文件不一致")
            layout = self._layout_from_fields()
            median_corners = {
                tag_id: np.median(np.asarray(tag_samples), axis=0)
                for tag_id, tag_samples in self.workspace_samples.items()
            }
            object_points, image_points, point_tag_ids = calibration_correspondences(
                layout, median_corners
            )
            camera_from_workspace, workspace_from_camera, errors = solve_workspace_pose(
                object_points, image_points, camera_matrix, distortion
            )
            rms = float(np.sqrt(np.mean(errors ** 2)))
            if rms > 2.0:
                per_tag = {
                    int(tag_id): float(
                        np.sqrt(np.mean(errors[point_tag_ids == tag_id] ** 2))
                    )
                    for tag_id in (100, 101, 102)
                }
                raise ValueError(
                    "RMS {:.3f} px 超过 2.0 px；分 Tag RMS："
                    "100={:.2f}, 101={:.2f}, 102={:.2f} px。"
                    "请检查黑框边长、左上角基准点距离、画面中的 +X/+Y、Yaw 和纸面平整度".format(
                        rms, per_tag[100], per_tag[101], per_tag[102]
                    )
                )
            output = build_workspace_output(
                layout,
                camera_from_workspace,
                workspace_from_camera,
                errors,
                point_tag_ids,
                self.workspace_valid_frames,
                self.intrinsics_path,
                LAYOUT_PATH,
            )
            save_workspace_output(self.extrinsics_path, output)
            debug = self.current_frame.copy()
            draw_tag_detections(debug, median_corners)
            draw_tag_coordinate_axes(
                debug,
                median_corners,
                layout_origins_mm(layout),
                camera_matrix,
                distortion,
                float(layout["tag_size_mm"]),
            )
            self._draw_axes(
                debug, camera_from_workspace, camera_matrix, distortion, length_m=0.1
            )
            cv2.imwrite(str(self.extrinsics_path.with_suffix(".png")), debug)
            camera_mm = workspace_from_camera[:3, 3] * 1000.0
            self.workspace_result.setText(
                "RMS {:.4f} px · 相机原点 [{:.1f}, {:.1f}, {:.1f}] mm".format(
                    rms, camera_mm[0], camera_mm[1], camera_mm[2]
                )
            )
            self._mark_stage(2, True)
            self.task_metric.set_value("外参 RMS {:.3f}px".format(rms))
            self._log("工作平面外参已保存，RMS {:.4f} px".format(rms))
        except Exception as error:
            self.workspace_result.setText("外参失败 · {}".format(error))
            self._show_error("外参计算失败", str(error))
        finally:
            self.workspace_capture_button.setEnabled(True)
            self.workspace_capture_button.setText("重新采集外参")
            self.workspace_capture_button.setIcon(standard_icon(self, "SP_MediaPlay"))
            QApplication.restoreOverrideCursor()

    def _draw_axes(self, image, camera_from_workspace, camera_matrix, distortion, length_m):
        draw_workspace_coordinate_axes(
            image,
            camera_from_workspace,
            camera_matrix,
            distortion,
            length_m,
        )

    def _process_validation_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = detect_apriltags(frame)
        self.current_tag_detections = detections
        preview = frame.copy()
        draw_tag_detections(preview, detections)
        display_origins = layout_origins_mm(self.layout_data, include_validation=False)
        camera_matrix, distortion = self._hybrid_intrinsics()
        reference_ids = tuple(sorted(int(value) for value in self.layout_data["calibration_tags"]))
        visible_references = tuple(tag_id for tag_id in reference_ids if tag_id in detections)
        self.detection_metric.set_value(
            "参考 {}/{} · ID103 {}".format(
                len(visible_references),
                len(reference_ids),
                "可见" if 103 in detections else "不可见",
            )
        )
        if self.validation_runtime is not None:
            runtime = self.validation_runtime
            camera_matrix = runtime["camera_matrix"]
            distortion = runtime["distortion"]
            visual_pose, estimate_mm = estimate_dynamic_validation(
                detections,
                runtime["pose_estimator"],
                103,
                camera_matrix,
                distortion,
                runtime["plane_z_m"],
            )
            measured_corners_mm = None
            if estimate_mm is not None:
                measured_corners_mm = estimate_tag_corners_mm(
                    detections[103],
                    camera_matrix,
                    distortion,
                    visual_pose.camera_from_workspace,
                    runtime["plane_z_m"],
                )
                self.validation_yaw_value.setText(
                    "{:.2f}".format(tag_yaw_deg_from_corners(measured_corners_mm))
                )
            else:
                self.validation_yaw_value.setText("--")
            if visual_pose.valid:
                self._draw_axes(
                    preview,
                    visual_pose.camera_from_workspace,
                    camera_matrix,
                    distortion,
                    length_m=0.1,
                )
                camera_xyz_m, camera_rpy_deg = xyz_rpy_from_transform(
                    visual_pose.workspace_from_camera
                )
                self.validation_camera_xyz_value.setText(
                    "{:.1f}, {:.1f}, {:.1f}".format(*(camera_xyz_m * 1000.0))
                )
                self.validation_camera_rpy_value.setText(
                    "{:.1f}, {:.1f}, {:.1f}".format(*camera_rpy_deg)
                )
                self._publish_rviz_pose(
                    visual_pose.workspace_from_camera,
                    estimate_mm,
                    runtime["expected_mm"],
                    measured_corners_mm,
                )
            else:
                self.validation_camera_xyz_value.setText("--")
                self.validation_camera_rpy_value.setText("--")
                self.validation_yaw_value.setText("--")
                self._hide_rviz_validation_measurement()
            if estimate_mm is not None:
                display_origins[103] = estimate_mm
                expected_mm = runtime["expected_mm"]
                xy_error = float(np.linalg.norm(estimate_mm[:2] - expected_mm[:2]))
                self.validation_xyz_value.setText(
                    "{:.1f}, {:.1f}, {:.1f}".format(*estimate_mm)
                )
                self.validation_error_value.setText(
                    "{:.2f} mm · Ref {:.2f} px".format(
                        xy_error, visual_pose.rms_reprojection_error_px
                    )
                )
                if self.validation_running:
                    self.validation_estimates.append(estimate_mm)
                    self.validation_reference_rms.append(
                        visual_pose.rms_reprojection_error_px
                    )
                    self.validation_progress.setValue(len(self.validation_estimates))
                    self.task_metric.set_value(
                        "动态验证 {}/{}".format(
                            len(self.validation_estimates),
                            self.validation_sample_target.value(),
                        )
                    )
                    if len(self.validation_estimates) >= self.validation_sample_target.value():
                        self.validation_running = False
                        QTimer.singleShot(0, self.finish_validation)
            elif self.validation_running:
                self.task_metric.set_value(
                    "等待参考 100/101/102 与 ID103 同时可见"
                )
        draw_tag_coordinate_axes(
            preview,
            detections,
            display_origins,
            camera_matrix,
            distortion,
            float(self.layout_data["tag_size_mm"]),
        )
        return preview

    def _prepare_validation_runtime(self) -> dict:
        camera_matrix, distortion, image_size, _ = camera_matrix_from_yaml(
            str(self.intrinsics_path)
        )
        if self.current_frame is None:
            raise ValueError("当前没有相机画面")
        if (self.current_frame.shape[1], self.current_frame.shape[0]) != image_size:
            raise ValueError("当前图像分辨率与内参文件不一致")
        self._load_current_extrinsics()
        layout = self._layout_from_fields()
        expected_mm = np.asarray(
            layout["validation_tag"]["origin_mm"], dtype=np.float64
        )
        reference_ids = tuple(
            sorted(int(value) for value in layout["calibration_tags"])
        )
        return {
            "camera_matrix": camera_matrix,
            "distortion": distortion,
            "pose_estimator": TagMapPoseEstimator(
                layout,
                minimum_tags=len(reference_ids),
                max_rms_reprojection_error_px=2.0,
            ),
            "plane_z_m": float(expected_mm[2]) / 1000.0,
            "expected_mm": expected_mm,
            "reference_ids": reference_ids,
        }

    def toggle_validation(self) -> None:
        if self.validation_running:
            self.validation_running = False
            self.validation_button.setText("继续验证")
            self.validation_button.setIcon(standard_icon(self, "SP_MediaPlay"))
            self._log("验证采集已暂停")
            return
        if not self._require_camera():
            return
        if not self.intrinsics_path.exists() or not self.extrinsics_path.exists():
            self._show_error("标定文件不完整", "请先完成 RGB 内参和工作平面外参")
            return
        if not self.save_layout_fields(quiet=True):
            return
        try:
            self.validation_runtime = self._prepare_validation_runtime()
            self.validation_estimates = []
            self.validation_reference_rms = []
            target = self.validation_sample_target.value()
            self.validation_progress.setRange(0, target)
            self.validation_progress.setValue(0)
            self.validation_running = True
            self.validation_button.setText("暂停验证")
            self.validation_button.setIcon(standard_icon(self, "SP_MediaPause"))
            self._log("开始动态验证：100–102 实时定位相机，ID103 仅作为验证点")
        except Exception as error:
            self._show_error("无法开始验证", str(error))

    def finish_validation(self) -> None:
        try:
            expected_mm = self.validation_runtime["expected_mm"]
            summary = summarize_estimates(self.validation_estimates, expected_mm, 103)
            summary["localization_mode"] = "dynamic_camera_pose_from_reference_tags"
            summary["reference_tag_ids"] = list(
                self.validation_runtime["reference_ids"]
            )
            summary["reference_pose_rms_mean_px"] = float(
                np.mean(self.validation_reference_rms)
            )
            summary["reference_pose_rms_max_px"] = float(
                np.max(self.validation_reference_rms)
            )
            save_validation_summary(self.validation_path, summary)
            self.validation_result.setText(
                "XY 中值误差 {:.2f} mm · XY 抖动 [{:.2f}, {:.2f}] mm".format(
                    summary["median_error_xy_mm"],
                    summary["standard_deviation_mm"][0],
                    summary["standard_deviation_mm"][1],
                )
            )
            self._mark_stage(3, True)
            self.task_metric.set_value("验证误差 {:.2f}mm".format(summary["median_error_xy_mm"]))
            self._log("验证完成，XY 中值误差 {:.2f} mm".format(summary["median_error_xy_mm"]))
        except Exception as error:
            self._show_error("验证结果保存失败", str(error))
        finally:
            self.validation_button.setText("重新验证")
            self.validation_button.setIcon(standard_icon(self, "SP_MediaPlay"))

    def reset_validation(self) -> None:
        self.validation_running = False
        self.validation_estimates = []
        self.validation_reference_rms = []
        self.validation_progress.setValue(0)
        self.validation_xyz_value.setText("--")
        self.validation_error_value.setText("--")
        self.validation_yaw_value.setText("--")
        self.validation_result.setText("尚未运行")
        self.validation_button.setText("开始验证")
        self.validation_button.setIcon(standard_icon(self, "SP_MediaPlay"))
        self._log("验证结果已清空")

    def toggle_rviz_view(self) -> None:
        if self.rviz_visualizer is not None or (
            self.rviz_process is not None
            and self.rviz_process.state() != QProcess.NotRunning
        ):
            self.stop_rviz_view()
            return
        if not self._require_camera():
            return
        if not self.intrinsics_path.exists() or not self.extrinsics_path.exists():
            self._show_error("标定文件不完整", "请先完成 RGB 内参和工作平面外参")
            return
        if not self.save_layout_fields(quiet=True):
            return
        try:
            self.validation_runtime = self._prepare_validation_runtime()
        except Exception as error:
            self._show_error("无法准备 RViz 位姿", str(error))
            return

        self.rviz_button.setEnabled(False)
        self.rviz_status.setText("正在连接 ROS Master…")
        if self._ros_master_online():
            self._start_rviz_session()
            return

        self.roscore_process = QProcess(self)
        self.roscore_process.setProgram("/opt/ros/noetic/bin/roscore")
        self.roscore_process.start()
        if not self.roscore_process.waitForStarted(2000):
            self.rviz_button.setEnabled(True)
            self.rviz_status.setText("roscore 启动失败")
            self._show_error("无法启动 ROS Master", self.roscore_process.errorString())
            return
        self.rviz_master_attempts = 0
        self.rviz_status.setText("正在启动 roscore…")
        QTimer.singleShot(300, self._wait_for_ros_master)

    def _ros_master_online(self) -> bool:
        try:
            import rosgraph

            return bool(rosgraph.is_master_online())
        except Exception:
            return False

    def _wait_for_ros_master(self) -> None:
        if self._ros_master_online():
            self._start_rviz_session()
            return
        self.rviz_master_attempts += 1
        if self.rviz_master_attempts >= 20:
            self.rviz_button.setEnabled(True)
            self.rviz_status.setText("ROS Master 启动超时")
            self._stop_owned_roscore()
            return
        QTimer.singleShot(300, self._wait_for_ros_master)

    def _start_rviz_session(self) -> None:
        try:
            from rviz_visualization import RosPoseVisualizer

            self.rviz_visualizer = RosPoseVisualizer(
                self.layout_data.get("workspace_frame", "ruler_workspace"),
                self.layout_data.get("camera_frame", "camera_color_optical_frame"),
            )
            self.rviz_visualizer.publish_scene(self.layout_data)
            self.rviz_process = QProcess(self)
            self.rviz_process.finished.connect(self._rviz_finished)
            self.rviz_process.setProgram("/opt/ros/noetic/bin/rviz")
            self.rviz_process.setArguments(["-d", str(RVIZ_CONFIG_PATH)])
            self.rviz_process.start()
            if not self.rviz_process.waitForStarted(2500):
                raise RuntimeError(self.rviz_process.errorString())
            self.rviz_button.setText("关闭 RViz 位姿")
            self.rviz_button.setIcon(standard_icon(self, "SP_DialogCloseButton"))
            self.rviz_button.setEnabled(True)
            self.rviz_clear_path_button.setEnabled(True)
            self.rviz_status.setText("RViz 已启动 · 等待参考 Tag 位姿")
            self._log("RViz 位姿视图已启动")
        except Exception as error:
            if self.rviz_process is not None:
                self.rviz_process.deleteLater()
                self.rviz_process = None
            self.rviz_visualizer = None
            self.rviz_button.setEnabled(True)
            self.rviz_status.setText("RViz 启动失败")
            self._stop_owned_roscore()
            self._show_error("无法启动 RViz", str(error))

    def _publish_rviz_pose(
        self,
        workspace_from_camera,
        validation_origin_mm=None,
        expected_origin_mm=None,
        validation_corners_mm=None,
    ) -> None:
        if self.rviz_visualizer is None:
            return
        try:
            self.rviz_visualizer.publish_pose(
                workspace_from_camera,
                validation_origin_mm,
                expected_origin_mm,
                validation_corners_mm,
            )
            if validation_origin_mm is None:
                self.rviz_status.setText("相机 TF 有效 · 等待 ID103 实时测量")
            else:
                self.rviz_status.setText("相机 TF 与 ID103 实时测量均有效")
        except Exception as error:
            self.rviz_status.setText("RViz 位姿发布失败：{}".format(error))

    def _hide_rviz_validation_measurement(self) -> None:
        if self.rviz_visualizer is None:
            return
        try:
            self.rviz_visualizer.hide_validation_measurement()
            self.rviz_status.setText("参考 Tag 不完整 · 已隐藏 ID103 实时位置")
        except Exception as error:
            self.rviz_status.setText("RViz 标记清理失败：{}".format(error))

    def clear_rviz_path(self) -> None:
        if self.rviz_visualizer is None:
            return
        self.rviz_visualizer.clear_path()
        self.rviz_status.setText("相机轨迹已清空")
        self._log("RViz 相机轨迹已清空")

    def _rviz_finished(self, exit_code, _exit_status) -> None:
        process = self.rviz_process
        self.rviz_process = None
        self.rviz_visualizer = None
        if process is not None:
            process.deleteLater()
        self.rviz_button.setText("打开 RViz 位姿")
        self.rviz_button.setIcon(standard_icon(self, "SP_ComputerIcon"))
        self.rviz_button.setEnabled(True)
        self.rviz_clear_path_button.setEnabled(False)
        if not self.rviz_stopping:
            self.rviz_status.setText("RViz 已关闭（退出码 {}）".format(exit_code))
            self._log("RViz 位姿视图已关闭")

    def stop_rviz_view(self) -> None:
        self.rviz_stopping = True
        if self.rviz_process is not None:
            process = self.rviz_process
            if process.state() != QProcess.NotRunning:
                process.terminate()
                if not process.waitForFinished(2000):
                    process.kill()
                    process.waitForFinished(1000)
            if self.rviz_process is process:
                process.deleteLater()
                self.rviz_process = None
        self.rviz_visualizer = None
        self.rviz_button.setText("打开 RViz 位姿")
        self.rviz_button.setIcon(standard_icon(self, "SP_ComputerIcon"))
        self.rviz_button.setEnabled(True)
        self.rviz_clear_path_button.setEnabled(False)
        self.rviz_status.setText("RViz 未启动")
        self.rviz_stopping = False

    def _stop_owned_roscore(self) -> None:
        if self.roscore_process is None:
            return
        process = self.roscore_process
        self.roscore_process = None
        if process.state() != QProcess.NotRunning:
            process.terminate()
            if not process.waitForFinished(2000):
                process.kill()
                process.waitForFinished(1000)
        process.deleteLater()

    def _configure_hybrid_localizer(self) -> None:
        visual = self.hybrid_config["visual"]
        fallback = self.hybrid_config["fallback"]
        tag_estimator = TagMapPoseEstimator(
            self.layout_data,
            minimum_tags=int(visual.get("minimum_tags", 1)),
            max_rms_reprojection_error_px=float(
                visual.get("max_rms_reprojection_error_px", 2.5)
            ),
        )
        self.hybrid_localizer = HybridCameraLocalizer(
            tag_estimator,
            self.robot_pose_provider,
            matrix_from_config(
                fallback["workspace_from_base"], "workspace_from_base"
            ),
            matrix_from_config(
                fallback["gripper_from_camera"], "gripper_from_camera"
            ),
            hand_eye_calibrated=bool(
                fallback.get("hand_eye_calibrated", False)
            ),
            maximum_robot_pose_age_s=float(
                fallback.get("maximum_robot_pose_age_s", 0.25)
            ),
        )

    def _hybrid_intrinsics(self):
        if not self.intrinsics_path.exists():
            return None, None
        camera_matrix, distortion, _, _ = camera_matrix_from_yaml(
            str(self.intrinsics_path)
        )
        return camera_matrix, distortion

    def _evaluate_hybrid(self, detections):
        if self.hybrid_localizer is None:
            self._configure_hybrid_localizer()
        camera_matrix, distortion = self._hybrid_intrinsics()
        effective_detections = (
            {} if self.hybrid_hide_tags_check.isChecked() else detections
        )
        return self.hybrid_localizer.update(
            effective_detections, camera_matrix, distortion
        )

    def _process_hybrid_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = detect_apriltags(frame)
        self.current_tag_detections = detections
        preview = frame.copy()
        draw_tag_detections(preview, detections)
        camera_matrix, distortion = self._hybrid_intrinsics()
        draw_tag_coordinate_axes(
            preview,
            detections,
            layout_origins_mm(self.layout_data, include_validation=True),
            camera_matrix,
            distortion,
            float(self.layout_data["tag_size_mm"]),
        )
        estimate = self._evaluate_hybrid(detections)
        if estimate.valid and estimate.source == SOURCE_TAG_VISUAL:
            self._draw_axes(
                preview,
                np.linalg.inv(estimate.workspace_from_camera),
                camera_matrix,
                distortion,
                length_m=0.1,
            )
        self._present_hybrid_estimate(estimate)
        return preview

    def _present_hybrid_estimate(self, estimate) -> None:
        source_labels = {
            SOURCE_TAG_VISUAL: "Tag 视觉闭环",
            SOURCE_ROBOT_FALLBACK: "机器人反推",
            SOURCE_SIMULATED_ROBOT: "模拟机器人反推",
        }
        source_label = source_labels.get(estimate.source, "无有效位姿")
        self.hybrid_source_value.setText(source_label)
        visible_text = ", ".join(map(str, estimate.visible_tag_ids)) or "--"
        self.hybrid_visible_value.setText(visible_text)
        if estimate.rms_reprojection_error_px is None:
            self.hybrid_quality_value.setText("--")
        else:
            self.hybrid_quality_value.setText(
                "RMS {:.3f} px".format(estimate.rms_reprojection_error_px)
            )
        if estimate.valid:
            xyz_m, rpy_deg = xyz_rpy_from_transform(
                estimate.workspace_from_camera
            )
            xyz_mm = xyz_m * 1000.0
            self.hybrid_xyz_value.setText(
                "{:.1f}, {:.1f}, {:.1f}".format(*xyz_mm)
            )
            self.hybrid_rpy_value.setText(
                "{:.1f}, {:.1f}, {:.1f}".format(*rpy_deg)
            )
        else:
            self.hybrid_xyz_value.setText("--")
            self.hybrid_rpy_value.setText("--")

        if estimate.source == SOURCE_TAG_VISUAL:
            self.hybrid_result.setText(
                "Tag 绝对定位有效 · 当前输出可用于视觉反馈"
            )
            self.preview_state.setText("视觉闭环")
        elif estimate.source == SOURCE_ROBOT_FALLBACK:
            self.hybrid_result.setText(
                "Tag 不可用 · 当前由机器人位姿和已标定手眼矩阵反推"
            )
            self.preview_state.setText("机器人回退")
        elif estimate.source == SOURCE_SIMULATED_ROBOT:
            self.hybrid_result.setText(
                "模拟回退有效 · 只验证接口与矩阵链，不可发送给真实机械臂"
            )
            self.preview_state.setText("模拟回退")
        else:
            self.hybrid_result.setText("无有效定位：{}".format(estimate.reason))
            self.preview_state.setText("定位无效")
        self.detection_metric.set_value(
            "{} 个定位 Tag".format(len(estimate.visible_tag_ids))
        )
        self.task_metric.set_value(source_label)
        if estimate.source != self.hybrid_last_source:
            self._log("混合定位来源切换：{}".format(source_label))
            self.hybrid_last_source = estimate.source

    def apply_simulated_robot_pose(self) -> None:
        xyz_m = [
            self.hybrid_pose_spins[axis].value() / 1000.0
            for axis in ("X", "Y", "Z")
        ]
        rpy_deg = [
            self.hybrid_pose_spins[axis].value()
            for axis in ("R", "P", "Yaw")
        ]
        self.robot_pose_provider.set_pose(
            transform_from_xyz_rpy(xyz_m, rpy_deg)
        )
        self._log(
            "模拟末端位姿已更新：XYZ={} mm，RPY={} deg".format(
                [round(value * 1000.0, 2) for value in xyz_m],
                [round(value, 2) for value in rpy_deg],
            )
        )
        self._refresh_hybrid_without_frame()

    def _hybrid_simulation_toggled(self, enabled: bool) -> None:
        self.robot_pose_provider.set_available(enabled)
        self._log("模拟机器人回退已{}".format("启用" if enabled else "关闭"))
        self._refresh_hybrid_without_frame()

    def _refresh_hybrid_without_frame(self) -> None:
        if self.current_stage != 4:
            return
        if self.camera_connected and self.current_frame is not None:
            return
        try:
            self._present_hybrid_estimate(self._evaluate_hybrid({}))
        except Exception as error:
            self.hybrid_result.setText("混合定位配置错误：{}".format(error))

    def change_stage(self, index: int) -> None:
        if index < 0:
            return
        self.current_stage = index
        self.control_stack.setCurrentIndex(index)
        titles = (
            "打印标定板",
            "RGB 相机内参",
            "工作平面外参",
            "ID 103 独立验证",
            "混合定位",
        )
        self.preview_title.setText(titles[index])
        self.detection_metric.set_value("待机")
        self.sidebar_note.setText(
            "当前坐标系：{}".format(
                "workspace → camera" if index == 4 else "ruler_workspace"
            )
        )
        if index == 0:
            self.video_canvas.set_image(TARGET_PREVIEW)
            self.preview_state.setText("600 DPI 素材")
            self.task_metric.set_value("素材就绪")
        elif self.current_frame is not None and self.camera_connected:
            self.video_canvas.set_frame(self.current_frame)
            self.preview_state.setText("实时")
        else:
            self.video_canvas.clear()
            self.preview_state.setText("待机")
        if index == 4:
            self._refresh_hybrid_without_frame()

    def open_target_pdf(self) -> None:
        from PyQt5.QtGui import QDesktopServices

        if not TARGET_PDF.exists():
            self._show_error("打印文件不存在", str(TARGET_PDF))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(TARGET_PDF)))
        self._log("已打开 A4 打印 PDF")

    def verify_target_pdf(self) -> None:
        self.verify_button.setEnabled(False)
        self.print_verify_status.setText("正在执行 600 DPI 尺寸校验…")
        self.verify_process = QProcess(self)
        self.verify_process.setWorkingDirectory(str(ROOT))
        self.verify_process.finished.connect(self.verify_finished)
        self.verify_process.start(sys.executable, [str(ROOT / "verify_targets.py")])

    def verify_finished(self, exit_code: int) -> None:
        output = bytes(self.verify_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        error = bytes(self.verify_process.readAllStandardError()).decode("utf-8", errors="replace")
        self.verify_button.setEnabled(True)
        if exit_code == 0:
            lines = [line for line in output.splitlines() if line.startswith("PASS")]
            self.print_verify_status.setText("尺寸校验通过 · A4 / 24.0 mm / 70.0 mm / ID 100–103")
            self._log("；".join(lines))
        else:
            self.print_verify_status.setText("尺寸校验失败")
            self._show_error("打印素材校验失败", error or output)
        self.verify_process.deleteLater()

    def open_output_folder(self) -> None:
        from PyQt5.QtGui import QDesktopServices

        output = ROOT / "output"
        output.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))

    def _require_camera(self) -> bool:
        if not self.camera_connected or self.current_frame is None:
            self._show_error("相机未连接", "请先连接 Astra Pro 彩色相机")
            return False
        return True

    def _show_error(self, title: str, message: str) -> None:
        self.statusBar().showMessage(message, 6000)
        self._log("{}：{}".format(title, message.replace("\n", " ")))
        QMessageBox.critical(self, title, message)

    def _log(self, message: str) -> None:
        from datetime import datetime

        self.log_output.appendPlainText("{}  {}".format(datetime.now().strftime("%H:%M:%S"), message))

    def closeEvent(self, event) -> None:
        self.stop_rviz_view()
        self._stop_owned_roscore()
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker.wait(3500)
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", help="save an offscreen UI screenshot and exit")
    parser.add_argument(
        "--stage",
        choices=("print", "intrinsics", "workspace", "validation", "localization"),
        default="print",
        help="initial workflow page",
    )
    parser.add_argument(
        "--auto-connect",
        action="store_true",
        help="connect the configured Astra Pro V4L2 device after startup",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Astra Pro Calibration")
    stage_indices = {
        "print": 0,
        "intrinsics": 1,
        "workspace": 2,
        "validation": 3,
        "localization": 4,
    }
    window = CalibrationWindow(stage_indices[args.stage], args.auto_connect)
    window.show()
    if args.screenshot:
        destination = Path(args.screenshot).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        def save_and_exit():
            window.grab().save(str(destination))
            app.quit()

        QTimer.singleShot(1800 if args.auto_connect else 600, save_and_exit)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
