#!/usr/bin/env python3
"""抓取执行链路的无硬件冒烟测试。

为什么单独有这一组：这条路径在 2026-08-22 之前**从未真正跑过** —— 识别一直正常，
抓取从来没执行成功过。也就是说 ``_grasp_worker`` 里的胶水代码和结果文本的
格式化在现场是第一次运行，任何一个 KeyError / AttributeError 都会表现为
一条含糊的"执行抓取失败"，而现场只有 3 小时。

这里不碰 Tk，只用一个带必要属性的假 self 调用未绑定方法，把胶水跑通。
"""

import queue
import unittest

import numpy as np

from tool.visual_grasp_pipeline.oak_vision_node import (
    OakVisionNode,
    format_grasp_summary,
)
from tool.visual_grasp_pipeline.ucs_grasp import (
    UcsGraspExecutorError,
    UcsGraspSafetyError,
)


def _info(**overrides):
    info = {
        "available": True,
        "rule": "顶点与中心的中点",
        "z_mm": 109.9,
        "engage_mm": 36.6,
        "requested_engage_mm": 36.6,
        "clamped": False,
        "object_height_mm": 146.5,
        "object_top_mm": 146.5,
        "grasp_width_mm": 57.0,
        "cloud_top_mm": 145.2,
        "cloud_points_used": 843,
        "reasons": [],
    }
    info.update(overrides)
    return info


class GraspSummaryTest(unittest.TestCase):
    """结果框里那段文本，操作者要靠它拿尺子核对后才放行。"""

    def test_all_the_numbers_an_operator_needs_are_present(self):
        text = format_grasp_summary(
            [10.0, 20.0, 109.9], [-100.0, 100.0, 109.9], _info(), 80.0
        )
        for needle in ("146.5", "36.6", "57.0", "80", "顶点与中心的中点"):
            self.assertIn(needle, text)
        self.assertIn("顶面交叉校验", text)
        self.assertNotIn("nan", text)

    def test_a_clamped_grasp_says_so_and_shows_the_original_value(self):
        text = format_grasp_summary(
            [0.0, 0.0, 335.0], [-100.0, 100.0, 335.0],
            _info(clamped=True, engage_mm=65.0, requested_engage_mm=100.0,
                  object_height_mm=400.0, object_top_mm=400.0),
            80.0,
        )
        self.assertIn("已按腔体深度抬高", text)
        self.assertIn("100.0", text)
        self.assertIn("65.0", text)

    def test_a_missing_cloud_is_called_out_not_silently_omitted(self):
        text = format_grasp_summary(
            [0.0, 0.0, 100.0], [-100.0, 100.0, 100.0],
            _info(cloud_top_mm=None), 80.0,
        )
        self.assertIn("未经点云交叉校验", text)

    def test_a_partial_info_dict_does_not_raise(self):
        """格式化绝不能因为缺字段就把整轮变成"抓取失败"。"""
        text = format_grasp_summary([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], {}, 80.0)
        self.assertIn("一键抓取已就绪", text)


class FakeJog:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeRunner:
    """替身 UcsGraspRunner；真实执行序列由 test_ucs_grasp 覆盖。"""

    instances = []

    def __init__(self, jog, place_x_mm=0.0, place_y_mm=0.0, on_event=None):
        self.jog = jog
        self.place_x_mm = place_x_mm
        self.place_y_mm = place_y_mm
        self.on_event = on_event
        self.result = {"status": "ok", "grasp_xyz_mm": [0.0, 0.0, 60.0],
                       "place_xyz_mm": [-100.0, 100.0, 60.0]}
        self.raises = None
        FakeRunner.instances.append(self)

    def execute(self, grasp_mm, dry_run=True):
        if self.raises is not None:
            raise self.raises
        if dry_run:
            return dict(self.result, status="dry_run")
        return self.result


class _FakeSelf:
    """带齐 _grasp_worker 需要的属性，不牵扯 Tk。"""

    def __init__(self):
        self.ui_queue = queue.Queue()
        self._jog = FakeJog()
        self._live_reader = None
        self._competition_yaml = "unused-because-jog-is-preset"
        self._controller_host = ""
        self.place_x_mm = -100.0
        self.place_y_mm = 100.0

    def _grasp_error_with_hint(self, error):
        return OakVisionNode._grasp_error_with_hint(self, error)

    def drain(self):
        items = []
        while not self.ui_queue.empty():
            items.append(self.ui_queue.get_nowait())
        return items


class GraspWorkerGlueTest(unittest.TestCase):
    """``_grasp_worker`` 的胶水：分支、消息、busy 复位。"""

    def setUp(self):
        FakeRunner.instances = []
        self._patched = []
        import tool.visual_grasp_pipeline.oak_vision_node as module
        self._module = module
        self._original_runner = module.UcsGraspRunner
        module.UcsGraspRunner = FakeRunner

    def tearDown(self):
        self._module.UcsGraspRunner = self._original_runner

    def _run(self, dry_run=True, raises=None):
        target = _FakeSelf()

        original_init = FakeRunner.__init__

        def _init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.raises = raises

        FakeRunner.__init__ = _init
        try:
            OakVisionNode._grasp_worker(target, [0.0, 0.0, 60.0], dry_run)
        finally:
            FakeRunner.__init__ = original_init
        return target, target.drain()

    def test_a_successful_run_reports_the_actual_place_position(self):
        _target, messages = self._run(dry_run=False)
        kinds = [item[0] for item in messages]
        self.assertIn("status", kinds)
        joined = " ".join(str(item[1]) for item in messages if item[0] == "status")
        self.assertIn("抓取-放置完成", joined)
        self.assertIn("-100", joined)

    def test_a_dry_run_says_no_motion_was_sent(self):
        _target, messages = self._run(dry_run=True)
        joined = " ".join(str(item[1]) for item in messages if item[0] == "status")
        self.assertIn("未发送运动", joined)

    def test_the_runner_gets_the_configured_place_point(self):
        self._run(dry_run=True)
        self.assertEqual(len(FakeRunner.instances), 1)
        runner = FakeRunner.instances[0]
        self.assertAlmostEqual(runner.place_x_mm, -100.0)
        self.assertAlmostEqual(runner.place_y_mm, 100.0)

    def test_a_safety_error_is_surfaced_as_an_error_dialog(self):
        _target, messages = self._run(
            dry_run=False, raises=UcsGraspSafetyError("目标超出安全区间")
        )
        errors = [item[1] for item in messages if item[0] == "error"]
        self.assertTrue(errors)
        self.assertEqual(errors[0][0], "安全拦截")
        self.assertIn("超出安全区间", errors[0][1])

    def test_an_executor_error_carries_the_field_hint(self):
        _target, messages = self._run(
            dry_run=False,
            raises=UcsGraspExecutorError("伺服未能进入运行态(status=1)"),
        )
        errors = [item[1] for item in messages if item[0] == "error"]
        self.assertTrue(errors)
        self.assertEqual(errors[0][0], "执行抓取失败")

    def test_busy_is_always_cleared_even_when_the_run_explodes(self):
        """busy 不复位 = 界面永久卡住，现场只能重启程序。"""
        for raises in (None, RuntimeError("boom"),
                       UcsGraspSafetyError("nope")):
            with self.subTest(raises=type(raises).__name__):
                _target, messages = self._run(dry_run=False, raises=raises)
                self.assertIn(("busy", False), messages)


if __name__ == "__main__":
    unittest.main()
