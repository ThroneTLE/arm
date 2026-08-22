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

    def __init__(self, jog, place_x_mm=0.0, place_y_mm=0.0, on_event=None,
                 speed_mm_s=None, arrival_dwell_s=None, gripper_settle_s=None):
        self.jog = jog
        self.place_x_mm = place_x_mm
        self.place_y_mm = place_y_mm
        self.on_event = on_event
        # UI 参数必须真的传到执行器；漏传会表现为"界面上改了速度但机器人没变"
        self.speed_mm_s = speed_mm_s
        self.arrival_dwell_s = arrival_dwell_s
        self.gripper_settle_s = gripper_settle_s
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


class _Entry:
    """ttk.Entry 的最小替身：参数读取只用到 .get()。"""

    def __init__(self, text):
        self._text = str(text)

    def get(self):
        return self._text


class _FakeSelf:
    """带齐 _grasp_worker 需要的属性，不牵扯 Tk。"""

    def __init__(self):
        from competition_pipeline.place_layout import PlaceLayout

        self.ui_queue = queue.Queue()
        self._jog = FakeJog()
        self._live_reader = None
        self._competition_yaml = "unused-because-jog-is-preset"
        self._controller_host = ""
        self._place_layout = PlaceLayout(
            origin_xy_mm=(-170.0, 170.0), direction=(0.0, -1.0),
            pitch_mm=100.0, count=4, exclusion_radius_mm=45.0,
        )
        self._occupied_slots = []
        self.place_x_mm = -170.0
        self.place_y_mm = 170.0
        # 参数控件的最小替身：只需要 .get() 返回字符串，
        # 这样 _speed_mm_s / _dwell_s 走的是**真实现**而不是被 mock 掉。
        self.speed_entry = _Entry("50")
        self.dwell_entry = _Entry("0.20")
        self.settle_entry = _Entry("0.50")

    def _grasp_error_with_hint(self, error):
        return OakVisionNode._grasp_error_with_hint(self, error)

    def _ensure_jog(self):
        return OakVisionNode._ensure_jog(self)

    def _speed_mm_s(self):
        return OakVisionNode._speed_mm_s(self)

    def _dwell_s(self, entry, fallback):
        return OakVisionNode._dwell_s(self, entry, fallback)

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

    def test_the_runner_gets_the_next_free_place_slot(self):
        self._run(dry_run=True)
        self.assertEqual(len(FakeRunner.instances), 1)
        runner = FakeRunner.instances[0]
        self.assertAlmostEqual(runner.place_x_mm, -170.0)
        self.assertAlmostEqual(runner.place_y_mm, 170.0)

    def test_ui_parameters_reach_the_runner(self):
        """界面上改了速度/停顿，必须真的传到执行器。

        漏传的表现是"界面上改了但机器人没变"，现场很难当场看出来。
        """
        target = _FakeSelf()
        target.speed_entry = _Entry("135")
        target.dwell_entry = _Entry("0.35")
        target.settle_entry = _Entry("0.80")
        OakVisionNode._grasp_worker(target, [0.0, 0.0, 60.0], True)
        runner = FakeRunner.instances[0]
        self.assertAlmostEqual(runner.speed_mm_s, 135.0)
        self.assertAlmostEqual(runner.arrival_dwell_s, 0.35)
        self.assertAlmostEqual(runner.gripper_settle_s, 0.80)

    def test_a_garbled_parameter_falls_back_instead_of_crashing(self):
        """输入框里打错字不该让整轮抓取变成一条含糊的失败。"""
        from tool.visual_grasp_pipeline.ucs_grasp import (
            ARRIVAL_DWELL_S, DEFAULT_SPEED_MM_S, GRIPPER_SETTLE_S,
        )

        target = _FakeSelf()
        target.speed_entry = _Entry("很快")
        target.dwell_entry = _Entry("")
        target.settle_entry = _Entry("abc")
        OakVisionNode._grasp_worker(target, [0.0, 0.0, 60.0], True)
        runner = FakeRunner.instances[0]
        self.assertAlmostEqual(runner.speed_mm_s, DEFAULT_SPEED_MM_S)
        self.assertAlmostEqual(runner.arrival_dwell_s, ARRIVAL_DWELL_S)
        self.assertAlmostEqual(runner.gripper_settle_s, GRIPPER_SETTLE_S)

    def test_a_successful_place_advances_to_the_next_slot(self):
        """连抓多个时必须换槽位；都放同一点会堆叠，第二个落在第一个上面必倒。"""
        target = _FakeSelf()
        for expected_slot, expected_y in enumerate((170.0, 70.0, -30.0)):
            with self.subTest(slot=expected_slot):
                FakeRunner.instances = []
                OakVisionNode._grasp_worker(target, [0.0, 0.0, 60.0], False)
                runner = FakeRunner.instances[0]
                self.assertAlmostEqual(runner.place_y_mm, expected_y)
                self.assertEqual(
                    target._occupied_slots, list(range(expected_slot + 1))
                )

    def test_a_dry_run_does_not_consume_a_slot(self):
        """dry-run 没真放东西。占用槽位会让下一次跳过一个空位。"""
        target = _FakeSelf()
        OakVisionNode._grasp_worker(target, [0.0, 0.0, 60.0], True)
        self.assertEqual(target._occupied_slots, [])

    def test_a_failed_run_does_not_consume_a_slot(self):
        """失败也没放成。占了槽位会把空位当成"已放置"，后续目标被误剔除。"""
        target = _FakeSelf()

        original_init = FakeRunner.__init__

        def _init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.raises = UcsGraspExecutorError("控制器拒绝")

        FakeRunner.__init__ = _init
        try:
            OakVisionNode._grasp_worker(target, [0.0, 0.0, 60.0], False)
        finally:
            FakeRunner.__init__ = original_init
        self.assertEqual(target._occupied_slots, [])

    def test_running_out_of_slots_is_reported_instead_of_wrapping(self):
        target = _FakeSelf()
        target._occupied_slots = [0, 1, 2, 3]
        OakVisionNode._grasp_worker(target, [0.0, 0.0, 60.0], False)
        errors = [item[1] for item in target.drain() if item[0] == "error"]
        self.assertTrue(errors)
        self.assertIn("放置区已经用满", errors[0][1])

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


class _ToggleSelf:
    """只带 on_dry_run_toggle 需要的东西。"""

    class _Var:
        def __init__(self, value):
            self.value = bool(value)

        def get(self):
            return self.value

    class _Status:
        def __init__(self):
            self.text = ""

        def configure(self, text):
            self.text = str(text)

    def __init__(self, dry_run):
        self._dry_run_var = self._Var(dry_run)
        self.status = self._Status()
        self.enable_robot_motion = not dry_run


class MotionDefaultTest(unittest.TestCase):
    """真实运动是默认；空跑是运行期开关，不是启动参数。

    背景：现场只有三小时，"看完坐标才决定执行"是人来做的判断，把它编码成
    "必须加命令行参数重启"只会浪费那三小时里的时间，而且重启会丢掉已算好的坐标。
    """

    def test_launching_with_no_flags_allows_real_motion(self):
        from tool.visual_grasp_pipeline.oak_vision_node import build_parser

        arguments = build_parser().parse_args([])
        self.assertFalse(arguments.dry_run)

    def test_dry_run_flag_starts_in_dry_run(self):
        from tool.visual_grasp_pipeline.oak_vision_node import build_parser

        arguments = build_parser().parse_args(["--dry-run"])
        self.assertTrue(arguments.dry_run)

    def test_the_old_flag_still_parses(self):
        """旧脚本/旧文档里写着 --enable-robot-motion；9 点钟不能因为它报
        'unrecognized argument' 起不来。"""
        from tool.visual_grasp_pipeline.oak_vision_node import build_parser

        arguments = build_parser().parse_args(["--enable-robot-motion"])
        self.assertFalse(arguments.dry_run)

    def test_the_checkbox_flips_motion_without_a_restart(self):
        target = _ToggleSelf(dry_run=False)
        target._dry_run_var.value = True
        OakVisionNode.on_dry_run_toggle(target)
        self.assertFalse(target.enable_robot_motion)
        self.assertIn("空跑", target.status.text)

        target._dry_run_var.value = False
        OakVisionNode.on_dry_run_toggle(target)
        self.assertTrue(target.enable_robot_motion)
        self.assertIn("真实运动", target.status.text)


class ConfirmDialogGeometryTest(unittest.TestCase):
    """确认框是最后一道人工闸门，"会不会压爆"的那个数必须在框里。"""

    class _Self:
        def __init__(self, info):
            self._last_grasp_height_info = info
            self._gripper_geometry = {"jaw_cavity_depth_mm": 80.0}

    def test_it_shows_the_engage_depth_against_the_cavity(self):
        line = OakVisionNode._confirm_geometry_line(self._Self({
            "available": True, "object_height_mm": 146.5,
            "engage_mm": 36.6, "clamped": False,
        }))
        self.assertIn("146.5", line)
        self.assertIn("36.6", line)
        self.assertIn("80", line)

    def test_it_flags_a_clamped_grasp(self):
        line = OakVisionNode._confirm_geometry_line(self._Self({
            "available": True, "object_height_mm": 400.0,
            "engage_mm": 65.0, "clamped": True,
        }))
        self.assertIn("抬高", line)

    def test_it_says_nothing_rather_than_inventing_numbers(self):
        self.assertEqual(
            OakVisionNode._confirm_geometry_line(self._Self({})), ""
        )
        self.assertEqual(
            OakVisionNode._confirm_geometry_line(self._Self(None)), ""
        )


if __name__ == "__main__":
    unittest.main()
