#!/usr/bin/env python3

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from competition_pipeline.grasp_demo import GraspDemoPanel, GraspDemoWorker


class FakeJog:
    def __init__(self):
        self.commands = []
        self.xyz = [0.0, 0.0, 0.0]
        self.abc = [0.0, 0.0, 0.0]

    def gripper(self, open_):
        self.commands.append(("gripper", bool(open_)))

    def move_to_ucs(self, xyz, abc, vel_mm_s, tolerance_mm):
        self.commands.append(("move", tuple(xyz), tuple(abc)))
        self.xyz = list(xyz)
        self.abc = list(abc)

    def current_pose(self):
        import numpy as np

        return tuple(self.xyz), tuple(np.degrees(self.abc))

    def go_reset_position(self):
        self.commands.append(("reset",))


class GraspDemoWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_default_zero_targets_cannot_execute(self):
        panel = GraspDemoPanel(lambda: FakeJog())
        self.assertFalse(panel.btn_go.isEnabled())

    def test_sequence_opens_before_approach_and_places_after_grasp(self):
        jog = FakeJog()
        worker = GraspDemoWorker(
            lambda: jog,
            grasp=([0.0, 0.0, 40.0], [0.0, 0.0, 0.0]),
            place=([0.0, -50.0, 40.0], [0.0, 0.0, 0.0]),
            lift_mm=80.0,
        )
        worker.run()
        self.assertEqual(jog.commands[0], ("reset",))
        self.assertEqual(jog.commands[1], ("gripper", True))
        self.assertIn(("gripper", False), jog.commands)
        self.assertEqual(jog.commands[-3], ("gripper", True))
        self.assertEqual(jog.commands[-2][0], "move")
        self.assertEqual(jog.commands[-1], ("reset",))


if __name__ == "__main__":
    unittest.main()
