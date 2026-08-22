#!/usr/bin/env python3

import struct
import unittest

from arm_vision_framework.adapters.inexbot_modbus import point_from_joint_degrees
from arm_vision_framework.controller_state import (
    ControllerAlarm, ControllerState, VisualTaskCommand,
)
from arm_vision_framework.controller_state_reader import (
    ControllerStateReader, decode_registers,
)
from arm_vision_framework.oak_imu import config_from_camera
from arm_vision_framework.safety_recovery import SafeRecoveryManager


class FakeClient:
    def __init__(self, registers=None, discrete=None):
        self.registers = dict(registers or {})
        self.discrete = dict(discrete or {})

    def read_holding_registers(self, address, quantity):
        return [self.registers[address + index] for index in range(quantity)]

    read_input_registers = read_holding_registers

    def read_discrete_inputs(self, address, quantity):
        return [self.discrete[address + index] for index in range(quantity)]

    read_coils = read_discrete_inputs


class Robot:
    def __init__(self):
        self.calls = []

    def stop(self):
        self.calls.append("stop")
        return True

    def move_j(self, points, speed_scale):
        self.calls.append(("move_j", tuple(points), speed_scale))
        return True


class ControllerContractTest(unittest.TestCase):
    def test_visual_command_json_fixes_units_and_metadata(self):
        command = VisualTaskCommand(
            "vision-1", "movj", ({"joint_deg": [0] * 6},),
            tool_id=1, user_id=2, shape=5, safe_to_execute=False,
        )
        decoded = VisualTaskCommand.from_json(command.to_json())
        self.assertEqual(decoded.motion_type, "MOVJ")
        self.assertEqual(decoded.xyz_unit, "mm")
        self.assertEqual(decoded.angle_unit, "deg")
        self.assertFalse(decoded.safe_to_execute)
        self.assertEqual(decoded.safety_state, "unvalidated")
        with self.assertRaisesRegex(ValueError, "unsupported control schema"):
            VisualTaskCommand.from_dict({
                **command.to_dict(), "schema_version": "arm_vision.control.v2"
            })

    def test_state_codec_is_explicit(self):
        self.assertEqual(decode_registers([0xFFFF], "s16"), -1.0)
        words = struct.unpack(">HH", struct.pack(">f", 12.5))
        self.assertAlmostEqual(decode_registers(words, "f32_be"), 12.5)

    def test_state_reader_preserves_controller_fields(self):
        mapping = {}
        for index in range(6):
            mapping["joint_deg_{}".format(index + 1)] = {
                "address": index, "encoding": "s16", "scale": 0.1,
            }
        for index, name in enumerate(("tcp_x_mm", "tcp_y_mm", "tcp_z_mm", "tcp_rx_deg", "tcp_ry_deg", "tcp_rz_deg"), 10):
            mapping[name] = {"address": index, "encoding": "s16", "scale": 0.1}
        mapping.update({
            "tool_id": {"address": 20}, "user_id": {"address": 21},
            "shape": {"address": 22}, "reserved_1": {"address": 23},
            "reserved_2": {"address": 24},
            "moving": {"address": 30, "source": "discrete"},
            "emergency_stop": {"address": 31, "source": "discrete"},
        })
        registers = {index: index * 10 for index in range(6)}
        registers.update({index: 1000 + index for index in range(10, 16)})
        registers.update({20: 1, 21: 2, 22: 5, 23: 7, 24: 8})
        state = ControllerStateReader(
            FakeClient(registers, {30: False, 31: False}),
            {"controller": {"state_registers": mapping}},
        ).read()
        self.assertEqual(state.joint_deg, (0, 1, 2, 3, 4, 5))
        self.assertEqual(state.tool_id, 1)
        self.assertEqual(state.user_id, 2)
        self.assertEqual(state.shape, 5)
        self.assertEqual(state.reserved, (7, 8))
        self.assertFalse(state.moving)

    def test_recovery_stops_before_low_speed_movej(self):
        robot = Robot()
        state = ControllerState(
            connected=True, emergency_stop=False,
            tcp_xyz_mm=(1, 2, 3), joint_deg=(0, 0, 0, 0, 0, 0),
        )
        manager = SafeRecoveryManager(
            robot, auto_recover=True, state_provider=lambda: state,
            singularity_error_codes=(1234,),
        )
        manager.save([point_from_joint_degrees("P9000", [0] * 6)])
        self.assertTrue(manager.recover())
        self.assertEqual(robot.calls[0], "stop")
        self.assertEqual(robot.calls[1][0], "move_j")
        self.assertEqual(robot.calls[1][2], 0.05)
        self.assertTrue(manager.reason_is_singularity("逆运动学奇异点"))
        self.assertTrue(manager.reason_is_singularity("controller alarm 1234"))

    def test_recovery_re_enables_the_servo_between_stop_and_movej(self):
        """stop() 是 0x2314 = 下电; 不重新使能, 恢复动作只是空放。

        旧代码 stop -> move_j 中间什么都不做, 却把 recovered 置 True。
        """
        class RobotWithEnable(Robot):
            def enable_servo(self):
                self.calls.append("enable_servo")
                return 3

        robot = RobotWithEnable()
        state = ControllerState(
            connected=True, emergency_stop=False,
            tcp_xyz_mm=(1, 2, 3), joint_deg=(0, 0, 0, 0, 0, 0),
        )
        manager = SafeRecoveryManager(
            robot, auto_recover=True, state_provider=lambda: state,
        )
        manager.save([point_from_joint_degrees("P9000", [0] * 6)])
        self.assertTrue(manager.recover())
        self.assertEqual(robot.calls[0], "stop")
        self.assertEqual(robot.calls[1], "enable_servo")
        self.assertEqual(robot.calls[2][0], "move_j")

    def test_recovery_reports_failure_when_the_servo_will_not_re_enable(self):
        """安全闸门不放行时, 必须返回 False 而不是"已恢复"。"""
        class RefusingRobot(Robot):
            def enable_servo(self):
                self.calls.append("enable_servo")
                raise RuntimeError("伺服仍为 status=1")

        robot = RefusingRobot()
        state = ControllerState(
            connected=True, emergency_stop=False,
            tcp_xyz_mm=(1, 2, 3), joint_deg=(0, 0, 0, 0, 0, 0),
        )
        manager = SafeRecoveryManager(
            robot, auto_recover=True, state_provider=lambda: state,
        )
        manager.save([point_from_joint_degrees("P9000", [0] * 6)])
        self.assertFalse(manager.recover())
        self.assertFalse(manager.state.recovered)
        self.assertIn("status=1", manager.state.last_reason)
        self.assertNotIn(
            "move_j", [call[0] if isinstance(call, tuple) else call
                       for call in robot.calls],
        )

    def test_recovery_locks_on_unknown_tcp_state(self):
        robot = Robot()
        manager = SafeRecoveryManager(
            robot, auto_recover=True,
            state_provider=lambda: ControllerState(connected=True),
        )
        manager.save([point_from_joint_degrees("P9000", [0] * 6)])
        self.assertFalse(manager.recover())
        self.assertEqual(robot.calls, [])

    def test_recovery_locks_without_controller_state_provider(self):
        robot = Robot()
        manager = SafeRecoveryManager(robot, auto_recover=True)
        manager.save([point_from_joint_degrees("P9000", [0] * 6)])
        self.assertFalse(manager.recover())
        self.assertEqual(robot.calls, [])

    def test_imu_axis_transform_remains_unknown_by_default(self):
        config = config_from_camera({"imu": {"enabled": True}})
        self.assertTrue(config.enabled)
        self.assertFalse(config.axis_transform_known)


if __name__ == "__main__":
    unittest.main()
