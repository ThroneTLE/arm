#!/usr/bin/env python3

import unittest

from arm_vision_framework.adapters.inexbot_modbus import InexbotPoint
from arm_vision_framework.adapters.modbus_global_point import (
    ModbusFallbackError, ModbusGlobalPointRobotController,
)
from arm_vision_framework.controller_state import ControllerState


POINT_FIELDS = (
    "coordinate_system", "angle_unit", "shape", "tool_id", "user_id",
    "reserved_1", "reserved_2", "axis_1", "axis_2", "axis_3", "axis_4",
    "axis_5", "axis_6", "axis_7",
)


def settings(**overrides):
    fallback = {
        "enabled": True,
        "local_program_verified": True,
        "timeout_s": 0.1,
        "poll_interval_s": 0.01,
        "motion_codes": {"MOVJ": 11, "MOVL": 12},
        "global_point_fields": {
            name: {"address": 100 + index, "encoding": "s32", "scale": 0.001}
            for index, name in enumerate(POINT_FIELDS)
        },
        "command_fields": {
            "motion_code": {"address": 200},
            "sequence_id": {"address": 201},
        },
        "start_signal": {"address": 300, "source": "coil"},
        "complete_signal": {
            "address": 301, "source": "discrete", "active_value": True,
        },
        "stop_signal": {"address": 302, "source": "coil"},
    }
    fallback.update(overrides)
    return {"controller": {"modbus_global_point_fallback": fallback}}


class FakeClient:
    def __init__(self, complete=(True,)):
        self.register_writes = []
        self.coil_writes = []
        self.complete = iter(complete)

    def write_multiple_registers(self, address, words):
        self.register_writes.append((address, tuple(words)))
        return True

    def write_single_coil(self, address, value):
        self.coil_writes.append((address, bool(value)))
        return True

    def read_discrete_inputs(self, address, quantity):
        try:
            value = next(self.complete)
        except StopIteration:
            value = False
        return [value]


class StepClock:
    def __init__(self, step=0.02):
        self.value = 0.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


def state(shape=5):
    return ControllerState(
        connected=True, emergency_stop=False, shape=shape, initial_shape=5,
    )


def point(shape=1):
    return InexbotPoint(
        "P0001", coordinate_system=0, angle_unit=0, shape=shape,
        tool_id=1, user_id=2, axes=(0, 1, 2, 3, 4, 5, 0),
    )


class ModbusGlobalPointTest(unittest.TestCase):
    def test_unverified_local_program_is_rejected(self):
        with self.assertRaisesRegex(ModbusFallbackError, "local_program_verified"):
            ModbusGlobalPointRobotController(
                FakeClient(), settings(local_program_verified=False),
                state_provider=state,
            )

    def test_missing_gp_fields_are_rejected(self):
        with self.assertRaisesRegex(ModbusFallbackError, "missing"):
            ModbusGlobalPointRobotController(
                FakeClient(), settings(global_point_fields={}),
                state_provider=state,
            )

    def test_complete_point_is_written_before_start_pulse(self):
        client = FakeClient((False, True))
        controller = ModbusGlobalPointRobotController(
            client, settings(), state_provider=state, sleep=lambda _: None,
        )
        self.assertTrue(controller.move_j((point(),)))
        # Fourteen GP fields plus motion code and sequence are written.
        self.assertEqual(len(client.register_writes), len(POINT_FIELDS) + 2)
        shape_write = client.register_writes[POINT_FIELDS.index("shape")]
        self.assertEqual(shape_write[0], 102)
        # s32 scale 0.001 encodes the latched shape 5 as integer 5000.
        self.assertEqual(shape_write[1], (0, 5000))
        self.assertEqual(client.coil_writes, [(300, True), (300, False)])

    def test_completion_timeout_stops_and_raises(self):
        client = FakeClient((False,) * 20)
        controller = ModbusGlobalPointRobotController(
            client, settings(), state_provider=state, sleep=lambda _: None,
            clock=StepClock(),
        )
        with self.assertRaisesRegex(ModbusFallbackError, "timed out"):
            controller.move_l((point(),))
        self.assertFalse(client.coil_writes[-1][1])

    def test_changed_shape_is_rejected(self):
        controller = ModbusGlobalPointRobotController(
            FakeClient(), settings(), state_provider=lambda: state(6),
            sleep=lambda _: None,
        )
        with self.assertRaisesRegex(ModbusFallbackError, "unsafe|changed"):
            controller.move_j((point(),))


if __name__ == "__main__":
    unittest.main()
