#!/usr/bin/env python3
"""Read-only ROS bridge for the controller's official state-register map.

This node deliberately has no MOVJ/MOVL or IO write capability.  It publishes
the stable JSON state contract after the field team has configured the actual
Modbus addresses, encoding and scale in ``system_parameters.yaml``.
"""

import sys
from dataclasses import replace
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import rospy
from std_msgs.msg import String

from arm_vision_framework.adapters.inexbot_modbus import modbus_client_from_config
from arm_vision_framework.controller_state import ControllerState
from arm_vision_framework.controller_state_reader import ControllerStateReader
from arm_vision_framework.parameters import load_system_parameters
from arm_vision_framework.shape_latch import ShapeLatch


class ControllerStateNode:
    def __init__(self):
        config_path = Path(rospy.get_param(
            "~config", str(PACKAGE_ROOT / "config" / "system_parameters.yaml")
        ))
        self.settings = load_system_parameters(config_path)
        controller = self.settings.get("controller", {})
        if not bool(controller.get("enabled", False)):
            raise RuntimeError("controller.enabled must be true for controller_state_node")
        self.client = modbus_client_from_config(self.settings)
        if self.client is None:
            raise RuntimeError("controller Modbus-TCP endpoint is incomplete")
        self.reader = ControllerStateReader(self.client, self.settings)
        if not self.reader.mapping:
            raise RuntimeError(
                "controller.state_registers is empty; refuse to advertise an unverified TCP state"
            )
        self.shape_latch = ShapeLatch(controller.get("initial_shape"))
        self.publisher = rospy.Publisher(
            self.settings["outputs"].get("controller_state_topic", "/arm_vision/controller/state"),
            String, queue_size=5,
        )
        poll_hz = float(controller.get("state_poll_hz", 5.0))
        if poll_hz <= 0.0:
            raise ValueError("controller.state_poll_hz must be positive")
        self.timer = rospy.Timer(rospy.Duration(1.0 / poll_hz), self.on_timer)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("controller state bridge is read-only at %.1f Hz", poll_hz)

    def on_timer(self, _event):
        try:
            state = self.reader.read()
            latch = self.shape_latch.observe(state.shape)
            state = replace(
                state,
                initial_shape=latch.initial_shape,
                shape_changed=latch.changed,
                raw_registers={
                    **state.raw_registers,
                    "observed_shape": latch.observed_shape,
                    "initial_shape": latch.initial_shape,
                    "shape_changed": latch.changed,
                },
            )
        except Exception as error:
            self.client.close()
            state = ControllerState(connected=False, error=str(error))
            rospy.logwarn_throttle(2.0, "controller state read failed: %s", error)
        self.publisher.publish(String(data=state.to_json()))

    def shutdown(self):
        if getattr(self, "timer", None) is not None:
            self.timer.shutdown()
            self.timer = None
        if getattr(self, "client", None) is not None:
            self.client.close()
            self.client = None


def main():
    rospy.init_node("arm_controller_state")
    ControllerStateNode()
    rospy.spin()


if __name__ == "__main__":
    main()
