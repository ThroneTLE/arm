"""Pipeline-side access to the canonical ROS controller protocol module.

The implementation belongs to the formal ROS package under
``ros_ws/src/arm_vision_framework``.  This compatibility loader lets the
offline competition tests exercise exactly that code without requiring a
catkin overlay to be sourced.  It is not a second vendor implementation.
"""

import importlib.util
from pathlib import Path
import sys


_FORMAL_MODULE = (
    Path(__file__).resolve().parents[1]
    / "ros_ws"
    / "src"
    / "arm_vision_framework"
    / "src"
    / "arm_vision_framework"
    / "adapters"
    / "inexbot_modbus.py"
)

if not _FORMAL_MODULE.is_file():
    raise ImportError("formal ROS controller module is missing: {}".format(_FORMAL_MODULE))

_spec = importlib.util.spec_from_file_location(
    "arm_vision_framework.adapters.inexbot_modbus", str(_FORMAL_MODULE)
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

for _name in _module.__all__:
    globals()[_name] = getattr(_module, _name)

__all__ = list(_module.__all__)
