"""Pipeline-side access to the canonical NexBot TCP protocol module.

The implementation belongs to the formal ROS package under
``ros_ws/src/arm_vision_framework``.  This compatibility loader lets the
offline competition tests exercise exactly that code without requiring a
catkin overlay to be sourced.  It is not a second vendor implementation.
"""

import sys
from pathlib import Path


_FORMAL_MODULE = (
    Path(__file__).resolve().parents[1]
    / "ros_ws"
    / "src"
    / "arm_vision_framework"
    / "src"
    / "arm_vision_framework"
    / "adapters"
    / "nexbot_tcp.py"
)

if not _FORMAL_MODULE.is_file():
    raise ImportError("formal ROS NexBot TCP module is missing: {}".format(_FORMAL_MODULE))

_SOURCE_ROOT = str(_FORMAL_MODULE.parents[2])
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from arm_vision_framework.adapters import nexbot_tcp as _module  # noqa: E402

for _name in _module.__all__:
    globals()[_name] = getattr(_module, _name)

__all__ = list(_module.__all__)
