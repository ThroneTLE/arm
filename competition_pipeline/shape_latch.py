"""Compatibility import for the formal match-shape latch."""

import importlib.util
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "ros_ws" / "src" /
    "arm_vision_framework" / "src" / "arm_vision_framework" / "shape_latch.py"
)
_spec = importlib.util.spec_from_file_location(
    "arm_vision_framework.shape_latch", str(_MODULE_PATH)
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
for _name in _module.__all__:
    globals()[_name] = getattr(_module, _name)
__all__ = list(_module.__all__)
