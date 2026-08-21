"""Fail-closed recovery to a previously validated, non-singular MOVJ pose."""

from dataclasses import dataclass
import re
from typing import Callable, Sequence


SINGULARITY_RE = re.compile(
    r"singular|singularity|奇异|奇点|逆运动学|\bIK\b|configuration",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SafeRecoveryState:
    enabled: bool = False
    safe_points: tuple = ()
    last_reason: str = ""
    recovered: bool = False


class SafeRecoveryManager:
    """Store and optionally execute a validated safe MOVJ point list.

    ``auto_recover`` is deliberately false by default.  A field operator can
    call ``recover(explicit=True)`` from the UI after confirming the arm is
    clear.  No recovery is attempted when the emergency stop or TCP state is
    unknown/unsafe.
    """

    def __init__(self, robot, *, auto_recover=False, state_provider=None,
                 emergency_stop_is_safe=False, singularity_error_codes=()):
        self.robot = robot
        self.auto_recover = bool(auto_recover)
        self.state_provider = state_provider
        self.emergency_stop_is_safe = bool(emergency_stop_is_safe)
        self.singularity_error_codes = tuple(
            str(code).strip() for code in singularity_error_codes
        )
        self._safe_points = tuple()
        self._last_reason = ""
        self._recovered = False

    @property
    def state(self):
        return SafeRecoveryState(
            enabled=self.auto_recover,
            safe_points=self._safe_points,
            last_reason=self._last_reason,
            recovered=self._recovered,
        )

    def save(self, points: Sequence):
        points = tuple(points)
        if not points:
            raise ValueError("at least one safe MOVJ point is required")
        if any(getattr(point, "coordinate_system", None) != 0 for point in points):
            raise ValueError("safe recovery points must be joint points")
        self._safe_points = points
        self._recovered = False
        return self.state

    def clear(self):
        self._safe_points = tuple()
        self._recovered = False

    def reason_is_singularity(self, reason):
        text = str(reason or "")
        if SINGULARITY_RE.search(text):
            return True
        if any(
            re.search(r"(?<!\d){}(?!\d)".format(re.escape(code)), text)
            for code in self.singularity_error_codes if code
        ):
            return True
        if self.state_provider is None:
            return False
        try:
            alarm = getattr(self.state_provider(), "alarm", None)
            if alarm is None:
                return False
            if getattr(alarm, "is_suspected_singularity", lambda: False)():
                return True
            return str(getattr(alarm, "code", "")) in self.singularity_error_codes
        except Exception:
            return False

    def _state_allows_motion(self):
        if self.state_provider is None:
            return False
        state = self.state_provider()
        if state is None or not getattr(state, "connected", False):
            return False
        emergency_stop = getattr(state, "emergency_stop", None)
        if emergency_stop is not False and not self.emergency_stop_is_safe:
            return False
        if getattr(state, "tcp_xyz_mm", None) is None:
            return False
        return True

    def recover(self, *, explicit=False, speed_scale=0.05):
        self._last_reason = ""
        self._recovered = False
        if not self._safe_points:
            self._last_reason = "no validated safe point"
            return False
        if not explicit and not self.auto_recover:
            self._last_reason = "automatic recovery disabled"
            return False
        if not self._state_allows_motion():
            self._last_reason = "controller/TCP state is not safe for recovery"
            return False
        try:
            stopped = self.robot.stop()
            if stopped is False:
                self._last_reason = "controller rejected stop before recovery"
                return False
            self.robot.move_j(self._safe_points, speed_scale=float(speed_scale))
            self._recovered = True
            return True
        except Exception as error:
            self._last_reason = str(error)
            try:
                self.robot.stop()
            except Exception:
                pass
            return False


__all__ = ["SINGULARITY_RE", "SafeRecoveryState", "SafeRecoveryManager"]
