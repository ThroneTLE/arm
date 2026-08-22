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
        """Halt, re-energise, then MOVJ back to a validated joint point.

        ⚠️ ``robot.stop()`` 在 Inexbot C1102 上是 ``0x2314``, 实测映射到
        ``Deadan_End -> PowerOff`` —— **直接下电, 不是受控停止**。所以:

        1. 伸展着的手臂在这一步会失力下坠一段。调用方必须先确认下方无人无物;
           这也是 ``auto_recover`` 默认为 False、需要现场操作者显式点按的原因。
        2. 下电之后伺服**不再使能**, 紧跟着的 ``move_j`` 必然落空。旧代码就是
           stop -> move_j 中间什么都不做, 于是恢复动作从来没真正执行过, 却把
           ``_recovered`` 置成了 True —— 又一例"程序说成功了但机器人没动"。
           现在显式重新使能 (0x2311), 失败就如实返回 False。

        底层适配器的 ``_ensure_servo_enabled`` 也会在 MOVJ 前兜一次底; 这里
        仍然显式调用, 是为了让"下电后必须重新使能"这条因果关系留在可读代码里,
        并且对不带该兜底的 robot 实现同样成立。
        """
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
            # stop() 下电了 -> 不重新使能的话下面这条 MOVJ 只是空放。
            enable = getattr(self.robot, "enable_servo", None)
            if callable(enable):
                enable()
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
