"""Latch the controller's first valid six-axis configuration value."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShapeLatchState:
    initial_shape: Optional[int] = None
    observed_shape: Optional[int] = None
    changed: bool = False


class ShapeLatch:
    """Keep the match-start shape while exposing later branch changes."""

    def __init__(self, initial_shape=None):
        self._initial = None if initial_shape is None else self._validate(initial_shape)
        self._observed = None
        self._changed = False

    @staticmethod
    def _validate(shape):
        shape = int(shape)
        if not 1 <= shape <= 8:
            raise ValueError("shape must be within 1..8")
        return shape

    @property
    def value(self):
        return self._initial

    @property
    def state(self):
        return ShapeLatchState(self._initial, self._observed, self._changed)

    def observe(self, shape):
        if shape is None:
            return self.state
        self._observed = self._validate(shape)
        if self._initial is None:
            self._initial = self._observed
        elif self._observed != self._initial:
            self._changed = True
        return self.state

    def reset(self, initial_shape=None):
        self._initial = None if initial_shape is None else self._validate(initial_shape)
        self._observed = None
        self._changed = False
        return self.state


__all__ = ["ShapeLatch", "ShapeLatchState"]
