"""Named remote-IO gripper adapter with no controller-specific addresses."""

import time

from ..interfaces import GripperController


class RemoteIoGripper(GripperController):
    """Drive a gripper through already-configured named controller IO.

    The controller manual confirms remote digital IO but does not assign the
    gripper addresses or electrical polarity.  This adapter deliberately
    accepts only explicit output maps from the field configuration.
    """

    def __init__(
        self, remote_io, open_outputs, close_outputs, stop_outputs=None,
        done_input=None, command_timeout_s=2.0, poll_interval_s=0.05,
        sleep=time.sleep, clock=time.monotonic,
    ):
        self.remote_io = remote_io
        self.open_outputs = self._outputs(open_outputs, "open_outputs")
        self.close_outputs = self._outputs(close_outputs, "close_outputs")
        self.stop_outputs = self._outputs(
            stop_outputs or {}, "stop_outputs", allow_empty=True
        )
        self.done_input = None if done_input in (None, "") else str(done_input)
        self.command_timeout_s = float(command_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        if self.command_timeout_s <= 0.0 or self.poll_interval_s <= 0.0:
            raise ValueError("gripper timeout and poll interval must be positive")
        self._sleep = sleep
        self._clock = clock

    @staticmethod
    def _outputs(values, name, allow_empty=False):
        if not isinstance(values, dict) or (not values and not allow_empty):
            raise ValueError("gripper.{} must contain at least one named output".format(name))
        return {str(key): bool(value) for key, value in values.items()}

    def _write(self, values):
        for name, value in values.items():
            if self.remote_io.set_output(name, value) is False:
                raise RuntimeError("controller rejected gripper output {}".format(name))

    def _wait_done(self):
        if self.done_input is None:
            return True
        deadline = self._clock() + self.command_timeout_s
        while self._clock() <= deadline:
            if self.remote_io.read_input(self.done_input):
                return True
            self._sleep(self.poll_interval_s)
        raise RuntimeError("gripper feedback {} timed out".format(self.done_input))

    def open(self):
        self._write(self.open_outputs)
        return self._wait_done()

    def close(self, width_mm, maximum_effort=None):
        # Width/effort are recorded by the caller.  A digital gripper cannot
        # honour either value until a vendor-specific analogue/fieldbus map is
        # explicitly supplied.
        float(width_mm)
        if maximum_effort is not None:
            float(maximum_effort)
        self._write(self.close_outputs)
        return self._wait_done()

    def stop(self):
        if self.stop_outputs:
            self._write(self.stop_outputs)
        return True


def gripper_from_config(settings, remote_io):
    """Create a gripper only after every named IO mapping is configured."""

    data = settings.data if hasattr(settings, "data") else settings
    entry = data.get("gripper", {})
    if not bool(entry.get("enabled", False)):
        return None
    return RemoteIoGripper(
        remote_io,
        entry.get("open_outputs", {}),
        entry.get("close_outputs", {}),
        entry.get("stop_outputs", {}),
        entry.get("done_input"),
        entry.get("command_timeout_s", 2.0),
        entry.get("poll_interval_s", 0.05),
    )


__all__ = ["RemoteIoGripper", "gripper_from_config"]
