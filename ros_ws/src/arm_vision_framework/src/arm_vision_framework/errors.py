"""Framework-specific exceptions."""


class FrameworkError(RuntimeError):
    pass


class ConfigurationError(FrameworkError):
    pass


class BackendUnavailable(FrameworkError):
    pass


class SafetyInterlockError(FrameworkError):
    pass
