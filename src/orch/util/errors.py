"""Application-level error types."""


class OrchError(Exception):
    """Base error for orchestrator."""


class PlanError(OrchError):
    """Raised when plan loading/validation fails."""


class RunNotFoundError(OrchError):
    """Raised when run id is unknown."""


class RunLockError(OrchError):
    """Raised when run lock cannot be acquired."""
