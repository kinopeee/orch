class OrchError(Exception):
    """Base orchestrator exception."""


class PlanError(OrchError):
    """Raised when the plan is invalid."""


class StateError(OrchError):
    """Raised when state persistence/load fails."""


class RunConflictError(OrchError):
    """Raised when another process owns the run lock."""
