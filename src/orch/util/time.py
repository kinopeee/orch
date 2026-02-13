"""Time helpers."""

from datetime import datetime


def now_iso() -> str:
    """Return local time ISO string with timezone."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def duration_sec(start: datetime, end: datetime) -> float:
    """Calculate duration in seconds."""
    return (end - start).total_seconds()
