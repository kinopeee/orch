from __future__ import annotations

from datetime import datetime


def now_iso() -> str:
    """Return timezone-aware current local time in ISO format."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def duration_sec(start: datetime, end: datetime) -> float:
    """Calculate elapsed seconds."""
    return round((end - start).total_seconds(), 3)
