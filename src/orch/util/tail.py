"""Tail helpers for potentially large log files."""

from __future__ import annotations

from collections import deque
from pathlib import Path


def tail_lines(path: Path, n: int) -> list[str]:
    """Return the last n lines from file."""
    if n <= 0:
        return []
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return list(deque(f, maxlen=n))
