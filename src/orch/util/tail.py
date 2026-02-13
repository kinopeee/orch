from __future__ import annotations

from collections import deque
from pathlib import Path


def tail_lines(path: Path, n: int) -> list[str]:
    """Read last N lines without loading the full file in memory."""
    if n <= 0 or not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in deque(f, maxlen=n)]
