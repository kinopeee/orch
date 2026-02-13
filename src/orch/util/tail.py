from __future__ import annotations

from collections import deque
from pathlib import Path


def _has_symlink_ancestor(path: Path) -> bool:
    current = path.parent
    while True:
        try:
            if current.is_symlink():
                return True
        except OSError:
            return False
        if current == current.parent:
            return False
        current = current.parent


def tail_lines(path: Path, n: int) -> list[str]:
    """Read last N lines without loading the full file in memory."""
    if n <= 0 or not path.is_file() or path.is_symlink() or _has_symlink_ancestor(path):
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n") for line in deque(f, maxlen=n)]
    except OSError:
        return []
