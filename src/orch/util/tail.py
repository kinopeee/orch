from __future__ import annotations

import os
import stat
from collections import deque
from contextlib import suppress
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
    if n <= 0 or path.is_symlink() or _has_symlink_ancestor(path):
        return []
    flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(str(path), flags)
        opened_meta = os.fstat(fd)
        if not stat.S_ISREG(opened_meta.st_mode):
            return []
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as f:
            fd = None
            return [line.rstrip("\n") for line in deque(f, maxlen=n)]
    except OSError:
        return []
    finally:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
