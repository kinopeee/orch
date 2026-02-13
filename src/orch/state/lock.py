"""Run directory lock implementation."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from orch.util.errors import RunLockError


def _is_stale(lock_path: Path, stale_sec: int) -> bool:
    if not lock_path.exists():
        return False
    age = time.time() - lock_path.stat().st_mtime
    return age > stale_sec


@contextmanager
def run_lock(run_dir: Path, stale_sec: int = 3600) -> Iterator[None]:
    """Acquire exclusive lock for run dir."""
    lock_path = run_dir / ".lock"
    lock_fd: int | None = None
    attempts = 0
    while True:
        attempts += 1
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, f"{os.getpid()}\n".encode("utf-8"))
            break
        except FileExistsError:
            if _is_stale(lock_path, stale_sec):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                if attempts < 3:
                    continue
            raise RunLockError(f"Run lock exists: {lock_path}")
    try:
        yield
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
