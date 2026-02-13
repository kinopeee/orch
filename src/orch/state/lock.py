from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from orch.util.errors import RunConflictError


@contextmanager
def run_lock(
    run_dir: Path, stale_sec: int = 3600, *, retries: int = 0, retry_interval: float = 0.2
) -> Iterator[None]:
    lock_path = run_dir / ".lock"
    fd: int | None = None

    def _is_stale() -> bool:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            return False
        return age > stale_sec

    attempt = 0
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError as err:
            if _is_stale():
                lock_path.unlink(missing_ok=True)
                continue
            if attempt >= retries:
                raise RunConflictError(f"run is locked by another process: {lock_path}") from err
            attempt += 1
            time.sleep(retry_interval)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)
