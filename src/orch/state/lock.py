from __future__ import annotations

import errno
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from orch.util.errors import RunConflictError
from orch.util.path_guard import has_symlink_ancestor, is_symlink_path


@contextmanager
def run_lock(
    run_dir: Path, stale_sec: int = 3600, *, retries: int = 0, retry_interval: float = 0.2
) -> Iterator[None]:
    if has_symlink_ancestor(run_dir):
        raise OSError(f"run directory path contains symlink component: {run_dir}")
    if is_symlink_path(run_dir):
        raise OSError(f"run directory must not be symlink: {run_dir}")
    lock_path = run_dir / ".lock"
    fd: int | None = None
    lock_inode: int | None = None
    lock_dev: int | None = None
    open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW

    def _is_stale() -> bool:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            return False
        return age > stale_sec

    attempt = 0
    while True:
        lock_path_is_symlink = is_symlink_path(lock_path)
        if lock_path_is_symlink:
            raise OSError(f"lock path must not be symlink: {lock_path}")
        try:
            acquired_fd = os.open(lock_path, open_flags)
            stat_result = os.fstat(acquired_fd)
            try:
                os.write(acquired_fd, str(os.getpid()).encode("utf-8"))
            except OSError:
                with suppress(OSError):
                    os.close(acquired_fd)
                try:
                    current_lock = lock_path.stat()
                except OSError:
                    current_lock = None
                if (
                    current_lock is not None
                    and current_lock.st_ino == stat_result.st_ino
                    and current_lock.st_dev == stat_result.st_dev
                ):
                    with suppress(OSError):
                        lock_path.unlink(missing_ok=True)
                raise
            fd = acquired_fd
            lock_inode = stat_result.st_ino
            lock_dev = stat_result.st_dev
            break
        except FileExistsError as err:
            if _is_stale():
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    if attempt >= retries:
                        raise RunConflictError(
                            f"run is locked by another process: {lock_path}"
                        ) from err
                    attempt += 1
                    time.sleep(retry_interval)
                continue
            if attempt >= retries:
                raise RunConflictError(f"run is locked by another process: {lock_path}") from err
            attempt += 1
            time.sleep(retry_interval)
        except OSError as err:
            if err.errno == errno.ELOOP:
                raise OSError(f"lock path must not be symlink: {lock_path}") from err
            raise

    try:
        yield
    finally:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
        if lock_inode is not None and lock_dev is not None:
            current: os.stat_result | None
            try:
                current = lock_path.stat()
            except OSError:
                current = None
            if current is not None and current.st_ino == lock_inode and current.st_dev == lock_dev:
                with suppress(OSError):
                    lock_path.unlink(missing_ok=True)
