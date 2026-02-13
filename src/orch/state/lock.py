from __future__ import annotations

import errno
import os
import stat
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
            lock_meta = lock_path.lstat()
        except (OSError, RuntimeError):
            return False
        if stat.S_ISLNK(lock_meta.st_mode) or not stat.S_ISREG(lock_meta.st_mode):
            return False
        age = time.time() - lock_meta.st_mtime
        return age > stale_sec

    attempt = 0
    while True:
        lock_path_is_symlink = is_symlink_path(lock_path)
        if lock_path_is_symlink:
            raise OSError(f"lock path must not be symlink: {lock_path}")
        try:
            acquired_fd = os.open(lock_path, open_flags)
            stat_result: os.stat_result | None = None
            try:
                stat_result = os.fstat(acquired_fd)
                os.write(acquired_fd, str(os.getpid()).encode("utf-8"))
            except (OSError, RuntimeError) as exc:
                with suppress(OSError, RuntimeError):
                    os.close(acquired_fd)
                if stat_result is not None:
                    try:
                        current_lock = lock_path.lstat()
                    except (OSError, RuntimeError):
                        current_lock = None
                    if (
                        current_lock is not None
                        and stat.S_ISREG(current_lock.st_mode)
                        and current_lock.st_ino == stat_result.st_ino
                        and current_lock.st_dev == stat_result.st_dev
                    ):
                        with suppress(OSError, RuntimeError):
                            lock_path.unlink(missing_ok=True)
                if isinstance(exc, RuntimeError):
                    if stat_result is None:
                        raise OSError(f"failed to open lock path: {lock_path}") from exc
                    raise OSError(str(exc)) from exc
                raise
            assert stat_result is not None
            fd = acquired_fd
            lock_inode = stat_result.st_ino
            lock_dev = stat_result.st_dev
            break
        except FileExistsError as err:
            if _is_stale():
                try:
                    lock_path.unlink(missing_ok=True)
                except (OSError, RuntimeError):
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
        except RuntimeError as err:
            raise OSError(f"failed to open lock path: {lock_path}") from err

    try:
        yield
    finally:
        if fd is not None:
            with suppress(OSError, RuntimeError):
                os.close(fd)
        if lock_inode is not None and lock_dev is not None:
            current: os.stat_result | None
            try:
                current = lock_path.lstat()
            except (OSError, RuntimeError):
                current = None
            if (
                current is not None
                and stat.S_ISREG(current.st_mode)
                and current.st_ino == lock_inode
                and current.st_dev == lock_dev
            ):
                with suppress(OSError, RuntimeError):
                    lock_path.unlink(missing_ok=True)
