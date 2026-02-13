from __future__ import annotations

import errno
import os
import stat
from contextlib import suppress
from pathlib import Path


def _has_symlink_ancestor(path: Path) -> bool:
    current = path.parent
    while True:
        try:
            meta = current.lstat()
        except FileNotFoundError:
            pass
        except OSError:
            return True
        else:
            if stat.S_ISLNK(meta.st_mode):
                return True
        if current == current.parent:
            return False
        current = current.parent


def cancel_requested(run_dir: Path) -> bool:
    path = run_dir / "cancel.request"
    if _has_symlink_ancestor(path):
        return False
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def write_cancel_request(run_dir: Path) -> None:
    path = run_dir / "cancel.request"
    if _has_symlink_ancestor(path):
        raise OSError("cancel request path contains symlink component")
    if path.is_symlink():
        raise OSError("cancel request path must not be symlink")
    if path.exists() and not path.is_file():
        raise OSError("cancel request path must be regular file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(path, flags, 0o600)
        opened_meta = os.fstat(fd)
        if not stat.S_ISREG(opened_meta.st_mode):
            raise OSError("cancel request path must be regular file")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            f.write("cancel requested\n")
    except OSError as exc:
        try:
            is_symlink = path.is_symlink()
        except OSError:
            is_symlink = False
        if is_symlink or exc.errno == errno.ELOOP:
            raise OSError("cancel request path must not be symlink") from exc
        raise
    finally:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)


def clear_cancel_request(run_dir: Path) -> None:
    path = run_dir / "cancel.request"
    if _has_symlink_ancestor(path):
        return
    try:
        meta = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISDIR(meta.st_mode):
        return
    with suppress(OSError):
        path.unlink(missing_ok=True)
