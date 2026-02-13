from __future__ import annotations

import errno
import os
from pathlib import Path


def cancel_requested(run_dir: Path) -> bool:
    path = run_dir / "cancel.request"
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def write_cancel_request(run_dir: Path) -> None:
    path = run_dir / "cancel.request"
    if path.is_symlink():
        raise OSError("cancel request path must not be symlink")
    if path.exists() and not path.is_file():
        raise OSError("cancel request path must be regular file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        if path.is_symlink() or exc.errno == errno.ELOOP:
            raise OSError("cancel request path must not be symlink") from exc
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("cancel requested\n")


def clear_cancel_request(run_dir: Path) -> None:
    path = run_dir / "cancel.request"
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
    except OSError:
        return
