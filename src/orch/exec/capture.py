from __future__ import annotations

import asyncio
import os
import stat
from contextlib import suppress
from pathlib import Path

from orch.util.path_guard import has_symlink_ancestor, is_symlink_path


async def stream_to_file(stream: asyncio.StreamReader | None, file_path: Path) -> None:
    if stream is None:
        return
    if has_symlink_ancestor(file_path):
        return
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError):
        return
    if is_symlink_path(file_path.parent) or is_symlink_path(file_path):
        return

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd: int | None = None
    try:
        fd = os.open(str(file_path), flags, 0o600)
        opened_meta = os.fstat(fd)
        if not stat.S_ISREG(opened_meta.st_mode):
            with suppress(OSError, RuntimeError):
                os.close(fd)
            fd = None
            return
    except (OSError, RuntimeError):
        if fd is not None:
            with suppress(OSError, RuntimeError):
                os.close(fd)
        return

    try:
        assert fd is not None
        with os.fdopen(fd, "ab") as f:
            fd = None
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                f.write(chunk)
                f.flush()
    except (OSError, RuntimeError):
        if fd is not None:
            with suppress(OSError, RuntimeError):
                os.close(fd)
        return
