from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from pathlib import Path


async def stream_to_file(stream: asyncio.StreamReader | None, file_path: Path) -> None:
    if stream is None:
        return
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    if file_path.parent.is_symlink() or file_path.is_symlink():
        return

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(str(file_path), flags, 0o600)
    except OSError:
        return

    try:
        with os.fdopen(fd, "ab") as f:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                f.write(chunk)
                f.flush()
    except OSError:
        with suppress(OSError):
            os.close(fd)
        return
