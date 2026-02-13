from __future__ import annotations

import asyncio
from pathlib import Path


async def stream_to_file(stream: asyncio.StreamReader | None, file_path: Path) -> None:
    if stream is None:
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("ab") as f:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            f.write(chunk)
            f.flush()
