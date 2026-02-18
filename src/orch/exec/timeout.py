from __future__ import annotations

import asyncio


async def wait_with_timeout(
    proc: asyncio.subprocess.Process, timeout_sec: float | None
) -> tuple[bool, int | None]:
    if timeout_sec is None:
        return False, await proc.wait()
    try:
        code = await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
        return False, code
    except TimeoutError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        return True, None
