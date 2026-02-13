from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from orch.exec.cancel import cancel_requested, clear_cancel_request, write_cancel_request
from orch.exec.capture import stream_to_file
from orch.exec.timeout import wait_with_timeout


@pytest.mark.asyncio
async def test_wait_with_timeout_returns_exit_code_when_process_finishes() -> None:
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", "print('ok')")
    timed_out, exit_code = await wait_with_timeout(proc, timeout_sec=1.0)
    assert timed_out is False
    assert exit_code == 0


@pytest.mark.asyncio
async def test_wait_with_timeout_times_out_and_terminates_process() -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(10)",
    )
    timed_out, exit_code = await wait_with_timeout(proc, timeout_sec=0.1)
    assert timed_out is True
    assert exit_code is None
    assert proc.returncode is not None


@pytest.mark.asyncio
async def test_stream_to_file_writes_all_stream_data(tmp_path: Path) -> None:
    file_path = tmp_path / "capture.log"
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "print('line-a');print('line-b')",
        stdout=asyncio.subprocess.PIPE,
    )
    await stream_to_file(proc.stdout, file_path)
    await proc.wait()
    content = file_path.read_text(encoding="utf-8")
    assert "line-a" in content
    assert "line-b" in content


def test_cancel_request_helpers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert cancel_requested(run_dir) is False
    write_cancel_request(run_dir)
    assert cancel_requested(run_dir) is True
    clear_cancel_request(run_dir)
    assert cancel_requested(run_dir) is False


def test_cancel_requested_ignores_directory_and_clear_is_safe(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir_cancel_dir"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.mkdir()
    assert cancel_requested(run_dir) is False
    clear_cancel_request(run_dir)
    assert cancel_path.is_dir()


def test_cancel_requested_ignores_symlink_and_clear_removes_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir_cancel_symlink"
    run_dir.mkdir()
    target = tmp_path / "outside_cancel_request.txt"
    target.write_text("outside\n", encoding="utf-8")
    cancel_path = run_dir / "cancel.request"
    cancel_path.symlink_to(target)

    assert cancel_requested(run_dir) is False
    clear_cancel_request(run_dir)
    assert not cancel_path.exists()
    assert target.read_text(encoding="utf-8") == "outside\n"


def test_write_cancel_request_rejects_symlink_without_overwriting_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir_cancel_symlink_write"
    run_dir.mkdir()
    target = tmp_path / "outside_cancel_target.txt"
    target.write_text("keep me\n", encoding="utf-8")
    cancel_path = run_dir / "cancel.request"
    cancel_path.symlink_to(target)

    with pytest.raises(OSError, match="must not be symlink"):
        write_cancel_request(run_dir)
    assert target.read_text(encoding="utf-8") == "keep me\n"
