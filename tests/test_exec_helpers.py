from __future__ import annotations

import asyncio
import errno
import os
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


@pytest.mark.asyncio
async def test_stream_to_file_ignores_symlink_target_path(tmp_path: Path) -> None:
    target = tmp_path / "outside.log"
    target.write_text("keep\n", encoding="utf-8")
    symlink = tmp_path / "capture.log"
    symlink.symlink_to(target)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "print('line-a')",
        stdout=asyncio.subprocess.PIPE,
    )
    await stream_to_file(proc.stdout, symlink)
    await proc.wait()

    assert target.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.asyncio
async def test_stream_to_file_ignores_symlink_parent_directory(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside_logs"
    outside_dir.mkdir()
    target = outside_dir / "capture.log"
    target.write_text("keep-parent\n", encoding="utf-8")

    link_parent = tmp_path / "logs_link"
    link_parent.symlink_to(outside_dir, target_is_directory=True)
    linked_path = link_parent / "capture.log"

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "print('line-a')",
        stdout=asyncio.subprocess.PIPE,
    )
    await stream_to_file(proc.stdout, linked_path)
    await proc.wait()

    assert target.read_text(encoding="utf-8") == "keep-parent\n"


@pytest.mark.asyncio
async def test_stream_to_file_ignores_symlink_ancestor_path(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    (real_parent / "logs").mkdir(parents=True)
    target = real_parent / "logs" / "capture.log"
    target.write_text("keep-ancestor\n", encoding="utf-8")
    symlink_parent = tmp_path / "link_parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    linked_path = symlink_parent / "logs" / "capture.log"

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "print('line-a')",
        stdout=asyncio.subprocess.PIPE,
    )
    await stream_to_file(proc.stdout, linked_path)
    await proc.wait()

    assert target.read_text(encoding="utf-8") == "keep-ancestor\n"


@pytest.mark.asyncio
async def test_stream_to_file_rejects_non_regular_opened_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "capture.log"
    original_open = os.open

    def redirect_open(path: str, flags: int, mode: int = 0o777) -> int:
        if path == str(file_path):
            return original_open("/dev/null", os.O_WRONLY)
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", redirect_open)

    stream = asyncio.StreamReader()
    stream.feed_data(b"line-a\n")
    stream.feed_eof()
    await stream_to_file(stream, file_path)

    assert not file_path.exists()


@pytest.mark.asyncio
async def test_stream_to_file_uses_nonblock_open_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "capture.log"
    captured_flags: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str, flags: int, mode: int = 0o777) -> int:
        if path == str(file_path):
            captured_flags["flags"] = flags
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)

    stream = asyncio.StreamReader()
    stream.feed_data(b"line-a\n")
    stream.feed_eof()
    await stream_to_file(stream, file_path)

    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK


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


def test_clear_cancel_request_removes_non_regular_path(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        return

    run_dir = tmp_path / "run_dir_cancel_fifo"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    os.mkfifo(cancel_path)
    assert cancel_requested(run_dir) is False

    clear_cancel_request(run_dir)
    assert not cancel_path.exists()


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


def test_write_cancel_request_normalizes_eloop_as_symlink_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_eloop"
    run_dir.mkdir()

    def _raise_eloop(_path: os.PathLike[str] | str, _flags: int, _mode: int) -> int:
        raise OSError(errno.ELOOP, "Too many symbolic links")

    monkeypatch.setattr(os, "open", _raise_eloop)

    with pytest.raises(OSError, match="must not be symlink"):
        write_cancel_request(run_dir)


def test_write_cancel_request_rejects_non_regular_opened_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_non_regular_open"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    original_open = os.open

    def redirect_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        if str(path) == str(cancel_path):
            return original_open("/dev/null", os.O_WRONLY)
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", redirect_open)

    with pytest.raises(OSError, match="must be regular file"):
        write_cancel_request(run_dir)
    assert not cancel_path.exists()


def test_write_cancel_request_uses_nonblock_open_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_nonblock_flag"
    run_dir.mkdir()
    captured_flags: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        captured_flags["flags"] = flags
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    write_cancel_request(run_dir)

    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK


def test_cancel_helpers_ignore_symlink_ancestor_paths(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    symlink_home = tmp_path / "home_link"
    symlink_home.symlink_to(real_home, target_is_directory=True)
    linked_run_dir = symlink_home / "runs" / "run1"
    cancel_path = real_run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")

    assert cancel_requested(linked_run_dir) is False
    clear_cancel_request(linked_run_dir)
    assert cancel_path.exists()

    with pytest.raises(OSError, match="contains symlink component"):
        write_cancel_request(linked_run_dir)
    assert cancel_path.read_text(encoding="utf-8") == "cancel requested\n"
