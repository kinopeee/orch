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
async def test_stream_to_file_closes_fd_when_opened_target_not_regular(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "capture.log"
    original_open = os.open
    original_close = os.close
    tracked_fd: int | None = None
    closed_fd = False

    def redirect_open(path: str, flags: int, mode: int = 0o777) -> int:
        nonlocal tracked_fd
        if path == str(file_path):
            fd = original_open("/dev/null", os.O_WRONLY)
            tracked_fd = fd
            return fd
        return original_open(path, flags, mode)

    def capture_close(fd: int) -> None:
        nonlocal closed_fd
        if tracked_fd is not None and fd == tracked_fd:
            closed_fd = True
        original_close(fd)

    monkeypatch.setattr(os, "open", redirect_open)
    monkeypatch.setattr(os, "close", capture_close)

    stream = asyncio.StreamReader()
    stream.feed_data(b"line-a\n")
    stream.feed_eof()
    await stream_to_file(stream, file_path)

    assert closed_fd is True


@pytest.mark.asyncio
async def test_stream_to_file_ignores_fstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "capture.log"
    original_open = os.open
    original_fstat = os.fstat
    tracked_fd: int | None = None

    def capture_open(path: str, flags: int, mode: int = 0o777) -> int:
        nonlocal tracked_fd
        fd = original_open(path, flags, mode)
        if path == str(file_path):
            tracked_fd = fd
        return fd

    def flaky_fstat(fd: int) -> os.stat_result:
        if tracked_fd is not None and fd == tracked_fd:
            raise RuntimeError("simulated fstat runtime failure")
        return original_fstat(fd)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "fstat", flaky_fstat)

    stream = asyncio.StreamReader()
    stream.feed_data(b"line-a\n")
    stream.feed_eof()
    await stream_to_file(stream, file_path)

    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_stream_to_file_uses_nonblock_open_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "capture.log"
    captured_flags: dict[str, int] = {}
    captured_mode: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str, flags: int, mode: int = 0o777) -> int:
        if path == str(file_path):
            captured_flags["flags"] = flags
            captured_mode["mode"] = mode
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)

    stream = asyncio.StreamReader()
    stream.feed_data(b"line-a\n")
    stream.feed_eof()
    await stream_to_file(stream, file_path)

    assert "flags" in captured_flags
    assert captured_flags["flags"] & os.O_WRONLY
    assert captured_flags["flags"] & os.O_CREAT
    assert captured_flags["flags"] & os.O_APPEND
    assert captured_mode.get("mode") == 0o600
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_WRONLY
    if hasattr(os, "O_TRUNC"):
        assert not (captured_flags["flags"] & os.O_TRUNC)
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK


@pytest.mark.asyncio
async def test_stream_to_file_uses_nofollow_open_flag(
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

    assert "flags" in captured_flags
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


@pytest.mark.asyncio
async def test_stream_to_file_ignores_when_symlink_check_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "capture.log"
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj in {file_path, file_path.parent}:
            raise PermissionError("simulated lstat failure")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    stream = asyncio.StreamReader()
    stream.feed_data(b"line-a\n")
    stream.feed_eof()
    await stream_to_file(stream, file_path)

    assert not file_path.exists()


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


def test_cancel_requested_missing_run_dir_skips_target_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "missing_run_dir_cancel_requested"
    cancel_path = run_dir / "cancel.request"
    original_lstat = Path.lstat
    target_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstat_calls
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cancel_requested(run_dir) is False
    assert target_lstat_calls == 0


def test_cancel_requested_non_directory_run_path_skips_target_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_as_file_cancel_requested"
    run_dir.write_text("not a directory", encoding="utf-8")
    cancel_path = run_dir / "cancel.request"
    original_lstat = Path.lstat
    target_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstat_calls
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cancel_requested(run_dir) is False
    assert target_lstat_calls == 0


def test_cancel_requested_run_dir_lstat_oserror_skips_target_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_requested_lstat_oserror"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    run_dir_lstat_calls = 0
    target_lstat_calls = 0

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal run_dir_lstat_calls, target_lstat_calls
        if path_obj == run_dir:
            run_dir_lstat_calls += 1
            if run_dir_lstat_calls >= 2:
                raise PermissionError("simulated run_dir lstat failure")
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    assert cancel_requested(run_dir) is False
    assert run_dir_lstat_calls >= 2
    assert target_lstat_calls == 0
    assert cancel_path.exists()


def test_cancel_requested_run_dir_lstat_runtime_skips_target_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_requested_lstat_runtime"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    run_dir_lstat_calls = 0
    target_lstat_calls = 0

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal run_dir_lstat_calls, target_lstat_calls
        if path_obj == run_dir:
            run_dir_lstat_calls += 1
            if run_dir_lstat_calls >= 2:
                raise RuntimeError("simulated run_dir lstat runtime failure")
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    assert cancel_requested(run_dir) is False
    assert run_dir_lstat_calls >= 2
    assert target_lstat_calls == 0
    assert cancel_path.exists()


def test_cancel_requested_returns_false_when_lstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_runtime_error"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == cancel_path:
            raise RuntimeError("simulated lstat runtime failure")
        return original_lstat(path_obj)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    assert cancel_requested(run_dir) is False


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


def test_clear_cancel_request_missing_run_dir_skips_target_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "missing_run_dir_clear_cancel"
    cancel_path = run_dir / "cancel.request"
    original_lstat = Path.lstat
    original_unlink = Path.unlink
    target_lstat_calls = 0
    target_unlink_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstat_calls
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    def capture_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        nonlocal target_unlink_calls
        if path_obj == cancel_path:
            target_unlink_calls += 1
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)
    monkeypatch.setattr(Path, "unlink", capture_unlink)

    clear_cancel_request(run_dir)
    assert target_lstat_calls == 0
    assert target_unlink_calls == 0


def test_clear_cancel_request_non_directory_run_path_skips_target_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_as_file_clear_cancel"
    run_dir.write_text("not a directory", encoding="utf-8")
    cancel_path = run_dir / "cancel.request"
    original_lstat = Path.lstat
    original_unlink = Path.unlink
    target_lstat_calls = 0
    target_unlink_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstat_calls
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    def capture_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        nonlocal target_unlink_calls
        if path_obj == cancel_path:
            target_unlink_calls += 1
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)
    monkeypatch.setattr(Path, "unlink", capture_unlink)

    clear_cancel_request(run_dir)
    assert target_lstat_calls == 0
    assert target_unlink_calls == 0


def test_clear_cancel_request_run_dir_lstat_oserror_skips_target_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_clear_cancel_lstat_oserror"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    original_unlink = Path.unlink
    run_dir_lstat_calls = 0
    target_lstat_calls = 0
    target_unlink_calls = 0

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal run_dir_lstat_calls, target_lstat_calls
        if path_obj == run_dir:
            run_dir_lstat_calls += 1
            if run_dir_lstat_calls >= 2:
                raise PermissionError("simulated run_dir lstat failure")
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    def capture_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        nonlocal target_unlink_calls
        if path_obj == cancel_path:
            target_unlink_calls += 1
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)
    monkeypatch.setattr(Path, "unlink", capture_unlink)

    clear_cancel_request(run_dir)
    assert run_dir_lstat_calls >= 2
    assert target_lstat_calls == 0
    assert target_unlink_calls == 0
    assert cancel_path.exists()


def test_clear_cancel_request_run_dir_lstat_runtime_skips_target_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_clear_cancel_lstat_runtime"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    original_unlink = Path.unlink
    run_dir_lstat_calls = 0
    target_lstat_calls = 0
    target_unlink_calls = 0

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal run_dir_lstat_calls, target_lstat_calls
        if path_obj == run_dir:
            run_dir_lstat_calls += 1
            if run_dir_lstat_calls >= 2:
                raise RuntimeError("simulated run_dir lstat runtime failure")
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    def capture_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        nonlocal target_unlink_calls
        if path_obj == cancel_path:
            target_unlink_calls += 1
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)
    monkeypatch.setattr(Path, "unlink", capture_unlink)

    clear_cancel_request(run_dir)
    assert run_dir_lstat_calls >= 2
    assert target_lstat_calls == 0
    assert target_unlink_calls == 0
    assert cancel_path.exists()


def test_clear_cancel_request_ignores_unlink_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_unlink_runtime_error"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_unlink = Path.unlink

    def flaky_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        if path_obj == cancel_path:
            raise RuntimeError("simulated unlink runtime failure")
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    clear_cancel_request(run_dir)
    assert cancel_path.exists()


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


def test_write_cancel_request_rejects_missing_run_dir_without_open_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "missing_run"
    open_called = False
    original_open = os.open

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_called
        open_called = True
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)

    with pytest.raises(OSError, match="run directory not found"):
        write_cancel_request(run_dir)
    assert open_called is False
    assert not (run_dir / "cancel.request").exists()


def test_write_cancel_request_rejects_non_directory_run_path_without_open_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_as_file"
    run_dir.write_text("not a directory", encoding="utf-8")
    open_called = False
    original_open = os.open

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_called
        open_called = True
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)

    with pytest.raises(OSError, match="run directory must be directory"):
        write_cancel_request(run_dir)
    assert open_called is False


def test_write_cancel_request_normalizes_run_dir_lstat_oserror_without_open_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_lstat_oserror"
    run_dir.mkdir()
    open_called = False
    original_open = os.open
    original_lstat = Path.lstat
    run_dir_lstat_calls = 0

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_called
        open_called = True
        return original_open(path, flags, mode)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal run_dir_lstat_calls
        if path_obj == run_dir:
            run_dir_lstat_calls += 1
            if run_dir_lstat_calls >= 2:
                raise PermissionError("simulated run_dir lstat failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="failed to access cancel request run directory"):
        write_cancel_request(run_dir)
    assert open_called is False
    assert run_dir_lstat_calls >= 2
    assert not (run_dir / "cancel.request").exists()


def test_write_cancel_request_normalizes_run_dir_lstat_runtime_without_open_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_lstat_runtime"
    run_dir.mkdir()
    open_called = False
    original_open = os.open
    original_lstat = Path.lstat
    run_dir_lstat_calls = 0

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_called
        open_called = True
        return original_open(path, flags, mode)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal run_dir_lstat_calls
        if path_obj == run_dir:
            run_dir_lstat_calls += 1
            if run_dir_lstat_calls >= 2:
                raise RuntimeError("simulated run_dir lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="failed to access cancel request run directory"):
        write_cancel_request(run_dir)
    assert open_called is False
    assert run_dir_lstat_calls >= 2
    assert not (run_dir / "cancel.request").exists()


def test_write_cancel_request_rejects_symlink_without_open_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_symlink_write_no_open"
    run_dir.mkdir()
    target = tmp_path / "outside_cancel_target_no_open.txt"
    target.write_text("keep me\n", encoding="utf-8")
    cancel_path = run_dir / "cancel.request"
    cancel_path.symlink_to(target)
    open_called = False
    original_open = os.open

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_called
        open_called = True
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)

    with pytest.raises(OSError, match="must not be symlink"):
        write_cancel_request(run_dir)
    assert open_called is False
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


def test_write_cancel_request_normalizes_enxio_as_regular_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_enxio"
    run_dir.mkdir()

    def _raise_enxio(_path: os.PathLike[str] | str, _flags: int, _mode: int) -> int:
        raise OSError(errno.ENXIO, "No such device or address")

    monkeypatch.setattr(os, "open", _raise_enxio)

    with pytest.raises(OSError, match="must be regular file"):
        write_cancel_request(run_dir)


def test_write_cancel_request_normalizes_open_enoent_as_missing_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_enoent"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    original_open = os.open

    def flaky_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        if str(path) == str(cancel_path):
            raise FileNotFoundError("simulated missing run directory")
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", flaky_open)

    with pytest.raises(OSError, match="run directory not found"):
        write_cancel_request(run_dir)


def test_write_cancel_request_normalizes_open_enotdir_as_non_directory_run_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_enotdir"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    original_open = os.open

    def flaky_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        if str(path) == str(cancel_path):
            raise OSError(errno.ENOTDIR, "simulated non-directory run path")
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", flaky_open)

    with pytest.raises(OSError, match="run directory must be directory"):
        write_cancel_request(run_dir)


def test_write_cancel_request_fails_closed_when_symlink_check_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_symlink_check_error"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    original_is_symlink = Path.is_symlink
    calls = 0

    def flaky_is_symlink(path_obj: Path) -> bool:
        nonlocal calls
        if path_obj == cancel_path:
            calls += 1
            if calls >= 2:
                raise PermissionError("simulated lstat failure")
        return original_is_symlink(path_obj)

    def _raise_eio(_path: os.PathLike[str] | str, _flags: int, _mode: int) -> int:
        raise OSError(errno.EIO, "simulated open failure")

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)
    monkeypatch.setattr(os, "open", _raise_eio)

    with pytest.raises(OSError, match="must not be symlink"):
        write_cancel_request(run_dir)
    assert not cancel_path.exists()


def test_write_cancel_request_fails_closed_when_initial_symlink_check_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_symlink_check_error_precheck"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == cancel_path:
            raise PermissionError("simulated precheck lstat failure")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    with pytest.raises(OSError, match="must not be symlink"):
        write_cancel_request(run_dir)
    assert not cancel_path.exists()


def test_write_cancel_request_fails_closed_precheck_without_open_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_symlink_check_error_precheck_no_open"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    original_is_symlink = Path.is_symlink
    open_called = False
    original_open = os.open

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == cancel_path:
            raise PermissionError("simulated precheck lstat failure")
        return original_is_symlink(path_obj)

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_called
        open_called = True
        return original_open(path, flags, mode)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)
    monkeypatch.setattr(os, "open", capture_open)

    with pytest.raises(OSError, match="must not be symlink"):
        write_cancel_request(run_dir)
    assert open_called is False
    assert not cancel_path.exists()


def test_write_cancel_request_normalizes_lstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_exists_runtime_error"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == cancel_path:
            raise RuntimeError("simulated lstat runtime failure")
        return original_lstat(path_obj)

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj == cancel_path:
            return False
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="must be regular file"):
        write_cancel_request(run_dir)
    assert not cancel_path.exists()


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


def test_write_cancel_request_normalizes_runtime_open_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_close_runtime_error"
    run_dir.mkdir()

    def _raise_runtime(_path: os.PathLike[str] | str, _flags: int, _mode: int = 0o777) -> int:
        raise RuntimeError("simulated open runtime failure")

    monkeypatch.setattr(os, "open", _raise_runtime)

    with pytest.raises(OSError, match="failed to write cancel request"):
        write_cancel_request(run_dir)


def test_write_cancel_request_uses_nonblock_open_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_nonblock_flag"
    run_dir.mkdir()
    captured_flags: dict[str, int] = {}
    captured_mode: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        captured_flags["flags"] = flags
        captured_mode["mode"] = mode
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    write_cancel_request(run_dir)

    assert "flags" in captured_flags
    assert captured_flags["flags"] & os.O_WRONLY
    assert captured_flags["flags"] & os.O_CREAT
    assert captured_flags["flags"] & os.O_TRUNC
    assert captured_mode.get("mode") == 0o600
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_WRONLY
    if hasattr(os, "O_APPEND"):
        assert not (captured_flags["flags"] & os.O_APPEND)
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK


def test_write_cancel_request_uses_nofollow_open_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_nofollow_flag"
    run_dir.mkdir()
    captured_flags: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        captured_flags["flags"] = flags
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    write_cancel_request(run_dir)

    assert "flags" in captured_flags
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


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


def test_cancel_helpers_ignore_symlink_run_dir_paths(tmp_path: Path) -> None:
    real_run_dir = tmp_path / "real_run"
    real_run_dir.mkdir()
    linked_run_dir = tmp_path / "run_link"
    linked_run_dir.symlink_to(real_run_dir, target_is_directory=True)
    cancel_path = real_run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")

    assert cancel_requested(linked_run_dir) is False
    clear_cancel_request(linked_run_dir)
    assert cancel_path.exists()

    with pytest.raises(OSError, match="contains symlink component"):
        write_cancel_request(linked_run_dir)
    assert cancel_path.read_text(encoding="utf-8") == "cancel requested\n"


def test_cancel_requested_symlink_run_dir_skips_target_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run_dir = tmp_path / "real_run"
    real_run_dir.mkdir()
    linked_run_dir = tmp_path / "run_link"
    linked_run_dir.symlink_to(real_run_dir, target_is_directory=True)
    target_cancel_path = linked_run_dir / "cancel.request"
    real_cancel_path = real_run_dir / "cancel.request"
    real_cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    target_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstat_calls
        if path_obj == target_cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cancel_requested(linked_run_dir) is False
    assert target_lstat_calls == 0
    assert real_cancel_path.exists()


def test_clear_cancel_request_symlink_run_dir_skips_target_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run_dir = tmp_path / "real_run"
    real_run_dir.mkdir()
    linked_run_dir = tmp_path / "run_link"
    linked_run_dir.symlink_to(real_run_dir, target_is_directory=True)
    target_cancel_path = linked_run_dir / "cancel.request"
    real_cancel_path = real_run_dir / "cancel.request"
    real_cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_unlink = Path.unlink
    target_unlink_calls = 0

    def capture_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        nonlocal target_unlink_calls
        if path_obj == target_cancel_path:
            target_unlink_calls += 1
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", capture_unlink)

    clear_cancel_request(linked_run_dir)
    assert target_unlink_calls == 0
    assert real_cancel_path.exists()


def test_cancel_requested_symlink_ancestor_skips_target_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    symlink_home = tmp_path / "home_link"
    symlink_home.symlink_to(real_home, target_is_directory=True)
    linked_run_dir = symlink_home / "runs" / "run1"
    target_cancel_path = linked_run_dir / "cancel.request"
    real_cancel_path = real_run_dir / "cancel.request"
    real_cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    target_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstat_calls
        if path_obj == target_cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cancel_requested(linked_run_dir) is False
    assert target_lstat_calls == 0
    assert real_cancel_path.exists()


def test_clear_cancel_request_symlink_ancestor_skips_target_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    symlink_home = tmp_path / "home_link"
    symlink_home.symlink_to(real_home, target_is_directory=True)
    linked_run_dir = symlink_home / "runs" / "run1"
    target_cancel_path = linked_run_dir / "cancel.request"
    real_cancel_path = real_run_dir / "cancel.request"
    real_cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_unlink = Path.unlink
    target_unlink_calls = 0

    def capture_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        nonlocal target_unlink_calls
        if path_obj == target_cancel_path:
            target_unlink_calls += 1
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", capture_unlink)

    clear_cancel_request(linked_run_dir)
    assert target_unlink_calls == 0
    assert real_cancel_path.exists()


def test_write_cancel_request_rejects_symlink_ancestor_without_open_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    symlink_home = tmp_path / "home_link"
    symlink_home.symlink_to(real_home, target_is_directory=True)
    linked_run_dir = symlink_home / "runs" / "run1"
    cancel_path = real_run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    open_called = False
    original_open = os.open

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_called
        open_called = True
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)

    with pytest.raises(OSError, match="contains symlink component"):
        write_cancel_request(linked_run_dir)
    assert open_called is False
    assert cancel_path.read_text(encoding="utf-8") == "cancel requested\n"


def test_write_cancel_request_fails_closed_when_ancestor_lstat_oserror_without_open_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_ancestor_lstat_oserror"
    run_dir.mkdir()
    open_called = False
    original_open = os.open
    original_lstat = Path.lstat

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_called
        open_called = True
        return original_open(path, flags, mode)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == tmp_path:
            raise PermissionError("simulated ancestor lstat failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="contains symlink component"):
        write_cancel_request(run_dir)
    assert open_called is False
    assert not (run_dir / "cancel.request").exists()


def test_write_cancel_request_fails_closed_ancestor_runtime_without_open_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_ancestor_lstat_runtime_error"
    run_dir.mkdir()
    open_called = False
    original_open = os.open
    original_lstat = Path.lstat

    def capture_open(path: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_called
        open_called = True
        return original_open(path, flags, mode)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == tmp_path:
            raise RuntimeError("simulated ancestor lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="contains symlink component"):
        write_cancel_request(run_dir)
    assert open_called is False
    assert not (run_dir / "cancel.request").exists()


def test_cancel_requested_fails_closed_ancestor_oserror_without_target_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_requested_ancestor_oserror"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    target_lstat_calls = 0

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstat_calls
        if path_obj == tmp_path:
            raise PermissionError("simulated ancestor lstat failure")
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    assert cancel_requested(run_dir) is False
    assert target_lstat_calls == 0
    assert cancel_path.exists()


def test_cancel_requested_fails_closed_ancestor_runtime_without_target_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_cancel_requested_ancestor_runtime"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    target_lstat_calls = 0

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstat_calls
        if path_obj == tmp_path:
            raise RuntimeError("simulated ancestor lstat runtime failure")
        if path_obj == cancel_path:
            target_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    assert cancel_requested(run_dir) is False
    assert target_lstat_calls == 0
    assert cancel_path.exists()


def test_clear_cancel_request_fails_closed_ancestor_oserror_without_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_clear_cancel_ancestor_oserror"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    original_unlink = Path.unlink
    unlink_called = False

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == tmp_path:
            raise PermissionError("simulated ancestor lstat failure")
        return original_lstat(path_obj, *args, **kwargs)

    def capture_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        nonlocal unlink_called
        if path_obj == cancel_path:
            unlink_called = True
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)
    monkeypatch.setattr(Path, "unlink", capture_unlink)

    clear_cancel_request(run_dir)
    assert unlink_called is False
    assert cancel_path.exists()


def test_clear_cancel_request_fails_closed_ancestor_runtime_without_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_clear_cancel_ancestor_runtime"
    run_dir.mkdir()
    cancel_path = run_dir / "cancel.request"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    original_lstat = Path.lstat
    original_unlink = Path.unlink
    unlink_called = False

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == tmp_path:
            raise RuntimeError("simulated ancestor lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    def capture_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        nonlocal unlink_called
        if path_obj == cancel_path:
            unlink_called = True
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)
    monkeypatch.setattr(Path, "unlink", capture_unlink)

    clear_cancel_request(run_dir)
    assert unlink_called is False
    assert cancel_path.exists()
