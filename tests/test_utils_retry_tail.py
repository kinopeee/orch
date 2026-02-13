from __future__ import annotations

import os
from pathlib import Path

import pytest

from orch.exec.retry import backoff_for_attempt
from orch.util.tail import tail_lines


def test_backoff_for_attempt_uses_configured_values_and_clamps_to_last() -> None:
    backoff = [0.5, 1.0]
    assert backoff_for_attempt(0, backoff) == 0.5
    assert backoff_for_attempt(1, backoff) == 1.0
    assert backoff_for_attempt(5, backoff) == 1.0


def test_backoff_for_attempt_uses_exponential_default_with_cap() -> None:
    assert backoff_for_attempt(0, []) == 1.0
    assert backoff_for_attempt(1, []) == 2.0
    assert backoff_for_attempt(6, []) == 60.0


def test_tail_lines_returns_last_n_lines(tmp_path: Path) -> None:
    file_path = tmp_path / "log.txt"
    file_path.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    assert tail_lines(file_path, 2) == ["d", "e"]


def test_tail_lines_handles_nonexistent_or_nonpositive_requests(tmp_path: Path) -> None:
    missing = tmp_path / "missing.log"
    assert tail_lines(missing, 10) == []
    existing = tmp_path / "log.txt"
    existing.write_text("x\ny\n", encoding="utf-8")
    assert tail_lines(existing, 0) == []


def test_tail_lines_returns_empty_for_directory_path(tmp_path: Path) -> None:
    directory = tmp_path / "logs_dir"
    directory.mkdir()
    assert tail_lines(directory, 10) == []


def test_tail_lines_returns_empty_for_symlink_path(tmp_path: Path) -> None:
    target = tmp_path / "outside.log"
    target.write_text("secret\nline2\n", encoding="utf-8")
    symlink = tmp_path / "linked.log"
    symlink.symlink_to(target)

    assert tail_lines(symlink, 10) == []


def test_tail_lines_returns_empty_for_symlink_ancestor_path(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_logs"
    real_parent.mkdir()
    target = real_parent / "app.log"
    target.write_text("line1\nline2\n", encoding="utf-8")
    link_parent = tmp_path / "logs_link"
    link_parent.symlink_to(real_parent, target_is_directory=True)

    assert tail_lines(link_parent / "app.log", 10) == []


def test_tail_lines_returns_empty_for_non_regular_file(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        return

    fifo = tmp_path / "log.pipe"
    os.mkfifo(fifo)
    assert tail_lines(fifo, 10) == []


def test_tail_lines_returns_empty_when_fstat_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "log.txt"
    file_path.write_text("a\nb\n", encoding="utf-8")
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

    assert tail_lines(file_path, 10) == []


def test_tail_lines_uses_nonblock_and_nofollow_open_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "log.txt"
    file_path.write_text("a\nb\n", encoding="utf-8")
    captured_flags: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str, flags: int, mode: int = 0o777) -> int:
        if path == str(file_path):
            captured_flags["flags"] = flags
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    assert tail_lines(file_path, 10) == ["a", "b"]

    assert "flags" in captured_flags
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_RDONLY
    if hasattr(os, "O_CREAT"):
        assert not (captured_flags["flags"] & os.O_CREAT)
    if hasattr(os, "O_APPEND"):
        assert not (captured_flags["flags"] & os.O_APPEND)
    if hasattr(os, "O_TRUNC"):
        assert not (captured_flags["flags"] & os.O_TRUNC)
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


def test_tail_lines_returns_empty_when_symlink_check_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "log.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == target:
            raise PermissionError("simulated lstat failure")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    assert tail_lines(target, 10) == []
