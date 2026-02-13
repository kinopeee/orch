from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from orch.state.lock import run_lock
from orch.util.errors import RunConflictError


def test_run_lock_creates_and_releases_lock_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"

    assert not lock_path.exists()
    with run_lock(run_dir):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_run_lock_raises_on_conflict_with_fresh_lock(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    lock_path.write_text("other-process", encoding="utf-8")

    with pytest.raises(RunConflictError), run_lock(run_dir, stale_sec=3600, retries=0):
        pass

    assert lock_path.exists()


def test_run_lock_recovers_stale_lock(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    lock_path.write_text("stale-lock", encoding="utf-8")

    old = time.time() - 120
    os.utime(lock_path, (old, old))

    with run_lock(run_dir, stale_sec=1):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_run_lock_releases_lock_even_if_context_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"

    with pytest.raises(RuntimeError), run_lock(run_dir):
        assert lock_path.exists()
        raise RuntimeError("boom")

    assert not lock_path.exists()


def test_run_lock_uses_nofollow_open_flag_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    captured_flags: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if str(path) == str(lock_path):
            captured_flags["flags"] = flags
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    with run_lock(run_dir):
        pass

    assert "flags" in captured_flags
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


def test_run_lock_uses_create_exclusive_write_open_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_open_flags"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    captured_flags: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if str(path) == str(lock_path):
            captured_flags["flags"] = flags
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    with run_lock(run_dir):
        pass

    assert "flags" in captured_flags
    assert captured_flags["flags"] & os.O_CREAT
    assert captured_flags["flags"] & os.O_EXCL
    assert captured_flags["flags"] & os.O_WRONLY
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_WRONLY
    if hasattr(os, "O_TRUNC"):
        assert not (captured_flags["flags"] & os.O_TRUNC)
    if hasattr(os, "O_APPEND"):
        assert not (captured_flags["flags"] & os.O_APPEND)
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK


def test_run_lock_uses_secure_mode_when_creating_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_mode_flags"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    captured_mode: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if str(path) == str(lock_path):
            captured_mode["mode"] = mode
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    with run_lock(run_dir):
        pass

    assert captured_mode.get("mode") == 0o600


def test_run_lock_can_acquire_after_retry_when_lock_disappears(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    lock_path.write_text("holder", encoding="utf-8")

    timer = threading.Timer(0.15, lambda: lock_path.unlink(missing_ok=True))
    timer.start()
    try:
        with run_lock(run_dir, retries=10, retry_interval=0.05):
            assert lock_path.exists()
    finally:
        timer.cancel()


def test_run_lock_handles_stat_race_during_stale_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    lock_path.write_text("holder", encoding="utf-8")

    original_lstat = Path.lstat
    called = False

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal called
        if path_obj == lock_path and not called:
            called = True
            lock_path.unlink(missing_ok=True)
            raise FileNotFoundError("simulated race")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with run_lock(run_dir, retries=1, retry_interval=0.01):
        assert lock_path.exists()


def test_run_lock_raises_conflict_when_stale_lock_cannot_be_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    lock_path.write_text("stale-lock", encoding="utf-8")
    old = time.time() - 120
    os.utime(lock_path, (old, old))

    original_unlink = Path.unlink

    def deny_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        if path_obj == lock_path:
            raise PermissionError("simulated unlink denied")
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_unlink)

    with pytest.raises(RunConflictError), run_lock(run_dir, stale_sec=1, retries=0):
        pass


def test_run_lock_raises_conflict_when_stale_check_lstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    lock_path.write_text("holder", encoding="utf-8")
    old = time.time() - 120
    os.utime(lock_path, (old, old))
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj in {lock_path, run_dir}:
            return False
        return original_is_symlink(path_obj)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == lock_path:
            raise RuntimeError("simulated stale lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(RunConflictError), run_lock(run_dir, stale_sec=1, retries=0):
        pass
    assert lock_path.exists()


def test_run_lock_does_not_delete_replaced_foreign_lock(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"

    with run_lock(run_dir):
        assert lock_path.exists()
        lock_path.unlink()
        lock_path.write_text("foreign-holder", encoding="utf-8")

    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8") == "foreign-holder"


def test_run_lock_ignores_release_unlink_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_unlink = Path.unlink

    def flaky_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        if path_obj == lock_path:
            raise PermissionError("simulated release unlink failure")
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    with run_lock(run_dir):
        assert lock_path.exists()

    assert lock_path.exists()


def test_run_lock_ignores_release_close_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_close = os.close
    failed_once = False

    def flaky_close(fd: int) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("simulated close failure")
        original_close(fd)

    monkeypatch.setattr(os, "close", flaky_close)

    with run_lock(run_dir):
        assert lock_path.exists()


def test_run_lock_ignores_release_close_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_close = os.close
    failed_once = False

    def flaky_close(fd: int) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("simulated close runtime failure")
        original_close(fd)

    monkeypatch.setattr(os, "close", flaky_close)

    with run_lock(run_dir):
        assert lock_path.exists()


def test_run_lock_cleans_up_if_pid_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_write = os.write
    failed_once = False

    def flaky_write(fd: int, data: bytes) -> int:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("simulated write failure")
        return original_write(fd, data)

    monkeypatch.setattr(os, "write", flaky_write)

    with pytest.raises(OSError, match="simulated write failure"), run_lock(run_dir):
        pass

    assert not lock_path.exists()

    with run_lock(run_dir):
        assert lock_path.exists()


def test_run_lock_cleans_up_if_pid_write_runtime_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_write = os.write
    failed_once = False

    def flaky_write(fd: int, data: bytes) -> int:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("simulated write runtime failure")
        return original_write(fd, data)

    monkeypatch.setattr(os, "write", flaky_write)

    with pytest.raises(OSError, match="simulated write runtime failure"), run_lock(run_dir):
        pass

    assert not lock_path.exists()

    with run_lock(run_dir):
        assert lock_path.exists()


def test_run_lock_preserves_write_error_when_cleanup_close_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_write = os.write
    original_close = os.close
    write_failed = False
    close_failed = False

    def flaky_write(fd: int, data: bytes) -> int:
        nonlocal write_failed
        if not write_failed:
            write_failed = True
            raise OSError("simulated write failure")
        return original_write(fd, data)

    def flaky_close(fd: int) -> None:
        nonlocal close_failed
        if not close_failed:
            close_failed = True
            raise RuntimeError("simulated close runtime failure")
        original_close(fd)

    monkeypatch.setattr(os, "write", flaky_write)
    monkeypatch.setattr(os, "close", flaky_close)

    with pytest.raises(OSError, match="simulated write failure"), run_lock(run_dir):
        pass
    assert lock_path.exists() is False


def test_run_lock_wraps_runtime_open_error_as_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original_open = os.open

    def flaky_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if str(path) == str(run_dir / ".lock"):
            raise RuntimeError("simulated open runtime failure")
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", flaky_open)

    with pytest.raises(OSError, match="failed to open lock path"), run_lock(run_dir):
        pass


def test_run_lock_closes_fd_when_fstat_oserror_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_open = os.open
    original_fstat = os.fstat
    original_close = os.close
    tracked_fd: int | None = None
    closed_fd = False

    def capture_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        nonlocal tracked_fd
        fd = original_open(path, flags, mode)
        if str(path) == str(lock_path):
            tracked_fd = fd
        return fd

    def flaky_fstat(fd: int) -> os.stat_result:
        if tracked_fd is not None and fd == tracked_fd:
            raise OSError("simulated fstat failure")
        return original_fstat(fd)

    def capture_close(fd: int) -> None:
        nonlocal closed_fd
        if tracked_fd is not None and fd == tracked_fd:
            closed_fd = True
        original_close(fd)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "fstat", flaky_fstat)
    monkeypatch.setattr(os, "close", capture_close)

    with pytest.raises(OSError, match="simulated fstat failure"), run_lock(run_dir):
        pass
    assert closed_fd is True


def test_run_lock_closes_fd_when_fstat_runtime_error_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_open = os.open
    original_fstat = os.fstat
    original_close = os.close
    tracked_fd: int | None = None
    closed_fd = False

    def capture_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        nonlocal tracked_fd
        fd = original_open(path, flags, mode)
        if str(path) == str(lock_path):
            tracked_fd = fd
        return fd

    def flaky_fstat(fd: int) -> os.stat_result:
        if tracked_fd is not None and fd == tracked_fd:
            raise RuntimeError("simulated fstat runtime failure")
        return original_fstat(fd)

    def capture_close(fd: int) -> None:
        nonlocal closed_fd
        if tracked_fd is not None and fd == tracked_fd:
            closed_fd = True
        original_close(fd)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "fstat", flaky_fstat)
    monkeypatch.setattr(os, "close", capture_close)

    with pytest.raises(OSError, match="failed to open lock path"), run_lock(run_dir):
        pass
    assert closed_fd is True


def test_run_lock_preserves_write_error_when_cleanup_lstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_write = os.write
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink
    failed_once = False

    def flaky_write(fd: int, data: bytes) -> int:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("simulated write failure")
        return original_write(fd, data)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == lock_path:
            raise RuntimeError("simulated cleanup lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj in {lock_path, run_dir}:
            return False
        return original_is_symlink(path_obj)

    monkeypatch.setattr(os, "write", flaky_write)
    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="simulated write failure"), run_lock(run_dir):
        pass
    assert lock_path.exists()


def test_run_lock_rejects_symlink_run_directory(tmp_path: Path) -> None:
    real_run_dir = tmp_path / "real_run"
    real_run_dir.mkdir()
    run_dir = tmp_path / "run_link"
    run_dir.symlink_to(real_run_dir, target_is_directory=True)

    with pytest.raises(OSError, match="run directory must not be symlink"), run_lock(run_dir):
        pass


def test_run_lock_rejects_path_with_symlink_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    real_run_dir = real_parent / "run"
    real_run_dir.mkdir()
    link_parent = tmp_path / "link_parent"
    link_parent.symlink_to(real_parent, target_is_directory=True)

    run_dir = link_parent / "run"
    with pytest.raises(OSError, match="path contains symlink component"), run_lock(run_dir):
        pass
    assert not (real_run_dir / ".lock").exists()


def test_run_lock_rejects_missing_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_run"

    with pytest.raises(OSError, match="run directory not found"), run_lock(run_dir):
        pass
    assert not (run_dir / ".lock").exists()


def test_run_lock_rejects_non_directory_run_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_file"
    run_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OSError, match="run directory must be directory"), run_lock(run_dir):
        pass


def test_run_lock_rejects_symlink_lock_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside_lock"
    outside.write_text("outside\n", encoding="utf-8")
    lock_path = run_dir / ".lock"
    lock_path.symlink_to(outside)

    with pytest.raises(OSError, match="lock path must not be symlink"), run_lock(run_dir):
        pass
    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_run_lock_fails_closed_when_symlink_check_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == lock_path:
            raise PermissionError("simulated lstat failure")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    with pytest.raises(OSError, match="lock path must not be symlink"), run_lock(run_dir):
        pass
    assert not lock_path.exists()


def test_run_lock_fails_closed_when_run_dir_symlink_check_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == run_dir:
            raise PermissionError("simulated run_dir lstat failure")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    with pytest.raises(OSError, match="run directory must not be symlink"), run_lock(run_dir):
        pass
    assert not (run_dir / ".lock").exists()


def test_run_lock_ignores_release_lstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / ".lock"
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj in {lock_path, run_dir}:
            return False
        return original_is_symlink(path_obj)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == lock_path:
            raise RuntimeError("simulated release lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with run_lock(run_dir):
        assert lock_path.exists()
    assert lock_path.exists()
