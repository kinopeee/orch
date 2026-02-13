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

    original_stat = Path.stat
    called = False

    def flaky_stat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal called
        if path_obj == lock_path and not called:
            called = True
            lock_path.unlink(missing_ok=True)
            raise FileNotFoundError("simulated race")
        return original_stat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

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
