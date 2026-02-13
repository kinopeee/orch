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
