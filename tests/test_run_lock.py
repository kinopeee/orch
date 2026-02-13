from __future__ import annotations

import os
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
