from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orch.util.ids import new_run_id
from orch.util.paths import ensure_run_layout, run_dir
from orch.util.time import duration_sec, now_iso


def test_new_run_id_format_includes_timestamp_and_suffix() -> None:
    now = datetime(2026, 2, 13, 12, 34, 56, tzinfo=UTC)
    run_id = new_run_id(now)
    assert run_id.startswith("20260213_123456_")
    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", run_id) is not None


def test_now_iso_returns_timezone_aware_string() -> None:
    value = now_iso()
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None


def test_duration_sec_rounds_to_three_decimals() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    end = start + timedelta(seconds=1.23456)
    assert duration_sec(start, end) == 1.235


def test_run_dir_and_ensure_run_layout_create_expected_paths(tmp_path: Path) -> None:
    path = run_dir(tmp_path, "run_1")
    ensure_run_layout(path)
    assert path == tmp_path / "runs" / "run_1"
    assert (path / "logs").is_dir()
    assert (path / "artifacts").is_dir()
    assert (path / "report").is_dir()


def test_ensure_run_layout_rejects_symlink_run_dir(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    target = tmp_path / "external_run"
    target.mkdir()
    path = runs_dir / "run_1"
    path.symlink_to(target, target_is_directory=True)

    with pytest.raises(OSError, match="must not be symlink"):
        ensure_run_layout(path)


def test_ensure_run_layout_rejects_symlink_logs_dir(tmp_path: Path) -> None:
    path = run_dir(tmp_path, "run_1")
    path.mkdir(parents=True)
    external_logs = tmp_path / "external_logs"
    external_logs.mkdir()
    (path / "logs").symlink_to(external_logs, target_is_directory=True)

    with pytest.raises(OSError, match="must not be symlink"):
        ensure_run_layout(path)
    assert list(external_logs.iterdir()) == []


def test_ensure_run_layout_rejects_symlink_ancestor(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    linked_home = tmp_path / "home_link"
    linked_home.symlink_to(real_home, target_is_directory=True)
    path = run_dir(linked_home, "run_1")

    with pytest.raises(OSError, match="contains symlink component"):
        ensure_run_layout(path)
    assert not (real_home / "runs").exists()


def test_ensure_run_layout_normalizes_lstat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = run_dir(tmp_path, "run_1")
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == path:
            raise PermissionError("simulated lstat failure")
        return original_lstat(path_obj)

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj == path:
            return False
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="path must be directory"):
        ensure_run_layout(path)


def test_ensure_run_layout_normalizes_lstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = run_dir(tmp_path, "run_1")
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == path:
            raise RuntimeError("simulated lstat runtime failure")
        return original_lstat(path_obj)

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj == path:
            return False
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="path must be directory"):
        ensure_run_layout(path)
