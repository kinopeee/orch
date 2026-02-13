from __future__ import annotations

import os
from pathlib import Path

import pytest
import typer

from orch.cli import (
    _resolve_workdir_or_exit,
    _validate_home_or_exit,
    _write_plan_snapshot,
    _write_report,
)
from orch.config.loader import load_plan
from orch.config.schema import PlanSpec, TaskSpec
from orch.state.model import RunState


def test_write_plan_snapshot_roundtrips_to_valid_plan(tmp_path: Path) -> None:
    plan = PlanSpec(
        goal="snapshot goal",
        artifacts_dir="collected",
        tasks=[
            TaskSpec(
                id="build",
                cmd=["python3", "-c", "print('ok')"],
                depends_on=[],
                cwd=".",
                env={"KEY": "VALUE"},
                timeout_sec=1.5,
                retries=2,
                retry_backoff_sec=[0.1, 0.2],
                outputs=["dist/**"],
            ),
            TaskSpec(
                id="test",
                cmd=["python3", "-c", "print('test')"],
                depends_on=["build"],
                retries=0,
                retry_backoff_sec=[],
                outputs=[],
            ),
        ],
    )

    snapshot_path = tmp_path / "plan.yaml"
    _write_plan_snapshot(plan, snapshot_path)
    loaded = load_plan(snapshot_path)
    assert loaded == plan


def test_write_plan_snapshot_rejects_symlink_destination(tmp_path: Path) -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="only", cmd=["python3", "-c", "print('ok')"])],
    )
    target = tmp_path / "outside_plan.yaml"
    target.write_text("sentinel\n", encoding="utf-8")
    snapshot_path = tmp_path / "plan.yaml"
    snapshot_path.symlink_to(target)

    with pytest.raises(OSError, match="plan snapshot path must not be symlink"):
        _write_plan_snapshot(plan, snapshot_path)
    assert target.read_text(encoding="utf-8") == "sentinel\n"


def test_write_plan_snapshot_rejects_non_regular_destination(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")

    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="only", cmd=["python3", "-c", "print('ok')"])],
    )
    snapshot_path = tmp_path / "plan.yaml"
    os.mkfifo(snapshot_path)

    with pytest.raises(OSError, match="plan snapshot path must be regular file"):
        _write_plan_snapshot(plan, snapshot_path)


def test_write_plan_snapshot_rejects_symlink_ancestor_destination(tmp_path: Path) -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="only", cmd=["python3", "-c", "print('ok')"])],
    )
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "link_parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(OSError, match="contains symlink component"):
        _write_plan_snapshot(plan, symlink_parent / "plan.yaml")


def test_write_report_rejects_symlink_ancestor_path(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    real_run_dir = real_parent / "run"
    (real_run_dir / "report").mkdir(parents=True)
    symlink_parent = tmp_path / "link_parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    state = RunState(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        status="SUCCESS",
        goal=None,
        plan_relpath="plan.yaml",
        home=str(tmp_path),
        workdir=str(tmp_path),
        max_parallel=1,
        fail_fast=False,
        tasks={},
    )

    with pytest.raises(OSError, match="contains symlink component"):
        _write_report(state, symlink_parent / "run")
    assert not (real_run_dir / "report" / "final_report.md").exists()


def test_validate_home_or_exit_rejects_when_exists_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    original_exists = Path.exists

    def flaky_exists(path_obj: Path) -> bool:
        if path_obj == home:
            raise PermissionError("simulated exists failure")
        return original_exists(path_obj)

    monkeypatch.setattr(Path, "exists", flaky_exists)

    with pytest.raises(typer.Exit) as exc_info:
        _validate_home_or_exit(home)
    assert exc_info.value.exit_code == 2


def test_validate_home_or_exit_rejects_when_is_dir_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    original_exists = Path.exists
    original_is_dir = Path.is_dir

    def fake_exists(path_obj: Path) -> bool:
        if path_obj == home:
            return True
        return original_exists(path_obj)

    def flaky_is_dir(path_obj: Path) -> bool:
        if path_obj == home:
            raise PermissionError("simulated is_dir failure")
        return original_is_dir(path_obj)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "is_dir", flaky_is_dir)

    with pytest.raises(typer.Exit) as exc_info:
        _validate_home_or_exit(home)
    assert exc_info.value.exit_code == 2


def test_resolve_workdir_or_exit_rejects_when_resolve_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "workdir"
    original_resolve = Path.resolve

    def flaky_resolve(path_obj: Path, *args: object, **kwargs: object) -> Path:
        if path_obj == workdir:
            raise RuntimeError("simulated resolve failure")
        return original_resolve(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)

    with pytest.raises(typer.Exit) as exc_info:
        _resolve_workdir_or_exit(workdir)
    assert exc_info.value.exit_code == 2


def test_resolve_workdir_or_exit_rejects_when_is_dir_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    original_is_dir = Path.is_dir

    def flaky_is_dir(path_obj: Path) -> bool:
        if path_obj == workdir:
            raise PermissionError("simulated is_dir failure")
        return original_is_dir(path_obj)

    monkeypatch.setattr(Path, "is_dir", flaky_is_dir)

    with pytest.raises(typer.Exit) as exc_info:
        _resolve_workdir_or_exit(workdir)
    assert exc_info.value.exit_code == 2
