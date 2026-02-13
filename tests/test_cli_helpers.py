from __future__ import annotations

import os
from pathlib import Path

import pytest

from orch.cli import _write_plan_snapshot, _write_report
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
