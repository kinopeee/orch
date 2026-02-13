from __future__ import annotations

import sys
from pathlib import Path

import pytest

from orch.config.schema import PlanSpec, TaskSpec
from orch.exec.runner import run_plan
from orch.util.paths import ensure_run_layout


@pytest.mark.asyncio
async def test_runner_propagates_skipped_and_retries(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_retry"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    fail_cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]
    plan = PlanSpec(
        goal="retry test",
        artifacts_dir=None,
        tasks=[
            TaskSpec(id="unstable", cmd=fail_cmd, retries=1, retry_backoff_sec=[0.01]),
            TaskSpec(
                id="downstream", cmd=[sys.executable, "-c", "print('ok')"], depends_on=["unstable"]
            ),
        ],
    )

    state = await run_plan(
        plan,
        run_dir,
        max_parallel=2,
        fail_fast=False,
        workdir=workdir,
        resume=False,
        failed_only=False,
    )
    assert state.status == "FAILED"
    assert state.tasks["unstable"].status == "FAILED"
    assert state.tasks["unstable"].attempts == 2
    assert state.tasks["downstream"].status == "SKIPPED"


@pytest.mark.asyncio
async def test_runner_timeout_marks_task_failed(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_timeout"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    timeout_cmd = [sys.executable, "-c", "import time; time.sleep(2)"]
    plan = PlanSpec(
        goal="timeout test",
        artifacts_dir=None,
        tasks=[TaskSpec(id="slow", cmd=timeout_cmd, timeout_sec=0.2)],
    )
    state = await run_plan(
        plan,
        run_dir,
        max_parallel=1,
        fail_fast=False,
        workdir=workdir,
        resume=False,
        failed_only=False,
    )
    assert state.tasks["slow"].status == "FAILED"
    assert state.tasks["slow"].timed_out is True
