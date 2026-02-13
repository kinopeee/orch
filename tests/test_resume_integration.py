from __future__ import annotations

import sys
from pathlib import Path

import pytest

from orch.config.schema import PlanSpec, TaskSpec
from orch.exec.runner import run_plan
from orch.state.store import load_state, save_state_atomic
from orch.util.paths import ensure_run_layout


@pytest.mark.asyncio
async def test_resume_skips_success_and_reruns_failed(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_resume"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    gate_file = workdir / "gate.ok"
    fail_until_gate_exists = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; sys.exit(0 if Path('gate.ok').exists() else 1)",
    ]
    plan = PlanSpec(
        goal="resume test",
        artifacts_dir=None,
        tasks=[
            TaskSpec(id="root", cmd=[sys.executable, "-c", "print('root')"]),
            TaskSpec(id="flaky", cmd=fail_until_gate_exists, depends_on=["root"]),
        ],
    )

    first = await run_plan(
        plan,
        run_dir,
        max_parallel=1,
        fail_fast=False,
        workdir=workdir,
        resume=False,
        failed_only=False,
    )
    assert first.tasks["root"].status == "SUCCESS"
    assert first.tasks["flaky"].status == "FAILED"

    gate_file.write_text("ok", encoding="utf-8")
    resumed = await run_plan(
        plan,
        run_dir,
        max_parallel=1,
        fail_fast=False,
        workdir=workdir,
        resume=True,
        failed_only=True,
    )
    assert resumed.tasks["root"].status == "SUCCESS"
    assert resumed.tasks["flaky"].status == "SUCCESS"


@pytest.mark.asyncio
async def test_resume_converts_interrupted_running_and_reruns(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_resume_interrupted"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    plan = PlanSpec(
        goal="resume interrupted",
        artifacts_dir=None,
        tasks=[TaskSpec(id="single", cmd=[sys.executable, "-c", "print('ok')"])],
    )

    first = await run_plan(
        plan,
        run_dir,
        max_parallel=1,
        fail_fast=False,
        workdir=workdir,
        resume=False,
        failed_only=False,
    )
    assert first.tasks["single"].status == "SUCCESS"
    assert first.tasks["single"].attempts == 1

    interrupted = load_state(run_dir)
    interrupted.status = "RUNNING"
    interrupted.tasks["single"].status = "RUNNING"
    interrupted.tasks["single"].skip_reason = None
    save_state_atomic(run_dir, interrupted)

    resumed = await run_plan(
        plan,
        run_dir,
        max_parallel=1,
        fail_fast=False,
        workdir=workdir,
        resume=True,
        failed_only=True,
    )
    assert resumed.status == "SUCCESS"
    assert resumed.tasks["single"].status == "SUCCESS"
    assert resumed.tasks["single"].attempts == 2
