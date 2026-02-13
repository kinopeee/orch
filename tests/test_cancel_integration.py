from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from orch.config.schema import PlanSpec, TaskSpec
from orch.exec.cancel import write_cancel_request
from orch.exec.runner import run_plan
from orch.util.paths import ensure_run_layout


@pytest.mark.asyncio
async def test_cancel_request_stops_run(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_cancel"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    plan = PlanSpec(
        goal="cancel test",
        artifacts_dir=None,
        tasks=[
            TaskSpec(id="long", cmd=[sys.executable, "-c", "import time; time.sleep(5)"]),
            TaskSpec(
                id="downstream",
                cmd=[sys.executable, "-c", "print('never')"],
                depends_on=["long"],
            ),
        ],
    )

    run_future = asyncio.create_task(
        run_plan(
            plan,
            run_dir,
            max_parallel=1,
            fail_fast=False,
            workdir=workdir,
            resume=False,
            failed_only=False,
        )
    )
    await asyncio.sleep(0.4)
    write_cancel_request(run_dir)
    state = await run_future
    assert state.status == "CANCELED"
    assert state.tasks["long"].status in {"CANCELED", "FAILED"}
    assert state.tasks["downstream"].status == "CANCELED"
