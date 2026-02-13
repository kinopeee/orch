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


@pytest.mark.asyncio
async def test_runner_copies_declared_output_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('out/sub').mkdir(parents=True, exist_ok=True); "
        "Path('out/sub/a.txt').write_text('A', encoding='utf-8'); "
        "Path('out/b.txt').write_text('B', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="artifact copy test",
        artifacts_dir=None,
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["out/**/*.txt"])],
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
    assert state.status == "SUCCESS"
    assert state.tasks["publish"].status == "SUCCESS"
    assert set(state.tasks["publish"].artifact_paths) == {
        "artifacts/publish/out/b.txt",
        "artifacts/publish/out/sub/a.txt",
    }
    assert (run_dir / "artifacts" / "publish" / "out" / "b.txt").read_text(encoding="utf-8") == "B"
    assert (run_dir / "artifacts" / "publish" / "out" / "sub" / "a.txt").read_text(
        encoding="utf-8"
    ) == "A"


@pytest.mark.asyncio
async def test_runner_resolves_relative_task_cwd_against_workdir(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_cwd_relative"
    workdir = tmp_path / "wd"
    (workdir / "subdir").mkdir(parents=True)
    ensure_run_layout(run_dir)

    write_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('marker.txt').write_text('ok', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="relative cwd test",
        artifacts_dir=None,
        tasks=[
            TaskSpec(
                id="write",
                cmd=write_cmd,
                cwd="subdir",
                outputs=["marker.txt"],
            )
        ],
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
    assert state.status == "SUCCESS"
    assert state.tasks["write"].status == "SUCCESS"
    assert (workdir / "subdir" / "marker.txt").read_text(encoding="utf-8") == "ok"
    assert state.tasks["write"].artifact_paths == ["artifacts/write/marker.txt"]


@pytest.mark.asyncio
async def test_runner_uses_absolute_task_cwd_without_rebasing(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_cwd_absolute"
    workdir = tmp_path / "wd"
    absolute_cwd = tmp_path / "absolute_target"
    workdir.mkdir(parents=True)
    absolute_cwd.mkdir(parents=True)
    ensure_run_layout(run_dir)

    write_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('abs_marker.txt').write_text('abs', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="absolute cwd test",
        artifacts_dir=None,
        tasks=[
            TaskSpec(
                id="write_abs",
                cmd=write_cmd,
                cwd=str(absolute_cwd),
                outputs=["abs_marker.txt"],
            )
        ],
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
    assert state.status == "SUCCESS"
    assert state.tasks["write_abs"].status == "SUCCESS"
    assert (absolute_cwd / "abs_marker.txt").read_text(encoding="utf-8") == "abs"
    assert state.tasks["write_abs"].artifact_paths == ["artifacts/write_abs/abs_marker.txt"]
