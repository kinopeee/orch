from __future__ import annotations

import sys
from pathlib import Path

import pytest

from orch.config.schema import PlanSpec, TaskSpec
from orch.exec import runner as runner_module
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
async def test_runner_marks_task_failed_when_command_cannot_start(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_start_failure"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    missing_cmd = ["__definitely_missing_command__", "--version"]
    plan = PlanSpec(
        goal="start failure test",
        artifacts_dir=None,
        tasks=[TaskSpec(id="badcmd", cmd=missing_cmd)],
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
    assert state.status == "FAILED"
    assert state.tasks["badcmd"].status == "FAILED"
    assert state.tasks["badcmd"].exit_code == 127
    stderr_log = run_dir / "logs" / "badcmd.err.log"
    assert "failed to start process" in stderr_log.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_runner_does_not_retry_when_command_cannot_start(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_start_failure_no_retry"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    missing_cmd = ["__definitely_missing_command__", "--version"]
    plan = PlanSpec(
        goal="start failure no retry",
        artifacts_dir=None,
        tasks=[TaskSpec(id="badcmd", cmd=missing_cmd, retries=3, retry_backoff_sec=[0.01])],
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
    assert state.status == "FAILED"
    assert state.tasks["badcmd"].status == "FAILED"
    assert state.tasks["badcmd"].attempts == 1
    assert state.tasks["badcmd"].exit_code == 127


@pytest.mark.asyncio
async def test_runner_collects_declared_outputs_even_when_task_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_fail_artifacts"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    fail_with_output_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('out').mkdir(exist_ok=True); "
        "Path('out/fail.txt').write_text('FAIL', encoding='utf-8'); "
        "raise SystemExit(1)",
    ]
    plan = PlanSpec(
        goal="failed artifact collection",
        artifacts_dir="collected",
        tasks=[TaskSpec(id="publish", cmd=fail_with_output_cmd, outputs=["out/*.txt"])],
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
    assert state.status == "FAILED"
    assert state.tasks["publish"].status == "FAILED"
    assert state.tasks["publish"].artifact_paths == ["artifacts/publish/out/fail.txt"]
    run_copy = run_dir / "artifacts" / "publish" / "out" / "fail.txt"
    aggregated_copy = workdir / "collected" / "publish" / "out" / "fail.txt"
    assert run_copy.read_text(encoding="utf-8") == "FAIL"
    assert aggregated_copy.read_text(encoding="utf-8") == "FAIL"


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
async def test_runner_ignores_copy_failures_and_keeps_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_copy_error"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('out').mkdir(exist_ok=True); "
        "Path('out/ok.txt').write_text('OK', encoding='utf-8'); "
        "Path('out/fail.txt').write_text('FAIL', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="artifact copy best effort",
        artifacts_dir=None,
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["out/*.txt"])],
    )

    original_copy2 = runner_module.shutil.copy2

    def flaky_copy2(src: object, dst: object, *args: object, **kwargs: object) -> object:
        if Path(src).name == "fail.txt":
            raise OSError("simulated copy failure")
        return original_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(runner_module.shutil, "copy2", flaky_copy2)

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
    assert state.tasks["publish"].artifact_paths == ["artifacts/publish/out/ok.txt"]
    copied_ok = run_dir / "artifacts" / "publish" / "out" / "ok.txt"
    assert copied_ok.read_text(encoding="utf-8") == "OK"
    assert not (run_dir / "artifacts" / "publish" / "out" / "fail.txt").exists()


@pytest.mark.asyncio
async def test_runner_marks_task_failed_when_run_task_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_internal_exception"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    plan = PlanSpec(
        goal="runner exception handling",
        artifacts_dir=None,
        tasks=[
            TaskSpec(id="a", cmd=[sys.executable, "-c", "print('a')"]),
            TaskSpec(id="b", cmd=[sys.executable, "-c", "print('b')"], depends_on=["a"]),
        ],
    )

    async def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_module, "run_task", boom)

    state = await run_plan(
        plan,
        run_dir,
        max_parallel=1,
        fail_fast=False,
        workdir=workdir,
        resume=False,
        failed_only=False,
    )
    assert state.status == "FAILED"
    assert state.tasks["a"].status == "FAILED"
    assert state.tasks["a"].skip_reason == "runner_exception"
    assert state.tasks["a"].exit_code == 70
    assert state.tasks["a"].attempts == 1
    assert state.tasks["b"].status == "SKIPPED"
    stderr_log = run_dir / "logs" / "a.err.log"
    assert "runner exception: boom" in stderr_log.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_runner_does_not_retry_when_run_task_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_internal_exception_no_retry"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    plan = PlanSpec(
        goal="runner exception no retry",
        artifacts_dir=None,
        tasks=[TaskSpec(id="a", cmd=[sys.executable, "-c", "print('a')"], retries=5)],
    )

    async def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_module, "run_task", boom)

    state = await run_plan(
        plan,
        run_dir,
        max_parallel=1,
        fail_fast=False,
        workdir=workdir,
        resume=False,
        failed_only=False,
    )
    assert state.status == "FAILED"
    assert state.tasks["a"].status == "FAILED"
    assert state.tasks["a"].skip_reason == "runner_exception"
    assert state.tasks["a"].attempts == 1


@pytest.mark.asyncio
async def test_runner_ignores_artifact_dir_creation_failures_and_keeps_success(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_dir_blocked"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)
    blocked = run_dir / "artifacts" / "publish"
    blocked.write_text("blocked\n", encoding="utf-8")

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('out').mkdir(exist_ok=True); "
        "Path('out/ok.txt').write_text('OK', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="artifact mkdir best effort",
        artifacts_dir=None,
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["out/*.txt"])],
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
    assert state.tasks["publish"].artifact_paths == []
    assert blocked.read_text(encoding="utf-8") == "blocked\n"


@pytest.mark.asyncio
async def test_runner_sanitizes_parent_segments_in_outputs_patterns(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_outputs_parent_pattern"
    workdir = tmp_path / "wd"
    outside = tmp_path / "outside"
    workdir.mkdir(parents=True)
    outside.mkdir(parents=True)
    ensure_run_layout(run_dir)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            "Path('../outside').mkdir(exist_ok=True); "
            "Path('../outside/secret.txt').write_text('secret', encoding='utf-8')"
        ),
    ]
    plan = PlanSpec(
        goal="sanitize parent outputs",
        artifacts_dir=None,
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["../outside/*.txt"])],
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

    paths = state.tasks["publish"].artifact_paths
    assert len(paths) == 1
    assert ".." not in paths[0]
    assert "__abs__" in paths[0] or "__external__" in paths[0]
    assert not (run_dir / "artifacts" / "outside" / "secret.txt").exists()


@pytest.mark.asyncio
async def test_runner_ignores_invalid_output_glob_patterns(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_invalid_output_glob"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('out').mkdir(exist_ok=True); "
        "Path('out/ok.txt').write_text('OK', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="invalid outputs pattern",
        artifacts_dir=None,
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["**a"])],
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
    assert state.tasks["publish"].artifact_paths == []


@pytest.mark.asyncio
async def test_runner_copies_artifacts_to_plan_artifacts_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_dir"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('build').mkdir(exist_ok=True); "
        "Path('build/out.txt').write_text('OK', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="artifacts dir copy test",
        artifacts_dir="collected",
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["build/**/*.txt"])],
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
    run_copy = run_dir / "artifacts" / "publish" / "build" / "out.txt"
    aggregated_copy = workdir / "collected" / "publish" / "build" / "out.txt"
    assert run_copy.read_text(encoding="utf-8") == "OK"
    assert aggregated_copy.read_text(encoding="utf-8") == "OK"


@pytest.mark.asyncio
async def test_runner_copies_artifacts_to_absolute_plan_artifacts_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_dir_abs"
    workdir = tmp_path / "wd"
    absolute_aggregate = tmp_path / "aggregate_abs"
    workdir.mkdir(parents=True)
    absolute_aggregate.mkdir(parents=True)
    ensure_run_layout(run_dir)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('build').mkdir(exist_ok=True); "
        "Path('build/out.txt').write_text('OK', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="absolute artifacts dir copy test",
        artifacts_dir=str(absolute_aggregate),
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["build/**/*.txt"])],
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
    run_copy = run_dir / "artifacts" / "publish" / "build" / "out.txt"
    aggregated_copy = absolute_aggregate / "publish" / "build" / "out.txt"
    assert run_copy.read_text(encoding="utf-8") == "OK"
    assert aggregated_copy.read_text(encoding="utf-8") == "OK"


@pytest.mark.asyncio
async def test_runner_keeps_success_when_plan_artifacts_dir_is_unusable(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_dir_unusable"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    unusable = workdir / "blocked"
    unusable.write_text("this is a file, not a directory\n", encoding="utf-8")

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('build').mkdir(exist_ok=True); "
        "Path('build/out.txt').write_text('OK', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="artifacts dir unusable test",
        artifacts_dir="blocked",
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["build/**/*.txt"])],
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
    run_copy = run_dir / "artifacts" / "publish" / "build" / "out.txt"
    assert run_copy.read_text(encoding="utf-8") == "OK"
    assert not (workdir / "blocked" / "publish").exists()


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


@pytest.mark.asyncio
async def test_runner_collects_artifacts_from_absolute_output_glob(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_abs_output_glob"
    workdir = tmp_path / "wd"
    external_dir = tmp_path / "external_artifacts"
    workdir.mkdir(parents=True)
    external_dir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    absolute_file_a = external_dir / "a" / "result.txt"
    absolute_file_b = external_dir / "b" / "result.txt"
    write_cmd = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"Path({str(absolute_file_a.parent)!r}).mkdir(parents=True, exist_ok=True); "
            f"Path({str(absolute_file_b.parent)!r}).mkdir(parents=True, exist_ok=True); "
            f"Path({str(absolute_file_a)!r}).write_text('A', encoding='utf-8'); "
            f"Path({str(absolute_file_b)!r}).write_text('B', encoding='utf-8')"
        ),
    ]
    absolute_glob = str(external_dir / "**" / "result.txt")
    plan = PlanSpec(
        goal="absolute output glob test",
        artifacts_dir=None,
        tasks=[
            TaskSpec(
                id="collect_abs",
                cmd=write_cmd,
                outputs=[absolute_glob],
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
    assert state.tasks["collect_abs"].status == "SUCCESS"
    rel_a = Path("__abs__", *absolute_file_a.parts[1:])
    rel_b = Path("__abs__", *absolute_file_b.parts[1:])
    expected = {
        str(Path("artifacts") / "collect_abs" / rel_a),
        str(Path("artifacts") / "collect_abs" / rel_b),
    }
    assert set(state.tasks["collect_abs"].artifact_paths) == expected
    copied_a = run_dir / "artifacts" / "collect_abs" / rel_a
    copied_b = run_dir / "artifacts" / "collect_abs" / rel_b
    assert copied_a.read_text(encoding="utf-8") == "A"
    assert copied_b.read_text(encoding="utf-8") == "B"


@pytest.mark.asyncio
async def test_runner_rejects_non_positive_max_parallel(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_bad_parallel"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    plan = PlanSpec(
        goal="invalid parallelism",
        artifacts_dir=None,
        tasks=[TaskSpec(id="t1", cmd=[sys.executable, "-c", "print('ok')"])],
    )

    with pytest.raises(ValueError, match="max_parallel must be >= 1"):
        await run_plan(
            plan,
            run_dir,
            max_parallel=0,
            fail_fast=False,
            workdir=workdir,
            resume=False,
            failed_only=False,
        )
