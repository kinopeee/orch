from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from orch.config.schema import PlanSpec, TaskSpec
from orch.exec import runner as runner_module
from orch.exec.runner import run_plan
from orch.state.store import load_state
from orch.util.paths import ensure_run_layout


@pytest.mark.asyncio
async def test_runner_persists_absolute_home_and_workdir_for_relative_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path(".orch_rel") / "runs" / "run_abs_paths"
    ensure_run_layout(run_dir)
    plan = PlanSpec(
        goal="absolute path persistence",
        artifacts_dir=None,
        tasks=[TaskSpec(id="t1", cmd=[sys.executable, "-c", "print('ok')"])],
    )

    state = await run_plan(
        plan,
        run_dir,
        max_parallel=1,
        fail_fast=False,
        workdir=Path("."),
        resume=False,
        failed_only=False,
    )
    assert state.status == "SUCCESS"
    assert Path(state.home).is_absolute()
    assert Path(state.workdir).is_absolute()
    assert state.home == str((tmp_path / ".orch_rel").resolve())
    assert state.workdir == str(tmp_path.resolve())

    loaded = load_state(run_dir)
    assert loaded.home == state.home
    assert loaded.workdir == state.workdir


@pytest.mark.asyncio
async def test_run_plan_normalizes_workdir_resolve_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_workdir_resolve_error"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)
    plan = PlanSpec(
        goal="workdir resolve error normalization",
        artifacts_dir=None,
        tasks=[TaskSpec(id="t1", cmd=[sys.executable, "-c", "print('ok')"])],
    )
    original_resolve = Path.resolve

    def flaky_resolve(path_obj: Path, *args: object, **kwargs: object) -> Path:
        if path_obj == workdir:
            raise RuntimeError("simulated workdir resolve failure")
        return original_resolve(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)

    with pytest.raises(OSError, match="failed to resolve workdir"):
        await run_plan(
            plan,
            run_dir,
            max_parallel=1,
            fail_fast=False,
            workdir=workdir,
            resume=False,
            failed_only=False,
        )


@pytest.mark.asyncio
async def test_run_plan_normalizes_home_resolve_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_home_resolve_error"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)
    home_dir = run_dir.parent.parent
    plan = PlanSpec(
        goal="home resolve error normalization",
        artifacts_dir=None,
        tasks=[TaskSpec(id="t1", cmd=[sys.executable, "-c", "print('ok')"])],
    )
    original_resolve = Path.resolve

    def flaky_resolve(path_obj: Path, *args: object, **kwargs: object) -> Path:
        if path_obj == home_dir:
            raise RuntimeError("simulated home resolve failure")
        return original_resolve(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)

    with pytest.raises(OSError, match="failed to resolve home path"):
        await run_plan(
            plan,
            run_dir,
            max_parallel=1,
            fail_fast=False,
            workdir=workdir,
            resume=False,
            failed_only=False,
        )


@pytest.mark.asyncio
async def test_run_plan_rejects_non_directory_workdir(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_workdir_file"
    ensure_run_layout(run_dir)
    workdir_file = tmp_path / "workdir.txt"
    workdir_file.write_text("not-a-directory\n", encoding="utf-8")
    plan = PlanSpec(
        goal="workdir must be directory",
        artifacts_dir=None,
        tasks=[TaskSpec(id="t1", cmd=[sys.executable, "-c", "print('ok')"])],
    )

    with pytest.raises(OSError, match="workdir must be directory"):
        await run_plan(
            plan,
            run_dir,
            max_parallel=1,
            fail_fast=False,
            workdir=workdir_file,
            resume=False,
            failed_only=False,
        )


@pytest.mark.asyncio
async def test_run_plan_normalizes_workdir_lstat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_workdir_is_dir_error"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)
    plan = PlanSpec(
        goal="workdir is_dir error normalization",
        artifacts_dir=None,
        tasks=[TaskSpec(id="t1", cmd=[sys.executable, "-c", "print('ok')"])],
    )
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == workdir.resolve():
            raise PermissionError("simulated workdir lstat failure")
        return original_lstat(path_obj)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="failed to access workdir"):
        await run_plan(
            plan,
            run_dir,
            max_parallel=1,
            fail_fast=False,
            workdir=workdir,
            resume=False,
            failed_only=False,
        )


@pytest.mark.asyncio
async def test_run_plan_normalizes_workdir_lstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_workdir_is_dir_runtime_error"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)
    plan = PlanSpec(
        goal="workdir is_dir runtime error normalization",
        artifacts_dir=None,
        tasks=[TaskSpec(id="t1", cmd=[sys.executable, "-c", "print('ok')"])],
    )
    original_lstat = Path.lstat
    resolved_workdir = workdir.resolve()

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == resolved_workdir:
            raise RuntimeError("simulated workdir lstat runtime failure")
        return original_lstat(path_obj)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="failed to access workdir"):
        await run_plan(
            plan,
            run_dir,
            max_parallel=1,
            fail_fast=False,
            workdir=workdir,
            resume=False,
            failed_only=False,
        )


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
async def test_runner_clears_terminal_fields_before_retry_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_retry_reset"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    call_count = 0

    async def _fake_run_task(
        task: TaskSpec,
        run_dir_for_task: Path,
        *,
        attempt: int,
        default_cwd: Path,
    ) -> runner_module.TaskResult:
        nonlocal call_count
        assert task.id == "flaky"
        assert default_cwd == workdir
        call_count += 1
        if call_count == 1:
            assert attempt == 1
            return runner_module.TaskResult(
                exit_code=1,
                timed_out=False,
                canceled=False,
                start_failed=False,
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:00:01+00:00",
                duration_sec=1.0,
            )

        assert attempt == 2
        persisted = load_state(run_dir_for_task)
        task_state = persisted.tasks["flaky"]
        assert task_state.status == "RUNNING"
        assert task_state.attempts == 2
        assert task_state.started_at is not None
        assert task_state.ended_at is None
        assert task_state.duration_sec is None
        assert task_state.exit_code is None
        assert task_state.timed_out is False
        assert task_state.canceled is False
        assert task_state.skip_reason is None
        return runner_module.TaskResult(
            exit_code=0,
            timed_out=False,
            canceled=False,
            start_failed=False,
            started_at="2026-01-01T00:00:02+00:00",
            ended_at="2026-01-01T00:00:03+00:00",
            duration_sec=1.0,
        )

    monkeypatch.setattr(runner_module, "run_task", _fake_run_task)

    plan = PlanSpec(
        goal="retry reset test",
        artifacts_dir=None,
        tasks=[TaskSpec(id="flaky", cmd=[sys.executable, "-c", "print('x')"], retries=1)],
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
    assert call_count == 2
    assert state.status == "SUCCESS"
    assert state.tasks["flaky"].status == "SUCCESS"
    assert state.tasks["flaky"].attempts == 2


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
    assert state.tasks["badcmd"].skip_reason == "process_start_failed"
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
    assert state.tasks["badcmd"].skip_reason == "process_start_failed"


@pytest.mark.asyncio
async def test_runner_handles_subprocess_value_error_as_start_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_start_value_error"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    async def _raise_value_error(*args: object, **kwargs: object) -> object:
        raise ValueError("illegal environment variable name")

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", _raise_value_error)

    plan = PlanSpec(
        goal="start value error",
        artifacts_dir=None,
        tasks=[
            TaskSpec(
                id="badcmd",
                cmd=[sys.executable, "-c", "print('x')"],
                retries=3,
                retry_backoff_sec=[0.01],
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
    assert state.status == "FAILED"
    assert state.tasks["badcmd"].status == "FAILED"
    assert state.tasks["badcmd"].attempts == 1
    assert state.tasks["badcmd"].exit_code == 127
    assert state.tasks["badcmd"].skip_reason == "process_start_failed"
    stderr_log = run_dir / "logs" / "badcmd.err.log"
    assert "failed to start process" in stderr_log.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_runner_handles_subprocess_runtime_error_as_start_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_start_runtime_error"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    async def _raise_runtime_error(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated subprocess runtime failure")

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", _raise_runtime_error)

    plan = PlanSpec(
        goal="start runtime error",
        artifacts_dir=None,
        tasks=[
            TaskSpec(
                id="badcmd",
                cmd=[sys.executable, "-c", "print('x')"],
                retries=3,
                retry_backoff_sec=[0.01],
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
    assert state.status == "FAILED"
    assert state.tasks["badcmd"].status == "FAILED"
    assert state.tasks["badcmd"].attempts == 1
    assert state.tasks["badcmd"].exit_code == 127
    assert state.tasks["badcmd"].skip_reason == "process_start_failed"
    stderr_log = run_dir / "logs" / "badcmd.err.log"
    assert "failed to start process" in stderr_log.read_text(encoding="utf-8")


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
async def test_runner_skips_symlink_output_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_symlink"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; "
        "Path('outside').mkdir(exist_ok=True); "
        "Path('outside/secret.txt').write_text('SECRET', encoding='utf-8'); "
        "Path('out').mkdir(exist_ok=True); "
        "Path('out/link.txt').symlink_to('../outside/secret.txt')",
    ]
    plan = PlanSpec(
        goal="artifact symlink skip",
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
    assert not any((run_dir / "artifacts" / "publish").rglob("*.txt"))


@pytest.mark.asyncio
async def test_runner_skips_output_artifacts_under_symlink_ancestor(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_source_symlink_ancestor"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    outside = tmp_path / "outside_src"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    (workdir / "linked").symlink_to(outside, target_is_directory=True)

    plan = PlanSpec(
        goal="artifact source symlink ancestor skip",
        artifacts_dir=None,
        tasks=[
            TaskSpec(
                id="publish",
                cmd=[sys.executable, "-c", "print('ok')"],
                outputs=["linked/*.txt"],
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
    assert state.tasks["publish"].status == "SUCCESS"
    assert state.tasks["publish"].artifact_paths == []
    assert not any((run_dir / "artifacts" / "publish").rglob("*.txt"))


@pytest.mark.asyncio
async def test_runner_skips_copy_when_run_artifacts_root_is_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_root_symlink"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    outside = tmp_path / "outside_artifacts"
    outside.mkdir()
    artifacts_root = run_dir / "artifacts"
    artifacts_root.rmdir()
    artifacts_root.symlink_to(outside, target_is_directory=True)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('out').mkdir(exist_ok=True); "
        "Path('out/ok.txt').write_text('OK', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="artifact root symlink skip",
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
    assert state.tasks["publish"].artifact_paths == []
    assert list(outside.iterdir()) == []


@pytest.mark.asyncio
async def test_runner_does_not_write_logs_when_logs_dir_is_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_logs_symlink"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)
    outside_logs = tmp_path / "outside_logs"
    outside_logs.mkdir()
    logs_dir = run_dir / "logs"
    logs_dir.rmdir()
    logs_dir.symlink_to(outside_logs, target_is_directory=True)

    plan = PlanSpec(
        goal="log symlink safety",
        artifacts_dir=None,
        tasks=[TaskSpec(id="t1", cmd=[sys.executable, "-c", "print('ok')"])],
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
    assert state.tasks["t1"].status == "SUCCESS"
    assert list(outside_logs.iterdir()) == []


def test_runner_append_text_best_effort_ignores_symlink_ancestor_path(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    (real_parent / "logs").mkdir(parents=True)
    target = real_parent / "logs" / "task.err.log"
    target.write_text("keep\n", encoding="utf-8")
    link_parent = tmp_path / "link_parent"
    link_parent.symlink_to(real_parent, target_is_directory=True)

    runner_module._append_text_best_effort(link_parent / "logs" / "task.err.log", "new-line\n")
    assert target.read_text(encoding="utf-8") == "keep\n"


def test_runner_append_text_best_effort_ignores_symlink_check_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "logs" / "task.err.log"
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj in {log_path, log_path.parent}:
            raise PermissionError("simulated lstat failure")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    runner_module._append_text_best_effort(log_path, "new-line\n")
    assert not log_path.exists()


def test_runner_append_text_best_effort_ignores_fstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "logs" / "task.err.log"
    original_open = os.open
    original_fstat = os.fstat
    tracked_fd: int | None = None

    def capture_open(path: str, flags: int, mode: int = 0o777) -> int:
        nonlocal tracked_fd
        fd = original_open(path, flags, mode)
        if path == str(log_path):
            tracked_fd = fd
        return fd

    def flaky_fstat(fd: int) -> os.stat_result:
        if tracked_fd is not None and fd == tracked_fd:
            raise RuntimeError("simulated fstat runtime failure")
        return original_fstat(fd)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "fstat", flaky_fstat)

    runner_module._append_text_best_effort(log_path, "new-line\n")
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == ""


def test_runner_append_text_best_effort_uses_nonblock_and_nofollow_open_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "logs" / "task.err.log"
    captured_flags: dict[str, int] = {}
    captured_mode: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str, flags: int, mode: int = 0o777) -> int:
        if path == str(log_path):
            captured_flags["flags"] = flags
            captured_mode["mode"] = mode
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    runner_module._append_text_best_effort(log_path, "new-line\n")

    assert "flags" in captured_flags
    assert captured_flags["flags"] & os.O_WRONLY
    assert captured_flags["flags"] & os.O_CREAT
    assert captured_flags["flags"] & os.O_APPEND
    assert captured_mode.get("mode") == 0o600
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_WRONLY
    if hasattr(os, "O_TRUNC"):
        assert not (captured_flags["flags"] & os.O_TRUNC)
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


def test_runner_append_text_best_effort_closes_fd_when_target_not_regular(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "logs" / "task.err.log"
    original_open = os.open
    original_close = os.close
    tracked_fd: int | None = None
    closed_fd = False

    def redirect_open(path: str, flags: int, mode: int = 0o777) -> int:
        nonlocal tracked_fd
        if path == str(log_path):
            fd = original_open("/dev/null", os.O_WRONLY)
            tracked_fd = fd
            return fd
        return original_open(path, flags, mode)

    def capture_close(fd: int) -> None:
        nonlocal closed_fd
        if tracked_fd is not None and fd == tracked_fd:
            closed_fd = True
        original_close(fd)

    monkeypatch.setattr(os, "open", redirect_open)
    monkeypatch.setattr(os, "close", capture_close)

    runner_module._append_text_best_effort(log_path, "new-line\n")
    assert closed_fd is True
    assert not log_path.exists()


@pytest.mark.asyncio
async def test_runner_start_failure_does_not_write_symlinked_logs(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_logs_symlink_start_fail"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)
    outside_logs = tmp_path / "outside_logs_fail"
    outside_logs.mkdir()
    logs_dir = run_dir / "logs"
    logs_dir.rmdir()
    logs_dir.symlink_to(outside_logs, target_is_directory=True)

    plan = PlanSpec(
        goal="log symlink safety start failure",
        artifacts_dir=None,
        tasks=[TaskSpec(id="bad", cmd=["__definitely_missing_command__"])],
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
    assert state.tasks["bad"].status == "FAILED"
    assert list(outside_logs.iterdir()) == []


@pytest.mark.asyncio
async def test_runner_preserves_case_only_colliding_artifacts_with_suffix(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_case_collision"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('out').mkdir(exist_ok=True); "
        "Path('out/a.txt').write_text('a', encoding='utf-8'); "
        "Path('out/A.txt').write_text('A', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="artifact case-collision preservation",
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
    paths = state.tasks["publish"].artifact_paths
    assert len(paths) == 2
    assert len({path.casefold() for path in paths}) == 2
    copied_root = run_dir / "artifacts" / "publish" / "out"
    copied_contents = sorted(path.read_text(encoding="utf-8") for path in copied_root.iterdir())
    assert copied_contents == ["A", "a"]
    assert any("__case2" in path for path in paths)


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
async def test_runner_skips_copy_when_source_revalidation_runtime_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_source_revalidate_runtime"
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
        goal="artifact source revalidation runtime handling",
        artifacts_dir=None,
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["out/*.txt"])],
    )

    source_path = (workdir / "out" / "ok.txt").resolve()
    original_lstat = Path.lstat
    source_lstat_calls = 0

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        nonlocal source_lstat_calls
        if path_obj == source_path:
            source_lstat_calls += 1
            if source_lstat_calls >= 2:
                raise RuntimeError("simulated source revalidation runtime failure")
        return original_lstat(path_obj)

    original_copy2 = runner_module.shutil.copy2
    copy_calls = 0

    def tracking_copy2(src: object, dst: object, *args: object, **kwargs: object) -> object:
        nonlocal copy_calls
        copy_calls += 1
        return original_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)
    monkeypatch.setattr(runner_module.shutil, "copy2", tracking_copy2)

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
    assert source_lstat_calls >= 2
    assert copy_calls == 0


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
async def test_runner_ignores_runtime_errors_from_output_glob_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_runtime_output_glob"
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
        goal="runtime outputs glob failure",
        artifacts_dir=None,
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["out/*.txt"])],
    )

    original_glob = Path.glob
    resolved_workdir = workdir.resolve()

    def flaky_glob(path_obj: Path, pattern: str):
        if path_obj == resolved_workdir and pattern == "out/*.txt":
            raise RuntimeError("simulated glob runtime failure")
        return original_glob(path_obj, pattern)

    monkeypatch.setattr(Path, "glob", flaky_glob)

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
async def test_runner_skips_non_regular_output_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_non_regular_outputs"
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True)
    ensure_run_layout(run_dir)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "import os; "
        "from pathlib import Path; "
        "Path('out').mkdir(exist_ok=True); "
        "fifo = Path('out') / 'queue.pipe'; "
        "fifo.unlink(missing_ok=True); "
        "os.mkfifo(fifo)",
    ]
    plan = PlanSpec(
        goal="skip non-regular outputs",
        artifacts_dir=None,
        tasks=[TaskSpec(id="publish", cmd=create_outputs_cmd, outputs=["out/*"])],
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
async def test_runner_skips_aggregate_copy_when_artifacts_dir_is_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_dir_symlink"
    workdir = tmp_path / "wd"
    outside = tmp_path / "outside_collected"
    workdir.mkdir(parents=True)
    outside.mkdir(parents=True)
    ensure_run_layout(run_dir)

    aggregate_link = workdir / "collected"
    aggregate_link.symlink_to(outside, target_is_directory=True)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('build').mkdir(exist_ok=True); "
        "Path('build/out.txt').write_text('OK', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="artifacts dir symlink skip",
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
    run_copy = run_dir / "artifacts" / "publish" / "build" / "out.txt"
    assert run_copy.read_text(encoding="utf-8") == "OK"
    assert list(outside.iterdir()) == []


@pytest.mark.asyncio
async def test_runner_skips_aggregate_copy_when_artifacts_dir_has_symlink_ancestor(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_artifacts_dir_symlink_ancestor"
    workdir = tmp_path / "wd"
    real_parent = workdir / "real_parent"
    outside = tmp_path / "outside_collected"
    workdir.mkdir(parents=True)
    real_parent.mkdir(parents=True)
    outside.mkdir(parents=True)
    ensure_run_layout(run_dir)

    link_parent = workdir / "link_parent"
    link_parent.symlink_to(real_parent, target_is_directory=True)

    create_outputs_cmd = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('build').mkdir(exist_ok=True); "
        "Path('build/out.txt').write_text('OK', encoding='utf-8')",
    ]
    plan = PlanSpec(
        goal="artifacts dir symlink ancestor skip",
        artifacts_dir="link_parent/collected",
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
    run_copy = run_dir / "artifacts" / "publish" / "build" / "out.txt"
    assert run_copy.read_text(encoding="utf-8") == "OK"
    assert not (real_parent / "collected").exists()
    assert list(outside.iterdir()) == []


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
