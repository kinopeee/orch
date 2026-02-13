from __future__ import annotations

import asyncio
import glob as globlib
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from orch.config.schema import PlanSpec, TaskSpec
from orch.dag.build import build_adjacency
from orch.exec.cancel import cancel_requested
from orch.exec.capture import stream_to_file
from orch.exec.retry import backoff_for_attempt
from orch.state.model import RunState, TaskState
from orch.state.store import load_state, save_state_atomic
from orch.util.errors import StateError
from orch.util.time import duration_sec, now_iso


@dataclass(slots=True)
class TaskResult:
    exit_code: int | None
    timed_out: bool
    canceled: bool
    start_failed: bool
    started_at: str
    ended_at: str
    duration_sec: float


def _terminal_status(task: TaskState) -> bool:
    return task.status in {"SUCCESS", "FAILED", "SKIPPED", "CANCELED"}


def _should_retry(task: TaskSpec, result: TaskResult, attempt: int) -> bool:
    max_attempts = task.retries + 1
    if attempt >= max_attempts:
        return False
    if result.canceled:
        return False
    if result.start_failed:
        return False
    return result.timed_out or result.exit_code not in (0, None)


def _append_attempt_header(log_path: Path, attempt: int, max_attempts: int) -> None:
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n===== attempt {attempt} / {max_attempts} =====\n")
    except OSError:
        return


def _append_text_best_effort(log_path: Path, text: str) -> None:
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        return


def _resolve_task_cwd(task_cwd: str | None, default_cwd: Path) -> Path:
    if task_cwd is None:
        return default_cwd
    cwd = Path(task_cwd)
    if cwd.is_absolute():
        return cwd
    return default_cwd / cwd


def _artifact_relative_path(match: Path, cwd: Path) -> Path:
    def _sanitize_parts(path: Path) -> list[str]:
        anchor = path.anchor
        cleaned: list[str] = []
        for part in path.parts:
            if not part or part == anchor or part == ".":
                continue
            if part == "..":
                cleaned.append("__up__")
            else:
                cleaned.append(part.replace(":", "_"))
        return cleaned

    try:
        rel = match.relative_to(cwd)
    except ValueError:
        if match.is_absolute():
            parts = _sanitize_parts(match)
            if not parts:
                return Path("__abs__", "root")
            return Path("__abs__", *parts)
        parts = _sanitize_parts(match)
        if not parts:
            return Path("__external__", "root")
        return Path("__external__", *parts)

    if any(part == ".." for part in rel.parts):
        parts = _sanitize_parts(match)
        if match.is_absolute():
            return Path("__abs__", *(parts or ["root"]))
        return Path("__external__", *(parts or ["root"]))

    sanitized = _sanitize_parts(rel)
    if not sanitized:
        return Path("root")
    return Path(*sanitized)


def _resolve_artifacts_dir(artifacts_dir: str | None, workdir: Path) -> Path | None:
    if artifacts_dir is None:
        return None
    root = Path(artifacts_dir)
    if root.is_absolute():
        return root
    return workdir / root


def _iter_output_matches(pattern: str, cwd: Path) -> list[Path]:
    try:
        if Path(pattern).is_absolute():
            return [Path(p) for p in globlib.glob(pattern, recursive=True)]
        return list(cwd.glob(pattern))
    except (OSError, ValueError, re.error):
        return []


def _copy_artifacts(task: TaskSpec, run_dir: Path, cwd: Path) -> list[str]:
    copied: list[str] = []
    if not task.outputs:
        return copied
    task_root = run_dir / "artifacts" / task.id
    try:
        task_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return copied
    for pattern in task.outputs:
        matches = _iter_output_matches(pattern, cwd)
        for match in matches:
            if not match.exists() or match.is_dir():
                continue
            rel = _artifact_relative_path(match, cwd)
            dest = task_root / rel
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(match, dest)
            except OSError:
                continue
            copied.append(str(dest.relative_to(run_dir)))
    return sorted(set(copied))


def _copy_to_aggregate_dir(
    task: TaskSpec,
    cwd: Path,
    *,
    aggregate_root: Path,
) -> None:
    task_root = aggregate_root / task.id
    try:
        task_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    for pattern in task.outputs:
        matches = _iter_output_matches(pattern, cwd)
        for match in matches:
            if not match.exists() or match.is_dir():
                continue
            rel = _artifact_relative_path(match, cwd)
            dest = task_root / rel
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(match, dest)
            except OSError:
                continue


def _copy_to_aggregate_dir_best_effort(
    task: TaskSpec,
    cwd: Path,
    *,
    aggregate_root: Path,
) -> None:
    try:
        _copy_to_aggregate_dir(task, cwd, aggregate_root=aggregate_root)
    except OSError:
        return


def _finalize_run_status(state: RunState) -> None:
    statuses = [task.status for task in state.tasks.values()]
    if any(status == "CANCELED" for status in statuses):
        state.status = "CANCELED"
    elif any(status == "FAILED" for status in statuses) or any(
        status == "SKIPPED" for status in statuses
    ):
        state.status = "FAILED"
    elif statuses and all(status == "SUCCESS" for status in statuses):
        state.status = "SUCCESS"
    else:
        state.status = "FAILED"


def _persist(run_dir: Path, state: RunState) -> None:
    state.updated_at = now_iso()
    save_state_atomic(run_dir, state)


def _initial_state(
    plan: PlanSpec,
    run_dir: Path,
    *,
    max_parallel: int,
    fail_fast: bool,
    workdir: Path,
) -> RunState:
    ts = now_iso()
    run_id = run_dir.name
    resolved_home = run_dir.parent.parent.resolve()
    tasks = {
        task.id: TaskState(
            status="PENDING",
            depends_on=task.depends_on,
            cmd=task.cmd,
            cwd=task.cwd,
            env=task.env,
            timeout_sec=task.timeout_sec,
            retries=task.retries,
            retry_backoff_sec=task.retry_backoff_sec,
            outputs=task.outputs,
            stdout_path=f"logs/{task.id}.out.log",
            stderr_path=f"logs/{task.id}.err.log",
        )
        for task in plan.tasks
    }
    return RunState(
        run_id=run_id,
        created_at=ts,
        updated_at=ts,
        status="RUNNING",
        goal=plan.goal,
        plan_relpath="plan.yaml",
        home=str(resolved_home),
        workdir=str(workdir),
        max_parallel=max_parallel,
        fail_fast=fail_fast,
        tasks=tasks,
    )


def _prepare_resume_state(state: RunState) -> None:
    for task in state.tasks.values():
        if task.status == "RUNNING":
            task.status = "FAILED"
            task.canceled = False
            task.timed_out = False
            task.skip_reason = "previous_run_interrupted"
            task.ended_at = now_iso()


def _rerun_set(
    plan: PlanSpec,
    state: RunState,
    *,
    failed_only: bool,
    dependents: dict[str, list[str]],
) -> set[str]:
    if not failed_only:
        return {task.id for task in plan.tasks if state.tasks[task.id].status != "SUCCESS"}

    seeds = {task.id for task in plan.tasks if state.tasks[task.id].status == "FAILED"}
    to_rerun = set(seeds)
    queue = list(seeds)
    while queue:
        current = queue.pop()
        for child in dependents.get(current, []):
            if child in to_rerun:
                continue
            if state.tasks[child].status != "SUCCESS":
                to_rerun.add(child)
                queue.append(child)
    return to_rerun


def _reset_for_rerun(task_state: TaskState) -> None:
    task_state.status = "PENDING"
    task_state.started_at = None
    task_state.ended_at = None
    task_state.duration_sec = None
    task_state.exit_code = None
    task_state.timed_out = False
    task_state.canceled = False
    task_state.skip_reason = None
    task_state.artifact_paths = []


def _validate_resume_state_matches_plan(plan: PlanSpec, state: RunState) -> None:
    plan_ids = {task.id for task in plan.tasks}
    state_ids = set(state.tasks.keys())
    missing = sorted(plan_ids - state_ids)
    if missing:
        raise StateError(f"missing task state entries: {missing}")
    unknown = sorted(state_ids - plan_ids)
    if unknown:
        raise StateError(f"unknown task state entries: {unknown}")


async def run_task(
    task: TaskSpec,
    run_dir: Path,
    *,
    attempt: int,
    default_cwd: Path,
) -> TaskResult:
    started_dt = datetime.now().astimezone()
    started_iso = started_dt.isoformat(timespec="seconds")
    out_path = run_dir / "logs" / f"{task.id}.out.log"
    err_path = run_dir / "logs" / f"{task.id}.err.log"
    max_attempts = task.retries + 1
    _append_attempt_header(out_path, attempt, max_attempts)
    _append_attempt_header(err_path, attempt, max_attempts)

    merged_env = os.environ.copy()
    if task.env:
        merged_env.update(task.env)
    cwd = _resolve_task_cwd(task.cwd, default_cwd)
    try:
        proc = await asyncio.create_subprocess_exec(
            *task.cmd,
            cwd=str(cwd),
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, ValueError) as exc:
        _append_text_best_effort(err_path, f"failed to start process: {exc}\n")
        ended_dt = datetime.now().astimezone()
        return TaskResult(
            exit_code=127,
            timed_out=False,
            canceled=False,
            start_failed=True,
            started_at=started_iso,
            ended_at=ended_dt.isoformat(timespec="seconds"),
            duration_sec=duration_sec(started_dt, ended_dt),
        )

    out_stream = asyncio.create_task(stream_to_file(proc.stdout, out_path))
    err_stream = asyncio.create_task(stream_to_file(proc.stderr, err_path))
    timed_out = False
    canceled = False
    exit_code: int | None = None

    while True:
        if proc.returncode is not None:
            exit_code = proc.returncode
            break
        if cancel_requested(run_dir):
            canceled = True
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except TimeoutError:
                proc.kill()
                await proc.wait()
            exit_code = proc.returncode
            break
        if task.timeout_sec is not None:
            elapsed = (datetime.now().astimezone() - started_dt).total_seconds()
            if elapsed > task.timeout_sec:
                timed_out = True
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
                exit_code = None
                break
        await asyncio.sleep(0.1)

    await asyncio.gather(out_stream, err_stream, return_exceptions=True)
    ended_dt = datetime.now().astimezone()
    return TaskResult(
        exit_code=exit_code,
        timed_out=timed_out,
        canceled=canceled,
        start_failed=False,
        started_at=started_iso,
        ended_at=ended_dt.isoformat(timespec="seconds"),
        duration_sec=duration_sec(started_dt, ended_dt),
    )


async def run_plan(
    plan: PlanSpec,
    run_dir: Path,
    *,
    max_parallel: int,
    fail_fast: bool,
    workdir: Path,
    resume: bool,
    failed_only: bool,
) -> RunState:
    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    resolved_workdir = workdir.resolve()

    dependents, _ = build_adjacency(plan)
    spec_by_id = {task.id: task for task in plan.tasks}
    aggregate_root = _resolve_artifacts_dir(plan.artifacts_dir, resolved_workdir)

    if resume:
        (run_dir / "cancel.request").unlink(missing_ok=True)
        state = load_state(run_dir)
        _validate_resume_state_matches_plan(plan, state)
        _prepare_resume_state(state)
        state.status = "RUNNING"
        state.max_parallel = max_parallel
        state.fail_fast = fail_fast
        state.workdir = str(resolved_workdir)
        rerun = _rerun_set(plan, state, failed_only=failed_only, dependents=dependents)
        for task_id in rerun:
            _reset_for_rerun(state.tasks[task_id])
    else:
        state = _initial_state(
            plan,
            run_dir,
            max_parallel=max_parallel,
            fail_fast=fail_fast,
            workdir=resolved_workdir,
        )

    _persist(run_dir, state)

    rerunnable = {task.id for task in plan.tasks if state.tasks[task.id].status == "PENDING"}
    active = set(rerunnable)
    dep_remaining: dict[str, int] = {}
    for task in plan.tasks:
        if task.id not in active:
            continue
        dep_remaining[task.id] = sum(1 for dep in task.depends_on if dep in active)

    ready = [task_id for task_id, dep_count in dep_remaining.items() if dep_count == 0]
    running: dict[str, asyncio.Task[TaskResult]] = {}
    sem = asyncio.Semaphore(max_parallel)
    cancel_mode = False
    fail_fast_mode = False

    while active or running:
        if cancel_requested(run_dir):
            cancel_mode = True

        if cancel_mode:
            for task_id in list(active):
                if task_id in running:
                    continue
                task_state = state.tasks[task_id]
                task_state.status = "CANCELED"
                task_state.canceled = True
                task_state.skip_reason = "run_canceled"
                task_state.ended_at = now_iso()
                active.remove(task_id)
                for child in dependents.get(task_id, []):
                    if child in dep_remaining:
                        dep_remaining[child] -= 1
                        if dep_remaining[child] == 0 and child in active:
                            ready.append(child)
            _persist(run_dir, state)

        while ready and len(running) < max_parallel and not cancel_mode:
            task_id = ready.pop(0)
            if task_id not in active or task_id in running:
                continue
            task = spec_by_id[task_id]
            task_state = state.tasks[task_id]
            dep_states = [state.tasks[dep].status for dep in task.depends_on]
            if any(dep_status != "SUCCESS" for dep_status in dep_states):
                task_state.status = "SKIPPED"
                task_state.skip_reason = "dependency_not_success"
                task_state.ended_at = now_iso()
                active.remove(task_id)
                for child in dependents.get(task_id, []):
                    if child in dep_remaining:
                        dep_remaining[child] -= 1
                        if dep_remaining[child] == 0 and child in active:
                            ready.append(child)
                _persist(run_dir, state)
                continue
            if fail_fast_mode:
                task_state.status = "SKIPPED"
                task_state.skip_reason = "fail_fast"
                task_state.ended_at = now_iso()
                active.remove(task_id)
                _persist(run_dir, state)
                continue

            async def _run_with_sem(spec: TaskSpec, attempt: int) -> TaskResult:
                async with sem:
                    return await run_task(
                        spec,
                        run_dir,
                        attempt=attempt,
                        default_cwd=resolved_workdir,
                    )

            task_state.status = "RUNNING"
            task_state.started_at = now_iso()
            task_state.attempts += 1
            attempt = task_state.attempts
            _persist(run_dir, state)
            running[task_id] = asyncio.create_task(_run_with_sem(task, attempt))

        if not running:
            if not ready:
                if active:
                    for task_id in list(active):
                        task_state = state.tasks[task_id]
                        task_state.status = "SKIPPED"
                        task_state.skip_reason = "unresolvable_dependencies"
                        task_state.ended_at = now_iso()
                        active.remove(task_id)
                    _persist(run_dir, state)
                break
            await asyncio.sleep(0.05)
            continue

        done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
        done_by_id = {task_id: fut for task_id, fut in running.items() if fut in done}

        for task_id, fut in done_by_id.items():
            del running[task_id]
            task = spec_by_id[task_id]
            task_state = state.tasks[task_id]
            try:
                result = fut.result()
            except Exception as exc:
                ended_dt = datetime.now().astimezone()
                started_iso = task_state.started_at or ended_dt.isoformat(timespec="seconds")
                try:
                    started_dt = datetime.fromisoformat(started_iso)
                    elapsed = duration_sec(started_dt, ended_dt)
                except ValueError:
                    elapsed = 0.0
                if task_state.stderr_path is not None:
                    _append_text_best_effort(
                        run_dir / task_state.stderr_path,
                        f"runner exception: {exc}\n",
                    )
                task_state.skip_reason = "runner_exception"
                result = TaskResult(
                    exit_code=70,
                    timed_out=False,
                    canceled=False,
                    start_failed=True,
                    started_at=started_iso,
                    ended_at=ended_dt.isoformat(timespec="seconds"),
                    duration_sec=elapsed,
                )
            task_state.ended_at = result.ended_at
            task_state.duration_sec = result.duration_sec
            task_state.exit_code = result.exit_code
            task_state.timed_out = result.timed_out
            task_state.canceled = result.canceled
            task_cwd = _resolve_task_cwd(task.cwd, resolved_workdir)

            if _should_retry(task, result, task_state.attempts):
                delay = backoff_for_attempt(task_state.attempts - 1, task.retry_backoff_sec)
                task_state.status = "READY"
                _persist(run_dir, state)
                await asyncio.sleep(delay)
                task_state.status = "PENDING"
                ready.append(task_id)
                _persist(run_dir, state)
                continue

            if result.canceled:
                task_state.status = "CANCELED"
                task_state.skip_reason = "run_canceled"
                cancel_mode = True
            else:
                task_state.artifact_paths = _copy_artifacts(task, run_dir, task_cwd)
                if aggregate_root is not None:
                    _copy_to_aggregate_dir_best_effort(
                        task,
                        task_cwd,
                        aggregate_root=aggregate_root,
                    )
                if result.exit_code == 0 and not result.timed_out:
                    task_state.status = "SUCCESS"
                else:
                    task_state.status = "FAILED"
                    if result.start_failed and task_state.skip_reason is None:
                        task_state.skip_reason = "process_start_failed"
                    if fail_fast:
                        fail_fast_mode = True

            if task_id in active:
                active.remove(task_id)
            for child in dependents.get(task_id, []):
                if child in dep_remaining:
                    dep_remaining[child] -= 1
                    if dep_remaining[child] == 0 and child in active:
                        ready.append(child)

            if fail_fast_mode:
                for pending_id in list(active):
                    if pending_id in running:
                        continue
                    pending_state = state.tasks[pending_id]
                    pending_state.status = "SKIPPED"
                    pending_state.skip_reason = "fail_fast"
                    pending_state.ended_at = now_iso()
                    active.remove(pending_id)
                    for child in dependents.get(pending_id, []):
                        if child in dep_remaining:
                            dep_remaining[child] -= 1
                            if dep_remaining[child] == 0 and child in active:
                                ready.append(child)

            _persist(run_dir, state)

    _finalize_run_status(state)
    _persist(run_dir, state)
    return state
