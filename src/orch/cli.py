from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import stat
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from orch.config.loader import load_plan
from orch.config.schema import PlanSpec, TaskSpec
from orch.dag.build import build_adjacency
from orch.dag.validate import assert_acyclic
from orch.exec.cancel import write_cancel_request
from orch.exec.runner import run_plan
from orch.report.render_md import render_markdown
from orch.report.summarize import build_summary
from orch.state.lock import run_lock
from orch.state.model import RunState
from orch.state.store import load_state
from orch.util.errors import PlanError, RunConflictError, StateError
from orch.util.ids import new_run_id
from orch.util.path_guard import has_symlink_ancestor, is_symlink_path
from orch.util.paths import ensure_run_layout, run_dir
from orch.util.tail import tail_lines

app = typer.Typer(help="CLI agent task orchestrator")
console = Console()
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RUN_ID_MAX_LEN = 128
_SYMLINK_HINT_PATTERN = re.compile(r"\bsymlink\w*\b|\bsymbolic(?:[\s_-]+)?link\w*\b", re.IGNORECASE)


def _exit_code_for_state(state: RunState) -> int:
    if state.status == "SUCCESS":
        return 0
    if state.status == "CANCELED":
        return 4
    return 3


def _state_to_jsonable(state: RunState) -> dict[str, Any]:
    return state.to_dict()


def _mentions_symlink(detail: str) -> bool:
    return _SYMLINK_HINT_PATTERN.search(detail) is not None


def _render_plan_error(exc: PlanError) -> str:
    detail = str(exc)
    if _mentions_symlink(detail):
        return "invalid plan path"
    return detail


def _render_runtime_error_detail(exc: BaseException) -> str:
    detail = str(exc)
    if _mentions_symlink(detail):
        return "invalid run path"
    return detail


def _task_to_plan_dict(task: TaskSpec) -> dict[str, object]:
    task_data: dict[str, object] = {
        "id": task.id,
        "cmd": task.cmd,
        "depends_on": task.depends_on,
        "retries": task.retries,
        "retry_backoff_sec": task.retry_backoff_sec,
        "outputs": task.outputs,
    }
    if task.cwd is not None:
        task_data["cwd"] = task.cwd
    if task.env is not None:
        task_data["env"] = task.env
    if task.timeout_sec is not None:
        task_data["timeout_sec"] = task.timeout_sec
    return task_data


def _write_plan_snapshot(plan: PlanSpec, destination: Path) -> None:
    plan_data: dict[str, object] = {"tasks": [_task_to_plan_dict(task) for task in plan.tasks]}
    if plan.goal is not None:
        plan_data["goal"] = plan.goal
    if plan.artifacts_dir is not None:
        plan_data["artifacts_dir"] = plan.artifacts_dir
    payload = yaml.safe_dump(plan_data, sort_keys=False, allow_unicode=True)
    if has_symlink_ancestor(destination):
        raise OSError(f"plan snapshot path must not include symlink: {destination}")
    if is_symlink_path(destination.parent) or is_symlink_path(destination):
        raise OSError(f"plan snapshot path must not be symlink: {destination}")
    try:
        destination_meta = destination.lstat()
    except FileNotFoundError:
        destination_meta = None
    except (OSError, RuntimeError) as exc:
        raise OSError(f"failed to prepare plan snapshot path: {destination}") from exc
    if destination_meta is not None and not stat.S_ISREG(destination_meta.st_mode):
        raise OSError(f"plan snapshot path must be regular file: {destination}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(str(destination), flags, 0o600)
        opened_meta = os.fstat(fd)
        if not stat.S_ISREG(opened_meta.st_mode):
            raise OSError(f"plan snapshot path must be regular file: {destination}")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            f.write(payload)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError(f"plan snapshot path must not be symlink: {destination}") from exc
        if exc.errno == errno.ENXIO:
            raise OSError(f"plan snapshot path must be regular file: {destination}") from exc
        raise
    except RuntimeError as exc:
        raise OSError(f"failed to write plan snapshot: {destination}") from exc
    finally:
        if fd is not None:
            with suppress(OSError, RuntimeError):
                os.close(fd)


def _write_report(state: RunState, current_run_dir: Path) -> Path:
    summary = build_summary(state, current_run_dir)
    md = render_markdown(summary)
    report_path = current_run_dir / "report" / "final_report.md"
    if has_symlink_ancestor(report_path):
        raise OSError(f"report path must not include symlink: {report_path}")
    if is_symlink_path(report_path.parent) or is_symlink_path(report_path):
        raise OSError(f"report path must not be symlink: {report_path}")
    try:
        report_meta = report_path.lstat()
    except FileNotFoundError:
        report_meta = None
    except (OSError, RuntimeError) as exc:
        raise OSError(f"failed to prepare report path: {report_path}") from exc
    if report_meta is not None and not stat.S_ISREG(report_meta.st_mode):
        raise OSError(f"report path must be regular file: {report_path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(str(report_path), flags, 0o600)
        opened_meta = os.fstat(fd)
        if not stat.S_ISREG(opened_meta.st_mode):
            raise OSError(f"report path must be regular file: {report_path}")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            f.write(md + "\n")
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError(f"report path must not be symlink: {report_path}") from exc
        if exc.errno == errno.ENXIO:
            raise OSError(f"report path must be regular file: {report_path}") from exc
        raise
    except RuntimeError as exc:
        raise OSError(f"failed to write report: {report_path}") from exc
    finally:
        if fd is not None:
            with suppress(OSError, RuntimeError):
                os.close(fd)
    return report_path


def _run_exists(current_run_dir: Path) -> bool:
    if has_symlink_ancestor(current_run_dir):
        return False
    try:
        run_meta = current_run_dir.lstat()
    except (OSError, RuntimeError):
        return False
    if stat.S_ISLNK(run_meta.st_mode) or not stat.S_ISDIR(run_meta.st_mode):
        return False

    def _is_regular_non_symlink(path: Path) -> bool:
        try:
            meta = path.lstat()
        except (OSError, RuntimeError):
            return False
        return stat.S_ISREG(meta.st_mode)

    return _is_regular_non_symlink(current_run_dir / "state.json") or _is_regular_non_symlink(
        current_run_dir / "plan.yaml"
    )


def _resolve_workdir_or_exit(workdir: Path) -> Path:
    try:
        resolved = workdir.resolve()
        meta = resolved.lstat()
    except (OSError, RuntimeError) as exc:
        console.print(f"[red]Invalid workdir:[/red] {workdir}")
        raise typer.Exit(2) from exc
    if not stat.S_ISDIR(meta.st_mode):
        console.print(f"[red]Invalid workdir:[/red] {workdir}")
        raise typer.Exit(2)
    return resolved


def _validate_home_or_exit(home: Path) -> None:
    try:
        unsafe_home = is_symlink_path(home) or has_symlink_ancestor(home)
    except (OSError, RuntimeError) as exc:
        console.print(f"[red]Invalid home:[/red] {home}")
        raise typer.Exit(2) from exc
    if unsafe_home:
        console.print(f"[red]Invalid home:[/red] {home}")
        raise typer.Exit(2)
    to_check = [home, *home.parents]
    for candidate in to_check:
        try:
            meta = candidate.lstat()
        except FileNotFoundError:
            continue
        except (OSError, RuntimeError) as exc:
            console.print(f"[red]Invalid home:[/red] {home}")
            raise typer.Exit(2) from exc
        if stat.S_ISDIR(meta.st_mode):
            break
        console.print(f"[red]Invalid home:[/red] {home}")
        raise typer.Exit(2)


def _validate_run_id_or_exit(run_id: str) -> None:
    if len(run_id) > _RUN_ID_MAX_LEN or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        console.print(f"[red]Invalid run_id:[/red] {run_id}")
        raise typer.Exit(2)


@app.command()
def run(
    plan_path: Annotated[Path, typer.Argument(exists=True)],
    max_parallel: Annotated[int, typer.Option("--max-parallel", min=1)] = 4,
    home: Annotated[Path, typer.Option("--home")] = Path(".orch"),
    workdir: Annotated[Path, typer.Option("--workdir")] = Path("."),
    fail_fast: Annotated[bool, typer.Option("--fail-fast/--no-fail-fast")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    _validate_home_or_exit(home)
    try:
        plan = load_plan(plan_path)
        dependents, in_degree = build_adjacency(plan)
        order = assert_acyclic([task.id for task in plan.tasks], dependents, in_degree)
    except PlanError as exc:
        console.print(f"[red]Plan validation error:[/red] {_render_plan_error(exc)}")
        raise typer.Exit(2) from exc

    if dry_run:
        table = Table(title="Dry Run - Topological Order")
        table.add_column("#")
        table.add_column("task_id")
        for idx, task_id in enumerate(order, start=1):
            table.add_row(str(idx), task_id)
        console.print(table)
        raise typer.Exit(0)

    resolved_workdir = _resolve_workdir_or_exit(workdir)
    run_id = new_run_id(datetime.now().astimezone())
    current_run_dir = run_dir(home, run_id)
    try:
        ensure_run_layout(current_run_dir)
        _write_plan_snapshot(plan, current_run_dir / "plan.yaml")
    except (OSError, RuntimeError) as exc:
        console.print(f"[red]Failed to initialize run:[/red] {_render_runtime_error_detail(exc)}")
        raise typer.Exit(2) from exc

    try:
        state = asyncio.run(
            run_plan(
                plan,
                current_run_dir,
                max_parallel=max_parallel,
                fail_fast=fail_fast,
                workdir=resolved_workdir,
                resume=False,
                failed_only=False,
            )
        )
    except (OSError, RuntimeError) as exc:
        console.print(f"[red]Run execution failed:[/red] {_render_runtime_error_detail(exc)}")
        raise typer.Exit(2) from exc
    try:
        report_path = _write_report(state, current_run_dir)
    except (OSError, RuntimeError) as exc:
        console.print(
            f"[yellow]Warning:[/yellow] failed to write report: {_render_runtime_error_detail(exc)}"
        )
        report_path = current_run_dir / "report" / "final_report.md"
    console.print(f"run_id: [bold]{run_id}[/bold]")
    console.print(f"state: [bold]{state.status}[/bold]")
    console.print(f"report: {report_path}")
    raise typer.Exit(_exit_code_for_state(state))


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument()],
    home: Annotated[Path, typer.Option("--home")] = Path(".orch"),
    max_parallel: Annotated[int, typer.Option("--max-parallel", min=1)] = 4,
    workdir: Annotated[Path, typer.Option("--workdir")] = Path("."),
    fail_fast: Annotated[bool, typer.Option("--fail-fast/--no-fail-fast")] = False,
    failed_only: Annotated[bool, typer.Option("--failed-only")] = False,
) -> None:
    _validate_run_id_or_exit(run_id)
    _validate_home_or_exit(home)
    resolved_workdir = _resolve_workdir_or_exit(workdir)
    current_run_dir = run_dir(home, run_id)
    try:
        with run_lock(current_run_dir):
            plan = load_plan(current_run_dir / "plan.yaml")
            dependents, in_degree = build_adjacency(plan)
            assert_acyclic([task.id for task in plan.tasks], dependents, in_degree)
            state = asyncio.run(
                run_plan(
                    plan,
                    current_run_dir,
                    max_parallel=max_parallel,
                    fail_fast=fail_fast,
                    workdir=resolved_workdir,
                    resume=True,
                    failed_only=failed_only,
                )
            )
    except (StateError, FileNotFoundError, OSError, RuntimeError) as exc:
        console.print(f"[red]Run not found or broken:[/red] {_render_runtime_error_detail(exc)}")
        raise typer.Exit(2) from exc
    except PlanError as exc:
        console.print(f"[red]Plan validation error:[/red] {_render_plan_error(exc)}")
        raise typer.Exit(2) from exc
    except RunConflictError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc

    try:
        report_path = _write_report(state, current_run_dir)
    except (OSError, RuntimeError) as exc:
        console.print(
            f"[yellow]Warning:[/yellow] failed to write report: {_render_runtime_error_detail(exc)}"
        )
        report_path = current_run_dir / "report" / "final_report.md"
    console.print(f"run_id: [bold]{run_id}[/bold]")
    console.print(f"state: [bold]{state.status}[/bold]")
    console.print(f"report: {report_path}")
    raise typer.Exit(_exit_code_for_state(state))


@app.command()
def status(
    run_id: Annotated[str, typer.Argument()],
    home: Annotated[Path, typer.Option("--home")] = Path(".orch"),
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    _validate_run_id_or_exit(run_id)
    _validate_home_or_exit(home)
    current_run_dir = run_dir(home, run_id)
    try:
        with run_lock(current_run_dir, retries=5, retry_interval=0.1):
            state = load_state(current_run_dir)
    except (StateError, FileNotFoundError, OSError, RuntimeError) as exc:
        console.print(f"[red]Failed to load state:[/red] {_render_runtime_error_detail(exc)}")
        raise typer.Exit(2) from exc
    except RunConflictError:
        try:
            state = load_state(current_run_dir)
        except (StateError, FileNotFoundError, OSError, RuntimeError) as exc:
            console.print(f"[red]Failed to load state:[/red] {_render_runtime_error_detail(exc)}")
            raise typer.Exit(2) from exc

    if as_json:
        typer.echo(json.dumps(_state_to_jsonable(state), ensure_ascii=False, indent=2))
        raise typer.Exit(0)

    table = Table(title=f"Run Status: {run_id}")
    table.add_column("task_id")
    table.add_column("status")
    table.add_column("attempts", justify="right")
    table.add_column("duration_sec", justify="right")
    table.add_column("exit_code", justify="right")
    for task_id, task in state.tasks.items():
        table.add_row(
            task_id,
            task.status,
            str(task.attempts),
            "-" if task.duration_sec is None else str(task.duration_sec),
            "-" if task.exit_code is None else str(task.exit_code),
        )
    console.print(table)


@app.command()
def logs(
    run_id: Annotated[str, typer.Argument()],
    home: Annotated[Path, typer.Option("--home")] = Path(".orch"),
    task: Annotated[str | None, typer.Option("--task")] = None,
    tail: Annotated[int, typer.Option("--tail", min=1)] = 100,
) -> None:
    _validate_run_id_or_exit(run_id)
    _validate_home_or_exit(home)
    current_run_dir = run_dir(home, run_id)
    try:
        with run_lock(current_run_dir, retries=5, retry_interval=0.1):
            state = load_state(current_run_dir)
    except (StateError, FileNotFoundError, OSError, RuntimeError) as exc:
        console.print(f"[red]Failed to load state:[/red] {_render_runtime_error_detail(exc)}")
        raise typer.Exit(2) from exc
    except RunConflictError:
        try:
            state = load_state(current_run_dir)
        except (StateError, FileNotFoundError, OSError, RuntimeError) as exc:
            console.print(f"[red]Failed to load state:[/red] {_render_runtime_error_detail(exc)}")
            raise typer.Exit(2) from exc
    task_ids = [task] if task else list(state.tasks.keys())
    missing_task = False
    for task_id in task_ids:
        if task_id not in state.tasks:
            console.print(f"[yellow]unknown task:[/yellow] {task_id}")
            missing_task = True
            continue
        task_state = state.tasks[task_id]
        out_lines = (
            tail_lines(current_run_dir / task_state.stdout_path, tail)
            if task_state.stdout_path is not None
            else []
        )
        err_lines = (
            tail_lines(current_run_dir / task_state.stderr_path, tail)
            if task_state.stderr_path is not None
            else []
        )
        console.rule(f"{task_id} :: stdout")
        if out_lines:
            console.print("\n".join(out_lines))
        else:
            console.print("(empty)")
        console.rule(f"{task_id} :: stderr")
        if err_lines:
            console.print("\n".join(err_lines))
        else:
            console.print("(empty)")
    if task is not None and missing_task:
        raise typer.Exit(2)


@app.command()
def cancel(
    run_id: Annotated[str, typer.Argument()],
    home: Annotated[Path, typer.Option("--home")] = Path(".orch"),
) -> None:
    _validate_run_id_or_exit(run_id)
    _validate_home_or_exit(home)
    current_run_dir = run_dir(home, run_id)
    try:
        exists = _run_exists(current_run_dir)
    except (OSError, RuntimeError) as exc:
        console.print(f"[red]Failed to inspect run:[/red] {_render_runtime_error_detail(exc)}")
        raise typer.Exit(2) from exc
    if not exists:
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(2)
    try:
        write_cancel_request(current_run_dir)
    except (OSError, RuntimeError) as exc:
        console.print(f"[red]Failed to request cancel:[/red] {_render_runtime_error_detail(exc)}")
        raise typer.Exit(2) from exc
    console.print(f"cancel requested: [bold]{run_id}[/bold]")


if __name__ == "__main__":
    app()
