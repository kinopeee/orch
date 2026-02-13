from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from orch.config.loader import load_plan
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
from orch.util.paths import ensure_run_layout, run_dir
from orch.util.tail import tail_lines

app = typer.Typer(help="CLI agent task orchestrator")
console = Console()


def _exit_code_for_state(state: RunState) -> int:
    if state.status == "SUCCESS":
        return 0
    if state.status == "CANCELED":
        return 4
    return 3


def _state_to_jsonable(state: RunState) -> dict[str, Any]:
    return state.to_dict()


def _write_report(state: RunState, current_run_dir: Path) -> Path:
    summary = build_summary(state, current_run_dir)
    md = render_markdown(summary)
    report_path = current_run_dir / "report" / "final_report.md"
    report_path.write_text(md + "\n", encoding="utf-8")
    return report_path


def _run_exists(current_run_dir: Path) -> bool:
    return current_run_dir.is_dir() and (
        (current_run_dir / "state.json").exists() or (current_run_dir / "plan.yaml").exists()
    )


def _resolve_workdir_or_exit(workdir: Path) -> Path:
    resolved = workdir.resolve()
    if not resolved.is_dir():
        console.print(f"[red]Invalid workdir:[/red] {workdir}")
        raise typer.Exit(2)
    return resolved


@app.command()
def run(
    plan_path: Annotated[Path, typer.Argument(exists=True)],
    max_parallel: Annotated[int, typer.Option("--max-parallel", min=1)] = 4,
    home: Annotated[Path, typer.Option("--home")] = Path(".orch"),
    workdir: Annotated[Path, typer.Option("--workdir")] = Path("."),
    fail_fast: Annotated[bool, typer.Option("--fail-fast/--no-fail-fast")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    try:
        plan = load_plan(plan_path)
        dependents, in_degree = build_adjacency(plan)
        order = assert_acyclic([task.id for task in plan.tasks], dependents, in_degree)
    except PlanError as exc:
        console.print(f"[red]Plan validation error:[/red] {exc}")
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
    ensure_run_layout(current_run_dir)
    shutil.copy2(plan_path, current_run_dir / "plan.yaml")

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
    report_path = _write_report(state, current_run_dir)
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
    except (StateError, FileNotFoundError) as exc:
        console.print(f"[red]Run not found or broken:[/red] {exc}")
        raise typer.Exit(2) from exc
    except PlanError as exc:
        console.print(f"[red]Plan validation error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except RunConflictError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc

    report_path = _write_report(state, current_run_dir)
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
    current_run_dir = run_dir(home, run_id)
    try:
        with run_lock(current_run_dir, retries=5, retry_interval=0.1):
            state = load_state(current_run_dir)
    except (StateError, FileNotFoundError) as exc:
        console.print(f"[red]Failed to load state:[/red] {exc}")
        raise typer.Exit(2) from exc
    except RunConflictError:
        try:
            state = load_state(current_run_dir)
        except (StateError, FileNotFoundError) as exc:
            console.print(f"[red]Failed to load state:[/red] {exc}")
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
    tail: Annotated[int, typer.Option("--tail")] = 100,
) -> None:
    current_run_dir = run_dir(home, run_id)
    try:
        state = load_state(current_run_dir)
    except (StateError, FileNotFoundError) as exc:
        console.print(f"[red]Failed to load state:[/red] {exc}")
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
    current_run_dir = run_dir(home, run_id)
    if not _run_exists(current_run_dir):
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(2)
    write_cancel_request(current_run_dir)
    console.print(f"cancel requested: [bold]{run_id}[/bold]")


if __name__ == "__main__":
    app()
