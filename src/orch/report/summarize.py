from __future__ import annotations

from pathlib import Path

from orch.state.model import RunState
from orch.util.tail import tail_lines


def build_summary(state: RunState, run_dir: Path) -> dict[str, object]:
    tasks_rows: list[dict[str, object]] = []
    problem_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []

    for task_id, task in state.tasks.items():
        tasks_rows.append(
            {
                "id": task_id,
                "status": task.status,
                "attempts": task.attempts,
                "duration_sec": task.duration_sec,
                "exit_code": task.exit_code,
                "timed_out": task.timed_out,
                "stdout_path": task.stdout_path,
                "stderr_path": task.stderr_path,
            }
        )
        if task.status in {"FAILED", "SKIPPED", "CANCELED"}:
            stderr_tail = (
                tail_lines(run_dir / task.stderr_path, 50) if task.stderr_path is not None else []
            )
            problem_rows.append(
                {
                    "id": task_id,
                    "status": task.status,
                    "skip_reason": task.skip_reason,
                    "stderr_tail": stderr_tail,
                }
            )

        for artifact in task.artifact_paths:
            artifact_rows.append({"task_id": task_id, "path": artifact})

    return {
        "run": {
            "run_id": state.run_id,
            "goal": state.goal,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "status": state.status,
            "max_parallel": state.max_parallel,
            "fail_fast": state.fail_fast,
            "workdir": state.workdir,
        },
        "tasks": tasks_rows,
        "problems": problem_rows,
        "artifacts": artifact_rows,
    }
