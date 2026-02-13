"""Dataclasses representing run state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from orch.config.schema import TaskSpec

RunStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED", "CANCELED"]
TaskStatus = Literal["PENDING", "READY", "RUNNING", "SUCCESS", "FAILED", "SKIPPED", "CANCELED"]


@dataclass(slots=True)
class TaskState:
    """Mutable task execution state."""

    status: TaskStatus
    depends_on: list[str]
    cmd: list[str]
    cwd: str
    env: dict[str, str] | None
    timeout_sec: float | None
    retries: int
    retry_backoff_sec: list[float]
    outputs: list[str]
    attempts: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    duration_sec: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    canceled: bool = False
    skip_reason: str | None = None
    stdout_path: str = ""
    stderr_path: str = ""
    artifact_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, task: TaskSpec, default_cwd: str) -> "TaskState":
        cwd = task.cwd if task.cwd is not None else default_cwd
        return cls(
            status="PENDING",
            depends_on=list(task.depends_on),
            cmd=list(task.cmd),
            cwd=cwd,
            env=dict(task.env) if task.env is not None else None,
            timeout_sec=task.timeout_sec,
            retries=task.retries,
            retry_backoff_sec=list(task.retry_backoff_sec),
            outputs=list(task.outputs),
            stdout_path=f"logs/{task.id}.out.log",
            stderr_path=f"logs/{task.id}.err.log",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "depends_on": self.depends_on,
            "cmd": self.cmd,
            "cwd": self.cwd,
            "env": self.env,
            "timeout_sec": self.timeout_sec,
            "retries": self.retries,
            "retry_backoff_sec": self.retry_backoff_sec,
            "outputs": self.outputs,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_sec": self.duration_sec,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "canceled": self.canceled,
            "skip_reason": self.skip_reason,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "artifact_paths": self.artifact_paths,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "TaskState":
        return cls(
            status=raw["status"],  # type: ignore[arg-type]
            depends_on=list(raw["depends_on"]),  # type: ignore[arg-type]
            cmd=list(raw["cmd"]),  # type: ignore[arg-type]
            cwd=raw["cwd"],  # type: ignore[arg-type]
            env=raw.get("env"),  # type: ignore[arg-type]
            timeout_sec=raw.get("timeout_sec"),  # type: ignore[arg-type]
            retries=raw["retries"],  # type: ignore[arg-type]
            retry_backoff_sec=list(raw["retry_backoff_sec"]),  # type: ignore[arg-type]
            outputs=list(raw["outputs"]),  # type: ignore[arg-type]
            attempts=raw.get("attempts", 0),  # type: ignore[arg-type]
            started_at=raw.get("started_at"),  # type: ignore[arg-type]
            ended_at=raw.get("ended_at"),  # type: ignore[arg-type]
            duration_sec=raw.get("duration_sec"),  # type: ignore[arg-type]
            exit_code=raw.get("exit_code"),  # type: ignore[arg-type]
            timed_out=bool(raw.get("timed_out", False)),
            canceled=bool(raw.get("canceled", False)),
            skip_reason=raw.get("skip_reason"),  # type: ignore[arg-type]
            stdout_path=raw.get("stdout_path", ""),  # type: ignore[arg-type]
            stderr_path=raw.get("stderr_path", ""),  # type: ignore[arg-type]
            artifact_paths=list(raw.get("artifact_paths", [])),  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class RunState:
    """Run-level mutable state."""

    run_id: str
    created_at: str
    updated_at: str
    status: RunStatus
    goal: str | None
    plan_relpath: str
    home: str
    workdir: str
    max_parallel: int
    fail_fast: bool
    tasks: dict[str, TaskState]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "goal": self.goal,
            "plan_relpath": self.plan_relpath,
            "home": self.home,
            "workdir": self.workdir,
            "max_parallel": self.max_parallel,
            "fail_fast": self.fail_fast,
            "tasks": {task_id: task_state.to_dict() for task_id, task_state in self.tasks.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "RunState":
        tasks_raw = raw["tasks"]  # type: ignore[assignment]
        assert isinstance(tasks_raw, dict)
        return cls(
            run_id=raw["run_id"],  # type: ignore[arg-type]
            created_at=raw["created_at"],  # type: ignore[arg-type]
            updated_at=raw["updated_at"],  # type: ignore[arg-type]
            status=raw["status"],  # type: ignore[arg-type]
            goal=raw.get("goal"),  # type: ignore[arg-type]
            plan_relpath=raw["plan_relpath"],  # type: ignore[arg-type]
            home=raw["home"],  # type: ignore[arg-type]
            workdir=raw["workdir"],  # type: ignore[arg-type]
            max_parallel=raw["max_parallel"],  # type: ignore[arg-type]
            fail_fast=bool(raw["fail_fast"]),
            tasks={
                str(task_id): TaskState.from_dict(task_raw)  # type: ignore[arg-type]
                for task_id, task_raw in tasks_raw.items()
            },
        )
