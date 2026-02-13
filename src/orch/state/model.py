from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RunStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED", "CANCELED"]
TaskStatus = Literal["PENDING", "READY", "RUNNING", "SUCCESS", "FAILED", "SKIPPED", "CANCELED"]


@dataclass(slots=True)
class TaskState:
    status: TaskStatus
    depends_on: list[str]
    cmd: list[str]
    cwd: str | None
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
    stdout_path: str | None = None
    stderr_path: str | None = None
    artifact_paths: list[str] = field(default_factory=list)

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
    def from_dict(cls, data: dict[str, object]) -> "TaskState":
        return cls(
            status=str(data["status"]),  # type: ignore[arg-type]
            depends_on=list(data.get("depends_on", [])),  # type: ignore[arg-type]
            cmd=list(data.get("cmd", [])),  # type: ignore[arg-type]
            cwd=data.get("cwd") if isinstance(data.get("cwd"), str) else None,
            env=data.get("env") if isinstance(data.get("env"), dict) else None,  # type: ignore[arg-type]
            timeout_sec=(
                float(data["timeout_sec"])
                if isinstance(data.get("timeout_sec"), (int, float))
                else None
            ),
            retries=int(data.get("retries", 0)),
            retry_backoff_sec=[float(v) for v in data.get("retry_backoff_sec", [])],  # type: ignore[arg-type]
            outputs=list(data.get("outputs", [])),  # type: ignore[arg-type]
            attempts=int(data.get("attempts", 0)),
            started_at=data.get("started_at") if isinstance(data.get("started_at"), str) else None,
            ended_at=data.get("ended_at") if isinstance(data.get("ended_at"), str) else None,
            duration_sec=(
                float(data["duration_sec"])
                if isinstance(data.get("duration_sec"), (int, float))
                else None
            ),
            exit_code=int(data["exit_code"]) if isinstance(data.get("exit_code"), int) else None,
            timed_out=bool(data.get("timed_out", False)),
            canceled=bool(data.get("canceled", False)),
            skip_reason=data.get("skip_reason") if isinstance(data.get("skip_reason"), str) else None,
            stdout_path=(
                data.get("stdout_path") if isinstance(data.get("stdout_path"), str) else None
            ),
            stderr_path=(
                data.get("stderr_path") if isinstance(data.get("stderr_path"), str) else None
            ),
            artifact_paths=list(data.get("artifact_paths", [])),  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class RunState:
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
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RunState":
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, dict):
            raw_tasks = {}
        return cls(
            run_id=str(data["run_id"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            status=str(data["status"]),  # type: ignore[arg-type]
            goal=data.get("goal") if isinstance(data.get("goal"), str) else None,
            plan_relpath=str(data["plan_relpath"]),
            home=str(data["home"]),
            workdir=str(data["workdir"]),
            max_parallel=int(data["max_parallel"]),
            fail_fast=bool(data["fail_fast"]),
            tasks={
                task_id: TaskState.from_dict(task_data)  # type: ignore[arg-type]
                for task_id, task_data in raw_tasks.items()
            },
        )
