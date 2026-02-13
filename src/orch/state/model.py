from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

RunStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED", "CANCELED"]
TaskStatus = Literal["PENDING", "READY", "RUNNING", "SUCCESS", "FAILED", "SKIPPED", "CANCELED"]
RUN_STATUS_VALUES: set[str] = {"PENDING", "RUNNING", "SUCCESS", "FAILED", "CANCELED"}
TASK_STATUS_VALUES: set[str] = {
    "PENDING",
    "READY",
    "RUNNING",
    "SUCCESS",
    "FAILED",
    "SKIPPED",
    "CANCELED",
}


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_bool(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _as_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def _as_optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _as_optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_list_str(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _as_list_float(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, (int, float))]


def _as_env_map(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    env: dict[str, str] = {}
    for key, val in value.items():
        if isinstance(key, str) and isinstance(val, str):
            env[key] = val
    return env


def _parse_task_status(value: object) -> TaskStatus:
    status = _as_str(value, "PENDING")
    if status not in TASK_STATUS_VALUES:
        status = "PENDING"
    return cast(TaskStatus, status)


def _parse_run_status(value: object) -> RunStatus:
    status = _as_str(value, "PENDING")
    if status not in RUN_STATUS_VALUES:
        status = "PENDING"
    return cast(RunStatus, status)


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
    def from_dict(cls, data: dict[str, object]) -> TaskState:
        return cls(
            status=_parse_task_status(data.get("status")),
            depends_on=_as_list_str(data.get("depends_on")),
            cmd=_as_list_str(data.get("cmd")),
            cwd=_as_optional_str(data.get("cwd")),
            env=_as_env_map(data.get("env")),
            timeout_sec=_as_optional_float(data.get("timeout_sec")),
            retries=_as_int(data.get("retries")),
            retry_backoff_sec=_as_list_float(data.get("retry_backoff_sec")),
            outputs=_as_list_str(data.get("outputs")),
            attempts=_as_int(data.get("attempts")),
            started_at=_as_optional_str(data.get("started_at")),
            ended_at=_as_optional_str(data.get("ended_at")),
            duration_sec=_as_optional_float(data.get("duration_sec")),
            exit_code=_as_optional_int(data.get("exit_code")),
            timed_out=_as_bool(data.get("timed_out")),
            canceled=_as_bool(data.get("canceled")),
            skip_reason=_as_optional_str(data.get("skip_reason")),
            stdout_path=_as_optional_str(data.get("stdout_path")),
            stderr_path=_as_optional_str(data.get("stderr_path")),
            artifact_paths=_as_list_str(data.get("artifact_paths")),
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
    def from_dict(cls, data: dict[str, object]) -> RunState:
        raw_tasks = data.get("tasks")
        tasks: dict[str, TaskState] = {}
        if isinstance(raw_tasks, dict):
            for task_id, task_data in raw_tasks.items():
                if isinstance(task_id, str) and isinstance(task_data, dict):
                    tasks[task_id] = TaskState.from_dict(task_data)
        return cls(
            run_id=_as_str(data.get("run_id")),
            created_at=_as_str(data.get("created_at")),
            updated_at=_as_str(data.get("updated_at")),
            status=_parse_run_status(data.get("status")),
            goal=_as_optional_str(data.get("goal")),
            plan_relpath=_as_str(data.get("plan_relpath")),
            home=_as_str(data.get("home")),
            workdir=_as_str(data.get("workdir")),
            max_parallel=_as_int(data.get("max_parallel")),
            fail_fast=_as_bool(data.get("fail_fast")),
            tasks=tasks,
        )
