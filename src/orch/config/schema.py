from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TaskSpec:
    id: str
    cmd: list[str]
    depends_on: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout_sec: float | None = None
    retries: int = 0
    retry_backoff_sec: list[float] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlanSpec:
    goal: str | None
    artifacts_dir: str | None
    tasks: list[TaskSpec]
