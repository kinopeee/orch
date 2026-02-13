"""Plan loader and validator."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

from orch.config.schema import PlanSpec, TaskSpec
from orch.dag.build import build_adjacency
from orch.dag.validate import assert_acyclic
from orch.util.errors import PlanError


def normalize_cmd(cmd: str | list[str]) -> list[str]:
    """Normalize cmd into list form."""
    if isinstance(cmd, str):
        parts = shlex.split(cmd)
        if not parts:
            raise PlanError("Task cmd string is empty.")
        return parts
    if isinstance(cmd, list) and all(isinstance(x, str) for x in cmd):
        if not cmd:
            raise PlanError("Task cmd list is empty.")
        return cmd
    raise PlanError("Task cmd must be string or list[string].")


def _ensure_str_list(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise PlanError(f"Task '{name}' must be list[str].")
    return value


def _parse_task(task_raw: Any) -> TaskSpec:
    if not isinstance(task_raw, dict):
        raise PlanError("Each task must be a mapping.")
    raw_id = task_raw.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        raise PlanError("Task id is required and must be non-empty string.")
    if "cmd" not in task_raw:
        raise PlanError(f"Task '{raw_id}' is missing cmd.")
    cmd = normalize_cmd(task_raw["cmd"])
    depends_on = _ensure_str_list("depends_on", task_raw.get("depends_on"))
    outputs = _ensure_str_list("outputs", task_raw.get("outputs"))

    env_raw = task_raw.get("env")
    env: dict[str, str] | None
    if env_raw is None:
        env = None
    elif isinstance(env_raw, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()
    ):
        env = dict(env_raw)
    else:
        raise PlanError(f"Task '{raw_id}' env must be map[str,str].")

    cwd_raw = task_raw.get("cwd")
    if cwd_raw is not None and not isinstance(cwd_raw, str):
        raise PlanError(f"Task '{raw_id}' cwd must be string.")

    timeout_raw = task_raw.get("timeout_sec")
    timeout_sec: float | None
    if timeout_raw is None:
        timeout_sec = None
    elif isinstance(timeout_raw, (int, float)):
        timeout_sec = float(timeout_raw)
    else:
        raise PlanError(f"Task '{raw_id}' timeout_sec must be number.")

    retries_raw = task_raw.get("retries", 0)
    if not isinstance(retries_raw, int):
        raise PlanError(f"Task '{raw_id}' retries must be int.")

    backoff_raw = task_raw.get("retry_backoff_sec", [])
    if not isinstance(backoff_raw, list) or not all(
        isinstance(x, (int, float)) for x in backoff_raw
    ):
        raise PlanError(f"Task '{raw_id}' retry_backoff_sec must be list[number].")
    retry_backoff_sec = [float(x) for x in backoff_raw]

    return TaskSpec(
        id=raw_id,
        cmd=cmd,
        depends_on=depends_on,
        cwd=cwd_raw,
        env=env,
        timeout_sec=timeout_sec,
        retries=retries_raw,
        retry_backoff_sec=retry_backoff_sec,
        outputs=outputs,
    )


def validate_plan(plan: PlanSpec) -> None:
    """Validate plan with schema + DAG rules."""
    if len(plan.tasks) == 0:
        raise PlanError("Plan tasks must contain at least one item.")

    id_set: set[str] = set()
    for task in plan.tasks:
        if task.id in id_set:
            raise PlanError(f"Task id must be unique: {task.id}")
        id_set.add(task.id)
        if task.retries < 0:
            raise PlanError(f"Task '{task.id}' retries must be >= 0.")
        if task.timeout_sec is not None and task.timeout_sec <= 0:
            raise PlanError(f"Task '{task.id}' timeout_sec must be > 0.")

    for task in plan.tasks:
        for dep in task.depends_on:
            if dep not in id_set:
                raise PlanError(f"Task '{task.id}' depends on unknown task '{dep}'.")

    dependents, in_degree = build_adjacency(plan)
    assert_acyclic([task.id for task in plan.tasks], dependents, in_degree)


def load_plan(path: Path) -> PlanSpec:
    """Load and validate plan YAML."""
    if not path.exists():
        raise PlanError(f"Plan file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PlanError("Plan root must be mapping.")

    goal_raw = raw.get("goal")
    if goal_raw is not None and not isinstance(goal_raw, str):
        raise PlanError("goal must be string.")
    artifacts_raw = raw.get("artifacts_dir")
    if artifacts_raw is not None and not isinstance(artifacts_raw, str):
        raise PlanError("artifacts_dir must be string.")

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list):
        raise PlanError("tasks must be list.")
    tasks = [_parse_task(task_raw) for task_raw in tasks_raw]
    plan = PlanSpec(goal=goal_raw, artifacts_dir=artifacts_raw, tasks=tasks)
    validate_plan(plan)
    return plan
