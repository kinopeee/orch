from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

from orch.config.schema import PlanSpec, TaskSpec
from orch.util.errors import PlanError


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_cmd(cmd: str | list[str]) -> list[str]:
    if isinstance(cmd, str):
        parts = shlex.split(cmd)
        if not parts:
            raise PlanError("cmd string must not be empty")
        return parts
    if isinstance(cmd, list) and all(isinstance(p, str) and p for p in cmd):
        return cmd
    raise PlanError("cmd must be str or non-empty list[str]")


def _ensure_list_str(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PlanError(f"{name} must be list[str]")
    return value


def _parse_task(raw: Any) -> TaskSpec:
    if not isinstance(raw, dict):
        raise PlanError("task must be mapping")
    if "id" not in raw or not isinstance(raw["id"], str) or not raw["id"]:
        raise PlanError("task.id is required and must be non-empty string")
    if "cmd" not in raw:
        raise PlanError(f"task '{raw['id']}' missing cmd")

    retries = raw.get("retries", 0)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise PlanError(f"task '{raw['id']}' retries must be int >= 0")

    timeout_sec = raw.get("timeout_sec")
    if timeout_sec is not None:
        if not _is_real_number(timeout_sec) or timeout_sec <= 0:
            raise PlanError(f"task '{raw['id']}' timeout_sec must be > 0")
        timeout_sec = float(timeout_sec)

    raw_backoff = raw.get("retry_backoff_sec", [])
    if not isinstance(raw_backoff, list) or not all(
        _is_real_number(v) and v >= 0 for v in raw_backoff
    ):
        raise PlanError(f"task '{raw['id']}' retry_backoff_sec must be list[number>=0]")
    retry_backoff = [float(v) for v in raw_backoff]

    depends_on = _ensure_list_str("depends_on", raw.get("depends_on", []))
    outputs = _ensure_list_str("outputs", raw.get("outputs", []))

    cwd = raw.get("cwd")
    if cwd is not None and (not isinstance(cwd, str) or not cwd):
        raise PlanError(f"task '{raw['id']}' cwd must be non-empty string")

    env = raw.get("env")
    if env is not None and (
        not isinstance(env, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
    ):
        raise PlanError(f"task '{raw['id']}' env must be dict[str, str]")

    return TaskSpec(
        id=raw["id"],
        cmd=normalize_cmd(raw["cmd"]),
        depends_on=depends_on,
        cwd=cwd,
        env=env,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff_sec=retry_backoff,
        outputs=outputs,
    )


def validate_plan(plan: PlanSpec) -> None:
    if not plan.tasks:
        raise PlanError("plan.tasks must contain at least one task")

    ids = [task.id for task in plan.tasks]
    if len(set(ids)) != len(ids):
        raise PlanError("task.id must be unique")

    known = set(ids)
    for task in plan.tasks:
        unknown = [dep for dep in task.depends_on if dep not in known]
        if unknown:
            raise PlanError(f"task '{task.id}' has unknown dependencies: {unknown}")


def load_plan(path: Path) -> PlanSpec:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlanError(f"plan file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise PlanError(f"failed to parse yaml: {exc}") from exc

    if not isinstance(raw, dict):
        raise PlanError("plan root must be a mapping")

    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list):
        raise PlanError("plan.tasks must be a list")

    goal = raw.get("goal")
    if goal is not None and not isinstance(goal, str):
        raise PlanError("plan.goal must be string when provided")

    artifacts_dir = raw.get("artifacts_dir")
    if artifacts_dir is not None and not isinstance(artifacts_dir, str):
        raise PlanError("plan.artifacts_dir must be string when provided")

    tasks = [_parse_task(task) for task in raw_tasks]
    plan = PlanSpec(
        goal=goal,
        artifacts_dir=artifacts_dir,
        tasks=tasks,
    )
    validate_plan(plan)
    return plan
