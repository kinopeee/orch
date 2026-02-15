from __future__ import annotations

import errno
import math
import os
import re
import shlex
import stat
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

from orch.config.schema import PlanSpec, TaskSpec
from orch.dag.build import build_adjacency
from orch.dag.validate import assert_acyclic
from orch.util.errors import PlanError
from orch.util.path_guard import has_symlink_ancestor

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TASK_ID_MAX_LEN = 128
_ALLOWED_PLAN_KEYS = {"goal", "artifacts_dir", "tasks"}
_ALLOWED_TASK_KEYS = {
    "id",
    "cmd",
    "depends_on",
    "cwd",
    "env",
    "timeout_sec",
    "retries",
    "retry_backoff_sec",
    "outputs",
}


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_real_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isfinite(value)


def _is_non_blank_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _is_str_without_nul(value: object) -> bool:
    return isinstance(value, str) and "\x00" not in value


def _is_valid_env_key(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return _is_non_blank_str(value) and "=" not in value


def _is_safe_id(value: object) -> bool:
    return isinstance(value, str) and _SAFE_ID_PATTERN.fullmatch(value) is not None


def normalize_cmd(cmd: str | list[str]) -> list[str]:
    if isinstance(cmd, str):
        try:
            parts = shlex.split(cmd)
        except ValueError as exc:
            raise PlanError(f"invalid cmd string: {exc}") from exc
        if not parts:
            raise PlanError("cmd string must not be empty")
        if any("\x00" in part for part in parts):
            raise PlanError("cmd must not contain null bytes")
        return parts
    if isinstance(cmd, list) and cmd and all(_is_non_blank_str(p) for p in cmd):
        return cmd
    raise PlanError("cmd must be str or non-empty list[str]")


def _ensure_list_str(name: str, value: Any, *, non_empty_items: bool = False) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PlanError(f"{name} must be list[str]")
    if non_empty_items and any(not _is_non_blank_str(v) for v in value):
        raise PlanError(f"{name} must not contain empty strings")
    return value


def _parse_task(raw: Any) -> TaskSpec:
    if not isinstance(raw, dict):
        raise PlanError("task must be mapping")
    if any(not isinstance(key, str) for key in raw):
        raise PlanError("task fields must use string keys")
    if "id" not in raw or not _is_non_blank_str(raw["id"]):
        raise PlanError("task.id is required and must be non-empty string")
    if len(raw["id"]) > _TASK_ID_MAX_LEN:
        raise PlanError(f"task.id must be <= {_TASK_ID_MAX_LEN} characters")
    if not _is_safe_id(raw["id"]):
        raise PlanError("task.id must match ^[A-Za-z0-9][A-Za-z0-9._-]*$")
    unknown = set(raw.keys()) - _ALLOWED_TASK_KEYS
    if unknown:
        raise PlanError(f"task '{raw['id']}' has unknown fields: {sorted(unknown)}")
    if "cmd" not in raw:
        raise PlanError(f"task '{raw['id']}' missing cmd")

    retries = raw.get("retries", 0)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise PlanError(f"task '{raw['id']}' retries must be int >= 0")

    timeout_sec = raw.get("timeout_sec")
    if timeout_sec is not None:
        if not _is_finite_real_number(timeout_sec) or timeout_sec <= 0:
            raise PlanError(f"task '{raw['id']}' timeout_sec must be > 0")
        timeout_sec = float(timeout_sec)

    raw_backoff = raw.get("retry_backoff_sec", [])
    if not isinstance(raw_backoff, list) or not all(
        _is_finite_real_number(v) and v >= 0 for v in raw_backoff
    ):
        raise PlanError(f"task '{raw['id']}' retry_backoff_sec must be list[number>=0]")
    retry_backoff = [float(v) for v in raw_backoff]
    if len(retry_backoff) > retries:
        raise PlanError(f"task '{raw['id']}' retry_backoff_sec length must be <= retries")

    depends_on = _ensure_list_str("depends_on", raw.get("depends_on", []), non_empty_items=True)
    outputs = _ensure_list_str("outputs", raw.get("outputs", []), non_empty_items=True)

    cwd = raw.get("cwd")
    if cwd is not None and not _is_non_blank_str(cwd):
        raise PlanError(f"task '{raw['id']}' cwd must be non-empty string")

    env = raw.get("env")
    if env is not None and (
        not isinstance(env, dict)
        or not all(_is_valid_env_key(k) and _is_str_without_nul(v) for k, v in env.items())
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
    folded_ids = [task_id.casefold() for task_id in ids]
    if len(set(folded_ids)) != len(folded_ids):
        raise PlanError("task.id must be unique (case-insensitive)")

    known = set(ids)
    for task in plan.tasks:
        unknown = [dep for dep in task.depends_on if dep not in known]
        if unknown:
            raise PlanError(f"task '{task.id}' has unknown dependencies: {unknown}")
        if task.id in task.depends_on:
            raise PlanError(f"task '{task.id}' must not depend on itself")
        if len(set(task.depends_on)) != len(task.depends_on):
            raise PlanError(f"task '{task.id}' has duplicate dependencies")
        if len({output.casefold() for output in task.outputs}) != len(task.outputs):
            raise PlanError(f"task '{task.id}' has duplicate outputs")

    dependents, in_degree = build_adjacency(plan)
    assert_acyclic(ids, dependents, in_degree)


def load_plan(path: Path) -> PlanSpec:
    if has_symlink_ancestor(path):
        raise PlanError(f"plan file path must not include symlink: {path}")
    try:
        meta = path.lstat()
    except FileNotFoundError:
        meta = None
    except (OSError, RuntimeError) as exc:
        raise PlanError(f"failed to read plan file: {path}") from exc

    if meta is not None:
        if stat.S_ISLNK(meta.st_mode):
            raise PlanError(f"plan file must not be symlink: {path}")
        if not stat.S_ISREG(meta.st_mode):
            raise PlanError(f"failed to read plan file: {path}")

    open_flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        open_flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(str(path), open_flags)
        opened_meta = os.fstat(fd)
        if not stat.S_ISREG(opened_meta.st_mode):
            raise PlanError(f"failed to read plan file: {path}")
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            fd = None
            content = f.read()
    except FileNotFoundError as exc:
        raise PlanError(f"plan file not found: {path}") from exc
    except UnicodeError as exc:
        raise PlanError(f"failed to decode plan file as utf-8: {path}") from exc
    except RuntimeError as exc:
        raise PlanError(f"failed to read plan file: {path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PlanError(f"plan file must not be symlink: {path}") from exc
        raise PlanError(f"failed to read plan file: {path}") from exc
    finally:
        if fd is not None:
            with suppress(OSError, RuntimeError):
                os.close(fd)

    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise PlanError(f"failed to parse yaml: {exc}") from exc

    if not isinstance(raw, dict):
        raise PlanError("plan root must be a mapping")
    if any(not isinstance(key, str) for key in raw):
        raise PlanError("plan root keys must be strings")
    unknown_root = set(raw.keys()) - _ALLOWED_PLAN_KEYS
    if unknown_root:
        raise PlanError(f"plan contains unknown fields: {sorted(unknown_root)}")

    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list):
        raise PlanError("plan.tasks must be a list")

    goal = raw.get("goal")
    if goal is not None and not _is_non_blank_str(goal):
        raise PlanError("plan.goal must be non-empty string when provided")

    artifacts_dir = raw.get("artifacts_dir")
    if artifacts_dir is not None and not _is_non_blank_str(artifacts_dir):
        raise PlanError("plan.artifacts_dir must be non-empty string when provided")

    tasks = [_parse_task(task) for task in raw_tasks]
    plan = PlanSpec(
        goal=goal,
        artifacts_dir=artifacts_dir,
        tasks=tasks,
    )
    validate_plan(plan)
    return plan
