from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

from orch.state.model import RUN_STATUS_VALUES, TASK_STATUS_VALUES, RunState
from orch.util.errors import StateError

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TASK_ID_MAX_LEN = 128
_RUN_ID_MAX_LEN = 128


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _is_iso_datetime(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _is_non_negative_finite_number_list(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(item)
        and item >= 0
        for item in value
    )


def _validate_state_shape(raw: dict[str, object], run_dir: Path) -> None:
    required_str = ("run_id", "status", "plan_relpath", "home", "workdir")
    for key in required_str:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise StateError(f"invalid state field: {key}")

    for key in ("created_at", "updated_at"):
        if not _is_iso_datetime(raw.get(key)):
            raise StateError(f"invalid state field: {key}")

    run_id = raw["run_id"]
    if (
        not isinstance(run_id, str)
        or len(run_id) > _RUN_ID_MAX_LEN
        or _SAFE_ID_PATTERN.fullmatch(run_id) is None
    ):
        raise StateError("invalid state field: run_id")
    if run_id != run_dir.name:
        raise StateError("state run_id does not match directory")

    status = raw["status"]
    if status not in RUN_STATUS_VALUES:
        raise StateError("invalid state field: status")

    max_parallel = raw.get("max_parallel")
    if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or max_parallel < 1:
        raise StateError("invalid state field: max_parallel")

    if not isinstance(raw.get("fail_fast"), bool):
        raise StateError("invalid state field: fail_fast")

    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise StateError("invalid state field: tasks")
    task_ids = list(tasks.keys())
    if any(not isinstance(task_id, str) for task_id in task_ids):
        raise StateError("invalid state field: tasks")
    folded = {task_id.casefold() for task_id in task_ids}
    if len(folded) != len(task_ids):
        raise StateError("invalid state field: tasks")

    for task_id, task_data in tasks.items():
        if (
            not isinstance(task_id, str)
            or len(task_id) > _TASK_ID_MAX_LEN
            or _SAFE_ID_PATTERN.fullmatch(task_id) is None
            or not isinstance(task_data, dict)
        ):
            raise StateError("invalid state field: tasks")
        task_status = task_data.get("status")
        if not isinstance(task_status, str) or task_status not in TASK_STATUS_VALUES:
            raise StateError("invalid state field: tasks")
        attempts = task_data.get("attempts")
        if attempts is not None and not _is_non_negative_int(attempts):
            raise StateError("invalid state field: tasks")
        retries = task_data.get("retries")
        if retries is not None and not _is_non_negative_int(retries):
            raise StateError("invalid state field: tasks")
        timeout_sec = task_data.get("timeout_sec")
        if timeout_sec is not None and not _is_positive_finite_number(timeout_sec):
            raise StateError("invalid state field: tasks")
        backoff = task_data.get("retry_backoff_sec")
        if backoff is not None and not _is_non_negative_finite_number_list(backoff):
            raise StateError("invalid state field: tasks")
        for key in ("stdout_path", "stderr_path"):
            rel = task_data.get(key)
            if rel is None:
                continue
            if not isinstance(rel, str) or not rel or "\x00" in rel:
                raise StateError("invalid state field: tasks")
            rel_path = Path(rel)
            if (
                rel_path.is_absolute()
                or ".." in rel_path.parts
                or not rel_path.parts
                or rel_path.parts[0] != "logs"
            ):
                raise StateError("invalid state field: tasks")
            if len(rel_path.parts) != 2:
                raise StateError("invalid state field: tasks")
            filename = rel_path.parts[1]
            if not filename.startswith(f"{task_id}.") or not filename.endswith(".log"):
                raise StateError("invalid state field: tasks")

        artifact_paths = task_data.get("artifact_paths")
        if artifact_paths is None:
            continue
        if not isinstance(artifact_paths, list):
            raise StateError("invalid state field: tasks")
        for artifact_rel in artifact_paths:
            if not isinstance(artifact_rel, str) or not artifact_rel or "\x00" in artifact_rel:
                raise StateError("invalid state field: tasks")
            artifact_path = Path(artifact_rel)
            if (
                artifact_path.is_absolute()
                or ".." in artifact_path.parts
                or not artifact_path.parts
                or artifact_path.parts[0] != "artifacts"
            ):
                raise StateError("invalid state field: tasks")
            if len(artifact_path.parts) < 3 or artifact_path.parts[1] != task_id:
                raise StateError("invalid state field: tasks")


def load_state(run_dir: Path) -> RunState:
    state_path = run_dir / "state.json"
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StateError(f"state file not found: {state_path}") from exc
    except UnicodeError as exc:
        raise StateError(f"failed to decode state file as utf-8: {state_path}") from exc
    except OSError as exc:
        raise StateError(f"failed to read state file: {state_path}") from exc
    except json.JSONDecodeError as exc:
        raise StateError(f"invalid state json: {state_path}") from exc
    if not isinstance(raw, dict):
        raise StateError("state root must be object")
    _validate_state_shape(raw, run_dir)
    return RunState.from_dict(raw)


def save_state_atomic(run_dir: Path, state: RunState) -> None:
    state_path = run_dir / "state.json"
    tmp_path = run_dir / "state.json.tmp"
    payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(payload + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, state_path)
    _fsync_directory(run_dir)
