from __future__ import annotations

import json
import os
from pathlib import Path

from orch.state.model import RUN_STATUS_VALUES, RunState
from orch.util.errors import StateError


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


def _validate_state_shape(raw: dict[str, object]) -> None:
    required_str = (
        "run_id",
        "created_at",
        "updated_at",
        "status",
        "plan_relpath",
        "home",
        "workdir",
    )
    for key in required_str:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise StateError(f"invalid state field: {key}")

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

    for task_id, task_data in tasks.items():
        if not isinstance(task_id, str) or not isinstance(task_data, dict):
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
    _validate_state_shape(raw)
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
