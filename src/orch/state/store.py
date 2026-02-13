from __future__ import annotations

import errno
import json
import math
import os
import re
import stat
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from orch.state.model import RUN_STATUS_VALUES, TASK_STATUS_VALUES, RunState
from orch.util.errors import StateError

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TASK_ID_MAX_LEN = 128
_RUN_ID_MAX_LEN = 128
_ALLOWED_RUN_KEYS = {
    "run_id",
    "created_at",
    "updated_at",
    "status",
    "goal",
    "plan_relpath",
    "home",
    "workdir",
    "max_parallel",
    "fail_fast",
    "tasks",
}
_ALLOWED_TASK_KEYS = {
    "status",
    "depends_on",
    "cmd",
    "cwd",
    "env",
    "timeout_sec",
    "retries",
    "retry_backoff_sec",
    "outputs",
    "attempts",
    "started_at",
    "ended_at",
    "duration_sec",
    "exit_code",
    "timed_out",
    "canceled",
    "skip_reason",
    "stdout_path",
    "stderr_path",
    "artifact_paths",
}


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
        with suppress(OSError):
            os.close(fd)


def _is_iso_datetime(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return False
    return dt.tzinfo is not None


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


def _is_non_blank_str_without_nul(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _validate_state_shape(raw: dict[str, object], run_dir: Path) -> None:
    if any(not isinstance(key, str) for key in raw):
        raise StateError("invalid state field: root")
    unknown_root = set(raw.keys()) - _ALLOWED_RUN_KEYS
    if unknown_root:
        raise StateError("invalid state field: root")

    required_str = ("run_id", "status", "plan_relpath", "home", "workdir")
    for key in required_str:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise StateError(f"invalid state field: {key}")

    plan_relpath = raw["plan_relpath"]
    if not isinstance(plan_relpath, str) or "\x00" in plan_relpath:
        raise StateError("invalid state field: plan_relpath")
    plan_rel = Path(plan_relpath)
    if (
        plan_rel.is_absolute()
        or ".." in plan_rel.parts
        or len(plan_rel.parts) != 1
        or plan_rel.name != "plan.yaml"
    ):
        raise StateError("invalid state field: plan_relpath")
    if "goal" not in raw:
        raise StateError("invalid state field: goal")
    goal = raw.get("goal")
    if goal is not None and not _is_non_blank_str_without_nul(goal):
        raise StateError("invalid state field: goal")

    for key in ("created_at", "updated_at"):
        if not _is_iso_datetime(raw.get(key)):
            raise StateError(f"invalid state field: {key}")
    created_at = datetime.fromisoformat(str(raw["created_at"]))
    updated_at = datetime.fromisoformat(str(raw["updated_at"]))
    if updated_at < created_at:
        raise StateError("invalid state field: updated_at")

    run_id = raw["run_id"]
    if (
        not isinstance(run_id, str)
        or len(run_id) > _RUN_ID_MAX_LEN
        or _SAFE_ID_PATTERN.fullmatch(run_id) is None
    ):
        raise StateError("invalid state field: run_id")
    if run_id != run_dir.name:
        raise StateError("state run_id does not match directory")

    home = raw["home"]
    if not isinstance(home, str) or "\x00" in home:
        raise StateError("invalid state field: home")
    home_path = Path(home)
    if not home_path.is_absolute() or ".." in home_path.parts:
        raise StateError("invalid state field: home")
    try:
        resolved_home = home_path.resolve()
    except (OSError, RuntimeError) as exc:
        raise StateError("invalid state field: home") from exc
    if run_dir.parent.name == "runs":
        expected_home = run_dir.parent.parent.resolve()
        if resolved_home != expected_home:
            raise StateError("state home does not match directory")

    workdir = raw["workdir"]
    if not isinstance(workdir, str) or "\x00" in workdir:
        raise StateError("invalid state field: workdir")
    workdir_path = Path(workdir)
    if not workdir_path.is_absolute() or ".." in workdir_path.parts:
        raise StateError("invalid state field: workdir")

    status = raw["status"]
    if status not in RUN_STATUS_VALUES:
        raise StateError("invalid state field: status")
    if status == "PENDING":
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
    known_task_ids = set(task_ids)
    task_statuses: list[str] = []

    for task_id, task_data in tasks.items():
        if (
            not isinstance(task_id, str)
            or len(task_id) > _TASK_ID_MAX_LEN
            or _SAFE_ID_PATTERN.fullmatch(task_id) is None
            or not isinstance(task_data, dict)
        ):
            raise StateError("invalid state field: tasks")
        if any(not isinstance(key, str) for key in task_data):
            raise StateError("invalid state field: tasks")
        unknown_task = set(task_data.keys()) - _ALLOWED_TASK_KEYS
        if unknown_task:
            raise StateError("invalid state field: tasks")
        task_status = task_data.get("status")
        if not isinstance(task_status, str) or task_status not in TASK_STATUS_VALUES:
            raise StateError("invalid state field: tasks")
        task_statuses.append(task_status)
        depends_on = task_data.get("depends_on")
        if not isinstance(depends_on, list) or any(not isinstance(dep, str) for dep in depends_on):
            raise StateError("invalid state field: tasks")
        if len(set(depends_on)) != len(depends_on):
            raise StateError("invalid state field: tasks")
        if task_id in depends_on or any(dep not in known_task_ids for dep in depends_on):
            raise StateError("invalid state field: tasks")

        cmd = task_data.get("cmd")
        if (
            not isinstance(cmd, list)
            or not cmd
            or any(not _is_non_blank_str_without_nul(part) for part in cmd)
        ):
            raise StateError("invalid state field: tasks")

        outputs = task_data.get("outputs")
        if not isinstance(outputs, list) or any(
            not _is_non_blank_str_without_nul(output) for output in outputs
        ):
            raise StateError("invalid state field: tasks")
        if len({output.casefold() for output in outputs}) != len(outputs):
            raise StateError("invalid state field: tasks")

        if "cwd" not in task_data:
            raise StateError("invalid state field: tasks")
        cwd = task_data.get("cwd")
        if cwd is not None and not _is_non_blank_str_without_nul(cwd):
            raise StateError("invalid state field: tasks")

        if "env" not in task_data:
            raise StateError("invalid state field: tasks")
        env = task_data.get("env")
        if env is not None:
            if not isinstance(env, dict):
                raise StateError("invalid state field: tasks")
            for env_key, env_value in env.items():
                if (
                    not _is_non_blank_str_without_nul(env_key)
                    or "=" in env_key
                    or not isinstance(env_value, str)
                    or "\x00" in env_value
                ):
                    raise StateError("invalid state field: tasks")
        attempts_raw = task_data.get("attempts")
        if not _is_non_negative_int(attempts_raw):
            raise StateError("invalid state field: tasks")
        retries_raw = task_data.get("retries")
        if not _is_non_negative_int(retries_raw):
            raise StateError("invalid state field: tasks")
        assert isinstance(attempts_raw, int) and not isinstance(attempts_raw, bool)
        assert isinstance(retries_raw, int) and not isinstance(retries_raw, bool)
        attempts = attempts_raw
        retries = retries_raw
        if attempts > (retries + 1):
            raise StateError("invalid state field: tasks")
        if "timeout_sec" not in task_data:
            raise StateError("invalid state field: tasks")
        timeout_sec = task_data.get("timeout_sec")
        if timeout_sec is not None and not _is_positive_finite_number(timeout_sec):
            raise StateError("invalid state field: tasks")
        if "retry_backoff_sec" not in task_data:
            raise StateError("invalid state field: tasks")
        backoff = task_data.get("retry_backoff_sec")
        if not _is_non_negative_finite_number_list(backoff):
            raise StateError("invalid state field: tasks")
        if isinstance(backoff, list) and len(backoff) > retries:
            raise StateError("invalid state field: tasks")
        if "started_at" not in task_data:
            raise StateError("invalid state field: tasks")
        started_at = task_data.get("started_at")
        if started_at is not None and not _is_iso_datetime(started_at):
            raise StateError("invalid state field: tasks")
        if "ended_at" not in task_data:
            raise StateError("invalid state field: tasks")
        ended_at = task_data.get("ended_at")
        if ended_at is not None and not _is_iso_datetime(ended_at):
            raise StateError("invalid state field: tasks")
        if isinstance(started_at, str) and isinstance(ended_at, str):
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(ended_at)
            if end_dt < start_dt:
                raise StateError("invalid state field: tasks")
        if task_status in {"SUCCESS", "FAILED", "SKIPPED", "CANCELED"} and ended_at is None:
            raise StateError("invalid state field: tasks")
        if task_status == "SUCCESS" and started_at is None:
            raise StateError("invalid state field: tasks")
        if "duration_sec" not in task_data:
            raise StateError("invalid state field: tasks")
        duration_sec = task_data.get("duration_sec")
        if duration_sec is not None and not (
            isinstance(duration_sec, (int, float))
            and not isinstance(duration_sec, bool)
            and math.isfinite(duration_sec)
            and duration_sec >= 0
        ):
            raise StateError("invalid state field: tasks")
        if "exit_code" not in task_data:
            raise StateError("invalid state field: tasks")
        exit_code = task_data.get("exit_code")
        if exit_code is not None and (
            not isinstance(exit_code, int) or isinstance(exit_code, bool)
        ):
            raise StateError("invalid state field: tasks")
        bool_values: dict[str, bool | None] = {}
        for bool_field in ("timed_out", "canceled"):
            if bool_field not in task_data:
                raise StateError("invalid state field: tasks")
            bval = task_data.get(bool_field)
            if bval is not None and not isinstance(bval, bool):
                raise StateError("invalid state field: tasks")
            bool_values[bool_field] = bval if isinstance(bval, bool) else None
        if isinstance(exit_code, int) and task_status == "SUCCESS" and exit_code != 0:
            raise StateError("invalid state field: tasks")
        if bool_values["timed_out"] is True and task_status not in {"FAILED", "READY", "PENDING"}:
            raise StateError("invalid state field: tasks")
        if bool_values["timed_out"] is True and exit_code is not None:
            raise StateError("invalid state field: tasks")
        if (
            task_status == "PENDING"
            and bool_values["timed_out"] is True
            and (not isinstance(attempts, int) or attempts < 1)
        ):
            raise StateError("invalid state field: tasks")
        if task_status == "PENDING" and (
            bool_values["timed_out"] is None or bool_values["canceled"] is None
        ):
            raise StateError("invalid state field: tasks")
        pending_has_runtime_snapshot = (
            started_at is not None
            or ended_at is not None
            or duration_sec is not None
            or exit_code is not None
            or bool_values["timed_out"] is True
        )
        if task_status == "PENDING" and pending_has_runtime_snapshot:
            if not isinstance(attempts, int) or attempts < 1:
                raise StateError("invalid state field: tasks")
            if not isinstance(started_at, str) or not isinstance(ended_at, str):
                raise StateError("invalid state field: tasks")
            if duration_sec is None:
                raise StateError("invalid state field: tasks")
            if bool_values["timed_out"] is not True and (
                not isinstance(exit_code, int) or exit_code == 0
            ):
                raise StateError("invalid state field: tasks")
        if bool_values["canceled"] is True and task_status != "CANCELED":
            raise StateError("invalid state field: tasks")
        if task_status in {"SUCCESS", "FAILED", "SKIPPED"} and bool_values["canceled"] is not False:
            raise StateError("invalid state field: tasks")
        if task_status in {"SUCCESS", "SKIPPED"} and bool_values["timed_out"] is not False:
            raise StateError("invalid state field: tasks")
        if task_status == "FAILED" and bool_values["timed_out"] is None:
            raise StateError("invalid state field: tasks")
        if task_status == "CANCELED" and bool_values["canceled"] is not True:
            raise StateError("invalid state field: tasks")
        if task_status == "CANCELED" and bool_values["timed_out"] is not False:
            raise StateError("invalid state field: tasks")
        if task_status == "CANCELED" and isinstance(exit_code, int) and exit_code == 0:
            raise StateError("invalid state field: tasks")
        if "skip_reason" not in task_data:
            raise StateError("invalid state field: tasks")
        skip_reason = task_data.get("skip_reason")
        if skip_reason is not None and not _is_non_blank_str_without_nul(skip_reason):
            raise StateError("invalid state field: tasks")
        if task_status in {"SKIPPED", "CANCELED"} and not _is_non_blank_str_without_nul(
            skip_reason
        ):
            raise StateError("invalid state field: tasks")
        if (
            task_status == "PENDING"
            and isinstance(attempts, int)
            and attempts == 0
            and (
                started_at is not None
                or ended_at is not None
                or duration_sec is not None
                or exit_code is not None
                or bool_values["timed_out"] is True
                or bool_values["canceled"] is True
                or _is_non_blank_str_without_nul(skip_reason)
            )
        ):
            raise StateError("invalid state field: tasks")
        if task_status == "PENDING" and isinstance(exit_code, int) and exit_code == 0:
            raise StateError("invalid state field: tasks")
        if task_status == "PENDING" and _is_non_blank_str_without_nul(skip_reason):
            raise StateError("invalid state field: tasks")
        if (
            task_status == "CANCELED"
            and started_at is None
            and (
                not isinstance(attempts, int)
                or attempts != 0
                or exit_code is not None
                or duration_sec is not None
            )
        ):
            raise StateError("invalid state field: tasks")
        if (
            task_status == "CANCELED"
            and started_at is not None
            and (
                not isinstance(attempts, int)
                or attempts < 1
                or not isinstance(exit_code, int)
                or exit_code == 0
                or duration_sec is None
            )
        ):
            raise StateError("invalid state field: tasks")
        if task_status == "FAILED":
            if not isinstance(attempts, int) or attempts < 1:
                raise StateError("invalid state field: tasks")
            if bool_values["timed_out"] is True:
                if exit_code is not None:
                    raise StateError("invalid state field: tasks")
            elif isinstance(exit_code, int):
                if exit_code == 0:
                    raise StateError("invalid state field: tasks")
            elif not _is_non_blank_str_without_nul(skip_reason):
                raise StateError("invalid state field: tasks")
        if task_status == "SKIPPED" and (
            started_at is not None or exit_code is not None or duration_sec is not None
        ):
            raise StateError("invalid state field: tasks")
        if task_status == "SKIPPED" and (not isinstance(attempts, int) or attempts != 0):
            raise StateError("invalid state field: tasks")
        if task_status == "RUNNING":
            if not isinstance(started_at, str):
                raise StateError("invalid state field: tasks")
            if ended_at is not None:
                raise StateError("invalid state field: tasks")
            if not isinstance(attempts, int) or attempts < 1:
                raise StateError("invalid state field: tasks")
            if bool_values["timed_out"] is not False or bool_values["canceled"] is not False:
                raise StateError("invalid state field: tasks")
            if _is_non_blank_str_without_nul(skip_reason):
                raise StateError("invalid state field: tasks")
            if exit_code is not None or duration_sec is not None:
                raise StateError("invalid state field: tasks")
        if task_status == "READY":
            if not isinstance(started_at, str) or not isinstance(ended_at, str):
                raise StateError("invalid state field: tasks")
            if duration_sec is None:
                raise StateError("invalid state field: tasks")
            if bool_values["canceled"] is not False:
                raise StateError("invalid state field: tasks")
            if bool_values["timed_out"] is None:
                raise StateError("invalid state field: tasks")
            if (
                not isinstance(attempts, int)
                or not isinstance(retries, int)
                or attempts < 1
                or attempts > retries
            ):
                raise StateError("invalid state field: tasks")
            if _is_non_blank_str_without_nul(skip_reason):
                raise StateError("invalid state field: tasks")
            if bool_values["timed_out"] is True:
                if exit_code is not None:
                    raise StateError("invalid state field: tasks")
            elif not isinstance(exit_code, int) or exit_code == 0:
                raise StateError("invalid state field: tasks")
        if task_status == "SUCCESS":
            if not isinstance(attempts, int) or attempts < 1:
                raise StateError("invalid state field: tasks")
            if duration_sec is None:
                raise StateError("invalid state field: tasks")
            if exit_code != 0:
                raise StateError("invalid state field: tasks")
            if _is_non_blank_str_without_nul(skip_reason):
                raise StateError("invalid state field: tasks")
            if bool_values["timed_out"] is True or bool_values["canceled"] is True:
                raise StateError("invalid state field: tasks")
        for key in ("stdout_path", "stderr_path"):
            rel = task_data.get(key)
            if rel is None:
                raise StateError("invalid state field: tasks")
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
            expected = f"{task_id}.out.log" if key == "stdout_path" else f"{task_id}.err.log"
            if filename != expected:
                raise StateError("invalid state field: tasks")
        stdout_path = task_data.get("stdout_path")
        stderr_path = task_data.get("stderr_path")
        if (
            isinstance(stdout_path, str)
            and isinstance(stderr_path, str)
            and stdout_path == stderr_path
        ):
            raise StateError("invalid state field: tasks")

        artifact_paths = task_data.get("artifact_paths")
        if artifact_paths is None:
            raise StateError("invalid state field: tasks")
        if not isinstance(artifact_paths, list):
            raise StateError("invalid state field: tasks")
        seen_artifacts: set[str] = set()
        for artifact_rel in artifact_paths:
            if not isinstance(artifact_rel, str) or not artifact_rel or "\x00" in artifact_rel:
                raise StateError("invalid state field: tasks")
            artifact_key = artifact_rel.casefold()
            if artifact_key in seen_artifacts:
                raise StateError("invalid state field: tasks")
            seen_artifacts.add(artifact_key)
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
        if task_status in {"PENDING", "READY", "RUNNING", "SKIPPED", "CANCELED"} and artifact_paths:
            raise StateError("invalid state field: tasks")

    if status == "SUCCESS" and any(task_status != "SUCCESS" for task_status in task_statuses):
        raise StateError("invalid state field: status")
    if status == "CANCELED" and not any(task_status == "CANCELED" for task_status in task_statuses):
        raise StateError("invalid state field: status")
    if status == "FAILED" and not any(task_status == "FAILED" for task_status in task_statuses):
        raise StateError("invalid state field: status")
    if status == "FAILED" and any(task_status == "CANCELED" for task_status in task_statuses):
        raise StateError("invalid state field: status")
    terminal_statuses = {"SUCCESS", "FAILED", "SKIPPED", "CANCELED"}
    active_statuses = {"PENDING", "READY", "RUNNING"}
    if status in {"SUCCESS", "FAILED", "CANCELED"} and any(
        task_status in active_statuses for task_status in task_statuses
    ):
        raise StateError("invalid state field: status")
    if status == "RUNNING" and all(
        task_status in terminal_statuses for task_status in task_statuses
    ):
        raise StateError("invalid state field: status")


def load_state(run_dir: Path) -> RunState:
    state_path = run_dir / "state.json"
    try:
        meta = state_path.lstat()
    except FileNotFoundError:
        meta = None
    except OSError as exc:
        raise StateError(f"failed to read state file: {state_path}") from exc

    if meta is not None:
        if stat.S_ISLNK(meta.st_mode):
            raise StateError(f"state file must not be symlink: {state_path}")
        if not stat.S_ISREG(meta.st_mode):
            raise StateError(f"failed to read state file: {state_path}")
    open_flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        open_flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(str(state_path), open_flags)
        opened_meta = os.fstat(fd)
        if not stat.S_ISREG(opened_meta.st_mode):
            raise StateError(f"failed to read state file: {state_path}")
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            fd = None
            raw = json.loads(f.read())
    except FileNotFoundError as exc:
        raise StateError(f"state file not found: {state_path}") from exc
    except UnicodeError as exc:
        raise StateError(f"failed to decode state file as utf-8: {state_path}") from exc
    except json.JSONDecodeError as exc:
        raise StateError(f"invalid state json: {state_path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise StateError(f"state file must not be symlink: {state_path}") from exc
        raise StateError(f"failed to read state file: {state_path}") from exc
    finally:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
    if not isinstance(raw, dict):
        raise StateError("state root must be object")
    _validate_state_shape(raw, run_dir)
    return RunState.from_dict(raw)


def save_state_atomic(run_dir: Path, state: RunState) -> None:
    state_path = run_dir / "state.json"
    tmp_path = run_dir / "state.json.tmp"
    payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(tmp_path), flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError(f"temporary state path must not be symlink: {tmp_path}") from exc
        raise
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    try:
        os.replace(tmp_path, state_path)
    except OSError:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    _fsync_directory(run_dir)
