from __future__ import annotations

import json
from pathlib import Path

import pytest

from orch.state.model import RunState, TaskState
from orch.state.store import load_state, save_state_atomic
from orch.util.errors import StateError


def _minimal_state_payload(*, run_id: str) -> dict[str, object]:
    state = RunState(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        status="RUNNING",
        goal=None,
        plan_relpath="plan.yaml",
        home="/tmp/.orch_state_test",
        workdir="/tmp",
        max_parallel=1,
        fail_fast=False,
        tasks={
            "t1": TaskState(
                status="SUCCESS",
                depends_on=[],
                cmd=["echo", "ok"],
                cwd=".",
                env=None,
                timeout_sec=None,
                retries=0,
                retry_backoff_sec=[],
                outputs=[],
                attempts=1,
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:00:01+00:00",
                duration_sec=1.0,
                exit_code=0,
                stdout_path="logs/t1.out.log",
                stderr_path="logs/t1.err.log",
                artifact_paths=["artifacts/t1/out.txt"],
            )
        },
    )
    return state.to_dict()


def test_save_and_load_state_atomic(tmp_path: Path) -> None:
    home = tmp_path / ".orch"
    run_dir = home / "runs" / "run1"
    run_dir.mkdir(parents=True)
    state = RunState(
        run_id="run1",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        status="SUCCESS",
        goal="demo",
        plan_relpath="plan.yaml",
        home=str(home),
        workdir=str(tmp_path),
        max_parallel=2,
        fail_fast=False,
        tasks={
            "t1": TaskState(
                status="SUCCESS",
                depends_on=[],
                cmd=["echo", "ok"],
                cwd=".",
                env=None,
                timeout_sec=None,
                retries=0,
                retry_backoff_sec=[],
                outputs=[],
                attempts=1,
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:00:01+00:00",
                duration_sec=1.0,
                exit_code=0,
            )
        },
    )

    save_state_atomic(run_dir, state)
    loaded = load_state(run_dir)
    assert loaded.run_id == "run1"
    assert loaded.tasks["t1"].status == "SUCCESS"
    assert not (run_dir / "state.json.tmp").exists()


def test_load_state_rejects_invalid_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(StateError):
        load_state(run_dir)


def test_load_state_rejects_non_object_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_list"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(json.dumps(["not", "object"]), encoding="utf-8")

    with pytest.raises(StateError):
        load_state(run_dir)


def test_load_state_rejects_incomplete_object(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_incomplete"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{}", encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field"):
        load_state(run_dir)


def test_load_state_rejects_invalid_timestamps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["created_at"] = "not-iso"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: created_at"):
        load_state(run_dir)


def test_load_state_rejects_unsafe_plan_relpath(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_plan_relpath"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["plan_relpath"] = "../plan.yaml"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: plan_relpath"):
        load_state(run_dir)


def test_load_state_rejects_non_absolute_home_and_workdir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_paths"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["home"] = ".orch"
    payload["workdir"] = "."
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: home"):
        load_state(run_dir)


def test_load_state_rejects_home_mismatch_with_run_directory(tmp_path: Path) -> None:
    run_id = "run_bad_home_mismatch"
    run_dir = tmp_path / ".orch" / "runs" / run_id
    run_dir.mkdir(parents=True)
    payload = _minimal_state_payload(run_id=run_id)
    payload["home"] = str((tmp_path / "other_home").resolve())
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="state home does not match directory"):
        load_state(run_dir)


def test_load_state_rejects_naive_timestamp(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_naive_ts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["created_at"] = "2026-01-01T00:00:00"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: created_at"):
        load_state(run_dir)


def test_load_state_rejects_updated_at_earlier_than_created_at(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ts_order"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["created_at"] = "2026-01-02T00:00:00+00:00"
    payload["updated_at"] = "2026-01-01T00:00:00+00:00"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: updated_at"):
        load_state(run_dir)


def test_load_state_rejects_non_utf8_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_non_utf8"
    run_dir.mkdir()
    (run_dir / "state.json").write_bytes(b"\xff\xfe\xfd")

    with pytest.raises(StateError, match="failed to decode state file as utf-8"):
        load_state(run_dir)


def test_load_state_wraps_read_oserror(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_state_dir"
    run_dir.mkdir()
    (run_dir / "state.json").mkdir()

    with pytest.raises(StateError, match="failed to read state file"):
        load_state(run_dir)


def test_load_state_rejects_unsafe_task_log_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_log_path"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stdout_path"] = "../escape.log"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_log_path_for_different_task_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_log_task_binding"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stdout_path"] = "logs/t2.out.log"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_unexpected_log_filename_pattern(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_log_filename"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stderr_path"] = "logs/t1.custom.log"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_unsafe_artifact_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_artifact_path"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["artifact_paths"] = ["../outside.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_artifact_path_for_different_task_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_artifact_task_binding"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["artifact_paths"] = ["artifacts/t2/out.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_duplicate_artifact_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dup_artifacts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["artifact_paths"] = ["artifacts/t1/out.txt", "artifacts/t1/OUT.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_identical_stdout_and_stderr_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_same_log_paths"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stdout_path"] = "logs/t1.out.log"
    task["stderr_path"] = "logs/t1.out.log"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_case_insensitive_duplicate_task_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_case_dup_tasks"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["tasks"] = {
        "Build": {
            "status": "SUCCESS",
            "depends_on": [],
            "cmd": ["echo", "ok"],
            "cwd": ".",
            "env": None,
            "timeout_sec": None,
            "retries": 0,
            "retry_backoff_sec": [],
            "outputs": [],
            "attempts": 1,
            "stdout_path": "logs/Build.out.log",
            "stderr_path": "logs/Build.err.log",
            "artifact_paths": ["artifacts/Build/out.txt"],
        },
        "build": {
            "status": "SUCCESS",
            "depends_on": [],
            "cmd": ["echo", "ok"],
            "cwd": ".",
            "env": None,
            "timeout_sec": None,
            "retries": 0,
            "retry_backoff_sec": [],
            "outputs": [],
            "attempts": 1,
            "stdout_path": "logs/build.out.log",
            "stderr_path": "logs/build.err.log",
            "artifact_paths": ["artifacts/build/out.txt"],
        },
    }
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_run_id_mismatch_with_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_mismatch"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id="different_run")
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="run_id does not match"):
        load_state(run_dir)


def test_load_state_rejects_negative_attempts_in_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["attempts"] = -1
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_non_finite_backoff_in_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_backoff"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["retry_backoff_sec"] = [0.1, float("inf")]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_missing_timestamps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_timestamps"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["started_at"] = None
    task["ended_at"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_with_skip_reason(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_skip_reason"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["skip_reason"] = "should_not_exist"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_without_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_no_exit_code"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["exit_code"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_with_zero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_zero_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["attempts"] = 0
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_without_started_at(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_started"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_with_terminal_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_terminal_fields"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_with_ended_at(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_ended_at"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_with_zero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 0
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_accepts_ready_task_after_timeout_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_ready_after_timeout"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "READY"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = None
    task["timed_out"] = True
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_state(run_dir)
    assert loaded.status == "RUNNING"
    assert loaded.tasks["t1"].status == "READY"
    assert loaded.tasks["t1"].timed_out is True


def test_load_state_rejects_ready_task_with_attempts_exhausted(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ready_attempts_exhausted"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "READY"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 2
    task["retries"] = 1
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_ready_task_with_success_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ready_success_exit"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "READY"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 0
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_failed_timeout_task_with_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_timeout_exit_code"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "FAILED"
    task["exit_code"] = 124
    task["timed_out"] = True
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_inconsistent_task_exit_and_flags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_task_flags"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "SUCCESS"
    task["exit_code"] = 1
    task["timed_out"] = True
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_failed_task_with_zero_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_zero_exit"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "FAILED"
    task["exit_code"] = 0
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_failed_task_with_zero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_zero_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "FAILED"
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 0
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_skipped_task_with_runtime_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_skipped_runtime_fields"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "SKIPPED"
    task["skip_reason"] = "dependency_not_success"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_skipped_task_with_nonzero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_skipped_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "SKIPPED"
    task["skip_reason"] = "dependency_not_success"
    task["started_at"] = None
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["attempts"] = 1
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_attempts_exceeding_retry_budget(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_attempt_budget"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["retries"] = 0
    task["attempts"] = 3
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_invalid_task_timestamps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_task_timestamps"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["started_at"] = "not-iso"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_end_before_start(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_task_time_order"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["started_at"] = "2026-01-02T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:00+00:00"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_unknown_dependency_in_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_dep_ref"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["depends_on"] = ["missing"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_status_with_non_success_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_status"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "SUCCESS"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "FAILED"
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_canceled_status_without_canceled_tasks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_canceled_status"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "CANCELED"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_terminal_run_status_with_running_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_with_running"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_running_status_when_all_tasks_terminal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_all_terminal"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_pending_run_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_pending_status"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "PENDING"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_failed_run_status_with_canceled_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_with_canceled"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    t1 = tasks["t1"]
    assert isinstance(t1, dict)
    t1["status"] = "FAILED"
    t1["exit_code"] = 1
    t1["timed_out"] = False
    t1["canceled"] = False
    t1["skip_reason"] = None
    tasks["t2"] = {
        "status": "CANCELED",
        "depends_on": [],
        "cmd": ["echo", "ok"],
        "cwd": ".",
        "env": None,
        "timeout_sec": None,
        "retries": 0,
        "retry_backoff_sec": [],
        "outputs": [],
        "attempts": 1,
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:01+00:00",
        "duration_sec": 1.0,
        "exit_code": None,
        "timed_out": False,
        "canceled": True,
        "skip_reason": "run_canceled",
        "stdout_path": "logs/t2.out.log",
        "stderr_path": "logs/t2.err.log",
        "artifact_paths": [],
    }
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)
