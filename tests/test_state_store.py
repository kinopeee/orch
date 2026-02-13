from __future__ import annotations

import json
from pathlib import Path

import pytest

from orch.state.model import RunState, TaskState
from orch.state.store import load_state, save_state_atomic
from orch.util.errors import StateError


def _minimal_state_payload() -> dict[str, object]:
    state = RunState(
        run_id="run_min",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        status="RUNNING",
        goal=None,
        plan_relpath="plan.yaml",
        home=".orch",
        workdir=".",
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
                stdout_path="logs/t1.out.log",
                stderr_path="logs/t1.err.log",
                artifact_paths=["artifacts/t1/out.txt"],
            )
        },
    )
    return state.to_dict()


def test_save_and_load_state_atomic(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    state = RunState(
        run_id="run1",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        status="RUNNING",
        goal="demo",
        plan_relpath="plan.yaml",
        home=".orch",
        workdir=".",
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
    payload = _minimal_state_payload()
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
    payload = _minimal_state_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stdout_path"] = "logs/t2.out.log"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_unsafe_artifact_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_artifact_path"
    run_dir.mkdir()
    payload = _minimal_state_payload()
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
    payload = _minimal_state_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["artifact_paths"] = ["artifacts/t2/out.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_case_insensitive_duplicate_task_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_case_dup_tasks"
    run_dir.mkdir()
    payload = _minimal_state_payload()
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
