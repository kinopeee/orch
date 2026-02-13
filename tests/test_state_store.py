from __future__ import annotations

import json
from pathlib import Path

import pytest

from orch.state.model import RunState, TaskState
from orch.state.store import load_state, save_state_atomic
from orch.util.errors import StateError


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
