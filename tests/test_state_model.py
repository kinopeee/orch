from __future__ import annotations

from orch.state.model import RunState, TaskState


def test_task_state_from_dict_filters_invalid_types() -> None:
    raw: dict[str, object] = {
        "status": "SUCCESS",
        "depends_on": ["a", 1, "b"],
        "cmd": ["python3", "-c", "print(1)", None],
        "env": {"A": "1", "B": 2},
        "timeout_sec": 3,
        "retries": 2,
        "retry_backoff_sec": [0.1, "x", 0.2, True, float("inf")],
        "outputs": ["dist/**", 1],
        "attempts": 1,
        "exit_code": True,
        "duration_sec": float("nan"),
        "artifact_paths": ["a.txt", 2],
    }

    state = TaskState.from_dict(raw)

    assert state.status == "SUCCESS"
    assert state.depends_on == ["a", "b"]
    assert state.cmd == ["python3", "-c", "print(1)"]
    assert state.env == {"A": "1"}
    assert state.retry_backoff_sec == [0.1, 0.2]
    assert state.outputs == ["dist/**"]
    assert state.artifact_paths == ["a.txt"]
    assert state.exit_code is None
    assert state.duration_sec is None


def test_run_state_from_dict_uses_safe_defaults_for_invalid_values() -> None:
    raw: dict[str, object] = {
        "run_id": "r1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:10+00:00",
        "status": "UNKNOWN",
        "plan_relpath": "plan.yaml",
        "home": ".orch",
        "workdir": ".",
        "max_parallel": "x",
        "fail_fast": "no",
        "tasks": {
            "ok": {"status": "READY", "cmd": ["echo", "ok"], "depends_on": []},
            "bad": "skip",
        },
    }

    state = RunState.from_dict(raw)

    assert state.status == "PENDING"
    assert state.max_parallel == 0
    assert state.fail_fast is False
    assert set(state.tasks.keys()) == {"ok"}
    assert state.tasks["ok"].status == "READY"


def test_run_state_from_dict_ignores_bool_and_nonfinite_numbers() -> None:
    raw: dict[str, object] = {
        "run_id": "r2",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:10+00:00",
        "status": "RUNNING",
        "plan_relpath": "plan.yaml",
        "home": ".orch",
        "workdir": ".",
        "max_parallel": True,
        "fail_fast": False,
        "tasks": {
            "t1": {
                "status": "FAILED",
                "cmd": ["echo", "ok"],
                "depends_on": [],
                "timeout_sec": float("inf"),
                "retry_backoff_sec": [1, True, float("nan")],
            }
        },
    }

    state = RunState.from_dict(raw)

    assert state.max_parallel == 0
    assert state.tasks["t1"].timeout_sec is None
    assert state.tasks["t1"].retry_backoff_sec == [1.0]
