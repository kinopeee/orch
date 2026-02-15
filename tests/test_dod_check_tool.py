from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_dod_check_module() -> object:
    module_path = Path(__file__).resolve().parents[1] / "tools" / "dod_check.py"
    spec = importlib.util.spec_from_file_location("dod_check_tool_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dod_check_intervals_overlap_true_for_parallel_window() -> None:
    module = _load_dod_check_module()
    assert module._intervals_overlap(  # type: ignore[attr-defined]
        "2026-02-15T15:00:00+00:00",
        "2026-02-15T15:00:02+00:00",
        "2026-02-15T15:00:01+00:00",
        "2026-02-15T15:00:03+00:00",
    )


def test_dod_check_intervals_overlap_false_for_touching_boundaries() -> None:
    module = _load_dod_check_module()
    assert not module._intervals_overlap(  # type: ignore[attr-defined]
        "2026-02-15T15:00:00+00:00",
        "2026-02-15T15:00:01+00:00",
        "2026-02-15T15:00:01+00:00",
        "2026-02-15T15:00:02+00:00",
    )


def test_dod_check_parse_args_skip_quality_gates_flag() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args(["--skip-quality-gates"])  # type: ignore[attr-defined]
    assert parsed.skip_quality_gates is True
    assert parsed.home == (module.ROOT / ".orch").resolve()  # type: ignore[attr-defined]
    assert parsed.emit_json is False
    assert parsed.json_out is None


def test_dod_check_parse_args_defaults() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args([])  # type: ignore[attr-defined]
    assert parsed.skip_quality_gates is False
    assert parsed.home == (module.ROOT / ".orch").resolve()  # type: ignore[attr-defined]
    assert parsed.emit_json is False
    assert parsed.json_out is None


def test_dod_check_parse_args_resolves_relative_home() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args(["--home", "tmp/dod-home"])  # type: ignore[attr-defined]
    assert parsed.home == (module.ROOT / "tmp/dod-home").resolve()  # type: ignore[attr-defined]


def test_dod_check_parse_args_keeps_absolute_home() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args(["--home", "/tmp/dod-home-absolute"])  # type: ignore[attr-defined]
    assert parsed.home == Path("/tmp/dod-home-absolute")


def test_dod_check_parse_args_enables_json_summary() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args(["--json"])  # type: ignore[attr-defined]
    assert parsed.emit_json is True
    assert parsed.json_out is None


def test_dod_check_parse_args_resolves_json_out_relative_path() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args(["--json-out", "tmp/dod-summary.json"])  # type: ignore[attr-defined]
    assert parsed.json_out == (module.ROOT / "tmp/dod-summary.json").resolve()  # type: ignore[attr-defined]


def test_dod_check_parse_args_keeps_json_out_absolute_path() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args(["--json-out", "/tmp/dod-summary.json"])  # type: ignore[attr-defined]
    assert parsed.json_out == Path("/tmp/dod-summary.json")


def test_dod_check_parse_args_supports_json_and_json_out_together() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args(  # type: ignore[attr-defined]
        ["--json", "--json-out", "tmp/dod-summary.json"]
    )
    assert parsed.emit_json is True
    assert parsed.json_out == (module.ROOT / "tmp/dod-summary.json").resolve()  # type: ignore[attr-defined]


def test_dod_check_has_parallel_overlap_true_for_successful_root_tasks() -> None:
    module = _load_dod_check_module()
    state = {
        "tasks": {
            "inspect_a": {
                "status": "SUCCESS",
                "depends_on": [],
                "started_at": "2026-02-15T15:00:00+00:00",
                "ended_at": "2026-02-15T15:00:02+00:00",
            },
            "inspect_b": {
                "status": "SUCCESS",
                "depends_on": [],
                "started_at": "2026-02-15T15:00:01+00:00",
                "ended_at": "2026-02-15T15:00:03+00:00",
            },
        }
    }
    assert module._has_parallel_overlap(state)  # type: ignore[attr-defined]


def test_dod_check_has_parallel_overlap_false_without_root_overlap() -> None:
    module = _load_dod_check_module()
    state = {
        "tasks": {
            "inspect_a": {
                "status": "SUCCESS",
                "depends_on": [],
                "started_at": "2026-02-15T15:00:00+00:00",
                "ended_at": "2026-02-15T15:00:01+00:00",
            },
            "inspect_b": {
                "status": "SUCCESS",
                "depends_on": [],
                "started_at": "2026-02-15T15:00:01+00:00",
                "ended_at": "2026-02-15T15:00:02+00:00",
            },
            "build": {
                "status": "SUCCESS",
                "depends_on": ["inspect_a", "inspect_b"],
                "started_at": "2026-02-15T15:00:00+00:00",
                "ended_at": "2026-02-15T15:00:02+00:00",
            },
        }
    }
    assert not module._has_parallel_overlap(state)  # type: ignore[attr-defined]


def test_dod_check_intervals_overlap_rejects_naive_timestamps() -> None:
    module = _load_dod_check_module()
    with pytest.raises(RuntimeError, match="timezone offset"):
        module._intervals_overlap(  # type: ignore[attr-defined]
            "2026-02-15T15:00:00",
            "2026-02-15T15:00:02",
            "2026-02-15T15:00:01+00:00",
            "2026-02-15T15:00:03+00:00",
        )


def test_dod_check_intervals_overlap_rejects_invalid_timestamp_format() -> None:
    module = _load_dod_check_module()
    with pytest.raises(RuntimeError, match="invalid timestamp format"):
        module._intervals_overlap(  # type: ignore[attr-defined]
            "not-a-timestamp",
            "2026-02-15T15:00:02+00:00",
            "2026-02-15T15:00:01+00:00",
            "2026-02-15T15:00:03+00:00",
        )


def test_dod_check_detect_orch_prefix_prefers_orch_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_dod_check_module()

    class Completed:
        returncode = 0

    def fake_run(*args: object, **kwargs: object) -> Completed:
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)  # type: ignore[attr-defined]
    assert module._detect_orch_prefix() == ["orch"]  # type: ignore[attr-defined]


def test_dod_check_detect_orch_prefix_falls_back_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_dod_check_module()

    class Completed:
        returncode = 1

    def fake_run(*args: object, **kwargs: object) -> Completed:
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)  # type: ignore[attr-defined]
    assert module._detect_orch_prefix() == [sys.executable, "-m", "orch.cli"]  # type: ignore[attr-defined]


def test_dod_check_detect_orch_prefix_falls_back_when_orch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_dod_check_module()

    def fake_run(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(module.subprocess, "run", fake_run)  # type: ignore[attr-defined]
    assert module._detect_orch_prefix() == [sys.executable, "-m", "orch.cli"]  # type: ignore[attr-defined]


def test_dod_check_run_raises_runtime_error_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_dod_check_module()

    def fake_run(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["orch", "run"], timeout=1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="command timed out"):
        module._run(["orch", "run"], title="timeout test", timeout_sec=1)  # type: ignore[attr-defined]


def test_dod_check_run_passes_custom_timeout_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_dod_check_module()
    captured_timeout: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args: object, **kwargs: object) -> Completed:
        captured_timeout["value"] = kwargs.get("timeout")
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)  # type: ignore[attr-defined]
    module._run(["orch", "status"], title="timeout passthrough", timeout_sec=12.5)  # type: ignore[attr-defined]
    assert captured_timeout["value"] == 12.5


def test_dod_check_run_uses_default_timeout_when_not_specified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_dod_check_module()
    captured_timeout: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args: object, **kwargs: object) -> Completed:
        captured_timeout["value"] = kwargs.get("timeout")
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)  # type: ignore[attr-defined]
    module._run(["orch", "status"], title="default timeout")  # type: ignore[attr-defined]
    assert captured_timeout["value"] == 180.0


def test_dod_check_parse_run_id_extracts_value() -> None:
    module = _load_dod_check_module()
    run_id = module._parse_run_id("run_id: 20260215_000000_abcdef\nstate: SUCCESS\n")  # type: ignore[attr-defined]
    assert run_id == "20260215_000000_abcdef"


def test_dod_check_parse_run_id_raises_when_missing() -> None:
    module = _load_dod_check_module()
    with pytest.raises(RuntimeError, match="run_id not found"):
        module._parse_run_id("state: SUCCESS\n")  # type: ignore[attr-defined]


def test_dod_check_has_parallel_overlap_rejects_non_object_tasks() -> None:
    module = _load_dod_check_module()
    with pytest.raises(RuntimeError, match="state.tasks must be an object"):
        module._has_parallel_overlap({"tasks": []})  # type: ignore[attr-defined]


def test_dod_check_state_tasks_returns_task_mapping() -> None:
    module = _load_dod_check_module()
    tasks = module._state_tasks(  # type: ignore[attr-defined]
        {"tasks": {"inspect": {"status": "SUCCESS", "depends_on": []}}}
    )
    assert tasks["inspect"]["status"] == "SUCCESS"


def test_dod_check_state_tasks_rejects_non_object_task_state() -> None:
    module = _load_dod_check_module()
    with pytest.raises(RuntimeError, match="task state must be an object: inspect"):
        module._state_tasks({"tasks": {"inspect": []}})  # type: ignore[attr-defined]


def test_dod_check_state_tasks_rejects_non_string_task_id() -> None:
    module = _load_dod_check_module()
    with pytest.raises(RuntimeError, match="task id must be a string"):
        module._state_tasks({"tasks": {1: {"status": "SUCCESS"}}})  # type: ignore[attr-defined]


def test_dod_check_successful_task_snapshots_collects_success_entries() -> None:
    module = _load_dod_check_module()
    snapshots = module._successful_task_snapshots(  # type: ignore[attr-defined]
        {
            "tasks": {
                "inspect": {
                    "status": "SUCCESS",
                    "attempts": 1,
                    "started_at": "2026-02-15T17:00:00+00:00",
                },
                "build": {
                    "status": "FAILED",
                    "attempts": 1,
                    "started_at": "2026-02-15T17:00:01+00:00",
                },
            }
        }
    )
    assert snapshots == {"inspect": (1, "2026-02-15T17:00:00+00:00")}


def test_dod_check_successful_task_snapshots_rejects_non_int_attempts() -> None:
    module = _load_dod_check_module()
    with pytest.raises(RuntimeError, match="task attempts must be int: inspect"):
        module._successful_task_snapshots(  # type: ignore[attr-defined]
            {
                "tasks": {
                    "inspect": {
                        "status": "SUCCESS",
                        "attempts": "1",
                        "started_at": "2026-02-15T17:00:00+00:00",
                    }
                }
            }
        )


def test_dod_check_assert_resume_kept_successful_tasks_unchanged_passes() -> None:
    module = _load_dod_check_module()
    baseline = {"inspect": (1, "2026-02-15T17:00:00+00:00")}
    resumed_state = {
        "tasks": {
            "inspect": {
                "status": "SUCCESS",
                "attempts": 1,
                "started_at": "2026-02-15T17:00:00+00:00",
            }
        }
    }
    module._assert_resume_kept_successful_tasks_unchanged(  # type: ignore[attr-defined]
        baseline, resumed_state
    )


def test_dod_check_assert_resume_kept_successful_tasks_unchanged_rejects_attempt_change() -> None:
    module = _load_dod_check_module()
    baseline = {"inspect": (1, "2026-02-15T17:00:00+00:00")}
    resumed_state = {
        "tasks": {
            "inspect": {
                "status": "SUCCESS",
                "attempts": 2,
                "started_at": "2026-02-15T17:00:00+00:00",
            }
        }
    }
    with pytest.raises(RuntimeError, match="resume changed attempts"):
        module._assert_resume_kept_successful_tasks_unchanged(  # type: ignore[attr-defined]
            baseline, resumed_state
        )


def test_dod_check_assert_resume_kept_successful_tasks_unchanged_rejects_started_at_change() -> (
    None
):
    module = _load_dod_check_module()
    baseline = {"inspect": (1, "2026-02-15T17:00:00+00:00")}
    resumed_state = {
        "tasks": {
            "inspect": {
                "status": "SUCCESS",
                "attempts": 1,
                "started_at": "2026-02-15T17:00:01+00:00",
            }
        }
    }
    with pytest.raises(RuntimeError, match="resume changed started_at"):
        module._assert_resume_kept_successful_tasks_unchanged(  # type: ignore[attr-defined]
            baseline, resumed_state
        )


def test_dod_check_assert_resume_kept_successful_tasks_unchanged_rejects_missing_task() -> None:
    module = _load_dod_check_module()
    baseline = {"inspect": (1, "2026-02-15T17:00:00+00:00")}
    resumed_state = {"tasks": {}}
    with pytest.raises(RuntimeError, match="task missing after resume"):
        module._assert_resume_kept_successful_tasks_unchanged(  # type: ignore[attr-defined]
            baseline, resumed_state
        )


def test_dod_check_assert_report_exists_passes_when_file_exists(tmp_path: Path) -> None:
    module = _load_dod_check_module()
    run_id = "20260215_000000_abcdef"
    report = tmp_path / run_id / "report" / "final_report.md"
    report.parent.mkdir(parents=True)
    report.write_text("# report\n", encoding="utf-8")
    module._assert_report_exists(run_id, tmp_path)  # type: ignore[attr-defined]


def test_dod_check_assert_report_exists_raises_when_missing(tmp_path: Path) -> None:
    module = _load_dod_check_module()
    with pytest.raises(RuntimeError, match="final report file was not generated"):
        module._assert_report_exists("20260215_000000_missing", tmp_path)  # type: ignore[attr-defined]


def test_dod_check_assert_run_status_passes_when_status_matches(tmp_path: Path) -> None:
    module = _load_dod_check_module()
    run_id = "20260215_000000_status_ok"
    state_path = tmp_path / run_id / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text('{"status":"SUCCESS","tasks":{}}', encoding="utf-8")
    module._assert_run_status(run_id, tmp_path, "SUCCESS")  # type: ignore[attr-defined]


def test_dod_check_assert_run_status_raises_when_status_mismatch(tmp_path: Path) -> None:
    module = _load_dod_check_module()
    run_id = "20260215_000000_status_ng"
    state_path = tmp_path / run_id / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text('{"status":"FAILED","tasks":{}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="run status mismatch"):
        module._assert_run_status(run_id, tmp_path, "SUCCESS")  # type: ignore[attr-defined]


def test_dod_check_build_summary_payload_contains_all_keys() -> None:
    module = _load_dod_check_module()
    payload = module._build_summary_payload(  # type: ignore[attr-defined]
        basic_run_id="basic123",
        parallel_run_id="parallel123",
        fail_run_id="fail123",
        cancel_run_id="cancel123",
        home=Path("/tmp/dod-home"),
    )
    assert payload == {
        "result": "PASS",
        "basic_run_id": "basic123",
        "parallel_run_id": "parallel123",
        "fail_run_id": "fail123",
        "cancel_run_id": "cancel123",
        "home": "/tmp/dod-home",
    }


def test_dod_check_format_key_set_for_error_supports_mixed_key_types() -> None:
    module = _load_dod_check_module()
    keys = {"result", 1}
    formatted = module._format_key_set_for_error(keys)  # type: ignore[attr-defined]
    assert formatted == sorted(repr(key) for key in keys)


def test_dod_check_assert_summary_payload_consistent_accepts_valid_payload() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
    }
    module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_non_mapping_payload() -> None:
    module = _load_dod_check_module()
    payload = ["result", "PASS"]
    with pytest.raises(RuntimeError, match="invalid summary payload type: list"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[arg-type, attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_missing_key() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
    }
    with pytest.raises(RuntimeError, match="invalid summary keys"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_extra_key() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
        "unexpected": "value",
    }
    with pytest.raises(RuntimeError, match="invalid summary keys"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_handles_non_string_extra_key() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
        1: "value",
    }
    with pytest.raises(RuntimeError, match=r"invalid summary keys: .*'1'.*"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_invalid_run_id() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "invalid",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
    }
    with pytest.raises(RuntimeError, match="invalid summary run_id"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_non_string_run_id() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": 123456,
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
    }
    with pytest.raises(RuntimeError, match="invalid summary value type: basic_run_id=int"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_non_string_home() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": 100,
    }
    with pytest.raises(RuntimeError, match="invalid summary value type: home=int"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_non_string_result() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": 1,
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
    }
    with pytest.raises(RuntimeError, match="invalid summary value type: result=int"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_uppercase_run_id() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_A1B2C3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
    }
    with pytest.raises(RuntimeError, match="invalid summary run_id"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_duplicate_run_ids() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000001_d4e5f6",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
    }
    with pytest.raises(RuntimeError, match="invalid summary run_id uniqueness"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_invalid_result() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "FAILED",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
    }
    with pytest.raises(RuntimeError, match="invalid summary result"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_empty_home() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "",
    }
    with pytest.raises(RuntimeError, match="invalid summary home"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_whitespace_home() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "   ",
    }
    with pytest.raises(RuntimeError, match="invalid summary home"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_surrounding_whitespace_home() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": " /tmp/dod-home ",
    }
    with pytest.raises(RuntimeError, match="invalid summary home: surrounding whitespace"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_assert_summary_payload_consistent_rejects_relative_home() -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "tmp/dod-home",
    }
    with pytest.raises(RuntimeError, match="invalid summary home: not absolute"):
        module._assert_summary_payload_consistent(payload)  # type: ignore[attr-defined]


def test_dod_check_write_summary_json_creates_file(tmp_path: Path) -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
    }
    out_path = tmp_path / "nested" / "dod-summary.json"
    module._write_summary_json(out_path, payload)  # type: ignore[attr-defined]
    written = out_path.read_text(encoding="utf-8")
    assert written == json.dumps(payload, sort_keys=True) + "\n"


def test_dod_check_write_summary_json_raises_runtime_error_on_io_failure(tmp_path: Path) -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
        "home": "/tmp/dod-home",
    }
    out_dir = tmp_path / "as-directory"
    out_dir.mkdir()
    with pytest.raises(RuntimeError, match="failed to write summary json"):
        module._write_summary_json(out_dir, payload)  # type: ignore[attr-defined]


def test_dod_check_write_summary_json_rejects_invalid_payload(tmp_path: Path) -> None:
    module = _load_dod_check_module()
    payload = {
        "result": "PASS",
        "basic_run_id": "20260215_000000_a1b2c3",
        "parallel_run_id": "20260215_000001_d4e5f6",
        "fail_run_id": "20260215_000002_0a1b2c",
        "cancel_run_id": "20260215_000003_3d4e5f",
    }
    out_path = tmp_path / "nested" / "dod-summary.json"
    with pytest.raises(RuntimeError, match="invalid summary keys"):
        module._write_summary_json(out_path, payload)  # type: ignore[attr-defined]
    assert not out_path.exists()
    assert not out_path.parent.exists()


def test_dod_check_write_summary_json_rejects_non_mapping_payload(tmp_path: Path) -> None:
    module = _load_dod_check_module()
    out_path = tmp_path / "nested" / "dod-summary.json"
    payload = ["result", "PASS"]
    with pytest.raises(RuntimeError, match="invalid summary payload type: list"):
        module._write_summary_json(out_path, payload)  # type: ignore[arg-type, attr-defined]
    assert not out_path.exists()
    assert not out_path.parent.exists()
