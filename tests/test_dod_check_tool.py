from __future__ import annotations

import importlib.util
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


def test_dod_check_parse_args_defaults() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args([])  # type: ignore[attr-defined]
    assert parsed.skip_quality_gates is False
    assert parsed.home == (module.ROOT / ".orch").resolve()  # type: ignore[attr-defined]


def test_dod_check_parse_args_resolves_relative_home() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args(["--home", "tmp/dod-home"])  # type: ignore[attr-defined]
    assert parsed.home == (module.ROOT / "tmp/dod-home").resolve()  # type: ignore[attr-defined]


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
