from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def test_dod_check_parse_args_defaults() -> None:
    module = _load_dod_check_module()
    parsed = module._parse_args([])  # type: ignore[attr-defined]
    assert parsed.skip_quality_gates is False
