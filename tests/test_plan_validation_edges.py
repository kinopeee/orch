from __future__ import annotations

from pathlib import Path

import pytest

from orch.config.loader import load_plan
from orch.util.errors import PlanError


def _write(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_load_plan_rejects_non_string_cmd_entries(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", 1]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_non_str_or_list_cmd(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: 123
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_non_positive_timeout(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            timeout_sec: 0
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_negative_retries(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            retries: -1
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_invalid_retry_backoff_values(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            retry_backoff_sec: [1, -1]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)
