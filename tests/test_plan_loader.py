from __future__ import annotations

from pathlib import Path

import pytest

from orch.config.loader import load_plan
from orch.util.errors import PlanError


def test_load_plan_normalizes_string_cmd(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
goal: test
tasks:
  - id: a
    cmd: "python -c 'print(1)'"
""".strip(),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)
    assert plan.tasks[0].cmd == ["python", "-c", "print(1)"]


def test_load_plan_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: a
    cmd: ["echo", "x"]
  - id: a
    cmd: ["echo", "y"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(PlanError):
        load_plan(plan_path)
