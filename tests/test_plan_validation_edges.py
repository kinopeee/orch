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


def test_load_plan_rejects_empty_cmd_list(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: []
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_cmd_list_with_empty_string(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", ""]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_whitespace_only_task_id_and_cmd_entry(tmp_path: Path) -> None:
    plan_id = tmp_path / "plan_id.yaml"
    _write(
        plan_id,
        """
        tasks:
          - id: "   "
            cmd: ["python3", "-c", "print('x')"]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_id)

    plan_cmd = tmp_path / "plan_cmd.yaml"
    _write(
        plan_cmd,
        """
        tasks:
          - id: t1
            cmd: ["python3", "   "]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_cmd)


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


def test_load_plan_rejects_bool_for_retries(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            retries: true
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_bool_for_timeout_and_backoff(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            timeout_sec: true
            retry_backoff_sec: [1, false]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_non_string_goal(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        goal: 123
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_non_string_artifacts_dir(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        artifacts_dir: [1,2,3]
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_empty_string_cmd(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: "   "
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_empty_cwd(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            cwd: ""
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)

    plan_ws = tmp_path / "plan_ws.yaml"
    _write(
        plan_ws,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            cwd: "   "
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_ws)


def test_load_plan_rejects_non_string_env_entries(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            env: {"OK": "1", "BAD": 2}
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_non_list_depends_on_and_outputs(tmp_path: Path) -> None:
    plan_dep = tmp_path / "plan_dep.yaml"
    _write(
        plan_dep,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            depends_on: "other"
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_dep)

    plan_out = tmp_path / "plan_out.yaml"
    _write(
        plan_out,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            outputs: "dist/**"
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_out)


def test_load_plan_rejects_empty_string_items_in_depends_on_and_outputs(tmp_path: Path) -> None:
    plan_dep = tmp_path / "plan_dep.yaml"
    _write(
        plan_dep,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            depends_on: [""]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_dep)

    plan_out = tmp_path / "plan_out.yaml"
    _write(
        plan_out,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            outputs: [""]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_out)

    plan_dep_ws = tmp_path / "plan_dep_ws.yaml"
    _write(
        plan_dep_ws,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            depends_on: ["   "]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_dep_ws)
