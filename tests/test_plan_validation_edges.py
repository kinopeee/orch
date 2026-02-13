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


def test_load_plan_rejects_unsafe_task_id_for_paths(tmp_path: Path) -> None:
    plan_slash = tmp_path / "plan_id_slash.yaml"
    _write(
        plan_slash,
        """
        tasks:
          - id: "a/b"
            cmd: ["python3", "-c", "print('x')"]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_slash)

    plan_escape = tmp_path / "plan_id_escape.yaml"
    _write(
        plan_escape,
        """
        tasks:
          - id: "../escape"
            cmd: ["python3", "-c", "print('x')"]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_escape)


def test_load_plan_rejects_too_long_task_id(tmp_path: Path) -> None:
    plan = tmp_path / "plan_long_id.yaml"
    _write(
        plan,
        f"""
        tasks:
          - id: "{"a" * 129}"
            cmd: ["python3", "-c", "print('x')"]
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


def test_load_plan_rejects_non_finite_timeout(tmp_path: Path) -> None:
    plan_inf = tmp_path / "plan_timeout_inf.yaml"
    _write(
        plan_inf,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            timeout_sec: .inf
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_inf)

    plan_nan = tmp_path / "plan_timeout_nan.yaml"
    _write(
        plan_nan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            timeout_sec: .nan
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_nan)


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

    plan_inf = tmp_path / "plan_backoff_inf.yaml"
    _write(
        plan_inf,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            retry_backoff_sec: [1, .inf]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_inf)


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

    blank = tmp_path / "plan_blank_goal.yaml"
    _write(
        blank,
        """
        goal: "   "
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(blank)


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

    blank = tmp_path / "plan_blank_artifacts.yaml"
    _write(
        blank,
        """
        artifacts_dir: "   "
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(blank)


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


def test_load_plan_rejects_malformed_cmd_string(tmp_path: Path) -> None:
    plan = tmp_path / "plan_bad_cmd_quote.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: "python3 -c \"print('oops')"
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_null_byte_in_cmd(tmp_path: Path) -> None:
    plan = tmp_path / "plan_cmd_nul.yaml"
    _write(
        plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "\0bad"]
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

    blank_key = tmp_path / "plan_blank_env_key.yaml"
    _write(
        blank_key,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            env: {"   ": "1"}
        """,
    )
    with pytest.raises(PlanError):
        load_plan(blank_key)

    nul_value = tmp_path / "plan_nul_env_value.yaml"
    _write(
        nul_value,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            env: {"OK": "\0bad"}
        """,
    )
    with pytest.raises(PlanError):
        load_plan(nul_value)

    key_with_equals = tmp_path / "plan_env_key_equals.yaml"
    _write(
        key_with_equals,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            env: {"A=B": "1"}
        """,
    )
    with pytest.raises(PlanError):
        load_plan(key_with_equals)


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


def test_load_plan_rejects_dependency_cycle(tmp_path: Path) -> None:
    plan = tmp_path / "plan_cycle.yaml"
    _write(
        plan,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
            depends_on: ["b"]
          - id: b
            cmd: ["python3", "-c", "print('b')"]
            depends_on: ["a"]
        """,
    )
    with pytest.raises(PlanError, match="cycle"):
        load_plan(plan)


def test_load_plan_rejects_self_dependency(tmp_path: Path) -> None:
    plan = tmp_path / "plan_self_dep.yaml"
    _write(
        plan,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
            depends_on: ["a"]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


def test_load_plan_rejects_duplicate_dependencies(tmp_path: Path) -> None:
    plan = tmp_path / "plan_dup_dep.yaml"
    _write(
        plan,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
          - id: b
            cmd: ["python3", "-c", "print('b')"]
            depends_on: ["a", "a"]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan)


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

    plan_out_ws = tmp_path / "plan_out_ws.yaml"
    _write(
        plan_out_ws,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('x')"]
            outputs: ["   "]
        """,
    )
    with pytest.raises(PlanError):
        load_plan(plan_out_ws)
