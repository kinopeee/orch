from __future__ import annotations

from pathlib import Path

from orch.config.loader import load_plan


def test_load_plan_parses_full_task_fields(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
goal: "demo goal"
artifacts_dir: ".orch/artifacts"
tasks:
  - id: build
    cmd: ["python3", "-c", "print('x')"]
    depends_on: []
    cwd: "."
    env: {"KEY": "VALUE"}
    timeout_sec: 1
    retries: 2
    retry_backoff_sec: [0.1, 0.2]
    outputs: ["dist/**", "report.json"]
""".strip(),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)
    task = plan.tasks[0]
    assert plan.goal == "demo goal"
    assert plan.artifacts_dir == ".orch/artifacts"
    assert task.id == "build"
    assert task.cmd == ["python3", "-c", "print('x')"]
    assert task.cwd == "."
    assert task.env == {"KEY": "VALUE"}
    assert task.timeout_sec == 1.0
    assert task.retries == 2
    assert task.retry_backoff_sec == [0.1, 0.2]
    assert task.outputs == ["dist/**", "report.json"]


def test_load_plan_normalizes_quoted_string_cmd(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: cmd
    cmd: "python3 -c \\"print('hello world')\\""
""".strip(),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)
    assert plan.tasks[0].cmd == ["python3", "-c", "print('hello world')"]
