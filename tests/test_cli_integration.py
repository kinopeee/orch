from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _write_plan(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _extract_run_id(output: str) -> str:
    match = re.search(r"run_id:\s*([0-9]{8}_[0-9]{6}_[0-9a-f]{6})", output)
    assert match is not None, output
    return match.group(1)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_cli_run_dry_run_returns_zero(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('a')"]
          - id: t2
            cmd: ["python3", "-c", "print('b')"]
            depends_on: ["t1"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert "task_id" in output
    assert "t1" in output
    assert "t2" in output
    assert output.index("t1") < output.index("t2")
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_shows_dependency_chain_order(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_chain.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('a')"]
          - id: t2
            cmd: ["python3", "-c", "print('b')"]
            depends_on: ["t1"]
          - id: t3
            cmd: ["python3", "-c", "print('c')"]
            depends_on: ["t2"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert output.index("t1") < output.index("t2") < output.index("t3")
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_ignores_fail_fast_flag_for_output(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_fail_fast_dry.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: fail_first
            cmd: ["python3", "-c", "import sys; sys.exit(1)"]
          - id: after_fail
            cmd: ["python3", "-c", "print('after')"]
            depends_on: ["fail_first"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert output.index("fail_first") < output.index("after_fail")
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_fail_fast_still_rejects_invalid_home(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_fail_fast_invalid_home.yaml"
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_fail_fast_skips_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_fail_fast_invalid_workdir.yaml"
    home = tmp_path / ".orch_cli"
    invalid_workdir_file = tmp_path / "invalid_workdir_file"
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert "Invalid workdir" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_fail_fast_still_rejects_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_fail_fast_invalid_plan.yaml"
    home = tmp_path / ".orch_cli"
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_no_fail_fast_skips_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_no_fail_fast_invalid_workdir.yaml"
    home = tmp_path / ".orch_cli"
    invalid_workdir_file = tmp_path / "invalid_workdir_no_fail_fast"
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert "Invalid workdir" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_no_fail_fast_still_rejects_invalid_home(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_no_fail_fast_invalid_home.yaml"
    home_file = tmp_path / "home_file_no_fail_fast"
    home_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_no_fail_fast_still_rejects_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_no_fail_fast_invalid_plan.yaml"
    home = tmp_path / ".orch_cli"
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_accepts_both_fail_fast_toggles(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_both_fail_fast_toggles.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert "t1" in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_accepts_both_fail_fast_toggles_reverse_order(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_both_fail_fast_toggles_reverse.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert "t1" in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_fail_fast_toggles_still_rejects_invalid_home(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_both_toggles_invalid_home.yaml"
    home_file = tmp_path / "home_file_both_toggles"
    home_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_fail_fast_toggles_still_rejects_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_both_toggles_invalid_plan.yaml"
    home = tmp_path / ".orch_cli"
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_rejects_invalid_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_invalid_home.yaml"
    home_file = tmp_path / "home_file_both_toggles_reverse"
    home_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_rejects_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_invalid_plan.yaml"
    home = tmp_path / ".orch_cli"
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_fail_fast_toggles_skip_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_both_toggles_invalid_workdir.yaml"
    home = tmp_path / ".orch_cli"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles"
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert "Invalid workdir" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_fail_fast_toggles_preserve_dependency_chain_order(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_chain.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('a')"]
          - id: t2
            cmd: ["python3", "-c", "print('b')"]
            depends_on: ["t1"]
          - id: t3
            cmd: ["python3", "-c", "print('c')"]
            depends_on: ["t2"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert "task_id" in output
    assert output.index("t1") < output.index("t2") < output.index("t3")
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_preserve_dependency_chain_order(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_chain.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('a')"]
          - id: t2
            cmd: ["python3", "-c", "print('b')"]
            depends_on: ["t1"]
          - id: t3
            cmd: ["python3", "-c", "print('c')"]
            depends_on: ["t2"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert "task_id" in output
    assert output.index("t1") < output.index("t2") < output.index("t3")
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_toggles_rejects_symlink_plan_path(
    tmp_path: Path,
) -> None:
    real_plan = tmp_path / "real_plan_both_toggles.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
        """,
    )
    symlink_plan = tmp_path / "plan_symlink_both_toggles.yaml"
    symlink_plan.symlink_to(real_plan)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(symlink_plan),
            "--home",
            str(home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "invalid plan path" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_toggles_reverse_rejects_symlink_plan_path(
    tmp_path: Path,
) -> None:
    real_plan = tmp_path / "real_plan_both_toggles_reverse.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
        """,
    )
    symlink_plan = tmp_path / "plan_symlink_both_toggles_reverse.yaml"
    symlink_plan.symlink_to(real_plan)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(symlink_plan),
            "--home",
            str(home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "invalid plan path" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_toggles_rejects_plan_path_with_symlink_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_plan_parent_both_toggles"
    real_parent.mkdir()
    real_plan = real_parent / "plan.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
        """,
    )
    symlink_parent = tmp_path / "plan_parent_link_both_toggles"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(symlink_parent / "plan.yaml"),
            "--home",
            str(home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "invalid plan path" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_toggles_reverse_rejects_plan_path_with_symlink_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_plan_parent_both_toggles_reverse"
    real_parent.mkdir()
    real_plan = real_parent / "plan.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
        """,
    )
    symlink_parent = tmp_path / "plan_parent_link_both_toggles_reverse"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(symlink_parent / "plan.yaml"),
            "--home",
            str(home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "invalid plan path" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_toggles_symlinked_plan_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("symlink_plan", "symlink_ancestor_plan")

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"plan_vs_workdir_{plan_mode}_{order_label}"
            case_root.mkdir()
            invalid_workdir_file = case_root / "invalid_workdir_file"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            if plan_mode == "symlink_plan":
                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_path = case_root / "plan_symlink.yaml"
                plan_path.symlink_to(real_plan)
            else:
                real_parent = case_root / "real_parent"
                real_parent.mkdir()
                real_plan = real_parent / "plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                symlink_parent = case_root / "plan_parent_link"
                symlink_parent.symlink_to(real_parent, target_is_directory=True)
                plan_path = symlink_parent / "plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Plan validation error" in output, context
            assert "invalid plan path" in output, context
            assert "contains symlink component" not in output, context
            assert "must not include symlink" not in output, context
            assert "must not be symlink" not in output, context
            assert "Invalid workdir" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not home.exists(), context
            assert not (home / "runs").exists(), context


def test_cli_run_dry_run_both_fail_fast_toggles_invalid_home_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_home_vs_workdir.yaml"
    home_file = tmp_path / "home_file_both_toggles_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_home"
    home_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_fail_fast_toggles_invalid_home_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_home_vs_plan.yaml"
    home_file = tmp_path / "home_file_both_toggles_plan"
    home_file.write_text("not a dir\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_non_regular_plan(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    plan_path = tmp_path / "plan_fifo_both_toggles.yaml"
    os.mkfifo(plan_path)
    home_file = tmp_path / "home_file_both_toggles_fifo_plan"
    home_file.write_text("not a dir\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_non_regular_plan_matrix(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    case_names = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for case_name in case_names:
            case_root = tmp_path / f"fifo_precedence_{case_name}_{order_label}"
            case_root.mkdir()
            side_effect_roots: list[Path] = []

            if case_name == "home_file":
                home_path = case_root / "home_file"
                home_path.write_text("not a dir\n", encoding="utf-8")
                side_effect_roots.append(case_root)
            elif case_name == "file_ancestor":
                parent_file = case_root / "home_parent_file"
                parent_file.write_text("not a dir\n", encoding="utf-8")
                home_path = parent_file / "orch_home"
                side_effect_roots.append(case_root)
            elif case_name == "symlink_to_dir":
                real_home = case_root / "real_home"
                real_home.mkdir()
                home_path = case_root / "home_symlink_dir"
                home_path.symlink_to(real_home, target_is_directory=True)
                side_effect_roots.extend([case_root, real_home])
            elif case_name == "symlink_to_file":
                target_file = case_root / "home_target_file"
                target_file.write_text("not a dir\n", encoding="utf-8")
                home_path = case_root / "home_symlink_file"
                home_path.symlink_to(target_file)
                side_effect_roots.append(case_root)
            elif case_name == "dangling_symlink":
                home_path = case_root / "home_dangling_symlink"
                home_path.symlink_to(case_root / "missing-target", target_is_directory=True)
                side_effect_roots.append(case_root)
            else:
                real_parent = case_root / "real_parent"
                real_parent.mkdir()
                symlink_parent = case_root / "symlink_parent"
                symlink_parent.symlink_to(real_parent, target_is_directory=True)
                home_path = symlink_parent / "orch_home"
                side_effect_roots.extend([case_root, real_parent])

            plan_path = case_root / "plan_fifo.yaml"
            os.mkfifo(plan_path)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = proc.stdout + proc.stderr
            context = f"{case_name}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid home" in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            for root in side_effect_roots:
                assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_non_regular_plan_and_workdir_matrix(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    case_names = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for case_name in case_names:
            case_root = tmp_path / f"fifo_workdir_precedence_{case_name}_{order_label}"
            case_root.mkdir()
            side_effect_roots: list[Path] = []

            if case_name == "home_file":
                home_path = case_root / "home_file"
                home_path.write_text("not a dir\n", encoding="utf-8")
                side_effect_roots.append(case_root)
            elif case_name == "file_ancestor":
                parent_file = case_root / "home_parent_file"
                parent_file.write_text("not a dir\n", encoding="utf-8")
                home_path = parent_file / "orch_home"
                side_effect_roots.append(case_root)
            elif case_name == "symlink_to_dir":
                real_home = case_root / "real_home"
                real_home.mkdir()
                home_path = case_root / "home_symlink_dir"
                home_path.symlink_to(real_home, target_is_directory=True)
                side_effect_roots.extend([case_root, real_home])
            elif case_name == "symlink_to_file":
                target_file = case_root / "home_target_file"
                target_file.write_text("not a dir\n", encoding="utf-8")
                home_path = case_root / "home_symlink_file"
                home_path.symlink_to(target_file)
                side_effect_roots.append(case_root)
            elif case_name == "dangling_symlink":
                home_path = case_root / "home_dangling_symlink"
                home_path.symlink_to(case_root / "missing-target", target_is_directory=True)
                side_effect_roots.append(case_root)
            else:
                real_parent = case_root / "real_parent"
                real_parent.mkdir()
                symlink_parent = case_root / "symlink_parent"
                symlink_parent.symlink_to(real_parent, target_is_directory=True)
                home_path = symlink_parent / "orch_home"
                side_effect_roots.extend([case_root, real_parent])

            plan_path = case_root / "plan_fifo.yaml"
            os.mkfifo(plan_path)
            invalid_workdir_file = case_root / "invalid_workdir_file"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home_path),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = proc.stdout + proc.stderr
            context = f"{case_name}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid home" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            for root in side_effect_roots:
                assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_non_regular_plan_and_workdir(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    plan_path = tmp_path / "plan_fifo_both_toggles_with_workdir.yaml"
    os.mkfifo(plan_path)
    home_file = tmp_path / "home_file_both_toggles_fifo_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_fifo_plan_home"
    home_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_symlink_plan_path(
    tmp_path: Path,
) -> None:
    real_plan = tmp_path / "real_plan_both_toggles_home_vs_symlink_plan.yaml"
    symlink_plan = tmp_path / "symlink_plan_both_toggles_home_vs_symlink_plan.yaml"
    home_file = tmp_path / "home_file_both_toggles_symlink_plan"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    symlink_plan.symlink_to(real_plan)
    home_file.write_text("not a dir\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(symlink_plan),
            "--home",
            str(home_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_symlink_ancestor_plan_path(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_plan_parent_both_toggles_home_vs_plan_ancestor"
    real_parent.mkdir()
    real_plan = real_parent / "plan.yaml"
    symlink_parent = tmp_path / "plan_parent_link_both_toggles_home_vs_plan_ancestor"
    home_file = tmp_path / "home_file_both_toggles_plan_ancestor"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    home_file.write_text("not a dir\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(symlink_parent / "plan.yaml"),
            "--home",
            str(home_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_invalid_home_never_emits_runtime_summary(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_invalid_home_runtime_summary.yaml"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]

    case_names = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for case_name in case_names:
            case_root = tmp_path / f"{case_name}_{order_label}"
            case_root.mkdir()
            side_effect_roots: list[Path] = []

            if case_name == "home_file":
                home_path = case_root / "home_file"
                home_path.write_text("not a dir\n", encoding="utf-8")
                side_effect_roots.append(case_root)
            elif case_name == "file_ancestor":
                parent_file = case_root / "home_parent_file"
                parent_file.write_text("not a dir\n", encoding="utf-8")
                home_path = parent_file / "orch_home"
                side_effect_roots.append(case_root)
            elif case_name == "symlink_to_dir":
                real_home = case_root / "real_home"
                real_home.mkdir()
                home_path = case_root / "home_symlink_dir"
                home_path.symlink_to(real_home, target_is_directory=True)
                side_effect_roots.extend([case_root, real_home])
            elif case_name == "symlink_to_file":
                target_file = case_root / "home_target_file"
                target_file.write_text("not a dir\n", encoding="utf-8")
                home_path = case_root / "home_symlink_file"
                home_path.symlink_to(target_file)
                side_effect_roots.append(case_root)
            elif case_name == "dangling_symlink":
                home_path = case_root / "home_dangling_symlink"
                home_path.symlink_to(case_root / "missing-target", target_is_directory=True)
                side_effect_roots.append(case_root)
            else:
                real_parent = case_root / "real_parent"
                real_parent.mkdir()
                symlink_parent = case_root / "symlink_parent"
                symlink_parent.symlink_to(real_parent, target_is_directory=True)
                home_path = symlink_parent / "orch_home"
                side_effect_roots.extend([case_root, real_parent])

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{case_name}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid home" in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            for root in side_effect_roots:
                assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_plan_and_workdir_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]

    case_names = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for case_name in case_names:
            case_root = tmp_path / f"precedence_{case_name}_{order_label}"
            case_root.mkdir()
            side_effect_roots: list[Path] = []

            if case_name == "home_file":
                home_path = case_root / "home_file"
                home_path.write_text("not a dir\n", encoding="utf-8")
                side_effect_roots.append(case_root)
            elif case_name == "file_ancestor":
                parent_file = case_root / "home_parent_file"
                parent_file.write_text("not a dir\n", encoding="utf-8")
                home_path = parent_file / "orch_home"
                side_effect_roots.append(case_root)
            elif case_name == "symlink_to_dir":
                real_home = case_root / "real_home"
                real_home.mkdir()
                home_path = case_root / "home_symlink_dir"
                home_path.symlink_to(real_home, target_is_directory=True)
                side_effect_roots.extend([case_root, real_home])
            elif case_name == "symlink_to_file":
                target_file = case_root / "home_target_file"
                target_file.write_text("not a dir\n", encoding="utf-8")
                home_path = case_root / "home_symlink_file"
                home_path.symlink_to(target_file)
                side_effect_roots.append(case_root)
            elif case_name == "dangling_symlink":
                home_path = case_root / "home_dangling_symlink"
                home_path.symlink_to(case_root / "missing-target", target_is_directory=True)
                side_effect_roots.append(case_root)
            else:
                real_parent = case_root / "real_parent"
                real_parent.mkdir()
                symlink_parent = case_root / "symlink_parent"
                symlink_parent.symlink_to(real_parent, target_is_directory=True)
                home_path = symlink_parent / "orch_home"
                side_effect_roots.extend([case_root, real_parent])

            invalid_workdir_file = case_root / "invalid_workdir_file"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")
            invalid_plan = case_root / "invalid_plan.yaml"
            invalid_plan.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(invalid_plan),
                    "--home",
                    str(home_path),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{case_name}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid home" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            for root in side_effect_roots:
                assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_invalid_plan_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]

    case_names = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for case_name in case_names:
            case_root = tmp_path / f"plan_precedence_{case_name}_{order_label}"
            case_root.mkdir()
            side_effect_roots: list[Path] = []

            if case_name == "home_file":
                home_path = case_root / "home_file"
                home_path.write_text("not a dir\n", encoding="utf-8")
                side_effect_roots.append(case_root)
            elif case_name == "file_ancestor":
                parent_file = case_root / "home_parent_file"
                parent_file.write_text("not a dir\n", encoding="utf-8")
                home_path = parent_file / "orch_home"
                side_effect_roots.append(case_root)
            elif case_name == "symlink_to_dir":
                real_home = case_root / "real_home"
                real_home.mkdir()
                home_path = case_root / "home_symlink_dir"
                home_path.symlink_to(real_home, target_is_directory=True)
                side_effect_roots.extend([case_root, real_home])
            elif case_name == "symlink_to_file":
                target_file = case_root / "home_target_file"
                target_file.write_text("not a dir\n", encoding="utf-8")
                home_path = case_root / "home_symlink_file"
                home_path.symlink_to(target_file)
                side_effect_roots.append(case_root)
            elif case_name == "dangling_symlink":
                home_path = case_root / "home_dangling_symlink"
                home_path.symlink_to(case_root / "missing-target", target_is_directory=True)
                side_effect_roots.append(case_root)
            else:
                real_parent = case_root / "real_parent"
                real_parent.mkdir()
                symlink_parent = case_root / "symlink_parent"
                symlink_parent.symlink_to(real_parent, target_is_directory=True)
                home_path = symlink_parent / "orch_home"
                side_effect_roots.extend([case_root, real_parent])

            invalid_plan = case_root / "invalid_plan.yaml"
            invalid_plan.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(invalid_plan),
                    "--home",
                    str(home_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{case_name}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid home" in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            for root in side_effect_roots:
                assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_invalid_workdir_matrix(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_invalid_home_workdir_matrix.yaml"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]

    case_names = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for case_name in case_names:
            case_root = tmp_path / f"workdir_precedence_{case_name}_{order_label}"
            case_root.mkdir()
            side_effect_roots: list[Path] = []

            if case_name == "home_file":
                home_path = case_root / "home_file"
                home_path.write_text("not a dir\n", encoding="utf-8")
                side_effect_roots.append(case_root)
            elif case_name == "file_ancestor":
                parent_file = case_root / "home_parent_file"
                parent_file.write_text("not a dir\n", encoding="utf-8")
                home_path = parent_file / "orch_home"
                side_effect_roots.append(case_root)
            elif case_name == "symlink_to_dir":
                real_home = case_root / "real_home"
                real_home.mkdir()
                home_path = case_root / "home_symlink_dir"
                home_path.symlink_to(real_home, target_is_directory=True)
                side_effect_roots.extend([case_root, real_home])
            elif case_name == "symlink_to_file":
                target_file = case_root / "home_target_file"
                target_file.write_text("not a dir\n", encoding="utf-8")
                home_path = case_root / "home_symlink_file"
                home_path.symlink_to(target_file)
                side_effect_roots.append(case_root)
            elif case_name == "dangling_symlink":
                home_path = case_root / "home_dangling_symlink"
                home_path.symlink_to(case_root / "missing-target", target_is_directory=True)
                side_effect_roots.append(case_root)
            else:
                real_parent = case_root / "real_parent"
                real_parent.mkdir()
                symlink_parent = case_root / "symlink_parent"
                symlink_parent.symlink_to(real_parent, target_is_directory=True)
                home_path = symlink_parent / "orch_home"
                side_effect_roots.extend([case_root, real_parent])

            invalid_workdir_file = case_root / "invalid_workdir_file"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home_path),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{case_name}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid home" in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            for root in side_effect_roots:
                assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_symlinked_plan_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("symlink_plan", "symlink_ancestor_plan")
    case_names = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            for case_name in case_names:
                case_root = tmp_path / f"home_vs_plan_{plan_mode}_{case_name}_{order_label}"
                case_root.mkdir()
                side_effect_roots: list[Path] = []

                if case_name == "home_file":
                    home_path = case_root / "home_file"
                    home_path.write_text("not a dir\n", encoding="utf-8")
                    side_effect_roots.append(case_root)
                elif case_name == "file_ancestor":
                    parent_file = case_root / "home_parent_file"
                    parent_file.write_text("not a dir\n", encoding="utf-8")
                    home_path = parent_file / "orch_home"
                    side_effect_roots.append(case_root)
                elif case_name == "symlink_to_dir":
                    real_home = case_root / "real_home"
                    real_home.mkdir()
                    home_path = case_root / "home_symlink_dir"
                    home_path.symlink_to(real_home, target_is_directory=True)
                    side_effect_roots.extend([case_root, real_home])
                elif case_name == "symlink_to_file":
                    target_file = case_root / "home_target_file"
                    target_file.write_text("not a dir\n", encoding="utf-8")
                    home_path = case_root / "home_symlink_file"
                    home_path.symlink_to(target_file)
                    side_effect_roots.append(case_root)
                elif case_name == "dangling_symlink":
                    home_path = case_root / "home_dangling_symlink"
                    home_path.symlink_to(case_root / "missing-target", target_is_directory=True)
                    side_effect_roots.append(case_root)
                else:
                    real_parent = case_root / "real_parent"
                    real_parent.mkdir()
                    symlink_parent = case_root / "symlink_parent"
                    symlink_parent.symlink_to(real_parent, target_is_directory=True)
                    home_path = symlink_parent / "orch_home"
                    side_effect_roots.extend([case_root, real_parent])

                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                if plan_mode == "symlink_plan":
                    plan_path = case_root / "plan_symlink.yaml"
                    plan_path.symlink_to(real_plan)
                else:
                    real_plan_parent = case_root / "real_plan_parent"
                    real_plan_parent.mkdir()
                    plan_file = real_plan_parent / "plan.yaml"
                    _write_plan(
                        plan_file,
                        """
                        tasks:
                          - id: t1
                            cmd: ["python3", "-c", "print('ok')"]
                        """,
                    )
                    plan_parent_link = case_root / "plan_parent_link"
                    plan_parent_link.symlink_to(real_plan_parent, target_is_directory=True)
                    plan_path = plan_parent_link / "plan.yaml"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "run",
                        str(plan_path),
                        "--home",
                        str(home_path),
                        "--dry-run",
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                context = f"{plan_mode}-{case_name}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid home" in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "Dry Run" not in output, context
                assert "run_id:" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                for root in side_effect_roots:
                    assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_invalid_home_precedes_symlinked_plan_and_workdir_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("symlink_plan", "symlink_ancestor_plan")
    case_names = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            for case_name in case_names:
                case_root = tmp_path / f"home_vs_plan_workdir_{plan_mode}_{case_name}_{order_label}"
                case_root.mkdir()
                side_effect_roots: list[Path] = []

                if case_name == "home_file":
                    home_path = case_root / "home_file"
                    home_path.write_text("not a dir\n", encoding="utf-8")
                    side_effect_roots.append(case_root)
                elif case_name == "file_ancestor":
                    parent_file = case_root / "home_parent_file"
                    parent_file.write_text("not a dir\n", encoding="utf-8")
                    home_path = parent_file / "orch_home"
                    side_effect_roots.append(case_root)
                elif case_name == "symlink_to_dir":
                    real_home = case_root / "real_home"
                    real_home.mkdir()
                    home_path = case_root / "home_symlink_dir"
                    home_path.symlink_to(real_home, target_is_directory=True)
                    side_effect_roots.extend([case_root, real_home])
                elif case_name == "symlink_to_file":
                    target_file = case_root / "home_target_file"
                    target_file.write_text("not a dir\n", encoding="utf-8")
                    home_path = case_root / "home_symlink_file"
                    home_path.symlink_to(target_file)
                    side_effect_roots.append(case_root)
                elif case_name == "dangling_symlink":
                    home_path = case_root / "home_dangling_symlink"
                    home_path.symlink_to(case_root / "missing-target", target_is_directory=True)
                    side_effect_roots.append(case_root)
                else:
                    real_parent = case_root / "real_parent"
                    real_parent.mkdir()
                    symlink_parent = case_root / "symlink_parent"
                    symlink_parent.symlink_to(real_parent, target_is_directory=True)
                    home_path = symlink_parent / "orch_home"
                    side_effect_roots.extend([case_root, real_parent])

                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                if plan_mode == "symlink_plan":
                    plan_path = case_root / "plan_symlink.yaml"
                    plan_path.symlink_to(real_plan)
                else:
                    real_plan_parent = case_root / "real_plan_parent"
                    real_plan_parent.mkdir()
                    plan_file = real_plan_parent / "plan.yaml"
                    _write_plan(
                        plan_file,
                        """
                        tasks:
                          - id: t1
                            cmd: ["python3", "-c", "print('ok')"]
                        """,
                    )
                    plan_parent_link = case_root / "plan_parent_link"
                    plan_parent_link.symlink_to(real_plan_parent, target_is_directory=True)
                    plan_path = plan_parent_link / "plan.yaml"

                invalid_workdir_file = case_root / "invalid_workdir_file"
                invalid_workdir_file.write_text("file\n", encoding="utf-8")

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "run",
                        str(plan_path),
                        "--home",
                        str(home_path),
                        "--workdir",
                        str(invalid_workdir_file),
                        "--dry-run",
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                context = f"{plan_mode}-{case_name}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid home" in output, context
                assert "Plan validation error" not in output, context
                assert "Invalid workdir" not in output, context
                assert "contains symlink component" not in output, context
                assert "Dry Run" not in output, context
                assert "run_id:" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                for root in side_effect_roots:
                    assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_symlink_home_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_home_vs_plan.yaml"
    real_home = tmp_path / "real_home_both_toggles"
    home_symlink = tmp_path / "home_symlink_both_toggles"
    real_home.mkdir()
    home_symlink.symlink_to(real_home, target_is_directory=True)
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_dry_run_both_toggles_reject_symlink_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_home_only.yaml"
    real_home = tmp_path / "real_home_both_toggles_only"
    home_symlink = tmp_path / "home_symlink_both_toggles_only"
    real_home.mkdir()
    home_symlink.symlink_to(real_home, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_dry_run_both_toggles_symlink_to_file_home_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_file_home_vs_plan.yaml"
    home_target_file = tmp_path / "home_target_file_both_toggles_plan"
    home_symlink = tmp_path / "home_symlink_to_file_both_toggles_plan"
    home_target_file.write_text("home-target-file\n", encoding="utf-8")
    home_symlink.symlink_to(home_target_file)
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reject_symlink_to_file_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_file_home_only.yaml"
    home_target_file = tmp_path / "home_target_file_both_toggles_only"
    home_symlink = tmp_path / "home_symlink_to_file_both_toggles_only"
    home_target_file.write_text("home-target-file\n", encoding="utf-8")
    home_symlink.symlink_to(home_target_file)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_symlink_to_file_home_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_file_home_vs_workdir.yaml"
    home_target_file = tmp_path / "home_target_file_both_toggles_workdir"
    home_symlink = tmp_path / "home_symlink_to_file_both_toggles_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_symlink_file_home"
    home_target_file.write_text("home-target-file\n", encoding="utf-8")
    home_symlink.symlink_to(home_target_file)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_symlink_to_file_home_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_file_home_vs_plan_workdir.yaml"
    home_target_file = tmp_path / "home_target_file_both_toggles_plan_workdir"
    home_symlink = tmp_path / "home_symlink_to_file_both_toggles_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_symlink_file_plan"
    home_target_file.write_text("home-target-file\n", encoding="utf-8")
    home_symlink.symlink_to(home_target_file)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reject_dangling_symlink_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_dangling_symlink_home_only.yaml"
    dangling_home = tmp_path / "dangling_home_both_toggles_only"
    dangling_home.symlink_to(tmp_path / "missing-home-target-only", target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(dangling_home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reject_symlink_ancestor_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_ancestor_home_only.yaml"
    real_parent = tmp_path / "real_parent_both_toggles_only"
    symlink_parent = tmp_path / "home_parent_link_both_toggles_only"
    nested_home = symlink_parent / "orch_home"
    real_parent.mkdir()
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reject_file_ancestor_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_file_ancestor_home_only.yaml"
    home_parent_file = tmp_path / "home_parent_file_both_toggles_only"
    nested_home = home_parent_file / "orch_home"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_dangling_symlink_home_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_dangling_symlink_home_vs_plan.yaml"
    dangling_home = tmp_path / "dangling_home_both_toggles_plan"
    dangling_home.symlink_to(tmp_path / "missing-home-target-plan", target_is_directory=True)
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(dangling_home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_dangling_symlink_home_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_dangling_symlink_home_vs_workdir.yaml"
    dangling_home = tmp_path / "dangling_home_both_toggles_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_dangling_home"
    dangling_home.symlink_to(tmp_path / "missing-home-target-workdir", target_is_directory=True)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(dangling_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_dangling_symlink_home_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_dangling_symlink_home_vs_plan_workdir.yaml"
    dangling_home = tmp_path / "dangling_home_both_toggles_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_dangling_plan"
    dangling_home.symlink_to(
        tmp_path / "missing-home-target-plan-workdir", target_is_directory=True
    )
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(dangling_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_symlink_home_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_home_vs_workdir.yaml"
    real_home = tmp_path / "real_home_both_toggles_workdir"
    home_symlink = tmp_path / "home_symlink_both_toggles_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_symlink_home"
    real_home.mkdir()
    home_symlink.symlink_to(real_home, target_is_directory=True)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_dry_run_both_toggles_symlink_home_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_home_vs_plan_workdir.yaml"
    real_home = tmp_path / "real_home_both_toggles_plan_workdir"
    home_symlink = tmp_path / "home_symlink_both_toggles_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_symlink_plan"
    real_home.mkdir()
    home_symlink.symlink_to(real_home, target_is_directory=True)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_dry_run_both_toggles_symlink_ancestor_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_ancestor_vs_plan_workdir.yaml"
    real_parent = tmp_path / "real_parent_both_toggles"
    symlink_parent = tmp_path / "home_parent_link_both_toggles"
    nested_home = symlink_parent / "orch_home"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_symlink_ancestor"
    real_parent.mkdir()
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_symlink_ancestor_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_ancestor_vs_plan.yaml"
    real_parent = tmp_path / "real_parent_both_toggles_plan"
    symlink_parent = tmp_path / "home_parent_link_both_toggles_plan"
    nested_home = symlink_parent / "orch_home"
    real_parent.mkdir()
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_symlink_ancestor_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_symlink_ancestor_vs_workdir.yaml"
    real_parent = tmp_path / "real_parent_both_toggles_workdir"
    symlink_parent = tmp_path / "home_parent_link_both_toggles_workdir"
    nested_home = symlink_parent / "orch_home"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_symlink_ancestor_only"
    real_parent.mkdir()
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_home_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_home_vs_plan_workdir.yaml"
    home_file = tmp_path / "home_file_both_toggles_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_plan_home"
    home_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reject_home_file(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_home_file_only.yaml"
    home_file = tmp_path / "home_file_both_toggles_only"
    home_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_file_ancestor_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_file_ancestor_vs_plan_workdir.yaml"
    home_parent_file = tmp_path / "home_parent_file_both_toggles"
    nested_home = home_parent_file / "orch_home"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_file_ancestor"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_file_ancestor_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_file_ancestor_vs_plan.yaml"
    home_parent_file = tmp_path / "home_parent_file_both_toggles_plan"
    nested_home = home_parent_file / "orch_home"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_file_ancestor_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_file_ancestor_vs_workdir.yaml"
    home_parent_file = tmp_path / "home_parent_file_both_toggles_workdir"
    nested_home = home_parent_file / "orch_home"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_file_ancestor_only"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_fail_fast_toggles_invalid_plan_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_plan_vs_workdir.yaml"
    home = tmp_path / ".orch_cli"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_plan"
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--fail-fast",
            "--no-fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "PLAN_PATH" not in output
    assert "Invalid home" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_toggles_invalid_plan_precedes_invalid_workdir_matrix(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = (
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"plan_vs_workdir_{plan_mode}_{order_label}"
            case_root.mkdir()
            home = case_root / ".orch_cli"
            invalid_workdir_file = case_root / "invalid_workdir"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            if plan_mode == "invalid_yaml":
                plan_path = case_root / "invalid_plan.yaml"
                plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")
            elif plan_mode == "unknown_root_field":
                plan_path = case_root / "unknown_root_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    unexpected_root: true
                    """,
                )
            elif plan_mode == "unknown_task_field":
                plan_path = case_root / "unknown_task_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                        unexpected_task_field: 1
                    """,
                )
            elif plan_mode == "non_regular_fifo":
                plan_path = case_root / "plan_fifo.yaml"
                os.mkfifo(plan_path)
            elif plan_mode == "symlink_plan":
                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_path = case_root / "plan_symlink.yaml"
                plan_path.symlink_to(real_plan)
            else:
                real_parent = case_root / "real_plan_parent"
                real_parent.mkdir()
                plan_file = real_parent / "plan.yaml"
                _write_plan(
                    plan_file,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_parent_link = case_root / "plan_parent_link"
                plan_parent_link.symlink_to(real_parent, target_is_directory=True)
                plan_path = plan_parent_link / "plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Plan validation error" in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "must not include symlink" not in output, context
            assert "must not be symlink" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not home.exists(), context
            assert not (home / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = (
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"invalid_plan_existing_home_workdir_{plan_mode}_{order_label}"
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()
            invalid_workdir_file = case_root / "invalid_workdir"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            if plan_mode == "invalid_yaml":
                plan_path = case_root / "invalid_plan.yaml"
                plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")
            elif plan_mode == "unknown_root_field":
                plan_path = case_root / "unknown_root_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    unexpected_root: true
                    """,
                )
            elif plan_mode == "unknown_task_field":
                plan_path = case_root / "unknown_task_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                        unexpected_task_field: 1
                    """,
                )
            elif plan_mode == "non_regular_fifo":
                plan_path = case_root / "plan_fifo.yaml"
                os.mkfifo(plan_path)
            elif plan_mode == "symlink_plan":
                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_path = case_root / "plan_symlink.yaml"
                plan_path.symlink_to(real_plan)
            else:
                real_parent = case_root / "real_plan_parent"
                real_parent.mkdir()
                plan_file = real_parent / "plan.yaml"
                _write_plan(
                    plan_file,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_parent_link = case_root / "plan_parent_link"
                plan_parent_link.symlink_to(real_parent, target_is_directory=True)
                plan_path = plan_parent_link / "plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Plan validation error" in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "must not include symlink" not in output, context
            assert "must not be symlink" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert not (home / "runs").exists(), context
            assert sorted(path.name for path in home.iterdir()) == [], context


def test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = (
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"invalid_plan_existing_home_only_{plan_mode}_{order_label}"
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()

            if plan_mode == "invalid_yaml":
                plan_path = case_root / "invalid_plan.yaml"
                plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")
            elif plan_mode == "unknown_root_field":
                plan_path = case_root / "unknown_root_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    unexpected_root: true
                    """,
                )
            elif plan_mode == "unknown_task_field":
                plan_path = case_root / "unknown_task_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                        unexpected_task_field: 1
                    """,
                )
            elif plan_mode == "non_regular_fifo":
                plan_path = case_root / "plan_fifo.yaml"
                os.mkfifo(plan_path)
            elif plan_mode == "symlink_plan":
                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_path = case_root / "plan_symlink.yaml"
                plan_path.symlink_to(real_plan)
            else:
                real_parent = case_root / "real_plan_parent"
                real_parent.mkdir()
                plan_file = real_parent / "plan.yaml"
                _write_plan(
                    plan_file,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_parent_link = case_root / "plan_parent_link"
                plan_parent_link.symlink_to(real_parent, target_is_directory=True)
                plan_path = plan_parent_link / "plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Plan validation error" in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "must not include symlink" not in output, context
            assert "must not be symlink" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert not (home / "runs").exists(), context
            assert sorted(path.name for path in home.iterdir()) == [], context


def test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_home_matrix(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = (
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"invalid_plan_default_home_workdir_{plan_mode}_{order_label}"
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            invalid_workdir_file = case_root / "invalid_workdir"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            if plan_mode == "invalid_yaml":
                plan_path = case_root / "invalid_plan.yaml"
                plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")
            elif plan_mode == "unknown_root_field":
                plan_path = case_root / "unknown_root_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    unexpected_root: true
                    """,
                )
            elif plan_mode == "unknown_task_field":
                plan_path = case_root / "unknown_task_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                        unexpected_task_field: 1
                    """,
                )
            elif plan_mode == "non_regular_fifo":
                plan_path = case_root / "plan_fifo.yaml"
                os.mkfifo(plan_path)
            elif plan_mode == "symlink_plan":
                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_path = case_root / "plan_symlink.yaml"
                plan_path.symlink_to(real_plan)
            else:
                real_parent = case_root / "real_plan_parent"
                real_parent.mkdir()
                plan_file = real_parent / "plan.yaml"
                _write_plan(
                    plan_file,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_parent_link = case_root / "plan_parent_link"
                plan_parent_link.symlink_to(real_parent, target_is_directory=True)
                plan_path = plan_parent_link / "plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Plan validation error" in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "must not include symlink" not in output, context
            assert "must not be symlink" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not default_home.exists(), context
            assert not (default_home / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_existing_home_matrix(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = (
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"invalid_plan_default_existing_home_{plan_mode}_{order_label}"
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            default_home.mkdir()
            invalid_workdir_file = case_root / "invalid_workdir"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            if plan_mode == "invalid_yaml":
                plan_path = case_root / "invalid_plan.yaml"
                plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")
            elif plan_mode == "unknown_root_field":
                plan_path = case_root / "unknown_root_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    unexpected_root: true
                    """,
                )
            elif plan_mode == "unknown_task_field":
                plan_path = case_root / "unknown_task_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                        unexpected_task_field: 1
                    """,
                )
            elif plan_mode == "non_regular_fifo":
                plan_path = case_root / "plan_fifo.yaml"
                os.mkfifo(plan_path)
            elif plan_mode == "symlink_plan":
                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_path = case_root / "plan_symlink.yaml"
                plan_path.symlink_to(real_plan)
            else:
                real_parent = case_root / "real_plan_parent"
                real_parent.mkdir()
                plan_file = real_parent / "plan.yaml"
                _write_plan(
                    plan_file,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_parent_link = case_root / "plan_parent_link"
                plan_parent_link.symlink_to(real_parent, target_is_directory=True)
                plan_path = plan_parent_link / "plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Plan validation error" in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "must not include symlink" not in output, context
            assert "must not be symlink" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert not (default_home / "runs").exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [], context


def test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_home_matrix(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = (
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"invalid_plan_default_home_only_{plan_mode}_{order_label}"
            case_root.mkdir()
            default_home = case_root / ".orch_cli"

            if plan_mode == "invalid_yaml":
                plan_path = case_root / "invalid_plan.yaml"
                plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")
            elif plan_mode == "unknown_root_field":
                plan_path = case_root / "unknown_root_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    unexpected_root: true
                    """,
                )
            elif plan_mode == "unknown_task_field":
                plan_path = case_root / "unknown_task_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                        unexpected_task_field: 1
                    """,
                )
            elif plan_mode == "non_regular_fifo":
                plan_path = case_root / "plan_fifo.yaml"
                os.mkfifo(plan_path)
            elif plan_mode == "symlink_plan":
                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_path = case_root / "plan_symlink.yaml"
                plan_path.symlink_to(real_plan)
            else:
                real_parent = case_root / "real_plan_parent"
                real_parent.mkdir()
                plan_file = real_parent / "plan.yaml"
                _write_plan(
                    plan_file,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_parent_link = case_root / "plan_parent_link"
                plan_parent_link.symlink_to(real_parent, target_is_directory=True)
                plan_path = plan_parent_link / "plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Plan validation error" in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "must not include symlink" not in output, context
            assert "must not be symlink" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not default_home.exists(), context
            assert not (default_home / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = (
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = (
                tmp_path / f"invalid_plan_default_existing_home_only_{plan_mode}_{order_label}"
            )
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            default_home.mkdir()

            if plan_mode == "invalid_yaml":
                plan_path = case_root / "invalid_plan.yaml"
                plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")
            elif plan_mode == "unknown_root_field":
                plan_path = case_root / "unknown_root_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    unexpected_root: true
                    """,
                )
            elif plan_mode == "unknown_task_field":
                plan_path = case_root / "unknown_task_field_plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                        unexpected_task_field: 1
                    """,
                )
            elif plan_mode == "non_regular_fifo":
                plan_path = case_root / "plan_fifo.yaml"
                os.mkfifo(plan_path)
            elif plan_mode == "symlink_plan":
                real_plan = case_root / "real_plan.yaml"
                _write_plan(
                    real_plan,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_path = case_root / "plan_symlink.yaml"
                plan_path.symlink_to(real_plan)
            else:
                real_parent = case_root / "real_plan_parent"
                real_parent.mkdir()
                plan_file = real_parent / "plan.yaml"
                _write_plan(
                    plan_file,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                plan_parent_link = case_root / "plan_parent_link"
                plan_parent_link.symlink_to(real_parent, target_is_directory=True)
                plan_path = plan_parent_link / "plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Plan validation error" in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "must not include symlink" not in output, context
            assert "must not be symlink" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert not (default_home / "runs").exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [], context


def test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_invalid_workdir_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"missing_plan_vs_workdir_{plan_mode}_{order_label}"
            case_root.mkdir()
            home = case_root / ".orch_cli"
            invalid_workdir_file = case_root / "invalid_workdir"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            if plan_mode == "missing_path":
                plan_path = case_root / "missing_plan.yaml"
            elif plan_mode == "dangling_symlink_path":
                plan_path = case_root / "dangling_plan_link.yaml"
                plan_path.symlink_to(case_root / "missing_plan_target.yaml")
            else:
                real_missing_parent = case_root / "real_missing_parent"
                real_missing_parent.mkdir()
                missing_parent_link = case_root / "missing_parent_link"
                missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                plan_path = missing_parent_link / "missing_plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "PLAN_PATH" in output, context
            assert "Invalid value for 'PLAN_PATH'" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not home.exists(), context
            assert not (home / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_missing_plan_precedes_workdir_existing_home_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = (
                tmp_path / f"missing_plan_vs_workdir_existing_home_{plan_mode}_{order_label}"
            )
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()
            invalid_workdir_file = case_root / "invalid_workdir"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            if plan_mode == "missing_path":
                plan_path = case_root / "missing_plan.yaml"
            elif plan_mode == "dangling_symlink_path":
                plan_path = case_root / "dangling_plan_link.yaml"
                plan_path.symlink_to(case_root / "missing_plan_target.yaml")
            else:
                real_missing_parent = case_root / "real_missing_parent"
                real_missing_parent.mkdir()
                missing_parent_link = case_root / "missing_parent_link"
                missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                plan_path = missing_parent_link / "missing_plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "PLAN_PATH" in output, context
            assert "Invalid value for 'PLAN_PATH'" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert not (home / "runs").exists(), context
            assert sorted(path.name for path in home.iterdir()) == [], context


def test_cli_run_dry_run_both_toggles_reject_missing_plan_path_existing_home_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"missing_plan_only_existing_home_{plan_mode}_{order_label}"
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()

            if plan_mode == "missing_path":
                plan_path = case_root / "missing_plan.yaml"
            elif plan_mode == "dangling_symlink_path":
                plan_path = case_root / "dangling_plan_link.yaml"
                plan_path.symlink_to(case_root / "missing_plan_target.yaml")
            else:
                real_missing_parent = case_root / "real_missing_parent"
                real_missing_parent.mkdir()
                missing_parent_link = case_root / "missing_parent_link"
                missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                plan_path = missing_parent_link / "missing_plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "PLAN_PATH" in output, context
            assert "Invalid value for 'PLAN_PATH'" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert not (home / "runs").exists(), context
            assert sorted(path.name for path in home.iterdir()) == [], context


def test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_plan_error_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    case_modes = (
        ("invalid_only", "invalid_plan", False),
        ("invalid_with_workdir", "invalid_plan", True),
        ("missing_only", "missing_plan", False),
        ("missing_with_workdir", "missing_plan", True),
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for case_name, plan_kind, needs_workdir in case_modes:
            case_root = tmp_path / f"preserve_home_entries_{case_name}_{order_label}"
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")

            if plan_kind == "invalid_plan":
                plan_path = case_root / "invalid_plan.yaml"
                plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")
            else:
                plan_path = case_root / "missing_plan.yaml"

            command = [
                sys.executable,
                "-m",
                "orch.cli",
                "run",
                str(plan_path),
                "--home",
                str(home),
                "--dry-run",
                *order,
            ]
            if needs_workdir:
                invalid_workdir_file = case_root / "invalid_workdir"
                invalid_workdir_file.write_text("file\n", encoding="utf-8")
                command.extend(["--workdir", str(invalid_workdir_file)])

            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = proc.stdout + proc.stderr
            context = f"{case_name}-{order_label}"
            assert proc.returncode == 2, context

            if plan_kind == "invalid_plan":
                assert "Plan validation error" in output, context
                assert "PLAN_PATH" not in output, context
            else:
                assert "PLAN_PATH" in output, context
                assert "Invalid value for 'PLAN_PATH'" in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "must not include symlink" not in output, context
                assert "must not be symlink" not in output, context

            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert not (home / "runs").exists(), context
            assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir"], context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_run_dry_run_both_toggles_reject_missing_plan_path_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"missing_plan_only_{plan_mode}_{order_label}"
            case_root.mkdir()
            home = case_root / ".orch_cli"

            if plan_mode == "missing_path":
                plan_path = case_root / "missing_plan.yaml"
            elif plan_mode == "dangling_symlink_path":
                plan_path = case_root / "dangling_plan_link.yaml"
                plan_path.symlink_to(case_root / "missing_plan_target.yaml")
            else:
                real_missing_parent = case_root / "real_missing_parent"
                real_missing_parent.mkdir()
                missing_parent_link = case_root / "missing_parent_link"
                missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                plan_path = missing_parent_link / "missing_plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "PLAN_PATH" in output, context
            assert "Invalid value for 'PLAN_PATH'" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not home.exists(), context
            assert not (home / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_reject_missing_plan_path_default_home_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"missing_plan_default_home_{plan_mode}_{order_label}"
            case_root.mkdir()
            default_home = case_root / ".orch_cli"

            if plan_mode == "missing_path":
                plan_path = case_root / "missing_plan.yaml"
            elif plan_mode == "dangling_symlink_path":
                plan_path = case_root / "dangling_plan_link.yaml"
                plan_path.symlink_to(case_root / "missing_plan_target.yaml")
            else:
                real_missing_parent = case_root / "real_missing_parent"
                real_missing_parent.mkdir()
                missing_parent_link = case_root / "missing_parent_link"
                missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                plan_path = missing_parent_link / "missing_plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "PLAN_PATH" in output, context
            assert "Invalid value for 'PLAN_PATH'" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not default_home.exists(), context
            assert not (default_home / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_reject_missing_plan_default_existing_home_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"missing_plan_default_existing_home_{plan_mode}_{order_label}"
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            default_home.mkdir()

            if plan_mode == "missing_path":
                plan_path = case_root / "missing_plan.yaml"
            elif plan_mode == "dangling_symlink_path":
                plan_path = case_root / "dangling_plan_link.yaml"
                plan_path.symlink_to(case_root / "missing_plan_target.yaml")
            else:
                real_missing_parent = case_root / "real_missing_parent"
                real_missing_parent.mkdir()
                missing_parent_link = case_root / "missing_parent_link"
                missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                plan_path = missing_parent_link / "missing_plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "PLAN_PATH" in output, context
            assert "Invalid value for 'PLAN_PATH'" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert not (default_home / "runs").exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [], context


def test_cli_run_dry_run_both_toggles_missing_plan_default_home_precedes_workdir_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = tmp_path / f"missing_plan_default_home_workdir_{plan_mode}_{order_label}"
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            invalid_workdir_file = case_root / "invalid_workdir"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            if plan_mode == "missing_path":
                plan_path = case_root / "missing_plan.yaml"
            elif plan_mode == "dangling_symlink_path":
                plan_path = case_root / "dangling_plan_link.yaml"
                plan_path.symlink_to(case_root / "missing_plan_target.yaml")
            else:
                real_missing_parent = case_root / "real_missing_parent"
                real_missing_parent.mkdir()
                missing_parent_link = case_root / "missing_parent_link"
                missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                plan_path = missing_parent_link / "missing_plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "PLAN_PATH" in output, context
            assert "Invalid value for 'PLAN_PATH'" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not default_home.exists(), context
            assert not (default_home / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_missing_plan_default_existing_home_precedes_workdir_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            case_root = (
                tmp_path / f"missing_plan_default_existing_home_workdir_{plan_mode}_{order_label}"
            )
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            default_home.mkdir()
            invalid_workdir_file = case_root / "invalid_workdir"
            invalid_workdir_file.write_text("file\n", encoding="utf-8")

            if plan_mode == "missing_path":
                plan_path = case_root / "missing_plan.yaml"
            elif plan_mode == "dangling_symlink_path":
                plan_path = case_root / "dangling_plan_link.yaml"
                plan_path.symlink_to(case_root / "missing_plan_target.yaml")
            else:
                real_missing_parent = case_root / "real_missing_parent"
                real_missing_parent.mkdir()
                missing_parent_link = case_root / "missing_parent_link"
                missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                plan_path = missing_parent_link / "missing_plan.yaml"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir_file),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{plan_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "PLAN_PATH" in output, context
            assert "Invalid value for 'PLAN_PATH'" in output, context
            assert "Plan validation error" not in output, context
            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert not (default_home / "runs").exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [], context


def test_cli_run_dry_run_both_toggles_default_existing_home_preserves_entries_plan_error_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    case_modes = (
        ("invalid_only", "invalid_plan", False),
        ("invalid_with_workdir", "invalid_plan", True),
        ("missing_only", "missing_plan", False),
        ("missing_with_workdir", "missing_plan", True),
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for case_name, plan_kind, needs_workdir in case_modes:
            case_root = tmp_path / f"preserve_default_home_entries_{case_name}_{order_label}"
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")

            if plan_kind == "invalid_plan":
                plan_path = case_root / "invalid_plan.yaml"
                plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")
            else:
                plan_path = case_root / "missing_plan.yaml"

            command = [
                sys.executable,
                "-m",
                "orch.cli",
                "run",
                str(plan_path),
                "--dry-run",
                *order,
            ]
            if needs_workdir:
                invalid_workdir_file = case_root / "invalid_workdir"
                invalid_workdir_file.write_text("file\n", encoding="utf-8")
                command.extend(["--workdir", str(invalid_workdir_file)])

            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{case_name}-{order_label}"
            assert proc.returncode == 2, context

            if plan_kind == "invalid_plan":
                assert "Plan validation error" in output, context
                assert "PLAN_PATH" not in output, context
            else:
                assert "PLAN_PATH" in output, context
                assert "Invalid value for 'PLAN_PATH'" in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "must not include symlink" not in output, context
                assert "must not be symlink" not in output, context

            assert "Invalid home" not in output, context
            assert "Invalid workdir" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert not (default_home / "runs").exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
            ], context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_invalid_workdir(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path / f"preserve_home_entries_invalid_workdir_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")

            plan_path = case_root / "valid_plan.yaml"
            plan_path.write_text(
                'tasks:\n  - id: t1\n    cmd: ["python3", "-c", "print(\'ok\')"]\n',
                encoding="utf-8",
            )

            side_effect_files: list[Path] = []
            if workdir_mode == "file_path":
                invalid_workdir_path = case_root / "invalid_workdir_file"
                invalid_workdir_path.write_text("file\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir_path)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = case_root / "workdir_symlink_to_file"
                invalid_workdir_path.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir_path = case_root / "workdir_dangling_symlink"
                invalid_workdir_path.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir_path = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = proc.stdout + proc.stderr
            context = f"invalid_workdir-{workdir_mode}-{order_label}"
            assert proc.returncode == 0, context
            assert "Dry Run" in output, context
            assert "Invalid workdir" not in output, context
            assert "Plan validation error" not in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert not (home / "runs").exists(), context
            assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir"], context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context
            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "file\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_dry_run_both_toggles_default_existing_home_preserves_entries_invalid_workdir(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path
                / f"preserve_default_home_entries_invalid_workdir_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")

            plan_path = case_root / "valid_plan.yaml"
            plan_path.write_text(
                'tasks:\n  - id: t1\n    cmd: ["python3", "-c", "print(\'ok\')"]\n',
                encoding="utf-8",
            )

            side_effect_files: list[Path] = []
            if workdir_mode == "file_path":
                invalid_workdir_path = case_root / "invalid_workdir_file"
                invalid_workdir_path.write_text("file\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir_path)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = case_root / "workdir_symlink_to_file"
                invalid_workdir_path.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir_path = case_root / "workdir_dangling_symlink"
                invalid_workdir_path.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir_path = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"invalid_workdir-{workdir_mode}-{order_label}"
            assert proc.returncode == 0, context
            assert "Dry Run" in output, context
            assert "Invalid workdir" not in output, context
            assert "Plan validation error" not in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert not (default_home / "runs").exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
            ], context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context
            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "file\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_dry_run_both_toggles_existing_home_with_runs_preserves_entries_invalid_workdir(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path
                / f"preserve_home_with_runs_entries_invalid_workdir_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")

            plan_path = case_root / "valid_plan.yaml"
            plan_path.write_text(
                'tasks:\n  - id: t1\n    cmd: ["python3", "-c", "print(\'ok\')"]\n',
                encoding="utf-8",
            )

            side_effect_files: list[Path] = []
            if workdir_mode == "file_path":
                invalid_workdir_path = case_root / "invalid_workdir_file"
                invalid_workdir_path.write_text("file\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir_path)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = case_root / "workdir_symlink_to_file"
                invalid_workdir_path.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir_path = case_root / "workdir_dangling_symlink"
                invalid_workdir_path.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir_path = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = proc.stdout + proc.stderr
            context = f"invalid_workdir-{workdir_mode}-{order_label}"
            assert proc.returncode == 0, context
            assert "Dry Run" in output, context
            assert "Invalid workdir" not in output, context
            assert "Plan validation error" not in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert sorted(path.name for path in home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], context
            assert sorted(path.name for path in existing_run.iterdir()) == ["plan.yaml"], context
            assert not (existing_run / ".lock").exists(), context
            assert not (existing_run / "cancel.request").exists(), context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context
            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "file\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_dry_run_toggles_default_home_with_runs_invalid_workdir_preserve(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path / "preserve_default_home_with_runs_entries_invalid_workdir_"
                f"{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = default_home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")

            plan_path = case_root / "valid_plan.yaml"
            plan_path.write_text(
                'tasks:\n  - id: t1\n    cmd: ["python3", "-c", "print(\'ok\')"]\n',
                encoding="utf-8",
            )

            side_effect_files: list[Path] = []
            if workdir_mode == "file_path":
                invalid_workdir_path = case_root / "invalid_workdir_file"
                invalid_workdir_path.write_text("file\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir_path)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = case_root / "workdir_symlink_to_file"
                invalid_workdir_path.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir_path = case_root / "workdir_dangling_symlink"
                invalid_workdir_path.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir_path = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"invalid_workdir-{workdir_mode}-{order_label}"
            assert proc.returncode == 0, context
            assert "Dry Run" in output, context
            assert "Invalid workdir" not in output, context
            assert "Plan validation error" not in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (default_home / "runs").iterdir()) == [
                "keep_run"
            ], context
            assert sorted(path.name for path in existing_run.iterdir()) == ["plan.yaml"], context
            assert not (existing_run / ".lock").exists(), context
            assert not (existing_run / "cancel.request").exists(), context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context
            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "file\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_dry_run_both_toggles_existing_home_run_artifacts_preserved_invalid_workdir(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path / "preserve_home_run_artifacts_invalid_workdir_"
                f"{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")
            lock_file = existing_run / ".lock"
            lock_file.write_text("lock\n", encoding="utf-8")
            cancel_request = existing_run / "cancel.request"
            cancel_request.write_text("cancel\n", encoding="utf-8")
            run_log = existing_run / "task.log"
            run_log.write_text("log\n", encoding="utf-8")

            plan_path = case_root / "valid_plan.yaml"
            plan_path.write_text(
                'tasks:\n  - id: t1\n    cmd: ["python3", "-c", "print(\'ok\')"]\n',
                encoding="utf-8",
            )

            side_effect_files: list[Path] = []
            if workdir_mode == "file_path":
                invalid_workdir_path = case_root / "invalid_workdir_file"
                invalid_workdir_path.write_text("file\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir_path)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = case_root / "workdir_symlink_to_file"
                invalid_workdir_path.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir_path = case_root / "workdir_dangling_symlink"
                invalid_workdir_path.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir_path = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = proc.stdout + proc.stderr
            context = f"invalid_workdir-{workdir_mode}-{order_label}"
            assert proc.returncode == 0, context
            assert "Dry Run" in output, context
            assert "Invalid workdir" not in output, context
            assert "Plan validation error" not in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert sorted(path.name for path in home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], context
            assert sorted(path.name for path in existing_run.iterdir()) == [
                ".lock",
                "cancel.request",
                "plan.yaml",
                "task.log",
            ], context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert lock_file.read_text(encoding="utf-8") == "lock\n", context
            assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
            assert run_log.read_text(encoding="utf-8") == "log\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context
            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "file\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_dry_run_both_toggles_default_existing_home_run_artifacts_preserved_invalid_workdir(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path / "preserve_default_home_run_artifacts_invalid_workdir_"
                f"{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            default_home = case_root / ".orch_cli"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = default_home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")
            lock_file = existing_run / ".lock"
            lock_file.write_text("lock\n", encoding="utf-8")
            cancel_request = existing_run / "cancel.request"
            cancel_request.write_text("cancel\n", encoding="utf-8")
            run_log = existing_run / "task.log"
            run_log.write_text("log\n", encoding="utf-8")

            plan_path = case_root / "valid_plan.yaml"
            plan_path.write_text(
                'tasks:\n  - id: t1\n    cmd: ["python3", "-c", "print(\'ok\')"]\n',
                encoding="utf-8",
            )

            side_effect_files: list[Path] = []
            if workdir_mode == "file_path":
                invalid_workdir_path = case_root / "invalid_workdir_file"
                invalid_workdir_path.write_text("file\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir_path)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("file\n", encoding="utf-8")
                invalid_workdir_path = case_root / "workdir_symlink_to_file"
                invalid_workdir_path.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir_path = case_root / "workdir_dangling_symlink"
                invalid_workdir_path.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir_path = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir_path),
                    "--dry-run",
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"invalid_workdir-{workdir_mode}-{order_label}"
            assert proc.returncode == 0, context
            assert "Dry Run" in output, context
            assert "Invalid workdir" not in output, context
            assert "Plan validation error" not in output, context
            assert "PLAN_PATH" not in output, context
            assert "Invalid home" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (default_home / "runs").iterdir()) == [
                "keep_run"
            ], context
            assert sorted(path.name for path in existing_run.iterdir()) == [
                ".lock",
                "cancel.request",
                "plan.yaml",
                "task.log",
            ], context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert lock_file.read_text(encoding="utf-8") == "lock\n", context
            assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
            assert run_log.read_text(encoding="utf-8") == "log\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context
            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "file\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_invalid_home_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")
    home_modes = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            for home_mode in home_modes:
                case_root = tmp_path / f"missing_plan_vs_home_{plan_mode}_{home_mode}_{order_label}"
                case_root.mkdir()
                side_effect_roots: list[Path] = []

                if home_mode == "home_file":
                    home_path = case_root / "home_file"
                    home_path.write_text("not a dir\n", encoding="utf-8")
                    side_effect_roots.append(case_root)
                elif home_mode == "file_ancestor":
                    parent_file = case_root / "home_parent_file"
                    parent_file.write_text("not a dir\n", encoding="utf-8")
                    home_path = parent_file / "orch_home"
                    side_effect_roots.append(case_root)
                elif home_mode == "symlink_to_dir":
                    real_home = case_root / "real_home"
                    real_home.mkdir()
                    home_path = case_root / "home_symlink_dir"
                    home_path.symlink_to(real_home, target_is_directory=True)
                    side_effect_roots.extend([case_root, real_home])
                elif home_mode == "symlink_to_file":
                    target_file = case_root / "home_target_file"
                    target_file.write_text("not a dir\n", encoding="utf-8")
                    home_path = case_root / "home_symlink_file"
                    home_path.symlink_to(target_file)
                    side_effect_roots.append(case_root)
                elif home_mode == "dangling_symlink":
                    home_path = case_root / "home_dangling_symlink"
                    home_path.symlink_to(
                        case_root / "missing-home-target", target_is_directory=True
                    )
                    side_effect_roots.append(case_root)
                else:
                    real_parent = case_root / "real_parent"
                    real_parent.mkdir()
                    symlink_parent = case_root / "symlink_parent"
                    symlink_parent.symlink_to(real_parent, target_is_directory=True)
                    home_path = symlink_parent / "orch_home"
                    side_effect_roots.extend([case_root, real_parent])

                if plan_mode == "missing_path":
                    plan_path = case_root / "missing_plan.yaml"
                elif plan_mode == "dangling_symlink_path":
                    plan_path = case_root / "dangling_plan_link.yaml"
                    plan_path.symlink_to(case_root / "missing_plan_target.yaml")
                else:
                    real_missing_parent = case_root / "real_missing_parent"
                    real_missing_parent.mkdir()
                    missing_parent_link = case_root / "missing_parent_link"
                    missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                    plan_path = missing_parent_link / "missing_plan.yaml"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "run",
                        str(plan_path),
                        "--home",
                        str(home_path),
                        "--dry-run",
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                context = f"{plan_mode}-{home_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "PLAN_PATH" in output, context
                assert "Invalid value for 'PLAN_PATH'" in output, context
                assert "Plan validation error" not in output, context
                assert "Invalid home" not in output, context
                assert "Invalid workdir" not in output, context
                assert "contains symlink component" not in output, context
                assert "Dry Run" not in output, context
                assert "run_id:" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                for root in side_effect_roots:
                    assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_home_and_workdir_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    plan_modes = ("missing_path", "dangling_symlink_path", "symlink_ancestor_missing_path")
    home_modes = (
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for plan_mode in plan_modes:
            for home_mode in home_modes:
                case_root = (
                    tmp_path / f"missing_plan_vs_home_workdir_{plan_mode}_{home_mode}_{order_label}"
                )
                case_root.mkdir()
                side_effect_roots: list[Path] = []
                invalid_workdir_file = case_root / "invalid_workdir"
                invalid_workdir_file.write_text("file\n", encoding="utf-8")

                if home_mode == "home_file":
                    home_path = case_root / "home_file"
                    home_path.write_text("not a dir\n", encoding="utf-8")
                    side_effect_roots.append(case_root)
                elif home_mode == "file_ancestor":
                    parent_file = case_root / "home_parent_file"
                    parent_file.write_text("not a dir\n", encoding="utf-8")
                    home_path = parent_file / "orch_home"
                    side_effect_roots.append(case_root)
                elif home_mode == "symlink_to_dir":
                    real_home = case_root / "real_home"
                    real_home.mkdir()
                    home_path = case_root / "home_symlink_dir"
                    home_path.symlink_to(real_home, target_is_directory=True)
                    side_effect_roots.extend([case_root, real_home])
                elif home_mode == "symlink_to_file":
                    target_file = case_root / "home_target_file"
                    target_file.write_text("not a dir\n", encoding="utf-8")
                    home_path = case_root / "home_symlink_file"
                    home_path.symlink_to(target_file)
                    side_effect_roots.append(case_root)
                elif home_mode == "dangling_symlink":
                    home_path = case_root / "home_dangling_symlink"
                    home_path.symlink_to(
                        case_root / "missing-home-target", target_is_directory=True
                    )
                    side_effect_roots.append(case_root)
                else:
                    real_parent = case_root / "real_parent"
                    real_parent.mkdir()
                    symlink_parent = case_root / "symlink_parent"
                    symlink_parent.symlink_to(real_parent, target_is_directory=True)
                    home_path = symlink_parent / "orch_home"
                    side_effect_roots.extend([case_root, real_parent])

                if plan_mode == "missing_path":
                    plan_path = case_root / "missing_plan.yaml"
                elif plan_mode == "dangling_symlink_path":
                    plan_path = case_root / "dangling_plan_link.yaml"
                    plan_path.symlink_to(case_root / "missing_plan_target.yaml")
                else:
                    real_missing_parent = case_root / "real_missing_parent"
                    real_missing_parent.mkdir()
                    missing_parent_link = case_root / "missing_parent_link"
                    missing_parent_link.symlink_to(real_missing_parent, target_is_directory=True)
                    plan_path = missing_parent_link / "missing_plan.yaml"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "run",
                        str(plan_path),
                        "--home",
                        str(home_path),
                        "--workdir",
                        str(invalid_workdir_file),
                        "--dry-run",
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                context = f"{plan_mode}-{home_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "PLAN_PATH" in output, context
                assert "Invalid value for 'PLAN_PATH'" in output, context
                assert "Plan validation error" not in output, context
                assert "Invalid home" not in output, context
                assert "Invalid workdir" not in output, context
                assert "contains symlink component" not in output, context
                assert "Dry Run" not in output, context
                assert "run_id:" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                for root in side_effect_roots:
                    assert not (root / "runs").exists(), context


def test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_invalid_home_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_home_vs_plan.yaml"
    home_file = tmp_path / "home_file_both_toggles_reverse_plan"
    home_file.write_text("not a dir\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_invalid_home_precedes_non_regular_plan(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    plan_path = tmp_path / "plan_fifo_both_toggles_reverse.yaml"
    os.mkfifo(plan_path)
    home_file = tmp_path / "home_file_both_toggles_reverse_fifo_plan"
    home_file.write_text("not a dir\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_invalid_home_precedes_non_regular_plan_and_workdir(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    plan_path = tmp_path / "plan_fifo_both_toggles_reverse_with_workdir.yaml"
    os.mkfifo(plan_path)
    home_file = tmp_path / "home_file_both_toggles_reverse_fifo_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_fifo_plan_home"
    home_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_invalid_home_precedes_symlink_plan_path(
    tmp_path: Path,
) -> None:
    real_plan = tmp_path / "real_plan_both_toggles_reverse_home_vs_symlink_plan.yaml"
    symlink_plan = tmp_path / "symlink_plan_both_toggles_reverse_home_vs_symlink_plan.yaml"
    home_file = tmp_path / "home_file_both_toggles_reverse_symlink_plan"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    symlink_plan.symlink_to(real_plan)
    home_file.write_text("not a dir\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(symlink_plan),
            "--home",
            str(home_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_invalid_home_precedes_symlink_ancestor_plan_path(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_plan_parent_both_toggles_reverse_home_vs_plan_ancestor"
    real_parent.mkdir()
    real_plan = real_parent / "plan.yaml"
    symlink_parent = tmp_path / "plan_parent_link_both_toggles_reverse_home_vs_plan_ancestor"
    home_file = tmp_path / "home_file_both_toggles_reverse_plan_ancestor"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    home_file.write_text("not a dir\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(symlink_parent / "plan.yaml"),
            "--home",
            str(home_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_symlink_home_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_home_vs_plan.yaml"
    real_home = tmp_path / "real_home_both_toggles_reverse"
    home_symlink = tmp_path / "home_symlink_both_toggles_reverse"
    real_home.mkdir()
    home_symlink.symlink_to(real_home, target_is_directory=True)
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_dry_run_both_toggles_reverse_reject_symlink_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_home_only.yaml"
    real_home = tmp_path / "real_home_both_toggles_reverse_only"
    home_symlink = tmp_path / "home_symlink_both_toggles_reverse_only"
    real_home.mkdir()
    home_symlink.symlink_to(real_home, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_dry_run_both_toggles_reverse_symlink_to_file_home_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_file_home_vs_plan.yaml"
    home_target_file = tmp_path / "home_target_file_both_toggles_reverse_plan"
    home_symlink = tmp_path / "home_symlink_to_file_both_toggles_reverse_plan"
    home_target_file.write_text("home-target-file\n", encoding="utf-8")
    home_symlink.symlink_to(home_target_file)
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_reject_symlink_to_file_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_file_home_only.yaml"
    home_target_file = tmp_path / "home_target_file_both_toggles_reverse_only"
    home_symlink = tmp_path / "home_symlink_to_file_both_toggles_reverse_only"
    home_target_file.write_text("home-target-file\n", encoding="utf-8")
    home_symlink.symlink_to(home_target_file)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_symlink_to_file_home_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_file_home_vs_workdir.yaml"
    home_target_file = tmp_path / "home_target_file_both_toggles_reverse_workdir"
    home_symlink = tmp_path / "home_symlink_to_file_both_toggles_reverse_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_symlink_file_home"
    home_target_file.write_text("home-target-file\n", encoding="utf-8")
    home_symlink.symlink_to(home_target_file)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_symlink_to_file_home_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_file_home_vs_plan_workdir.yaml"
    home_target_file = tmp_path / "home_target_file_both_toggles_reverse_plan_workdir"
    home_symlink = tmp_path / "home_symlink_to_file_both_toggles_reverse_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_symlink_file_plan"
    home_target_file.write_text("home-target-file\n", encoding="utf-8")
    home_symlink.symlink_to(home_target_file)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_reject_dangling_symlink_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_dangling_symlink_home_only.yaml"
    dangling_home = tmp_path / "dangling_home_both_toggles_reverse_only"
    dangling_home.symlink_to(
        tmp_path / "missing-home-target-reverse-only", target_is_directory=True
    )
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(dangling_home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_reject_symlink_ancestor_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_ancestor_home_only.yaml"
    real_parent = tmp_path / "real_parent_both_toggles_reverse_only"
    symlink_parent = tmp_path / "home_parent_link_both_toggles_reverse_only"
    nested_home = symlink_parent / "orch_home"
    real_parent.mkdir()
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_reject_file_ancestor_home(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_file_ancestor_home_only.yaml"
    home_parent_file = tmp_path / "home_parent_file_both_toggles_reverse_only"
    nested_home = home_parent_file / "orch_home"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_dangling_symlink_home_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_dangling_symlink_home_vs_plan.yaml"
    dangling_home = tmp_path / "dangling_home_both_toggles_reverse_plan"
    dangling_home.symlink_to(
        tmp_path / "missing-home-target-reverse-plan", target_is_directory=True
    )
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(dangling_home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_dangling_symlink_home_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_dangling_symlink_home_vs_workdir.yaml"
    dangling_home = tmp_path / "dangling_home_both_toggles_reverse_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_dangling_home"
    dangling_home.symlink_to(
        tmp_path / "missing-home-target-reverse-workdir", target_is_directory=True
    )
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(dangling_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_dangling_symlink_home_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_dangling_symlink_home_vs_plan_workdir.yaml"
    dangling_home = tmp_path / "dangling_home_both_toggles_reverse_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_dangling_plan"
    dangling_home.symlink_to(
        tmp_path / "missing-home-target-reverse-plan-workdir", target_is_directory=True
    )
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(dangling_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_symlink_home_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_home_vs_workdir.yaml"
    real_home = tmp_path / "real_home_both_toggles_reverse_workdir"
    home_symlink = tmp_path / "home_symlink_both_toggles_reverse_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_symlink_home"
    real_home.mkdir()
    home_symlink.symlink_to(real_home, target_is_directory=True)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_dry_run_both_toggles_reverse_symlink_home_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_home_vs_plan_workdir.yaml"
    real_home = tmp_path / "real_home_both_toggles_reverse_plan_workdir"
    home_symlink = tmp_path / "home_symlink_both_toggles_reverse_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_symlink_plan"
    real_home.mkdir()
    home_symlink.symlink_to(real_home, target_is_directory=True)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_symlink),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_dry_run_both_toggles_reverse_symlink_ancestor_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_ancestor_vs_plan_workdir.yaml"
    real_parent = tmp_path / "real_parent_both_toggles_reverse"
    symlink_parent = tmp_path / "home_parent_link_both_toggles_reverse"
    nested_home = symlink_parent / "orch_home"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_symlink_ancestor"
    real_parent.mkdir()
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_symlink_ancestor_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_ancestor_vs_plan.yaml"
    real_parent = tmp_path / "real_parent_both_toggles_reverse_plan"
    symlink_parent = tmp_path / "home_parent_link_both_toggles_reverse_plan"
    nested_home = symlink_parent / "orch_home"
    real_parent.mkdir()
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_toggles_reverse_symlink_ancestor_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_symlink_ancestor_vs_workdir.yaml"
    real_parent = tmp_path / "real_parent_both_toggles_reverse_workdir"
    symlink_parent = tmp_path / "home_parent_link_both_toggles_reverse_workdir"
    nested_home = symlink_parent / "orch_home"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_symlink_ancestor_only"
    real_parent.mkdir()
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output


def test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_invalid_plan_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_plan_vs_workdir.yaml"
    home = tmp_path / ".orch_cli"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_plan"
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "PLAN_PATH" not in output
    assert "Invalid home" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Dry Run" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_invalid_home_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_home_vs_workdir.yaml"
    home_file = tmp_path / "home_file_both_toggles_reverse_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_home"
    home_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_home_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_home_vs_plan_workdir.yaml"
    home_file = tmp_path / "home_file_both_toggles_reverse_plan_workdir"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_plan_home"
    home_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_reject_home_file(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_home_file_only.yaml"
    home_file = tmp_path / "home_file_both_toggles_reverse_only"
    home_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_file_ancestor_precedes_plan_and_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_file_ancestor_vs_plan_workdir.yaml"
    home_parent_file = tmp_path / "home_parent_file_both_toggles_reverse"
    nested_home = home_parent_file / "orch_home"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_file_ancestor"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_file_ancestor_precedes_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_file_ancestor_vs_plan.yaml"
    home_parent_file = tmp_path / "home_parent_file_both_toggles_reverse_plan"
    nested_home = home_parent_file / "orch_home"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    plan_path.write_text("tasks:\n  - id: t1\n    cmd: [\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Plan validation error" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_toggles_reverse_file_ancestor_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_file_ancestor_vs_workdir.yaml"
    home_parent_file = tmp_path / "home_parent_file_both_toggles_reverse_workdir"
    nested_home = home_parent_file / "orch_home"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse_file_ancestor_only"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Invalid workdir" not in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_skip_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_both_toggles_reverse_invalid_workdir.yaml"
    home = tmp_path / ".orch_cli"
    invalid_workdir_file = tmp_path / "invalid_workdir_both_toggles_reverse"
    invalid_workdir_file.write_text("file\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(invalid_workdir_file),
            "--dry-run",
            "--no-fail-fast",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Dry Run" in output
    assert "Invalid workdir" not in output
    assert "run_id:" not in output
    assert "state:" not in output
    assert "report:" not in output
    assert not (home / "runs").exists()


def test_cli_run_rejects_non_positive_max_parallel(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--max-parallel",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = _strip_ansi(proc.stdout + proc.stderr).lower()
    assert proc.returncode == 2
    assert "invalid value" in output
    assert "x>=1" in output


def test_cli_status_succeeds_with_case_only_duplicate_artifact_names(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_case_artifacts.yaml"
    home = tmp_path / ".orch_cli"
    write_case_colliding_outputs = (
        "from pathlib import Path; "
        "Path('out').mkdir(exist_ok=True); "
        "Path('out/a.txt').write_text('a', encoding='utf-8'); "
        "Path('out/A.txt').write_text('A', encoding='utf-8')"
    )
    _write_plan(
        plan_path,
        """
        tasks:
          - id: publish
            cmd:
              - "python3"
              - "-c"
              - "__REPLACE_CMD__"
            outputs: ["out/*.txt"]
        """.replace("__REPLACE_CMD__", write_case_colliding_outputs),
    )

    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    status_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "status",
            run_id,
            "--home",
            str(home),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert status_proc.returncode == 0

    payload = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    publish = tasks["publish"]
    assert isinstance(publish, dict)
    artifact_paths = publish["artifact_paths"]
    assert isinstance(artifact_paths, list)
    assert len(artifact_paths) == 2
    keys = {artifact.casefold() for artifact in artifact_paths}
    assert len(keys) == 2
    assert "artifacts/publish/out/a.txt" in keys
    assert any("__case2" in artifact for artifact in artifact_paths)


def test_cli_run_rejects_file_home_path(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_file),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output


def test_cli_run_rejects_home_with_file_ancestor(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output


def test_cli_run_with_relative_home_writes_absolute_state_home(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    relative_home = ".orch_rel"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            relative_home,
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    status_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "status",
            run_id,
            "--home",
            relative_home,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert status_proc.returncode == 0

    state_path = tmp_path / relative_home / "runs" / run_id / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["home"] == str((tmp_path / relative_home).resolve())


def test_cli_run_with_relative_workdir_writes_absolute_state_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home = tmp_path / ".orch_cli"
    relative_workdir = "."
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            relative_workdir,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    state_path = home / "runs" / run_id / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["workdir"] == str(tmp_path.resolve())


def test_cli_run_rejects_symlink_file_workdir_without_creating_run_dir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home = tmp_path / ".orch_cli"
    workdir_target_file = tmp_path / "workdir_target_file.txt"
    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
    linked_workdir = tmp_path / "linked_workdir"
    linked_workdir.symlink_to(workdir_target_file)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(linked_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid workdir" in output
    assert "contains symlink component" not in output
    assert not (home / "runs").exists()
    assert workdir_target_file.read_text(encoding="utf-8") == "not a directory\n"


def test_cli_run_accepts_symlink_directory_workdir_and_persists_resolved(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.yaml"
    home = tmp_path / ".orch_cli"
    real_workdir = tmp_path / "real_workdir"
    real_workdir.mkdir()
    linked_workdir = tmp_path / "linked_workdir"
    linked_workdir.symlink_to(real_workdir, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(linked_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    state_path = home / "runs" / run_id / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["workdir"] == str(real_workdir.resolve())


def test_cli_run_rejects_missing_workdir_without_creating_run_dir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home = tmp_path / ".orch_cli"
    missing_workdir = tmp_path / "does_not_exist"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid workdir" in output
    assert "contains symlink component" not in output
    assert not (home / "runs").exists()


def test_cli_run_rejects_invalid_workdir_modes_without_creating_run_dir_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = tmp_path / f"run_invalid_workdir_modes_{workdir_mode}_{order_label}"
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            home = case_root / ".orch_cli"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not home.exists(), context
            assert not (home / "runs").exists(), context

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_default_home_rejects_invalid_workdir_modes_without_creating_run_dir_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path / f"run_default_home_invalid_workdir_modes_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            default_home = case_root / ".orch_cli"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert not default_home.exists(), context
            assert not (default_home / "runs").exists(), context

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_invalid_workdir_existing_home_preserves_entries_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = tmp_path / f"run_invalid_workdir_existing_home_{workdir_mode}_{order_label}"
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert not (home / "runs").exists(), context
            assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir"], context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_default_home_invalid_workdir_preserves_entries_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path
                / f"run_default_home_invalid_workdir_existing_home_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )
            default_home = case_root / ".orch"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert not (default_home / "runs").exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
            ], context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_invalid_workdir_existing_home_with_runs_preserves_entries_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path
                / f"run_invalid_workdir_existing_home_with_runs_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert sorted(path.name for path in home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], context
            assert sorted(path.name for path in existing_run.iterdir()) == ["plan.yaml"], context
            assert not (existing_run / ".lock").exists(), context
            assert not (existing_run / "cancel.request").exists(), context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_default_home_invalid_workdir_with_runs_preserves_entries_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path
                / f"run_default_home_invalid_workdir_with_runs_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )
            default_home = case_root / ".orch"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = default_home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (default_home / "runs").iterdir()) == [
                "keep_run"
            ], context
            assert sorted(path.name for path in existing_run.iterdir()) == ["plan.yaml"], context
            assert not (existing_run / ".lock").exists(), context
            assert not (existing_run / "cancel.request").exists(), context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_invalid_workdir_existing_home_run_artifacts_preserved_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path
                / f"run_invalid_workdir_existing_home_run_artifacts_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")
            lock_file = existing_run / ".lock"
            lock_file.write_text("lock\n", encoding="utf-8")
            cancel_request = existing_run / "cancel.request"
            cancel_request.write_text("cancel\n", encoding="utf-8")
            run_log = existing_run / "task.log"
            run_log.write_text("log\n", encoding="utf-8")

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert home.exists(), context
            assert sorted(path.name for path in home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], context
            assert sorted(path.name for path in existing_run.iterdir()) == [
                ".lock",
                "cancel.request",
                "plan.yaml",
                "task.log",
            ], context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert lock_file.read_text(encoding="utf-8") == "lock\n", context
            assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
            assert run_log.read_text(encoding="utf-8") == "log\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_default_home_invalid_workdir_run_artifacts_preserved_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path
                / f"run_default_home_invalid_workdir_run_artifacts_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )
            default_home = case_root / ".orch"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = default_home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")
            lock_file = existing_run / ".lock"
            lock_file.write_text("lock\n", encoding="utf-8")
            cancel_request = existing_run / "cancel.request"
            cancel_request.write_text("cancel\n", encoding="utf-8")
            run_log = existing_run / "task.log"
            run_log.write_text("log\n", encoding="utf-8")

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "Dry Run" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert default_home.exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (default_home / "runs").iterdir()) == [
                "keep_run"
            ], context
            assert sorted(path.name for path in existing_run.iterdir()) == [
                ".lock",
                "cancel.request",
                "plan.yaml",
                "task.log",
            ], context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert lock_file.read_text(encoding="utf-8") == "lock\n", context
            assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
            assert run_log.read_text(encoding="utf-8") == "log\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_run_symlink_ancestor_invalid_workdir_suppresses_component_detail_matrix(
    tmp_path: Path,
) -> None:
    modes = ("plain", "with_runs", "artifacts")
    for dry_run in (False, True):
        for use_default_home in (False, True):
            for mode in modes:
                case_root = (
                    tmp_path / f"run_symlink_ancestor_detail_{dry_run}_{use_default_home}_{mode}"
                )
                case_root.mkdir()
                plan_path = case_root / "plan.yaml"
                _write_plan(
                    plan_path,
                    """
                    tasks:
                      - id: t1
                        cmd: ["python3", "-c", "print('ok')"]
                    """,
                )
                home = case_root / (".orch" if use_default_home else ".orch_cli")
                home.mkdir()
                sentinel_file = home / "keep.txt"
                sentinel_file.write_text("keep\n", encoding="utf-8")
                sentinel_dir = home / "keep_dir"
                sentinel_dir.mkdir()
                nested_file = sentinel_dir / "nested.txt"
                nested_file.write_text("nested\n", encoding="utf-8")

                existing_run = home / "runs" / "keep_run"
                plan_file = existing_run / "plan.yaml"
                lock_file = existing_run / ".lock"
                cancel_request = existing_run / "cancel.request"
                run_log = existing_run / "task.log"
                if mode in {"with_runs", "artifacts"}:
                    existing_run.mkdir(parents=True)
                    plan_file.write_text("tasks: []\n", encoding="utf-8")
                if mode == "artifacts":
                    lock_file.write_text("lock\n", encoding="utf-8")
                    cancel_request.write_text("cancel\n", encoding="utf-8")
                    run_log.write_text("log\n", encoding="utf-8")

                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

                cmd = [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(invalid_workdir),
                ]
                if dry_run:
                    cmd.append("--dry-run")
                if not use_default_home:
                    cmd += ["--home", str(home)]

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=case_root if use_default_home else None,
                )
                output = proc.stdout + proc.stderr
                context = f"{dry_run}-{use_default_home}-{mode}"
                assert "contains symlink component" not in output, context
                assert "run_id:" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                assert home.exists(), context
                assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
                assert sentinel_dir.is_dir(), context
                assert nested_file.read_text(encoding="utf-8") == "nested\n", context
                assert not (real_workdir_parent / "child_workdir").exists(), context

                if dry_run:
                    assert proc.returncode == 0, context
                    assert "Dry Run" in output, context
                    assert "Invalid workdir" not in output, context
                else:
                    assert proc.returncode == 2, context
                    assert "Dry Run" not in output, context
                    assert "Invalid workdir" in output, context

                if mode == "plain":
                    assert not (home / "runs").exists(), context
                elif mode == "with_runs":
                    assert sorted(path.name for path in (home / "runs").iterdir()) == [
                        "keep_run"
                    ], context
                    assert sorted(path.name for path in existing_run.iterdir()) == ["plan.yaml"], (
                        context
                    )
                    assert not lock_file.exists(), context
                    assert not cancel_request.exists(), context
                    assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
                else:
                    assert mode == "artifacts"
                    assert sorted(path.name for path in (home / "runs").iterdir()) == [
                        "keep_run"
                    ], context
                    assert sorted(path.name for path in existing_run.iterdir()) == [
                        ".lock",
                        "cancel.request",
                        "plan.yaml",
                        "task.log",
                    ], context
                    assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
                    assert lock_file.read_text(encoding="utf-8") == "lock\n", context
                    assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
                    assert run_log.read_text(encoding="utf-8") == "log\n", context


def test_cli_run_invalid_home_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home = tmp_path / "home_file"
    home.write_text("not a directory\n", encoding="utf-8")
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert home.read_text(encoding="utf-8") == "not a directory\n"


def test_cli_run_home_file_ancestor_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_home_symlink_to_file_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_link),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_run_home_symlink_directory_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_link),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_home_symlink_ancestor_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / "orch_home"
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not (real_parent / "orch_home" / "runs").exists()


def test_cli_run_home_dangling_symlink_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_link),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not missing_home_target.exists()


def test_cli_run_invalid_plan_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid.yaml"
    home = tmp_path / ".orch_cli"
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert "Invalid workdir" not in output
    assert not (home / "runs").exists()


def test_cli_run_invalid_home_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid.yaml"
    home = tmp_path / "home_file"
    home.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert home.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_dry_run_invalid_home_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid_dry.yaml"
    home = tmp_path / "home_file"
    home.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert home.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_home_file_ancestor_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid.yaml"
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_home_symlink_to_file_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid.yaml"
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_link),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_run_home_symlink_directory_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid.yaml"
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_link),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert not (real_home / "runs").exists()


def test_cli_dry_run_home_symlink_directory_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid_dry.yaml"
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_link),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_home_symlink_ancestor_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid.yaml"
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / "orch_home"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert not (real_parent / "orch_home" / "runs").exists()


def test_cli_dry_run_home_symlink_ancestor_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid_dry.yaml"
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / "orch_home"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(nested_home),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert not (real_parent / "orch_home" / "runs").exists()


def test_cli_dry_run_home_symlink_to_file_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid_dry.yaml"
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_link),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_run_home_dangling_symlink_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid.yaml"
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_link),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert not missing_home_target.exists()


def test_cli_dry_run_home_dangling_symlink_precedes_invalid_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid_dry.yaml"
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_link),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Plan validation error" not in output
    assert not missing_home_target.exists()


def test_cli_dry_run_valid_plan_ignores_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    home = tmp_path / ".orch_cli"
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "task_id" in output
    assert "t1" in output
    assert "Invalid workdir" not in output
    assert not (home / "runs").exists()


def test_cli_dry_run_invalid_home_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_file),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_dry_run_home_dangling_symlink_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_link),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not missing_home_target.exists()


def test_cli_dry_run_home_symlink_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_link),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not (real_home / "runs").exists()


def test_cli_dry_run_home_file_ancestor_precedes_invalid_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(nested_home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_dry_run_home_symlink_to_file_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_link),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_dry_run_home_symlink_ancestor_precedes_invalid_workdir(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / "orch_home"
    missing_workdir = tmp_path / "missing_workdir"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(nested_home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not (real_parent / "orch_home" / "runs").exists()


def test_cli_run_rejects_home_with_symlink_ancestor_without_side_effect(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home = tmp_path / "home_link"
    home.symlink_to(real_home, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert not (real_home / "runs").exists()


def test_cli_run_rejects_home_symlink_to_file_without_side_effect(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_link),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_run_rejects_home_with_symlink_ancestor_component_without_side_effect(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.yaml"
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home_name = "orch_home"
    nested_home = symlink_parent / nested_home_name
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(nested_home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert not (real_parent / nested_home_name / "runs").exists()


def test_cli_dry_run_rejects_home_with_symlink_ancestor_component(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home_name = "orch_home"
    nested_home = symlink_parent / nested_home_name
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(nested_home),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Dry Run" not in output
    assert "contains symlink component" not in output
    assert not (real_parent / nested_home_name / "runs").exists()


def test_cli_dry_run_rejects_symlink_home_path(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_link),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Dry Run" not in output
    assert "contains symlink component" not in output
    assert not (real_home / "runs").exists()


def test_cli_dry_run_rejects_home_symlink_to_file_path(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_link),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Dry Run" not in output
    assert "contains symlink component" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_dry_run_rejects_home_file_path(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_dry.yaml"
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Dry Run" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_run_handles_non_directory_runs_path(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    home = tmp_path / ".orch_cli"
    home.mkdir(parents=True)
    (home / "runs").write_text("not a directory\n", encoding="utf-8")
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Failed to initialize run" in output


def test_cli_run_sanitizes_runs_symlink_path_initialize_error(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_runs_symlink_error.yaml"
    home = tmp_path / ".orch_cli"
    home.mkdir(parents=True)
    real_runs = tmp_path / "real_runs"
    real_runs.mkdir()
    (home / "runs").symlink_to(real_runs, target_is_directory=True)
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Failed to initialize run" in output
    assert "invalid run path" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert sorted(path.name for path in real_runs.iterdir()) == []


def test_cli_resume_rejects_non_positive_max_parallel(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            "dummy_run",
            "--max-parallel",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = _strip_ansi(proc.stdout + proc.stderr).lower()
    assert proc.returncode == 2
    assert "invalid value" in output
    assert "x>=1" in output


def test_cli_logs_rejects_non_positive_tail(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_logs_tail_invalid.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    logs_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "logs",
            run_id,
            "--home",
            str(home),
            "--tail",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = _strip_ansi(logs_proc.stdout + logs_proc.stderr).lower()
    assert logs_proc.returncode == 2
    assert "invalid value" in output
    assert "x>=1" in output


def test_cli_resume_rejects_missing_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_resume_wd.yaml"
    home = tmp_path / ".orch_cli"
    missing_workdir = tmp_path / "missing_resume_wd"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    resume_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            run_id,
            "--home",
            str(home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resume_proc.stdout + resume_proc.stderr
    assert resume_proc.returncode == 2
    assert "Invalid workdir" in output
    assert "contains symlink component" not in output


def test_cli_resume_invalid_home_precedes_invalid_workdir(tmp_path: Path) -> None:
    home = tmp_path / "home_file"
    home.write_text("not a dir\n", encoding="utf-8")
    missing_workdir = tmp_path / "missing_resume_wd"

    resume_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            "20260101_000000_abcdef",
            "--home",
            str(home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resume_proc.stdout + resume_proc.stderr
    assert resume_proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert home.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_resume_home_file_ancestor_precedes_invalid_workdir(tmp_path: Path) -> None:
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    missing_workdir = tmp_path / "missing_resume_wd"

    resume_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            "20260101_000000_abcdef",
            "--home",
            str(nested_home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resume_proc.stdout + resume_proc.stderr
    assert resume_proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_resume_home_symlink_to_file_precedes_invalid_workdir(tmp_path: Path) -> None:
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    missing_workdir = tmp_path / "missing_resume_wd"

    resume_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            "20260101_000000_abcdef",
            "--home",
            str(home_link),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resume_proc.stdout + resume_proc.stderr
    assert resume_proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_resume_home_symlink_directory_precedes_invalid_workdir(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    missing_workdir = tmp_path / "missing_resume_wd"

    resume_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            "20260101_000000_abcdef",
            "--home",
            str(home_link),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resume_proc.stdout + resume_proc.stderr
    assert resume_proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not (real_home / "runs").exists()


def test_cli_resume_home_symlink_ancestor_precedes_invalid_workdir(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / "orch_home"
    missing_workdir = tmp_path / "missing_resume_wd"

    resume_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            "20260101_000000_abcdef",
            "--home",
            str(nested_home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resume_proc.stdout + resume_proc.stderr
    assert resume_proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not (real_parent / "orch_home" / "runs").exists()


def test_cli_resume_home_dangling_symlink_precedes_invalid_workdir(tmp_path: Path) -> None:
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)
    missing_workdir = tmp_path / "missing_resume_wd"

    resume_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            "20260101_000000_abcdef",
            "--home",
            str(home_link),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resume_proc.stdout + resume_proc.stderr
    assert resume_proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not missing_home_target.exists()


def test_cli_resume_rejects_symlink_file_workdir(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_resume_symlink_file_wd.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    workdir_target_file = tmp_path / "workdir_target_file.txt"
    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
    linked_workdir = tmp_path / "linked_workdir"
    linked_workdir.symlink_to(workdir_target_file)

    resume_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            run_id,
            "--home",
            str(home),
            "--workdir",
            str(linked_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resume_proc.stdout + resume_proc.stderr
    assert resume_proc.returncode == 2
    assert "Invalid workdir" in output
    assert "contains symlink component" not in output
    assert workdir_target_file.read_text(encoding="utf-8") == "not a directory\n"


def test_cli_resume_rejects_invalid_workdir_modes_matrix(tmp_path: Path) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = tmp_path / f"resume_invalid_workdir_modes_{workdir_mode}_{order_label}"
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            home = case_root / ".orch_cli"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )
            run_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--home",
                    str(home),
                    "--workdir",
                    str(case_root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert run_proc.returncode == 0
            run_id = _extract_run_id(run_proc.stdout)
            state_path = home / "runs" / run_id / "state.json"
            baseline_state = state_path.read_text(encoding="utf-8")

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            resume_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "resume",
                    run_id,
                    "--home",
                    str(home),
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = resume_proc.stdout + resume_proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert resume_proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Run not found or broken" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert state_path.read_text(encoding="utf-8") == baseline_state, context
            assert sorted(path.name for path in (home / "runs").iterdir()) == [run_id], context

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_resume_default_home_rejects_invalid_workdir_modes_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for workdir_mode in workdir_modes:
            case_root = (
                tmp_path / f"resume_default_home_invalid_workdir_modes_{workdir_mode}_{order_label}"
            )
            case_root.mkdir()
            plan_path = case_root / "plan.yaml"
            default_home = case_root / ".orch"
            _write_plan(
                plan_path,
                """
                tasks:
                  - id: t1
                    cmd: ["python3", "-c", "print('ok')"]
                """,
            )
            run_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "run",
                    str(plan_path),
                    "--workdir",
                    str(case_root),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            assert run_proc.returncode == 0
            run_id = _extract_run_id(run_proc.stdout)
            state_path = default_home / "runs" / run_id / "state.json"
            baseline_state = state_path.read_text(encoding="utf-8")

            side_effect_files: list[Path] = []
            if workdir_mode == "missing_path":
                invalid_workdir = case_root / "missing_workdir"
            elif workdir_mode == "file_path":
                invalid_workdir = case_root / "workdir_file"
                invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                side_effect_files.append(invalid_workdir)
            elif workdir_mode == "file_ancestor":
                workdir_parent_file = case_root / "workdir_parent_file"
                workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = workdir_parent_file / "child_workdir"
                side_effect_files.append(workdir_parent_file)
            elif workdir_mode == "symlink_to_file":
                workdir_target_file = case_root / "workdir_target_file"
                workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                invalid_workdir = case_root / "workdir_symlink_to_file"
                invalid_workdir.symlink_to(workdir_target_file)
                side_effect_files.append(workdir_target_file)
            elif workdir_mode == "dangling_symlink":
                invalid_workdir = case_root / "workdir_dangling_symlink"
                invalid_workdir.symlink_to(
                    case_root / "missing_workdir_target", target_is_directory=True
                )
            else:
                real_workdir_parent = case_root / "real_workdir_parent"
                real_workdir_parent.mkdir()
                workdir_parent_link = case_root / "workdir_parent_link"
                workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                invalid_workdir = workdir_parent_link / "child_workdir"

            resume_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    "resume",
                    run_id,
                    "--workdir",
                    str(invalid_workdir),
                    *order,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = resume_proc.stdout + resume_proc.stderr
            context = f"{workdir_mode}-{order_label}"
            assert resume_proc.returncode == 2, context
            assert "Invalid workdir" in output, context
            assert "Invalid home" not in output, context
            assert "Run not found or broken" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id:" not in output, context
            assert "state:" not in output, context
            assert "report:" not in output, context
            assert state_path.read_text(encoding="utf-8") == baseline_state, context
            assert sorted(path.name for path in (default_home / "runs").iterdir()) == [run_id], (
                context
            )

            for file_path in side_effect_files:
                assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
            if workdir_mode == "dangling_symlink":
                assert not (case_root / "missing_workdir_target").exists(), context
            if workdir_mode == "symlink_ancestor":
                assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_resume_accepts_symlink_directory_workdir_and_persists_resolved(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_resume_symlink_dir_wd.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    real_workdir = tmp_path / "real_workdir"
    real_workdir.mkdir()
    linked_workdir = tmp_path / "linked_workdir"
    linked_workdir.symlink_to(real_workdir, target_is_directory=True)

    resume_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            run_id,
            "--home",
            str(home),
            "--workdir",
            str(linked_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resume_proc.returncode == 0

    state_path = home / "runs" / run_id / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["workdir"] == str(real_workdir.resolve())


def test_cli_run_failure_returns_three_and_writes_state(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_fail.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: fail
            cmd: ["python3", "-c", "import sys; sys.exit(1)"]
          - id: skipped
            cmd: ["python3", "-c", "print('never')"]
            depends_on: ["fail"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 3
    run_id = _extract_run_id(proc.stdout)
    state_path = home / "runs" / run_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["tasks"]["fail"]["status"] == "FAILED"
    assert state["tasks"]["skipped"]["status"] == "SKIPPED"


def test_cli_cancel_stops_running_process(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_cancel.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: long
            cmd: ["python3", "-c", "import time; time.sleep(10)"]
          - id: next
            cmd: ["python3", "-c", "print('next')"]
            depends_on: ["long"]
        """,
    )

    run_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    run_id = ""
    deadline = time.time() + 8
    while time.time() < deadline and not run_id:
        runs_dir = home / "runs"
        if runs_dir.exists():
            candidates = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])
            if candidates:
                run_id = candidates[-1]
                break
        time.sleep(0.1)
    assert run_id, "run_id was not created in time"

    cancel_proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cancel_proc.returncode == 0

    stdout, stderr = run_proc.communicate(timeout=20)
    assert run_proc.returncode == 4, f"stdout={stdout}\nstderr={stderr}"
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "CANCELED"


def test_cli_resume_failed_only_does_not_rerun_success(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_resume.yaml"
    home = tmp_path / ".orch_cli"
    gate = tmp_path / "gate.ok"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: root
            cmd: ["python3", "-c", "print('root')"]
          - id: flaky
            cmd:
              [
                "python3",
                "-c",
                "import os,sys;sys.exit(0 if os.path.exists('gate.ok') else 1)",
              ]
            depends_on: ["root"]
        """,
    )

    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 3
    run_id = _extract_run_id(first.stdout)

    gate.write_text("ok\n", encoding="utf-8")
    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            run_id,
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
            "--failed-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "SUCCESS"
    assert state["tasks"]["root"]["status"] == "SUCCESS"
    assert state["tasks"]["root"]["attempts"] == 1
    assert state["tasks"]["flaky"]["status"] == "SUCCESS"
    assert state["tasks"]["flaky"]["attempts"] == 2


def test_cli_resume_uses_copied_plan_not_modified_source(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_resume_copy.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: flaky
            cmd:
              [
                "python3",
                "-c",
                "import os,sys;sys.exit(0 if os.path.exists('gate.ok') else 1)",
              ]
        """,
    )

    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 3
    run_id = _extract_run_id(first.stdout)

    _write_plan(
        plan_path,
        """
        tasks:
          - id: flaky
            cmd: ["python3", "-c", "print('changed plan should not be used')"]
        """,
    )

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            run_id,
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
            "--failed-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 3
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["tasks"]["flaky"]["attempts"] == 2


def test_cli_resume_continues_when_report_write_fails(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_resume_report_fail.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0
    run_id = _extract_run_id(first.stdout)
    report_path = home / "runs" / run_id / "report" / "final_report.md"
    report_path.unlink()
    report_path.mkdir()

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            run_id,
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
            "--failed-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resumed.stdout + resumed.stderr
    assert resumed.returncode == 0
    assert "failed to write report" in output
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "SUCCESS"


def test_cli_resume_continues_when_report_path_is_symlink(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_resume_report_symlink.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0
    run_id = _extract_run_id(first.stdout)
    report_path = home / "runs" / run_id / "report" / "final_report.md"
    outside = tmp_path / "outside_report.md"
    outside.write_text("outside\n", encoding="utf-8")
    report_path.unlink()
    report_path.symlink_to(outside)

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            run_id,
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
            "--failed-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = resumed.stdout + resumed.stderr
    assert resumed.returncode == 0
    assert "failed to write report" in output
    assert outside.read_text(encoding="utf-8") == "outside\n"
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "SUCCESS"


def test_cli_resume_continues_when_report_path_is_fifo(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        return

    plan_path = tmp_path / "plan_resume_report_fifo.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0
    run_id = _extract_run_id(first.stdout)
    report_path = home / "runs" / run_id / "report" / "final_report.md"
    report_path.unlink()
    os.mkfifo(report_path)

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            run_id,
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
            "--failed-only",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    output = resumed.stdout + resumed.stderr
    assert resumed.returncode == 0
    assert "failed to write report" in output
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "SUCCESS"


def test_cli_logs_missing_run_returns_two(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "logs", "missing_run", "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_cancel_missing_run_returns_two_without_creating_run_dir(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_dir = home / "runs" / "missing_run"
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", "missing_run", "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not run_dir.exists()


def test_cli_cancel_handles_non_directory_runs_path(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    home.mkdir(parents=True)
    runs_path = home / "runs"
    runs_path.write_text("not a directory\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", "20260101_000000_abcdef", "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert runs_path.read_text(encoding="utf-8") == "not a directory\n"


def test_cli_cancel_handles_non_directory_run_target_path(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.parent.mkdir(parents=True)
    run_dir.write_text("not a directory\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert run_dir.read_text(encoding="utf-8") == "not a directory\n"
    assert not (run_dir.parent / "cancel.request").exists()


def test_cli_cancel_accepts_existing_run_dir_with_plan_copy(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert (run_dir / "cancel.request").exists()


def test_cli_cancel_accepts_existing_run_dir_with_state_marker_only(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert (run_dir / "cancel.request").exists()


def test_cli_cancel_rejects_existing_run_dir_without_markers(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()


def test_cli_cancel_returns_two_when_cancel_request_write_fails(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    (run_dir / "cancel.request").mkdir()

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to request cancel" in proc.stdout


def test_cli_cancel_sanitizes_symlink_cancel_request_write_error(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    real_cancel = tmp_path / "real_cancel_request.txt"
    real_cancel.write_text("sentinel\n", encoding="utf-8")
    (run_dir / "cancel.request").symlink_to(real_cancel)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Failed to request cancel" in output
    assert "invalid run path" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert real_cancel.read_text(encoding="utf-8") == "sentinel\n"


def test_cli_cancel_rejects_symlink_run_dir_without_side_effect(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    runs_dir = home / "runs"
    runs_dir.mkdir(parents=True)
    outside = tmp_path / "outside_run"
    outside.mkdir()
    (outside / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    symlink_run_dir = runs_dir / run_id
    symlink_run_dir.symlink_to(outside, target_is_directory=True)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (outside / "cancel.request").exists()


def test_cli_cancel_rejects_run_when_runs_path_is_symlink_without_side_effect(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    real_runs = tmp_path / "real_runs"
    real_run_dir = real_runs / run_id
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    home.mkdir(parents=True)
    (home / "runs").symlink_to(real_runs, target_is_directory=True)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (real_run_dir / "cancel.request").exists()


def test_cli_cancel_rejects_run_when_runs_path_symlinks_to_file_without_side_effect(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    outside_file = tmp_path / "outside_runs_file.txt"
    outside_file.write_text("outside\n", encoding="utf-8")
    home.mkdir(parents=True)
    (home / "runs").symlink_to(outside_file)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert outside_file.read_text(encoding="utf-8") == "outside\n"


def test_cli_cancel_rejects_symlink_run_dir_to_file_without_side_effect(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    runs_dir = home / "runs"
    runs_dir.mkdir(parents=True)
    outside_file = tmp_path / "outside_run_file.txt"
    outside_file.write_text("outside\n", encoding="utf-8")
    symlink_run_dir = runs_dir / run_id
    symlink_run_dir.symlink_to(outside_file)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert outside_file.read_text(encoding="utf-8") == "outside\n"


def test_cli_cancel_rejects_run_with_symlink_plan_marker(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    outside_plan = tmp_path / "outside_plan.yaml"
    outside_plan.write_text("tasks: []\n", encoding="utf-8")
    (run_dir / "plan.yaml").symlink_to(outside_plan)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()


def test_cli_cancel_rejects_run_with_symlink_state_marker(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    outside_state = tmp_path / "outside_state.json"
    outside_state.write_text("{}", encoding="utf-8")
    (run_dir / "state.json").symlink_to(outside_state)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()
    assert outside_state.read_text(encoding="utf-8") == "{}"


def test_cli_cancel_rejects_run_with_symlink_state_and_directory_plan_markers(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    outside_state = tmp_path / "outside_state.json"
    outside_state.write_text("{}", encoding="utf-8")
    (run_dir / "state.json").symlink_to(outside_state)
    (run_dir / "plan.yaml").mkdir()

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()
    assert outside_state.read_text(encoding="utf-8") == "{}"


def test_cli_cancel_rejects_run_with_symlink_plan_and_directory_state_markers(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    outside_plan = tmp_path / "outside_plan.yaml"
    outside_plan.write_text("tasks: []\n", encoding="utf-8")
    (run_dir / "state.json").mkdir()
    (run_dir / "plan.yaml").symlink_to(outside_plan)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()
    assert outside_plan.read_text(encoding="utf-8") == "tasks: []\n"


def test_cli_cancel_rejects_run_with_fifo_state_and_directory_plan_markers(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    os.mkfifo(run_dir / "state.json")
    (run_dir / "plan.yaml").mkdir()

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()


def test_cli_cancel_rejects_run_with_directory_state_and_fifo_plan_markers(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").mkdir()
    os.mkfifo(run_dir / "plan.yaml")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()


def test_cli_cancel_rejects_run_with_symlink_state_and_fifo_plan_markers(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    outside_state = tmp_path / "outside_state.json"
    outside_state.write_text("{}", encoding="utf-8")
    (run_dir / "state.json").symlink_to(outside_state)
    os.mkfifo(run_dir / "plan.yaml")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()
    assert outside_state.read_text(encoding="utf-8") == "{}"


def test_cli_cancel_rejects_run_with_fifo_state_and_symlink_plan_markers(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        return

    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    outside_plan = tmp_path / "outside_plan.yaml"
    outside_plan.write_text("tasks: []\n", encoding="utf-8")
    os.mkfifo(run_dir / "state.json")
    (run_dir / "plan.yaml").symlink_to(outside_plan)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()
    assert outside_plan.read_text(encoding="utf-8") == "tasks: []\n"


def test_cli_cancel_accepts_regular_state_with_symlink_plan_marker(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    outside_plan = tmp_path / "outside_plan.yaml"
    outside_plan.write_text("tasks: []\n", encoding="utf-8")
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    (run_dir / "plan.yaml").symlink_to(outside_plan)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert (run_dir / "cancel.request").exists()
    assert outside_plan.read_text(encoding="utf-8") == "tasks: []\n"


def test_cli_cancel_accepts_regular_plan_with_symlink_state_marker(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    outside_state = tmp_path / "outside_state.json"
    outside_state.write_text("{}", encoding="utf-8")
    (run_dir / "state.json").symlink_to(outside_state)
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert (run_dir / "cancel.request").exists()
    assert outside_state.read_text(encoding="utf-8") == "{}"


def test_cli_cancel_rejects_run_with_directory_only_markers(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").mkdir()
    (run_dir / "plan.yaml").mkdir()

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()


def test_cli_cancel_accepts_regular_state_with_directory_plan_marker(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    (run_dir / "plan.yaml").mkdir()

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert (run_dir / "cancel.request").exists()


def test_cli_cancel_accepts_regular_plan_with_directory_state_marker(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").mkdir()
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert (run_dir / "cancel.request").exists()


def test_cli_cancel_rejects_run_with_fifo_only_markers(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        return

    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    os.mkfifo(run_dir / "state.json")
    os.mkfifo(run_dir / "plan.yaml")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert proc.returncode == 2
    assert "Run not found" in proc.stdout
    assert not (run_dir / "cancel.request").exists()


def test_cli_cancel_accepts_regular_state_with_fifo_plan_marker(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        return

    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    os.mkfifo(run_dir / "plan.yaml")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert proc.returncode == 0
    assert (run_dir / "cancel.request").exists()


def test_cli_cancel_accepts_regular_plan_with_fifo_state_marker(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        return

    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    os.mkfifo(run_dir / "state.json")
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert proc.returncode == 0
    assert (run_dir / "cancel.request").exists()


def test_cli_cancel_rejects_run_with_symlink_ancestor_home(tmp_path: Path) -> None:
    run_id = "20260101_000000_abcdef"
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / run_id
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    home = tmp_path / "home_link"
    home.symlink_to(real_home, target_is_directory=True)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert not (real_run_dir / "cancel.request").exists()


def test_cli_cancel_rejects_home_with_symlink_ancestor_without_side_effect(
    tmp_path: Path,
) -> None:
    run_id = "20260101_000000_abcdef"
    real_parent = tmp_path / "real_parent"
    nested_home_name = "orch_home"
    real_run_dir = real_parent / nested_home_name / "runs" / run_id
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / nested_home_name

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", run_id, "--home", str(nested_home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert not (real_run_dir / "cancel.request").exists()


def test_cli_cancel_rejects_home_file_without_side_effect(tmp_path: Path) -> None:
    home = tmp_path / "home_file"
    home.write_text("not a directory\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "cancel",
            "20260101_000000_abcdef",
            "--home",
            str(home),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert home.read_text(encoding="utf-8") == "not a directory\n"


def test_cli_cancel_rejects_home_with_file_ancestor_without_side_effect(
    tmp_path: Path,
) -> None:
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "cancel",
            "20260101_000000_abcdef",
            "--home",
            str(nested_home),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_cancel_rejects_home_symlink_to_file_without_side_effect(
    tmp_path: Path,
) -> None:
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "cancel",
            "20260101_000000_abcdef",
            "--home",
            str(home_link),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_status_missing_run_returns_two(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", "missing_run", "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_handles_non_directory_runs_path(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    home.mkdir(parents=True)
    (home / "runs").write_text("not a directory\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", "20260101_000000_abcdef", "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_logs_handles_non_directory_runs_path(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    home.mkdir(parents=True)
    (home / "runs").write_text("not a directory\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "logs", "20260101_000000_abcdef", "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_incomplete_state_object(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_run_id_mismatch(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": "different_run",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_invalid_timestamp(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "not-iso",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_running_state_with_all_terminal_tasks(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_pending_run_status(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "PENDING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_failed_status_when_any_task_canceled(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "FAILED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "FAILED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 1,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    },
                    "t2": {
                        "status": "CANCELED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 130,
                        "timed_out": False,
                        "canceled": True,
                        "skip_reason": "run_canceled",
                        "stdout_path": "logs/t2.out.log",
                        "stderr_path": "logs/t2.err.log",
                        "artifact_paths": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_failed_status_without_failed_tasks(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "FAILED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SKIPPED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": "upstream_failed",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_canceled_task_without_start_with_runtime_fields(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "CANCELED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "CANCELED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": None,
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": True,
                        "skip_reason": "run_canceled",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_started_canceled_task_with_zero_attempts(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "CANCELED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "CANCELED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 130,
                        "timed_out": False,
                        "canceled": True,
                        "skip_reason": "run_canceled",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_started_canceled_task_without_exit_code(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "CANCELED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "CANCELED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": True,
                        "skip_reason": "run_canceled",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_started_canceled_task_without_duration(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "CANCELED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "CANCELED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": None,
                        "exit_code": 130,
                        "timed_out": False,
                        "canceled": True,
                        "skip_reason": "run_canceled",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_canceled_task_with_artifact_paths(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "CANCELED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "CANCELED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 130,
                        "timed_out": False,
                        "canceled": True,
                        "skip_reason": "run_canceled",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_canceled_task_with_zero_exit_code(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "CANCELED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "CANCELED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": True,
                        "skip_reason": "run_canceled",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_canceled_task_with_missing_timed_out_flag(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "CANCELED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "CANCELED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 130,
                        "timed_out": None,
                        "canceled": True,
                        "skip_reason": "run_canceled",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_running_state_with_terminal_result_fields(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "RUNNING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 2,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": 1,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_running_state_with_artifact_paths(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "RUNNING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 2,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_running_state_with_ended_at(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "RUNNING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 2,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_running_state_with_zero_attempts(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "RUNNING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_accepts_running_state_with_ready_timeout_task(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "READY",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": None,
                        "timed_out": True,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "READY" in proc.stdout


def test_cli_status_rejects_running_state_with_missing_bool_flags(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "RUNNING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": None,
                        "canceled": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_accepts_running_state_with_pending_timeout_task(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": None,
                        "timed_out": True,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "PENDING" in proc.stdout


def test_cli_status_accepts_running_state_with_pending_non_timeout_retry_task(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 1,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "PENDING" in proc.stdout


def test_cli_status_rejects_pending_timeout_task_with_zero_attempts(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": True,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_pending_timeout_task_without_timestamps(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": 1.0,
                        "exit_code": None,
                        "timed_out": True,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_running_state_with_ready_missing_timed_out_flag(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "READY",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 1,
                        "timed_out": None,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_fresh_pending_task_with_runtime_fields(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 1,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_pending_task_with_success_exit_code(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_pending_task_with_skip_reason(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 1,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": "previous_failure",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_pending_task_with_missing_bool_flags(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": None,
                        "canceled": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_missing_attempts(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": None,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_missing_retries(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": None,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_missing_timeout_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_missing_retry_backoff_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_backoff_longer_than_retries(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.1, 0.2],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_missing_cwd_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_missing_env_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_case_insensitive_duplicate_outputs(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": ["dist/report.txt", "dist/REPORT.txt"],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_unknown_root_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
                "unexpected_root": "noise",
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_unknown_task_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                        "unexpected_task_field": "noise",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_missing_started_at_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 0,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_missing_skip_reason_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_running_state_with_ready_attempts_exhausted(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "READY",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 2,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 1,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_running_state_with_ready_without_duration(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "READY",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 2,
                        "retry_backoff_sec": [0.5, 1.0],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": None,
                        "exit_code": 1,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_unsafe_plan_relpath(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "../plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_non_string_goal(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": ["invalid"],
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_missing_goal_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_relative_home_workdir(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": ".orch",
                "workdir": ".",
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_symlink_state_file(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    real_state = tmp_path / "external_state.json"
    real_state.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "state.json").symlink_to(real_state)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_path_with_symlink_ancestor(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    runs_dir = home / "runs"
    runs_dir.mkdir(parents=True)
    real_run_dir = tmp_path / "real_run"
    real_run_dir.mkdir()
    symlink_run_dir = runs_dir / run_id
    symlink_run_dir.symlink_to(real_run_dir, target_is_directory=True)
    (real_run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_run_dir_with_symlink_ancestor_without_lock_side_effect(
    tmp_path: Path,
) -> None:
    run_id = "20260101_000000_abcdef"
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / run_id
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(real_home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "PENDING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    home = tmp_path / "home_link"
    home.symlink_to(real_home, target_is_directory=True)
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert not (real_run_dir / ".lock").exists()


def test_cli_status_logs_resume_sanitize_runs_symlink_path_error(tmp_path: Path) -> None:
    run_id = "20260101_000000_abcdef"
    home = tmp_path / ".orch_cli"
    home.mkdir()
    real_runs = tmp_path / "real_runs"
    real_runs.mkdir()
    (home / "runs").symlink_to(real_runs, target_is_directory=True)
    real_run_dir = real_runs / run_id
    real_run_dir.mkdir()
    (real_run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home.resolve()),
                "workdir": str(tmp_path.resolve()),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {},
            }
        ),
        encoding="utf-8",
    )
    (real_run_dir / "plan.yaml").write_text(
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
        encoding="utf-8",
    )

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, run_id, "--home", str(home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        if command == "resume":
            assert "Run not found or broken" in output, command
        else:
            assert "Failed to load state" in output, command
        assert "invalid run path" in output, command
        assert "contains symlink component" not in output, command
        assert "must not include symlink" not in output, command
        assert "must not be symlink" not in output, command

    assert not (real_run_dir / ".lock").exists()


def test_cli_logs_rejects_symlink_home_path(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    run_id = "20260101_000000_abcdef"
    real_run_dir = real_home / "runs" / run_id
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(real_home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {},
            }
        ),
        encoding="utf-8",
    )

    home = tmp_path / "home_link"
    home.symlink_to(real_home, target_is_directory=True)
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "logs", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert not (real_run_dir / ".lock").exists()


def test_cli_run_rejects_dangling_symlink_home_without_side_effect(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "dangling_home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home_link),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert not missing_home_target.exists()


def test_cli_dry_run_rejects_dangling_symlink_home_without_side_effect(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.yaml"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "dangling_home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--dry-run",
            "--home",
            str(home_link),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "Dry Run" not in output
    assert "contains symlink component" not in output
    assert not missing_home_target.exists()


def test_cli_resume_rejects_symlink_home_path(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    run_id = "20260101_000000_abcdef"
    real_run_dir = real_home / "runs" / run_id
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text(
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
        encoding="utf-8",
    )

    home = tmp_path / "home_link"
    home.symlink_to(real_home, target_is_directory=True)
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "resume", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output
    assert not (real_run_dir / ".lock").exists()


def test_cli_status_logs_resume_cancel_reject_dangling_symlink_home(
    tmp_path: Path,
) -> None:
    run_id = "20260101_000000_abcdef"
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "dangling_home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)

    for command in ("status", "logs", "resume", "cancel"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, run_id, "--home", str(home_link)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid home" in output, command
        assert "contains symlink component" not in output, command

    assert not missing_home_target.exists()


def test_cli_status_logs_resume_reject_home_symlink_to_file_path(tmp_path: Path) -> None:
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "orch.cli",
                command,
                "20260101_000000_abcdef",
                "--home",
                str(home_link),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid home" in output, command
        assert "contains symlink component" not in output, command

    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_status_logs_resume_reject_home_with_symlink_ancestor_without_lock_side_effect(
    tmp_path: Path,
) -> None:
    run_id = "20260101_000000_abcdef"
    real_parent = tmp_path / "real_parent"
    nested_home_name = "orch_home"
    real_run_dir = real_parent / nested_home_name / "runs" / run_id
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(real_parent / nested_home_name),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {},
            }
        ),
        encoding="utf-8",
    )
    (real_run_dir / "plan.yaml").write_text(
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
        encoding="utf-8",
    )

    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / nested_home_name

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, run_id, "--home", str(nested_home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid home" in output, command
        assert "contains symlink component" not in output, command

    assert not (real_run_dir / ".lock").exists()


def test_cli_status_rejects_non_regular_state_file(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        return

    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    os.mkfifo(run_dir / "state.json")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_home_mismatch(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(tmp_path / "other_home"),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "RUNNING",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": None,
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_unknown_dependency_ref(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": ["missing"],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_inconsistent_task_flags(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "exit_code": 1,
                        "timed_out": True,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_failed_timeout_task_with_exit_code(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "FAILED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "FAILED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 124,
                        "timed_out": True,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_failed_task_with_missing_timed_out_flag(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "FAILED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "FAILED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": 1.0,
                        "retries": 1,
                        "retry_backoff_sec": [0.5],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 1,
                        "timed_out": None,
                        "canceled": False,
                        "skip_reason": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_skipped_task_with_runtime_fields(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "FAILED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SKIPPED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": "dependency_not_success",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_skipped_task_with_nonzero_attempts(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "FAILED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SKIPPED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": None,
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": "dependency_not_success",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_skipped_task_with_missing_bool_flags(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "FAILED",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SKIPPED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": None,
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": None,
                        "exit_code": None,
                        "timed_out": None,
                        "canceled": None,
                        "skip_reason": "dependency_not_success",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_success_task_with_skip_reason(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "skip_reason": "unexpected",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_success_task_without_exit_code(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": None,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_success_task_without_duration(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": None,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_success_task_with_zero_attempts(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 0,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_success_task_with_missing_bool_flags(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": None,
                        "canceled": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_task_with_missing_artifact_paths_field(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "ended_at": "2026-01-01T00:00:01+00:00",
                        "duration_sec": 1.0,
                        "exit_code": 0,
                        "timed_out": False,
                        "canceled": False,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_success_task_missing_timestamps(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": None,
                        "ended_at": None,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_attempts_over_retry_budget(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 3,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_invalid_task_timestamp(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "started_at": "not-iso",
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_inverted_timestamps(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-02T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_inconsistent_success_state(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "SUCCESS",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "FAILED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_case_insensitive_duplicate_state_tasks(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "Build": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/Build.out.log",
                        "stderr_path": "logs/Build.err.log",
                        "artifact_paths": ["artifacts/Build/out.txt"],
                    },
                    "build": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/build.out.log",
                        "stderr_path": "logs/build.err.log",
                        "artifact_paths": ["artifacts/build/out.txt"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_state_with_duplicate_artifact_paths(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt", "artifacts/t1/OUT.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_logs_rejects_state_with_unsafe_log_path(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "../escape.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "logs", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_logs_rejects_state_with_log_path_bound_to_other_task(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/t2.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "logs", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_logs_rejects_state_with_missing_log_path(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": None,
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "logs", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_logs_rejects_state_with_unexpected_log_filename(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "SUCCESS",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": 1,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.custom.log",
                        "artifact_paths": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "logs", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Failed to load state" in proc.stdout


def test_cli_status_rejects_file_home_path(tmp_path: Path) -> None:
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", "any_run", "--home", str(home_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output


def test_cli_status_rejects_home_with_file_ancestor(tmp_path: Path) -> None:
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "status", "any_run", "--home", str(nested_home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid home" in output
    assert "contains symlink component" not in output


def test_cli_logs_resume_reject_home_file_path(tmp_path: Path) -> None:
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    for command in ("logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, "any_run", "--home", str(home_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid home" in output, command
        assert "contains symlink component" not in output, command
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_logs_resume_reject_home_with_file_ancestor(tmp_path: Path) -> None:
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    for command in ("logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, "any_run", "--home", str(nested_home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid home" in output, command
        assert "contains symlink component" not in output, command
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_status_logs_resume_reject_path_like_run_id(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    bad_run_id = "../escape"
    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "contains symlink component" not in output, command


def test_cli_status_logs_resume_run_id_precedes_invalid_home(tmp_path: Path) -> None:
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    bad_run_id = "../escape"
    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_status_logs_resume_too_long_run_id_precedes_invalid_home(
    tmp_path: Path,
) -> None:
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    bad_run_id = "a" * 129
    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_status_logs_resume_invalid_run_id_precedes_symlink_home(
    tmp_path: Path,
) -> None:
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    bad_run_id = "../escape"
    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home_link)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command


def test_cli_status_logs_resume_too_long_run_id_precedes_symlink_home(
    tmp_path: Path,
) -> None:
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    bad_run_id = "a" * 129
    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home_link)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command


def test_cli_status_logs_resume_invalid_run_id_precedes_dangling_symlink_home(
    tmp_path: Path,
) -> None:
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)
    bad_run_id = "../escape"

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home_link)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command

    assert not missing_home_target.exists()


def test_cli_status_logs_resume_too_long_run_id_precedes_dangling_symlink_home(
    tmp_path: Path,
) -> None:
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)
    bad_run_id = "a" * 129

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home_link)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command

    assert not missing_home_target.exists()


def test_cli_status_logs_resume_invalid_run_id_precedes_home_symlink_to_file(
    tmp_path: Path,
) -> None:
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    bad_run_id = "../escape"

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home_link)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command

    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_status_logs_resume_too_long_run_id_precedes_home_symlink_to_file(
    tmp_path: Path,
) -> None:
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    bad_run_id = "a" * 129

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home_link)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command

    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_status_logs_resume_invalid_run_id_precedes_home_file_ancestor(
    tmp_path: Path,
) -> None:
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    bad_run_id = "../escape"
    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(nested_home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_status_logs_resume_too_long_run_id_precedes_home_file_ancestor(
    tmp_path: Path,
) -> None:
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    bad_run_id = "a" * 129
    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(nested_home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_status_logs_resume_invalid_run_id_precedes_home_symlink_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / "orch_home"
    bad_run_id = "../escape"

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(nested_home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command

    assert not (real_parent / "orch_home" / "runs").exists()


def test_cli_status_logs_resume_too_long_run_id_precedes_home_symlink_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / "orch_home"
    bad_run_id = "a" * 129

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(nested_home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command

    assert not (real_parent / "orch_home" / "runs").exists()


def test_cli_status_logs_resume_invalid_run_id_precedes_home_symlink_ancestor_directory(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home_name = "orch_home"
    nested_home = symlink_parent / nested_home_name
    real_run_dir = real_parent / nested_home_name / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    bad_run_id = "../escape"

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(nested_home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command

    assert not (real_run_dir / ".lock").exists()


def test_cli_status_logs_resume_too_long_run_id_precedes_home_symlink_ancestor_directory(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home_name = "orch_home"
    nested_home = symlink_parent / nested_home_name
    real_run_dir = real_parent / nested_home_name / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    bad_run_id = "a" * 129

    for command in ("status", "logs", "resume"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(nested_home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "Invalid home" not in output, command
        assert "contains symlink component" not in output, command

    assert not (real_run_dir / ".lock").exists()


def test_cli_status_logs_resume_invalid_run_id_precedence_invalid_home_shape_matrix(
    tmp_path: Path,
) -> None:
    commands = ("status", "logs", "resume")
    run_id_modes = ("path_escape", "too_long")
    home_modes = (
        "file_path",
        "symlink",
        "dangling_symlink",
        "symlink_to_file",
        "file_ancestor",
        "symlink_ancestor",
        "symlink_ancestor_directory",
    )

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        for home_mode in home_modes:
            for command in commands:
                case_root = tmp_path / f"run_id_precedence_{run_id_mode}_{home_mode}_{command}"
                case_root.mkdir()
                cmd = [sys.executable, "-m", "orch.cli", command, bad_run_id]

                if home_mode == "file_path":
                    home_file = case_root / "home_file"
                    home_file.write_text("not a dir\n", encoding="utf-8")
                    cmd += ["--home", str(home_file)]
                elif home_mode == "symlink":
                    real_home = case_root / "real_home"
                    real_run_dir = real_home / "runs" / "run1"
                    real_run_dir.mkdir(parents=True)
                    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
                    home_link = case_root / "home_link"
                    home_link.symlink_to(real_home, target_is_directory=True)
                    cmd += ["--home", str(home_link)]
                elif home_mode == "dangling_symlink":
                    missing_home_target = case_root / "missing_home_target"
                    home_link = case_root / "home_link"
                    home_link.symlink_to(missing_home_target, target_is_directory=True)
                    cmd += ["--home", str(home_link)]
                elif home_mode == "symlink_to_file":
                    home_target_file = case_root / "home_target_file.txt"
                    home_target_file.write_text("not a home dir\n", encoding="utf-8")
                    home_link = case_root / "home_link"
                    home_link.symlink_to(home_target_file)
                    cmd += ["--home", str(home_link)]
                elif home_mode == "file_ancestor":
                    home_parent_file = case_root / "home_parent_file"
                    home_parent_file.write_text("not a dir\n", encoding="utf-8")
                    nested_home = home_parent_file / "orch_home"
                    cmd += ["--home", str(nested_home)]
                elif home_mode == "symlink_ancestor":
                    real_parent = case_root / "real_parent"
                    real_parent.mkdir()
                    symlink_parent = case_root / "home_parent_link"
                    symlink_parent.symlink_to(real_parent, target_is_directory=True)
                    nested_home = symlink_parent / "orch_home"
                    cmd += ["--home", str(nested_home)]
                else:
                    assert home_mode == "symlink_ancestor_directory"
                    real_parent = case_root / "real_parent"
                    real_parent.mkdir()
                    symlink_parent = case_root / "home_parent_link"
                    symlink_parent.symlink_to(real_parent, target_is_directory=True)
                    nested_home_name = "orch_home"
                    nested_home = symlink_parent / nested_home_name
                    real_run_dir = real_parent / nested_home_name / "runs" / "run1"
                    real_run_dir.mkdir(parents=True)
                    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
                    cmd += ["--home", str(nested_home)]

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                context = f"{run_id_mode}-{home_mode}-{command}"
                assert proc.returncode == 2, context
                assert "Invalid run_id" in output, context
                assert "Invalid home" not in output, context
                assert "contains symlink component" not in output, context
                assert "run_id: [bold]" not in output, context
                assert "state: [bold]" not in output, context
                assert "report: [bold]" not in output, context

                if home_mode == "file_path":
                    assert home_file.read_text(encoding="utf-8") == "not a dir\n", context
                elif home_mode == "symlink":
                    assert not (real_run_dir / ".lock").exists(), context
                elif home_mode == "dangling_symlink":
                    assert not missing_home_target.exists(), context
                elif home_mode == "symlink_to_file":
                    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n", (
                        context
                    )
                elif home_mode == "file_ancestor":
                    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n", context
                elif home_mode == "symlink_ancestor":
                    assert "contains symlink component" not in output, context
                    assert not (real_parent / "orch_home" / "runs").exists(), context
                else:
                    assert home_mode == "symlink_ancestor_directory"
                    assert "contains symlink component" not in output, context
                    assert not (real_run_dir / ".lock").exists(), context


def test_cli_status_logs_resume_invalid_run_id_existing_home_preserve_entries_matrix(
    tmp_path: Path,
) -> None:
    commands = ("status", "logs", "resume")
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        for command in commands:
            case_root = tmp_path / f"invalid_run_id_existing_home_{run_id_mode}_{command}"
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    command,
                    bad_run_id,
                    "--home",
                    str(home),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{run_id_mode}-{command}"
            assert proc.returncode == 2, context
            assert "Invalid run_id" in output, context
            assert "Invalid home" not in output, context
            assert "Run not found or broken" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id: [bold]" not in output, context
            assert "state: [bold]" not in output, context
            assert "report: [bold]" not in output, context
            assert home.exists(), context
            assert not (home / "runs").exists(), context
            assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir"], context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_status_logs_resume_invalid_run_id_default_home_preserve_entries_matrix(
    tmp_path: Path,
) -> None:
    commands = ("status", "logs", "resume")
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        for command in commands:
            case_root = tmp_path / f"invalid_run_id_default_home_{run_id_mode}_{command}"
            case_root.mkdir()
            default_home = case_root / ".orch"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    command,
                    bad_run_id,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{run_id_mode}-{command}"
            assert proc.returncode == 2, context
            assert "Invalid run_id" in output, context
            assert "Invalid home" not in output, context
            assert "Run not found or broken" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id: [bold]" not in output, context
            assert "state: [bold]" not in output, context
            assert "report: [bold]" not in output, context
            assert default_home.exists(), context
            assert not (default_home / "runs").exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
            ], context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_status_logs_resume_invalid_run_id_existing_home_with_runs_preserve_entries_matrix(
    tmp_path: Path,
) -> None:
    commands = ("status", "logs", "resume")
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        for command in commands:
            case_root = tmp_path / f"invalid_run_id_existing_home_with_runs_{run_id_mode}_{command}"
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            (existing_run / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    command,
                    bad_run_id,
                    "--home",
                    str(home),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{run_id_mode}-{command}"
            assert proc.returncode == 2, context
            assert "Invalid run_id" in output, context
            assert "Invalid home" not in output, context
            assert "Run not found or broken" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id: [bold]" not in output, context
            assert "state: [bold]" not in output, context
            assert "report: [bold]" not in output, context
            assert home.exists(), context
            assert sorted(path.name for path in home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], context
            assert not (existing_run / ".lock").exists(), context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_status_logs_resume_invalid_run_id_default_home_with_runs_preserve_entries_matrix(
    tmp_path: Path,
) -> None:
    commands = ("status", "logs", "resume")
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        for command in commands:
            case_root = tmp_path / f"invalid_run_id_default_home_with_runs_{run_id_mode}_{command}"
            case_root.mkdir()
            default_home = case_root / ".orch"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = default_home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            (existing_run / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    command,
                    bad_run_id,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{run_id_mode}-{command}"
            assert proc.returncode == 2, context
            assert "Invalid run_id" in output, context
            assert "Invalid home" not in output, context
            assert "Run not found or broken" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id: [bold]" not in output, context
            assert "state: [bold]" not in output, context
            assert "report: [bold]" not in output, context
            assert default_home.exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (default_home / "runs").iterdir()) == [
                "keep_run"
            ], context
            assert not (existing_run / ".lock").exists(), context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_status_logs_resume_invalid_run_id_existing_home_run_artifacts_preserved_matrix(
    tmp_path: Path,
) -> None:
    commands = ("status", "logs", "resume")
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        for command in commands:
            case_root = (
                tmp_path / f"invalid_run_id_existing_home_run_artifacts_{run_id_mode}_{command}"
            )
            case_root.mkdir()
            home = case_root / ".orch_cli"
            home.mkdir()
            sentinel_file = home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")
            lock_file = existing_run / ".lock"
            lock_file.write_text("lock\n", encoding="utf-8")
            cancel_request = existing_run / "cancel.request"
            cancel_request.write_text("cancel\n", encoding="utf-8")
            run_log = existing_run / "task.log"
            run_log.write_text("log\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    command,
                    bad_run_id,
                    "--home",
                    str(home),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{run_id_mode}-{command}"
            assert proc.returncode == 2, context
            assert "Invalid run_id" in output, context
            assert "Invalid home" not in output, context
            assert "Run not found or broken" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id: [bold]" not in output, context
            assert "state: [bold]" not in output, context
            assert "report: [bold]" not in output, context
            assert home.exists(), context
            assert sorted(path.name for path in home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], context
            assert sorted(path.name for path in existing_run.iterdir()) == [
                ".lock",
                "cancel.request",
                "plan.yaml",
                "task.log",
            ], context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert lock_file.read_text(encoding="utf-8") == "lock\n", context
            assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
            assert run_log.read_text(encoding="utf-8") == "log\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_status_logs_resume_invalid_run_id_default_home_run_artifacts_preserved_matrix(
    tmp_path: Path,
) -> None:
    commands = ("status", "logs", "resume")
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        for command in commands:
            case_root = (
                tmp_path / f"invalid_run_id_default_home_run_artifacts_{run_id_mode}_{command}"
            )
            case_root.mkdir()
            default_home = case_root / ".orch"
            default_home.mkdir()
            sentinel_file = default_home / "keep.txt"
            sentinel_file.write_text("keep\n", encoding="utf-8")
            sentinel_dir = default_home / "keep_dir"
            sentinel_dir.mkdir()
            nested_file = sentinel_dir / "nested.txt"
            nested_file.write_text("nested\n", encoding="utf-8")
            existing_run = default_home / "runs" / "keep_run"
            existing_run.mkdir(parents=True)
            plan_file = existing_run / "plan.yaml"
            plan_file.write_text("tasks: []\n", encoding="utf-8")
            lock_file = existing_run / ".lock"
            lock_file.write_text("lock\n", encoding="utf-8")
            cancel_request = existing_run / "cancel.request"
            cancel_request.write_text("cancel\n", encoding="utf-8")
            run_log = existing_run / "task.log"
            run_log.write_text("log\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orch.cli",
                    command,
                    bad_run_id,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=case_root,
            )
            output = proc.stdout + proc.stderr
            context = f"{run_id_mode}-{command}"
            assert proc.returncode == 2, context
            assert "Invalid run_id" in output, context
            assert "Invalid home" not in output, context
            assert "Run not found or broken" not in output, context
            assert "Plan validation error" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id: [bold]" not in output, context
            assert "state: [bold]" not in output, context
            assert "report: [bold]" not in output, context
            assert default_home.exists(), context
            assert sorted(path.name for path in default_home.iterdir()) == [
                "keep.txt",
                "keep_dir",
                "runs",
            ], context
            assert sorted(path.name for path in (default_home / "runs").iterdir()) == [
                "keep_run"
            ], context
            assert sorted(path.name for path in existing_run.iterdir()) == [
                ".lock",
                "cancel.request",
                "plan.yaml",
                "task.log",
            ], context
            assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
            assert lock_file.read_text(encoding="utf-8") == "lock\n", context
            assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
            assert run_log.read_text(encoding="utf-8") == "log\n", context
            assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
            assert sentinel_dir.is_dir(), context
            assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_resume_invalid_run_id_precedes_invalid_workdir(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    missing_workdir = tmp_path / "missing_workdir"
    bad_run_id = "../escape"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            bad_run_id,
            "--home",
            str(home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not (home / "runs").exists()


def test_cli_resume_too_long_run_id_precedes_invalid_workdir(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    missing_workdir = tmp_path / "missing_workdir"
    bad_run_id = "a" * 129

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "resume",
            bad_run_id,
            "--home",
            str(home),
            "--workdir",
            str(missing_workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "contains symlink component" not in output
    assert "Invalid workdir" not in output
    assert not (home / "runs").exists()


def test_cli_resume_invalid_run_id_precedes_invalid_workdir_modes_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    run_id_modes = ("path_escape", "too_long")
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for run_id_mode in run_id_modes:
            bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
            for workdir_mode in workdir_modes:
                case_root = (
                    tmp_path
                    / f"resume_invalid_run_id_vs_workdir_{run_id_mode}_{workdir_mode}_{order_label}"
                )
                case_root.mkdir()
                home = case_root / ".orch_cli"

                side_effect_files: list[Path] = []
                if workdir_mode == "missing_path":
                    invalid_workdir = case_root / "missing_workdir"
                elif workdir_mode == "file_path":
                    invalid_workdir = case_root / "workdir_file"
                    invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                    side_effect_files.append(invalid_workdir)
                elif workdir_mode == "file_ancestor":
                    workdir_parent_file = case_root / "workdir_parent_file"
                    workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = workdir_parent_file / "child_workdir"
                    side_effect_files.append(workdir_parent_file)
                elif workdir_mode == "symlink_to_file":
                    workdir_target_file = case_root / "workdir_target_file"
                    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = case_root / "workdir_symlink_to_file"
                    invalid_workdir.symlink_to(workdir_target_file)
                    side_effect_files.append(workdir_target_file)
                elif workdir_mode == "dangling_symlink":
                    invalid_workdir = case_root / "workdir_dangling_symlink"
                    invalid_workdir.symlink_to(
                        case_root / "missing_workdir_target", target_is_directory=True
                    )
                else:
                    real_workdir_parent = case_root / "real_workdir_parent"
                    real_workdir_parent.mkdir()
                    workdir_parent_link = case_root / "workdir_parent_link"
                    workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                    invalid_workdir = workdir_parent_link / "child_workdir"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "resume",
                        bad_run_id,
                        "--home",
                        str(home),
                        "--workdir",
                        str(invalid_workdir),
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                context = f"{run_id_mode}-{workdir_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid run_id" in output, context
                assert "Invalid workdir" not in output, context
                assert "Invalid home" not in output, context
                assert "Run not found or broken" not in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "run_id: [bold]" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                assert not home.exists(), context
                assert not (home / "runs").exists(), context

                for file_path in side_effect_files:
                    assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
                if workdir_mode == "dangling_symlink":
                    assert not (case_root / "missing_workdir_target").exists(), context
                if workdir_mode == "symlink_ancestor":
                    assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_modes_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    run_id_modes = ("path_escape", "too_long")
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for run_id_mode in run_id_modes:
            bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
            for workdir_mode in workdir_modes:
                case_root = (
                    tmp_path / "resume_default_home_invalid_run_id_vs_workdir_"
                    f"{run_id_mode}_{workdir_mode}_{order_label}"
                )
                case_root.mkdir()
                default_home = case_root / ".orch"

                side_effect_files: list[Path] = []
                if workdir_mode == "missing_path":
                    invalid_workdir = case_root / "missing_workdir"
                elif workdir_mode == "file_path":
                    invalid_workdir = case_root / "workdir_file"
                    invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                    side_effect_files.append(invalid_workdir)
                elif workdir_mode == "file_ancestor":
                    workdir_parent_file = case_root / "workdir_parent_file"
                    workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = workdir_parent_file / "child_workdir"
                    side_effect_files.append(workdir_parent_file)
                elif workdir_mode == "symlink_to_file":
                    workdir_target_file = case_root / "workdir_target_file"
                    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = case_root / "workdir_symlink_to_file"
                    invalid_workdir.symlink_to(workdir_target_file)
                    side_effect_files.append(workdir_target_file)
                elif workdir_mode == "dangling_symlink":
                    invalid_workdir = case_root / "workdir_dangling_symlink"
                    invalid_workdir.symlink_to(
                        case_root / "missing_workdir_target", target_is_directory=True
                    )
                else:
                    real_workdir_parent = case_root / "real_workdir_parent"
                    real_workdir_parent.mkdir()
                    workdir_parent_link = case_root / "workdir_parent_link"
                    workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                    invalid_workdir = workdir_parent_link / "child_workdir"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "resume",
                        bad_run_id,
                        "--workdir",
                        str(invalid_workdir),
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=case_root,
                )
                output = proc.stdout + proc.stderr
                context = f"{run_id_mode}-{workdir_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid run_id" in output, context
                assert "Invalid workdir" not in output, context
                assert "Invalid home" not in output, context
                assert "Run not found or broken" not in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "run_id: [bold]" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                assert not default_home.exists(), context
                assert not (default_home / "runs").exists(), context

                for file_path in side_effect_files:
                    assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
                if workdir_mode == "dangling_symlink":
                    assert not (case_root / "missing_workdir_target").exists(), context
                if workdir_mode == "symlink_ancestor":
                    assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_preserves_entries_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    run_id_modes = ("path_escape", "too_long")
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for run_id_mode in run_id_modes:
            bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
            for workdir_mode in workdir_modes:
                case_root = (
                    tmp_path / "resume_invalid_run_id_vs_workdir_existing_home_"
                    f"{run_id_mode}_{workdir_mode}_{order_label}"
                )
                case_root.mkdir()
                home = case_root / ".orch_cli"
                home.mkdir()
                sentinel_file = home / "keep.txt"
                sentinel_file.write_text("keep\n", encoding="utf-8")
                sentinel_dir = home / "keep_dir"
                sentinel_dir.mkdir()
                nested_file = sentinel_dir / "nested.txt"
                nested_file.write_text("nested\n", encoding="utf-8")

                side_effect_files: list[Path] = []
                if workdir_mode == "missing_path":
                    invalid_workdir = case_root / "missing_workdir"
                elif workdir_mode == "file_path":
                    invalid_workdir = case_root / "workdir_file"
                    invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                    side_effect_files.append(invalid_workdir)
                elif workdir_mode == "file_ancestor":
                    workdir_parent_file = case_root / "workdir_parent_file"
                    workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = workdir_parent_file / "child_workdir"
                    side_effect_files.append(workdir_parent_file)
                elif workdir_mode == "symlink_to_file":
                    workdir_target_file = case_root / "workdir_target_file"
                    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = case_root / "workdir_symlink_to_file"
                    invalid_workdir.symlink_to(workdir_target_file)
                    side_effect_files.append(workdir_target_file)
                elif workdir_mode == "dangling_symlink":
                    invalid_workdir = case_root / "workdir_dangling_symlink"
                    invalid_workdir.symlink_to(
                        case_root / "missing_workdir_target", target_is_directory=True
                    )
                else:
                    real_workdir_parent = case_root / "real_workdir_parent"
                    real_workdir_parent.mkdir()
                    workdir_parent_link = case_root / "workdir_parent_link"
                    workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                    invalid_workdir = workdir_parent_link / "child_workdir"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "resume",
                        bad_run_id,
                        "--home",
                        str(home),
                        "--workdir",
                        str(invalid_workdir),
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                context = f"{run_id_mode}-{workdir_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid run_id" in output, context
                assert "Invalid workdir" not in output, context
                assert "Invalid home" not in output, context
                assert "Run not found or broken" not in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "run_id: [bold]" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                assert home.exists(), context
                assert not (home / "runs").exists(), context
                assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir"], (
                    context
                )
                assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
                assert sentinel_dir.is_dir(), context
                assert nested_file.read_text(encoding="utf-8") == "nested\n", context

                for file_path in side_effect_files:
                    assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
                if workdir_mode == "dangling_symlink":
                    assert not (case_root / "missing_workdir_target").exists(), context
                if workdir_mode == "symlink_ancestor":
                    assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_preserve_entries_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    run_id_modes = ("path_escape", "too_long")
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for run_id_mode in run_id_modes:
            bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
            for workdir_mode in workdir_modes:
                case_root = (
                    tmp_path / "resume_default_home_invalid_run_id_vs_workdir_existing_home_"
                    f"{run_id_mode}_{workdir_mode}_{order_label}"
                )
                case_root.mkdir()
                default_home = case_root / ".orch"
                default_home.mkdir()
                sentinel_file = default_home / "keep.txt"
                sentinel_file.write_text("keep\n", encoding="utf-8")
                sentinel_dir = default_home / "keep_dir"
                sentinel_dir.mkdir()
                nested_file = sentinel_dir / "nested.txt"
                nested_file.write_text("nested\n", encoding="utf-8")

                side_effect_files: list[Path] = []
                if workdir_mode == "missing_path":
                    invalid_workdir = case_root / "missing_workdir"
                elif workdir_mode == "file_path":
                    invalid_workdir = case_root / "workdir_file"
                    invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                    side_effect_files.append(invalid_workdir)
                elif workdir_mode == "file_ancestor":
                    workdir_parent_file = case_root / "workdir_parent_file"
                    workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = workdir_parent_file / "child_workdir"
                    side_effect_files.append(workdir_parent_file)
                elif workdir_mode == "symlink_to_file":
                    workdir_target_file = case_root / "workdir_target_file"
                    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = case_root / "workdir_symlink_to_file"
                    invalid_workdir.symlink_to(workdir_target_file)
                    side_effect_files.append(workdir_target_file)
                elif workdir_mode == "dangling_symlink":
                    invalid_workdir = case_root / "workdir_dangling_symlink"
                    invalid_workdir.symlink_to(
                        case_root / "missing_workdir_target", target_is_directory=True
                    )
                else:
                    real_workdir_parent = case_root / "real_workdir_parent"
                    real_workdir_parent.mkdir()
                    workdir_parent_link = case_root / "workdir_parent_link"
                    workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                    invalid_workdir = workdir_parent_link / "child_workdir"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "resume",
                        bad_run_id,
                        "--workdir",
                        str(invalid_workdir),
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=case_root,
                )
                output = proc.stdout + proc.stderr
                context = f"{run_id_mode}-{workdir_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid run_id" in output, context
                assert "Invalid workdir" not in output, context
                assert "Invalid home" not in output, context
                assert "Run not found or broken" not in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "run_id: [bold]" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                assert default_home.exists(), context
                assert not (default_home / "runs").exists(), context
                assert sorted(path.name for path in default_home.iterdir()) == [
                    "keep.txt",
                    "keep_dir",
                ], context
                assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
                assert sentinel_dir.is_dir(), context
                assert nested_file.read_text(encoding="utf-8") == "nested\n", context

                for file_path in side_effect_files:
                    assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
                if workdir_mode == "dangling_symlink":
                    assert not (case_root / "missing_workdir_target").exists(), context
                if workdir_mode == "symlink_ancestor":
                    assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_with_runs_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    run_id_modes = ("path_escape", "too_long")
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for run_id_mode in run_id_modes:
            bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
            for workdir_mode in workdir_modes:
                case_root = (
                    tmp_path / "resume_invalid_run_id_vs_workdir_existing_home_with_runs_"
                    f"{run_id_mode}_{workdir_mode}_{order_label}"
                )
                case_root.mkdir()
                home = case_root / ".orch_cli"
                home.mkdir()
                sentinel_file = home / "keep.txt"
                sentinel_file.write_text("keep\n", encoding="utf-8")
                sentinel_dir = home / "keep_dir"
                sentinel_dir.mkdir()
                nested_file = sentinel_dir / "nested.txt"
                nested_file.write_text("nested\n", encoding="utf-8")
                existing_run = home / "runs" / "keep_run"
                existing_run.mkdir(parents=True)
                plan_file = existing_run / "plan.yaml"
                plan_file.write_text("tasks: []\n", encoding="utf-8")

                side_effect_files: list[Path] = []
                if workdir_mode == "missing_path":
                    invalid_workdir = case_root / "missing_workdir"
                elif workdir_mode == "file_path":
                    invalid_workdir = case_root / "workdir_file"
                    invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                    side_effect_files.append(invalid_workdir)
                elif workdir_mode == "file_ancestor":
                    workdir_parent_file = case_root / "workdir_parent_file"
                    workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = workdir_parent_file / "child_workdir"
                    side_effect_files.append(workdir_parent_file)
                elif workdir_mode == "symlink_to_file":
                    workdir_target_file = case_root / "workdir_target_file"
                    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = case_root / "workdir_symlink_to_file"
                    invalid_workdir.symlink_to(workdir_target_file)
                    side_effect_files.append(workdir_target_file)
                elif workdir_mode == "dangling_symlink":
                    invalid_workdir = case_root / "workdir_dangling_symlink"
                    invalid_workdir.symlink_to(
                        case_root / "missing_workdir_target", target_is_directory=True
                    )
                else:
                    real_workdir_parent = case_root / "real_workdir_parent"
                    real_workdir_parent.mkdir()
                    workdir_parent_link = case_root / "workdir_parent_link"
                    workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                    invalid_workdir = workdir_parent_link / "child_workdir"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "resume",
                        bad_run_id,
                        "--home",
                        str(home),
                        "--workdir",
                        str(invalid_workdir),
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                context = f"{run_id_mode}-{workdir_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid run_id" in output, context
                assert "Invalid workdir" not in output, context
                assert "Invalid home" not in output, context
                assert "Run not found or broken" not in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "run_id: [bold]" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                assert home.exists(), context
                assert sorted(path.name for path in home.iterdir()) == [
                    "keep.txt",
                    "keep_dir",
                    "runs",
                ], context
                assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], (
                    context
                )
                assert sorted(path.name for path in existing_run.iterdir()) == ["plan.yaml"], (
                    context
                )
                assert not (existing_run / ".lock").exists(), context
                assert not (existing_run / "cancel.request").exists(), context
                assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
                assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
                assert sentinel_dir.is_dir(), context
                assert nested_file.read_text(encoding="utf-8") == "nested\n", context

                for file_path in side_effect_files:
                    assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
                if workdir_mode == "dangling_symlink":
                    assert not (case_root / "missing_workdir_target").exists(), context
                if workdir_mode == "symlink_ancestor":
                    assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_with_runs_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    run_id_modes = ("path_escape", "too_long")
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for run_id_mode in run_id_modes:
            bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
            for workdir_mode in workdir_modes:
                case_root = (
                    tmp_path / "resume_default_home_invalid_run_id_vs_workdir_with_runs_"
                    f"{run_id_mode}_{workdir_mode}_{order_label}"
                )
                case_root.mkdir()
                default_home = case_root / ".orch"
                default_home.mkdir()
                sentinel_file = default_home / "keep.txt"
                sentinel_file.write_text("keep\n", encoding="utf-8")
                sentinel_dir = default_home / "keep_dir"
                sentinel_dir.mkdir()
                nested_file = sentinel_dir / "nested.txt"
                nested_file.write_text("nested\n", encoding="utf-8")
                existing_run = default_home / "runs" / "keep_run"
                existing_run.mkdir(parents=True)
                plan_file = existing_run / "plan.yaml"
                plan_file.write_text("tasks: []\n", encoding="utf-8")

                side_effect_files: list[Path] = []
                if workdir_mode == "missing_path":
                    invalid_workdir = case_root / "missing_workdir"
                elif workdir_mode == "file_path":
                    invalid_workdir = case_root / "workdir_file"
                    invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                    side_effect_files.append(invalid_workdir)
                elif workdir_mode == "file_ancestor":
                    workdir_parent_file = case_root / "workdir_parent_file"
                    workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = workdir_parent_file / "child_workdir"
                    side_effect_files.append(workdir_parent_file)
                elif workdir_mode == "symlink_to_file":
                    workdir_target_file = case_root / "workdir_target_file"
                    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = case_root / "workdir_symlink_to_file"
                    invalid_workdir.symlink_to(workdir_target_file)
                    side_effect_files.append(workdir_target_file)
                elif workdir_mode == "dangling_symlink":
                    invalid_workdir = case_root / "workdir_dangling_symlink"
                    invalid_workdir.symlink_to(
                        case_root / "missing_workdir_target", target_is_directory=True
                    )
                else:
                    real_workdir_parent = case_root / "real_workdir_parent"
                    real_workdir_parent.mkdir()
                    workdir_parent_link = case_root / "workdir_parent_link"
                    workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                    invalid_workdir = workdir_parent_link / "child_workdir"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "resume",
                        bad_run_id,
                        "--workdir",
                        str(invalid_workdir),
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=case_root,
                )
                output = proc.stdout + proc.stderr
                context = f"{run_id_mode}-{workdir_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid run_id" in output, context
                assert "Invalid workdir" not in output, context
                assert "Invalid home" not in output, context
                assert "Run not found or broken" not in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "run_id: [bold]" not in output, context
                assert "state:" not in output, context
                assert "report:" not in output, context
                assert default_home.exists(), context
                assert sorted(path.name for path in default_home.iterdir()) == [
                    "keep.txt",
                    "keep_dir",
                    "runs",
                ], context
                assert sorted(path.name for path in (default_home / "runs").iterdir()) == [
                    "keep_run"
                ], context
                assert sorted(path.name for path in existing_run.iterdir()) == ["plan.yaml"], (
                    context
                )
                assert not (existing_run / ".lock").exists(), context
                assert not (existing_run / "cancel.request").exists(), context
                assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
                assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
                assert sentinel_dir.is_dir(), context
                assert nested_file.read_text(encoding="utf-8") == "nested\n", context

                for file_path in side_effect_files:
                    assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
                if workdir_mode == "dangling_symlink":
                    assert not (case_root / "missing_workdir_target").exists(), context
                if workdir_mode == "symlink_ancestor":
                    assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_resume_invalid_run_id_invalid_workdir_existing_home_run_artifacts_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    run_id_modes = ("path_escape", "too_long")
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for run_id_mode in run_id_modes:
            bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
            for workdir_mode in workdir_modes:
                case_root = (
                    tmp_path / "resume_invalid_run_id_vs_workdir_existing_home_run_artifacts_"
                    f"{run_id_mode}_{workdir_mode}_{order_label}"
                )
                case_root.mkdir()
                home = case_root / ".orch_cli"
                home.mkdir()
                sentinel_file = home / "keep.txt"
                sentinel_file.write_text("keep\n", encoding="utf-8")
                sentinel_dir = home / "keep_dir"
                sentinel_dir.mkdir()
                nested_file = sentinel_dir / "nested.txt"
                nested_file.write_text("nested\n", encoding="utf-8")
                existing_run = home / "runs" / "keep_run"
                existing_run.mkdir(parents=True)
                plan_file = existing_run / "plan.yaml"
                plan_file.write_text("tasks: []\n", encoding="utf-8")
                lock_file = existing_run / ".lock"
                lock_file.write_text("lock\n", encoding="utf-8")
                cancel_request = existing_run / "cancel.request"
                cancel_request.write_text("cancel\n", encoding="utf-8")
                run_log = existing_run / "task.log"
                run_log.write_text("log\n", encoding="utf-8")

                side_effect_files: list[Path] = []
                if workdir_mode == "missing_path":
                    invalid_workdir = case_root / "missing_workdir"
                elif workdir_mode == "file_path":
                    invalid_workdir = case_root / "workdir_file"
                    invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                    side_effect_files.append(invalid_workdir)
                elif workdir_mode == "file_ancestor":
                    workdir_parent_file = case_root / "workdir_parent_file"
                    workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = workdir_parent_file / "child_workdir"
                    side_effect_files.append(workdir_parent_file)
                elif workdir_mode == "symlink_to_file":
                    workdir_target_file = case_root / "workdir_target_file"
                    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = case_root / "workdir_symlink_to_file"
                    invalid_workdir.symlink_to(workdir_target_file)
                    side_effect_files.append(workdir_target_file)
                elif workdir_mode == "dangling_symlink":
                    invalid_workdir = case_root / "workdir_dangling_symlink"
                    invalid_workdir.symlink_to(
                        case_root / "missing_workdir_target", target_is_directory=True
                    )
                else:
                    real_workdir_parent = case_root / "real_workdir_parent"
                    real_workdir_parent.mkdir()
                    workdir_parent_link = case_root / "workdir_parent_link"
                    workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                    invalid_workdir = workdir_parent_link / "child_workdir"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "resume",
                        bad_run_id,
                        "--home",
                        str(home),
                        "--workdir",
                        str(invalid_workdir),
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout + proc.stderr
                context = f"{run_id_mode}-{workdir_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid run_id" in output, context
                assert "Invalid workdir" not in output, context
                assert "Invalid home" not in output, context
                assert "Run not found or broken" not in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "run_id: [bold]" not in output, context
                assert "state: [bold]" not in output, context
                assert "report: [bold]" not in output, context
                assert home.exists(), context
                assert sorted(path.name for path in home.iterdir()) == [
                    "keep.txt",
                    "keep_dir",
                    "runs",
                ], context
                assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], (
                    context
                )
                assert sorted(path.name for path in existing_run.iterdir()) == [
                    ".lock",
                    "cancel.request",
                    "plan.yaml",
                    "task.log",
                ], context
                assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
                assert lock_file.read_text(encoding="utf-8") == "lock\n", context
                assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
                assert run_log.read_text(encoding="utf-8") == "log\n", context
                assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
                assert sentinel_dir.is_dir(), context
                assert nested_file.read_text(encoding="utf-8") == "nested\n", context

                for file_path in side_effect_files:
                    assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
                if workdir_mode == "dangling_symlink":
                    assert not (case_root / "missing_workdir_target").exists(), context
                if workdir_mode == "symlink_ancestor":
                    assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_resume_default_home_invalid_run_id_invalid_workdir_run_artifacts_matrix(
    tmp_path: Path,
) -> None:
    flag_orders: list[list[str]] = [
        ["--fail-fast", "--no-fail-fast"],
        ["--no-fail-fast", "--fail-fast"],
    ]
    run_id_modes = ("path_escape", "too_long")
    workdir_modes = (
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    )

    for order in flag_orders:
        order_label = "forward" if order[0] == "--fail-fast" else "reverse"
        for run_id_mode in run_id_modes:
            bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
            for workdir_mode in workdir_modes:
                case_root = (
                    tmp_path / "resume_default_home_invalid_run_id_vs_workdir_run_artifacts_"
                    f"{run_id_mode}_{workdir_mode}_{order_label}"
                )
                case_root.mkdir()
                default_home = case_root / ".orch"
                default_home.mkdir()
                sentinel_file = default_home / "keep.txt"
                sentinel_file.write_text("keep\n", encoding="utf-8")
                sentinel_dir = default_home / "keep_dir"
                sentinel_dir.mkdir()
                nested_file = sentinel_dir / "nested.txt"
                nested_file.write_text("nested\n", encoding="utf-8")
                existing_run = default_home / "runs" / "keep_run"
                existing_run.mkdir(parents=True)
                plan_file = existing_run / "plan.yaml"
                plan_file.write_text("tasks: []\n", encoding="utf-8")
                lock_file = existing_run / ".lock"
                lock_file.write_text("lock\n", encoding="utf-8")
                cancel_request = existing_run / "cancel.request"
                cancel_request.write_text("cancel\n", encoding="utf-8")
                run_log = existing_run / "task.log"
                run_log.write_text("log\n", encoding="utf-8")

                side_effect_files: list[Path] = []
                if workdir_mode == "missing_path":
                    invalid_workdir = case_root / "missing_workdir"
                elif workdir_mode == "file_path":
                    invalid_workdir = case_root / "workdir_file"
                    invalid_workdir.write_text("not a directory\n", encoding="utf-8")
                    side_effect_files.append(invalid_workdir)
                elif workdir_mode == "file_ancestor":
                    workdir_parent_file = case_root / "workdir_parent_file"
                    workdir_parent_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = workdir_parent_file / "child_workdir"
                    side_effect_files.append(workdir_parent_file)
                elif workdir_mode == "symlink_to_file":
                    workdir_target_file = case_root / "workdir_target_file"
                    workdir_target_file.write_text("not a directory\n", encoding="utf-8")
                    invalid_workdir = case_root / "workdir_symlink_to_file"
                    invalid_workdir.symlink_to(workdir_target_file)
                    side_effect_files.append(workdir_target_file)
                elif workdir_mode == "dangling_symlink":
                    invalid_workdir = case_root / "workdir_dangling_symlink"
                    invalid_workdir.symlink_to(
                        case_root / "missing_workdir_target", target_is_directory=True
                    )
                else:
                    real_workdir_parent = case_root / "real_workdir_parent"
                    real_workdir_parent.mkdir()
                    workdir_parent_link = case_root / "workdir_parent_link"
                    workdir_parent_link.symlink_to(real_workdir_parent, target_is_directory=True)
                    invalid_workdir = workdir_parent_link / "child_workdir"

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "orch.cli",
                        "resume",
                        bad_run_id,
                        "--workdir",
                        str(invalid_workdir),
                        *order,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=case_root,
                )
                output = proc.stdout + proc.stderr
                context = f"{run_id_mode}-{workdir_mode}-{order_label}"
                assert proc.returncode == 2, context
                assert "Invalid run_id" in output, context
                assert "Invalid workdir" not in output, context
                assert "Invalid home" not in output, context
                assert "Run not found or broken" not in output, context
                assert "Plan validation error" not in output, context
                assert "contains symlink component" not in output, context
                assert "run_id: [bold]" not in output, context
                assert "state: [bold]" not in output, context
                assert "report: [bold]" not in output, context
                assert default_home.exists(), context
                assert sorted(path.name for path in default_home.iterdir()) == [
                    "keep.txt",
                    "keep_dir",
                    "runs",
                ], context
                assert sorted(path.name for path in (default_home / "runs").iterdir()) == [
                    "keep_run"
                ], context
                assert sorted(path.name for path in existing_run.iterdir()) == [
                    ".lock",
                    "cancel.request",
                    "plan.yaml",
                    "task.log",
                ], context
                assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
                assert lock_file.read_text(encoding="utf-8") == "lock\n", context
                assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
                assert run_log.read_text(encoding="utf-8") == "log\n", context
                assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
                assert sentinel_dir.is_dir(), context
                assert nested_file.read_text(encoding="utf-8") == "nested\n", context

                for file_path in side_effect_files:
                    assert file_path.read_text(encoding="utf-8") == "not a directory\n", context
                if workdir_mode == "dangling_symlink":
                    assert not (case_root / "missing_workdir_target").exists(), context
                if workdir_mode == "symlink_ancestor":
                    assert not (real_workdir_parent / "child_workdir").exists(), context


def test_cli_cancel_rejects_absolute_run_id_without_side_effect(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    outside_run_dir = tmp_path / "outside_run"
    outside_run_dir.mkdir()
    (outside_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "cancel",
            str(outside_run_dir),
            "--home",
            str(home),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "contains symlink component" not in output
    assert not (outside_run_dir / "cancel.request").exists()


def test_cli_cancel_invalid_run_id_existing_home_preserves_entries_matrix(
    tmp_path: Path,
) -> None:
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        case_root = tmp_path / f"cancel_invalid_run_id_existing_home_{run_id_mode}"
        case_root.mkdir()
        home = case_root / ".orch_cli"
        home.mkdir()
        sentinel_file = home / "keep.txt"
        sentinel_file.write_text("keep\n", encoding="utf-8")
        sentinel_dir = home / "keep_dir"
        sentinel_dir.mkdir()
        nested_file = sentinel_dir / "nested.txt"
        nested_file.write_text("nested\n", encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "orch.cli",
                "cancel",
                bad_run_id,
                "--home",
                str(home),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        context = run_id_mode
        assert proc.returncode == 2, context
        assert "Invalid run_id" in output, context
        assert "Invalid home" not in output, context
        assert "Run not found or broken" not in output, context
        assert "Plan validation error" not in output, context
        assert "contains symlink component" not in output, context
        assert "Cancel request written" not in output, context
        assert "run_id: [bold]" not in output, context
        assert "state: [bold]" not in output, context
        assert "report: [bold]" not in output, context
        assert home.exists(), context
        assert not (home / "runs").exists(), context
        assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir"], context
        assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
        assert sentinel_dir.is_dir(), context
        assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_cancel_invalid_run_id_default_home_preserves_entries_matrix(
    tmp_path: Path,
) -> None:
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        case_root = tmp_path / f"cancel_invalid_run_id_default_home_{run_id_mode}"
        case_root.mkdir()
        default_home = case_root / ".orch"
        default_home.mkdir()
        sentinel_file = default_home / "keep.txt"
        sentinel_file.write_text("keep\n", encoding="utf-8")
        sentinel_dir = default_home / "keep_dir"
        sentinel_dir.mkdir()
        nested_file = sentinel_dir / "nested.txt"
        nested_file.write_text("nested\n", encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "orch.cli",
                "cancel",
                bad_run_id,
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=case_root,
        )
        output = proc.stdout + proc.stderr
        context = run_id_mode
        assert proc.returncode == 2, context
        assert "Invalid run_id" in output, context
        assert "Invalid home" not in output, context
        assert "Run not found or broken" not in output, context
        assert "Plan validation error" not in output, context
        assert "contains symlink component" not in output, context
        assert "Cancel request written" not in output, context
        assert "run_id: [bold]" not in output, context
        assert "state: [bold]" not in output, context
        assert "report: [bold]" not in output, context
        assert default_home.exists(), context
        assert not (default_home / "runs").exists(), context
        assert sorted(path.name for path in default_home.iterdir()) == [
            "keep.txt",
            "keep_dir",
        ], context
        assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
        assert sentinel_dir.is_dir(), context
        assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_cancel_invalid_run_id_existing_home_with_runs_preserves_entries_matrix(
    tmp_path: Path,
) -> None:
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        case_root = tmp_path / f"cancel_invalid_run_id_existing_home_with_runs_{run_id_mode}"
        case_root.mkdir()
        home = case_root / ".orch_cli"
        home.mkdir()
        sentinel_file = home / "keep.txt"
        sentinel_file.write_text("keep\n", encoding="utf-8")
        sentinel_dir = home / "keep_dir"
        sentinel_dir.mkdir()
        nested_file = sentinel_dir / "nested.txt"
        nested_file.write_text("nested\n", encoding="utf-8")
        existing_run = home / "runs" / "keep_run"
        existing_run.mkdir(parents=True)
        (existing_run / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "orch.cli",
                "cancel",
                bad_run_id,
                "--home",
                str(home),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        context = run_id_mode
        assert proc.returncode == 2, context
        assert "Invalid run_id" in output, context
        assert "Invalid home" not in output, context
        assert "Run not found or broken" not in output, context
        assert "Plan validation error" not in output, context
        assert "contains symlink component" not in output, context
        assert "Cancel request written" not in output, context
        assert "run_id: [bold]" not in output, context
        assert "state: [bold]" not in output, context
        assert "report: [bold]" not in output, context
        assert home.exists(), context
        assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir", "runs"], (
            context
        )
        assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], context
        assert not (existing_run / "cancel.request").exists(), context
        assert not (existing_run / ".lock").exists(), context
        assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
        assert sentinel_dir.is_dir(), context
        assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_cancel_invalid_run_id_default_home_with_runs_preserves_entries_matrix(
    tmp_path: Path,
) -> None:
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        case_root = tmp_path / f"cancel_invalid_run_id_default_home_with_runs_{run_id_mode}"
        case_root.mkdir()
        default_home = case_root / ".orch"
        default_home.mkdir()
        sentinel_file = default_home / "keep.txt"
        sentinel_file.write_text("keep\n", encoding="utf-8")
        sentinel_dir = default_home / "keep_dir"
        sentinel_dir.mkdir()
        nested_file = sentinel_dir / "nested.txt"
        nested_file.write_text("nested\n", encoding="utf-8")
        existing_run = default_home / "runs" / "keep_run"
        existing_run.mkdir(parents=True)
        (existing_run / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "orch.cli",
                "cancel",
                bad_run_id,
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=case_root,
        )
        output = proc.stdout + proc.stderr
        context = run_id_mode
        assert proc.returncode == 2, context
        assert "Invalid run_id" in output, context
        assert "Invalid home" not in output, context
        assert "Run not found or broken" not in output, context
        assert "Plan validation error" not in output, context
        assert "contains symlink component" not in output, context
        assert "Cancel request written" not in output, context
        assert "run_id: [bold]" not in output, context
        assert "state: [bold]" not in output, context
        assert "report: [bold]" not in output, context
        assert default_home.exists(), context
        assert sorted(path.name for path in default_home.iterdir()) == [
            "keep.txt",
            "keep_dir",
            "runs",
        ], context
        assert sorted(path.name for path in (default_home / "runs").iterdir()) == ["keep_run"], (
            context
        )
        assert not (existing_run / "cancel.request").exists(), context
        assert not (existing_run / ".lock").exists(), context
        assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
        assert sentinel_dir.is_dir(), context
        assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_cancel_invalid_run_id_existing_home_run_artifacts_preserved_matrix(
    tmp_path: Path,
) -> None:
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        case_root = tmp_path / f"cancel_invalid_run_id_existing_home_run_artifacts_{run_id_mode}"
        case_root.mkdir()
        home = case_root / ".orch_cli"
        home.mkdir()
        sentinel_file = home / "keep.txt"
        sentinel_file.write_text("keep\n", encoding="utf-8")
        sentinel_dir = home / "keep_dir"
        sentinel_dir.mkdir()
        nested_file = sentinel_dir / "nested.txt"
        nested_file.write_text("nested\n", encoding="utf-8")
        existing_run = home / "runs" / "keep_run"
        existing_run.mkdir(parents=True)
        plan_file = existing_run / "plan.yaml"
        plan_file.write_text("tasks: []\n", encoding="utf-8")
        lock_file = existing_run / ".lock"
        lock_file.write_text("lock\n", encoding="utf-8")
        cancel_request = existing_run / "cancel.request"
        cancel_request.write_text("cancel\n", encoding="utf-8")
        run_log = existing_run / "task.log"
        run_log.write_text("log\n", encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "orch.cli",
                "cancel",
                bad_run_id,
                "--home",
                str(home),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        context = run_id_mode
        assert proc.returncode == 2, context
        assert "Invalid run_id" in output, context
        assert "Invalid home" not in output, context
        assert "Run not found or broken" not in output, context
        assert "Plan validation error" not in output, context
        assert "contains symlink component" not in output, context
        assert "Cancel request written" not in output, context
        assert "run_id: [bold]" not in output, context
        assert "state: [bold]" not in output, context
        assert "report: [bold]" not in output, context
        assert home.exists(), context
        assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir", "runs"], (
            context
        )
        assert sorted(path.name for path in (home / "runs").iterdir()) == ["keep_run"], context
        assert sorted(path.name for path in existing_run.iterdir()) == [
            ".lock",
            "cancel.request",
            "plan.yaml",
            "task.log",
        ], context
        assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
        assert lock_file.read_text(encoding="utf-8") == "lock\n", context
        assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
        assert run_log.read_text(encoding="utf-8") == "log\n", context
        assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
        assert sentinel_dir.is_dir(), context
        assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_cancel_invalid_run_id_default_home_run_artifacts_preserved_matrix(
    tmp_path: Path,
) -> None:
    run_id_modes = ("path_escape", "too_long")

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        case_root = tmp_path / f"cancel_invalid_run_id_default_home_run_artifacts_{run_id_mode}"
        case_root.mkdir()
        default_home = case_root / ".orch"
        default_home.mkdir()
        sentinel_file = default_home / "keep.txt"
        sentinel_file.write_text("keep\n", encoding="utf-8")
        sentinel_dir = default_home / "keep_dir"
        sentinel_dir.mkdir()
        nested_file = sentinel_dir / "nested.txt"
        nested_file.write_text("nested\n", encoding="utf-8")
        existing_run = default_home / "runs" / "keep_run"
        existing_run.mkdir(parents=True)
        plan_file = existing_run / "plan.yaml"
        plan_file.write_text("tasks: []\n", encoding="utf-8")
        lock_file = existing_run / ".lock"
        lock_file.write_text("lock\n", encoding="utf-8")
        cancel_request = existing_run / "cancel.request"
        cancel_request.write_text("cancel\n", encoding="utf-8")
        run_log = existing_run / "task.log"
        run_log.write_text("log\n", encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "orch.cli",
                "cancel",
                bad_run_id,
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=case_root,
        )
        output = proc.stdout + proc.stderr
        context = run_id_mode
        assert proc.returncode == 2, context
        assert "Invalid run_id" in output, context
        assert "Invalid home" not in output, context
        assert "Run not found or broken" not in output, context
        assert "Plan validation error" not in output, context
        assert "contains symlink component" not in output, context
        assert "Cancel request written" not in output, context
        assert "run_id: [bold]" not in output, context
        assert "state: [bold]" not in output, context
        assert "report: [bold]" not in output, context
        assert default_home.exists(), context
        assert sorted(path.name for path in default_home.iterdir()) == [
            "keep.txt",
            "keep_dir",
            "runs",
        ], context
        assert sorted(path.name for path in (default_home / "runs").iterdir()) == ["keep_run"], (
            context
        )
        assert sorted(path.name for path in existing_run.iterdir()) == [
            ".lock",
            "cancel.request",
            "plan.yaml",
            "task.log",
        ], context
        assert plan_file.read_text(encoding="utf-8") == "tasks: []\n", context
        assert lock_file.read_text(encoding="utf-8") == "lock\n", context
        assert cancel_request.read_text(encoding="utf-8") == "cancel\n", context
        assert run_log.read_text(encoding="utf-8") == "log\n", context
        assert sentinel_file.read_text(encoding="utf-8") == "keep\n", context
        assert sentinel_dir.is_dir(), context
        assert nested_file.read_text(encoding="utf-8") == "nested\n", context


def test_cli_cancel_invalid_run_id_takes_precedence_over_invalid_home(tmp_path: Path) -> None:
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    bad_run_id = "../escape"

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(home_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_cancel_too_long_run_id_takes_precedence_over_invalid_home(tmp_path: Path) -> None:
    home_file = tmp_path / "home_file"
    home_file.write_text("not a dir\n", encoding="utf-8")
    bad_run_id = "a" * 129

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(home_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert home_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_cancel_invalid_run_id_takes_precedence_over_symlink_home(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    bad_run_id = "../escape"

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(home_link)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert not (real_run_dir / "cancel.request").exists()


def test_cli_cancel_invalid_run_id_takes_precedence_over_home_file_ancestor(
    tmp_path: Path,
) -> None:
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    bad_run_id = "../escape"

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(nested_home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_cancel_invalid_run_id_takes_precedence_over_home_symlink_to_file(
    tmp_path: Path,
) -> None:
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    bad_run_id = "../escape"

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(home_link)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_cancel_too_long_run_id_takes_precedence_over_symlink_home(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(real_home, target_is_directory=True)
    bad_run_id = "a" * 129

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(home_link)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert not (real_run_dir / "cancel.request").exists()


def test_cli_cancel_invalid_run_id_takes_precedence_over_dangling_symlink_home(
    tmp_path: Path,
) -> None:
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)
    bad_run_id = "../escape"

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(home_link)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert not missing_home_target.exists()


def test_cli_cancel_too_long_run_id_takes_precedence_over_dangling_symlink_home(
    tmp_path: Path,
) -> None:
    missing_home_target = tmp_path / "missing_home_target"
    home_link = tmp_path / "home_link"
    home_link.symlink_to(missing_home_target, target_is_directory=True)
    bad_run_id = "a" * 129

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(home_link)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert not missing_home_target.exists()


def test_cli_cancel_too_long_run_id_takes_precedence_over_home_symlink_to_file(
    tmp_path: Path,
) -> None:
    home_target_file = tmp_path / "home_target_file.txt"
    home_target_file.write_text("not a home dir\n", encoding="utf-8")
    home_link = tmp_path / "home_link"
    home_link.symlink_to(home_target_file)
    bad_run_id = "a" * 129

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(home_link)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n"


def test_cli_cancel_too_long_run_id_takes_precedence_over_home_file_ancestor(
    tmp_path: Path,
) -> None:
    home_parent_file = tmp_path / "home_parent_file"
    home_parent_file.write_text("not a dir\n", encoding="utf-8")
    nested_home = home_parent_file / "orch_home"
    bad_run_id = "a" * 129

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(nested_home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n"


def test_cli_cancel_invalid_run_id_takes_precedence_over_home_symlink_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / "orch_home"
    bad_run_id = "../escape"

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(nested_home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert not (real_parent / "orch_home" / "runs").exists()


def test_cli_cancel_invalid_run_id_takes_precedence_over_home_symlink_ancestor_directory(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home_name = "orch_home"
    nested_home = symlink_parent / nested_home_name
    run_id = "run1"
    real_run_dir = real_parent / nested_home_name / "runs" / run_id
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    bad_run_id = "../escape"

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(nested_home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert not (real_run_dir / "cancel.request").exists()
    assert not (real_run_dir / ".lock").exists()


def test_cli_cancel_too_long_run_id_takes_precedence_over_home_symlink_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home = symlink_parent / "orch_home"
    bad_run_id = "a" * 129

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(nested_home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert not (real_parent / "orch_home" / "runs").exists()


def test_cli_cancel_too_long_run_id_takes_precedence_over_home_symlink_ancestor_directory(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "home_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    nested_home_name = "orch_home"
    nested_home = symlink_parent / nested_home_name
    run_id = "run1"
    real_run_dir = real_parent / nested_home_name / "runs" / run_id
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    bad_run_id = "a" * 129

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "cancel", bad_run_id, "--home", str(nested_home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Invalid run_id" in output
    assert "Invalid home" not in output
    assert "contains symlink component" not in output
    assert not (real_run_dir / "cancel.request").exists()
    assert not (real_run_dir / ".lock").exists()


def test_cli_cancel_invalid_run_id_precedence_invalid_home_shape_matrix(
    tmp_path: Path,
) -> None:
    run_id_modes = ("path_escape", "too_long")
    home_modes = (
        "file_path",
        "symlink",
        "dangling_symlink",
        "symlink_to_file",
        "file_ancestor",
        "symlink_ancestor",
        "symlink_ancestor_directory",
    )

    for run_id_mode in run_id_modes:
        bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129
        for home_mode in home_modes:
            case_root = tmp_path / f"cancel_run_id_precedence_{run_id_mode}_{home_mode}"
            case_root.mkdir()
            cmd = [sys.executable, "-m", "orch.cli", "cancel", bad_run_id]

            if home_mode == "file_path":
                home_file = case_root / "home_file"
                home_file.write_text("not a dir\n", encoding="utf-8")
                cmd += ["--home", str(home_file)]
            elif home_mode == "symlink":
                real_home = case_root / "real_home"
                real_run_dir = real_home / "runs" / "run1"
                real_run_dir.mkdir(parents=True)
                (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
                home_link = case_root / "home_link"
                home_link.symlink_to(real_home, target_is_directory=True)
                cmd += ["--home", str(home_link)]
            elif home_mode == "dangling_symlink":
                missing_home_target = case_root / "missing_home_target"
                home_link = case_root / "home_link"
                home_link.symlink_to(missing_home_target, target_is_directory=True)
                cmd += ["--home", str(home_link)]
            elif home_mode == "symlink_to_file":
                home_target_file = case_root / "home_target_file.txt"
                home_target_file.write_text("not a home dir\n", encoding="utf-8")
                home_link = case_root / "home_link"
                home_link.symlink_to(home_target_file)
                cmd += ["--home", str(home_link)]
            elif home_mode == "file_ancestor":
                home_parent_file = case_root / "home_parent_file"
                home_parent_file.write_text("not a dir\n", encoding="utf-8")
                nested_home = home_parent_file / "orch_home"
                cmd += ["--home", str(nested_home)]
            elif home_mode == "symlink_ancestor":
                real_parent = case_root / "real_parent"
                real_parent.mkdir()
                symlink_parent = case_root / "home_parent_link"
                symlink_parent.symlink_to(real_parent, target_is_directory=True)
                nested_home = symlink_parent / "orch_home"
                cmd += ["--home", str(nested_home)]
            else:
                assert home_mode == "symlink_ancestor_directory"
                real_parent = case_root / "real_parent"
                real_parent.mkdir()
                symlink_parent = case_root / "home_parent_link"
                symlink_parent.symlink_to(real_parent, target_is_directory=True)
                nested_home_name = "orch_home"
                nested_home = symlink_parent / nested_home_name
                real_run_dir = real_parent / nested_home_name / "runs" / "run1"
                real_run_dir.mkdir(parents=True)
                (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
                cmd += ["--home", str(nested_home)]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            output = proc.stdout + proc.stderr
            context = f"{run_id_mode}-{home_mode}"
            assert proc.returncode == 2, context
            assert "Invalid run_id" in output, context
            assert "Invalid home" not in output, context
            assert "contains symlink component" not in output, context
            assert "run_id: [bold]" not in output, context
            assert "state: [bold]" not in output, context
            assert "report: [bold]" not in output, context

            if home_mode == "file_path":
                assert home_file.read_text(encoding="utf-8") == "not a dir\n", context
            elif home_mode == "symlink":
                assert not (real_run_dir / "cancel.request").exists(), context
                assert not (real_run_dir / ".lock").exists(), context
            elif home_mode == "dangling_symlink":
                assert not missing_home_target.exists(), context
            elif home_mode == "symlink_to_file":
                assert home_target_file.read_text(encoding="utf-8") == "not a home dir\n", context
            elif home_mode == "file_ancestor":
                assert home_parent_file.read_text(encoding="utf-8") == "not a dir\n", context
            elif home_mode == "symlink_ancestor":
                assert "contains symlink component" not in output, context
                assert not (real_parent / "orch_home" / "runs").exists(), context
            else:
                assert home_mode == "symlink_ancestor_directory"
                assert "contains symlink component" not in output, context
                assert not (real_run_dir / "cancel.request").exists(), context
                assert not (real_run_dir / ".lock").exists(), context


def test_cli_rejects_too_long_run_id_for_all_commands(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    bad_run_id = "a" * 129
    for command in ("status", "logs", "resume", "cancel"):
        proc = subprocess.run(
            [sys.executable, "-m", "orch.cli", command, bad_run_id, "--home", str(home)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 2, command
        assert "Invalid run_id" in output, command
        assert "contains symlink component" not in output, command
    assert not (home / "runs").exists()


def test_cli_logs_unknown_task_returns_two(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_logs_task.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)
    logs_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "logs",
            run_id,
            "--home",
            str(home),
            "--task",
            "missing_task",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert logs_proc.returncode == 2
    assert "unknown task" in logs_proc.stdout


def test_cli_fail_fast_skips_unstarted_ready_tasks(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_fail_fast.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: fail_first
            cmd: ["python3", "-c", "import sys; sys.exit(1)"]
          - id: independent
            cmd: ["python3", "-c", "print('independent')"]
          - id: dependent
            cmd: ["python3", "-c", "print('dependent')"]
            depends_on: ["fail_first"]
        """,
    )

    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
            "--fail-fast",
            "--max-parallel",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 3
    run_id = _extract_run_id(run_proc.stdout)
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["tasks"]["fail_first"]["status"] == "FAILED"
    assert state["tasks"]["independent"]["status"] == "SKIPPED"
    assert state["tasks"]["independent"]["skip_reason"] == "fail_fast"
    assert state["tasks"]["dependent"]["status"] == "SKIPPED"


def test_cli_no_fail_fast_runs_independent_tasks(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_no_fail_fast.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: fail_first
            cmd: ["python3", "-c", "import sys; sys.exit(1)"]
          - id: independent
            cmd: ["python3", "-c", "print('independent')"]
          - id: dependent
            cmd: ["python3", "-c", "print('dependent')"]
            depends_on: ["fail_first"]
        """,
    )

    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
            "--no-fail-fast",
            "--max-parallel",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 3
    run_id = _extract_run_id(run_proc.stdout)
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["tasks"]["fail_first"]["status"] == "FAILED"
    assert state["tasks"]["independent"]["status"] == "SUCCESS"
    assert state["tasks"]["dependent"]["status"] == "SKIPPED"
    assert state["tasks"]["dependent"]["skip_reason"] == "dependency_not_success"


def test_cli_run_invalid_plan_returns_two_and_creates_no_run(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
            depends_on: ["missing_dep"]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output
    assert not (home / "runs").exists()


def test_cli_dry_run_invalid_plan_returns_two(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_invalid_dry.yaml"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
            depends_on: ["b"]
        """,
    )

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "run", str(plan_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_dry_run_rejects_plan_with_unknown_root_field(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_unknown_root_dry.yaml"
    _write_plan(
        plan_path,
        """
        goal: demo
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
        unexpected_root: true
        """,
    )

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "run", str(plan_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_dry_run_rejects_plan_with_unknown_task_field(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_unknown_task_dry.yaml"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
            unexpected_task_field: 1
        """,
    )

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "run", str(plan_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_dry_run_rejects_symlink_plan_path(tmp_path: Path) -> None:
    real_plan = tmp_path / "real_plan.yaml"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
        """,
    )
    symlink_plan = tmp_path / "plan_symlink.yaml"
    symlink_plan.symlink_to(real_plan)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "run", str(symlink_plan), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "invalid plan path" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_dry_run_rejects_plan_path_with_symlink_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    real_plan = real_parent / "plan.yaml"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
        """,
    )
    symlink_parent = tmp_path / "plan_parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "run", str(symlink_parent / "plan.yaml"), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "invalid plan path" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_dry_run_rejects_non_regular_plan_path(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        return

    plan_path = tmp_path / "plan_fifo.yaml"
    os.mkfifo(plan_path)
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "run", str(plan_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_dry_run_rejects_plan_with_case_insensitive_duplicate_outputs(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan_dup_outputs_case_dry.yaml"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
            outputs: ["dist/report.txt", "dist/REPORT.txt"]
        """,
    )

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "run", str(plan_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_dry_run_rejects_plan_with_backoff_longer_than_retries(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_backoff_too_long_dry.yaml"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: a
            cmd: ["python3", "-c", "print('a')"]
            retries: 1
            retry_backoff_sec: [0.1, 0.2]
        """,
    )

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "run", str(plan_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_timeout_with_retry_records_attempts_and_fails(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_timeout_retry.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: slow
            cmd: ["python3", "-c", "import time; time.sleep(1.0)"]
            timeout_sec: 0.1
            retries: 1
            retry_backoff_sec: [0.01]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 3
    run_id = _extract_run_id(proc.stdout)
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["tasks"]["slow"]["status"] == "FAILED"
    assert state["tasks"]["slow"]["attempts"] == 2
    assert state["tasks"]["slow"]["timed_out"] is True


def test_cli_run_missing_command_fails_without_retry_and_logs_reason(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_missing_cmd.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: badcmd
            cmd: ["__definitely_missing_command__", "--version"]
            retries: 3
            retry_backoff_sec: [0.01, 0.01, 0.01]
        """,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 3
    run_id = _extract_run_id(proc.stdout)
    state = json.loads((home / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    badcmd = state["tasks"]["badcmd"]
    assert badcmd["status"] == "FAILED"
    assert badcmd["attempts"] == 1
    assert badcmd["exit_code"] == 127
    assert badcmd["skip_reason"] == "process_start_failed"
    stderr_log = home / "runs" / run_id / badcmd["stderr_path"]
    assert "failed to start process" in stderr_log.read_text(encoding="utf-8")


def test_cli_status_json_outputs_valid_state_payload(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_status_json.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: only
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )

    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    status_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "status",
            run_id,
            "--home",
            str(home),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert status_proc.returncode == 0
    payload = json.loads(status_proc.stdout)
    assert payload["run_id"] == run_id
    assert payload["status"] == "SUCCESS"
    assert payload["tasks"]["only"]["status"] == "SUCCESS"


def test_cli_dry_run_cycle_plan_returns_two(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_cycle.yaml"
    _write_plan(
        plan_path,
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

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "run", str(plan_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_resume_missing_run_returns_two(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "resume", "missing_run", "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Run not found or broken" in output


def test_cli_resume_handles_non_directory_runs_path(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    home.mkdir(parents=True)
    (home / "runs").write_text("not a directory\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "resume", "20260101_000000_abcdef", "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Run not found or broken" in output


def test_cli_resume_invalid_plan_copy_returns_two(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    (run_dir / "plan.yaml").write_text("tasks: [", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "resume", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_resume_rejects_symlink_plan_file(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    real_plan = tmp_path / "real_plan.yaml"
    _write_plan(
        real_plan,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """,
    )
    (run_dir / "plan.yaml").symlink_to(real_plan)

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "resume", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Plan validation error" in output
    assert "invalid plan path" in output
    assert "contains symlink component" not in output
    assert "must not include symlink" not in output
    assert "must not be symlink" not in output


def test_cli_resume_rejects_state_with_invalid_attempts(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "RUNNING",
                "goal": None,
                "plan_relpath": "plan.yaml",
                "home": str(home),
                "workdir": str(tmp_path),
                "max_parallel": 1,
                "fail_fast": False,
                "tasks": {
                    "t1": {
                        "status": "FAILED",
                        "depends_on": [],
                        "cmd": ["python3", "-c", "print('ok')"],
                        "cwd": None,
                        "env": None,
                        "timeout_sec": None,
                        "retries": 0,
                        "retry_backoff_sec": [],
                        "outputs": [],
                        "attempts": -1,
                        "stdout_path": "logs/t1.out.log",
                        "stderr_path": "logs/t1.err.log",
                        "artifact_paths": ["artifacts/t1/out.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "plan.yaml").write_text(
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "resume", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Run not found or broken" in output


def test_cli_resume_state_missing_plan_tasks_returns_two(tmp_path: Path) -> None:
    home = tmp_path / ".orch_cli"
    run_id = "20260101_000000_abcdef"
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    (run_dir / "plan.yaml").write_text(
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('ok')"]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "orch.cli", "resume", run_id, "--home", str(home)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "Run not found or broken" in output


def test_cli_logs_tail_limits_output_lines(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_logs_tail.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: noisy
            cmd:
              [
                "python3",
                "-c",
                "print('line1');print('line2');print('line3');print('line4');print('line5')",
              ]
        """,
    )

    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)

    logs_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "logs",
            run_id,
            "--home",
            str(home),
            "--task",
            "noisy",
            "--tail",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert logs_proc.returncode == 0
    out = logs_proc.stdout
    assert "line5" in out
    assert "line4" in out
    assert "line1" not in out
    assert "line2" not in out


def test_cli_logs_falls_back_to_unlocked_read_when_lock_conflicted(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_logs_lock_fallback.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: t1
            cmd: ["python3", "-c", "print('from-log')"]
        """,
    )

    run_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "run",
            str(plan_path),
            "--home",
            str(home),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    run_id = _extract_run_id(run_proc.stdout)
    run_dir = home / "runs" / run_id
    (run_dir / ".lock").write_text("other-holder", encoding="utf-8")

    logs_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orch.cli",
            "logs",
            run_id,
            "--home",
            str(home),
            "--task",
            "t1",
            "--tail",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert logs_proc.returncode == 0
    assert "from-log" in logs_proc.stdout
