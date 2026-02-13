from __future__ import annotations

import json
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
    assert proc.returncode == 0
    assert "Dry Run" in proc.stdout
    assert "t1" in proc.stdout
    assert "t2" in proc.stdout
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
    assert not (home / "runs").exists()


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
    assert not (outside_run_dir / "cancel.request").exists()


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
