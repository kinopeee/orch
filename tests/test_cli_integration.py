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


def test_cli_run_dry_run_returns_zero(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
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
        [sys.executable, "-m", "orch.cli", "run", str(plan_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Dry Run" in proc.stdout
    assert "t1" in proc.stdout
    assert "t2" in proc.stdout


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
