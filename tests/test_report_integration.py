from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _write_plan(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _extract_run_id(output: str) -> str:
    match = re.search(r"run_id:\s*([0-9]{8}_[0-9]{6}_[0-9a-f]{6})", output)
    assert match is not None, output
    return match.group(1)


def test_success_run_writes_report_with_artifacts(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_report_success.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: build
            cmd:
              [
                "python3",
                "-c",
                "from pathlib import Path; Path('artifact.txt').write_text('ok', encoding='utf-8')",
              ]
            outputs: ["artifact.txt"]
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
    assert proc.returncode == 0
    run_id = _extract_run_id(proc.stdout)
    report = (home / "runs" / run_id / "report" / "final_report.md").read_text(encoding="utf-8")
    assert "# Final Run Report" in report
    assert "status: **SUCCESS**" in report
    assert "artifacts/build/artifact.txt" in report


def test_failed_run_writes_report_with_problem_section(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_report_fail.yaml"
    home = tmp_path / ".orch_cli"
    _write_plan(
        plan_path,
        """
        tasks:
          - id: failing
            cmd:
              [
                "python3",
                "-c",
                "import sys; print('boom', file=sys.stderr); sys.exit(1)",
              ]
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
    report = (home / "runs" / run_id / "report" / "final_report.md").read_text(encoding="utf-8")
    assert "status: **FAILED**" in report
    assert "### failing (FAILED)" in report
    assert "boom" in report
