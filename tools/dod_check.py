#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Options:
    skip_quality_gates: bool
    home: Path


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _run(args: Sequence[str], *, expected: int | None = None, title: str) -> CommandResult:
    _print_header(title)
    print("+", " ".join(args))
    completed = subprocess.run(
        list(args),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"exit: {completed.returncode}")
    if completed.stdout.strip():
        print("stdout:")
        print(completed.stdout.rstrip())
    if completed.stderr.strip():
        print("stderr:")
        print(completed.stderr.rstrip())
    if expected is not None and completed.returncode != expected:
        raise RuntimeError(f"unexpected exit code: got={completed.returncode}, want={expected}")
    return CommandResult(
        args=list(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _detect_orch_prefix() -> list[str]:
    try:
        probe = subprocess.run(
            ["orch", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        probe = None

    if probe is not None and probe.returncode == 0:
        return ["orch"]
    return [sys.executable, "-m", "orch.cli"]


def _parse_args(argv: Sequence[str]) -> Options:
    parser = argparse.ArgumentParser(description="Release 0.1 DoD self-check runner")
    parser.add_argument(
        "--skip-quality-gates",
        action="store_true",
        help="Skip ruff/pytest checks (useful when these are run separately in CI)",
    )
    parser.add_argument(
        "--home",
        default=".orch",
        help="Home directory used for DoD runs (default: .orch)",
    )
    parsed = parser.parse_args(list(argv))
    home = Path(parsed.home)
    resolved_home = home.resolve() if home.is_absolute() else (ROOT / home).resolve()
    return Options(skip_quality_gates=parsed.skip_quality_gates, home=resolved_home)


def _parse_run_id(output: str) -> str:
    match = re.search(r"run_id:\s*([A-Za-z0-9_-]+)", output)
    if match is None:
        raise RuntimeError("run_id not found in command output")
    return match.group(1)


def _runs_dir(home: Path) -> Path:
    return home / "runs"


def _load_state(run_id: str, runs_dir: Path) -> dict[str, object]:
    state_path = runs_dir / run_id / "state.json"
    if not state_path.exists():
        raise RuntimeError(f"state file not found: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _assert_report_exists(run_id: str, runs_dir: Path) -> None:
    report_path = runs_dir / run_id / "report" / "final_report.md"
    _assert(report_path.exists(), f"final report file was not generated: {report_path}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _parse_iso_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"timestamp must be string, got {type(value).__name__}")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError("timestamp must include timezone offset")
    return parsed


def _intervals_overlap(
    left_start: object, left_end: object, right_start: object, right_end: object
) -> bool:
    ls = _parse_iso_timestamp(left_start)
    le = _parse_iso_timestamp(left_end)
    rs = _parse_iso_timestamp(right_start)
    re_ = _parse_iso_timestamp(right_end)
    return ls < re_ and rs < le


def _has_parallel_overlap(state: dict[str, object]) -> bool:
    tasks_obj = state.get("tasks")
    if not isinstance(tasks_obj, dict):
        raise RuntimeError("state.tasks must be an object")

    windows: list[tuple[object, object]] = []
    for task_obj in tasks_obj.values():
        if not isinstance(task_obj, dict):
            raise RuntimeError("task state must be an object")
        if task_obj.get("status") != "SUCCESS":
            continue
        depends_on = task_obj.get("depends_on")
        if not isinstance(depends_on, list):
            raise RuntimeError("task depends_on must be a list")
        if depends_on:
            continue

        started_at = task_obj.get("started_at")
        ended_at = task_obj.get("ended_at")
        if started_at is None or ended_at is None:
            continue
        windows.append((started_at, ended_at))

    for i, (left_start, left_end) in enumerate(windows):
        for right_start, right_end in windows[i + 1 :]:
            if _intervals_overlap(left_start, left_end, right_start, right_end):
                return True
    return False


def _run_cancel_scenario(orch_prefix: list[str], home_str: str, runs_dir: Path) -> str:
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in runs_dir.iterdir() if p.is_dir()}
    proc = subprocess.Popen(
        [*orch_prefix, "run", "examples/plan_cancel.yaml", "--home", home_str],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    detected_run_id: str | None = None
    deadline = time.time() + 20
    while time.time() < deadline and detected_run_id is None:
        for path in runs_dir.iterdir():
            if not path.is_dir() or path.name in existing:
                continue
            state_path = path / "state.json"
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if state.get("status") == "RUNNING":
                detected_run_id = path.name
                break
        time.sleep(0.05)

    if detected_run_id is None:
        proc.terminate()
        proc.communicate(timeout=10)
        raise RuntimeError("could not detect running cancel scenario run_id")

    cancel = _run(
        [*orch_prefix, "cancel", detected_run_id, "--home", home_str],
        expected=0,
        title="cancel running run",
    )
    run_stdout, run_stderr = proc.communicate(timeout=30)
    print("run(exit):", proc.returncode)
    if run_stdout.strip():
        print("run(stdout):")
        print(run_stdout.rstrip())
    if run_stderr.strip():
        print("run(stderr):")
        print(run_stderr.rstrip())

    _assert(proc.returncode == 4, "cancel scenario run must exit with code 4")
    state = _load_state(detected_run_id, runs_dir)
    _assert(state.get("status") == "CANCELED", "cancel scenario state must be CANCELED")
    _assert_report_exists(detected_run_id, runs_dir)
    _assert(
        "cancel requested" in cancel.stdout.lower(),
        "cancel command must print cancel requested message",
    )
    return detected_run_id


def main(options: Options) -> int:
    orch_prefix = _detect_orch_prefix()
    runs_dir = _runs_dir(options.home)
    home_str = str(options.home)
    print("orch command:", " ".join(orch_prefix))
    print("home:", home_str)

    basic = _run(
        [*orch_prefix, "run", "examples/plan_basic.yaml", "--home", home_str],
        expected=0,
        title="run basic plan",
    )
    basic_run_id = _parse_run_id(basic.stdout)
    _assert_report_exists(basic_run_id, runs_dir)

    parallel = _run(
        [
            *orch_prefix,
            "run",
            "examples/plan_parallel.yaml",
            "--max-parallel",
            "2",
            "--home",
            home_str,
        ],
        expected=0,
        title="run parallel plan",
    )
    parallel_run_id = _parse_run_id(parallel.stdout)
    _assert_report_exists(parallel_run_id, runs_dir)
    parallel_state = _load_state(parallel_run_id, runs_dir)
    _assert(_has_parallel_overlap(parallel_state), "parallel evidence check failed")

    fail = _run(
        [*orch_prefix, "run", "examples/plan_fail_retry.yaml", "--home", home_str],
        expected=3,
        title="run failure plan for skip propagation",
    )
    fail_run_id = _parse_run_id(fail.stdout)
    _assert_report_exists(fail_run_id, runs_dir)
    fail_state = _load_state(fail_run_id, runs_dir)
    fail_tasks = fail_state["tasks"]  # type: ignore[index]
    downstream = fail_tasks["downstream"]  # type: ignore[index]
    _assert(downstream["status"] == "SKIPPED", "downstream must be SKIPPED")
    _assert(
        downstream["skip_reason"] == "dependency_not_success",
        "downstream skip_reason must be dependency_not_success",
    )

    resume = _run(
        [*orch_prefix, "resume", basic_run_id, "--home", home_str],
        expected=0,
        title="resume completed run",
    )
    _ = resume
    resumed_state = _load_state(basic_run_id, runs_dir)
    for task_id, task_state in resumed_state["tasks"].items():  # type: ignore[index]
        attempts = task_state["attempts"]  # type: ignore[index]
        _assert(attempts == 1, f"resume reran successful task: {task_id}")

    _run(
        [*orch_prefix, "status", basic_run_id, "--home", home_str],
        expected=0,
        title="status command",
    )
    _run(
        [
            *orch_prefix,
            "logs",
            basic_run_id,
            "--home",
            home_str,
            "--task",
            "inspect",
            "--tail",
            "5",
        ],
        expected=0,
        title="logs command",
    )

    cancel_run_id = _run_cancel_scenario(orch_prefix, home_str, runs_dir)

    if not options.skip_quality_gates:
        _run(
            [sys.executable, "-m", "ruff", "format", "--check", "."],
            expected=0,
            title="ruff format check",
        )
        _run(
            [sys.executable, "-m", "ruff", "check", "--no-fix", "."],
            expected=0,
            title="ruff lint check",
        )
        _run(
            [sys.executable, "-m", "pytest"],
            expected=0,
            title="pytest",
        )
    else:
        _print_header("quality gates")
        print("skipped (requested by --skip-quality-gates)")

    _print_header("DoD check summary")
    print(f"basic_run_id={basic_run_id}")
    print(f"parallel_run_id={parallel_run_id}")
    print(f"fail_run_id={fail_run_id}")
    print(f"cancel_run_id={cancel_run_id}")
    print("result=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(_parse_args(sys.argv[1:])))
    except Exception as exc:  # noqa: BLE001
        print(f"DoD check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
