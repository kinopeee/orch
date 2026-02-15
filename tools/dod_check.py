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
    emit_json: bool
    json_out: Path | None


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _run(
    args: Sequence[str],
    *,
    expected: int | None = None,
    title: str,
    timeout_sec: float = 180.0,
) -> CommandResult:
    _print_header(title)
    print("+", " ".join(args))
    try:
        completed = subprocess.run(
            list(args),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout_sec}s: {' '.join(args)}") from exc
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary at the end",
    )
    parser.add_argument(
        "--json-out",
        help="Write machine-readable JSON summary to this file path",
    )
    parsed = parser.parse_args(list(argv))
    home = Path(parsed.home)
    resolved_home = home.resolve() if home.is_absolute() else (ROOT / home).resolve()
    json_out: Path | None = None
    if parsed.json_out:
        output = Path(parsed.json_out)
        json_out = output.resolve() if output.is_absolute() else (ROOT / output).resolve()
    return Options(
        skip_quality_gates=parsed.skip_quality_gates,
        home=resolved_home,
        emit_json=parsed.json,
        json_out=json_out,
    )


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


def _state_tasks(state: dict[str, object]) -> dict[str, dict[str, object]]:
    tasks_obj = state.get("tasks")
    if not isinstance(tasks_obj, dict):
        raise RuntimeError("state.tasks must be an object")

    tasks: dict[str, dict[str, object]] = {}
    for task_id, task_obj in tasks_obj.items():
        if not isinstance(task_id, str):
            raise RuntimeError("task id must be a string")
        if not isinstance(task_obj, dict):
            raise RuntimeError(f"task state must be an object: {task_id}")
        tasks[task_id] = task_obj
    return tasks


def _successful_task_snapshots(state: dict[str, object]) -> dict[str, tuple[int, object]]:
    snapshots: dict[str, tuple[int, object]] = {}
    for task_id, task_state in _state_tasks(state).items():
        if task_state.get("status") != "SUCCESS":
            continue
        attempts = task_state.get("attempts")
        if not isinstance(attempts, int):
            raise RuntimeError(f"task attempts must be int: {task_id}")
        snapshots[task_id] = (attempts, task_state.get("started_at"))
    return snapshots


def _assert_resume_kept_successful_tasks_unchanged(
    baseline: dict[str, tuple[int, object]], resumed_state: dict[str, object]
) -> None:
    resumed_tasks = _state_tasks(resumed_state)
    for task_id, (baseline_attempts, baseline_started_at) in baseline.items():
        task_state = resumed_tasks.get(task_id)
        if task_state is None:
            raise RuntimeError(f"task missing after resume: {task_id}")

        attempts = task_state.get("attempts")
        if not isinstance(attempts, int):
            raise RuntimeError(f"task attempts must be int: {task_id}")
        _assert(
            attempts == baseline_attempts,
            f"resume changed attempts for successful task: {task_id}",
        )
        _assert(
            task_state.get("started_at") == baseline_started_at,
            f"resume changed started_at for successful task: {task_id}",
        )


def _assert_report_exists(run_id: str, runs_dir: Path) -> None:
    report_path = runs_dir / run_id / "report" / "final_report.md"
    _assert(report_path.exists(), f"final report file was not generated: {report_path}")


def _assert_run_status(run_id: str, runs_dir: Path, expected_status: str) -> None:
    state = _load_state(run_id, runs_dir)
    actual_status = state.get("status")
    _assert(
        actual_status == expected_status,
        f"run status mismatch: run_id={run_id}, expected={expected_status}, actual={actual_status}",
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _parse_iso_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"timestamp must be string, got {type(value).__name__}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid timestamp format: {value!r}") from exc
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
    windows: list[tuple[object, object]] = []
    for task_obj in _state_tasks(state).values():
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
        timeout_sec=30.0,
    )
    try:
        run_stdout, run_stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired as exc:
        proc.terminate()
        proc.communicate(timeout=10)
        raise RuntimeError("cancel scenario run did not stop within timeout") from exc
    print("run(exit):", proc.returncode)
    if run_stdout.strip():
        print("run(stdout):")
        print(run_stdout.rstrip())
    if run_stderr.strip():
        print("run(stderr):")
        print(run_stderr.rstrip())

    _assert(proc.returncode == 4, "cancel scenario run must exit with code 4")
    _assert_run_status(detected_run_id, runs_dir, "CANCELED")
    _assert_report_exists(detected_run_id, runs_dir)
    _assert(
        "cancel requested" in cancel.stdout.lower(),
        "cancel command must print cancel requested message",
    )
    return detected_run_id


def _build_summary_payload(
    *,
    basic_run_id: str,
    parallel_run_id: str,
    fail_run_id: str,
    cancel_run_id: str,
    home: Path,
) -> dict[str, str]:
    return {
        "result": "PASS",
        "basic_run_id": basic_run_id,
        "parallel_run_id": parallel_run_id,
        "fail_run_id": fail_run_id,
        "cancel_run_id": cancel_run_id,
        "home": str(home),
    }


def _assert_summary_payload_consistent(payload: dict[str, str]) -> None:
    required_keys = {
        "result",
        "basic_run_id",
        "parallel_run_id",
        "fail_run_id",
        "cancel_run_id",
        "home",
    }
    actual_keys = set(payload)
    if actual_keys != required_keys:
        missing = sorted(required_keys - actual_keys)
        extra = sorted(actual_keys - required_keys)
        raise RuntimeError(f"invalid summary keys: missing={missing}, extra={extra}")

    if payload["result"] != "PASS":
        raise RuntimeError(f"invalid summary result: {payload['result']!r}")
    if payload["home"] == "":
        raise RuntimeError("invalid summary home: empty")

    run_id_pattern = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")
    run_id_keys = ("basic_run_id", "parallel_run_id", "fail_run_id", "cancel_run_id")
    run_ids: list[str] = []
    for key in run_id_keys:
        run_id = payload[key]
        if run_id_pattern.fullmatch(run_id) is None:
            raise RuntimeError(f"invalid summary run_id: {key}={run_id!r}")
        run_ids.append(run_id)
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError(f"invalid summary run_id uniqueness: {run_ids!r}")


def _write_summary_json(path: Path, payload: dict[str, str]) -> None:
    _assert_summary_payload_consistent(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"failed to write summary json: {path}") from exc


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
        timeout_sec=180.0,
    )
    basic_run_id = _parse_run_id(basic.stdout)
    _assert_run_status(basic_run_id, runs_dir, "SUCCESS")
    _assert_report_exists(basic_run_id, runs_dir)
    baseline_successful_tasks = _successful_task_snapshots(_load_state(basic_run_id, runs_dir))

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
        timeout_sec=180.0,
    )
    parallel_run_id = _parse_run_id(parallel.stdout)
    _assert_run_status(parallel_run_id, runs_dir, "SUCCESS")
    _assert_report_exists(parallel_run_id, runs_dir)
    parallel_state = _load_state(parallel_run_id, runs_dir)
    _assert(_has_parallel_overlap(parallel_state), "parallel evidence check failed")

    fail = _run(
        [*orch_prefix, "run", "examples/plan_fail_retry.yaml", "--home", home_str],
        expected=3,
        title="run failure plan for skip propagation",
        timeout_sec=180.0,
    )
    fail_run_id = _parse_run_id(fail.stdout)
    _assert_run_status(fail_run_id, runs_dir, "FAILED")
    _assert_report_exists(fail_run_id, runs_dir)
    fail_state = _load_state(fail_run_id, runs_dir)
    fail_tasks = _state_tasks(fail_state)
    downstream = fail_tasks.get("downstream")
    if downstream is None:
        raise RuntimeError("downstream task not found in fail_retry state")
    _assert(downstream["status"] == "SKIPPED", "downstream must be SKIPPED")
    _assert(
        downstream["skip_reason"] == "dependency_not_success",
        "downstream skip_reason must be dependency_not_success",
    )

    resume = _run(
        [*orch_prefix, "resume", basic_run_id, "--home", home_str],
        expected=0,
        title="resume completed run",
        timeout_sec=180.0,
    )
    _ = resume
    _assert_run_status(basic_run_id, runs_dir, "SUCCESS")
    _assert_resume_kept_successful_tasks_unchanged(
        baseline_successful_tasks,
        _load_state(basic_run_id, runs_dir),
    )

    _run(
        [*orch_prefix, "status", basic_run_id, "--home", home_str],
        expected=0,
        title="status command",
        timeout_sec=60.0,
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
        timeout_sec=60.0,
    )

    cancel_run_id = _run_cancel_scenario(orch_prefix, home_str, runs_dir)

    if not options.skip_quality_gates:
        _run(
            [sys.executable, "-m", "ruff", "format", "--check", "."],
            expected=0,
            title="ruff format check",
            timeout_sec=120.0,
        )
        _run(
            [sys.executable, "-m", "ruff", "check", "--no-fix", "."],
            expected=0,
            title="ruff lint check",
            timeout_sec=120.0,
        )
        _run(
            [sys.executable, "-m", "pytest"],
            expected=0,
            title="pytest",
            timeout_sec=900.0,
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
    payload = _build_summary_payload(
        basic_run_id=basic_run_id,
        parallel_run_id=parallel_run_id,
        fail_run_id=fail_run_id,
        cancel_run_id=cancel_run_id,
        home=options.home,
    )
    _assert_summary_payload_consistent(payload)
    if options.emit_json:
        print(json.dumps(payload, sort_keys=True))
    if options.json_out is not None:
        _write_summary_json(options.json_out, payload)
        print(f"summary_json_path={options.json_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(_parse_args(sys.argv[1:])))
    except Exception as exc:  # noqa: BLE001
        print(f"DoD check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
