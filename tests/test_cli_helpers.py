from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import typer

import orch.cli as cli_module
from orch.cli import (
    _resolve_workdir_or_exit,
    _validate_home_or_exit,
    _write_plan_snapshot,
    _write_report,
)
from orch.config.loader import load_plan
from orch.config.schema import PlanSpec, TaskSpec
from orch.state.model import RunState


def test_write_plan_snapshot_roundtrips_to_valid_plan(tmp_path: Path) -> None:
    plan = PlanSpec(
        goal="snapshot goal",
        artifacts_dir="collected",
        tasks=[
            TaskSpec(
                id="build",
                cmd=["python3", "-c", "print('ok')"],
                depends_on=[],
                cwd=".",
                env={"KEY": "VALUE"},
                timeout_sec=1.5,
                retries=2,
                retry_backoff_sec=[0.1, 0.2],
                outputs=["dist/**"],
            ),
            TaskSpec(
                id="test",
                cmd=["python3", "-c", "print('test')"],
                depends_on=["build"],
                retries=0,
                retry_backoff_sec=[],
                outputs=[],
            ),
        ],
    )

    snapshot_path = tmp_path / "plan.yaml"
    _write_plan_snapshot(plan, snapshot_path)
    loaded = load_plan(snapshot_path)
    assert loaded == plan


def test_write_plan_snapshot_rejects_symlink_destination(tmp_path: Path) -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="only", cmd=["python3", "-c", "print('ok')"])],
    )
    target = tmp_path / "outside_plan.yaml"
    target.write_text("sentinel\n", encoding="utf-8")
    snapshot_path = tmp_path / "plan.yaml"
    snapshot_path.symlink_to(target)

    with pytest.raises(OSError, match="plan snapshot path must not be symlink"):
        _write_plan_snapshot(plan, snapshot_path)
    assert target.read_text(encoding="utf-8") == "sentinel\n"


def test_write_plan_snapshot_rejects_non_regular_destination(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")

    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="only", cmd=["python3", "-c", "print('ok')"])],
    )
    snapshot_path = tmp_path / "plan.yaml"
    os.mkfifo(snapshot_path)

    with pytest.raises(OSError, match="plan snapshot path must be regular file"):
        _write_plan_snapshot(plan, snapshot_path)


def test_write_plan_snapshot_rejects_symlink_ancestor_destination(tmp_path: Path) -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="only", cmd=["python3", "-c", "print('ok')"])],
    )
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "link_parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(OSError, match="contains symlink component"):
        _write_plan_snapshot(plan, symlink_parent / "plan.yaml")


def test_write_plan_snapshot_wraps_runtime_lstat_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="only", cmd=["python3", "-c", "print('ok')"])],
    )
    snapshot_path = tmp_path / "plan_runtime_lstat.yaml"
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj in {snapshot_path, snapshot_path.parent}:
            return False
        return original_is_symlink(path_obj)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> object:
        if path_obj == snapshot_path:
            raise RuntimeError("simulated snapshot lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="failed to prepare plan snapshot path"):
        _write_plan_snapshot(plan, snapshot_path)


def test_write_plan_snapshot_wraps_runtime_open_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="only", cmd=["python3", "-c", "print('ok')"])],
    )
    snapshot_path = tmp_path / "plan_runtime_close.yaml"

    def _raise_runtime(_path: str, _flags: int, _mode: int = 0o777) -> int:
        raise RuntimeError("simulated open runtime failure")

    monkeypatch.setattr(os, "open", _raise_runtime)

    with pytest.raises(OSError, match="failed to write plan snapshot"):
        _write_plan_snapshot(plan, snapshot_path)


def test_write_plan_snapshot_uses_nonblock_and_nofollow_open_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="only", cmd=["python3", "-c", "print('ok')"])],
    )
    snapshot_path = tmp_path / "plan_flags.yaml"
    captured_flags: dict[str, int] = {}
    captured_mode: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str, flags: int, mode: int = 0o777) -> int:
        if path == str(snapshot_path):
            captured_flags["flags"] = flags
            captured_mode["mode"] = mode
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    _write_plan_snapshot(plan, snapshot_path)

    assert "flags" in captured_flags
    assert captured_flags["flags"] & os.O_WRONLY
    assert captured_flags["flags"] & os.O_CREAT
    assert captured_flags["flags"] & os.O_TRUNC
    assert captured_mode.get("mode") == 0o600
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_WRONLY
    if hasattr(os, "O_APPEND"):
        assert not (captured_flags["flags"] & os.O_APPEND)
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


def test_write_report_rejects_symlink_ancestor_path(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    real_run_dir = real_parent / "run"
    (real_run_dir / "report").mkdir(parents=True)
    symlink_parent = tmp_path / "link_parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    state = RunState(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        status="SUCCESS",
        goal=None,
        plan_relpath="plan.yaml",
        home=str(tmp_path),
        workdir=str(tmp_path),
        max_parallel=1,
        fail_fast=False,
        tasks={},
    )

    with pytest.raises(OSError, match="contains symlink component"):
        _write_report(state, symlink_parent / "run")
    assert not (real_run_dir / "report" / "final_report.md").exists()


def test_write_report_wraps_runtime_lstat_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "report").mkdir(parents=True)
    state = RunState(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        status="SUCCESS",
        goal=None,
        plan_relpath="plan.yaml",
        home=str(tmp_path),
        workdir=str(tmp_path),
        max_parallel=1,
        fail_fast=False,
        tasks={},
    )
    report_path = run_dir / "report" / "final_report.md"
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj in {report_path, report_path.parent}:
            return False
        return original_is_symlink(path_obj)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> object:
        if path_obj == report_path:
            raise RuntimeError("simulated report lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="failed to prepare report path"):
        _write_report(state, run_dir)


def test_write_report_wraps_runtime_open_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "report").mkdir(parents=True)
    state = RunState(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        status="SUCCESS",
        goal=None,
        plan_relpath="plan.yaml",
        home=str(tmp_path),
        workdir=str(tmp_path),
        max_parallel=1,
        fail_fast=False,
        tasks={},
    )

    def _raise_runtime(_path: str, _flags: int, _mode: int = 0o777) -> int:
        raise RuntimeError("simulated open runtime failure")

    monkeypatch.setattr(os, "open", _raise_runtime)

    with pytest.raises(OSError, match="failed to write report"):
        _write_report(state, run_dir)


def test_write_report_uses_nonblock_and_nofollow_open_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "report").mkdir(parents=True)
    state = RunState(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        status="SUCCESS",
        goal=None,
        plan_relpath="plan.yaml",
        home=str(tmp_path),
        workdir=str(tmp_path),
        max_parallel=1,
        fail_fast=False,
        tasks={},
    )
    report_path = run_dir / "report" / "final_report.md"
    captured_flags: dict[str, int] = {}
    captured_mode: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str, flags: int, mode: int = 0o777) -> int:
        if path == str(report_path):
            captured_flags["flags"] = flags
            captured_mode["mode"] = mode
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    _write_report(state, run_dir)

    assert "flags" in captured_flags
    assert captured_flags["flags"] & os.O_WRONLY
    assert captured_flags["flags"] & os.O_CREAT
    assert captured_flags["flags"] & os.O_TRUNC
    assert captured_mode.get("mode") == 0o600
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_WRONLY
    if hasattr(os, "O_APPEND"):
        assert not (captured_flags["flags"] & os.O_APPEND)
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


def test_validate_home_or_exit_rejects_when_lstat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == home:
            raise PermissionError("simulated lstat failure")
        return original_lstat(path_obj)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(typer.Exit) as exc_info:
        _validate_home_or_exit(home)
    assert exc_info.value.exit_code == 2


def test_validate_home_or_exit_rejects_when_lstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == home:
            raise RuntimeError("simulated lstat runtime failure")
        return original_lstat(path_obj)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(typer.Exit) as exc_info:
        _validate_home_or_exit(home)
    assert exc_info.value.exit_code == 2


def test_validate_home_or_exit_rejects_when_existing_component_is_not_directory(
    tmp_path: Path,
) -> None:
    home_file = tmp_path / "home_file"
    home_file.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(typer.Exit) as exc_info:
        _validate_home_or_exit(home_file)
    assert exc_info.value.exit_code == 2


def test_resolve_workdir_or_exit_rejects_when_resolve_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "workdir"
    original_resolve = Path.resolve

    def flaky_resolve(path_obj: Path, *args: object, **kwargs: object) -> Path:
        if path_obj == workdir:
            raise RuntimeError("simulated resolve failure")
        return original_resolve(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)

    with pytest.raises(typer.Exit) as exc_info:
        _resolve_workdir_or_exit(workdir)
    assert exc_info.value.exit_code == 2


def test_resolve_workdir_or_exit_rejects_when_lstat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == workdir:
            raise PermissionError("simulated lstat failure")
        return original_lstat(path_obj)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(typer.Exit) as exc_info:
        _resolve_workdir_or_exit(workdir)
    assert exc_info.value.exit_code == 2


def test_cli_run_normalizes_runtime_initialize_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: t1
    cmd: ["python3", "-c", "print('ok')"]
""".strip(),
        encoding="utf-8",
    )
    home = tmp_path / ".orch"
    workdir = tmp_path / "wd"
    workdir.mkdir()

    def boom_initialize(_run_dir: Path) -> None:
        raise RuntimeError("simulated initialize runtime failure")

    monkeypatch.setattr(cli_module, "ensure_run_layout", boom_initialize)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.run(
            plan_path,
            max_parallel=1,
            home=home,
            workdir=workdir,
            fail_fast=False,
            dry_run=False,
        )
    assert exc_info.value.exit_code == 2


def test_cli_run_normalizes_runtime_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: t1
    cmd: ["python3", "-c", "print('ok')"]
""".strip(),
        encoding="utf-8",
    )
    home = tmp_path / ".orch"
    workdir = tmp_path / "wd"
    workdir.mkdir()

    async def boom_run_plan(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated run execution runtime failure")

    monkeypatch.setattr(cli_module, "run_plan", boom_run_plan)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.run(
            plan_path,
            max_parallel=1,
            home=home,
            workdir=workdir,
            fail_fast=False,
            dry_run=False,
        )
    assert exc_info.value.exit_code == 2


def test_cli_resume_normalizes_runtime_lock_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    home.mkdir()
    workdir = tmp_path / "wd"
    workdir.mkdir()

    @contextmanager
    def boom_lock(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated lock runtime failure")
        yield

    monkeypatch.setattr(cli_module, "run_lock", boom_lock)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.resume(
            "run1",
            home=home,
            max_parallel=1,
            workdir=workdir,
            fail_fast=False,
            failed_only=False,
        )
    assert exc_info.value.exit_code == 2


def test_cli_status_normalizes_runtime_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    home.mkdir()

    @contextmanager
    def fake_lock(*args: object, **kwargs: object) -> object:
        yield

    def boom_load_state(_run_dir: Path) -> object:
        raise RuntimeError("simulated state runtime failure")

    monkeypatch.setattr(cli_module, "run_lock", fake_lock)
    monkeypatch.setattr(cli_module, "load_state", boom_load_state)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.status("run1", home=home, as_json=False)
    assert exc_info.value.exit_code == 2


def test_cli_logs_normalizes_runtime_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    home.mkdir()

    @contextmanager
    def fake_lock(*args: object, **kwargs: object) -> object:
        yield

    def boom_load_state(_run_dir: Path) -> object:
        raise RuntimeError("simulated state runtime failure")

    monkeypatch.setattr(cli_module, "run_lock", fake_lock)
    monkeypatch.setattr(cli_module, "load_state", boom_load_state)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.logs("run1", home=home, task=None, tail=10)
    assert exc_info.value.exit_code == 2


def test_cli_cancel_normalizes_runtime_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    run_dir = home / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")

    def boom_write_cancel(_run_dir: Path) -> None:
        raise RuntimeError("simulated cancel runtime failure")

    monkeypatch.setattr(cli_module, "write_cancel_request", boom_write_cancel)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.cancel("run1", home=home)
    assert exc_info.value.exit_code == 2


def test_run_exists_short_circuits_on_symlink_ancestor_without_marker_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "state.json").write_text("{}", encoding="utf-8")
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    linked_home = tmp_path / "home_link"
    linked_home.symlink_to(real_home, target_is_directory=True)
    linked_run_dir = linked_home / "runs" / "run1"
    linked_state = linked_run_dir / "state.json"
    linked_plan = linked_run_dir / "plan.yaml"

    original_lstat = Path.lstat
    marker_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal marker_lstat_calls
        if path_obj in {linked_state, linked_plan}:
            marker_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(linked_run_dir) is False
    assert marker_lstat_calls == 0


def test_run_exists_short_circuits_on_symlink_ancestor_without_run_dir_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_home = tmp_path / "real_home"
    real_run_dir = real_home / "runs" / "run1"
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "state.json").write_text("{}", encoding="utf-8")

    linked_home = tmp_path / "home_link"
    linked_home.symlink_to(real_home, target_is_directory=True)
    linked_run_dir = linked_home / "runs" / "run1"

    original_lstat = Path.lstat
    run_dir_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal run_dir_lstat_calls
        if path_obj == linked_run_dir:
            run_dir_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(linked_run_dir) is False
    assert run_dir_lstat_calls == 0


def test_run_exists_short_circuits_when_runs_component_is_symlink_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_runs = tmp_path / "real_runs"
    real_run_dir = real_runs / "run1"
    real_run_dir.mkdir(parents=True)
    (real_run_dir / "state.json").write_text("{}", encoding="utf-8")

    home = tmp_path / ".orch"
    home.mkdir()
    (home / "runs").symlink_to(real_runs, target_is_directory=True)
    linked_run_dir = home / "runs" / "run1"
    linked_state = linked_run_dir / "state.json"
    linked_plan = linked_run_dir / "plan.yaml"

    original_lstat = Path.lstat
    run_dir_lstat_calls = 0
    marker_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal run_dir_lstat_calls, marker_lstat_calls
        if path_obj == linked_run_dir:
            run_dir_lstat_calls += 1
        if path_obj in {linked_state, linked_plan}:
            marker_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(linked_run_dir) is False
    assert run_dir_lstat_calls == 0
    assert marker_lstat_calls == 0


def test_run_exists_short_circuits_when_runs_component_symlinks_to_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    home.mkdir()
    outside_file = tmp_path / "outside_runs_file.txt"
    outside_file.write_text("outside\n", encoding="utf-8")
    (home / "runs").symlink_to(outside_file)
    linked_run_dir = home / "runs" / "run1"
    linked_state = linked_run_dir / "state.json"
    linked_plan = linked_run_dir / "plan.yaml"

    original_lstat = Path.lstat
    run_dir_lstat_calls = 0
    marker_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal run_dir_lstat_calls, marker_lstat_calls
        if path_obj == linked_run_dir:
            run_dir_lstat_calls += 1
        if path_obj in {linked_state, linked_plan}:
            marker_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(linked_run_dir) is False
    assert run_dir_lstat_calls == 0
    assert marker_lstat_calls == 0
    assert outside_file.read_text(encoding="utf-8") == "outside\n"


def test_run_exists_short_circuits_on_non_directory_without_marker_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.parent.mkdir(parents=True)
    run_dir.write_text("not a directory\n", encoding="utf-8")
    marker_state = run_dir / "state.json"
    marker_plan = run_dir / "plan.yaml"

    original_lstat = Path.lstat
    marker_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal marker_lstat_calls
        if path_obj in {marker_state, marker_plan}:
            marker_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is False
    assert marker_lstat_calls == 0


def test_run_exists_short_circuits_on_run_dir_lstat_runtime_error_without_marker_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    marker_state = run_dir / "state.json"
    marker_plan = run_dir / "plan.yaml"

    monkeypatch.setattr(cli_module, "has_symlink_ancestor", lambda _path: False)

    original_lstat = Path.lstat
    marker_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal marker_lstat_calls
        if path_obj in {marker_state, marker_plan}:
            marker_lstat_calls += 1
        if path_obj == run_dir:
            raise RuntimeError("simulated run_dir lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is False
    assert marker_lstat_calls == 0


def test_run_exists_short_circuits_on_run_dir_lstat_oserror_without_marker_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    marker_state = run_dir / "state.json"
    marker_plan = run_dir / "plan.yaml"

    monkeypatch.setattr(cli_module, "has_symlink_ancestor", lambda _path: False)

    original_lstat = Path.lstat
    marker_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal marker_lstat_calls
        if path_obj in {marker_state, marker_plan}:
            marker_lstat_calls += 1
        if path_obj == run_dir:
            raise OSError("simulated run_dir lstat os failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is False
    assert marker_lstat_calls == 0


def test_run_exists_short_circuits_on_symlink_run_dir_without_marker_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run_dir = tmp_path / "outside_run"
    real_run_dir.mkdir()
    (real_run_dir / "state.json").write_text("{}", encoding="utf-8")
    (real_run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    linked_run_dir = tmp_path / ".orch" / "runs" / "run1"
    linked_run_dir.parent.mkdir(parents=True)
    linked_run_dir.symlink_to(real_run_dir, target_is_directory=True)

    linked_state = linked_run_dir / "state.json"
    linked_plan = linked_run_dir / "plan.yaml"

    original_lstat = Path.lstat
    marker_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal marker_lstat_calls
        if path_obj in {linked_state, linked_plan}:
            marker_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(linked_run_dir) is False
    assert marker_lstat_calls == 0


def test_run_exists_short_circuits_on_missing_run_dir_without_marker_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    marker_state = run_dir / "state.json"
    marker_plan = run_dir / "plan.yaml"

    original_lstat = Path.lstat
    marker_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal marker_lstat_calls
        if path_obj in {marker_state, marker_plan}:
            marker_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is False
    assert marker_lstat_calls == 0


def test_run_exists_accepts_regular_plan_marker_without_state_marker(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_accepts_regular_state_marker_without_plan_marker(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_rejects_symlink_only_markers(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    outside_state = tmp_path / "outside_state.json"
    outside_plan = tmp_path / "outside_plan.yaml"
    outside_state.write_text("{}", encoding="utf-8")
    outside_plan.write_text("tasks: []\n", encoding="utf-8")
    (run_dir / "state.json").symlink_to(outside_state)
    (run_dir / "plan.yaml").symlink_to(outside_plan)

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_accepts_regular_state_with_symlink_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    outside_plan = tmp_path / "outside_plan.yaml"
    outside_plan.write_text("tasks: []\n", encoding="utf-8")
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    (run_dir / "plan.yaml").symlink_to(outside_plan)

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_accepts_regular_plan_with_symlink_state(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    outside_state = tmp_path / "outside_state.json"
    outside_state.write_text("{}", encoding="utf-8")
    (run_dir / "state.json").symlink_to(outside_state)
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_rejects_directory_only_markers(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").mkdir()
    (run_dir / "plan.yaml").mkdir()

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_accepts_regular_state_with_directory_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    (run_dir / "plan.yaml").mkdir()

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_accepts_regular_plan_with_directory_state(tmp_path: Path) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").mkdir()
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_rejects_fifo_only_markers(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    os.mkfifo(run_dir / "state.json")
    os.mkfifo(run_dir / "plan.yaml")

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_rejects_mixed_non_regular_symlink_and_directory_markers(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    outside_state = tmp_path / "outside_state.json"
    outside_state.write_text("{}", encoding="utf-8")
    (run_dir / "state.json").symlink_to(outside_state)
    (run_dir / "plan.yaml").mkdir()

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_rejects_mixed_non_regular_directory_and_symlink_markers(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    outside_plan = tmp_path / "outside_plan.yaml"
    outside_plan.write_text("tasks: []\n", encoding="utf-8")
    (run_dir / "state.json").mkdir()
    (run_dir / "plan.yaml").symlink_to(outside_plan)

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_rejects_mixed_non_regular_fifo_and_directory_markers(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    os.mkfifo(run_dir / "state.json")
    (run_dir / "plan.yaml").mkdir()

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_rejects_mixed_non_regular_directory_and_fifo_markers(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").mkdir()
    os.mkfifo(run_dir / "plan.yaml")

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_rejects_mixed_non_regular_symlink_and_fifo_markers(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    outside_state = tmp_path / "outside_state.json"
    outside_state.write_text("{}", encoding="utf-8")
    (run_dir / "state.json").symlink_to(outside_state)
    os.mkfifo(run_dir / "plan.yaml")

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_rejects_mixed_non_regular_fifo_and_symlink_markers(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    outside_plan = tmp_path / "outside_plan.yaml"
    outside_plan.write_text("tasks: []\n", encoding="utf-8")
    os.mkfifo(run_dir / "state.json")
    (run_dir / "plan.yaml").symlink_to(outside_plan)

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_accepts_regular_state_with_fifo_plan(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    os.mkfifo(run_dir / "plan.yaml")

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_accepts_regular_plan_with_fifo_state(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    os.mkfifo(run_dir / "state.json")
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_short_circuits_plan_marker_when_state_is_regular(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    plan_marker = run_dir / "plan.yaml"

    original_lstat = Path.lstat
    plan_lstat_calls = 0

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal plan_lstat_calls
        if path_obj == plan_marker:
            plan_lstat_calls += 1
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is True
    assert plan_lstat_calls == 0


def test_run_exists_accepts_regular_state_when_plan_lstat_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    plan_marker = run_dir / "plan.yaml"

    original_lstat = Path.lstat

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == plan_marker:
            raise RuntimeError("simulated plan marker runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_accepts_regular_state_when_plan_lstat_raises_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    plan_marker = run_dir / "plan.yaml"

    original_lstat = Path.lstat

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == plan_marker:
            raise OSError("simulated plan marker os failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_accepts_regular_plan_when_state_lstat_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    state_marker = run_dir / "state.json"

    original_lstat = Path.lstat

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == state_marker:
            raise RuntimeError("simulated state marker runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_rejects_when_both_marker_lstat_raise_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    state_marker = run_dir / "state.json"
    plan_marker = run_dir / "plan.yaml"

    original_lstat = Path.lstat

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj in {state_marker, plan_marker}:
            raise RuntimeError("simulated marker runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is False


def test_run_exists_accepts_regular_plan_when_state_lstat_raises_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "plan.yaml").write_text("tasks: []\n", encoding="utf-8")
    state_marker = run_dir / "state.json"

    original_lstat = Path.lstat

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == state_marker:
            raise OSError("simulated state marker os failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is True


def test_run_exists_rejects_when_both_marker_lstat_raise_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run1"
    run_dir.mkdir(parents=True)
    state_marker = run_dir / "state.json"
    plan_marker = run_dir / "plan.yaml"

    original_lstat = Path.lstat

    def capture_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj in {state_marker, plan_marker}:
            raise OSError("simulated marker os failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", capture_lstat)

    assert cli_module._run_exists(run_dir) is False


def test_cli_cancel_skips_write_when_run_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    write_called = False

    def fake_run_exists(_run_dir: Path) -> bool:
        return False

    def fake_write_cancel(_run_dir: Path) -> None:
        nonlocal write_called
        write_called = True

    monkeypatch.setattr(cli_module, "_run_exists", fake_run_exists)
    monkeypatch.setattr(cli_module, "write_cancel_request", fake_write_cancel)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.cancel("run1", home=home)
    assert exc_info.value.exit_code == 2
    assert write_called is False


def test_cli_cancel_normalizes_runtime_run_exists_error_without_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / ".orch"
    write_called = False

    def boom_run_exists(_run_dir: Path) -> bool:
        raise RuntimeError("simulated run exists runtime failure")

    def fake_write_cancel(_run_dir: Path) -> None:
        nonlocal write_called
        write_called = True

    monkeypatch.setattr(cli_module, "_run_exists", boom_run_exists)
    monkeypatch.setattr(cli_module, "write_cancel_request", fake_write_cancel)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.cancel("run1", home=home)
    assert exc_info.value.exit_code == 2
    assert write_called is False
    captured = capsys.readouterr()
    assert "Failed to inspect run" in captured.out


def test_cli_cancel_normalizes_oserror_run_exists_error_without_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / ".orch"
    write_called = False

    def boom_run_exists(_run_dir: Path) -> bool:
        raise OSError("simulated run exists os failure")

    def fake_write_cancel(_run_dir: Path) -> None:
        nonlocal write_called
        write_called = True

    monkeypatch.setattr(cli_module, "_run_exists", boom_run_exists)
    monkeypatch.setattr(cli_module, "write_cancel_request", fake_write_cancel)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.cancel("run1", home=home)
    assert exc_info.value.exit_code == 2
    assert write_called is False
    captured = capsys.readouterr()
    assert "Failed to inspect run" in captured.out


def test_cli_cancel_rejects_invalid_run_id_before_run_exists_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    run_exists_called = False
    write_called = False

    def fake_run_exists(_run_dir: Path) -> bool:
        nonlocal run_exists_called
        run_exists_called = True
        return True

    def fake_write_cancel(_run_dir: Path) -> None:
        nonlocal write_called
        write_called = True

    monkeypatch.setattr(cli_module, "_run_exists", fake_run_exists)
    monkeypatch.setattr(cli_module, "write_cancel_request", fake_write_cancel)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.cancel("../bad", home=home)
    assert exc_info.value.exit_code == 2
    assert run_exists_called is False
    assert write_called is False


def test_cli_cancel_invalid_run_id_short_circuits_before_home_and_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    validate_home_called = False
    run_dir_called = False
    run_exists_called = False
    write_called = False

    def fake_validate_home(_home: Path) -> None:
        nonlocal validate_home_called
        validate_home_called = True

    def fake_run_dir(_home: Path, _run_id: str) -> Path:
        nonlocal run_dir_called
        run_dir_called = True
        return _home / "runs" / "run1"

    def fake_run_exists(_run_dir: Path) -> bool:
        nonlocal run_exists_called
        run_exists_called = True
        return True

    def fake_write_cancel(_run_dir: Path) -> None:
        nonlocal write_called
        write_called = True

    monkeypatch.setattr(cli_module, "_validate_home_or_exit", fake_validate_home)
    monkeypatch.setattr(cli_module, "run_dir", fake_run_dir)
    monkeypatch.setattr(cli_module, "_run_exists", fake_run_exists)
    monkeypatch.setattr(cli_module, "write_cancel_request", fake_write_cancel)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.cancel("../bad", home=home)
    assert exc_info.value.exit_code == 2
    assert validate_home_called is False
    assert run_dir_called is False
    assert run_exists_called is False
    assert write_called is False


def test_cli_cancel_rejects_invalid_home_before_run_exists_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home_as_file"
    home.write_text("not a directory\n", encoding="utf-8")
    run_dir_called = False
    run_exists_called = False
    write_called = False

    def fake_run_dir(_home: Path, _run_id: str) -> Path:
        nonlocal run_dir_called
        run_dir_called = True
        return _home / "runs" / "run1"

    def fake_run_exists(_run_dir: Path) -> bool:
        nonlocal run_exists_called
        run_exists_called = True
        return True

    def fake_write_cancel(_run_dir: Path) -> None:
        nonlocal write_called
        write_called = True

    monkeypatch.setattr(cli_module, "run_dir", fake_run_dir)
    monkeypatch.setattr(cli_module, "_run_exists", fake_run_exists)
    monkeypatch.setattr(cli_module, "write_cancel_request", fake_write_cancel)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.cancel("run1", home=home)
    assert exc_info.value.exit_code == 2
    assert run_dir_called is False
    assert run_exists_called is False
    assert write_called is False


def test_cli_cancel_calls_write_when_run_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    write_called = False
    captured_run_dir: Path | None = None

    def fake_run_exists(_run_dir: Path) -> bool:
        return True

    def fake_write_cancel(run_dir: Path) -> None:
        nonlocal write_called, captured_run_dir
        write_called = True
        captured_run_dir = run_dir

    monkeypatch.setattr(cli_module, "_run_exists", fake_run_exists)
    monkeypatch.setattr(cli_module, "write_cancel_request", fake_write_cancel)

    cli_module.cancel("run1", home=home)
    assert write_called is True
    assert captured_run_dir == home / "runs" / "run1"


def test_cli_run_ignores_runtime_report_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: t1
    cmd: ["python3", "-c", "print('ok')"]
""".strip(),
        encoding="utf-8",
    )
    home = tmp_path / ".orch"
    workdir = tmp_path / "wd"
    workdir.mkdir()

    async def fake_run_plan(*args: object, **kwargs: object) -> RunState:
        return RunState(
            run_id="run1",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:01+00:00",
            status="SUCCESS",
            goal=None,
            plan_relpath="plan.yaml",
            home=str(home),
            workdir=str(workdir),
            max_parallel=1,
            fail_fast=False,
            tasks={},
        )

    def boom_write_report(_state: RunState, _run_dir: Path) -> Path:
        raise RuntimeError("simulated report runtime failure")

    monkeypatch.setattr(cli_module, "run_plan", fake_run_plan)
    monkeypatch.setattr(cli_module, "_write_report", boom_write_report)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.run(
            plan_path,
            max_parallel=1,
            home=home,
            workdir=workdir,
            fail_fast=False,
            dry_run=False,
        )
    assert exc_info.value.exit_code == 0


def test_cli_resume_ignores_runtime_report_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".orch"
    home.mkdir()
    workdir = tmp_path / "wd"
    workdir.mkdir()
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[TaskSpec(id="t1", cmd=["python3", "-c", "print('ok')"])],
    )

    @contextmanager
    def fake_lock(*args: object, **kwargs: object) -> object:
        yield

    def fake_load_plan(_path: Path) -> PlanSpec:
        return plan

    async def fake_run_plan(*args: object, **kwargs: object) -> RunState:
        return RunState(
            run_id="run1",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:01+00:00",
            status="SUCCESS",
            goal=None,
            plan_relpath="plan.yaml",
            home=str(home),
            workdir=str(workdir),
            max_parallel=1,
            fail_fast=False,
            tasks={},
        )

    def boom_write_report(_state: RunState, _run_dir: Path) -> Path:
        raise RuntimeError("simulated report runtime failure")

    monkeypatch.setattr(cli_module, "run_lock", fake_lock)
    monkeypatch.setattr(cli_module, "load_plan", fake_load_plan)
    monkeypatch.setattr(cli_module, "run_plan", fake_run_plan)
    monkeypatch.setattr(cli_module, "_write_report", boom_write_report)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module.resume(
            "run1",
            home=home,
            max_parallel=1,
            workdir=workdir,
            fail_fast=False,
            failed_only=False,
        )
    assert exc_info.value.exit_code == 0
