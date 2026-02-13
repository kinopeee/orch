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
