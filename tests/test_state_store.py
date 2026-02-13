from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from orch.state.model import RunState, TaskState
from orch.state.store import load_state, save_state_atomic
from orch.util.errors import StateError


def _minimal_state_payload(*, run_id: str) -> dict[str, object]:
    state = RunState(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        status="RUNNING",
        goal=None,
        plan_relpath="plan.yaml",
        home="/tmp/.orch_state_test",
        workdir="/tmp",
        max_parallel=1,
        fail_fast=False,
        tasks={
            "t1": TaskState(
                status="SUCCESS",
                depends_on=[],
                cmd=["echo", "ok"],
                cwd=".",
                env=None,
                timeout_sec=None,
                retries=0,
                retry_backoff_sec=[],
                outputs=[],
                attempts=1,
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:00:01+00:00",
                duration_sec=1.0,
                exit_code=0,
                stdout_path="logs/t1.out.log",
                stderr_path="logs/t1.err.log",
                artifact_paths=["artifacts/t1/out.txt"],
            )
        },
    )
    return state.to_dict()


def test_save_and_load_state_atomic(tmp_path: Path) -> None:
    home = tmp_path / ".orch"
    run_dir = home / "runs" / "run1"
    run_dir.mkdir(parents=True)
    state = RunState(
        run_id="run1",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        status="SUCCESS",
        goal="demo",
        plan_relpath="plan.yaml",
        home=str(home),
        workdir=str(tmp_path),
        max_parallel=2,
        fail_fast=False,
        tasks={
            "t1": TaskState(
                status="SUCCESS",
                depends_on=[],
                cmd=["echo", "ok"],
                cwd=".",
                env=None,
                timeout_sec=None,
                retries=0,
                retry_backoff_sec=[],
                outputs=[],
                attempts=1,
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:00:01+00:00",
                duration_sec=1.0,
                exit_code=0,
                stdout_path="logs/t1.out.log",
                stderr_path="logs/t1.err.log",
            )
        },
    )

    save_state_atomic(run_dir, state)
    loaded = load_state(run_dir)
    assert loaded.run_id == "run1"
    assert loaded.tasks["t1"].status == "SUCCESS"
    assert not (run_dir / "state.json.tmp").exists()


def test_save_state_atomic_cleans_tmp_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_replace_fail"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))

    def failing_replace(_src: str | Path, _dst: str | Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        save_state_atomic(run_dir, state)
    assert not (run_dir / "state.json.tmp").exists()


def test_save_state_atomic_cleans_tmp_when_replace_runtime_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_replace_runtime_fail"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))

    def failing_replace(_src: str | Path, _dst: str | Path) -> None:
        raise RuntimeError("simulated replace runtime failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="failed to replace state file"):
        save_state_atomic(run_dir, state)
    assert not (run_dir / "state.json.tmp").exists()


def test_save_state_atomic_cleans_tmp_when_write_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_write_fsync_fail"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    original_fsync = os.fsync
    failed_once = False

    def flaky_fsync(fd: int) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("simulated write fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", flaky_fsync)

    with pytest.raises(OSError, match="simulated write fsync failure"):
        save_state_atomic(run_dir, state)
    assert not (run_dir / "state.json.tmp").exists()


def test_save_state_atomic_cleans_tmp_when_write_fsync_runtime_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_write_fsync_runtime_fail"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    original_fsync = os.fsync
    failed_once = False

    def flaky_fsync(fd: int) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("simulated write fsync runtime failure")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", flaky_fsync)

    with pytest.raises(OSError, match="failed to write temporary state file"):
        save_state_atomic(run_dir, state)
    assert not (run_dir / "state.json.tmp").exists()


def test_save_state_atomic_rejects_symlink_tmp_without_overwriting_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_symlink_tmp"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    target = tmp_path / "outside.txt"
    target.write_text("do-not-touch\n", encoding="utf-8")
    tmp_symlink = run_dir / "state.json.tmp"
    tmp_symlink.symlink_to(target)

    with pytest.raises(OSError, match="temporary state path must not be symlink"):
        save_state_atomic(run_dir, state)
    assert tmp_symlink.is_symlink()
    assert target.read_text(encoding="utf-8") == "do-not-touch\n"


def test_save_state_atomic_rejects_symlink_state_target_without_overwriting(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_symlink_state_target"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    target = tmp_path / "outside_state.json"
    target.write_text("keep-target\n", encoding="utf-8")
    state_symlink = run_dir / "state.json"
    state_symlink.symlink_to(target)

    with pytest.raises(OSError, match="state file path must not be symlink"):
        save_state_atomic(run_dir, state)
    assert state_symlink.is_symlink()
    assert target.read_text(encoding="utf-8") == "keep-target\n"
    assert not (run_dir / "state.json.tmp").exists()


def test_save_state_atomic_wraps_runtime_state_lstat_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_state_lstat_runtime_error"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    state_path = run_dir / "state.json"
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj == state_path:
            return False
        return original_is_symlink(path_obj)

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == state_path:
            raise RuntimeError("simulated state lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="failed to prepare state file path"):
        save_state_atomic(run_dir, state)
    assert not state_path.exists()
    assert not (run_dir / "state.json.tmp").exists()


def test_save_state_atomic_fails_closed_when_state_symlink_check_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_symlink_check_error"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    state_path = run_dir / "state.json"
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == state_path:
            raise PermissionError("simulated state_path lstat failure")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    with pytest.raises(OSError, match="state file path must not be symlink"):
        save_state_atomic(run_dir, state)
    assert not state_path.exists()
    assert not (run_dir / "state.json.tmp").exists()


def test_save_state_atomic_rejects_state_path_with_symlink_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    run_real = real_parent / "run_linked"
    run_real.mkdir()
    linked_parent = tmp_path / "parent_link"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    run_dir = linked_parent / "run_linked"
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))

    with pytest.raises(OSError, match="contains symlink component"):
        save_state_atomic(run_dir, state)
    assert not (run_real / "state.json").exists()
    assert not (run_real / "state.json.tmp").exists()


def test_save_state_atomic_rejects_non_regular_tmp_path(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")

    run_dir = tmp_path / "run_fifo_tmp"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    os.mkfifo(run_dir / "state.json.tmp")

    with pytest.raises(OSError):
        save_state_atomic(run_dir, state)
    assert not (run_dir / "state.json").exists()


def test_save_state_atomic_normalizes_open_enxio_as_regular_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_tmp_open_enxio"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))

    def _raise_enxio(_path: str, _flags: int, _mode: int = 0o777) -> int:
        raise OSError(errno.ENXIO, "No such device or address")

    monkeypatch.setattr(os, "open", _raise_enxio)

    with pytest.raises(OSError, match="temporary state path must be regular file"):
        save_state_atomic(run_dir, state)


def test_save_state_atomic_wraps_runtime_open_error_for_tmp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_tmp_open_runtime_error"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))

    def _raise_runtime(_path: str, _flags: int, _mode: int = 0o777) -> int:
        raise RuntimeError("simulated tmp open runtime failure")

    monkeypatch.setattr(os, "open", _raise_runtime)

    with pytest.raises(OSError, match="failed to open temporary state path"):
        save_state_atomic(run_dir, state)


def test_save_state_atomic_cleans_tmp_when_tmp_path_changes_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_tmp_swapped"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    tmp_state_path = run_dir / "state.json.tmp"
    imposter = tmp_path / "imposter.tmp"
    imposter.write_text("imposter\n", encoding="utf-8")
    imposter_meta = imposter.lstat()
    original_lstat = Path.lstat

    def swapped_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == tmp_state_path:
            return imposter_meta
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", swapped_lstat)

    with pytest.raises(OSError, match="temporary state path changed before replace"):
        save_state_atomic(run_dir, state)
    assert not (run_dir / "state.json").exists()
    assert not tmp_state_path.exists()


def test_save_state_atomic_cleans_tmp_when_tmp_lstat_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_tmp_lstat_runtime_error"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    tmp_state_path = run_dir / "state.json.tmp"
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == tmp_state_path:
            raise RuntimeError("simulated tmp lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="failed to validate temporary state path"):
        save_state_atomic(run_dir, state)
    assert not (run_dir / "state.json").exists()
    assert not tmp_state_path.exists()


def test_save_state_atomic_ignores_directory_close_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_close_fail"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    state.status = "SUCCESS"
    original_close = os.close
    failed_once = False

    def flaky_close(fd: int) -> None:
        nonlocal failed_once
        original_close(fd)
        if not failed_once:
            failed_once = True
            raise OSError("simulated directory close failure")

    monkeypatch.setattr(os, "close", flaky_close)

    save_state_atomic(run_dir, state)
    loaded = load_state(run_dir)
    assert loaded.run_id == run_dir.name


def test_save_state_atomic_ignores_directory_close_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_close_runtime_fail"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    state.status = "SUCCESS"
    original_close = os.close
    failed_once = False

    def flaky_close(fd: int) -> None:
        nonlocal failed_once
        original_close(fd)
        if not failed_once:
            failed_once = True
            raise RuntimeError("simulated directory close runtime failure")

    monkeypatch.setattr(os, "close", flaky_close)

    save_state_atomic(run_dir, state)
    loaded = load_state(run_dir)
    assert loaded.run_id == run_dir.name


def test_save_state_atomic_uses_nofollow_for_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_nofollow_open"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    state.status = "SUCCESS"
    captured_flags: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if str(path) == str(run_dir):
            captured_flags["flags"] = flags
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)

    save_state_atomic(run_dir, state)
    assert "flags" in captured_flags
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_RDONLY
    if hasattr(os, "O_CREAT"):
        assert not (captured_flags["flags"] & os.O_CREAT)
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


def test_save_state_atomic_uses_nonblock_and_nofollow_for_tmp_state_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_dir_tmp_open_flags"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    tmp_state_path = run_dir / "state.json.tmp"
    captured_flags: dict[str, int] = {}
    captured_mode: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if str(path) == str(tmp_state_path):
            captured_flags["flags"] = flags
            captured_mode["mode"] = mode
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    save_state_atomic(run_dir, state)

    assert "flags" in captured_flags
    assert captured_flags["flags"] & os.O_WRONLY
    assert captured_flags["flags"] & os.O_CREAT
    assert captured_flags["flags"] & os.O_TRUNC
    assert captured_mode.get("mode") == 0o600
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


def test_load_state_uses_nonblock_and_nofollow_open_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".orch" / "runs" / "run_open_flags"
    run_dir.mkdir(parents=True)
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    state.status = "SUCCESS"
    state.home = str(run_dir.parent.parent.resolve())
    state.workdir = str(tmp_path.resolve())
    save_state_atomic(run_dir, state)

    state_path = run_dir / "state.json"
    captured_flags: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if str(path) == str(state_path):
            captured_flags["flags"] = flags
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    loaded = load_state(run_dir)
    assert loaded.run_id == run_dir.name
    assert "flags" in captured_flags
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_RDONLY
    if hasattr(os, "O_CREAT"):
        assert not (captured_flags["flags"] & os.O_CREAT)
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW


def test_save_state_atomic_preserves_write_error_when_tmp_cleanup_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_tmp_cleanup_runtime_fail"
    run_dir.mkdir()
    state = RunState.from_dict(_minimal_state_payload(run_id=run_dir.name))
    tmp_state_path = run_dir / "state.json.tmp"
    original_fsync = os.fsync
    original_unlink = Path.unlink
    failed_once = False

    def flaky_fsync(fd: int) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("simulated write fsync failure")
        original_fsync(fd)

    def flaky_unlink(path_obj: Path, *args: object, **kwargs: object) -> None:
        if path_obj == tmp_state_path:
            raise RuntimeError("simulated tmp cleanup runtime failure")
        original_unlink(path_obj, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", flaky_fsync)
    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    with pytest.raises(OSError, match="simulated write fsync failure"):
        save_state_atomic(run_dir, state)
    assert tmp_state_path.exists()


def test_load_state_rejects_invalid_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(StateError):
        load_state(run_dir)


def test_load_state_rejects_non_object_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_list"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(json.dumps(["not", "object"]), encoding="utf-8")

    with pytest.raises(StateError):
        load_state(run_dir)


def test_load_state_rejects_incomplete_object(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_incomplete"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{}", encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field"):
        load_state(run_dir)


def test_load_state_rejects_unknown_root_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_unknown_root_field"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "SUCCESS"
    payload["unexpected_root"] = "noise"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: root"):
        load_state(run_dir)


def test_load_state_rejects_unknown_task_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_unknown_task_field"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "SUCCESS"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["unexpected_task_field"] = "noise"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_invalid_timestamps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["created_at"] = "not-iso"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: created_at"):
        load_state(run_dir)


def test_load_state_rejects_unsafe_plan_relpath(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_plan_relpath"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["plan_relpath"] = "../plan.yaml"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: plan_relpath"):
        load_state(run_dir)


def test_load_state_rejects_non_string_goal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_goal"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["goal"] = ["invalid"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: goal"):
        load_state(run_dir)


def test_load_state_rejects_missing_goal_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_goal"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload.pop("goal", None)
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: goal"):
        load_state(run_dir)


def test_load_state_rejects_non_absolute_home_and_workdir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_paths"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["home"] = ".orch"
    payload["workdir"] = "."
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: home"):
        load_state(run_dir)


def test_load_state_rejects_non_canonical_home_path(tmp_path: Path) -> None:
    run_id = "run_bad_home_non_canonical"
    home_dir = tmp_path / ".orch"
    run_dir = home_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    home_link = tmp_path / "orch_link"
    home_link.symlink_to(home_dir, target_is_directory=True)
    payload = _minimal_state_payload(run_id=run_id)
    payload["home"] = str(home_link)
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: home"):
        load_state(run_dir)


def test_load_state_rejects_non_canonical_workdir_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_workdir_non_canonical"
    run_dir.mkdir()
    real_workdir = tmp_path / "real_wd"
    real_workdir.mkdir()
    link_workdir = tmp_path / "wd_link"
    link_workdir.symlink_to(real_workdir, target_is_directory=True)
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["workdir"] = str(link_workdir)
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: workdir"):
        load_state(run_dir)


def test_load_state_rejects_home_mismatch_with_run_directory(tmp_path: Path) -> None:
    run_id = "run_bad_home_mismatch"
    run_dir = tmp_path / ".orch" / "runs" / run_id
    run_dir.mkdir(parents=True)
    payload = _minimal_state_payload(run_id=run_id)
    payload["home"] = str((tmp_path / "other_home").resolve())
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="state home does not match directory"):
        load_state(run_dir)


def test_load_state_rejects_when_expected_home_resolve_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run_home_resolve_error"
    run_dir = tmp_path / ".orch" / "runs" / run_id
    run_dir.mkdir(parents=True)
    home_dir = tmp_path / ".orch"
    payload = _minimal_state_payload(run_id=run_id)
    payload["home"] = str(home_dir.resolve())
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    original_resolve = Path.resolve
    calls = 0

    def flaky_resolve(path_obj: Path, *args: object, **kwargs: object) -> Path:
        nonlocal calls
        if path_obj == home_dir:
            calls += 1
            if calls >= 2:
                raise PermissionError("simulated expected_home resolve failure")
        return original_resolve(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)

    with pytest.raises(StateError, match="invalid state field: home"):
        load_state(run_dir)


def test_load_state_rejects_naive_timestamp(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_naive_ts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["created_at"] = "2026-01-01T00:00:00"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: created_at"):
        load_state(run_dir)


def test_load_state_rejects_updated_at_earlier_than_created_at(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ts_order"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["created_at"] = "2026-01-02T00:00:00+00:00"
    payload["updated_at"] = "2026-01-01T00:00:00+00:00"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: updated_at"):
        load_state(run_dir)


def test_load_state_rejects_non_utf8_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_non_utf8"
    run_dir.mkdir()
    (run_dir / "state.json").write_bytes(b"\xff\xfe\xfd")

    with pytest.raises(StateError, match="failed to decode state file as utf-8"):
        load_state(run_dir)


def test_load_state_wraps_read_oserror(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_state_dir"
    run_dir.mkdir()
    (run_dir / "state.json").mkdir()

    with pytest.raises(StateError, match="failed to read state file"):
        load_state(run_dir)


def test_load_state_wraps_runtime_open_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_state_runtime_open_error"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    def _raise_runtime(_path: str, _flags: int) -> int:
        raise RuntimeError("simulated open runtime failure")

    monkeypatch.setattr(os, "open", _raise_runtime)

    with pytest.raises(StateError, match="failed to read state file"):
        load_state(run_dir)


def test_load_state_wraps_runtime_lstat_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_runtime_lstat"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    state_path = run_dir / "state.json"
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == state_path:
            raise RuntimeError("simulated state lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(StateError, match="failed to read state file"):
        load_state(run_dir)


def test_load_state_rejects_non_regular_state_file(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")

    run_dir = tmp_path / "run_state_fifo"
    run_dir.mkdir()
    os.mkfifo(run_dir / "state.json")

    with pytest.raises(StateError, match="failed to read state file"):
        load_state(run_dir)


def test_load_state_normalizes_open_eloop_as_symlink_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_state_eloop"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(_minimal_state_payload(run_id=run_dir.name)),
        encoding="utf-8",
    )

    def _raise_eloop(_path: str, _flags: int) -> int:
        raise OSError(errno.ELOOP, "Too many symbolic links")

    monkeypatch.setattr(os, "open", _raise_eloop)

    with pytest.raises(StateError, match="state file must not be symlink"):
        load_state(run_dir)


def test_load_state_rejects_symlink_state_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_state_symlink"
    run_dir.mkdir()
    real_state = tmp_path / "real_state.json"
    real_state.write_text(
        json.dumps(_minimal_state_payload(run_id=run_dir.name)),
        encoding="utf-8",
    )
    (run_dir / "state.json").symlink_to(real_state)

    with pytest.raises(StateError, match="state file must not be symlink"):
        load_state(run_dir)


def test_load_state_rejects_state_path_with_symlink_ancestor(tmp_path: Path) -> None:
    real_run_dir = tmp_path / "real_run"
    real_run_dir.mkdir()
    symlink_run_dir = tmp_path / "run_link"
    symlink_run_dir.symlink_to(real_run_dir, target_is_directory=True)
    payload = _minimal_state_payload(run_id=symlink_run_dir.name)
    (real_run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="contains symlink component"):
        load_state(symlink_run_dir)


def test_load_state_rejects_when_ancestor_stat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run_state_ancestor_error"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(_minimal_state_payload(run_id=run_dir.name)),
        encoding="utf-8",
    )
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == run_dir.parent:
            raise PermissionError("simulated ancestor lstat failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(StateError, match="contains symlink component"):
        load_state(run_dir)


def test_load_state_rejects_unsafe_task_log_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_log_path"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stdout_path"] = "../escape.log"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_missing_task_log_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_log_path"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stdout_path"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_log_path_for_different_task_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_log_task_binding"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stdout_path"] = "logs/t2.out.log"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_unexpected_log_filename_pattern(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_log_filename"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stderr_path"] = "logs/t1.custom.log"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_unsafe_artifact_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_artifact_path"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["artifact_paths"] = ["../outside.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_missing_artifact_paths_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_artifact_paths"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task.pop("artifact_paths", None)
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_artifact_path_for_different_task_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_artifact_task_binding"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["artifact_paths"] = ["artifacts/t2/out.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_duplicate_artifact_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dup_artifacts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["artifact_paths"] = ["artifacts/t1/out.txt", "artifacts/t1/OUT.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_case_insensitive_duplicate_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dup_outputs_case"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["outputs"] = ["dist/report.txt", "dist/REPORT.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_identical_stdout_and_stderr_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_same_log_paths"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["stdout_path"] = "logs/t1.out.log"
    task["stderr_path"] = "logs/t1.out.log"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_case_insensitive_duplicate_task_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_case_dup_tasks"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["tasks"] = {
        "Build": {
            "status": "SUCCESS",
            "depends_on": [],
            "cmd": ["echo", "ok"],
            "cwd": ".",
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
            "cmd": ["echo", "ok"],
            "cwd": ".",
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
    }
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_run_id_mismatch_with_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_mismatch"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id="different_run")
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="run_id does not match"):
        load_state(run_dir)


def test_load_state_rejects_negative_attempts_in_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["attempts"] = -1
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_non_finite_backoff_in_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_backoff"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["retry_backoff_sec"] = [0.1, float("inf")]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_backoff_longer_than_retries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_backoff_length"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["retries"] = 1
    task["retry_backoff_sec"] = [0.1, 0.2]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_missing_timestamps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_timestamps"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["started_at"] = None
    task["ended_at"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_with_skip_reason(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_skip_reason"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["skip_reason"] = "should_not_exist"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_without_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_no_exit_code"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["exit_code"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_without_duration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_no_duration"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["duration_sec"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_with_zero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_zero_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["attempts"] = 0
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_task_with_missing_bool_flags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_missing_bool_flags"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["timed_out"] = None
    task["canceled"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_without_started_at(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_started"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_with_terminal_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_terminal_fields"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_with_artifact_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_artifact_paths"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["artifact_paths"] = ["artifacts/t1/out.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_with_ended_at(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_ended_at"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_with_zero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 0
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_running_task_with_missing_bool_flags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_missing_bool_flags"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = None
    task["canceled"] = None
    task["skip_reason"] = None
    task["attempts"] = 1
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_accepts_ready_task_after_timeout_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_ready_after_timeout"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "READY"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = None
    task["timed_out"] = True
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_state(run_dir)
    assert loaded.status == "RUNNING"
    assert loaded.tasks["t1"].status == "READY"
    assert loaded.tasks["t1"].timed_out is True


def test_load_state_accepts_pending_task_after_timeout_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_pending_after_timeout"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = None
    task["timed_out"] = True
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_state(run_dir)
    assert loaded.status == "RUNNING"
    assert loaded.tasks["t1"].status == "PENDING"
    assert loaded.tasks["t1"].timed_out is True


def test_load_state_accepts_pending_task_after_non_timeout_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_pending_after_non_timeout_failure"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_state(run_dir)
    assert loaded.status == "RUNNING"
    assert loaded.tasks["t1"].status == "PENDING"
    assert loaded.tasks["t1"].exit_code == 1
    assert loaded.tasks["t1"].timed_out is False


def test_load_state_rejects_pending_timeout_task_with_zero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_pending_timeout_without_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = True
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 0
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_pending_timeout_task_without_timestamps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_pending_timeout_without_timestamps"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = 1.0
    task["exit_code"] = None
    task["timed_out"] = True
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_ready_task_with_missing_timed_out_flag(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ready_missing_timed_out_flag"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "READY"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 1
    task["timed_out"] = None
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_fresh_pending_task_with_runtime_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_pending_fresh_runtime_fields"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 0
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_pending_task_with_success_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_pending_success_exit"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 0
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_pending_task_with_skip_reason(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_pending_with_skip_reason"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = "previous_failure"
    task["attempts"] = 1
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_pending_task_with_missing_bool_flags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_pending_missing_bool_flags"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = None
    task["canceled"] = None
    task["skip_reason"] = None
    task["attempts"] = 0
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_ready_task_with_attempts_exhausted(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ready_attempts_exhausted"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "READY"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 2
    task["retries"] = 1
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_ready_task_without_duration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ready_without_duration"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "READY"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = None
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_ready_task_with_success_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_ready_success_exit"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "READY"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 0
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 1
    task["retries"] = 2
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_failed_timeout_task_with_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_timeout_exit_code"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "FAILED"
    task["exit_code"] = 124
    task["timed_out"] = True
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_inconsistent_task_exit_and_flags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_task_flags"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "SUCCESS"
    task["exit_code"] = 1
    task["timed_out"] = True
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_failed_task_with_zero_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_zero_exit"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "FAILED"
    task["exit_code"] = 0
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_failed_task_with_zero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_zero_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "FAILED"
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["attempts"] = 0
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_failed_task_with_missing_timed_out_flag(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_missing_timed_out_flag"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "FAILED"
    task["exit_code"] = 1
    task["timed_out"] = None
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_skipped_task_with_runtime_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_skipped_runtime_fields"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "SKIPPED"
    task["skip_reason"] = "dependency_not_success"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_skipped_task_with_nonzero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_skipped_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "SKIPPED"
    task["skip_reason"] = "dependency_not_success"
    task["started_at"] = None
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["attempts"] = 1
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_skipped_task_with_missing_bool_flags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_skipped_missing_bool_flags"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "SKIPPED"
    task["skip_reason"] = "dependency_not_success"
    task["started_at"] = None
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = None
    task["canceled"] = None
    task["attempts"] = 0
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_with_missing_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["attempts"] = None
    task["retries"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["skip_reason"] = None
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_with_missing_retries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_retries"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task["attempts"] = 0
    task["retries"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["skip_reason"] = None
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_with_missing_timeout_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_timeout_field"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task.pop("timeout_sec", None)
    task["attempts"] = 0
    task["retries"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["skip_reason"] = None
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_with_missing_retry_backoff_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_retry_backoff_field"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task.pop("retry_backoff_sec", None)
    task["attempts"] = 0
    task["retries"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["skip_reason"] = None
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_with_missing_cwd_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_cwd_field"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task.pop("cwd", None)
    task["attempts"] = 0
    task["retries"] = 1
    task["timeout_sec"] = None
    task["retry_backoff_sec"] = []
    task["timed_out"] = False
    task["canceled"] = False
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["skip_reason"] = None
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_with_missing_env_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_env_field"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task.pop("env", None)
    task["attempts"] = 0
    task["retries"] = 1
    task["timeout_sec"] = None
    task["retry_backoff_sec"] = []
    task["timed_out"] = False
    task["canceled"] = False
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["skip_reason"] = None
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_with_missing_started_at_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_started_at_field"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task.pop("started_at", None)
    task["attempts"] = 0
    task["retries"] = 1
    task["timeout_sec"] = None
    task["retry_backoff_sec"] = []
    task["timed_out"] = False
    task["canceled"] = False
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["skip_reason"] = None
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_with_missing_skip_reason_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_missing_skip_reason_field"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "PENDING"
    task.pop("skip_reason", None)
    task["attempts"] = 0
    task["retries"] = 1
    task["timeout_sec"] = None
    task["retry_backoff_sec"] = []
    task["timed_out"] = False
    task["canceled"] = False
    task["started_at"] = None
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_attempts_exceeding_retry_budget(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_attempt_budget"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["retries"] = 0
    task["attempts"] = 3
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_invalid_task_timestamps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_task_timestamps"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["started_at"] = "not-iso"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_task_end_before_start(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_task_time_order"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["started_at"] = "2026-01-02T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:00+00:00"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_unknown_dependency_in_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_dep_ref"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["depends_on"] = ["missing"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_success_status_with_non_success_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_success_status"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "SUCCESS"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "FAILED"
    task["exit_code"] = 1
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_canceled_status_without_canceled_tasks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_canceled_status"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "CANCELED"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_canceled_task_with_no_start_and_runtime_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_canceled_runtime_without_start"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "CANCELED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "CANCELED"
    task["attempts"] = 1
    task["started_at"] = None
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = True
    task["skip_reason"] = "run_canceled"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_started_canceled_task_with_zero_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_canceled_started_zero_attempts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "CANCELED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "CANCELED"
    task["attempts"] = 0
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 130
    task["timed_out"] = False
    task["canceled"] = True
    task["skip_reason"] = "run_canceled"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_started_canceled_task_without_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_canceled_started_no_exit"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "CANCELED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "CANCELED"
    task["attempts"] = 1
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = True
    task["skip_reason"] = "run_canceled"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_started_canceled_task_without_duration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_canceled_started_no_duration"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "CANCELED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "CANCELED"
    task["attempts"] = 1
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = None
    task["exit_code"] = 130
    task["timed_out"] = False
    task["canceled"] = True
    task["skip_reason"] = "run_canceled"
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_canceled_task_with_artifact_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_canceled_artifacts"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "CANCELED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "CANCELED"
    task["attempts"] = 1
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 130
    task["timed_out"] = False
    task["canceled"] = True
    task["skip_reason"] = "run_canceled"
    task["artifact_paths"] = ["artifacts/t1/out.txt"]
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_canceled_task_with_zero_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_canceled_zero_exit"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "CANCELED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "CANCELED"
    task["attempts"] = 1
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 0
    task["timed_out"] = False
    task["canceled"] = True
    task["skip_reason"] = "run_canceled"
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_canceled_task_with_missing_timed_out_flag(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_canceled_missing_timed_out_flag"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "CANCELED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "CANCELED"
    task["attempts"] = 1
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = "2026-01-01T00:00:01+00:00"
    task["duration_sec"] = 1.0
    task["exit_code"] = 130
    task["timed_out"] = None
    task["canceled"] = True
    task["skip_reason"] = "run_canceled"
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: tasks"):
        load_state(run_dir)


def test_load_state_rejects_terminal_run_status_with_running_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_with_running"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    task = tasks["t1"]
    assert isinstance(task, dict)
    task["status"] = "RUNNING"
    task["started_at"] = "2026-01-01T00:00:00+00:00"
    task["ended_at"] = None
    task["duration_sec"] = None
    task["exit_code"] = None
    task["timed_out"] = False
    task["canceled"] = False
    task["skip_reason"] = None
    task["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_running_status_when_all_tasks_terminal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_running_all_terminal"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "RUNNING"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_pending_run_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_pending_status"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "PENDING"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_failed_run_status_with_canceled_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_with_canceled"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    t1 = tasks["t1"]
    assert isinstance(t1, dict)
    t1["status"] = "FAILED"
    t1["exit_code"] = 1
    t1["timed_out"] = False
    t1["canceled"] = False
    t1["skip_reason"] = None
    tasks["t2"] = {
        "status": "CANCELED",
        "depends_on": [],
        "cmd": ["echo", "ok"],
        "cwd": ".",
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
    }
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)


def test_load_state_rejects_failed_run_status_without_failed_tasks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_failed_without_failed_tasks"
    run_dir.mkdir()
    payload = _minimal_state_payload(run_id=run_dir.name)
    payload["status"] = "FAILED"
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    t1 = tasks["t1"]
    assert isinstance(t1, dict)
    t1["status"] = "SKIPPED"
    t1["attempts"] = 0
    t1["started_at"] = None
    t1["ended_at"] = "2026-01-01T00:00:01+00:00"
    t1["duration_sec"] = None
    t1["exit_code"] = None
    t1["timed_out"] = False
    t1["canceled"] = False
    t1["skip_reason"] = "upstream_failed"
    t1["artifact_paths"] = []
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError, match="invalid state field: status"):
        load_state(run_dir)
