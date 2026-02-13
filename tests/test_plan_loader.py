from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from orch.config.loader import load_plan
from orch.util.errors import PlanError


def test_load_plan_normalizes_string_cmd(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
goal: test
tasks:
  - id: a
    cmd: "python -c 'print(1)'"
""".strip(),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)
    assert plan.tasks[0].cmd == ["python", "-c", "print(1)"]


def test_load_plan_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: a
    cmd: ["echo", "x"]
  - id: a
    cmd: ["echo", "y"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(PlanError):
        load_plan(plan_path)


def test_load_plan_rejects_case_insensitive_duplicate_task_ids(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan_case_dup.yaml"
    plan_path.write_text(
        """
tasks:
  - id: Build
    cmd: ["echo", "x"]
  - id: build
    cmd: ["echo", "y"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(PlanError):
        load_plan(plan_path)


def test_load_plan_rejects_unreadable_path_like_directory(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan_dir"
    plan_dir.mkdir()

    with pytest.raises(PlanError, match="failed to read plan file"):
        load_plan(plan_dir)


def test_load_plan_rejects_invalid_yaml_syntax(tmp_path: Path) -> None:
    plan_path = tmp_path / "bad.yaml"
    plan_path.write_text("tasks: [\n", encoding="utf-8")

    with pytest.raises(PlanError, match="failed to parse yaml"):
        load_plan(plan_path)


def test_load_plan_rejects_non_utf8_file(tmp_path: Path) -> None:
    plan_path = tmp_path / "bad_encoding.yaml"
    plan_path.write_bytes(b"\xff\xfe\xfd")

    with pytest.raises(PlanError, match="failed to decode plan file as utf-8"):
        load_plan(plan_path)


def test_load_plan_rejects_symlink_path(tmp_path: Path) -> None:
    real_plan = tmp_path / "real_plan.yaml"
    real_plan.write_text(
        """
tasks:
  - id: a
    cmd: ["echo", "x"]
""".strip(),
        encoding="utf-8",
    )
    symlink_plan = tmp_path / "plan_symlink.yaml"
    symlink_plan.symlink_to(real_plan)

    with pytest.raises(PlanError, match="plan file must not be symlink"):
        load_plan(symlink_plan)


def test_load_plan_rejects_path_with_symlink_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    plan_path = real_parent / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: a
    cmd: ["echo", "x"]
""".strip(),
        encoding="utf-8",
    )
    symlink_parent = tmp_path / "link_parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(PlanError, match="contains symlink component"):
        load_plan(symlink_parent / "plan.yaml")


def test_load_plan_rejects_when_ancestor_stat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: a
    cmd: ["echo", "x"]
""".strip(),
        encoding="utf-8",
    )
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == plan_path.parent:
            raise PermissionError("simulated ancestor lstat failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(PlanError, match="contains symlink component"):
        load_plan(plan_path)


def test_load_plan_wraps_runtime_lstat_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan_runtime_lstat.yaml"
    plan_path.write_text(
        """
tasks:
  - id: a
    cmd: ["echo", "x"]
""".strip(),
        encoding="utf-8",
    )
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path_obj == plan_path:
            raise RuntimeError("simulated plan lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(PlanError, match="failed to read plan file"):
        load_plan(plan_path)


def test_load_plan_rejects_non_regular_file_path(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not supported on this platform")

    plan_path = tmp_path / "plan_fifo.yaml"
    os.mkfifo(plan_path)

    with pytest.raises(PlanError, match="failed to read plan file"):
        load_plan(plan_path)


def test_load_plan_wraps_runtime_open_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan_runtime_open.yaml"
    plan_path.write_text(
        """
tasks:
  - id: a
    cmd: ["echo", "x"]
""".strip(),
        encoding="utf-8",
    )

    def _raise_runtime(_path: str, _flags: int) -> int:
        raise RuntimeError("simulated open runtime failure")

    monkeypatch.setattr(os, "open", _raise_runtime)

    with pytest.raises(PlanError, match="failed to read plan file"):
        load_plan(plan_path)


def test_load_plan_normalizes_open_eloop_as_symlink_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: a
    cmd: ["echo", "x"]
""".strip(),
        encoding="utf-8",
    )

    def _raise_eloop(_path: str, _flags: int) -> int:
        raise OSError(errno.ELOOP, "Too many symbolic links")

    monkeypatch.setattr(os, "open", _raise_eloop)

    with pytest.raises(PlanError, match="plan file must not be symlink"):
        load_plan(plan_path)


def test_load_plan_uses_nonblock_and_nofollow_open_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
tasks:
  - id: a
    cmd: ["echo", "x"]
""".strip(),
        encoding="utf-8",
    )
    captured_flags: dict[str, int] = {}
    original_open = os.open

    def capture_open(path: str, flags: int) -> int:
        if path == str(plan_path):
            captured_flags["flags"] = flags
        return original_open(path, flags)

    monkeypatch.setattr(os, "open", capture_open)
    load_plan(plan_path)

    assert "flags" in captured_flags
    if hasattr(os, "O_ACCMODE"):
        assert captured_flags["flags"] & os.O_ACCMODE == os.O_RDONLY
    if hasattr(os, "O_CREAT"):
        assert not (captured_flags["flags"] & os.O_CREAT)
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags["flags"] & os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags["flags"] & os.O_NOFOLLOW
