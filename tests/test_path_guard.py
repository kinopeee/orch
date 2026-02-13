from __future__ import annotations

from pathlib import Path

import pytest

from orch.util.path_guard import has_symlink_ancestor, is_symlink_path


def test_is_symlink_path_fails_closed_on_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "target"
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == path:
            raise RuntimeError("simulated is_symlink runtime failure")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    assert is_symlink_path(path) is True


def test_is_symlink_path_can_fail_open_on_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "target"
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == path:
            raise RuntimeError("simulated is_symlink runtime failure")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    assert is_symlink_path(path, fail_closed=False) is False


def test_has_symlink_ancestor_fails_closed_on_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "a"
    parent.mkdir()
    path = parent / "file.txt"
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> object:
        if path_obj == parent:
            raise RuntimeError("simulated lstat runtime failure")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    assert has_symlink_ancestor(path) is True
