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


def test_is_symlink_path_fails_closed_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "target"
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == path:
            raise OSError("simulated is_symlink os error")
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", flaky_is_symlink)

    assert is_symlink_path(path) is True


def test_is_symlink_path_can_fail_open_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "target"
    original_is_symlink = Path.is_symlink

    def flaky_is_symlink(path_obj: Path) -> bool:
        if path_obj == path:
            raise OSError("simulated is_symlink os error")
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


def test_has_symlink_ancestor_fails_closed_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "a"
    parent.mkdir()
    path = parent / "file.txt"
    original_lstat = Path.lstat

    def flaky_lstat(path_obj: Path, *args: object, **kwargs: object) -> object:
        if path_obj == parent:
            raise OSError("simulated lstat os error")
        return original_lstat(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    assert has_symlink_ancestor(path) is True


def test_is_symlink_path_returns_false_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    assert is_symlink_path(missing) is False


def test_has_symlink_ancestor_detects_real_symlink_component(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "parent_link"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    target = symlink_parent / "child" / "file.txt"

    assert has_symlink_ancestor(target) is True


def test_has_symlink_ancestor_returns_false_when_no_symlink_in_ancestors(
    tmp_path: Path,
) -> None:
    target = tmp_path / "a" / "b" / "file.txt"

    assert has_symlink_ancestor(target) is False
