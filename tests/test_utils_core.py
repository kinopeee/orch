from __future__ import annotations

import ast
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orch.util.ids import new_run_id
from orch.util.paths import ensure_run_layout, run_dir
from orch.util.time import duration_sec, now_iso


def test_new_run_id_format_includes_timestamp_and_suffix() -> None:
    now = datetime(2026, 2, 13, 12, 34, 56, tzinfo=UTC)
    run_id = new_run_id(now)
    assert run_id.startswith("20260213_123456_")
    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", run_id) is not None


def test_now_iso_returns_timezone_aware_string() -> None:
    value = now_iso()
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None


def test_duration_sec_rounds_to_three_decimals() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    end = start + timedelta(seconds=1.23456)
    assert duration_sec(start, end) == 1.235


def test_run_dir_and_ensure_run_layout_create_expected_paths(tmp_path: Path) -> None:
    path = run_dir(tmp_path, "run_1")
    ensure_run_layout(path)
    assert path == tmp_path / "runs" / "run_1"
    assert (path / "logs").is_dir()
    assert (path / "artifacts").is_dir()
    assert (path / "report").is_dir()


def test_ensure_run_layout_rejects_symlink_run_dir(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    target = tmp_path / "external_run"
    target.mkdir()
    path = runs_dir / "run_1"
    path.symlink_to(target, target_is_directory=True)

    with pytest.raises(OSError, match="must not be symlink"):
        ensure_run_layout(path)


def test_ensure_run_layout_rejects_symlink_logs_dir(tmp_path: Path) -> None:
    path = run_dir(tmp_path, "run_1")
    path.mkdir(parents=True)
    external_logs = tmp_path / "external_logs"
    external_logs.mkdir()
    (path / "logs").symlink_to(external_logs, target_is_directory=True)

    with pytest.raises(OSError, match="must not be symlink"):
        ensure_run_layout(path)
    assert list(external_logs.iterdir()) == []


def test_ensure_run_layout_rejects_symlink_ancestor(tmp_path: Path) -> None:
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    linked_home = tmp_path / "home_link"
    linked_home.symlink_to(real_home, target_is_directory=True)
    path = run_dir(linked_home, "run_1")

    with pytest.raises(OSError, match="contains symlink component"):
        ensure_run_layout(path)
    assert not (real_home / "runs").exists()


def test_ensure_run_layout_normalizes_lstat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = run_dir(tmp_path, "run_1")
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == path:
            raise PermissionError("simulated lstat failure")
        return original_lstat(path_obj)

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj == path:
            return False
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="path must be directory"):
        ensure_run_layout(path)


def test_ensure_run_layout_normalizes_lstat_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = run_dir(tmp_path, "run_1")
    original_lstat = Path.lstat
    original_is_symlink = Path.is_symlink

    def flaky_lstat(path_obj: Path) -> os.stat_result:
        if path_obj == path:
            raise RuntimeError("simulated lstat runtime failure")
        return original_lstat(path_obj)

    def fake_is_symlink(path_obj: Path) -> bool:
        if path_obj == path:
            return False
        return original_is_symlink(path_obj)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    with pytest.raises(OSError, match="path must be directory"):
        ensure_run_layout(path)


def test_source_does_not_use_direct_exists_is_file_is_dir_predicates() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    predicate_patterns = (".exists(", ".is_file(", ".is_dir(")
    violations: list[str] = []

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if any(pattern in line for pattern in predicate_patterns):
                violations.append(f"{relative_path}:{line_number}: {stripped}")

    assert not violations, "direct Path predicate usage found:\n" + "\n".join(violations)


def test_source_uses_path_guard_for_is_symlink_checks() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        if relative_path == Path("util/path_guard.py"):
            continue
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if ".is_symlink(" in line:
                violations.append(f"{relative_path}:{line_number}: {stripped}")

    assert not violations, "direct is_symlink usage found outside path_guard:\n" + "\n".join(
        violations
    )


def _collect_unguarded_calls(
    method_name: str,
    *,
    receiver_name: str | None = None,
    allow_suppress_guard: bool = False,
) -> list[str]:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    class GuardVisitor(ast.NodeVisitor):
        def __init__(self, relative_path: Path) -> None:
            self.relative_path = relative_path
            self.guard_stack: list[bool] = []
            self.suppress_guard_stack: list[bool] = []

        @staticmethod
        def _covers(name: str, handler_type: ast.expr | None) -> bool:
            if handler_type is None:
                return True
            if isinstance(handler_type, ast.Name):
                return handler_type.id in {name, "Exception", "BaseException"}
            if isinstance(handler_type, ast.Attribute):
                return handler_type.attr in {name, "Exception", "BaseException"}
            if isinstance(handler_type, ast.Tuple):
                return any(GuardVisitor._covers(name, elem) for elem in handler_type.elts)
            return False

        def visit_Try(self, node: ast.Try) -> None:
            handles_oserror = any(
                self._covers("OSError", handler.type) for handler in node.handlers
            )
            handles_runtime = any(
                self._covers("RuntimeError", handler.type) for handler in node.handlers
            )
            self.guard_stack.append(handles_oserror and handles_runtime)
            for stmt in node.body:
                self.visit(stmt)
            self.guard_stack.pop()
            for handler in node.handlers:
                self.visit(handler)
            for stmt in node.orelse:
                self.visit(stmt)
            for stmt in node.finalbody:
                self.visit(stmt)

        def visit_Call(self, node: ast.Call) -> None:
            matches_receiver = True
            if receiver_name is not None:
                matches_receiver = (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == receiver_name
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == method_name
                and matches_receiver
                and not (any(self.guard_stack) or any(self.suppress_guard_stack))
            ):
                violations.append(f"{self.relative_path}:{node.lineno}")
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:
            suppresses = False
            if allow_suppress_guard:
                for item in node.items:
                    context_expr = item.context_expr
                    if not isinstance(context_expr, ast.Call):
                        continue
                    func = context_expr.func
                    if isinstance(func, ast.Name) and func.id == "suppress":
                        handles_oserror = any(
                            self._covers("OSError", handler_type)
                            for handler_type in context_expr.args
                        )
                        handles_runtime = any(
                            self._covers("RuntimeError", handler_type)
                            for handler_type in context_expr.args
                        )
                        suppresses = handles_oserror and handles_runtime
                        if suppresses:
                            break
            self.suppress_guard_stack.append(suppresses)
            for stmt in node.body:
                self.visit(stmt)
            self.suppress_guard_stack.pop()
            for item in node.items:
                self.visit(item)

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            suppresses = False
            if allow_suppress_guard:
                for item in node.items:
                    context_expr = item.context_expr
                    if not isinstance(context_expr, ast.Call):
                        continue
                    func = context_expr.func
                    if isinstance(func, ast.Name) and func.id == "suppress":
                        handles_oserror = any(
                            self._covers("OSError", handler_type)
                            for handler_type in context_expr.args
                        )
                        handles_runtime = any(
                            self._covers("RuntimeError", handler_type)
                            for handler_type in context_expr.args
                        )
                        suppresses = handles_oserror and handles_runtime
                        if suppresses:
                            break
            self.suppress_guard_stack.append(suppresses)
            for stmt in node.body:
                self.visit(stmt)
            self.suppress_guard_stack.pop()
            for item in node.items:
                self.visit(item)

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))
        GuardVisitor(relative_path).visit(module)

    return violations


def test_source_wraps_lstat_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("lstat")

    assert not violations, "unguarded lstat calls found:\n" + "\n".join(violations)


def test_source_wraps_resolve_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("resolve")

    assert not violations, "unguarded resolve calls found:\n" + "\n".join(violations)


def test_source_wraps_os_open_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("open", receiver_name="os")

    assert not violations, "unguarded os.open calls found:\n" + "\n".join(violations)


def test_source_wraps_os_fstat_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("fstat", receiver_name="os")

    assert not violations, "unguarded os.fstat calls found:\n" + "\n".join(violations)


def test_source_wraps_os_replace_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("replace", receiver_name="os")

    assert not violations, "unguarded os.replace calls found:\n" + "\n".join(violations)


def test_source_guards_unlink_calls_with_try_or_suppress() -> None:
    violations = _collect_unguarded_calls("unlink", allow_suppress_guard=True)

    assert not violations, "unguarded unlink calls found:\n" + "\n".join(violations)


def test_source_guards_os_close_calls_with_try_or_suppress() -> None:
    violations = _collect_unguarded_calls("close", receiver_name="os", allow_suppress_guard=True)

    assert not violations, "unguarded os.close calls found:\n" + "\n".join(violations)
