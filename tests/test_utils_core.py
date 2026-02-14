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


def _collect_direct_attribute_calls(
    method_names: set[str], *, excluded_relative_paths: set[Path] | None = None
) -> list[str]:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    excluded = excluded_relative_paths or set()
    violations: list[str] = []

    class DirectCallVisitor(ast.NodeVisitor):
        def __init__(self, relative_path: Path) -> None:
            self.relative_path = relative_path

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr in method_names:
                violations.append(f"{self.relative_path}:{node.lineno}: {node.func.attr}()")
            self.generic_visit(node)

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        if relative_path in excluded:
            continue
        module = ast.parse(file_path.read_text(encoding="utf-8"))
        DirectCallVisitor(relative_path).visit(module)

    return violations


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


def test_ensure_run_layout_normalizes_mkdir_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = run_dir(tmp_path, "run_1")
    original_mkdir = Path.mkdir
    target_path = path

    def flaky_mkdir(path_obj: Path, *args: object, **kwargs: object) -> None:
        if path_obj == target_path:
            raise PermissionError("simulated mkdir failure")
        original_mkdir(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)

    with pytest.raises(OSError, match="failed to create directory path"):
        ensure_run_layout(path)


def test_ensure_run_layout_normalizes_mkdir_runtime_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = run_dir(tmp_path, "run_1")
    original_mkdir = Path.mkdir
    target_path = path

    def flaky_mkdir(path_obj: Path, *args: object, **kwargs: object) -> None:
        if path_obj == target_path:
            raise RuntimeError("simulated mkdir runtime failure")
        original_mkdir(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)

    with pytest.raises(OSError, match="failed to create directory path"):
        ensure_run_layout(path)


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
    violations = _collect_direct_attribute_calls({"exists", "is_file", "is_dir"})

    assert not violations, "direct Path predicate usage found:\n" + "\n".join(violations)


def test_source_uses_path_guard_for_is_symlink_checks() -> None:
    violations = _collect_direct_attribute_calls(
        {"is_symlink"},
        excluded_relative_paths={Path("util/path_guard.py")},
    )

    assert not violations, "direct is_symlink usage found outside path_guard:\n" + "\n".join(
        violations
    )


def _collect_except_type_names(handler_type: ast.expr | None) -> set[str]:
    if handler_type is None:
        return set()
    if isinstance(handler_type, ast.Name):
        return {handler_type.id}
    if isinstance(handler_type, ast.Attribute):
        return {handler_type.attr}
    if isinstance(handler_type, ast.Tuple):
        names: set[str] = set()
        for elem in handler_type.elts:
            names.update(_collect_except_type_names(elem))
        return names
    return set()


def test_source_path_guard_is_symlink_path_catches_oserror_and_runtimeerror() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "orch" / "util" / "path_guard.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    is_symlink_func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "is_symlink_path"
    )
    try_nodes = [node for node in ast.walk(is_symlink_func) if isinstance(node, ast.Try)]
    assert try_nodes, "is_symlink_path should wrap predicate in try/except"
    except_type_names = {
        name
        for try_node in try_nodes
        for handler in try_node.handlers
        for name in _collect_except_type_names(handler.type)
    }
    assert "OSError" in except_type_names
    assert "RuntimeError" in except_type_names


def test_source_path_guard_has_symlink_ancestor_catches_oserror_and_runtimeerror() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "orch" / "util" / "path_guard.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    has_ancestor_func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "has_symlink_ancestor"
    )
    try_nodes = [node for node in ast.walk(has_ancestor_func) if isinstance(node, ast.Try)]
    assert try_nodes, "has_symlink_ancestor should wrap ancestor walk in try/except"
    except_type_names = {
        name
        for try_node in try_nodes
        for handler in try_node.handlers
        for name in _collect_except_type_names(handler.type)
    }
    assert "OSError" in except_type_names
    assert "RuntimeError" in except_type_names


def test_source_path_guard_is_symlink_path_uses_fail_closed_return_on_error() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "orch" / "util" / "path_guard.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    is_symlink_func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "is_symlink_path"
    )
    try_nodes = [node for node in ast.walk(is_symlink_func) if isinstance(node, ast.Try)]
    assert try_nodes, "is_symlink_path should include guarded try/except"

    handler = next(
        (
            candidate
            for try_node in try_nodes
            for candidate in try_node.handlers
            if {"OSError", "RuntimeError"}.issubset(_collect_except_type_names(candidate.type))
        ),
        None,
    )
    assert handler is not None, "is_symlink_path must catch OSError and RuntimeError together"
    assert handler.body, "is_symlink_path error handler should not be empty"
    return_stmt = handler.body[0]
    assert isinstance(return_stmt, ast.Return)
    assert isinstance(return_stmt.value, ast.Name)
    assert return_stmt.value.id == "fail_closed"


def test_source_path_guard_has_symlink_ancestor_returns_true_on_error() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "orch" / "util" / "path_guard.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    has_ancestor_func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "has_symlink_ancestor"
    )
    try_nodes = [node for node in ast.walk(has_ancestor_func) if isinstance(node, ast.Try)]
    assert try_nodes, "has_symlink_ancestor should include guarded try/except"

    handler = next(
        (
            candidate
            for try_node in try_nodes
            for candidate in try_node.handlers
            if {"OSError", "RuntimeError"}.issubset(_collect_except_type_names(candidate.type))
        ),
        None,
    )
    assert handler is not None, "has_symlink_ancestor must catch OSError and RuntimeError together"
    assert handler.body, "has_symlink_ancestor error handler should not be empty"
    return_stmt = handler.body[0]
    assert isinstance(return_stmt, ast.Return)
    assert isinstance(return_stmt.value, ast.Constant)
    assert return_stmt.value.value is True


def test_source_path_guard_is_symlink_path_returns_false_on_filenotfound() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "orch" / "util" / "path_guard.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    is_symlink_func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "is_symlink_path"
    )
    try_nodes = [node for node in ast.walk(is_symlink_func) if isinstance(node, ast.Try)]
    assert try_nodes, "is_symlink_path should include guarded try/except"

    handler = next(
        (
            candidate
            for try_node in try_nodes
            for candidate in try_node.handlers
            if "FileNotFoundError" in _collect_except_type_names(candidate.type)
        ),
        None,
    )
    assert handler is not None, "is_symlink_path must explicitly handle FileNotFoundError"
    assert handler.body, "FileNotFoundError handler should not be empty"
    return_stmt = handler.body[0]
    assert isinstance(return_stmt, ast.Return)
    assert isinstance(return_stmt.value, ast.Constant)
    assert return_stmt.value.value is False


def test_source_path_guard_has_symlink_ancestor_ignores_filenotfound() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "orch" / "util" / "path_guard.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    has_ancestor_func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "has_symlink_ancestor"
    )
    try_nodes = [node for node in ast.walk(has_ancestor_func) if isinstance(node, ast.Try)]
    assert try_nodes, "has_symlink_ancestor should include guarded try/except"

    handler = next(
        (
            candidate
            for try_node in try_nodes
            for candidate in try_node.handlers
            if "FileNotFoundError" in _collect_except_type_names(candidate.type)
        ),
        None,
    )
    assert handler is not None, "has_symlink_ancestor must explicitly handle FileNotFoundError"
    assert handler.body, "FileNotFoundError handler should not be empty"
    assert len(handler.body) == 1
    assert isinstance(handler.body[0], ast.Pass)


def test_source_path_guard_has_symlink_ancestor_starts_from_parent() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "orch" / "util" / "path_guard.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    has_ancestor_func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "has_symlink_ancestor"
    )
    assign_stmt = next(
        (
            stmt
            for stmt in has_ancestor_func.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "current" for target in stmt.targets
            )
        ),
        None,
    )
    assert assign_stmt is not None
    assert isinstance(assign_stmt.value, ast.Attribute)
    assert isinstance(assign_stmt.value.value, ast.Name)
    assert assign_stmt.value.value.id == "path"
    assert assign_stmt.value.attr == "parent"


def test_source_path_guard_has_symlink_ancestor_has_root_termination_guard() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "orch" / "util" / "path_guard.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    has_ancestor_func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "has_symlink_ancestor"
    )

    termination_if = next(
        (
            stmt
            for stmt in ast.walk(has_ancestor_func)
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and len(stmt.test.ops) == 1
            and isinstance(stmt.test.ops[0], ast.Eq)
            and isinstance(stmt.test.left, ast.Name)
            and stmt.test.left.id == "current"
            and len(stmt.test.comparators) == 1
            and isinstance(stmt.test.comparators[0], ast.Attribute)
            and isinstance(stmt.test.comparators[0].value, ast.Name)
            and stmt.test.comparators[0].value.id == "current"
            and stmt.test.comparators[0].attr == "parent"
        ),
        None,
    )
    assert termination_if is not None
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in termination_if.body
    )


def _collect_unguarded_calls(
    method_name: str,
    *,
    receiver_name: str | None = None,
    allow_suppress_guard: bool = False,
    required_exceptions: tuple[str, ...] = ("OSError", "RuntimeError"),
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
            handles_required = all(
                any(self._covers(exc_name, handler.type) for handler in node.handlers)
                for exc_name in required_exceptions
            )
            self.guard_stack.append(handles_required)
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


def test_source_wraps_os_fdopen_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("fdopen", receiver_name="os")

    assert not violations, "unguarded os.fdopen calls found:\n" + "\n".join(violations)


def test_source_wraps_mkdir_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("mkdir")

    assert not violations, "unguarded mkdir calls found:\n" + "\n".join(violations)


def test_source_wraps_subprocess_creation_calls_with_expected_handlers() -> None:
    violations = _collect_unguarded_calls(
        "create_subprocess_exec",
        receiver_name="asyncio",
        required_exceptions=("OSError", "RuntimeError", "ValueError"),
    )

    assert not violations, "unguarded asyncio.create_subprocess_exec calls found:\n" + "\n".join(
        violations
    )


def test_source_wraps_glob_calls_with_expected_handlers() -> None:
    violations = _collect_unguarded_calls(
        "glob",
        required_exceptions=("OSError", "RuntimeError", "ValueError"),
    )

    assert not violations, "unguarded glob calls found:\n" + "\n".join(violations)


def test_source_wraps_os_write_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("write", receiver_name="os")

    assert not violations, "unguarded os.write calls found:\n" + "\n".join(violations)


def test_source_wraps_os_fsync_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("fsync", receiver_name="os")

    assert not violations, "unguarded os.fsync calls found:\n" + "\n".join(violations)


def test_source_wraps_os_replace_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("replace", receiver_name="os")

    assert not violations, "unguarded os.replace calls found:\n" + "\n".join(violations)


def test_source_wraps_shutil_copy2_calls_with_oserror_and_runtimeerror_handlers() -> None:
    violations = _collect_unguarded_calls("copy2", receiver_name="shutil")

    assert not violations, "unguarded shutil.copy2 calls found:\n" + "\n".join(violations)


def test_source_guards_unlink_calls_with_try_or_suppress() -> None:
    violations = _collect_unguarded_calls("unlink", allow_suppress_guard=True)

    assert not violations, "unguarded unlink calls found:\n" + "\n".join(violations)


def test_source_guards_os_close_calls_with_try_or_suppress() -> None:
    violations = _collect_unguarded_calls("close", receiver_name="os", allow_suppress_guard=True)

    assert not violations, "unguarded os.close calls found:\n" + "\n".join(violations)


def test_source_oserror_handlers_also_cover_runtimeerror() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    def _covers(name: str, handler_type: ast.expr | None) -> bool:
        if handler_type is None:
            return True
        if isinstance(handler_type, ast.Name):
            return handler_type.id in {name, "Exception", "BaseException"}
        if isinstance(handler_type, ast.Attribute):
            return handler_type.attr in {name, "Exception", "BaseException"}
        if isinstance(handler_type, ast.Tuple):
            return any(_covers(name, elem) for elem in handler_type.elts)
        return False

    class HandlerVisitor(ast.NodeVisitor):
        def __init__(self, relative_path: Path) -> None:
            self.relative_path = relative_path

        def visit_Try(self, node: ast.Try) -> None:
            handles_oserror = any(_covers("OSError", handler.type) for handler in node.handlers)
            if handles_oserror:
                handles_runtime = any(
                    _covers("RuntimeError", handler.type) for handler in node.handlers
                )
                if not handles_runtime:
                    violations.append(f"{self.relative_path}:{node.lineno}")
            self.generic_visit(node)

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))
        HandlerVisitor(relative_path).visit(module)

    assert not violations, "try blocks with OSError-only handling found:\n" + "\n".join(violations)


def test_source_three_arg_os_open_calls_use_secure_mode() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "open"
            ):
                continue
            if len(node.args) < 3:
                continue
            mode_arg = node.args[2]
            if not (isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, int)):
                violations.append(f"{relative_path}:{node.lineno}: non-literal open mode")
                continue
            if mode_arg.value != 0o600:
                violations.append(
                    f"{relative_path}:{node.lineno}: unexpected open mode {mode_arg.value:#o}"
                )

    assert not violations, "os.open secure mode violations found:\n" + "\n".join(violations)


def test_source_does_not_use_os_ordwr_flag() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr == "O_RDWR"
            ):
                violations.append(f"{relative_path}:{node.lineno}")

    assert not violations, "unexpected os.O_RDWR usage found:\n" + "\n".join(violations)


def test_source_os_open_functions_define_nonblock_and_nofollow_flags() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(module):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        open_functions: dict[ast.AST, list[int]] = {}
        for node in ast.walk(module):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "open"
            ):
                continue
            current: ast.AST | None = node
            while current is not None and not isinstance(
                current, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                current = parents.get(current)
            if current is None:
                violations.append(f"{relative_path}:{node.lineno}: os.open outside function")
                continue
            open_functions.setdefault(current, []).append(node.lineno)

        for function_node, open_lines in open_functions.items():
            os_attrs = {
                attr.attr
                for attr in ast.walk(function_node)
                if isinstance(attr, ast.Attribute)
                and isinstance(attr.value, ast.Name)
                and attr.value.id == "os"
            }
            for required_flag in ("O_NONBLOCK", "O_NOFOLLOW"):
                if required_flag not in os_attrs:
                    function_name = getattr(function_node, "name", "<unknown>")
                    lines = ",".join(str(line) for line in sorted(open_lines))
                    violations.append(
                        f"{relative_path}:{function_name}:os.open@{lines} missing {required_flag}"
                    )

    assert not violations, "os.open flag policy violations found:\n" + "\n".join(violations)


def test_source_os_open_uses_flag_variable_with_nonblock_and_nofollow() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    def _is_os_attr(expr: ast.AST, attr: str) -> bool:
        return (
            isinstance(expr, ast.Attribute)
            and isinstance(expr.value, ast.Name)
            and expr.value.id == "os"
            and expr.attr == attr
        )

    def _expr_contains_os_attr(expr: ast.AST, attr: str) -> bool:
        return any(_is_os_attr(node, attr) for node in ast.walk(expr))

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))

        for function_node in ast.walk(module):
            if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            flag_assignments: dict[str, set[str]] = {}
            for node in ast.walk(function_node):
                if isinstance(node, ast.Assign):
                    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    for target_name in targets:
                        recorded = flag_assignments.setdefault(target_name, set())
                        if _expr_contains_os_attr(node.value, "O_NONBLOCK"):
                            recorded.add("O_NONBLOCK")
                        if _expr_contains_os_attr(node.value, "O_NOFOLLOW"):
                            recorded.add("O_NOFOLLOW")
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    recorded = flag_assignments.setdefault(node.target.id, set())
                    if _expr_contains_os_attr(node.value, "O_NONBLOCK"):
                        recorded.add("O_NONBLOCK")
                    if _expr_contains_os_attr(node.value, "O_NOFOLLOW"):
                        recorded.add("O_NOFOLLOW")

            for node in ast.walk(function_node):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "open"
                    and len(node.args) >= 2
                ):
                    continue

                open_arg = node.args[1]
                if not isinstance(open_arg, ast.Name):
                    violations.append(
                        f"{relative_path}:{node.lineno}: os.open flags must use named variable"
                    )
                    continue

                seen_flags = flag_assignments.get(open_arg.id, set())
                missing = {"O_NONBLOCK", "O_NOFOLLOW"} - seen_flags
                if missing:
                    missing_text = ",".join(sorted(missing))
                    violations.append(
                        f"{relative_path}:{node.lineno}: {open_arg.id} missing {missing_text}"
                    )

    assert not violations, "os.open flag-variable policy violations found:\n" + "\n".join(
        violations
    )


def test_source_read_only_os_open_calls_do_not_include_write_flags() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    def _is_os_attr(expr: ast.AST, attr: str) -> bool:
        return (
            isinstance(expr, ast.Attribute)
            and isinstance(expr.value, ast.Name)
            and expr.value.id == "os"
            and expr.attr == attr
        )

    def _expr_contains_os_attr(expr: ast.AST, attr: str) -> bool:
        return any(_is_os_attr(node, attr) for node in ast.walk(expr))

    forbidden = {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND", "O_EXCL"}

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))

        for function_node in ast.walk(module):
            if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            flag_assignments: dict[str, set[str]] = {}
            for node in ast.walk(function_node):
                if isinstance(node, ast.Assign):
                    targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                    for target_name in targets:
                        recorded = flag_assignments.setdefault(target_name, set())
                        for attr_name in ("O_RDONLY", *sorted(forbidden)):
                            if _expr_contains_os_attr(node.value, attr_name):
                                recorded.add(attr_name)
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    recorded = flag_assignments.setdefault(node.target.id, set())
                    for attr_name in ("O_RDONLY", *sorted(forbidden)):
                        if _expr_contains_os_attr(node.value, attr_name):
                            recorded.add(attr_name)

            for node in ast.walk(function_node):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "open"
                    and len(node.args) == 2
                ):
                    continue
                open_arg = node.args[1]
                if not isinstance(open_arg, ast.Name):
                    violations.append(
                        f"{relative_path}:{node.lineno}: read os.open flags must use named variable"
                    )
                    continue
                seen_flags = flag_assignments.get(open_arg.id, set())
                if "O_RDONLY" not in seen_flags:
                    violations.append(
                        f"{relative_path}:{node.lineno}: {open_arg.id} missing O_RDONLY"
                    )
                used_forbidden = sorted(seen_flags & forbidden)
                if used_forbidden:
                    violations.append(
                        f"{relative_path}:{node.lineno}: {open_arg.id} uses forbidden flags "
                        + ",".join(used_forbidden)
                    )

    assert not violations, "read-only os.open policy violations found:\n" + "\n".join(violations)


def test_source_three_arg_os_open_calls_use_write_only_create_modes() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    def _is_os_attr(expr: ast.AST, attr: str) -> bool:
        return (
            isinstance(expr, ast.Attribute)
            and isinstance(expr.value, ast.Name)
            and expr.value.id == "os"
            and expr.attr == attr
        )

    def _expr_contains_os_attr(expr: ast.AST, attr: str) -> bool:
        return any(_is_os_attr(node, attr) for node in ast.walk(expr))

    required = {"O_WRONLY", "O_CREAT"}
    forbidden = {"O_RDONLY", "O_RDWR"}
    mode_variants = {"O_TRUNC", "O_APPEND", "O_EXCL"}

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))

        for function_node in ast.walk(module):
            if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            flag_assignments: dict[str, set[str]] = {}
            tracked_attrs = required | forbidden | mode_variants
            for node in ast.walk(function_node):
                if isinstance(node, ast.Assign):
                    targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                    for target_name in targets:
                        recorded = flag_assignments.setdefault(target_name, set())
                        for attr_name in sorted(tracked_attrs):
                            if _expr_contains_os_attr(node.value, attr_name):
                                recorded.add(attr_name)
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    recorded = flag_assignments.setdefault(node.target.id, set())
                    for attr_name in sorted(tracked_attrs):
                        if _expr_contains_os_attr(node.value, attr_name):
                            recorded.add(attr_name)

            for node in ast.walk(function_node):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "open"
                    and len(node.args) >= 3
                ):
                    continue
                open_arg = node.args[1]
                if not isinstance(open_arg, ast.Name):
                    violations.append(
                        f"{relative_path}:{node.lineno}: "
                        "write os.open flags must use named variable"
                    )
                    continue
                seen_flags = flag_assignments.get(open_arg.id, set())
                missing_required = sorted(required - seen_flags)
                if missing_required:
                    violations.append(
                        f"{relative_path}:{node.lineno}: {open_arg.id} missing "
                        + ",".join(missing_required)
                    )
                used_forbidden = sorted(seen_flags & forbidden)
                if used_forbidden:
                    violations.append(
                        f"{relative_path}:{node.lineno}: {open_arg.id} uses forbidden flags "
                        + ",".join(used_forbidden)
                    )
                if not (seen_flags & mode_variants):
                    violations.append(
                        f"{relative_path}:{node.lineno}: {open_arg.id} missing mode variant "
                        "(O_TRUNC/O_APPEND/O_EXCL)"
                    )

    assert not violations, "three-arg os.open policy violations found:\n" + "\n".join(violations)


def test_source_os_open_flag_variable_has_single_base_assignment() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))

        for function_node in ast.walk(module):
            if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for node in ast.walk(function_node):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "open"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Name)
                ):
                    continue

                flag_name = node.args[1].id
                base_assignments = [
                    assign
                    for assign in ast.walk(function_node)
                    if isinstance(assign, ast.Assign)
                    and assign.lineno <= node.lineno
                    and any(
                        isinstance(target, ast.Name) and target.id == flag_name
                        for target in assign.targets
                    )
                ]
                if len(base_assignments) != 1:
                    violations.append(
                        f"{relative_path}:{node.lineno}: {flag_name} has "
                        f"{len(base_assignments)} base assignments"
                    )

                invalid_augassigns = [
                    aug
                    for aug in ast.walk(function_node)
                    if isinstance(aug, ast.AugAssign)
                    and aug.lineno <= node.lineno
                    and isinstance(aug.target, ast.Name)
                    and aug.target.id == flag_name
                    and not isinstance(aug.op, ast.BitOr)
                ]
                if invalid_augassigns:
                    violations.append(
                        f"{relative_path}:{node.lineno}: {flag_name} has non-|= augmented assign"
                    )

    assert not violations, "os.open flag-assignment policy violations found:\n" + "\n".join(
        violations
    )


def test_source_os_open_calls_use_supported_positional_signature() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    violations: list[str] = []

    for file_path in src_root.rglob("*.py"):
        relative_path = file_path.relative_to(src_root)
        module = ast.parse(file_path.read_text(encoding="utf-8"))

        for node in ast.walk(module):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "open"
            ):
                continue
            if node.keywords:
                keyword_names = ",".join(keyword.arg or "<**kwargs>" for keyword in node.keywords)
                violations.append(
                    f"{relative_path}:{node.lineno}: os.open uses keyword args ({keyword_names})"
                )
            if len(node.args) not in {2, 3}:
                violations.append(
                    f"{relative_path}:{node.lineno}: unsupported os.open arg count {len(node.args)}"
                )

    assert not violations, "os.open signature policy violations found:\n" + "\n".join(violations)


def test_source_critical_writers_check_symlink_guards_before_open() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    targets = [
        (Path("exec/cancel.py"), "write_cancel_request"),
        (Path("state/lock.py"), "run_lock"),
    ]
    violations: list[str] = []

    for relative_path, function_name in targets:
        file_path = src_root / relative_path
        module = ast.parse(file_path.read_text(encoding="utf-8"))
        function_node = next(
            (
                node
                for node in ast.walk(module)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        if function_node is None:
            violations.append(f"{relative_path}:{function_name}: function not found")
            continue

        open_lines = [
            node.lineno
            for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "open"
        ]
        ancestor_guard_lines = [
            node.lineno
            for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "has_symlink_ancestor"
        ]
        symlink_guard_lines = [
            node.lineno
            for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "is_symlink_path"
        ]

        if not open_lines:
            violations.append(f"{relative_path}:{function_name}: os.open not found")
            continue
        if not ancestor_guard_lines:
            violations.append(f"{relative_path}:{function_name}: has_symlink_ancestor not found")
        if not symlink_guard_lines:
            violations.append(f"{relative_path}:{function_name}: is_symlink_path not found")

        first_open = min(open_lines)
        if ancestor_guard_lines and min(ancestor_guard_lines) >= first_open:
            violations.append(
                f"{relative_path}:{function_name}: has_symlink_ancestor occurs after os.open"
            )
        if symlink_guard_lines and min(symlink_guard_lines) >= first_open:
            violations.append(
                f"{relative_path}:{function_name}: is_symlink_path occurs after os.open"
            )

    assert not violations, "critical writer guard-order violations found:\n" + "\n".join(violations)


def test_source_cancel_helpers_check_ancestor_guard_before_target_ops() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cancel_module = ast.parse((src_root / "exec/cancel.py").read_text(encoding="utf-8"))
    targets: list[tuple[str, tuple[str, ...]]] = [
        ("cancel_requested", ("lstat",)),
        ("clear_cancel_request", ("lstat", "unlink")),
        ("write_cancel_request", ("lstat", "open")),
    ]
    violations: list[str] = []

    for function_name, operation_names in targets:
        function_node = next(
            (
                node
                for node in ast.walk(cancel_module)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        if function_node is None:
            violations.append(f"exec/cancel.py:{function_name}: function not found")
            continue

        guard_lines = [
            node.lineno
            for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "has_symlink_ancestor"
        ]
        if not guard_lines:
            violations.append(f"exec/cancel.py:{function_name}: has_symlink_ancestor not found")
            continue
        first_guard = min(guard_lines)

        for operation_name in operation_names:
            operation_lines: list[int]
            if operation_name == "open":
                operation_lines = [
                    node.lineno
                    for node in ast.walk(function_node)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "open"
                ]
            else:
                operation_lines = [
                    node.lineno
                    for node in ast.walk(function_node)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == operation_name
                ]
            if not operation_lines:
                violations.append(
                    f"exec/cancel.py:{function_name}: {operation_name} call not found"
                )
                continue
            if first_guard >= min(operation_lines):
                violations.append(
                    f"exec/cancel.py:{function_name}: has_symlink_ancestor occurs "
                    f"after {operation_name}"
                )

    assert not violations, "cancel helper guard-order violations found:\n" + "\n".join(violations)


def test_source_cancel_helpers_check_run_dir_lstat_before_target_ops() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cancel_module = ast.parse((src_root / "exec/cancel.py").read_text(encoding="utf-8"))
    targets: list[tuple[str, tuple[tuple[str, str], ...]]] = [
        ("cancel_requested", (("path", "lstat"),)),
        ("clear_cancel_request", (("path", "lstat"), ("path", "unlink"))),
        ("write_cancel_request", (("path", "lstat"), ("os", "open"))),
    ]
    violations: list[str] = []

    for function_name, operations in targets:
        function_node = next(
            (
                node
                for node in ast.walk(cancel_module)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        if function_node is None:
            violations.append(f"exec/cancel.py:{function_name}: function not found")
            continue

        run_dir_lstat_lines = [
            node.lineno
            for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "run_dir"
            and node.func.attr == "lstat"
        ]
        if not run_dir_lstat_lines:
            violations.append(f"exec/cancel.py:{function_name}: run_dir.lstat not found")
            continue
        first_run_dir_lstat = min(run_dir_lstat_lines)

        for receiver_name, method_name in operations:
            operation_lines: list[int] = []
            if receiver_name == "os":
                operation_lines = [
                    node.lineno
                    for node in ast.walk(function_node)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == method_name
                ]
            else:
                operation_lines = [
                    node.lineno
                    for node in ast.walk(function_node)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == receiver_name
                    and node.func.attr == method_name
                ]
            if not operation_lines:
                violations.append(
                    f"exec/cancel.py:{function_name}: {receiver_name}.{method_name} not found"
                )
                continue
            if first_run_dir_lstat >= min(operation_lines):
                violations.append(
                    f"exec/cancel.py:{function_name}: run_dir.lstat occurs after "
                    f"{receiver_name}.{method_name}"
                )

    assert not violations, "cancel helper run_dir-lstat order violations found:\n" + "\n".join(
        violations
    )


def test_source_run_lock_checks_run_dir_lstat_before_lock_ops() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    lock_module = ast.parse((src_root / "state/lock.py").read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in ast.walk(lock_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_lock"
        ),
        None,
    )
    assert function_node is not None

    run_dir_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_dir"
        and node.func.attr == "lstat"
    ]
    assert run_dir_lstat_lines
    first_run_dir_lstat = min(run_dir_lstat_lines)

    lock_path_symlink_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_symlink_path"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "lock_path"
    ]
    assert lock_path_symlink_lines
    assert first_run_dir_lstat < min(lock_path_symlink_lines)

    os_open_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr == "open"
    ]
    assert os_open_lines
    assert first_run_dir_lstat < min(os_open_lines)


def test_source_run_lock_checks_ancestor_guard_before_run_dir_lstat() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    lock_module = ast.parse((src_root / "state/lock.py").read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in ast.walk(lock_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_lock"
        ),
        None,
    )
    assert function_node is not None

    ancestor_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "has_symlink_ancestor"
    ]
    assert ancestor_guard_lines
    first_ancestor_guard = min(ancestor_guard_lines)

    run_dir_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_dir"
        and node.func.attr == "lstat"
    ]
    assert run_dir_lstat_lines
    assert first_ancestor_guard < min(run_dir_lstat_lines)


def test_source_run_lock_checks_ancestor_guard_before_run_dir_symlink_guard() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    lock_module = ast.parse((src_root / "state/lock.py").read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in ast.walk(lock_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_lock"
        ),
        None,
    )
    assert function_node is not None

    ancestor_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "has_symlink_ancestor"
    ]
    assert ancestor_guard_lines
    first_ancestor_guard = min(ancestor_guard_lines)

    run_dir_symlink_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_symlink_path"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "run_dir"
    ]
    assert run_dir_symlink_guard_lines
    assert first_ancestor_guard < min(run_dir_symlink_guard_lines)


def test_source_write_cancel_request_checks_run_dir_lstat_before_path_symlink_guard() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cancel_module = ast.parse((src_root / "exec/cancel.py").read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in ast.walk(cancel_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "write_cancel_request"
        ),
        None,
    )
    assert function_node is not None

    run_dir_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_dir"
        and node.func.attr == "lstat"
    ]
    assert run_dir_lstat_lines
    first_run_dir_lstat = min(run_dir_lstat_lines)

    path_symlink_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_symlink_path"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "path"
    ]
    assert path_symlink_guard_lines
    assert first_run_dir_lstat < min(path_symlink_guard_lines)


def test_source_write_cancel_request_checks_ancestor_guard_before_path_symlink_guard() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cancel_module = ast.parse((src_root / "exec/cancel.py").read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in ast.walk(cancel_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "write_cancel_request"
        ),
        None,
    )
    assert function_node is not None

    ancestor_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "has_symlink_ancestor"
    ]
    assert ancestor_guard_lines
    first_ancestor_guard = min(ancestor_guard_lines)

    path_symlink_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_symlink_path"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "path"
    ]
    assert path_symlink_guard_lines
    assert first_ancestor_guard < min(path_symlink_guard_lines)


def test_source_cancel_helpers_check_ancestor_guard_before_run_dir_lstat() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cancel_module = ast.parse((src_root / "exec/cancel.py").read_text(encoding="utf-8"))
    targets = ("cancel_requested", "clear_cancel_request", "write_cancel_request")
    violations: list[str] = []

    for function_name in targets:
        function_node = next(
            (
                node
                for node in ast.walk(cancel_module)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        if function_node is None:
            violations.append(f"exec/cancel.py:{function_name}: function not found")
            continue

        ancestor_guard_lines = [
            node.lineno
            for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "has_symlink_ancestor"
        ]
        run_dir_lstat_lines = [
            node.lineno
            for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "run_dir"
            and node.func.attr == "lstat"
        ]
        if not ancestor_guard_lines:
            violations.append(f"exec/cancel.py:{function_name}: has_symlink_ancestor not found")
            continue
        if not run_dir_lstat_lines:
            violations.append(f"exec/cancel.py:{function_name}: run_dir.lstat not found")
            continue
        if min(ancestor_guard_lines) >= min(run_dir_lstat_lines):
            violations.append(
                f"exec/cancel.py:{function_name}: has_symlink_ancestor occurs after run_dir.lstat"
            )

    assert not violations, "cancel helper ancestor-before-run_dir violations found:\n" + "\n".join(
        violations
    )


def test_source_run_lock_checks_full_guard_sequence_before_open() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    lock_module = ast.parse((src_root / "state/lock.py").read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in ast.walk(lock_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_lock"
        ),
        None,
    )
    assert function_node is not None

    ancestor_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "has_symlink_ancestor"
    ]
    run_dir_symlink_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_symlink_path"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "run_dir"
    ]
    run_dir_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_dir"
        and node.func.attr == "lstat"
    ]
    lock_path_symlink_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_symlink_path"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "lock_path"
    ]
    open_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr == "open"
    ]

    assert ancestor_guard_lines
    assert run_dir_symlink_guard_lines
    assert run_dir_lstat_lines
    assert lock_path_symlink_guard_lines
    assert open_lines

    assert min(ancestor_guard_lines) < min(run_dir_symlink_guard_lines)
    assert min(run_dir_symlink_guard_lines) < min(run_dir_lstat_lines)
    assert min(run_dir_lstat_lines) < min(lock_path_symlink_guard_lines)
    assert min(lock_path_symlink_guard_lines) < min(open_lines)


def test_source_write_cancel_request_checks_full_guard_sequence_before_open() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cancel_module = ast.parse((src_root / "exec/cancel.py").read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in ast.walk(cancel_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "write_cancel_request"
        ),
        None,
    )
    assert function_node is not None

    ancestor_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "has_symlink_ancestor"
    ]
    run_dir_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_dir"
        and node.func.attr == "lstat"
    ]
    path_symlink_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_symlink_path"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "path"
    ]
    path_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "path"
        and node.func.attr == "lstat"
    ]
    open_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr == "open"
    ]

    assert ancestor_guard_lines
    assert run_dir_lstat_lines
    assert path_symlink_guard_lines
    assert path_lstat_lines
    assert open_lines

    assert min(ancestor_guard_lines) < min(run_dir_lstat_lines)
    assert min(run_dir_lstat_lines) < min(path_symlink_guard_lines)
    assert min(path_symlink_guard_lines) < min(path_lstat_lines)
    assert min(path_lstat_lines) < min(open_lines)


def test_source_cancel_requested_checks_full_guard_sequence_before_target_lstat() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cancel_module = ast.parse((src_root / "exec/cancel.py").read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in ast.walk(cancel_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "cancel_requested"
        ),
        None,
    )
    assert function_node is not None

    ancestor_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "has_symlink_ancestor"
    ]
    run_dir_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_dir"
        and node.func.attr == "lstat"
    ]
    path_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "path"
        and node.func.attr == "lstat"
    ]

    assert ancestor_guard_lines
    assert run_dir_lstat_lines
    assert path_lstat_lines

    assert min(ancestor_guard_lines) < min(run_dir_lstat_lines)
    assert min(run_dir_lstat_lines) < min(path_lstat_lines)


def test_source_clear_cancel_request_checks_full_guard_sequence_before_unlink() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cancel_module = ast.parse((src_root / "exec/cancel.py").read_text(encoding="utf-8"))
    function_node = next(
        (
            node
            for node in ast.walk(cancel_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "clear_cancel_request"
        ),
        None,
    )
    assert function_node is not None

    ancestor_guard_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "has_symlink_ancestor"
    ]
    run_dir_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "run_dir"
        and node.func.attr == "lstat"
    ]
    path_lstat_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "path"
        and node.func.attr == "lstat"
    ]
    path_unlink_lines = [
        node.lineno
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "path"
        and node.func.attr == "unlink"
    ]

    assert ancestor_guard_lines
    assert run_dir_lstat_lines
    assert path_lstat_lines
    assert path_unlink_lines

    assert min(ancestor_guard_lines) < min(run_dir_lstat_lines)
    assert min(run_dir_lstat_lines) < min(path_lstat_lines)
    assert min(path_lstat_lines) < min(path_unlink_lines)


def test_source_cli_cancel_checks_run_exists_before_write_cancel_request() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    cancel_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "cancel"
        ),
        None,
    )
    assert cancel_function is not None

    run_exists_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_exists"
    ]
    write_cancel_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_cancel_request"
    ]
    assert run_exists_lines
    assert write_cancel_lines
    assert min(run_exists_lines) < min(write_cancel_lines)


def test_source_cli_cancel_checks_full_guard_sequence_before_cancel_write() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    cancel_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "cancel"
        ),
        None,
    )
    assert cancel_function is not None

    validate_run_id_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_run_id_or_exit"
    ]
    validate_home_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_home_or_exit"
    ]
    run_dir_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_dir"
    ]
    run_exists_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_exists"
    ]
    write_cancel_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_cancel_request"
    ]

    assert validate_run_id_lines
    assert validate_home_lines
    assert run_dir_lines
    assert run_exists_lines
    assert write_cancel_lines

    assert min(validate_run_id_lines) < min(validate_home_lines)
    assert min(validate_home_lines) < min(run_dir_lines)
    assert min(run_dir_lines) < min(run_exists_lines)
    assert min(run_exists_lines) < min(write_cancel_lines)


def test_source_cli_cancel_rejects_invalid_run_id_before_run_exists_and_write() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    cancel_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "cancel"
        ),
        None,
    )
    assert cancel_function is not None

    validate_run_id_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_run_id_or_exit"
    ]
    run_exists_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_exists"
    ]
    write_cancel_lines = [
        node.lineno
        for node in ast.walk(cancel_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_cancel_request"
    ]

    assert validate_run_id_lines
    assert run_exists_lines
    assert write_cancel_lines
    assert min(validate_run_id_lines) < min(run_exists_lines)
    assert min(validate_run_id_lines) < min(write_cancel_lines)


def test_source_cli_status_logs_resume_validate_run_id_before_home_and_run_dir() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))

    for function_name in ("status", "logs", "resume"):
        cli_function = next(
            (
                node
                for node in ast.walk(cli_module)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        assert cli_function is not None

        validate_run_id_lines = [
            node.lineno
            for node in ast.walk(cli_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validate_run_id_or_exit"
        ]
        validate_home_lines = [
            node.lineno
            for node in ast.walk(cli_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validate_home_or_exit"
        ]
        run_dir_lines = [
            node.lineno
            for node in ast.walk(cli_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_dir"
        ]

        assert validate_run_id_lines, function_name
        assert validate_home_lines, function_name
        assert run_dir_lines, function_name

        assert min(validate_run_id_lines) < min(validate_home_lines), function_name
        assert min(validate_home_lines) < min(run_dir_lines), function_name


def test_source_cli_resume_validates_and_resolves_workdir_before_run_dir() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    resume_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "resume"
        ),
        None,
    )
    assert resume_function is not None

    validate_run_id_lines = [
        node.lineno
        for node in ast.walk(resume_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_run_id_or_exit"
    ]
    validate_home_lines = [
        node.lineno
        for node in ast.walk(resume_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_home_or_exit"
    ]
    resolve_workdir_lines = [
        node.lineno
        for node in ast.walk(resume_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_resolve_workdir_or_exit"
    ]
    run_dir_lines = [
        node.lineno
        for node in ast.walk(resume_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_dir"
    ]

    assert validate_run_id_lines
    assert validate_home_lines
    assert resolve_workdir_lines
    assert run_dir_lines

    assert min(validate_run_id_lines) < min(validate_home_lines)
    assert min(validate_home_lines) < min(resolve_workdir_lines)
    assert min(resolve_workdir_lines) < min(run_dir_lines)


def test_source_cli_run_validates_home_before_plan_load_and_workdir_resolution() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    validate_home_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_home_or_exit"
    ]
    load_plan_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_plan"
    ]
    resolve_workdir_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_resolve_workdir_or_exit"
    ]

    assert validate_home_lines
    assert load_plan_lines
    assert resolve_workdir_lines
    assert min(validate_home_lines) < min(load_plan_lines)
    assert min(load_plan_lines) < min(resolve_workdir_lines)


def test_source_cli_run_builds_dag_before_dry_run_branch() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    load_plan_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_plan"
    ]
    build_adjacency_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_adjacency"
    ]
    assert_acyclic_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_acyclic"
    ]
    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )

    assert load_plan_lines
    assert build_adjacency_lines
    assert assert_acyclic_lines
    assert dry_run_if is not None
    assert min(load_plan_lines) < min(build_adjacency_lines)
    assert min(build_adjacency_lines) < min(assert_acyclic_lines)
    assert min(assert_acyclic_lines) < dry_run_if.lineno


def test_source_cli_run_resolves_workdir_before_run_dir_creation() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    resolve_workdir_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_resolve_workdir_or_exit"
    ]
    run_dir_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_dir"
    ]
    assert resolve_workdir_lines
    assert run_dir_lines
    assert min(resolve_workdir_lines) < min(run_dir_lines)


def test_source_cli_run_has_dry_run_exit_before_workdir_resolution() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    has_exit_zero = any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Attribute)
        and isinstance(node.exc.func.value, ast.Name)
        and node.exc.func.value.id == "typer"
        and node.exc.func.attr == "Exit"
        and len(node.exc.args) == 1
        and isinstance(node.exc.args[0], ast.Constant)
        and node.exc.args[0].value == 0
        for node in ast.walk(dry_run_if)
    )
    assert has_exit_zero

    resolve_workdir_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_resolve_workdir_or_exit"
    ]
    assert resolve_workdir_lines
    assert dry_run_if.lineno < min(resolve_workdir_lines)


def test_source_cli_run_dry_run_branch_has_single_exit_zero_raise() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    exit_zero_raises = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Attribute)
        and isinstance(node.exc.func.value, ast.Name)
        and node.exc.func.value.id == "typer"
        and node.exc.func.attr == "Exit"
        and len(node.exc.args) == 1
        and isinstance(node.exc.args[0], ast.Constant)
        and node.exc.args[0].value == 0
    ]
    assert len(exit_zero_raises) == 1


def test_source_cli_run_has_single_top_level_dry_run_branch() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_ifs = [
        stmt
        for stmt in run_function.body
        if isinstance(stmt, ast.If)
        and isinstance(stmt.test, ast.Name)
        and stmt.test.id == "dry_run"
    ]
    assert len(dry_run_ifs) == 1


def test_source_cli_run_top_level_dry_run_branch_has_no_else_block() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None
    assert not dry_run_if.orelse


def test_source_cli_run_dry_run_branch_builds_expected_table_title() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    table_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Table"
    ]
    assert table_calls
    assert any(
        any(
            keyword.arg == "title"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "Dry Run - Topological Order"
            for keyword in table_call.keywords
        )
        for table_call in table_calls
    )


def test_source_cli_run_dry_run_branch_assigns_table_variable_from_table_ctor() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    table_assignments = [
        stmt
        for stmt in dry_run_if.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == "table"
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "Table"
    ]
    assert table_assignments


def test_source_cli_run_dry_run_branch_has_single_table_constructor_call() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    table_ctor_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Table"
    ]
    assert len(table_ctor_calls) == 1


def test_source_cli_run_dry_run_branch_starts_with_table_assignment() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None
    assert dry_run_if.body

    first_stmt = dry_run_if.body[0]
    assert isinstance(first_stmt, ast.Assign)
    assert len(first_stmt.targets) == 1
    assert isinstance(first_stmt.targets[0], ast.Name)
    assert first_stmt.targets[0].id == "table"
    assert isinstance(first_stmt.value, ast.Call)
    assert isinstance(first_stmt.value.func, ast.Name)
    assert first_stmt.value.func.id == "Table"


def test_source_cli_run_dry_run_branch_adds_expected_table_columns() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    add_column_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "table"
        and node.func.attr == "add_column"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    column_names = {node.args[0].value for node in add_column_calls}
    assert {"#", "task_id"}.issubset(column_names)


def test_source_cli_run_dry_run_branch_adds_columns_in_expected_order() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    add_column_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "table"
        and node.func.attr == "add_column"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    column_line_by_name = {node.args[0].value: node.lineno for node in add_column_calls}
    assert "#" in column_line_by_name
    assert "task_id" in column_line_by_name
    assert column_line_by_name["#"] < column_line_by_name["task_id"]


def test_source_cli_run_dry_run_branch_adds_row_with_index_and_task_id() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    add_row_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "table"
        and node.func.attr == "add_row"
        and len(node.args) == 2
    ]
    assert add_row_calls
    assert any(
        isinstance(call.args[0], ast.Call)
        and isinstance(call.args[0].func, ast.Name)
        and call.args[0].func.id == "str"
        and len(call.args[0].args) == 1
        and isinstance(call.args[0].args[0], ast.Name)
        and call.args[0].args[0].id == "idx"
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "task_id"
        for call in add_row_calls
    )


def test_source_cli_run_dry_run_branch_adds_columns_before_rows() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    add_column_lines = [
        node.lineno
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "table"
        and node.func.attr == "add_column"
    ]
    add_row_lines = [
        node.lineno
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "table"
        and node.func.attr == "add_row"
    ]
    assert add_column_lines
    assert add_row_lines
    assert min(add_column_lines) < min(add_row_lines)


def test_source_cli_run_dry_run_branch_enumerates_order_with_start_one() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    enumerate_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "enumerate"
    ]
    assert enumerate_calls
    assert any(
        (len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and call.args[1].value == 1)
        or any(
            keyword.arg == "start"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 1
            for keyword in call.keywords
        )
        for call in enumerate_calls
    )


def test_source_cli_run_dry_run_branch_iterates_enumerate_order_into_idx_task_id() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    for_nodes = [node for node in ast.walk(dry_run_if) if isinstance(node, ast.For)]
    assert for_nodes
    assert any(
        isinstance(for_node.target, ast.Tuple)
        and len(for_node.target.elts) == 2
        and isinstance(for_node.target.elts[0], ast.Name)
        and for_node.target.elts[0].id == "idx"
        and isinstance(for_node.target.elts[1], ast.Name)
        and for_node.target.elts[1].id == "task_id"
        and isinstance(for_node.iter, ast.Call)
        and isinstance(for_node.iter.func, ast.Name)
        and for_node.iter.func.id == "enumerate"
        and for_node.iter.args
        and isinstance(for_node.iter.args[0], ast.Name)
        and for_node.iter.args[0].id == "order"
        for for_node in for_nodes
    )


def test_source_cli_run_dry_run_branch_add_row_occurs_inside_for_loop() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    for_nodes = [node for node in ast.walk(dry_run_if) if isinstance(node, ast.For)]
    assert for_nodes
    assert any(
        any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "table"
            and child.func.attr == "add_row"
            for child in ast.walk(for_node)
        )
        for for_node in for_nodes
    )


def test_source_cli_run_dry_run_branch_uses_single_add_row_callsite() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    add_row_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "table"
        and node.func.attr == "add_row"
    ]
    assert len(add_row_calls) == 1


def test_source_cli_run_dry_run_branch_prints_table_object() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    print_table_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "console"
        and node.func.attr == "print"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "table"
    ]
    assert print_table_calls


def test_source_cli_run_dry_run_branch_console_prints_only_table() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    console_print_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "console"
        and node.func.attr == "print"
    ]
    assert console_print_calls
    assert all(
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "table"
        and len(call.args) == 1
        and not call.keywords
        for call in console_print_calls
    )


def test_source_cli_run_dry_run_branch_has_single_console_print_call() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    console_print_calls = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "console"
        and node.func.attr == "print"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "table"
    ]
    assert len(console_print_calls) == 1


def test_source_cli_run_dry_run_branch_prints_table_before_exit_zero() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    print_table_lines = [
        node.lineno
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "console"
        and node.func.attr == "print"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "table"
    ]
    exit_zero_lines = [
        node.lineno
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Attribute)
        and isinstance(node.exc.func.value, ast.Name)
        and node.exc.func.value.id == "typer"
        and node.exc.func.attr == "Exit"
        and len(node.exc.args) == 1
        and isinstance(node.exc.args[0], ast.Constant)
        and node.exc.args[0].value == 0
    ]
    assert print_table_lines
    assert exit_zero_lines
    assert min(print_table_lines) < min(exit_zero_lines)


def test_source_cli_run_dry_run_branch_prints_table_after_row_population() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    add_row_lines = [
        node.lineno
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "table"
        and node.func.attr == "add_row"
    ]
    print_table_lines = [
        node.lineno
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "console"
        and node.func.attr == "print"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "table"
    ]
    assert add_row_lines
    assert print_table_lines
    assert min(add_row_lines) < min(print_table_lines)


def test_source_cli_run_dry_run_branch_is_independent_of_fail_fast_name() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    fail_fast_name_nodes = [
        node
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Name) and node.id == "fail_fast"
    ]
    assert not fail_fast_name_nodes


def test_source_cli_run_dry_run_branch_has_no_runtime_execution_calls() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    call_names: set[str] = set()
    for node in ast.walk(dry_run_if):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            call_names.add(node.func.attr)

    forbidden = {
        "_resolve_workdir_or_exit",
        "run_dir",
        "ensure_run_layout",
        "_write_plan_snapshot",
        "run_plan",
        "_write_report",
        "new_run_id",
        "_exit_code_for_state",
    }
    assert not (call_names & forbidden)


def test_source_cli_run_dry_run_branch_has_no_runtime_summary_literals() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    literals = [
        node.value
        for node in ast.walk(dry_run_if)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    forbidden_markers = ("run_id:", "state:", "report:")
    assert not any(marker in literal for literal in literals for marker in forbidden_markers)


def test_source_cli_run_has_dry_run_exit_before_run_dir_creation() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    run_dir_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_dir"
    ]
    assert run_dir_lines
    assert dry_run_if.lineno < min(run_dir_lines)


def test_source_cli_run_has_dry_run_exit_before_ensure_run_layout() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    ensure_layout_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ensure_run_layout"
    ]
    assert ensure_layout_lines
    assert dry_run_if.lineno < min(ensure_layout_lines)


def test_source_cli_run_has_dry_run_exit_before_plan_snapshot_write() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    snapshot_write_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_plan_snapshot"
    ]
    assert snapshot_write_lines
    assert dry_run_if.lineno < min(snapshot_write_lines)


def test_source_cli_run_has_dry_run_exit_before_run_plan_execution() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    run_plan_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_plan"
    ]
    assert run_plan_lines
    assert dry_run_if.lineno < min(run_plan_lines)


def test_source_cli_run_has_dry_run_exit_before_report_write() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    write_report_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_report"
    ]
    assert write_report_lines
    assert dry_run_if.lineno < min(write_report_lines)


def test_source_cli_run_has_dry_run_exit_before_run_id_generation() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    new_run_id_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "new_run_id"
    ]
    assert new_run_id_lines
    assert dry_run_if.lineno < min(new_run_id_lines)


def test_source_cli_run_has_dry_run_exit_before_runtime_summary_prints() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    def _arg_contains_token(arg: ast.AST, token: str) -> bool:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return token in arg.value
        if isinstance(arg, ast.JoinedStr):
            return any(
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and token in value.value
                for value in arg.values
            )
        return False

    runtime_summary_tokens = ("run_id:", "state:", "report:")
    runtime_summary_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "console"
        and node.func.attr == "print"
        and node.args
        and any(_arg_contains_token(node.args[0], token) for token in runtime_summary_tokens)
    ]
    assert runtime_summary_lines
    assert dry_run_if.lineno < min(runtime_summary_lines)


def test_source_cli_run_has_dry_run_exit_before_exit_code_mapping() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )
    assert dry_run_if is not None

    exit_code_mapping_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_exit_code_for_state"
    ]
    assert exit_code_mapping_lines
    assert dry_run_if.lineno < min(exit_code_mapping_lines)


def test_source_cli_run_validates_home_before_dry_run_branch() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    validate_home_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_home_or_exit"
    ]
    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )

    assert validate_home_lines
    assert dry_run_if is not None
    assert min(validate_home_lines) < dry_run_if.lineno


def test_source_cli_run_load_plan_before_dry_run_branch() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    assert run_function is not None

    load_plan_lines = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_plan"
    ]
    dry_run_if = next(
        (
            stmt
            for stmt in run_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "dry_run"
        ),
        None,
    )

    assert load_plan_lines
    assert dry_run_if is not None
    assert min(load_plan_lines) < dry_run_if.lineno


def test_source_validate_home_checks_symlink_guards_before_lstat_loop() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    validate_home_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_validate_home_or_exit"
        ),
        None,
    )
    assert validate_home_function is not None

    symlink_guard_lines = [
        node.lineno
        for node in ast.walk(validate_home_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"is_symlink_path", "has_symlink_ancestor"}
    ]
    lstat_lines = [
        node.lineno
        for node in ast.walk(validate_home_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.attr == "lstat"
    ]
    assert symlink_guard_lines
    assert lstat_lines
    assert min(symlink_guard_lines) < min(lstat_lines)

    guard_try = next(
        (
            stmt
            for stmt in validate_home_function.body
            if isinstance(stmt, ast.Try)
            and any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "unsafe_home"
                    for target in node.targets
                )
                and isinstance(node.value, ast.BoolOp)
                and isinstance(node.value.op, ast.Or)
                for node in stmt.body
            )
        ),
        None,
    )
    assert guard_try is not None

    raises_exit_two = any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Attribute)
        and isinstance(node.exc.func.value, ast.Name)
        and node.exc.func.value.id == "typer"
        and node.exc.func.attr == "Exit"
        and len(node.exc.args) == 1
        and isinstance(node.exc.args[0], ast.Constant)
        and node.exc.args[0].value == 2
        for handler in guard_try.handlers
        for node in ast.walk(handler)
    )
    assert raises_exit_two


def test_source_validate_home_guard_orders_is_symlink_before_ancestor_walk() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    validate_home_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_validate_home_or_exit"
        ),
        None,
    )
    assert validate_home_function is not None

    guard_try = next(
        (stmt for stmt in validate_home_function.body if isinstance(stmt, ast.Try)),
        None,
    )
    assert guard_try is not None

    guard_assign = next(
        (
            stmt
            for stmt in guard_try.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "unsafe_home"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.BoolOp)
            and isinstance(stmt.value.op, ast.Or)
        ),
        None,
    )
    assert guard_assign is not None

    values = guard_assign.value.values
    assert len(values) == 2
    first, second = values
    assert isinstance(first, ast.Call)
    assert isinstance(first.func, ast.Name)
    assert first.func.id == "is_symlink_path"
    assert len(first.args) == 1
    assert isinstance(first.args[0], ast.Name)
    assert first.args[0].id == "home"

    assert isinstance(second, ast.Call)
    assert isinstance(second.func, ast.Name)
    assert second.func.id == "has_symlink_ancestor"
    assert len(second.args) == 1
    assert isinstance(second.args[0], ast.Name)
    assert second.args[0].id == "home"


def test_source_validate_home_guard_reports_invalid_home_message() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    validate_home_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_validate_home_or_exit"
        ),
        None,
    )
    assert validate_home_function is not None

    guard_if = next(
        (
            stmt
            for stmt in validate_home_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "unsafe_home"
        ),
        None,
    )
    assert guard_if is not None

    message_found = False
    for node in ast.walk(guard_if):
        if not isinstance(node, ast.Call):
            continue
        if (
            not isinstance(node.func, ast.Attribute)
            or not isinstance(node.func.value, ast.Name)
            or node.func.value.id != "console"
            or node.func.attr != "print"
            or not node.args
        ):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.JoinedStr):
            has_message = any(
                isinstance(part, ast.Constant)
                and isinstance(part.value, str)
                and "Invalid home" in part.value
                for part in arg.values
            )
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            has_message = "Invalid home" in arg.value
        else:
            has_message = False
        if has_message:
            message_found = True
            break

    assert message_found


def test_source_validate_home_guard_precedes_to_check_assignment_and_lstat_loop() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    validate_home_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_validate_home_or_exit"
        ),
        None,
    )
    assert validate_home_function is not None

    guard_try = next(
        (
            stmt
            for stmt in validate_home_function.body
            if isinstance(stmt, ast.Try)
            and any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "unsafe_home"
                    for target in node.targets
                )
                and isinstance(node.value, ast.BoolOp)
                and isinstance(node.value.op, ast.Or)
                for node in stmt.body
            )
        ),
        None,
    )
    assert guard_try is not None

    to_check_assign = next(
        (
            stmt
            for stmt in validate_home_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "to_check" for target in stmt.targets
            )
        ),
        None,
    )
    assert to_check_assign is not None

    lstat_lines = [
        node.lineno
        for node in ast.walk(validate_home_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.attr == "lstat"
    ]
    assert lstat_lines
    assert guard_try.lineno < to_check_assign.lineno
    assert guard_try.lineno < min(lstat_lines)


def test_source_validate_home_wraps_guard_calls_with_oserror_runtimeerror_handler() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    validate_home_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_validate_home_or_exit"
        ),
        None,
    )
    assert validate_home_function is not None

    guard_try = next(
        (
            stmt
            for stmt in validate_home_function.body
            if isinstance(stmt, ast.Try)
            and any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"is_symlink_path", "has_symlink_ancestor"}
                for node in ast.walk(stmt)
            )
        ),
        None,
    )
    assert guard_try is not None

    catches_expected = False
    for handler in guard_try.handlers:
        if handler.type is None:
            continue
        names: set[str] = set()
        if isinstance(handler.type, ast.Name):
            names.add(handler.type.id)
        elif isinstance(handler.type, ast.Tuple):
            for elt in handler.type.elts:
                if isinstance(elt, ast.Name):
                    names.add(elt.id)
        if {"OSError", "RuntimeError"}.issubset(names):
            raises_exit_two = any(
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and isinstance(node.exc.func, ast.Attribute)
                and isinstance(node.exc.func.value, ast.Name)
                and node.exc.func.value.id == "typer"
                and node.exc.func.attr == "Exit"
                and len(node.exc.args) == 1
                and isinstance(node.exc.args[0], ast.Constant)
                and node.exc.args[0].value == 2
                for node in ast.walk(handler)
            )
            assert raises_exit_two
            catches_expected = True
            break

    assert catches_expected


def test_source_cli_status_logs_validate_home_before_lock_and_load_state() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))

    for function_name in ("status", "logs"):
        cli_function = next(
            (
                node
                for node in ast.walk(cli_module)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        assert cli_function is not None

        validate_home_lines = [
            node.lineno
            for node in ast.walk(cli_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validate_home_or_exit"
        ]
        run_lock_lines = [
            node.lineno
            for node in ast.walk(cli_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_lock"
        ]
        load_state_lines = [
            node.lineno
            for node in ast.walk(cli_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_state"
        ]

        assert validate_home_lines, function_name
        assert run_lock_lines, function_name
        assert load_state_lines, function_name
        assert min(validate_home_lines) < min(run_lock_lines), function_name
        assert min(run_lock_lines) < min(load_state_lines), function_name


def test_source_cli_resume_validates_home_before_lock_and_plan_load() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    resume_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "resume"
        ),
        None,
    )
    assert resume_function is not None

    validate_home_lines = [
        node.lineno
        for node in ast.walk(resume_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_home_or_exit"
    ]
    run_lock_lines = [
        node.lineno
        for node in ast.walk(resume_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_lock"
    ]
    load_plan_lines = [
        node.lineno
        for node in ast.walk(resume_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_plan"
    ]

    assert validate_home_lines
    assert run_lock_lines
    assert load_plan_lines
    assert min(validate_home_lines) < min(run_lock_lines)
    assert min(run_lock_lines) < min(load_plan_lines)


def test_source_cli_cancel_catches_run_exists_oserror_and_runtimeerror() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    cancel_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "cancel"
        ),
        None,
    )
    assert cancel_function is not None

    run_exists_try = next(
        (
            stmt
            for stmt in cancel_function.body
            if isinstance(stmt, ast.Try)
            and any(
                isinstance(body_stmt, ast.Assign)
                and isinstance(body_stmt.value, ast.Call)
                and isinstance(body_stmt.value.func, ast.Name)
                and body_stmt.value.func.id == "_run_exists"
                for body_stmt in stmt.body
            )
        ),
        None,
    )
    assert run_exists_try is not None

    matched = False
    for handler in run_exists_try.handlers:
        if handler.type is None:
            continue
        names: set[str] = set()
        if isinstance(handler.type, ast.Name):
            names.add(handler.type.id)
        elif isinstance(handler.type, ast.Tuple):
            for elt in handler.type.elts:
                if isinstance(elt, ast.Name):
                    names.add(elt.id)
        if {"OSError", "RuntimeError"}.issubset(names):
            raises_exit_two = any(
                isinstance(stmt, ast.Raise)
                and isinstance(stmt.exc, ast.Call)
                and isinstance(stmt.exc.func, ast.Attribute)
                and isinstance(stmt.exc.func.value, ast.Name)
                and stmt.exc.func.value.id == "typer"
                and stmt.exc.func.attr == "Exit"
                and len(stmt.exc.args) == 1
                and isinstance(stmt.exc.args[0], ast.Constant)
                and stmt.exc.args[0].value == 2
                for stmt in handler.body
            )
            assert raises_exit_two
            matched = True
            break

    assert matched


def test_source_cli_cancel_reports_run_exists_inspection_failure_message() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    cancel_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "cancel"
        ),
        None,
    )
    assert cancel_function is not None

    run_exists_try = next(
        (
            stmt
            for stmt in cancel_function.body
            if isinstance(stmt, ast.Try)
            and any(
                isinstance(body_stmt, ast.Assign)
                and isinstance(body_stmt.value, ast.Call)
                and isinstance(body_stmt.value.func, ast.Name)
                and body_stmt.value.func.id == "_run_exists"
                for body_stmt in stmt.body
            )
        ),
        None,
    )
    assert run_exists_try is not None

    message_found = False
    for handler in run_exists_try.handlers:
        if handler.type is None:
            continue
        names: set[str] = set()
        if isinstance(handler.type, ast.Name):
            names.add(handler.type.id)
        elif isinstance(handler.type, ast.Tuple):
            for elt in handler.type.elts:
                if isinstance(elt, ast.Name):
                    names.add(elt.id)
        if not {"OSError", "RuntimeError"}.issubset(names):
            continue
        for stmt in handler.body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            if (
                not isinstance(call.func, ast.Attribute)
                or not isinstance(call.func.value, ast.Name)
                or call.func.value.id != "console"
                or call.func.attr != "print"
                or not call.args
            ):
                continue
            message_arg = call.args[0]
            if isinstance(message_arg, ast.JoinedStr):
                has_prefix = any(
                    isinstance(piece, ast.Constant)
                    and isinstance(piece.value, str)
                    and "Failed to inspect run" in piece.value
                    for piece in message_arg.values
                )
            elif isinstance(message_arg, ast.Constant) and isinstance(message_arg.value, str):
                has_prefix = "Failed to inspect run" in message_arg.value
            else:
                has_prefix = False
            if has_prefix:
                message_found = True
                break
        if message_found:
            break

    assert message_found


def test_source_run_exists_checks_guard_sequence_before_marker_checks() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_exists_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_exists"
        ),
        None,
    )
    assert run_exists_function is not None

    ancestor_guard_lines = [
        node.lineno
        for node in ast.walk(run_exists_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "has_symlink_ancestor"
    ]
    run_dir_lstat_lines = [
        node.lineno
        for node in ast.walk(run_exists_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "current_run_dir"
        and node.func.attr == "lstat"
    ]
    marker_check_lines = [
        node.lineno
        for node in ast.walk(run_exists_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_is_regular_non_symlink"
    ]

    assert ancestor_guard_lines
    assert run_dir_lstat_lines
    assert marker_check_lines
    assert min(ancestor_guard_lines) < min(run_dir_lstat_lines)
    assert min(run_dir_lstat_lines) < min(marker_check_lines)


def test_source_run_exists_checks_run_dir_shape_before_marker_checks() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_exists_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_exists"
        ),
        None,
    )
    assert run_exists_function is not None

    run_shape_guard_lines = [
        node.lineno
        for node in ast.walk(run_exists_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "stat"
        and node.func.attr in {"S_ISLNK", "S_ISDIR"}
    ]
    marker_check_lines = [
        node.lineno
        for node in ast.walk(run_exists_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_is_regular_non_symlink"
    ]

    assert run_shape_guard_lines
    assert marker_check_lines
    assert max(run_shape_guard_lines) < min(marker_check_lines)


def test_source_run_exists_marker_helper_uses_regular_file_predicate() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_exists_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_exists"
        ),
        None,
    )
    assert run_exists_function is not None

    marker_helper = next(
        (
            node
            for node in ast.walk(run_exists_function)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_is_regular_non_symlink"
        ),
        None,
    )
    assert marker_helper is not None

    marker_lstat_lines = [
        node.lineno
        for node in ast.walk(marker_helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "lstat"
    ]
    regular_predicate_lines = [
        node.lineno
        for node in ast.walk(marker_helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "stat"
        and node.func.attr == "S_ISREG"
    ]

    assert marker_lstat_lines
    assert regular_predicate_lines
    assert min(marker_lstat_lines) < min(regular_predicate_lines)


def test_source_run_exists_uses_ordered_or_for_state_then_plan_markers() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_exists_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_exists"
        ),
        None,
    )
    assert run_exists_function is not None

    marker_return = next(
        (
            node
            for node in ast.walk(run_exists_function)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.BoolOp)
            and isinstance(node.value.op, ast.Or)
        ),
        None,
    )
    assert marker_return is not None
    assert isinstance(marker_return.value, ast.BoolOp)

    marker_calls = [
        value
        for value in marker_return.value.values
        if isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "_is_regular_non_symlink"
    ]
    assert len(marker_calls) == 2

    marker_names: list[str] = []
    for call in marker_calls:
        assert len(call.args) == 1
        arg = call.args[0]
        assert isinstance(arg, ast.BinOp)
        assert isinstance(arg.op, ast.Div)
        assert isinstance(arg.left, ast.Name)
        assert arg.left.id == "current_run_dir"
        assert isinstance(arg.right, ast.Constant)
        assert isinstance(arg.right.value, str)
        marker_names.append(arg.right.value)

    assert marker_names == ["state.json", "plan.yaml"]


def test_source_run_exists_marker_helper_catches_oserror_and_runtimeerror_as_false() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_exists_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_exists"
        ),
        None,
    )
    assert run_exists_function is not None

    marker_helper = next(
        (
            node
            for node in ast.walk(run_exists_function)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_is_regular_non_symlink"
        ),
        None,
    )
    assert marker_helper is not None

    matched = False
    for handler in [n for n in ast.walk(marker_helper) if isinstance(n, ast.ExceptHandler)]:
        if handler.type is None:
            continue
        names: set[str] = set()
        if isinstance(handler.type, ast.Name):
            names.add(handler.type.id)
        elif isinstance(handler.type, ast.Tuple):
            for elt in handler.type.elts:
                if isinstance(elt, ast.Name):
                    names.add(elt.id)
        if {"OSError", "RuntimeError"}.issubset(names):
            returns_false = any(
                isinstance(stmt, ast.Return)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is False
                for stmt in handler.body
            )
            assert returns_false
            matched = True
            break

    assert matched


def test_source_run_exists_run_dir_lstat_catches_oserror_and_runtimeerror_as_false() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_exists_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_exists"
        ),
        None,
    )
    assert run_exists_function is not None

    run_dir_try = next(
        (
            node
            for node in run_exists_function.body
            if isinstance(node, ast.Try)
            and any(
                isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "run_meta"
                    for target in stmt.targets
                )
                for stmt in node.body
            )
        ),
        None,
    )
    assert run_dir_try is not None

    matched = False
    for handler in run_dir_try.handlers:
        if handler.type is None:
            continue
        names: set[str] = set()
        if isinstance(handler.type, ast.Name):
            names.add(handler.type.id)
        elif isinstance(handler.type, ast.Tuple):
            for elt in handler.type.elts:
                if isinstance(elt, ast.Name):
                    names.add(elt.id)
        if {"OSError", "RuntimeError"}.issubset(names):
            returns_false = any(
                isinstance(stmt, ast.Return)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is False
                for stmt in handler.body
            )
            assert returns_false
            matched = True
            break

    assert matched


def test_source_run_exists_ancestor_guard_returns_false_before_run_dir_lstat() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_module = ast.parse((src_root / "cli.py").read_text(encoding="utf-8"))
    run_exists_function = next(
        (
            node
            for node in ast.walk(cli_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_exists"
        ),
        None,
    )
    assert run_exists_function is not None

    ancestor_if = next(
        (
            stmt
            for stmt in run_exists_function.body
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Call)
            and isinstance(stmt.test.func, ast.Name)
            and stmt.test.func.id == "has_symlink_ancestor"
        ),
        None,
    )
    assert ancestor_if is not None
    assert len(ancestor_if.test.args) == 1
    assert isinstance(ancestor_if.test.args[0], ast.Name)
    assert ancestor_if.test.args[0].id == "current_run_dir"
    assert any(
        isinstance(stmt, ast.Return)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is False
        for stmt in ancestor_if.body
    )

    run_dir_lstat_lines = [
        node.lineno
        for node in ast.walk(run_exists_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "current_run_dir"
        and node.func.attr == "lstat"
    ]
    assert run_dir_lstat_lines
    assert ancestor_if.lineno < min(run_dir_lstat_lines)


def test_cli_integration_invalid_plan_workdir_matrix_keeps_plan_modes_and_toggle_orders() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_invalid_workdir_matrix"
        ),
        None,
    )
    assert matrix_function is not None

    flag_orders_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "flag_orders"
            and isinstance(stmt.value, ast.List)
        ),
        None,
    )
    assert flag_orders_assign is not None
    assert isinstance(flag_orders_assign.value, ast.List)
    toggle_orders: set[tuple[str, ...]] = set()
    for order_node in flag_orders_assign.value.elts:
        assert isinstance(order_node, ast.List)
        order_values: list[str] = []
        for item in order_node.elts:
            assert isinstance(item, ast.Constant)
            assert isinstance(item.value, str)
            order_values.append(item.value)
        toggle_orders.add(tuple(order_values))
    assert toggle_orders == {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }

    plan_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "plan_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert plan_modes_assign is not None
    assert isinstance(plan_modes_assign.value, ast.Tuple)
    plan_modes: set[str] = set()
    for mode_node in plan_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        plan_modes.add(mode_node.value)
    assert plan_modes == {
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    }


def test_cli_integration_invalid_plan_workdir_matrix_asserts_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_invalid_workdir_matrix"

    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef) and node.name == matrix_name
        ),
        None,
    )
    assert matrix_function is not None

    source_segment = ast.get_source_segment(integration_source, matrix_function)
    assert source_segment is not None
    assert '"Plan validation error" in output' in source_segment
    assert '"PLAN_PATH" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment


def test_cli_integration_missing_plan_path_matrix_keeps_plan_modes_and_toggle_orders() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_invalid_workdir_matrix"
        ),
        None,
    )
    assert matrix_function is not None

    flag_orders_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "flag_orders"
            and isinstance(stmt.value, ast.List)
        ),
        None,
    )
    assert flag_orders_assign is not None
    assert isinstance(flag_orders_assign.value, ast.List)
    toggle_orders: set[tuple[str, ...]] = set()
    for order_node in flag_orders_assign.value.elts:
        assert isinstance(order_node, ast.List)
        order_values: list[str] = []
        for item in order_node.elts:
            assert isinstance(item, ast.Constant)
            assert isinstance(item.value, str)
            order_values.append(item.value)
        toggle_orders.add(tuple(order_values))
    assert toggle_orders == {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }

    plan_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "plan_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert plan_modes_assign is not None
    assert isinstance(plan_modes_assign.value, ast.Tuple)
    plan_modes: set[str] = set()
    for mode_node in plan_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        plan_modes.add(mode_node.value)
    assert plan_modes == {
        "missing_path",
        "dangling_symlink_path",
        "symlink_ancestor_missing_path",
    }


def test_cli_integration_single_plan_precedence_tests_assert_no_runtime_summary() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    target_names = {
        "test_cli_run_dry_run_both_fail_fast_toggles_invalid_plan_precedes_invalid_workdir",
        "test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_invalid_plan_precedes_invalid_workdir",
    }

    matched_names: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in target_names:
            continue
        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert '"run_id:" not in output' in source_segment
        assert '"state:" not in output' in source_segment
        assert '"report:" not in output' in source_segment
        matched_names.add(node.name)

    assert matched_names == target_names


def test_cli_integration_single_plan_precedence_tests_assert_no_planpath_error() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    target_names = {
        "test_cli_run_dry_run_both_fail_fast_toggles_invalid_plan_precedes_invalid_workdir",
        "test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_invalid_plan_precedes_invalid_workdir",
    }

    matched_names: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in target_names:
            continue
        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert '"PLAN_PATH" not in output' in source_segment
        matched_names.add(node.name)

    assert matched_names == target_names


def test_cli_integration_missing_plan_path_matrix_asserts_no_symlink_component_message() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_invalid_workdir_matrix"
        ),
        None,
    )
    assert matrix_function is not None

    source_segment = ast.get_source_segment(integration_source, matrix_function)
    assert source_segment is not None
    assert '"contains symlink component" not in output' in source_segment


def test_cli_integration_missing_plan_path_matrix_asserts_no_runs_side_effect_check() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_invalid_workdir_matrix"
        ),
        None,
    )
    assert matrix_function is not None

    source_segment = ast.get_source_segment(integration_source, matrix_function)
    assert source_segment is not None
    assert 'assert not (home / "runs").exists(), context' in source_segment


def test_cli_integration_missing_plan_workdir_matrix_asserts_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_invalid_workdir_matrix"
        ),
        None,
    )
    assert matrix_function is not None

    source_segment = ast.get_source_segment(integration_source, matrix_function)
    assert source_segment is not None
    assert '"PLAN_PATH" in output' in source_segment
    assert "\"Invalid value for 'PLAN_PATH'\" in output" in source_segment
    assert '"Plan validation error" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"contains symlink component" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment


def test_cli_integration_missing_plan_vs_home_matrix_keeps_mode_and_toggle_sets() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_invalid_home_matrix"
        ),
        None,
    )
    assert matrix_function is not None

    flag_orders_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "flag_orders"
            and isinstance(stmt.value, ast.List)
        ),
        None,
    )
    assert flag_orders_assign is not None
    assert isinstance(flag_orders_assign.value, ast.List)
    toggle_orders: set[tuple[str, ...]] = set()
    for order_node in flag_orders_assign.value.elts:
        assert isinstance(order_node, ast.List)
        order_values: list[str] = []
        for item in order_node.elts:
            assert isinstance(item, ast.Constant)
            assert isinstance(item.value, str)
            order_values.append(item.value)
        toggle_orders.add(tuple(order_values))
    assert toggle_orders == {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }

    plan_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "plan_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert plan_modes_assign is not None
    assert isinstance(plan_modes_assign.value, ast.Tuple)
    plan_modes: set[str] = set()
    for mode_node in plan_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        plan_modes.add(mode_node.value)
    assert plan_modes == {
        "missing_path",
        "dangling_symlink_path",
        "symlink_ancestor_missing_path",
    }

    home_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "home_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert home_modes_assign is not None
    assert isinstance(home_modes_assign.value, ast.Tuple)
    home_modes: set[str] = set()
    for mode_node in home_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        home_modes.add(mode_node.value)
    assert home_modes == {
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }


def test_cli_integration_missing_plan_vs_home_matrix_asserts_planpath_and_home_exclusion() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_invalid_home_matrix"
        ),
        None,
    )
    assert matrix_function is not None

    source_segment = ast.get_source_segment(integration_source, matrix_function)
    assert source_segment is not None
    assert '"PLAN_PATH" in output' in source_segment
    assert "\"Invalid value for 'PLAN_PATH'\" in output" in source_segment
    assert '"Plan validation error" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"contains symlink component" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment


def test_cli_integration_missing_plan_vs_home_workdir_matrix_keeps_mode_sets() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_home_and_workdir_matrix"
    )
    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef) and node.name == matrix_name
        ),
        None,
    )
    assert matrix_function is not None

    flag_orders_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "flag_orders"
            and isinstance(stmt.value, ast.List)
        ),
        None,
    )
    assert flag_orders_assign is not None
    assert isinstance(flag_orders_assign.value, ast.List)
    toggle_orders: set[tuple[str, ...]] = set()
    for order_node in flag_orders_assign.value.elts:
        assert isinstance(order_node, ast.List)
        order_values: list[str] = []
        for item in order_node.elts:
            assert isinstance(item, ast.Constant)
            assert isinstance(item.value, str)
            order_values.append(item.value)
        toggle_orders.add(tuple(order_values))
    assert toggle_orders == {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }

    plan_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "plan_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert plan_modes_assign is not None
    assert isinstance(plan_modes_assign.value, ast.Tuple)
    plan_modes: set[str] = set()
    for mode_node in plan_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        plan_modes.add(mode_node.value)
    assert plan_modes == {
        "missing_path",
        "dangling_symlink_path",
        "symlink_ancestor_missing_path",
    }

    home_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "home_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert home_modes_assign is not None
    assert isinstance(home_modes_assign.value, ast.Tuple)
    home_modes: set[str] = set()
    for mode_node in home_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        home_modes.add(mode_node.value)
    assert home_modes == {
        "home_file",
        "file_ancestor",
        "symlink_to_dir",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }


def test_cli_integration_missing_plan_vs_home_workdir_matrix_asserts_priority_and_exclusion() -> (
    None
):
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_home_and_workdir_matrix"
    )

    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef) and node.name == matrix_name
        ),
        None,
    )
    assert matrix_function is not None

    source_segment = ast.get_source_segment(integration_source, matrix_function)
    assert source_segment is not None
    assert '"PLAN_PATH" in output' in source_segment
    assert "\"Invalid value for 'PLAN_PATH'\" in output" in source_segment
    assert '"Plan validation error" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"contains symlink component" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment


def test_cli_integration_missing_plan_vs_home_matrix_asserts_no_runs_side_effect_check() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_invalid_home_matrix"

    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef) and node.name == matrix_name
        ),
        None,
    )
    assert matrix_function is not None

    source_segment = ast.get_source_segment(integration_source, matrix_function)
    assert source_segment is not None
    assert 'assert not (root / "runs").exists(), context' in source_segment


def test_cli_integration_missing_plan_home_workdir_matrix_asserts_no_runs_side_effect() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_missing_plan_path_precedes_home_and_workdir_matrix"
    )

    matrix_function = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef) and node.name == matrix_name
        ),
        None,
    )
    assert matrix_function is not None

    source_segment = ast.get_source_segment(integration_source, matrix_function)
    assert source_segment is not None
    assert 'assert not (root / "runs").exists(), context' in source_segment
