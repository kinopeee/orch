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
