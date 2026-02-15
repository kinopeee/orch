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


def test_source_does_not_emit_symlink_component_detail_literal() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    offending_files: list[str] = []
    for file_path in src_root.rglob("*.py"):
        source = file_path.read_text(encoding="utf-8")
        if "contains symlink component" in source:
            offending_files.append(str(file_path.relative_to(src_root)))
    assert offending_files == []


def test_source_does_not_emit_symbolic_links_detail_literal() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    offending_files: list[str] = []
    forbidden_literals = (
        "symbolic link",
        "symbolic links",
        "symbolic-link",
        "symbolic_link",
        "symboliclink",
        "symboliclinks",
        "symbolic-linking",
        "symbolic_linking",
        "symboliclinking",
        "symbolically links",
        "symbolically-links",
        "symbolically_links",
        "symbolicallylinks",
        "symbolically linked",
        "symbolically-linked",
        "symbolically_linked",
        "symbolically--linked",
        "symbolically__linked",
        "symbolicallylinked",
    )
    for file_path in src_root.rglob("*.py"):
        source = file_path.read_text(encoding="utf-8").lower()
        if any(literal in source for literal in forbidden_literals):
            offending_files.append(str(file_path.relative_to(src_root)))
    assert offending_files == []


def test_cli_error_output_paths_use_sanitizer_helpers() -> None:
    cli_source = (Path(__file__).resolve().parents[1] / "src" / "orch" / "cli.py").read_text(
        encoding="utf-8"
    )

    required_runtime_fragments = {
        "[red]Failed to initialize run:[/red] {_render_runtime_error_detail(exc)}",
        "[red]Run execution failed:[/red] {_render_runtime_error_detail(exc)}",
        "[red]Run not found or broken:[/red] {_render_runtime_error_detail(exc)}",
        "[red]Failed to load state:[/red] {_render_runtime_error_detail(exc)}",
        "[red]Failed to inspect run:[/red] {_render_runtime_error_detail(exc)}",
        "[red]Failed to request cancel:[/red] {_render_runtime_error_detail(exc)}",
        "[red]{_render_runtime_error_detail(exc)}[/red]",
    }
    for fragment in required_runtime_fragments:
        assert fragment in cli_source

    assert (
        cli_source.count(
            "[yellow]Warning:[/yellow] failed to write report: {_render_runtime_error_detail(exc)}"
        )
        >= 2
    )

    forbidden_runtime_fragments = {
        "[red]Failed to initialize run:[/red] {exc}",
        "[red]Run execution failed:[/red] {exc}",
        "[red]Run not found or broken:[/red] {exc}",
        "[red]Failed to load state:[/red] {exc}",
        "[red]Failed to inspect run:[/red] {exc}",
        "[red]Failed to request cancel:[/red] {exc}",
        "[red]{exc}[/red]",
        "[yellow]Warning:[/yellow] failed to write report: {exc}",
    }
    for fragment in forbidden_runtime_fragments:
        assert fragment not in cli_source

    assert cli_source.count("[red]Plan validation error:[/red] {_render_plan_error(exc)}") >= 2
    assert "[red]Plan validation error:[/red] {exc}" not in cli_source


def test_source_cli_resume_conflict_handler_uses_runtime_sanitizer() -> None:
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

    conflict_handler = next(
        (
            handler
            for node in ast.walk(resume_function)
            if isinstance(node, ast.Try)
            for handler in node.handlers
            if isinstance(handler, ast.ExceptHandler)
            and isinstance(handler.type, ast.Name)
            and handler.type.id == "RunConflictError"
            and handler.name == "exc"
        ),
        None,
    )
    assert conflict_handler is not None

    print_calls = [
        stmt
        for stmt in conflict_handler.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id == "console"
        and stmt.value.func.attr == "print"
    ]
    assert len(print_calls) == 1
    print_call = print_calls[0].value
    assert isinstance(print_call, ast.Call)
    assert print_call.args

    rendered_segment = ast.get_source_segment(
        (src_root / "cli.py").read_text(encoding="utf-8"), print_call.args[0]
    )
    assert rendered_segment is not None
    assert "_render_runtime_error_detail(exc)" in rendered_segment
    assert "{exc}" not in rendered_segment


def test_source_cli_console_print_never_interpolates_exc_directly() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "orch"
    cli_source = (src_root / "cli.py").read_text(encoding="utf-8")
    cli_module = ast.parse(cli_source)

    violations: list[int] = []
    for node in ast.walk(cli_module):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "console"
            and node.func.attr == "print"
            and node.args
            and isinstance(node.args[0], ast.JoinedStr)
        ):
            continue

        if any(
            isinstance(part, ast.FormattedValue)
            and isinstance(part.value, ast.Name)
            and part.value.id == "exc"
            for part in node.args[0].values
        ):
            violations.append(node.lineno)

    assert violations == []


def test_cli_symlink_hint_pattern_shape_and_usage_are_stable() -> None:
    cli_source = (Path(__file__).resolve().parents[1] / "src" / "orch" / "cli.py").read_text(
        encoding="utf-8"
    )

    assert "_SYMLINK_HINT_PATTERN = re.compile(" in cli_source
    assert r"\bsymlink\w*\b|\bsymbolic(?:ally)?(?:[\s_-]+)?link(?:s|ed|ing)?\b" in cli_source
    assert "re.IGNORECASE" in cli_source

    assert "def _mentions_symlink(detail: str) -> bool:" in cli_source
    assert "return _SYMLINK_HINT_PATTERN.search(detail) is not None" in cli_source

    assert 'return "symlink" in normalized or "symbolic link" in normalized' not in cli_source
    assert 'return "symlink" in detail.lower()' not in cli_source


def test_cli_helpers_mentions_symlink_detection_matrix_exists() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    helpers_source = (tests_root / "test_cli_helpers.py").read_text(encoding="utf-8")
    helpers_module = ast.parse(helpers_source)

    expected: dict[str, tuple[str, ...]] = {
        "test_mentions_symlink_detects_supported_variants": (
            '"path contains symlink component",',
            '"path has symbolic-link reference",',
            '"path has symbolic_link reference",',
            '"path has symbolic\\rlink reference",',
            '"path has symbolic\\r\\nlinks reference",',
            '"path has symbolic\\flink reference",',
            '"path has symbolic\\vlinks reference",',
            '"path has symbolic-_link reference",',
            '"path has symbolic_-links reference",',
            '"PATH HAS SYMBOLIC_-LINKS REFERENCE",',
            '"PATH HAS SYMBOLIC-_LINK REFERENCE",',
            '"path has symbolic--link reference",',
            '"path has symbolic__links reference",',
            '"path has symbolic-linking issue",',
            '"path has symbolic_linking issue",',
            '"path is symbolically-linked",',
            '"path is symbolically--linked",',
            '"path is symbolically__linked",',
            '"PATH IS SYMBOLICALLY-LINKED",',
            '"PATH IS SYMBOLICALLY_LINKED",',
            '"PATH IS SYMBOLICALLYLINKED",',
            '"path has symbolically links issue",',
            '"path has symbolically-links issue",',
            '"path has symbolically_links issue",',
            '"path has symbolicallylinks issue",',
            '"PATH HAS SYMBOLICALLYLINKS ISSUE",',
            '"PATH HAS SYMBOLICALLY-LINKS ISSUE",',
            '"PATH HAS SYMBOLICALLY_LINKS ISSUE",',
            '"RUN PATH HAS SYMBOLIC_LINK REFERENCE",',
            "assert _mentions_symlink(detail) is True",
        ),
        "test_mentions_symlink_rejects_non_symlink_variants": (
            '"path has hardlink reference",',
            '"path references linker script",',
            '"path has symbolic-linker issue",',
            '"path has symbolic_linkless issue",',
            '"this error is about permissions only",',
            "assert _mentions_symlink(detail) is False",
        ),
    }

    matched: set[str] = set()
    source_lines = helpers_source.splitlines()
    for node in ast.walk(helpers_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected:
            continue

        start_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        source_segment = "\n".join(source_lines[start_line - 1 : node.end_lineno])
        for fragment in expected[node.name]:
            assert fragment in source_segment
        matched.add(node.name)

    assert matched == set(expected)


def test_cli_helpers_cover_symbolic_link_variant_sanitization_cases() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    helpers_source = (tests_root / "test_cli_helpers.py").read_text(encoding="utf-8")
    helpers_module = ast.parse(helpers_source)

    required_cases = {
        "test_render_plan_error_sanitizes_symbolic_link_detail": (
            'err = PlanError("plan path has symbolic link reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_uppercase_detail": (
            'err = PlanError("PLAN PATH HAS SYMBOLIC LINK REFERENCE: /TMP/PLAN.YAML")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_hyphenated_detail": (
            'err = PlanError("plan path has symbolic-link reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_hyphenated_uppercase_detail": (
            'err = PlanError("PLAN PATH HAS SYMBOLIC-LINK REFERENCE: /TMP/PLAN.YAML")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_underscored_detail": (
            'err = PlanError("plan path has symbolic_link reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_tab_separated_detail": (
            'err = PlanError("plan path has symbolic\\tlink reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_newline_separated_detail": (
            'err = PlanError("plan path has symbolic\\nlink reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_carriage_return_separated_detail": (
            'err = PlanError("plan path has symbolic\\rlink reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_links_crlf_separated_detail": (
            'err = PlanError("plan path has symbolic\\r\\nlinks reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_form_feed_separated_detail": (
            'err = PlanError("plan path has symbolic\\flink reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_links_vertical_tab_separated_detail": (
            'err = PlanError("plan path has symbolic\\vlinks reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_mixed_separator_detail": (
            'err = PlanError("plan path has symbolic-_link reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_links_mixed_separator_uppercase_detail": (
            'err = PlanError("PLAN PATH HAS SYMBOLIC_-LINKS REFERENCE: /TMP/PLAN.YAML")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_link_double_hyphen_separator_detail": (
            'err = PlanError("plan path has symbolic--link reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_linking_hyphenated_detail": (
            'err = PlanError("plan path has symbolic-linking issue: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_links_tab_separated_detail": (
            'err = PlanError("plan path has symbolic\\tlinks reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_links_newline_separated_detail": (
            'err = PlanError("plan path has symbolic\\nlinks reference: /tmp/plan.yaml")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_links_plural_detail": (
            'err = PlanError("too many levels of symbolic links in plan path")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_links_uppercase_detail": (
            'err = PlanError("TOO MANY LEVELS OF SYMBOLIC LINKS IN PLAN PATH")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_links_hyphenated_uppercase_detail": (
            'err = PlanError("TOO MANY LEVELS OF SYMBOLIC-LINKS IN PLAN PATH")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolic_links_underscored_uppercase_detail": (
            'err = PlanError("TOO MANY LEVELS OF SYMBOLIC_LINKS IN PLAN PATH")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_linked_detail": (
            'err = PlanError("plan path is symbolically linked to another location")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_linked_hyphenated_detail": (
            'err = PlanError("plan path is symbolically-linked to another location")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_linked_hyphenated_uppercase_detail": (
            'err = PlanError("PLAN PATH IS SYMBOLICALLY-LINKED TO ANOTHER LOCATION")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_linked_underscored_detail": (
            'err = PlanError("plan path is symbolically_linked to another location")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_linked_double_hyphen_detail": (
            'err = PlanError("plan path is symbolically--linked to another location")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolicallylinked_compact_detail": (
            'err = PlanError("plan path is symbolicallylinked to another location")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_links_plural_detail": (
            'err = PlanError("plan path has symbolically links issue")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_links_hyphenated_plural_detail": (
            'err = PlanError("plan path has symbolically-links issue")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_links_hyphenated_uppercase_detail": (
            'err = PlanError("PLAN PATH HAS SYMBOLICALLY-LINKS ISSUE")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_links_underscored_plural_detail": (
            'err = PlanError("plan path has symbolically_links issue")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolically_links_underscored_uppercase_detail": (
            'err = PlanError("PLAN PATH HAS SYMBOLICALLY_LINKS ISSUE")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolicallylinks_compact_uppercase_detail": (
            'err = PlanError("PLAN PATH HAS SYMBOLICALLYLINKS ISSUE")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symbolicallylinked_compact_uppercase_detail": (
            'err = PlanError("PLAN PATH IS SYMBOLICALLYLINKED TO ANOTHER LOCATION")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symboliclinks_compact_detail": (
            'err = PlanError("plan path has symboliclinks issue")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symboliclinks_compact_uppercase_detail": (
            'err = PlanError("PLAN PATH HAS SYMBOLICLINKS ISSUE")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symboliclink_compact_detail": (
            'err = PlanError("plan path has symboliclink issue")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symboliclink_compact_uppercase_detail": (
            'err = PlanError("PLAN PATH HAS SYMBOLICLINK ISSUE")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_plan_error_sanitizes_symlinked_detail": (
            'err = PlanError("plan path is symlinked to another location")',
            'assert _render_plan_error(err) == "invalid plan path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_link_detail": (
            'err = OSError("run path has symbolic link reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_link_uppercase_detail": (
            'err = OSError("RUN PATH HAS SYMBOLIC LINK REFERENCE")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_link_hyphenated_detail": (
            'err = OSError("run path has symbolic-link reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_link_underscored_uppercase_detail": (
            'err = OSError("RUN PATH HAS SYMBOLIC_LINK REFERENCE")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_link_underscored_detail": (
            'err = OSError("run path has symbolic_link reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_link_tab_separated_detail": (
            'err = OSError("run path has symbolic\\tlink reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_link_newline_separated_detail": (
            'err = OSError("run path has symbolic\\nlink reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        (
            "test_render_runtime_error_detail_sanitizes_symbolic_link_"
            "carriage_return_separated_detail"
        ): (
            'err = OSError("run path has symbolic\\rlink reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_crlf_separated_detail": (
            'err = OSError("run path has symbolic\\r\\nlinks reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_link_form_feed_separated_detail": (
            'err = OSError("run path has symbolic\\flink reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_vertical_tab_separated_detail": (
            'err = OSError("run path has symbolic\\vlinks reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_mixed_separator_detail": (
            'err = OSError("run path has symbolic_-links reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        (
            "test_render_runtime_error_detail_sanitizes_symbolic_link_"
            "mixed_separator_uppercase_detail"
        ): (
            'err = OSError("RUN PATH HAS SYMBOLIC-_LINK REFERENCE")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_double_underscore_detail": (
            'err = OSError("run path has symbolic__links reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_linking_underscored_detail": (
            'err = OSError("run path has symbolic_linking issue")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_tab_separated_detail": (
            'err = OSError("run path has symbolic\\tlinks reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_newline_separated_detail": (
            'err = OSError("run path has symbolic\\nlinks reference")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_plural_detail": (
            'err = OSError("too many levels of symbolic links in run path")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_uppercase_detail": (
            'err = OSError("TOO MANY LEVELS OF SYMBOLIC LINKS IN RUN PATH")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_hyphenated_uppercase_detail": (
            'err = OSError("TOO MANY LEVELS OF SYMBOLIC-LINKS IN RUN PATH")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolic_links_underscored_uppercase_detail": (
            'err = OSError("TOO MANY LEVELS OF SYMBOLIC_LINKS IN RUN PATH")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolically_linked_detail": (
            'err = OSError("run path is symbolically linked to another location")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolically_linked_hyphenated_detail": (
            'err = OSError("run path is symbolically-linked to another location")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolically_linked_uppercase_underscored": (
            'err = OSError("RUN PATH IS SYMBOLICALLY_LINKED TO ANOTHER LOCATION")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolically_linked_underscored_detail": (
            'err = OSError("run path is symbolically_linked to another location")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        (
            "test_render_runtime_error_detail_sanitizes_symbolically_linked_"
            "double_underscore_detail"
        ): (
            'err = OSError("run path is symbolically__linked to another location")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolicallylinked_compact_detail": (
            'err = OSError("run path is symbolicallylinked to another location")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolicallylinks_compact_detail": (
            'err = OSError("run path has symbolicallylinks issue")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        (
            "test_render_runtime_error_detail_sanitizes_symbolically_links_hyphenated_plural_detail"
        ): (
            'err = OSError("run path has symbolically-links issue")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        (
            "test_render_runtime_error_detail_sanitizes_symbolically_links_"
            "hyphenated_uppercase_detail"
        ): (
            'err = OSError("RUN PATH HAS SYMBOLICALLY-LINKS ISSUE")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        (
            "test_render_runtime_error_detail_sanitizes_symbolically_links_"
            "underscored_plural_detail"
        ): (
            'err = OSError("run path has symbolically_links issue")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        ("test_render_runtime_error_detail_sanitizes_symbolically_links_uppercase_underscored"): (
            'err = OSError("RUN PATH HAS SYMBOLICALLY_LINKS ISSUE")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        ("test_render_runtime_error_detail_sanitizes_symbolicallylinks_compact_uppercase_detail"): (
            'err = OSError("RUN PATH HAS SYMBOLICALLYLINKS ISSUE")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symbolicallylinked_uppercase_compact": (
            'err = OSError("RUN PATH IS SYMBOLICALLYLINKED TO ANOTHER LOCATION")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symboliclinks_compact_detail": (
            'err = OSError("run path has symboliclinks issue")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symboliclinks_compact_uppercase_detail": (
            'err = OSError("RUN PATH HAS SYMBOLICLINKS ISSUE")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symboliclink_compact_detail": (
            'err = OSError("run path has symboliclink issue")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symboliclink_compact_uppercase_detail": (
            'err = OSError("RUN PATH HAS SYMBOLICLINK ISSUE")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
        "test_render_runtime_error_detail_sanitizes_symlinked_detail": (
            'err = OSError("run path is symlinked to another location")',
            'assert _render_runtime_error_detail(err) == "invalid run path"',
        ),
    }

    matched: set[str] = set()
    for node in ast.walk(helpers_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in required_cases:
            continue

        source_segment = ast.get_source_segment(helpers_source, node)
        assert source_segment is not None
        expected_error_line, expected_assert_line = required_cases[node.name]
        assert expected_error_line in source_segment
        assert expected_assert_line in source_segment
        matched.add(node.name)

    assert matched == set(required_cases)


def test_cli_helpers_run_resume_plan_error_symbolic_variants_are_sanitized() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    helpers_source = (tests_root / "test_cli_helpers.py").read_text(encoding="utf-8")
    helpers_module = ast.parse(helpers_source)

    expected_names = {
        "test_cli_run_sanitizes_symbolically_linked_plan_error",
        "test_cli_resume_sanitizes_symbolically_linked_plan_error",
    }

    matched: set[str] = set()
    for node in ast.walk(helpers_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_names:
            continue

        source_segment = ast.get_source_segment(helpers_source, node)
        assert source_segment is not None
        assert 'assert "Plan validation error" in captured.out' in source_segment
        assert 'assert "invalid plan path" in captured.out' in source_segment
        assert 'assert "symbolic links" not in captured.out.lower()' in source_segment
        assert 'assert "contains symlink component" not in captured.out' in source_segment
        assert 'assert "must not include symlink" not in captured.out' in source_segment
        assert 'assert "must not be symlink" not in captured.out' in source_segment
        matched.add(node.name)

    assert matched == expected_names


def test_cli_helpers_run_resume_non_symlink_symbolic_details_are_preserved() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    helpers_source = (tests_root / "test_cli_helpers.py").read_text(encoding="utf-8")
    helpers_module = ast.parse(helpers_source)

    expected_checks = {
        "test_cli_run_keeps_symbolic_linker_plan_error_detail": "symbolic-linker issue",
        "test_cli_run_keeps_symbolic_linkless_plan_error_detail": "symbolic_linkless issue",
        "test_cli_resume_keeps_symbolic_linker_plan_error_detail": "symbolic-linker issue",
        "test_cli_resume_keeps_symbolic_linkless_plan_error_detail": "symbolic_linkless issue",
        "test_cli_resume_keeps_symbolic_linker_conflict_error_detail": "symbolic-linker issue",
        "test_cli_resume_keeps_symbolic_linkless_conflict_error_detail": "symbolic_linkless issue",
        "test_cli_status_keeps_symbolic_linkless_runtime_load_error_detail": (
            "symbolic_linkless issue"
        ),
        "test_cli_status_keeps_symbolic_linker_runtime_load_error_detail": (
            "symbolic-linker issue"
        ),
        "test_cli_logs_keeps_symbolic_linker_runtime_load_error_detail": ("symbolic-linker issue"),
        "test_cli_logs_keeps_symbolic_linkless_runtime_load_error_detail": (
            "symbolic_linkless issue"
        ),
        "test_cli_cancel_keeps_symbolic_linkless_write_error_detail": ("symbolic_linkless issue"),
        "test_cli_cancel_keeps_symbolic_linker_write_error_detail": ("symbolic-linker issue"),
    }

    matched: set[str] = set()
    for node in ast.walk(helpers_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_checks:
            continue

        source_segment = ast.get_source_segment(helpers_source, node)
        assert source_segment is not None
        expected_fragment = expected_checks[node.name]
        assert expected_fragment in source_segment
        if "runtime_load_error" in node.name:
            assert 'assert "Failed to load state" in captured.out' in source_segment
            assert 'assert "invalid run path" not in captured.out' in source_segment
        elif "write_error" in node.name:
            assert 'assert "Failed to request cancel" in captured.out' in source_segment
            assert 'assert "invalid run path" not in captured.out' in source_segment
        elif "conflict_error" in node.name:
            assert "exc_info.value.exit_code == 3" in source_segment
            assert 'assert "invalid run path" not in captured.out' in source_segment
        else:
            assert 'assert "Plan validation error" in captured.out' in source_segment
            assert 'assert "invalid plan path" not in captured.out' in source_segment
        matched.add(node.name)

    assert matched == set(expected_checks)


def test_cli_helpers_runtime_symbolic_links_variants_are_sanitized() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    helpers_source = (tests_root / "test_cli_helpers.py").read_text(encoding="utf-8")
    helpers_module = ast.parse(helpers_source)

    expected_checks = {
        "test_cli_run_sanitizes_symbolic_links_execution_error": "Run execution failed",
        "test_cli_resume_sanitizes_symbolic_links_runtime_lock_error": "Run not found or broken",
        "test_cli_resume_sanitizes_symbolic_links_conflict_error": "invalid run path",
        "test_cli_status_sanitizes_symbolic_links_runtime_load_error": "Failed to load state",
        "test_cli_logs_sanitizes_symbolic_links_runtime_load_error": "Failed to load state",
        "test_cli_cancel_sanitizes_symbolic_links_write_error": "Failed to request cancel",
        "test_cli_run_sanitizes_symbolic_links_report_write_warning": "failed to write report",
        "test_cli_resume_sanitizes_symbolic_links_report_write_warning": "failed to write report",
    }

    matched: set[str] = set()
    for node in ast.walk(helpers_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_checks:
            continue

        source_segment = ast.get_source_segment(helpers_source, node)
        assert source_segment is not None
        assert f'assert "{expected_checks[node.name]}" in captured.out' in source_segment
        assert 'assert "invalid run path" in captured.out' in source_segment
        assert 'assert "symbolic links" not in captured.out.lower()' in source_segment
        assert 'assert "must not include symlink" not in captured.out' in source_segment
        assert 'assert "must not be symlink" not in captured.out' in source_segment
        matched.add(node.name)

    assert matched == set(expected_checks)


def test_cli_helpers_runtime_symlink_named_tests_suppress_symbolic_links_detail() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    helpers_source = (tests_root / "test_cli_helpers.py").read_text(encoding="utf-8")
    helpers_module = ast.parse(helpers_source)

    expected_names = {
        "test_cli_run_sanitizes_symlink_initialize_error",
        "test_cli_run_sanitizes_symlink_execution_error",
        "test_cli_resume_sanitizes_symlink_runtime_lock_error",
        "test_cli_status_sanitizes_symlink_runtime_load_error",
        "test_cli_logs_sanitizes_symlink_runtime_load_error",
        "test_cli_cancel_sanitizes_symlink_runtime_run_exists_error_without_write",
        "test_cli_cancel_sanitizes_symlink_write_error",
        "test_cli_run_sanitizes_symlink_report_write_warning",
    }

    matched: set[str] = set()
    for node in ast.walk(helpers_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_names:
            continue

        source_segment = ast.get_source_segment(helpers_source, node)
        assert source_segment is not None
        assert 'assert "invalid run path" in captured.out' in source_segment
        assert 'assert "symbolic links" not in captured.out.lower()' in source_segment
        assert 'assert "must not include symlink" not in captured.out' in source_segment
        assert 'assert "must not be symlink" not in captured.out' in source_segment
        matched.add(node.name)

    assert matched == expected_names


def test_cli_helpers_sanitizer_output_tests_require_symbolic_links_suppression() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    helpers_source = (tests_root / "test_cli_helpers.py").read_text(encoding="utf-8")
    helpers_module = ast.parse(helpers_source)

    examined: set[str] = set()
    for node in ast.walk(helpers_module):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if "sanitizes_" not in node.name:
            continue
        if "symlink" not in node.name and "symbolic" not in node.name:
            continue

        source_segment = ast.get_source_segment(helpers_source, node)
        assert source_segment is not None
        requires_plan = 'assert "invalid plan path" in captured.out' in source_segment
        requires_run = 'assert "invalid run path" in captured.out' in source_segment
        if not requires_plan and not requires_run:
            continue

        assert 'assert "symbolic links" not in captured.out.lower()' in source_segment
        assert 'assert "symbolic link" not in captured.out.lower()' in source_segment
        assert 'assert "must not include symlink" not in captured.out' in source_segment
        assert 'assert "must not be symlink" not in captured.out' in source_segment
        examined.add(node.name)

    assert examined


def test_cli_helpers_invalid_plan_path_output_requires_full_suppression() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    helpers_source = (tests_root / "test_cli_helpers.py").read_text(encoding="utf-8")
    helpers_module = ast.parse(helpers_source)

    examined: set[str] = set()
    for node in ast.walk(helpers_module):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue

        source_segment = ast.get_source_segment(helpers_source, node)
        assert source_segment is not None
        if 'assert "invalid plan path" in captured.out' not in source_segment:
            continue

        assert 'assert "symbolic links" not in captured.out.lower()' in source_segment
        assert 'assert "symbolic link" not in captured.out.lower()' in source_segment
        assert 'assert "contains symlink component" not in captured.out' in source_segment
        assert 'assert "must not include symlink" not in captured.out' in source_segment
        assert 'assert "must not be symlink" not in captured.out' in source_segment
        examined.add(node.name)

    assert examined


def test_cli_helpers_plural_symbolic_links_assertions_require_singular_pair() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    helpers_source = (tests_root / "test_cli_helpers.py").read_text(encoding="utf-8")
    helper_lines = helpers_source.splitlines()

    examined = 0
    for index, line in enumerate(helper_lines):
        stripped = line.strip()
        if stripped != 'assert "symbolic links" not in captured.out.lower()':
            continue

        examined += 1
        next_line = helper_lines[index + 1].strip() if index + 1 < len(helper_lines) else ""
        assert next_line == 'assert "symbolic link" not in captured.out.lower()'

    assert examined > 0


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

    with pytest.raises(OSError, match="must not include symlink"):
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


def test_cli_integration_invalid_plan_existing_home_workdir_matrix_keeps_modes_and_toggles() -> (
    None
):
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix"
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
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    }


def test_cli_integration_invalid_plan_existing_home_workdir_matrix_asserts_output_contract() -> (
    None
):
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix"
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
    assert '"Plan validation error" in output' in source_segment
    assert '"PLAN_PATH" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert home.exists(), context" in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment
    assert "assert sorted(path.name for path in home.iterdir()) == [], context" in source_segment
    assert "cwd=case_root" not in source_segment
    assert '"--home"' in source_segment
    assert '"--workdir"' in source_segment


def test_cli_integration_invalid_plan_existing_home_matrix_keeps_modes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix"

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
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    }


def test_cli_integration_invalid_plan_existing_home_matrix_asserts_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix"

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
    assert "assert home.exists(), context" in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment
    assert "assert sorted(path.name for path in home.iterdir()) == [], context" in source_segment
    assert "cwd=case_root" not in source_segment
    assert '"--home"' in source_segment
    assert '"--workdir"' not in source_segment


def test_cli_integration_invalid_plan_default_home_workdir_matrix_keeps_modes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_home_matrix"
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
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    }


def test_cli_integration_invalid_plan_default_home_workdir_matrix_asserts_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_home_matrix"
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
    assert '"Plan validation error" in output' in source_segment
    assert '"PLAN_PATH" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert not default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert '"--workdir"' in source_segment
    assert "cwd=case_root" in source_segment


def test_cli_integration_invalid_plan_default_home_existing_workdir_matrix_keeps_modes() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_"
        "default_existing_home_matrix"
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
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    }


def test_cli_integration_invalid_plan_defhome_existing_workdir_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_"
        "default_existing_home_matrix"
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
    assert '"Plan validation error" in output' in source_segment
    assert '"PLAN_PATH" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert (
        "assert sorted(path.name for path in default_home.iterdir()) == [], context"
        in source_segment
    )
    assert '"--home"' not in source_segment
    assert '"--workdir"' in source_segment
    assert "cwd=case_root" in source_segment


def test_cli_integration_invalid_plan_default_home_matrix_keeps_modes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_home_matrix"

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
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    }


def test_cli_integration_invalid_plan_default_home_matrix_asserts_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_home_matrix"

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
    assert "assert not default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert "cwd=case_root" in source_segment


def test_cli_integration_invalid_plan_defhome_existing_matrix_keeps_modes() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix"
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
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    }


def test_cli_integration_invalid_plan_defhome_existing_matrix_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix"
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
    assert '"Plan validation error" in output' in source_segment
    assert '"PLAN_PATH" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert (
        "assert sorted(path.name for path in default_home.iterdir()) == [], context"
        in source_segment
    )
    assert '"--home"' not in source_segment
    assert "cwd=case_root" in source_segment


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
    assert "assert not home.exists(), context" in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment


def test_cli_integration_missing_plan_reject_matrix_keeps_mode_and_toggle_sets() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_matrix"

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


def test_cli_integration_missing_plan_reject_matrix_asserts_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_matrix"

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
    assert "assert not home.exists(), context" in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment


def test_cli_integration_missing_plan_reject_default_home_matrix_keeps_mode_and_toggle_sets() -> (
    None
):
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_default_home_matrix"

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


def test_cli_integration_missing_plan_reject_default_home_matrix_asserts_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_default_home_matrix"

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
    assert "assert not default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert "cwd=case_root" in source_segment


def test_cli_integration_missing_plan_default_home_workdir_matrix_keeps_mode_and_toggle_sets() -> (
    None
):
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_home_precedes_workdir_matrix"
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


def test_cli_integration_missing_plan_default_home_workdir_matrix_asserts_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_home_precedes_workdir_matrix"
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
    assert "assert not default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert "cwd=case_root" in source_segment
    assert '"--workdir"' in source_segment


def test_cli_integration_missing_plan_reject_defhome_existing_keeps_modes() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_default_existing_home_matrix"
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


def test_cli_integration_missing_plan_reject_defhome_existing_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_default_existing_home_matrix"
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
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert (
        "assert sorted(path.name for path in default_home.iterdir()) == [], context"
        in source_segment
    )
    assert '"--home"' not in source_segment
    assert "cwd=case_root" in source_segment


def test_cli_integration_missing_plan_defhome_existing_workdir_keeps_modes() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_existing_home_"
        "precedes_workdir_matrix"
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


def test_cli_integration_missing_plan_defhome_existing_workdir_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_existing_home_"
        "precedes_workdir_matrix"
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
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert (
        "assert sorted(path.name for path in default_home.iterdir()) == [], context"
        in source_segment
    )
    assert '"--home"' not in source_segment
    assert "cwd=case_root" in source_segment
    assert '"--workdir"' in source_segment


def test_cli_integration_missing_plan_default_home_cases_keep_cwd_and_home_contracts() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    reject_default_home = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_default_home_matrix"
    )
    reject_default_existing_home = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_default_existing_home_matrix"
    )
    workdir_default_home = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_home_precedes_workdir_matrix"
    )
    workdir_default_existing_home = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_existing_home_"
        "precedes_workdir_matrix"
    )

    expectations = {
        reject_default_home: {
            "home_assert": "assert not default_home.exists(), context",
            "needs_workdir": False,
            "expects_home_empty_assert": False,
        },
        workdir_default_home: {
            "home_assert": "assert not default_home.exists(), context",
            "needs_workdir": True,
            "expects_home_empty_assert": False,
        },
        reject_default_existing_home: {
            "home_assert": "assert default_home.exists(), context",
            "needs_workdir": False,
            "expects_home_empty_assert": True,
        },
        workdir_default_existing_home: {
            "home_assert": "assert default_home.exists(), context",
            "needs_workdir": True,
            "expects_home_empty_assert": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]

        assert "cwd=case_root" in source_segment
        assert expected["home_assert"] in source_segment
        assert 'assert not (default_home / "runs").exists(), context' in source_segment
        assert '"--home"' not in source_segment
        if expected["expects_home_empty_assert"]:
            assert (
                "assert sorted(path.name for path in default_home.iterdir()) == [], context"
                in source_segment
            )
        else:
            assert (
                "assert sorted(path.name for path in default_home.iterdir()) == [], context"
                not in source_segment
            )

        if expected["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_invalid_plan_default_home_cases_keep_cwd_and_home_contracts() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    reject_default_home = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_home_matrix"
    )
    reject_default_existing_home = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix"
    )
    workdir_default_home = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_home_matrix"
    )
    workdir_default_existing_home = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_"
        "default_existing_home_matrix"
    )

    expectations = {
        reject_default_home: {
            "home_assert": "assert not default_home.exists(), context",
            "needs_workdir": False,
            "expects_home_empty_assert": False,
        },
        workdir_default_home: {
            "home_assert": "assert not default_home.exists(), context",
            "needs_workdir": True,
            "expects_home_empty_assert": False,
        },
        reject_default_existing_home: {
            "home_assert": "assert default_home.exists(), context",
            "needs_workdir": False,
            "expects_home_empty_assert": True,
        },
        workdir_default_existing_home: {
            "home_assert": "assert default_home.exists(), context",
            "needs_workdir": True,
            "expects_home_empty_assert": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]

        assert "cwd=case_root" in source_segment
        assert expected["home_assert"] in source_segment
        assert 'assert not (default_home / "runs").exists(), context' in source_segment
        assert '"--home"' not in source_segment
        if expected["expects_home_empty_assert"]:
            assert (
                "assert sorted(path.name for path in default_home.iterdir()) == [], context"
                in source_segment
            )
        else:
            assert (
                "assert sorted(path.name for path in default_home.iterdir()) == [], context"
                not in source_segment
            )

        if expected["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_invalid_plan_existing_home_cases_keep_home_contracts() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    reject_existing_home = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix"
    )
    workdir_existing_home = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix"
    )

    expectations = {
        reject_existing_home: {"needs_workdir": False},
        workdir_existing_home: {"needs_workdir": True},
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]

        assert "cwd=case_root" not in source_segment
        assert "assert home.exists(), context" in source_segment
        assert 'assert not (home / "runs").exists(), context' in source_segment
        assert (
            "assert sorted(path.name for path in home.iterdir()) == [], context" in source_segment
        )
        assert '"--home"' in source_segment

        if expected["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_explicit_existing_home_plan_error_cases_keep_contracts() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expectations = {
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix": {
            "anchor": '"Plan validation error" in output',
            "needs_workdir": True,
            "present": [],
            "absent": [
                '"PLAN_PATH" not in output',
                '"Invalid home" not in output',
                '"Invalid workdir" not in output',
            ],
        },
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix": {
            "anchor": '"Plan validation error" in output',
            "needs_workdir": False,
            "present": [],
            "absent": [
                '"PLAN_PATH" not in output',
                '"Invalid home" not in output',
                '"Invalid workdir" not in output',
            ],
        },
        "test_cli_run_dry_run_both_toggles_missing_plan_precedes_workdir_existing_home_matrix": {
            "anchor": '"PLAN_PATH" in output',
            "needs_workdir": True,
            "present": ["\"Invalid value for 'PLAN_PATH'\" in output"],
            "absent": [
                '"Plan validation error" not in output',
                '"Invalid home" not in output',
                '"Invalid workdir" not in output',
                '"contains symlink component" not in output',
            ],
        },
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_existing_home_matrix": {
            "anchor": '"PLAN_PATH" in output',
            "needs_workdir": False,
            "present": ["\"Invalid value for 'PLAN_PATH'\" in output"],
            "absent": [
                '"Plan validation error" not in output',
                '"Invalid home" not in output',
                '"Invalid workdir" not in output',
                '"contains symlink component" not in output',
            ],
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]

        assert expected["anchor"] in source_segment
        assert "cwd=case_root" not in source_segment
        assert "assert home.exists(), context" in source_segment
        assert 'assert not (home / "runs").exists(), context' in source_segment
        assert (
            "assert sorted(path.name for path in home.iterdir()) == [], context" in source_segment
        )
        assert '"--home"' in source_segment
        assert '"Dry Run" not in output' in source_segment
        assert '"run_id:" not in output' in source_segment
        assert '"state:" not in output' in source_segment
        assert '"report:" not in output' in source_segment

        for snippet in expected["present"]:
            assert snippet in source_segment
        for snippet in expected["absent"]:
            assert snippet in source_segment

        if expected["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_explicit_existing_home_plan_error_cases_keep_modes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expected_toggle_orders = {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }
    expectations = {
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix": {
            "plan_modes": {
                "invalid_yaml",
                "unknown_root_field",
                "unknown_task_field",
                "non_regular_fifo",
                "symlink_plan",
                "symlink_ancestor_plan",
            },
            "needs_workdir": True,
        },
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix": {
            "plan_modes": {
                "invalid_yaml",
                "unknown_root_field",
                "unknown_task_field",
                "non_regular_fifo",
                "symlink_plan",
                "symlink_ancestor_plan",
            },
            "needs_workdir": False,
        },
        "test_cli_run_dry_run_both_toggles_missing_plan_precedes_workdir_existing_home_matrix": {
            "plan_modes": {
                "missing_path",
                "dangling_symlink_path",
                "symlink_ancestor_missing_path",
            },
            "needs_workdir": True,
        },
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_existing_home_matrix": {
            "plan_modes": {
                "missing_path",
                "dangling_symlink_path",
                "symlink_ancestor_missing_path",
            },
            "needs_workdir": False,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        flag_orders_assign = next(
            (
                stmt
                for stmt in node.body
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
        assert toggle_orders == expected_toggle_orders

        plan_modes_assign = next(
            (
                stmt
                for stmt in node.body
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
        actual_plan_modes: set[str] = set()
        for mode_node in plan_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            actual_plan_modes.add(mode_node.value)
        assert actual_plan_modes == expectations[node.name]["plan_modes"]

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert "cwd=case_root" not in source_segment
        assert '"--home"' in source_segment
        if expectations[node.name]["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_default_existing_home_plan_error_cases_keep_contracts() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    invalid_plan_with_workdir = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_"
        "default_existing_home_matrix"
    )
    invalid_plan_only = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix"
    )
    missing_plan_with_workdir = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_existing_home_"
        "precedes_workdir_matrix"
    )
    missing_plan_only = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_default_existing_home_matrix"
    )

    expectations = {
        invalid_plan_with_workdir: {
            "anchor": '"Plan validation error" in output',
            "needs_workdir": True,
            "present": [],
            "absent": [
                '"PLAN_PATH" not in output',
                '"Invalid home" not in output',
                '"Invalid workdir" not in output',
            ],
        },
        invalid_plan_only: {
            "anchor": '"Plan validation error" in output',
            "needs_workdir": False,
            "present": [],
            "absent": [
                '"PLAN_PATH" not in output',
                '"Invalid home" not in output',
                '"Invalid workdir" not in output',
            ],
        },
        missing_plan_with_workdir: {
            "anchor": '"PLAN_PATH" in output',
            "needs_workdir": True,
            "present": ["\"Invalid value for 'PLAN_PATH'\" in output"],
            "absent": [
                '"Plan validation error" not in output',
                '"Invalid home" not in output',
                '"Invalid workdir" not in output',
                '"contains symlink component" not in output',
            ],
        },
        missing_plan_only: {
            "anchor": '"PLAN_PATH" in output',
            "needs_workdir": False,
            "present": ["\"Invalid value for 'PLAN_PATH'\" in output"],
            "absent": [
                '"Plan validation error" not in output',
                '"Invalid home" not in output',
                '"Invalid workdir" not in output',
                '"contains symlink component" not in output',
            ],
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]

        assert expected["anchor"] in source_segment
        assert "cwd=case_root" in source_segment
        assert "assert default_home.exists(), context" in source_segment
        assert 'assert not (default_home / "runs").exists(), context' in source_segment
        assert (
            "assert sorted(path.name for path in default_home.iterdir()) == [], context"
            in source_segment
        )
        assert '"--home"' not in source_segment
        assert '"Dry Run" not in output' in source_segment
        assert '"run_id:" not in output' in source_segment
        assert '"state:" not in output' in source_segment
        assert '"report:" not in output' in source_segment

        for snippet in expected["present"]:
            assert snippet in source_segment
        for snippet in expected["absent"]:
            assert snippet in source_segment

        if expected["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_default_existing_home_plan_error_cases_keep_modes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expected_toggle_orders = {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }
    invalid_plan_with_workdir = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_"
        "default_existing_home_matrix"
    )
    invalid_plan_only = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix"
    )
    missing_plan_with_workdir = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_existing_home_"
        "precedes_workdir_matrix"
    )
    missing_plan_only = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_default_existing_home_matrix"
    )

    expectations = {
        invalid_plan_with_workdir: {
            "plan_modes": {
                "invalid_yaml",
                "unknown_root_field",
                "unknown_task_field",
                "non_regular_fifo",
                "symlink_plan",
                "symlink_ancestor_plan",
            },
            "needs_workdir": True,
        },
        invalid_plan_only: {
            "plan_modes": {
                "invalid_yaml",
                "unknown_root_field",
                "unknown_task_field",
                "non_regular_fifo",
                "symlink_plan",
                "symlink_ancestor_plan",
            },
            "needs_workdir": False,
        },
        missing_plan_with_workdir: {
            "plan_modes": {
                "missing_path",
                "dangling_symlink_path",
                "symlink_ancestor_missing_path",
            },
            "needs_workdir": True,
        },
        missing_plan_only: {
            "plan_modes": {
                "missing_path",
                "dangling_symlink_path",
                "symlink_ancestor_missing_path",
            },
            "needs_workdir": False,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        flag_orders_assign = next(
            (
                stmt
                for stmt in node.body
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
        assert toggle_orders == expected_toggle_orders

        plan_modes_assign = next(
            (
                stmt
                for stmt in node.body
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
        actual_plan_modes: set[str] = set()
        for mode_node in plan_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            actual_plan_modes.add(mode_node.value)
        assert actual_plan_modes == expectations[node.name]["plan_modes"]

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert "cwd=case_root" in source_segment
        assert '"--home"' not in source_segment
        if expectations[node.name]["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_existing_home_plan_error_groups_keep_modes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expected_toggle_orders = {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }
    invalid_plan_modes = {
        "invalid_yaml",
        "unknown_root_field",
        "unknown_task_field",
        "non_regular_fifo",
        "symlink_plan",
        "symlink_ancestor_plan",
    }
    missing_plan_modes = {
        "missing_path",
        "dangling_symlink_path",
        "symlink_ancestor_missing_path",
    }
    invalid_plan_default_with_workdir = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_"
        "default_existing_home_matrix"
    )
    invalid_plan_default_only = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix"
    )
    missing_plan_default_with_workdir = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_existing_home_"
        "precedes_workdir_matrix"
    )
    missing_plan_default_only = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_default_existing_home_matrix"
    )

    expectations = {
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix": {
            "plan_modes": invalid_plan_modes,
            "needs_workdir": True,
            "has_cwd": False,
            "uses_home_flag": True,
        },
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix": {
            "plan_modes": invalid_plan_modes,
            "needs_workdir": False,
            "has_cwd": False,
            "uses_home_flag": True,
        },
        "test_cli_run_dry_run_both_toggles_missing_plan_precedes_workdir_existing_home_matrix": {
            "plan_modes": missing_plan_modes,
            "needs_workdir": True,
            "has_cwd": False,
            "uses_home_flag": True,
        },
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_existing_home_matrix": {
            "plan_modes": missing_plan_modes,
            "needs_workdir": False,
            "has_cwd": False,
            "uses_home_flag": True,
        },
        invalid_plan_default_with_workdir: {
            "plan_modes": invalid_plan_modes,
            "needs_workdir": True,
            "has_cwd": True,
            "uses_home_flag": False,
        },
        invalid_plan_default_only: {
            "plan_modes": invalid_plan_modes,
            "needs_workdir": False,
            "has_cwd": True,
            "uses_home_flag": False,
        },
        missing_plan_default_with_workdir: {
            "plan_modes": missing_plan_modes,
            "needs_workdir": True,
            "has_cwd": True,
            "uses_home_flag": False,
        },
        missing_plan_default_only: {
            "plan_modes": missing_plan_modes,
            "needs_workdir": False,
            "has_cwd": True,
            "uses_home_flag": False,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        flag_orders_assign = next(
            (
                stmt
                for stmt in node.body
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
        assert toggle_orders == expected_toggle_orders

        plan_modes_assign = next(
            (
                stmt
                for stmt in node.body
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
        actual_plan_modes: set[str] = set()
        for mode_node in plan_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            actual_plan_modes.add(mode_node.value)
        expected = expectations[node.name]
        assert actual_plan_modes == expected["plan_modes"]

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        if expected["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_existing_home_plan_error_groups_keep_home_and_cwd_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_with_workdir = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix"
    )
    explicit_without_workdir = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix"
    )
    explicit_missing_with_workdir = (
        "test_cli_run_dry_run_both_toggles_missing_plan_precedes_workdir_existing_home_matrix"
    )
    explicit_missing_without_workdir = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_existing_home_matrix"
    )

    default_with_workdir = (
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_"
        "default_existing_home_matrix"
    )
    default_without_workdir = (
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix"
    )
    default_missing_with_workdir = (
        "test_cli_run_dry_run_both_toggles_missing_plan_default_existing_home_"
        "precedes_workdir_matrix"
    )
    default_missing_without_workdir = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_default_existing_home_matrix"
    )

    expectations = {
        explicit_with_workdir: {
            "home_var": "home",
            "has_cwd": False,
            "needs_workdir": True,
            "anchor": '"Plan validation error" in output',
            "present": [],
            "absent": ['"PLAN_PATH" not in output'],
        },
        explicit_without_workdir: {
            "home_var": "home",
            "has_cwd": False,
            "needs_workdir": False,
            "anchor": '"Plan validation error" in output',
            "present": [],
            "absent": ['"PLAN_PATH" not in output'],
        },
        explicit_missing_with_workdir: {
            "home_var": "home",
            "has_cwd": False,
            "needs_workdir": True,
            "anchor": '"PLAN_PATH" in output',
            "present": ["\"Invalid value for 'PLAN_PATH'\" in output"],
            "absent": ['"Plan validation error" not in output'],
        },
        explicit_missing_without_workdir: {
            "home_var": "home",
            "has_cwd": False,
            "needs_workdir": False,
            "anchor": '"PLAN_PATH" in output',
            "present": ["\"Invalid value for 'PLAN_PATH'\" in output"],
            "absent": ['"Plan validation error" not in output'],
        },
        default_with_workdir: {
            "home_var": "default_home",
            "has_cwd": True,
            "needs_workdir": True,
            "anchor": '"Plan validation error" in output',
            "present": [],
            "absent": ['"PLAN_PATH" not in output'],
        },
        default_without_workdir: {
            "home_var": "default_home",
            "has_cwd": True,
            "needs_workdir": False,
            "anchor": '"Plan validation error" in output',
            "present": [],
            "absent": ['"PLAN_PATH" not in output'],
        },
        default_missing_with_workdir: {
            "home_var": "default_home",
            "has_cwd": True,
            "needs_workdir": True,
            "anchor": '"PLAN_PATH" in output',
            "present": [
                "\"Invalid value for 'PLAN_PATH'\" in output",
                '"contains symlink component" not in output',
            ],
            "absent": ['"Plan validation error" not in output'],
        },
        default_missing_without_workdir: {
            "home_var": "default_home",
            "has_cwd": True,
            "needs_workdir": False,
            "anchor": '"PLAN_PATH" in output',
            "present": [
                "\"Invalid value for 'PLAN_PATH'\" in output",
                '"contains symlink component" not in output',
            ],
            "absent": ['"Plan validation error" not in output'],
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert expected["anchor"] in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert (
            f"assert sorted(path.name for path in {home_var}.iterdir()) == [], context"
            in source_segment
        )
        assert '"Dry Run" not in output' in source_segment
        assert '"run_id:" not in output' in source_segment
        assert '"state:" not in output' in source_segment
        assert '"report:" not in output' in source_segment
        for snippet in expected["present"]:
            assert snippet in source_segment
        for snippet in expected["absent"]:
            assert snippet in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
            assert '"--home"' not in source_segment
        else:
            assert "cwd=case_root" not in source_segment
            assert '"--home"' in source_segment

        if expected["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


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


def test_cli_integration_missing_plan_workdir_existing_home_matrix_keeps_modes_and_toggles() -> (
    None
):
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_missing_plan_precedes_workdir_existing_home_matrix"
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


def test_cli_integration_missing_plan_workdir_existing_home_matrix_asserts_output_contract() -> (
    None
):
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_missing_plan_precedes_workdir_existing_home_matrix"
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
    assert "assert home.exists(), context" in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment
    assert "assert sorted(path.name for path in home.iterdir()) == [], context" in source_segment
    assert "cwd=case_root" not in source_segment
    assert '"--home"' in source_segment
    assert '"--workdir"' in source_segment


def test_cli_integration_missing_plan_existing_home_matrix_keeps_modes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_existing_home_matrix"

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


def test_cli_integration_missing_plan_existing_home_matrix_asserts_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_existing_home_matrix"

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
    assert "assert home.exists(), context" in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment
    assert "assert sorted(path.name for path in home.iterdir()) == [], context" in source_segment
    assert "cwd=case_root" not in source_segment
    assert '"--home"' in source_segment
    assert '"--workdir"' not in source_segment


def test_cli_integration_existing_home_preserve_entries_matrix_keeps_axes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_plan_error_matrix"
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

    case_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "case_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert case_modes_assign is not None
    assert isinstance(case_modes_assign.value, ast.Tuple)

    case_names: set[str] = set()
    plan_kinds: set[str] = set()
    workdir_modes: set[bool] = set()
    for case_node in case_modes_assign.value.elts:
        assert isinstance(case_node, ast.Tuple)
        assert len(case_node.elts) == 3

        name_node = case_node.elts[0]
        kind_node = case_node.elts[1]
        workdir_node = case_node.elts[2]

        assert isinstance(name_node, ast.Constant)
        assert isinstance(name_node.value, str)
        case_names.add(name_node.value)

        assert isinstance(kind_node, ast.Constant)
        assert isinstance(kind_node.value, str)
        plan_kinds.add(kind_node.value)

        assert isinstance(workdir_node, ast.Constant)
        assert isinstance(workdir_node.value, bool)
        workdir_modes.add(workdir_node.value)

    assert case_names == {
        "invalid_only",
        "invalid_with_workdir",
        "missing_only",
        "missing_with_workdir",
    }
    assert plan_kinds == {"invalid_plan", "missing_plan"}
    assert workdir_modes == {False, True}


def test_cli_integration_existing_home_preserve_entries_matrix_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_plan_error_matrix"
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
    assert '"Plan validation error" in output' in source_segment
    assert '"PLAN_PATH" in output' in source_segment
    assert "\"Invalid value for 'PLAN_PATH'\" in output" in source_segment
    assert '"contains symlink component" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert home.exists(), context" in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment
    assert (
        'assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir"], context'
        in source_segment
    )
    assert (
        'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in source_segment
    )
    assert "assert sentinel_dir.is_dir(), context" in source_segment
    assert (
        'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context' in source_segment
    )
    assert '"--home"' in source_segment
    assert '"--workdir"' in source_segment
    assert "cwd=case_root" not in source_segment


def test_cli_integration_default_existing_home_preserve_entries_matrix_keeps_axes_and_toggles() -> (
    None
):
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_default_existing_home_"
        "preserves_entries_plan_error_matrix"
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

    case_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "case_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert case_modes_assign is not None
    assert isinstance(case_modes_assign.value, ast.Tuple)

    case_names: set[str] = set()
    plan_kinds: set[str] = set()
    workdir_modes: set[bool] = set()
    for case_node in case_modes_assign.value.elts:
        assert isinstance(case_node, ast.Tuple)
        assert len(case_node.elts) == 3

        name_node = case_node.elts[0]
        kind_node = case_node.elts[1]
        workdir_node = case_node.elts[2]

        assert isinstance(name_node, ast.Constant)
        assert isinstance(name_node.value, str)
        case_names.add(name_node.value)

        assert isinstance(kind_node, ast.Constant)
        assert isinstance(kind_node.value, str)
        plan_kinds.add(kind_node.value)

        assert isinstance(workdir_node, ast.Constant)
        assert isinstance(workdir_node.value, bool)
        workdir_modes.add(workdir_node.value)

    assert case_names == {
        "invalid_only",
        "invalid_with_workdir",
        "missing_only",
        "missing_with_workdir",
    }
    assert plan_kinds == {"invalid_plan", "missing_plan"}
    assert workdir_modes == {False, True}


def test_cli_integration_default_existing_home_preserve_entries_matrix_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_default_existing_home_"
        "preserves_entries_plan_error_matrix"
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
    assert '"Plan validation error" in output' in source_segment
    assert '"PLAN_PATH" in output' in source_segment
    assert "\"Invalid value for 'PLAN_PATH'\" in output" in source_segment
    assert '"contains symlink component" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert "default_home.iterdir()" in source_segment
    assert '"keep.txt"' in source_segment
    assert '"keep_dir"' in source_segment
    assert (
        'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in source_segment
    )
    assert "assert sentinel_dir.is_dir(), context" in source_segment
    assert (
        'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context' in source_segment
    )
    assert '"--home"' not in source_segment
    assert '"--workdir"' in source_segment
    assert "cwd=case_root" in source_segment


def test_cli_integration_existing_home_preserve_entries_invalid_workdir_keeps_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_invalid_workdir"
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

    workdir_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "workdir_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert workdir_modes_assign is not None
    assert isinstance(workdir_modes_assign.value, ast.Tuple)
    workdir_modes: set[str] = set()
    for mode_node in workdir_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        workdir_modes.add(mode_node.value)
    assert workdir_modes == {
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }


def test_cli_integration_existing_home_preserve_entries_invalid_workdir_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_invalid_workdir"
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
    assert "assert proc.returncode == 0, context" in source_segment
    assert '"Dry Run" in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"Plan validation error" not in output' in source_segment
    assert '"PLAN_PATH" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert home.exists(), context" in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment
    assert (
        'assert sorted(path.name for path in home.iterdir()) == ["keep.txt", "keep_dir"], context'
        in source_segment
    )
    assert (
        'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in source_segment
    )
    assert "assert sentinel_dir.is_dir(), context" in source_segment
    assert (
        'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context' in source_segment
    )
    assert '"--home"' in source_segment
    assert '"--workdir"' in source_segment
    assert "cwd=case_root" not in source_segment


def test_cli_integration_default_existing_home_invalid_workdir_keeps_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_default_existing_home_preserves_entries_invalid_workdir"
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

    workdir_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "workdir_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert workdir_modes_assign is not None
    assert isinstance(workdir_modes_assign.value, ast.Tuple)
    workdir_modes: set[str] = set()
    for mode_node in workdir_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        workdir_modes.add(mode_node.value)
    assert workdir_modes == {
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }


def test_cli_integration_default_existing_home_invalid_workdir_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_dry_run_both_toggles_default_existing_home_preserves_entries_invalid_workdir"
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
    assert "assert proc.returncode == 0, context" in source_segment
    assert '"Dry Run" in output' in source_segment
    assert '"Invalid workdir" not in output' in source_segment
    assert '"Plan validation error" not in output' in source_segment
    assert '"PLAN_PATH" not in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert "default_home.iterdir()" in source_segment
    assert '"keep.txt"' in source_segment
    assert '"keep_dir"' in source_segment
    assert (
        'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in source_segment
    )
    assert "assert sentinel_dir.is_dir(), context" in source_segment
    assert (
        'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context' in source_segment
    )
    assert '"--home"' not in source_segment
    assert '"--workdir"' in source_segment
    assert "cwd=case_root" in source_segment


def test_cli_integration_preserve_entries_invalid_workdir_matrices_keep_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = (
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_invalid_workdir"
    )
    default_matrix = (
        "test_cli_run_dry_run_both_toggles_default_existing_home_preserves_entries_invalid_workdir"
    )
    explicit_with_runs_matrix = (
        "test_cli_run_dry_run_both_toggles_existing_home_with_runs_"
        "preserves_entries_invalid_workdir"
    )
    default_with_runs_matrix = (
        "test_cli_run_dry_run_toggles_default_home_with_runs_invalid_workdir_preserve"
    )
    explicit_artifacts_matrix = (
        "test_cli_run_dry_run_both_toggles_existing_home_run_artifacts_preserved_invalid_workdir"
    )
    default_artifacts_matrix = (
        "test_cli_run_dry_run_both_toggles_default_existing_home_run_artifacts_"
        "preserved_invalid_workdir"
    )
    expectations = {
        explicit_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_with_runs_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        default_with_runs_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        explicit_artifacts_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
        default_artifacts_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "assert proc.returncode == 0, context" in source_segment
        assert 'assert "Dry Run" in output, context' in source_segment
        assert 'assert "Invalid workdir" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "PLAN_PATH" not in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "run_id:" not in output, context' in source_segment
        assert 'assert "state:" not in output, context' in source_segment
        assert 'assert "report:" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert (
            'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context'
            in source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )
        if expected["has_existing_runs"]:
            assert '"runs"' in source_segment
            assert "existing_run" in source_segment
            assert f'({home_var} / "runs").iterdir()' in source_segment
            assert '"keep_run"' in source_segment
            assert '"plan.yaml"' in source_segment
            assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
            if expected["has_artifacts"]:
                assert '"cancel.request"' in source_segment
                assert '"task.log"' in source_segment
                assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
            else:
                assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
                assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                    source_segment
                )
        else:
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment
        assert 'file_path.read_text(encoding="utf-8") == "file\\n"' in source_segment
        assert 'if workdir_mode == "dangling_symlink":' in source_segment
        assert '"missing_workdir_target"' in source_segment
        assert 'if workdir_mode == "symlink_ancestor":' in source_segment
        assert '"child_workdir"' in source_segment
        assert "workdir_modes" in source_segment
        assert "for workdir_mode in workdir_modes:" in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment
        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_preserve_entries_invalid_workdir_matrices_keep_wiring() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    matrix_names = {
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_invalid_workdir",
        "test_cli_run_dry_run_both_toggles_default_existing_home_preserves_entries_invalid_workdir",
        "test_cli_run_dry_run_both_toggles_existing_home_with_runs_preserves_entries_invalid_workdir",
        "test_cli_run_dry_run_toggles_default_home_with_runs_invalid_workdir_preserve",
        "test_cli_run_dry_run_both_toggles_existing_home_run_artifacts_preserved_invalid_workdir",
        "test_cli_run_dry_run_both_toggles_default_existing_home_run_artifacts_preserved_invalid_workdir",
    }

    expected_toggle_orders = {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }
    expected_workdir_modes = {
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in matrix_names:
            continue

        flag_orders_assign = next(
            (
                stmt
                for stmt in node.body
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
        assert toggle_orders == expected_toggle_orders

        workdir_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "workdir_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert workdir_modes_assign is not None
        assert isinstance(workdir_modes_assign.value, ast.Tuple)
        workdir_modes: set[str] = set()
        for mode_node in workdir_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            workdir_modes.add(mode_node.value)
        assert workdir_modes == expected_workdir_modes

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert "for workdir_mode in workdir_modes:" in source_segment
        assert 'plan_path = case_root / "valid_plan.yaml"' in source_segment
        assert "plan_path.write_text(" in source_segment
        assert "cmd:" in source_segment
        assert 'if workdir_mode == "file_path":' in source_segment
        assert 'invalid_workdir_path = case_root / "invalid_workdir_file"' in source_segment
        assert 'elif workdir_mode == "file_ancestor":' in source_segment
        assert 'workdir_parent_file = case_root / "workdir_parent_file"' in source_segment
        assert 'invalid_workdir_path = workdir_parent_file / "child_workdir"' in source_segment
        assert 'elif workdir_mode == "symlink_to_file":' in source_segment
        assert 'workdir_target_file = case_root / "workdir_target_file"' in source_segment
        assert 'invalid_workdir_path = case_root / "workdir_symlink_to_file"' in source_segment
        assert 'elif workdir_mode == "dangling_symlink":' in source_segment
        assert 'invalid_workdir_path = case_root / "workdir_dangling_symlink"' in source_segment
        assert "invalid_workdir_path.symlink_to(" in source_segment
        assert 'workdir_parent_link = case_root / "workdir_parent_link"' in source_segment
        assert 'invalid_workdir_path = workdir_parent_link / "child_workdir"' in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment
        assert 'file_path.read_text(encoding="utf-8") == "file\\n"' in source_segment
        assert 'if workdir_mode == "dangling_symlink":' in source_segment
        assert '"missing_workdir_target"' in source_segment
        assert 'if workdir_mode == "symlink_ancestor":' in source_segment
        assert '"child_workdir"' in source_segment
        assert '"--dry-run"' in source_segment
        assert "*order" in source_segment
        assert 'nested_file = sentinel_dir / "nested.txt"' in source_segment
        assert 'nested_file.write_text("nested\\n", encoding="utf-8")' in source_segment
        if "with_runs" in node.name:
            assert "existing_run" in source_segment
            assert '"plan.yaml"' in source_segment
            assert 'assert plan_file.read_text(encoding="utf-8") == "tasks: []\\n", context' in (
                source_segment
            )
            assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
            assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                source_segment
            )
        if "run_artifacts" in node.name:
            assert "existing_run" in source_segment
            assert '"cancel.request"' in source_segment
            assert '"task.log"' in source_segment
            assert 'assert lock_file.read_text(encoding="utf-8") == "lock\\n", context' in (
                source_segment
            )
            assert (
                'assert cancel_request.read_text(encoding="utf-8") == "cancel\\n", context'
                in source_segment
            )
            assert 'assert run_log.read_text(encoding="utf-8") == "log\\n", context' in (
                source_segment
            )
        matched.add(node.name)

    assert matched == matrix_names


def test_cli_integration_run_invalid_workdir_modes_matrix_keeps_axes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = "test_cli_run_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"

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

    workdir_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "workdir_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert workdir_modes_assign is not None
    assert isinstance(workdir_modes_assign.value, ast.Tuple)
    workdir_modes: set[str] = set()
    for mode_node in workdir_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        workdir_modes.add(mode_node.value)
    assert workdir_modes == {
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }


def test_cli_integration_run_invalid_workdir_modes_matrix_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"

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
    assert '"Invalid workdir" in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Plan validation error" not in output' in source_segment
    assert '"contains symlink component" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert not home.exists(), context" in source_segment
    assert 'assert not (home / "runs").exists(), context' in source_segment
    assert "side_effect_files" in source_segment
    assert "for file_path in side_effect_files:" in source_segment
    assert (
        'assert file_path.read_text(encoding="utf-8") == "not a directory\\n", context'
        in source_segment
    )
    assert 'if workdir_mode == "dangling_symlink":' in source_segment
    assert '"missing_workdir_target"' in source_segment
    assert 'if workdir_mode == "symlink_ancestor":' in source_segment
    assert '"child_workdir"' in source_segment


def test_cli_integration_run_invalid_workdir_modes_matrix_wiring() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"

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
    assert "for workdir_mode in workdir_modes:" in source_segment
    assert 'if workdir_mode == "missing_path":' in source_segment
    assert 'elif workdir_mode == "file_path":' in source_segment
    assert 'elif workdir_mode == "file_ancestor":' in source_segment
    assert 'elif workdir_mode == "symlink_to_file":' in source_segment
    assert 'elif workdir_mode == "dangling_symlink":' in source_segment
    assert "else:" in source_segment
    assert 'invalid_workdir = case_root / "missing_workdir"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_file"' in source_segment
    assert 'workdir_parent_file = case_root / "workdir_parent_file"' in source_segment
    assert 'invalid_workdir = workdir_parent_file / "child_workdir"' in source_segment
    assert 'workdir_target_file = case_root / "workdir_target_file"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_symlink_to_file"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_dangling_symlink"' in source_segment
    assert 'workdir_parent_link = case_root / "workdir_parent_link"' in source_segment
    assert 'invalid_workdir = workdir_parent_link / "child_workdir"' in source_segment
    assert '"--workdir"' in source_segment
    assert '"--dry-run"' not in source_segment
    assert "*order" in source_segment


def test_cli_integration_run_default_home_invalid_workdir_modes_keeps_axes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = (
        "test_cli_run_default_home_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"
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

    workdir_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "workdir_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert workdir_modes_assign is not None
    assert isinstance(workdir_modes_assign.value, ast.Tuple)
    workdir_modes: set[str] = set()
    for mode_node in workdir_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        workdir_modes.add(mode_node.value)
    assert workdir_modes == {
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }


def test_cli_integration_run_default_home_invalid_workdir_modes_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_default_home_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"
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
    assert '"Invalid workdir" in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Plan validation error" not in output' in source_segment
    assert '"contains symlink component" not in output' in source_segment
    assert '"Dry Run" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert "assert not default_home.exists(), context" in source_segment
    assert 'assert not (default_home / "runs").exists(), context' in source_segment
    assert "side_effect_files" in source_segment
    assert "for file_path in side_effect_files:" in source_segment
    assert (
        'assert file_path.read_text(encoding="utf-8") == "not a directory\\n", context'
        in source_segment
    )
    assert 'if workdir_mode == "dangling_symlink":' in source_segment
    assert '"missing_workdir_target"' in source_segment
    assert 'if workdir_mode == "symlink_ancestor":' in source_segment
    assert '"child_workdir"' in source_segment
    assert "cwd=case_root" in source_segment
    assert '"--home"' not in source_segment


def test_cli_integration_run_default_home_invalid_workdir_modes_wiring() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = (
        "test_cli_run_default_home_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"
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
    assert "for workdir_mode in workdir_modes:" in source_segment
    assert 'if workdir_mode == "missing_path":' in source_segment
    assert 'elif workdir_mode == "file_path":' in source_segment
    assert 'elif workdir_mode == "file_ancestor":' in source_segment
    assert 'elif workdir_mode == "symlink_to_file":' in source_segment
    assert 'elif workdir_mode == "dangling_symlink":' in source_segment
    assert "else:" in source_segment
    assert 'invalid_workdir = case_root / "missing_workdir"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_file"' in source_segment
    assert 'workdir_parent_file = case_root / "workdir_parent_file"' in source_segment
    assert 'invalid_workdir = workdir_parent_file / "child_workdir"' in source_segment
    assert 'workdir_target_file = case_root / "workdir_target_file"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_symlink_to_file"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_dangling_symlink"' in source_segment
    assert 'workdir_parent_link = case_root / "workdir_parent_link"' in source_segment
    assert 'invalid_workdir = workdir_parent_link / "child_workdir"' in source_segment
    assert '"--workdir"' in source_segment
    assert '"--dry-run"' not in source_segment
    assert "*order" in source_segment
    assert "cwd=case_root" in source_segment
    assert '"--home"' not in source_segment


def test_cli_integration_run_invalid_workdir_mode_matrices_keep_home_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = "test_cli_run_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"
    default_matrix = (
        "test_cli_run_default_home_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"
    )
    expectations = {
        explicit_matrix: {"home_var": "home", "has_cwd": False, "uses_home_flag": True},
        default_matrix: {"home_var": "default_home", "has_cwd": True, "uses_home_flag": False},
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert f"assert not {home_var}.exists(), context" in source_segment
        assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert 'assert "Invalid workdir" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "run_id:" not in output, context' in source_segment
        assert 'assert "state:" not in output, context' in source_segment
        assert 'assert "report:" not in output, context' in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment
        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_run_invalid_workdir_with_runs_matrices_keep_axes_and_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_preserve_matrix = "test_cli_run_invalid_workdir_existing_home_preserves_entries_matrix"
    default_preserve_matrix = "test_cli_run_default_home_invalid_workdir_preserves_entries_matrix"
    explicit_matrix = (
        "test_cli_run_invalid_workdir_existing_home_with_runs_preserves_entries_matrix"
    )
    default_matrix = "test_cli_run_default_home_invalid_workdir_with_runs_preserves_entries_matrix"
    explicit_artifacts_matrix = (
        "test_cli_run_invalid_workdir_existing_home_run_artifacts_preserved_matrix"
    )
    default_artifacts_matrix = (
        "test_cli_run_default_home_invalid_workdir_run_artifacts_preserved_matrix"
    )
    expectations = {
        explicit_preserve_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_preserve_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        default_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        explicit_artifacts_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
        default_artifacts_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
    }
    expected_toggle_orders = {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }
    expected_workdir_modes = {
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        flag_orders_assign = next(
            (
                stmt
                for stmt in node.body
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
        assert toggle_orders == expected_toggle_orders

        workdir_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "workdir_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert workdir_modes_assign is not None
        assert isinstance(workdir_modes_assign.value, ast.Tuple)
        workdir_modes: set[str] = set()
        for mode_node in workdir_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            workdir_modes.add(mode_node.value)
        assert workdir_modes == expected_workdir_modes

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert f"assert {home_var}.exists(), context" in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        if expected["has_existing_runs"]:
            assert '"runs"' in source_segment
            assert "existing_run" in source_segment
            assert f'({home_var} / "runs").iterdir()' in source_segment
            assert '"keep_run"' in source_segment
            assert "sorted(path.name for path in existing_run.iterdir())" in source_segment
            assert '"plan.yaml"' in source_segment
            assert 'assert plan_file.read_text(encoding="utf-8") == "tasks: []\\n", context' in (
                source_segment
            )
            if expected["has_artifacts"]:
                assert '"cancel.request"' in source_segment
                assert '"task.log"' in source_segment
                assert 'assert lock_file.read_text(encoding="utf-8") == "lock\\n", context' in (
                    source_segment
                )
                assert (
                    'assert cancel_request.read_text(encoding="utf-8") == "cancel\\n", context'
                    in source_segment
                )
                assert 'assert run_log.read_text(encoding="utf-8") == "log\\n", context' in (
                    source_segment
                )
            else:
                assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
                assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                    source_segment
                )
        else:
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert 'assert "Invalid workdir" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "Dry Run" not in output, context' in source_segment
        assert 'assert "run_id:" not in output, context' in source_segment
        assert 'assert "state:" not in output, context' in source_segment
        assert 'assert "report:" not in output, context' in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment
        assert "*order" in source_segment
        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_run_invalid_workdir_preserve_supergroup_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_non_preserve = (
        "test_cli_run_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"
    )
    default_non_preserve = (
        "test_cli_run_default_home_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"
    )
    explicit_preserve = "test_cli_run_invalid_workdir_existing_home_preserves_entries_matrix"
    default_preserve = "test_cli_run_default_home_invalid_workdir_preserves_entries_matrix"
    explicit_with_runs = (
        "test_cli_run_invalid_workdir_existing_home_with_runs_preserves_entries_matrix"
    )
    default_with_runs = (
        "test_cli_run_default_home_invalid_workdir_with_runs_preserves_entries_matrix"
    )
    explicit_artifacts = "test_cli_run_invalid_workdir_existing_home_run_artifacts_preserved_matrix"
    default_artifacts = "test_cli_run_default_home_invalid_workdir_run_artifacts_preserved_matrix"

    expectations = {
        explicit_non_preserve: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_non_preserve: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_preserve: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_preserve: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_with_runs: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        default_with_runs: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        explicit_artifacts: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
        default_artifacts: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "flag_orders" in source_segment
        assert "workdir_modes" in source_segment
        assert "for workdir_mode in workdir_modes:" in source_segment
        assert 'assert "Invalid workdir" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "Dry Run" not in output, context' in source_segment
        assert 'assert "run_id:" not in output, context' in source_segment
        assert 'assert "state:" not in output, context' in source_segment
        assert 'assert "report:" not in output, context' in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment
        assert '"--workdir"' in source_segment
        assert "*order" in source_segment

        if expected["preserves_entries"]:
            assert f"assert {home_var}.exists(), context" in source_segment
            assert f"{home_var}.iterdir()" in source_segment
            assert '"keep.txt"' in source_segment
            assert '"keep_dir"' in source_segment
            assert (
                'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context'
                in source_segment
            )
            assert "assert sentinel_dir.is_dir(), context" in source_segment
            assert (
                'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
                in source_segment
            )
            if expected["has_existing_runs"]:
                assert '"runs"' in source_segment
                assert "existing_run" in source_segment
                assert f'({home_var} / "runs").iterdir()' in source_segment
                assert '"keep_run"' in source_segment
                assert '"plan.yaml"' in source_segment
                assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
                if expected["has_artifacts"]:
                    assert '"cancel.request"' in source_segment
                    assert '"task.log"' in source_segment
                    assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
                    assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
                    assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
                else:
                    assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
                    assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                        source_segment
                    )
            else:
                assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        else:
            assert f"assert not {home_var}.exists(), context" in source_segment
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_run_invalid_workdir_preserve_dry_run_non_dry_parity() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expectations = {
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_invalid_workdir": {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": False,
            "has_artifacts": False,
            "is_dry_run": True,
        },
        (
            "test_cli_run_dry_run_both_toggles_default_existing_home_"
            "preserves_entries_invalid_workdir"
        ): {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": False,
            "has_artifacts": False,
            "is_dry_run": True,
        },
        (
            "test_cli_run_dry_run_both_toggles_existing_home_with_runs_"
            "preserves_entries_invalid_workdir"
        ): {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": False,
            "is_dry_run": True,
        },
        "test_cli_run_dry_run_toggles_default_home_with_runs_invalid_workdir_preserve": {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": False,
            "is_dry_run": True,
        },
        "test_cli_run_dry_run_both_toggles_existing_home_run_artifacts_preserved_invalid_workdir": {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": True,
            "is_dry_run": True,
        },
        (
            "test_cli_run_dry_run_both_toggles_default_existing_home_run_artifacts_"
            "preserved_invalid_workdir"
        ): {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": True,
            "is_dry_run": True,
        },
        "test_cli_run_invalid_workdir_existing_home_preserves_entries_matrix": {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": False,
            "has_artifacts": False,
            "is_dry_run": False,
        },
        "test_cli_run_default_home_invalid_workdir_preserves_entries_matrix": {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": False,
            "has_artifacts": False,
            "is_dry_run": False,
        },
        "test_cli_run_invalid_workdir_existing_home_with_runs_preserves_entries_matrix": {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": False,
            "is_dry_run": False,
        },
        "test_cli_run_default_home_invalid_workdir_with_runs_preserves_entries_matrix": {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": False,
            "is_dry_run": False,
        },
        "test_cli_run_invalid_workdir_existing_home_run_artifacts_preserved_matrix": {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": True,
            "is_dry_run": False,
        },
        "test_cli_run_default_home_invalid_workdir_run_artifacts_preserved_matrix": {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": True,
            "is_dry_run": False,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "flag_orders" in source_segment
        assert "workdir_modes" in source_segment
        assert "for workdir_mode in workdir_modes:" in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment
        if expected["is_dry_run"]:
            assert 'file_path.read_text(encoding="utf-8") == "file\\n"' in source_segment
        else:
            assert 'file_path.read_text(encoding="utf-8") == "not a directory\\n"' in (
                source_segment
            )
        assert 'if workdir_mode == "dangling_symlink":' in source_segment
        assert '"missing_workdir_target"' in source_segment
        assert 'if workdir_mode == "symlink_ancestor":' in source_segment
        assert '"child_workdir"' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "run_id:" not in output, context' in source_segment
        assert 'assert "state:" not in output, context' in source_segment
        assert 'assert "report:" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert (
            'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context'
            in source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )

        if expected["is_dry_run"]:
            assert "assert proc.returncode == 0, context" in source_segment
            assert 'assert "Dry Run" in output, context' in source_segment
            assert 'assert "Invalid workdir" not in output, context' in source_segment
            assert 'assert "PLAN_PATH" not in output, context' in source_segment
        else:
            assert "assert proc.returncode == 2, context" in source_segment
            assert 'assert "Dry Run" not in output, context' in source_segment
            assert 'assert "Invalid workdir" in output, context' in source_segment

        if expected["has_existing_runs"]:
            assert '"runs"' in source_segment
            assert "existing_run" in source_segment
            assert f'({home_var} / "runs").iterdir()' in source_segment
            assert '"keep_run"' in source_segment
            assert '"plan.yaml"' in source_segment
            assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
            if expected["has_artifacts"]:
                assert '"cancel.request"' in source_segment
                assert '"task.log"' in source_segment
                assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
            else:
                assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
                assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                    source_segment
                )
        else:
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_run_symlink_ancestor_invalid_workdir_detail_matrix_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_run_symlink_ancestor_invalid_workdir_suppresses_component_detail_matrix"

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

    assert 'modes = ("plain", "with_runs", "artifacts")' in source_segment
    assert "for dry_run in (False, True):" in source_segment
    assert "for use_default_home in (False, True):" in source_segment
    assert "for mode in modes:" in source_segment
    assert 'home = case_root / (".orch" if use_default_home else ".orch_cli")' in source_segment
    assert '"workdir_parent_link"' in source_segment
    assert '"child_workdir"' in source_segment
    assert 'assert "contains symlink component" not in output, context' in source_segment
    assert 'assert "run_id:" not in output, context' in source_segment
    assert 'assert "state:" not in output, context' in source_segment
    assert 'assert "report:" not in output, context' in source_segment
    assert 'assert not (real_workdir_parent / "child_workdir").exists(), context' in (
        source_segment
    )
    assert "if dry_run:" in source_segment
    assert "assert proc.returncode == 0, context" in source_segment
    assert "assert proc.returncode == 2, context" in source_segment
    assert 'assert "Dry Run" in output, context' in source_segment
    assert 'assert "Dry Run" not in output, context' in source_segment
    assert 'assert "Invalid workdir" in output, context' in source_segment
    assert 'assert "Invalid workdir" not in output, context' in source_segment
    assert 'if mode == "plain":' in source_segment
    assert 'elif mode == "with_runs":' in source_segment
    assert "else:" in source_segment
    assert 'assert mode == "artifacts"' in source_segment
    assert "assert not lock_file.exists(), context" in source_segment
    assert "assert not cancel_request.exists(), context" in source_segment
    assert 'assert lock_file.read_text(encoding="utf-8") == "lock\\n", context' in (source_segment)
    assert 'assert cancel_request.read_text(encoding="utf-8") == "cancel\\n", context' in (
        source_segment
    )
    assert 'assert run_log.read_text(encoding="utf-8") == "log\\n", context' in source_segment
    assert "cwd=case_root if use_default_home else None" in source_segment
    assert "if not use_default_home:" in source_segment
    assert '"--home"' in source_segment
    assert '"--workdir"' in source_segment
    assert 'cmd.append("--dry-run")' in source_segment


def test_cli_integration_resume_invalid_workdir_modes_matrix_keeps_axes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = "test_cli_resume_rejects_invalid_workdir_modes_matrix"

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

    workdir_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "workdir_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert workdir_modes_assign is not None
    assert isinstance(workdir_modes_assign.value, ast.Tuple)
    workdir_modes: set[str] = set()
    for mode_node in workdir_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        workdir_modes.add(mode_node.value)
    assert workdir_modes == {
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }


def test_cli_integration_resume_invalid_workdir_modes_matrix_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_resume_rejects_invalid_workdir_modes_matrix"

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
    assert '"Invalid workdir" in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Run not found or broken" not in output' in source_segment
    assert '"Plan validation error" not in output' in source_segment
    assert '"contains symlink component" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert 'state_path.read_text(encoding="utf-8") == baseline_state' in source_segment
    assert 'sorted(path.name for path in (home / "runs").iterdir()) == [run_id]' in source_segment
    assert "side_effect_files" in source_segment
    assert "for file_path in side_effect_files:" in source_segment
    assert (
        'assert file_path.read_text(encoding="utf-8") == "not a directory\\n", context'
        in source_segment
    )
    assert 'if workdir_mode == "dangling_symlink":' in source_segment
    assert '"missing_workdir_target"' in source_segment
    assert 'if workdir_mode == "symlink_ancestor":' in source_segment
    assert '"child_workdir"' in source_segment


def test_cli_integration_resume_invalid_workdir_modes_matrix_wiring() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_resume_rejects_invalid_workdir_modes_matrix"

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
    assert "for workdir_mode in workdir_modes:" in source_segment
    assert 'if workdir_mode == "missing_path":' in source_segment
    assert 'elif workdir_mode == "file_path":' in source_segment
    assert 'elif workdir_mode == "file_ancestor":' in source_segment
    assert 'elif workdir_mode == "symlink_to_file":' in source_segment
    assert 'elif workdir_mode == "dangling_symlink":' in source_segment
    assert "else:" in source_segment
    assert 'invalid_workdir = case_root / "missing_workdir"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_file"' in source_segment
    assert 'workdir_parent_file = case_root / "workdir_parent_file"' in source_segment
    assert 'invalid_workdir = workdir_parent_file / "child_workdir"' in source_segment
    assert 'workdir_target_file = case_root / "workdir_target_file"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_symlink_to_file"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_dangling_symlink"' in source_segment
    assert 'workdir_parent_link = case_root / "workdir_parent_link"' in source_segment
    assert 'invalid_workdir = workdir_parent_link / "child_workdir"' in source_segment
    assert "run_id = _extract_run_id(run_proc.stdout)" in source_segment
    assert '"resume"' in source_segment
    assert '"--workdir"' in source_segment
    assert "*order" in source_segment
    assert '"--dry-run"' not in source_segment


def test_cli_integration_resume_default_home_invalid_workdir_modes_keeps_axes_and_toggles() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )
    matrix_name = "test_cli_resume_default_home_rejects_invalid_workdir_modes_matrix"

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

    workdir_modes_assign = next(
        (
            stmt
            for stmt in matrix_function.body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "workdir_modes"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Tuple)
        ),
        None,
    )
    assert workdir_modes_assign is not None
    assert isinstance(workdir_modes_assign.value, ast.Tuple)
    workdir_modes: set[str] = set()
    for mode_node in workdir_modes_assign.value.elts:
        assert isinstance(mode_node, ast.Constant)
        assert isinstance(mode_node.value, str)
        workdir_modes.add(mode_node.value)
    assert workdir_modes == {
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }


def test_cli_integration_resume_default_home_invalid_workdir_modes_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_resume_default_home_rejects_invalid_workdir_modes_matrix"

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
    assert '"Invalid workdir" in output' in source_segment
    assert '"Invalid home" not in output' in source_segment
    assert '"Run not found or broken" not in output' in source_segment
    assert '"Plan validation error" not in output' in source_segment
    assert '"contains symlink component" not in output' in source_segment
    assert '"run_id:" not in output' in source_segment
    assert '"state:" not in output' in source_segment
    assert '"report:" not in output' in source_segment
    assert 'state_path.read_text(encoding="utf-8") == baseline_state' in source_segment
    assert 'sorted(path.name for path in (default_home / "runs").iterdir()) == [run_id]' in (
        source_segment
    )
    assert "side_effect_files" in source_segment
    assert "for file_path in side_effect_files:" in source_segment
    assert (
        'assert file_path.read_text(encoding="utf-8") == "not a directory\\n", context'
        in source_segment
    )
    assert 'if workdir_mode == "dangling_symlink":' in source_segment
    assert '"missing_workdir_target"' in source_segment
    assert 'if workdir_mode == "symlink_ancestor":' in source_segment
    assert '"child_workdir"' in source_segment
    assert "cwd=case_root" in source_segment
    assert '"--home"' not in source_segment


def test_cli_integration_resume_default_home_invalid_workdir_modes_wiring() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    matrix_name = "test_cli_resume_default_home_rejects_invalid_workdir_modes_matrix"

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
    assert "for workdir_mode in workdir_modes:" in source_segment
    assert 'if workdir_mode == "missing_path":' in source_segment
    assert 'elif workdir_mode == "file_path":' in source_segment
    assert 'elif workdir_mode == "file_ancestor":' in source_segment
    assert 'elif workdir_mode == "symlink_to_file":' in source_segment
    assert 'elif workdir_mode == "dangling_symlink":' in source_segment
    assert "else:" in source_segment
    assert 'invalid_workdir = case_root / "missing_workdir"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_file"' in source_segment
    assert 'workdir_parent_file = case_root / "workdir_parent_file"' in source_segment
    assert 'invalid_workdir = workdir_parent_file / "child_workdir"' in source_segment
    assert 'workdir_target_file = case_root / "workdir_target_file"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_symlink_to_file"' in source_segment
    assert 'invalid_workdir = case_root / "workdir_dangling_symlink"' in source_segment
    assert 'workdir_parent_link = case_root / "workdir_parent_link"' in source_segment
    assert 'invalid_workdir = workdir_parent_link / "child_workdir"' in source_segment
    assert "run_id = _extract_run_id(run_proc.stdout)" in source_segment
    assert '"resume"' in source_segment
    assert '"--workdir"' in source_segment
    assert "*order" in source_segment
    assert '"--dry-run"' not in source_segment
    assert "cwd=case_root" in source_segment
    assert '"--home"' not in source_segment


def test_cli_integration_resume_invalid_workdir_mode_matrices_keep_home_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = "test_cli_resume_rejects_invalid_workdir_modes_matrix"
    default_matrix = "test_cli_resume_default_home_rejects_invalid_workdir_modes_matrix"
    expectations = {
        explicit_matrix: {"home_var": "home", "has_cwd": False, "uses_home_flag": True},
        default_matrix: {"home_var": "default_home", "has_cwd": True, "uses_home_flag": False},
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert 'assert "Invalid workdir" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "run_id:" not in output, context' in source_segment
        assert 'assert "state:" not in output, context' in source_segment
        assert 'assert "report:" not in output, context' in source_segment
        assert 'state_path.read_text(encoding="utf-8") == baseline_state' in source_segment
        assert f'sorted(path.name for path in ({home_var} / "runs").iterdir()) == [run_id]' in (
            source_segment
        )
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment
        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_resume_invalid_run_id_workdir_mode_matrices_keep_axes_and_boundaries() -> (
    None
):
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = "test_cli_resume_invalid_run_id_precedes_invalid_workdir_modes_matrix"
    default_matrix = (
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_modes_matrix"
    )
    expectations = {
        explicit_matrix: {"home_var": "home", "has_cwd": False, "uses_home_flag": True},
        default_matrix: {"home_var": "default_home", "has_cwd": True, "uses_home_flag": False},
    }
    expected_toggle_orders = {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }
    expected_run_id_modes = {"path_escape", "too_long"}
    expected_workdir_modes = {
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        flag_orders_assign = next(
            (
                stmt
                for stmt in node.body
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
        assert toggle_orders == expected_toggle_orders

        run_id_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "run_id_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert run_id_modes_assign is not None
        assert isinstance(run_id_modes_assign.value, ast.Tuple)
        run_id_modes: set[str] = set()
        for mode_node in run_id_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            run_id_modes.add(mode_node.value)
        assert run_id_modes == expected_run_id_modes

        workdir_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "workdir_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert workdir_modes_assign is not None
        assert isinstance(workdir_modes_assign.value, ast.Tuple)
        workdir_modes: set[str] = set()
        for mode_node in workdir_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            workdir_modes.add(mode_node.value)
        assert workdir_modes == expected_workdir_modes

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert f"assert not {home_var}.exists(), context" in source_segment
        assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert "for workdir_mode in workdir_modes:" in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment
        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_resume_invalid_run_id_workdir_mode_matrices_output_contract() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    matrix_names = {
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_modes_matrix",
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_modes_matrix",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in matrix_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid workdir" not in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "run_id: [bold]" not in output, context' in source_segment
        assert 'assert "state:" not in output, context' in source_segment
        assert 'assert "report:" not in output, context' in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment
        assert (
            'assert file_path.read_text(encoding="utf-8") == "not a directory\\n", context'
            in source_segment
        )
        assert 'if workdir_mode == "dangling_symlink":' in source_segment
        assert '"missing_workdir_target"' in source_segment
        assert 'if workdir_mode == "symlink_ancestor":' in source_segment
        assert '"child_workdir"' in source_segment
        matched.add(node.name)

    assert matched == matrix_names


def test_cli_integration_resume_invalid_run_id_workdir_mode_matrices_wiring() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    matrix_names = {
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_modes_matrix",
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_modes_matrix",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in matrix_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert 'bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129' in (
            source_segment
        )
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert "for workdir_mode in workdir_modes:" in source_segment
        assert 'if workdir_mode == "missing_path":' in source_segment
        assert 'elif workdir_mode == "file_path":' in source_segment
        assert 'elif workdir_mode == "file_ancestor":' in source_segment
        assert 'elif workdir_mode == "symlink_to_file":' in source_segment
        assert 'elif workdir_mode == "dangling_symlink":' in source_segment
        assert "else:" in source_segment
        assert 'invalid_workdir = case_root / "missing_workdir"' in source_segment
        assert 'invalid_workdir = case_root / "workdir_file"' in source_segment
        assert 'workdir_parent_file = case_root / "workdir_parent_file"' in source_segment
        assert 'invalid_workdir = workdir_parent_file / "child_workdir"' in source_segment
        assert 'workdir_target_file = case_root / "workdir_target_file"' in source_segment
        assert 'invalid_workdir = case_root / "workdir_symlink_to_file"' in source_segment
        assert 'invalid_workdir = case_root / "workdir_dangling_symlink"' in source_segment
        assert 'workdir_parent_link = case_root / "workdir_parent_link"' in source_segment
        assert 'invalid_workdir = workdir_parent_link / "child_workdir"' in source_segment
        assert '"resume"' in source_segment
        assert '"--workdir"' in source_segment
        assert "*order" in source_segment
        assert '"--dry-run"' not in source_segment
        matched.add(node.name)

    assert matched == matrix_names


def test_cli_integration_resume_invalid_run_id_preserve_entries_matrices_keep_axes() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = (
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_"
        "preserves_entries_matrix"
    )
    default_matrix = (
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_"
        "preserve_entries_matrix"
    )
    explicit_with_runs_matrix = (
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_with_runs_matrix"
    )
    default_with_runs_matrix = (
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_with_runs_matrix"
    )
    explicit_artifacts_matrix = (
        "test_cli_resume_invalid_run_id_invalid_workdir_existing_home_run_artifacts_matrix"
    )
    default_artifacts_matrix = (
        "test_cli_resume_default_home_invalid_run_id_invalid_workdir_run_artifacts_matrix"
    )
    expectations = {
        explicit_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_with_runs_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        default_with_runs_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        explicit_artifacts_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
        default_artifacts_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
    }
    expected_toggle_orders = {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }
    expected_run_id_modes = {"path_escape", "too_long"}
    expected_workdir_modes = {
        "missing_path",
        "file_path",
        "file_ancestor",
        "symlink_to_file",
        "dangling_symlink",
        "symlink_ancestor",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        flag_orders_assign = next(
            (
                stmt
                for stmt in node.body
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
        assert toggle_orders == expected_toggle_orders

        run_id_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "run_id_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert run_id_modes_assign is not None
        assert isinstance(run_id_modes_assign.value, ast.Tuple)
        run_id_modes: set[str] = set()
        for mode_node in run_id_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            run_id_modes.add(mode_node.value)
        assert run_id_modes == expected_run_id_modes

        workdir_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "workdir_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert workdir_modes_assign is not None
        assert isinstance(workdir_modes_assign.value, ast.Tuple)
        workdir_modes: set[str] = set()
        for mode_node in workdir_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            workdir_modes.add(mode_node.value)
        assert workdir_modes == expected_workdir_modes

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert f"assert {home_var}.exists(), context" in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert 'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in (
            source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )
        if expected["has_existing_runs"]:
            assert '"runs"' in source_segment
            assert "existing_run" in source_segment
            assert f'({home_var} / "runs").iterdir()' in source_segment
            assert '"keep_run"' in source_segment
            if expected["has_artifacts"]:
                assert '"cancel.request"' in source_segment
                assert '"task.log"' in source_segment
                assert '"plan.yaml"' in source_segment
                assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
            else:
                assert (
                    'assert sorted(path.name for path in existing_run.iterdir()) == ["plan.yaml"]'
                    in (source_segment)
                )
                assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
                assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                    source_segment
                )
        else:
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert "for workdir_mode in workdir_modes:" in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment
        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_resume_invalid_run_id_preserve_entries_matrices_output_wiring() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    matrix_names = {
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_preserves_entries_matrix",
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_preserve_entries_matrix",
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_with_runs_matrix",
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_with_runs_matrix",
        "test_cli_resume_invalid_run_id_invalid_workdir_existing_home_run_artifacts_matrix",
        "test_cli_resume_default_home_invalid_run_id_invalid_workdir_run_artifacts_matrix",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in matrix_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid workdir" not in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "run_id: [bold]" not in output, context' in source_segment
        if "run_artifacts" in node.name:
            assert 'assert "state: [bold]" not in output, context' in source_segment
            assert 'assert "report: [bold]" not in output, context' in source_segment
            assert "existing_run" in source_segment
            assert '"cancel.request"' in source_segment
            assert '"plan.yaml"' in source_segment
            assert '"task.log"' in source_segment
            assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
            assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
            assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
            assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
        elif "with_runs" in node.name:
            assert 'assert "state:" not in output, context' in source_segment
            assert 'assert "report:" not in output, context' in source_segment
            assert "existing_run" in source_segment
            assert (
                'assert sorted(path.name for path in existing_run.iterdir()) == ["plan.yaml"]'
                in (source_segment)
            )
            assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
            assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                source_segment
            )
        else:
            assert 'assert "state:" not in output, context' in source_segment
            assert 'assert "report:" not in output, context' in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment
        assert (
            'assert file_path.read_text(encoding="utf-8") == "not a directory\\n", context'
            in source_segment
        )
        assert 'if workdir_mode == "dangling_symlink":' in source_segment
        assert '"missing_workdir_target"' in source_segment
        assert 'if workdir_mode == "symlink_ancestor":' in source_segment
        assert '"child_workdir"' in source_segment
        assert 'bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129' in (
            source_segment
        )
        assert 'if workdir_mode == "missing_path":' in source_segment
        assert 'elif workdir_mode == "file_path":' in source_segment
        assert 'elif workdir_mode == "file_ancestor":' in source_segment
        assert 'elif workdir_mode == "symlink_to_file":' in source_segment
        assert 'elif workdir_mode == "dangling_symlink":' in source_segment
        assert "else:" in source_segment
        assert '"resume"' in source_segment
        assert '"--workdir"' in source_segment
        assert "*order" in source_segment
        assert '"--dry-run"' not in source_segment
        matched.add(node.name)

    assert matched == matrix_names


def test_cli_integration_status_logs_resume_invalid_run_id_preserve_axes() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )

    matrix_names = {
        "test_cli_status_logs_resume_invalid_run_id_existing_home_preserve_entries_matrix",
        "test_cli_status_logs_resume_invalid_run_id_default_home_preserve_entries_matrix",
    }
    expected_commands = {"status", "logs", "resume"}
    expected_run_id_modes = {"path_escape", "too_long"}

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in matrix_names:
            continue

        commands_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "commands"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert commands_assign is not None
        assert isinstance(commands_assign.value, ast.Tuple)
        commands: set[str] = set()
        for command_node in commands_assign.value.elts:
            assert isinstance(command_node, ast.Constant)
            assert isinstance(command_node.value, str)
            commands.add(command_node.value)
        assert commands == expected_commands

        run_id_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "run_id_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert run_id_modes_assign is not None
        assert isinstance(run_id_modes_assign.value, ast.Tuple)
        run_id_modes: set[str] = set()
        for mode_node in run_id_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            run_id_modes.add(mode_node.value)
        assert run_id_modes == expected_run_id_modes

        source_segment = ast.get_source_segment(
            (tests_root / "test_cli_integration.py").read_text(encoding="utf-8"),
            node,
        )
        assert source_segment is not None
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert "for command in commands:" in source_segment
        matched.add(node.name)

    assert matched == matrix_names


def test_cli_integration_status_logs_resume_invalid_run_id_preserve_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_preserve_entries_matrix"
    )
    default_matrix = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_preserve_entries_matrix"
    )
    expectations = {
        explicit_matrix: {"home_var": "home", "has_cwd": False, "uses_home_flag": True},
        default_matrix: {"home_var": "default_home", "has_cwd": True, "uses_home_flag": False},
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "run_id: [bold]" not in output, context' in source_segment
        assert 'assert "state: [bold]" not in output, context' in source_segment
        assert 'assert "report: [bold]" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert 'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in (
            source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_status_logs_resume_invalid_run_id_with_runs_preserve_axes() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )

    matrix_names = {
        "test_cli_status_logs_resume_invalid_run_id_existing_home_with_runs_preserve_entries_matrix",
        "test_cli_status_logs_resume_invalid_run_id_default_home_with_runs_preserve_entries_matrix",
        "test_cli_status_logs_resume_invalid_run_id_existing_home_run_artifacts_preserved_matrix",
        "test_cli_status_logs_resume_invalid_run_id_default_home_run_artifacts_preserved_matrix",
    }
    expected_commands = {"status", "logs", "resume"}
    expected_run_id_modes = {"path_escape", "too_long"}

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in matrix_names:
            continue

        commands_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "commands"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert commands_assign is not None
        assert isinstance(commands_assign.value, ast.Tuple)
        commands: set[str] = set()
        for command_node in commands_assign.value.elts:
            assert isinstance(command_node, ast.Constant)
            assert isinstance(command_node.value, str)
            commands.add(command_node.value)
        assert commands == expected_commands

        run_id_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "run_id_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert run_id_modes_assign is not None
        assert isinstance(run_id_modes_assign.value, ast.Tuple)
        run_id_modes: set[str] = set()
        for mode_node in run_id_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            run_id_modes.add(mode_node.value)
        assert run_id_modes == expected_run_id_modes

        source_segment = ast.get_source_segment(
            (tests_root / "test_cli_integration.py").read_text(encoding="utf-8"),
            node,
        )
        assert source_segment is not None
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert "for command in commands:" in source_segment
        assert "existing_run = " in source_segment
        assert ".mkdir(parents=True)" in source_segment
        matched.add(node.name)

    assert matched == matrix_names


def test_cli_integration_status_logs_resume_invalid_run_id_with_runs_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_with_runs_preserve_entries_matrix"
    )
    default_matrix = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_with_runs_preserve_entries_matrix"
    )
    explicit_artifacts_matrix = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_run_artifacts_preserved_matrix"
    )
    default_artifacts_matrix = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_run_artifacts_preserved_matrix"
    )
    expectations = {
        explicit_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_extra_artifacts": False,
        },
        default_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_extra_artifacts": False,
        },
        explicit_artifacts_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_extra_artifacts": True,
        },
        default_artifacts_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_extra_artifacts": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert '"runs"' in source_segment
        assert f'({home_var} / "runs").iterdir()' in source_segment
        assert '"keep_run"' in source_segment
        if expected["has_extra_artifacts"]:
            assert "assert sorted(path.name for path in existing_run.iterdir())" in source_segment
            assert '"cancel.request"' in source_segment
            assert '"plan.yaml"' in source_segment
            assert '"task.log"' in source_segment
            assert 'assert plan_file.read_text(encoding="utf-8") == "tasks: []\\n", context' in (
                source_segment
            )
            assert 'assert lock_file.read_text(encoding="utf-8") == "lock\\n", context' in (
                source_segment
            )
            assert (
                'assert cancel_request.read_text(encoding="utf-8") == "cancel\\n", context'
                in source_segment
            )
            assert 'assert run_log.read_text(encoding="utf-8") == "log\\n", context' in (
                source_segment
            )
        else:
            assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
        assert 'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in (
            source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_status_logs_resume_invalid_run_id_preserve_groups() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_preserve_entries_matrix"
    )
    default_matrix = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_preserve_entries_matrix"
    )
    explicit_with_runs = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_with_runs_preserve_entries_matrix"
    )
    default_with_runs = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_with_runs_preserve_entries_matrix"
    )
    explicit_with_artifacts = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_run_artifacts_preserved_matrix"
    )
    default_with_artifacts = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_run_artifacts_preserved_matrix"
    )

    expectations = {
        explicit_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_with_runs: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        default_with_runs: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        explicit_with_artifacts: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
        default_with_artifacts: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "commands = (" in source_segment
        assert "run_id_modes" in source_segment
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert "for command in commands:" in source_segment
        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment

        if expected["has_existing_runs"]:
            assert '"runs"' in source_segment
            assert "existing_run" in source_segment
            assert ".lock" in source_segment
            assert f'({home_var} / "runs").iterdir()' in source_segment
            assert '"keep_run"' in source_segment
            if expected["has_artifacts"]:
                assert '"cancel.request"' in source_segment
                assert '"task.log"' in source_segment
                assert '"plan.yaml"' in source_segment
                assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
            else:
                assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
        else:
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_cancel_invalid_run_id_preserve_matrices_keep_axes() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )

    matrix_names = {
        "test_cli_cancel_invalid_run_id_existing_home_preserves_entries_matrix",
        "test_cli_cancel_invalid_run_id_default_home_preserves_entries_matrix",
    }
    expected_run_id_modes = {"path_escape", "too_long"}

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in matrix_names:
            continue

        run_id_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "run_id_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert run_id_modes_assign is not None
        assert isinstance(run_id_modes_assign.value, ast.Tuple)
        run_id_modes: set[str] = set()
        for mode_node in run_id_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            run_id_modes.add(mode_node.value)
        assert run_id_modes == expected_run_id_modes

        source_segment = ast.get_source_segment(
            (tests_root / "test_cli_integration.py").read_text(encoding="utf-8"),
            node,
        )
        assert source_segment is not None
        assert "for run_id_mode in run_id_modes:" in source_segment
        matched.add(node.name)

    assert matched == matrix_names


def test_cli_integration_cancel_invalid_run_id_preserve_matrices_output_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = "test_cli_cancel_invalid_run_id_existing_home_preserves_entries_matrix"
    default_matrix = "test_cli_cancel_invalid_run_id_default_home_preserves_entries_matrix"
    expectations = {
        explicit_matrix: {"home_var": "home", "has_cwd": False, "uses_home_flag": True},
        default_matrix: {"home_var": "default_home", "has_cwd": True, "uses_home_flag": False},
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "Cancel request written" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert 'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in (
            source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_invalid_run_id_preserve_matrix_groups_keep_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    status_explicit = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_preserve_entries_matrix"
    )
    status_default = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_preserve_entries_matrix"
    )
    cancel_explicit = "test_cli_cancel_invalid_run_id_existing_home_preserves_entries_matrix"
    cancel_default = "test_cli_cancel_invalid_run_id_default_home_preserves_entries_matrix"

    expectations = {
        status_explicit: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_commands_axis": True,
            "checks_cancel_message": False,
        },
        status_default: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_commands_axis": True,
            "checks_cancel_message": False,
        },
        cancel_explicit: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_commands_axis": False,
            "checks_cancel_message": True,
        },
        cancel_default: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_commands_axis": False,
            "checks_cancel_message": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "run_id_modes" in source_segment
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert 'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in (
            source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )

        if expected["has_commands_axis"]:
            assert "commands = (" in source_segment
            assert "for command in commands:" in source_segment
            assert '"status", "logs", "resume"' in source_segment
        else:
            assert "commands = (" not in source_segment
            assert "for command in commands:" not in source_segment

        if expected["checks_cancel_message"]:
            assert 'assert "Cancel request written" not in output, context' in source_segment
            assert '"cancel"' in source_segment
        else:
            assert 'assert "Cancel request written" not in output, context' not in source_segment
            assert '"cancel"' not in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_cancel_invalid_run_id_with_runs_preserve_matrices_keep_axes() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_module = ast.parse(
        (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    )

    matrix_names = {
        "test_cli_cancel_invalid_run_id_existing_home_with_runs_preserves_entries_matrix",
        "test_cli_cancel_invalid_run_id_default_home_with_runs_preserves_entries_matrix",
        "test_cli_cancel_invalid_run_id_existing_home_run_artifacts_preserved_matrix",
        "test_cli_cancel_invalid_run_id_default_home_run_artifacts_preserved_matrix",
    }
    expected_run_id_modes = {"path_escape", "too_long"}

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in matrix_names:
            continue

        run_id_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "run_id_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert run_id_modes_assign is not None
        assert isinstance(run_id_modes_assign.value, ast.Tuple)
        run_id_modes: set[str] = set()
        for mode_node in run_id_modes_assign.value.elts:
            assert isinstance(mode_node, ast.Constant)
            assert isinstance(mode_node.value, str)
            run_id_modes.add(mode_node.value)
        assert run_id_modes == expected_run_id_modes

        source_segment = ast.get_source_segment(
            (tests_root / "test_cli_integration.py").read_text(encoding="utf-8"),
            node,
        )
        assert source_segment is not None
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert "existing_run = " in source_segment
        assert ".mkdir(parents=True)" in source_segment
        matched.add(node.name)

    assert matched == matrix_names


def test_cli_integration_cancel_invalid_run_id_with_runs_preserve_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = (
        "test_cli_cancel_invalid_run_id_existing_home_with_runs_preserves_entries_matrix"
    )
    default_matrix = (
        "test_cli_cancel_invalid_run_id_default_home_with_runs_preserves_entries_matrix"
    )
    explicit_artifacts_matrix = (
        "test_cli_cancel_invalid_run_id_existing_home_run_artifacts_preserved_matrix"
    )
    default_artifacts_matrix = (
        "test_cli_cancel_invalid_run_id_default_home_run_artifacts_preserved_matrix"
    )
    expectations = {
        explicit_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_extra_artifacts": False,
        },
        default_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_extra_artifacts": False,
        },
        explicit_artifacts_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_extra_artifacts": True,
        },
        default_artifacts_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_extra_artifacts": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "Cancel request written" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert f'({home_var} / "runs").iterdir()' in source_segment
        assert '"keep_run"' in source_segment
        if expected["has_extra_artifacts"]:
            assert "assert sorted(path.name for path in existing_run.iterdir())" in source_segment
            assert '"cancel.request"' in source_segment
            assert '"plan.yaml"' in source_segment
            assert '"task.log"' in source_segment
            assert 'assert plan_file.read_text(encoding="utf-8") == "tasks: []\\n", context' in (
                source_segment
            )
            assert 'assert lock_file.read_text(encoding="utf-8") == "lock\\n", context' in (
                source_segment
            )
            assert (
                'assert cancel_request.read_text(encoding="utf-8") == "cancel\\n", context'
                in source_segment
            )
            assert 'assert run_log.read_text(encoding="utf-8") == "log\\n", context' in (
                source_segment
            )
        else:
            assert (
                'assert not (existing_run / "cancel.request").exists(), context' in source_segment
            )
            assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert '"runs"' in source_segment
        assert 'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in (
            source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_cancel_invalid_run_id_preserve_matrix_groups_keep_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = "test_cli_cancel_invalid_run_id_existing_home_preserves_entries_matrix"
    default_matrix = "test_cli_cancel_invalid_run_id_default_home_preserves_entries_matrix"
    explicit_with_runs = (
        "test_cli_cancel_invalid_run_id_existing_home_with_runs_preserves_entries_matrix"
    )
    default_with_runs = (
        "test_cli_cancel_invalid_run_id_default_home_with_runs_preserves_entries_matrix"
    )
    explicit_with_artifacts = (
        "test_cli_cancel_invalid_run_id_existing_home_run_artifacts_preserved_matrix"
    )
    default_with_artifacts = (
        "test_cli_cancel_invalid_run_id_default_home_run_artifacts_preserved_matrix"
    )

    expectations = {
        explicit_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_with_runs: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        default_with_runs: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        explicit_with_artifacts: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
        default_with_artifacts: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "run_id_modes" in source_segment
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "Cancel request written" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )

        if expected["has_existing_runs"]:
            assert '"runs"' in source_segment
            assert "existing_run" in source_segment
            assert "cancel.request" in source_segment
            assert ".lock" in source_segment
            if expected["has_artifacts"]:
                assert '"task.log"' in source_segment
                assert '"plan.yaml"' in source_segment
                assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
            else:
                assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                    source_segment
                )
                assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
        else:
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_invalid_run_id_preserve_matrix_supergroup_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    status_explicit = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_preserve_entries_matrix"
    )
    status_default = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_preserve_entries_matrix"
    )
    status_explicit_with_runs = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_with_runs_preserve_entries_matrix"
    )
    status_default_with_runs = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_with_runs_preserve_entries_matrix"
    )
    status_explicit_run_artifacts = (
        "test_cli_status_logs_resume_invalid_run_id_existing_home_run_artifacts_preserved_matrix"
    )
    status_default_run_artifacts = (
        "test_cli_status_logs_resume_invalid_run_id_default_home_run_artifacts_preserved_matrix"
    )
    cancel_explicit = "test_cli_cancel_invalid_run_id_existing_home_preserves_entries_matrix"
    cancel_default = "test_cli_cancel_invalid_run_id_default_home_preserves_entries_matrix"
    cancel_explicit_with_runs = (
        "test_cli_cancel_invalid_run_id_existing_home_with_runs_preserves_entries_matrix"
    )
    cancel_default_with_runs = (
        "test_cli_cancel_invalid_run_id_default_home_with_runs_preserves_entries_matrix"
    )
    cancel_explicit_run_artifacts = (
        "test_cli_cancel_invalid_run_id_existing_home_run_artifacts_preserved_matrix"
    )
    cancel_default_run_artifacts = (
        "test_cli_cancel_invalid_run_id_default_home_run_artifacts_preserved_matrix"
    )

    expectations = {
        status_explicit: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_commands_axis": True,
            "checks_cancel_message": False,
            "has_existing_runs": False,
        },
        status_default: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_commands_axis": True,
            "checks_cancel_message": False,
            "has_existing_runs": False,
        },
        status_explicit_with_runs: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_commands_axis": True,
            "checks_cancel_message": False,
            "has_existing_runs": True,
            "has_extra_run_artifacts": False,
        },
        status_default_with_runs: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_commands_axis": True,
            "checks_cancel_message": False,
            "has_existing_runs": True,
            "has_extra_run_artifacts": False,
        },
        status_explicit_run_artifacts: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_commands_axis": True,
            "checks_cancel_message": False,
            "has_existing_runs": True,
            "has_extra_run_artifacts": True,
        },
        status_default_run_artifacts: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_commands_axis": True,
            "checks_cancel_message": False,
            "has_existing_runs": True,
            "has_extra_run_artifacts": True,
        },
        cancel_explicit: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_commands_axis": False,
            "checks_cancel_message": True,
            "has_existing_runs": False,
            "has_extra_run_artifacts": False,
        },
        cancel_default: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_commands_axis": False,
            "checks_cancel_message": True,
            "has_existing_runs": False,
            "has_extra_run_artifacts": False,
        },
        cancel_explicit_with_runs: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_commands_axis": False,
            "checks_cancel_message": True,
            "has_existing_runs": True,
            "has_extra_run_artifacts": False,
        },
        cancel_default_with_runs: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_commands_axis": False,
            "checks_cancel_message": True,
            "has_existing_runs": True,
            "has_extra_run_artifacts": False,
        },
        cancel_explicit_run_artifacts: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "has_commands_axis": False,
            "checks_cancel_message": True,
            "has_existing_runs": True,
            "has_extra_run_artifacts": True,
        },
        cancel_default_run_artifacts: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "has_commands_axis": False,
            "checks_cancel_message": True,
            "has_existing_runs": True,
            "has_extra_run_artifacts": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "run_id_modes" in source_segment
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert f"assert {home_var}.exists(), context" in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert 'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in (
            source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )

        if expected["has_commands_axis"]:
            assert "commands = (" in source_segment
            assert "for command in commands:" in source_segment
            assert '"status", "logs", "resume"' in source_segment
        else:
            assert "commands = (" not in source_segment
            assert "for command in commands:" not in source_segment

        if expected["checks_cancel_message"]:
            assert 'assert "Cancel request written" not in output, context' in source_segment
            assert '"cancel"' in source_segment
        else:
            assert 'assert "Cancel request written" not in output, context' not in source_segment
            assert '"cancel"' not in source_segment

        if expected["has_existing_runs"]:
            assert '"runs"' in source_segment
            assert "existing_run" in source_segment
            assert "keep_run" in source_segment
            assert ".lock" in source_segment
            if expected["has_extra_run_artifacts"]:
                assert '"cancel.request"' in source_segment
                assert '"task.log"' in source_segment
                assert '"plan.yaml"' in source_segment
                assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
                assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
        else:
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_run_id_precedence_over_invalid_home_shape_families() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    status_entries = (
        (
            "test_cli_status_logs_resume_run_id_precedes_invalid_home",
            "path_escape",
            "file_path",
        ),
        (
            "test_cli_status_logs_resume_too_long_run_id_precedes_invalid_home",
            "too_long",
            "file_path",
        ),
        (
            "test_cli_status_logs_resume_invalid_run_id_precedes_symlink_home",
            "path_escape",
            "symlink",
        ),
        (
            "test_cli_status_logs_resume_too_long_run_id_precedes_symlink_home",
            "too_long",
            "symlink",
        ),
        (
            "test_cli_status_logs_resume_invalid_run_id_precedes_dangling_symlink_home",
            "path_escape",
            "dangling_symlink",
        ),
        (
            "test_cli_status_logs_resume_too_long_run_id_precedes_dangling_symlink_home",
            "too_long",
            "dangling_symlink",
        ),
        (
            "test_cli_status_logs_resume_invalid_run_id_precedes_home_symlink_to_file",
            "path_escape",
            "symlink_to_file",
        ),
        (
            "test_cli_status_logs_resume_too_long_run_id_precedes_home_symlink_to_file",
            "too_long",
            "symlink_to_file",
        ),
        (
            "test_cli_status_logs_resume_invalid_run_id_precedes_home_file_ancestor",
            "path_escape",
            "file_ancestor",
        ),
        (
            "test_cli_status_logs_resume_too_long_run_id_precedes_home_file_ancestor",
            "too_long",
            "file_ancestor",
        ),
        (
            "test_cli_status_logs_resume_invalid_run_id_precedes_home_symlink_ancestor",
            "path_escape",
            "symlink_ancestor",
        ),
        (
            "test_cli_status_logs_resume_too_long_run_id_precedes_home_symlink_ancestor",
            "too_long",
            "symlink_ancestor",
        ),
        (
            "test_cli_status_logs_resume_invalid_run_id_precedes_home_symlink_ancestor_directory",
            "path_escape",
            "symlink_ancestor_directory",
        ),
        (
            "test_cli_status_logs_resume_too_long_run_id_precedes_home_symlink_ancestor_directory",
            "too_long",
            "symlink_ancestor_directory",
        ),
        (
            "test_cli_status_logs_resume_invalid_run_id_precedence_invalid_home_shape_matrix",
            "matrix",
            "matrix",
        ),
    )
    cancel_entries = (
        (
            "test_cli_cancel_invalid_run_id_takes_precedence_over_invalid_home",
            "path_escape",
            "file_path",
        ),
        (
            "test_cli_cancel_too_long_run_id_takes_precedence_over_invalid_home",
            "too_long",
            "file_path",
        ),
        (
            "test_cli_cancel_invalid_run_id_takes_precedence_over_symlink_home",
            "path_escape",
            "symlink",
        ),
        (
            "test_cli_cancel_too_long_run_id_takes_precedence_over_symlink_home",
            "too_long",
            "symlink",
        ),
        (
            "test_cli_cancel_invalid_run_id_takes_precedence_over_dangling_symlink_home",
            "path_escape",
            "dangling_symlink",
        ),
        (
            "test_cli_cancel_too_long_run_id_takes_precedence_over_dangling_symlink_home",
            "too_long",
            "dangling_symlink",
        ),
        (
            "test_cli_cancel_invalid_run_id_takes_precedence_over_home_symlink_to_file",
            "path_escape",
            "symlink_to_file",
        ),
        (
            "test_cli_cancel_too_long_run_id_takes_precedence_over_home_symlink_to_file",
            "too_long",
            "symlink_to_file",
        ),
        (
            "test_cli_cancel_invalid_run_id_takes_precedence_over_home_file_ancestor",
            "path_escape",
            "file_ancestor",
        ),
        (
            "test_cli_cancel_too_long_run_id_takes_precedence_over_home_file_ancestor",
            "too_long",
            "file_ancestor",
        ),
        (
            "test_cli_cancel_invalid_run_id_takes_precedence_over_home_symlink_ancestor",
            "path_escape",
            "symlink_ancestor",
        ),
        (
            "test_cli_cancel_too_long_run_id_takes_precedence_over_home_symlink_ancestor",
            "too_long",
            "symlink_ancestor",
        ),
        (
            "test_cli_cancel_invalid_run_id_takes_precedence_over_home_symlink_ancestor_directory",
            "path_escape",
            "symlink_ancestor_directory",
        ),
        (
            "test_cli_cancel_too_long_run_id_takes_precedence_over_home_symlink_ancestor_directory",
            "too_long",
            "symlink_ancestor_directory",
        ),
        (
            "test_cli_cancel_invalid_run_id_precedence_invalid_home_shape_matrix",
            "matrix",
            "matrix",
        ),
    )

    expectations: dict[str, dict[str, str]] = {}
    for name, run_id_mode, home_shape in status_entries:
        expectations[name] = {
            "command_family": (
                "status_logs_resume_matrix" if run_id_mode == "matrix" else "status_logs_resume"
            ),
            "run_id_mode": run_id_mode,
            "home_shape": home_shape,
        }
    for name, run_id_mode, home_shape in cancel_entries:
        expectations[name] = {
            "command_family": "cancel_matrix" if run_id_mode == "matrix" else "cancel",
            "run_id_mode": run_id_mode,
            "home_shape": home_shape,
        }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]

        assert "subprocess.run(" in source_segment
        assert "assert proc.returncode == 2" in source_segment
        assert 'assert "Invalid run_id" in output' in source_segment
        assert 'assert "Invalid home" not in output' in source_segment
        assert 'assert "contains symlink component" not in output' in source_segment

        if expected["command_family"] == "status_logs_resume":
            assert 'for command in ("status", "logs", "resume"):' in source_segment
        elif expected["command_family"] == "status_logs_resume_matrix":
            assert 'commands = ("status", "logs", "resume")' in source_segment
            assert "for command in commands:" in source_segment
        else:
            assert 'for command in ("status", "logs", "resume"):' not in source_segment
            if expected["command_family"] == "cancel_matrix":
                assert 'cmd = [sys.executable, "-m", "orch.cli", "cancel", bad_run_id]' in (
                    source_segment
                )
            else:
                assert '"cancel"' in source_segment

        if expected["run_id_mode"] == "path_escape":
            assert 'bad_run_id = "../escape"' in source_segment
        elif expected["run_id_mode"] == "too_long":
            assert 'bad_run_id = "a" * 129' in source_segment
        else:
            assert expected["run_id_mode"] == "matrix"
            assert 'run_id_modes = ("path_escape", "too_long")' in source_segment
            assert "for run_id_mode in run_id_modes:" in source_segment

        if expected["home_shape"] == "file_path":
            assert 'home_file.read_text(encoding="utf-8") == "not a dir\\n"' in source_segment
        elif expected["home_shape"] == "symlink":
            assert 'real_home = tmp_path / "real_home"' in source_segment
            assert "home_link.symlink_to(real_home, target_is_directory=True)" in source_segment
            if expected["command_family"] == "cancel":
                assert 'assert not (real_run_dir / "cancel.request").exists()' in source_segment
        elif expected["home_shape"] == "dangling_symlink":
            assert 'missing_home_target = tmp_path / "missing_home_target"' in source_segment
            assert "home_link.symlink_to(missing_home_target, target_is_directory=True)" in (
                source_segment
            )
            assert "assert not missing_home_target.exists()" in source_segment
        elif expected["home_shape"] == "symlink_to_file":
            assert 'home_target_file = tmp_path / "home_target_file.txt"' in source_segment
            assert "home_link.symlink_to(home_target_file)" in source_segment
            assert (
                'home_target_file.read_text(encoding="utf-8") == "not a home dir\\n"'
                in source_segment
            )
        elif expected["home_shape"] == "file_ancestor":
            assert 'home_parent_file = tmp_path / "home_parent_file"' in source_segment
            assert 'nested_home = home_parent_file / "orch_home"' in source_segment
            assert 'home_parent_file.read_text(encoding="utf-8") == "not a dir\\n"' in (
                source_segment
            )
        elif expected["home_shape"] == "symlink_ancestor":
            assert 'symlink_parent = tmp_path / "home_parent_link"' in source_segment
            assert 'nested_home = symlink_parent / "orch_home"' in source_segment
            assert 'assert "contains symlink component" not in output' in source_segment
            assert 'assert not (real_parent / "orch_home" / "runs").exists()' in source_segment
        else:
            if expected["home_shape"] == "symlink_ancestor_directory":
                assert 'nested_home_name = "orch_home"' in source_segment
                assert 'real_run_dir = real_parent / nested_home_name / "runs"' in source_segment
                assert 'assert "contains symlink component" not in output' in source_segment
                if expected["command_family"] == "cancel":
                    assert 'assert not (real_run_dir / "cancel.request").exists()' in source_segment
                    assert 'assert not (real_run_dir / ".lock").exists()' in source_segment
                else:
                    assert 'assert not (real_run_dir / ".lock").exists()' in source_segment
            else:
                assert expected["home_shape"] == "matrix"
                assert "home_modes = (" in source_segment
                assert "for home_mode in home_modes:" in source_segment
                assert '"file_path"' in source_segment
                assert '"symlink"' in source_segment
                assert '"dangling_symlink"' in source_segment
                assert '"symlink_to_file"' in source_segment
                assert '"file_ancestor"' in source_segment
                assert '"symlink_ancestor"' in source_segment
                assert '"symlink_ancestor_directory"' in source_segment
                assert 'assert "contains symlink component" not in output, context' in (
                    source_segment
                )
                assert 'assert "run_id: [bold]" not in output, context' in source_segment
                assert 'assert "state: [bold]" not in output, context' in source_segment
                assert 'assert "report: [bold]" not in output, context' in source_segment
                if expected["command_family"] == "cancel_matrix":
                    assert (
                        'assert not (real_run_dir / "cancel.request").exists(), context'
                        in source_segment
                    )
                    assert 'assert not (real_run_dir / ".lock").exists(), context' in source_segment
                else:
                    assert 'assert not (real_run_dir / ".lock").exists(), context' in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_run_invalid_home_families_suppress_symlink_detail() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expected_names = {
        "test_cli_run_dry_run_no_fail_fast_still_rejects_invalid_home",
        "test_cli_run_dry_run_both_fail_fast_toggles_still_rejects_invalid_home",
        "test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_rejects_invalid_home",
        "test_cli_run_rejects_file_home_path",
        "test_cli_run_rejects_home_with_file_ancestor",
        "test_cli_run_invalid_home_precedes_invalid_workdir",
        "test_cli_run_home_file_ancestor_precedes_invalid_workdir",
        "test_cli_run_home_symlink_to_file_precedes_invalid_workdir",
        "test_cli_run_home_symlink_directory_precedes_invalid_workdir",
        "test_cli_run_home_symlink_ancestor_precedes_invalid_workdir",
        "test_cli_run_home_dangling_symlink_precedes_invalid_workdir",
        "test_cli_run_invalid_home_precedes_invalid_plan",
        "test_cli_dry_run_invalid_home_precedes_invalid_plan",
        "test_cli_run_home_file_ancestor_precedes_invalid_plan",
        "test_cli_run_home_symlink_to_file_precedes_invalid_plan",
        "test_cli_run_home_symlink_directory_precedes_invalid_plan",
        "test_cli_dry_run_home_symlink_directory_precedes_invalid_plan",
        "test_cli_run_home_symlink_ancestor_precedes_invalid_plan",
        "test_cli_dry_run_home_symlink_ancestor_precedes_invalid_plan",
        "test_cli_dry_run_home_symlink_to_file_precedes_invalid_plan",
        "test_cli_run_home_dangling_symlink_precedes_invalid_plan",
        "test_cli_dry_run_home_dangling_symlink_precedes_invalid_plan",
        "test_cli_dry_run_invalid_home_precedes_invalid_workdir",
        "test_cli_dry_run_home_dangling_symlink_precedes_invalid_workdir",
        "test_cli_dry_run_home_symlink_precedes_invalid_workdir",
        "test_cli_dry_run_home_file_ancestor_precedes_invalid_workdir",
        "test_cli_dry_run_home_symlink_to_file_precedes_invalid_workdir",
        "test_cli_dry_run_home_symlink_ancestor_precedes_invalid_workdir",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert 'assert "Invalid home" in output' in source_segment
        assert 'assert "contains symlink component" not in output' in source_segment

        if "invalid_workdir" in node.name:
            assert 'assert "Invalid workdir" not in output' in source_segment

        if "invalid_plan" in node.name:
            assert 'assert "Plan validation error" not in output' in source_segment

        if "dry_run" in node.name:
            assert '"--dry-run"' in source_segment

        matched.add(node.name)

    assert matched == expected_names


def test_cli_integration_other_invalid_home_families_suppress_symlink_detail() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expected_names = {
        "test_cli_run_rejects_home_with_symlink_ancestor_without_side_effect",
        "test_cli_run_rejects_home_symlink_to_file_without_side_effect",
        "test_cli_run_rejects_home_with_symlink_ancestor_component_without_side_effect",
        "test_cli_resume_invalid_home_precedes_invalid_workdir",
        "test_cli_resume_home_file_ancestor_precedes_invalid_workdir",
        "test_cli_resume_home_symlink_to_file_precedes_invalid_workdir",
        "test_cli_resume_home_symlink_directory_precedes_invalid_workdir",
        "test_cli_resume_home_symlink_ancestor_precedes_invalid_workdir",
        "test_cli_resume_home_dangling_symlink_precedes_invalid_workdir",
        "test_cli_cancel_rejects_home_with_symlink_ancestor_without_side_effect",
        "test_cli_cancel_rejects_home_with_file_ancestor_without_side_effect",
        "test_cli_cancel_rejects_home_symlink_to_file_without_side_effect",
        "test_cli_run_rejects_dangling_symlink_home_without_side_effect",
        "test_cli_status_logs_resume_reject_home_symlink_to_file_path",
        "test_cli_status_logs_resume_reject_home_with_symlink_ancestor_without_lock_side_effect",
        "test_cli_status_rejects_file_home_path",
        "test_cli_status_rejects_home_with_file_ancestor",
        "test_cli_logs_resume_reject_home_file_path",
        "test_cli_logs_resume_reject_home_with_file_ancestor",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert 'assert "Invalid home" in output' in source_segment
        assert 'assert "contains symlink component" not in output' in source_segment

        if "invalid_workdir" in node.name:
            assert 'assert "Invalid workdir" not in output' in source_segment

        if "symlink_to_file" in node.name:
            assert "home_target_file" in source_segment

        if "file_ancestor" in node.name:
            assert "home_parent_file" in source_segment

        matched.add(node.name)

    assert matched == expected_names


def test_cli_integration_invalid_home_symlink_detail_suppression_extended() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    invalid_home_absent_names = {
        "test_cli_run_dry_run_both_fail_fast_toggles_invalid_plan_precedes_invalid_workdir",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_invalid_workdir_matrix",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_home_matrix",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_home_matrix",
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix",
        "test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_invalid_plan_precedes_invalid_workdir",
    }
    invalid_home_present_names = {
        "test_cli_cancel_rejects_run_with_symlink_ancestor_home",
        "test_cli_cancel_rejects_home_file_without_side_effect",
        "test_cli_status_rejects_run_dir_with_symlink_ancestor_without_lock_side_effect",
        "test_cli_logs_rejects_symlink_home_path",
        "test_cli_resume_rejects_symlink_home_path",
    }

    expected_names = invalid_home_absent_names | invalid_home_present_names

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert 'assert "contains symlink component" not in output' in source_segment

        if node.name in invalid_home_absent_names:
            assert 'assert "Invalid home" not in output' in source_segment
            assert '"Plan validation error" in output' in source_segment
        else:
            assert 'assert "Invalid home" in output' in source_segment

        matched.add(node.name)

    assert matched == expected_names


def test_cli_integration_plan_validation_errors_suppress_symlink_detail() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expected_names = {
        "test_cli_run_dry_run_fail_fast_still_rejects_invalid_plan",
        "test_cli_run_dry_run_no_fail_fast_still_rejects_invalid_plan",
        "test_cli_run_dry_run_both_fail_fast_toggles_still_rejects_invalid_plan",
        "test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_rejects_invalid_plan",
        "test_cli_run_dry_run_both_toggles_rejects_symlink_plan_path",
        "test_cli_run_dry_run_both_toggles_reverse_rejects_symlink_plan_path",
        "test_cli_run_dry_run_both_toggles_rejects_plan_path_with_symlink_ancestor",
        "test_cli_run_dry_run_both_toggles_reverse_rejects_plan_path_with_symlink_ancestor",
        "test_cli_run_dry_run_both_toggles_symlinked_plan_precedes_invalid_workdir",
        "test_cli_run_dry_run_both_fail_fast_toggles_invalid_plan_precedes_invalid_workdir",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_invalid_workdir_matrix",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_home_matrix",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_home_matrix",
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_plan_error_matrix",
        "test_cli_run_dry_run_both_toggles_default_existing_home_preserves_entries_plan_error_matrix",
        "test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_invalid_plan_precedes_invalid_workdir",
        "test_cli_run_invalid_plan_precedes_invalid_workdir",
        "test_cli_run_invalid_plan_returns_two_and_creates_no_run",
        "test_cli_dry_run_invalid_plan_returns_two",
        "test_cli_dry_run_rejects_plan_with_unknown_root_field",
        "test_cli_dry_run_rejects_plan_with_unknown_task_field",
        "test_cli_dry_run_rejects_symlink_plan_path",
        "test_cli_dry_run_rejects_plan_path_with_symlink_ancestor",
        "test_cli_dry_run_rejects_non_regular_plan_path",
        "test_cli_dry_run_rejects_plan_with_case_insensitive_duplicate_outputs",
        "test_cli_dry_run_rejects_plan_with_backoff_longer_than_retries",
        "test_cli_dry_run_cycle_plan_returns_two",
        "test_cli_resume_invalid_plan_copy_returns_two",
        "test_cli_resume_rejects_symlink_plan_file",
    }
    matrix_mode_names = {
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_invalid_workdir_matrix",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_home_matrix",
        "test_cli_run_dry_run_both_toggles_invalid_plan_precedes_workdir_default_existing_home_matrix",
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_home_matrix",
        "test_cli_run_dry_run_both_toggles_reject_invalid_plan_default_existing_home_matrix",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert 'assert "Plan validation error" in output' in source_segment
        assert 'assert "contains symlink component" not in output' in source_segment
        assert 'assert "must not include symlink" not in output' in source_segment
        assert 'assert "must not be symlink" not in output' in source_segment
        assert (
            'assert "symbolic links" not in output.lower()' in source_segment
            or 'assert "symbolic links" not in output.lower(), context' in source_segment
        )
        if "symlink" in node.name:
            assert 'assert "invalid plan path" in output' in source_segment
            assert 'assert "symbolic links" not in output.lower()' in source_segment
        if node.name in matrix_mode_names:
            assert 'if plan_mode in {"symlink_plan", "symlink_ancestor_plan"}:' in source_segment
            assert 'assert "invalid plan path" in output, context' in source_segment
            assert 'assert "invalid plan path" not in output, context' in source_segment
            assert 'assert "symbolic links" not in output.lower(), context' in source_segment
        matched.add(node.name)

    assert matched == expected_names


def test_cli_integration_plural_symbolic_links_assertions_require_singular_pair() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_lines = integration_source.splitlines()

    expected_suffixes = ("", ", context", ", command")
    examined = 0
    for index, line in enumerate(integration_lines):
        stripped = line.strip()
        expected: str | None = None
        for suffix in expected_suffixes:
            if stripped == f'assert "symbolic links" not in output.lower(){suffix}':
                expected = f'assert "symbolic link" not in output.lower(){suffix}'
                break
            if stripped == f'assert "symbolic links" not in proc.stdout.lower(){suffix}':
                expected = f'assert "symbolic link" not in proc.stdout.lower(){suffix}'
                break
        if expected is None:
            continue

        examined += 1
        next_line = (
            integration_lines[index + 1].strip() if index + 1 < len(integration_lines) else ""
        )
        assert next_line == expected

    assert examined > 0


def test_cli_integration_early_plan_cases_suppress_singular_symbolic_link_detail() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expected_names = {
        "test_cli_run_dry_run_fail_fast_still_rejects_invalid_plan",
        "test_cli_run_dry_run_no_fail_fast_still_rejects_invalid_plan",
        "test_cli_run_dry_run_both_fail_fast_toggles_still_rejects_invalid_plan",
        "test_cli_run_dry_run_both_fail_fast_toggles_reverse_order_rejects_invalid_plan",
        "test_cli_run_dry_run_both_toggles_rejects_symlink_plan_path",
        "test_cli_run_dry_run_both_toggles_reverse_rejects_symlink_plan_path",
        "test_cli_run_dry_run_both_toggles_rejects_plan_path_with_symlink_ancestor",
        "test_cli_run_dry_run_both_toggles_reverse_rejects_plan_path_with_symlink_ancestor",
        "test_cli_run_dry_run_both_toggles_symlinked_plan_precedes_invalid_workdir",
    }
    context_names = {
        "test_cli_run_dry_run_both_toggles_symlinked_plan_precedes_invalid_workdir",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert (
            'assert "symbolic links" not in output.lower()' in source_segment
            or 'assert "symbolic links" not in output.lower(), context' in source_segment
        )
        if node.name in context_names:
            assert 'assert "symbolic link" not in output.lower(), context' in source_segment
        else:
            assert 'assert "symbolic link" not in output.lower()' in source_segment
        matched.add(node.name)

    assert matched == expected_names


def test_cli_integration_invalid_plan_path_assertions_require_symbolic_suppression() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    examined: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        if (
            'assert "invalid plan path" in output' not in source_segment
            and 'assert "invalid plan path" in proc.stdout' not in source_segment
        ):
            continue

        assert (
            'assert "contains symlink component" not in output' in source_segment
            or 'assert "contains symlink component" not in proc.stdout' in source_segment
        )
        assert (
            'assert "must not include symlink" not in output' in source_segment
            or 'assert "must not include symlink" not in proc.stdout' in source_segment
        )
        assert (
            'assert "must not be symlink" not in output' in source_segment
            or 'assert "must not be symlink" not in proc.stdout' in source_segment
        )
        assert (
            'assert "symbolic links" not in output.lower()' in source_segment
            or 'assert "symbolic links" not in proc.stdout.lower()' in source_segment
            or 'assert "symbolic links" not in output.lower(), context' in source_segment
        )
        examined.add(node.name)

    assert examined


def test_cli_integration_path_validation_output_markers_require_symlink_suppression() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    markers = {
        'assert "Invalid home" in output',
        'assert "Invalid workdir" in output',
        'assert "Invalid run_id" in output',
        'assert "Plan validation error" in output',
        'assert "Invalid home" in proc.stdout',
        'assert "Invalid workdir" in proc.stdout',
        'assert "Invalid run_id" in proc.stdout',
        'assert "Plan validation error" in proc.stdout',
    }

    examined: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        if not any(marker in source_segment for marker in markers):
            continue

        assert (
            'assert "contains symlink component" not in output' in source_segment
            or 'assert "contains symlink component" not in proc.stdout' in source_segment
        )

        if (
            'assert "Plan validation error" in output' in source_segment
            or 'assert "Plan validation error" in proc.stdout' in source_segment
        ):
            assert (
                'assert "must not include symlink" not in output' in source_segment
                or 'assert "must not include symlink" not in proc.stdout' in source_segment
            )
            assert (
                'assert "must not be symlink" not in output' in source_segment
                or 'assert "must not be symlink" not in proc.stdout' in source_segment
            )
            assert (
                'assert "symbolic links" not in output.lower()' in source_segment
                or 'assert "symbolic links" not in proc.stdout.lower()' in source_segment
                or 'assert "symbolic links" not in output.lower(), context' in source_segment
            )
        examined.add(node.name)

    assert examined


def test_cli_integration_runs_symlink_path_error_is_sanitized() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    target_name = "test_cli_status_logs_resume_sanitize_runs_symlink_path_error"

    target = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef) and node.name == target_name
        ),
        None,
    )
    assert target is not None

    source_segment = ast.get_source_segment(integration_source, target)
    assert source_segment is not None
    assert 'for command in ("status", "logs", "resume")' in source_segment
    assert 'assert "invalid run path" in output, command' in source_segment
    assert 'assert "contains symlink component" not in output, command' in source_segment
    assert 'assert "must not include symlink" not in output, command' in source_segment
    assert 'assert "must not be symlink" not in output, command' in source_segment
    assert 'assert "symbolic links" not in output.lower(), command' in source_segment
    assert 'assert "symbolic link" not in output.lower(), command' in source_segment


def test_cli_integration_cancel_symlink_cancel_request_error_is_sanitized() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    target_name = "test_cli_cancel_sanitizes_symlink_cancel_request_write_error"

    target = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef) and node.name == target_name
        ),
        None,
    )
    assert target is not None

    source_segment = ast.get_source_segment(integration_source, target)
    assert source_segment is not None
    assert 'assert "Failed to request cancel" in output' in source_segment
    assert 'assert "invalid run path" in output' in source_segment
    assert 'assert "contains symlink component" not in output' in source_segment
    assert 'assert "must not include symlink" not in output' in source_segment
    assert 'assert "must not be symlink" not in output' in source_segment
    assert 'assert "symbolic links" not in output.lower()' in source_segment
    assert 'assert "symbolic link" not in output.lower()' in source_segment


def test_cli_integration_run_initialize_symlink_error_is_sanitized() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    target_name = "test_cli_run_sanitizes_runs_symlink_path_initialize_error"

    target = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef) and node.name == target_name
        ),
        None,
    )
    assert target is not None

    source_segment = ast.get_source_segment(integration_source, target)
    assert source_segment is not None
    assert 'assert "Failed to initialize run" in output' in source_segment
    assert 'assert "invalid run path" in output' in source_segment
    assert 'assert "contains symlink component" not in output' in source_segment
    assert 'assert "must not include symlink" not in output' in source_segment
    assert 'assert "must not be symlink" not in output' in source_segment
    assert 'assert "symbolic links" not in output.lower()' in source_segment
    assert 'assert "symbolic link" not in output.lower()' in source_segment


def test_cli_integration_resume_report_symlink_warning_is_sanitized() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)
    target_name = "test_cli_resume_continues_when_report_path_is_symlink"

    target = next(
        (
            node
            for node in ast.walk(integration_module)
            if isinstance(node, ast.FunctionDef) and node.name == target_name
        ),
        None,
    )
    assert target is not None

    source_segment = ast.get_source_segment(integration_source, target)
    assert source_segment is not None
    assert 'assert "failed to write report" in output' in source_segment
    assert 'assert "invalid run path" in output' in source_segment
    assert 'assert "contains symlink component" not in output' in source_segment
    assert 'assert "must not include symlink" not in output' in source_segment
    assert 'assert "must not be symlink" not in output' in source_segment
    assert 'assert "symbolic links" not in output.lower()' in source_segment
    assert 'assert "symbolic link" not in output.lower()' in source_segment


def test_cli_integration_status_state_symlink_errors_are_sanitized() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    expected_names = {
        "test_cli_status_rejects_symlink_state_file",
        "test_cli_status_rejects_state_path_with_symlink_ancestor",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expected_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        assert 'assert "Failed to load state" in output' in source_segment
        assert 'assert "invalid run path" in output' in source_segment
        assert 'assert "contains symlink component" not in output' in source_segment
        assert 'assert "must not include symlink" not in output' in source_segment
        assert 'assert "must not be symlink" not in output' in source_segment
        assert 'assert "symbolic links" not in output.lower()' in source_segment
        assert 'assert "symbolic link" not in output.lower()' in source_segment
        matched.add(node.name)

    assert matched == expected_names


def test_cli_integration_runtime_symlink_errors_require_sanitized_run_path() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    runtime_markers = {
        'assert "Failed to initialize run" in output',
        'assert "Failed to load state" in output',
        'assert "Run not found or broken" in output',
        'assert "Failed to request cancel" in output',
        'assert "Run execution failed" in output',
        'assert "failed to write report" in output',
    }

    examined: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_") or "symlink" not in node.name:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        if not any(marker in source_segment for marker in runtime_markers):
            continue

        assert 'assert "invalid run path" in output' in source_segment
        assert 'assert "contains symlink component" not in output' in source_segment
        assert 'assert "must not include symlink" not in output' in source_segment
        assert 'assert "must not be symlink" not in output' in source_segment
        assert 'assert "symbolic links" not in output.lower()' in source_segment
        examined.add(node.name)

    assert examined


def test_cli_integration_positive_runtime_symlink_errors_require_sanitized_run_path() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    positive_runtime_markers = {
        'assert "Failed to initialize run" in output',
        'assert "Failed to load state" in output',
        'assert "Run not found or broken" in output',
        'assert "Failed to request cancel" in output',
        'assert "Run execution failed" in output',
        'assert "failed to write report" in output',
        'assert "Failed to initialize run" in proc.stdout',
        'assert "Failed to load state" in proc.stdout',
        'assert "Run not found or broken" in proc.stdout',
        'assert "Failed to request cancel" in proc.stdout',
        'assert "Run execution failed" in proc.stdout',
        'assert "failed to write report" in proc.stdout',
    }

    examined: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        if not any(marker in source_segment for marker in positive_runtime_markers):
            continue
        if "symlink" not in source_segment.lower():
            continue

        assert (
            'assert "invalid run path" in output' in source_segment
            or 'assert "invalid run path" in proc.stdout' in source_segment
        )
        assert (
            'assert "contains symlink component" not in output' in source_segment
            or 'assert "contains symlink component" not in proc.stdout' in source_segment
        )
        assert (
            'assert "must not include symlink" not in output' in source_segment
            or 'assert "must not include symlink" not in proc.stdout' in source_segment
        )
        assert (
            'assert "must not be symlink" not in output' in source_segment
            or 'assert "must not be symlink" not in proc.stdout' in source_segment
        )
        assert (
            'assert "symbolic links" not in output.lower()' in source_segment
            or 'assert "symbolic links" not in proc.stdout.lower()' in source_segment
        )
        examined.add(node.name)

    assert examined


def test_cli_integration_resume_invalid_run_id_workdir_preserve_supergroup_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_non_preserve = "test_cli_resume_invalid_run_id_precedes_invalid_workdir_modes_matrix"
    default_non_preserve = (
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_modes_matrix"
    )
    explicit_preserve = (
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_"
        "preserves_entries_matrix"
    )
    default_preserve = (
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_"
        "preserve_entries_matrix"
    )
    explicit_with_runs = (
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_with_runs_matrix"
    )
    default_with_runs = (
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_with_runs_matrix"
    )
    explicit_artifacts = (
        "test_cli_resume_invalid_run_id_invalid_workdir_existing_home_run_artifacts_matrix"
    )
    default_artifacts = (
        "test_cli_resume_default_home_invalid_run_id_invalid_workdir_run_artifacts_matrix"
    )

    expectations = {
        explicit_non_preserve: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_non_preserve: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_preserve: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_preserve: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_with_runs: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        default_with_runs: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        explicit_artifacts: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
        default_artifacts: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "flag_orders" in source_segment
        assert "run_id_modes" in source_segment
        assert "workdir_modes" in source_segment
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert "for workdir_mode in workdir_modes:" in source_segment
        assert 'bad_run_id = "../escape" if run_id_mode == "path_escape" else "a" * 129' in (
            source_segment
        )
        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid workdir" not in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "run_id: [bold]" not in output, context' in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment
        assert '"--workdir"' in source_segment
        assert "*order" in source_segment

        if expected["has_artifacts"]:
            assert 'assert "state: [bold]" not in output, context' in source_segment
            assert 'assert "report: [bold]" not in output, context' in source_segment
        else:
            assert 'assert "state:" not in output, context' in source_segment
            assert 'assert "report:" not in output, context' in source_segment

        if expected["preserves_entries"]:
            assert f"assert {home_var}.exists(), context" in source_segment
            assert f"{home_var}.iterdir()" in source_segment
            assert '"keep.txt"' in source_segment
            assert '"keep_dir"' in source_segment
            assert (
                'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context'
                in source_segment
            )
            assert "assert sentinel_dir.is_dir(), context" in source_segment
            assert (
                'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
                in source_segment
            )
            if expected["has_existing_runs"]:
                assert '"runs"' in source_segment
                assert "existing_run" in source_segment
                assert f'({home_var} / "runs").iterdir()' in source_segment
                assert '"keep_run"' in source_segment
                if expected["has_artifacts"]:
                    assert '"cancel.request"' in source_segment
                    assert '"task.log"' in source_segment
                    assert '"plan.yaml"' in source_segment
                    assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
                    assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
                    assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
                    assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
                else:
                    assert "sorted(path.name for path in existing_run.iterdir())" in source_segment
                    assert '"plan.yaml"' in source_segment
                    assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
                    assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                        source_segment
                    )
            else:
                assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        else:
            assert f"assert not {home_var}.exists(), context" in source_segment
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_resume_invalid_run_id_workdir_matrix_groups_keep_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = "test_cli_resume_invalid_run_id_precedes_invalid_workdir_modes_matrix"
    default_matrix = (
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_modes_matrix"
    )
    explicit_preserve_matrix = (
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_"
        "preserves_entries_matrix"
    )
    default_preserve_matrix = (
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_"
        "preserve_entries_matrix"
    )
    explicit_with_runs_matrix = (
        "test_cli_resume_invalid_run_id_precedes_invalid_workdir_existing_home_with_runs_matrix"
    )
    default_with_runs_matrix = (
        "test_cli_resume_default_home_invalid_run_id_precedes_invalid_workdir_with_runs_matrix"
    )
    explicit_artifacts_matrix = (
        "test_cli_resume_invalid_run_id_invalid_workdir_existing_home_run_artifacts_matrix"
    )
    default_artifacts_matrix = (
        "test_cli_resume_default_home_invalid_run_id_invalid_workdir_run_artifacts_matrix"
    )

    expectations = {
        explicit_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": False,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_preserve_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        default_preserve_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": True,
            "has_existing_runs": False,
            "has_artifacts": False,
        },
        explicit_with_runs_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        default_with_runs_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": False,
        },
        explicit_artifacts_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
        default_artifacts_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
            "preserves_entries": True,
            "has_existing_runs": True,
            "has_artifacts": True,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "flag_orders" in source_segment
        assert "run_id_modes" in source_segment
        assert "workdir_modes" in source_segment
        assert "for run_id_mode in run_id_modes:" in source_segment
        assert "for workdir_mode in workdir_modes:" in source_segment
        assert 'assert "Invalid run_id" in output, context' in source_segment
        assert 'assert "Invalid workdir" not in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Run not found or broken" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "run_id: [bold]" not in output, context' in source_segment
        if expected["has_artifacts"]:
            assert 'assert "state: [bold]" not in output, context' in source_segment
            assert 'assert "report: [bold]" not in output, context' in source_segment
        else:
            assert 'assert "state:" not in output, context' in source_segment
            assert 'assert "report:" not in output, context' in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment

        if expected["preserves_entries"]:
            assert f"assert {home_var}.exists(), context" in source_segment
            assert f"{home_var}.iterdir()" in source_segment
            assert '"keep.txt"' in source_segment
            assert '"keep_dir"' in source_segment
            assert (
                'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context'
                in source_segment
            )
            assert "assert sentinel_dir.is_dir(), context" in source_segment
            assert (
                'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
                in source_segment
            )
            if expected["has_existing_runs"]:
                assert '"runs"' in source_segment
                assert "existing_run" in source_segment
                assert f'({home_var} / "runs").iterdir()' in source_segment
                assert '"keep_run"' in source_segment
                if expected["has_artifacts"]:
                    assert '"cancel.request"' in source_segment
                    assert '"task.log"' in source_segment
                    assert '"plan.yaml"' in source_segment
                    assert 'read_text(encoding="utf-8") == "tasks: []\\n"' in source_segment
                    assert 'read_text(encoding="utf-8") == "lock\\n"' in source_segment
                    assert 'read_text(encoding="utf-8") == "cancel\\n"' in source_segment
                    assert 'read_text(encoding="utf-8") == "log\\n"' in source_segment
                else:
                    assert "sorted(path.name for path in existing_run.iterdir())" in source_segment
                    assert '"plan.yaml"' in source_segment
                    assert 'assert not (existing_run / ".lock").exists(), context' in source_segment
                    assert 'assert not (existing_run / "cancel.request").exists(), context' in (
                        source_segment
                    )
            else:
                assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        else:
            assert f"assert not {home_var}.exists(), context" in source_segment
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment
        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_run_and_resume_invalid_workdir_mode_matrices_keep_parity() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    run_matrix = "test_cli_run_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"
    run_default_matrix = (
        "test_cli_run_default_home_rejects_invalid_workdir_modes_without_creating_run_dir_matrix"
    )
    resume_matrix = "test_cli_resume_rejects_invalid_workdir_modes_matrix"
    resume_default_matrix = "test_cli_resume_default_home_rejects_invalid_workdir_modes_matrix"
    expectations = {
        run_matrix: {
            "has_resume_state_guard": False,
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
        },
        run_default_matrix: {
            "has_resume_state_guard": False,
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
        },
        resume_matrix: {
            "has_resume_state_guard": True,
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
        },
        resume_default_matrix: {
            "has_resume_state_guard": True,
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert "flag_orders" in source_segment
        assert "workdir_modes" in source_segment
        assert "for workdir_mode in workdir_modes:" in source_segment
        assert 'if workdir_mode == "missing_path":' in source_segment
        assert 'elif workdir_mode == "file_path":' in source_segment
        assert 'elif workdir_mode == "file_ancestor":' in source_segment
        assert 'elif workdir_mode == "symlink_to_file":' in source_segment
        assert 'elif workdir_mode == "dangling_symlink":' in source_segment
        assert "else:" in source_segment
        assert 'assert "Invalid workdir" in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "run_id:" not in output, context' in source_segment
        assert 'assert "state:" not in output, context' in source_segment
        assert 'assert "report:" not in output, context' in source_segment
        assert '"--workdir"' in source_segment
        assert "*order" in source_segment
        assert '"--dry-run"' not in source_segment
        assert "side_effect_files" in source_segment
        assert "for file_path in side_effect_files:" in source_segment
        assert 'if workdir_mode == "dangling_symlink":' in source_segment
        assert 'if workdir_mode == "symlink_ancestor":' in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        if expected["has_resume_state_guard"]:
            assert '"resume"' in source_segment
            assert "run_id = _extract_run_id(run_proc.stdout)" in source_segment
            assert 'assert "Run not found or broken" not in output, context' in source_segment
            assert "baseline_state" in source_segment
            assert 'state_path.read_text(encoding="utf-8") == baseline_state' in source_segment
            assert f'sorted(path.name for path in ({home_var} / "runs").iterdir()) == [run_id]' in (
                source_segment
            )
        else:
            assert '"run"' in source_segment
            assert "run_id = _extract_run_id(run_proc.stdout)" not in source_segment
            assert 'assert "Run not found or broken" not in output, context' not in (source_segment)
            assert "baseline_state" not in source_segment
            assert 'state_path.read_text(encoding="utf-8") == baseline_state' not in (
                source_segment
            )
            assert f"assert not {home_var}.exists(), context" in source_segment
            assert f'assert not ({home_var} / "runs").exists(), context' in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_preserve_entries_matrices_keep_mode_and_toggle_contracts() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = (
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_plan_error_matrix"
    )
    default_matrix = (
        "test_cli_run_dry_run_both_toggles_default_existing_home_"
        "preserves_entries_plan_error_matrix"
    )
    expected_case_names = {
        "invalid_only",
        "invalid_with_workdir",
        "missing_only",
        "missing_with_workdir",
    }
    expected_plan_kinds = {"invalid_plan", "missing_plan"}
    expected_workdir_modes = {False, True}
    expected_toggle_orders = {
        ("--fail-fast", "--no-fail-fast"),
        ("--no-fail-fast", "--fail-fast"),
    }

    expectations = {
        explicit_matrix: {"has_cwd": False, "uses_home_flag": True},
        default_matrix: {"has_cwd": True, "uses_home_flag": False},
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        flag_orders_assign = next(
            (
                stmt
                for stmt in node.body
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
        assert toggle_orders == expected_toggle_orders

        case_modes_assign = next(
            (
                stmt
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "case_modes"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Tuple)
            ),
            None,
        )
        assert case_modes_assign is not None
        assert isinstance(case_modes_assign.value, ast.Tuple)

        case_names: set[str] = set()
        plan_kinds: set[str] = set()
        workdir_modes: set[bool] = set()
        for case_node in case_modes_assign.value.elts:
            assert isinstance(case_node, ast.Tuple)
            assert len(case_node.elts) == 3

            name_node = case_node.elts[0]
            kind_node = case_node.elts[1]
            workdir_node = case_node.elts[2]

            assert isinstance(name_node, ast.Constant)
            assert isinstance(name_node.value, str)
            case_names.add(name_node.value)

            assert isinstance(kind_node, ast.Constant)
            assert isinstance(kind_node.value, str)
            plan_kinds.add(kind_node.value)

            assert isinstance(workdir_node, ast.Constant)
            assert isinstance(workdir_node.value, bool)
            workdir_modes.add(workdir_node.value)

        assert case_names == expected_case_names
        assert plan_kinds == expected_plan_kinds
        assert workdir_modes == expected_workdir_modes

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment
        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_preserve_entries_matrices_keep_error_branch_contracts() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = (
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_plan_error_matrix"
    )
    default_matrix = (
        "test_cli_run_dry_run_both_toggles_default_existing_home_"
        "preserves_entries_plan_error_matrix"
    )
    expectations = {
        explicit_matrix: {"home_var": "home", "has_cwd": False, "uses_home_flag": True},
        default_matrix: {"home_var": "default_home", "has_cwd": True, "uses_home_flag": False},
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert 'if plan_kind == "invalid_plan":' in source_segment
        assert 'assert "Plan validation error" in output, context' in source_segment
        assert 'assert "PLAN_PATH" not in output, context' in source_segment
        assert 'assert "PLAN_PATH" in output, context' in source_segment
        assert "assert \"Invalid value for 'PLAN_PATH'\" in output, context" in source_segment
        assert 'assert "Plan validation error" not in output, context' in source_segment
        assert 'assert "contains symlink component" not in output, context' in source_segment
        assert 'assert "Invalid home" not in output, context' in source_segment
        assert 'assert "Invalid workdir" not in output, context' in source_segment
        assert 'assert "Dry Run" not in output, context' in source_segment
        assert 'assert "run_id:" not in output, context' in source_segment
        assert 'assert "state:" not in output, context' in source_segment
        assert 'assert "report:" not in output, context' in source_segment

        assert f"assert {home_var}.exists(), context" in source_segment
        assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert 'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context' in (
            source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment
        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_preserve_entries_matrices_keep_branch_and_workdir_conditionals() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    matrix_names = {
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_plan_error_matrix",
        "test_cli_run_dry_run_both_toggles_default_existing_home_preserves_entries_plan_error_matrix",
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in matrix_names:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None

        assert 'if plan_kind == "invalid_plan":' in source_segment
        assert "plan_path.write_text(" in source_segment
        assert 'plan_path = case_root / "missing_plan.yaml"' in source_segment
        assert "command = [" in source_segment
        assert '"--dry-run"' in source_segment
        assert "*order" in source_segment

        assert "if needs_workdir:" in source_segment
        assert 'invalid_workdir_file = case_root / "invalid_workdir"' in source_segment
        assert 'command.extend(["--workdir", str(invalid_workdir_file)])' in source_segment
        assert 'nested_file = sentinel_dir / "nested.txt"' in source_segment
        assert 'nested_file.write_text("nested\\n", encoding="utf-8")' in source_segment

        matched.add(node.name)

    assert matched == matrix_names


def test_cli_integration_existing_home_preserve_entries_matrices_keep_boundaries() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    explicit_matrix = (
        "test_cli_run_dry_run_both_toggles_existing_home_preserves_entries_plan_error_matrix"
    )
    default_matrix = (
        "test_cli_run_dry_run_both_toggles_default_existing_home_"
        "preserves_entries_plan_error_matrix"
    )

    expectations = {
        explicit_matrix: {
            "home_var": "home",
            "has_cwd": False,
            "uses_home_flag": True,
        },
        default_matrix: {
            "home_var": "default_home",
            "has_cwd": True,
            "uses_home_flag": False,
        },
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]
        home_var = expected["home_var"]

        assert f"assert {home_var}.exists(), context" in source_segment
        assert f'assert not ({home_var} / "runs").exists(), context' in source_segment
        assert f"{home_var}.iterdir()" in source_segment
        assert '"keep.txt"' in source_segment
        assert '"keep_dir"' in source_segment
        assert (
            'assert sentinel_file.read_text(encoding="utf-8") == "keep\\n", context'
            in source_segment
        )
        assert "assert sentinel_dir.is_dir(), context" in source_segment
        assert (
            'assert nested_file.read_text(encoding="utf-8") == "nested\\n", context'
            in source_segment
        )
        assert '"run_id:" not in output' in source_segment
        assert '"state:" not in output' in source_segment
        assert '"report:" not in output' in source_segment

        if expected["has_cwd"]:
            assert "cwd=case_root" in source_segment
        else:
            assert "cwd=case_root" not in source_segment

        if expected["uses_home_flag"]:
            assert '"--home"' in source_segment
        else:
            assert '"--home"' not in source_segment

        assert '"--workdir"' in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


def test_cli_integration_missing_plan_existing_home_cases_keep_home_contracts() -> None:
    tests_root = Path(__file__).resolve().parents[1] / "tests"
    integration_source = (tests_root / "test_cli_integration.py").read_text(encoding="utf-8")
    integration_module = ast.parse(integration_source)

    reject_existing_home = (
        "test_cli_run_dry_run_both_toggles_reject_missing_plan_path_existing_home_matrix"
    )
    workdir_existing_home = (
        "test_cli_run_dry_run_both_toggles_missing_plan_precedes_workdir_existing_home_matrix"
    )

    expectations = {
        reject_existing_home: {"needs_workdir": False},
        workdir_existing_home: {"needs_workdir": True},
    }

    matched: set[str] = set()
    for node in ast.walk(integration_module):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in expectations:
            continue

        source_segment = ast.get_source_segment(integration_source, node)
        assert source_segment is not None
        expected = expectations[node.name]

        assert "cwd=case_root" not in source_segment
        assert "assert home.exists(), context" in source_segment
        assert 'assert not (home / "runs").exists(), context' in source_segment
        assert (
            "assert sorted(path.name for path in home.iterdir()) == [], context" in source_segment
        )
        assert '"--home"' in source_segment

        if expected["needs_workdir"]:
            assert '"--workdir"' in source_segment
        else:
            assert '"--workdir"' not in source_segment

        matched.add(node.name)

    assert matched == set(expectations)


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
