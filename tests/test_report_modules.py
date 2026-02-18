from __future__ import annotations

from pathlib import Path

from orch.report.render_md import render_markdown
from orch.report.summarize import build_summary
from orch.state.model import RunState, TaskState


def _make_state() -> RunState:
    return RunState(
        run_id="r1",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:10+00:00",
        status="FAILED",
        goal="demo",
        plan_relpath="plan.yaml",
        home=".orch",
        workdir=".",
        max_parallel=2,
        fail_fast=False,
        tasks={
            "ok": TaskState(
                status="SUCCESS",
                depends_on=[],
                cmd=["echo", "ok"],
                cwd=".",
                env=None,
                timeout_sec=None,
                retries=0,
                retry_backoff_sec=[],
                outputs=["out.txt"],
                attempts=1,
                stdout_path="logs/ok.out.log",
                stderr_path="logs/ok.err.log",
                artifact_paths=["artifacts/ok/out.txt"],
            ),
            "bad": TaskState(
                status="FAILED",
                depends_on=[],
                cmd=["echo", "bad"],
                cwd=".",
                env=None,
                timeout_sec=None,
                retries=0,
                retry_backoff_sec=[],
                outputs=[],
                attempts=1,
                exit_code=1,
                skip_reason="failed",
                stdout_path="logs/bad.out.log",
                stderr_path="logs/bad.err.log",
            ),
        },
    )


def _make_success_state() -> RunState:
    return RunState(
        run_id="r2",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:05+00:00",
        status="SUCCESS",
        goal=None,
        plan_relpath="plan.yaml",
        home=".orch",
        workdir=".",
        max_parallel=1,
        fail_fast=False,
        tasks={
            "ok": TaskState(
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
                stdout_path="logs/ok.out.log",
                stderr_path="logs/ok.err.log",
            )
        },
    )


def test_build_summary_collects_problems_and_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "bad.err.log").write_text("line1\nline2\n", encoding="utf-8")
    (run_dir / "logs" / "ok.err.log").write_text("", encoding="utf-8")
    state = _make_state()

    summary = build_summary(state, run_dir)
    assert summary["run"]["status"] == "FAILED"  # type: ignore[index]
    assert len(summary["tasks"]) == 2  # type: ignore[arg-type]
    problems = summary["problems"]  # type: ignore[assignment]
    assert len(problems) == 1
    assert problems[0]["id"] == "bad"
    assert problems[0]["stderr_tail"] == ["line1", "line2"]
    artifacts = summary["artifacts"]  # type: ignore[assignment]
    assert artifacts == [{"task_id": "ok", "path": "artifacts/ok/out.txt"}]


def test_render_markdown_includes_problem_and_artifact_sections(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "bad.err.log").write_text("boom\n", encoding="utf-8")
    state = _make_state()
    summary = build_summary(state, run_dir)

    markdown = render_markdown(summary)
    assert "# Final Run Report" in markdown
    assert "status: **FAILED**" in markdown
    assert "### bad (FAILED)" in markdown
    assert "boom" in markdown
    assert "`artifacts/ok/out.txt` (task: `ok`)" in markdown


def test_render_markdown_shows_empty_sections_for_success_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "ok.err.log").write_text("", encoding="utf-8")
    state = _make_success_state()
    summary = build_summary(state, run_dir)

    markdown = render_markdown(summary)
    assert "status: **SUCCESS**" in markdown
    assert "No failed/skipped/canceled tasks." in markdown
    assert "\n- (none)\n" in markdown
