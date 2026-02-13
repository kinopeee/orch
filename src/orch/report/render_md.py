from __future__ import annotations

from typing import Any


def _bool_mark(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(summary: dict[str, Any]) -> str:
    run = summary["run"]
    tasks = summary["tasks"]
    problems = summary["problems"]
    artifacts = summary["artifacts"]

    lines: list[str] = []
    lines.append("# Final Run Report")
    lines.append("")
    lines.append("## Run Overview")
    lines.append("")
    lines.append(f"- run_id: `{run['run_id']}`")
    lines.append(f"- goal: {run['goal'] or '(none)'}")
    lines.append(f"- status: **{run['status']}**")
    lines.append(f"- started: {run['created_at']}")
    lines.append(f"- ended: {run['updated_at']}")
    lines.append(f"- max_parallel: {run['max_parallel']}")
    lines.append(f"- fail_fast: {_bool_mark(run['fail_fast'])}")
    lines.append(f"- workdir: `{run['workdir']}`")
    lines.append("")
    lines.append("## Task Results")
    lines.append("")
    lines.append("| id | status | attempts | duration_sec | exit_code | timed_out | logs |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in tasks:
        logs = f"`{row['stdout_path']}` / `{row['stderr_path']}`"
        lines.append(
            f"| {row['id']} | {row['status']} | {row['attempts']} | "
            f"{row['duration_sec']} | {row['exit_code']} | {row['timed_out']} | {logs} |"
        )
    lines.append("")
    lines.append("## Failed / Skipped / Canceled Details")
    lines.append("")
    if problems:
        for row in problems:
            lines.append(f"### {row['id']} ({row['status']})")
            if row["skip_reason"]:
                lines.append(f"- skip_reason: `{row['skip_reason']}`")
            lines.append("- stderr tail:")
            lines.append("```")
            stderr_tail = row["stderr_tail"] or ["(empty)"]
            lines.extend(stderr_tail)
            lines.append("```")
            lines.append("")
    else:
        lines.append("No failed/skipped/canceled tasks.")
        lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    if artifacts:
        for artifact in artifacts:
            lines.append(f"- `{artifact['path']}` (task: `{artifact['task_id']}`)")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)
