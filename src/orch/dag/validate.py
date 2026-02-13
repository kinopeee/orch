"""DAG validation helpers."""

from __future__ import annotations

from collections import deque

from orch.util.errors import PlanError


def assert_acyclic(
    task_ids: list[str], dependents: dict[str, list[str]], in_degree: dict[str, int]
) -> None:
    """Validate graph has no cycle using Kahn's algorithm."""
    degrees = dict(in_degree)
    q = deque([task_id for task_id in task_ids if degrees.get(task_id, 0) == 0])
    seen = 0

    while q:
        current = q.popleft()
        seen += 1
        for nxt in dependents.get(current, []):
            degrees[nxt] = degrees[nxt] - 1
            if degrees[nxt] == 0:
                q.append(nxt)

    if seen != len(task_ids):
        raise PlanError("Plan has cyclic dependencies.")
