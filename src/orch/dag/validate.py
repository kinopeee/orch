from __future__ import annotations

from collections import deque

from orch.util.errors import PlanError


def assert_acyclic(
    task_ids: list[str], dependents: dict[str, list[str]], in_degree: dict[str, int]
) -> list[str]:
    """Validate DAG and return one topological order."""
    queue = deque([tid for tid in task_ids if in_degree[tid] == 0])
    seen: list[str] = []
    local_in = dict(in_degree)
    while queue:
        current = queue.popleft()
        seen.append(current)
        for nxt in dependents.get(current, []):
            local_in[nxt] -= 1
            if local_in[nxt] == 0:
                queue.append(nxt)
    if len(seen) != len(task_ids):
        raise PlanError("plan contains dependency cycle")
    return seen
