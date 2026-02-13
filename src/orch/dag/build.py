from __future__ import annotations

from collections import defaultdict

from orch.config.schema import PlanSpec


def build_adjacency(plan: PlanSpec) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Build dependents adjacency list and in-degree map."""
    dependents: dict[str, list[str]] = defaultdict(list)
    in_degree = {task.id: len(task.depends_on) for task in plan.tasks}
    for task in plan.tasks:
        dependents.setdefault(task.id, [])
        for dep in task.depends_on:
            dependents[dep].append(task.id)
    return dict(dependents), in_degree
