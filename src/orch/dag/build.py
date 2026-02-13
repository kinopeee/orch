"""Build graph structures from plan."""

from __future__ import annotations

from collections import defaultdict

from orch.config.schema import PlanSpec


def build_adjacency(plan: PlanSpec) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Return dependents adjacency and in-degree by task id."""
    dependents: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {}

    for task in plan.tasks:
        in_degree[task.id] = len(task.depends_on)
        dependents.setdefault(task.id, [])
        for dep in task.depends_on:
            dependents[dep].append(task.id)

    return dict(dependents), in_degree
