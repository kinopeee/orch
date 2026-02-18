from __future__ import annotations

import pytest

from orch.config.schema import PlanSpec, TaskSpec
from orch.dag.build import build_adjacency
from orch.dag.validate import assert_acyclic
from orch.util.errors import PlanError


def test_assert_acyclic_returns_topological_order() -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[
            TaskSpec(id="a", cmd=["echo", "a"]),
            TaskSpec(id="b", cmd=["echo", "b"], depends_on=["a"]),
            TaskSpec(id="c", cmd=["echo", "c"], depends_on=["b"]),
        ],
    )
    dependents, in_degree = build_adjacency(plan)
    order = assert_acyclic([task.id for task in plan.tasks], dependents, in_degree)
    assert order == ["a", "b", "c"]


def test_assert_acyclic_detects_cycle() -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[
            TaskSpec(id="a", cmd=["echo", "a"], depends_on=["c"]),
            TaskSpec(id="b", cmd=["echo", "b"], depends_on=["a"]),
            TaskSpec(id="c", cmd=["echo", "c"], depends_on=["b"]),
        ],
    )
    dependents, in_degree = build_adjacency(plan)
    with pytest.raises(PlanError):
        assert_acyclic([task.id for task in plan.tasks], dependents, in_degree)
