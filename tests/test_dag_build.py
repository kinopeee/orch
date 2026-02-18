from __future__ import annotations

from orch.config.schema import PlanSpec, TaskSpec
from orch.dag.build import build_adjacency
from orch.dag.validate import assert_acyclic


def test_build_adjacency_includes_leaf_nodes_and_correct_in_degree() -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[
            TaskSpec(id="root", cmd=["echo", "root"]),
            TaskSpec(id="child_a", cmd=["echo", "a"], depends_on=["root"]),
            TaskSpec(id="child_b", cmd=["echo", "b"], depends_on=["root"]),
            TaskSpec(id="leaf", cmd=["echo", "leaf"], depends_on=["child_a", "child_b"]),
        ],
    )

    dependents, in_degree = build_adjacency(plan)
    assert dependents["root"] == ["child_a", "child_b"]
    assert dependents["leaf"] == []
    assert in_degree == {"root": 0, "child_a": 1, "child_b": 1, "leaf": 2}


def test_assert_acyclic_keeps_input_in_degree_unchanged() -> None:
    plan = PlanSpec(
        goal=None,
        artifacts_dir=None,
        tasks=[
            TaskSpec(id="a", cmd=["echo", "a"]),
            TaskSpec(id="b", cmd=["echo", "b"], depends_on=["a"]),
            TaskSpec(id="c", cmd=["echo", "c"], depends_on=["a"]),
        ],
    )

    dependents, in_degree = build_adjacency(plan)
    original = dict(in_degree)
    order = assert_acyclic([task.id for task in plan.tasks], dependents, in_degree)
    idx = {task_id: i for i, task_id in enumerate(order)}
    assert idx["a"] < idx["b"]
    assert idx["a"] < idx["c"]
    assert in_degree == original
