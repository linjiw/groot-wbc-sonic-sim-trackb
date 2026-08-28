"""Both feet's contact force must be recorded under the same condition.

This cannot be unit-tested by running the recorder -- it needs Isaac -- but the bug it guards
against was purely structural and cost a full experiment to find.

``left_foot_contact_force_n`` was appended under the ``contact_force_norm is not None``
guard, and ``right_foot_contact_force_n`` under ``right_foot_ground_contact_force_w is not
None``. Whenever the ground-contact sensor was absent -- every bare-plane rollout -- the right
foot's scalar was silently dropped while the left's was kept, and every such episode failed
evaluation with "missing required field: right_foot_contact_force_n" and was classified
unevaluable. That reads as a recording failure and was really a two-line asymmetry.

So the invariant is asserted on the source: whatever guards one foot must guard the other.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RECORDER = Path(__file__).resolve().parents[2] / "gear_sonic/envs/manager_env/mdp/recorders.py"

FIELDS = ("left_foot_contact_force_n", "right_foot_contact_force_n")


def guards_for(field: str) -> list[str]:
    """The `if` conditions enclosing every append to ``data[field]``, as source text."""
    tree = ast.parse(RECORDER.read_text())
    found: list[str] = []

    def walk(node: ast.AST, conditions: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                condition = ast.unparse(child.test)
                for statement in child.body:
                    walk(statement, conditions + [condition])
                for statement in child.orelse:
                    walk(statement, conditions + [f"not ({condition})"])
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "append"
                and isinstance(child.func.value, ast.Subscript)
                and isinstance(child.func.value.slice, ast.Constant)
                and child.func.value.slice.value == field
            ):
                found.append(" and ".join(conditions))
            walk(child, conditions)

    walk(tree, [])
    return found


@pytest.mark.skipif(not RECORDER.exists(), reason="recorder module not present")
def test_both_feet_are_appended_somewhere():
    for field in FIELDS:
        assert guards_for(field), f"{field} is never recorded"


@pytest.mark.skipif(not RECORDER.exists(), reason="recorder module not present")
def test_the_two_feet_share_their_guard():
    """The asymmetry itself. If these diverge again, one foot goes missing on some scenes."""
    left, right = (guards_for(field) for field in FIELDS)
    assert sorted(left) == sorted(right), (
        "the feet are recorded under different conditions, so one can go missing while the "
        f"other is kept:\n  left  guarded by {left}\n  right guarded by {right}"
    )


@pytest.mark.skipif(not RECORDER.exists(), reason="recorder module not present")
def test_neither_foot_depends_on_the_ground_contact_sensor():
    """A contact-force scalar must not be gated on a *ground*-force array being available:
    that is exactly the coupling that made every bare-plane episode unevaluable."""
    for field in FIELDS:
        for guard in guards_for(field):
            assert (
                "ground_contact_force_w" not in guard
            ), f"{field} is gated on a ground-contact sensor: {guard}"
