"""Source-level claims cannot be rescued by pooling or ceiling ties."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_fresh_audit import predicates  # noqa: E402
from motion2scene_fresh_sources import SOURCE_IDS  # noqa: E402


def rows():
    return [
        dict(
            arm=arm,
            budget=0,
            carrier_seed=source,
            phase="unseen",
            split="fresh_audit",
            seed=8421,
            valid=8 if arm != "uniform_pattern" else 1,
            proposals=8,
            target_failures=0,
            neutral_failures=0,
            rescued_gradient=0,
            lost_gradient=0,
        )
        for source in SOURCE_IDS
        for arm in ["pattern", "gradient", "uniform_pattern"]
    ]


def test_ceiling_ties_fail_strict_gradient_gain():
    _, outcome = predicates(rows())
    assert outcome["learned_pattern_strictly_better_on_every_source"]
    assert outcome["no_paired_gradient_loss"]
    assert not outcome["no_source_loss_and_total_gain_over_gradient"]


def test_one_source_tie_fails_even_when_pooled_uniform_gain_large():
    data = rows()
    next(r for r in data if r["arm"] == "uniform_pattern")["valid"] = 8
    _, outcome = predicates(data)
    assert sum(outcome["pattern_minus_uniform_by_source"].values()) > 0
    assert not outcome["learned_pattern_strictly_better_on_every_source"]
