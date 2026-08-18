"""The selector's controls must collapse. These tests fail if they stop collapsing.

A compatibility model is only evidence about geometry if removing the geometry destroys it. Two
of these tests are therefore inverted: they assert the model gets *worse*.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))

from gear_sonic.dataset_generation.compatibility_selector import (  # noqa: E402
    Candidate,
    CompatibilityModel,
    Family,
    evaluate,
    features,
    holdout,
    select,
    training_matrix,
)

SEED = 20260818


@pytest.fixture(scope="module")
def synthetic():
    from evaluate_selector import synthetic_families

    return synthetic_families(240, seed=SEED)


@pytest.fixture(scope="module")
def trained(synthetic):
    train, test = holdout(synthetic, by="family", fold=0)
    x, y = training_matrix(train)
    return CompatibilityModel(seed=SEED).fit(x, y), test


def test_the_privileged_model_beats_always_answering_nominal(trained):
    model, test = trained
    scorable = [f for f in test if f.optimum() is not None]
    baseline = sum(1 for f in scorable if f.optimum().cost == 0.0) / len(scorable)
    m = evaluate(model, test, control="none", seed=SEED)
    assert m.choice_accuracy > baseline + 0.3, (m.choice_accuracy, baseline)


def test_hiding_the_scene_destroys_the_selector(trained):
    """The no-scene control. If this passes while the model is blind, geometry is not the signal."""
    model, test = trained
    seeing = evaluate(model, test, control="none", seed=SEED).choice_accuracy
    blind = evaluate(model, test, control="no-scene", seed=SEED).choice_accuracy
    assert blind < seeing / 2.0, (blind, seeing)


def test_pairing_a_family_with_the_wrong_scene_destroys_the_selector(trained):
    model, test = trained
    real = evaluate(model, test, control="none", seed=SEED).choice_accuracy
    shuffled = evaluate(model, test, control="scene-shuffle", seed=SEED).choice_accuracy
    assert shuffled < real - 0.2, (shuffled, real)


def test_candidate_order_changes_nothing(trained):
    """The lexicographic rule is order-invariant, so this control must be a no-op exactly."""
    model, test = trained
    a = evaluate(model, test, control="none", seed=SEED)
    b = evaluate(model, test, control="candidate-order", seed=SEED)
    assert a.choice_accuracy == b.choice_accuracy
    assert a.false_safe_rate == b.false_safe_rate


def test_holdout_by_nominal_leaks_no_nominal(synthetic):
    train, test = holdout(synthetic, by="nominal", fold=1)
    seen = {c.nominal_id for f in train for c in f.candidates}
    held = {c.nominal_id for f in test for c in f.candidates}
    assert held and not (seen & held)


def test_holdout_by_family_leaks_no_family(synthetic):
    train, test = holdout(synthetic, by="family", fold=2)
    assert {f.family_id for f in train}.isdisjoint({f.family_id for f in test})


def test_the_blind_feature_vector_keeps_its_width(synthetic):
    """The no-scene control must zero the scene, not shorten the vector -- a different width
    would retrain a different model and stop being a control."""
    fam = synthetic[0]
    seeing = features(fam.scene, fam.candidates[0].profile)
    blind = features(fam.scene, fam.candidates[0].profile, blind=True)
    assert seeing.shape == blind.shape
    assert np.count_nonzero(blind[len(seeing) - 9 :]) == 0


def test_selection_prefers_the_cheaper_of_two_viable_candidates():
    """The lexicographic rule, isolated from any model error."""
    prof = {
        "peak_height_m": 1.0,
        "half_width_left_m": 0.2,
        "half_width_right_m": 0.2,
        "foot_clearance_m": 0.2,
    }
    scene = {
        "overhead_clearance_m": 2.0,
        "left_gap_m": 0.5,
        "right_gap_m": 0.5,
        "floor_height_m": 0.0,
    }
    fam = Family(
        "f",
        scene,
        (
            Candidate("adapted", "n0", prof, 1.0, True),
            Candidate("nominal", "n0", prof, 0.0, True),
        ),
    )

    class AlwaysViable(CompatibilityModel):
        def predict_proba(self, x):
            return np.ones(len(x))

    assert select(AlwaysViable(), fam).motion_id == "nominal"
    assert fam.optimum().motion_id == "nominal"


def test_an_impassable_family_is_excluded_rather_than_scored(synthetic):
    """A scene no candidate survives cannot test a selector, and counting it as a miss would
    understate the selector while counting it as a hit would overstate it."""
    prof = {
        "peak_height_m": 3.0,
        "half_width_left_m": 0.2,
        "half_width_right_m": 0.2,
        "foot_clearance_m": 0.2,
    }
    scene = {
        "overhead_clearance_m": 1.0,
        "left_gap_m": 0.5,
        "right_gap_m": 0.5,
        "floor_height_m": 0.0,
    }
    impassable = Family("dead", scene, (Candidate("m", "n0", prof, 0.0, False),))
    assert impassable.optimum() is None
    m = evaluate(CompatibilityModel(), [impassable], control="none")
    assert m.families == 0
    assert any("impassable" in n for n in m.notes)
