"""Which candidates a hallucinated scene has to rule out, at a regret tolerance of epsilon.

The project's inverse set is `S_{eps,delta}(tau) = {S : Feasible, Regret <= eps, Necessity >= delta}`.
Necessity is charged against the *cheapest* candidate; regret against any candidate cheaper than the
executed motion by more than `eps`. Reading regret as "every cheaper candidate must be struck by the
full margin" instead collapses the achievable margin from 38 mm to 3 mm at `crouch_070`, because a
70 mm crouch has a 55 mm crouch sitting 15 mm away from it -- so this boundary is load-bearing and
is pinned here.
"""

import torch

from scripts.research.hallucination.train_lflh_sdf import set_regret_rivals

#: The real ladder from `lflh_candidates.json`, in label order.
LABELS = [
    "nominal",
    "crouch_040",
    "crouch_055",
    "crouch_070",
    "tuck_left_040",
    "tuck_left_070",
    "tuck_right_040",
    "tuck_right_070",
]
COSTS = [0.0, 0.5712866, 0.7855250, 0.9999171, 0.5714286, 1.0, 0.5714286, 1.0]


def _clip(target: str) -> dict:
    return {
        "costs": torch.tensor(COSTS, dtype=torch.float32),
        "observed": LABELS.index(target),
        "cheapest": 0,
    }


def _names(clip: dict) -> set[str]:
    return {LABELS[index] for index in clip["regret_rivals"]}


def test_the_cheapest_candidate_is_never_a_regret_rival():
    """It is charged to necessity instead; double-counting it would demand the impossible."""
    for epsilon in (0.0, 0.1, 0.25, 0.5):
        clip = _clip("crouch_070")
        set_regret_rivals([clip], epsilon)
        assert "nominal" not in _names(clip)
        assert clip["observed"] not in clip["regret_rivals"]


def test_the_neighbouring_rung_is_tolerated_once_epsilon_exceeds_its_cost_gap():
    """`crouch_055` is 0.214 cheaper than `crouch_070`, so it flips between 0.1 and 0.25."""
    tight, loose = _clip("crouch_070"), _clip("crouch_070")
    set_regret_rivals([tight], 0.1)
    set_regret_rivals([loose], 0.25)
    assert "crouch_055" in _names(tight)
    assert "crouch_055" not in _names(loose)
    # The shallower rung is 0.429 cheaper and survives both tolerances.
    assert "crouch_040" in _names(tight)
    assert "crouch_040" in _names(loose)


def test_more_expensive_candidates_are_never_rivals():
    """A costlier edit needs no excluding: the robot would not have preferred it anyway."""
    clip = _clip("crouch_040")
    set_regret_rivals([clip], 0.0)
    assert not _names(clip) & {"crouch_055", "crouch_070", "tuck_left_070", "tuck_right_070"}


def test_a_shallow_target_has_no_regret_rivals_at_all():
    """`crouch_040` is the cheapest edit, so only necessity against the nominal binds it."""
    clip = _clip("crouch_040")
    set_regret_rivals([clip], 0.0)
    assert _names(clip) == set()


def test_the_rival_set_grows_monotonically_as_epsilon_shrinks():
    previous: set[str] = set()
    for epsilon in (0.5, 0.25, 0.1, 0.0):
        clip = _clip("crouch_070")
        set_regret_rivals([clip], epsilon)
        assert previous <= _names(clip), "a smaller tolerance must never forgive a rival"
        previous = _names(clip)
