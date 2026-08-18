"""Tests for deciding whether a reference deserves a rollout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.motion_envelope import EnvelopeSignature
from gear_sonic.dataset_generation.reference_gate import (
    MIN_FUNCTIONAL_SEPARATION_M,
    ReferenceVerdict,
    screen_reference,
)
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_G1_MJCF).exists(), reason="G1 MJCF not present"
)


def standing(frames: int = 40) -> np.ndarray:
    qpos = np.zeros((frames, 36))
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0
    return qpos


def signature(**kw) -> EnvelopeSignature:
    base = dict(
        episode_id="nominal", behaviour="walk", frames=40, duration_s=1.3,
        start_xy=(0.0, 0.0), goal_xy=(2.0, 0.0), path_length_m=2.0,
        net_displacement_m=2.0, heading_change_rad=0.0, mean_speed_mps=1.5,
        min_silhouette_peak_m=1.30, min_half_width_m=0.30, min_foot_apex_m=0.13,
    )
    base.update(kw)
    return EnvelopeSignature(**base)


def verdict(**kw) -> ReferenceVerdict:
    base = dict(
        motion_id="m", body_mode="walk", reference_semantic_valid=True,
        embodiment_feasible=True, self_collision_free=True, functionally_separated=True,
    )
    base.update(kw)
    return ReferenceVerdict(**base)


# ---- the four facts are independent -------------------------------------------------------

def test_an_unreachable_clip_is_told_to_retarget_not_rephrase():
    """crouch_walk's failure: semantically correct, physically impossible. Rewording the
    prompt cannot fix a joint that has no range."""
    v = verdict(embodiment_feasible=False, worst_joint="waist_pitch_joint",
                worst_joint_fraction=1.0)
    assert not v.worth_a_rollout
    assert "retarget" in v.diagnosis
    assert "waist_pitch_joint" in v.diagnosis


def test_a_semantically_empty_clip_is_told_to_rephrase_not_retarget():
    """step_over's failure: perfectly reachable, and the behaviour simply is not there.
    Retargeting a walk produces a walk."""
    v = verdict(reference_semantic_valid=False)
    assert not v.worth_a_rollout
    assert "rephrase" in v.diagnosis
    assert "retarget" not in v.diagnosis


def test_self_collision_blocks_a_rollout():
    assert not verdict(self_collision_free=False).worth_a_rollout


def test_an_unchecked_behaviour_is_not_a_failure():
    """"Nobody wrote the predicate" must not read the same as "the behaviour is absent"."""
    v = verdict(reference_semantic_valid=None)
    assert v.worth_a_rollout
    assert "no predicate exists" in v.diagnosis


def test_poor_separation_does_not_block_a_rollout_but_is_reported():
    """A clip too close to the nominal motion is still a valid motion. It just cannot
    anchor a family, which is a fact about the pair rather than about the clip."""
    v = verdict(functionally_separated=False, separation_m=0.01)
    assert v.worth_a_rollout
    assert "cannot anchor a family" in v.diagnosis


def test_a_clean_clip_says_so():
    assert verdict().worth_a_rollout
    assert verdict().diagnosis == "worth a rollout"


# ---- running the real checks ---------------------------------------------------------------

def test_a_standing_reference_is_reachable_and_collision_free():
    v = screen_reference(standing(), "still", "walk")
    assert v.embodiment_feasible
    assert v.self_collision_free
    assert v.signature is not None


def test_separation_is_measured_along_the_regime_axis():
    """Overhead wants the adapted motion lower; the sign must not be inverted."""
    tall = signature(min_silhouette_peak_m=1.30)
    v = screen_reference(standing(), "low", "crouch_walk", nominal=tall, regime="overhead")
    assert v.separation_m is not None
    # A standing clip is not lower than a 1.30 m nominal, so it must not read as separated.
    assert v.separation_m < MIN_FUNCTIONAL_SEPARATION_M
    assert v.functionally_separated is False


def test_separation_is_not_computed_without_a_nominal_to_compare_against():
    v = screen_reference(standing(), "x", "walk")
    assert v.functionally_separated is None
    assert v.separation_m is None


def test_an_unknown_regime_is_an_error_rather_than_a_silent_pass():
    with pytest.raises(ValueError, match="unknown regime"):
        screen_reference(standing(), "x", "walk", nominal=signature(), regime="ceiling")


def test_a_malformed_clip_fails_closed():
    """A gate that cannot run must refuse, not wave the clip through to the GPU."""
    v = screen_reference(np.zeros((10, 12)), "bad", "walk")
    assert not v.worth_a_rollout
    assert not v.embodiment_feasible
