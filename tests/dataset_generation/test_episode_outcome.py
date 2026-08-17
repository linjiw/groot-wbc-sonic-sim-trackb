"""Tests for the three-way episode outcome."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.dataset_generation import episode_outcome as module
from gear_sonic.dataset_generation.episode_outcome import (
    ACCEPTED,
    REJECTED,
    UNEVALUABLE,
    EpisodeOutcome,
    OutcomeTally,
    classify_episode,
)


def payload(times: list[float]) -> dict:
    """A minimal capture that still satisfies the real payload contract.

    `reference_g1_qpos` is not optional: the gate policy measures drift against it, and an
    episode without a reference genuinely cannot be assessed for stability.
    """
    frames = len(times)
    return {
        "motion_time_s": np.asarray(times, dtype=np.float64),
        "root_pos_w": np.zeros((frames, 3)),
        "reference_g1_qpos": np.zeros((frames, 36)),
        "total_frames": frames,
        "fps": 50.0,
    }


def ramp(count: int, start: float = 0.0) -> list[float]:
    return [start + i * 0.02 for i in range(count)]


def fake_report(*, accepted=True, reasons=(), errors=()):
    return SimpleNamespace(
        accepted=accepted, rejection_reasons=tuple(reasons), errors=tuple(errors),
        diagnostics={},
    )


def test_all_gates_passing_is_accepted(monkeypatch):
    monkeypatch.setattr(module, "evaluate_locomotion_trajectory", lambda p: fake_report())
    outcome = classify_episode("e", payload(ramp(80)))
    assert outcome.outcome == ACCEPTED
    assert outcome.accepted and outcome.evaluated
    assert outcome.frames == 80


def test_a_failing_gate_is_rejected_with_its_reason(monkeypatch):
    monkeypatch.setattr(
        module,
        "evaluate_locomotion_trajectory",
        lambda p: fake_report(accepted=False, reasons=("excessive_self_contact",)),
    )
    outcome = classify_episode("e", payload(ramp(80)))
    assert outcome.outcome == REJECTED
    assert outcome.rejection_reasons == ("excessive_self_contact",)
    assert outcome.evaluated and not outcome.accepted


def test_an_evaluation_error_is_unevaluable_not_rejected(monkeypatch):
    """This is the confusion the module exists to prevent.

    An unevaluable capture reported accepted=False with an empty reasons tuple, while
    carrying 1776 N of scene contact that no gate ever saw.
    """
    monkeypatch.setattr(
        module,
        "evaluate_locomotion_trajectory",
        lambda p: fake_report(accepted=False, errors=("frame count mismatch",)),
    )
    outcome = classify_episode("e", payload(ramp(80)))
    assert outcome.outcome == UNEVALUABLE
    assert not outcome.evaluated
    assert outcome.errors == ("frame count mismatch",)
    assert outcome.rejection_reasons == ()


def test_errors_win_even_when_the_report_claims_acceptance(monkeypatch):
    """No gate ran, so there is no verdict to report -- not even a positive one."""
    monkeypatch.setattr(
        module,
        "evaluate_locomotion_trajectory",
        lambda p: fake_report(accepted=True, errors=("no gates ran",)),
    )
    assert classify_episode("e", payload(ramp(80))).outcome == UNEVALUABLE


def test_a_reset_spanning_capture_is_split_and_evaluated(monkeypatch):
    monkeypatch.setattr(module, "evaluate_locomotion_trajectory", lambda p: fake_report())
    outcome = classify_episode("e", payload(ramp(50) + ramp(90)))
    assert outcome.outcome == ACCEPTED
    assert outcome.recovered_from_split is True
    assert outcome.frames == 90


def test_a_single_pass_is_not_marked_as_recovered(monkeypatch):
    monkeypatch.setattr(module, "evaluate_locomotion_trajectory", lambda p: fake_report())
    assert classify_episode("e", payload(ramp(80))).recovered_from_split is False


def test_a_capture_with_no_usable_pass_is_unevaluable(monkeypatch):
    monkeypatch.setattr(module, "evaluate_locomotion_trajectory", lambda p: fake_report())
    outcome = classify_episode("e", payload(ramp(10) + ramp(12)))
    assert outcome.outcome == UNEVALUABLE
    assert "SegmentError" in outcome.errors[0]


def test_a_malformed_payload_is_unevaluable_rather_than_an_exception(monkeypatch):
    monkeypatch.setattr(module, "evaluate_locomotion_trajectory", lambda p: fake_report())
    outcome = classify_episode("e", {"total_frames": 5})
    assert outcome.outcome == UNEVALUABLE


def test_tally_keeps_the_three_outcomes_apart():
    tally = OutcomeTally()
    for outcome in (ACCEPTED, ACCEPTED, REJECTED, UNEVALUABLE):
        tally.add(EpisodeOutcome(episode_id="e", outcome=outcome))
    assert (tally.accepted, tally.rejected, tally.unevaluable) == (2, 1, 1)
    assert tally.total == 4
    assert tally.evaluated == 3


def test_acceptance_rate_divides_by_evaluated_not_total():
    """A recording bug must not depress the reported acceptance rate."""
    tally = OutcomeTally()
    tally.add(EpisodeOutcome("a", ACCEPTED))
    tally.add(EpisodeOutcome("b", REJECTED))
    for index in range(8):
        tally.add(EpisodeOutcome(f"u{index}", UNEVALUABLE))
    assert tally.acceptance_rate == pytest.approx(0.5)
    assert tally.total == 10


def test_acceptance_rate_of_nothing_evaluated_is_zero_not_a_crash():
    tally = OutcomeTally()
    tally.add(EpisodeOutcome("u", UNEVALUABLE))
    assert tally.acceptance_rate == 0.0


def test_tally_counts_recoveries():
    tally = OutcomeTally()
    tally.add(EpisodeOutcome("a", ACCEPTED, recovered_from_split=True))
    tally.add(EpisodeOutcome("b", ACCEPTED))
    assert tally.recovered_from_split == 1
    assert "recovered by splitting" in tally.summary()


def test_summary_omits_clean_categories():
    tally = OutcomeTally()
    tally.add(EpisodeOutcome("a", ACCEPTED))
    summary = tally.summary()
    assert "unevaluable" not in summary
    assert "recovered" not in summary


def test_a_capture_that_resets_constantly_is_unevaluable_not_a_diverse_episode(monkeypatch):
    """A real capture reset 22 times in 199 frames, every 0.18 s.

    Stitched, it read as 10.16 m of path with 0.25 m of displacement, -106.9 rad of heading
    change and a tortuosity of 41 -- numbers that were reported as diversity until the
    exclusion was added. No segment is long enough to evaluate, so the outcome must be
    unevaluable and every downstream statistic must skip it.
    """
    times: list[float] = []
    for _ in range(22):
        times.extend(ramp(9))
    outcome = classify_episode("thrashing", payload(times))
    assert outcome.outcome == UNEVALUABLE
    assert not outcome.evaluated
    assert "SegmentError" in outcome.errors[0]


def test_evaluated_is_the_flag_downstream_statistics_must_gate_on():
    """`accepted` alone is not enough: a rejected episode has a valid trajectory to
    measure, while an unevaluable one does not."""
    rejected = EpisodeOutcome("r", REJECTED, rejection_reasons=("path_error",))
    unevaluable = EpisodeOutcome("u", UNEVALUABLE, errors=("no gates ran",))
    assert rejected.evaluated and not rejected.accepted
    assert not unevaluable.evaluated
