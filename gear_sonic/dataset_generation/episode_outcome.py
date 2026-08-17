"""Classify an episode three ways, because "not accepted" hides two different facts.

The acceptance report answers a boolean, and a capture that could not be evaluated at all
answers it the same way a motion that failed does: ``accepted=False``. The two mean
opposite things. A rejection says the controller or the reference failed; an unevaluable
capture says the *recording* failed and the motion was never assessed. Counting the second
as the first inflates the rejection rate and hides a recording bug behind what looks like a
controller problem.

This was not hypothetical. A capture spanning an environment reset reported
``accepted=False`` with an empty ``rejection_reasons`` tuple -- rejected for no stated
reason -- while carrying 1776 N of scene contact that no gate ever saw, because no gate
ran. It appeared in the batch log as ``RUN-OK``.

So an episode is one of:

* **accepted** -- evaluated, every gate passed
* **rejected** -- evaluated, at least one gate failed, and the reasons say which
* **unevaluable** -- not assessed; the capture itself is the problem

Where a reset-spanning capture holds one complete pass, that pass is recovered and
evaluated rather than discarded, and the outcome records that a split happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .trajectory_acceptance import evaluate_locomotion_trajectory
from .trajectory_segments import SegmentError, best_evaluable_payload

#: The three outcomes. Callers should switch on these rather than on a boolean.
ACCEPTED = "accepted"
REJECTED = "rejected"
UNEVALUABLE = "unevaluable"


@dataclass(frozen=True)
class EpisodeOutcome:
    """What happened to one capture, and why."""

    episode_id: str
    outcome: str
    #: Gate names that failed. Empty for accepted and for unevaluable episodes.
    rejection_reasons: tuple[str, ...] = ()
    #: Why the capture could not be assessed. Empty unless unevaluable.
    errors: tuple[str, ...] = ()
    #: True when a reset-spanning capture was split and one pass recovered.
    recovered_from_split: bool = False
    frames: int = 0
    diagnostics: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.outcome == ACCEPTED

    @property
    def evaluated(self) -> bool:
        """Whether any gate actually ran. Unevaluable episodes carry no verdict."""
        return self.outcome in (ACCEPTED, REJECTED)


def classify_episode(episode_id: str, payload: dict, *, min_frames: int = 40) -> EpisodeOutcome:
    """Evaluate one capture, recovering a single pass from a reset-spanning one."""
    recovered = False
    try:
        payload, recovered = best_evaluable_payload(payload, min_frames=min_frames)
    except (SegmentError, KeyError) as error:
        return EpisodeOutcome(
            episode_id=episode_id,
            outcome=UNEVALUABLE,
            errors=(f"{type(error).__name__}: {error}",),
        )

    report = evaluate_locomotion_trajectory(payload)
    frames = int(payload.get("total_frames", 0))

    # Errors mean gates did not run, so there is no verdict to report -- not even a
    # negative one. Checking errors before `accepted` is what keeps the two apart.
    if report.errors:
        return EpisodeOutcome(
            episode_id=episode_id,
            outcome=UNEVALUABLE,
            errors=tuple(report.errors),
            recovered_from_split=recovered,
            frames=frames,
        )

    return EpisodeOutcome(
        episode_id=episode_id,
        outcome=ACCEPTED if report.accepted else REJECTED,
        rejection_reasons=tuple(report.rejection_reasons),
        recovered_from_split=recovered,
        frames=frames,
        diagnostics=dict(report.diagnostics or {}),
    )


@dataclass
class OutcomeTally:
    """Counts over a set of episodes, keeping the three outcomes distinct."""

    accepted: int = 0
    rejected: int = 0
    unevaluable: int = 0
    recovered_from_split: int = 0

    @property
    def total(self) -> int:
        return self.accepted + self.rejected + self.unevaluable

    @property
    def evaluated(self) -> int:
        return self.accepted + self.rejected

    @property
    def acceptance_rate(self) -> float:
        """Accepted over *evaluated*, not over total.

        Dividing by total would let a recording bug depress the reported acceptance rate,
        which is the confusion this module exists to prevent.
        """
        return self.accepted / self.evaluated if self.evaluated else 0.0

    def add(self, outcome: EpisodeOutcome) -> None:
        setattr(self, outcome.outcome, getattr(self, outcome.outcome) + 1)
        if outcome.recovered_from_split:
            self.recovered_from_split += 1

    def summary(self) -> str:
        parts = [
            f"{self.accepted}/{self.evaluated} accepted ({self.acceptance_rate:.0%})",
        ]
        if self.unevaluable:
            parts.append(f"{self.unevaluable} unevaluable")
        if self.recovered_from_split:
            parts.append(f"{self.recovered_from_split} recovered by splitting a reset")
        return ", ".join(parts)
