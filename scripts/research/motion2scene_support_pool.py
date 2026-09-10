"""Support-preserving acquisition queues for the post-M8 pilot; geometry only.

Three nested proposal controls over one fresh common reservoir drawn from the
declared task domain:

``support_strict``
    The unchanged executed-contrast rule. Delegated verbatim to
    :func:`motion2scene_acquisition_pool.acquisition_queues` so the strict arm
    cannot drift from the rule the M8 study measured.
``support_broad``
    Every validly drawn candidate in its original draw order. Neither the
    positive 81-offset robustness bar nor the negative nominal-interference test
    is applied, so this queue has support outside both screens.
``support_mixture``
    A fixed finite alternation of the two channels above, epsilon = 1/2,
    committed before any new physical outcome exists.

Nothing here reads a physical outcome, a sensor trace, a learner, or a reserved
layout. Candidate metadata never becomes a policy input.
"""

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_pool import (
    DRAW_KEYS,
    SEEDS,
    acquisition_queues,
    draw_pool,
    proposal_strata,
)

PILOT_ARMS = ("support_strict", "support_broad", "support_mixture")

# The M8 candidate pool drew 256 rounds per stratum. Restricting the pilot's
# selection to later rounds keeps the reservoir fresh while reusing the frozen
# draw stream, so the shared prefix of the pool reproduces bit-for-bit.
M8_POOL_ROUNDS_PER_STRATUM = 256

# Committed before any new outcome: half of the mixture's new encounters are
# proposed by the strict channel and half by the exploration channel.
MIXTURE_STRICT_FRACTION = 0.5

# Slot -> channel for the mixture arm, applied to the shared stratum schedule.
# Four new encounters give exactly two guided and two exploratory proposals.
MIXTURE_CHANNELS = ("support_strict", "support_broad", "support_strict", "support_broad")

STRICT_SOURCE_ARM = "analytic_contrast"


def fresh_candidates(generation, seed, total_rounds_per_stratum, option_ids, *, prior_rounds):
    """Draw the locked domain deeper and keep only never-screened rounds.

    The returned candidates carry the same identifiers, strata, branch orders and
    assigned positives they would have had in the original stream, because the
    frozen :func:`draw_pool` produces the whole prefix and only the tail is kept.
    """
    if (
        type(total_rounds_per_stratum) is not int
        or type(prior_rounds) is not int
        or prior_rounds < 0
        or total_rounds_per_stratum <= prior_rounds
    ):
        raise ValueError("the fresh reservoir must extend strictly beyond the screened rounds")
    drawn = draw_pool(generation, seed, total_rounds_per_stratum, option_ids)
    prefix = [row for row in drawn if row["stratum_round"] < prior_rounds]
    fresh = [row for row in drawn if row["stratum_round"] >= prior_rounds]
    expected = prior_rounds * len(proposal_strata(generation))
    if len(prefix) != expected or len(prefix) + len(fresh) != len(drawn):
        raise ValueError("the frozen draw stream did not reproduce the screened prefix")
    return dict(drawn=drawn, screened_prefix=prefix, fresh=fresh)


def gate_membership(minimum, offsets, negative, *, offset_count=81, margin_m=0.01):
    """Record which of the two original screens a candidate row would pass."""
    robust_positive = [
        bool(count == offset_count and value >= margin_m)
        for count, value in zip(offsets, minimum, strict=True)
    ]
    interfering_negative = [bool(value <= -margin_m) for value in negative]
    return dict(
        inside_positive_screen=any(robust_positive),
        inside_negative_screen=any(interfering_negative),
        robust_positive_option_count=sum(robust_positive),
        interfering_negative_option_count=sum(interfering_negative),
    )


def support_queues(
    candidates,
    option_ids,
    evaluated_positive_min,
    evaluated_offsets,
    nominal_inner,
    *,
    offset_count=81,
    margin_m=0.01,
):
    """Build the three nested proposal queues over one shared candidate list.

    ``support_strict`` is the unchanged contrast rule and ranking.
    ``support_broad`` keeps every candidate in draw order with no geometric gate.
    ``support_mixture`` is not a ranked queue: it is realised by the plan's fixed
    slot-to-channel schedule, so it is returned as the two source channels plus
    the committed schedule rather than as a third ordering.
    """
    original = acquisition_queues(
        candidates,
        option_ids,
        evaluated_positive_min,
        evaluated_offsets,
        nominal_inner,
        offset_count=offset_count,
        margin_m=margin_m,
    )
    membership_by_id = {
        candidate["candidate_id"]: gate_membership(
            evaluated_positive_min[index],
            evaluated_offsets[index],
            nominal_inner[index],
            offset_count=offset_count,
            margin_m=margin_m,
        )
        for index, candidate in enumerate(candidates)
    }
    # The frozen entry() builder predates the pilot and records no screen
    # membership. Attach it explicitly so a strict row is never read as
    # "outside the screen" merely because the field was absent.
    strict = [
        dict(row, proposal_channel="support_strict", **membership_by_id[row["candidate_id"]])
        for row in original[STRICT_SOURCE_ARM]
    ]
    by_id = {row["candidate_id"]: row for row in strict}
    broad = []
    for index, candidate in enumerate(candidates):
        membership = membership_by_id[candidate["candidate_id"]]
        ranked = by_id.get(candidate["candidate_id"])
        broad.append(
            dict(
                candidate_id=candidate["candidate_id"],
                candidate_index=candidate["candidate_index"],
                stratum=candidate["stratum"],
                proposal_channel="support_broad",
                # No positive is required, so none is asserted. The seven-branch
                # order is already frozen per candidate and is what gets executed.
                positive_option_id=None,
                negative_option_id=None,
                geometric_screen_passed=False,
                geometric_slack_m=None if ranked is None else ranked["geometric_slack_m"],
                physical_measurement_admitted=None,
                physical_passage=None,
                observation_availability=None,
                verified_learner_gap=None,
                curriculum_ready=False,
                future_physics_seed=candidate["future_physics_seed"],
                **membership,
            )
        )
    strict_ids = {row["candidate_id"] for row in strict}
    for row in broad:
        row["also_eligible_for_strict"] = row["candidate_id"] in strict_ids
        if row["also_eligible_for_strict"] != (
            row["inside_positive_screen"] and row["inside_negative_screen"]
        ):
            raise ValueError("strict eligibility disagrees with the recorded screen membership")
    return dict(
        support_strict=strict,
        support_broad=broad,
        mixture_schedule=list(MIXTURE_CHANNELS),
        mixture_strict_fraction=MIXTURE_STRICT_FRACTION,
        offset_count=offset_count,
        margin_m=margin_m,
    )


def stratum_schedule(prefix_strata, additions, strata_order):
    """Balance new encounters over strata, shared identically by all three arms.

    Reproduces the balancing intent of the M8 plan's ``balanced_extension`` while
    fixing the stratum sequence *before* any arm consults its own queue, so the
    three arms differ only in which channel proposes each slot.
    """
    if (
        type(additions) is not int
        or additions < 1
        or not prefix_strata
        or any(name not in strata_order for name in prefix_strata)
    ):
        raise ValueError("a nonempty prefix composition and positive addition count are required")
    counts = {name: 0 for name in strata_order}
    for name in prefix_strata:
        counts[name] += 1
    schedule = []
    for _ in range(additions):
        chosen = min(strata_order, key=lambda name: (counts[name], strata_order.index(name)))
        schedule.append(chosen)
        counts[chosen] += 1
    return schedule


def select_pilot_encounters(queues, schedule, *, mixture_channels=MIXTURE_CHANNELS):
    """Assign one candidate per arm per slot from the committed channel.

    Returns the realised finite sampling schedule. A slot whose channel has no
    remaining candidate in the scheduled stratum is retained as a shortfall; it
    is never refilled from the other channel and no margin is relaxed.
    """
    if len(mixture_channels) < len(schedule):
        raise ValueError("the committed mixture schedule is shorter than the addition budget")
    channels = {name: queues[name] for name in ("support_strict", "support_broad")}
    selected = {arm: [] for arm in PILOT_ARMS}
    used = {arm: set() for arm in PILOT_ARMS}
    shortfalls = []
    for slot, stratum in enumerate(schedule):
        assignment = {
            "support_strict": "support_strict",
            "support_broad": "support_broad",
            "support_mixture": mixture_channels[slot],
        }
        for arm in PILOT_ARMS:
            channel = assignment[arm]
            rank, row = next(
                (
                    (i, r)
                    for i, r in enumerate(channels[channel])
                    if r["stratum"] == stratum and r["candidate_id"] not in used[arm]
                ),
                (None, None),
            )
            if row is None:
                shortfalls.append(dict(arm=arm, slot=slot, stratum=stratum, channel=channel))
                continue
            selected[arm].append(
                dict(row, slot=slot, original_queue_rank=rank, proposal_channel=channel)
            )
            used[arm].add(row["candidate_id"])
    return dict(
        selected=selected,
        schedule=schedule,
        mixture_channels=list(mixture_channels[: len(schedule)]),
        shortfalls=shortfalls,
        physical_outcomes_consulted=False,
        realised_mixture_strict_fraction=(
            None
            if not selected["support_mixture"]
            else sum(
                row["proposal_channel"] == "support_strict" for row in selected["support_mixture"]
            )
            / len(selected["support_mixture"])
        ),
    )


__all__ = [
    "DRAW_KEYS",
    "MIXTURE_CHANNELS",
    "MIXTURE_STRICT_FRACTION",
    "M8_POOL_ROUNDS_PER_STRATUM",
    "PILOT_ARMS",
    "SEEDS",
    "STRICT_SOURCE_ARM",
    "fresh_candidates",
    "gate_membership",
    "proposal_strata",
    "select_pilot_encounters",
    "stratum_schedule",
    "support_queues",
]
