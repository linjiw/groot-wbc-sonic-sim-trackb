"""Development-validation sample drawn from task geometry alone; no predicates.

The M8 development panel is development/regression evidence, and one of its six
contexts was minted by the same clearance predicate and max-min-slack ranking an
acquisition arm uses. This module draws a *separate* validation sample for the
support-preserving pilot using only the declared task domain:

* the locked generation ranges and beam families of the overhead-traversal task,
* a joint stratification over (family x underside band), both task geometry,
* a declared RNG stream fixed before any physics runs.

It issues no clearance query, reads no physical outcome, no teacher table, no
policy and no reserved layout geometry beyond exact-identity exclusion. Because
it never consults feasibility, the sample will contain contexts that no schedule
solves; those are retained and reported, never redrawn.
"""

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_pool import (
    DRAW_KEYS,
    proposal_strata,
)

# Fixed before any outcome on this sample is observed.
VALIDATION_RNG_SEED = 940910
UNDERSIDE_BANDS = ("low", "high")
CONTEXTS_PER_CELL = 1


def underside_bands(common_ranges):
    """Split the declared underside range at its midpoint: task geometry only."""
    low, high = (float(value) for value in common_ranges["underside_m"])
    if not high > low:
        raise ValueError("an ordered underside range is required")
    middle = round((low + high) / 2, 9)
    return {"low": [low, middle], "high": [middle, high]}


def validation_cells(generation, *, bands=UNDERSIDE_BANDS, per_cell=CONTEXTS_PER_CELL):
    """Enumerate the (family x underside band) cells in a declared fixed order."""
    strata = proposal_strata(generation)
    return [
        dict(stratum=stratum["name"], underside_band=band, index=index)
        for stratum in strata
        for band in bands
        for index in range(per_cell)
    ]


def draw_validation_specifications(
    generation, *, seed=VALIDATION_RNG_SEED, bands=UNDERSIDE_BANDS, per_cell=CONTEXTS_PER_CELL
):
    """Draw one beam specification set per (family x band) cell.

    Uses the same seven draw keys, the same ordered ranges and the same nine
    decimal rounding as the training-domain draw, so the validation sample and
    the acquisition reservoir inhabit one declared domain. No rejection
    sampling: every draw is retained exactly as produced.
    """
    import random

    if type(seed) is not int or type(per_cell) is not int or per_cell < 1:
        raise ValueError("an explicit integer RNG seed and positive cell size are required")
    common = generation["common_ranges"]
    band_ranges = underside_bands(common)
    if any(band not in band_ranges for band in bands):
        raise ValueError("unknown underside band requested")
    strata = {row["name"]: row for row in proposal_strata(generation)}
    rng = random.Random(seed)
    drawn = []
    for cell in validation_cells(generation, bands=bands, per_cell=per_cell):
        stratum = strata[cell["stratum"]]
        beams = []
        for station_range, length_range in stratum["beam_ranges"]:
            ranges = [
                station_range,
                band_ranges[cell["underside_band"]],
                length_range,
                common["width_m"],
                common["thickness_m"],
                common["lateral_offset_m"],
                common["yaw_offset_rad"],
            ]
            beams.append(
                dict(
                    zip(
                        DRAW_KEYS,
                        [round(rng.uniform(*bounds), 9) for bounds in ranges],
                        strict=True,
                    )
                )
            )
        layout_id = (
            f"support_validation_{cell['stratum']}_{cell['underside_band']}_{cell['index']:02d}"
        )
        drawn.append(
            dict(
                layout_id=layout_id,
                base_layout_id=layout_id,
                stratum=cell["stratum"],
                underside_band=cell["underside_band"],
                cell_index=cell["index"],
                underside_band_range_m=band_ranges[cell["underside_band"]],
                beam_specifications=beams,
                rng_seed=seed,
                draw_order=list(DRAW_KEYS),
                rejection_sampling=False,
                clearance_queries=0,
                physical_outcomes_used=False,
                selected_by_policy_failure=False,
            )
        )
    return drawn


def ancestry_groups(layouts):
    """Group layouts by base layout so descendants never split across a split.

    Every drawn layout is an independent geometry draw, so each is its own base
    layout here; the grouping is emitted anyway so a later perturbation variant
    cannot be reported as an independent validation item.
    """
    groups = {}
    for layout in layouts:
        groups.setdefault(layout["base_layout_id"], []).append(layout["layout_id"])
    return {key: sorted(value) for key, value in sorted(groups.items())}


__all__ = [
    "CONTEXTS_PER_CELL",
    "UNDERSIDE_BANDS",
    "VALIDATION_RNG_SEED",
    "ancestry_groups",
    "draw_validation_specifications",
    "underside_bands",
    "validation_cells",
]
