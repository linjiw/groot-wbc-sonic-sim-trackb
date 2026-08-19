#!/usr/bin/env python3
"""Adjudicate the hygiene x sampler matched-compute matrix -- written before the matrix is run.

WHY this file exists before any outcome does: this repo records the decision rule ahead of the
measurement (``docs/prediction_register.md``), so that a rule cannot be reverse-fitted to a result
that already happened. Nothing here reads a training log; ``--synthetic`` exercises every decision
branch on fabricated numbers so the logic is testable while the GPUs are still idle.

THE MATRIX. Single-factor by construction, because SONIC already ships a 10% uniform sampling floor
and a two-factor change would confound the bank with the sampler:

  bank axis   (sampler pinned at the SONIC default ``uniform_sampling_rate=0.1``)
    A_raw        the corpus as retargeted                                   -- baseline
    B_pruned     clips with ``infeasible_frac > 0.10`` removed              -- N shrinks
    C_repaired   those same clips replaced by their contact-projected repair -- N preserved
  sampler axis (bank pinned at raw)
    D_raw_uniform  ``uniform_sampling_rate = 1.0``; the adaptive path stays ENABLED, because
                   ``enable=false`` would also change the start-time distribution and confound
                   three things at once
    E_raw_cap      ``max_prob_per_motion`` at 5x fair share; the config-only remedy that already
                   exists in ``motion_lib_base.py`` but is ``None`` in every shipped yaml

THE CONFOUND THIS FILE IS BUILT AROUND. B trains on FEWER clips, so at matched compute it takes more
gradient steps per surviving clip. "It converged faster because there was less data" has to be
separable from "hygiene helped". The separator is a CONCENTRATION SIGNATURE, ported from the sibling
project's sealed pre-registration (``/data/robotixx/climb/tools/analyze_ehyg.py`` lines 52-66):

    order the feasible eval clips ascending by the BASELINE arm's offset-mean survival
    worst      = order[: max(1, n // 10)]
    best_half  = order[n // 2 :]
    easy       = clips whose baseline offset-mean survival >= 0.95
    the hygiene claim passes iff  delta_worst >= 2 * delta_best_half  AND  |delta_easy| <= 0.02

A pure data-volume effect lifts every stratum by roughly the same amount, so it fails the ratio test
and moves the easy stratum off its ceiling. Two additions to the sealed rule are made explicit here
rather than hidden: a MATERIALITY FLOOR on ``delta_worst`` (the literal ratio test is vacuously true
when both deltas are ~0, and perversely true when both are negative), and a split-half robustness
column that re-ranks the strata on a disjoint set of start offsets so that regression-to-the-mean in
the selection step can be seen rather than assumed away. The sealed sub-results are still reported
individually so a reviewer can apply the original rule unmodified.

C_repaired is the arm that sidesteps the confound outright: N is preserved, so a lift there is not
explicable by fewer clips. That is stated in the output, not left to the reader.

ENDPOINTS, in the priority order of the design:
  1. stratified survival at fixed start offsets on a RAW held-out eval set (the eval set is never
     repaired -- repairing what you are graded on is a self-serving endpoint, and this script
     refuses to run if an arm's eval bank says otherwise);
  2. the exposure ledger -- normalized Shannon entropy of the sampling distribution, top-1 clip
     share, and the fraction of sampling mass spent on flagged clips ("wasted exposure");
  3. torque health -- actuator saturation and contact-force discontinuity at tracking transitions.

Only endpoint 1 is adjudicated. 2 and 3 are reported as mechanism evidence, because neither has a
pre-registered threshold.

INFERENCE. Every arm is n=1 seed. There is therefore NO seed-level inference in this file and no
claim about run-to-run reproducibility; the permutation tests are paired over CLIPS and describe
clip-level variability conditional on the single training run that produced each arm. The sign-flip
permutation is Monte Carlo with nperm=5000 under ``numpy.random.default_rng(0)`` -- the seed is
pinned and echoed into every output. ``scripts/research/paired_stats.py`` holds the exact (2^n
enumeration) seed-level counterpart used for multi-seed comparisons; it is deliberately not reused
here because n=1 seed admits no seed-level test at all.

Usage::

    python scripts/research/analyze_hygiene_matrix.py --synthetic

    python scripts/research/analyze_hygiene_matrix.py \\
        --strat A_raw=eval/A.csv B_pruned=eval/B.csv C_repaired=eval/C.csv \\
                D_raw_uniform=eval/D.csv E_raw_cap=eval/E.csv \\
        --eval-screen screens/eval_raw.csv \\
        --exposure A_raw=exp/A.csv ... --torque A_raw=tq/A.csv ... \\
        --screen all=screens/train_raw.csv --screen C_repaired=screens/train_repaired.csv \\
        --ground-clips tiers/ground_contact.txt \\
        --out-json reports/hygiene_matrix.json --out-md reports/hygiene_matrix.md
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = 1

DEFAULT_NPERM = 5000
DEFAULT_SEED = 0

#: A clip is "flagged" when the dynamic screen cannot support the wrench its reference demands on
#: more than this fraction of frames. Same threshold the pruned/repaired banks are built from.
FLAG_INFEASIBLE_FRAC = 0.10
#: "easy" stratum membership, on the BASELINE arm's offset-mean survival.
EASY_SURVIVAL = 0.95
#: how far the easy stratum may drift before the result stops looking like targeted repair.
EASY_DRIFT_TOL = 0.02
#: the worst decile must move by at least this much for the ratio test to mean anything.
MIN_WORST_DECILE_EFFECT = 0.02
#: sealed ratio: worst-decile gain must be at least this multiple of the best-half gain.
CONCENTRATION_RATIO = 2.0

BASELINE_ARM = "A_raw"
REQUIRED_ARMS: tuple[str, ...] = ("A_raw", "B_pruned", "C_repaired", "D_raw_uniform", "E_raw_cap")

ARM_ROLES: dict[str, dict[str, str]] = {
    "A_raw": {
        "axis": "baseline",
        "confound_status": "baseline",
        "note": "corpus as retargeted; sampler at the shipped default uniform_sampling_rate=0.1",
    },
    "B_pruned": {
        "axis": "bank",
        "confound_status": "volume-confounded",
        "note": (
            "N shrinks, so at matched compute this arm takes more gradient steps per surviving "
            "clip; a lift here is a hygiene result only if the concentration signature holds"
        ),
    },
    "C_repaired": {
        "axis": "bank",
        "confound_status": "volume-controlled",
        "note": (
            "N is preserved -- flagged clips are replaced by their contact-projected repair, not "
            "removed -- so a lift here cannot be explained by training on less data"
        ),
    },
    "D_raw_uniform": {
        "axis": "sampler",
        "confound_status": "bank-fixed",
        "note": (
            "bank fixed at raw; uniform_sampling_rate=1.0 with the adaptive path still enabled so "
            "the start-time distribution is unchanged"
        ),
    },
    "E_raw_cap": {
        "axis": "sampler",
        "confound_status": "bank-fixed",
        "note": "bank fixed at raw; max_prob_per_motion at 5x fair share",
    },
}

INFERENCE_NOTE = (
    "n=1 seed per arm: there is NO seed-level inference in this report. The permutation tests are "
    "paired over CLIPS and describe clip-level variability conditional on one training run per "
    "arm; they say nothing about run-to-run reproducibility. Treat every verdict as a single-run "
    "screen."
)

CAVEATS = (
    "Selection bias: the worst decile is chosen BY the baseline arm, so regression to the mean "
    "inflates delta_worst even under a null. The split-half column re-ranks on a disjoint set of "
    "start offsets; the gap between the sealed and split-half delta_worst is the visible part of "
    "that bias.",
    "The sign-flip null assumes per-clip deltas are symmetric about zero under H0. Survival is "
    "bounded in [0, 1], so clips at the ceiling can only move down and clips at the floor can only "
    "move up; the easy stratum is the most affected and its p-value is the least trustworthy.",
    "Five arms are compared against one baseline with no multiplicity correction. Each contrast is "
    "pre-registered and read on its own; do not read 'at least one arm passed' as a single test.",
    "Normalized Shannon entropy is comparable across arms with different bank sizes only up to the "
    "normalization; B_pruned's ledger is over a smaller support by construction, and its wasted "
    "exposure is ~0 by construction rather than by learning.",
    "Torque health and the exposure ledger have no pre-registered thresholds. They are mechanism "
    "evidence for whichever way the survival endpoint lands, never a substitute for it.",
)


class AnalysisInputError(RuntimeError):
    """An input is missing, ragged, or inconsistent.

    Raised instead of dropping a column: a matrix silently missing an arm still prints a tidy
    table, and that table is wrong in a way nobody notices.
    """


# --------------------------------------------------------------------------- readers


_CLIP_COLUMNS = ("motion_key", "clip", "motion", "key")
_SURVIVAL_COLUMNS = ("survival", "survival_rate")
_OFFSET_COLUMNS = ("offset_s", "offset", "start_offset_s")
_PROB_COLUMNS = ("prob", "sampling_prob", "probability", "mass", "p")
_COUNT_COLUMNS = ("count", "n_samples", "samples")
_SATURATION_COLUMNS = ("saturation_frac", "sat_frac", "torque_saturation_frac")
_CONTACT_JUMP_COLUMNS = ("contact_force_jump_N", "contact_jump_N", "contact_discontinuity_N")
_INFEASIBLE_COLUMNS = ("infeasible_frac",)


def _read_csv_rows(source: str | Path) -> list[dict[str, str]]:
    """Read CSV rows from a path, or from inline CSV text (handy for tests)."""
    text = str(source)
    if "\n" in text:
        with io.StringIO(text) as handle:
            return list(csv.DictReader(handle))
    path = Path(source)
    if not path.is_file():
        raise AnalysisInputError(f"missing input file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AnalysisInputError(f"empty CSV: {path}")
    return rows


def _column(row: dict[str, str], candidates: Sequence[str], *, where: str) -> str:
    for name in candidates:
        value = row.get(name)
        if value not in (None, ""):
            return value
    raise AnalysisInputError(
        f"{where}: none of the columns {tuple(candidates)} are present/populated; saw {sorted(row)}"
    )


@dataclass(frozen=True)
class StratifiedEval:
    """One arm's stratified-start eval table: survival per (clip, start offset)."""

    source: str
    by_clip: dict[str, dict[float, float]]
    offsets: tuple[float, ...]
    eval_bank: str | None

    def clips(self) -> set[str]:
        return set(self.by_clip)

    def offset_mean(self, offsets: Iterable[float] | None = None) -> dict[str, float]:
        keep = None if offsets is None else set(offsets)
        out: dict[str, float] = {}
        for clip, per_offset in self.by_clip.items():
            values = [v for o, v in per_offset.items() if keep is None or o in keep]
            if not values:
                raise AnalysisInputError(f"{self.source}: clip {clip} has no survival at {keep}")
            out[clip] = float(np.mean(values))
        return out


def read_stratified(source: str | Path) -> StratifiedEval:
    """Parse a stratified-start eval CSV: clip, offset_s, survival (+ optional eval_bank tag)."""
    rows = _read_csv_rows(source)
    where = str(source)[:120]
    by_clip: dict[str, dict[float, float]] = {}
    banks: set[str] = set()
    for row in rows:
        raw_offset = _column(row, _OFFSET_COLUMNS, where=where)
        if str(raw_offset).strip().lower() in {"mean", "all", "aggregate"}:
            continue  # summary row emitted by the eval writer; recomputed here from the raw grid
        clip = _column(row, _CLIP_COLUMNS, where=where)
        offset = float(raw_offset)
        survival = float(_column(row, _SURVIVAL_COLUMNS, where=where))
        per_offset = by_clip.setdefault(clip, {})
        if offset in per_offset:
            raise AnalysisInputError(f"{where}: duplicate row for clip {clip} at offset {offset}")
        per_offset[offset] = survival
        bank = row.get("eval_bank") or row.get("eval_set")
        if bank:
            banks.add(bank)
    if not by_clip:
        raise AnalysisInputError(f"{where}: no per-offset rows (only summary rows?)")
    if len(banks) > 1:
        raise AnalysisInputError(f"{where}: rows disagree about the eval bank: {sorted(banks)}")
    offsets = tuple(sorted({o for per in by_clip.values() for o in per}))
    ragged = sorted(c for c, per in by_clip.items() if len(per) != len(offsets))
    if ragged:
        raise AnalysisInputError(
            f"{where}: {len(ragged)} clips are missing start offsets (grid is {offsets}); "
            f"first few: {ragged[:5]}. A ragged grid silently reweights the offset mean."
        )
    return StratifiedEval(
        source=where,
        by_clip=by_clip,
        offsets=offsets,
        eval_bank=next(iter(banks)) if banks else None,
    )


def read_screen(source: str | Path) -> dict[str, float]:
    """Parse a hygiene screen CSV into ``{motion_key: infeasible_frac}``."""
    rows = _read_csv_rows(source)
    where = str(source)[:120]
    out: dict[str, float] = {}
    for row in rows:
        clip = _column(row, _CLIP_COLUMNS, where=where)
        out[clip] = float(_column(row, _INFEASIBLE_COLUMNS, where=where))
    if not out:
        raise AnalysisInputError(f"{where}: screen CSV has no rows")
    return out


def read_exposure(source: str | Path) -> dict[str, float]:
    """Parse an exposure ledger into a normalized ``{motion_key: sampling probability}``."""
    rows = _read_csv_rows(source)
    where = str(source)[:120]
    raw: dict[str, float] = {}
    for row in rows:
        clip = _column(row, _CLIP_COLUMNS, where=where)
        try:
            value = float(_column(row, _PROB_COLUMNS, where=where))
        except AnalysisInputError:
            value = float(_column(row, _COUNT_COLUMNS, where=where))
        if value < 0:
            raise AnalysisInputError(f"{where}: negative sampling mass for {clip}: {value}")
        raw[clip] = raw.get(clip, 0.0) + value
    total = sum(raw.values())
    if total <= 0:
        raise AnalysisInputError(f"{where}: exposure ledger sums to {total}")
    return {clip: value / total for clip, value in raw.items()}


def read_torque(source: str | Path) -> dict[str, dict[str, float]]:
    """Parse a torque-health CSV: saturation fraction and contact-force jump per clip."""
    rows = _read_csv_rows(source)
    where = str(source)[:120]
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        clip = _column(row, _CLIP_COLUMNS, where=where)
        record: dict[str, float] = {
            "saturation_frac": float(_column(row, _SATURATION_COLUMNS, where=where))
        }
        for name in _CONTACT_JUMP_COLUMNS:
            if row.get(name) not in (None, ""):
                record["contact_force_jump_N"] = float(row[name])
                break
        out[clip] = record
    if not out:
        raise AnalysisInputError(f"{where}: torque CSV has no rows")
    return out


def read_clip_list(source: str | Path) -> tuple[str, ...]:
    """Read a newline-delimited motion-key list (``#`` comments and blanks ignored)."""
    path = Path(source)
    if not path.is_file():
        raise AnalysisInputError(f"missing clip list: {path}")
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            keys.append(stripped)
    return tuple(keys)


# --------------------------------------------------------------------------- statistics


def sign_flip_permutation_p(
    deltas: Sequence[float],
    *,
    nperm: int = DEFAULT_NPERM,
    seed: int = DEFAULT_SEED,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Paired sign-flip permutation test on per-clip deltas, RNG pinned to ``default_rng(seed)``.

    The p-value uses the ``(hits + 1) / (nperm + 1)`` convention so it can never be reported as
    exactly zero -- a Monte Carlo test cannot license "p = 0", and 5000 draws cannot resolve below
    1/5001 no matter how large the effect. ``discrete_floor_p`` records the *other* floor: with n
    clips there are only 2^n sign assignments, so for small strata the discrete lattice, not the
    number of draws, is what limits the smallest attainable p.
    """
    values = np.asarray(list(deltas), dtype=float)
    if values.size == 0:
        raise AnalysisInputError("permutation test needs at least one paired delta")
    if not np.all(np.isfinite(values)):
        raise AnalysisInputError("permutation test received a non-finite delta")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(nperm), values.size))
    null = (signs * values).mean(axis=1)
    if alternative == "greater":
        hits = int(np.count_nonzero(null >= observed))
    elif alternative == "less":
        hits = int(np.count_nonzero(null <= observed))
    elif alternative == "two-sided":
        hits = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return {
        "n_clips": int(values.size),
        "observed_mean": observed,
        "alternative": alternative,
        "p": float((hits + 1) / (int(nperm) + 1)),
        "nperm": int(nperm),
        "rng_seed": int(seed),
        "monte_carlo_floor_p": 1.0 / (int(nperm) + 1),
        "discrete_floor_p": float(2.0**-values.size) if values.size <= 30 else 0.0,
    }


@dataclass(frozen=True)
class Strata:
    """The eval strata, all defined by the BASELINE arm so no arm can redefine its own grading."""

    ranked_on: str
    feasible: tuple[str, ...]
    infeasible: tuple[str, ...]
    worst_decile: tuple[str, ...]
    best_half: tuple[str, ...]
    easy: tuple[str, ...]
    ground_contact: tuple[str, ...]

    def sizes(self) -> dict[str, int]:
        return {
            "feasible": len(self.feasible),
            "infeasible": len(self.infeasible),
            "worst_decile": len(self.worst_decile),
            "best_half": len(self.best_half),
            "easy": len(self.easy),
            "ground_contact": len(self.ground_contact),
        }


def build_strata(
    baseline_means: dict[str, float],
    *,
    ranking_means: dict[str, float] | None = None,
    infeasible_frac: dict[str, float] | None = None,
    ground_clips: Iterable[str] = (),
    ranked_on: str = "all start offsets",
) -> Strata:
    """Build the sealed strata from the baseline arm's survival.

    ``ranking_means`` lets the strata be defined on a disjoint subset of start offsets from the one
    the deltas are scored on -- the split-half control for regression to the mean.
    """
    ranking = baseline_means if ranking_means is None else ranking_means
    flags = infeasible_frac or {}
    clips = sorted(baseline_means)
    missing = [c for c in clips if c not in ranking]
    if missing:
        raise AnalysisInputError(f"ranking is missing {len(missing)} clips, e.g. {missing[:5]}")
    feasible = tuple(c for c in clips if flags.get(c, 0.0) <= FLAG_INFEASIBLE_FRAC)
    feasible_set = set(feasible)
    infeasible = tuple(c for c in clips if c not in feasible_set)
    if not feasible:
        raise AnalysisInputError(
            "every eval clip is flagged infeasible; the concentration test has no support"
        )
    order = sorted(feasible, key=lambda c: (ranking[c], c))
    worst = tuple(order[: max(1, len(order) // 10)])
    best_half = tuple(order[len(order) // 2 :])
    easy = tuple(c for c in feasible if ranking[c] >= EASY_SURVIVAL)
    wanted_ground = set(ground_clips)
    ground = tuple(c for c in clips if c in wanted_ground)
    return Strata(
        ranked_on=ranked_on,
        feasible=feasible,
        infeasible=infeasible,
        worst_decile=worst,
        best_half=best_half,
        easy=easy,
        ground_contact=ground,
    )


def stratum_delta(
    baseline_means: dict[str, float],
    arm_means: dict[str, float],
    keys: Sequence[str],
    *,
    nperm: int = DEFAULT_NPERM,
    seed: int = DEFAULT_SEED,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Paired per-clip delta summary for one stratum (arm minus baseline)."""
    keys = tuple(keys)
    if not keys:
        return {"n_clips": 0, "delta_mean": float("nan"), "note": "stratum is empty"}
    deltas = [arm_means[c] - baseline_means[c] for c in keys]
    return {
        "n_clips": len(keys),
        "baseline_mean": float(np.mean([baseline_means[c] for c in keys])),
        "arm_mean": float(np.mean([arm_means[c] for c in keys])),
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "n_improved": int(sum(1 for d in deltas if d > 0)),
        "n_worsened": int(sum(1 for d in deltas if d < 0)),
        "permutation": sign_flip_permutation_p(
            deltas, nperm=nperm, seed=seed, alternative=alternative
        ),
    }


def classify_direction(delta_worst: float, delta_best_half: float, delta_easy: float) -> str:
    """Name what the arm actually did, so a failure is never reported as a bare 'no'."""
    if delta_worst <= -MIN_WORST_DECILE_EFFECT:
        return "harm"
    if (
        abs(delta_worst) < MIN_WORST_DECILE_EFFECT
        and abs(delta_best_half) < MIN_WORST_DECILE_EFFECT
        and abs(delta_easy) <= EASY_DRIFT_TOL
    ):
        return "null"
    if (
        delta_worst >= MIN_WORST_DECILE_EFFECT
        and delta_worst >= CONCENTRATION_RATIO * delta_best_half
    ):
        return "improves-worst"
    if delta_best_half >= MIN_WORST_DECILE_EFFECT or abs(delta_easy) > EASY_DRIFT_TOL:
        return "uniform-lift"
    return "mixed"


def concentration_signature(
    baseline_means: dict[str, float],
    arm_means: dict[str, float],
    strata: Strata,
    *,
    nperm: int = DEFAULT_NPERM,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """The volume-confound control: does the gain concentrate where hygiene predicts it should?"""
    worst = stratum_delta(
        baseline_means,
        arm_means,
        strata.worst_decile,
        nperm=nperm,
        seed=seed,
        alternative="greater",
    )
    best = stratum_delta(baseline_means, arm_means, strata.best_half, nperm=nperm, seed=seed)
    easy = stratum_delta(baseline_means, arm_means, strata.easy, nperm=nperm, seed=seed)
    delta_worst = float(worst["delta_mean"])
    delta_best = float(best["delta_mean"])
    delta_easy = 0.0 if easy["n_clips"] == 0 else float(easy["delta_mean"])
    ratio_pass = bool(delta_worst >= CONCENTRATION_RATIO * delta_best)
    materiality_pass = bool(delta_worst >= MIN_WORST_DECILE_EFFECT)
    easy_flat_pass = bool(abs(delta_easy) <= EASY_DRIFT_TOL)
    return {
        "ranked_on": strata.ranked_on,
        "delta_worst_decile": delta_worst,
        "delta_best_half": delta_best,
        "delta_easy": delta_easy,
        "worst_decile": worst,
        "best_half": best,
        "easy": easy,
        "easy_stratum_empty": bool(easy["n_clips"] == 0),
        "ratio_pass": ratio_pass,
        "materiality_pass": materiality_pass,
        "easy_flat_pass": easy_flat_pass,
        "pass": bool(ratio_pass and materiality_pass and easy_flat_pass),
        "direction": classify_direction(delta_worst, delta_best, delta_easy),
        "rule": (
            f"delta_worst >= {CONCENTRATION_RATIO:g} * delta_best_half (sealed ratio) AND "
            f"delta_worst >= {MIN_WORST_DECILE_EFFECT} (materiality floor, added here because the "
            "bare ratio is vacuous at zero and perverse when both deltas are negative) AND "
            f"|delta_easy| <= {EASY_DRIFT_TOL} (easy stratum stays flat)"
        ),
    }


# --------------------------------------------------------------------------- endpoints 2 and 3


def exposure_metrics(
    probs: dict[str, float], *, infeasible_frac: dict[str, float] | None = None
) -> dict[str, Any]:
    """Normalized Shannon entropy, top-1 share, and mass spent on flagged clips."""
    if not probs:
        raise AnalysisInputError("exposure ledger is empty")
    flags = infeasible_frac or {}
    keys = sorted(probs)
    values = np.array([probs[k] for k in keys], dtype=float)
    nonzero = values[values > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    n = len(keys)
    top_index = int(np.argmax(values))
    flagged = [k for k in keys if flags.get(k, 0.0) > FLAG_INFEASIBLE_FRAC]
    wasted = float(sum(probs[k] for k in flagged))
    return {
        "n_clips": n,
        "shannon_entropy_nats": entropy,
        "normalized_entropy": float(entropy / math.log(n)) if n > 1 else float("nan"),
        "effective_num_clips": float(math.exp(entropy)),
        "top1_share": float(values[top_index]),
        "top1_motion_key": keys[top_index],
        "fair_share": 1.0 / n,
        "top1_over_fair_share": float(values[top_index] * n),
        "n_flagged_in_bank": len(flagged),
        "wasted_exposure_frac": wasted,
        "flag_threshold_infeasible_frac": FLAG_INFEASIBLE_FRAC,
    }


def torque_metrics(records: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Actuator saturation and contact-force discontinuity, summarized over clips."""
    if not records:
        raise AnalysisInputError("torque CSV is empty")
    saturation = np.array([r["saturation_frac"] for r in records.values()], dtype=float)
    jumps = np.array(
        [r["contact_force_jump_N"] for r in records.values() if "contact_force_jump_N" in r],
        dtype=float,
    )
    out: dict[str, Any] = {
        "n_clips": len(records),
        "saturation_frac_mean": float(np.mean(saturation)),
        "saturation_frac_p95": float(np.percentile(saturation, 95)),
    }
    if jumps.size:
        out["contact_force_jump_N_mean"] = float(np.mean(jumps))
        out["contact_force_jump_N_p95"] = float(np.percentile(jumps, 95))
    else:
        out["contact_force_jump_N_mean"] = float("nan")
        out["contact_force_jump_N_p95"] = float("nan")
    return out


# --------------------------------------------------------------------------- assembly


def _require_complete(
    stratified: dict[str, StratifiedEval],
    *,
    baseline: str,
    required_arms: Sequence[str],
    exposure: dict[str, dict[str, float]] | None,
    torque: dict[str, dict[str, dict[str, float]]] | None,
) -> None:
    """Refuse to produce a table that is quietly missing a column."""
    missing = [arm for arm in required_arms if arm not in stratified]
    if missing:
        raise AnalysisInputError(
            "incomplete matrix: no stratified eval for arm(s) "
            f"{missing}. Present: {sorted(stratified)}. Pass --require-arms to analyze a "
            "deliberately partial matrix; a missing arm is never dropped silently."
        )
    if baseline not in stratified:
        raise AnalysisInputError(f"baseline arm {baseline!r} absent; present: {sorted(stratified)}")
    for label, table in (("exposure", exposure), ("torque", torque)):
        if not table:
            continue
        gaps = [arm for arm in stratified if arm not in table]
        if gaps:
            raise AnalysisInputError(
                f"{label} ledger given for {sorted(table)} but not for {gaps}; a partial "
                f"{label} table would compare arms on different evidence"
            )


def _check_eval_set(stratified: dict[str, StratifiedEval], *, baseline: str) -> dict[str, Any]:
    """The eval set must be identical across arms, and must be the RAW bank."""
    base = stratified[baseline]
    base_clips = base.clips()
    banks = {arm: ev.eval_bank for arm, ev in stratified.items()}
    tagged = {arm: bank for arm, bank in banks.items() if bank}
    repaired = sorted(arm for arm, bank in tagged.items() if "repair" in bank.lower())
    if repaired:
        raise AnalysisInputError(
            f"arms {repaired} report a repaired eval bank ({[tagged[a] for a in repaired]}). "
            "Grading a repair on repaired references is a self-serving endpoint; the eval set "
            "must stay raw for every arm."
        )
    distinct = sorted(set(tagged.values()))
    if len(distinct) > 1:
        raise AnalysisInputError(f"arms disagree about the eval bank: {tagged}")
    for arm, ev in stratified.items():
        if ev.clips() != base_clips:
            only_base = sorted(base_clips - ev.clips())
            only_arm = sorted(ev.clips() - base_clips)
            raise AnalysisInputError(
                f"arm {arm!r} evaluates a different clip set than {baseline!r}: "
                f"{len(only_base)} missing (e.g. {only_base[:3]}), "
                f"{len(only_arm)} extra (e.g. {only_arm[:3]}). Pass --allow-clip-mismatch to "
                "intersect deliberately."
            )
        if ev.offsets != base.offsets:
            raise AnalysisInputError(
                f"arm {arm!r} uses start offsets {ev.offsets}, baseline uses {base.offsets}; "
                "the survival endpoint is only paired at matched offsets"
            )
    return {
        "eval_bank": distinct[0] if distinct else None,
        "eval_bank_declared": bool(distinct),
        "n_clips": len(base_clips),
        "start_offsets_s": list(base.offsets),
        "raw_eval_guarantee": (
            "eval bank tag checked for 'repair'"
            if distinct
            else "NO eval_bank column present -- the raw-eval guarantee is asserted by the "
            "caller, not verified by this script"
        ),
    }


def _intersect_clip_sets(stratified: dict[str, StratifiedEval]) -> dict[str, StratifiedEval]:
    """Deliberate, recorded intersection for --allow-clip-mismatch."""
    shared = set.intersection(*(ev.clips() for ev in stratified.values()))
    if not shared:
        raise AnalysisInputError("arms share no eval clips at all")
    return {
        arm: StratifiedEval(
            source=ev.source,
            by_clip={c: v for c, v in ev.by_clip.items() if c in shared},
            offsets=ev.offsets,
            eval_bank=ev.eval_bank,
        )
        for arm, ev in stratified.items()
    }


def analyze(
    *,
    stratified: dict[str, StratifiedEval],
    eval_flags: dict[str, float] | None = None,
    exposure: dict[str, dict[str, float]] | None = None,
    torque: dict[str, dict[str, dict[str, float]]] | None = None,
    screens: dict[str, dict[str, float]] | None = None,
    ground_clips: Iterable[str] = (),
    baseline: str = BASELINE_ARM,
    required_arms: Sequence[str] = REQUIRED_ARMS,
    nperm: int = DEFAULT_NPERM,
    seed: int = DEFAULT_SEED,
    allow_clip_mismatch: bool = False,
    assume_all_feasible: bool = False,
    script_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the whole matrix analysis and return the machine-readable report."""
    _require_complete(
        stratified,
        baseline=baseline,
        required_arms=required_arms,
        exposure=exposure,
        torque=torque,
    )
    if eval_flags is None and not assume_all_feasible:
        raise AnalysisInputError(
            "no --eval-screen given: the concentration test is defined on FEASIBLE eval clips, and "
            "the raw eval set contains clips the reference itself cannot support. Pass "
            "--eval-screen, or --assume-all-feasible to record that choice in the output."
        )
    if allow_clip_mismatch:
        stratified = _intersect_clip_sets(stratified)
    eval_info = _check_eval_set(stratified, baseline=baseline)
    eval_info["clip_sets_intersected"] = bool(allow_clip_mismatch)
    eval_info["feasibility_source"] = (
        "eval screen CSV"
        if eval_flags is not None
        else "ASSUMED all-feasible (--assume-all-feasible)"
    )

    base_eval = stratified[baseline]
    base_means = base_eval.offset_mean()
    ground = tuple(ground_clips)
    strata = build_strata(
        base_means,
        infeasible_frac=eval_flags,
        ground_clips=ground,
        ranked_on="all start offsets (sealed)",
    )

    # Split-half control: rank the strata on one half of the start offsets, score the deltas on the
    # other half, so the noise that selects the worst decile is independent of the noise that scores
    # it. The gap between this and the sealed column is the visible regression-to-the-mean.
    ordered = list(base_eval.offsets)
    rank_offsets, score_offsets = tuple(ordered[0::2]), tuple(ordered[1::2])
    split_ok = bool(rank_offsets) and bool(score_offsets)
    split_strata = None
    base_score_means = None
    if split_ok:
        split_strata = build_strata(
            base_eval.offset_mean(score_offsets),
            ranking_means=base_eval.offset_mean(rank_offsets),
            infeasible_frac=eval_flags,
            ground_clips=ground,
            ranked_on=f"rank offsets {list(rank_offsets)} / score offsets {list(score_offsets)}",
        )
        base_score_means = base_eval.offset_mean(score_offsets)

    arms: dict[str, Any] = {}
    for arm, ev in sorted(stratified.items()):
        role = ARM_ROLES.get(
            arm, {"axis": "unregistered", "confound_status": "unknown", "note": ""}
        )
        record: dict[str, Any] = {
            "source": ev.source,
            "axis": role["axis"],
            "confound_status": role["confound_status"],
            "design_note": role["note"],
            "is_baseline": arm == baseline,
        }
        arm_means = ev.offset_mean()
        record["survival_overall_mean"] = float(np.mean([arm_means[c] for c in sorted(arm_means)]))
        if arm != baseline:
            record["concentration"] = concentration_signature(
                base_means, arm_means, strata, nperm=nperm, seed=seed
            )
            if split_ok and split_strata is not None and base_score_means is not None:
                record["concentration_split_half"] = concentration_signature(
                    base_score_means,
                    ev.offset_mean(score_offsets),
                    split_strata,
                    nperm=nperm,
                    seed=seed,
                )
                record["regression_to_mean_gap"] = float(
                    record["concentration"]["delta_worst_decile"]
                    - record["concentration_split_half"]["delta_worst_decile"]
                )
            else:
                record["concentration_split_half"] = None
                record["regression_to_mean_gap"] = float("nan")
            record["strata_deltas"] = {
                "all_feasible": stratum_delta(
                    base_means, arm_means, strata.feasible, nperm=nperm, seed=seed
                ),
                "infeasible_eval_clips": stratum_delta(
                    base_means, arm_means, strata.infeasible, nperm=nperm, seed=seed
                ),
                "ground_contact": stratum_delta(
                    base_means, arm_means, strata.ground_contact, nperm=nperm, seed=seed
                ),
            }
        if exposure and arm in exposure:
            record["exposure"] = exposure_metrics(
                exposure[arm], infeasible_frac=(screens or {}).get(arm)
            )
            record["exposure"]["flags_source"] = (
                "per-arm training-bank screen"
                if (screens or {}).get(arm)
                else "NONE (wasted " "exposure not computable without a --screen for this arm)"
            )
        if torque and arm in torque:
            record["torque_health"] = torque_metrics(torque[arm])
        arms[arm] = record

    if torque and baseline in torque:
        base_torque = arms[baseline]["torque_health"]
        for arm, record in arms.items():
            if arm == baseline or "torque_health" not in record:
                continue
            record["torque_health_delta_vs_baseline"] = {
                key: record["torque_health"][key] - base_torque[key]
                for key in (
                    "saturation_frac_mean",
                    "saturation_frac_p95",
                    "contact_force_jump_N_p95",
                )
                if key in record["torque_health"] and key in base_torque
            }

    verdicts = {
        arm: bool(record["concentration"]["pass"])
        for arm, record in arms.items()
        if "concentration" in record
    }
    directions = {
        arm: record["concentration"]["direction"]
        for arm, record in arms.items()
        if "concentration" in record
    }
    passing = sorted(arm for arm, ok in verdicts.items() if ok)
    headline = (
        "no arm shows the concentration signature; hygiene is not supported by this matrix"
        if not passing
        else "concentration signature present in: " + ", ".join(passing)
    )
    if "C_repaired" in passing:
        headline += (
            " -- C_repaired preserves N, so its lift is not attributable to training on less data"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": {
            "script": Path(__file__).name,
            "script_sha256": script_sha256 or _self_sha256(),
            "nperm": int(nperm),
            "rng_seed": int(seed),
            "rng": f"numpy.random.default_rng({int(seed)})",
            "p_value_convention": "(hits + 1) / (nperm + 1); never reported as exactly zero",
        },
        "design": {
            "baseline_arm": baseline,
            "required_arms": list(required_arms),
            "arms_present": sorted(stratified),
            "roles": {arm: ARM_ROLES.get(arm, {}) for arm in sorted(stratified)},
            "adjudicated_endpoint": "stratified survival on the raw held-out eval set",
            "reported_not_adjudicated": ["exposure ledger", "torque health"],
        },
        "eval_set": eval_info,
        "strata": {
            "sizes": strata.sizes(),
            "definition": (
                "feasible eval clips ordered ascending by the BASELINE arm's offset-mean survival; "
                f"worst = order[:max(1, n//10)], best_half = order[n//2:], easy = baseline >= "
                f"{EASY_SURVIVAL}; flagged when infeasible_frac > {FLAG_INFEASIBLE_FRAC}"
            ),
            "worst_decile_members": list(strata.worst_decile),
            "split_half_available": split_ok,
            "split_half_offsets": {"rank": list(rank_offsets), "score": list(score_offsets)},
        },
        "arms": arms,
        "decision": {
            "hygiene_claim_passes": verdicts,
            "direction": directions,
            "rule": (
                f"per arm vs {baseline}: delta_worst >= {CONCENTRATION_RATIO:g} * delta_best_half "
                f"AND delta_worst >= {MIN_WORST_DECILE_EFFECT} AND |delta_easy| <= {EASY_DRIFT_TOL}"
            ),
            "headline": headline,
        },
        "inference_note": INFERENCE_NOTE,
        "caveats": list(CAVEATS),
    }


def _self_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - only when the script is executed from memory
        return "unavailable"


# --------------------------------------------------------------------------- rendering


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:+.{digits}f}" if abs(value) < 1e3 else f"{value:.3g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Readable companion to the JSON; same numbers, no extra conclusions."""
    lines: list[str] = []
    decision = report["decision"]
    lines.append("# Hygiene x sampler matrix -- stratified survival")
    lines.append("")
    lines.append(f"**Headline.** {decision['headline']}")
    lines.append("")
    lines.append(f"- baseline arm: `{report['design']['baseline_arm']}`")
    lines.append(f"- decision rule: {decision['rule']}")
    lines.append(
        f"- permutation: sign-flip over clips, nperm={report['analysis']['nperm']}, "
        f"{report['analysis']['rng']}, p = {report['analysis']['p_value_convention']}"
    )
    lines.append(
        f"- eval set: {report['eval_set']['n_clips']} clips at start offsets "
        f"{report['eval_set']['start_offsets_s']} s; {report['eval_set']['raw_eval_guarantee']}"
    )
    lines.append(f"- feasibility: {report['eval_set']['feasibility_source']}")
    lines.append(f"- strata sizes: {report['strata']['sizes']}")
    lines.append("")
    lines.append(f"> {report['inference_note']}")
    lines.append("")

    lines.append("## Concentration signature (sealed strata, all start offsets)")
    lines.append("")
    lines.append(
        "| arm | confound status | n worst | d worst | d best-half | d easy | ratio | material | "
        "easy flat | VERDICT | direction |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|---|---|")
    for arm in sorted(report["arms"]):
        record = report["arms"][arm]
        conc = record.get("concentration")
        if conc is None:
            lines.append(
                f"| `{arm}` | {record['confound_status']} | - | - | - | - | - | - | - | "
                "baseline | - |"
            )
            continue
        lines.append(
            f"| `{arm}` | {record['confound_status']} | {conc['worst_decile']['n_clips']} | "
            f"{_fmt(conc['delta_worst_decile'])} | {_fmt(conc['delta_best_half'])} | "
            f"{_fmt(conc['delta_easy'])} | {_fmt(conc['ratio_pass'])} | "
            f"{_fmt(conc['materiality_pass'])} | {_fmt(conc['easy_flat_pass'])} | "
            f"**{'PASS' if conc['pass'] else 'FAIL'}** | {conc['direction']} |"
        )
    lines.append("")

    if report["strata"]["split_half_available"]:
        lines.append("## Split-half robustness (strata ranked on offsets held out from scoring)")
        lines.append("")
        lines.append("| arm | d worst (sealed) | d worst (split-half) | selection gap | verdict |")
        lines.append("|---|---:|---:|---:|---|")
        for arm in sorted(report["arms"]):
            split = report["arms"][arm].get("concentration_split_half")
            conc = report["arms"][arm].get("concentration")
            if not split or not conc:
                continue
            lines.append(
                f"| `{arm}` | {_fmt(conc['delta_worst_decile'])} | "
                f"{_fmt(split['delta_worst_decile'])} | "
                f"{_fmt(report['arms'][arm]['regression_to_mean_gap'])} | "
                f"{'PASS' if split['pass'] else 'FAIL'} |"
            )
        lines.append("")

    lines.append("## Per-stratum paired deltas (permutation p, clip-level only)")
    lines.append("")
    lines.append("| arm | stratum | n | d mean | improved/worsened | p |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for arm in sorted(report["arms"]):
        deltas = report["arms"][arm].get("strata_deltas")
        if not deltas:
            continue
        for name, rec in deltas.items():
            if rec["n_clips"] == 0:
                lines.append(f"| `{arm}` | {name} | 0 | n/a | - | - |")
                continue
            lines.append(
                f"| `{arm}` | {name} | {rec['n_clips']} | {_fmt(rec['delta_mean'])} | "
                f"{rec['n_improved']}/{rec['n_worsened']} | {rec['permutation']['p']:.4f} |"
            )
    lines.append("")

    if any("exposure" in r for r in report["arms"].values()):
        lines.append("## Exposure ledger (reported, not adjudicated)")
        lines.append("")
        lines.append(
            "| arm | bank clips | normalized entropy | effective clips | top-1 share | "
            "top-1 / fair share | wasted exposure |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm in sorted(report["arms"]):
            exp = report["arms"][arm].get("exposure")
            if not exp:
                continue
            lines.append(
                f"| `{arm}` | {exp['n_clips']} | {exp['normalized_entropy']:.4f} | "
                f"{exp['effective_num_clips']:.1f} | {exp['top1_share']:.4f} | "
                f"{exp['top1_over_fair_share']:.1f}x | {exp['wasted_exposure_frac']:.4f} |"
            )
        lines.append("")

    if any("torque_health" in r for r in report["arms"].values()):
        lines.append("## Torque health (reported, not adjudicated)")
        lines.append("")
        lines.append("| arm | saturation mean | saturation p95 | contact-force jump p95 (N) |")
        lines.append("|---|---:|---:|---:|")
        for arm in sorted(report["arms"]):
            tq = report["arms"][arm].get("torque_health")
            if not tq:
                continue
            lines.append(
                f"| `{arm}` | {tq['saturation_frac_mean']:.4f} | {tq['saturation_frac_p95']:.4f} | "
                f"{tq['contact_force_jump_N_p95']:.1f} |"
            )
        lines.append("")

    lines.append("## Arm roles")
    lines.append("")
    for arm in sorted(report["arms"]):
        record = report["arms"][arm]
        lines.append(
            f"- `{arm}` ({record['axis']} axis, {record['confound_status']}): "
            f"{record['design_note']}"
        )
    lines.append("")
    lines.append("## What a reviewer should challenge")
    lines.append("")
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- synthetic dry-run

#: Start-offset grid the fabricated eval tables use. Seven offsets, so the split-half control has a
#: 4/3 partition of the grid to re-rank on.
SYNTHETIC_OFFSETS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)

#: Per-offset survival wobble. Constructed to sum to EXACTLY zero over the full offset grid, so a
#: clip's sealed offset-mean is exactly its designed level -- while any HALF of the grid does not,
#: which is what gives the split-half control something real to disagree about.
SYNTHETIC_JITTER_AMPLITUDE = 0.012

#: Larger wobble for clips near the survival floor. This is not decoration: a clip that barely
#: survives is exactly the clip whose survival depends most on where the episode starts, and it is
#: the low end that the worst decile is selected from. Without heteroscedastic noise there the
#: selection step is noiseless, the split-half control reproduces the sealed column exactly, and the
#: regression-to-the-mean column would look reassuring for the wrong reason.
SYNTHETIC_SELECTION_NOISE = 0.10
SYNTHETIC_NOISY_BELOW = 0.5

SYNTHETIC_EVAL_BANK = "raw_heldout_v1"
SYNTHETIC_TRAIN_CLEAN = 170
SYNTHETIC_TRAIN_FLAGGED = 30

#: Sampling weight the fabricated ledger gives a FLAGGED training clip, per arm. ``B_pruned`` has no
#: flagged clips in its bank at all, so it gets ``None``. ``E_raw_cap`` is absent on purpose: it is
#: not a weight choice, it is ``A_raw``'s own ledger with the per-motion cap applied to it, which is
#: what ``max_prob_per_motion`` actually does to a run.
SYNTHETIC_FLAGGED_WEIGHT: dict[str, float | None] = {
    "A_raw": 8.0,
    "B_pruned": None,
    "C_repaired": 1.6,
    "D_raw_uniform": 1.0,
}

#: ``max_prob_per_motion``, expressed the way the intervention config sets it: a multiple of fair share.
SYNTHETIC_CAP_MULTIPLE = 5.0

#: (mean actuator-saturation fraction, mean contact-force jump [N]) per arm, fabricated.
SYNTHETIC_TORQUE: dict[str, tuple[float, float]] = {
    "A_raw": (0.184, 61.0),
    "B_pruned": (0.132, 38.0),
    "C_repaired": (0.129, 35.0),
    "D_raw_uniform": (0.171, 52.0),
    "E_raw_cap": (0.166, 49.0),
}

#: effect name -> (expected concentration verdict, expected direction label).
SYNTHETIC_EXPECTATIONS: dict[str, tuple[bool, str]] = {
    "concentrated": (True, "improves-worst"),
    "uniform": (False, "uniform-lift"),
    "flat": (False, "null"),
    "harmful": (False, "harm"),
    "perverse_ratio": (False, "harm"),
}

_TREATMENT_ARMS = ("B_pruned", "C_repaired", "D_raw_uniform", "E_raw_cap")


def _all_arms(effect: str) -> dict[str, str]:
    return dict.fromkeys(_TREATMENT_ARMS, effect)


#: (branch id, what it fabricates, arm -> effect). The first four are the required branches; the
#: last two exist because they are the two ways the SEALED rule alone would misfire.
SYNTHETIC_BRANCHES: tuple[tuple[str, str, dict[str, str]], ...] = (
    (
        "i_hygiene_works",
        "worst decile lifts (+0.25), best half and easy stratum barely move (+0.008)",
        _all_arms("concentrated"),
    ),
    (
        "ii_volume_confound",
        "every stratum lifts by the same +0.06 -- what training on less data looks like",
        _all_arms("uniform"),
    ),
    (
        "iii_null",
        "nothing moves (+/-0.004 alternating); the honest empty result",
        _all_arms("flat"),
    ),
    (
        "iv_harm",
        "worst decile DROPS (-0.15) while the rest drifts -0.01",
        _all_arms("harmful"),
    ),
    (
        "v_mixed_matrix",
        "four different outcomes inside one matrix; no arm may contaminate another",
        {
            "B_pruned": "concentrated",
            "C_repaired": "flat",
            "D_raw_uniform": "uniform",
            "E_raw_cap": "harmful",
        },
    ),
    (
        "vi_perverse_ratio",
        "worst -0.05, rest -0.10: the SEALED ratio test alone would PASS this; the "
        "materiality floor is what refuses it",
        _all_arms("perverse_ratio"),
    ),
)


def _synthetic_jitter(clip_index: int, offset_index: int, level: float) -> float:
    """Zero-mean-over-the-grid per-offset wobble (stride coprime with the 7-offset grid).

    Amplitude is larger below :data:`SYNTHETIC_NOISY_BELOW` so the worst decile is selected out of
    a noisy tail, as it will be in the real eval.
    """
    stride = (clip_index % 6) + 1
    amplitude = (
        SYNTHETIC_SELECTION_NOISE if level < SYNTHETIC_NOISY_BELOW else SYNTHETIC_JITTER_AMPLITUDE
    )
    return amplitude * (((offset_index * stride) % 7) - 3) / 3.0


def synthetic_eval_bank() -> tuple[dict[str, float], dict[str, float], tuple[str, ...]]:
    """Fabricated raw held-out eval set: designed offset-mean survival, flags, ground-contact keys.

    76 feasible clips (so the worst decile is 7 and the best half is 38) plus 8 clips the screen
    flags, which are evaluated but excluded from the concentration strata by construction.
    """
    levels: dict[str, float] = {}
    for i, value in enumerate(np.linspace(0.30, 0.90, 60)):
        levels[f"FEAS_{i:02d}"] = float(value)
    for i, value in enumerate(np.linspace(0.955, 0.975, 10)):
        levels[f"EASY_{i:02d}"] = float(
            value
        )  # sits above the 0.95 easy-stratum cut, with headroom
    for i, value in enumerate(np.linspace(0.327, 0.807, 6)):
        levels[f"GROUND_{i:02d}"] = float(value)
    for i in range(8):
        levels[f"FLAG_{i:02d}"] = 0.40 + 0.01 * i
    flags = {key: (0.45 if key.startswith("FLAG_") else 0.02) for key in levels}
    ground = tuple(f"GROUND_{i:02d}" for i in range(6))
    return levels, flags, ground


def synthetic_gains(
    effect: str, levels: dict[str, float], flags: dict[str, float]
) -> dict[str, float]:
    """Per-clip survival shift for one fabricated outcome, keyed off the baseline ranking."""
    feasible = [k for k in sorted(levels) if flags.get(k, 0.0) <= FLAG_INFEASIBLE_FRAC]
    order = sorted(feasible, key=lambda c: (levels[c], c))
    worst = set(order[: max(1, len(order) // 10)])
    gains: dict[str, float] = {}
    for index, key in enumerate(sorted(levels)):
        in_worst = key in worst
        if effect == "concentrated":
            gain = 0.25 if in_worst else 0.008
        elif effect == "uniform":
            gain = 0.06
        elif effect == "flat":
            gain = 0.004 if index % 2 == 0 else -0.004
        elif effect == "harmful":
            gain = -0.15 if in_worst else -0.01
        elif effect == "perverse_ratio":
            gain = -0.05 if in_worst else -0.10
        else:
            raise ValueError(f"unknown synthetic effect {effect!r}")
        gains[key] = gain
    return gains


def synthetic_stratified_csv(
    levels: dict[str, float],
    gains: dict[str, float] | None = None,
    *,
    eval_bank: str = SYNTHETIC_EVAL_BANK,
    omit_clips: Sequence[str] = (),
    ragged_clip: str | None = None,
) -> str:
    """Render one arm's stratified-start eval CSV, including the summary rows the reader skips."""
    gains = gains or {}
    rows = ["clip,offset_s,survival,mean_survival_s,n,window_s,eval_bank"]
    for index, key in enumerate(sorted(levels)):
        if key in set(omit_clips):
            continue
        values = []
        for offset_index, offset in enumerate(SYNTHETIC_OFFSETS):
            if key == ragged_clip and offset_index == 0:
                continue  # a hole in the grid: the reader must refuse, not average over it
            level = levels[key]
            wobble = _synthetic_jitter(index, offset_index, level)
            survival = float(np.clip(level + wobble + gains.get(key, 0.0), 0.0, 1.0))
            values.append(survival)
            rows.append(f"{key},{offset:g},{survival:.6f},2.5,8,3.0,{eval_bank}")
        rows.append(f"{key},mean,{float(np.mean(values)):.6f},,56,3.0,{eval_bank}")
    return "\n".join(rows) + "\n"


def synthetic_screen_csv(levels: dict[str, float], flags: dict[str, float]) -> str:
    rows = ["motion_key,infeasible_frac,airborne_frac"]
    for key in sorted(levels):
        rows.append(f"{key},{flags[key]:.4f},{flags[key] * 0.5:.4f}")
    return "\n".join(rows) + "\n"


def synthetic_training_bank() -> tuple[tuple[str, ...], tuple[str, ...]]:
    clean = tuple(f"TRAIN_{i:03d}" for i in range(SYNTHETIC_TRAIN_CLEAN))
    flagged = tuple(f"TFLAG_{i:02d}" for i in range(SYNTHETIC_TRAIN_FLAGGED))
    return clean, flagged


def _synthetic_weights(arm: str) -> list[tuple[str, float]]:
    clean, flagged = synthetic_training_bank()
    weight = SYNTHETIC_FLAGGED_WEIGHT[arm]
    uniform = arm == "D_raw_uniform"
    pairs: list[tuple[str, float]] = []
    for i, key in enumerate(clean):
        pairs.append((key, 1.0 if uniform else 1.0 + 0.4 * ((i % 5) / 4.0)))
    if weight is not None:
        for i, key in enumerate(flagged):
            # one clip absorbs 5x what its already-inflated peers do: the single impossible clip
            pairs.append((key, weight * (5.0 if (i == 0 and arm == "A_raw") else 1.0)))
    return pairs


def apply_per_motion_cap(
    pairs: Sequence[tuple[str, float]], *, multiple: float = SYNTHETIC_CAP_MULTIPLE
) -> list[tuple[str, float]]:
    """Bound every motion at ``multiple`` x fair share and water-fill the excess pro rata.

    Written out rather than hand-waved because the result is the point of arm E: the cap is a bound
    on the TOP of the distribution, so it collapses top-1 share while barely touching the aggregate
    mass spent on flagged clips. An arm-E ledger that showed both improving would have been a
    fabricated result quietly arguing for its own intervention.
    """
    keys = [key for key, _ in pairs]
    probs = np.array([value for _, value in pairs], dtype=float)
    probs = probs / probs.sum()
    cap = multiple / probs.size
    for _ in range(64):
        over = probs > cap + 1e-12
        if not over.any():
            break
        excess = float((probs[over] - cap).sum())
        probs[over] = cap
        under = ~over
        probs[under] += excess * probs[under] / probs[under].sum()
    return list(zip(keys, (probs / probs.sum()).tolist()))


def synthetic_exposure_csv(arm: str) -> str:
    """One arm's sampling ledger. ``A_raw`` emits raw counts to exercise the count fallback."""
    if arm == "E_raw_cap":
        pairs = apply_per_motion_cap(_synthetic_weights("A_raw"))
    else:
        pairs = _synthetic_weights(arm)
    column = "count" if arm == "A_raw" else "prob"
    total = sum(value for _, value in pairs)
    rows = [f"motion_key,{column}"]
    for key, value in pairs:
        rows.append(f"{key},{value:.8f}" if column == "count" else f"{key},{value / total:.10f}")
    return "\n".join(rows) + "\n"


def synthetic_bank_screen_csv(*, repaired: bool) -> str:
    """Training-bank screen. ``repaired=True`` is C's bank: the same clips, re-screened after repair."""
    clean, flagged = synthetic_training_bank()
    rows = ["motion_key,infeasible_frac"]
    for key in clean:
        rows.append(f"{key},0.0200")
    for key in flagged:
        rows.append(f"{key},{0.0300 if repaired else 0.4000:.4f}")
    return "\n".join(rows) + "\n"


def synthetic_torque_csv(arm: str, levels: dict[str, float]) -> str:
    saturation, jump = SYNTHETIC_TORQUE[arm]
    rows = ["motion_key,saturation_frac,contact_force_jump_N"]
    for index, key in enumerate(sorted(levels)):
        sat = saturation + 0.012 * (((index * 3) % 7) - 3) / 3.0
        jmp = jump * (1.0 + 0.10 * (((index * 5) % 7) - 3) / 3.0)
        rows.append(f"{key},{max(sat, 0.0):.6f},{max(jmp, 0.0):.4f}")
    return "\n".join(rows) + "\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def materialize_branch(
    root: Path,
    effects: dict[str, str],
    *,
    eval_bank_overrides: dict[str, str] | None = None,
    ragged_arm: str | None = None,
    drop_clip_from: str | None = None,
) -> dict[str, Any]:
    """Write a whole fabricated matrix to ``root`` and return the paths, arm by arm.

    Everything goes through real files and the real readers -- a dry-run that bypassed the parsers
    would not be a dry-run of this script.
    """
    levels, flags, ground = synthetic_eval_bank()
    banks = eval_bank_overrides or {}
    strat: dict[str, str] = {}
    for arm in (BASELINE_ARM, *_TREATMENT_ARMS):
        gains = None if arm == BASELINE_ARM else synthetic_gains(effects[arm], levels, flags)
        text = synthetic_stratified_csv(
            levels,
            gains,
            eval_bank=banks.get(arm, SYNTHETIC_EVAL_BANK),
            omit_clips=("FEAS_07",) if arm == drop_clip_from else (),
            ragged_clip="FEAS_09" if arm == ragged_arm else None,
        )
        strat[arm] = str(_write(root / f"strat_{arm}.csv", text))
    paths: dict[str, Any] = {
        "strat": strat,
        "eval_screen": str(_write(root / "eval_screen.csv", synthetic_screen_csv(levels, flags))),
        "ground": str(
            _write(
                root / "ground_clips.txt",
                "# ground contact / kneel / crawl\n" + "\n".join(ground) + "\n",
            )
        ),
        "exposure": {
            arm: str(_write(root / f"exposure_{arm}.csv", synthetic_exposure_csv(arm)))
            for arm in (BASELINE_ARM, *_TREATMENT_ARMS)
        },
        "torque": {
            arm: str(_write(root / f"torque_{arm}.csv", synthetic_torque_csv(arm, levels)))
            for arm in (BASELINE_ARM, *_TREATMENT_ARMS)
        },
        "screen": {
            "all": str(
                _write(root / "screen_raw_bank.csv", synthetic_bank_screen_csv(repaired=False))
            ),
            "C_repaired": str(
                _write(root / "screen_repaired_bank.csv", synthetic_bank_screen_csv(repaired=True))
            ),
        },
    }
    return paths


def expand_screens(
    raw: dict[str, dict[str, float]], arms: Iterable[str]
) -> dict[str, dict[str, float]]:
    """``all=<csv>`` supplies the training-bank screen for every arm that has no screen of its own."""
    default = raw.get("all")
    out: dict[str, dict[str, float]] = {}
    for arm in arms:
        if arm in raw:
            out[arm] = raw[arm]
        elif default is not None:
            out[arm] = default
    return out


def load_case(paths: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Turn a path bundle from :func:`materialize_branch` into ``analyze()`` keyword arguments."""
    stratified = {arm: read_stratified(path) for arm, path in paths["strat"].items()}
    exposure = {arm: read_exposure(path) for arm, path in paths.get("exposure", {}).items()}
    torque = {arm: read_torque(path) for arm, path in paths.get("torque", {}).items()}
    screens = expand_screens(
        {arm: read_screen(path) for arm, path in paths.get("screen", {}).items()}, stratified
    )
    kwargs: dict[str, Any] = {
        "stratified": stratified,
        "eval_flags": read_screen(paths["eval_screen"]) if paths.get("eval_screen") else None,
        "exposure": exposure or None,
        "torque": torque or None,
        "screens": screens or None,
        "ground_clips": read_clip_list(paths["ground"]) if paths.get("ground") else (),
    }
    kwargs.update(overrides)
    return kwargs


def _expect_refusal(label: str, thunk: Any) -> tuple[bool, str]:
    try:
        thunk()
    except AnalysisInputError as exc:
        return True, str(exc).splitlines()[0][:96]
    except Exception as exc:  # noqa: BLE001 - any other exception is itself the failure
        return False, f"raised {type(exc).__name__} instead of AnalysisInputError: {exc}"
    return False, "returned normally -- the bad input was accepted"


def _run_guards(root: Path) -> list[tuple[str, bool, str]]:
    """Every way an incomplete or self-serving input must be refused rather than quietly dropped."""
    results: list[tuple[str, bool, str]] = []
    base = materialize_branch(root / "guards", _all_arms("concentrated"))

    def missing_arm() -> None:
        kwargs = load_case(base)
        kwargs["stratified"] = {a: v for a, v in kwargs["stratified"].items() if a != "C_repaired"}
        kwargs["exposure"] = {a: v for a, v in kwargs["exposure"].items() if a != "C_repaired"}
        kwargs["torque"] = {a: v for a, v in kwargs["torque"].items() if a != "C_repaired"}
        analyze(**kwargs)

    def partial_exposure() -> None:
        kwargs = load_case(base)
        kwargs["exposure"] = {a: v for a, v in kwargs["exposure"].items() if a != "E_raw_cap"}
        analyze(**kwargs)

    def no_eval_screen() -> None:
        analyze(**load_case(base, eval_flags=None))

    def repaired_eval_bank() -> None:
        paths = materialize_branch(
            root / "guard_repaired_eval",
            _all_arms("concentrated"),
            eval_bank_overrides={"C_repaired": "repaired_heldout_v1"},
        )
        analyze(**load_case(paths))

    def ragged_grid() -> None:
        paths = materialize_branch(
            root / "guard_ragged", _all_arms("concentrated"), ragged_arm="B_pruned"
        )
        load_case(paths)

    def clip_set_mismatch() -> None:
        paths = materialize_branch(
            root / "guard_mismatch", _all_arms("concentrated"), drop_clip_from="B_pruned"
        )
        analyze(**load_case(paths))

    def missing_file() -> None:
        read_stratified(root / "does_not_exist.csv")

    for label, thunk in (
        ("missing arm (C_repaired absent)", missing_arm),
        ("exposure ledger given for only some arms", partial_exposure),
        ("no --eval-screen and no --assume-all-feasible", no_eval_screen),
        ("an arm graded on a REPAIRED eval bank", repaired_eval_bank),
        ("ragged start-offset grid", ragged_grid),
        ("arms evaluate different clip sets", clip_set_mismatch),
        ("missing input file", missing_file),
    ):
        ok, detail = _expect_refusal(label, thunk)
        results.append((label, ok, detail))
    return results


def run_synthetic(
    *,
    nperm: int = DEFAULT_NPERM,
    seed: int = DEFAULT_SEED,
    out_json: str | None = None,
    out_md: str | None = None,
) -> int:
    """Fabricate outcomes for every decision branch and check the verdicts. No real data is read.

    Returns a POSIX exit status: 0 only if every branch and every input guard behaves as sealed.
    """
    print("=" * 100)
    print("analyze_hygiene_matrix.py --synthetic : decision-logic dry-run on FABRICATED outcomes")
    print(f"  no training log, no checkpoint, no GPU. sha256(this script) = {_self_sha256()}")
    print(f"  permutation: sign-flip over clips, nperm={nperm}, numpy.random.default_rng({seed})")
    levels, flags, _ = synthetic_eval_bank()
    n_feasible = sum(1 for k in levels if flags[k] <= FLAG_INFEASIBLE_FRAC)
    print(
        f"  fabricated eval set: {len(levels)} clips ({n_feasible} feasible, "
        f"{len(levels) - n_feasible} flagged) x {len(SYNTHETIC_OFFSETS)} start offsets"
    )
    print("=" * 100)

    failures = 0
    first_report: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="hygiene_synthetic_") as tmp:
        root = Path(tmp)
        for branch, blurb, effects in SYNTHETIC_BRANCHES:
            paths = materialize_branch(root / branch, effects)
            report = analyze(**load_case(paths), nperm=nperm, seed=seed)
            if first_report is None:
                first_report = report
            json.dumps(report)  # the JSON endpoint must actually serialize, not just look like it
            markdown = render_markdown(report)
            print(f"\nbranch {branch}  --  {blurb}")
            for arm in _TREATMENT_ARMS:
                conc = report["arms"][arm]["concentration"]
                want_pass, want_dir = SYNTHETIC_EXPECTATIONS[effects[arm]]
                got_pass, got_dir = bool(conc["pass"]), conc["direction"]
                ok = got_pass == want_pass and got_dir == want_dir
                failures += 0 if ok else 1
                split = report["arms"][arm]["concentration_split_half"]
                print(
                    f"  {'PASS' if ok else 'FAIL'}  {arm:<14s} "
                    f"d_worst={conc['delta_worst_decile']:+.4f} "
                    f"d_best={conc['delta_best_half']:+.4f} "
                    f"d_easy={conc['delta_easy']:+.4f} | "
                    f"ratio={'y' if conc['ratio_pass'] else 'n'} "
                    f"material={'y' if conc['materiality_pass'] else 'n'} "
                    f"easy_flat={'y' if conc['easy_flat_pass'] else 'n'} "
                    f"-> claim={'PASSES' if got_pass else 'fails'} ({got_dir}); "
                    f"expected {'PASSES' if want_pass else 'fails'} ({want_dir}); "
                    f"split-half d_worst={split['delta_worst_decile']:+.4f}"
                )
            for check, ok in (
                ("markdown renders", bool(markdown) and "Concentration signature" in markdown),
                ("every arm in table", all(f"`{a}`" in markdown for a in report["arms"])),
                ("seed echoed", f"default_rng({seed})" in json.dumps(report["analysis"])),
                ("n=1-seed caveat present", "NO seed-level inference" in report["inference_note"]),
            ):
                failures += 0 if ok else 1
                if not ok:
                    print(f"  FAIL  output contract: {check}")
            print(f"        headline: {report['decision']['headline']}")

        print(
            "\nnote on the split-half column above: in fabricated data the sealed ranking is exactly "
            "the designed level,\n     so its selection step is noise-free and the printed "
            "regression-to-the-mean gap is a LOWER bound on the real one."
        )
        print("\ninput guards (an incomplete matrix must be a loud error, never a dropped column)")
        for label, ok, detail in _run_guards(root):
            failures += 0 if ok else 1
            print(f"  {'PASS' if ok else 'FAIL'}  refuses: {label}\n          -> {detail}")

    assert first_report is not None
    exposure = first_report["arms"]
    print("\nexposure ledger + torque health (reported, never adjudicated) -- branch i:")
    for arm in (BASELINE_ARM, *_TREATMENT_ARMS):
        exp = exposure[arm]["exposure"]
        tq = exposure[arm]["torque_health"]
        print(
            f"  {arm:<14s} bank={exp['n_clips']:4d}  H_norm={exp['normalized_entropy']:.4f}  "
            f"top1={exp['top1_share']:.4f} ({exp['top1_over_fair_share']:.1f}x fair)  "
            f"wasted={exp['wasted_exposure_frac']:.4f}  sat_mean={tq['saturation_frac_mean']:.4f}"
        )

    if out_json:
        Path(out_json).write_text(json.dumps(first_report, indent=1), encoding="utf-8")
        print(f"\nwrote synthetic branch-i JSON to {out_json}")
    if out_md:
        Path(out_md).write_text(render_markdown(first_report), encoding="utf-8")
        print(f"wrote synthetic branch-i markdown to {out_md}")

    print("\n" + "=" * 100)
    if failures:
        print(f"SYNTHETIC DRY-RUN FAILED: {failures} check(s) reached the wrong verdict.")
        return 1
    print(
        f"SYNTHETIC DRY-RUN OK: {len(SYNTHETIC_BRANCHES)} fabricated branches x "
        f"{len(_TREATMENT_ARMS)} arms all decide as designed, and all 7 input guards refuse."
    )
    print("No real data was touched: every table above was fabricated in a temporary directory.")
    print("=" * 100)
    return 0


# --------------------------------------------------------------------------- CLI


def _parse_pairs(items: Sequence[str], *, flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise AnalysisInputError(f"{flag} expects ARM=PATH, got {item!r}")
        arm, path = item.split("=", 1)
        if arm in out:
            raise AnalysisInputError(f"{flag}: arm {arm!r} given twice")
        out[arm] = path
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="dry-run the decision logic on fabricated outcomes; touches no real data",
    )
    parser.add_argument(
        "--strat",
        nargs="*",
        default=[],
        metavar="ARM=CSV",
        help="stratified-start eval CSV per arm",
    )
    parser.add_argument(
        "--eval-screen",
        default=None,
        help="hygiene screen of the RAW eval bank; defines the feasible strata",
    )
    parser.add_argument(
        "--assume-all-feasible",
        action="store_true",
        help="run without --eval-screen and record that choice in the output",
    )
    parser.add_argument("--exposure", nargs="*", default=[], metavar="ARM=CSV")
    parser.add_argument(
        "--screen",
        nargs="*",
        default=[],
        metavar="ARM|all=CSV",
        help="training-bank screen per arm; 'all' is the default for unlisted arms",
    )
    parser.add_argument("--torque", nargs="*", default=[], metavar="ARM=CSV")
    parser.add_argument(
        "--ground-clips",
        default=None,
        help="newline-delimited motion keys for the ground/kneel/crawl stratum",
    )
    parser.add_argument("--baseline", default=BASELINE_ARM)
    parser.add_argument(
        "--require-arms",
        default=",".join(REQUIRED_ARMS),
        help="comma-separated arms that must be present; refuses otherwise",
    )
    parser.add_argument(
        "--allow-clip-mismatch",
        action="store_true",
        help="intersect the arms' eval clip sets on purpose, and record it",
    )
    parser.add_argument("--nperm", type=int, default=DEFAULT_NPERM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args(argv)

    try:
        if args.synthetic:
            return run_synthetic(
                nperm=args.nperm, seed=args.seed, out_json=args.out_json, out_md=args.out_md
            )
        if not args.strat:
            raise AnalysisInputError(
                "--strat ARM=CSV is required (or --synthetic for the fabricated dry-run)"
            )
        stratified = {
            arm: read_stratified(path)
            for arm, path in _parse_pairs(args.strat, flag="--strat").items()
        }
        exposure = {
            arm: read_exposure(path)
            for arm, path in _parse_pairs(args.exposure, flag="--exposure").items()
        }
        torque = {
            arm: read_torque(path)
            for arm, path in _parse_pairs(args.torque, flag="--torque").items()
        }
        screens = expand_screens(
            {
                arm: read_screen(path)
                for arm, path in _parse_pairs(args.screen, flag="--screen").items()
            },
            stratified,
        )
        report = analyze(
            stratified=stratified,
            eval_flags=read_screen(args.eval_screen) if args.eval_screen else None,
            exposure=exposure or None,
            torque=torque or None,
            screens=screens or None,
            ground_clips=read_clip_list(args.ground_clips) if args.ground_clips else (),
            baseline=args.baseline,
            required_arms=tuple(a for a in args.require_arms.split(",") if a),
            nperm=args.nperm,
            seed=args.seed,
            allow_clip_mismatch=args.allow_clip_mismatch,
            assume_all_feasible=args.assume_all_feasible,
        )
    except AnalysisInputError as exc:
        print(f"ERROR (refusing to analyze an incomplete matrix): {exc}", file=sys.stderr)
        return 2

    print(render_markdown(report))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {args.out_json}", file=sys.stderr)
    if args.out_md:
        Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
        print(f"wrote {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
