#!/usr/bin/env python
"""Freeze a stratified 100-episode cohort for independent human review.

Four things have already slipped past every numeric gate: an acceptance rate that moved when gate
semantics changed, a horizontal-only contact test that could not see an overhead collision, prompt
labels widely inconsistent with what the reference actually does, and visual marker leakage. So the
reviewed pilot is part of the dataset's credibility, not optional polish.

Sampling is stratified rather than random because the informative episodes are rare. A uniform draw
from a corpus that is mostly accepted-easy would spend most of its budget confirming the easy case
and leave the near-boundary and contact strata with one or two episodes each -- exactly the cells
where an automatic verdict is most likely to be wrong.

The cohort is frozen the same way the scene-first test set is: a fixed seed, a SHA-256 fingerprint
over the selected ids, and a refusal to overwrite. A review whose sample can be re-drawn after
seeing the results is not evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

SEED = 20260818

#: Minimum episodes per stratum, and the reason each one is worth a person's time. Totals below the
#: target are filled from the largest strata; a stratum with fewer episodes than its minimum
#: contributes all of them and is reported short rather than silently topped up from elsewhere.
STRATA = {
    "accepted_easy": (10, "the common case; a false accept here would be systematic"),
    "accepted_adapted_hard": (12, "the cells a family's claim rests on"),
    "nominal_hard_negative": (12, "the matched negatives; the whole counterfactual turns on these"),
    "near_boundary": (12, "outcome decided by a small margin, where the gate is least reliable"),
    "tracking_drift_failure": (10, "rejected with no contact at all -- is it really a failure?"),
    "overhead_contact": (10, "the collision type the gate could not see until it was fixed"),
    "lateral_contact": (8, "the other collision type, for comparison"),
    "high_self_contact": (8, "large paired forces that are correctly *not* disqualifying"),
    "unevaluable": (6, "episodes no gate could grade; are they recoverable?"),
    "other_rejected": (12, "everything else rejected, so no failure mode is excluded by design"),
}

#: Newtons. Below this an external contact is a graze rather than a collision.
CONTACT_FLOOR_N = 1.0
#: Metres per second. The gate's own drift threshold is bracketed at (0.140, 0.223); an episode
#: within this band of either side of a verdict is near-boundary.
NEAR_DRIFT_BAND = (0.12, 0.26)


def stratum_of(outcome, diagnostics: dict) -> str:
    """Which review stratum an episode belongs to. First match wins, most specific first."""
    decomposition = diagnostics.get("contact_decomposition") or {}
    overhead = float(decomposition.get("max_overhead_contact_force_n", 0.0))
    lateral = float(decomposition.get("max_lateral_contact_force_n", 0.0))
    self_force = float(decomposition.get("max_self_contact_force_n", 0.0))
    drift = abs(float(diagnostics.get("drift_rate_mps", 0.0)))
    reasons = set(outcome.rejection_reasons)

    if not outcome.evaluated:
        return "unevaluable"
    if NEAR_DRIFT_BAND[0] <= drift <= NEAR_DRIFT_BAND[1]:
        return "near_boundary"
    if overhead > CONTACT_FLOOR_N and overhead >= lateral:
        return "overhead_contact"
    if lateral > CONTACT_FLOOR_N:
        return "lateral_contact"
    if reasons == {"unstable_reference_drift"}:
        return "tracking_drift_failure"
    if outcome.outcome == "rejected":
        return "other_rejected"
    if self_force > 100.0:
        return "high_self_contact"
    return "accepted_easy"


def role_of(cell: str, outcome) -> str | None:
    """The episode's role in a family, read from its name, or None if it has none.

    Role is decided *before* physics, because a matched negative is review material precisely for
    being a rejection with contact -- classifying by physics first swallowed every one of them into
    the contact strata and left ``nominal_hard_negative`` empty, which is the one stratum the whole
    counterfactual claim rests on.
    """
    if not cell.endswith("_hard"):
        return None
    adapted = any(key in cell for key in ("crouch", "tuck", "adapted"))
    if adapted and outcome.outcome == "accepted":
        return "accepted_adapted_hard"
    if not adapted and "nominal" in cell and outcome.outcome == "rejected":
        return "nominal_hard_negative"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.out.exists():
        print(f"refusing to overwrite {args.out}; a cohort that can be re-drawn is not evidence")
        return 2

    buckets: dict[str, list[dict]] = defaultdict(list)
    scanned = 0
    for path in sorted(args.rollouts.rglob("*.trajectory.pkl")):
        cell = path.parents[1].name
        try:
            with open(path, "rb") as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))
        except Exception:  # noqa: BLE001 - a corrupt capture is itself review material
            buckets["unevaluable"].append({"cell": cell, "path": str(path), "note": "unreadable"})
            continue
        scanned += 1
        if payload is None:
            buckets["unevaluable"].append({"cell": cell, "path": str(path), "note": "no pass"})
            continue
        outcome = classify_episode(cell, payload)
        diagnostics = outcome.diagnostics or {}
        stratum = role_of(cell, outcome) or stratum_of(outcome, diagnostics)
        decomposition = diagnostics.get("contact_decomposition") or {}
        buckets[stratum].append(
            {
                "cell": cell,
                "path": str(path),
                "outcome": outcome.outcome,
                "reasons": list(outcome.rejection_reasons),
                "drift_mps": round(float(diagnostics.get("drift_rate_mps", 0.0)), 4),
                "overhead_n": round(
                    float(decomposition.get("max_overhead_contact_force_n", 0.0)), 1
                ),
                "lateral_n": round(float(decomposition.get("max_lateral_contact_force_n", 0.0)), 1),
                "self_n": round(float(decomposition.get("max_self_contact_force_n", 0.0)), 1),
            }
        )

    rng = np.random.default_rng(args.seed)
    selected, short = [], []
    for name, (minimum, _why) in STRATA.items():
        pool = buckets.get(name, [])
        take = min(minimum, len(pool))
        if take < minimum:
            short.append((name, len(pool), minimum))
        if pool:
            index = rng.choice(len(pool), size=take, replace=False)
            selected.extend(pool[int(i)] for i in sorted(index))

    # Top up toward the target from whatever is left, largest strata first, so the cohort reaches
    # its size without any stratum being quietly over-weighted beyond what is available.
    chosen = {row["path"] for row in selected}
    remainder = [
        r
        for name in sorted(buckets, key=lambda n: -len(buckets[n]))
        for r in buckets[name]
        if r["path"] not in chosen
    ]
    if len(selected) < args.target and remainder:
        extra = rng.choice(
            len(remainder), size=min(args.target - len(selected), len(remainder)), replace=False
        )
        selected.extend(remainder[int(i)] for i in sorted(extra))

    selected.sort(key=lambda r: r["path"])
    fingerprint = hashlib.sha256("\n".join(r["path"] for r in selected).encode("utf-8")).hexdigest()

    cohort = {
        "seed": args.seed,
        "target": args.target,
        "selected": len(selected),
        "scanned_episodes": scanned,
        "fingerprint_sha256": fingerprint,
        "strata_available": {k: len(v) for k, v in sorted(buckets.items())},
        "strata_rationale": {k: v[1] for k, v in STRATA.items()},
        "short_strata": [{"stratum": n, "available": a, "wanted": w} for n, a, w in short],
        "episodes": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cohort, indent=2) + "\n")

    print(f"scanned {scanned} evaluable episodes")
    print(f"\n{'stratum':>24s}{'available':>11s}{'wanted':>8s}")
    for name, (minimum, _why) in STRATA.items():
        print(
            f"{name:>24s}{len(buckets.get(name, [])):11d}{minimum:8d}"
            f"{'   SHORT' if len(buckets.get(name, [])) < minimum else ''}"
        )
    print(f"\nselected {len(selected)} episodes, fingerprint {fingerprint[:16]}...")
    if short:
        print(
            "Strata below their minimum are reported, not topped up from elsewhere: a cohort "
            "that silently substitutes easy episodes for rare ones measures the wrong thing."
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
