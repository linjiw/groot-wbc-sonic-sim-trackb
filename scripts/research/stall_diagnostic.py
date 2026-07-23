#!/usr/bin/env python3
"""Stall diagnostic: capacity-limited vs curriculum-limited (plan D12).

If frontier bins stall under the ZPD teacher, this analysis must run BEFORE any
sampler-knob revision. The expert's transferable result (SONIC_RESPONSE.md,
closing note): their teacher walked the frontier to the policy's *execution
ceiling* and stalled there — the diagnostic that exonerated the teacher was
per-step accuracy vs reach geometry. Our analogue: fit the per-step
tracking-error growth rate within each stalled bin against bin length.

Interpretation rule (frozen here, before any M5 run):
- CAPACITY-LIMITED: error grows steadily from episode start within the bin —
  the fitted per-step growth rate is positive and the projected error crosses
  the termination threshold at a horizon shorter than the bin, regardless of
  where the episode starts. More sampling of this bin cannot help; the policy's
  execution ceiling is binding. The teacher is exonerated.
- CURRICULUM-LIMITED: error stays flat/bounded for most of the bin and only
  spikes near a localized segment, or the growth rate is near zero while the
  bin still fails — failure is concentrated where practice is missing, so
  allocation (or RSI placement) is the lever. The teacher (or its evidence) is
  implicated.

Input: per-bin, per-step tracking-error series recorded at eval time
(JSON; see ``--input-json`` schema below). The recorder side does not exist
yet — it rides along with the M5 eval runs on robotixx; this script and its
tests freeze the classification rule so the interpretation is preregistered.

Input schema (one record per evaluated bin)::

    {
      "records": [
        {
          "bin_id": 123,               # global bin index
          "motion_key": "walk_x1.5",
          "bin_length": 50,            # frames
          "failed": true,              # early-terminated within this bin
          "error_series": [0.02, ...], # per-step tracking error (anchor_pos or
                                       # the gating termination metric), from
                                       # bin entry to exit/termination
        }, ...
      ],
      "termination_threshold": 0.15    # the verifier bound the series is judged by
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Classification constants (frozen; changing them is a preregistration change).
_MIN_SERIES_STEPS = 8  # too-short series are "insufficient_data", never classified
_FLAT_GROWTH_FRACTION = 0.25  # |slope|*bin_length < this fraction of threshold => flat
_LOCALIZED_SPIKE_QUANTILE = 0.75  # spike must start after this fraction of the series


def fit_error_growth(error_series: list[float]) -> dict[str, float]:
    """Least-squares per-step growth rate of a tracking-error series."""
    y = np.asarray(error_series, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    return {
        "slope_per_step": float(slope),
        "intercept": float(intercept),
        "residual_std": float(residuals.std()),
        "final_error": float(y[-1]),
        "max_error": float(y.max()),
    }


def classify_bin(record: dict[str, Any], termination_threshold: float) -> dict[str, Any]:
    """Classify one stalled bin as capacity- or curriculum-limited (D12 rule)."""
    series = record.get("error_series") or []
    result: dict[str, Any] = {
        "bin_id": record.get("bin_id"),
        "motion_key": record.get("motion_key"),
        "failed": bool(record.get("failed", False)),
    }
    if len(series) < _MIN_SERIES_STEPS:
        result["classification"] = "insufficient_data"
        return result

    bin_length = float(record.get("bin_length") or len(series))

    # Where does the series first cross the threshold (if it does)?
    y = np.asarray(series, dtype=float)
    crossings = np.nonzero(y >= termination_threshold)[0]
    crossing_position = float(crossings[0]) / len(y) if len(crossings) else None
    result["threshold_crossing_position"] = crossing_position

    # Fit growth on the PRE-crossing portion: the drift the policy exhibits
    # before the verifier fires. Fitting through the spike itself would read
    # every localized failure as steep "growth" and misclassify it as capacity.
    fit_end = int(crossings[0]) if len(crossings) else len(y)
    fit_series = series if fit_end < _MIN_SERIES_STEPS else series[:fit_end]
    fit = fit_error_growth(fit_series)
    result["fit"] = fit

    # Projected error accumulated over one full bin traversal at the fitted rate.
    projected_growth = fit["slope_per_step"] * bin_length
    growth_is_flat = abs(projected_growth) < _FLAT_GROWTH_FRACTION * termination_threshold

    if not result["failed"]:
        result["classification"] = "not_stalled"
    elif growth_is_flat and (
        crossing_position is None or crossing_position >= _LOCALIZED_SPIKE_QUANTILE
    ):
        # Error bounded for most of the bin; failure localized late => the
        # missing skill is a specific segment, an allocation/RSI problem.
        result["classification"] = "curriculum_limited"
    elif projected_growth >= termination_threshold:
        # Steady drift that crosses the verifier within one traversal from any
        # start point => execution ceiling; more visits cannot fix it.
        result["classification"] = "capacity_limited"
    else:
        result["classification"] = "ambiguous"
    return result


def diagnose(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the D12 classification over all recorded bins and aggregate."""
    threshold = float(payload["termination_threshold"])
    records = payload.get("records") or []
    classified = [classify_bin(r, threshold) for r in records]
    counts: dict[str, int] = {}
    for c in classified:
        counts[c["classification"]] = counts.get(c["classification"], 0) + 1

    failed = [c for c in classified if c["failed"]]
    capacity = counts.get("capacity_limited", 0)
    curriculum = counts.get("curriculum_limited", 0)
    if not failed:
        verdict = "no_stall"
    elif capacity > 2 * curriculum:
        verdict = "capacity_limited"  # teacher exonerated (expert's case)
    elif curriculum > 2 * capacity:
        verdict = "curriculum_limited"  # sampler/RSI implicated
    else:
        verdict = "mixed_or_ambiguous"

    return {
        "schema_version": 1,
        "kind": "stall_diagnostic",
        "is_measurement": True,  # consumes real eval recordings (unlike the sims)
        "termination_threshold": threshold,
        "counts": counts,
        "verdict": verdict,
        "rule": (
            "capacity_limited iff fitted per-step error growth projected over one bin "
            f"crosses the threshold; curriculum_limited iff growth is flat (<{_FLAT_GROWTH_FRACTION}"
            "x threshold per traversal) and failure is localized late in the bin. "
            "Verdict requires a 2:1 majority among failed bins; otherwise mixed."
        ),
        "bins": classified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    result = diagnose(payload)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"verdict: {result['verdict']} ({result['counts']})")
    print(f"wrote stall diagnostic to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
