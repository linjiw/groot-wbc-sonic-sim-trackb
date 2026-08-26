#!/usr/bin/env python3
"""Fit and cross-validate D_phi: executed overhead reach from a reference clip.

The critical window is an order statistic over *executed* reach, so proposing a scene for a
freshly generated motion costs two rollouts before any geometry can be authored.  This model
predicts executed reach from the reference clip's own kinematic reach, which is what lets an
amplitude be chosen -- and a window checked for emptiness -- before the spend.

Evaluation is leave-one-motion-out against an explicit baseline.  The baseline is *identity*
(executed = commanded), because that is what the pipeline assumes today whenever it reasons about
a reference clip.  A model that cannot beat identity is not worth carrying.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def _rows(path: Path, *, empty_only: bool, accepted_only: bool) -> list[dict]:
    rows = []
    for row in csv.DictReader(path.open()):
        if accepted_only and row["accepted"] != "1":
            continue
        if empty_only and row["scene"] != "screen_empty":
            continue
        rows.append(row)
    return rows


def _errors(x: np.ndarray, y: np.ndarray, motion: np.ndarray, fit, predict) -> np.ndarray:
    errors = []
    for held in sorted(set(motion)):
        test = motion == held
        train = ~test
        if train.sum() < 3:
            continue
        parameters = fit(x[train], y[train])
        errors.extend(predict(parameters, x[test]) - y[test])
    return np.asarray(errors) * 1000.0


MODELS = {
    "identity": (lambda x, y: None, lambda p, x: x),
    "constant_offset": (lambda x, y: float((y - x).mean()), lambda p, x: x + p),
    "linear": (lambda x, y: np.polyfit(x, y, 1), lambda p, x: np.polyval(p, x)),
    "executed_mean": (lambda x, y: float(y.mean()), lambda p, x: np.full_like(x, p)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=REPO_ROOT / "docs/hallucination/reach_response_corpus.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/reach_delivery_model.json"
    )
    parser.add_argument("--include-scene-cells", action="store_true")
    args = parser.parse_args()

    rows = _rows(args.corpus, empty_only=not args.include_scene_cells, accepted_only=True)
    x = np.asarray([float(row["commanded_reach_m"]) for row in rows])
    y = np.asarray([float(row["executed_reach_m"]) for row in rows])
    motion = np.asarray([row["reference_csv"] for row in rows])
    if len(set(motion)) < 5:
        raise SystemExit("too few distinct reference clips to cross-validate")

    scores = {}
    for name, (fit, predict) in MODELS.items():
        errors = _errors(x, y, motion, fit, predict)
        scores[name] = {
            "rmse_mm": float(np.sqrt((errors**2).mean())),
            "median_abs_mm": float(np.median(np.abs(errors))),
            "q90_abs_mm": float(np.quantile(np.abs(errors), 0.9)),
            "held_out_points": int(errors.size),
        }

    baseline = scores["identity"]["rmse_mm"]
    best = min(scores, key=lambda name: scores[name]["rmse_mm"])
    slope, intercept = np.polyfit(x, y, 1)
    residual = (y - np.polyval((slope, intercept), x)) * 1000.0

    report = {
        "schema_version": "lfh_reach_delivery_model_v1",
        "role": "proposal-only screening estimate; physics remains the verdict",
        "population": {
            "condition": "accepted rollouts"
            + ("" if args.include_scene_cells else ", empty scene only"),
            "rows": len(rows),
            "distinct_reference_clips": len(set(motion)),
        },
        "cross_validation": {
            "scheme": "leave-one-motion-out",
            "baseline": "identity (executed = commanded reference reach)",
            "scores": scores,
            "best_model": best,
            "rmse_reduction_vs_identity": float(1.0 - scores[best]["rmse_mm"] / baseline),
        },
        "fitted_linear": {
            "slope": float(slope),
            "intercept_m": float(intercept),
            "in_sample_residual_sd_mm": float(residual.std(ddof=1)),
            "in_sample_residual_q90_abs_mm": float(np.quantile(np.abs(residual), 0.9)),
        },
        "caveats": [
            "fitted on empty-scene accepted rollouts; it does not describe a drifting execution",
            "predictive band is the leave-one-motion-out q90, not a calibrated confidence interval",
            "a predicted window still requires executed physics before any scene is promoted",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} rows / {len(set(motion))} clips -> {args.out}")
    for name, score in sorted(scores.items(), key=lambda item: item[1]["rmse_mm"]):
        print(
            f"  {name:16s} RMSE {score['rmse_mm']:6.2f} mm  "
            f"median {score['median_abs_mm']:6.2f}  q90 {score['q90_abs_mm']:6.2f}"
        )
    print(
        f"best={best}; RMSE reduction vs identity "
        f"{100 * report['cross_validation']['rmse_reduction_vs_identity']:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
