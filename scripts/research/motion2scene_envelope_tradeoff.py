#!/usr/bin/env python3
"""Measure contrast availability against placement-robustness envelope size.

Post hoc CPU diagnostic over already recorded achieved transitions. It adds no
physics, no label and no proposal, and it changes no registered threshold. The
question is design-facing: the acquisition funnel refused 47/48 analytic and 48/48
learned slots under the inherited 113-offset audit, while nominal witnesses exist.
This maps how many centres survive as the audit envelope is scaled, so a later
separately registered contract can choose an envelope with a measured yield instead
of inheriting one.

Scaling the inherited offsets is a diagnostic sweep, not a new acceptance rule. A
surviving centre is a sampled finite witness on this grid, never a feasibility proof
and never a physical outcome.
"""

import argparse
import json
from pathlib import Path
import time

from motion2scene_development_bank import DATA
from motion2scene_timing_diagnostic import ROOT, artifact, write_new
from motion2scene_transition_construction import cases
from motion2scene_uncertainty_learning import numpy_perturbed
import numpy as np

PROTOCOL = ROOT / "docs/motion2scene/ENVELOPE_TRADEOFF_V1.md"
BANK = DATA / "m2s-development-transition-bank-v2"
REFINEMENT = DATA / "m2s-refinement-v1/registration.json"
SOURCES = (41001, 41002, 41003)
SCALES = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)
CLEARANCE_M = 0.01
CPU_CEILING_S = 1800


def grid():
    """Fixed diagnostic grid: 0.01 station steps, 2.5 mm height steps."""
    stations = np.round(np.arange(0.10, 0.9001, 0.01), 4)
    heights = np.round(np.arange(1.10, 1.45001, 0.0025), 5)
    return np.array([[s, h] for s in stations for h in heights])


def margins(values, mask):
    """Worst-case target clearance and walk interference over the offset set."""
    target = values[..., mask].min(-1).min(1)
    walk = values[..., ~mask].max(-1).max(1)
    return target, walk


def sweep(out):
    started = time.monotonic()
    bank = json.loads((BANK / "result.json").read_text())
    offsets = np.array(json.loads(REFINEMENT.read_text())["audit_offsets"])
    points = grid()
    rows, per_source = [], {}
    for source in SOURCES:
        case, _, _, _ = cases(source, bank)
        mask = np.asarray(case["mask"])
        route, yaw = case["route"], case["yaw"].item()
        nominal = numpy_perturbed(points, np.zeros((1, 4)), case["states"], route, yaw)[:, 0, :]
        target, walk = nominal[..., mask].min(-1), nominal[..., ~mask].max(-1)
        witness = (target >= CLEARANCE_M) & (walk <= -CLEARANCE_M)
        # Only nominal witnesses can survive a larger envelope: the offset set
        # contains the zero offset, so robustness is monotone in the envelope.
        assert np.allclose(offsets[np.abs(offsets).sum(1).argmin()], 0.0)
        candidates = points[witness]
        per_source[source] = {
            "nominal_witnesses": int(witness.sum()),
            "grid_points": int(len(points)),
            "best_nominal_joint_margin_m": (
                float(np.minimum(target, -walk)[witness].max()) if witness.any() else None
            ),
        }
        for scale in SCALES:
            if time.monotonic() - started > CPU_CEILING_S:
                raise TimeoutError("registered diagnostic ceiling")
            if not len(candidates):
                rows.append(
                    {"source": source, "scale": scale, "witnesses": 0, "best_joint_margin_m": None}
                )
                continue
            values = numpy_perturbed(candidates, offsets * scale, case["states"], route, yaw)
            tgt, wlk = margins(values, mask)
            joint = np.minimum(tgt, -wlk)
            survive = (tgt >= CLEARANCE_M) & (wlk <= -CLEARANCE_M)
            rows.append(
                {
                    "source": source,
                    "scale": float(scale),
                    "envelope_xy_mm": float(20 * scale),
                    "envelope_z_mm": float(10 * scale),
                    "envelope_yaw_rad": float(0.02 * scale),
                    "witnesses": int(survive.sum()),
                    "best_joint_margin_m": float(joint.max()),
                }
            )
    write_new(
        out / "envelope-tradeoff.json",
        {
            "protocol": artifact(PROTOCOL) if PROTOCOL.exists() else None,
            "bank": artifact(BANK / "result.json"),
            "refinement": artifact(REFINEMENT),
            "clearance_m": CLEARANCE_M,
            "grid": {
                "stations": "0.10:0.90 step 0.01",
                "heights": "1.100:1.450 step 0.0025",
                "points": int(len(points)),
            },
            "scales": list(SCALES),
            "per_source": {str(k): v for k, v in per_source.items()},
            "rows": rows,
            "seconds": time.monotonic() - started,
            "scope": (
                "Sampled finite witnesses on one grid over recorded seed-8721 achieved "
                "transitions; not a feasibility proof, not a new acceptance rule, and "
                "not a physical outcome"
            ),
            "analysis_only": True,
        },
    )
    for source in SOURCES:
        line = [r for r in rows if r["source"] == source]
        print(f"source {source}: nominal {per_source[source]['nominal_witnesses']} witnesses")
        for r in line:
            print(
                f"   envelope +-{r.get('envelope_xy_mm', 0):4.1f} mm xy "
                f"+-{r.get('envelope_z_mm', 0):4.1f} mm z: {r['witnesses']:5d} witnesses"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sweep(args.out)


if __name__ == "__main__":
    main()
