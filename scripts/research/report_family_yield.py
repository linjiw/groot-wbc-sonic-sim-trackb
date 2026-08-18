#!/usr/bin/env python3
"""Aggregate every counterfactual family attempt into a yield table.

A method claim needs a rate, not an example. Four things can be true of a family and they
are progressively harder, so reporting only the last one hides where the attrition is:

1. **geometry predicted** -- the two swept volumes separate, so a window exists on paper
2. **physics verified** -- the 2x2 came out as predicted when actually rolled out
3. **attribution pure** -- and the one rejection was the intended obstacle hitting the
   intended body, not drift or a wall or a fall
4. **perturbation robust** -- and it still holds from jittered start poses

Levels 1 and 2 disagreeing is the interesting number: it is how often the geometry lies, and
it is the honest measure of whether swept-volume prediction can stand in for a rollout. So
far it has disagreed once, and that turned out to be a plumbing fault rather than the
geometry being wrong -- which is itself worth recording, because a yield table that silently
counts infrastructure failures as method failures understates the method.

Usage::

    python scripts/research/report_family_yield.py --root /data/.../counterfactual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGES = (
    ("geometry_predicted", "geometry predicted a window"),
    ("physics_verified", "physics produced the 2x2"),
    ("attribution_pure", "the failure was the intended one"),
    ("perturbation_robust", "it survived a jittered start"),
)


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def assess(family_dir: Path) -> dict | None:
    family = read(family_dir / "family.json")
    if not family:
        return None
    attribution = read(family_dir / "attribution.json")
    robustness = read(family_dir / "robustness.json")
    diagnosis = read(family_dir / "diagnosis.json")

    window = family.get("window_m")
    record = {
        "family": family_dir.name,
        "family_id": family.get("family_id", ""),
        "regime": family.get("regime", "overhead"),
        "window_m": window,
        "geometry_predicted": bool(window and window > 0),
        "physics_verified": bool(family.get("counterfactual_established")),
        "attribution_pure": bool(attribution.get("attribution_pure")),
        "perturbation_robust": bool(robustness.get("perturbation_robust")),
        "perturbation_ran": bool(robustness),
        "cause": diagnosis.get("cause", ""),
        "note": diagnosis.get("note", ""),
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    records = [
        record for record in (assess(d) for d in sorted(args.root.iterdir()) if d.is_dir())
        if record is not None
    ]
    if not records:
        print(f"no family.json under {args.root}")
        return 1

    print(f"{len(records)} family attempt(s) under {args.root}\n")
    print(f"{'family':12s}{'regime':10s}{'window':>9s}"
          f"{'geom':>7s}{'phys':>7s}{'pure':>7s}{'robust':>8s}")
    for record in records:
        window = f"{record['window_m']:.3f}" if record["window_m"] else "-"
        robust = (
            "yes" if record["perturbation_robust"]
            else ("no" if record["perturbation_ran"] else "-")
        )
        print(
            f"{record['family'][:11]:12s}{record['regime'][:9]:10s}{window:>9s}"
            f"{('yes' if record['geometry_predicted'] else 'no'):>7s}"
            f"{('yes' if record['physics_verified'] else 'no'):>7s}"
            f"{('yes' if record['attribution_pure'] else 'no'):>7s}"
            f"{robust:>8s}"
        )

    print("\nfunnel:")
    total = len(records)
    for key, description in STAGES:
        passing = sum(1 for record in records if record[key])
        share = passing / total if total else 0.0
        print(f"  {passing:3d}/{total:<3d} ({share:4.0%})  {description}")

    predicted = [r for r in records if r["geometry_predicted"]]
    verified = [r for r in predicted if r["physics_verified"]]
    if predicted:
        print(f"\ngeometry-to-physics agreement: {len(verified)}/{len(predicted)} "
              f"({len(verified) / len(predicted):.0%})")
        print("  how often a predicted window survived contact with a rollout")

    # A run lost to a bug in the harness is not evidence about the geometry, and counting it
    # as such understates the method. Excluding it silently would overstate the method, so
    # both numbers are printed and every exclusion is named.
    plumbing = [r for r in predicted if not r["physics_verified"]
                and r["cause"] == "infrastructure"]
    if plumbing:
        remaining = [r for r in predicted if r not in plumbing]
        rate = len(verified) / len(remaining) if remaining else 0.0
        print(f"\nexcluding {len(plumbing)} attempt(s) diagnosed as harness faults: "
              f"{len(verified)}/{len(remaining)} ({rate:.0%})")
        for record in plumbing:
            print(f"  excluded {record['family']}: {record['note'] or 'no note recorded'}")
        print("  both rates are shown because either alone misleads")

    by_regime: dict[str, list] = {}
    for record in records:
        by_regime.setdefault(record["regime"], []).append(record)
    if len(by_regime) > 1:
        print("\nby regime:")
        for regime, group in sorted(by_regime.items()):
            robust = sum(1 for r in group if r["perturbation_robust"])
            print(f"  {regime:10s} {len(group):2d} attempted, {robust:2d} robust")

    if args.json:
        args.json.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
