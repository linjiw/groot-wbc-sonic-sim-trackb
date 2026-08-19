#!/usr/bin/env python
"""Retime the adapted clips that cleared their obstacle and were rejected for arriving short.

The 2026-08-19 banded batch produced the counterfactual it was designed for and could not bank it.
`n_013_ceiling_overhead_left` walks its nominal torso into the ceiling at 1543.6 N and clears the
same ceiling adapted at 49 N -- and *both* adapted cells were rejected, for
`reference_endpoint_tracking_error` rather than for contact. The cause is in the operator: a local
crouch lowers the root and leaves the forward schedule alone, so the reference asks a crouched G1
to walk at an upright pace and it arrives 0.52 m short of a 0.35 m budget. Since a ceiling can only
be relieved by lowering the robot, that bounds the whole overhead band structurally.

This script does not touch the causal reference and does not move the gate. It writes a *second*
artifact from the same pair -- the deployable skill -- by retiming the adapted clip so the stretch
where the adaptation is active is commanded at the pace the robot was measured to deliver. Route,
start, goal, joint angles at each route position and the obstacle's station in route progress are
all held; duration is not, which is exactly the thing a matched counterfactual must hold and a
deployable skill must not.

The pace comes from the family's own rollout: the adapted cell's endpoint error minus the
nominal's, divided by the arclength the adaptation was active over. Nothing is inferred from the
clip -- four attempts to predict trackability from a clip rather than measure it are recorded as
wrong in docs/prediction_register.md.

Selection is deliberately narrow. A family is retimed only when its adapted cell failed on
*tracking alone*; one that also struck the obstacle has an operator problem, not a transport
problem, and slowing it down would hide that behind a longer clip.

Usage::

    python scripts/research/build_deployable_skill.py \\
        --scores /data/.../wsA/family_scores.json \\
        --clips /data/.../wsA/clips \\
        --nominals /data/.../taxonomy/motions_4s \\
        --out /data/.../wsA/deployable

    # or one pair directly, for a sweep across paces rather than a measured correction
    python scripts/research/build_deployable_skill.py \\
        --nominal 013.csv --adapted n_013_ceiling_overhead_left.csv --ratio 0.7 --out DIR
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))

import batch_preflight as preflight  # noqa: E402

from gear_sonic.dataset_generation.deployable_retiming import (  # noqa: E402
    RETIME_SAFETY,
    deployable_clip,
)

#: Which measured lag curve belongs to which operator. The curves live in `batch_preflight` and are
#: read from there rather than copied: two copies of a measured quantity drift, and this one is
#: measured rather than mirrored from a builder, so there is nothing for a duplicate to police.
LAG_CURVE = {
    "local_crouch": "CROUCH_LAG_CURVE",
    "local_arm_tuck": "TUCK_LAG_CURVE",
}


def predicted_excess_lag_m(operator: str, excursion_rad: float) -> float | None:
    """What the amplitude sweeps say an edit of this size costs in forward progress.

    The curves give *absolute* endpoint lag, and their value at zero excursion is the clip's own
    tracking lag rather than anything the adaptation caused. The excess is therefore the difference
    from that baseline, which is a property of the operator and transfers between nominals; the
    baseline is not, and would import one clip's tracking behaviour into another's correction.

    This is a prediction, and it is used only where no rollout exists yet. On the one family where
    both are available it reads 0.252 m against a measured 0.307 m -- 82% -- so it under-corrects,
    in the same direction and for the same reason the geometric delivery ratio does. A clip retimed
    from a prediction should be expected to need a second pass.
    """
    name = LAG_CURVE.get(operator)
    if name is None:
        return None
    curve = getattr(preflight, name)
    return preflight.interpolate(curve, excursion_rad) - preflight.interpolate(curve, 0.0)


#: The rejection reason this script exists to answer. Any other reason on the adapted cell means
#: the clip failed at something retiming does not address.
TRANSPORT_REASONS = {"reference_endpoint_tracking_error", "reference_path_tracking_error"}

#: The cell the correction is measured from. The easy scene, because there the adapted clip is
#: alone with its own transport cost: nothing it might have struck is in the room, so the whole
#: endpoint error is the pace.
MEASURED_FROM = "adapted_easy"
BASELINE = "nominal_easy"


def nominal_csv(nominals: Path, family: str) -> Path | None:
    """Find the taxonomy clip a family was bent from, by its index."""
    index = family.split("_")[1] if "_" in family else family
    matches = sorted(glob.glob(str(nominals / f"{index}_*.csv")))
    return Path(matches[0]) if matches else None


def selectable(cells: dict) -> tuple[bool, str]:
    """Whether retiming is the right answer for this family, and why not when it is not."""
    adapted = cells.get(MEASURED_FROM)
    nominal = cells.get(BASELINE)
    if adapted is None or nominal is None:
        return False, "no easy-scene cells"
    if adapted.get("accepted"):
        return False, "the adapted clip already tracks"
    reasons = set(adapted.get("reasons", ()))
    if not reasons:
        return False, "rejected without a reason"
    if not reasons <= TRANSPORT_REASONS:
        return False, f"not a transport failure: {sorted(reasons - TRANSPORT_REASONS)}"
    if not nominal.get("accepted"):
        return False, "the nominal does not survive its own easy scene"
    excess = adapted["endpoint_error_m"] - nominal["endpoint_error_m"]
    if excess <= 0.0:
        return False, "the adapted clip already tracks better than its nominal"
    return True, ""


def write_pair(
    out: Path, name: str, retimed: np.ndarray, report, extra: dict | None = None
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{name}_deployable.csv"
    np.savetxt(csv, retimed, delimiter=",", fmt="%.6f")
    record = {"clip": csv.name, "source": name, **asdict(report)}
    record["holds_route"] = report.holds_route
    if extra:
        record.update(extra)
    (out / f"{name}_deployable.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="family_scores.json from a scored batch")
    parser.add_argument("--clips", type=Path, help="directory of adapted clips, one per family")
    parser.add_argument(
        "--nominals", type=Path, help="taxonomy motions the families were bent from"
    )
    parser.add_argument("--nominal", type=Path, help="single-pair mode: the nominal clip")
    parser.add_argument("--adapted", type=Path, help="single-pair mode: the adapted clip")
    parser.add_argument("--ratio", type=float, help="command this pace instead of a measured one")
    parser.add_argument("--excess-lag", type=float, help="single-pair mode: the measured shortfall")
    parser.add_argument(
        "--predict",
        choices=tuple(LAG_CURVE),
        help="single-pair mode: derive the shortfall from the amplitude sweeps instead of a "
        "rollout, for a clip that has never been run. Under-corrects; expect a second pass.",
    )
    parser.add_argument("--safety", type=float, default=RETIME_SAFETY)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.nominal and args.adapted:
        given = sum(x is not None for x in (args.ratio, args.excess_lag, args.predict))
        if given != 1:
            parser.error("give exactly one of --ratio, --excess-lag or --predict")
        nominal = np.loadtxt(args.nominal, delimiter=",")
        adapted = np.loadtxt(args.adapted, delimiter=",")
        if args.predict:
            excursion = float(np.abs(adapted[:, 7:] - nominal[:, 7:]).max())
            args.excess_lag = predicted_excess_lag_m(args.predict, excursion)
            print(
                f"predicted from the {args.predict} sweep: {excursion:.4f} rad of excursion "
                f"costs {args.excess_lag:.4f} m -- a prediction, not a measurement"
            )
            if args.excess_lag <= 0.0:
                print("the sweep says this edit costs no forward progress; nothing to retime")
                return 0
        retimed, report = deployable_clip(
            nominal,
            adapted,
            ratio=args.ratio,
            excess_lag_m=args.excess_lag,
            safety=args.safety,
        )
        csv = write_pair(args.out, args.adapted.stem, retimed, report)
        print(
            f"{args.adapted.stem}: {report.source_frames} -> {report.frames} frames at "
            f"{report.commanded_ratio:.3f} of nominal pace, route held to "
            f"{report.route_deviation_m * 1000:.2f} mm"
        )
        print(f"wrote {csv}")
        return 0

    if not (args.scores and args.clips and args.nominals):
        parser.error("give --scores, --clips and --nominals, or --nominal and --adapted")

    families = json.loads(args.scores.read_text())
    print(f"{'family':38s}{'pace':>7s}{'frames':>10s}{'window':>9s}{'route':>8s}  note")
    written = 0
    for entry in families:
        name = entry["family"]
        keep, why = selectable(entry.get("cells", {}))
        if not keep:
            print(f"{name:38s}{'':7s}{'':10s}{'':9s}{'':8s}  skipped -- {why}")
            continue

        adapted_csv = args.clips / f"{name}.csv"
        source = nominal_csv(args.nominals, name)
        if not adapted_csv.exists() or source is None:
            missing = adapted_csv.name if not adapted_csv.exists() else "its nominal"
            print(f"{name:38s}{'':7s}{'':10s}{'':9s}{'':8s}  skipped -- {missing} not found")
            continue

        cells = entry["cells"]
        excess = cells[MEASURED_FROM]["endpoint_error_m"] - cells[BASELINE]["endpoint_error_m"]
        retimed, report = deployable_clip(
            np.loadtxt(source, delimiter=","),
            np.loadtxt(adapted_csv, delimiter=","),
            excess_lag_m=excess,
            safety=args.safety,
        )
        write_pair(
            args.out,
            name,
            retimed,
            report,
            {
                "operator": entry.get("operator"),
                "nominal_clip": source.name,
                "measured_endpoint_error_m": cells[MEASURED_FROM]["endpoint_error_m"],
                "nominal_endpoint_error_m": cells[BASELINE]["endpoint_error_m"],
            },
        )
        written += 1
        note = "FLOORED -- expected to still arrive short" if report.floored else ""
        if not report.holds_route:
            note = "route moved -- do not reuse the family's scene"
        print(
            f"{name:38s}{report.commanded_ratio:7.3f}"
            f"{report.source_frames:5d}->{report.frames:<4d}"
            f"{report.weighted_window_m:9.3f}"
            f"{report.route_deviation_m * 1000:7.2f}mm  {note}"
        )

    print(f"\nwrote {written} deployable clip(s) to {args.out}")
    if written:
        print(
            "these are skills, not counterfactual evidence: the causal references they were "
            "retimed from are unchanged and remain what the 2x2 is reported on.\n"
            "next: convert with gear_sonic/data_process/convert_kimodo_to_motion_lib.py "
            "(--source-fps 30, the frame count carries the retiming) and roll out against the "
            "family's own scenes with --max-steps auto."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
