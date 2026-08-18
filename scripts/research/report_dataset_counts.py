#!/usr/bin/env python
"""The five counts that must never be collapsed into one number.

Five shelf heights on a single nominal motion are five *scene instances*. They are not five
independent behaviour families, and reporting them as such would make the corpus look an order of
magnitude larger than the evidence it carries. So every count is emitted separately and the
inflation factor between them is printed, because that ratio is the thing a reader needs to judge
independence:

* **verified families** -- certified 2x2s, the unit of counterfactual evidence
* **distinct nominal motions** -- how many different journeys the corpus really covers
* **distinct adapted motions** -- how many different behaviours were constructed
* **distinct routes** -- how many different paths through space
* **scene instances** -- how many rooms were built, which is the largest and least meaningful

A family is counted as verified only when its four cells came out as a counterfactual requires:
both motions accepted in the easy scene, the nominal rejected in the hard one, the adapted motion
accepted there. A family whose cells all passed is a *failed* family -- no contrast -- and is
reported separately rather than dropped, since its cost was paid and its absence from the total is
what the yield number means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)


def verdict(directory: Path) -> str:
    paths = sorted(directory.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        return "missing"
    with open(paths[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))
    if payload is None:
        return "unevaluable"
    return classify_episode(directory.name, payload).outcome


def classify_family(cells: dict[str, str]) -> str:
    """A family is verified only when the 2x2 forms a counterfactual."""
    needed = ("nominal_easy", "nominal_hard", "adapted_easy", "adapted_hard")
    if any(cells.get(k, "missing") in ("missing", "unevaluable") for k in needed):
        return "incomplete"
    if (
        cells["nominal_easy"] == "accepted"
        and cells["adapted_easy"] == "accepted"
        and cells["nominal_hard"] == "rejected"
        and cells["adapted_hard"] == "accepted"
    ):
        return "verified"
    if cells["nominal_hard"] == "accepted":
        return "no_contrast"
    return "failed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--families", type=Path, required=True, help="directory of matched-family working dirs"
    )
    ap.add_argument(
        "--scenes", type=Path, default=REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rows = []
    for manifest in sorted(args.families.glob("*.json")):
        work = args.families / manifest.stem
        if not work.is_dir():
            continue
        data = json.loads(manifest.read_text())
        prefix = manifest.stem.replace("mf_", "").rsplit("_c", 1)[0]
        cells = {}
        for role, kinds in (
            ("nominal", ("nominal", "w_nominal")),
            ("adapted", ("crouch", "w_crouch08", "tuck")),
        ):
            for scene in ("easy", "hard"):
                found = "missing"
                for kind in kinds:
                    for name in (f"{prefix}_{kind}_{scene}", f"{kind}_{scene}"):
                        if (work / name).is_dir():
                            found = verdict(work / name)
                            break
                    if found != "missing":
                        break
                cells[f"{role}_{scene}"] = found
        rows.append(
            {
                "family_id": manifest.stem,
                "nominal_motion": prefix,
                "status": classify_family(cells),
                "window_m": round(float(data.get("window_m", 0.0)), 4),
                "cells": cells,
            }
        )

    verified = [r for r in rows if r["status"] == "verified"]
    counts = {
        "verified_families": len(verified),
        "distinct_nominal_motions": len({r["nominal_motion"] for r in verified}),
        "distinct_adapted_motions": len({f"{r['nominal_motion']}_adapted" for r in verified}),
        "distinct_routes": len({r["nominal_motion"] for r in verified}),
        "scene_instances": len(list(args.scenes.glob("*.usda"))),
        "families_attempted": len(rows),
        "no_contrast": len([r for r in rows if r["status"] == "no_contrast"]),
        "incomplete": len([r for r in rows if r["status"] == "incomplete"]),
    }

    print(f"{'count':>28s}  value")
    for key in (
        "verified_families",
        "distinct_nominal_motions",
        "distinct_adapted_motions",
        "distinct_routes",
        "scene_instances",
    ):
        print(f"{key:>28s}  {counts[key]}")
    print(
        f"\n{'families attempted':>28s}  {counts['families_attempted']}"
        f"   (no contrast: {counts['no_contrast']}, incomplete: {counts['incomplete']})"
    )
    if counts["distinct_nominal_motions"]:
        ratio = counts["scene_instances"] / counts["distinct_nominal_motions"]
        print(
            f"{'scene instances per motion':>28s}  {ratio:.1f}x"
            "   <- the inflation factor, and why these are reported apart"
        )
    print(f"\n{'family':>18s}{'status':>13s}{'window':>10s}  cells")
    for row in rows:
        cells = " ".join(
            f"{k.split('_')[0][0]}{k.split('_')[1][0]}={v[:4]}" for k, v in row["cells"].items()
        )
        print(
            f"{row['family_id']:>18s}{row['status']:>13s}{row['window_m'] * 1000:8.1f}mm  {cells}"
        )

    print(
        "\nAgainst the dataset target: "
        f"{counts['verified_families']}/24 verified families, "
        f"{counts['distinct_nominal_motions']}/8 nominal motions."
    )
    if args.json:
        args.json.write_text(json.dumps({"counts": counts, "families": rows}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
