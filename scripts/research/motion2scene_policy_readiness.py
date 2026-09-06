#!/usr/bin/env python3
"""Package geometry-only decision features; expose missing physics alternatives."""

import json
from pathlib import Path

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_policy_features import (
    feasibility_label,
    packet_features,
)

CONDITIONS = (
    "nominal",
    "height_minus10mm",
    "height_plus10mm",
    "station_minus15cm",
    "station_plus15cm",
    "absent",
    "raised",
    "blocked",
)


def main():
    data = ROOT.parent / "research-data/groot-wbc"
    source = data / "m2s-overhang-variation-v1/result.json"
    out = data / "m2s-policy-readiness-v1"
    result = json.loads(source.read_text())
    if len(result["rows"]) != 42 or not result["predictions"]["p5_measurement_contract"]:
        raise ValueError("requires complete measurement-valid variation panel")
    out.mkdir(parents=True, exist_ok=False)
    refs = [
        artifact(source),
        artifact(Path(__file__)),
        artifact(
            ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_policy_features.py"
        ),
    ]
    refs += [r["sensor"] for r in result["rows"]]
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "conditions": CONDITIONS,
            "seeds": [8041, 8042],
            "feature_count": 144,
            "decision_time_s": 0.2,
            "training_eligible": False,
            "purpose": "Data-contract readiness on one observed development ancestor; no fitting",
        },
    )
    rows = []
    features = []
    for seed in (8041, 8042):
        for condition in CONDITIONS:
            cells = [r for r in result["rows"] if r["seed"] == seed and r["condition"] == condition]
            neutral = next((r for r in cells if r["mode"] == "blind"), None)
            if neutral is None:
                candidate = next(r for r in cells if r["mode"] == "reactive")
                sensor = json.loads(Path(candidate["sensor"]["path"]).read_text())
                if not candidate["switches"] and all(
                    o["active"] == 0 for o in sensor["observations"]
                ):
                    neutral = candidate
            oracle = next((r for r in cells if r["mode"] == "oracle"), None)
            if oracle is not None and not any(
                s["to"] == 1 and 0.2 <= s["time_s"] <= 0.4 for s in oracle["switches"]
            ):
                raise ValueError("oracle did not execute crouch")
            feature_cell = neutral or next(r for r in cells if r["mode"] == "reactive")
            sensor = json.loads(Path(feature_cell["sensor"]["path"]).read_text())
            packet = next(o for o in sensor["observations"] if o["time_s"] == 0.2)
            if any(s["time_s"] < 0.2 for s in feature_cell["switches"]):
                raise ValueError("post-treatment features")
            features.append(packet_features(packet))
            label = feasibility_label(
                None if neutral is None else neutral["pass"],
                None if oracle is None else oracle["pass"],
            )
            rows.append(
                {
                    "source_ancestor": 41002,
                    "physics_seed": seed,
                    "condition": condition,
                    "label": label,
                    "neutral_cell": neutral["cell_id"] if neutral else None,
                    "crouch_cell": oracle["cell_id"] if oracle else None,
                    "feature_cell": feature_cell["cell_id"],
                    "sensor": feature_cell["sensor"],
                    "feature_row": len(features) - 1,
                    "split": "interface_development_only",
                }
            )
    feature_path = out / "decision_features.npz"
    np.savez_compressed(feature_path, x=np.stack(features))
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "source_result": artifact(source),
            "rows": rows,
            "features": artifact(feature_path),
            "complete_labels": sum(r["label"]["feasible"] is not None for r in rows),
            "requested_scene_seed_groups": 16,
            "missing_comparators": [
                {"condition": r["condition"], "seed": r["physics_seed"]}
                for r in rows
                if r["label"]["feasible"] is None
            ],
            "training_eligible": False,
            "learning_runs": 0,
            "scope": (
                "144 observed range/height/mask values only; metadata is separate. "
                "One development ancestor; no fresh-source or learning benefit claim."
            ),
        },
    )
    print(
        json.dumps(
            {"complete_labels": sum(r["label"]["feasible"] is not None for r in rows), "groups": 16}
        )
    )


if __name__ == "__main__":
    main()
