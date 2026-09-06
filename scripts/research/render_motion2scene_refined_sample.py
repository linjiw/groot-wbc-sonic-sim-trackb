#!/usr/bin/env python3
"""Export a fixed-plan fresh sampler run after verifying its raw audit and selection."""

import argparse
import json
from pathlib import Path

from motion2scene_refinement_study import load
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import verdict
import numpy as np
from render_motion2scene_refinement import slack
from render_motion2scene_timing_report import portable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    result_path = Path(plan["out"]) / "result.json"
    sample = json.loads(result_path.read_text())
    for key in ["registration", "sampler"]:
        assert plan[key] == sample[key]
        checked(Path(sample[key]["path"]), sample[key]["sha256"])
    for key in ["case_id", "checkpoint_seed", "draw_seed", "count"]:
        assert plan[key] == sample[key]
    reg, cases, _ = load(Path(sample["registration"]["path"]))
    case = cases[sample["case_id"]]
    raw = np.load(
        checked(Path(sample["trace"]["path"]), sample["trace"]["sha256"]), allow_pickle=False
    )
    checked(Path(sample["checkpoint"]["path"]), sample["checkpoint"]["sha256"])
    np.testing.assert_array_equal(raw["offsets"], reg["audit_offsets"])
    np.testing.assert_allclose(raw["scenes"][0], sample["initial_scenes"], rtol=0, atol=1e-14)
    selected = np.argmax(slack(raw["clearances"]), axis=0)
    np.testing.assert_array_equal(selected, raw["selected"])
    np.testing.assert_array_equal(raw["outputs"], raw["scenes"][selected, np.arange(8)])
    np.testing.assert_array_equal(raw["outputs"], sample["refined_scenes"])
    assert np.isfinite(raw["audit_clearances"]).all()
    valid = verdict(raw["audit_clearances"], case["mask"].numpy()).all(1)
    assert sample["accepted_indices"] == np.flatnonzero(valid).tolist()
    assert sample["accepted_count"] == int(valid.sum())
    np.testing.assert_array_equal(sample["target_min_m"], raw["audit_clearances"][:, :, 1].min(1))
    np.testing.assert_array_equal(sample["neutral_max_m"], raw["audit_clearances"][:, :, 0].max(1))
    outputs = []
    for name, value in [("refinement-sampler-plan", plan), ("refinement-fresh-sampler", sample)]:
        path = args.out / "evidence" / (name + ".json")
        write_new(path, portable(value))
        outputs.append(artifact(path))
    write_new(
        args.out / "assets/refinement-sampler-manifest.json",
        {
            "plan": artifact(args.plan),
            "source": artifact(result_path),
            "renderer": artifact(Path(__file__)),
            "outputs": outputs,
        },
    )
    print(json.dumps({"accepted": int(valid.sum()), "count": 8, "verified": True}))


if __name__ == "__main__":
    main()
