#!/usr/bin/env python3
"""Audit the declared 20 mm local-certificate cap against the original 100 mm oracle."""

import argparse
import json
from pathlib import Path

from motion2scene_fresh_sources import load_cases
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_uncertainty_learning import numpy_perturbed
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_local_pose_certificate import (
    local_pose_certificate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = json.loads(args.result.read_text())
    ref = result["registration"]
    registration = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    source_ref = registration["source_result"]
    source = json.loads(checked(Path(source_ref["path"]), source_ref["sha256"]).read_text())
    _, _, cases = load_cases(Path(registration["source_registration"]["path"]))
    jobs = {(r["seed"], r["case_id"]): r for r in source["rows"] if r["arm"] == "pattern"}
    rows = []
    for row in result["rows"]:
        case = cases[row["case_id"]]
        mask = case["mask"].numpy().astype(bool)
        reference = numpy_perturbed(
            np.array([row["scene"]]),
            np.zeros((1, 4)),
            case["states"],
            case["route"],
            case["yaw"].item(),
        )[0, 0]
        job_ref = jobs[(row["seed"], row["case_id"])]["raw"]
        with np.load(checked(Path(job_ref["path"]), job_ref["sha256"]), allow_pickle=False) as raw:
            zero = np.flatnonzero((raw["offsets"] == 0).all(1))[0]
            saved = raw["clearances"][row["output_index"], zero]
        ref = row["raw"]
        with np.load(checked(Path(ref["path"]), ref["sha256"]), allow_pickle=False) as raw:
            cap_error = float(np.abs(raw["nominal"] - np.minimum(reference, 0.02)).max())
            cert = local_pose_certificate(
                float(raw["nominal"][mask].min()),
                float(raw["nominal"][~mask].max()),
                registration["half_extents_m"],
            )
            cert_matches = cert == row["certificate"]
            valid = (raw["scores"][:, mask].min(-1) >= 0.01) & (
                raw["scores"][:, ~mask].max(-1) <= -0.01
            )
            probes_match = (
                len(valid) == row["diagnostic_probes"] and int(valid.sum()) == row["passing_probes"]
            )
        rows.append(
            {
                "case_id": row["case_id"],
                "seed": row["seed"],
                "output_index": row["output_index"],
                "original_100mm_oracle_error_m": float(np.abs(reference - saved).max()),
                "capped_20mm_oracle_error_m": cap_error,
                "certificate_matches": cert_matches,
                "probe_counts_match": probes_match,
            }
        )
    write_new(
        args.out,
        {
            "original_result": artifact(args.result),
            "audit_driver": artifact(Path(__file__)),
            "original_p1_reproduction_prediction_passed": result["predictions"]["p1"],
            "diagnosis": (
                "The new protocol/driver used a 20 mm clearance cap; the original fresh-source "
                "oracle uses 100 mm. Direct uncapped comparison was therefore an invalid "
                "reproduction predicate above 20 mm. Capping is 1-Lipschitz and above the "
                "required 10 mm margin, so it preserves these conservative local-domain bounds."
            ),
            "rows": rows,
            "all_original_scores_reproduced": len(rows) == 384
            and all(r["original_100mm_oracle_error_m"] <= 1e-12 for r in rows),
            "all_capped_scores_reproduced": len(rows) == 384
            and all(r["capped_20mm_oracle_error_m"] <= 1e-12 for r in rows),
            "all_certificates_and_probes_reproduced": all(
                r["certificate_matches"] and r["probe_counts_match"] for r in rows
            ),
            "additional_whole_motion_queries": 768,
            "new_physics_runs": 0,
            "historical_results_modified": False,
        },
    )


if __name__ == "__main__":
    main()
