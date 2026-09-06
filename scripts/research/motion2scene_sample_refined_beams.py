#!/usr/bin/env python3
"""Sample eight beams, refine within fixed bounds, audit once and retain refusals."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_refinement_study import load
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_uncertainty_learning import numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_events import (
    EventMixture,
    sample_scenes,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import route_centers
from gear_sonic.dataset_generation.hallucination.motion2scene_refinement import bounded_refine
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--checkpoint-seed", required=True, type=int)
    parser.add_argument("--draw-seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    reg, cases, _ = load(args.registration)
    if args.case not in cases or args.checkpoint_seed not in reg["seeds"]:
        raise ValueError("unregistered case or checkpoint")
    cells = [json.loads(Path(ref["path"]).read_text()) for ref in reg["cells"]]
    cell = next(c for c in cells if c["seed"] == args.checkpoint_seed)
    checked(Path(cell["checkpoint"]["path"]), cell["checkpoint"]["sha256"])
    saved = torch.load(cell["checkpoint"]["path"], map_location="cpu", weights_only=False)
    torch.set_num_threads(2)
    model = EventMixture(width=32).double()
    model.load_state_dict(saved["model"])
    model.eval()
    case = cases[args.case]
    feature = normalized(raw_features({args.case: case}), saved["config"])[args.case]
    with torch.no_grad():
        initial = sample_scenes(model(feature), 256, torch.Generator().manual_seed(args.draw_seed))[
            :8
        ]
    setup_seconds = time.monotonic() - start
    tick = time.monotonic()

    def query(scenes):
        return perturbed_clearances(
            scenes,
            tensor(reg["search_offsets"]),
            case["clouds"],
            case["rt"],
            case["progress"],
            case["yaw"],
            DOMAIN,
        )

    scenes, trace = bounded_refine(initial, query)
    refinement_seconds = time.monotonic() - tick
    tick = time.monotonic()
    offsets = np.array(reg["audit_offsets"])
    values = numpy_perturbed(
        scenes.numpy(), offsets, case["states"], case["route"], case["yaw"].item()
    )
    with torch.no_grad():
        other = torch.cat(
            [
                perturbed_clearances(
                    scenes[:2],
                    tensor(chunk),
                    case["clouds"],
                    case["rt"],
                    case["progress"],
                    case["yaw"],
                    DOMAIN,
                )
                for chunk in np.array_split(offsets, 23)
            ],
            dim=1,
        ).numpy()
    error = float(abs(values[:2] - other).max())
    if not np.isfinite(values).all() or not np.isfinite(error) or error > 1e-8:
        raise ValueError("invalid or disagreeing audit")
    valid = verdict(values, case["mask"].numpy()).all(1)
    audit_seconds = time.monotonic() - tick
    origin = np.loadtxt(case["metadata"]["parent"]["path"], delimiter=",")[0, :2]
    xy = route_centers(scenes[:, 0], case["rt"], case["progress"]).numpy() + origin
    beams = [
        {
            "proposal_index": i,
            "world_center_xyz": [*xy[i].tolist(), float(scenes[i, 1]) + 0.05],
            "world_yaw_rad": case["yaw"].item(),
            "size_xyz_m": [0.1, 1.2, 0.1],
            "reference_audit_accepted": bool(valid[i]),
        }
        for i in range(8)
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "trace.npz"
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{k: v.numpy() for k, v in trace.items()},
            outputs=scenes.numpy(),
            offsets=offsets,
            audit_clearances=values
        )
    write_new(
        args.out / "result.json",
        {
            "registration": artifact(args.registration),
            "sampler": artifact(Path(__file__)),
            "checkpoint": cell["checkpoint"],
            "case_id": args.case,
            "checkpoint_seed": args.checkpoint_seed,
            "draw_seed": args.draw_seed,
            "count": 8,
            "initial_scenes": initial.numpy().tolist(),
            "refined_scenes": scenes.numpy().tolist(),
            "beams": beams,
            "accepted_indices": np.flatnonzero(valid).tolist(),
            "accepted_count": int(valid.sum()),
            "target_min_m": values[:, :, 1].min(1).tolist(),
            "neutral_max_m": values[:, :, 0].max(1).tolist(),
            "trace": artifact(path),
            "query_error_m": error,
            "setup_seconds": setup_seconds,
            "refinement_seconds": refinement_seconds,
            "audit_seconds": audit_seconds,
            "search_queries": 4624,
            "audit_queries": 1808,
            "execution_eligible": False,
            "training_eligible": False,
            "role": "fresh_development_draws_fixed_count_no_retries",
        },
    )
    print(json.dumps({"accepted": int(valid.sum()), "count": 8, "output": str(args.out)}))


if __name__ == "__main__":
    main()
