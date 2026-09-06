#!/usr/bin/env python3
"""Sample a saved uncertainty model and reject failures under its finite pose audit.

Uses the original development carrier only. Output is diagnostic geometry, not an
executable or certified robot scene. An independent NumPy checker audits all draws.
"""

import argparse
import json
from pathlib import Path
import time

from motion2scene_inverse_learning import DOMAIN, inputs
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_uncertainty_learning import load, numpy_perturbed, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import (
    BeamMixture,
    sample_latents,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("robust_full", "robust_no_kl"), default="robust_full")
    parser.add_argument("--model-seed", type=int, default=8121)
    parser.add_argument("--sample-seed", type=int, default=9421)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.out.exists():
        parser.error("count must be positive and output must not exist")
    manifest, source, states, route = load(args.manifest)
    completion = json.loads((args.manifest.parent / "runs/complete.json").read_text())
    checked(args.manifest, completion["manifest"]["sha256"])
    if args.model_seed not in manifest["seeds"]:
        parser.error("model seed is not registered")
    cell = args.manifest.parent / f"runs/{args.arm}_{args.model_seed}"
    result = json.loads((cell / "result.json").read_text())
    ref = result["checkpoint"]
    checkpoint_path = checked(Path(ref["path"]), ref["sha256"])
    torch.set_num_threads(manifest["threads"])
    feature, _, _, _, yaw, _, _ = inputs(states, route, source["records"])
    model = BeamMixture(feature.shape[1]).double()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["arm"] != args.arm or checkpoint["seed"] != args.model_seed:
        raise ValueError("checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    started = time.monotonic()
    with torch.no_grad():
        scenes = DOMAIN.physical(
            sample_latents(
                model(feature), args.count, torch.Generator().manual_seed(args.sample_seed)
            )
        ).numpy()
    offsets = np.array(manifest["evaluation_offsets"])
    values = numpy_perturbed(scenes, offsets, states, route, yaw.item())
    target = np.array([record["label"] == "d055" for record in source["records"]])
    valid = verdict(values, target).all(-1)
    write_new(
        args.out,
        {
            "manifest": artifact(args.manifest),
            "checkpoint": artifact(checkpoint_path),
            "sampler": artifact(Path(__file__)),
            "sample_seed": args.sample_seed,
            "sample_count": args.count,
            "scenes_station_height": scenes.tolist(),
            "source_order": [r["name"] for r in source["records"]],
            "offsets_xyzh": offsets.tolist(),
            "per_source_clearance_m": values.tolist(),
            "accepted_by_finite_audit": valid.tolist(),
            "accepted_count": int(valid.sum()),
            "rejected_count": int((~valid).sum()),
            "elapsed_s": time.monotonic() - started,
            "independent_carriers": 1,
            "training_eligible": False,
            "execution_eligible": False,
            "scope": "All recorded capsule poses at 113 finite placements; no continuous certificate",
        },
    )
    print(
        json.dumps(
            {"out": str(args.out), "accepted": int(valid.sum()), "rejected": int((~valid).sum())}
        )
    )


if __name__ == "__main__":
    main()
