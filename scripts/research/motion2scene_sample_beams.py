#!/usr/bin/env python3
"""Sample a saved development model and refuse proposals failing independent geometry.

Uses the registered training carrier only. A new-carrier API awaits a qualified
multi-carrier bank. Accepted proposals are capsule diagnostics, not physics assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motion2scene_inverse_learning import DOMAIN, BeamMixture, evaluate, inputs, load_bank
from motion2scene_timing_diagnostic import artifact, checked, write_new
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import sample_latents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("full", "no_kl", "clearance_only", "per_motion"), default="full"
    )
    parser.add_argument("--model-seed", type=int, default=8121)
    parser.add_argument("--sample-seed", type=int, default=9321)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 4096:
        parser.error("count must be in [1,4096]")
    manifest, states, route = load_bank(args.manifest)
    torch.set_num_threads(manifest["threads"])
    cell = args.manifest.parent / f"runs/{args.arm}_{args.model_seed}"
    result = json.loads((cell / "result.json").read_text())
    ref = result["checkpoint"]
    checkpoint = checked(Path(ref["path"]), ref["sha256"])
    feature, _, _, _, yaw, _, _ = inputs(states, route, manifest["records"])
    model = BeamMixture(
        feature.shape[1], motion_conditioned=manifest["arms"][args.arm]["conditioned"]
    ).double()
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if saved["arm"] != args.arm or saved["seed"] != args.model_seed:
        raise ValueError("checkpoint metadata mismatch")
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        samples = DOMAIN.physical(
            sample_latents(
                model(feature), args.count, torch.Generator().manual_seed(args.sample_seed)
            )
        ).numpy()
    audit = evaluate(samples, states, route, yaw.item(), manifest["records"])
    selected = [scene for scene, valid in zip(audit["scenes"], audit["valid"]) if valid]
    write_new(
        args.out,
        {
            "role": "sampled_development_capsule_geometry",
            "manifest": artifact(args.manifest),
            "checkpoint": ref,
            "sampler": artifact(Path(__file__)),
            "sample_seed": args.sample_seed,
            "requested": args.count,
            "accepted_count": len(selected),
            "rejected_count": args.count - len(selected),
            "accepted_station_height": selected,
            "audit": audit,
            "domain": manifest["domain"],
            "refused": len(selected) == 0,
            "training_eligible": False,
            "execution_eligible": False,
            "limits": [manifest["split_note"], manifest["geometry_note"]],
        },
    )
    print(
        json.dumps(
            {
                "accepted": len(selected),
                "rejected": args.count - len(selected),
                "output": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
