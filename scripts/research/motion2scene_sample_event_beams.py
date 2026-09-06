#!/usr/bin/env python3
"""Sample a registered event model and independently reject failed proxy placements."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN
from motion2scene_event_scaling import load, normalized
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_uncertainty_learning import numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_events import (
    EventMixture,
    sample_scenes,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import route_centers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True, help="Saved checkpoint result.json")
    parser.add_argument("--case", required=True, help="Case from the registered reference bank")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 256:
        raise ValueError("count must be in [1,256]")
    reg, cases, raw_features = load(args.registration)
    cell = json.loads(args.cell.read_text())
    complete = json.loads((args.registration.parent / "runs/complete.json").read_text())
    checked(args.registration, complete["registration"]["sha256"])
    ref = artifact(args.cell)
    if ref not in complete["cells"]:
        raise ValueError("cell is not a registered completed snapshot")
    ref = cell["checkpoint"]
    checkpoint = torch.load(checked(Path(ref["path"]), ref["sha256"]), weights_only=True)
    config = reg["arms"][cell["arm"]]
    if checkpoint["config"] != config or checkpoint["budget"] != cell["budget"]:
        raise ValueError("checkpoint configuration mismatch")
    torch.set_num_threads(reg["threads"])
    model = EventMixture(width=config["width"], pooled=config["pooled"]).double()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    feature = normalized(raw_features, config)[args.case]
    if config["input"] == "constant":
        feature = torch.zeros_like(feature)
    case = cases[args.case]
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with torch.no_grad():
        scenes = sample_scenes(model(feature), 256, torch.Generator().manual_seed(args.seed))
        scenes = scenes.numpy()[: args.count]
    sampling_seconds = time.monotonic() - start
    offsets = np.array(reg["evaluation_offsets"])
    start = time.monotonic()
    values = numpy_perturbed(scenes, offsets, case["states"], case["route"], case["yaw"].item())
    valid = verdict(values, np.array([False, True])).all(1)
    checking_seconds = time.monotonic() - start
    parent = case["metadata"]["parent"]
    parent_path = checked(Path(parent["path"]), parent["sha256"])
    origin = np.loadtxt(parent_path, delimiter=",")[0, :2]
    xy = route_centers(tensor(scenes[:, 0]), case["rt"], case["progress"]).numpy() + origin
    proposals = []
    for i, (station, height) in enumerate(scenes):
        proposals.append(
            {
                "index": i,
                "station": float(station),
                "underside_height_m": float(height),
                "center_xyz_m": [*xy[i].tolist(), float(height + DOMAIN.thickness / 2)],
                "yaw_rad": case["yaw"].item(),
                "dimensions_xyz_m": [DOMAIN.depth, DOMAIN.width, DOMAIN.thickness],
                "frame": "original_parent_reference_world",
                "target_min_clearance_m": float(values[i, :, 1].min()),
                "upright_max_clearance_m": float(values[i, :, 0].max()),
                "all_113_proxy_valid": bool(valid[i]),
            }
        )
    raw = args.out / "clearances.npz"
    with raw.open("xb") as handle:
        np.savez_compressed(handle, scenes=scenes, offsets=offsets, clearances=values)
    write_new(
        args.out / "result.json",
        {
            "registration": artifact(args.registration),
            "cell": artifact(args.cell),
            "sampler": artifact(Path(__file__)),
            "case": args.case,
            "seed": args.seed,
            "requested": args.count,
            "accepted": int(valid.sum()),
            "proposals": proposals,
            "accepted_indices": np.flatnonzero(valid).tolist(),
            "raw": artifact(raw),
            "sampling_seconds": sampling_seconds,
            "checking_seconds": checking_seconds,
            "canonical_to_parent_translation_xy_m": origin.tolist(),
            "reference_only": True,
            "execution_eligible": False,
            "training_eligible": False,
            "contract": "sampled reference capsules at 113 placements; no continuous or execution certificate",
        },
    )
    print(
        json.dumps(
            {
                "sampled": args.count,
                "accepted": int(valid.sum()),
                "result": str(args.out / "result.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
