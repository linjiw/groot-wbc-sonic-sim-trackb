#!/usr/bin/env python3
"""Train LfLH against true capsule-cloud geometry, with a held-out split.

Replaces the directional-envelope decoder, which an audit showed could not tell a wall standing
across the walking path from the same wall ten metres in the sky, and whose clearance term had no
reachable minimum. Here clearance is a real signed distance in metres, so:

* the barrier loss ``relu(margin - clearance)^2`` is zero exactly when the observed motion is clear
  by ``margin``, and therefore actually enforces separation;
* "does this scene keep the robot clear" is measured in the same units it is trained on;
* the reported numbers are on **held-out clips**, not the ones fitted.

Every arm is scored on the same two axes: how often a sampled scene makes the observed motion the
preferred one, and whether the robot is geometrically clear of every obstacle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.lflh import (  # noqa: E402
    MultiObstacleHallucinator,
    ObstacleGeometry,
    PlausibleShape,
    _kl,
)
from gear_sonic.dataset_generation.hallucination.sdf_decoder import (  # noqa: E402
    SdfChoiceDecoder,
    candidate_cloud,
)
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    local_arm_tuck,
    local_crouch,
)
from gear_sonic.dataset_generation.reference_payload import (  # noqa: E402
    payload_from_reference,
)
from scripts.research.hallucination.build_candidate_sets import (  # noqa: E402
    STATION_FRACTION,
    WINDOW_FRACTION,
)

CLEARANCE_MARGIN_M = 0.05


def rebuild(qpos: np.ndarray, label: str) -> np.ndarray:
    if label == "nominal":
        return qpos
    kind, _, magnitude = label.rpartition("_")
    amount = int(magnitude) / 1000.0
    if kind == "crouch":
        return local_crouch(qpos, STATION_FRACTION, target_drop_m=amount, window=WINDOW_FRACTION)[0]
    side = "left" if "left" in kind else "right"
    return local_arm_tuck(
        qpos, STATION_FRACTION, target_reduction_m=amount, window=WINDOW_FRACTION, side=side
    )[0]


def build_clip(item: dict, target: str, *, stride: int) -> dict | None:
    if target not in item["labels"]:
        return None
    qpos = np.loadtxt(item["source_csv"], delimiter=",")
    clouds = []
    for label in item["labels"]:
        points, radii = candidate_cloud(
            payload_from_reference(rebuild(qpos, label)), frame_stride=stride
        )
        clouds.append(
            (torch.tensor(points, dtype=torch.float32), torch.tensor(radii, dtype=torch.float32))
        )
    return {
        "motion_index": item["motion_index"],
        "body_mode": item.get("body_mode", ""),
        "labels": item["labels"],
        "observed": item["labels"].index(target),
        "costs": torch.tensor(item["costs"], dtype=torch.float32),
        "extents": torch.tensor(np.asarray(item["extents"]), dtype=torch.float32),
        "clouds": clouds,
        "station_xy": np.asarray(item["station_xy_m"]),
        "yaw": np.asarray(item["yaw_rad"]),
        "source_csv": item["source_csv"],
    }


def to_world(boxes: dict[str, torch.Tensor], clip: dict) -> dict[str, torch.Tensor]:
    """Route-frame obstacle parameters -> world box tensors for the SDF decoder."""
    index = torch.clamp(boxes["station"].round().long(), 0, len(clip["yaw"]) - 1)
    yaw = torch.tensor(clip["yaw"], dtype=torch.float32)[index]
    station = torch.tensor(clip["station_xy"], dtype=torch.float32)[index]
    lateral = boxes["lateral_m"]
    centre_x = station[:, 0] - torch.sin(yaw) * lateral
    centre_y = station[:, 1] + torch.cos(yaw) * lateral
    return {
        "centre_x": centre_x,
        "centre_y": centre_y,
        "centre_z": boxes["height_m"],
        "half_along_m": boxes["half_along_m"],
        "half_lateral_m": boxes["half_lateral_m"],
        "half_vertical_m": boxes["half_vertical_m"],
        "yaw": yaw,
    }


def profiles_of(clips: list[dict]) -> torch.Tensor:
    return torch.stack(
        [torch.stack((c["extents"][0], c["extents"][c["observed"]]), dim=0) for c in clips], dim=0
    )


def evaluate(model, clips, geometry, decoder, *, draws: int, seed: int) -> dict:
    """Selection rate and true geometric clearance, on whatever clips are passed."""
    torch.manual_seed(seed)
    selected, clear, worst = [], [], []
    with torch.no_grad():
        mean, log_sigma = model(profiles_of(clips))
        sigma = log_sigma.exp()
        for row, clip in enumerate(clips):
            for _ in range(draws):
                latent = mean[row] + sigma[row] * torch.randn_like(mean[row])
                boxes = to_world(geometry.decode(latent), clip)
                weights = decoder(clip["clouds"], boxes, clip["costs"])
                selected.append(int(weights.argmax()) == clip["observed"])
                points, radii = clip["clouds"][clip["observed"]]
                gap = decoder.clearance(points, radii, boxes)
                clear.append(bool((gap > 0).all()))
                worst.append(float(gap.min()))
    return {
        "selection_rate": float(np.mean(selected)),
        "robot_clear_rate": float(np.mean(clear)),
        "worst_clearance_m": float(np.min(worst)),
        "median_worst_clearance_m": float(np.median(worst)),
        "scenes": len(selected),
    }


def train_sdf(
    clips,
    geometry,
    decoder,
    *,
    obstacles,
    steps,
    samples,
    kl_weight,
    clearance_weight,
    shape_weight,
    seed,
):
    torch.manual_seed(seed)
    model = MultiObstacleHallucinator(geometry.stations, obstacles=obstacles)
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    profiles = profiles_of(clips)
    shape = PlausibleShape()
    history = []
    for step in range(steps):
        mean, log_sigma = model(profiles)
        sigma = log_sigma.exp()
        ramp = float(np.clip((step / max(steps - 1, 1) - 0.33) / 0.5, 0.0, 1.0))
        reconstruction = torch.zeros(())
        barrier = torch.zeros(())
        shape_term = torch.zeros(())
        for _ in range(samples):
            latent = mean + sigma * torch.randn_like(mean)
            for row, clip in enumerate(clips):
                local = geometry.decode(latent[row])
                boxes = to_world(local, clip)
                weights = decoder(clip["clouds"], boxes, clip["costs"])
                reconstruction = reconstruction - torch.log(weights[clip["observed"]] + 1e-8)
                # A real barrier: zero exactly when the observed motion clears every obstacle by
                # the margin. The previous term used a strictly positive blocked-ness score, so
                # its relu never fired and its only escape was to push obstacles skyward.
                points, radii = clip["clouds"][clip["observed"]]
                gap = decoder.clearance(points, radii, boxes)
                barrier = barrier + torch.relu(CLEARANCE_MARGIN_M - gap).pow(2).sum()
                if ramp:
                    shape_term = shape_term + ramp * geometry.size_penalty(local, shape)
        scale = samples * len(clips)
        loss = (
            reconstruction / scale
            + clearance_weight * barrier / scale
            + shape_weight * shape_term / scale
            + kl_weight * _kl(mean, log_sigma)
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "reconstruction": float(reconstruction.detach() / scale),
                    "barrier": float(barrier.detach() / scale),
                }
            )
    return model, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_candidates.json"
    )
    parser.add_argument("--target", default="crouch_040")
    parser.add_argument("--clips", type=int, default=12)
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--obstacles", type=int, default=4)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--clearance-weight", type=float, default=40.0)
    parser.add_argument("--shape-weight", type=float, default=6.0)
    parser.add_argument("--kl-weight", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_sdf.json")
    parser.add_argument("--model-out", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.candidates.read_text())
    built = []
    for item in payload["candidate_sets"]:
        if len(built) >= args.clips:
            break
        clip = build_clip(item, args.target, stride=args.stride)
        if clip is not None:
            built.append(clip)
    if len(built) <= args.holdout:
        raise SystemExit("not enough clips for a held-out split")
    train_clips, test_clips = built[: -args.holdout], built[-args.holdout :]
    print(f"train {len(train_clips)} clips, held out {len(test_clips)}")

    geometry = ObstacleGeometry(stations=payload["stations"])
    decoder = SdfChoiceDecoder()
    model, history = train_sdf(
        train_clips,
        geometry,
        decoder,
        obstacles=args.obstacles,
        steps=args.steps,
        samples=args.samples,
        kl_weight=args.kl_weight,
        clearance_weight=args.clearance_weight,
        shape_weight=args.shape_weight,
        seed=args.seed,
    )
    print(
        f"  final reconstruction {history[-1]['reconstruction']:.3f}  "
        f"barrier {history[-1]['barrier']:.5f}"
    )

    fitted = evaluate(model, train_clips, geometry, decoder, draws=args.draws, seed=1)
    heldout = evaluate(model, test_clips, geometry, decoder, draws=args.draws, seed=2)

    # Control: obstacles drawn from the prior, no learning.
    class _Prior(torch.nn.Module):
        def __init__(self, obstacles):
            super().__init__()
            self.obstacles = obstacles

        def forward(self, profiles):
            generator = torch.Generator().manual_seed(args.seed)
            mean = torch.randn(profiles.shape[0], self.obstacles, 6, generator=generator)
            return mean, torch.zeros_like(mean)

    control = evaluate(
        _Prior(args.obstacles), test_clips, geometry, decoder, draws=args.draws, seed=2
    )

    summary = {
        "schema_version": "lflh_sdf_v1",
        "target": args.target,
        "train_clips": len(train_clips),
        "heldout_clips": len(test_clips),
        "obstacles": args.obstacles,
        "steps": args.steps,
        "clearance_margin_m": CLEARANCE_MARGIN_M,
        "history": history,
        "arms": {
            "fitted_clips": fitted,
            "held_out_clips": heldout,
            "random_from_prior_held_out": control,
        },
        "contract": (
            "clearance is a true signed capsule-to-box distance in metres; selection is the "
            "differentiable choice rule, not a physics verdict"
        ),
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.model_out:
        torch.save({"state_dict": model.state_dict(), "obstacles": args.obstacles}, args.model_out)

    print("\n%-30s %10s %12s %14s" % ("arm", "selection", "robot clear", "worst gap"))
    print("-" * 70)
    for name, values in summary["arms"].items():
        print(
            "%-30s %9.1f%% %11.1f%% %12.3f m"
            % (
                name,
                100 * values["selection_rate"],
                100 * values["robot_clear_rate"],
                values["worst_clearance_m"],
            )
        )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
