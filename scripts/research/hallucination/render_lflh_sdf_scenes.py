#!/usr/bin/env python3
"""Render scenes from the signed-clearance LfLH model, ranked by measured quality.

Every scene is scored on the three things that make a counterfactual real, all in metres and all
against true capsule geometry rather than the abstraction being optimised:

* **clearance** -- the smallest signed distance from the observed motion to any obstacle. Positive
  means the robot actually fits through the scene it is supposed to explain.
* **strike** -- how deeply the cheapest rival is penetrated. A scene that no one hits explains
  nothing.
* **selection** -- whether the fixed decoder, shown this scene, prefers the observed motion.

Scenes are ranked by ``min(clearance, strike)``, so a scene only scores well when it *both* admits
the motion and forces it. Picking a best example is a selection, so the summary always reports the
whole distribution beside the winner.
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
)
from gear_sonic.dataset_generation.hallucination.sdf_decoder import (  # noqa: E402
    SdfChoiceDecoder,
)
from scripts.research.hallucination.render_proposal_video import (  # noqa: E402
    render_pair,
)
from scripts.research.hallucination.train_lflh_sdf import (  # noqa: E402
    build_clip,
    profiles_of,
    rebuild,
    set_regret_rivals,
    to_world,
    train_sdf,
)


def score_scene(clip: dict, boxes: dict, decoder: SdfChoiceDecoder) -> dict:
    points, radii = clip["clouds"][clip["observed"]]
    # Hard minimum: the soft one the trainer uses is never below the true distance.
    clearance = float(decoder.exact_clearance(points, radii, boxes).min())
    # Necessity against the cheapest candidate, plus every candidate regret rules out. Rungs
    # within epsilon are deliberately excluded; see set_regret_rivals.
    rivals = [clip["cheapest"]] + list(clip["regret_rivals"])
    # Score the least-struck of them: the scene is only as good as its weakest exclusion.
    strike, binding = float("inf"), 0
    for rival in rivals:
        rival_points, rival_radii = clip["clouds"][rival]
        depths = -decoder.exact_clearance(rival_points, rival_radii, boxes)
        if float(depths.max()) < strike:
            strike = float(depths.max())
            # Which obstacle does the work? The figure is only interpretable if the box it
            # highlights is the one that forces the edit, so record it rather than assuming it is
            # the first.
            binding = int(depths.argmax())
    if not rivals:
        strike = 0.0
    weights = decoder(clip["clouds"], boxes, clip["costs"])
    return {
        "clearance_m": clearance,
        "strike_m": strike,
        "selected": clip["labels"][int(weights.argmax())],
        "match": int(weights.argmax()) == clip["observed"],
        "quality_m": min(clearance, strike),
        "binding_obstacle": binding,
    }


def world_boxes(boxes: dict) -> list[dict]:
    out = []
    for index in range(boxes["centre_x"].shape[0]):
        out.append(
            {
                "name": f"sdf_{index}",
                "center_m": (
                    float(boxes["centre_x"][index]),
                    float(boxes["centre_y"][index]),
                    float(boxes["centre_z"][index]),
                ),
                "full_size_m": (
                    2 * float(boxes["half_along_m"][index]),
                    2 * float(boxes["half_lateral_m"][index]),
                    2 * float(boxes["half_vertical_m"][index]),
                ),
                "yaw_rad": float(boxes["yaw"][index]),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_candidates.json"
    )
    parser.add_argument("--target", default="crouch_070")
    parser.add_argument("--clips", type=int, default=16)
    parser.add_argument("--holdout", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--obstacles", type=int, default=4)
    parser.add_argument("--draws", type=int, default=24)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--render-top", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.25, help="regret tolerance")
    parser.add_argument("--model-in", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--ceiling",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/scene_ceiling.json",
        help="per-clip best achievable margin, from measure_scene_ceiling.py",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.candidates.read_text())
    built = []
    for item in payload["candidate_sets"]:
        if len(built) >= args.clips:
            break
        clip = build_clip(item, args.target, stride=args.stride)
        if clip is not None:
            built.append(clip)
    set_regret_rivals(built, args.epsilon)
    train_clips, test_clips = built[: -args.holdout], built[-args.holdout :]
    geometry = ObstacleGeometry(stations=payload["stations"])
    decoder = SdfChoiceDecoder()

    if args.model_in and args.model_in.exists():
        blob = torch.load(args.model_in, weights_only=False)
        model = MultiObstacleHallucinator(payload["stations"], obstacles=blob["obstacles"])
        model.load_state_dict(blob["state_dict"])
        print(f"loaded {args.model_in}")
    else:
        print(f"training on {len(train_clips)} clips ...")
        model, history = train_sdf(
            train_clips,
            geometry,
            decoder,
            obstacles=args.obstacles,
            steps=args.steps,
            samples=args.samples,
            kl_weight=0.004,
            clearance_weight=40.0,
            shape_weight=6.0,
            seed=args.seed,
        )
        print(f"  reconstruction {history[-1]['reconstruction']:.3f}")

    torch.manual_seed(args.seed)
    scored = []
    with torch.no_grad():
        for split, clips in (("held_out", test_clips), ("fitted", train_clips)):
            if not clips:
                continue
            mean, log_sigma = model(profiles_of(clips))
            sigma = log_sigma.exp()
            for row, clip in enumerate(clips):
                for draw in range(args.draws):
                    latent = mean[row] + sigma[row] * torch.randn_like(mean[row])
                    boxes = to_world(geometry.decode(latent), clip)
                    entry = score_scene(clip, boxes, decoder)
                    entry.update(
                        {
                            "split": split,
                            "motion_index": clip["motion_index"],
                            "body_mode": clip["body_mode"],
                            "draw": draw,
                            "boxes": world_boxes(boxes),
                        }
                    )
                    scored.append(entry)

    # Margin efficiency: the achieved two-sided margin as a fraction of the best any single box
    # could reach on that clip. A raw margin is uninterpretable without it -- a small margin may be
    # the task rather than the model, which is exactly what the first run of this trainer got wrong.
    ceilings = {}
    if args.ceiling.exists():
        blob = json.loads(args.ceiling.read_text())
        for row in blob.get("targets", {}).get(args.target, {}).get("clips", []):
            # Compare like with like: the achieved margin uses the all-rivals rule, so the
            # ceiling must too.
            ceilings[row["motion_index"]] = row.get("score_all_m", row["score_m"])
    for entry in scored:
        ceiling = ceilings.get(entry["motion_index"])
        entry["ceiling_m"] = ceiling
        entry["efficiency"] = entry["quality_m"] / ceiling if ceiling and ceiling > 0 else None

    held = [s for s in scored if s["split"] == "held_out"]
    print(
        f"\n{len(scored)} scenes scored ({len(held)} held out). "
        f"held-out match {np.mean([s['match'] for s in held]):.1%}, "
        f"robot clear {np.mean([s['clearance_m'] > 0 for s in held]):.1%}"
    )

    efficiencies = [s["efficiency"] for s in held if s["efficiency"] is not None]
    if efficiencies:
        best_per_clip = {}
        for entry in held:
            if entry["efficiency"] is None:
                continue
            index = entry["motion_index"]
            if entry["efficiency"] > best_per_clip.get(index, -9):
                best_per_clip[index] = entry["efficiency"]
        print(
            f"margin efficiency vs the per-clip ceiling: median draw {np.median(efficiencies):.1%}, "
            f"best draw per clip {np.median(list(best_per_clip.values())):.1%} "
            f"(over {len(best_per_clip)} clips)"
        )

    admissible = [s for s in held if s["match"] and s["clearance_m"] > 0 and s["strike_m"] > 0]
    # `match` is the decoder actually preferring the observed motion; the two margins say how
    # robustly. All three are required, so no arm of the criterion can be satisfied trivially.
    admissible.sort(key=lambda s: -s["quality_m"])
    print(
        f"held-out scenes that admit the motion AND strike a rival AND select it: "
        f"{len(admissible)}/{len(held)}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for rank, entry in enumerate(admissible[: args.render_top]):
        clip = next(c for c in test_clips if c["motion_index"] == entry["motion_index"])
        qpos = np.loadtxt(clip["source_csv"], delimiter=",")
        adapted = rebuild(qpos, args.target)
        out_path = args.out_dir / f"rank{rank}_{entry['motion_index']:03d}_{args.target}.mp4"
        shared = [
            f"#{rank + 1} of {len(admissible)} admissible held-out scenes",
            f"clip {entry['motion_index']:03d} {entry['body_mode']}  "
            f"{len(entry['boxes'])} generated obstacles",
            f"robot clears by {1000 * entry['clearance_m']:.0f} mm; "
            f"every cheaper motion struck by >= {1000 * entry['strike_m']:.0f} mm"
            + (
                f"  ({entry['efficiency']:.0%} of the best any box could do)"
                if entry.get("efficiency")
                else ""
            ),
            "distances are analytic capsule-to-box; props are not simulated contacts",
        ]
        ordered = [entry["boxes"][entry["binding_obstacle"]]] + [
            box for index, box in enumerate(entry["boxes"]) if index != entry["binding_obstacle"]
        ]
        render_pair(
            {"a_nominal": qpos, "b_adapted": adapted},
            ordered[0],
            out_path,
            width=args.width,
            height=args.height,
            fps=30,
            context=ordered[1:],
            captions={
                "a_nominal": ["NOMINAL  (struck -> must adapt)"] + shared,
                "b_adapted": [f"ADAPTED {args.target}  (clears)"] + shared,
            },
        )
        entry["video"] = str(out_path)
        rendered.append(entry)
        print(
            f"  rank {rank}: clip {entry['motion_index']:03d} clear "
            f"{1000 * entry['clearance_m']:+.0f} mm strike {1000 * entry['strike_m']:+.0f} mm "
            f"-> {out_path.name}"
        )

    if args.manifest:
        args.manifest.write_text(
            json.dumps(
                {
                    "schema_version": "lflh_sdf_scenes_v1",
                    "target": args.target,
                    "held_out_scenes": len(held),
                    "held_out_match_rate": float(np.mean([s["match"] for s in held])),
                    "held_out_robot_clear_rate": float(
                        np.mean([s["clearance_m"] > 0 for s in held])
                    ),
                    "held_out_admissible": len(admissible),
                    "median_margin_efficiency": (
                        float(np.median(efficiencies)) if efficiencies else None
                    ),
                    "clearance_distribution_mm": sorted(
                        round(1000 * s["clearance_m"], 1) for s in held
                    ),
                    "rendered": rendered,
                    "contract": (
                        "clearance and strike are true signed capsule-to-box distances; the "
                        "rendered scenes are the best-scoring of the held-out draws and the full "
                        "distribution is reported beside them"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
