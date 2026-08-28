#!/usr/bin/env python3
"""What is one forward pass of the hallucinator worth, in units of search?

Section 2 of ``REPORT_SCENE_CEILING.md`` shows that whether a usable scene *exists* is a property of
the edit amplitude, not of any model: an exhaustive box search finds the ceiling every time. So the
learned hallucinator cannot be justified by existence. Its only defensible claim is **amortisation**
-- that it returns a near-ceiling scene for an unseen motion in one forward pass, with no search.

This script measures that claim in the only currency that makes it falsifiable: how many random
draws a search needs before its best-so-far matches the model's median margin on the same clip. If
the answer is "one or two", the model is worth nothing. Every distance here is a hard minimum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

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
from scripts.research.hallucination.train_lflh_sdf import (  # noqa: E402
    build_clip,
    profiles_of,
    set_regret_rivals,
    to_world,
)


def margin_of(clip: dict, boxes: dict, decoder: SdfChoiceDecoder) -> float:
    points, radii = clip["clouds"][clip["observed"]]
    clear = float(decoder.exact_clearance(points, radii, boxes).min())
    # Necessity against the cheapest candidate, plus every candidate regret rules out at the
    # configured epsilon; the scene is only as good as its weakest exclusion.
    rivals = [clip["cheapest"]] + list(clip["regret_rivals"])
    strike = min(
        (float(-decoder.exact_clearance(*clip["clouds"][r], boxes).min()) for r in rivals),
        default=0.0,
    )
    return min(clear, strike)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_candidates.json"
    )
    parser.add_argument("--target", default="crouch_070")
    parser.add_argument("--clips", type=int, default=16)
    parser.add_argument("--holdout", type=int, default=5)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--model-in", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=16, help="model draws per clip")
    parser.add_argument("--search-budget", type=int, default=4096, help="random draws per clip")
    parser.add_argument("--epsilon", type=float, default=0.25, help="regret tolerance")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/amortisation.json"
    )
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
    test_clips = built[-args.holdout :]
    geometry = ObstacleGeometry(stations=payload["stations"])
    decoder = SdfChoiceDecoder()

    blob = torch.load(args.model_in, weights_only=False)
    model = MultiObstacleHallucinator(payload["stations"], obstacles=blob["obstacles"])
    model.load_state_dict(blob["state_dict"])

    rows = []
    with torch.no_grad():
        profiles = profiles_of(test_clips)
        started = time.perf_counter()
        mean, log_sigma = model(profiles)
        forward_s = (time.perf_counter() - started) / len(test_clips)
        sigma = log_sigma.exp()
        generator = torch.Generator().manual_seed(args.seed)
        for row, clip in enumerate(test_clips):
            model_margins = []
            for _ in range(args.draws):
                latent = mean[row] + sigma[row] * torch.randn(mean[row].shape, generator=generator)
                model_margins.append(
                    margin_of(clip, to_world(geometry.decode(latent), clip), decoder)
                )
            reference = float(np.median(model_margins))

            # The control has the same parameterisation and the same decoder; it simply does not
            # condition on the motion. Anything the model gains over it is what conditioning buys.
            best, draws_to_match, search_started = -9.0, None, time.perf_counter()
            for draw in range(args.search_budget):
                latent = torch.randn(mean[row].shape, generator=generator)
                value = margin_of(clip, to_world(geometry.decode(latent), clip), decoder)
                best = max(best, value)
                if draws_to_match is None and best >= reference:
                    draws_to_match = draw + 1
            search_s = (time.perf_counter() - search_started) / args.search_budget

            rows.append(
                {
                    "motion_index": clip["motion_index"],
                    "model_median_margin_m": reference,
                    "model_best_margin_m": float(np.max(model_margins)),
                    "search_best_margin_m": best,
                    "draws_to_match": draws_to_match,
                    "search_budget": args.search_budget,
                }
            )
            print(
                f"clip {clip['motion_index']:03d}  model median {1000 * reference:6.1f} mm  "
                f"search best of {args.search_budget} {1000 * best:6.1f} mm  "
                f"draws to match: "
                f"{draws_to_match if draws_to_match is not None else '>' + str(args.search_budget)}"
            )

    # Standing diagnostic from the first retraction: if replacing the trajectory with the corpus
    # mean does not change the output, the model is not conditional and nothing was learned.
    with torch.no_grad():
        blind = profiles.mean(dim=0, keepdim=True).expand_as(profiles)
        blind_mean, _ = model(blind)
        spread = float(mean.std(dim=0).mean())
        shift = float((mean - blind_mean).abs().mean())
        blind_margins = []
        for row, clip in enumerate(test_clips):
            values = [
                margin_of(clip, to_world(geometry.decode(blind_mean[row]), clip), decoder)
                for _ in range(1)
            ]
            blind_margins.append(float(np.median(values)))
    conditioned = [r["model_median_margin_m"] for r in rows]
    print(
        f"\ninput ablation: latent spread across clips {spread:.4f}; "
        f"mean shift when the motion is replaced by the corpus mean {shift:.4f}"
    )
    print(
        f"  margin conditioned {1000 * np.median(conditioned):.1f} mm vs "
        f"motion-blind {1000 * np.median(blind_margins):.1f} mm"
    )

    matched = [r["draws_to_match"] for r in rows if r["draws_to_match"] is not None]
    never = len(rows) - len(matched)
    print(
        f"\n{len(rows)} held-out clips | search matched the model on {len(matched)}, "
        f"never within budget on {never}"
    )
    if matched:
        print(f"median draws for search to match one forward pass: {int(np.median(matched))}")
    print(
        f"model forward pass {1000 * forward_s:.2f} ms/clip; one search draw {1000 * search_s:.2f} ms"
    )

    args.out.write_text(
        json.dumps(
            {
                "schema_version": "amortisation_v1",
                "target": args.target,
                "clips": rows,
                "median_draws_to_match": (int(np.median(matched)) if matched else None),
                "clips_search_never_matched": never,
                "model_forward_ms_per_clip": 1000 * forward_s,
                "latent_spread_across_clips": spread,
                "latent_shift_when_motion_blinded": shift,
                "median_margin_m_conditioned": float(np.median(conditioned)),
                "median_margin_m_motion_blind": float(np.median(blind_margins)),
                "search_draw_ms": 1000 * search_s,
                "contract": (
                    "the control shares the model's parameterisation and decoder and differs only "
                    "in not conditioning on the motion; margins are hard minima"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
