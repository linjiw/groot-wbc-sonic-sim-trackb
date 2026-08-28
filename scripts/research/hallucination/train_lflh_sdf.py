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

#: The counterfactual the scene must express, as two margins in metres. The observed motion has to
#: clear every obstacle by ``m_clear``; the cheapest rival has to be struck by at least ``m_hit``.
#: One-sided training -- only "the observed motion clears" -- lets the model satisfy the barrier by
#: pushing obstacles away entirely, which is what drove every box to the height ceiling.
#:
#: Both must be *feasible*. ``measure_scene_ceiling.py`` brute-forces the best single box per clip
#: and finds the achievable ``min(clear, strike)`` is roughly half the edit amplitude: 31.6 mm
#: median at a 40 mm crouch, 39.5 at 55 mm, 46.0 at 70 mm. A 40/30 mm demand at ``crouch_040`` is
#: therefore unsatisfiable on 7 clips in 8, and the first run of this trainer spent its whole budget
#: chasing it -- its final barrier of 0.011 per clip-sample is exactly a 55 mm violation, which is
#: the median worst clearance it reported. Margins are now taken as a fraction of the amplitude.
MARGIN_FRACTION_OF_CEILING = 0.6
#: Fallback only, for a target the ceiling sweep has not covered. Fitted on the *any-rival* reading,
#: which is an upper bound: prefer the measured value whenever `scene_ceiling.json` has one.
CEILING_SLOPE = 0.578
CEILING_INTERCEPT_MM = 10.0
CEILING_PATH = REPO_ROOT / "docs/hallucination/scene_ceiling.json"


def measured_ceiling_mm(target: str, epsilon: float) -> float | None:
    """Median achievable two-sided margin for ``target`` at regret tolerance ``epsilon``, if known.

    Reading this from the artifact rather than hard-coding it keeps the training objective inside
    the feasible set by construction. The first run of this trainer demanded 70 mm of total margin
    where 33 mm was achievable, and spent its whole budget on an unsatisfiable barrier.
    """
    if not CEILING_PATH.exists():
        return None
    blob = json.loads(CEILING_PATH.read_text())
    entry = blob.get("targets", {}).get(target)
    if not entry:
        return None
    by_epsilon = entry.get("median_ceiling_by_epsilon_mm") or {}
    if by_epsilon:
        # The nearest tolerance at or below the requested one; a tighter tolerance is never
        # optimistic, so rounding down is the safe direction.
        candidates = [float(key) for key in by_epsilon if float(key) <= epsilon + 1e-9]
        if candidates:
            return float(by_epsilon[f"{max(candidates):g}"])
    return entry.get("median_ceiling_all_rivals_mm")


def default_margins(target: str, epsilon: float = 0.25) -> tuple[float, float]:
    """Feasible ``(m_clear, m_hit)`` in metres for an edit target, from the measured ceiling."""
    ceiling_mm = measured_ceiling_mm(target, epsilon)
    if ceiling_mm is None:
        suffix = target.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            return 0.02, 0.015
        ceiling_mm = CEILING_SLOPE * int(suffix) + CEILING_INTERCEPT_MM
    share = max(MARGIN_FRACTION_OF_CEILING * ceiling_mm, 1.0) / 1000.0
    return share, share


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
    costs = torch.tensor(item["costs"], dtype=torch.float32)
    observed = item["labels"].index(target)
    cheapest = int(costs.argmin())
    return {
        "cheapest": cheapest,
        "motion_index": item["motion_index"],
        "body_mode": item.get("body_mode", ""),
        "labels": item["labels"],
        "observed": observed,
        "costs": costs,
        "extents": torch.tensor(np.asarray(item["extents"]), dtype=torch.float32),
        "clouds": clouds,
        "station_xy": np.asarray(item["station_xy_m"]),
        "yaw": np.asarray(item["yaw_rad"]),
        "source_csv": item["source_csv"],
    }


def set_regret_rivals(clips: list[dict], epsilon: float) -> None:
    """Which candidates the scene must render infeasible, at a regret tolerance of ``epsilon``.

    A candidate cheaper than the executed motion by more than ``epsilon`` is a counterexample: had
    it been feasible, the robot wasted more than ``epsilon`` of edit cost. Candidates within
    ``epsilon`` -- typically the neighbouring rung of the same operator -- are tolerated, which is
    the difference between a ceiling of 38 mm and one of 3 mm at ``crouch_070``.
    """
    for clip in clips:
        observed_cost = float(clip["costs"][clip["observed"]])
        clip["regret_rivals"] = [
            index
            for index in range(len(clip["costs"]))
            if index not in (clip["observed"], clip["cheapest"])
            and observed_cost - float(clip["costs"][index]) > epsilon
        ]


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
    selected, clear, worst, struck, expressed = [], [], [], [], []
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
                # Hard minimum, not the smooth surrogate the barrier is trained on: a soft
                # minimum is a weighted average and never below the true one, so reporting
                # "clear" from it would be optimistic.
                gap = decoder.exact_clearance(points, radii, boxes)
                clear.append(bool((gap > 0).all()))
                worst.append(float(gap.min()))
                rivals = [
                    index
                    for index in range(len(clip["clouds"]))
                    if index != clip["observed"]
                    and clip["costs"][index] <= clip["costs"][clip["observed"]] + 1e-9
                ]
                depths = [
                    float(-decoder.exact_clearance(*clip["clouds"][r], boxes).min()) for r in rivals
                ]
                # Necessity against the cheapest candidate, plus every candidate that regret
                # rules out. Rungs within epsilon are deliberately absent -- tolerating them is
                # what makes the objective feasible at all.
                struck.append(min(depths) if depths else 0.0)
                # The unambiguous criterion, free of either reading: the decoder actually prefers
                # the observed motion, and the observed motion actually fits.
                expressed.append(bool(gap.min() > 0 and int(weights.argmax()) == clip["observed"]))
    return {
        "selection_rate": float(np.mean(selected)),
        "robot_clear_rate": float(np.mean(clear)),
        "worst_clearance_m": float(np.min(worst)),
        "median_worst_clearance_m": float(np.median(worst)),
        "median_strike_of_every_rival_m": float(np.median(struck)),
        # The scene expresses the counterfactual only if it admits the observed motion AND the
        # decoder prefers it. Either alone is trivially satisfiable: an empty scene admits
        # everything, a solid wall excludes everything.
        "counterfactual_rate": float(np.mean(expressed)),
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
    margin_weight=20.0,
    m_clear=0.02,
    m_hit=0.015,
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
        margin = torch.zeros(())
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
                barrier = barrier + torch.relu(m_clear - gap).pow(2).sum()
                # The other half of the counterfactual: without it the loss is satisfied by a
                # scene that simply contains nothing near anybody.
                # Necessity: the motion that would have been executed in an *empty* scene must be
                # excluded by m_hit. Measured on the cheapest candidate alone, not on every rival.
                cheap_points, cheap_radii = clip["clouds"][clip["cheapest"]]
                cheap_gap = decoder.clearance(cheap_points, cheap_radii, boxes)
                # Softmin over obstacles: the candidate is struck if *any* obstacle strikes it.
                struck = -(torch.softmax(-cheap_gap / 0.01, dim=0) * cheap_gap).sum()
                margin = margin + torch.relu(m_hit - struck).pow(2)
                # Regret: a candidate cheaper than the observed one by more than epsilon must not
                # be feasible, or the executed motion wasted more than epsilon of edit cost for
                # nothing. Candidates within epsilon -- the neighbouring rungs of the same
                # operator -- are tolerated. Demanding that *every* cheaper candidate be struck by
                # the full margin instead is what collapses the achievable ceiling from 50 mm to
                # 2.7 mm at crouch_070; see REPORT_SCENE_CEILING.md section 2.
                for rival in clip["regret_rivals"]:
                    rival_points, rival_radii = clip["clouds"][rival]
                    rival_gap = decoder.clearance(rival_points, rival_radii, boxes)
                    blocked = -(torch.softmax(-rival_gap / 0.01, dim=0) * rival_gap).sum()
                    margin = margin + torch.relu(-blocked).pow(2)
                if ramp:
                    shape_term = shape_term + ramp * geometry.size_penalty(local, shape)
        scale = samples * len(clips)
        loss = (
            reconstruction / scale
            + clearance_weight * barrier / scale
            + margin_weight * margin / scale
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
                    "margin": float(margin.detach() / scale),
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
    parser.add_argument("--margin-weight", type=float, default=20.0)
    parser.add_argument("--shape-weight", type=float, default=6.0)
    parser.add_argument("--kl-weight", type=float, default=0.004)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.25,
        help="regret tolerance: candidates cheaper by more than this must be made infeasible",
    )
    parser.add_argument("--m-clear", type=float, help="metres the observed motion must clear by")
    parser.add_argument("--m-hit", type=float, help="metres the cheapest rival must be struck by")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_sdf.json")
    parser.add_argument("--model-out", type=Path)
    args = parser.parse_args()

    fallback_clear, fallback_hit = default_margins(args.target, args.epsilon)
    m_clear = args.m_clear if args.m_clear is not None else fallback_clear
    m_hit = args.m_hit if args.m_hit is not None else fallback_hit
    ceiling_mm = measured_ceiling_mm(args.target, args.epsilon)
    print(
        f"margins: clear {1000 * m_clear:.1f} mm, strike {1000 * m_hit:.1f} mm; "
        f"measured ceiling at regret <= {args.epsilon:g}: "
        + (f"{ceiling_mm:.1f} mm" if ceiling_mm else "not measured, using the fitted law")
    )

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
    set_regret_rivals(built, args.epsilon)
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
        margin_weight=args.margin_weight,
        m_clear=m_clear,
        m_hit=m_hit,
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
        "m_clear_m": m_clear,
        "m_hit_m": m_hit,
        "regret_epsilon": args.epsilon,
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

    print(
        "\n%-30s %10s %12s %14s %12s"
        % ("arm", "selection", "robot clear", "counterfactual", "worst gap")
    )
    print("-" * 84)
    for name, values in summary["arms"].items():
        print(
            "%-30s %9.1f%% %11.1f%% %13.1f%% %10.3f m"
            % (
                name,
                100 * values["selection_rate"],
                100 * values["robot_clear_rate"],
                100 * values["counterfactual_rate"],
                values["worst_clearance_m"],
            )
        )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
