#!/usr/bin/env python3
"""Fit the matched target-only generator baseline; no downstream/test evaluation."""

import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN
from motion2scene_distill_study import checkpoint, load, original_cells
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import tensor
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_events import (
    EventMixture,
    stratified_scenes,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_no_contrast import target_penalty
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)


def main():
    data = ROOT.parent / "research-data/groot-wbc"
    source = data / "m2s-distillation-v1/registration.json"
    _, parent, train, _ = load(source)
    cell = original_cells(parent)[0]
    saved, _ = checkpoint(cell["checkpoint"])
    config = saved["config"]
    ids = config["training_ids"]
    assert list(train) == ids and cell["seed"] == 8421
    fit_source = data / "m2s-source-phase-v1/registration.json"
    original = json.loads(fit_source.read_text())
    out = data / "m2s-no-contrast-baseline-v1"
    out.mkdir(parents=True, exist_ok=False)
    refs = [
        artifact(source),
        artifact(fit_source),
        cell["checkpoint"],
        artifact(Path(__file__)),
        artifact(ROOT / "docs/motion2scene/NO_CONTRAST_BASELINE_V1.md"),
        artifact(ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_no_contrast.py"),
        artifact(ROOT / "scripts/research/motion2scene_event_scaling.py"),
        artifact(ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_uncertainty.py"),
        artifact(ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_events.py"),
    ]
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "config": config,
            "seed": 8421,
            "steps": 1200,
            "learning_rate": original["learning_rate"],
            "terms": {"target": 5, "preference": 0, "neutral_interference": 0, "kl": 0},
            "selection": "fixed_final_step",
            "cpu_ceiling_s": 300,
            "gpu_seconds": 0,
            "scope": "baseline generator fit; no robot-data selector fitting",
        },
    )
    features = normalized(raw_features(train), config)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(8421)
    model = EventMixture(width=32).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=original["learning_rate"])
    rng = torch.Generator().manual_seed(18421)
    pose_rng = torch.Generator().manual_seed(38421)
    order_rng = torch.Generator().manual_seed(48421)
    corners = tensor(original["training_corners"])
    visits = {name: 0 for name in ids}
    history = []
    start = time.monotonic()
    for step in range(1200):
        if time.monotonic() - start > 300:
            raise TimeoutError("registered baseline CPU cap")
        if step % len(ids) == 0:
            order = torch.randperm(len(ids), generator=order_rng).tolist()
        name = ids[order[step % len(ids)]]
        visits[name] += 1
        case = train[name]
        optimizer.zero_grad(set_to_none=True)
        scenes, weights = stratified_scenes(model(features[name]), rng)
        offsets = torch.cat(
            (
                torch.zeros(1, 4, dtype=torch.float64),
                corners[torch.randperm(16, generator=pose_rng)[:4]],
            )
        )
        values = perturbed_clearances(
            scenes, offsets, case["clouds"], case["rt"], case["progress"], case["yaw"], DOMAIN
        )
        # Channel zero is intentionally not consumed by loss or selection.
        loss = (weights * 5 * target_penalty(values[:, :, 1])).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite target loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if step % 200 == 0 or step == 1199:
            history.append({"step": step + 1, "loss": float(loss.detach())})
            print(json.dumps(history[-1]), flush=True)
    seconds = time.monotonic() - start
    with (out / "checkpoint.pt").open("xb") as f:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "seed": 8421,
                "budget": 1200,
                "arm": "no_contrast_target_only",
            },
            f,
        )
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "checkpoint": artifact(out / "checkpoint.pt"),
            "history": history,
            "visits": visits,
            "training_seconds": seconds,
            "gpu_seconds": 0,
            "fixed_budget_completed": True,
            "robot_data_selector_fits": 0,
            "downstream_utility_measured": False,
        },
    )
    print(
        json.dumps(
            {"training_seconds": seconds, "generator_fits": 1, "robot_data_selector_fits": 0}
        )
    )


if __name__ == "__main__":
    main()
