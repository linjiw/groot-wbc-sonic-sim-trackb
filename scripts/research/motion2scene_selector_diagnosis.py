#!/usr/bin/env python3
"""Registered post hoc normalization, feature-group and empirical-ceiling diagnosis."""

import argparse
import json
from pathlib import Path
import time

from bundle_motion2scene_sources import closure
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_empirical_ambiguity import (
    empirical_limits,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import load_readout
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import select_action

DATA = ROOT.parent / "research-data/groot-wbc"
CORPUS = DATA / "m2s-comparative-corpus-completion-v1"
ORIGINAL = DATA / "m2s-independent-layout-v1"
PROTOCOL = ROOT / "docs/motion2scene/SELECTOR_BREAKPOINT_STUDY_V1.md"
GROUPS = {
    "rays": (0, 144),
    "phase_skill_age": (144, 147),
    "gravity": (147, 150),
    "velocity": (150, 156),
    "joint_position": (156, 185),
    "joint_velocity": (185, 214),
    "all_non_ray": (144, 214),
}


def names(joints):
    result = []
    for i in range(12):
        result += [f"ray_{i}_{k}" for k in ("hit", "distance_over_3", "height_over_2.5")]
        for j in range(3):
            result += [f"ray_{i}_lower_{j}_{k}" for k in ("observed", "hit", "distance_over_3")]
    result += ["phase_over_4", "active_skill", "observation_age_s"]
    result += [
        f"{g}_{a}" for g in ("gravity_b", "linear_velocity_w", "angular_velocity_w") for a in "xyz"
    ]
    result += [f"joint_position_{j}" for j in joints] + [f"joint_velocity_{j}" for j in joints]
    assert len(result) == 214
    return result


def register(out):
    fits = json.loads((CORPUS / "fits/result.json").read_text())
    packets = []
    master = json.loads((ORIGINAL / "master.json").read_text())
    for block in master["blocks"][:6]:
        r = json.loads((Path(block["directory"]) / "result.json").read_text())
        packets.append({"block": block["index"], "decision": r["rows"][0]["decision"]})
    refs = [
        artifact(PROTOCOL),
        artifact(CORPUS / "admission.json"),
        artifact(CORPUS / "registration.json"),
        artifact(CORPUS / "fits/result.json"),
        artifact(ROOT / "docs/motion2scene/evidence/independent-layout-decisions.json"),
    ]
    refs += [r["checkpoint"] for r in fits["rows"]] + [p["decision"] for p in packets]
    refs += [artifact(p) for p in sorted(closure([Path(__file__)]))]
    for r in refs:
        checked(Path(r["path"]), r["sha256"])
    out.mkdir(exist_ok=False)
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "packets": packets,
            "groups": GROUPS,
            "physical_scales": {"default": 1, "joint_velocity": 5},
            "linear_controls": 4,
            "cpu_ceiling_s": 300,
            "scope": "post hoc development diagnosis; no original model changes or new physics",
        },
    )


def network_details(policy, x):
    model, mean, std = policy
    z = (torch.as_tensor(x, dtype=torch.float32) - mean) / std
    with torch.no_grad():
        h1 = model.network[0](z)
        h2 = model.network[2](h1.relu())
        logits = model.network[4](h2.relu())
        p = logits.sigmoid().numpy()
    return z.numpy(), h1.numpy(), h2.numpy(), logits.numpy(), p


def low_capacity(x, y, eval_x):
    scale = torch.ones(214)
    scale[185:] = 5
    weights = torch.zeros((214, 2), requires_grad=True)
    bias = torch.zeros(2, requires_grad=True)
    tx, ty = torch.tensor(x) / scale, torch.tensor(y)
    optimizer = torch.optim.Adam([weights, bias], lr=0.01)
    for _ in range(2000):
        optimizer.zero_grad(set_to_none=True)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(tx @ weights + bias, ty)
        loss = bce + 0.01 * weights.square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = (torch.tensor(eval_x) / scale) @ weights + bias
        p = logits.sigmoid().numpy()
        final_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            tx @ weights + bias, ty
        ).item()
    return (
        {
            "weights": weights.detach().numpy(),
            "bias": bias.detach().numpy(),
            "scale": scale.numpy(),
        },
        p,
        final_bce,
    )


def analyze(out):
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    reg = json.loads((out / "registration.json").read_text())
    for ref in reg["references"]:
        checked(Path(ref["path"]), ref["sha256"])
    corpus = json.loads((CORPUS / "admission.json").read_text())
    fit_reg = json.loads((CORPUS / "registration.json").read_text())
    fits = json.loads((CORPUS / "fits/result.json").read_text())
    evaluation = json.loads(
        (ROOT / "docs/motion2scene/evidence/independent-layout-decisions.json").read_text()
    )["rows"]
    pairs = {p["group_id"]: p for p in corpus["pairs"]}
    first_packet = json.loads(Path(reg["packets"][0]["decision"]["path"]).read_text())
    feature_names = names(first_packet["joint_names"])
    limits = {}
    for arm, ids in {"all_corpus": list(pairs), **fit_reg["fit_ids"]}.items():
        ps = [pairs[k] for k in ids]
        limits[arm] = empirical_limits(
            [p["features"] for p in ps], [p["outcomes"] for p in ps], [p["mask"] for p in ps]
        )
        limits[arm]["ids"] = ids
    models, interventions, readouts = [], [], []
    for fit in fits["rows"]:
        if time.monotonic() - start > reg["cpu_ceiling_s"]:
            raise TimeoutError("bounded selector diagnosis")
        arm, seed = fit["arm"], fit["seed"]
        ids = fit_reg["fit_ids"][arm]
        x = np.array([pairs[k]["features"] for k in ids], dtype=np.float32)
        erows = [r for r in evaluation if r["arm"] == arm and r["optimizer_seed"] == seed]
        e = np.array([r["features"] for r in erows], dtype=np.float32)
        policy = load_readout(fit["checkpoint"]["path"], fit["checkpoint"]["sha256"])
        model, mean, scale = policy
        z, h1, h2, logits, probabilities = network_details(policy, e)
        assert np.allclose(
            probabilities, [r["readout"]["probabilities"] for r in erows], atol=1e-6, rtol=0
        )
        raw_std = torch.tensor(x).std(0, correction=0).numpy()
        stored = scale.numpy()
        train_z = (x - mean.numpy()) / stored
        features = [
            {
                "index": i,
                "name": name,
                "train_min": float(x[:, i].min()),
                "train_max": float(x[:, i].max()),
                "train_std": float(raw_std[i]),
                "stored_mean": float(mean[i]),
                "stored_scale": float(stored[i]),
                "scale_clamped": bool(raw_std[i] < 1e-3),
                "training_max_abs_normalized": float(abs(train_z[:, i]).max()),
                "evaluation_max_abs_normalized": float(abs(z[:, i]).max()),
                "evaluation_max_displacement": float(abs(e[:, i] - mean.numpy()[i]).max()),
                "evaluation_outside_training_range": int(
                    ((e[:, i] < x[:, i].min()) | (e[:, i] > x[:, i].max())).sum()
                ),
            }
            for i, name in enumerate(feature_names)
        ]
        group_contributions = {}
        for g, (lo, hi) in GROUPS.items():
            with torch.no_grad():
                contribution = torch.tensor(z[:, lo:hi]) @ model.network[0].weight[:, lo:hi].T
            group_contributions[g] = float(contribution.abs().max())
        models.append(
            {
                "arm": arm,
                "seed": seed,
                "checkpoint": fit["checkpoint"],
                "features": features,
                "first_layer_group_max_abs_contribution": group_contributions,
                "hidden1_pre_max_abs": float(abs(h1).max()),
                "hidden2_pre_max_abs": float(abs(h2).max()),
                "logit_min": logits.min(0).tolist(),
                "logit_max": logits.max(0).tolist(),
                "saturated_heads_at_1e_minus_6": int(
                    ((probabilities <= 1e-6) | (probabilities >= 1 - 1e-6)).sum()
                ),
                "training_bce": fit["final_training_loss"],
                "bce_above_empirical_infimum": fit["final_training_loss"]
                - limits[arm]["masked_bce_infimum"],
            }
        )
        for i, r in enumerate(erows):
            readouts.append(
                {
                    "cell_id": r["cell_id"],
                    "arm": arm,
                    "seed": seed,
                    "logits": logits[i].tolist(),
                    "probabilities": probabilities[i].tolist(),
                    "top_normalized_features": [
                        {
                            "index": int(j),
                            "name": feature_names[j],
                            "normalized_value": float(z[i, j]),
                        }
                        for j in np.argsort(abs(z[i]))[-10:][::-1]
                    ],
                }
            )
            mixed, labels = [], []
            for group, (lo, hi) in GROUPS.items():
                for k, donor in enumerate(ids):
                    value = e[i].copy()
                    value[lo:hi] = x[k, lo:hi]
                    mixed.append(value)
                    labels.append((group, donor))
            _, _, _, mix_logits, mix_p = network_details(policy, np.array(mixed))
            selected = select_action(mix_p)
            for j, (group, donor) in enumerate(labels):
                action = int(selected[j])
                interventions.append(
                    {
                        "cell_id": r["cell_id"],
                        "arm": arm,
                        "seed": seed,
                        "group": group,
                        "donor": donor,
                        "logits": mix_logits[j].tolist(),
                        "probabilities": mix_p[j].tolist(),
                        "selected_action": action,
                        "requested_action": max(action, 0),
                        "refusal": action == -1,
                        "request_changed": max(action, 0) != r["readout"]["requested_action"],
                        "refusal_changed": (action == -1) != r["readout"]["refusal"],
                        "max_abs_logit_change": float(abs(mix_logits[j] - logits[i]).max()),
                    }
                )
    controls = []
    for arm, ids in fit_reg["fit_ids"].items():
        if time.monotonic() - start > reg["cpu_ceiling_s"]:
            raise TimeoutError("bounded linear control fitting")
        x = np.array([pairs[k]["features"] for k in ids], dtype=np.float32)
        y = np.array([pairs[k]["outcomes"] for k in ids], dtype=np.float32)
        erows = [r for r in evaluation if r["arm"] == arm and r["optimizer_seed"] == 8501]
        e = np.array([r["features"] for r in erows], dtype=np.float32)
        saved, p, bce = low_capacity(x, y, e)
        path = out / f"linear_{arm}.npz"
        np.savez_compressed(path, **saved)
        actions = select_action(p)
        controls.append(
            {
                "arm": arm,
                "model": artifact(path),
                "training_ids": ids,
                "training_bce": bce,
                "conditions": [
                    {
                        "cell_id": r["cell_id"],
                        "probabilities": v.tolist(),
                        "selected_action": int(a),
                        "requested_action": max(int(a), 0),
                        "refusal": bool(a == -1),
                    }
                    for r, v, a in zip(erows, p, actions)
                ],
            }
        )
    visibility = []
    for item in reg["packets"]:
        d = json.loads(Path(item["decision"]["path"]).read_text())
        rays = d["packet"]["rays"]
        hits = [r["hit"] for r in rays if r["hit"]]
        visibility.append(
            {
                "block": item["block"],
                "phase_s": d["phase_s"],
                "upper_hits": len(hits),
                "upper_beam_hits": sum(
                    "CounterfactualBeam" in str(h.get("path", "")) for h in hits
                ),
                "lower_queries": sum(len(r["lower_rays"]) for r in rays),
                "upper_candidates": sum(bool(r["upper_candidate"]) for r in rays),
                "occupied": d["packet"]["occupied"],
                "source": item["decision"],
            }
        )
    aggregates = {}
    for arm in fit_reg["fit_ids"]:
        ms = [m for m in models if m["arm"] == arm]
        aggregates[arm] = {
            "models": len(ms),
            "max_abs_normalized_feature": max(
                f["evaluation_max_abs_normalized"] for m in ms for f in m["features"]
            ),
            "min_logit": min(min(m["logit_min"]) for m in ms),
            "max_logit": max(max(m["logit_max"]) for m in ms),
            "d040_recovery_by_group": {
                g: len(
                    {
                        r["cell_id"]
                        for r in interventions
                        if r["arm"] == arm and r["group"] == g and r["requested_action"] == 1
                    }
                )
                for g in GROUPS
            },
            "measured_readouts": 30,
            "floor": limits[arm]["masked_bce_infimum"],
        }
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "limits": limits,
            "models": models,
            "readouts": readouts,
            "interventions": interventions,
            "linear_controls": controls,
            "visibility": visibility,
            "aggregates": aggregates,
            "seconds": time.monotonic() - start,
            "new_physics_executions": 0,
            "new_original_model_fits": 0,
            "new_linear_control_fits": 4,
            "scope": "post hoc CPU diagnosis and four development controls; no physical policy performance",
        },
    )
    print(
        json.dumps(
            {
                "aggregates": aggregates,
                "linear_actions": {
                    r["arm"]: [c["selected_action"] for c in r["conditions"]] for r in controls
                },
                "visibility": visibility,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args.out)
