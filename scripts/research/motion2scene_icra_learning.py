#!/usr/bin/env python3
"""Fit the one-budget four-arm corpus and instantiate actual policy evaluation assignments."""

import argparse
import copy
import json
from pathlib import Path

from bundle_motion2scene_sources import closure
from motion2scene_development_bank import DATA
from motion2scene_icra_compare import summarize
from motion2scene_icra_eval_audit import execute
from motion2scene_icra_study import ARMS, BACKGROUNDS, BANK, physical_scene
from motion2scene_linear_execution_v2 import embed_linear
from motion2scene_selector_diagnosis_v2 import low_capacity
from motion2scene_source_execution import beam_at
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_transition_construction import cases
from motion2scene_uncertainty_learning import numpy_perturbed
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import select_action

PROTOCOL = ROOT / "docs/motion2scene/M2S_ICRA_EXECUTION_V1.md"
RUNTIME = ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_icra_execution.py"
STUDY = DATA / "m2s-icra-v1"
METHODS = (*ARMS, "scripted_rays", "privileged_geometry")


def register(out):
    out.mkdir(exist_ok=False)
    refs = [
        artifact(PROTOCOL),
        artifact(STUDY / "registration.json"),
        artifact(STUDY / "prepared.json"),
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME]))]
    for r in refs:
        checked(Path(r["path"]), r["sha256"])
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "arms": ARMS,
            "methods": METHODS,
            "maximum_fit_groups_per_arm": 24,
            "primary_fits": 4,
            "diagnostic_refits": 24,
            "traversal_assignments": 432,
            "background_assignments": 108,
            "no_refits_on_test": True,
        },
    )


def fit(out):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    reg = json.loads((out / "registration.json").read_text())
    for r in reg["references"]:
        checked(Path(r["path"]), r["sha256"])
    prepared = json.loads((STUDY / "prepared.json").read_text())
    proposals = json.loads((STUDY / "proposals.json").read_text())
    pairs, records = [], []
    for b in prepared["batches"]:
        folder = Path(b["directory"])
        a = json.loads((folder / "admission.json").read_text())
        assert a["admitted"], folder
        result = json.loads(checked(Path(a["result"]["path"]), a["result"]["sha256"]).read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        for p in result["pairs"]:
            assert p["valid"] and all(p["commands_executed"])
            c = next(c for c in manifest["cells"] if c["group_id"] == p["group_id"])
            pairs.append({**p, "arm": c["arm"], "suite": c["suite"]})
        records.append(artifact(folder / "admission.json"))
    assert len(pairs) * 2 == prepared["physics_assigned"]
    by_id = {p["group_id"]: p for p in pairs}
    assert len(by_id) == len(pairs)
    evaluation_inputs = np.array([p["features"] for p in pairs], np.float32)
    (out / "models").mkdir()
    fits, folds = [], []
    for arm in ARMS:
        assigned = sorted(
            [r["group_id"] for r in proposals["rows"] if r["assigned"] and r["arm"] == arm]
            + [r["group_id"] for r in proposals["backgrounds"]]
        )
        assert len(assigned) == 24
        ids = [k for k in assigned if k in by_id]
        assert len(ids) >= 6
        x = np.array([by_id[k]["features"] for k in ids], np.float32)
        y = np.array([by_id[k]["outcomes"] for k in ids], np.float32)
        saved, predictions, loss = low_capacity(x, y, evaluation_inputs)
        path = out / "models" / f"{arm}.npz"
        np.savez_compressed(path, **saved)
        embedded = embed_linear(saved["weights"], saved["bias"])
        compiled = path.with_suffix(".pt")
        torch.save(
            {
                "model": embedded.state_dict(),
                "mean": np.zeros(214, np.float32),
                "std": saved["scale"],
                "linear_source": artifact(path),
            },
            compiled,
        )
        ref = artifact(compiled)
        runtime = load_readout(ref["path"], ref["sha256"])
        actual = [readout(runtime, v) for v in evaluation_inputs]
        error = float(abs(np.array([r["probabilities"] for r in actual]) - predictions).max())
        assert error <= 2e-6
        assert np.array_equal([r["selected_action"] for r in actual], select_action(predictions))
        fits.append(
            {
                "arm": arm,
                "training_ids": ids,
                "requested_ids": assigned,
                "missing_ids": [k for k in assigned if k not in by_id],
                "training_bce": loss,
                "model": ref,
                "linear": artifact(path),
                "compile_probability_error": error,
                "outcome_counts": {
                    str(key): sum(tuple(by_id[k]["outcomes"]) == key for k in ids)
                    for key in ((False, False), (False, True), (True, False), (True, True))
                },
            }
        )
        for fi in range(6):
            held = assigned[4 * fi : 4 * fi + 4]
            train_ids = [k for k in ids if k not in held]
            held_ids = [k for k in held if k in by_id]
            tx = np.array([by_id[k]["features"] for k in train_ids], np.float32)
            ty = np.array([by_id[k]["outcomes"] for k in train_ids], np.float32)
            ex = np.array([by_id[k]["features"] for k in held_ids], np.float32).reshape(-1, 214)
            weights, pred, bce = low_capacity(tx, ty, ex)
            fp = out / "models" / f"{arm}_leave4_{fi}.npz"
            np.savez_compressed(fp, **weights)
            folds.append(
                {
                    "arm": arm,
                    "fold": fi,
                    "withheld_assigned": held,
                    "withheld_labelled": held_ids,
                    "train_ids": train_ids,
                    "probabilities": pred.tolist(),
                    "selected_actions": select_action(pred).tolist(),
                    "training_bce": bce,
                    "model": artifact(fp),
                }
            )
    write_new(
        out / "fit.json",
        {
            "registration": artifact(out / "registration.json"),
            "admissions": records,
            "pairs": pairs,
            "fits": fits,
            "folds": folds,
            "equal_24_complete_labels": all(len(f["training_ids"]) == 24 for f in fits),
            "scope": "Four actual deterministic fits; leave-four refits are offline sensitivity, not rollouts",
        },
    )
    print(
        json.dumps({"training_counts": {f["arm"]: len(f["training_ids"]) for f in fits}}),
        flush=True,
    )


def prepare(out):
    fitted = json.loads((out / "fit.json").read_text())
    reg = json.loads((out / "registration.json").read_text())
    for r in reg["references"]:
        checked(Path(r["path"]), r["sha256"])
    construction = json.loads((STUDY / "registration.json").read_text())
    bank = json.loads((BANK / "result.json").read_text())
    parent = json.loads((BANK / "manifest.json").read_text())
    models = {f["arm"]: f["model"] for f in fitted["fits"]}
    (out / "scenes").mkdir()
    (out / "comparators").mkdir()
    scripted = out / "comparators/scripted.json"
    write_new(scripted, {"kind": "scripted_rays", "decision_time_s": 0.3})
    layouts = construction["layouts"] + [
        {"id": kind, "suite": kind, "station": 0.6, "underside_m": h}
        for kind, h in [("absent", 1.5), ("raised", 2.0), ("blocked", 0.0)]
    ]
    clouds = {s: cases(s, bank)[0] for s in BACKGROUNDS}
    batches = []
    for layout in layouts:
        for seed in construction["evaluation_seeds"]:
            for source in BACKGROUNDS:
                case = clouds[source]
                group = f"icra_eval_{source}_{layout['id']}_p{seed}"
                folder = out.with_name(out.name + f"-eval-{len(batches):03d}")
                folder.mkdir()
                beam = {
                    **beam_at(case, layout["station"], layout["underside_m"]),
                    "thickness_m": 1.4 if layout["suite"] == "blocked" else 0.1,
                }
                scene = physical_scene(
                    out / "scenes" / f"{group}.usda", beam, layout["suite"] == "absent"
                )
                if layout["suite"] == "blocked":
                    feasible = np.array([False, False])
                    scope = (
                        "blocked family predeclared command-bank inadequacy; not a successful stop"
                    )
                elif layout["suite"] == "absent":
                    feasible = np.array([True, True])
                    scope = "empty-bank command qualification"
                else:
                    points = np.array([[layout["station"], layout["underside_m"]]])
                    clearance = numpy_perturbed(
                        points, np.zeros((1, 4)), case["states"], case["route"], case["yaw"].item()
                    )[0, 0]
                    feasible = clearance >= 0.01
                    scope = "nominal achieved-capsule forecast with 10 mm clearance, not an optimal policy"
                action = 0 if feasible[0] else 1 if feasible[1] else -1
                privileged = out / "comparators" / f"{group}.json"
                write_new(
                    privileged,
                    {
                        "kind": "privileged_geometry",
                        "selected_action": action,
                        "forecast_feasible": feasible.tolist(),
                        "scope": scope,
                        "beam": beam,
                    },
                )
                cells = []
                for method in METHODS:
                    old = next(
                        c
                        for c in parent["cells"]
                        if c["generation_seed"] == source and c["encounter_action"] == 0
                    )
                    prior = next(r for r in bank["rows"] if r["cell_id"] == old["cell_id"])
                    c = copy.deepcopy(old)
                    ref = (
                        artifact(scripted)
                        if method == "scripted_rays"
                        else (
                            artifact(privileged)
                            if method == "privileged_geometry"
                            else models[method]
                        )
                    )
                    ident = group + "_" + method
                    c.update(
                        cell_id=ident,
                        group_id=group,
                        arm=method,
                        suite=layout["suite"],
                        layout=layout["id"],
                        runtime_seed=seed,
                        condition="absent" if layout["suite"] == "absent" else "present",
                        beam=beam,
                        scene=scene,
                        policy=ref,
                        qualified_bank=prior["bank"],
                        role="icra_policy_evaluation",
                        output=str(folder / "rollouts" / ident),
                    )
                    c["hydra_overrides"] = [
                        v
                        for v in c["hydra_overrides"]
                        if not any(
                            k in v
                            for k in (
                                "++seed=",
                                "++manager_env._target_=",
                                "++manager_env.recorders.trajectory._target_=",
                            )
                        )
                    ]
                    module = (
                        "gear_sonic.dataset_generation.hallucination.motion2scene_icra_execution"
                    )
                    c["hydra_overrides"] += [
                        f"++seed={seed}",
                        f"++manager_env._target_={module}.LearnedEnvCfg",
                        f"++manager_env.recorders.trajectory._target_={module}.LearnedRecorderCfg",
                        f"++manager_env.config.learned_policy_path={ref['path']}",
                        f"++manager_env.config.learned_policy_sha256={ref['sha256']}",
                    ]
                    cells.append(c)
                m = copy.deepcopy(parent)
                m.update(
                    cells=cells,
                    experiment=group,
                    registered_predictions=artifact(PROTOCOL),
                    purpose="Matched fixed-policy ICRA evaluation",
                )
                m["new_dependencies"] += (
                    reg["references"]
                    + [artifact(out / "fit.json")]
                    + [c["scene"] for c in cells]
                    + [c["policy"] for c in cells]
                    + [c["qualified_bank"] for c in cells]
                )
                m["execution_policy"]["cost_ceiling"].update(
                    rollouts=6, gpu_hours_contended=6 * 375 / 3600
                )
                write_new(folder / "manifest.json", m)
                batches.append(
                    {
                        "directory": str(folder),
                        "manifest": artifact(folder / "manifest.json"),
                        "assigned": 6,
                        "source": source,
                        "layout": layout,
                        "physics_seed": seed,
                    }
                )
    assert len(batches) == 90
    write_new(
        out / "evaluation_master.json",
        {
            "batches": batches,
            "traversal": 432,
            "background": 108,
            "total": 540,
            "role": "Serial assignment index, not a single executable batch",
        },
    )


def run(out):
    master = json.loads((out / "evaluation_master.json").read_text())
    for b in master["batches"]:
        folder = Path(b["directory"])
        if (folder / "admission.json").exists():
            assert json.loads((folder / "admission.json").read_text())["admitted"]
            continue
        execute(folder, preflight=True)
        execute(folder)
        if not (folder / "admission.json").exists():
            return
        assert json.loads((folder / "admission.json").read_text())["admitted"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("register", "fit", "prepare", "run", "summarize"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {
        "register": register,
        "fit": fit,
        "prepare": prepare,
        "run": run,
        "summarize": summarize,
    }[a.command](a.out)
