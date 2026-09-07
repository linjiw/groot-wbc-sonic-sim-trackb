#!/usr/bin/env python3
"""Fit the nominal-contract corpus and instantiate its reduced policy evaluation.

Mirrors the M2S-ICRA-v1 learning stage with three registered differences, and no
others: the training corpus is the nominal-contract labels, the six shared background
groups are reused from M2S-ICRA-v1 by reference rather than re-executed, and the
evaluation runs the twelve reserved layouts at physics seed 8511 only, for the four
fitted arms. The scripted-ray and privileged-geometry comparators do not depend on the
training arm, so their existing M2S-ICRA-v1 measurements on exactly these conditions
are reused and reported as such; comparisons against them are restricted to seed 8511.

The learner, its scaling, its optimizer, its selection rule, the 214 features, the
source banks, the scorer and the reserved layouts are unchanged.
"""

import argparse
import copy
import json
from pathlib import Path

from bundle_motion2scene_sources import closure
from motion2scene_development_bank import DATA
from motion2scene_icra_eval_audit import execute
from motion2scene_icra_study import ARMS, BACKGROUNDS, BANK, physical_scene
from motion2scene_linear_execution_v2 import embed_linear
from motion2scene_selector_diagnosis_v2 import low_capacity
from motion2scene_source_execution import beam_at
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_transition_construction import cases
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import select_action

PROTOCOL = ROOT / "docs/motion2scene/M2S_ICRA_NOMINAL_V1.md"
RUNTIME = ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_icra_execution.py"
PARENT = DATA / "m2s-icra-v1"
STUDY = DATA / "m2s-icra-nominal-v1"
EVALUATION_SEED = 8511


def background_pairs():
    """The six shared background groups, taken from M2S-ICRA-v1's admitted labels."""
    prepared = json.loads((PARENT / "prepared.json").read_text())
    pairs = {}
    for b in prepared["batches"]:
        folder = Path(b["directory"])
        a = json.loads((folder / "admission.json").read_text())
        assert a["admitted"], folder
        result = json.loads(checked(Path(a["result"]["path"]), a["result"]["sha256"]).read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        for p in result["pairs"]:
            cell = next(c for c in manifest["cells"] if c["group_id"] == p["group_id"])
            if cell["arm"] == "shared":
                assert p["valid"] and all(p["commands_executed"])
                pairs[p["group_id"]] = {**p, "arm": "shared", "suite": cell["suite"]}
    assert len(pairs) == 6, sorted(pairs)
    return pairs


def register(out):
    out.mkdir(exist_ok=False)
    refs = [
        artifact(PROTOCOL),
        artifact(STUDY / "registration.json"),
        artifact(STUDY / "prepared.json"),
        artifact(STUDY / "proposals.json"),
        artifact(PARENT / "prepared.json"),
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME]))]
    for r in refs:
        checked(Path(r["path"]), r["sha256"])
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "parent_study": str(PARENT),
            "construction": str(STUDY),
            "arms": list(ARMS),
            "methods": list(ARMS),
            "comparators": "scripted_rays and privileged_geometry reused from M2S-ICRA-v1 at seed 8511",
            "evaluation_seed": EVALUATION_SEED,
            "traversal_assignments": 144,
            "background_assignments": 0,
            "backgrounds": "six M2S-ICRA-v1 label groups reused by reference in fitting",
            "primary_fits": 4,
            "no_refits_on_test": True,
            "scope": (
                "Reduced single-seed evaluation of the nominal contract on the same "
                "reserved layouts and inspected development carriers"
            ),
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
    by_id = dict(background_pairs())
    records = []
    for b in prepared["batches"]:
        folder = Path(b["directory"])
        a = json.loads((folder / "admission.json").read_text())
        assert a["admitted"], folder
        result = json.loads(checked(Path(a["result"]["path"]), a["result"]["sha256"]).read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        for p in result["pairs"]:
            assert p["valid"] and all(p["commands_executed"])
            cell = next(c for c in manifest["cells"] if c["group_id"] == p["group_id"])
            by_id[p["group_id"]] = {**p, "arm": cell["arm"], "suite": cell["suite"]}
        records.append(artifact(folder / "admission.json"))
    generated = [k for k, v in by_id.items() if v["arm"] != "shared"]
    assert len(generated) * 2 == prepared["physics_assigned"]
    evaluation_inputs = np.array([by_id[k]["features"] for k in sorted(by_id)], np.float32)
    (out / "models").mkdir()
    fits = []
    for arm in ARMS:
        assigned = sorted(
            [r["group_id"] for r in proposals["rows"] if r["assigned"] and r["arm"] == arm]
            + [r["group_id"] for r in proposals["reused_backgrounds"]]
        )
        ids = [k for k in assigned if k in by_id]
        assert len(ids) >= 6, arm
        x = np.array([by_id[k]["features"] for k in ids], np.float32)
        y = np.array([by_id[k]["outcomes"] for k in ids], np.float32)
        saved, predictions, loss = low_capacity(x, y, evaluation_inputs)
        path = out / "models" / f"{arm}.npz"
        np.savez_compressed(path, **saved)
        compiled = path.with_suffix(".pt")
        torch.save(
            {
                "model": embed_linear(saved["weights"], saved["bias"]).state_dict(),
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
        counts = {}
        for k in ids:
            counts[str(tuple(by_id[k]["outcomes"]))] = (
                counts.get(str(tuple(by_id[k]["outcomes"])), 0) + 1
            )
        fits.append(
            {
                "arm": arm,
                "requested_ids": assigned,
                "training_ids": ids,
                "missing_ids": [k for k in assigned if k not in by_id],
                "outcome_counts": counts,
                "training_bce": float(loss),
                "compile_probability_error": error,
                "model": ref,
            }
        )
    write_new(
        out / "fit.json",
        {
            "registration": artifact(out / "registration.json"),
            "admissions": records,
            "fits": fits,
            "reused_background_ids": sorted(background_pairs()),
            "equal_label_counts": len({len(f["training_ids"]) for f in fits}) == 1,
            "scope": (
                "Nominal-contract corpus with M2S-ICRA-v1 backgrounds reused by reference; "
                "unequal acquired counts are reported, never refilled"
            ),
        },
    )
    print(
        json.dumps(
            {f["arm"]: {"groups": len(f["training_ids"]), **f["outcome_counts"]} for f in fits},
            indent=1,
        ),
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
    clouds = {s: cases(s, bank)[0] for s in BACKGROUNDS}
    batches = []
    for layout in construction["layouts"]:
        for source in BACKGROUNDS:
            case = clouds[source]
            group = f"nom_eval_{source}_{layout['id']}_p{EVALUATION_SEED}"
            folder = out.with_name(out.name + f"-eval-{len(batches):03d}")
            folder.mkdir()
            beam = {
                **beam_at(case, layout["station"], layout["underside_m"]),
                "thickness_m": 0.1,
            }
            scene = physical_scene(out / "scenes" / f"{group}.usda", beam)
            cells = []
            for method in ARMS:
                old = next(
                    c
                    for c in parent["cells"]
                    if c["generation_seed"] == source and c["encounter_action"] == 0
                )
                prior = next(r for r in bank["rows"] if r["cell_id"] == old["cell_id"])
                c = copy.deepcopy(old)
                ref = models[method]
                ident = group + "_" + method
                c.update(
                    cell_id=ident,
                    group_id=group,
                    arm=method,
                    suite="traversal",
                    layout=layout["id"],
                    runtime_seed=EVALUATION_SEED,
                    condition="present",
                    beam=beam,
                    scene=scene,
                    policy=ref,
                    qualified_bank=prior["bank"],
                    role="icra_nominal_policy_evaluation",
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
                module = "gear_sonic.dataset_generation.hallucination.motion2scene_icra_execution"
                c["hydra_overrides"] += [
                    f"++seed={EVALUATION_SEED}",
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
                purpose="Nominal-contract fixed-policy evaluation",
            )
            m["new_dependencies"] += (
                reg["references"]
                + [artifact(out / "fit.json")]
                + [c["scene"] for c in cells]
                + [c["policy"] for c in cells]
                + [c["qualified_bank"] for c in cells]
            )
            m["execution_policy"]["cost_ceiling"].update(
                rollouts=len(cells), gpu_hours_contended=len(cells) * 375 / 3600
            )
            write_new(folder / "manifest.json", m)
            batches.append(
                {
                    "directory": str(folder),
                    "manifest": artifact(folder / "manifest.json"),
                    "assigned": len(cells),
                    "source": source,
                    "layout": layout,
                    "physics_seed": EVALUATION_SEED,
                }
            )
    assert len(batches) == 36 and sum(b["assigned"] for b in batches) == 144
    write_new(
        out / "evaluation_master.json",
        {
            "batches": batches,
            "traversal": 144,
            "background": 0,
            "total": 144,
            "comparators": (
                "scripted_rays and privileged_geometry reused from M2S-ICRA-v1 at this seed; "
                "not re-executed here"
            ),
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
    p.add_argument("command", choices=("register", "fit", "prepare", "run"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"register": register, "fit": fit, "prepare": prepare, "run": run}[a.command](a.out)
