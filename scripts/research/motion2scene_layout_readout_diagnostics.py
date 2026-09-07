#!/usr/bin/env python3
"""Post hoc sensor intervention and constant-phase readout, without physics claims."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_check_learned_commands import CORPUS, DATA, FIRST
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import select_action
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)

PROTOCOL = ROOT / "docs/motion2scene/LAYOUT_READOUT_DIAGNOSTIC_V1.md"


def register(out):
    fit = json.loads((CORPUS / "fits/manifest.json").read_text())
    corpus = json.loads((CORPUS / "admission.json").read_text())
    pairs = {p["group_id"]: p for p in corpus["pairs"]}
    first = json.loads((FIRST / "result.json").read_text())
    absent = next(r for r in first["rows"] if r["group_id"] == "shared_absent" and r["action"] == 0)
    models = json.loads((CORPUS / "fits/result.json").read_text())
    frequencies = {
        arm: np.array([pairs[k]["outcomes"] for k in ids], dtype=float).mean(0).tolist()
        for arm, ids in fit["fit_ids"].items()
    }
    write_new(
        out / "readout_diagnostic_registration.json",
        {
            "protocol": artifact(PROTOCOL),
            "implementation": artifact(Path(__file__)),
            "training": artifact(CORPUS / "admission.json"),
            "fit_assignment": artifact(CORPUS / "fits/manifest.json"),
            "models": [r["checkpoint"] for r in models["rows"]],
            "absent_capture": absent["decision"],
            "constant_probabilities": frequencies,
            "phase_s": 0.3,
            "scope": "post hoc diagnostic after the first completed block, not an independent hypothesis test",
            "first_inspected_block": artifact(DATA / "m2s-independent-layout-v1-b00/result.json"),
        },
    )


def analyze(out):
    started = time.monotonic()
    reg = json.loads((out / "readout_diagnostic_registration.json").read_text())
    for key in ("protocol", "implementation", "training", "fit_assignment", "absent_capture"):
        checked(Path(reg[key]["path"]), reg[key]["sha256"])
    absent = json.loads(Path(reg["absent_capture"]["path"]).read_text())
    master = json.loads((out / "master.json").read_text())
    rows, causal, histories, models = [], [], {}, {}
    for block in master["blocks"][:6]:
        if time.monotonic() - started > 180:
            raise TimeoutError("bounded CPU readout diagnostic")
        folder = Path(block["directory"])
        admission = json.loads((folder / "admission.json").read_text())
        assert admission["admitted"]
        result = json.loads(
            checked(Path(admission["result"]["path"]), admission["result"]["sha256"]).read_text()
        )
        ref = result["manifest"]
        manifest = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        for r, c in zip(result["rows"], manifest["cells"]):
            assert r["cell_id"] == c["cell_id"]
            d = json.loads(
                checked(Path(r["decision"]["path"]), r["decision"]["sha256"]).read_text()
            )
            assert c["policy"] in reg["models"]
            key = c["policy"]["sha256"]
            if key not in models:
                models[key] = load_readout(c["policy"]["path"], key)
            original = readout(models[key], d["features"])
            assert np.allclose(
                original["probabilities"], d["learned_decision"]["probabilities"], atol=1e-6, rtol=0
            )
            assert original["requested_action"] == d["requested_action"]
            x = np.array(d["features"], dtype=np.float32)
            x[:144] = absent["features"][:144]
            swapped = readout(models[key], x)
            p = reg["constant_probabilities"][r["arm"]]
            selected = int(select_action(p))
            rows.append(
                {
                    "cell_id": r["cell_id"],
                    "arm": r["arm"],
                    "original": original,
                    "absent_rays": swapped,
                    "constant_probabilities": p,
                    "constant_selected_action": selected,
                    "constant_requested_action": max(selected, 0),
                    "request_changed_by_absent_rays": original["requested_action"]
                    != swapped["requested_action"],
                    "refusal_changed_by_absent_rays": original["refusal"] != swapped["refusal"],
                    "constant_request_agrees": original["requested_action"] == max(selected, 0),
                    "constant_refusal_agrees": original["refusal"] == (selected == -1),
                }
            )
        first = result["rows"][0]
        d = json.loads(Path(first["decision"]["path"]).read_text())
        trajectory = load_reset_capture(
            checked(Path(first["trajectory"]["path"]), first["trajectory"]["sha256"])
        )
        seed = block["physics_seed"]
        if seed not in histories:
            histories[seed] = (trajectory, d["features"][144:])
        baseline, state = histories[seed]
        causal.append(
            {
                "block": block["index"],
                "physics_seed": seed,
                "prefix": paired_prefix(baseline, trajectory, 0.3),
                "non_ray_features_exact": state == d["features"][144:],
            }
        )
    assert len(rows) == 120
    arms = {}
    for arm in reg["constant_probabilities"]:
        selected = [r for r in rows if r["arm"] == arm]
        arms[arm] = {
            "measured_readouts": len(selected),
            **{
                k: sum(r[k] for r in selected)
                for k in (
                    "request_changed_by_absent_rays",
                    "refusal_changed_by_absent_rays",
                    "constant_request_agrees",
                    "constant_refusal_agrees",
                )
            },
        }
    write_new(
        out / "readout_diagnostic_result.json",
        {
            "registration": artifact(out / "readout_diagnostic_registration.json"),
            "rows": rows,
            "arms": arms,
            "cross_layout_inputs": causal,
            "seconds": time.monotonic() - started,
            "physics_executions": 0,
            "scope": "post hoc readout sensitivity; no baseline avoidance outcome inferred",
        },
    )
    print(json.dumps(arms))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args.out)
