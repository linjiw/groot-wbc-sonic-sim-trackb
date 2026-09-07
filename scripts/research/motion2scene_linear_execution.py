#!/usr/bin/env python3
"""Execute the four P0 linear controls through the frozen learned-command interface."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from bundle_motion2scene_sources import closure
from motion2scene_layout_budget_review import review
from motion2scene_reactive_interface import physics_windows
from motion2scene_source_execution import read_cell
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import (
    OutcomePredictor,
    select_action,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)

DATA = ROOT.parent / "research-data/groot-wbc"
P0 = DATA / "m2s-selector-input-diagnostic-v2"
P1 = DATA / "m2s-selector-breakpoint-v1"
ORIGINAL = DATA / "m2s-independent-layout-v1"
PROTOCOL = ROOT / "docs/motion2scene/LINEAR_SELECTOR_EXECUTION_V1.md"
AUDITOR = ROOT / "scripts/research/audit_motion2scene_comparison_capture.py"
DRIVER = ROOT / "scripts/research/hallucination/run_approved_manifest.py"
RUNTIME = ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_learned_execution.py"
ARMS = ("uniform", "analytic", "no_contrast", "motion2scene")


def embed_linear(weights, bias):
    with torch.random.fork_rng(devices=[]):
        model = OutcomePredictor()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for action in range(2):
            for sign, offset in ((1, 0), (-1, 1)):
                i = 2 * action + offset
                model.network[0].weight[i] = sign * torch.as_tensor(weights[:, action])
                model.network[0].bias[i] = sign * float(bias[action])
                model.network[2].weight[i, i] = 1
                model.network[4].weight[action, i] = sign
    return model.eval()


def prepare(out):
    torch.set_num_threads(2)
    p0 = json.loads((P0 / "result.json").read_text())
    p1 = json.loads((P1 / "result.json").read_text())
    assert json.loads((P1 / "admission.json").read_text())["admitted"]
    parent = json.loads((DATA / "m2s-learned-command-check-v1/manifest.json").read_text())
    master = json.loads((ORIGINAL / "master.json").read_text())
    corpus = json.loads((DATA / "m2s-comparative-corpus-completion-v1/admission.json").read_text())
    evaluation = json.loads(
        (ROOT / "docs/motion2scene/evidence/independent-layout-decisions.json").read_text()
    )["rows"]
    inputs = np.array(
        [p["features"] for p in corpus["pairs"]] + [p["features"] for p in evaluation],
        dtype=np.float32,
    )
    out.mkdir(exist_ok=False)
    (out / "models").mkdir()
    policies = {}
    equivalence = []
    for control in p0["linear_controls"]:
        ref = control["model"]
        with np.load(checked(Path(ref["path"]), ref["sha256"])) as v:
            weights, bias, scale = v["weights"], v["bias"], v["scale"]
        model = embed_linear(weights, bias)
        path = out / "models" / f"{control['arm']}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "mean": np.zeros(214, np.float32),
                "std": scale,
                "role": "exact linear-function embedding, not a new fit",
                "linear_source": ref,
            },
            path,
        )
        policy_ref = artifact(path)
        policy = load_readout(path, policy_ref["sha256"])
        errors = []
        actions = True
        for x in inputs:
            with torch.no_grad():
                p = (
                    (
                        (torch.tensor(x) / torch.tensor(scale)) @ torch.tensor(weights)
                        + torch.tensor(bias)
                    )
                    .sigmoid()
                    .numpy()
                )
            actual = readout(policy, x)
            errors.append(float(abs(np.array(actual["probabilities"]) - p).max()))
            actions &= actual["selected_action"] == int(select_action(p))
        assert max(errors) <= 1e-6 and actions
        equivalence.append(
            {
                "arm": control["arm"],
                "inputs": len(inputs),
                "max_probability_error": max(errors),
                "actions_exact": bool(actions),
                "source": ref,
                "compiled": policy_ref,
            }
        )
        policies[control["arm"]] = (policy_ref, policy)
    write_new(
        out / "compile_equivalence.json",
        {
            "rows": equivalence,
            "new_fits": 0,
            "scope": "same linear functions in the existing frozen runtime container",
        },
    )
    refs = [
        artifact(PROTOCOL),
        artifact(P0 / "result.json"),
        artifact(P1 / "result.json"),
        artifact(P1 / "admission.json"),
        artifact(out / "compile_equivalence.json"),
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME, AUDITOR]))]
    refs += [v[0] for v in policies.values()]
    cells = []
    for block in master["blocks"][:6]:
        for arm in ARMS:
            c = copy.deepcopy(parent["cells"][0])
            policy_ref, policy = policies[arm]
            pair = next(p for p in p1["pairs"] if p["block"] == block["index"])
            expected = readout(policy, pair["features"])
            old = next(
                r
                for r in p1["rows"]
                if r["original_block"] == block["index"]
                and r["action"] == expected["requested_action"]
            )
            ident = f"{pair['group_id']}_{arm}"
            c.update(
                cell_id=ident,
                group_id=pair["group_id"],
                original_block=block["index"],
                beam=block["spec"]["beam"],
                scene=block["spec"]["scene"],
                policy=policy_ref,
                policy_arm=arm,
                optimizer_seed=None,
                runtime_seed=block["physics_seed"],
                expected_readout=expected,
                paired_outcome=old,
                condition="generated",
                role="shared_linear_control_execution",
                output=str(out / "rollouts" / ident),
            )
            c["hydra_overrides"] = [
                s
                for s in c["hydra_overrides"]
                if not any(
                    k in s for k in ("++seed=", "learned_policy_path=", "learned_policy_sha256=")
                )
            ]
            c["hydra_overrides"] += [
                f"++seed={block['physics_seed']}",
                f"++manager_env.config.learned_policy_path={policy_ref['path']}",
                f"++manager_env.config.learned_policy_sha256={policy_ref['sha256']}",
            ]
            cells.append(c)
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    m = copy.deepcopy(parent)
    m.update(
        experiment="M2S-shared-linear-control-v1",
        cells=cells,
        purpose="Execute four shared linear controls on six development conditions",
        registered_predictions=artifact(PROTOCOL),
    )
    m["new_dependencies"] += refs
    m["execution_policy"]["cost_ceiling"].update(rollouts=24, gpu_hours_contended=2.5)
    m["execution_policy"][
        "timing_override"
    ] = "User-prioritized shared learner control; fresh budget gate."
    write_new(out / "manifest.json", m)
    subprocess.run([sys.executable, str(AUDITOR), "register", "--out", str(out)], check=True)
    print(json.dumps({"cells": len(cells), "equivalence": equivalence}))


def analyze(out):
    m = json.loads((out / "manifest.json").read_text())
    rec = json.loads((out / "run_record.json").read_text())
    assert (
        rec["status"] == "completed"
        and rec["manifest_sha256"] == artifact(out / "manifest.json")["sha256"]
    )
    rows = []
    for c in m["cells"]:
        _, sampled, r = read_cell({**c, "condition": "present"}, rec)
        p = load_reset_capture(checked(Path(r["trajectory"]["path"]), r["trajectory"]["sha256"]))
        expected = c["paired_outcome"]
        old = load_reset_capture(
            checked(Path(expected["trajectory"]["path"]), expected["trajectory"]["sha256"])
        )
        old_d = json.loads(
            checked(Path(expected["decision"]["path"]), expected["decision"]["sha256"]).read_text()
        )
        folder = Path(c["output"]) / "trajectories"
        d = json.loads((folder / "decision_capture.json").read_text())
        s = json.loads((folder / "reactive_interface.json").read_text())
        with np.load(folder / "physics_beam_contacts.npz") as f:
            _, forces, sync = physics_windows(f, sampled)
        outcome = score_passage(p, forces, c["beam"])
        prefix = paired_prefix(old, p, 0.3)
        full = all(np.array_equal(old[k], p[k]) for k in prefix["max_abs_errors"])
        prediction = readout(
            load_readout(c["policy"]["path"], c["policy"]["sha256"]), d["features"]
        )
        match = all(
            prediction[k] == d["learned_decision"][k] == c["expected_readout"][k]
            for k in ("selected_action", "requested_action", "refusal")
        )
        error = float(
            abs(
                np.array(prediction["probabilities"]) - d["learned_decision"]["probabilities"]
            ).max()
        )
        flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
        legal = all(all(o["transition"][k] for k in flags) for o in s["observations"]) and all(
            all(x[k] for k in flags)
            and x["joint_reference_jump_rad"] <= 0.05
            and x["root_reference_jump_m"] <= 0.01
            and (0.2 <= x["time_s"] <= 0.4 if x["to"] else 3.3 <= x["time_s"] <= 3.5)
            for x in s["switches"]
        )
        matrix = np.array(r["imported_beam"]["local_to_world_at_capture_start"])
        beam = c["beam"]
        yaw = beam["yaw_rad"]
        rot = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        placement = np.allclose(
            matrix[3, :3],
            [*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2],
            atol=1e-6,
            rtol=0,
        ) and np.allclose(
            matrix[:3, :3],
            np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]])[:, None] * rot,
            atol=1e-6,
            rtol=0,
        )
        predicates = {
            "paired_prefix": prefix["exact_match"],
            "features_exact": d["features"] == old_d["features"],
            "model_request_matches": match and error <= 1e-6,
            "legal": legal,
            "placement": bool(placement),
            "contact_sync": sync <= 1e-6,
            "trace_exact": bool(full),
            "outcome_matches": all(
                outcome[k] == expected[k]
                for k in ("pass", "maximum_beam_normal_force_n_through_passage", "reset_count")
            ),
        }
        r.update(
            outcome,
            arm=c["policy_arm"],
            group_id=c["group_id"],
            original_block=c["original_block"],
            physics_seed=c["runtime_seed"],
            beam=beam,
            readout=d["learned_decision"],
            predicates=predicates,
            probability_error=error,
            return_logged=any(x["to"] == 0 for x in s["switches"]),
            entry_logged=any(x["to"] == 1 for x in s["switches"]),
            decision=artifact(folder / "decision_capture.json"),
            sensor=artifact(folder / "reactive_interface.json"),
            physics=artifact(folder / "physics_beam_contacts.npz"),
        )
        rows.append(r)
    predicates = {k: all(r["predicates"][k] for r in rows) for k in rows[0]["predicates"]}
    write_new(
        out / "result.json",
        {
            "manifest": artifact(out / "manifest.json"),
            "run_record": artifact(out / "run_record.json"),
            "rows": rows,
            "predictions": predicates,
            "actual_contended_gpu_hours": rec["budget"]["actual_contended_gpu_hours"],
            "scope": (
                "24 real shared-linear-control executions on inspected development conditions; "
                "no new source or Motion2Scene advantage claim"
            ),
        },
    )
    subprocess.run([sys.executable, str(AUDITOR), "analyze", "--out", str(out)], check=True)
    audit = json.loads((out / "capture_audit.json").read_text())
    write_new(
        out / "admission.json",
        {
            "admitted": all(predicates.values()) and all(audit["predictions"].values()),
            "result": artifact(out / "result.json"),
            "capture_audit": artifact(out / "capture_audit.json"),
        },
    )
    print(
        json.dumps(
            {
                "predictions": predicates,
                "per_arm": {
                    a: {
                        "pass": sum(r["pass"] for r in rows if r["arm"] == a),
                        "d040_requests": sum(
                            r["readout"]["requested_action"] for r in rows if r["arm"] == a
                        ),
                    }
                    for a in ARMS
                },
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("prepare", "analyze"):
        globals()[args.command](args.out)
        return
    m = json.loads((args.out / "manifest.json").read_text())
    for r in m["new_dependencies"]:
        checked(Path(r["path"]), r["sha256"])
    record = args.out / "run_record.json"
    if args.command == "run":
        old = json.loads(record.read_text()) if record.exists() else {"cells": {}}
        assert all(c["status"] in ("not_started", "completed") for c in old["cells"].values())
        gate = review(DATA, 24 - sum(c["status"] == "completed" for c in old["cells"].values()))
        write_new(
            args.out / f"budget_gate_{len(list(args.out.glob('budget_gate_*.json'))):03d}.json",
            gate,
        )
        if not gate["admitted"]:
            print("BUDGET YIELD")
            return
    cmd = [
        sys.executable,
        str(DRIVER),
        "--manifest",
        str(args.out / "manifest.json"),
        "--run-record",
        str(record),
    ]
    if args.command == "preflight":
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if args.command == "run" and json.loads(record.read_text())["status"] == "completed":
        analyze(args.out)


if __name__ == "__main__":
    main()
