#!/usr/bin/env python3
"""Qualify learned requests against existing explicit-action traces on development data."""

import argparse
import copy
import datetime
import json
from pathlib import Path
import subprocess
import sys

from bundle_motion2scene_sources import closure
from motion2scene_comparative_acquisition import DATA
from motion2scene_reactive_interface import physics_windows
from motion2scene_source_execution import read_cell
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)

FIRST = DATA / "m2s-comparative-acquisition-v1"
CORPUS = DATA / "m2s-comparative-corpus-completion-v1"
PROTOCOL = ROOT / "docs/motion2scene/LEARNED_COMMAND_CHECK_V1.md"
RUNTIME = ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_learned_execution.py"
GROUPS = ("analytic_01", "shared_absent", "shared_blocked")


def prepare(out):
    fits = json.loads((CORPUS / "fits/result.json").read_text())
    admission = json.loads((CORPUS / "admission.json").read_text())
    assert fits["fits"] == 20 and admission["all_inputs_admitted"]
    out.mkdir(exist_ok=False)
    parent = json.loads((FIRST / "manifest.json").read_text())
    results = json.loads((FIRST / "result.json").read_text())
    refs = [
        artifact(PROTOCOL),
        artifact(CORPUS / "fits/result.json"),
        artifact(CORPUS / "admission.json"),
        artifact(FIRST / "result.json"),
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME]))]
    cells = []
    for fit in fits["rows"]:
        if fit["seed"] != 8501:
            continue
        ref = fit["checkpoint"]
        refs.append(ref)
        model = load_readout(ref["path"], ref["sha256"])
        for group in GROUPS:
            base = next(
                c for c in parent["cells"] if c["group_id"] == group and c["encounter_action"] == 0
            )
            first = next(r for r in results["rows"] if r["group_id"] == group and r["action"] == 0)
            capture = json.loads(
                checked(Path(first["decision"]["path"]), first["decision"]["sha256"]).read_text()
            )
            expected = readout(model, capture["features"])
            paired = next(
                r
                for r in results["rows"]
                if r["group_id"] == group and r["action"] == expected["requested_action"]
            )
            c = copy.deepcopy(base)
            c.update(
                cell_id=f"{fit['arm']}_{group}",
                policy_arm=fit["arm"],
                policy=ref,
                expected_readout=expected,
                paired_outcome=paired,
                mode="learned",
            )
            c["output"] = str(out / "rollouts" / c["cell_id"])
            c["hydra_overrides"] = [
                s.replace(
                    "motion2scene_comparison_execution.ComparisonEnvCfg",
                    "motion2scene_learned_execution.LearnedEnvCfg",
                ).replace(
                    "motion2scene_comparison_execution.ComparisonRecorderCfg",
                    "motion2scene_learned_execution.LearnedRecorderCfg",
                )
                for s in c["hydra_overrides"]
            ]
            c["hydra_overrides"] += [
                f"++manager_env.config.learned_policy_path={ref['path']}",
                f"++manager_env.config.learned_policy_sha256={ref['sha256']}",
            ]
            cells.append(c)
    assert len(cells) == 12
    m = copy.deepcopy(parent)
    m.update(
        experiment="M2S-learned-command-check-v1",
        cells=cells,
        purpose="Learned command integration on observed scenes, not unseen evaluation",
        registered_predictions=artifact(PROTOCOL),
    )
    m["new_dependencies"] += refs
    m["execution_policy"]["cost_ceiling"].update(rollouts=12, gpu_hours_contended=1.25)
    m["execution_policy"][
        "timing_override"
    ] = "Bounded learned-command integration after admitted corpus and fits."
    write_new(out / "manifest.json", m)


def analyze(out):
    m = json.loads((out / "manifest.json").read_text())
    record = json.loads((out / "run_record.json").read_text())
    assert (
        record["status"] == "completed"
        and record["manifest_sha256"] == artifact(out / "manifest.json")["sha256"]
    )
    rows = []
    for c in m["cells"]:
        p, sampled, r = read_cell(
            {**c, "condition": "absent" if c["condition"] == "absent" else "present"}, record
        )
        p = load_reset_capture(checked(Path(r["trajectory"]["path"]), r["trajectory"]["sha256"]))
        expected = c["paired_outcome"]
        old = load_reset_capture(
            checked(Path(expected["trajectory"]["path"]), expected["trajectory"]["sha256"])
        )
        old_capture = json.loads(
            checked(Path(expected["decision"]["path"]), expected["decision"]["sha256"]).read_text()
        )
        folder = Path(c["output"]) / "trajectories"
        capture = json.loads((folder / "decision_capture.json").read_text())
        keys = (
            "dof_pos",
            "dof_vel",
            "root_pos_w",
            "root_quat_w",
            "applied_joint_action",
            "action_motion_token",
            "reference_g1_qpos",
            "motion_time_s",
        )
        trace = {k: bool(np.array_equal(p[k], old[k])) for k in keys}
        with np.load(folder / "physics_beam_contacts.npz") as f:
            _, forces, sync = physics_windows(f, sampled)
        outcome = score_passage(p, forces, c["beam"])
        d = capture["learned_decision"]
        probability_error = float(
            np.max(abs(np.array(d["probabilities"]) - c["expected_readout"]["probabilities"]))
        )
        predicates = {
            "feature_exact": capture["features"] == old_capture["features"],
            "model_request_matches": probability_error <= 1e-6
            and all(
                d[k] == c["expected_readout"][k]
                for k in ("selected_action", "requested_action", "refusal")
            ),
            "trace_exact": all(trace.values()),
            "outcome_matches": all(
                outcome[k] == expected[k]
                for k in (
                    "pass",
                    "observed_beam_contact",
                    "reset_count",
                    "maximum_beam_normal_force_n_through_passage",
                )
            ),
        }
        rows.append(
            {
                **r,
                **outcome,
                "policy_arm": c["policy_arm"],
                "group_id": c["group_id"],
                "readout": d,
                "probability_error": probability_error,
                "trace_fields": trace,
                "predicates": predicates,
                "contact_sync_error_n": sync,
                "decision": artifact(folder / "decision_capture.json"),
            }
        )
    predictions = {key: all(r["predicates"][key] for r in rows) for key in rows[0]["predicates"]}
    write_new(
        out / "result.json",
        {
            "manifest": artifact(out / "manifest.json"),
            "run_record": artifact(out / "run_record.json"),
            "rows": rows,
            "predictions": predictions,
            "actual_contended_gpu_hours": record["budget"]["actual_contended_gpu_hours"],
            "scope": (
                "12 real learned-policy executions on observed development scenes; "
                "not learning utility or independent-layout evaluation"
            ),
        },
    )
    print(json.dumps(predictions))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command in ("prepare", "analyze"):
        globals()[a.command](a.out)
        return
    if a.command == "run":
        now = datetime.datetime.now(datetime.timezone.utc)
        entries = []
        for path in DATA.glob("m2s-*/*run_record*.json"):
            r = json.loads(path.read_text())
            hours = float(r.get("budget", {}).get("actual_contended_gpu_hours", 0))
            if hours:
                age = (
                    now - datetime.datetime.fromisoformat(r["started_at"])
                ).total_seconds() / 86400
                entries.append({"path": str(path), "hours": hours, "age_days": age})
        daily = sum(r["hours"] for r in entries if r["age_days"] <= 1)
        weekly = sum(r["hours"] for r in entries if r["age_days"] <= 7)
        assert daily + 1.25 <= 8 and weekly + 1.25 <= 24, (daily, weekly)
        write_new(
            a.out / "launch_gate.json",
            {
                "records": entries,
                "daily_hours": daily,
                "weekly_hours": weekly,
                "reserved_hours": 1.25,
            },
        )
    cmd = [
        sys.executable,
        str(ROOT / "scripts/research/hallucination/run_approved_manifest.py"),
        "--manifest",
        str(a.out / "manifest.json"),
        "--run-record",
        str(a.out / "run_record.json"),
    ]
    if a.command == "preflight":
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if a.command == "run":
        analyze(a.out)


if __name__ == "__main__":
    main()
