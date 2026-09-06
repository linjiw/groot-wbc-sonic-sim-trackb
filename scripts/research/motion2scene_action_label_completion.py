#!/usr/bin/env python3
"""Complete paired action labels and test three existing legal transition times."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from motion2scene_reactive_interface import physics_windows
from motion2scene_source_execution import read_cell
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_policy_features import (
    feasibility_label,
)
from gear_sonic.dataset_generation.trajectory_export import load_validated_trajectory

DATA = ROOT.parent / "research-data/groot-wbc"
PRIOR = DATA / "m2s-overhang-variation-v1"
PREFIX = "gear_sonic.dataset_generation.hallucination.motion2scene_action_execution"
PROTOCOL = ROOT / "docs/motion2scene/ACTION_LABEL_COMPLETION_V1.md"


def prepare(out):
    m = json.loads((PRIOR / "manifest.json").read_text())
    r = json.loads((PRIOR / "result.json").read_text())
    assert len(r["rows"]) == 42 and r["predictions"]["p5_measurement_contract"]
    for ref in m["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    out.mkdir(parents=True, exist_ok=False)
    cells = []
    for seed in (8041, 8042):
        for condition in ("absent", "raised", "blocked"):
            c = copy.deepcopy(
                next(
                    c
                    for c in m["cells"]
                    if c["runtime_seed"] == seed and c["condition"] == condition
                )
            )
            c.update(mode="oracle", role="missing_control", decision_time_s=0.2)
            c["hydra_overrides"] = [
                s.replace("reactive_mode=reactive", "reactive_mode=oracle")
                for s in c["hydra_overrides"]
            ]
            c["cell_id"] = f"labels_{seed}_{condition}_oracle"
            c["output"] = str(out / "rollouts" / c["cell_id"])
            cells.append(c)
    for seed in (8041, 8042):
        for t in (0.2, 0.3, 0.4):
            c = copy.deepcopy(
                next(
                    c
                    for c in m["cells"]
                    if c["runtime_seed"] == seed
                    and c["condition"] == "station_minus15cm"
                    and c["mode"] == "oracle"
                )
            )
            c.update(role="timing", decision_time_s=t)
            c["cell_id"] = f"timing_{seed}_{int(t*1000)}ms"
            c["output"] = str(out / "rollouts" / c["cell_id"])
            c["hydra_overrides"] = [
                s.replace(
                    "motion2scene_delayed_overhang_execution.DelayedOverhangEnvCfg",
                    "motion2scene_action_execution.ActionRequestEnvCfg",
                ).replace(
                    "motion2scene_delayed_overhang_execution.DelayedOverhangRecorderCfg",
                    "motion2scene_action_execution.ActionRequestRecorderCfg",
                )
                for s in c["hydra_overrides"]
            ]
            c["hydra_overrides"] += [
                "++manager_env.config.encounter_action=1",
                f"++manager_env.config.decision_time_s={t}",
            ]
            cells.append(c)
    m.update(
        experiment="M2S-action-label-completion-v1",
        cells=cells,
        purpose="Six missing d040 controls and six explicitly commanded timing cells",
        registered_predictions=artifact(PROTOCOL),
        prior_variation_result=artifact(PRIOR / "result.json"),
    )
    m["new_dependencies"] += [
        artifact(Path(__file__)),
        artifact(PROTOCOL),
        artifact(PRIOR / "manifest.json"),
        artifact(PRIOR / "result.json"),
    ]
    m["new_dependencies"] += [
        artifact(ROOT / "gear_sonic/dataset_generation/hallucination" / f"{n}.py")
        for n in (
            "motion2scene_action_contract",
            "motion2scene_action_execution",
            "motion2scene_policy_features",
        )
    ]
    m["execution_policy"]["cost_ceiling"].update(rollouts=12, gpu_hours_contended=1.25)
    m["execution_policy"]["timing_override"] = (
        "User directs label completion and bounded timing study; prior variation total 0.400278 h; "
        "12-cell ceiling 1.25 h within standing envelope."
    )
    m["comparison_contract"] = (
        "Same scene, seed and recorded pre-decision state/action/token history. One development ancestor. "
        "Whole-motion and wait/reconsider labels excluded."
    )
    write_new(out / "manifest.json", m)


def payload(row):
    ref = row["trajectory"]
    return load_validated_trajectory(checked(Path(ref["path"]), ref["sha256"]))


def sensor(row):
    ref = row["sensor"]
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


def pairing(a, b, t):
    audit = paired_prefix(payload(a), payload(b), t)
    sa, sb = sensor(a), sensor(b)
    pa = next(o for o in sa["observations"] if o["time_s"] == t)
    pb = next(o for o in sb["observations"] if o["time_s"] == t)
    audit["decision_packet_matches"] = all(pa[k] == pb[k] for k in ("origin", "rays", "occupied"))
    audit["valid"] = audit["exact_match"] and audit["decision_packet_matches"]
    return audit


def analyze(out):
    m = json.loads((out / "manifest.json").read_text())
    record = json.loads((out / "run_record.json").read_text())
    assert (
        record["status"] == "completed"
        and record["manifest_sha256"] == artifact(out / "manifest.json")["sha256"]
    )
    old = json.loads(
        checked(
            Path(m["prior_variation_result"]["path"]), m["prior_variation_result"]["sha256"]
        ).read_text()
    )
    rows = []
    for c in m["cells"]:
        p, sampled, row = read_cell(
            {**c, "condition": "absent" if c["condition"] == "absent" else "present"}, record
        )
        folder = Path(c["output"]) / "trajectories"
        beam = c["beam"]
        matrix = np.asarray(row["imported_beam"]["local_to_world_at_capture_start"])
        a = beam["yaw_rad"]
        rot = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
        assert np.allclose(
            matrix[3, :3],
            [*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2],
            atol=1e-6,
        )
        assert np.allclose(
            matrix[:3, :3],
            np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]])[:, None] * rot,
            atol=1e-6,
        )
        s = json.loads((folder / "reactive_interface.json").read_text())
        obs = s["observations"]
        assert len(obs) == 199
        for i, o in enumerate(obs):
            expected = {k: o[k] for k in ("time_s", "occupied", "origin", "rays")}
            expected.update(capture_frame=i, capture_elapsed_s=i / 50)
            assert o["capture_frame"] == i and o["delivered"] == expected
        with np.load(folder / "physics_beam_contacts.npz") as f:
            blocks, aggregate, error = physics_windows(f, sampled)
        assert blocks.shape == (199, 4, 30, 3)
        score = score_passage(p, aggregate, beam)
        horizon = score["passage_finish_frame_exclusive"] or score["first_episode_frames"]
        norms = np.linalg.norm(blocks[:horizon].reshape(-1, 30, 3), axis=-1)
        neutral = next(
            r
            for r in old["rows"]
            if r["seed"] == c["runtime_seed"]
            and r["condition"] == c["condition"]
            and r["mode"] == ("blind" if c["role"] == "timing" else "reactive")
        )
        with np.load(folder / "loaded_reference_bank.npz") as bank, np.load(
            checked(Path(neutral["bank"]["path"]), neutral["bank"]["sha256"])
        ) as base:
            assert all(np.array_equal(bank[k], base[k]) for k in ("root_xyz", "joint_pos", "fps"))
        flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
        legal = all(all(o["transition"][k] for k in flags) for o in obs) and all(
            all(s[k] for k in flags)
            and s["joint_reference_jump_rad"] <= 0.05
            and s["root_reference_jump_m"] <= 0.01
            and (0.2 <= s["time_s"] <= 0.4 if s["to"] else 3.3 <= s["time_s"] <= 3.5)
            for s in s["switches"]
        )
        row.update(
            score,
            condition=c["condition"],
            role=c["role"],
            seed=c["runtime_seed"],
            decision_time_s=c["decision_time_s"],
            beam=beam,
            sensor=artifact(folder / "reactive_interface.json"),
            physics_contacts=artifact(folder / "physics_beam_contacts.npz"),
            bank=artifact(folder / "loaded_reference_bank.npz"),
            switches=s["switches"],
            legal_switch_contract=legal,
            causal_packet_audit=True,
            contact_sync_max_error_n=error,
            reference_bank_matches_prior=True,
            physics_contact_duration_s=float((norms.max(1) > 1).sum() * 0.005),
            sum_body_normal_magnitude_impulse_ns=float(norms.sum() * 0.005),
        )
        row["pairing"] = pairing(neutral, row, c["decision_time_s"])
        row["command_executed"] = any(
            s["to"] == 1 and s["time_s"] == c["decision_time_s"] for s in row["switches"]
        )
        if c["role"] == "timing" and c["decision_time_s"] == 0.2:
            oracle = next(
                r
                for r in old["rows"]
                if r["seed"] == c["runtime_seed"]
                and r["condition"] == c["condition"]
                and r["mode"] == "oracle"
            )
            q = payload(oracle)
            row["old_oracle_full_trace_matches"] = all(
                np.array_equal(p[k], q[k])
                for k in (
                    "root_pos_w",
                    "root_quat_w",
                    "dof_pos",
                    "dof_vel",
                    "applied_joint_action",
                    "action_motion_token",
                )
            )
            row["old_oracle_outcome_matches"] = row["pass"] == oracle["pass"]
        rows.append(row)
    labels = []
    for seed in (8041, 8042):
        for condition in (
            "nominal",
            "height_minus10mm",
            "height_plus10mm",
            "station_minus15cm",
            "station_plus15cm",
            "absent",
            "raised",
            "blocked",
        ):
            group = [r for r in old["rows"] if r["seed"] == seed and r["condition"] == condition]
            neutral = next((r for r in group if r["mode"] == "blind"), None) or next(
                r for r in group if r["mode"] == "reactive"
            )
            assert not neutral["switches"]
            crouch = next((r for r in group if r["mode"] == "oracle"), None) or next(
                r
                for r in rows
                if r["seed"] == seed
                and r["condition"] == condition
                and r["role"] == "missing_control"
            )
            pair = pairing(neutral, crouch, 0.2)
            executed = any(s["to"] == 1 and s["time_s"] == 0.2 for s in crouch["switches"])
            labels.append(
                {
                    "source_ancestor": 41002,
                    "seed": seed,
                    "condition": condition,
                    "decision_time_s": 0.2,
                    "neutral": neutral["cell_id"],
                    "crouch": crouch["cell_id"],
                    "pairing": pair,
                    "label": feasibility_label(
                        neutral["pass"] if pair["valid"] else None,
                        crouch["pass"] if pair["valid"] and executed else None,
                    ),
                    "training_eligible": False,
                }
            )
    controls = [r for r in rows if r["role"] == "missing_control"]
    reproduction = [r for r in rows if "old_oracle_full_trace_matches" in r]
    predictions = {
        "p1_complete_paired_labels": all(r["label"]["feasible"] is not None for r in labels),
        "p2_control_outcomes": all(r["pass"] == (r["condition"] != "blocked") for r in controls),
        "p3_explicit_command_reproduction": all(
            r["old_oracle_full_trace_matches"] and r["old_oracle_outcome_matches"]
            for r in reproduction
        ),
        "p4_legal_time_rescue_8042": any(
            r["pass"] and r["command_executed"]
            for r in rows
            if r["role"] == "timing" and r["seed"] == 8042
        ),
        "p5_measurement_contract": all(
            r["legal_switch_contract"] and r["pairing"]["valid"] for r in rows
        ),
    }
    write_new(
        out / "result.json",
        {
            "manifest": artifact(out / "manifest.json"),
            "run_record": artifact(out / "run_record.json"),
            "prior_result": m["prior_variation_result"],
            "rows": rows,
            "labels": labels,
            "predictions": predictions,
            "new_contended_gpu_hours": record["budget"]["actual_contended_gpu_hours"],
            "scope": (
                "12 development cells; exact recorded prefix audit, not hidden-state snapshots; "
                "no policy fitting or fresh-source evidence"
            ),
        },
    )
    print(json.dumps(predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "prepare":
        prepare(a.out)
        return
    m = json.loads((a.out / "manifest.json").read_text())
    for r in m["new_dependencies"]:
        checked(Path(r["path"]), r["sha256"])
    if a.command == "analyze":
        analyze(a.out)
        return
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
        record = json.loads((a.out / "run_record.json").read_text())
        if record["status"] == "completed":
            analyze(a.out)


if __name__ == "__main__":
    main()
