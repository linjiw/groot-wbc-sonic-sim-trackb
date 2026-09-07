#!/usr/bin/env python3
"""Complete immutable proposal assignments; fit only the prespecified admitted subsets."""

import argparse
import copy
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from bundle_motion2scene_sources import closure
from motion2scene_comparative_acquisition import ARMS, DATA
from motion2scene_source_execution import scene_file
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import fit

FIRST = DATA / "m2s-comparative-acquisition-v1"
PROTOCOL = ROOT / "docs/motion2scene/COMPARATIVE_CORPUS_COMPLETION_V1.md"


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    first = json.loads((FIRST / "manifest.json").read_text())
    assignment = json.loads((FIRST / "proposals.json").read_text())
    rows = assignment["rows"]
    selected_ids = {
        arm: [r["id"] for r in rows if r["arm"] == arm and r["assigned"] and r["eligible"]][:8]
        + ["shared_absent", "shared_raised", "shared_blocked"]
        for arm in ARMS
    }
    assert all(len(ids) == 11 for ids in selected_ids.values())
    refs = [
        artifact(PROTOCOL),
        artifact(FIRST / "manifest.json"),
        artifact(FIRST / "proposals.json"),
        artifact(FIRST / "capture_audit_registration.json"),
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__)]))]
    for r in refs:
        checked(Path(r["path"]), r["sha256"])
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "fit_ids": selected_ids,
            "optimizer_seeds": list(range(8501, 8506)),
            "fit_encounters_per_arm": 11,
            "fit_updates": 1000,
            "fit_cpu_ceiling_s": 600,
            "scope": "observed-bank development fitting, no held-out evaluation",
            "physics_launch_requires": "completed first slice and all capture audit predicates pass",
        },
    )
    (out / "scenes").mkdir()
    specs = []
    for row in rows:
        if row["assigned"] and row["eligible"] and row["slot"] >= 2:
            specs.append(
                {
                    "group_id": row["id"],
                    "condition": "generated",
                    "arm": row["arm"],
                    "beam": row["beam"],
                    "scene": scene_file(out / "scenes" / f"{row['id']}.usda", row["beam"]),
                }
            )
    variation = json.loads((DATA / "m2s-overhang-variation-v1/manifest.json").read_text())
    raised = next(c for c in variation["cells"] if c["condition"] == "raised")
    specs.append(
        {
            "group_id": "shared_raised",
            "condition": "raised",
            "arm": "shared",
            "beam": raised["beam"],
            "scene": raised["scene"],
        }
    )
    cells = []
    for spec in specs:
        for action in (0, 1):
            base = next(c for c in first["cells"] if c["encounter_action"] == action)
            c = {**copy.deepcopy(base), **spec}
            c["cell_id"] = f"{spec['group_id']}_a{action}"
            c["output"] = str(out / "rollouts" / c["cell_id"])
            cells.append(c)
    assert len(cells) == 56
    m = copy.deepcopy(first)
    m.update(
        experiment="M2S-comparative-corpus-completion-v1",
        cells=cells,
        purpose="Remaining fixed proposal assignments and shared raised control",
        registered_predictions=artifact(PROTOCOL),
    )
    m["new_dependencies"] += [*refs, artifact(out / "registration.json")]
    m["execution_policy"]["cost_ceiling"].update(rollouts=56, gpu_hours_contended=56 * 375 / 3600)
    m["execution_policy"]["timing_override"] = (
        "Completion of fixed assignment; launch requires first-slice input audit "
        "and a fresh cumulative budget check."
    )
    write_new(out / "manifest.json", m)
    print(json.dumps({"cells": len(cells), "fit_ids": selected_ids}))


def launch(out, dry=False):
    reg = json.loads((out / "registration.json").read_text())
    for r in reg["references"]:
        checked(Path(r["path"]), r["sha256"])
    first = json.loads((FIRST / "result.json").read_text())
    audit = json.loads((FIRST / "capture_audit.json").read_text())
    assert len(first["pairs"]) == 10 and all(p["valid"] for p in first["pairs"])
    assert all(audit["predictions"].values())
    now = datetime.datetime.now(datetime.timezone.utc)
    entries = []
    for p in DATA.glob("m2s-*/*run_record*.json"):
        r = json.loads(p.read_text())
        cost = float(r.get("budget", {}).get("actual_contended_gpu_hours", 0))
        if not cost:
            continue
        start = datetime.datetime.fromisoformat(r["started_at"])
        entries.append(
            {"path": str(p), "hours": cost, "age_days": (now - start).total_seconds() / 86400}
        )
    # Rolling 24 h / 7 day accounting is conservative relative to calendar-day limits.
    daily = sum(r["hours"] for r in entries if r["age_days"] <= 1)
    weekly = sum(r["hours"] for r in entries if r["age_days"] <= 7)
    reserve = 56 * 375 / 3600
    assert daily + reserve <= 8 and weekly + reserve <= 24, (daily, weekly, reserve)
    if not dry:
        write_new(
            out / "launch_gate.json",
            {
                "first_result": artifact(FIRST / "result.json"),
                "first_audit": artifact(FIRST / "capture_audit.json"),
                "rolling_day_hours": daily,
                "rolling_week_hours": weekly,
                "reserved_hours": reserve,
                "entries": entries,
                "timestamp_utc": now.isoformat(),
            },
        )
    cmd = [
        sys.executable,
        str(ROOT / "scripts/research/hallucination/run_approved_manifest.py"),
        "--manifest",
        str(out / "manifest.json"),
        "--run-record",
        str(out / "run_record.json"),
    ]
    if dry:
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)


def analyze(out):
    """A clearly marked, nonexecutable union of two completed physics records."""
    source_dirs = [FIRST, out]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in source_dirs]
    records = [json.loads((p / "run_record.json").read_text()) for p in source_dirs]
    for path, record in zip(source_dirs, records):
        assert record["status"] == "completed"
        assert record["manifest_sha256"] == artifact(path / "manifest.json")["sha256"]
    union = out / "analysis_union"
    union.mkdir(exist_ok=False)
    write_new(
        union / "derivation.json",
        {
            "role": "analysis-only union; not a third physics run",
            "source_manifests": [artifact(p / "manifest.json") for p in source_dirs],
            "source_records": [artifact(p / "run_record.json") for p in source_dirs],
            "implementation": artifact(Path(__file__)),
            "protocol": artifact(PROTOCOL),
        },
    )
    manifest = copy.deepcopy(manifests[0])
    manifest["cells"] = [c for m in manifests for c in m["cells"]]
    ids = [c["cell_id"] for c in manifest["cells"]]
    assert len(ids) == len(set(ids)) == 76
    manifest["execution_policy"]["not_authorized"] = True
    manifest["analysis_only"] = True
    manifest["derivation"] = artifact(union / "derivation.json")
    write_new(union / "manifest.json", manifest)
    record = {
        "schema_version": "motion2scene_analysis_union_v1",
        "analysis_only": True,
        "status": "completed",
        "manifest_sha256": artifact(union / "manifest.json")["sha256"],
        "cells": {k: v for r in records for k, v in r["cells"].items()},
        "budget": {
            "actual_contended_gpu_hours": sum(
                r["budget"]["actual_contended_gpu_hours"] for r in records
            )
        },
        "source_records": [artifact(p / "run_record.json") for p in source_dirs],
    }
    assert set(record["cells"]) == set(ids)
    write_new(union / "run_record.json", record)
    from motion2scene_comparative_acquisition import analyze as analyze_slice

    analyze_slice(union)
    audit_driver = ROOT / "scripts/research/audit_motion2scene_comparison_capture.py"
    for command in ("register", "analyze"):
        subprocess.run(
            [sys.executable, str(audit_driver), command, "--out", str(union)], check=True
        )
    result = json.loads((union / "result.json").read_text())
    audit = json.loads((union / "capture_audit.json").read_text())
    returns = []
    for row in result["rows"]:
        if row["action"] != 1:
            continue
        sensors = json.loads(
            checked(Path(row["sensor"]["path"]), row["sensor"]["sha256"]).read_text()
        )
        exits = [s for s in sensors["switches"] if s["to"] == 0]
        returns.append(
            {
                "cell_id": row["cell_id"],
                "entry_executed": row["command_executed"],
                "return_logged": bool(exits),
                "return_times_s": [s["time_s"] for s in exits],
                "reset_count": row["reset_count"],
                "pass": row["pass"],
                "scope": "logged command return; no claim that a reset-containing capture is one uninterrupted encounter",
            }
        )
    admitted = (
        len(result["pairs"]) == 38
        and all(p["valid"] for p in result["pairs"])
        and all(audit["predictions"].values())
    )
    write_new(
        out / "admission.json",
        {
            "registration": artifact(out / "registration.json"),
            "result": artifact(union / "result.json"),
            "capture_audit": artifact(union / "capture_audit.json"),
            "all_inputs_admitted": admitted,
            "pairs": result["pairs"],
            "command_returns": returns,
            "complete_scene_pairs": len(result["pairs"]),
            "physics_predictions": {
                "all_pairs_match": all(p["valid"] for p in result["pairs"]),
                "all_capture_checks": all(audit["predictions"].values()),
            },
            "promotion": "complete matched development corpus only; explicit fitting subset in registration",
            "held_out_evaluation": False,
        },
    )
    print(
        json.dumps(
            {
                "admitted": admitted,
                "pairs": len(result["pairs"]),
                "returns_logged": sum(r["return_logged"] for r in returns),
            }
        )
    )


def fitting(out):
    reg = json.loads((out / "registration.json").read_text())
    for ref in reg["references"]:
        checked(Path(ref["path"]), ref["sha256"])
    admission = json.loads((out / "admission.json").read_text())
    assert admission["all_inputs_admitted"]
    by_id = {p["group_id"]: p for p in admission["pairs"]}
    ids = reg["fit_ids"]
    for arm in ARMS:
        assert len(ids[arm]) == 11 and all(by_id[k]["valid"] for k in ids[arm])
    directory = out / "fits"
    directory.mkdir(exist_ok=False)
    write_new(
        directory / "manifest.json",
        {
            "registration": artifact(out / "registration.json"),
            "admission": artifact(out / "admission.json"),
            "implementation": artifact(Path(__file__)),
            "fit_ids": ids,
            "seeds": reg["optimizer_seeds"],
            "interpretation": "training diagnostics only, no evaluation performance",
        },
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    results = []
    for arm in ARMS:
        pairs = [by_id[k] for k in ids[arm]]
        x = np.array([p["features"] for p in pairs], dtype=np.float32)
        y = np.array([p["outcomes"] for p in pairs], dtype=np.float32)
        mask = np.ones_like(y, dtype=bool)
        assert x.shape == (11, 214) and y.shape == (11, 2) and np.isfinite(x).all()
        for seed in reg["optimizer_seeds"]:
            if time.monotonic() - start > reg["fit_cpu_ceiling_s"]:
                raise TimeoutError("CPU fitting ceiling")
            tick = time.monotonic()
            model, mean, std, loss = fit(x, y, mask, seed, steps=1000)
            path = directory / f"{arm}_{seed}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "arm": arm,
                    "seed": seed,
                    "fit_ids": ids[arm],
                    "admission": artifact(out / "admission.json"),
                },
                path,
            )
            results.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "encounters": 11,
                    "final_training_loss": loss,
                    "seconds": time.monotonic() - tick,
                    "checkpoint": artifact(path),
                }
            )
    write_new(
        directory / "result.json",
        {
            "manifest": artifact(directory / "manifest.json"),
            "rows": results,
            "seconds": time.monotonic() - start,
            "fits": 20,
            "scope": "one data size, development training diagnostics; no held-out or policy rollout result",
        },
    )
    print(json.dumps({"fits": 20, "seconds": time.monotonic() - start}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "preflight", "run", "analyze", "fit"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.out)
    elif a.command == "analyze":
        analyze(a.out)
    elif a.command == "fit":
        fitting(a.out)
    else:
        launch(a.out, dry=a.command == "preflight")


if __name__ == "__main__":
    main()
