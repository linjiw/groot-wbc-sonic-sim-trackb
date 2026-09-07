#!/usr/bin/env python3
"""Compare reference and achieved empty-transition geometry against existing labels."""

import argparse
import json
from pathlib import Path
import time

from bundle_motion2scene_sources import closure
from motion2scene_beam_teacher import capsules
from motion2scene_comparative_acquisition import reference_pair
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance
from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)

DATA = ROOT.parent / "research-data/groot-wbc"
CORPUS = DATA / "m2s-comparative-corpus-completion-v1"
PROTOCOL = ROOT / "docs/motion2scene/TRANSITION_FORECAST_AUDIT_V1.md"


def register(out):
    result = json.loads((CORPUS / "analysis_union/result.json").read_text())
    rows = [r for r in result["rows"] if r["group_id"] == "shared_absent"]
    refs = [artifact(PROTOCOL), artifact(CORPUS / "analysis_union/result.json")]
    refs += [artifact(CORPUS / "admission.json"), result["proposals"]]
    refs += [artifact(DATA / "m2s-action-label-completion-v1/manifest.json")]
    for row in rows:
        refs += [row[k] for k in ("trajectory", "sensor", "decision")]
    refs += [artifact(p) for p in sorted(closure([Path(__file__)]))]
    # reference_pair independently checks its nested motion and conversion hashes.
    out.mkdir(exist_ok=False)
    write_new(out / "registration.json", {"references": refs, "cpu_ceiling_s": 180})


def minimum(state, beam):
    yaw = beam["yaw_rad"]
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    origin = np.array([*beam["center_xy_m"], 0])
    a, b = [(state[k] - origin) @ rot for k in ("starts", "ends")]
    half = np.array([beam["length_m"], beam["width_m"]]) / 2
    low = np.r_[-half, beam["underside_m"]]
    high = np.r_[half, beam["underside_m"] + beam["thickness_m"]]
    return float(capsule_box_clearance(a, b, state["radii"], low, high).min())


def analyze(out):
    start = time.monotonic()
    reg = json.loads((out / "registration.json").read_text())
    for ref in reg["references"]:
        checked(Path(ref["path"]), ref["sha256"])
    corpus = json.loads((CORPUS / "analysis_union/result.json").read_text())
    proposals = json.loads(Path(corpus["proposals"]["path"]).read_text())
    prior = json.loads((DATA / "m2s-action-label-completion-v1/manifest.json").read_text())
    case, _, binding = reference_pair(prior)
    empty = sorted(
        [r for r in corpus["rows"] if r["group_id"] == "shared_absent"], key=lambda r: r["action"]
    )
    assert len(empty) == 2 and all(r["reset_count"] == 0 and r["legal"] for r in empty)
    payloads = [load_reset_capture(Path(r["trajectory"]["path"])) for r in empty]
    prefix = paired_prefix(*payloads, 0.3)
    assert prefix["exact_match"]
    states = {
        "reference": [case["states"][k] for k in ("neutral", "d040")],
        "empty_transition": [capsules(p) for p in payloads],
    }
    beams = {r["id"]: r["beam"] for r in proposals["rows"]}
    rows = []
    for observed in corpus["rows"]:
        if observed["arm"] == "shared":
            continue
        if time.monotonic() - start > reg["cpu_ceiling_s"]:
            raise TimeoutError("transition forecast CPU ceiling")
        action = observed["action"]
        clear = {k: minimum(v[action], beams[observed["group_id"]]) for k, v in states.items()}
        rows.append(
            {
                "group_id": observed["group_id"],
                "arm": observed["arm"],
                "action": action,
                "clearance_m": clear,
                "screen_clear": {k: v >= 0.01 for k, v in clear.items()},
                "physical_contact_qualified": observed[
                    "maximum_beam_normal_force_n_through_passage"
                ]
                <= 1,
                "physical_task_pass": observed["pass"],
                "physics": observed["physics"],
            }
        )
    assert len(rows) == 70 and len({r["group_id"] for r in rows}) == 35
    summary = {}
    for model in states:
        summary[model] = {}
        for arm in sorted({r["arm"] for r in rows}):
            rr = [r for r in rows if r["arm"] == arm]
            summary[model][arm] = {
                "commands": len(rr),
                "false_clear_contact": sum(
                    r["screen_clear"][model] and not r["physical_contact_qualified"] for r in rr
                ),
                "missed_contact_qualified": sum(
                    not r["screen_clear"][model] and r["physical_contact_qualified"] for r in rr
                ),
                "screen_clear_task_fail": sum(
                    r["screen_clear"][model] and not r["physical_task_pass"] for r in rr
                ),
            }
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "reference_binding": binding,
            "empty_transition": empty,
            "prefix": prefix,
            "first_root_positions": [p["root_pos_w"][0].tolist() for p in payloads],
            "reference_first_xy": case["route"][0].tolist(),
            "rows": rows,
            "summary": summary,
            "cpu_seconds": time.monotonic() - start,
            "scope": "post hoc 35-scene forecast audit; no new labels or admitted placements",
        },
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args.out)
