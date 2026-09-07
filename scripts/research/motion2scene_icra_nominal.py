#!/usr/bin/env python3
"""Acquire the nominal-contract construction arms without altering M2S-ICRA-v1.

M2S-ICRA-v1 inherited a 113-offset placement-perturbation audit and refused 47/48
analytic and 48/48 learned generated slots. The measured cause is two-sided: nominal
witnesses exist while the inherited envelope removes nearly all of them
(TRANSITION_SUPPORT_DIAGNOSTIC_V1, ENVELOPE_TRADEOFF_V1).

This is a separately registered contract, not a relaxation of the original one. The
obstacle here is authored at an exact simulator pose, so the inherited envelope models
a placement uncertainty this study does not have. Acceptance therefore uses the
nominal pose. The 17-offset and 113-offset survivals are still computed and recorded
for every proposal, as diagnostics that never gate acceptance.

Nothing in M2S-ICRA-v1 is edited, replaced or re-scored, and its six shared background
label groups are reused by reference rather than re-executed.
"""

import argparse
import copy
import json
from pathlib import Path
import time

from bundle_motion2scene_sources import closure
from motion2scene_carrier_learning import DOMAIN
from motion2scene_development_bank import DATA, RUNTIME
from motion2scene_distill_study import checkpoint, independent, query_for
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_icra_audit import execute
from motion2scene_icra_study import ARMS, physical_scene
from motion2scene_source_execution import beam_at
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_transition_construction import BANK, cases
from motion2scene_uncertainty_learning import numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_distinct import (
    distinct_indices,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_global import global_search
from gear_sonic.dataset_generation.hallucination.motion2scene_beam_visibility import beam_ray_hits
from gear_sonic.dataset_generation.hallucination.motion2scene_comparison_contract import (
    reserved_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_coverage import analytic_proposals
from gear_sonic.dataset_generation.hallucination.motion2scene_events import sample_scenes
from gear_sonic.dataset_generation.hallucination.motion2scene_no_contrast import target_only_pattern
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import station_search

PROTOCOL = ROOT / "docs/motion2scene/M2S_ICRA_NOMINAL_V1.md"
STUDY = DATA / "m2s-icra-v1"
ASSIGNED_PER_SOURCE_ARM = 3
CLEARANCE_M = 0.01


def eligibility(arm, point, preclear, critical, target_clear, duplicate, layouts):
    """Identical to M2S-ICRA-v1 except that criticality is judged at the nominal pose."""
    reasons = []
    if not np.isfinite(point).all() or np.any(point < [0.1, 1.1]) or np.any(point > [0.9, 1.45]):
        reasons.append("domain")
    if preclear < CLEARANCE_M:
        reasons.append("predecision_clearance")
    if reserved_layout(*point, layouts):
        reasons.append("reserved_layout")
    if duplicate:
        reasons.append("duplicate_within_arm")
    if arm in ("analytic", "motion2scene") and not critical:
        reasons.append("nominal_contrast_audit")
    if arm == "no_contrast" and not target_clear:
        reasons.append("target_audit")
    return reasons


def register(out):
    out.mkdir(exist_ok=False)
    parent = json.loads((STUDY / "registration.json").read_text())
    refs = [
        artifact(PROTOCOL),
        artifact(STUDY / "registration.json"),
        artifact(STUDY / "proposals.json"),
        artifact(STUDY / "prepared.json"),
        artifact(BANK / "result.json"),
        artifact(BANK / "manifest.json"),
        parent["generator"],
        parent["baseline"],
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME]))]
    for r in refs:
        checked(Path(r["path"]), r["sha256"])
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "parent_study": str(STUDY),
            "sources": parent["sources"],
            "arms": list(ARMS),
            "generator": parent["generator"],
            "baseline": parent["baseline"],
            "layouts": parent["layouts"],
            "proposal_seed": parent["proposal_seed"],
            "physics_seed": parent["physics_seed"],
            "contract": "nominal pose, 10 mm target clearance and 10 mm walk interference",
            "diagnostic_offsets": "17 search and 113 audit offsets recorded, never gating",
            "generated_slots_per_source_arm": 6,
            "assigned_per_source_arm": ASSIGNED_PER_SOURCE_ARM,
            "backgrounds": "reused by reference from M2S-ICRA-v1; no new background physics",
            "evaluation_seed": 8511,
            "refitting": False,
            "cpu_ceiling_s": 900,
            "scope": (
                "Separately registered nominal contract on the same inspected development "
                "carriers and reserved layouts; not a revision or re-score of M2S-ICRA-v1"
            ),
        },
    )


def prepare(out):
    tick = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    reg = json.loads((out / "registration.json").read_text())
    for r in reg["references"]:
        checked(Path(r["path"]), r["sha256"])
    bank = json.loads((BANK / "result.json").read_text())
    parent_manifest = json.loads((BANK / "manifest.json").read_text())
    search = json.loads((DATA / "m2s-refinement-v1/registration.json").read_text())
    saved, model = checkpoint(reg["generator"])
    _, baseline = checkpoint(reg["baseline"])
    (out / "scenes").mkdir()
    rows, costs, specs = [], [], []
    zero = np.zeros((1, 4))  # nominal pose for pre-decision clearance only
    for source in reg["sources"]:
        case, legacy, packet, _ = cases(source, bank)
        feature = normalized(raw_features({"case": case}), saved["config"])["case"]
        mask = np.asarray(case["mask"])
        for arm in ARMS:
            if time.monotonic() - tick > reg["cpu_ceiling_s"]:
                raise TimeoutError("registered preparation ceiling")
            start = time.monotonic()
            charged = []
            cloud_case = {**case, "clouds": case["clouds"][1:]} if arm == "no_contrast" else case
            base_query = query_for(cloud_case, tensor(search["search_offsets"]))

            def query(points):
                charged.append(len(points) * 17 * (1 if arm == "no_contrast" else 2))
                return base_query(points)

            rng = torch.Generator().manual_seed(reg["proposal_seed"])
            with torch.no_grad():
                if arm == "uniform":
                    output = (
                        tensor([0.1, 1.1])
                        + torch.rand((16, 2), generator=rng, dtype=torch.float64)
                        * tensor([0.8, 0.35])
                    ).numpy()
                    trace = {}
                elif arm == "analytic":
                    ranked, envelopes = analytic_proposals(
                        legacy,
                        case["route"],
                        case["yaw"].item(),
                        depth=DOMAIN.depth,
                        width=DOMAIN.width,
                        count=20,
                    )
                    nominal = np.array([[r["station"], r["height"]] for r in envelopes])
                    ranking = np.array(
                        [int(np.argmin(abs(nominal[:, 0] - s))) for s in ranked[:, 0]]
                    )
                    _, trace = global_search(nominal, ranking, lambda x: query(tensor(x)).numpy())
                    trace["selected"] = distinct_indices(trace, count=16)
                    output = trace["candidates"][trace["selected"]]
                else:
                    initial = sample_scenes(
                        (baseline if arm == "no_contrast" else model)(feature), 16, rng
                    )
                    if arm == "no_contrast":
                        output, trace = target_only_pattern(initial, lambda x: query(x)[..., 0])
                    else:
                        output, trace = station_search(initial, query, "pattern")
                    output = output.numpy()
            # Acceptance geometry: nominal pose only. Inherited envelopes are recorded
            # alongside it as diagnostics and never enter eligibility. The nominal pose
            # is taken from the audit set's own zero-offset row rather than evaluated
            # separately, so acceptance and diagnostics cannot diverge and the audit's
            # independent cross-check covers both.
            # The 17 search offsets are a subset of the 113 audit offsets and the zero
            # offset is the first row of both, so a single cross-checked evaluation
            # supplies all three views. `independent` chunks its cross-check into 23
            # parts and therefore requires at least 23 offsets; evaluating the smaller
            # sets separately would pass it empty chunks.
            audit_offsets = np.array(search["audit_offsets"])
            search_offsets = np.array(search["search_offsets"])
            assert (audit_offsets[0] == 0).all() and (search_offsets[0] == 0).all()
            subset = [
                int(np.flatnonzero((audit_offsets == row).all(1))[0]) for row in search_offsets
            ]
            audit_values, audit_error = independent(output, case, audit_offsets)
            search_values = audit_values[:, subset, :]
            search_error = audit_error
            nominal_values = audit_values[:, :1, :]
            nominal_error = audit_error
            critical = verdict(nominal_values, mask).all(1)
            robust_113 = verdict(audit_values, mask).all(1)
            robust_17 = verdict(search_values, mask).all(1)
            targets = nominal_values[:, :, 1].min(1) >= CLEARANCE_M
            pre = {
                k: {**v, "starts": v["starts"][:16], "ends": v["ends"][:16]}
                for k, v in case["states"].items()
            }
            preclear = numpy_perturbed(output, zero, pre, case["route"], case["yaw"].item())[
                :, 0, :
            ]
            seen, taken = set(), 0
            for slot, point in enumerate(output):
                reasons = eligibility(
                    arm,
                    point,
                    preclear[slot].min(),
                    critical[slot],
                    targets[slot],
                    tuple(point) in seen,
                    reg["layouts"],
                )
                seen.add(tuple(point))
                eligible = not reasons
                assign = eligible and taken < ASSIGNED_PER_SOURCE_ARM
                taken += bool(assign)
                beam = {**beam_at(case, *point), "thickness_m": 0.1}
                row = {
                    "group_id": f"nom_{source}_{arm}_{slot:02d}",
                    "source": source,
                    "arm": arm,
                    "slot": slot,
                    "assigned": bool(assign),
                    "eligible": eligible,
                    "reasons": reasons,
                    "beam": beam,
                    "nominal_critical": bool(critical[slot]),
                    "robust_17_offset": bool(robust_17[slot]),
                    "robust_113_offset": bool(robust_113[slot]),
                    "nominal_joint_margin_m": float(
                        min(
                            nominal_values[slot, :, mask].min(),
                            -nominal_values[slot, :, ~mask].max(),
                        )
                    ),
                    "target_clear_geometry": bool(targets[slot]),
                    "predecision_clearance_m": float(preclear[slot].min()),
                    "visibility_proposal_hits": beam_ray_hits(packet, beam),
                    "suite": "generated",
                }
                rows.append(row)
                if assign:
                    specs.append(row)
            np.savez_compressed(
                out / f"{source}_{arm}.npz",
                output=output,
                nominal=nominal_values,
                audit=audit_values,
                search=search_values,
                predecision_clearance=preclear,
                **{k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in trace.items()},
            )
            costs.append(
                {
                    "source": source,
                    "arm": arm,
                    "search_queries": sum(charged),
                    "audit_queries": int(nominal_values.size + audit_values.size),
                    "predecision_queries": int(preclear.size),
                    "seconds": time.monotonic() - start,
                    "trace": artifact(out / f"{source}_{arm}.npz"),
                    "independent_error_m": max(nominal_error, audit_error, search_error),
                }
            )
    batches = []
    for bi, start in enumerate(range(0, len(specs), 4)):
        folder = out.with_name(out.name + f"-labels-{bi:02d}")
        folder.mkdir()
        cells = []
        for spec in specs[start : start + 4]:
            scene = physical_scene(out / "scenes" / (spec["group_id"] + ".usda"), spec["beam"])
            for action in (0, 1):
                old = next(
                    c
                    for c in parent_manifest["cells"]
                    if c["generation_seed"] == spec["source"]
                    and c["runtime_seed"] == reg["physics_seed"]
                    and c["encounter_action"] == action
                )
                prior = next(r for r in bank["rows"] if r["cell_id"] == old["cell_id"])
                c = copy.deepcopy(old)
                ident = f"{spec['group_id']}_a{action}"
                c.update(
                    cell_id=ident,
                    group_id=spec["group_id"],
                    arm=spec["arm"],
                    suite=spec["suite"],
                    condition="present",
                    beam=spec["beam"],
                    scene=scene,
                    qualified_bank=prior["bank"],
                    role="icra_nominal_training_label",
                    output=str(folder / "rollouts" / ident),
                )
                cells.append(c)
        manifest = copy.deepcopy(parent_manifest)
        manifest.update(
            cells=cells,
            experiment=f"M2S-ICRA-nominal-v1-labels-{bi:02d}",
            registered_predictions=artifact(PROTOCOL),
            purpose="Nominal-contract four-arm paired labels",
        )
        manifest["new_dependencies"] += (
            reg["references"] + [c["scene"] for c in cells] + [c["qualified_bank"] for c in cells]
        )
        manifest["execution_policy"]["cost_ceiling"].update(
            rollouts=len(cells), gpu_hours_contended=len(cells) * 375 / 3600
        )
        write_new(folder / "manifest.json", manifest)
        batches.append(
            {
                "directory": str(folder),
                "manifest": artifact(folder / "manifest.json"),
                "assigned": len(cells),
            }
        )
    write_new(
        out / "proposals.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "costs": costs,
            "reused_backgrounds": [
                s for s in json.loads((STUDY / "proposals.json").read_text())["backgrounds"]
            ],
            "requested_generated": 72,
            "seconds": time.monotonic() - tick,
        },
    )
    write_new(
        out / "prepared.json",
        {
            "batches": batches,
            "physics_assigned": sum(b["assigned"] for b in batches),
            "proposals": artifact(out / "proposals.json"),
        },
    )
    summary = {
        "eligible_generated": {a: sum(r["eligible"] and r["arm"] == a for r in rows) for a in ARMS},
        "assigned_generated": {a: sum(r["assigned"] and r["arm"] == a for r in rows) for a in ARMS},
        "nominal_but_not_113_robust": sum(
            r["nominal_critical"] and not r["robust_113_offset"] for r in rows
        ),
        "physics_assigned": len(specs) * 2,
    }
    print(json.dumps(summary), flush=True)


def run(out):
    prepared = json.loads((out / "prepared.json").read_text())
    for b in prepared["batches"]:
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
    p.add_argument("command", choices=("register", "prepare", "run"))
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    globals()[a.command](a.out)
