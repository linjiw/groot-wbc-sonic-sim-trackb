#!/usr/bin/env python3
"""Physically execute the pilot's frozen policies on the development-validation sample.

Reuses the tested collection preparation and cell execution unchanged. Every
assignment is declared before any of this sample's physics runs, and all assigned
outcomes are retained, including contexts no schedule solves and unknowns.

Rows executed on each validation context:

* the nine pilot policies (three proposal arms x three corpus seeds) at M8,
* the three frozen M8 executed-contrast checkpoints, reused as a located
  comparator and never counted as independently trained pilot policies,
* the unchanged strong sensor script,
* all seven fixed schedules, which establish bank solvability per context.

Passage is the success criterion; matched passage time is secondary and is
reported only over mutually successful assignments. Passage time is crossing plus
stabilization, not whole adaptation and recovery completion.
"""

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_collect_timed_schedules as collection  # noqa: E402
from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_run_extension_tasks import execute_cell  # noqa: E402
from motion2scene_storage import deduplicate_inventories  # noqa: E402
from motion2scene_support_pool import (  # noqa: E402
    PILOT_ARMS,
)
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402, E501
    load_schedule_policy,
    load_script_parameters,
)

PHYSICS_SEED = 8732
ORDER_SEED = 202609101500
EXPECTED_STEPS = 1192
MINIMUM_FREE_GPU_MIB = 7500
MINIMUM_FREE_DISK_GIB = 30
DRIVER_FILENAME = "gear_sonic/" + "eval_agent" + "_trl.py"


def native_driver_pids():
    """Detect a live native simulator by inspecting /proc, not by pattern matching.

    The panel's own shell commands, and any grep or pgrep that merely mentions the
    driver path, must not be mistaken for an occupied simulator. Only a process
    whose executable is a Python interpreter and one of whose arguments actually
    ends with the driver path counts.
    """
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        argv = [value for value in argv if value]
        if len(argv) < 2 or "python" not in Path(argv[0]).name:
            continue
        if any(value.endswith(DRIVER_FILENAME) for value in argv[1:]):
            pids.append(int(entry.name))
    return pids


def resource_ready(panel):
    return (
        not native_driver_pids()
        and collection.free_gpu_mib() >= MINIMUM_FREE_GPU_MIB
        and shutil.disk_usage(panel).free >= MINIMUM_FREE_DISK_GIB * 1024**3
    )


def pilot_models(plan, plan_ref):
    """Locate each pilot corpus's final checkpoint from its own plan rounds."""
    models = []
    for run in plan["runs"]:
        last = run["rounds"][-1]
        if last["index"] != plan["stop_round"]:
            raise ValueError("the pilot plan does not end at its declared stop round")
        models.append(
            dict(
                policy_id=run["run_id"],
                run_id=run["run_id"],
                arm=run["arm"],
                seed=run["seed"],
                origin="pilot",
                plan=plan_ref,
                expected_training_result=str(Path(last["model_directory"]) / "result.json"),
            )
        )
    if {row["arm"] for row in models} != set(PILOT_ARMS) or len(models) != 9:
        raise ValueError("all three proposal arms at all three corpus seeds are required")
    return models


def reused_models(reference_panel):
    """Reuse the frozen M8 executed-contrast checkpoints as a located comparator."""
    study = read_checked(artifact(reference_panel / "study.json"))
    rows = [row for row in study["models"] if row["arm"] == "analytic_contrast"]
    if len(rows) != 3:
        raise ValueError("the three frozen executed-contrast M8 checkpoints are required")
    return [
        dict(
            policy_id=f"m8_reused_{row['run_id']}",
            run_id=row["run_id"],
            arm=row["arm"],
            seed=row["seed"],
            origin="reused_m8_checkpoint",
            plan=study["expanded_plan"],
            expected_training_result=row["expected_training_result"],
        )
        for row in rows
    ]


def assignments(models, contexts, option_ids):
    if len(option_ids) != 7 or len(set(option_ids)) != 7 or option_ids[0] != "neutral":
        raise ValueError("the full seven-schedule repertoire is required")
    if not contexts:
        raise ValueError("at least one validation context is required")
    policies = [
        dict(policy_id=row["policy_id"], mode="learned", run_id=row["run_id"])
        for row in sorted(models, key=lambda x: x["policy_id"])
    ]
    policies += [dict(policy_id="script", mode="scripted_multi")]
    policies += [
        dict(policy_id="fixed_" + option, mode="forced", option_id=option) for option in option_ids
    ]
    rows = [
        dict(
            policy,
            scene_id=context["scene_id"],
            base_layout_id=context["base_layout_id"],
            stratum=context["stratum"],
            underside_band=context["underside_band"],
            scene_definition=context["scene_definition"],
            physics_seed=PHYSICS_SEED,
        )
        for policy in policies
        for context in sorted(contexts, key=lambda x: x["scene_id"])
    ]
    random.Random(ORDER_SEED).shuffle(rows)
    return [dict(row, assignment_id=f"episode_{i:03d}") for i, row in enumerate(rows)]


def declare(args):
    plan_ref = artifact(args.plan)
    plan = read_checked(plan_ref)
    if plan["schema"] != "motion2scene_support_pilot_plan_v1":
        raise ValueError("the support-preserving pilot plan is required")
    sample_ref = artifact(args.sample / "registration.json")
    sample = read_checked(sample_ref)
    if sample["schema"] != "motion2scene_support_validation_sample_v1":
        raise ValueError("the declared development-validation sample is required")
    if sample["exact_identity_collisions"] or sample["clearance_queries"]:
        raise ValueError("the validation sample must be collision-free and predicate-free")
    reference = read_checked(artifact(args.reference_panel / "study.json"))
    bank_ref = plan["registry"]
    if reference["registry"] != bank_ref:
        raise ValueError("the pilot and the reused M8 checkpoints must share one bank")
    bank = load_verified_registry(bank_ref["path"], bank_ref["sha256"])
    script_ref = reference["script"]
    load_script_parameters(script_ref["path"], script_ref["sha256"], bank)
    contexts = sample["contexts"]
    models = pilot_models(plan, plan_ref) + reused_models(args.reference_panel)
    rows = assignments(models, contexts, bank.option_ids)
    args.panel.mkdir(parents=True, exist_ok=False)
    return write_new(
        args.panel / "study.json",
        dict(
            schema="motion2scene_support_validation_panel_v1",
            pilot_plan=plan_ref,
            validation_sample=sample_ref,
            reference_panel=artifact(args.reference_panel / "study.json"),
            registry=bank_ref,
            request=reference["request"],
            template=reference["template"],
            script=script_ref,
            models=models,
            assignments=rows,
            contexts=contexts,
            ancestry_groups=sample["ancestry_groups"],
            checkpoint=plan["stop_round"],
            assigned_episodes=len(rows),
            maximum_physics_steps=EXPECTED_STEPS * len(rows),
            physics_seed=PHYSICS_SEED,
            implementation=artifact(Path(__file__)),
            scope=(
                "development-validation comparison of three proposal-support controls; no "
                "reserved layouts, no reserved outcomes, no held-out claim"
            ),
            comparison=(
                "nine pilot policies, three reused frozen M8 executed-contrast checkpoints, "
                "the unchanged strong script and all seven fixed schedules"
            ),
            matching=(
                "same validation geometries, same physics seed, same bank, same ideal "
                "observations and the same physical scorer for every row"
            ),
            causal_question=(
                "C versus A tests whether preserving exploratory proposal support improves the "
                "strict-gate method. C versus B tests whether the geometry-guided portion adds "
                "value beyond broad coverage. Equal improvement by B and C supports broadening "
                "support, not the mixture's superiority."
            ),
            primary_metric="actual passage over all assigned contexts, reported per corpus",
            secondary_metrics=[
                "matched passage-time differences over mutually successful assignments only",
                "bank solvability per context",
                "passing-set patterns",
                "added response coverage",
                "presence of usable continuation targets",
                "physical outcomes of off-gate proposals",
            ],
            retention=(
                "every assigned context is retained with its measured outcome, including "
                "bank-unsolved and unknown cases; the sample is never redrawn"
            ),
            selection="all assigned models; no checkpoint or context selection using this panel",
        ),
    )


def bind_models(study):
    bank = load_verified_registry(study["registry"]["path"], study["registry"]["sha256"])
    bindings = {}
    for model in study["models"]:
        ref = artifact(Path(model["expected_training_result"]))
        result = read_checked(ref)
        if result["status"] != "complete" or result["audit_errors"]:
            raise ValueError("complete acquired checkpoint required before physical evaluation")
        if (
            Path(result["policy"]["path"]).resolve()
            != Path(ref["path"]).with_name("policy.npz").resolve()
        ):
            raise ValueError("checkpoint's policy path differs from its acquisition assignment")
        teachers = read_checked(result["teachers"])
        registration = read_checked(result["registration"])
        if registration["collections"] != [group["collection"] for group in teachers]:
            raise ValueError("checkpoint does not bind the exact teacher prefix it was fitted on")
        if registration["registry"] != study["registry"]:
            raise ValueError("checkpoint was fitted under a different bank")
        load_schedule_policy(result["policy"]["path"], result["policy"]["sha256"], bank)
        bindings[model["policy_id"]] = dict(
            training_result=ref,
            policy=result["policy"],
            origin=model["origin"],
            encounters=len(teachers),
        )
    return bindings


def prepare(panel):
    study_ref = artifact(panel / "study.json")
    study = read_checked(study_ref)
    checked(Path(study["implementation"]["path"]), study["implementation"]["sha256"])
    models = bind_models(study)
    by_policy = {row["policy_id"]: row for row in study["models"]}
    prepared = []
    for row in study["assignments"]:
        target = panel / "episodes" / row["assignment_id"]
        policy = models[row["policy_id"]]["policy"] if row["mode"] == "learned" else None
        args = SimpleNamespace(
            registry=Path(study["registry"]["path"]),
            request=Path(study["request"]["path"]),
            template=Path(study["template"]["path"]),
            cell="neutral",
            scene_definition=Path(row["scene_definition"]["path"]),
            policy_mode=row["mode"],
            policy=None if policy is None else Path(policy["path"]),
            script_parameters=(
                Path(study["script"]["path"]) if row["mode"] == "scripted_multi" else None
            ),
            forced_option_ids=[row["option_id"]] if row["mode"] == "forced" else None,
            preferred_option_id="neutral",
            preferred_reference_id="sustained",
            seed=study["physics_seed"],
            out=target,
        )
        if target.exists():
            raise ValueError("partial preparation retained; inspect before continuing")
        collection.prepare(args)
        prepared.append(
            dict(
                **row,
                origin=by_policy[row["policy_id"]]["origin"] if row["mode"] == "learned" else None,
                collection=artifact(target / "manifest.json"),
            )
        )
    return write_new(
        panel / "prepared.json", dict(study=study_ref, models=models, assignments=prepared)
    )


def check_assignment(row, common, study, models):
    if (
        common["split"] != "development"
        or common["registry"] != study["registry"]
        or common["scene_definition"] != row["scene_definition"]
        or len(common["cells"]) != 1
        or common["expected_physics_steps"] != EXPECTED_STEPS
    ):
        raise ValueError("physical evaluation assignment changed")
    cell = common["cells"][0]
    if cell["runtime_seed"] != study["physics_seed"] or cell["timed_schedule_mode"] != row["mode"]:
        raise ValueError("policy mode or paired execution seed differs")
    expected = models[row["policy_id"]]["policy"] if row["mode"] == "learned" else None
    if common["policy"] != expected:
        raise ValueError("frozen policy differs from the bound checkpoint")
    script = study["script"] if row["mode"] == "scripted_multi" else None
    if common["script_parameters"] != script:
        raise ValueError("strong script parameters changed")
    if row["mode"] == "forced" and cell["forced_option_id"] != row["option_id"]:
        raise ValueError("fixed reference schedule changed")
    return cell


def run(panel):
    prepared = read_checked(artifact(panel / "prepared.json"))
    study = read_checked(prepared["study"])
    checked(Path(study["implementation"]["path"]), study["implementation"]["sha256"])
    if prepared["models"] != bind_models(study):
        raise ValueError("prepared policies differ from the bound checkpoints")
    stripped = [
        {k: v for k, v in row.items() if k not in ("collection", "origin")}
        for row in prepared["assignments"]
    ]
    if stripped != study["assignments"]:
        raise ValueError("prepared assignments differ from the declared panel")
    results, unknowns = [], []
    for row in prepared["assignments"]:
        out = Path(row["collection"]["path"]).parent
        check_assignment(row, read_checked(row["collection"]), study, prepared["models"])
        if not (out / "result.json").exists():
            while not resource_ready(panel):
                print(
                    json.dumps(
                        dict(
                            status="waiting_for_native_capacity",
                            assignment=row["assignment_id"],
                            free_gpu_mib=collection.free_gpu_mib(),
                            native_driver_pids=native_driver_pids(),
                        )
                    ),
                    flush=True,
                )
                time.sleep(45)
            common, bank, scene = collection.verify_manifest(out, execution=True)
            cell = check_assignment(row, common, study, prepared["models"])
            execute_cell(cell, common, bank, scene, out / "launch.json")
            collection.analyze(out)
            write_new(out / "storage.json", deduplicate_inventories([out]))
        ref = artifact(out / "result.json")
        result = read_checked(ref)
        if result["manifest"] != row["collection"] or len(result["rows"]) != 1:
            raise ValueError("physical result belongs to another assignment")
        outcome = result["rows"][0]["outcome"]["task_outcome"]
        results.append(dict(assignment_id=row["assignment_id"], outcome=outcome, result=ref))
        print(
            json.dumps(
                dict(
                    status="validation_episode_complete",
                    assignment=row["assignment_id"],
                    policy_id=row["policy_id"],
                    scene_id=row["scene_id"],
                    outcome=outcome,
                    completed=len(results),
                    assigned=len(prepared["assignments"]),
                )
            ),
            flush=True,
        )
        if outcome == "unknown":
            # Retained, never retried and never replaced by another context.
            unknowns.append(row["assignment_id"])
            raise RuntimeError(f"unknown assigned outcome retained: {row['assignment_id']}")
    return write_new(
        panel / "result.json",
        dict(
            prepared=artifact(panel / "prepared.json"),
            assignments=results,
            retained_unknowns=unknowns,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "prepare", "run"))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--reference-panel", type=Path)
    args = parser.parse_args()
    if args.action == "declare":
        result = declare(args)
    else:
        result = {"prepare": prepare, "run": run}[args.action](args.panel)
    print(json.dumps(result))
