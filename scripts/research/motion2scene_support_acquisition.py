#!/usr/bin/env python3
"""Execute the support-preserving pilot's three nested acquisition controls.

Reuses the tested attempt execution, complete-continuation teacher structure and
phase ridge fit without modifying them: the controller, the collection backend
and the fit are the frozen ones. Only the proposal support of the four new
encounters per corpus differs across arms.

Every arm imports the identical executed-contrast M4 prefix and its checkpoint,
so the inherited measured cost is shared and reported once per arm. New
acquisition cost is reported separately. Failures and unknowns are retained; an
unknown outcome pauses the run rather than being retried.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

import motion2scene_build_timed_replay as replay  # noqa: E402
from motion2scene_expanded_acquisition import (  # noqa: E402
    ExpandedController,
    validate_training_prefix,
)
from motion2scene_run_primary_acquisition import (  # noqa: E402
    Controller,
    Paused,
    acquisition_lock,
    commit,
    load_context,
)
from motion2scene_storage import deduplicate_inventories  # noqa: E402
from motion2scene_support_pool import (  # noqa: E402
    PILOT_ARMS,
)
from motion2scene_timing_diagnostic import artifact, checked  # noqa: E402
from motion2scene_train_timed_schedules import audit_collection  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)

EPISODES_PER_ROUND = 8
STEPS_PER_EPISODE = 1192


def bound(ref):
    return json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())


class SupportController(ExpandedController):
    """Correct the assigned-slot arithmetic for this pilot's bounded addition."""

    def __init__(self, context, run, inherited_ledger, new_slots):
        super().__init__(context, run, inherited_ledger)
        self.new_slots = new_slots

    def accounting(self):
        report = super().accounting()
        launched = len(report["attempts"])
        report["run_id"] = self.expanded_run["run_id"]
        report["prefix_source_run_id"] = self.expanded_run["prefix_source_run_id"]
        report["new_assigned_episode_slots"] = self.new_slots
        report["unlaunched_assigned_episode_slots"] = self.new_slots - launched
        report["reserved_future_assigned_steps"] = (
            report["unlaunched_assigned_episode_slots"] * STEPS_PER_EPISODE
        )
        report["inherited_assigned_episode_slots"] = None
        report["cost_interpretation"] = (
            "inherited totals are the shared executed-contrast M4 prefix, identical for all "
            "three arms; new_recorded_physics_steps is this arm's own acquisition cost"
        )
        return report

    def acquire_expanded(self, stop):
        """The frozen walk, with the inherited-slot path repaired.

        ``ExpandedController.acquire_expanded`` calls ``artifact()`` on
        ``slot["inherited_model"]``, which arrives from JSON as ``str`` while
        ``artifact()`` requires a ``Path``. Both files match the hashes recorded
        in the M8 plan and the adopted runtime, so the defect is latent rather
        than introduced here, and it is repaired in this subclass instead of in
        the frozen file, whose sha256 the M8 plan and every resume depend on.
        Nothing else about the walk changes.
        """
        groups, students, previous, previous_result = [], [], None, None
        bank_ref = self.context["plan"]["registry"]
        bank = load_verified_registry(bank_ref["path"], bank_ref["sha256"])
        for slot in self.expanded_run["rounds"][: stop + 1]:
            index = slot["index"]
            if slot["inherited_model"]:
                previous_result = artifact(Path(slot["inherited_model"]))
                previous = bound(previous_result)
                registered = bound(previous["registration"])
                imported_groups = bound(previous["teachers"])
                if imported_groups[:-1] != groups:
                    raise ValueError("inherited models do not form one teaching prefix")
                if registered["collections"] != [g["collection"] for g in imported_groups]:
                    raise ValueError("inherited teaching order differs")
                groups = imported_groups
                if index:
                    actual, _, unknown = replay.audit_students(
                        Path(slot["inherited_student"]), bank, bank_ref
                    )
                    if unknown or len(actual) != 1:
                        raise Paused("inherited student measurement unavailable")
                    students.append(actual[0])
                else:
                    students.append(None)
                continue
            row = dict(
                round_index=index,
                scene_definition=slot["scene"],
                physics_seed=self.expanded_run["seed"],
                teacher_branch_order=slot["branch_order"],
                teacher_directory=slot["teacher_directory"],
                student_directory=slot["student_directory"],
            )
            current = Path(slot["teacher_directory"]).parent
            student = None
            if index:
                binding_path = current / "preupdate.json"
                expected = dict(
                    model=previous["policy"],
                    training_result=previous_result,
                    earlier_collections=[g["collection"] for g in groups],
                    round_index=index,
                    scene=slot["scene"],
                    expanded_plan=self.context["expanded_plan_ref"],
                )
                if not binding_path.exists() and (
                    Path(slot["teacher_directory"]).exists()
                    or Path(slot["student_directory"]).exists()
                ):
                    raise Paused("current collection exists without pre-update model commitment")
                validate_training_prefix(
                    bound(previous["registration"]), expected["earlier_collections"], bank_ref
                )
                commit(binding_path, expected)
                student_ref = self.collect(row, "student", previous["policy"])
                release_path = current / "student_complete.json"
                if not release_path.exists() and Path(slot["teacher_directory"]).exists():
                    raise Paused("teachers exist without a completed-student release")
                commit(release_path, dict(preupdate=artifact(binding_path), student=student_ref))
                actual, _, unknown = replay.audit_students(
                    Path(student_ref["path"]), bank, bank_ref
                )
                if (
                    unknown
                    or len(actual) != 1
                    or actual[0]["manifest"]["policy"] != previous["policy"]
                ):
                    raise Paused("actual student does not bind the pre-update policy")
                student = actual[0]
            teacher_ref = self.collect(row, "teacher")
            group = audit_collection(Path(teacher_ref["path"]), bank, bank_ref)
            if (
                group["scene_id"] != bound(slot["scene"])["scene_id"]
                or group["physics_seed"] != self.expanded_run["seed"]
            ):
                raise ValueError("measured encounter differs from the assigned scene and seed")
            groups.append(group)
            students.append(student)
            storage_path = current / "storage.json"
            if not storage_path.exists():
                collections = [Path(slot["teacher_directory"])]
                if index:
                    collections.append(Path(slot["student_directory"]))
                write_new(storage_path, deduplicate_inventories(collections))
            previous = self.fit_expanded(groups, students, Path(slot["model_directory"]))
            previous_result = artifact(Path(slot["model_directory"]) / "result.json")
        output = dict(
            expanded_plan=self.context["expanded_plan_ref"],
            run_id=self.expanded_run["run_id"],
            completed_through_round=stop,
            model=previous["policy"],
            training_result=previous_result,
            accounting=self.accounting(),
            reserved_evaluation_started=False,
        )
        return commit(self.folder / f"complete_{stop:03d}.json", output)


def run(plan_path, adoption_path, stop):
    plan_ref = artifact(plan_path)
    plan = bound(plan_ref)
    if plan.get("schema") != "motion2scene_support_pilot_plan_v1":
        raise ValueError("the support-preserving pilot plan is required")
    if stop != plan["stop_round"]:
        raise ValueError("the declared bounded addition budget is the only permitted stop round")
    for source in plan["implementation"]:
        checked(Path(source["path"]), source["sha256"])
    prefix_plan = bound(plan["prefix_plan"])
    if prefix_plan["predecessor"] != plan["predecessor"]:
        raise ValueError("the prefix plan and this pilot must share one predecessor")
    for boundary in range(stop + 1):
        for assignment in plan["runs"]:
            folder = Path(plan["execution_root"]) / assignment["run_id"]
            done = folder / "controller" / f"complete_{boundary:03d}.json"
            if done.exists():
                completed = bound(artifact(done))
                if completed["expanded_plan"] != plan_ref:
                    raise ValueError("completed boundary belongs to a different pilot plan")
                bound(completed["training_result"])
                continue
            if boundary < plan["shared_prefix_rounds"]:
                # Rounds 0-4 are imported, never re-acquired; nothing to execute.
                continue
            source_id = assignment["prefix_source_run_id"]
            context = load_context(Path(plan["predecessor"]["path"]), adoption_path, source_id)
            inherited_ledger = Controller(context).accounting()
            source_root = context["execution_root"]
            new_slots = assignment["new_rounds"] * EPISODES_PER_ROUND
            budget = inherited_ledger["conservative_charged_steps"] + new_slots * STEPS_PER_EPISODE
            context = dict(
                context,
                plan={**context["plan"], "budget": {"budget_physics_steps": budget}},
                run={**context["run"], "run_id": assignment["run_id"]},
                expanded_plan_ref=plan_ref,
                run_root=folder,
                execution_root=Path(plan["execution_root"]),
                reused_bootstrap=None,
            )
            controller = SupportController(context, assignment, inherited_ledger, new_slots)
            try:
                with acquisition_lock(source_root / ".expanded_acquisition.lock"):
                    controller.acquire_expanded(boundary)
                # commit() returns an artifact ref on an idempotent re-commit and
                # nothing on a first write, so read the receipt itself.
                receipt = json.loads(done.read_text())
                print(
                    json.dumps(
                        dict(
                            status="completed_boundary",
                            run_id=assignment["run_id"],
                            arm=assignment["arm"],
                            boundary=boundary,
                            new_recorded_physics_steps=receipt["accounting"][
                                "new_recorded_physics_steps"
                            ],
                            actual_recorded_physics_steps=receipt["accounting"][
                                "actual_recorded_physics_steps"
                            ],
                        )
                    ),
                    flush=True,
                )
            except Paused as error:
                return dict(
                    status="paused",
                    reason=str(error),
                    run_id=assignment["run_id"],
                    boundary=boundary,
                )
    receipt = dict(
        status="complete",
        plan=plan_ref,
        encounters_per_corpus=stop,
        corpora=len(plan["runs"]),
        arms=list(PILOT_ARMS),
    )
    commit(Path(plan["execution_root"]) / "pilot_complete.json", receipt)
    return receipt


def watch(plan_path, adoption_path, stop):
    """Wait only for free capacity; never retry an unknown physical outcome."""
    previous = None
    while True:
        result = run(plan_path, adoption_path, stop)
        if result != previous:
            print(json.dumps(result), flush=True)
            previous = result
        waitable = result["status"] == "paused" and result["reason"] in (
            "resource preflight paused before intent; assigned cell remains unlaunched",
            "fewer than 30 GiB free; remaining physics stays unlaunched",
            "another acquisition controller holds the common lock",
        )
        if not waitable:
            return result
        time.sleep(45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--until-budget", type=int, default=8)
    parser.add_argument("--watch", action="store_true", help="wait for free capacity only")
    args = parser.parse_args()
    result = (watch if args.watch else run)(args.plan, args.adoption, args.until_budget)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "complete" else 75)
