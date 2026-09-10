#!/usr/bin/env python3
"""Plan the support-preserving pilot: one shared M4 prefix, three proposal arms.

All three arms import the *identical* four-encounter executed-contrast prefix and
its M4 checkpoint, then acquire four new encounters over a shared stratum
schedule. The only thing that differs across arms is which channel proposes each
new slot, so the comparison varies one thing.

The declared M16/M32 trajectory and every completed corpus are untouched: this
plan writes a new execution root and never names an existing run directory.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_prepare_acquisition_pools import read_bound  # noqa: E402
from motion2scene_support_pool import (  # noqa: E402
    MIXTURE_CHANNELS,
    MIXTURE_STRICT_FRACTION,
    PILOT_ARMS,
    select_pilot_encounters,
    stratum_schedule,
)
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_pool import (  # noqa: E402
    SEEDS,
)

STRATA = ("short", "sustained", "early_constraint", "short_then_short", "short_then_sustained")
SOURCE_ARM = "analytic_contrast"
PREFIX_ROUNDS = 4
ADDITIONS = 4
EPISODES_PER_ROUND = 8
STEPS_PER_EPISODE = 1192


def build(expanded_plan, support_pools, execution_root, out):
    old_ref = artifact(expanded_plan)
    old = read_bound(old_ref)
    if old.get("schema") != "motion2scene_expanded_acquisition_v1":
        raise ValueError("the expanded acquisition plan is required as the prefix source")
    pool_ref = artifact(support_pools / "result.json")
    pools = read_bound(pool_ref)
    if pools.get("schema") != "motion2scene_support_preserving_pool_result_v1":
        raise ValueError("the support-preserving candidate pool result is required")
    if execution_root.exists():
        raise ValueError("a new pilot execution root is required")
    pool_by_seed = {row["seed"]: row for row in pools["pools"]}
    runs, selections = [], []
    for seed in SEEDS:
        source = next(r for r in old["runs"] if r["run_id"] == f"seed{seed}_{SOURCE_ARM}")
        prefix = source["rounds"][: PREFIX_ROUNDS + 1]
        if any(row["inherited_model"] is None for row in prefix):
            raise ValueError("the shared prefix must be fully inherited, never re-acquired")
        composition = [row["stratum"] for row in prefix[1:]]
        schedule = stratum_schedule(composition, ADDITIONS, STRATA)
        pool = pool_by_seed[seed]
        queues = {
            arm: read_bound(ref)["rows"]
            for arm, ref in pool["arms"].items()
            if arm in ("support_strict", "support_broad")
        }
        chosen = select_pilot_encounters(queues, schedule, mixture_channels=MIXTURE_CHANNELS)
        selections.append(
            dict(
                seed=seed,
                shared_prefix_strata=composition,
                shared_prefix_candidates=[row["candidate_id"] for row in prefix[1:]],
                stratum_schedule=schedule,
                mixture_channels=chosen["mixture_channels"],
                shortfalls=chosen["shortfalls"],
                realised_mixture_strict_fraction=chosen["realised_mixture_strict_fraction"],
                physical_outcomes_consulted=False,
                selected={
                    arm: [
                        dict(
                            slot=row["slot"],
                            candidate_id=row["candidate_id"],
                            stratum=row["stratum"],
                            proposal_channel=row["proposal_channel"],
                            original_queue_rank=row["original_queue_rank"],
                            inside_positive_screen=row.get("inside_positive_screen"),
                            inside_negative_screen=row.get("inside_negative_screen"),
                            geometric_screen_passed=row["geometric_screen_passed"],
                        )
                        for row in rows
                    ]
                    for arm, rows in chosen["selected"].items()
                },
            )
        )
        if chosen["shortfalls"]:
            # Retained, never refilled from the other channel and no margin moved.
            print(
                json.dumps(
                    dict(status="acquisition_shortfall", seed=seed, shortfalls=chosen["shortfalls"])
                ),
                flush=True,
            )
        for arm in PILOT_ARMS:
            run_id = f"seed{seed}_{arm}"
            rounds = []
            for row in prefix:
                rounds.append(dict(row, shared_prefix=True, proposal_channel=None))
            for offset, row in enumerate(chosen["selected"][arm], start=PREFIX_ROUNDS + 1):
                location = execution_root / run_id / f"round_{offset:03d}"
                rounds.append(
                    dict(
                        index=offset,
                        candidate_id=row["candidate_id"],
                        stratum=row["stratum"],
                        scene=row["definition"],
                        branch_order=row["future_branch_order"],
                        inherited_teacher=None,
                        inherited_student=None,
                        inherited_model=None,
                        teacher_directory=str(location / "teachers"),
                        student_directory=str(location / "student"),
                        model_directory=str(
                            execution_root / run_id / "models" / f"model_{offset:03d}"
                        ),
                        shared_prefix=False,
                        proposal_channel=row["proposal_channel"],
                        slot=row["slot"],
                        inside_positive_screen=row.get("inside_positive_screen"),
                        inside_negative_screen=row.get("inside_negative_screen"),
                    )
                )
            runs.append(
                dict(
                    run_id=run_id,
                    seed=seed,
                    arm=arm,
                    prefix_source_run_id=source["run_id"],
                    rounds=rounds,
                    new_rounds=len(rounds) - PREFIX_ROUNDS - 1,
                )
            )
    new_rounds = sum(row["new_rounds"] for row in runs)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "selection.json", dict(rows=selections))
    return write_new(
        out / "plan.json",
        dict(
            schema="motion2scene_support_pilot_plan_v1",
            question=(
                "Does preserving exploratory proposal support outside the executed-envelope "
                "acceptance gate produce better teaching data and better physically executed "
                "traversal decisions at matched measured acquisition cost?"
            ),
            predecessor=old["predecessor"],
            prefix_plan=old_ref,
            candidate_pools=pool_ref,
            registry=old["registry"],
            execution_root=str(execution_root.resolve()),
            runs=runs,
            arms=list(PILOT_ARMS),
            acquisition_seeds=list(SEEDS),
            shared_prefix_rounds=PREFIX_ROUNDS,
            shared_prefix_source_arm=SOURCE_ARM,
            additions_per_corpus=ADDITIONS,
            stop_round=PREFIX_ROUNDS + ADDITIONS,
            mixture_channels=list(MIXTURE_CHANNELS),
            mixture_strict_fraction=MIXTURE_STRICT_FRACTION,
            selection=artifact(out / "selection.json"),
            new_assigned_episodes_per_corpus=ADDITIONS * EPISODES_PER_ROUND,
            new_assigned_steps_per_corpus=ADDITIONS * EPISODES_PER_ROUND * STEPS_PER_EPISODE,
            new_assigned_episodes=new_rounds * EPISODES_PER_ROUND,
            new_assigned_steps=new_rounds * EPISODES_PER_ROUND * STEPS_PER_EPISODE,
            fixed_common=(
                "frozen tracker, seven qualified schedules, causal 114D student input, "
                "complete-continuation teacher, phase ridge learner at l2=10, uniform phase "
                "weighting, three decision phases, one adaptation and recovery"
            ),
            varies=(
                "only the proposal support of the four new encounters per corpus; the shared "
                "prefix, its M4 checkpoint, the physics seed and the branch structure are "
                "identical across arms"
            ),
            not_varied=(
                "no information-consistent teacher, no feasibility classifier, no new sensor, "
                "no new skill, no replay rule and no change to the success rule"
            ),
            physical_outcomes_used_for_new_selection=False,
            reserved_evaluation_started=False,
            budget_rule=(
                "Bounded addition committed before any new outcome. Count every attempt and "
                "all measured steps; preserve failures and unknowns; never replace a scene "
                "after an outcome and never expand the budget to reach a desired result."
            ),
            preserved=(
                "The declared M16/M32 trajectory, the five-arm M8 corpora and every completed "
                "primary result stay unchanged; this plan writes a separate execution root."
            ),
            implementation=[
                artifact(ROOT / "scripts/research" / name)
                for name in (
                    "motion2scene_support_plan.py",
                    "motion2scene_support_acquisition.py",
                    "motion2scene_prepare_support_pools.py",
                )
            ],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expanded-plan", "support-pools", "execution-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.expanded_plan, args.support_pools, args.execution_root, args.out)))
