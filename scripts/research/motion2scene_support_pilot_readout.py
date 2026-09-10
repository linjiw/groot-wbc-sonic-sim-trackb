#!/usr/bin/env python3
"""Read out the support-preserving pilot: passage first, mechanism second.

Primary comparison is actual physical passage on the development-validation
sample, reported per corpus and never as a pooled mean alone. Matched passage
time is reported only over mutually successful assignments and is kept distinct
from whole-episode adaptation and recovery completion.

Secondary mechanism readouts explain a result; they are never the criterion:
bank solvability, passing-set patterns, added response coverage, whether an
encounter offered a usable continuation target, and the physical outcomes of
proposals that fell outside the original geometric screens.

Every assigned context is retained, including bank-unsolved and unknown cases.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from motion2scene_decision_study import read_checked  # noqa: E402
from motion2scene_support_pool import (  # noqa: E402
    PILOT_ARMS,
)
from motion2scene_timing_diagnostic import artifact  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_plan import (  # noqa: E402
    write_new,
)


def episode_rows(panel):
    """Collect every assigned cell that has a physical result, and say which do not."""
    prepared = read_checked(artifact(panel / "prepared.json"))
    study = read_checked(prepared["study"])
    rows, missing = [], []
    for assignment in prepared["assignments"]:
        out = Path(assignment["collection"]["path"]).parent
        if not (out / "result.json").exists():
            missing.append(assignment["assignment_id"])
            continue
        result = read_checked(artifact(out / "result.json"))
        row = result["rows"][0]
        rows.append(
            dict(
                assignment_id=assignment["assignment_id"],
                policy_id=assignment["policy_id"],
                mode=assignment["mode"],
                origin=assignment.get("origin"),
                option_id=assignment.get("option_id"),
                scene_id=assignment["scene_id"],
                base_layout_id=assignment["base_layout_id"],
                stratum=assignment["stratum"],
                underside_band=assignment["underside_band"],
                outcome=row["outcome"]["task_outcome"],
                passage_time_s=row["costs"]["passage_time_s"],
                whole_episode_time_s=row["costs"]["whole_episode_time_s"],
                chosen_option_id=(row.get("schedule_audit") or {}).get("chosen_option_id"),
                physics_steps=row["physics_steps"],
                measurement_admitted=row["measurement_admitted"],
            )
        )
    return study, prepared, rows, missing


def bank_capability(rows, contexts):
    """Which fixed schedules physically pass each context, measured not predicted."""
    capability = {}
    for scene in contexts:
        forced = [r for r in rows if r["mode"] == "forced" and r["scene_id"] == scene]
        passing = sorted(r["option_id"] for r in forced if r["outcome"] == "pass")
        best = [r["passage_time_s"] for r in forced if r["outcome"] == "pass"]
        capability[scene] = dict(
            measured_branches=len(forced),
            passing_schedules=passing,
            bank_solvable=bool(passing),
            best_bank_passage_time_s=min(best) if best else None,
            unknown_branches=sum(r["outcome"] == "unknown" for r in forced),
        )
    return capability


def policy_summary(rows, contexts, capability):
    summaries = {}
    for policy_id in sorted({r["policy_id"] for r in rows}):
        own = {r["scene_id"]: r for r in rows if r["policy_id"] == policy_id}
        assigned = [own.get(scene) for scene in contexts]
        measured = [r for r in assigned if r is not None]
        passes = [r for r in measured if r["outcome"] == "pass"]
        selection_failures = [
            r["scene_id"]
            for r in measured
            if r["outcome"] == "failure" and capability[r["scene_id"]]["bank_solvable"]
        ]
        chosen = [r["chosen_option_id"] for r in measured if r["chosen_option_id"]]
        summaries[policy_id] = dict(
            assigned=len(contexts),
            measured=len(measured),
            unmeasured=len(contexts) - len(measured),
            passages=len(passes),
            failures=sum(r["outcome"] == "failure" for r in measured),
            unknowns=sum(r["outcome"] == "unknown" for r in measured),
            selection_failures=selection_failures,
            distinct_schedules_used=len(set(chosen)),
            schedules_used=sorted(set(chosen)),
            per_context={
                r["scene_id"]: dict(
                    outcome=r["outcome"],
                    passage_time_s=r["passage_time_s"],
                    chosen_option_id=r["chosen_option_id"],
                    stratum=r["stratum"],
                    underside_band=r["underside_band"],
                )
                for r in measured
            },
        )
    return summaries


def matched_difference(first, second, summaries):
    """Passage difference plus paired time over mutually successful contexts only."""
    left, right = summaries.get(first), summaries.get(second)
    if left is None or right is None:
        return None
    mutual = [
        scene
        for scene, row in left["per_context"].items()
        if row["outcome"] == "pass"
        and right["per_context"].get(scene, {}).get("outcome") == "pass"
        and row["passage_time_s"] is not None
        and right["per_context"][scene]["passage_time_s"] is not None
    ]
    paired = [
        left["per_context"][scene]["passage_time_s"] - right["per_context"][scene]["passage_time_s"]
        for scene in sorted(mutual)
    ]
    return dict(
        first=first,
        second=second,
        passage_difference=left["passages"] - right["passages"],
        mutually_successful=len(mutual),
        mean_paired_passage_time_difference_s=(
            None if not paired else round(sum(paired) / len(paired), 4)
        ),
        interpretation="negative paired time favours the first named policy",
    )


def encounter_mechanism(plan):
    """What each new proposal physically taught, including the off-gate ones."""
    records = []
    for run in plan["runs"]:
        for slot in run["rounds"]:
            if slot.get("shared_prefix", False):
                continue
            teacher = Path(slot["teacher_directory"]) / "result.json"
            record = dict(
                run_id=run["run_id"],
                arm=run["arm"],
                seed=run["seed"],
                round_index=slot["index"],
                candidate_id=slot["candidate_id"],
                stratum=slot["stratum"],
                proposal_channel=slot["proposal_channel"],
                inside_positive_screen=slot.get("inside_positive_screen"),
                inside_negative_screen=slot.get("inside_negative_screen"),
                off_gate=not (
                    slot.get("inside_positive_screen") and slot.get("inside_negative_screen")
                ),
                executed=teacher.exists(),
            )
            if teacher.exists():
                result = read_checked(artifact(teacher))
                branches = {
                    row["forced_option_id"]: row["outcome"]["task_outcome"]
                    for row in result["rows"]
                }
                passing = sorted(k for k, v in branches.items() if v == "pass")
                record.update(
                    measured_branches=len(branches),
                    passing_branches=passing,
                    passing_branch_count=len(passing),
                    failing_branch_count=sum(v == "failure" for v in branches.values()),
                    unknown_branch_count=sum(v == "unknown" for v in branches.values()),
                    # A usable continuation target requires at least one branch
                    # the teacher can actually recommend.
                    usable_continuation_target=bool(passing),
                    all_branches_failed=not passing,
                    recorded_physics_steps=result["physics_steps"],
                )
            records.append(record)
    return records


def corpus_coverage(records):
    """Added response coverage: how many distinct passing sets a corpus teaches."""
    coverage = {}
    for run_id in sorted({r["run_id"] for r in records}):
        own = [r for r in records if r["run_id"] == run_id and r.get("executed")]
        sets = {tuple(r["passing_branches"]) for r in own if r["passing_branches"]}
        covering = (
            set.intersection(*[set(r["passing_branches"]) for r in own])
            if own and all(r["passing_branches"] for r in own)
            else set()
        )
        coverage[run_id] = dict(
            executed_encounters=len(own),
            distinct_passing_sets=len(sets),
            single_schedule_covers_corpus=bool(covering),
            covering_schedules=sorted(covering),
            encounters_with_no_passing_branch=sum(r.get("all_branches_failed", False) for r in own),
            off_gate_encounters=sum(r["off_gate"] for r in own),
            off_gate_with_usable_target=sum(
                r["off_gate"] and r.get("usable_continuation_target", False) for r in own
            ),
        )
    return coverage


def acquisition_cost(plan):
    """Shared prefix cost and new cost stay separately visible."""
    costs = {}
    for run in plan["runs"]:
        folder = Path(plan["execution_root"]) / run["run_id"] / "controller"
        markers = sorted(folder.glob("complete_*.json"))
        ledger = None
        if markers:
            ledger = read_checked(artifact(markers[-1]))["accounting"]
        costs[run["run_id"]] = dict(
            arm=run["arm"],
            seed=run["seed"],
            completed_boundaries=[int(p.stem.split("_")[1]) for p in markers],
            new_recorded_physics_steps=(
                None if ledger is None else ledger.get("new_recorded_physics_steps")
            ),
            actual_recorded_physics_steps=(
                None if ledger is None else ledger.get("actual_recorded_physics_steps")
            ),
            conservative_charged_steps=(
                None if ledger is None else ledger.get("conservative_charged_steps")
            ),
            unlaunched_assigned_episode_slots=(
                None if ledger is None else ledger.get("unlaunched_assigned_episode_slots")
            ),
            within_budget=None if ledger is None else ledger.get("within_budget"),
        )
    return costs


def cost_denominator(records, costs):
    """Pair each arm's measured steps with the encounters that taught nothing.

    Measured physics steps are NOT a neutral efficiency denominator in this
    design. An encounter no schedule solves terminates early, so it costs fewer
    measured steps than a solvable one; dividing improvement by measured steps
    would therefore reward an arm for collecting unsolvable scenes. The declared
    budget is matched in *assigned episodes*, and that is the comparison basis.
    """
    rollup = {}
    for run_id, cost in costs.items():
        own = [r for r in records if r["run_id"] == run_id and r.get("executed")]
        useless = [r for r in own if r.get("all_branches_failed")]
        rollup[run_id] = dict(
            arm=cost["arm"],
            seed=cost["seed"],
            new_recorded_physics_steps=cost["new_recorded_physics_steps"],
            executed_encounters=len(own),
            encounters_with_no_passing_branch=len(useless),
            steps_in_encounters_that_taught_nothing=sum(
                r.get("recorded_physics_steps") or 0 for r in useless
            ),
        )
    return dict(
        rule=(
            "compare at matched assigned episodes; report measured steps descriptively and "
            "never as an efficiency denominator"
        ),
        why=(
            "an encounter with no passing branch terminates early and so costs fewer measured "
            "steps; per-step normalisation would reward collecting unsolvable scenes"
        ),
        per_corpus=rollup,
    )


def render(result):
    """Render the readout so a partial or null result stays legible and honest."""
    contexts = [row["scene_id"] for row in result["contexts"]]
    short = {scene: scene.replace("support_validation_", "") for scene in contexts}
    out = ["# Support-preserving acquisition pilot — development-validation readout", ""]
    out += [
        f"**{result['measured_episodes']} of {result['assigned_episodes']} assigned episodes "
        f"measured**, {result['measured_physics_steps']} physics steps. "
        f"Unmeasured assignments are listed, never dropped: "
        f"{len(result['unmeasured_assignments'])}.",
        "",
        "Passage is the criterion. Matched passage time is reported only over mutually "
        "successful assignments and is crossing plus stabilization, not whole adaptation "
        "and recovery completion. No reserved layout was touched and no held-out claim is "
        "made.",
        "",
        "## 1. Measured bank capability per context",
        "",
        "| Context | Stratum · band | Bank solvable | Passing schedules | Best bank time (s) |",
        "|---|---|:--:|---|---:|",
    ]
    by_scene = {row["scene_id"]: row for row in result["contexts"]}
    for scene in contexts:
        cap = result["bank_capability"].get(scene, {})
        meta = by_scene[scene]
        passing = cap.get("passing_schedules") or []
        out.append(
            f"| `{short[scene]}` | {meta['stratum']} · {meta['underside_band']} | "
            f"{'yes' if cap.get('bank_solvable') else '**no**'} | "
            f"{', '.join(f'`{p}`' for p in passing) if passing else '—'} | "
            f"{cap.get('best_bank_passage_time_s') if cap.get('best_bank_passage_time_s') else '—'} |"
        )
    unsolved = result["bank_unsolved_contexts"]
    out += [
        "",
        f"Contexts no schedule in the bank solves: **{len(unsolved)}**"
        + (f" (`{'`, `'.join(short[s] for s in unsolved)}`)" if unsolved else "")
        + ". These were drawn from task geometry without consulting feasibility and are "
        "retained, not redrawn.",
        "",
        "## 2. Passage per corpus",
        "",
        "| Policy | Arm | Seed | Passages | Selection failures | Distinct schedules |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for policy_id, row in sorted(
        result["policies"].items(), key=lambda kv: (kv[1].get("arm") or "", kv[1].get("seed") or 0)
    ):
        out.append(
            f"| `{policy_id}` | {row.get('arm') or '—'} | {row.get('seed') or '—'} | "
            f"{row['passages']}/{row['measured']} | {len(row['selection_failures'])} | "
            f"{row['distinct_schedules_used']} |"
        )
    out += [
        "",
        "Individual corpora, never a pooled mean as the primary figure. A selection failure is "
        "a context where the bank demonstrably contains a passing schedule but the policy "
        "failed. Non-constant action is not the criterion; correct selection is.",
        "",
        "## 3. Declared contrasts",
        "",
        "| Seed | Comparison | Passage difference | Mutually successful | Paired time (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["primary_contrasts"]:
        if not row.get("first"):
            continue
        first = (row["first"].split("_", 1)[1]) if "_" in row["first"] else row["first"]
        second = (row["second"].split("_", 1)[1]) if "_" in row["second"] else row["second"]
        paired = row["mean_paired_passage_time_difference_s"]
        out.append(
            f"| {row['seed']} | {first} − {second} | {row['passage_difference']:+d} | "
            f"{row['mutually_successful']} | {paired if paired is not None else '—'} |"
        )
    out += [
        "",
        "Negative paired time favours the first named arm. If broad and mixture improve "
        "equally, the supported statement is that broadening support helps — not that the "
        "mixture is superior.",
        "",
        "## 4. What each new proposal taught, by gate membership",
        "",
        "Stratification recorded in the plan before any physics ran, so this is a "
        "pre-registered mechanism readout and not a post hoc split.",
        "",
        "| Gate membership | Encounters | With a usable continuation target | All branches failed |",
        "|---|---:|---:|---:|",
    ]
    buckets = {}
    for row in result["encounter_mechanism"]:
        if not row.get("executed"):
            continue
        key = (
            f"{'inside' if row.get('inside_positive_screen') else 'outside'} positive · "
            f"{'inside' if row.get('inside_negative_screen') else 'outside'} negative"
        )
        entry = buckets.setdefault(key, [0, 0, 0])
        entry[0] += 1
        entry[1] += bool(row.get("usable_continuation_target"))
        entry[2] += bool(row.get("all_branches_failed"))
    for key in sorted(buckets):
        n, usable, dead = buckets[key]
        out.append(f"| {key} | {n} | {usable} | {dead} |")
    if not buckets:
        out.append("| _no encounter executed yet_ | 0 | 0 | 0 |")
    warning = result["cost_denominator_warning"]
    out += [
        "",
        "## 5. Measured acquisition cost",
        "",
        f"**{warning['rule']}** — {warning['why']}.",
        "",
        "| Corpus | Arm | New steps | Encounters | No passing branch | Steps that taught nothing |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run_id, row in sorted(warning["per_corpus"].items()):
        out.append(
            f"| `{run_id}` | {row['arm']} | {row['new_recorded_physics_steps']} | "
            f"{row['executed_encounters']} | {row['encounters_with_no_passing_branch']} | "
            f"{row['steps_in_encounters_that_taught_nothing']} |"
        )
    prefix = result["shared_prefix"]
    out += [
        "",
        f"The shared {prefix['rounds']}-encounter `{prefix['source_arm']}` prefix is identical "
        "for all three arms; its cost is inherited, not new.",
        "",
        "## 6. Interpretation rules applied",
        "",
    ]
    out += [f"- {rule}" for rule in result["interpretation_rules"]]
    out.append("")
    return "\n".join(out)


def build(args):
    study, prepared, rows, missing = episode_rows(args.panel)
    plan = read_checked(study["pilot_plan"])
    contexts = sorted(row["scene_id"] for row in study["contexts"])
    capability = bank_capability(rows, contexts)
    summaries = policy_summary(rows, contexts, capability)
    by_arm = {row["policy_id"]: row for row in study["models"]}
    contrasts = []
    for seed in plan["acquisition_seeds"]:
        named = {arm: f"seed{seed}_{arm}" for arm in PILOT_ARMS}
        contrasts.extend(
            [
                dict(
                    seed=seed,
                    tests=(
                        "whether preserving exploratory support improves the strict-gate method"
                    ),
                    **(
                        matched_difference(
                            named["support_mixture"], named["support_strict"], summaries
                        )
                        or {}
                    ),
                ),
                dict(
                    seed=seed,
                    tests="whether the geometry-guided portion adds value beyond broad coverage",
                    **(
                        matched_difference(
                            named["support_mixture"], named["support_broad"], summaries
                        )
                        or {}
                    ),
                ),
                dict(
                    seed=seed,
                    tests="whether broadening support alone changes the strict-gate method",
                    **(
                        matched_difference(
                            named["support_broad"], named["support_strict"], summaries
                        )
                        or {}
                    ),
                ),
            ]
        )
    records = encounter_mechanism(plan)
    result = dict(
        schema="motion2scene_support_pilot_readout_v1",
        panel=artifact(args.panel / "prepared.json"),
        pilot_plan=study["pilot_plan"],
        validation_sample=study["validation_sample"],
        contexts=study["contexts"],
        ancestry_groups=study["ancestry_groups"],
        assigned_episodes=len(prepared["assignments"]),
        measured_episodes=len(rows),
        unmeasured_assignments=missing,
        measured_physics_steps=sum(row["physics_steps"] for row in rows),
        bank_capability=capability,
        bank_unsolved_contexts=[
            scene for scene, row in capability.items() if not row["bank_solvable"]
        ],
        policies={
            policy_id: dict(
                summary,
                arm=by_arm.get(policy_id, {}).get("arm"),
                seed=by_arm.get(policy_id, {}).get("seed"),
                origin=by_arm.get(policy_id, {}).get("origin"),
            )
            for policy_id, summary in summaries.items()
        },
        primary_contrasts=contrasts,
        encounter_mechanism=records,
        corpus_coverage=corpus_coverage(records),
        off_gate_summary=dict(
            Counter(
                f"{r['proposal_channel']}:{'off_gate' if r['off_gate'] else 'on_gate'}"
                for r in records
            )
        ),
        acquisition_cost=acquisition_cost(plan),
        cost_denominator_warning=cost_denominator(records, acquisition_cost(plan)),
        shared_prefix=dict(
            rounds=plan["shared_prefix_rounds"],
            source_arm=plan["shared_prefix_source_arm"],
            note=(
                "identical for all three arms; its measured cost is inherited, not new, and is "
                "reported once per arm rather than attributed to any one of them"
            ),
        ),
        interpretation_rules=[
            "passage is the criterion; nonconstant actions and label counts are not",
            "a development-panel null is not proof of population equality at every budget",
            "equal improvement by broad and mixture supports broadening support, not the mixture",
            "screen recall or action diversity improving without physical policy improvement "
            "leaves the central method hypothesis unsupported",
            "association between selected encounters and a corpus result does not identify a "
            "causal effect of those encounters",
            "passage time is crossing plus stabilization, not whole adaptation completion",
            "measured physics steps are descriptive, never an efficiency denominator: an "
            "encounter no schedule solves terminates early and therefore costs less, so "
            "per-step normalisation rewards collecting unsolvable scenes",
            "an off-gate encounter with no passing branch offers the teacher no verified "
            "continuation, so it adds cost without adding a lesson",
        ],
        reserved_layouts_touched=False,
        held_out_claim=False,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    ref = write_new(args.out / "result.json", result)
    (args.out / "report.md").write_text(render(result))
    print(
        json.dumps(
            dict(
                result=ref,
                measured=len(rows),
                assigned=len(prepared["assignments"]),
                bank_unsolved=len(result["bank_unsolved_contexts"]),
                passages={k: v["passages"] for k, v in summaries.items()},
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args())
