#!/usr/bin/env python3
"""Prepare the support-preserving pilot's fresh reservoir and three queues; CPU only.

Draws the locked task domain deeper than the M8 pool, keeps only rounds that were
never screened, and builds three nested proposal controls over that one shared
reservoir: the unchanged strict executed-contrast rule, a broad draw-order queue
with no geometric gate, and a committed half-and-half mixture schedule.

The screened prefix of the draw stream is audited against the existing M8 pool by
exact geometry comparison, so domain drift is detected without spending a single
extra clearance query. No physics, no sensor query, no physical outcome and no
reserved-layout geometry beyond exact-identity exclusion.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bundle_motion2scene_sources import closure  # noqa: E402
import joblib  # noqa: E402
from motion2scene_materialize_evaluation import place_beams  # noqa: E402
from motion2scene_prepare_acquisition_pools import beam_key, read_bound  # noqa: E402
from motion2scene_support_pool import (  # noqa: E402
    M8_POOL_ROUNDS_PER_STRATUM,
    PILOT_ARMS,
    fresh_candidates,
    support_queues,
)
from motion2scene_timed_scene_screen import beam_clearance  # noqa: E402
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_pool import (  # noqa: E402
    SEEDS,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    author_course,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (  # noqa: E402
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    CollisionCapsule,
    body_capsules_world,
)

OFFSET_COUNT = 81
MARGIN_M = 0.01


def offsets_world_xyz_yaw():
    """The unchanged 81 placement offsets: nominal plus the 80-point product grid."""
    import itertools

    values = [[0.0, 0.0, 0.0, 0.0]] + [
        list(v)
        for v in itertools.product(
            (-0.02, 0, 0.02), (-0.02, 0, 0.02), (-0.005, 0, 0.005), (-0.02, 0, 0.02)
        )
        if any(v)
    ]
    if len(values) != OFFSET_COUNT:
        raise ValueError("the original 81-offset placement envelope is required")
    return values


def prepare(args):
    """Register the domain, the fresh draw and the exclusions before any query."""
    started = time.monotonic()
    result_ref = artifact(args.executions)
    result = read_bound(result_ref)
    manifest = read_bound(result["manifest"])
    registry_ref = result["registry"]
    bank = load_verified_registry(registry_ref["path"], registry_ref["sha256"])
    if (
        len(bank.option_ids) != 7
        or bank.frame_count != 299
        or any(not r["qualified"] for r in result["rows"])
    ):
        raise ValueError("the physically verified complete seven-schedule bank is required")
    geometry = read_bound(manifest["geometry"])
    references = [result_ref, result["manifest"], registry_ref, manifest["geometry"]]
    for layer in geometry["layers"]:
        checked(Path(layer["path"]), layer["sha256"])
        references.append(layer)
    for row in result["rows"]:
        evidence = read_bound(row["evidence"])
        ref = evidence["artifacts"]["trajectory"]
        checked(Path(ref["path"]), ref["sha256"])
        references.extend([row["evidence"], ref])
    neutral_ref = manifest["cells"][0]["motion"]
    neutral = next(
        iter(joblib.load(checked(Path(neutral_ref["path"]), neutral_ref["sha256"])).values())
    )
    route = np.asarray(neutral["root_trans_offset"])[:, :2]
    scene_ref = manifest["cells"][0]["scene"]
    checked(Path(scene_ref["path"]), scene_ref["sha256"])
    lock_ref = artifact(args.lock)
    lock = read_bound(lock_ref)
    reserved = {
        beam_key(beam)
        for layout in lock["evaluation"]["layouts"]
        for variant in layout["fixed_world_variants"]
        for beam in variant["beams"]
    }
    old_contexts, integration_beams = [], set()
    for path in args.integration_definitions:
        ref = artifact(path)
        definition = read_bound(ref)
        old_contexts.append(ref)
        integration_beams.update(
            beam_key(b) for b in definition.get("beams", [definition.get("beam")]) if b is not None
        )
    previous_ref = artifact(args.previous_pools / "result.json")
    previous = read_bound(previous_ref)
    previous_by_seed = {row["seed"]: row for row in previous["pools"]}
    pools, audits = [], []
    for seed in SEEDS:
        draw = fresh_candidates(
            lock["generation"],
            seed,
            args.rounds_per_stratum,
            bank.option_ids,
            prior_rounds=M8_POOL_ROUNDS_PER_STRATUM,
        )
        # Domain-drift audit: the screened prefix of this deeper draw must equal
        # the M8 pool's candidates exactly. Pure comparison; zero new queries.
        earlier = read_bound(previous_by_seed[seed]["candidates"])["rows"]
        earlier_by_id = {row["candidate_id"]: row for row in earlier}
        mismatches = []
        for row in draw["screened_prefix"]:
            other = earlier_by_id.get(row["candidate_id"])
            if other is None:
                continue
            if (
                other["beam_specifications"] != row["beam_specifications"]
                or other["stratum"] != row["stratum"]
                or other["future_branch_order"] != row["future_branch_order"]
                or other["assigned_positive_option_id"] != row["assigned_positive_option_id"]
            ):
                mismatches.append(row["candidate_id"])
        audits.append(
            dict(
                seed=seed,
                screened_prefix=len(draw["screened_prefix"]),
                compared_against_previous=len(
                    [r for r in draw["screened_prefix"] if r["candidate_id"] in earlier_by_id]
                ),
                geometry_mismatches=mismatches,
                fresh_candidates=len(draw["fresh"]),
            )
        )
        if mismatches:
            raise ValueError(f"the locked draw stream changed at seed {seed}: {mismatches[:3]}")
        for candidate in draw["fresh"]:
            candidate["beams"] = place_beams(route, candidate["beam_specifications"])
            keys = {beam_key(beam) for beam in candidate["beams"]}
            candidate["excluded_reason"] = (
                "exact_reserved_beam"
                if keys & reserved
                else "exact_integration_context_beam" if keys & integration_beams else None
            )
        pools.append(dict(seed=seed, candidates=draw["fresh"]))
    args.out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        destination = args.out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(destination)})
    write_new(args.out / "draws.json", dict(pools=pools, draw_stream_audit=audits))
    write_new(
        args.out / "registration.json",
        dict(
            schema="motion2scene_support_preserving_pool_v1",
            registered_before_clearance=True,
            status="candidate_geometry_only",
            hypothesis=(
                "The hard executed-envelope acceptance gate suppresses useful physical "
                "teaching encounters. Keeping its geometry as a proposal bias while allowing "
                "bounded exploration outside it may produce better teaching data. Proposed, "
                "not demonstrated."
            ),
            domain_lock=lock_ref,
            generation_domain=lock["generation"],
            previous_pools=previous_ref,
            reserved_layouts_excluded_by_exact_beam_identity=len(reserved),
            excluded_integration_definitions=old_contexts,
            new_scene_or_heldout_outcome_files_loaded=False,
            option_qualification_inputs=references,
            neutral_route=neutral_ref,
            room_template=scene_ref,
            registry=registry_ref,
            implementation=sources,
            draws=artifact(args.out / "draws.json"),
            acquisition_seeds=list(SEEDS),
            rounds_per_stratum=args.rounds_per_stratum,
            screened_rounds_per_stratum_reserved=M8_POOL_ROUNDS_PER_STRATUM,
            reservoir_freshness=(
                "Only draw rounds beyond the M8 pool's 256 are selectable; the earlier rounds "
                "are reproduced and compared, never reselected."
            ),
            proposal_rng="Python random.Random MT19937; the frozen draw_pool stream, extended",
            arms=list(PILOT_ARMS),
            margin_m=MARGIN_M,
            capped_clearance_m=0.20,
            offsets_world_xyz_yaw=offsets_world_xyz_yaw(),
            positive_screen=(
                "Unchanged: all 81 offsets, every beam, whole recorded 50Hz trajectory outer "
                "envelope >=.01m. Applied to the strict arm only."
            ),
            negative_screen=(
                "Unchanged: at least one distinct option has native primitive-subset nominal "
                "interference <=-.01m at some beam. Applied to the strict arm only."
            ),
            arm_selection={
                "support_strict": (
                    "The unchanged executed-contrast rule and ranking, delegated verbatim to "
                    "acquisition_queues; all executable pairs, maximum min(positive slack, "
                    "negative nominal slack)"
                ),
                "support_broad": (
                    "Every validly drawn candidate in original draw order. Neither geometric "
                    "gate applied; environment validity, interior station, finite locked "
                    "bounds and the same seven-branch action interface are preserved."
                ),
                "support_mixture": (
                    "Committed finite alternation of the two channels above at "
                    "epsilon=1/2, fixed before any new outcome; slots 0,2 guided and 1,3 "
                    "exploratory over the shared stratum schedule."
                ),
            },
            support_claim_sacrificed=(
                "The mixture no longer guarantees that every proposed scene satisfies the old "
                "geometric screen. The physical passage/contact/recovery contract is unchanged "
                "and exploration is simulation only."
            ),
            physics_steps=0,
            sensor_queries=0,
            clearance_queries=0,
            independent_physical_corpora=0,
        ),
    )
    write_new(
        args.out / "preparation_execution.json",
        dict(
            registration=artifact(args.out / "registration.json"),
            preparation_wall_seconds=time.monotonic() - started,
            clearance_queries=0,
            physics_steps=0,
            draw_stream_audit=audits,
        ),
    )
    print(
        json.dumps(
            dict(
                registration=artifact(args.out / "registration.json"),
                fresh_proposals=sum(len(p["candidates"]) for p in pools),
                draw_stream_audit=audits,
            )
        ),
        flush=True,
    )


def screen(candidates, executions, bank, offsets):
    """The unchanged two-stage clearance screen over the shared reservoir."""
    count, actions = len(candidates), len(bank.option_ids)
    outer = np.full((count, OFFSET_COUNT, actions), np.nan)
    inner = np.zeros((count, actions))
    evaluated = np.ones((count, actions), dtype=np.int64)
    query_outer = np.zeros((count, actions), dtype=np.int64)
    query_inner = np.zeros((count, actions), dtype=np.int64)
    started = time.monotonic()
    for i, candidate in enumerate(candidates):
        for a, option_id in enumerate(bank.option_ids):
            for kind in ("outer", "inner"):
                value = min(
                    beam_clearance(executions[option_id][kind], beam, offsets[0])
                    for beam in candidate["beams"]
                )
                if kind == "outer":
                    outer[i, 0, a] = value
                    query_outer[i, a] += len(candidate["beams"])
                else:
                    inner[i, a] = value
                    query_inner[i, a] += len(candidate["beams"])
            if outer[i, 0, a] >= MARGIN_M:
                for j, offset in enumerate(offsets[1:], 1):
                    outer[i, j, a] = min(
                        beam_clearance(executions[option_id]["outer"], beam, offset)
                        for beam in candidate["beams"]
                    )
                    query_outer[i, a] += len(candidate["beams"])
                evaluated[i, a] = OFFSET_COUNT
        if (i + 1) % 40 == 0:
            print(
                json.dumps(
                    dict(screened=i + 1, total=count, seconds=round(time.monotonic() - started, 1))
                ),
                flush=True,
            )
    return outer, inner, evaluated, query_outer, query_inner


def load_executions(registration, bank):
    """Rebuild the recorded whole-body capsule sweeps for the seven schedules."""
    source = read_bound(registration["option_qualification_inputs"][0])
    manifest = read_bound(source["manifest"])
    geometry = read_bound(manifest["geometry"])
    shapes = {"outer": {}, "inner": {}}
    for shape in geometry["shapes"]:
        capsule = CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        shapes["outer"].setdefault(shape["owner"], []).append(capsule)
        if shape["role"] == "native_primitive_subset":
            shapes["inner"].setdefault(shape["owner"], []).append(capsule)
    executions = {}
    for row in source["rows"]:
        ref = read_bound(row["evidence"])["artifacts"]["trajectory"]
        payload = load_reset_capture(checked(Path(ref["path"]), ref["sha256"]))
        if len(payload["body_pos_w"]) != bank.frame_count - 1:
            raise ValueError("complete 299-frame-bank execution required")
        executions[row["cell_id"]] = {
            kind: body_capsules_world(
                payload["body_pos_w"], payload["body_quat_w"], payload["body_names"], capsules=value
            )
            for kind, value in shapes.items()
        }
    return executions


def run(out):
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    for ref in registration["implementation"]:
        checked(Path(ref["path"]), ref["sha256"])
    for ref in registration["option_qualification_inputs"]:
        checked(Path(ref["path"]), ref["sha256"])
    checked(Path(registration["domain_lock"]["path"]), registration["domain_lock"]["sha256"])
    bank = load_verified_registry(
        registration["registry"]["path"], registration["registry"]["sha256"]
    )
    executions = load_executions(registration, bank)
    template = checked(
        Path(registration["room_template"]["path"]), registration["room_template"]["sha256"]
    ).read_text()
    draws = read_bound(registration["draws"])
    offsets = registration["offsets_world_xyz_yaw"]
    summaries = []
    for pool in draws["pools"]:
        folder = out / f"seed_{pool['seed']}"
        folder.mkdir(exist_ok=False)
        start = time.monotonic()
        candidates = [c for c in pool["candidates"] if c["excluded_reason"] is None]
        outer, inner, evaluated, query_outer, query_inner = screen(
            candidates, executions, bank, offsets
        )
        clearance_seconds = time.monotonic() - start
        minima = np.nanmin(outer, axis=1)
        selection_started = time.monotonic()
        queues = support_queues(
            candidates,
            bank.option_ids,
            minima,
            evaluated,
            inner,
            offset_count=OFFSET_COUNT,
            margin_m=MARGIN_M,
        )
        selection_seconds = time.monotonic() - selection_started
        raw_path = folder / "geometry_queries.npz"
        np.savez_compressed(
            raw_path,
            candidate_ids=np.asarray([c["candidate_id"] for c in candidates]),
            option_ids=np.asarray(bank.option_ids),
            outer_clearance_by_offset_m=outer,
            nominal_inner_clearance_m=inner,
            evaluated_offset_counts=evaluated,
            outer_beam_queries=query_outer,
            inner_beam_queries=query_inner,
        )
        candidate_rows = []
        for i, candidate in enumerate(candidates):
            scene_path = folder / "scenes" / f"{candidate['candidate_id']}.usda"
            scene_path.parent.mkdir(exist_ok=True)
            scene_path.write_text(
                author_course(template, candidate["beams"], course_id=candidate["candidate_id"])
            )
            definition_path = folder / "definitions" / f"{candidate['candidate_id']}.json"
            write_new(
                definition_path,
                dict(
                    schema="motion2scene_timed_schedule_scene_v1",
                    split="development",
                    usage=(
                        "support-preserving pilot acquisition candidate; no outcomes yet; "
                        "proposal support is the only thing that varies across arms"
                    ),
                    scene_id=candidate["candidate_id"],
                    scene=artifact(scene_path),
                    beams=candidate["beams"],
                    beam_collision_enabled=[True] * len(candidate["beams"]),
                    provenance=dict(
                        pool_registration=artifact(registration_path),
                        candidate_id=candidate["candidate_id"],
                        source_ancestry="source41002 development motion ancestry",
                    ),
                    physical_labels=None,
                    observation_timing_status="pending native 65-ray execution",
                ),
            )
            candidate_rows.append(
                dict(
                    **candidate,
                    definition=artifact(definition_path),
                    positive_offset_counts=evaluated[i].tolist(),
                    minimum_evaluated_positive_clearance_m=minima[i].tolist(),
                    robust_positive_eligible=(
                        (evaluated[i] == OFFSET_COUNT) & (minima[i] >= MARGIN_M)
                    ).tolist(),
                    nominal_negative_clearance_m=inner[i].tolist(),
                )
            )
        definitions = {r["candidate_id"]: r["definition"] for r in candidate_rows}
        branch_orders = {r["candidate_id"]: r["future_branch_order"] for r in candidate_rows}
        for arm in ("support_strict", "support_broad"):
            for row in queues[arm]:
                row["definition"] = definitions[row["candidate_id"]]
                row["future_branch_order"] = branch_orders[row["candidate_id"]]
        support = dict(
            reservoir=len(candidate_rows),
            strict_eligible=len(queues["support_strict"]),
            broad_eligible=len(queues["support_broad"]),
            outside_positive_screen=sum(
                not row["inside_positive_screen"] for row in queues["support_broad"]
            ),
            outside_negative_screen=sum(
                not row["inside_negative_screen"] for row in queues["support_broad"]
            ),
            outside_both_screens=sum(
                not row["inside_positive_screen"] and not row["inside_negative_screen"]
                for row in queues["support_broad"]
            ),
        )
        if not support["outside_positive_screen"] or not support["outside_negative_screen"]:
            raise ValueError(
                "the exploration channel must have support outside both original screens"
            )
        arm_refs = {}
        for arm in PILOT_ARMS:
            path = folder / f"{arm}.json"
            rows = queues.get(arm, [])
            write_new(
                path,
                dict(
                    schema="motion2scene_support_candidate_queue_v1",
                    arm=arm,
                    pool_registration=artifact(registration_path),
                    geometry=artifact(raw_path),
                    acquisition_seed=pool["seed"],
                    future_physics_seed=pool["seed"],
                    proposed_candidates=len(pool["candidates"]),
                    screened_candidates=len(candidate_rows),
                    eligible_candidates=len(rows),
                    rows=rows,
                    realised_by=(
                        "the committed slot-to-channel schedule over the two source queues"
                        if arm == "support_mixture"
                        else "this ranked queue"
                    ),
                    mixture_schedule=queues["mixture_schedule"],
                    mixture_strict_fraction=queues["mixture_strict_fraction"],
                    coverage_counts=dict(Counter(r["stratum"] for r in rows)),
                    physical_rollout_steps=0,
                    actual_sensor_queries=0,
                    acquired_corpora=0,
                ),
            )
            arm_refs[arm] = artifact(path)
        write_new(
            folder / "candidates.json",
            dict(
                rows=candidate_rows,
                excluded=[c for c in pool["candidates"] if c["excluded_reason"] is not None],
            ),
        )
        summary = dict(
            seed=pool["seed"],
            proposed_candidates=len(pool["candidates"]),
            screened_candidates=len(candidate_rows),
            excluded_before_geometry=len(pool["candidates"]) - len(candidate_rows),
            actual_shared_outer_queries=int(query_outer.sum()),
            actual_shared_inner_queries=int(query_inner.sum()),
            actual_shared_clearance_queries=int(query_outer.sum() + query_inner.sum()),
            shared_clearance_search_seconds=clearance_seconds,
            all_arm_queue_selection_seconds=selection_seconds,
            candidates=artifact(folder / "candidates.json"),
            geometry=artifact(raw_path),
            arms=arm_refs,
            eligible_counts={arm: len(queues.get(arm, [])) for arm in PILOT_ARMS},
            support=support,
            physical_steps=0,
        )
        write_new(folder / "result.json", summary)
        summaries.append({**summary, "result": artifact(folder / "result.json")})
        print(json.dumps(dict(seed=pool["seed"], support=support)), flush=True)
    write_new(
        out / "result.json",
        dict(
            schema="motion2scene_support_preserving_pool_result_v1",
            registration=artifact(registration_path),
            pools=summaries,
            total_fresh_proposals=sum(r["proposed_candidates"] for r in summaries),
            actual_shared_clearance_queries=sum(
                r["actual_shared_clearance_queries"] for r in summaries
            ),
            physical_steps=0,
            actual_sensor_queries=0,
            independent_physical_corpora=0,
            pending=(
                "Matched seven-branch encounters, causal sensor timing, student decisions and "
                "the physical outcomes of off-gate proposals; all shortfalls retained"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executions", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--previous-pools", type=Path)
    parser.add_argument("--integration-definitions", type=Path, nargs="*", default=[])
    parser.add_argument("--rounds-per-stratum", type=int, default=512)
    args = parser.parse_args()
    prepare(args) if args.action == "prepare" else run(args.out)
