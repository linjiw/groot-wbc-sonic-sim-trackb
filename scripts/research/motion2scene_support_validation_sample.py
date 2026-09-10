#!/usr/bin/env python3
"""Mint the pilot's development-validation sample from task geometry alone.

Declared before any physics on this sample. The draw uses only the locked
overhead-traversal domain: its beam families, its common ranges, and a joint
stratification over (family x underside band). It issues zero clearance queries,
consults no physical outcome, no teacher table, no policy, no known policy
failure and no acceptance predicate of any acquisition arm.

Consequences that are accepted, not engineered away:

* The sample will contain contexts that no schedule in the bank solves. Those are
  retained with their measured outcomes and reported; the panel is never redrawn
  because it turned out easy, hard or unfavourable.
* Every layout is an independent draw, so each is its own base layout. The
  ancestry grouping is emitted so a later perturbation variant can never be
  reported as an independent validation item.

Exact-identity exclusion is applied against the eighteen reserved layouts and
their stress variants, the six M8 development contexts, and every candidate in
both training reservoirs. That is the only use made of reserved geometry.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]

from bundle_motion2scene_sources import closure  # noqa: E402
import joblib  # noqa: E402
from motion2scene_materialize_evaluation import place_beams  # noqa: E402
from motion2scene_prepare_acquisition_pools import beam_key, read_bound  # noqa: E402
from motion2scene_support_validation import (  # noqa: E402
    CONTEXTS_PER_CELL,
    UNDERSIDE_BANDS,
    VALIDATION_RNG_SEED,
    ancestry_groups,
    draw_validation_specifications,
    underside_bands,
)
from motion2scene_timing_diagnostic import artifact, checked, write_new  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_course import (  # noqa: E402
    author_course,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_options import (  # noqa: E402
    load_verified_registry,
)


def build(args):
    started = time.monotonic()
    result_ref = artifact(args.executions)
    result = read_bound(result_ref)
    manifest = read_bound(result["manifest"])
    registry_ref = result["registry"]
    bank = load_verified_registry(registry_ref["path"], registry_ref["sha256"])
    if len(bank.option_ids) != 7 or bank.frame_count != 299:
        raise ValueError("the complete seven-schedule bank is required")
    neutral_ref = manifest["cells"][0]["motion"]
    neutral = next(
        iter(joblib.load(checked(Path(neutral_ref["path"]), neutral_ref["sha256"])).values())
    )
    route = np.asarray(neutral["root_trans_offset"])[:, :2]
    scene_ref = manifest["cells"][0]["scene"]
    template = checked(Path(scene_ref["path"]), scene_ref["sha256"]).read_text()
    lock_ref = artifact(args.lock)
    lock = read_bound(lock_ref)
    excluded = {}
    for layout in lock["evaluation"]["layouts"]:
        for variant in layout["fixed_world_variants"]:
            for beam in variant["beams"]:
                excluded[beam_key(beam)] = f"reserved:{layout['layout_id']}"
    development = []
    for path in args.development_definitions:
        ref = artifact(path)
        definition = read_bound(ref)
        development.append(ref)
        for beam in definition.get("beams", []):
            excluded[beam_key(beam)] = f"development:{definition['scene_id']}"
    reservoirs = []
    for pools in args.reservoirs:
        pool_ref = artifact(pools / "result.json")
        reservoirs.append(pool_ref)
        for pool in read_bound(pool_ref)["pools"]:
            for row in read_bound(pool["candidates"])["rows"]:
                for beam in row["beams"]:
                    excluded.setdefault(beam_key(beam), f"training:{row['candidate_id']}")
    drawn = draw_validation_specifications(
        lock["generation"],
        seed=args.rng_seed,
        bands=UNDERSIDE_BANDS,
        per_cell=args.contexts_per_cell,
    )
    collisions = []
    for layout in drawn:
        layout["beams"] = place_beams(route, layout["beam_specifications"])
        for beam in layout["beams"]:
            owner = excluded.get(beam_key(beam))
            if owner is not None:
                collisions.append(dict(layout_id=layout["layout_id"], collides_with=owner))
    if collisions:
        raise ValueError(f"validation draw collides with excluded geometry: {collisions}")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in sorted(closure([Path(__file__)])):
        destination = args.out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        sources.append({**artifact(path), "snapshot": artifact(destination)})
    contexts = []
    for layout in drawn:
        scene_path = args.out / "scenes" / f"{layout['layout_id']}.usda"
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        scene_path.write_text(
            author_course(template, layout["beams"], course_id=layout["layout_id"])
        )
        definition_path = args.out / "definitions" / f"{layout['layout_id']}.json"
        write_new(
            definition_path,
            dict(
                schema="motion2scene_timed_schedule_scene_v1",
                # The frozen collector accepts only "development" or
                # "reserved_evaluation_v3"; widening it would break every
                # acquisition arm, so the validation role is carried in
                # provenance instead of in the split field.
                split="development",
                sample_role="development_validation",
                usage=(
                    "support-preserving pilot development-validation context; drawn from task "
                    "geometry only; no acceptance predicate, outcome or policy failure used"
                ),
                scene_id=layout["layout_id"],
                scene=artifact(scene_path),
                beams=layout["beams"],
                beam_collision_enabled=[True] * len(layout["beams"]),
                provenance=dict(
                    sample_registration=str(args.out / "registration.json"),
                    base_layout_id=layout["base_layout_id"],
                    stratum=layout["stratum"],
                    underside_band=layout["underside_band"],
                    underside_band_range_m=layout["underside_band_range_m"],
                    rng_seed=layout["rng_seed"],
                    draw_order=layout["draw_order"],
                    clearance_queries=0,
                    physical_outcomes_used=False,
                    selected_by_policy_failure=False,
                    source_ancestry="source41002 development motion ancestry",
                ),
                physical_labels=None,
                observation_timing_status="pending native 65-ray execution",
            ),
        )
        contexts.append(
            dict(
                scene_id=layout["layout_id"],
                base_layout_id=layout["base_layout_id"],
                stratum=layout["stratum"],
                underside_band=layout["underside_band"],
                scene_definition=artifact(definition_path),
            )
        )
    write_new(
        args.out / "registration.json",
        dict(
            schema="motion2scene_support_validation_sample_v1",
            declared_before_any_physics_on_this_sample=True,
            purpose=(
                "A development-validation sample separate from the six-context M8 panel, so "
                "the pilot is not read out on contexts minted by an acquisition arm's own "
                "predicate. It is development evidence, not the reserved held-out claim."
            ),
            domain_lock=lock_ref,
            generation_domain=lock["generation"],
            stratification="joint over declared beam family x underside band; both task geometry",
            underside_bands=underside_bands(lock["generation"]["common_ranges"]),
            rng_seed=args.rng_seed,
            contexts_per_cell=args.contexts_per_cell,
            contexts=contexts,
            ancestry_groups=ancestry_groups(drawn),
            excluded_reserved_layouts=len(lock["evaluation"]["layouts"]),
            excluded_development_definitions=development,
            excluded_training_reservoirs=reservoirs,
            exact_identity_collisions=collisions,
            neutral_route=neutral_ref,
            room_template=scene_ref,
            registry=registry_ref,
            implementation=sources,
            retention=(
                "All drawn contexts are retained with their measured outcomes, including "
                "bank-unsolved and unknown cases. The sample is never redrawn on the basis of "
                "its outcomes."
            ),
            not_used=[
                "predicted clearance or interference of any option",
                "the positive 81-offset robustness bar",
                "the negative nominal-interference test",
                "max-min-slack ranking",
                "any physical outcome, teacher table or policy",
                "known policy failures",
                "reserved layout geometry beyond exact-identity exclusion",
            ],
            clearance_queries=0,
            physics_steps=0,
            wall_seconds=time.monotonic() - started,
        ),
    )
    print(
        json.dumps(
            dict(
                registration=artifact(args.out / "registration.json"),
                contexts=len(contexts),
                exact_identity_collisions=collisions,
                clearance_queries=0,
                physics_steps=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--development-definitions", type=Path, nargs="*", default=[])
    parser.add_argument("--reservoirs", type=Path, nargs="*", default=[])
    parser.add_argument("--rng-seed", type=int, default=VALIDATION_RNG_SEED)
    parser.add_argument("--contexts-per-cell", type=int, default=CONTEXTS_PER_CELL)
    build(parser.parse_args())
