# LFH Repository Reconnaissance

**Status:** Phase 0 accepted on 2026-08-20; amended with the review conditions
in `REVIEW_PHASE0.md`. This document is a read-only reconciliation of the LFH
design with the current SweepCF code and artifacts. No generator code, frozen
split, prediction register, or rollout artifact was changed.

## Executive Verdict

The LFH direction is implementable, but not literally from the current plan.
The repository can support physics-verified, single-constraint scene synthesis
with its executed body poses and capsule model. Four corrections are required:

1. Treat semantic keypoints as groups of existing collision capsules, not new
   single-radius spheres.
2. Repair the release/index identity and coverage contract before optimizing
   coverage. Scene variants must not count as independent causal families.
3. Restrict v1 edits to `local_crouch` and `local_arm_tuck`; side-step,
   narrow-stance, and step-high are predicates or proposals, not implemented
   trajectory edit operators.
4. Require paired, executed empty-room trajectories for measured delivery.
   A Phase-2 artifact audit supersedes the initial reading: `mf_005_c08` has
   neither matching nominal nor adapted plane evidence, and its legacy manifest
   also lacks explicit axis/motion identities and contact attribution.

Physics remains the only pass/fail verdict. Geometry, semantic groups, and any
delivery model may propose or diagnose scenes only.

## Invariant Audit

| Invariant | Repository status | LFH consequence |
|---|---|---|
| Frozen SONIC main experiment | Intact | Do not alter SONIC artifacts or claims. |
| Frozen evaluation splits | Primary `gear_sonic/data/splits/scene_first_v1.json` and contingency `scene_first_v2.json` are parameter manifests; scenes are not materialized | V1 is the active 30-scene primary split. V2 remains an unlabelled, immutable second vintage unless the preregistered ambiguous-zone rule activates it. |
| Claim 5 / prediction register | Existing files are dirty and user-owned | Do not edit or adjudicate them during LFH work. |
| Physics-only verdict | Existing scores use rollout outcomes; geometry supplies explanation | Preserve this separation in every stage. |
| Executed empty-room inputs | Available for `cf_005_056`; neither matching member is available for local `mf_005_c08` | Refuse synthesis when the required executed pair is absent. |
| Single changed cause | No general `C \\ B` validator exists | Add a CPU geometric validator before any rollout. |
| Measured-back placement | Shelf-specific read-back exists | Generalize to emitted primitive world AABBs and enforce 0.5 mm tolerance. |
| Additive schemas | Current release and flat index are separate builders | Extend both; retain all existing fields and artifacts. |

## Concept-to-Repository Map

| LFH concept | Current implementation or artifact | Reconciliation |
|---|---|---|
| Semantic and embodiment gate | `gear_sonic/dataset_generation/reference_gate.py`; `scripts/research/gate_references_for_rollout.py` | Embodiment/self-collision/predicate checks exist. Physics trackability is not joined into `motions.csv`; candidate selection must join gate results with accepted plane rollouts. |
| Delivery model `D_phi` | Ratios and caps in `gear_sonic/dataset_generation/local_adaptation.py` | Use only as a proposal/calibration layer. It cannot supply a verdict or replace an executed trajectory. |
| Edit operators | `local_crouch`, `local_arm_tuck` in `gear_sonic/dataset_generation/local_adaptation.py` | These are the only v1 operators. Existing side/narrow/step checks are behavior predicates. |
| Executed state contract | Pickled payloads consumed by `gear_sonic/dataset_generation/trajectory_segments.py` | Payloads contain `(T,30,3)` body positions, body quaternions, names, root states, joints, reference qpos, forces, and tokens. `best_evaluable_payload` handles reset splits. |
| Tier-2 collision geometry | `body_capsules_world`, `swept_point_cloud`, and clearance helpers in `gear_sonic/dataset_generation/swept_volume.py` | Twenty-nine capsules across 14 collision-bearing links provide the authoritative geometric screen. |
| Critical interval/binding anatomy | `gear_sonic/dataset_generation/criticality_map.py`; station-envelope functions in `gear_sonic/dataset_generation/motion_envelope.py` | Height bands, station envelopes, one-sided width, and binding capsule owners exist. They need semantic grouping and paired executed inputs. |
| Constraint distance | `gear_sonic/dataset_generation/constraint_distance.py` | Retains per-frame arclength, footprint, and capsule-clearance series; suitable for diagnostics, never verdicts. |
| Paired boundary synthesis | `gear_sonic/dataset_generation/counterfactual_family.py`; `scripts/research/build_counterfactual_family.py` | Binary-searches capsule/box boundary to 5 mm and measures a rendered shelf back. Generalize the solver and authoring handle. |
| Graded-family planner | `scripts/research/plan_graded_families.py` | Not valid LFH evidence: it uses reference-derived nominal/adapted envelopes rather than paired executed adaptations. |
| Primitive scene authoring | `gear_sonic/dataset_generation/clutter_scene_builder.py`; `scripts/research/build_graded_scene.py` | Self-contained Z-up USDA with Plane/Cube collision primitives is compatible with v1. Current obstacle face extents are fixed, not derived from both swept bodies. |
| Route validation | `gear_sonic/dataset_generation/scene_route_check.py` | Confirms the root route enters an obstacle footprint, but does not prove anti-skirt face coverage. |
| CPU preflight | `scripts/research/batch_preflight.py`; `gear_sonic/dataset_generation/scene_asset_preflight.py` | Useful base for semantic, geometry, route, provenance, and package checks. Add non-binding keepout and generalized measured-back checks. |
| Family scoring | `scripts/research/score_family_batch.py` | Keep outcome scoring, but group by stable causal family plus variant identity. |
| Coverage/indexing | `scripts/research/build_dataset_release.py`, `build_dataset_index.py`, and `render_corpus_stats.py` | Current flat fields cannot derive the full DCS. Add explicit axis, constraint coordinate, binding group, source family, and variant fields before coverage claims. |
| Fixed evaluation set | `scene_first_v1.json` | Thirty records: 10 overhead, 10 lateral, 10 floor. It is a sealed evaluation artifact, not a generation queue. |
| Qualitative views | rollout ego video; `render_room_camera.py`, `render_family_views.py`; `render_multiview.sh` | Use ego plus CPU room/side reconstruction for v1 contact sheets. Chase/wrist require extra simulation passes; the configured overhead pass is blank. |

## Planned Component Disposition

| Plan concept or experiment | Phase-0 finding | Disposition |
|---|---|---|
| `ConstraintSpec` | No versioned schema currently represents paired executed inputs, binding axis, semantic group, face coordinate, and keepout together | Add only after the index identity contract. References must be content-addressable artifacts, not inferred directory names. |
| Archetype library | The primitive catalog has 17 furniture types, but no archetype interface with a binding-face handle and complete authored-prim manifest | Reuse the deterministic USDA writer; add the interface in Phase 2. Baseline `shelf_plank` must be the first golden archetype. |
| Keepout validator | Capsule distance queries exist; `C \\ B`, authored-prim enumeration, and refusal records do not | Implement as a CPU gate reusing `swept_volume.py`; do not approximate the robot with a new geometry model. |
| Verdict transfer | Four-cell outcome scoring exists. Native contact records do not identify a scene prim | Require the 2×2 physics pattern and anatomy-compatible contact, then geometrically attribute the external contact to a unique binding primitive or refuse. |
| Workstream A | Duck shelf heights demonstrate variants, but current identities incorrectly make them look independent | Extract one spec from `cf_005_056`; variants support archetype generalization only. `mf_005_c08` is blocked as a synthesis seed by missing executed data. |
| Workstream B | Crouch and arm-tuck editors exist; coverage-gated paired execution and generalized window search do not | Start only after A's CPU and golden gates. Empty-room rollout requests require their own approved manifest. |
| Workstream C | Margins, behaviors, operators, and roles are partly present; axis/binding/source identity are not derivable for every row | Phase 1 first: additive schema repair, honest unknown bins, DCS, and `targets.json`; zero scene authoring or GPU work. |
| E1a repeatability | Not run for LFH; one of the proposed source pairs is incomplete | Redesign the manifest after the source audit. A physics rerun remains prediction-register- and user-approval-gated. |
| E1b golden re-instantiation | No generalized measure-back test or LFH archetype exists | Separate a CPU authoring golden test from the later four-cell physics regression. Do not call E1b green until both pass. |
| E2/E3 | No approved manifest exists | Do not run. Their counts and hypotheses must be revisited after E1 and coverage results. |
| E4 | Existing corpus statistics are not the proposed DCS | Produce a current-index baseline in Phase 1; a before/after comparison waits for accepted new data. |
| E5 learner probe | Explicitly out of scope in the design | Keep as a post-claim-5 design stub. Do not train or claim trajectory optimality. |

## R1: Authoritative Executed States

Executed rollout payloads are sufficient for LFH geometry. The authoritative
state is `body_pos_w`, `body_quat_w`, and `body_names` from the best evaluable
segment, with root path and force data retained for route/contact diagnostics.
`reference_g1_qpos` is useful for provenance but must not replace executed body
poses in window measurement.

`cf_005_056` has distinct nominal (`005`) and adapted (`056`) plane probes.
The initial reconnaissance incorrectly described a nominal plane trajectory for
`mf_005_c08`. An exhaustive Phase-2 match over its exact `w_nominal*.pkl` and
`w_crouch08*.pkl` runtime logs found no `scene=plane` execution: the apparent
modebank pair ran in `cf_005_056_easy`, while all local executions used shelf
scenes. LFH therefore requires both plane probes. Its parent manifest also lacks
an explicit constraint axis, nominal/adapted motion identities, and a contact-
attribution artifact; plane success alone will not license spec extraction.

## R2: Semantic Critical Set

The capsule model has no explicit head link and is not reducible to ten honest
single-center/single-radius keypoints. Feet use seven capsules each and torso
uses four. The v1 semantic layer should therefore aggregate existing capsules:

| Semantic group | Collision owners |
|---|---|
| `head_torso` | `torso_link` (four capsules; no separate head geometry) |
| `shoulder_left/right` | corresponding shoulder-yaw and elbow links |
| `wrist_left/right` | corresponding wrist-yaw link |
| `pelvis` | pelvis link |
| `knee_left/right` | corresponding hip-roll and knee links |
| `foot_left/right` | corresponding ankle-roll link capsules |

Every collision-bearing link must map to exactly one semantic group. Tier 1
then evaluates group extrema and margin, reports a semantic binding label, and
cannot pass geometry that Tier 2 rejects. Ground support needs a separate
support-polygon model; capsule radii must not be treated as foot support radii.

## R3: Edit Operators

`local_crouch` is a coupled hip/knee/ankle edit over local route progress;
`local_arm_tuck` edits shoulder/elbow/wrist joints. Their existing angle caps
and measured-delivery ratios may seed proposals. No code implements a lateral
side-step, narrowed stance, raised step, or changed stride trajectory. Thus:

- overhead v1 may propose `local_crouch`;
- lateral v1 may propose `local_arm_tuck` only when the binding anatomy agrees;
- floor/support scenes remain v2 and require a separately reviewed operator and
  support model.

## Golden Sources and Evidence Accounting

| Causal family | Independent? | Executed evidence | Current disposition |
|---|---:|---|---|
| `cf_005_056` | Yes | Paired plane probes and all four easy/hard cells | Valid overhead source. |
| `mf_005_c08` | Yes | Four physics cells, but no matching nominal/adapted plane pair and incomplete extraction metadata | Valid historical family; blocked as an LFH delivery seed pending both probes and an evidence-preserving metadata repair. |
| `duck_002`, `duck_003` | No | Alternate shelf heights over the same `005`/`056` source | Variants of `cf_005_056`, never two independent families. |

For `mf_005_c08`, measured peak height changes from 1.30515 m to
1.20743 m; the hard shelf is 1.25743 m. Its four recorded cells show nominal
hard contact and the other three accepted. For `cf_005_056`, the measured
delivery window is 0.178125 m and its four cells have the same expected
nominal-hard-only rejection pattern.

The release views currently disagree: `episodes.csv` has 42 rows (34 graded),
`families.csv` labels both duck variants verified, while the JSONL family index
also contains `mf_005_c08`. `motions.csv` has 150 rows, but semantic validity is
not populated and trackability is not joined. These inconsistencies must be
reported and repaired before DCS drives generation.

This count agrees independently with `docs/gate_audit_2026-08-19.md` and the
executed-clearance audit in `docs/constraint_distance_findings.md`. Phase 1 must
repair all three previously recorded defects: `report_dataset_counts.py`
re-admits void `mf_x003_c08` cells because its verdict path omits the route
check; `families.csv` omits `mf_005_c08`; and the progress board reports three
families. The contract acceptance test is exactly **two independent causal
families**. Any other result is a stop condition, not a number to normalize.

Phase-1 reproduction showed that the route repair alone is insufficient:
`mf_x002_c08` and `mf_x003_c08` both intersect the route because the shelf
footprint is broad, while the loaded shelf center is exactly 2.0 m from the
manifest binding station. The implemented identity contract therefore requires
both route intersection and a manifest-to-loaded-station offset of at most
0.5 mm. This evidence-driven adjustment is recorded as D1-001.

## Sealed-Split Resolution

The primary pre-registered evaluation artifact is
`gear_sonic/data/splits/scene_first_v1.json`, fingerprint
`9ce6f16f15eb950ded4a476ff75fd520cacfdc5d49960728d987ad3b0c316d29`.
This identity is corroborated by `docs/dataset_release_layout.md`,
`scripts/research/render_release_page.py`, and
`tests/dataset_generation/test_scene_first_testset.py`. The analysis
pre-registration in `docs/paper/analysis_prereg.md` defines 30 primary scenes
and a separately fingerprinted second 30-scene batch. The latter is
`scene_first_v2.json`, fingerprint
`5811692709022a4f993c90466360837ade385799cbf9633695ed50446c0a5080`,
and may be labelled only if the primary result enters the predeclared ambiguous
zone. `review_cohort_v1.json` is instead the 100-episode human-review cohort.

LFH reads neither V1 nor V2 for coverage, target selection, training, or scene
generation. Both remain immutable; only V1 is the active primary evaluation
split. `docs/g0_real_dataset_runbook.md` concerns an unrelated G0B pick/place
golden path and supplies no competing split identity.

## Scene and Asset Reality

The repository currently contains 157 USD-family files across 11 asset
directories. Generated counterfactual scenes use deterministic axis-aligned
Cube and Plane primitives with collision APIs; there is no mesh-to-convex-hull
approximation in this path. Existing overhead obstacles use a fixed 3.0 m
lateral span and 0.5 m route depth; walls use a fixed 1.2 m route span and
0.12 m thickness. Those constants are historical conventions, not proof that
the non-binding body cannot skirt a new constraint.

Generated USDA files do not author an obstacle physics material. Scene-USD
terrain also has no attached numeric material in its configuration, so the
honest baseline is the runtime/default material, not a claimed friction value.
Variants should omit explicit material exactly as current shelf scenes do and
record this provenance as `runtime_default`/unknown.

Contact attribution adopts `docs/constraint_distance_findings.md` without
retuning. The capsule proxy reaches its practical resolution floor at roughly
10 micrometres and can precede force contact by about 14 frames because its
envelope is conservative relative to the collision mesh. Physics supplies the
verdict; the proxy explains it. Disagreements and timing leads are reported,
never tuned away. Attribution at or below the 10 micrometre floor, or without a
unique binding primitive, is `unattributed` and is a refusal rather than a
judgement call.

## `[VERIFY]` Resolutions

| Design-plan tag | Resolution |
|---|---|
| Exact tool interfaces | `python3 check_environment.py [--training|--deploy|--sim]`; `python3 scripts/research/batch_preflight.py --plans … --rollouts … --out …`; `python3 scripts/research/score_family_batch.py ROOT --json …`; `python3 scripts/research/render_corpus_stats.py --index … --out …`; `python3 scripts/research/check_g1_dataset_scenes.py [PACKAGE_DIR] --json …`. Help output remains the authority when a stage is implemented. |
| Obstacle face convention | Existing fixed dimensions are listed above. LFH must derive each face extent from the union of nominal and adapted swept geometry over the critical interval plus a recorded margin. |
| Collision approximation | Generated LFH-compatible assets use exact authored Cube/Plane primitives. Mesh collision approximation does not apply unless a later design delta admits mesh assets. |
| Baseline shelf material values | No explicit values are authored. Preserve the same omission/runtime default and do not invent friction or restitution metadata. |
| 50 mm keepout convention | Fifty millimetres is used as an easy-scene clearance convention, not as an implemented non-binding keepout proof. Make it a configurable v1 default and verify `C \\ B` geometrically. |
| Clutter availability | A 17-type primitive furniture catalog and generated clutter scenes exist. Optional clutter is eligible only through reject-sampling against the same keepout, route, and single-cause checks. |
| Exact placement rule | Existing paired code uses executed collision boundaries and a midpoint hard parameter; newer graded planning uses easy = nominal boundary + 50 mm. LFH should solve from paired executed bounds, then measure the emitted world AABB back to within 0.5 mm. |
| CI integration | No repository CI currently runs `tests/dataset_generation`; only documentation CI is present. Add focused local tests first. A new CI workflow is a separate scope decision, not an implicit Phase-2 edit. |

## Phase 1 Recommendation

Phase 1 should remain CPU-only and additive:

1. Define one stable causal `source_family_id`, a separate `variant_id`,
   `constraint_axis`, constraint coordinate, and semantic binding group.
2. Populate them deterministically in both nested release and flat index views,
   retaining unknown values instead of inferring them from directory names.
3. Join semantic/embodiment gate records with accepted empty-room rollout
   records to expose controller trackability.
4. Compute DCS with unknown-bin reporting and separate counts for independent
   families, variants, motions, operators, axes, and binding groups.
5. Emit a ranked CPU-only candidate report; do not author scenes or schedule
   simulation in this phase.

## Review Gate

Before Phase 1, review and accept the deltas in `DESIGN_DELTAS.md`, especially
semantic capsule groups, stable family identity, v1 operator scope, material
provenance, and the missing `mf_005_c08` plane rollout. GPU work remains gated
by a written manifest, estimated budget, stop conditions, and explicit user
approval.
