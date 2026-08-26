# SweepCF-Hallucination: Constraint-Conditioned Scene Synthesis

**Design plan v0.1 — 2026-08-20**
Target repo: `groot-wbc-sonic-sim-trackb` (SweepCF corpus pipeline; Unitree G1, frozen SONIC, Isaac Lab 2.3.2)

---

## 0. Summary

SweepCF currently measures, per counterfactual family, the exact geometric window in which one motion strikes and its single-edit twin clears, then instantiates that window as one obstacle (a shelf) in two rooms. This plan adapts the core factorization of **LfH-CP** (Learning from Hallucinating Critical Points, Ghani et al.) to SweepCF: *measure the critical constraint once, then procedurally generate many diverse realizations around it, and let physics verify each one.*

Three workstreams:

- **A — Scene multiplication.** From each verified family's measured window, generate K visually distinct obstacle instantiations (beam, duct, lintel, hanging panel, …) whose binding face sits at exactly the same coordinate, plus non-binding clutter outside a keep-out corridor. Every variant inherits the family's counterfactual cause and is re-verified by rollout. This is the ammunition for claim 6 (held-out generalization) and de-risks claim 5's "did the learner key on the shelf or on the geometry" question.
- **B — Inverse synthesis for tail behaviors.** For under-represented behavior classes (arms: 6/150, crouch: 9/150, side/narrow: 3/150), construct scenes that *force* the behavior: pick a base motion + existing operator, compute both executed swept volumes, search for the face coordinate that separates them by a window, instantiate via A, verify by rollout. Behavior diversity then comes from (edit × hallucinated scene), not generator sampling luck.
- **C — SweepCF-DCS coverage metric.** A joint coverage score over (binding link group × constraint axis/coordinate × margin bucket × behaviour class × operator × cell role), computed purely from the release index, LfH-CP Tab.-II style. Empty bins are emitted as machine-readable generation targets that feed Workstream B. Closes the loop: gaps → targeted synthesis → verified families → re-measured coverage.

The conceptual mapping, for orientation:

| LfH-CP | SweepCF equivalent |
|---|---|
| plan `p` collected in open space | executed motion from empty-room rollout |
| robot footprint (2D cylinder) | 29 capsules over 14 links, swept per frame |
| critical point `(x, y, t_crit)` | binding link + `clearance_frame` + measured window |
| fixed classical decoder verifies optimality | frozen SONIC + Isaac physics verifies the 2×2 verdict |
| hallucinate `C_obst` s.t. `p` optimal | place geometry s.t. orig strikes ∧ edit clears (hard) / both clear (easy) |
| ≥1% loss-reduction contribution filter | preflight refusal + single-cause keep-out validator |
| DCS coverage metric | SweepCF-DCS (Workstream C) |

Two deliberate departures from LfH: (1) supervision here is **edit necessity/sufficiency**, not trajectory optimality — use that vocabulary everywhere; (2) LfH's labels are correct by construction, ours are not (the controller delivers 30–70% of a commanded edit), so **geometry only proposes; physics disposes**.

---

## 1. Invariants — non-negotiable

These hold regardless of what reconnaissance finds. Violating any of them invalidates the work.

1. **Physics is the only verdict source.** No episode ships with a geometry-derived outcome label. Hallucinated/variant scenes get `outcome`, `min_clearance_mm`, contact columns, etc. exclusively from rollouts through the existing scoring path.
2. **Frozen SONIC stays frozen.** No fine-tuning, no controller changes, no changes to the token stream the deployment runtime consumes.
3. **The 30 pre-registered evaluation scenes are read-only.** Generated scenes never enter, replace, or modify them. New scenes live in clearly separated packages.
4. **The Aug 24 pre-registered claim-5 run is not disturbed.** Default policy: variant/hallucinated families are **excluded** from the claim-5 dataset and target claim 6 / corpus growth. Inclusion requires an explicit, documented opt-in from the user consistent with the pre-registration text ("whatever families exist then").
5. **Single-cause discipline.** In every generated scene, the only geometry permitted inside the keep-out corridor (Section 3.3) is the intended binding face. Any rollout contact with non-binding geometry is an automatic refusal (`secondary_contact`), never a shipped episode.
6. **Executed, not commanded.** All reach profiles, sweeps, and window searches use executed empty-room rollouts, never the reference clip and never the commanded edit amplitude.
7. **Semantic gate first.** Hallucinate only around motions that pass `embodiment_feasible ∧ controller_trackable ∧ semantically_valid` (the last is 32% — this filter is not optional).
8. **Millimetre discipline.** Authored binding-face world coordinate must match the spec to ≤0.5 mm, verified by measuring back from the composed USD stage, not by trusting authoring parameters.
9. **Determinism & provenance.** Every generated scene is reproducible from (spec JSON + archetype id + seed). Scenes ship beside episodes as USD, per existing convention. Every figure/table regenerates from the release index.
10. **Honest accounting.** Refusals are results. Every batch reports its full funnel (proposed → window found → preflight pass → rolled out → verified). Hypotheses for E2/E3 are filed in the prediction register *before* running; misses are reported, not buried.
11. **Additive schema only.** New index columns are appended with documented semantics; the existing 35 columns and the three CSV grains keep their meaning. Existing consumers must not break.

---

## 2. What exists (assumed baseline — verify all of it in recon)

From the project page; internal names/paths are `[VERIFY]`:

- Pipeline stages: empty-room rollout → capsule sweep (29 capsules / 14 links, per frame) → per-height-band criticality map (which link binds) → apply edit, sweep again → window = band where orig strikes and edit clears → place easy face (50 mm clear of orig's reach) and hard face (window center) → verify against executed body → rollout → 35-column episode row.
- Measured facts usable as golden-test targets: family `duck_003` — shelf underside 1.3906 m (easy) / 1.2125 m (hard); clearances: easy/orig +89.7 mm, easy/edit +193.4 mm, hard/orig −0.0 mm (strike), hard/edit +55.6 mm. Windows across bands run 21–105 mm. 12/12 cells survived start-pose jitter (claim 3). 14 configurations refused by preflight in one batch. Historic scalar survival columns remain diagnostics only; CAL1/CAL2 show they cannot bound per-keypoint placement.
- Corpus: 96 rollouts, 150 screened motions, 2 verified families, 117 scenes / 11 packages, behavior histogram turn 67 / walk 42 / stop-start 23 / crouch 9 / arms 6 / side-narrow 3.
- Tooling: `scripts/research/score_family_batch.py`, `scripts/research/batch_preflight.py`, `check_environment.py`, `render_corpus_stats.py` `[VERIFY exact interfaces]`.
- Cameras: fixed observer per room, plus ego/chase/wrist; overhead camera renders blank (known broken) — never schedule it.
- Index columns (grouped): identity (`episode_id`, `family_id`, `cell_role`, `motion_role`), behaviour (`operator`, `behaviour_class`), scene (`scene_id`, `scene_path`, `obstacle_underside_m`, `route_reaches_obstacle`), verdict (`outcome`, `rejection_reasons`), clearance (`min_clearance_mm`, `clearance_frame`, `closest_body`), contact (directional forces, `contact_body`, `contact_frame`), fidelity (`drift_rate_mps`, commanded/executed amplitude, `operator_survival`), media (five video columns, `n_videos`).

Anything in this plan that contradicts the actual repo: reality wins, and the delta is documented (see prompt, Phase 0).

---

## 3. Core abstractions

### 3.1 ConstraintSpec (JSON)

The unit of hallucination. Extracted from a verified family (Workstream A) or produced by window search (Workstream B).

```json
{
  "spec_version": "0.1",
  "spec_id": "cs_duck_003",
  "source": {"kind": "verified_family", "family_id": "duck_003"},
  "motions": {
    "orig": {"motion_id": "...", "empty_room_rollout": "<artifact ref>"},
    "edit": {"motion_id": "...", "operator": "crouch", "empty_room_rollout": "<artifact ref>"}
  },
  "crossing": {"frame_range": [f0, f1]},
  "binding": {
    "axis_type": "overhead",
    "band_m": [lo, hi],
    "expected_link_group": "head_torso",
    "reach_orig": "<from stored family measurement>",
    "reach_edit": "<from stored family measurement>",
    "window_mm": "<from stored family measurement>"
  },
  "faces": {"easy": "<coordinate>", "hard": "<coordinate>"},
  "face_extent": {"lateral_m": w, "along_route_m": d},
  "keepout": {"delta_mm": 50, "exempt_prims": ["floor", "room_shell"]},
  "camera": {"observer_pose_ref": "<source room camera>"}
}
```

Rules:
- `axis_type` v1 supports exactly two values: `overhead` (scalar coordinate = underside height; binding face is a downward-facing horizontal plane) and `lateral_gap` (scalar coordinate = gap width at a band; realized as two vertical faces, one fixed one placed, or symmetric). Curved undersides, slots with 2-D coordinates, and dynamic obstacles are explicitly **out of scope** for v1.
- For A-specs, `faces`/`reach`/`window` are copied from the family's stored measurements — do **not** re-derive them; the shipped numbers are the ground truth the golden tests assert.
- `face_extent` must be wide enough that the route cannot skirt the face (`route_reaches_obstacle` must hold); default: cover the lateral extent of both swept volumes at the crossing plus margin `[VERIFY convention]`.

### 3.2 Archetype library

An archetype = a parametric, seeded USD generator that (a) declares a named **binding-face handle** (the prim + face whose world coordinate realizes the spec), and (b) declares a full manifest of every other prim it authors (so the validator can check each against the corridor).

v1 set — overhead: `shelf_plank` (baseline, must reproduce existing scenes), `ibeam`, `hvac_duct`, `door_lintel` (side jambs allowed **only if** they pass keep-out; otherwise auto-degrade to lintel-only), `hanging_panel` (rigid, fixed — no pendulum). Lateral-gap: `pinch_panels`, `doorway`, `rack_aisle`.

Constraints: static colliders only (kinematic, no articulations, no deformables); collision approximation consistent with existing scenes `[VERIFY: convex hull vs mesh]`; visual variation (materials, textures, non-binding dimensions, small decorative offsets) is seeded and free **only along non-binding axes**; physics materials (friction/restitution) match the baseline shelf so contact-force readings stay comparable `[VERIFY values]`.

### 3.3 Keep-out validator — the single-cause guarantee

Definitions, per spec:
- `S_orig`, `S_edit`: executed swept volumes at the crossing (reuse the existing capsule-sweep module; do not reimplement).
- Corridor `C = dilate(S_orig ∪ S_edit, δ)`, δ = `keepout.delta_mm` (default 50 mm, matching the easy-face clearance convention `[VERIFY]`).
- Binding region `B` = the intended face volume ⊕ placement tolerance.

**Rule:** for every non-exempt prim in the generated scene, `prim ∩ (C \ B) = ∅`. Exempt: floor/room shell inherited unchanged from the source room (feet must touch the floor; the shell is part of the family's controlled geometry, not ours to move).

Implementation sketch: sample or SDF the prim surfaces; distance-query against the frame-wise capsule set (≈ frames × 29 capsules; vectorized CPU is fine, Warp if convenient). Also verify: realized binding-face world coordinate within 0.5 mm of spec (measure back from composed stage, correct up-axis/units/xform stack); face extent covers the route; nothing added below foot-clearance height on the route.

Outputs per scene: pass/fail, `binding_face_offset_mm`, `keepout_min_clearance_mm` (min distance of any non-binding prim to the corridor), violation report.

### 3.4 Verdict-transfer criteria (what "the variant worked" means)

A variant scene set (easy+hard) is **verified** iff rollouts reproduce the source pattern:
- hard/orig → `strike`, with `contact_body` in the expected link group and contact on the binding prim (contact with any other prim ⇒ refuse `secondary_contact`);
- hard/edit → `clear`; easy/orig → `clear`; easy/edit → `clear`.

Clearance drift vs the source family is **reported, not gated** (the gate is the outcome pattern + contact identity), judged against the physics-repeatability noise floor measured in E1a. New `rejection_reasons` values: `keepout_violation`, `binding_face_offset`, `secondary_contact`, `verdict_flip`, `window_below_min`.

---

## 4. Workstream A — scene multiplication from verified families

**Input:** ConstraintSpec extracted from each verified family. **Output:** K verified variant scene-pairs per family, each a 4-cell family sharing the source's counterfactual cause but a different visual/physical realization.

Steps:
1. `extract_spec` — read the family's stored measurements + empty-room rollouts, emit spec JSON. Golden check: re-instantiating `shelf_plank` from `cs_duck_003` must reproduce the shipped scene geometry (face coordinates exact) and, on rollout, the shipped 2×2 outcomes.
2. `instantiate` — spec × archetype × seed → easy USD + hard USD, authored into a new scene package (e.g. `packages/hallucinated_variants_v1/`), never touching source or frozen-eval packages. Optional clutter pass: place clutter-package assets `[VERIFY availability]` rejected-sampled against the corridor.
3. `validate_keepout` — Section 3.3; refusals logged with reasons before any GPU spend.
4. Preflight via the existing `batch_preflight` pattern; manifest of surviving cells presented for **user approval** (GPU gate).
5. Rollout + scoring via the existing family scorer; verdict-transfer evaluation (3.4); index rows appended with new columns (Section 7).
6. Render ego/chase strips for accepted variants (skip overhead camera) for the visual-diversity check in E2.

Index additions (episodes grain, additive): `spec_id`, `source_family_id`, `variant_id`, `obstacle_archetype`, `visual_seed`, `binding_face_commanded_m`, `binding_face_realized_m`, `binding_face_offset_mm`, `keepout_min_clearance_mm`. Families grain: `spec_id`, `n_variants`, `variant_pattern_rate`.

---

## 5. Workstream B — inverse synthesis for tail behaviors

Runs only after E1–E2 validate the machinery.

1. **Operator inventory (recon output).** Confirmed to exist: arm-tuck and crouch. Their delivery is measured per motion/keypoint from exact executions; historical scalar survival ranges do not certify placement. Side-step/narrow-stance operator existence is unknown — if absent, side/narrow is **v2** and requires a user decision before any new operator work.
2. **v1 targets:** (a) *arms* — `lateral_gap` at chest/shoulder band forcing tuck; tuck's high survival makes it the best-behaved first target; (b) *crouch* — new `overhead` families at bands/heights/margin-buckets the coverage tool flags empty, exploiting crouch's known fidelity columns.
3. **Candidate selection:** base motions from the gated pool (Invariant 7) with a clean straight crossing segment; pair with operator at 2–3 amplitudes.
4. **Window search (cheap, no GPU):** roll both motions in the empty room (or reuse cached rollouts), compute executed reach profiles along the constraint axis within the band at the crossing, sweep the face coordinate, find the interval where orig strikes ∧ edit clears. Keep candidates with window ≥ `w_min` (default 20 mm — just below the narrowest verified window, 21 mm, and defensible against claim-3 jitter; configurable). Expect heavy attrition; that is signal, not failure — LfH-CP's contribution filter plays exactly this role.
5. **Face placement per existing convention** (easy = orig reach + 50 mm; hard = window center) `[VERIFY exact rule in code]`, then hand off to Workstream A steps 2–6.
6. **Funnel report per behavior:** candidates → windows found → instantiated → preflight pass → rolled out → verified families. Defaults: ≥12 candidates per behavior, top 4 instantiated.

---

## 6. Workstream C — SweepCF-DCS coverage metric

Pure reporting over the release index; zero risk; can land first.

Dimensions (all derivable from existing columns; bucket edges in one config file):
- `binding_link_group`: map 14 links → {head, torso, shoulder_upper_arm, forearm_wrist, pelvis_thigh, shank_foot} from `closest_body`;
- `constraint_axis` × coordinate bucket: overhead via `obstacle_underside_m` (e.g. 0.1 m bins over the traversal-relevant range); `lateral_gap` bucketed by gap width once such families exist;
- `margin_bucket` from signed `min_clearance_mm`: {strike ≤0, (0,10], (10,25], (25,50], (50,100], >100};
- `behaviour_class`, `operator`, `cell_role`.

Outputs: marginal coverage per dimension; selected joint tables; a coverage-vs-corpus-size curve;
and `targets.json`. SweepCF-DCS v2 uses only canonical cells of exact verified variants for causal
occupancy and masks impossible operator-keypoint cells. Workstream B first computes
trajectory-feasible support (including route phase and finite extent), then uses DCS as a novelty
term. CLI `render_coverage.py --episodes <index> --out coverage/` is deterministic.

---

## 7. Experiments & validation

File prediction-register entries for E2/E3 (with the agreed numeric thresholds) **before** running them.

**E1a — physics repeatability noise floor.** Re-roll the two source families' 8 cells with fresh seeds (default 2 extra repeats each ⇒ 16 rollouts). Report per-cell clearance spread and outcome stability. This defines the tolerance every later "drift" is judged against. If any *outcome* flips here, stop and escalate — that is a finding about the corpus itself.

**E1b — golden re-instantiation.** Regenerate both families via `extract_spec` + `instantiate(shelf_plank)`. Acceptance: face coordinates match shipped values to ≤0.5 mm; rollout outcomes match the shipped 2×2; clearances within the E1a noise floor. This is the permanent regression test for the whole machinery.

**E2 — variant transfer.** Default K=3 archetypes × 2 families × 4 cells = 24 rollouts (user-approved manifest). Metrics: pattern-reproduction rate (per variant, per cell), clearance-drift distribution vs noise floor, `secondary_contact` rate, `binding_face_offset_mm` distribution, and an ego-view contact sheet demonstrating visual diversity at fixed cause. Suggested pre-registered hypothesis (finalize with user): ≥2/3 of keep-out-passing variants reproduce the full 2×2 pattern; failures are explained by `secondary_contact` or offset, not silent.

**E3 — tail-behavior funnel.** For arms and crouch: ≥12 candidates each through the Section 5 funnel; 4 instantiated each ⇒ ≤32 rollouts. Suggested hypothesis: ≥1 newly verified family per targeted behavior. Report the funnel even (especially) where it dies.

**E4 — coverage before/after.** SweepCF-DCS tables on the current index; re-render after E2/E3; coverage-vs-samples curve; updated `targets.json`.

**E5 — controlled distribution-value probe (design stub).** At four independent sources × three
**common** verified archetypes forming a complete crossed matrix, compare equal-budget
critical-support, uniform-feasible, DCS-ranked, and visual-only sampling on held-out source ×
archetype splits. Write the design before the readiness gate; run nothing early.

**E6 — independent-source critical-window pilot.** Reopen the three accepted CAL3 pairs under DCS
v2. Instantiate one canonical shelf family per source (12 cells maximum), with the E1a empirical
engineering margin applied symmetrically. Primary prediction: at least 2/3 complete 2×2 patterns.
E6 confirmed exactly 2/3 in 10 rollouts. The separately registered E6c ablation recovered source
086 by reducing finite route exposure from 0.30 m to 0.10 m, without changing E6's denominator.

**E7 — archetype transfer for new sources.** Transfer `door_lintel` and `ibeam` to 086, 089, and
090 while preserving each verified station, exposure, and coordinate (24 cells). Primary:
at least 4/6 complete source-archetype patterns and at least one transfer per source. E7 confirmed
5/6; E7c replaced the refused source-086 lintel with a verified hanging panel. This satisfies the
count-only gate, but only two archetypes are common across all four sources.

**E8 — crossed-support completion.** Transfer `hanging_panel` to `cf_005_056`, 089, and 090 while
holding each source's trajectory-conditioned atom fixed. Stage the six easy cells, recertify, then
run six hard cells only for survivors. E5 remains closed until the resulting matrix has at least
three common verified archetypes.

**E9 — generated-motion transaction.** Sample a new Kimodo motion, screen it, apply one deterministic
operator, and execute both references in empty-scene physics before computing support. E9a's curved
walking nominal accepted, but its 80 mm crouch twin rejected on path/endpoint tracking with zero
external contact. No obstacle was authored. Next calibration work must improve motion/adaptation
delivery before another generated source can enter `P_feas`.

**E10 — multi-obstacle context composition.** Around one verified hanging-panel atom, add four
route-relative context obstacles spanning left, right, floor-level, and high-overhead placements.
E10 changed seed and context together and was non-identifying. E10b restored the source seed and
verified the full 2×2 pattern: context clearance 288.78 mm, unique binding contact before drift, and
zero context contact. This validates one-binding-plus-context composition, not non-overhead critical
support or population-level context invariance.

Default GPU budget across E1–E3: ≈72 rollouts (16 + 24 + 32), each batch behind an approval gate. All numbers adjustable by the user before the corresponding prediction-register entry is filed.

---

## 8. Sequencing and the Aug 24 boundary

Today is Aug 20; claim 5 runs Aug 24 on a pre-registered analysis; the 100-episode human review is submission-blocking. Therefore:

1. **Now / zero-risk:** Workstream C; Phase-0 recon; spec extraction; archetype + validator code with unit tests. None of this touches rollouts, eval scenes, or the register.
2. **This week, gated:** E1a/E1b (small, and E1a doubles as a corpus-repeatability check worth having before claim 5 regardless).
3. **Default after Aug 24:** E2, E3, E4 re-render. If the user explicitly opts variants into claim 5 (Invariant 4), only E1b-verified, fully re-rolled families qualify, and the opt-in is documented in the register.
4. **Roadmap only:** E5; dynamic/temporal obstacles (`clearance_frame` as t_crit — a closing door, a crossing agent — the LfH-CP phase-2 analogue); curved binding faces; multi-binding rooms with an LfH-CP-style incremental-contribution filter (keep only obstacles that flip or tighten a verdict). One-binding multi-obstacle context composition is verified by E10b.

---

## 9. Engineering plan

Proposed layout (conform to conventions found in recon; do not fight the repo):

```
scripts/research/hallucination/
  constraint_spec.py      # schema, load/save, validation
  extract_spec.py         # verified family -> spec
  archetypes/             # base.py + one module per archetype
  instantiate.py          # spec x archetype x seed -> USD pair
  validate_keepout.py     # Section 3.3
  search_windows.py       # Workstream B step 4
  render_coverage.py      # Workstream C
docs/hallucination/
  RECON.md  DESIGN_DELTAS.md  REPORT_E1.md  REPORT_E2.md  REPORT_E3.md  COVERAGE.md
specs/            scenes/packages/hallucinated_variants_v1/
```

Reuse, never reimplement: capsule-sweep module, preflight, family scorer, index writers, USD scene conventions of the existing 117 scenes. Tests: mm-level unit asserts on placement math (incl. up-axis/units/xform composition); property tests on the validator (inject a violating prim ⇒ must refuse); E1b as the golden integration test wired into CI if one exists `[VERIFY]`. Docs: every new column defined where the existing 35 are defined; schema version note.

---

## 10. Risks & mitigations

- **Secondary contact from decorative geometry** → strict keep-out + auto-refusal + report; lintel jambs auto-degrade.
- **Physics nondeterminism blurs mm margins** → E1a noise floor first; gate on outcomes + contact identity, report drift as distributions.
- **USD authoring pitfalls (units, up-axis, nested xforms, collision approximation)** → measure-back verification (Invariant 8) is the defense; never trust authoring parameters.
- **Windows too narrow vs jitter** → `w_min` filter; sub-threshold candidates refused as `window_below_min`.
- **Archetype diversity too tame to matter downstream** → seeded material/texture/non-binding-dimension jitter; E2 ego contact sheet as the check; E5 as the real test later.
- **Side-step operator missing** → scoped to v2 behind a user decision; do not silently invent operators.
- **Index-consumer breakage** → additive columns only, schema note, run existing render scripts as a smoke test.
- **Timeline collision with Aug 24** → sequencing in Section 8; nothing GPU-heavy or register-touching without explicit approval.

---

## 11. Definition of done

- [x] `RECON.md` mapping every `[VERIFY]` to reality; `DESIGN_DELTAS.md` for every deviation from this plan (invariants excepted — those cannot be deviated from).
- [x] Workstream C shipped: coverage tables + curve + `targets.json`, regenerable from the index alone.
- [x] Spec extraction, ≥5 overhead + ≥2 lateral archetypes, keep-out validator — all unit-tested; E1b golden test green.
- [x] E1a noise-floor report.
- [x] E2 executed behind approval gate; report with funnel, drift distributions, ego contact sheet; prediction-register entry filed beforehand and reconciled afterward.
- [x] E3 executed for arms + crouch; funnel report; any new verified families indexed with full provenance columns.
- [x] E4 before/after coverage; E5 design stub written.
- [x] All refusals carry machine-readable reasons; all figures regenerate from the index; no invariant violated.
- [x] Final audit shipped SweepCF-DCS v2 and restored the CAL3 critical-support census.
- [x] E6 confirmed its ≥2/3 primary; E6c verified the exposure-conditioned 086 source separately.
- [x] E7/E7c reach four sources with three verified archetypes each (count-only gate).
- [x] E9a enforces generate → adapt → empty-scene physics and refuses an invalid fresh pair.
- [x] E10b verifies one binding obstacle plus four non-causal route-relative context obstacles.
- [ ] E8 completes four sources × three common archetypes; only then may E5 run.
- [ ] A fresh generated motion/adaptation pair passes both empty-scene physics cells.
- [ ] A lateral or oblique operator produces executed critical support and a verified family.
