# Prompt: implement constraint-conditioned scene synthesis in SweepCF

You are working inside the SweepCF repo (`groot-wbc-sonic-sim-trackb`): a counterfactual humanoid-traversal corpus on a Unitree G1 with a frozen SONIC controller in Isaac Lab 2.3.2, where every episode has a twin and physics — not an annotator — decides which motion survives.

**Mission.** Implement, experimentally validate, and document the design in `docs/plans/sweepcf_hallucination_design_plan.md` (read it in full before anything else; if the file is missing, stop and ask for it). In one line: adapt LfH-CP's factorization — *measure the critical constraint, then procedurally generate diverse realizations around it, physics-verified* — to this corpus, via (A) scene multiplication from verified families, (B) inverse scene synthesis targeting tail behaviors, (C) a coverage metric that closes the loop.

The bar is: a reviewer of this dataset paper should look at your output and find it *more* rigorous than the existing pipeline, not less. This corpus's credibility rests on measurement discipline and honest refusals; your work must extend that culture, not dilute it.

## Ground rules (from the plan's Invariants — never trade these away)

1. Physics rollouts are the only source of verdict labels. Your geometry proposes; the existing scorer disposes.
2. Frozen SONIC and its runtime token stream are untouchable.
3. The 30 pre-registered evaluation scenes are read-only. Generated scenes live in new, clearly separated packages.
4. **Today is Aug 20, 2026; claim 5 runs Aug 24 on a pre-registered analysis.** Default: your generated families are excluded from claim 5. Do nothing that touches the pre-registered analysis, its inputs, or the prediction register without an explicit user decision.
5. Single cause per scene: only the binding face may enter the keep-out corridor; any rollout contact with other geometry ⇒ refusal, never a shipped episode.
6. Use executed empty-room rollouts everywhere (never reference clips, never commanded amplitudes), and only motions passing the semantic gate.
7. Binding-face placement verified to ≤0.5 mm by measuring back from the composed USD stage.
8. Determinism: (spec + archetype + seed) reproduces the scene. Additive index columns only. Every figure regenerates from the release index.
9. Refusals are results: report full funnels; file prediction-register entries before E2/E3, and reconcile after.

## Working protocol — phases with hard gates

Work phase by phase. **Stop at the end of each phase, present the named artifact, and wait for user review before continuing.** Never spend GPU rollouts without presenting a preflight manifest and receiving explicit approval.

**Phase 0 — Reconnaissance (no code changes).**
Read the repo: the capsule-sweep implementation, window/placement measurement code, USD scene authoring conventions across the 117 scenes / 11 packages, `batch_preflight.py`, `score_family_batch.py`, `render_corpus_stats.py`, the index writers and the exact 35-column schema, the two verified families' stored measurements, and the operator implementations (confirm: crouch, arm-tuck; determine whether any side-step/narrow operator exists).
Deliver `docs/hallucination/RECON.md`: a table mapping every plan concept and every `[VERIFY]` tag to actual file paths / functions / conventions, plus a list of places where the plan's assumptions are wrong and your proposed adjustment. Record all deviations you adopt, now and later, in `docs/hallucination/DESIGN_DELTAS.md`. The plan is a strong prior; the repo is reality; the Invariants outrank both. **Gate: user reviews RECON.md.**

**Phase 1 — Coverage tool (Workstream C).** Zero-risk, index-only. Deliver `render_coverage.py`, marginal + joint coverage tables, the coverage-vs-corpus-size curve, and `targets.json` of empty bins. Run the existing render scripts afterward as a smoke test that nothing broke. **Gate: user reviews COVERAGE.md.**

**Phase 2 — Core machinery + golden reproduction.** Implement ConstraintSpec, `extract_spec`, the archetype library (≥5 overhead, ≥2 lateral-gap), `instantiate`, and the keep-out validator — with unit tests (mm-level placement asserts incl. up-axis/units/xform composition; property test: an injected corridor-violating prim must be refused). Then E1: (a) present the E1a repeatability manifest (≈16 rollouts) for approval, run it, report the per-cell noise floor; (b) E1b golden re-instantiation of both verified families via `shelf_plank` — face coordinates must match shipped values (duck_003: 1.3906 m / 1.2125 m) to ≤0.5 mm and rollouts must reproduce the shipped 2×2 (+89.7 / +193.4 / −0.0 / +55.6 mm within the E1a noise floor). Wire E1b in as the permanent regression test. **Gate: REPORT_E1.md. If any source-family outcome flips in E1a, stop everything and escalate — that is a corpus-level finding.**

**2026-08-20 continuation:** the accepted review was followed by the required LFH finite-face recomputation. It found and repaired D2-005 before physics: the old extended plank reversed adapted-hard capsule clearance. `REPORT_LFH_KCS.md` is the renewed CPU gate; its V2 E1b/probe manifests are not authorized and Phase 3 remains unopened.

**Phase 3 — Variant transfer (Workstream A, E2).** Draft the prediction-register entry (thresholds per the plan; user finalizes numbers), then the preflight manifest (default 3 archetypes × 2 families × 4 cells = 24 rollouts) for approval. Run, score through the existing path, deliver REPORT_E2.md: pattern-reproduction rate, clearance drift vs noise floor, secondary-contact and face-offset distributions, ego-view contact sheet (never schedule the broken overhead camera), and index rows with the new provenance columns. **Gate: user review.**

**2026-08-20 autonomous continuation:** `GOVERNANCE.md` superseded the earlier authorization pause
without changing the scientific invariants. E1a established an 18.044 mm source-inclusive floor;
E1b passed 4/4; empty-room probes refused `mf_005_c08` and both under-delivered arm-tuck scene
proposals. E2 therefore declared its one-source denominator before spend (D2-008). It reproduced
12/12 outcomes and verified 2/3 variants exactly at threshold: door-lintel and I-beam passed, while
HVAC was refused for pre-contact drift. `REPORT_E2.md` contains the full funnel and ego evidence.
Phase 4 must begin with stronger, multi-level existing-operator calibration per D2-007; the current
arm scenes must not be instantiated from the disproved scalar delivery prior.

**2026-08-20 Phase-4 calibration:** the two-motion/three-level empty-room cohort completed 12/12
accepted captures and fit four supported per-keypoint D_phi curves. Its conservative 60 mm wrist
delivery bounds are 1.10 mm left and 4.76 mm right after matched-level pooling, so D2-009 keeps E3 lateral scene synthesis
refused. This is a useful negative result: motion trackability is not enough, and LFH must not
author geometry against commanded edits that the executed humanoid does not deliver.
The follow-on CPU-only strength screen kept the operator fixed and passed 20/20 reference gates;
it found candidate windows up to 117.06 mm left and 100.15 mm right. Those are the bounded next
calibration targets, not delivery claims (`REPORT_ARM_TUCK_STRENGTH_SCREEN.md`).

**2026-08-20 Phase-4 strong calibration and lateral CPU gate:** CAL2 produced three accepted
strong edits, one rejected nominal, and one dependency skip at 0.030 GPU-h. The matched right-wrist
lower bound rose to 23.84 mm, falsifying that side of the below-20 prediction; strong-left remains
unidentifiable. Review then repaired the lateral coordinate contract: gap coordinates are full
world-fixed face separation, not root-relative half-width, and `screen_empty` evidence now needs a
hash-pinned zero-external-contact adjudication. The fixed 24-trial E3 CPU funnel retained 1 candidate:
`084/right`, target rank 48, with a 165.80 mm raw and 56.41 mm noise-certified window. `092/right`
is refused by temporal/anatomy gates. `REPORT_E3_LATERAL_CPU.md` and the four-cell
`E3_LATERAL_PILOT_PROPOSED.json` are the next physics gate; they do not yet establish a family.

**2026-08-20 E3 lateral pilot:** The registered four-cell pattern was falsified at 3/4: both easy
cells and nominal-hard matched, while adapted-hard contacted the binding panel before tracking
drift and rejected. The target is refused and remains empty. Recomputing exact full-gap reach from
the two accepted scene-conditioned easy trajectories rules out a same-bin retry: adapted reach is
0.906489 m before the 36.088 mm uncertainty allowance, already beyond the `0.8_0.9` upper edge.
D2-015 therefore promotes accepted, zero-contact context trajectories into the retry gate and
forbids clearance certification from collision-contaminated paths. No further lateral physics is
justified for this target; `REPORT_E3_LATERAL_PILOT.md` preserves the 24 -> 1 -> 4 -> 0 funnel.

**2026-08-20 E3 crouch branch:** A fixed 15-motion x four-strength CPU screen passed 60/60 strict
reference gates and selected four route-straight, uncapped 80 mm edits. CAL3 then accepted all four
nominals and three adapted motions at 0.063 GPU-h. Exact finite overhead faces produced 36 trials
and a 72.26 mm maximum raw head/torso window, confirming the registered calibration prediction.
The initial v1 analysis reported zero empty-target intersections. Final review found that probe
rows had filled a causal target and that the denominator included impossible crouch/shoulder cells.
SweepCF-DCS v2 restores 20 CPU-valid proposals across all three accepted pairs. No family is yet
verified; D2-017 records the correction and `critical_support_cal3.json` applies the E1a empirical
engineering margin symmetrically.

**Phase-4 closeout / final-review gate:** `REPORT_E3.md` consolidates both behavior funnels: 84
CPU trials/settings, five selections, 12 serial rollouts, and zero verified families at 0.102
GPU-h. E4 adds the two E2-verified archetypes to an isolated index, increasing verified variants
from 3 to 5 while independent families remain 2. Under corrected v2 accounting, verified-family
occupancy changes from 2/120 to 3/120 because the new variants promote a previously probe-only
margin bin; this deepens one source without adding source independence. `REPORT_E4.md` is fully
regenerated. `E5_LEARNER_PROBE_DESIGN.md` now compares critical, uniform-feasible, DCS-ranked, and
visual-only sampling and remains blocked until four sources × three common archetypes form a
complete cross.

**2026-08-20 final-review continuation:** `critical-distribution.md` defines trajectory-conditioned
support and separates unknown natural `P_env`, deterministic feasible `P_feas`, and designed
`q_LFH`. This review registered E6 as three CAL3 sources × one canonical shelf × four cells, with
a primary ≥2/3 pattern threshold and a 12-rollout maximum. E7 was conditioned on enough source
evidence to make the E5 readiness gate attainable.

**2026-08-20 E6/E6c closeout:** E6 confirmed its primary at exactly 2/3 sources in 10 rollouts and
0.148 contended GPU-h. Sources 089 and 090 have complete, uniquely attributed shelf patterns;
source 086 failed adapted-easy trackability at 0.30 m exposure. The registered E6c paired ablation
held all other design variables fixed, shortened exposure to 0.10 m, and verified all four cells in
0.056 GPU-h. Four sources are now extractable, but E5 remains blocked until E7 transfers two more
archetypes to each new source. `REPORT_E6.md`, `REPORT_E6C.md`, and
`critical-distribution.md` are the evidence and next-phase gate.

**2026-08-21 E7/E7c closeout:** E7 staged 12 easy and 12 hard cells, verified 5/6 causal
archetype transfers across all three new sources, and cost 0.340 contended GPU-h. Source 086's
door-lintel matched outcome labels but was refused because contact followed drift. E7c changed
only that archetype to `hanging_panel` and verified the complete 2x2 pattern in four rollouts and
0.057 GPU-h. `q_lfh_v1.json` now records four source-balanced atoms and 12 verified
source-archetype pairs. The count-only gate is satisfied, but strict E5 readiness is still blocked:
only shelf and I-beam are common across all sources. E8 must transfer hanging panels to
`cf_005_056`, 089, and 090 before any learner comparison.

**2026-08-21 E9/E10 closeout:** E9a connects fresh Kimodo generation to the fail-closed LFH
transaction. The curved nominal accepted in empty-scene Isaac, while its 80 mm crouch twin rejected
on reference path/endpoint tracking with zero external contact; therefore no obstacle was authored.
E10b separately verifies multi-obstacle scene composition on source 086: one binding hanging panel
plus four route-relative context obstacles reproduced the full 2×2 pattern at the source seed, with
288.78 mm CPU context clearance and no secondary contact. The first E10 run at a changed seed failed
both adapted cells without contact, so simulator seed is a required context variable rather than a
nuisance to average away silently. `materialize_motion_scene_loop.py`,
`materialize_context_rich_scene.py`, `analyze_e10_context.py`, and the Isaac wide-camera assets on
the LFH progress page are the executable evidence.

The next implementation sequence is: (1) calibrate a fresh motion/operator pair until both
empty-scene cells accept; (2) compute route-frame support using face normal and finite 3D extent;
(3) pursue a lateral or oblique executed separation instead of more overhead variants; (4) finish
E8's common hanging-panel cross; then (5) run the equal-budget E5 proposal-distribution comparison.

**Phase 4 — Tail synthesis (Workstream B, E3) for arms and crouch.** Window search over the gated motion pool (executed reach profiles, `w_min` = 20 mm default), consuming Phase-1 `targets.json`; ≥12 candidates per behavior, top 4 instantiated; preflight manifest for approval (≤32 rollouts); full funnel in REPORT_E3.md whether or not families verify. Then E4 (coverage before/after) and the E5 design stub (write it; run nothing). **Gate: final review.**

## Ask, don't assume

Bring these to the user rather than deciding yourself: GPU/rollout budgets beyond the stated defaults; whether any generated family may enter claim 5 (default no); final prediction-register thresholds; whether lintel side-jambs are acceptable when they pass keep-out; anything requiring a *new* operator (side-step) — that is a scope decision, not yours; any conflict between the plan and the repo where both readings are defensible.

## Style and quality

Match the repo's naming, CLI, and USD conventions exactly — this must read like the same authors wrote it. Write in the corpus's vocabulary: edits, windows, binding faces, verdicts, refusals; say "edit necessity/sufficiency," never "optimality" (this is not LfH's guarantee and must not borrow its language). Tests before trust: no measurement path ships without a unit test, and E1b guards everything. Keep every report reproducible: numbers in reports must be regenerable from the index or from a committed script, never hand-typed. When something fails, the failure goes in the report with a reason code — a refused batch honestly reported is a success of the methodology.

Begin with Phase 0 now.
