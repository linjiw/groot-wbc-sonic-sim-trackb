# Keypoint Critical-Set Method: theory, computation, and model layer

**Method spec v0.1 — 2026-08-20 — addendum to `sweepcf_hallucination_design_plan.md`**
Read together with the design plan; its Invariants apply verbatim here. Section 7 amends the plan's phases.

---

## 0. Positioning: what transfers from LfH, and what cannot

LfH-CP supplies four things: (1) a theory of *critical configurations* — the minimal spatio-temporal facts about obstacles that determine behavior; (2) a two-stage factorization — identify the critical point, then procedurally generate diverse realizations through it; (3) a self-supervised learning loop closed by a **fixed differentiable decoder** (a classical planner); (4) a coverage metric (DCS).

Three obstructions in the humanoid case dictate where each piece lands:

- **The body is not a footprint.** A cylinder has one reach; a humanoid has many, and *which body part binds* is part of the label. → The critical point (x, y, t) generalizes to a **keypoint critical set** (k, t*, axis, coordinate).
- **The decoder is real physics through a frozen controller.** Command-to-execution response is
  keypoint-, motion-, and context-dependent. CAL1/CAL2 disproved the historical scalar arm-tuck
  ratios as placement bounds, and the current delivery fits lack leave-one-motion-out calibration.
  D_φ is therefore proposal-only screening evidence until validated; exact accepted executions are
  authoritative. Physics remains the verdict (Invariant 1).
- **The target is a verdict pattern, not optimality.** We do not hallucinate scenes where a plan is optimal; we hallucinate scenes where an edit is *necessary and sufficient* (hard cell) or *unnecessary* (easy cell).

**Design stance:** compute trajectory-feasible support, learn only a source-balanced proposal
distribution over that support, and let physics judge. We deliberately do **not** learn a scene
generator or verdict model: the placement window is available in closed form (Section 2.4), while
appearance comes from deterministic archetypes.

---

## 1. Keypoint body model

### 1.1 Keypoint set

The implemented semantic set is `head_torso`, `wrist_left/right`,
`knee_left/right`, `foot_left/right`, `shoulder_left/right`, and `pelvis`.
`head_torso` is intentionally combined because the authoritative collision model
has no head link. Each label owns a disjoint group of the existing 29 capsules;
all 14 collision-bearing links map exactly once (D0-002). A group retains every
executed capsule start/end/radius track rather than inventing one center or radius.

Ground-support constraints remain v2: they require a separate conservative sole
support polygon. Collision-capsule radii are never treated as support radii.

### 1.2 Soundness hierarchy — what keypoints may and may not certify

Semantic grouping preserves the full capsule members, but tier-1 face extrema do
not certify compound scene geometry. Therefore:

| Tier | Instrument | Role | Cost |
|---|---|---|---|
| 1 | Semantic capsule-group screen (this spec) | search, attribution, proposal; **necessary-condition filter only** | O(T·29) |
| 2 | Full capsule sweep + keep-out validator (plan §3.3) | geometric certificate | existing |
| 3 | Physics rollout via existing scorer | verdict | GPU |

No scene skips tier 2 because tier 1 passed. Tier 1's job is to make tiers 2–3 cheap by proposing only configurations that can work.

---

## 2. Theory

### 2.1 Margins, reach, and the critical set

Let the crossing segment be T_x, route arclength u(t), and C_k(t) the
authoritative capsule surfaces assigned to semantic group k. For a candidate
face F = F(axis, s), define the signed group margin

  μ_k(m; F) = min over t ∈ T_x, c ∈ C_k(t), active(c, t, F) of d_signed(c, F).

`active` requires the capsule projection to overlap the finite face footprint;
thus along-route extent is part of the intervention, not context. Body margin
μ(m; F) = min_k μ_k(m; F); the **binding keypoint** k*(m; F) and
**critical frame** t*(m; F) are the argmins. (k*, t*, axis, s) is the
humanoid critical set. t* refines `clearance_frame`; k* maps the 14-link
`closest_body` attribution into the semantic group set.

**Positional reach profile.** For an axis and band B, R_k(u) is the maximum
support of every capsule in C_k over frames whose projection overlaps the face
at u. Hands oscillate with gait, so R_hand(u) is genuinely positional; faces
shorter than one gait cycle must be windowed against R(u), not a global max.
Body reach R(u) = max_k R_k(u), retaining the group, owner, and frame argmax.

**Route-relative 3D frame.** Let `T(u) = [t(u), l(u), z]`, where `t` is the executed route tangent,
`l = z × t`, and `z` is world up. A candidate face uses `n_world = T(u*) n_route` plus finite
tangent/lateral/vertical extent. This represents lateral, floor, overhead, and oblique geometry in
one coordinate system. Representation is not evidence: a direction enters `P_feas` only when an
accepted executed nominal/adapted pair has a robust margin separation along that exact normal.

### 2.2 Constraint classes and the keypoint role table

| axis_type | Binding sense | Coordinate s | Primary keypoints | Edit operators | Notes |
|---|---|---|---|---|---|
| `overhead` | body must stay **below** face | underside height | head, shoulders, (knees under deep crouch) | crouch | existing duck families |
| `lateral_gap` | body must stay **inside** gap | gap width | wrists, shoulders, pelvis, thighs | arm-tuck; side/narrow (v2, operator TBD) | arm-tuck delivery is motion-specific; CAL1/CAL2 and scene-conditioned recertification gate every proposal |
| `ground_support` | feet must land **inside** allowed region | beam width / gap length / rail height | feet (inscribed footprint); shank for step-over apex | step-high / stride edits (inventory TBD) | **dual constraint**: intended foot–ground contact is the point, so keep-out exempts support surfaces and the verdict is traverse/fall/timeout, not strike/clear. **v2, pilot-gated** (Section 6) |

**Knee-band structure.** A bar at height h partitions into regimes: h below the original gait's foot-swing apex ⇒ no edit needed; h ∈ (apex_orig, apex_edit) ⇒ a step-high edit is necessary and sufficient (a foot-keypoint window, measured from *below* — reach of the swing apex); h above any feasible step ⇒ refusal-rich infeasible zone. This yields a natural difficulty ladder and a second crouch-independent leg behavior.

**Coupling.** Edits move keypoints jointly: crouch lowers head/shoulders but pushes knees forward and can widen stance. Define the measured **edit response field** Δ_k(O, α) — per-keypoint displacement (by axis) under operator O at amplitude α — and its sign/magnitude summary, the coupling matrix C[k, O]. The proposer must screen *all* keypoints of the edited motion against *all* scene faces (a crouch family with incidental knee-height geometry is a compound scene, refused or intentionally designed, never accidental).

### 2.3 The counterfactual inverse problem and its factorization

Forward map (physics): V(m, G) ∈ {clear, strike(k, t)} for motion m in scene G. Inverse (hallucination): given the pair (m, m′ = O_α(m)) and target pattern P — hard: (strike, clear); easy: (clear, clear) — find G with (V(m,G), V(m′,G)) = P.

Factorize G = F ⊎ G_ctx (binding face ⊎ context), with G_ctx ∩ dilate(S(m) ∪ S(m′), δ_ko) = ∅ (the keep-out condition, plan §3.3). Then:

- the feasible set for F is the **window** (2.4) — the analogue of LfH-CP's critical point;
- G_ctx is the free diversity dimension — the analogue of LfH-CP's g(K) generating unlimited trajectories through a fixed critical point. Archetype, materials, clutter, room dressing all live here.

**Context-transfer hypothesis (empirical, not a lemma).** For a fixed source, axis, finite exposure,
and binding station, keep-out-compliant archetypes may preserve the verdict pattern. E2 supported
this for 2/3 overhead variants of one source; E3 falsified it for the tested lateral adapted-hard
cell because the binding face changed the executed path before drift. Every axis/source/context is
therefore measured by physics, and accepted zero-external-contact context trajectories are used to
recertify any retry.

E10/E10b sharpen this hypothesis. A scene with one binding hanging panel and four keep-out-compliant
route-relative context obstacles failed both adapted cells at seed 33101 but reproduced the full
2×2 pattern at the source seed 32301. The matched run had 288.78 mm minimum CPU context clearance,
no context contact, and binding contact before drift. Thus context transfer is seed-conditioned even
when geometric clearance is large; one matching seed does not certify population invariance.

**Temporal criticality.** The binding constraint is active only on a short interval around t* — LfH-CP's phase-2 insight, here measured rather than learned (no Gumbel-Softmax needed). This licenses (a) faces of finite along-route extent and (b) the future dynamic-obstacle extension (a face present only near t*), which stays roadmap-only per the plan.

### 2.4 The window, closed form, and engineering margins

Overhead case (others analogous, with reach replaced by the class-appropriate extremum): for underside height s, μ(m; F_s) = s − R↑(m). Hence

  W = [ R↑(m′) + δ_clear , R↑(m) − δ_strike ],  |W| = (R↑(m) − R↑(m′)) − δ_clear − δ_strike.

The window is an order statistic of executed reach values — **no sampling, no learning, O(T·|K|)**. Existence ⇔ executed edit depth at the binding surface exceeds δ_clear + δ_strike. Current corpus conventions (hard = window center; easy = R↑(m) + 50 mm; refusal `window_below_min` for |W| < w_min = 20 mm) are unchanged.

**Predictive proposal window.** When m′ has not been rolled yet, replace R↑(m′) by a screened upper
estimate R̂↑_hi(m′) from D_φ (Section 4):

  W_cert = [ R̂↑_hi(m′) + δ_clear , R↑(m) − δ_strike ].

This interval is not called certified until its predictive coverage is validated. Empty support
still implies refusal; otherwise an empty-room calibration rollout collapses the response estimate.
For a target coordinate s*, solve α* = min{ α : R̂↑_hi(m, O, α) ≤ s* − δ_clear } as a proposal,
then require exact executed reach and physics before promotion.

**Constraint-conditioned recertification.** Empty-room `W_cert` is a proposal certificate, not an
invariance claim about the controller under authored geometry. Once an accepted, zero-external-
contact execution exists in the candidate context, recompute the reach as R^G and require any retry
coordinate to clear R^G + δ_clear while remaining inside its registered DCS bucket. Rejected or
contact-contaminated executions cannot certify clearance because contact may have already changed
the path. If the accepted-context bound lies outside the target bucket, refuse that target rather
than spending a boundary retry. The E3 lateral pilot motivated this rule: its accepted adapted-easy
path required a 0.906489 m gap before applying the empirical engineering margin, outside the
targeted 0.8–0.9 m bin.

**Target feasibility is distinct from target emptiness.** SweepCF-DCS v2 removes operator-keypoint
combinations that cannot bind and lets only canonical cells of an exact verified variant occupy a
causal target. The final audit found that v1 probe rows had incorrectly suppressed CAL3's
`clear_25_50` target: replaying the same immutable trials restored 20 proposals across three
sources. DCS is now a novelty/reporting term; measured route phase, finite extent, and the exact
executed window define proposal support.

---

## 3. Computation

1. **Extraction** (`keypoints.py`): lossless semantic capsule tracks from stored rollout body states (R1/R2); cache by rollout path + hash; unit-test the exact 29-capsule partition.
2. **Reach profiles** (`reach.py`): R_k(u; axis, band) at fixed route stations (default 2 cm spacing); exact min/max within each station via frame maxima.
3. **Window solver** (`window.py`): interval arithmetic over reach profiles per §2.4; emits window, binding keypoint, critical frame, per-keypoint margin table; feeds ConstraintSpec directly (plan §3.1 gains fields `binding.keypoint`, `binding.per_keypoint_margins`).
4. **Proposer** (`propose.py`, `critical_support.py`): builds trajectory-feasible support first,
   including route progress and finite extent; DCS novelty ranks only within that support. It applies
   the multi-keypoint screen and hands the spec to instantiate → validate → preflight.
5. **Tier discipline**: proposer output is tier-1 only; the capsule keep-out validator (tier 2) and physics scorer (tier 3) run unchanged.

---

## 4. Model layer

### 4.1 Delivery model D_φ (the learned component)

**Role.** The seat LfH gives its fixed decoder — the map from commanded to realized — is occupied here by real physics; D_φ is its cheap, differentiable, uncertainty-carrying *predictor*, used only to propose.

**Data.** Generalize the existing fidelity columns (`commanded_amplitude_rad`, `executed_amplitude_rad`, `operator_survival`) to per-keypoint vectors: for each empty-room rollout of an edited motion, record commanded keypoint displacement (from the kinematically retargeted edit) vs executed displacement (from tier-1 extraction), per axis. New sidecar table `keypoint_response.csv` (motion_id, operator, α, keypoint, axis, commanded_mm, executed_mm). **Every future rollout appends to it — corpus construction is the active-learning loop.**

**Form.** Per (operator, keypoint, axis): 1-D monotone regression of executed vs commanded
displacement with saturation and partial pooling only when data supports it. Historic scalar
response ratios are diagnostics, not priors or bounds. The current implementation's in-sample
residual q90 is a screening statistic, not a calibrated predictive interval.

**Validation (E-D1, first evidence 2026-08-26).** Leave-one-motion-out CV over **28 distinct
reference clips** (accepted empty-scene rollouts) on the quantity the window is actually built
from — executed overhead reach at a finite face:

| model | RMSE | median abs | q90 abs |
|---|---:|---:|---:|
| **linear in commanded reach** | **9.07 mm** | **5.75 mm** | **14.78 mm** |
| identity (`executed = commanded`, the implicit baseline) | 14.04 mm | 10.69 mm | 20.15 mm |
| feature-blind executed mean | 22.82 mm | 9.27 mm | 35.03 mm |

A 35.4% RMSE reduction against identity, with the feature-blind control the *worst* model. After
LFH-E12 added 18 clips the model had never seen, across eight further body modes, the refit over
**46 distinct clips** gives RMSE 9.71 mm against identity's 15.95 mm — the advantage *grew* to
**39.2%**. Corpus:
`docs/hallucination/reach_response_corpus.csv` (118 paired rows); fit and scores:
`reach_delivery_model.json`; write-up: `REPORT_DELIVERY_MODEL.md`.

Two limits are load-bearing. The fit is conditioned on *acceptance* — it estimates reach given that
the clip tracked, and says nothing about whether it will track, which is the gate E9a actually
failed. And it does not describe a drifting execution: in-scene cells with 0.30–0.37 m endpoint
error show residuals up to +200 mm, because a drifted robot reaches the commanded station at a
different gait phase. D_φ still cannot certify geometry or replace an executed calibration pair;
it can now choose an amplitude and refuse an empty window before the spend
(`scripts/research/hallucination/propose_scene_from_motion.py`).

**Calibration of the whole proposal chain (E-D2).** Report the fraction of proposed windows whose
physics verdict matches the target pattern, stratified by source, axis, and context. No universal
threshold is asserted from the current one-source/small-cohort evidence.

### 4.2 Critical-support proposal distribution

Parameterize an obstacle atom by

`c = (operator, u*, n_route, keypoint, extent_tlv, mu_nom, mu_edit, xi, archetype, context)`.

For exact executed reaches and engineering margins, let
`L = R_edit + delta_clear`, `U = R_nominal - delta_strike`, and
`xi = (coordinate - L) / (U - L)`. `xi` expresses placement within the pair-local critical window
without confusing robot height with difficulty.

Keep three distributions separate:

- `P_env`: the unknown natural-world obstacle distribution, which this corpus cannot identify.
- `P_feas(c | executed pair)`: deterministic support emitted by geometry, timing, keep-out, and
  observability gates.
- `q_LFH`: the designed training/proposal distribution over `P_feas`.

For a multi-obstacle scene, `q_LFH` samples one causal binding atom and a separate
`q_ctx(G_ctx | trajectory, binding)` samples context subject to the joint swept-volume keep-out.
Context objects do not receive causal labels or extra source weight. Multi-binding scenes remain
unsupported until per-obstacle ablations establish incremental causal contribution.

At current scale, `q_LFH_v1` is a source-balanced deterministic sampler, not a fitted neural model.
E7/E7c provide four sources and 12 verified source-archetype pairs: source mass is 0.25, archetypes
are balanced within source, and easy/hard roles split each atom equally. Only `shelf_plank` and
`ibeam` are common to all four sources, so the strict crossed learner gate remains closed. After E8
adds a third common archetype, compare a maximum-entropy or hierarchical density over route phase,
extent, keypoint, and `xi`, with an exploration component and per-source caps. Physics outcomes
evaluate the sampler but never become a learned verdict gate.

`q_LFH_conditional_v1` was the first small-data realization of that interface, and its conditioning
claim is **withdrawn** (audit, 2026-08-26). It improves on a uniform prior — 1.51373 against
1.60944 nats — but a feature-blind marginal-counting sampler with the same source balancing and
exploration floor scores **1.46416**, so the kernel costs 0.0496 nats and the entire gain over
uniform belongs to frequency counting. Its top-3 recall and 400/400 support-validity are true by
construction. Use source-balanced counting as the archetype proposal until a better target is
chosen.

The diagnosis generalizes, and it is why §4.1's D_φ is now the retained learned component: the
executed pair determines *where* the face must be — closed form, §2.4, deliberately unlearned —
and does not determine *what the face looks like*. Archetype identity was the one part of the
scene the conditioning inputs do not constrain. Executed reach is a part they do.

### 4.3 What we deliberately do not learn

No learned scene/face generator (Dyna-LfLH's mode collapse; closed-form window makes it unnecessary). No learned verdict predictor shipping labels (Invariant 1). No controller fine-tuning (Invariant 2).

### 4.4 Downstream learner hooks (E5 design stub — written now, run later)

The keypoint formalism gives the claim-5/6 learner supervision that is free from the index:

- **Binding-aware auxiliary heads** on the policy trunk: classify binding keypoint ∈ K ∪ {none} and regress signed margin (labels: k* mapped from `closest_body`, `min_clearance_mm`) from ego observations. "Which part of me does this room bind, and by how much" is precisely the scene-conditioning claim 5 asks about, made into a measurable head.
- **Twin-contrastive loss**: embeddings of a family's easy/hard episodes should differ along a direction predictive of (axis, s); pairs are free by construction.
- **Probe metric**: auxiliary-head accuracy on held-out archetypes (E2's variants) — a cheap early read on geometry-use that doesn't wait for full policy evaluation.

Out of execution scope per the plan; this section exists so the E5 stub has an architecture to point at.

---

## 5. Index & artifact additions (additive only)

Episode grain: `binding_keypoint` (semantic class), `binding_keypoint_margin_mm`; per-keypoint margin vectors go to a sidecar (`keypoint_margins/<episode_id>.npz`), not 10 new columns. New tables: `keypoint_response.csv` (§4.1). ConstraintSpec: fields per §3.3. Coverage (plan Workstream C): add `binding_keypoint` as a dimension → keypoint-DCS; `targets.json` bins become (behaviour × keypoint × band × margin bucket), which is what the proposer consumes.

---

## 6. Risks specific to this layer

- **Keypoint screen unsoundness** → tier 2 mandatory (never certify from keypoints); unit test that every historical strike's `closest_body` maps into K's coverage, else extend K.
- **Small-data D_φ** → monotone/saturating priors, published-ratio priors, LOMO-CV, report-first thresholds; never gate on D_φ where a cheap rollout can measure directly.
- **Coupling surprises** (crouch → knee protrusion strikes) → multi-keypoint screen in the proposer; tier 2/3 catch what it misses; any such catch is logged and added to C[k, O].
- **Body-state logs may not exist in stored rollouts (R1)** → Phase 0 answers this first; if absent, D_φ v0 trains from a small dedicated empty-room re-roll batch (budget flagged to user).
- **`ground_support` semantics** (verdict = traverse/fall/timeout; keep-out exemptions; SONIC may simply fail on beams) → single pilot family attempt, explicitly allowed to fail, behind its own approval gate; success optional, funnel reported either way.

## 7. Integration into the plan's phases

- **Phase 0 (+recon items):** R1 — are per-frame body states (joint q or link poses) recoverable from stored rollout artifacts? R2 — capsule→keypoint mapping and radii. R3 — does any step-high / stride operator exist (for the knee-bar and `ground_support` rows)? Answers go in RECON.md.
- **Phase 1:** ship keypoint-DCS alongside the base coverage tool (same CLI, one more dimension).
- **Phase 2:** implement §3 modules with the plan's machinery; the plan's window search is *replaced* by the closed-form solver; fit D_φ v0 from whatever R1 yields; E1b additionally asserts the recomputed duck_003 window and binding keypoint match stored family measurements.
- **Phase 3 (E2):** report pattern reproduction as source/axis-conditioned context transfer
  (§2.3), never as a general invariance theorem.
- **Phase 4 (E3):** proposer + α* solver drive candidate generation from keypoint-DCS targets; add E-D2 to the register before running; optional `ground_support` pilot behind its own gate.
- **E5 stub:** include §4.4.
- **E6:** verify one canonical finite shelf for each of the three CAL3 sources, using symmetric
  engineering margins and context-conditioned retry gates; then update `q_LFH` support.
- **Distribution-value experiment:** once four sources × three common archetypes form a complete
  crossed matrix,
  compare source-balanced critical sampling against equal-budget uniform-feasible, DCS-ranked, and
  visual-only controls on held-out source × archetype splits.
