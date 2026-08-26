# Fable — research guidance for the ICRA paper

**Motion → scene generation with Learning-from-Hallucination for humanoids**
Written 2026-08-26 after reading `docs/lfh/*`, `docs/hallucination/REPORT_*`, the prediction
register (LFH-E1a … E10b), `docs/paper/sweepcf_draft.md`, `E5_LEARNER_PROBE_DESIGN.md`, and the
2026-08-22 autoresearch iteration. Where I make a judgement rather than restate a measurement,
I say so.

Assumed deadline: ICRA 2027 submission ≈ **15 Sep 2026** (three weeks). If it is later, keep
the ordering below and stretch the scale, not the scope.

---

## Update — 2026-08-26, after the code audit and first model fits

Everything below this section was written before auditing the code. Five things changed, and two
of them change the paper. Full detail: `docs/hallucination/REPORT_AUDIT_2026-08-26.md` and
`docs/hallucination/REPORT_DELIVERY_MODEL.md`.

**1. The overhead results survive; seven fail-open defects were fixed.** After the fixes,
recomputing `lfh_089_crouch`'s support from the stored trajectories reproduces
`critical_support_cal3.json` bit-exact, and the offline suite passes at 828. But
`lateral_face_reach` was measuring whole-capsule endpoints after a mere AABB test (up to **8×**
overestimate), and `solve_window` emitted `+inf` margins that fail to serialise — together, **no
lateral window could ever populate a spec**. The "lateral is unsupported" claim is therefore
confounded between the E3 physics observation and a broken instrument. Re-derive it on fixed code
before writing it as C7.

**2. The conditional `q_LFH` result is refuted.** Feature-blind archetype counting with the same
source balancing and exploration floor scores **1.46416 nats** against the kernel's 1.51373. The
conditioning *costs* 0.05 nats; the whole reported gain over uniform belongs to frequency counting.
Top-3 11/12 and 400/400 in-support are true by construction. The cause is the **learning target**:
the executed pair determines where the face must be (closed form, unlearned) and does not determine
what it looks like. Archetype identity is the one part of the scene the inputs do not constrain.
Docs corrected; the register carries the refutation.

**3. A learned component that does survive audit: `D_phi`.** Fitted on the quantity the window is
actually built from — executed overhead reach at a finite face — over 118 paired rows,
leave-one-motion-out across **28 distinct clips**: RMSE **9.07 mm** against identity's 14.04 mm, a
**35.4% reduction**, with the feature-blind control the *worst* model. This is E-D1's first
evidence. `propose_scene_from_motion.py` now runs the full motion→scene inference on CPU: apply
the ladder, predict both reaches, solve the window, refuse when it does not survive the model's own
q90 band. On the three verified sources it lands within 8–15 mm of their known executed windows.
**Report this as the paper's learned component, together with the baseline that killed the first
one.** Two models, one refuted by its own honest baseline and one that beats it, is a stronger and
more credible section than either alone.

**4. Source supply is not 4 — it is ~58 candidates.** CAL3 screened only the 15 `stand_to_walk`
clips and then discarded three of its four ladder rungs. The pool holds 150 clips, 94 gated, of
which **58 have straightness ≥ 0.95** across ten body modes. Of 36 screened so far, every one has
at least two usable crouch rungs and a third reach the full 85 mm. The 24–30 family bar is a
screening problem, not a generation problem.

**5. E9a's conclusion is confounded and must not be quoted.** The fresh motion has route
straightness **0.747** against the `MIN_ROUTE_STRAIGHTNESS = 0.95` the CAL3 selection itself
imposes (verified sources are 0.985–0.991), and it was commanded at the top of the ladder. It
varied straightness and amplitude together, outside the calibrated envelope. "A fresh generated
motion is not yet a valid LFH source" is not supported by it.

**Two new measurements worth a figure each.** The three accepted crouches deliver **63–84%** of
their commanded window (83–86 mm commanded → 52–72 mm executed), consistently optimistic in one
direction. And the eight arm-tuck pairs are a genuine null control on the overhead axis — their
commanded window is exactly zero — giving executed windows within **2.73 mm** of zero. That is a
measured noise floor, and it is six times tighter than the 18.044 mm engineering margin applied on
both sides of every window. The two are different quantities (E1a's margin is an in-scene clearance
range) so the margin is **not** being changed, but it currently spends 36 mm of every 52–72 mm raw
window, and that is now a registered question rather than an assumption.

**6. E12 ran, and its primary prediction is falsified — which localised the real bottleneck.**
47 cells, 0.427 GPU-h. Predicted ≥8 new sources; got **5**. Predicted ≥6 clearing a 20 mm window;
got **1**. Funnel: 12 motions → 8 nominals accepted → 5 delivered an amplitude → 1 entered
`P_feas`. Two predictions did hold, and one of them matters: the median delivered amplitude is
**55 mm**, not 80, and the five motions deliver 40/40/55/70/70 — a fixed command is the wrong
instrument. Acceptance was also monotone in amplitude for all 8 accepted nominals, which was only
observable because I made the rungs independent before spending.

But the ladder did not rescue supply, because a shallow crouch tracks without separating. The five
raw executed windows are 31.5, 43.7, 51.3, 54.5, 70.2 mm — the same range as the three verified
CAL3 sources — and the symmetric 18.044 mm margin removes 36.1 mm from each:

| margin, each side | pairs clearing the 20 mm floor |
|---|---|
| 18.044 mm (current) | **1 / 5** |
| 10 mm | 4 / 5 |
| 5 mm | **5 / 5** |

**The binding constraint is the imported engineering margin, not motion supply and not amplitude.**
Four of five pairs are refused by a margin rule rather than by physics. E1a derived 18.044 mm from
in-scene *clearance* ranges on overhead cells; the paired null control measures the empty-scene
window estimator reproducing a true zero to 2.73 mm. Different quantities — so the margin is not
being changed — but this is now the cheapest large win available.

**7. `D_phi` generalised out of sample.** It was fitted before E12 ran. E12 added 18 unseen clips
across eight body modes absent from the fit; the refit over 46 clips gives RMSE 9.71 mm against
identity's 15.95, and the advantage **grew** from 35.4% to 39.2%. That is a relation holding, not a
fit chasing points.

**8. A verified family from a ladder-discovered source (LFH-E17), and the visuals.** The first
scene authored by the generalised synthesiser, from `ladder_138` — a `reach_walk` clip the CAL3
cohort never contained. All four registered predictions confirmed at 0.078 GPU-h:
accepted/accepted/rejected/accepted, `hard/nominal` rejected for **contact alone** (415.9 N on
`torso_link` at frame 101, no drift reason recorded at all), zero external contact in the three
clear cells, and `hard/adapted` endpoint identical to its empty-scene value. The two adapted cells
come back bit-identical across two genuinely separate rollouts — because the crouched robot never
touches either plank, so the 85 mm height difference is causally irrelevant to it. That is the
counterfactual stated as sharply as this corpus can state it.

**9. Acceptance is seed-dependent at the deepest rung, and that reframes E12 (LFH-E16, partial).**
E16 stopped at 6/18 cells when another user's job took 23.8 GB of the shared card; the runner's
contention gate and hang timeout held, and no cell was mis-scored as a rejection. The six that ran
falsify E16's own third prediction and displace its primary. `ladder_138`'s adapted rung accepted
**1 of 3** seeds, with endpoint error 0.3298 / 0.3300 / 0.3823 / 0.4355 m across four seeds against
a 0.35 m gate — it straddles the threshold, spread 105.7 mm, twice the nominal's. So E12's
"delivered amplitude" is a single-seed point estimate at the edge of a cliff, and its best source
is the one that fails to reproduce. Nothing in E12 is withdrawn, but **delivered amplitude must be
redefined as accepted at k of n seeds**, and the operating point should sit a rung below the cliff
rather than on it. That now outranks trimming the placement margin.

**10. The model and its metric.** `scene_distribution.py` implements `q(S | executed pair)` with
support computed and only the free remainder sampled, plus the eps-delta scorecard. Two closed
forms fell out and are verified in tests: **regret = xi * |W|**, identically equal to the adapted
motion's clearance under the face; and **regret + necessity = delta_strike + |W|**, invariant in
xi. So `xi` *is* normalised regret, the corpus convention of placing hard at the window centre
spends half the window on it, and neither term improves except by a deeper delivered adaptation —
which is exactly why E12 and E16 are the experiments that matter. Scored against baselines on 2000
samples: random placement admits **3.8%**, uniform-in-support **45.9%**, shaped `q_LFH` **100%** at
half the median regret. Note honestly that most of the jump is the closed-form support, which is
computed rather than learned.

**11. Archetype freedom is verified (LFH-E18).** Four visually distinct obstacles — a 40 mm
plank, a lintel with jambs, an I-beam with web and flange, a 0.42 m panel — placed at the *same*
coordinate inferred from the *same* executed pair, all reproduce accepted/accepted/rejected/
accepted. `hard/adapted` endpoint error is **0.3297638984283594 m in all four**, identical to the
empty-scene value: the adaptation is completely insensitive to what the obstacle is, because it
never touches any of them. This is the two-stage factorisation — critical point computed,
realisation free — demonstrated rather than assumed, and it is the empirical basis for treating
archetype as a design choice rather than something to condition on.

One prediction of mine was falsified by its own wording: I required the hard-cell rejection to be
`disallowed_robot_contact` *alone*, and `ibeam` also trips both tracking gates. But its contact is
at frame 101 on `torso_link`, the same frame and body as `door_lintel`, with endpoint error 0.4208
against 0.3432 — the tracking failure is the *consequence* of a harder strike, not a rival cause.
The predicate conflated "contact is the cause" with "contact is the only line in the report". Fixed
in the analyzer; the result stands.

Also worth keeping: peak force varies 410–538 N across archetypes at an identical face coordinate,
and the peak frame moves from 101 to 117 for the thick panel. **Verdict is invariant; contact
dynamics are not** — which is the standing reason force is never reported as a graded quantity.

**12. The margin question is answered (LFH-E16b).** 24 cells, 0.746 GPU-h. Three motions x three
rungs x three seeds. Two results matter.

*Acceptance is not a property of a motion.* `ladder_034` accepts 3/3 at **every** rung including
70 mm; `ladder_126` accepts 3/3 only at 40 mm; `ladder_148` accepts 3/3 at **no** amplitude despite
a stable 3/3 nominal and an E12 record of "delivered 55 mm". So single-seed delivery is not
systematically optimistic — E12 was exactly right for two of three — it is *unreliable*, and the
failures concentrate where rungs sit near the gate. Seed spread also grows with amplitude in all
three motions (nominal 17–94 mm → deepest rung 89–170 mm): crouching makes tracking not just worse
but **less predictable**, which is why an operating point near the gate is unsafe even when its
mean sits below it.

*The executed window is highly reproducible where it is measurable.* On rungs accepting 3/3, the
three-seed window range is **1.18, 2.66, 3.39, 7.94 mm** — worst case 7.94 mm, against a margin of
18.044 mm applied to *each* side. Same order as the 2.73 mm null control, now measured by E1a's own
statistic on the quantity the margin is actually applied to. **The margin removes 36.1 mm from
every window to guard something that moves by at most 8 mm.** That is now a measured claim, not an
inference — and it is the registered basis for proposing an empty-scene margin, which is a separate
decision I did not take.

Net: two **stable** sources established in a sense no earlier source has been — `ladder_034`
(`walk_look`, 70 mm at 3/3, window 51–55 mm) and `ladder_126` (`carry_walk`, 40 mm at 3/3, window
36–44 mm). Delivered amplitude is redefined as *deepest rung accepted at 3/3 seeds*; E12's yield
figure is now an upper bound.

**Revised priorities, in order.**

1. **Register the margin decision** (no GPU). E16b measured what E16 asked. Propose an
   empty-scene margin derived from empty-scene repeatability — the worst observed three-seed range
   is 7.94 mm — with the E1a value retained for every existing artifact so nothing is retroactively
   re-graded. At 10 mm/side, four of E12's five pairs clear the 20 mm floor instead of one. This is
   the single largest yield gain available and it costs no rollouts.
2. **Run the ladder + 3-seed protocol over the remaining 55 screened clips.** The pipeline is now
   fully scripted end to end (screen -> prepare -> manifest -> run -> analyse -> synthesise ->
   verify -> render). At ~0.25 GPU-h per motion for a 3-seed ladder, 20 more motions is ~5 GPU-h
   and would take the stable-source count from 2 toward the 24-30 bar.
2. **E14 — lateral repeatability, redesigned.** Not an E3-style retry. The fixed instrument yields
   raw lateral gap windows of 8–51 mm on existing executions (grid-maxima, so optimistically
   biased), all refused by the same imported margin. Measure lateral repeatability first, then
   pre-register a station and band.
3. **E11 (regret)** and **E15 (seeds)** unchanged. Note E16 and E15 share machinery — the same
   three-seed re-rolls answer both.
4. **E5** keeps its go/no-go, but the paper no longer needs it for a learned component: `D_phi` is
   one, and the refuted archetype model is a second, more interesting result.

**What to write in the paper about supply.** Not "we have N families". The honest and more
interesting claim is the funnel with its causes: the clip pool offers 58 straight gated candidates,
the controller rejects whole gaits (`side_step`, `backward` nominals fail at 0.45 and 0.38 m
endpoint error on straight routes), the crouch costs tracking budget so delivered amplitude varies
2× across motions, and the margin rule then removes most of what survives. Each of those four is
measured, and three of them are fixable.

---

## 中文摘要（详细内容见下文英文）

1. 你写的 $\tau,K,G \to p_\psi(S\mid\tau,K,G)$ 本身是 non-identifiable 的，这一点你已经意识到了。
   代码库里实际做的、也是**正确**的做法，是把条件改成 **counterfactual pair**
   $(\tau_0,\tau)$ ——原始动作 + 被编辑的动作。给定这一对，critical window 有闭式解，问题就变成
   identifiable 的。论文的核心主张应该是这一句：*单个动作不能解释场景，但一对反事实动作可以。*
2. 你定义的 $\mathcal S_{\epsilon,\delta}$ 四个条件里，现有的 2×2 physics 模式
   (accepted/accepted/rejected/accepted) 已经**离散地**验证了 Feasible 和 Necessity；
   Regret ≤ ε（"这个动作是否接近最小编辑"）和 Realism 还**没有**被测量。两者都能便宜地补上（下面 E11、E13）。
3. 现在的真正瓶颈**不是 GPU**（一个 4-cell family ≈ 0.06 GPU-h），而是**能被控制器跟踪的
   motion pair 供给**：E9a 的新动作 80 mm crouch 在空场景就被拒了。下一步最重要的实验是
   **amplitude-ladder source screening**（E12），目标 2 周内从 4 个 source 增到 ≥12。
4. 三周内能诚实写出的论文：**"Counterfactual inverse scene design"** ——方法 + 物理证伪 +
   小规模 learning-value（E5）。不要承诺 lateral/floor 的 critical support，除非 E14 在
   2 天预算内成功；把 E3 的失败作为结果写进去。
5. Seed 必须当 random effect 报告（E10 vs E10b）。每个 family 至少 3 个 sim seed。

---

## 1. Where the theory stands against what is built

### 1.1 Your formal target, mapped onto the pipeline

Your set

$$\mathcal S_{\epsilon,\delta}(\tau)=\{S:\ \mathrm{Feasible}=1,\ \mathrm{Regret}\le\epsilon,\ \mathrm{Necessity}\ge\delta,\ \mathrm{Realism}\ge r_0\}$$

is the right object. Here is what each term currently is, in the code and in the evidence:

| term | what implements it today | status |
|---|---|---|
| $\mathrm{Feasible}(\tau,S,K)=1$ | `adapted_hard` cell **accepted** by the frozen SONIC + Isaac gate (tracking error, drift, contact) | measured, physics-verified. Note that *K* enters twice: the 29-capsule body (`keypoints.py`) and the **controller's tracking capability** — E9a shows the second dominates |
| $\mathrm{Necessity}(\tau,S)\ge\delta$ | `nominal_hard` **rejected** with contact uniquely attributed to the binding face *before* drift; `nominal_easy` **accepted** (i.e. in $S\setminus e$ the edit is unnecessary) | measured; the "remove $e$" ablation is literally the easy scene |
| $\mathrm{Regret}(\tau;S)\le\epsilon$ | **nothing** — the 2×2 shows the adapted motion is feasible and the nominal is not, but not that a *smaller* edit would fail | **not measured** (fix: E11) |
| $\mathrm{Realism}(S)\ge r_0$ | deterministic archetypes (`shelf_plank`, `ibeam`, `door_lintel`, `hanging_panel`, `hvac_duct`) + keep-out-certified context (`q_ctx`, E10b) | designed, not scored (fix: E13) |

With the lexicographic cost the draft already uses —
$J(\tau;S)=\infty$ if infeasible, else $D(\tau,\tau_0)$ — the 2×2 verdict pattern is **exactly** the
finite-sample membership test for $\mathcal S_{\epsilon,\delta}$ with $\epsilon=0$ over the
two-candidate set $\{\tau_0,\tau\}$ and $\delta=D(\tau,\tau_0)>0$. Say this in the paper. It ties
your formalism to the thing that is already verified, and it makes clear what is missing: the
minimum over *all* $\tau'$ (Regret) has only been taken over two candidates.

### 1.2 The non-identifiability point, and how the pipeline already resolves it

You are right that $p(S\mid\tau)$ is badly non-identifiable: a crouch can be explained by a beam,
a doorway, a branch, or theatre. The pipeline does not try to learn it. What it does instead —
and this is the paper's real theoretical move, so state it as such — is **condition on the
counterfactual pair** $(\tau_0,\tau)$ rather than on $\tau$ alone:

- Given executed $\tau_0$ (nominal) and $\tau=O_\alpha(\tau_0)$ (adapted), the set of overhead
  faces that make $\tau$ necessary and sufficient is a **closed-form interval**
  $W=[R^\uparrow(\tau)+\delta_{clear},\ R^\uparrow(\tau_0)-\delta_{strike}]$ (`lfh.md` §2.4).
  That is an identifiable object. Nothing is learned to get it.
- What is genuinely unidentified — *which archetype, which context, where inside $W$* — is
  handled by a **designed** proposal distribution $q_{LFH}$ over that support, and the physics
  gate falsifies each proposal. `q_LFH` is explicitly not $P_{env}$.

So the honest one-line thesis is:

> *A single humanoid motion does not identify the scene that explains it; a counterfactual pair
> does. LFH computes the identifiable critical support from the pair in closed form, samples
> appearance and difficulty from a designed distribution over that support, and lets a frozen
> physics-in-the-loop controller falsify every proposal.*

Two consequences for how you write the problem statement:

1. Replace $p_\psi(S\mid\tau,K,G)$ with $p_\psi(S\mid\tau_0,\tau,K,G)$ and say where $\tau_0$
   comes from. Today it comes from the *operator*: $\tau=O_\alpha(\tau_0)$, so the pair is
   constructed. The "from a single observed motion" version needs an **inverse operator**
   ($\tau_0=O^{-1}(\tau)$ — un-crouch the clip) and is future work. Do not let a reviewer discover
   this scope; declare it.
2. $K$ is not just morphology. The decoder in LfH is a planner; here it is a **frozen tracking
   controller**, and E9a/CAL1/CAL2/P9 show its response is motion- and context-dependent. That
   is the paper's second contribution (the draft's "controller is a feasibility oracle"
   section) — keep it.

### 1.3 What is actually established (do not overstate)

- **Overhead axis only.** 4 independent sources (`cf_005_056`, `lfh_086/089/090_crouch`),
  12 physics-verified source×archetype pairs, all `local_crouch`. Lateral critical support is
  **zero** (E3 falsified: the binding face changed the executed path). Floor/oblique: zero.
- **Finite exposure is a necessary design variable** (E6 → E6c: 0.30 m face fails without
  contact, 0.10 m passes). Real finding; a paragraph in the paper.
- **Context composition works at one seed** (E10b, 5 obstacles, 288.78 mm keep-out) and
  **fails at another** (E10, seed 33101, both adapted cells reject with zero contact). Seed is
  a random effect. You have n=1 per condition.
- **Fresh motion → pair fails on delivery, not geometry** (E9a: nominal accepted, 80 mm crouch
  twin rejected, endpoint 0.376 m, zero contact). This is the supply bottleneck.
- **Conditional `q_LFH`:** LOO log-loss 1.514 vs 1.609 uniform, top-3 11/12, 400/400
  in-support. The report says correctly that this is a mechanism check, not a density claim.
  Do **not** put it in the abstract as a learning result.
- **Claim 5 (learner) has not been run.** The selector harness is validated on synthetic
  families (0.963 privileged, 0.126 scene-hidden, 0.593 wrong-scene). The gate is closed on a
  crossed-matrix technicality (`hanging_panel` missing on 3 sources).

---

## 2. The paper you can honestly write in three weeks

**Title direction:** *Counterfactual Inverse Scene Design: Hallucinating the Obstacles that
Explain Humanoid Whole-Body Motion.*

**Claim ladder for the paper (each row must have a number in the paper):**

| # | claim | evidence needed | have it? |
|---|---|---|---|
| C1 | From an executed motion pair, the critical support is closed-form and non-empty for a measurable fraction of pairs | window yield vs random/midpoint/max-separation baselines (draft has 0/55, 0/55, 5/55) | yes — refresh with all sources |
| C2 | Proposals from that support are **physics-verified** at a reported rate (E-D2 calibration) | fraction of proposed 2×2s whose physics matches target, stratified by source/archetype/seed | partly — needs the seed replication (E12/E15) |
| C3 | The scene **explains** the motion: necessity (remove $e$) and near-minimality (regret) | 2×2 + amplitude ladder | necessity yes; **regret no** → E11 |
| C4 | Appearance/context is a free dimension: archetype and clutter transfer preserve verdicts, with measured exceptions | E2/E7/E7c/E10b + seed replication | yes at n=1 per cell; needs seeds |
| C5 | The controller is part of the inverse problem (delivery ≠ command; fresh motions fail on tracking) | CAL1/2/3, P9, E9a, ladder yields from E12 | yes |
| C6 | Counterfactual-critical sampling has **learning value** over uniform-feasible / DCS / visual-only at equal budget | E5 on ≥8–12 sources, held-out source×archetype | **no** — decide by day 12 whether it makes the paper |
| C7 (negative) | Lateral binding is not a scene-invariant intervention for this controller | E3 + one boxed retry (E14) | yes (as negative) |

Ship C1–C5 + C7 with certainty. C6 is the difference between a strong method paper and a
paper with a learning result; the schedule below gives it exactly one shot.

**Do not** claim: natural obstacle density, simulator invariance (MuJoCo is a renderer here),
lateral/floor generality, or policy-level (as opposed to selector-level) improvement.

---

## 3. Experiments, in priority order

Costs use the measured rate: one rollout ≈ 0.015 contended GPU-h; a 4-cell family ≈ 0.06 GPU-h.
GPU is not the bottleneck. **Register each of these in `docs/prediction_register.md` before
spending**, house style, predictions first.

### E12 — Source supply via amplitude-ladder screening  (start today; highest value)

*Why first:* everything downstream (C2, C4, C6) scales with independent sources, and E9a says the
failure is delivery at α=80 mm, not geometry. The draft's own numbers say crouch delivery ≈ 58%
and the controller resists leg departures, so a fixed 80 mm is the wrong ask.

*Protocol:*
1. Generate N=30 fresh Kimodo motions across ≥3 prompt families (straight walk, gentle curve,
   walk-and-turn), 2 seeds each, CPU gate as in E9a.
2. For each accepted nominal, build `local_crouch` twins at α ∈ {40, 55, 70, 85} mm at route
   progress 0.55 (or the station selector's choice).
3. Empty-scene physics on nominal + all twins (5 rollouts/motion ≈ 150 rollouts ≈ 2.3 GPU-h).
4. Keep the **largest accepted α** per motion. Compute the engineering window with the
   symmetric 18.044 mm margins; a source enters `P_feas` iff $|W|\ge 20$ mm.

*Predictions to register:* (a) ≥40% of fresh nominals accept; (b) of those, ≥50% have some
accepted twin; (c) median largest-accepted α is between 40 and 70 mm; (d) ≥8 new sources clear
the 20 mm window bar. If (d) fails, the paper's C1 yield number becomes the headline and C6 is
dropped — that is still a result.

*Output:* the source count for everything below, and the C5 "delivery ladder" figure (accepted
α vs motion), which is new evidence in its own right.

### E11 — Regret / tightness ladder  (the missing term in your definition)

*Question:* is the adapted motion close to the *minimal* edit the scene demands?

*Protocol:* for each verified hard scene (start with the 4 existing sources at their canonical
hard coordinate), roll the adapted motion at fractional amplitudes
$\alpha\in\{0.25,0.5,0.75\}\alpha^*$. Let $\alpha_{min}$ be the smallest accepted with zero
binding contact. Define

$$\widehat{\mathrm{Regret}} = D(O_{\alpha^*}\tau_0,\tau_0)-D(O_{\alpha_{min}}\tau_0,\tau_0)$$

with $D$ = the draft's continuous edit cost (joint-space or capsule-height drop). Report it in mm
of head/torso drop and as a fraction of $\alpha^*$. 3 rollouts × 4 sources ≈ 0.2 GPU-h.

*Prediction:* at hard = window centre (ξ=0.5), $\alpha_{min}\approx 0.5$–$0.75\,\alpha^*$; regret
is not zero. Then **the design lever is ξ**: placing hard at ξ→0 drives regret → 0 at the cost
of tracking margin. One extra family at ξ=0.2 per source (≈0.06 GPU-h each) gives the paper a
regret-vs-ξ curve. This is the single most direct answer to "the model just put an object next
to the motion" — the object is placed *as tight as physics allows*.

### E15 — Seeds as random effects  (cheap, mandatory for C2/C4)

For every verified family (existing 12 pairs + E12 additions), rerun the 2×2 at 3 sim seeds
with geometry seed **fixed** (separate the two seeds in the manifest — E10 conflated them).
≈ 12 × 3 × 4 × 0.015 ≈ 2.2 GPU-h. Report **per-source, per-seed** pattern survival, never a
pooled rate. Prediction: ≥80% of (source, archetype, seed) cells reproduce the pattern; failures
are drift-without-contact (the E10 mode), not context contact.

### E8 — Close the crossed matrix  (prerequisite for E5)

Transfer `hanging_panel` to `cf_005_056`, 089, 090 with easy/context recertification before
hard. ≈ 0.2 GPU-h. `q_LFH_conditional_v1` already ranks it first — treat that as prioritisation
only, as the report says.

### E5 — Learning value  (one shot; go/no-go on day 12)

*Gate to run:* ≥8 verified sources with ≥3 common archetypes, all E15-replicated.

*Design:* as in `E5_LEARNER_PROBE_DESIGN.md` — four equal-budget samplers (q_LFH critical,
uniform-feasible, DCS-ranked, visual-only), held-out source × held-out archetype, 5 training
seeds, the already-validated selector harness, primary metrics counterfactual choice accuracy /
unsafe rate / unnecessary-adaptation rate / success-minus-cost. Register the minimum worthwhile
effect after a power calculation at the attained N.

*Scope decision you must make now, in writing:* the draft's claim 5 currently **excludes
generated/LFH families**. For an LFH paper that exclusion is self-defeating — the LFH families
*are* the corpus. Register a scope amendment: "E5 for the ICRA submission is run on LFH-verified
families; the SweepCF claim-5 pre-registration (mined families, 24–30 bar) is unchanged and is
not what this paper reports." Two pre-registrations, two papers, no contamination.

*Input modality:* run the privileged-geometry selector first (it is what the harness validates).
If ≥4 days remain, add an ego-depth variant from the Isaac renders; if not, state plainly that
the learning result is about **data construction**, with perception held privileged, and that
the ego-depth version is the next paper. Reviewers accept a declared limitation far more readily
than a rushed perception model.

### E13 — Realism, scored not asserted  (half a day, no GPU)

Do not run a user study. Do two cheap things:
1. **Dimensional plausibility:** each archetype carries a published dimension prior (door
   lintel 2.0–2.1 m, shelf plank 0.3–0.6 m deep, I-beam flange 0.1–0.3 m …). Report the fraction
   of generated scenes whose *non-binding* dimensions fall inside the prior. This is $r_0$.
2. **Context density:** number of keep-out-certified context objects per scene and the
   CPU minimum clearance distribution (E10b's 288.78 mm becomes a histogram).

### E14 — Lateral, time-boxed to 2 days  (optional; write the negative either way)

D2-012 introduced exact `lateral_gap_reach` and the context-recertification rule that E3 lacked.
One registered retry on the best CAL1/CAL2 arm-tuck source, 0.10 m exposure, recertify easy in
context before hard. If it passes: one lateral family, reported as n=1. If it fails: C7 is a
finding — *for a frozen tracking controller, lateral faces alter the executed path before
contact, so lateral hallucination requires controller-in-the-loop recertification that
overhead does not.* Either outcome is a paragraph. Do not spend a third day.

### Not now

- Learned trajectory encoder replacing the kernel `q_LFH` — the report's own model-selection
  criteria (nested LOMO, calibration, support violation) need ≥10 sources first.
- Multi-binding scenes; floor/`ground_support`; dynamic obstacles; the inverse operator.
- MuJoCo as a physics oracle. Keep it as the cross-render figure generator.

---

## 4. Three-week schedule

| days | GPU (contended h) | work | gate |
|---|---:|---|---|
| 1–3 | ~2.5 | **E12** generate + ladder screen; register E11/E15/E8 | ≥8 new sources? |
| 3–5 | ~1.5 | **E11** regret ladder on existing 4 + ξ=0.2 families; **E8** hanging_panel | regret curve exists |
| 5–8 | ~3 | **E15** seed replication over all verified pairs; verify new E12 sources' 2×2s (≈0.06 each) | per-seed survival table |
| 6–8 | 0 | **E13** realism scoring; refresh C1 yield tables over all sources; figures 1–4 | — |
| 8–9 | ~0.5 | **E14** lateral box (optional) | pass/negative written |
| 10–12 | — | **Go/no-go on E5** at attained source count | decide |
| 12–17 | ~2–4 | **E5** if go; otherwise expand E11/E15 and write | — |
| 15–21 | 0 | write, internal review (`/ars-reviewer`), freeze evidence hashes | submit |

Total ≈ 10–12 contended GPU-h. The constraint is your attention and the register discipline,
not compute.

---

## 5. Paper skeleton (6+n pages, ICRA)

1. **Introduction** — the failure mode (decorated scenes: image changes, trajectory bit-identical);
   the inverse problem; why a single motion is non-identifiable; the pair fix. Figure 1: the same
   nominal/crouch pair, the closed-form window drawn on the reach profile, three archetype
   realisations, and the 2×2 physics verdicts (E7c/E10b renders already exist).
2. **Problem formulation** — your $\mathcal S_{\epsilon,\delta}$ with the table from §1.1
   mapping each term to a measurement; lexicographic $J$; the 2×2 as the finite membership test.
3. **Related work** — LfH, LfLH/Dyna-LfLH, LfH-CP (critical configurations), scene generation
   for embodied AI (Holodeck/ProcTHOR/Infinigen-style), humanoid motion-tracking controllers
   (SONIC etc.), counterfactual data generation. Position: *we do not learn a scene generator;
   we compute identifiable support and learn only the proposal over it.*
4. **Method** — keypoint critical set; closed-form window; finite exposure; $q_{LFH}$ factorised
   as binding atom × context; tier discipline (screen → capsule keep-out → physics); the frozen
   controller as decoder.
5. **Experiments** — C1 yield; C2 physics agreement stratified by source/archetype/seed (E15);
   C3 necessity + regret (E11) with the regret-vs-ξ curve; C4 context/archetype transfer incl.
   the E10/E10b seed lesson; C5 delivery ladder (E12) and CAL1–3; C7 lateral negative; C6 (E5) if
   it ran.
6. **Limitations** — overhead only; operator-constructed pairs (no inverse operator); privileged
   perception in E5; seeds as random effects with small n; one controller.

Figures to make from existing assets: E10b Isaac + MuJoCo cross-render (side-by-side),
`route-map-2d.png`/`scene-map-3d.png`, `direction-support.png` (make its caption say *only −z is
verified*), `proposal-distribution.png`. New figures: delivery ladder (E12), regret vs ξ (E11),
per-seed survival heatmap (E15).

---

## 6. Rules that the last two weeks re-taught (keep them)

- Register before spend; record void attempts; infra failure ≠ rejection.
- Physics is the verdict; geometry, `D_φ`, `q_LFH`, and MuJoCo never re-grade.
- Accepted-in-context, zero-contact executions are the only certificates for a retry coordinate.
- Fix the geometry seed and the sim seed separately; report per-seed.
- One binding obstacle per scene until per-obstacle ablation exists.
- Causal-reference artifacts and deployable (retimed) artifacts never share a table.
- Quote source counts by the pre-registered definition (scene instances ≠ families).

---

## 7. If I had to pick one thing

Run **E12 today**. Every claim in the paper gets stronger with sources, the E9a failure mode is
a fixed-amplitude problem the ladder is designed to route around, and the result of the ladder
(accepted-α per motion) is itself the cleanest evidence yet that the controller, not the
geometry, defines what a humanoid scene can explain.
