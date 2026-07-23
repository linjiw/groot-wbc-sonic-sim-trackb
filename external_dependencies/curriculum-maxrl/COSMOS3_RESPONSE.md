# Response: Curriculum-MaxRL × Cosmos3 Self-Verified Frontier RL

*From the curriculum-MaxRL side, 2026-07-23. Answers keyed to your §5 asks;
Part II is the design plan you requested. We verified your reading of our side
against our repo — your §1.3 mapping table (TaskSpace ↔ policy server + vector
env, relabel ↔ dynamics modes, update ↔ weighted flow step) is the one we would
have written, and your §2 correspondence table is accurate. Three of your calls
we endorse explicitly before anything else: making the self-verifier's
poison-rate measurement Pilot 0 (it is the load-bearing number, and "valuable
even as a negative result" is exactly right); the "sim oracle as measurement
instrument, self-verified variant as scientific object" framing (that sentence
should survive into the paper); and refusing the three broader novelty claims
your review refuted. This is the most careful external read of our work we've
received.*

*One headline before the numbered answers, because it upgrades your Q1: your
Phase-1 "weighted RFT" framing is not a compromise. The positive part of the
MaxRL weights is itself an unbiased estimator — of the pass@k tail objective —
and its expected mass is exactly our teacher utility. Details in Q1; it means
Phase 1 keeps more of the theory than your proposal claims.*

---

## Part 0 — Calibrations to the proposal itself (before the asks)

Mostly your document is right about us. Five adjustments, ordered by how much
they change your plan:

**0.1 — The cold-start ordering: hindsight carries the few-demo regime,
the teacher earns its keep only after ignition.** Your §2 item 1 argues "the
compute economics of the teacher are better in robotics." True in
mixed-difficulty pools — but few-demo LIBERO-Long is our *frontier-heavy*
regime, and there the measured result is stark: uniform, DAPO, **and the plain
teacher all score exactly 0.000** (V5, max pool pass rate 10⁻⁵ — reallocating
among unlearnable tasks cannot help; our teacher's own code falls back to
uniform when all utilities are ~0). Only hindsight scores (0.928 AUC / 0.981
final). The traced mechanism: hindsight relabels dead groups into sub-goals
below the pool, ignites in-pool learnability within ~400 groups, then goes
nearly silent while the teacher takes over allocation. So sequence your claims
accordingly: in the few-demo protocol, **hindsight is load-bearing from step 0;
the teacher's 22–35% efficiency gain materializes after ignition.** Your §3
core claim already credits the combination — just don't let §2's teacher
economics paragraph promise teacher-alone wins on few-demo LIBERO-Long; our
data says teacher-alone ≈ uniform there, and a reviewer who runs that arm will
find it. (It's also the right *prediction* to preregister: teacher-alone ≈
uniform on few-demo, teacher ≫ uniform after warm start or on balanced pools.)

**0.2 — Arm-count arithmetic: your (task, init-state) pool is starved at the
matched budget.** 2,500 trajectories/suite at N=8 is **312 groups**; at N=16,
156. Over 90 tasks × even 5 init bins = 450 arms, that is <1 group per arm —
the posterior never localizes the frontier (our README guidance: ~8–30 bins,
a few groups per bin; SONIC's F3 and our V3 agree that *nothing* discriminates
in a starved regime). This doesn't kill the init-state idea — see Q2 for the
version that works — but "hundreds of arms" (§3A) should not survive into the
Phase-1 config.

**0.3 — Flow-GRPO is the collapse configuration.** Your §3B route table lists
Flow-GRPO/Flow-SDE as a Phase-2 alternative to ReinFlow. Take the **machinery**
(ODE→SDE conversion, per-step Gaussian chain likelihoods) if it ablates well —
but do not take the **weighting**. GRPO's std-normalized group advantages under
a frontier teacher is exactly our H6 reversal (pass@8 0.332→0.269, every seed,
easy-retention lost). If Phase 2 ships any denoising-MDP route, the group
weights on top must stay MaxRL's success-conditioned 1/K form. One sentence for
your route table: *Flow-SDE the chain, MaxRL the weights.*

**0.4 — Inverse-dynamics consistency is a prefilter, not a verifier.** ID-mode
recovering the executed actions certifies the model *understands the dynamics
of the trajectory* — it says nothing about whether a goal predicate holds. Use
it as the cheap OOD gate before the reasoner-mode predicate query (which is the
actual verifier), and score Pilot 0's precision/recall on the predicate query
alone, with and without the ID prefilter, so you know what each stage buys.

**0.5 — Good news your proposal undersells: you are in the compounding
regime.** Our single most important regime variable is whether relabeled skill
can compound: fixed task set +0.22 AUC, one-shot task stream +0.01. LIBERO is a
*fixed pool* whose sub-goals recur across tasks (`open(microwave)` appears in
many task conjunctions), so a relabeled success transfers to every task sharing
the predicate — this is the CPU-skill-chain end of our bracket, not the maze
end. Your §4 risk table treats hindsight's gain as uncertain within our
[+0.01, +0.22] endpoints; the regime map says you should expect the high end,
and that's a preregisterable prediction, not a hope.

---

## Q1 — Estimator framing under the FPO surrogate: your RFT framing is better than you think; here is the exact statement to use

**Rank: ship Phase 1 as weighted RFT with the *positive part* of the MaxRL
weights — `w_i = 1/K − 1/N` on verified successes, 0 on failures — and claim
the following, which is exact, not order-preserving:**

1. **The positive part is itself a principled estimator.** MaxRL's
   variance-reduced form (paper eq. 10) is `w_i = r_i/K − 1/N`; the failure
   term `−(1/N)Σ S_i` is a zero-mean control variate (unconditional score mean
   is 0), so dropping *all* weights' baseline gives the paper's own eq. 9 —
   unbiased, higher variance. Dropping only the **negative** weights gives
   something sharper. Per prompt (exact likelihoods):

   ```
   E[Σ_succ (1/K − 1/N) S_i] = (w_T(p) − 1)·∇p = Σ_{k=2}^{N} (1/k) ∇pass@k
   ```

   i.e. the positive-part update is an **unbiased estimator of the pass@k tail
   objective** — maximum likelihood minus its first-order (REINFORCE) term. Its
   cross-prompt weight `w_T(p)−1` vanishes at p→1 and grows like N−1 as p→0:
   the coverage-first behavior you want, in the currency (pass@k) your
   evaluation already speaks. This is a *feature* for a robotics paper whose
   headline is coverage, not a degraded MaxRL.

2. **P1 survives untouched on the sampling side.** The expected mass of the
   positive-part weights is `E[Σ w⁺] = E[(1−K/N)·1{K≥1}] = pass@N − pass@1`
   — *exactly* the teacher utility (we re-verified by MC while drafting this,
   200k trials, matches to 4 decimals). So the teacher's algebra governs the
   Phase-1 update exactly, not approximately. The only thing the surrogate
   degrades is the update direction (surrogate score vs true score), not the
   curriculum.

3. **Two practical properties you get free.** (a) All-pass groups self-retire:
   K=N ⇒ every weight is 0 — mastered tasks stop producing gradient without
   any filtering rule. (b) Hindsight groups need no special casing: relabeled
   K′ successes get the same formula on the relabeled task; K′=1 gives weight
   ≈ 1−1/N ≈ our `hindsight_scale=1.0` natural K=1 group weight.

4. **Never negative-weight the CFM loss.** Maximizing flow-matching loss on
   failures is an unbounded objective (likelihood → −∞); this is the flow-side
   reason to drop the failure terms, on top of the control-variate argument.
   If you later want a contrastive term, bound it explicitly and treat it as a
   new method, not as MaxRL.

5. **You don't need FPO's ratio machinery in Phase 1 at all.** FPO's
   loss-difference surrogate exists to build PPO-style clipped ratios. Weighted
   RFT needs only per-sample weights multiplying the CFM loss — which is your
   existing trainer plus one multiply. Save FPO for a Phase-2 arm if you want
   clipped off-policy reuse; default Phase 2 to ReinFlow chain likelihoods as
   you proposed.

6. **The unbiasedness analysis we'd want for the surrogate case is empirical,
   and cheap: a V1-style fidelity probe (add it to Pilot 0).** On ~50 mixed
   groups, compute (a) the weighted-CFM update direction and (b) the ReinFlow
   exact-chain PG direction on the same data, and report the cosine. This is
   exactly how we validated hindsight gradients (V1: relabeled cosine 0.956 vs
   fresh 0.958, mean → 1.000) — one number that either licenses the Phase-1
   framing or tells you to go to chain likelihoods immediately, before three
   weeks of training depend on it.

**What's load-bearing in our framing, ranked:** (1) success-conditioned
1/K-style weighting with the K=0 drop — this is what makes the teacher utility
exact and what H6 shows a curriculum *requires*; (2) posterior hygiene
(relabels never touch the teacher, Q3); (3) the two hindsight contracts.
**Not load-bearing for Phase 1:** exact policy-gradient unbiasedness. Frame it
as above and no honest reviewer can call it "MaxRL in name only" — the
estimator, the utility identity, and the self-retirement property all carry
over exactly.

One flow-specific warning that has no LLM analogue: **group-based anything
requires within-group variance.** With a deterministic simulator and fixed
init state, p is purely the diffusion sampler's stochasticity. If rollout-time
sampling is near-deterministic (low noise, aggressive guidance), K collapses
to {0, N}, the band empties, and both the teacher and the estimator starve.
Check the within-group action variance of the released checkpoint *first*
(this can be Pilot 0's first afternoon), and treat sampler
noise/temperature as a first-class exploration knob, as SimpleVLA-RL did.

## Q2 — Group semantics and (task, init-state) arms: task-level arms first; init-state bins are for manufacturing frontier, not for day-0 granularity

1. **Phase 1: 90 task-level arms, N=8.** The budget arithmetic (0.2) forces
   it: 312 groups over 90 arms is ~3.5 groups/arm — enough for a decayed Beta
   to localize a frontier band, not enough for any finer partition. N=8 puts
   the utility peak at p\* ≈ 0.26 with the >50%-utility band covering p ∈
   [0.06, 0.68] (N=16: peak 0.17, band [0.03, 0.61]) — at 220–520 sim steps
   per episode, N=8 buys twice the arm coverage per budget and targets a band
   better matched to few-demo pass rates. Unlike SONIC's reset-stream setting,
   you have *real episodic groups*, so use `utility="advmass"` with your real
   N — the band is then derived, not tuned. Thompson sampling is fine here
   (small pool, episodic groups — exactly the validated regime; SONIC's
   determinism guardrail doesn't apply to you). Defaults from `teacher.py`:
   decay 0.7, floor 0.1, and **γ=1** — libero_90 is a broad heterogeneous
   pool, and γ=4 failed to transfer even to the maze's 13 *ordered* levels
   (F2; the ODE model predicted it: no shared unlock structure ⇒ no
   compounding to concentrate on).

2. **Init-state bins enter when a task masters, not before.** The elegant use
   of your per-episode init-state control is *frontier manufacturing*: when a
   task's posterior saturates (p̂ > 0.9 ⇒ u ≈ 0 ⇒ the teacher retires it),
   split it into init-state-difficulty bins (e.g. distance/clutter quantiles)
   and hand the teacher the sub-arms — new frontier from mastered material,
   the same role init-state randomization plays in your Phase-3 imagination
   plan but verified in the real sim. This keeps the live arm count bounded
   (~90 + splits) instead of starting at 450.

3. **If/when you do want bins under one task, share hierarchically — the cheap
   version, not the full hierarchical Beta.** Shrink each bin's posterior
   toward its task's: `α_b = λ·α_task + s_b`, `β_b = λ·β_task + f_b` with
   λ ≈ 0.3 as pseudo-count fraction — one line, and it's the same move as the
   kernel evidence-sharing we recommended for SONIC's speed-factor families.
   If your init-state axis is continuous and ordered by difficulty,
   `StreamingFrontierTeacher` (kernel posterior + isotonic projection) already
   exists and validated *identical* to discrete bins (AUC 0.684 = 0.684);
   don't build a new mechanism.

4. **Group semantics fine print:** a group must be N episodes of one arm. If
   the arm is (task, bin), fix the *same* init state for all N episodes of the
   group when possible (your `_get_initial_state` supports it) — pooling
   heterogeneous inits into one arm is tolerable but note the direction of the
   error: u is concave, so utility-at-pooled-p̂ *over*states the bin's true
   expected mass (Jensen). Mild at Phase-1 scale; worth a sentence in the
   methods section, not a mechanism.

## Q3 — Hindsight goal-space design: dense-to-deepest-predicate is right; here are the contracts instantiated, plus the hygiene confirmation you asked for

1. **Posterior hygiene: confirmed, and it's measured, not theoretical.**
   Relabels must never update the teacher's posterior. We tested the
   alternative twice: V4 (CPU, pre-registered "safe but redundant, risk =
   optimism inflation") and the GPU A/B/C config C, which reproduced the
   predicted signature exactly — posterior p̂ 0.81 vs true eval 0.47 at level
   2, sampling pushed deeper prematurely, worse final. In your setting there's
   a tempting trap we want to flag explicitly: the relabeled target (e.g.
   "open the microwave") may literally *be* another arm in libero_90. Even
   then, don't credit it — the trajectory was sampled under a *different*
   task's conditioning, so it is evidence about the wrong conditional law
   precisely in the sense of P6. The telemetry that catches violations:
   teacher p̂ vs held-out eval pass rate per arm (we ship this comparison in
   the maze harness; port it).

2. **Contract 2 (conditioning rewrite) at VLA scale: use templates, not the
   reasoner.** LIBERO's closed BDDL predicate vocabulary is the easiest
   possible version of the rewrite — exploit that by mapping each achievable
   predicate/conjunction to a *fixed canonical instruction string* (one
   template per predicate, written once, reviewed by a human). Do not
   free-generate the rewritten instruction with the AR tower: it injects
   instruction-distribution drift into exactly the channel the contract
   exists to protect, and it makes the rewrite unverifiable. Our measured cost
   of skipping/botching the rewrite: hindsight flips from best (0.703) to
   actively harmful (0.600 < teacher-only 0.658). Unit-test the rewrite the
   way you'd unit-test a parser: predicate → template → (new language goal,
   new task_id), asserted against the BDDL ground truth. This is also why
   `Policy.update` takes `task_id` explicitly in our interface — the relabeled
   conditioning is rebuilt, never patched.

3. **Relabel target: deepest achieved sub-conjunction along the task's own
   goal, plus cross-task predicate credit.** Your instinct (dense, deepest
   achieved predicate — like our skill-chain deepest-prefix) is the validated
   variant (dense hindsight is the GPU champion; sparse tied baseline). Two
   refinements: (a) prefer sub-conjunctions of the failed task's goal over
   arbitrary achieved predicates — they're the on-the-path targets that
   compound toward the sampled task; (b) where the achieved predicate matches
   *another pool task's* goal (the recurrence from 0.5), relabeling to that
   task is legitimate **as training data** (with rewritten conditioning) —
   that's the cross-task compounding channel — it's only the *posterior* that
   must not hear about it (point 1).

4. **Poison management: per-predicate precision, and an asymmetric gate.**
   Contract 1 says a relabeled success must be a true success under the
   verifier. Your ≥90% precision gate is the right shape; make it
   **per-predicate-class**, because a VLM-style verifier will be systematically
   better at `open(microwave)` than at `on(bowl,plate)`-type spatial
   relations. The action on a failing class is to *remove it from the relabel
   vocabulary*, not to lower the global gate — you keep clean hindsight on the
   predicates the verifier can see. Recall costs you only recyclable signal;
   precision costs you contract 1. And one rule that eliminates the worst
   poison class outright: **the self-verifier may only relabel to
   strictly-easier achieved sub-goals — it must never upgrade a failed rollout
   into a success of the original task.** The simulator's own
   `info["success"]` remains the sole verifier for live groups.

5. **`hindsight_scale`: start 1.0, expect a knee, watch diversity.** On the
   CPU toy the scale was monotone to 8× only because relabels are exact and
   gradients exact; at VLA scale over-weighted self-imitation entrenches the
   policy's own reachable set (the HER drift of P6's practical reading — the
   floor and live original-task groups are your anchors). Track within-group
   action variance and relabeled-fraction-per-step; if relabels stop declining
   after ignition (V5's signature is 81 relabels from draw 800 *to 3200*,
   i.e. near-silence), the recycler has become a crutch and the scale is too
   high.

## Q4 — Evaluation protocol: matched rollouts is necessary; four additions make a robotics claim clean by our standards

1. **Report two currencies: matched rollout budget AND matched wall-clock.**
   Matched-rollouts (your SimpleVLA-RL-protocol 2,500/suite) is the fair
   apples-to-apples for the estimator question — and it's the currency where
   hindsight is *free* (relabels consume zero extra rollouts; say this
   explicitly and also report update counts, so no reviewer discovers the
   extra gradient steps for you). But matched-rollouts *hides* the teacher's
   real-world advantage: frontier episodes terminate earlier and dead groups
   are avoided, which bought us 22–35% more optimization steps per hour. Our
   maze protocol switched to matched wall-clock for exactly this reason
   (fixed-step comparisons were unfair to teachers). Both numbers, one table.

2. **Coverage currency, per-seed, from day one.** Three separate times a
   metric hid what our method was doing (the "meter lesson"): likelihood-style
   training moves the distribution's tail first, and any single-sample metric
   under-reports it. Your eval loop needs success@k per task (k independent
   episodes; the robotics pass@k), not just mean success — the GRPO-collapse
   headline ablation is *invisible* without per-seed pass@k curves, and
   "coverage grows vs collapses" was 6/6-seeds systematic for us while mean
   pass barely separated. Add easy-decile retention (mean success on the k
   easiest tasks by SFT baseline) in every arm — it's the collapse tripwire —
   plus dead-group rate and teacher-p̂-vs-eval calibration as standing
   telemetry.

3. **Preregister the baseline strength.** You diagnosed SimpleVLA-RL's
   weakened-baseline headroom yourself (§1.4); don't inherit the same critique.
   State the few-demo SFT protocol (demos/task, exact checkpoints) before RL
   runs, report the full-SFT reference number next to it, and let the oracle
   arms bound the claims: **oracle-relabel** (simulator predicates) isolates
   curriculum+hindsight value from verifier quality; **self-verified relabel**
   is the transfer claim; the gap between them *is* the poison-rate cost,
   measured end-to-end. Run both arms in Phase 1 — it's the same
   infrastructure with one function swapped, and it decouples your two
   contributions cleanly.

4. **Seeds and pairing:** ≥3 seeds, paired via a shared per-seed SFT
   warmstart (our maze protocol) so paired deltas, not overlapping error
   bars, carry the claim. Our seed noise on finals was ±0.01–0.015 —
   assume yours is worse and size claims accordingly; single-seed numbers
   get an explicit caveat or stay out of the abstract.

## Q5 — verl vs cosmos-framework: your Phase-1 default is correct; keep Phase 2 on cosmos too, and share the teacher as code

**Bolt `frontier_rl`'s TaskSpace onto the cosmos-framework HTTP loop.**
Reasons, in strength order: (1) Phase 1 is weighted-CFM SFT — verl's PPO
machinery buys nothing, and your 24 GB stack is already built on the cosmos
trainer; (2) porting a 4B MoT video+action model with a lazily-loaded VAE into
veRL's rollout worker is a large engineering fight orthogonal to every claim
in the paper; (3) comparability to SimpleVLA-RL is protocol-level (budgets,
suites, metrics), not framework-level — nobody sane demands framework parity.
Our verl fork matters to you in exactly one way: the teacher there
(`verl_integration/curriculum.py`) and `frontier_rl/teacher.py` are the same
~100-line object — import the package rather than vendoring a copy, so V2b/V3
default changes propagate. For Phase 2, revisit only if you adopt FPO-style
clipped updates and genuinely want verl's PPO plumbing; ReinFlow's noise-net +
chain-likelihood loop is small enough to live in the cosmos trainer as another
loss module, next to `flow_matching.py`.

---

# Part II — Design plan (what we'd build, in order)

## II.1 The wiring (Phase 1, all pieces exist on one side or the other)

```python
class LiberoTaskSpace:                        # frontier_rl.TaskSpace
    n_tasks = 90                              # libero_90; + split bins later (Q2.2)

    def rollout_group(self, task_id, n=8):
        # one wave: n parallel episodes of one (task, init) via
        # SubprocVectorEnv + POST /predict_batch; binary info["success"]
        return GroupResult(task_id, rewards, trajectories,
                           infos=[{"final_frames": ..., "video": ...}])

    def relabel(self, group):                 # dead groups only
        preds = verifier(group)               # oracle: sim BDDL predicates
                                              # self: ID-prefilter -> reasoner query
        g_star = deepest_subconjunction(preds, task_goal(group.task_id))  # Q3.3
        if g_star is None: return None
        new_task = task_of_predicate(g_star)
        new_traj = rewrite_language_goal(group.trajectories,
                                         TEMPLATE[g_star])               # Q3.2
        return new_task, success_under(g_star, preds), new_traj

class CosmosPolicy:                           # frontier_rl.Policy
    def update(self, task_id, trajectories, weights):
        # weighted CFM step in the existing 24GB trainer:
        # loss = sum_i weights[i] * cfm_loss(traj_i | rewritten goal)
        # weights from estimators.maxrl_weights(r), CLIPPED AT ZERO (Q1)
```

`FrontierTrainer` (`frontier_rl/trainer.py`) runs this loop as-is — teacher
sees requested-task evidence only, dead groups route through `relabel`, both
contracts enforced at the interface. The one deviation from the shipped
trainer: clip `maxrl_weights` to the positive part before `Policy.update`
(one line; we'll add a `positive_part=True` flag to `estimators.py` so both
teams run the same code).

## II.2 Config, with provenance

| knob | value | why |
|---|---|---|
| group size N | 8 | band p\*≈0.26, [0.06, 0.68]; budget arithmetic (0.2) |
| utility | advmass (real N exists) | P1/P2; unlike SONIC's stream case |
| posterior | Beta rows, 90 arms; Thompson | validated regime; Q2.1 |
| decay | 0.7 | V2b (tracking > memory); revisit if rounds are long |
| floor | 0.1 | P7/V3 — staleness insurance, HER-drift anchor |
| γ | 1 | F2: γ=4 failed to transfer even to 13 ordered levels |
| weights | positive-part MaxRL, `1/K − 1/N` on successes | Q1 — mass = u(p) exactly; all-pass self-retire |
| hindsight | dense, deepest sub-conjunction, templates | GPU champion config; Q3.2–3 |
| hindsight_scale | 1.0 | natural K=1 weight; watch the knee (Q3.5) |
| relabel→posterior | **never** | V4 + GPU config C, measured |
| rollout sampler noise | tuned for within-group variance | flow-specific (Q1 warning) |

## II.3 Milestones and gates

**Pilot 0 (A10G, ~days) — three measurements, not one:**
(a) within-group variance of the released checkpoint (afternoon; if K ∈ {0,N}
always, fix sampler noise before anything else); (b) your poison-rate
measurement, per predicate class, ID-prefilter on/off (gate: ≥90% precision on
the *surviving* vocabulary — prune classes, don't lower the bar); (c) the
surrogate-fidelity probe — cosine(weighted-CFM direction, ReinFlow-chain PG
direction) on ~50 mixed groups (gate: comparable to our V1 fresh-group cosine
~0.95; if far below, Phase 1 proceeds but Phase 2's chain likelihoods get
promoted to the main line).

**Phase 1 (A10G, 2–3 wks) — four arms, few-demo LIBERO-Long, both currencies:**
uniform / teacher / teacher+oracle-relabel / teacher+self-relabel.
Preregistered predictions from our regime map: uniform ≈ teacher ≈ 0 if the
pool is truly frontier-heavy (0.1); oracle-relabel ignites (the V5 dynamic:
relabel burst → live-group takeover → relabel silence); self-relabel lands
within poison-rate of oracle. Success = the ordering
`uniform ≤ teacher < teacher+relabel` with the ignition trace visible in the
relabel-rate telemetry. This is simultaneously your Rank-3 de-risking milestone
and the Rank-1 ablation baseline, as you said.

**Phase 2 (multi-GPU):** ReinFlow chain likelihoods with full MaxRL weights
(negative terms return — legal on exact log-probs); the MaxRL-vs-GRPO-
under-teacher collapse ablation with per-seed success@k (0.3: Flow-SDE chain,
MaxRL weights); matched-budget SimpleVLA-RL comparison; init-state frontier
manufacturing (Q2.2) on mastered tasks; RoboCasa headline.

**Phase 3 gate (unchanged from yours, plus one):** imagination rollouts only
if Pilot 0 poison is low *and* forward-dynamics divergence horizon (measured
against real sim rollouts of identical action sequences) covers the sub-goal
lengths you'd relabel — imagined relabels beyond the divergence horizon
violate contract 1 by construction.

## II.4 Telemetry checklist (every arm, from step 0)

dead-group rate · relabels/step (expect burst→silence) · success@k per task ·
easy-decile retention · teacher p̂ vs eval calibration ·
within-group action variance · update counts alongside rollout counts.

## II.5 Risk additions to your §4 table

| risk | mitigation |
|---|---|
| within-group determinism (flow sampler) | Pilot 0a; sampler noise as exploration knob |
| self-imitation entrenchment past ignition | relabel-rate telemetry; hindsight_scale down; floor |
| RFT diversity collapse over rounds | keep a small demo-replay fraction in every training mix; retention metric is the tripwire |
| arm starvation (0.2) | task-level arms; bins only via mastery splits |
| surrogate misalignment | Pilot 0c cosine probe *before* Phase-1 conclusions |

---

*One closing remark, matching the one we gave the SONIC team: our depth study
found the teacher walks the frontier to the policy's execution ceiling and
then stalls — and the diagnostic separating "curriculum problem" from
"capacity problem" was per-step accuracy extrapolated geometrically. Your
analogue: per-chunk success rate on achieved sub-goals vs sub-goal chain
length. If few-demo LIBERO-Long stalls under the full stack, run that before
blaming the sampler — in our case the teacher was exonerated and the answer
was capacity. With a 4B model on 24 GB, capacity may well be where your
frontier march ends, and knowing that early changes Phase 2's hardware ask.*

*Repo pointers for your integration: `frontier_rl/interfaces.py` (the three
contracts), `frontier_rl/trainer.py` (the loop II.1 reuses),
`frontier_rl/estimators.py` (weights; positive-part flag incoming),
`frontier_rl/streaming.py` (if your init-state axis goes continuous),
`SONIC_RESPONSE.md` (the dense-PPO sibling of this document — your Q2/Q4
have cousins there).*
