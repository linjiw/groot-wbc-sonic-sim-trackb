# The Estimator Is the Curriculum:
## Frontier Sampling and Failure Recycling for Likelihood-Based RL

*Draft. Experimental tables in REPORT.md; proofs in curriculum_maxrl/PROOFS.md;
all numbers reproduce from this repository.*

---

## The 30-second version

> When you post-train with verifiable rewards, your rollout compute is wasted
> twice. It's spent on prompts the model has already mastered — nothing left
> to learn — and on prompts it can't solve at all, where every rollout fails
> and group-based estimators emit exactly zero gradient. We measured the
> damage: under uniform sampling, **65–75% of rollout groups produce no
> learning signal**.
>
> Our fix needs no new machinery, because the fix is already inside the
> estimator. We prove that MaxRL's expected learning signal on a prompt is
> exactly **2·(pass@N − pass@1)** — the probability the prompt is solvable
> within N attempts *but not within one*. That formula **is** a curriculum:
> sample prompts by it (a Thompson posterior makes it practical) and the
> first kind of waste disappears. For the second kind — all-fail prompts —
> no sampling rule can help, because there is nothing to sample toward. So
> we recycle: a failed rollout is a verified success *for the goal it
> actually reached*, and relabeling dead groups manufactures exact learning
> signal from compute you already spent.
>
> The combination beats an **oracle** teacher that knows every true pass
> rate — because the oracle can only allocate the signal that exists, while
> failure recycling creates signal that didn't. One sentence: **read the
> curriculum off the estimator's own algebra, and salvage what the estimator
> throws away.**

---

## 1. Why this direction

MaxRL (Tajwar, Zeng, et al., 2026) showed that standard RL on binary rewards
optimizes only a first-order approximation of maximum likelihood, and that a
one-line change — normalize advantages by the group's success count instead
of its size — recovers a truncated-ML objective whose weight function
w(p) = (1−(1−p)ᵀ)/p pours gradient into hard, low-pass-rate prompts.

Reading that weight-function view raises an obvious question with a
non-obvious answer:

> *If the objective already reweights toward hard prompts, is there anything
> left for a curriculum to do?*

Either curricula are redundant with likelihood-based RL (a useful negative),
or they do something gradient reweighting cannot (a useful positive). The
answer turned out to be sharply the second, for a structural reason:
**weights act on prompts after they are sampled and only when at least one
rollout succeeds.** However w(p) is shaped, it cannot rescue a prompt whose
group came back all-fail (the group is dropped — that drop is what makes the
estimator unbiased, Theorem 2 of the paper), and it cannot un-spend the
compute burned re-confirming mastered prompts. The curriculum's true job is
therefore not "emphasize hard prompts" — the objective already does that —
but to manage exactly the two regimes the estimator is blind to.

This reframing is the project's origin: **the estimator's blind spots define
the curriculum's job description.** Everything else follows from taking that
sentence literally.

## 2. Three insights, one algorithm

**Insight 1 — the estimator only learns from successes.** MaxRL's Theorem 1
says the ML gradient is the average score function *conditioned on success*;
the estimator implements this by averaging over successful rollouts only.
Consequence (a): its behavior on a prompt is a function of the success count
K alone, so its expected signal has a closed form. Consequence (b): failed
rollouts are pure waste — *unless something turns them into successes*.

**Insight 2 — the closed form is a ZPD functional (Proposition 1).**
Conditioning on K and telescoping, the expected total advantage magnitude a
prompt receives from a group of N rollouts is exactly

    E[Σ|w|] = 2·(pass@N(p) − pass@1(p)) = 2·((1−(1−p)ᴺ) − p),

strictly concave with peak p\* = 1 − N^(−1/(N−1)) ≈ ln N / N. This is a
zone-of-proximal-development functional — zero on mastered prompts, zero on
unreachable ones, maximal on "solvable with effort" — and the estimator
computes it *implicitly on every batch*. A curriculum built on it is not a
heuristic bolted onto MaxRL; it is MaxRL's own bookkeeping, surfaced. Three
corollaries fall out of the same algebra:

- RLOO's expected signal is exactly 2p(1−p) — the "learnability" objective
  of Rutherford et al. (2024). The learnability-curriculum literature and
  the estimator algebra are one object seen from two sides (Prop. 4).
- MaxRL concentrates ≈(N−1)× more expected signal than RLOO on frontier
  prompts as p→0 (Prop. 5) — the finite-sample mechanism behind the paper's
  "extracts more learning signal" observation, and the reason a frontier
  curriculum is *safe* with MaxRL specifically.
- Optimal rollout allocation across prompts is greedy water-filling on the
  marginal p(1−p)ᴺ — the probability the next rollout is a group's *first
  success* (Prop. 3).

**Insight 3 — failures are recyclable, and exactly so (Proposition 6).**
Hindsight Experience Replay meets Theorem 1: if the estimator learns only
from successes, manufacture successes. A failed trajectory is a verified
success for the sub-goal it actually achieved; relabeling a dead group to
that sub-goal and applying the same success-conditioned weights yields — we
prove — the ML gradient of the relabeled task under a shifted conditional
law, which is *exact* when the conditional laws match and empirically
indistinguishable from unbiased fresh groups where they do (measured
per-group cosine 0.956 vs 0.958 against the true gradient; the mean
relabeled gradient reaches cosine 1.000). Two contracts keep it exact in
practice: relabeled successes must be true successes under the env's own
verifier, and goal-conditioned trajectories must have their conditioning
rewritten to the achieved goal (skipping the rewrite makes hindsight
actively hurt — we measured the cost).

**The algorithm (FrontierMax).** A decayed Beta posterior tracks each
prompt's pass rate from observed group outcomes; Thompson sampling scores
prompts by u(p̃) = (1−(1−p̃)ᴺ) − p̃, concentrated as u^γ (γ≈4 when tasks share
skills — learning compounds; γ=1 on flat pools) and mixed with a 10% uniform
floor; live groups train with unmodified MaxRL advantages; dead groups are
densely relabeled to their achieved sub-goals. The estimator is never
modified, so every unbiasedness result of the base paper carries over.

## 2b. The three channels (how to think about the method)

Everything the method does flows through three channels, and every experiment
we ran gains its effect through exactly one of them:

**Channel 1 — waste avoidance (the teacher).** Don't roll out where the
estimator will emit nothing. Worth a consistent but bounded +0.05–0.08 AUC —
bounded by the oracle ceiling, because allocation can only redistribute
signal that exists (a perfect sampler collects just 0.4% more advantage mass
than our posterior).

**Channel 2 — signal creation (hindsight).** Manufacture verified successes
from failures already paid for. The only channel that breaks the oracle
ceiling and the only one that scores in frontier-heavy regimes (0 → 0.98).
Its gain is proportional to how much a relabeled skill can *compound*:
+0.22 AUC on fixed task sets, +0.01 on one-shot task streams — the single
most important regime variable we found.

**Channel 3 — objective safety (MaxRL weighting underneath).** Channels 1–2
are not objective-agnostic add-ons: the identical teacher grew coverage under
MaxRL in every seed and amplified GRPO's collapse in every seed. The
objective decides whether a curriculum is safe at all.

One line: **the teacher allocates, hindsight creates, the objective decides
whether either is safe.** The regime map, practitioner playbook, and graded
claim inventory live in EVIDENCE.md; the interactive version of this section
(a live frontier-walk simulation) is on the project site.

## 3. What problem this actually resolves

**Compute allocation in RLVR.** Rollout generation dominates the cost of RL
post-training, and on hard task distributions most of it buys nothing. Prior
fixes either pay for the waste differently (DAPO's dynamic sampling redraws
until a live group appears — the discards still cost GPU-hours), or gate on
heuristic difficulty bands with their own hyperparameters (ADARFT), or
target learnability p(1−p) — the right instinct, but the N=1 shadow of the
real functional. Deriving the rule from the estimator's algebra gives the
band, its width, and its compute-scaling (ln N/N) with **zero new
hyperparameters**: the rollout budget N you already chose *is* the
curriculum knob.

**Coverage collapse, and a compatibility warning.** The field's two favorite
post-training tools are GRPO and difficulty curricula. We show they are
**actively incompatible**: a frontier curriculum amplifies GRPO's pass@k
collapse (pass@8 0.332→0.269 under the teacher vs 0.351→0.312 uniform, every
seed) because GRPO's inverted weight function was quietly *maintaining* easy
prompts, and the curriculum removes that maintenance. The same teacher grows
coverage under MaxRL in every seed (0.316→0.348). Curricula don't fix
objective-level pathologies — they magnify them. If you run a curriculum,
you need likelihood-style weighting underneath it.

**The information ceiling of sampling.** Our cleanest conceptual result: an
oracle teacher that knows every true pass rate — the best possible sampler —
reaches AUC 0.851 on our testbed, yet collects only 0.4% more advantage mass
than the Thompson teacher. Perfect allocation is nearly saturated; the
remaining gap is tracking latency. Then failure recycling **breaks the
ceiling entirely**: the full stack reaches 0.890, above the oracle, and in a
frontier-heavy regime (max pool pass rate 10⁻⁵) where uniform, DAPO, and
even the plain teacher all score *exactly zero*, the full stack reaches
0.98 — the missing curriculum below the pool is invented from failures,
ignites learnability within ~400 groups, then goes silent. Signal creation
beats signal allocation, categorically.

## 4. What you get (stated plainly)

Real, measured:
- **22–35% more optimization steps per wall-clock hour** at matched compute
  (dead groups avoided; frontier rollouts also terminate earlier), with the
  gains landing on frontier tasks specifically.
- **Coverage that grows instead of collapsing** (pass@k up in every seed
  where GRPO's falls).
- **Inference-time sampling efficiency that grows with difficulty**: at
  matched training compute, our checkpoint needs up to 11× fewer samples
  than GRPO's to hit target coverage on the hardest evaluated level
  (1.2×/2.7×/11× at levels 2/3/5) — the MaxRL paper's headline pattern
  (2.3–19.2×, their Fig. 5) reproduced at 1.26M scale with the teacher on
  top. GRPO's coverage curves also *flatten* at large k (saturating at
  0.88/0.56 where ours reach 1.00/0.62): collapse expressed in inference
  currency — extra samples stop helping.
- **Frontier-heavy regimes go from impossible to solved** (0 → 0.98 at equal
  compute).
- **Better than oracle allocation** when recycling is available (0.890 vs
  0.851).

Quieter, but arguably worth more:
- **No difficulty-band hyperparameters.** The band is derived; N sets it.
- **A safety diagnosis, not just a method.** The same algebra that builds
  the teacher tells you when a curriculum will backfire (GRPO). Negative
  knowledge that saves other people's compute.
- **Conceptual compression.** Learnability curricula, DAPO-style filtering,
  and HER stop being separate tricks: they are the N=1 slice, the sampling
  shadow, and the success-manufacturing complement of one identity.
- **Free telemetry.** The teacher's posterior is a live difficulty map of
  your task pool — mastered/frontier/dead fractions per step at no cost.

**A gym case study — MountainCar's flag, and where curricula actually act.**
Training on the flag alone (the standard sparse-reward setup) scores exactly
0.000 — the classic exploration wall. A positional curriculum with a
*shared* goal-conditioned policy breaks it: uniform mixing reaches flag pass
0.889, the teacher 0.944, and the full stack **1.000 in every seed** at
equal compute (150 steps, weak tabular policy). The instructive failure: the
same curriculum with *per-bin* policy parameters never reaches the flag —
curricula operate through shared parameters, and difficulty bins must share
the policy or there is no channel for competence to transfer. That is the
gym-scale version of our maze-size generalization cliff, and it is the first
thing to check when a curriculum "doesn't work."

## 5. Evidence in brief

Three testbeds, same ordering everywhere (uniform < teacher <
teacher+hindsight): a CPU skill-chain with exact gradients (5 seeds; where
all seven propositions are also Monte-Carlo verified), a 1.26M-parameter
transformer on procedurally generated mazes at matched wall-clock (>20 runs;
champion = teacher + dense hindsight at 0.252 ± 0.005 final / 0.229 ± 0.009
AUC over 3 seeds vs 0.230 ± 0.015 / 0.211 ± 0.011 uniform, paired deltas
positive in every seed), and real Gymnasium environments through the
env-agnostic `frontier_rl` package (MountainCar flag 0.000 → 1.000;
CartPole survival curriculum). Honest per-regime accounting: on the
infinite-data maze, dense hindsight's reliable gain is learning speed and
never being worse — its final-eval edge over the plain teacher is small
(one-shot mazes can't compound salvaged skill), while on fixed task sets
(CPU, MountainCar) it is decisive. Three CPU pre-registrations were tested
on GPU: two transferred, one (γ concentration) did not — and the ODE model
predicted which. Two further honest negatives (adaptive truncation order,
learning-progress teachers) are documented with diagnoses. Full tables:
REPORT.md.

## 6. Limits we know about

Deep frontiers at fixed budget remain uncrossed on the maze (level 6+ at
distance ≥16 stays ≈0 in 2400 s; a 4× duration run is executing). LLM-scale
transfer is implemented (drop-in verl integration) but blocked on multi-GPU
hardware; the data-scarce GSM8K regime is where fixed-pool recycling should
compound hardest. The teacher assumes a finite task pool; streaming/
procedural sources need the parametric-posterior variant (planned). And the
exactness of recycled gradients is a property of the environment's relabel
map — math and mazes admit exact relabels; noisier verifiers will land
between our toy (+0.22 AUC) and maze (+0.01) endpoints.

---

*Repo: https://github.com/linjiw/curriculum-maxrl · Site:
https://linjiw.github.io/curriculum-maxrl/ · Base: MaxRL (arXiv:2602.02710)*
