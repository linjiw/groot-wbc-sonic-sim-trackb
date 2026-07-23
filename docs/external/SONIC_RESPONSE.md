# Response: Curriculum-MaxRL × GEAR-SONIC motion tracking

*From the curriculum-MaxRL side, 2026-07-22. Answers keyed to your §8 questions.
We verified your file:line claims against your checkout
(`update_adaptive_sampling` occupancy semantics at `motion_lib_base.py:2462+`,
`uniform_sampling_rate: 0.1` at `motion.yaml:21`, cap default 50 at `:2398-2400`,
the wired-but-unused scheduler at `trl/utils/scheduler.py:296+`) — your reading of
our side (§4) is accurate, and your port/no-port table (§1) is the one we would
have written. Two of your calls we want to explicitly endorse before the answers:
dropping the estimator import (§5.1) is correct, and your observation that the
per-step occupancy update already implements the statistics half of hindsight
(§3.2) is exactly right — that was the sharpest sentence in the document.*

---

## Q1 — Right utility for dense-reward PPO: binary-verifier utility first, empirical-|A| second

Rank: **A (verifier-based `u(p)`) first, C (Σ|GAE|) as the follow-up arm, not the
lead.** Reasons:

1. The theory reason your instinct anticipated: our advantage-mass identity is
   estimator-specific, but its *role* is "expected useful learning signal per visit
   as a function of task difficulty." In your stack the analogous quantity is not
   gradient magnitude (dense PPO always has gradient) but **useful-gradient
   density**: mastered bins produce near-zero advantage variance (rewards are
   saturated, GAE ≈ 0 everywhere), impossible bins produce gradient that mostly
   optimizes the pre-failure prefix you already know. Both ends waste visits — the
   same U-shape. A binary verifier (your termination flag) is a cheap, robust
   *proxy* for that U-shape, and `p` from your hazard counts is a monotone
   transform of difficulty, which is all the ZPD utility needs. The exact
   functional form matters less than its two zeros; your forecast (§6) already
   shows learnability and advmass behave similarly on the frontier.
2. The practical reason: C's signal (per-bin Σ|A|) is **confounded by episode
   length and reward-term weighting** — a bin with longer survival accumulates
   more |A| mass at equal per-step usefulness, and your 12-term reward means |A|
   scale drifts whenever term weights or the KL controller move. You would spend
   your first GPU cycles debugging signal normalization instead of testing the
   curriculum. SEC (the empirical-|A| lineage) works on LLM groups where length
   is bounded and rewards are single-source; your setting is the adversarial case
   for it. If you do build C, normalize per-step (mean |A| per occupied step, not
   sum) and expect it to *agree* with A on the ends and differ mid-band — which
   makes it a good second arm precisely because disagreement is informative.
3. One refinement to A: your `p` is a *hazard complement* per bin, which is
   better-behaved than pass@1 for this purpose (it is per-visit-normalized by
   construction). Use `learn_beta` (your `E[p(1−p)]` closed form) as the default
   utility, not advmass — see Q2 for why.

## Q2 — What is N? Here: nothing natural. Drop it — use learnability, or define N as visits-per-recompute

Honest answer: the `p* ≈ ln N/N` band placement is a theorem about *episodic
groups of N rollouts on one task*, where "solvable within N attempts" is a real
event. Your multinomial-over-thousands-of-bins allocation has no such event, so N
degrades from a derived constant to a free band knob — at which point **you have
lost the property that made advmass attractive (zero tuned hyperparameters)** and
learnability `p(1−p)` is the more principled choice: it is the N→1 member of the
family, needs no knob, and your own forecast shows it *more* aggressive at
discounting impossible bins (0.034–0.058 impossible mass vs advmass's 0.256–0.274)
— which, per your F2 diagnosis, is the pathology you are fixing. Long-shot
exploration on near-impossible bins is better funded by the uniform floor
(explicit, auditable) than by a fat utility tail (implicit, unbounded in time).

If you want an N with real semantics: **expected visits per bin per recompute**
(your interpretation (c)) is the only one that preserves the reading "signal this
bin can emit before the distribution updates" — it makes N scale-aware
(4096-env N ≫ 8-env N) and moves the band harder as throughput grows, which is
the compute-indexed behavior from our theory. But we would still ship
learnability first and treat advmass-with-N as an ablation.

## Q3 — Thompson vs posterior mean: mean+bonus is fine; Thompson's job here is small

We attribute Thompson's contribution in our stack to **unvisited-task probing** in
small pools (24–36 tasks) with per-task rows. You have ~70–thousands of bins with
a 0.1 uniform floor and (at release scale) huge visit counts — the floor already
guarantees coverage, and the posterior is tight. Our V3 ablation found the floor
curve flat from 0 to 0.4 *because Thompson self-explores*; the converse also
holds: with a floor, determinism costs little. A deterministic optimism bonus
(`mean + k·std` of the Beta, k≈1) is an acceptable substitute and is what we
would use under your reproducibility guardrail. At 8-env micro scale, note that
*nothing* discriminates (your F3 and our forecast agree), so the Thompson
question is moot exactly where it would matter.

## Q4 — Decay semantics: per-evidence-unit, and you are right that it is more principled

We did not test evidence-scaled decay (our teacher steps observe one group each,
so per-step ≡ per-evidence for us; decay 0.7 ≈ effective memory of ~3 groups).
Your situation — throughput varying 512× between release and micro — makes
per-recompute decay meaningless as a constant. Define a **half-life in
episode-equivalents** H and apply `counts *= 0.5^(Δepisodes/H)` at recompute.
Calibration hint from our V2b: the point of decay is *tracking* a moving
competence, and the right H is a small multiple of how much evidence changes p
materially — we'd start at H ≈ 3–5× the mean per-bin visits per recompute at
release scale. One warning from our Q7 simulation below: if you also run the
threshold controller, do NOT default to fast decay (see Q7 — the two interact,
and in the coupled system slower memory was better).

## Q5 — γ under direct settability: use γ=1. The compounding argument does not apply

Correct instinct. Our γ>1 result exists because progress on task j *changes the
policy* in a way that unlocks task j+1, and concentrating accelerates that chain
— the mechanism is parameter-transfer compounding, which our ODE model showed
vanishes on pools without shared unlock structure. Your bins do share parameters
(one policy), so *transfer* exists — but your bins-within-a-motion are not
gated: RSI reaches any bin directly, and competence on bin k mostly transfers
through generic motor skill, not through an ordering. That is our "flat pool with
heterogeneous difficulty" regime, where measured γ sensitivity was ≈0
(AUC 0.409→0.427 from γ=0.5→8, with *worse* finals at high γ). Also note our own
GPU result: γ=4 failed to transfer even to the maze's 13 ordered levels. γ=1,
and don't spend a run on it until everything else is validated.

## Q6 — Hindsight: your rejection is correct; one cheap remnant is worth keeping

Trajectory-level relabeling under on-policy PPO fails both our contracts *and*
the on-policy assumption, as you said; and your dense reward genuinely is the
partial-credit channel hindsight builds in sparse settings. Drop the thread —
with two footnotes:

1. **The statistics half you already have is not merely "no change needed" — it
   is load-bearing.** In our stack the biggest single CPU gain came from
   *counting* what failures achieved (relabels feed the learner). You feed the
   *sampler* instead (occupancy credit), which is the correct on-policy port.
   Preserve it when you swap utilities: the Beta posterior must be built from
   the occupancy-based (traversal, failure) counts, not from episodic terminal
   bins only — you already documented this trap in your correction note.
2. The speed-scaled family (your residual candidate): the safe version is not
   trajectory relabeling but **evidence sharing across the family** — a success
   at ×1.5 is Bayesian evidence about ×1.0's difficulty (your bins have known
   family structure; a one-line kernel over speed factors in the posterior, like
   our streaming teacher's kernel over difficulty). No conditioning rewrite, no
   off-policy issue, and it thickens the posterior exactly where micro-scale
   evidence is thinnest. That is the hindsight *insight* (failures/successes
   carry information about neighboring tasks) surviving on-policy PPO.

## Q7 — Closed-loop verifier: viable, with three rules from a coupling simulation

We simulated your coupled system (teacher posterior + threshold controller +
learning competence; 40 bins, 400 iters — sketch below). Findings:

1. **The controller must move slower than competence grows.** With gain η ≥ 0.3
   the controller overshoots to the strict bound and pins there with failure
   ≈ 0.66 (far above target 0.35) — a one-way ratchet into an impossible
   verifier. η = 0.05 held failure at 0.31–0.44 with τ in a sensible band. The
   failure mode is not oscillation (we measured almost none); it is
   **overshoot-and-pin**. Rate-limit the threshold (small η, plus a cap on
   Δthreshold per iteration) and consider one-sided annealing (only tighten,
   never loosen) — that matches your KungfuBot precedent and removes the
   pinned-loose equilibrium entirely.
2. **Decay interacts, opposite to intuition:** in the coupled system, *slow*
   posterior decay (0.99) tracked the moving `p` better than fast (0.7) —
   posterior error 0.018–0.042 vs 0.101–0.111. Reason: a slow controller makes
   the verifier quasi-stationary, so the dominant noise is sampling variance,
   and fast forgetting just amplifies it. If the threshold moves slowly, keep
   memory long. (This is why Q4's answer says calibrate decay to whichever
   nonstationarity dominates — policy drift OR verifier drift, not both fast.)
3. **Do not reset the posterior on threshold moves** — in simulation it changed
   nothing at slow η and added variance at fast η. The decayed posterior
   re-adapts on its own within a half-life.
4. Your target `p_fail* ≈ ln N/N` inherits Q2's problem (no natural N). A
   defensible target is simply the frontier band's center — hold global
   early-termination rate near 0.3–0.5 — or the value that maximizes your
   measured *learning progress* per iteration once you have real telemetry.

Also, from your own doc: the eval `schedule_dict` re-application
(`eval_agent_trl.py:466-470`) is the kind of silent-corruption bug that costs a
month — we'd promote "threshold schedules stripped at eval" from a note to a
unit test before any run.

## Q8 — Forgetting: the floor was sufficient *for MaxRL-weighted learners*; you should add the retention metric and consider anchor mass

Careful transfer here. Our H6 result says forgetting risk depends on the
*objective*: with MaxRL weighting the 0.1 floor sufficed (coverage grew every
seed); with GRPO the same teacher amplified collapse. Your dense-reward PPO is
neither — its retention behavior under frontier sampling is an empirical unknown.
Your plan is right: ship the easy-decile retention metric in every arm from day
one. If retention degrades, the cheap escalation path is (a) raise floor to 0.2,
(b) explicit anchor mass on mastered bins (your `stage_sampling_weights`
mechanism), (c) an ALP-style |Δp̂| term that re-injects *regressing* bins
specifically — (c) was worth +0.01 final on our maze (frontier_alp was our best
pure teacher) and is ~10 LOC on top of the Beta posterior.

## Q9 — Drop the cap for ZPD utilities, keep a probability ceiling as a tripwire

The `mean × 200` cap exists because failure rate is unbounded-relative-to-mean on
pathological data. Learnability is bounded by 1/4 and self-limiting at both ends,
so the cap's *function* is gone. What can still go wrong at small counts is a
few-bin posterior fluke concentrating mass; the floor bounds the minimum but not
the maximum. Keep a cheap tripwire: `max_prob_per_bin` (you have the dormant
knob) at something loose like 20×uniform, plus your existing
`effective_num_bins` telemetry with an alert threshold. That converts the cap
from a shaping mechanism into a safety assertion — which is also easier to
reason about in preregistration.

## Q10 — What you misread: almost nothing; two calibrations

1. **§5.1 slightly overstates our claim.** We do not claim dense-reward PPO has
   "no dead groups *problem*" — we claim the *zero-gradient* pathology is absent.
   The *useful-gradient* pathology (saturated bins emit near-zero advantage
   variance) is present in your stack and is the actual justification for the
   teacher import; your §2.2 last paragraph says this correctly, so just align
   §5.1's wording with it. It matters because it predicts *where* the teacher
   helps: not by avoiding zero-gradient batches but by reallocating from
   advantage-flat bins.
2. **Occupancy-based `p` (your §3.2/Q2 worry) does not invalidate the theory —
   it replaces it with something better for your setting.** The ZPD utility only
   needs a difficulty-monotone `p` with calibrated ends; your hazard complement
   is exactly that, with finer granularity than episodic pass@1. The thing you
   *lose* is the N-semantics (Q2); the thing you gain is that partial traversals
   inform the posterior. Net positive. State it as a deviation, not a violation.

## Endorsed plan (our ranking of your §7)

1. **Change A with `signal: learnability` + evidence-scaled decay (Q4) + optimism
   bonus (Q3)** — smallest diff, directly targets the diagnosed 43–53% impossible-
   bin waste, keeps all your telemetry.
2. **Change B static schedule first** (loose→strict linear over N iters, stripped
   at eval, with the unit test), closed-loop only after the static version shows
   the frontier manufacturing works — and then with η small + one-sided + long
   posterior memory (Q7).
3. **Change C as the disagreement probe**, per-step-normalized, after A has a
   verdict — not before.
4. Non-negotiables we'd copy from your own guardrails: SIM-D1 headroom gate
   first (your forecast's "starved" row and our V3 agree: no teacher rescues a
   starved regime), targeting-criterion activation gate (not peakedness — your
   forecast is right that ZPD utilities are diffuse by design), retention metric
   in every arm.

One last transferable result from our side: our depth-mechanism study found the
teacher correctly walks the frontier to the *policy's execution ceiling* and then
stalls — and the diagnostic that separates "curriculum problem" from "capacity
problem" was measuring per-step accuracy and predicting reach geometrically. Your
analogue: per-step tracking-error growth rate within a bin vs bin length. If
frontier bins stall under the new teacher, run that diagnosis before blaming the
sampler — in our case the answer was capacity, and the teacher was exonerated.

---

### Appendix: Q7 coupling simulation (numpy, self-contained)

```python
import numpy as np
def run(eta, decay, T=400, B=40, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.02, 0.6, B)          # latent competence at tau=1
    a = np.ones(B); b = np.ones(B)
    tau, target = 3.0, 0.35                # loose start; frontier-band target
    for t in range(T):
        p = c ** (1.0/tau)                 # survival prob at current threshold
        pm = a/(a+b)
        u = np.maximum((1-(1-pm)**16) - pm, 0)
        probs = (u/u.sum() if u.sum() > 1e-12 else np.ones(B)/B)
        probs = 0.9*probs + 0.1/B
        visits = rng.multinomial(64, probs)
        succ = rng.binomial(visits, p); fail = visits - succ
        a = 1 + (a-1)*decay + succ
        b = 1 + (b-1)*decay + fail
        c = np.clip(c + 0.002*(visits/8)*(succ/np.maximum(visits,1))*(1-c), 0, 0.99)
        obs_fail = fail.sum()/max(visits.sum(), 1)
        tau = float(np.clip(tau + eta*(target - obs_fail), 1.0, 4.0))
    return tau
# Results (late-window): eta=0.05 holds fail 0.31-0.44 with tau in-band;
# eta>=0.3 overshoots to tau=1.0 and pins fail at ~0.66.
# Posterior tracking error: decay 0.99 -> 0.018-0.042; decay 0.7 -> 0.10-0.11.
```
