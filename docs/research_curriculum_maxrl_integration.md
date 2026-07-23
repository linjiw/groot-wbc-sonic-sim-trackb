# Curriculum-MaxRL × GEAR-SONIC motion tracking — analysis and integration design

Date: 2026-07-22
Status: **draft for external review** — written to be sent to the curriculum-MaxRL author/expert.
Repo under study: `groot-wbc-sonic-sim-trackb` (fork of NVIDIA GR00T-WholeBodyControl / GEAR-SONIC).
Reference implementation studied: local checkout of `curriculum-maxrl`
(`frontier_rl/` + `curriculum_maxrl/THEORY.md`, `PROOFS.md`).

**Purpose.** We evaluated whether the curriculum-MaxRL ideas (derived teacher utility
`u(p) = pass@N − pass@1`, decayed-Beta posterior teacher, hindsight recycling, MaxRL
estimator) can improve PPO training of a humanoid whole-body motion-tracking policy.
This document explains (a) our RL stack, (b) our existing adaptive sampler in full
mechanistic detail, (c) our reading of curriculum-MaxRL, (d) which parts we believe port
and which do not, (e) a GPU-free forecast experiment, (f) a concrete integration design,
and (g) **numbered open questions where we want expert input** (§8 — the ask).

Everything here is claim-checked against code with `file:line` references. If our
reading of the MaxRL side is wrong anywhere, §4 and §8 are the places to correct us.

---

## 1. TL;DR verdict

| curriculum-MaxRL component | Ports to SONIC motion tracking? | Why |
|---|---|---|
| MaxRL estimator (`w = r/K − 1/N`, drop K=0 groups) | **No** | Our PPO is dense-reward with GAE — every timestep carries tracking-reward gradient; there are no "dead groups" at the gradient level. The estimator solves an RLVR pathology we don't have. |
| Teacher layer (Beta posterior + ZPD utility + uniform floor) | **Yes, cleanly** | Our adaptive sampler is already a bin-level teacher with a binary success signal, a uniform floor, and checkpointed state. Only the *utility function* differs — and ours has a measurable pathology the ZPD utility fixes (§6). |
| Hindsight relabeling (HER-style trajectory relabel) | **Mostly no** (open question §8-Q6) | The policy is conditioned on reference-motion futures embedded in observations; relabeling requires an observation rewrite plus off-policy correction under PPO ratios. But the *dense per-step reward* already plays the partial-credit role hindsight plays in sparse RLVR. |
| Verifier-as-curriculum insight (the threshold defines `p`) | **Yes — the most novel import** | Our termination thresholds are the verifier. Annealing them manufactures a frontier on data where none exists — directly addressing our diagnosed "flat data ⇒ no mechanism can help" dead-end (§5.4). |

Expected benefit: **conditional, and forecastable**. On our current 2-motion easy dataset
at micro budget: none (all signals stay flat — confirmed by simulation, §6). On a dataset
with real difficulty spread that includes unlearnable-at-budget clips: the release
failure-rate sampler wastes 43–53% of its sampling mass on impossible bins in our
forecast; ZPD-shaped utilities cut this to 3–11%. That is the concrete mechanism by which
the import should help, and it is testable with our preregistered harness.

---

## 2. The motion-tracking RL stack (for the MaxRL expert)

### 2.1 Task and policy

A single policy tracks retargeted human motion clips on a simulated Unitree G1 humanoid
(Isaac Lab, 4096 parallel envs in release config; our research runs use 8–16). Each env
is assigned a **reference motion segment**; observations include proprioception histories
plus **10 future reference frames** of the assigned motion
(`sonic_release.yaml:48-51` — `num_future_frames: 10`, `dt_future_ref_frames: 0.1`).
So the policy is *goal-conditioned on the reference trajectory*, analogous to a
goal-conditioned prompt: the "task" is embedded in the observation.

The action is a joint-space target (plus a 64-D motion-token interface used downstream
by a VLA — frozen deploy contract, not relevant to training-time curriculum).

### 2.2 The RL algorithm — dense-reward PPO, not RLVR

- Trainer: a TRL fork's PPO (`gear_sonic/trl/trainer/ppo_trainer.py`), GAE with
  γ=0.99, λ=0.95 (`config/algo/ppo_im_phc.yaml:13-14`), adaptive-KL learning rate,
  24 steps/env per iteration (`sonic_release.yaml:79`), 4 minibatches.
- Reward: **dense, per-timestep**, 12 terms — exp-kernel tracking rewards
  (anchor pos/ori, relative body pos/ori, lin/ang velocity, VR 5-point) plus
  regularizers (action-rate, joint-limit, undesired-contact, anti-shake, feet-acc)
  (`config/manager_env/rewards/tracking/base_5point_local_feet_acc.yaml`).
- **Termination = the verifier.** An episode terminates early when tracking error
  exceeds thresholds: `anchor_pos 0.15`, `ee_body_pos 0.15`, `anchor_ori_full 0.2`,
  `foot_pos_xyz 0.2`
  (`config/manager_env/terminations/tracking/base_adaptive_strict_ori_foot_xyz.yaml`).
  Otherwise it runs to `motion_time_out`. The binary early-termination flag
  (`reset_terminated`) is the **only signal the adaptive sampler consumes**.

Key structural contrast with RLVR: an all-fail batch here still produces abundant
gradient (dense rewards up to the failure point, plus the failure transition itself).
The MaxRL "dead group ⇒ zero gradient" problem does not exist at the estimator level.
What *does* exist is the compute-allocation problem: rollouts spent on mastered or
impossible motion segments produce little *useful* gradient. That is exactly the
teacher's job description in curriculum-MaxRL — which is why the teacher layer ports
and the estimator does not.

### 2.3 Evaluation

Per-motion metrics from a paged evaluator (`trl/callbacks/im_eval_callback.py`):
`mpjpe_g` / `mpjpe_l` / `pa_mpjpe` (per-joint position error, mm-scale units),
per-motion `success` (no early termination) and `progress` fraction, aggregated
success-rate. Our preregistered experiments gate on `eval.all.mpjpe_g` deltas with
paired seeds, exact sign-flip permutation p, and bootstrap CIs
(`scripts/research/paired_stats.py`).

### 2.4 Research constraints (matter for any design)

- Micro budget: research runs are 3 seeds × 8–16 envs × 50–200 iterations on small
  datasets; effect claims require ≥5 seeds (guardrail — 3-seed min permutation p = 0.125).
- Guardrail: every new mechanism ships **flag-gated, default-off, byte-identical to
  release when unset** (`fable-next.md` §4.7).
- Preregistration: gates and thresholds are frozen in code before results
  (see `scripts/research/classify_sampler_diagnosis.py` for the pattern).
- Deploy contract (obs ordering, action dims) is frozen — curriculum work is
  training-time only, which all designs below respect.

---

## 3. The existing adaptive sampler, in full detail

This is the component a curriculum-MaxRL teacher would replace/extend. It is a
PHC-style failure-rate resampler (Luo et al. 2023 lineage), ON by default in the
release config. All code in `gear_sonic/utils/motion_lib/motion_lib_base.py` unless
noted.

### 3.1 Task space: 50-frame bins

Every motion clip is split into 50-frame bins (`bin_size: 50`,
`config/manager_env/commands/terms/motion.yaml:16-25`; construction at `:2258-2388`).
On our 2-motion sample dataset this yields ~70 bins; on a real dataset, thousands.
Bins are the "prompts". Crucially, **bins are directly settable**: reference-state
initialization can start an episode at any bin's start frame (with a
`pre_failure_sample_window: 200` backward offset when re-sampling a failed bin,
`:2861-2864`) — unlike LLM prompts, no prefix generation is needed to "reach" a task.

### 3.2 Statistics update — per-step occupancy semantics (important, subtle)

`update_adaptive_sampling(failure, motion_ids, time_steps)` (`:2462-2499`) is called
**every simulation step for all envs** from the motion command's `_update_command`
(`envs/manager_env/mdp/commands.py:3204-3217`), with `failure = reset_terminated`
(this step's termination flags).

Per step, for each env:
- its **currently occupied bin** gains `1 / bin_motion_length` episode mass
  (`counts / adp_samp_bin_motion_length`, `:2483-2486`) — so a *complete traversal*
  of a bin contributes exactly 1 episode-equivalent;
- if the env terminated this step, the terminal bin gains 1 failure count
  (× dormant `failure_counts_multiplier`, `:2496-2498`).

> **Correction note:** an earlier internal draft of this analysis claimed only the
> terminal bin is credited per episode. That was wrong — the per-step call means
> traversed-and-survived bins accumulate episode mass continuously. The denominator is
> occupancy-based; `failure_rate = failures / traversal-equivalents` is a per-bin
> **hazard estimate**. This matters for §5.3: the "statistics half" of hindsight
> (credit what a failed episode achieved before failing) is *already present*.

### 3.3 Probability computation

Per PPO iteration (`trl/trainer/ppo_trainer.py:1958-1970`, multi-GPU mean-sync every
200 iters):

1. `failure_rate = num_failures / num_episodes` per bin (`:2537-2541`, 0/0 guarded).
2. Clip at `mean(failure_rate) × cap`, cap = 200 in release
   (`sonic_release.yaml:70-71`; base default 50, `motion.yaml`).
3. Normalize → blend `0.9 × failure_based + 0.1 × uniform`
   (`uniform_sampling_rate: 0.1`) (`:2568-2714`).
4. Multiply by length-derived bin weights (with `sequence_length_agnostic: true`
   dividing by peer-bin count so each *sequence* is equally weighted, `:2391-2396`),
   renormalize.

**Priors:** every bin starts at 1 failure / 1 episode (`init_num_failures: 1`,
`:2397-2424`) → prior failure rate 1.0 everywhere. **Counts are cumulative and never
decay** (a `use_failure_rate_decay` backward-propagation option exists but is
dormant/disabled). There is no posterior, no optimism, no forgetting.

### 3.4 Knobs (active vs dormant)

| Knob | Release value | Notes |
|---|---|---|
| `bin_size` | 50 | task granularity |
| `init_num_failures` | 1 | prior; `=0` used to 0/0-NaN — fixed by us 2026-07-09, both divisions guarded |
| `uniform_sampling_rate` | 0.1 | the uniform floor — identical role to FrontierTeacher's `floor` |
| `adp_samp_failure_rate_max_over_mean` | 200 | clip cap; effectively never binds on flat data |
| `failure_counts_multiplier` | 1 (dormant) | multiplies observed failures — useless when failures ≈ 0 |
| `use_failure_rate_decay`, `decay_gamma` | off / 0.8 (dormant) | backward difficulty propagation to preceding bins |
| `max_prob_per_bin`, `max_prob_per_motion` | unset (dormant) | legacy caps |
| `pre_failure_sample_window` | 200 | resample offset before the failure frame |

### 3.5 Telemetry and persistence

- 16 `Env/adp_samp/*` scalars per iteration (episodes/failures/failure-rate min/max/mean,
  `prob_max_over_uniform`, `effective_num_bins` (1/Σp²), `num_concentrated_bins`
  (>10× uniform), `episodes_max_over_mean`) — `envs/wrapper/manager_env_wrapper.py:921-968`.
- Sampler state (counts) is saved in checkpoints (`env_state_dict['motion_lib']`,
  `get_state_dict`/`load_state_dict` `:2426-2459`) and restored on trainer resume
  (`ppo_trainer.py:2214-2215`) — but **eval never restores it**
  (`eval_agent_trl.py:439-440` loads policy only). Analysis must read checkpoints,
  not eval-time `sampling_prob`.

### 3.6 What we measured (the empirical starting point)

- **SIM-M3 (preregistered, 3 paired seeds, 50 iters, 2-motion easy data):** adaptive vs
  uniform, mean MPJPE-G delta **+0.111 (adaptive worse)**, 1/3 seeds improved, exact
  permutation p = 7/8, bootstrap CI [−0.012, +0.202]. A clean negative.
- **Telemetry:** the trained distribution stayed essentially flat —
  `prob_max_over_uniform ≈ 3`, `effective_num_bins ≈ 69–70 of 70`,
  `num_concentrated_bins = 0`. With easy motions, almost nothing terminates early;
  failure counts stay prior-dominated (1/1 seed dominates ~0 observed failures).
- **Mechanism simulation (GPU-free, `scripts/research/sampler_dynamics_sim.py`):**
  - F1: the release sampler stays flat in the starved regime (reproduces the negative).
  - F2: it *peaks* only on sparse extreme outliers; on a broad difficulty spread it
    targets correctly but stays diffuse — peakedness metrics misread it.
  - F3: on flat data, no signal-family swap can help — nothing discriminates.

So our diagnosis matches curriculum-MaxRL's opening observation almost verbatim: compute
is wasted on mastered tasks (no failures ⇒ no signal) — and, we add from F2/§6, on
*impossible* tasks (failure rate saturates at 1 and the monotone utility over-commits
to them forever).

---

## 4. Curriculum-MaxRL as we read it (please correct)

From the local repo (`frontier_rl/`, `curriculum_maxrl/THEORY.md`, `PROOFS.md`):

1. **MaxRL estimator** (`estimators.py:12-18`): group of N rollouts, K successes;
   weights `w_succ = 1/K − 1/N`, `w_fail = −1/N`, zero vector at K=0. Unbiased for the
   T=N-truncated maximum-likelihood objective; interpolates RL→ML.
2. **Advantage-mass identity** (THEORY.md §2): `E[Σ|w|] = 2(pass@N − pass@1)`,
   vanishing at p→0 and p→1, peaking at `p* ≈ ln N / N`. RLOO's mass is `2p(1−p)` =
   SFL "learnability"; GRPO's realized mass on hard prompts is ~½ its population curve
   because all-fail groups die.
3. **FrontierTeacher** (`teacher.py`): per-task decayed Beta posterior (decay 0.7),
   Thompson draw, utility `u(p̃) = (1−(1−p̃)^N) − p̃` raised to γ (≈4 on chained pools,
   1 on flat), mixed with a 10% uniform floor. Observes *requested-task evidence only*
   (feeding relabeled successes back inflates the posterior).
4. **Hindsight recycling**: a failed rollout is a verified success of the task it
   actually reached; dead groups are relabeled and trained with the same
   success-conditioned weights. Two contracts (interfaces.py): exactness (true success
   under the env's own verifier) and conditioning (goal-embedded trajectories must be
   rewritten to the relabeled goal). Their biggest single gain; beats an oracle
   allocator because it *creates* signal rather than reallocating it.
5. **Key negative** (H6): a frontier teacher *amplifies* GRPO's pass@k collapse —
   GRPO's inverted weighting was silently maintaining easy tasks; removing them via a
   curriculum breaks retention. Curricula need likelihood-style weighting to be safe.

The validated stack is estimator + teacher + hindsight on binary-verifier tasks
(RLVR-style: mazes, skill chains, MountainCar/CartPole with threshold verifiers).

---

## 5. The mapping — what ports, what doesn't, and why

### 5.1 The estimator does NOT port (and doesn't need to)

MaxRL's estimator repairs a *sparse-binary-reward group estimator*. Our PPO computes
per-timestep GAE advantages from dense rewards; an episode that fails still contributes
hundreds of gradient-bearing transitions. Swapping advantage normalization here would
be modifying a component that isn't broken and would violate our "sampler-only,
byte-identical-when-off" guardrail for no diagnosed benefit.

**Consequence for the theory (question for the expert, §8-Q1):** the advantage-mass
derivation is estimator-specific. When the downstream learner is dense-reward PPO, is
`u(p) = pass@N − pass@1` still the principled utility, or should the utility be the
*empirical* per-bin advantage magnitude (the SEC-style construction THEORY.md §3.2
identifies your formulas as the expectation of)? We have per-step GAE advantages and
know each env's current bin at every step — an empirical-|A| bandit signal is
implementable (§7, option C).

### 5.2 The teacher DOES port — structural correspondence table

| FrontierTeacher | GEAR-SONIC sampler | Status |
|---|---|---|
| task id | 50-frame motion bin | identical granularity concept |
| binary verifier reward | early-termination flag (threshold verifier) | identical |
| Beta(α,β) posterior, decay 0.7 | cumulative counts, 1/1 prior, **no decay** | gap → import |
| utility `(1−(1−p)^N) − p` (ZPD-shaped) | clipped failure rate (monotone in difficulty) | gap → import |
| Thompson sampling (optimism) | none (point estimate) | gap (maybe unneeded at 4096 envs — Q3) |
| uniform floor 0.1 | `uniform_sampling_rate 0.1` | already identical |
| γ sharpening | none | optional import |
| `observe()` requested-task-only | per-step occupancy update | ours is finer-grained (hazard) |
| `state_dict` persistence | `get_state_dict`/`load_state_dict` | already identical |
| dead/mastered/frontier metrics | 16 adp_samp telemetry keys | already richer |

The one *functional* difference is the utility shape, and it is exactly where our
diagnosed pathology lives: **failure rate is monotone in difficulty**, so a bin the
policy can never survive (hazard ≈ 1) receives maximal sampling mass indefinitely.
The ZPD utility zeroes both ends. Our F2 finding (peaks only on sparse outliers) is
this pathology seen from the telemetry side: the "outliers" the sampler locks onto are
precisely the near-impossible bins.

### 5.3 Hindsight: the statistics half already exists; the trajectory half is problematic

- **Statistics half — present.** Because the update is per-step occupancy (§3.2), a
  failed episode already credits every bin it survived. This is the skill-chain
  "deepest-prefix" credit, applied to the sampler's counts. No change needed.
- **Trajectory half — does not transfer cleanly.** The HER move ("this failed rollout
  is a success of the motion it actually performed") would require synthesizing a
  reference clip from the achieved trajectory *and* rewriting the observation's
  10-future-frame conditioning (contract 2), *and* the actions were sampled under the
  original conditioning, so PPO importance ratios would be computed against rewritten
  states — off-policy in a way our on-policy trainer doesn't support. We also note the
  dense per-step tracking reward already delivers the "partial credit for partial
  progress" that hindsight manufactures in sparse settings. We currently classify
  trajectory-level hindsight as **out of scope**, with one possible exception
  (speed-scaled motion families, §8-Q6).

### 5.4 The verifier-curriculum reframe — the most valuable conceptual import

Our F3 dead-end says: on easy data nothing fails, so no sampling signal exists, so no
teacher can help. The MaxRL lens reframes it: the pass rate `p` is a property of
(motion bin × **termination threshold**), and the threshold is ours to schedule. The
thresholds (`anchor_pos 0.15` etc.) are exactly MountainCar's `x*` targets in
`gym_classic.py`: a strictness knob that walks the verifier from easy to hard,
manufacturing a frontier where the dataset provides none.

This is implementable **today with zero new mechanism code**: the trainer has a fully
wired but never-used `schedule_dict` engine (`trl/utils/scheduler.py:296-353`,
applied per iteration at `ppo_trainer.py:1702-1706`) whose path syntax reaches
`get_term_cfg(...)` params. Published precedent: KungfuBot/PBHC (NeurIPS 2025) anneals
tracking tolerance with a hand-designed rule. The MaxRL theory suggests a *principled*
closed-loop rule instead: adjust the threshold to hold the induced failure rate near
the utility peak `p* ≈ ln N / N`. To our knowledge nobody has published that.

Known pitfall (ours, pre-existing): eval re-applies `schedule_dict` at the checkpoint's
global step (`eval_agent_trl.py:466-470`) — any threshold schedule must be scoped
train-only or stripped at eval.

---

## 6. GPU-free forecast: teacher utilities vs the release sampler

**Method.** We extended our existing sampler-dynamics simulation
(`scripts/research/sampler_dynamics_sim.py` — a numerical model of the sampler's update
math, explicitly labeled *simulation, never a headline*) with the candidate utilities.
Model: 70 bins with fixed per-episode failure probability `d` (difficulty); per
iteration, E episodes allocated ~ Multinomial(E, sampling_prob); failures ~
Binomial(visits, d); counts update; probability recomputes per signal mode; 50
iterations, 5 seeds. Signal modes:

- `failure_rate` — the release sampler (clip cap 200, 0.9/0.1 blend, 1/1 prior).
- `learn_point` — learnability `p(1−p)` from the same point-estimate counts.
- `learn_beta` — `E[p(1−p)]` under a per-iteration-decayed Beta(1+succ, 1+fail)
  posterior (decay 0.9): closed form `ab/((a+b)(a+b+1))`.
- `advmass_beta` — `(1−(1−p̄)^N) − p̄` with posterior-mean p̄ and band knob N=16.

**Results** (episodes/iter = 200; mass fractions of the final sampling distribution;
"impossible" = d > 0.9, "frontier" = 0.2 ≤ d ≤ 0.8):

| Regime | Metric | failure_rate | learn_point | learn_beta | advmass N=16 |
|---|---|---|---|---|---|
| sparse_outlier (F2's regime) | impossible mass | **0.533** | 0.051 | 0.058 | 0.274 |
| spread + 20% impossible | impossible mass | **0.433** | 0.034 | 0.107 | 0.256 |
| spread + 20% impossible | frontier mass | 0.511 | **0.869** | 0.741 | 0.638 |
| spread (no impossible) | frontier mass | 0.787 | 0.811 | 0.788 | 0.776 |
| starved (our sample_data) | all modes ≈ flat | — | — | — | — |

**Interpretation.**

1. The release sampler's waste is *specifically on unlearnable bins*: ~half its
   non-uniform mass in mixed regimes. ZPD utilities redirect it to the frontier. This
   is the concrete, mechanistic benefit prediction for the teacher import.
2. On datasets with **no** impossible content, all utilities perform similarly — the
   import is approximately free but not beneficial. The benefit is conditional on
   dataset composition (real retargeted mocap will contain infeasible segments;
   retargeting-artifact literature supports this).
3. At micro budget (8 episodes/iter) **everything is flat** — consistent with our F3.
   No teacher rescues a starved regime; our dataset-headroom gate (SIM-D1) remains a
   hard prerequisite.
4. `advmass` at N=16 retains meaningful mass on d=0.98 bins by design
   (u(p=0.02) ≈ 0.26 — "solvable within 16 tries" is genuinely nonzero); learnability
   discounts near-impossible more aggressively. Which is *correct* depends on whether
   near-impossible bins are worth long-shot exploration — a band-knob question (Q2).
5. Caveats: static difficulty (no learning dynamics — same limitation as our existing
   sim), posterior mean instead of Thompson, per-iteration decay 0.9 chosen ad hoc.
   The script is Appendix A; it is a sketch, not preregistered tooling. Before any GPU
   run we would fold these modes into `sampler_dynamics_sim.py` properly and
   preregister the forecast.

---

## 7. Proposed integration design

All changes flag-gated, default-off, byte-identical to release when unset. Ordered by
implementation cost.

### Change A — teacher signal family in the sampler (~80–120 LOC + tests)

New config keys under `adaptive_sampling` (consumed in
`update_adaptive_sampling_probabilities`, `motion_lib_base.py:2568`):

```yaml
adaptive_sampling:
  signal: failure_rate      # failure_rate | learnability | advantage_mass
  posterior_decay: 1.0      # per-iteration count decay; 1.0 = release behavior
  advmass_n: 16             # band knob for advantage_mass only
```

Implementation sketch (all tensor-level, same shapes as existing code):

```python
# in sync_and_compute_adaptive_sampling, after failure_rate computation
signal = cfg.get("signal", "failure_rate")
if signal != "failure_rate":
    # decayed pseudo-counts (replaces cumulative counts when decay < 1;
    # decay applied once per recompute, i.e., per PPO iteration)
    succ = (self.adp_samp_num_episodes - self.adp_samp_num_failures).clamp(min=0)
    a = 1.0 + succ                      # Beta posterior over per-bin survival prob
    b = 1.0 + self.adp_samp_num_failures
    p_mean = a / (a + b)
    if signal == "learnability":
        utility = a * b / ((a + b) * (a + b + 1.0))     # E[p(1-p)], closed form
    elif signal == "advantage_mass":
        N = cfg.get("advmass_n", 16)
        utility = ((1.0 - (1.0 - p_mean) ** N) - p_mean).clamp(min=0)
else:
    utility = clipped_failure_rate                      # existing path, untouched
# downstream unchanged: normalize -> 0.9/0.1 uniform blend -> bin weights -> renorm
```

Notes and deliberate deviations from the reference implementation (flagged for review):

- **Posterior mean, not Thompson.** At 4096 envs/iteration the posterior is tight and
  our reproducibility guardrails favor determinism. At 8-env micro scale this is more
  questionable — Q3.
- Decay applied at recompute time (per PPO iteration), not per teacher-step; the
  effective horizon therefore depends on iteration episode throughput — Q4.
- The `p` being tracked is per-bin *survival* (1 − hazard) from the occupancy-based
  counts (§3.2), not per-group pass@1 — the semantics the expected-mass formula assumes
  may differ — Q2.
- Everything downstream (clip is skipped for non-failure_rate signals or retained?
  — currently we'd skip the cap since utilities are bounded; flag for review),
  telemetry, checkpointing, and the uniform floor are unchanged, so all existing
  diagnosis tooling keeps working.

### Change B — verifier/threshold curriculum (config-only; +~30 LOC for closed-loop)

Static version — the first `schedule_dict` ever used in this repo (engine already
wired, `scheduler.py:296-353`):

```yaml
trainer:
  schedule_dict:
    "env@<path-to-termination-manager>@get_term_cfg('anchor_pos')@params@threshold":
      type: linear
      seg_steps: [0, 200]
      seg_vals: [0.30, 0.15]     # loose -> release-strict over 200 iterations
```

(The exact attribute path into the Isaac Lab termination manager must be verified
against the live object before launch; the scheduler module ships a worked example of
the path syntax for event terms.) **Mandatory guard:** strip or scope the schedule at
eval — `eval_agent_trl.py:466-470` re-applies it otherwise.

Closed-loop version (novel, needs expert sanity check — Q7): each iteration, nudge the
threshold toward the value holding the observed global early-termination rate near a
target `p_fail* ≈ ln N / N`. A ~30-line controller in the env wrapper; turns the
hand-designed anneal into a self-tuning verifier curriculum.

### Change C — empirical advantage-mass signal (SEC-style; contingent on Q1)

If the expert's answer to Q1 is "for dense PPO, use realized advantage magnitude":
accumulate `Σ|A_t|` per bin per iteration (the motion command knows each env's current
bin at every step; the trainer has per-step GAE advantages; join them in the wrapper),
decay-average, and use it as the utility. ~80 LOC across trainer/wrapper/motion_lib.
This is the only variant that needs no binary-success model at all and directly
measures "gradient budget commanded per bin". We have not built it; we want the
expert's ranking of A-vs-C first.

### Experiment plan (fits the existing preregistered pipeline)

Arms via `scripts/research/run_sonic_multiseed.py` (3-seed screens, then ≥5 seeds for
claims): uniform | failure_rate (release) | learnability | learnability+decay |
threshold-schedule | learnability × threshold-schedule. Prerequisites, in order:

1. SIM-M4b diagnosis closes (pending artifact sync — one command).
2. SIM-D1 dataset-headroom gate passes on a candidate dataset (**hard prerequisite** —
   forecast row "starved" shows every teacher is flat without difficulty spread).
3. Preregister activation + effect gates before launch. The activation gate must use
   the **targeting criterion** (hard-half/easy-half mass ratio ≥ 1.5), not peakedness —
   ZPD utilities are diffuse by design (forecast: pmax/uniform only 1.3–2.0; a
   peakedness gate would misclassify a working teacher as inactive).
4. Add an **easy-decile retention metric** (MPJPE on the easiest 10% of motions) to
   every arm — the direct analogue of the H6 forgetting failure mode. The 0.1 uniform
   floor is our only current mitigation.

---

## 8. Open questions for the curriculum-MaxRL expert (the ask)

**Q1 — Right utility for dense-reward PPO.** Your advantage-mass derivation is
estimator-specific (MaxRL/RLOO/GRPO weights on binary group rewards). Our learner is
dense-reward PPO+GAE: no dead groups, gradient from every timestep. Is the principled
teacher utility still a function of binary pass rate (`pass@N − pass@1`), or should we
build the SEC-style empirical version (per-bin Σ|GAE-advantage|, which we can measure
directly — §7 Change C)? Which would you rank first, and is there a theory reason to
expect the binary-verifier utility to remain a good proxy when the reward is dense but
the *verifier* (early termination) is what defines task success?

**Q2 — What is N here?** We have no natural "group of N rollouts per task": ~4096 envs
are allocated across thousands of bins each iteration by a multinomial draw, and our
`p` is a per-bin survival probability from per-step occupancy counts (a hazard
complement, not pass@1 of an episodic group). Candidate interpretations of N: (a) a
free band knob (our forecast used 16); (b) expected visits per bin per posterior
half-life; (c) expected visits per bin between probability recomputes. Which
semantics — if any — preserves the ZPD interpretation? Does the `p* ≈ ln N / N` band
placement carry over at all when tasks aren't grouped?

**Q3 — Thompson vs posterior mean.** At 4096 envs the posterior is tight; at our 8-env
micro-screen scale it is not. Our reproducibility guardrails favor determinism
(posterior mean). How much of the teacher's validated performance do you attribute to
Thompson optimism specifically (rarely-visited-task exploration), and is a
deterministic optimism bonus (e.g., mean + k·std) an acceptable substitute?

**Q4 — Decay semantics.** Your validated decay is 0.7 per teacher-step (one group
observed per step). Our natural cadence is one recompute per PPO iteration, with
episode throughput per bin varying by orders of magnitude between release (4096 envs)
and micro (8 envs) scale. Should decay be per-recompute (our current sketch, 0.9), or
per-evidence-unit (scale-invariant half-life in episode-equivalents)? The latter seems
more principled; did you test anything like it?

**Q5 — Chained bins vs independent prompts.** Our bins within a motion are sequential
(like your skill-chain prefixes) but — unlike LLM prompts — **directly settable**: RSI
can start an episode at any bin. Under direct settability, does the chained structure
still argue for γ≈4 sharpening ("steps on the highest-mass task unlock the next"), or
does the compounding argument collapse to γ=1 because no unlock ordering is enforced?

**Q6 — Any valid hindsight analogue?** We rejected trajectory-level relabeling
(conditioning rewrite + off-policy correction under on-policy PPO; §5.3). One residual
candidate: our planned synthetic difficulty set builds speed-scaled variants of each
motion (×0.75…×2.0) — a nested task family where a failed ×1.5 rollout may satisfy the
×1.0 verifier over the frames it survived. But actions were conditioned on the ×1.5
reference futures, so the conditioning contract is still violated unless observations
are rewritten and the update accepts the off-policy gap. Is there a version of the
recycling insight that survives on-policy PPO, or should we accept that dense rewards
already provide the partial credit and drop this thread?

**Q7 — Closed-loop verifier curriculum.** Scheduling the termination threshold to hold
the induced failure rate near a target (our §7 Change B closed-loop variant) couples
the verifier to the policy's current competence — the verifier is no longer stationary,
so the sampler's `p` estimates chase a moving definition of success, and episode
lengths (hence data distribution per env-step) change with the threshold. Do you see a
stability argument (or failure mode) for teacher + moving-verifier running
simultaneously? Should the teacher's posterior be reset or re-decayed on threshold
moves? Your MountainCar positional curriculum is the closest analogue — but there the
task family was explicit in the policy conditioning, whereas our threshold is invisible
to the policy.

**Q8 — Forgetting mitigation.** Your H6 result (frontier teacher amplifies GRPO
collapse because GRPO was silently maintaining easy tasks) has a plausible analogue
here: aggressive frontier sampling may degrade mastered motions. Is a 0.1 uniform floor
the validated mitigation, or did retention require more (explicit anchor mass on
mastered bins, as in your `stage_sampling_weights` anchor mechanism)?

**Q9 — Is the clip cap still needed?** The release sampler clips failure rate at
`mean × 200` to bound concentration. ZPD utilities are bounded and self-limiting near
both ends. Would you keep any cap for safety under posterior noise at small visit
counts, or is the uniform floor sufficient?

**Q10 — Anything we misread?** §4 is our summary of your method from the repo; §5 is
the port/no-port reasoning. If the estimator-relevance argument (§5.1) or the
occupancy-based `p` semantics (§3.2, Q2) invalidate an assumption your theory needs,
that is the most valuable thing to hear.

---

## 9. Constraints and guardrails any design must respect

1. Deploy contract frozen (obs ordering, 64D+7D×2 action, ZMQ layout) — training-time
   changes only. All designs above comply.
2. New mechanisms: flag-gated, default-off, byte-identical when unset, unit-tested
   (existing test suite: `tests/research/`, 117 tests, runs in a CPU-only venv).
3. Preregistration-in-code: gates/thresholds frozen before results (pattern:
   `scripts/research/classify_sampler_diagnosis.py`).
4. No mechanism experiments until the dataset-headroom gate (SIM-D1) passes — the
   forecast's "starved" row is why.
5. Effect claims need ≥5 seeds + bootstrap CI; 3-seed runs are screens (min exact
   permutation p at n=3 is 0.125).
6. Simulations (including §6) are forecasts, never headlines; they must be validated
   against real telemetry before routing decisions.

---

## Appendix A — forecast script (self-contained, numpy-only)

Run with any Python ≥3.10 + numpy. This is a sketch for §6, not preregistered tooling;
before GPU experiments the modes will be folded into
`scripts/research/sampler_dynamics_sim.py` with tests.

```python
"""GPU-free forecast: MaxRL-derived teacher utilities vs the release failure-rate
sampler, on the same bin/update model as scripts/research/sampler_dynamics_sim.py.

Model: 70 bins, each with a fixed per-episode failure probability d (difficulty).
Each iteration, E episodes are allocated across bins ~ Multinomial(E, prob).
An episode in bin b fails w.p. d[b]. Counts update -> prob recomputes per signal mode.
"""
import numpy as np

BINS = 70
CAP = 200.0
UNIFORM_RATE = 0.1
ITERS = 50


def regimes(seed):
    rng = np.random.default_rng(1000 + seed)
    out = {
        "starved": np.clip(rng.beta(1.2, 200.0, BINS), 0.0, 0.05),
        "spread": np.clip(rng.beta(1.5, 2.5, BINS), 0.0, 0.95),
    }
    d = np.full(BINS, 0.02)
    hard = rng.choice(BINS, size=max(1, BINS // 20), replace=False)
    d[hard] = 0.95
    out["sparse_outlier"] = d
    # spread_impossible: 30% mastered, 50% frontier, 20% impossible-at-budget
    d2 = np.empty(BINS)
    idx = rng.permutation(BINS)
    n_m, n_f = int(BINS * 0.3), int(BINS * 0.5)
    d2[idx[:n_m]] = 0.02
    d2[idx[n_m:n_m + n_f]] = rng.uniform(0.2, 0.7, n_f)
    d2[idx[n_m + n_f:]] = 0.98
    out["spread_impossible"] = d2
    return out


def prob_from_utility(u):
    total = u.sum()
    base = np.full(BINS, 1.0 / BINS) if total <= 0 else u / total
    return base * (1 - UNIFORM_RATE) + UNIFORM_RATE / BINS


def compute_prob(mode, fails, eps):
    if mode == "failure_rate":
        f = np.where(eps > 0, fails / eps, 0.0)
        upper = f.mean() * CAP
        return prob_from_utility(np.clip(f, 0.0, upper))
    if mode == "learn_point":
        f = np.where(eps > 0, fails / eps, 1.0)  # release prior 1/1 -> f_hat=1 unvisited
        p = 1.0 - f
        return prob_from_utility(p * (1 - p))
    succ = np.maximum(eps - fails, 0.0)
    a, b = 1.0 + succ, 1.0 + fails
    pm = a / (a + b)
    if mode == "learn_beta":
        return prob_from_utility(a * b / ((a + b) * (a + b + 1.0)))  # E[p(1-p)]
    if mode == "advmass_beta":
        N = 16  # band knob, not visit count
        return prob_from_utility(np.maximum((1.0 - (1.0 - pm) ** N) - pm, 0.0))
    raise ValueError(mode)


def simulate(difficulty, mode, episodes_per_iter, seed, decay=0.9):
    rng = np.random.default_rng(seed)
    if mode in ("failure_rate", "learn_point"):   # release init: 1 failure / 1 episode
        eps, fails = np.ones(BINS), np.ones(BINS)
    else:
        eps, fails = np.zeros(BINS), np.zeros(BINS)
    prob = np.full(BINS, 1.0 / BINS)
    for _ in range(ITERS):
        visits = rng.multinomial(episodes_per_iter, prob)
        f_new = rng.binomial(visits, difficulty)
        if mode in ("learn_beta", "advmass_beta"):
            eps *= decay
            fails *= decay
        eps, fails = eps + visits, fails + f_new
        prob = compute_prob(mode, fails, eps)
    return prob


def report(difficulty, prob):
    mastered = difficulty < 0.05
    frontier = (difficulty >= 0.2) & (difficulty <= 0.8)
    impossible = difficulty > 0.9
    order = np.argsort(difficulty)
    easy_half, hard_half = order[: BINS // 2], order[BINS - BINS // 2:]
    em = prob[easy_half].sum()
    return {
        "mastered_mass": prob[mastered].sum() if mastered.any() else float("nan"),
        "frontier_mass": prob[frontier].sum() if frontier.any() else float("nan"),
        "impossible_mass": prob[impossible].sum() if impossible.any() else float("nan"),
        "hard_half_ratio": prob[hard_half].sum() / em if em > 0 else float("inf"),
        "pmax_over_uniform": prob.max() * BINS,
    }


def main():
    modes = ["failure_rate", "learn_point", "learn_beta", "advmass_beta"]
    for episodes in (8, 200):
        print(f"\n=== episodes/iter = {episodes} ===")
        for regime in ("starved", "spread", "sparse_outlier", "spread_impossible"):
            print(f"\n-- {regime} --")
            print(f"{'mode':<14}{'mastered':>10}{'frontier':>10}{'imposs':>8}"
                  f"{'hardhalf':>10}{'pmax/U':>8}")
            for mode in modes:
                accs = [report(regimes(s)[regime],
                               simulate(regimes(s)[regime], mode, episodes, 7000 + s))
                        for s in range(5)]
                m = {k: np.nanmean([a[k] for a in accs]) for k in accs[0]}
                print(f"{mode:<14}{m['mastered_mass']:>10.3f}{m['frontier_mass']:>10.3f}"
                      f"{m['impossible_mass']:>8.3f}{m['hard_half_ratio']:>10.2f}"
                      f"{m['pmax_over_uniform']:>8.2f}")


if __name__ == "__main__":
    main()
```

## Appendix B — key file:line index (this repo)

| What | Where |
|---|---|
| Sampler bin construction | `gear_sonic/utils/motion_lib/motion_lib_base.py:2258-2424` |
| Per-step stats update (occupancy + failures) | `motion_lib_base.py:2462-2499`; call site `envs/manager_env/mdp/commands.py:3204-3217` |
| Failure rate + probability computation | `motion_lib_base.py:2501-2714` |
| Sampler state persistence | `motion_lib_base.py:2426-2460`; eval gap `eval_agent_trl.py:439-440` |
| Sampler telemetry (16 keys) | `envs/wrapper/manager_env_wrapper.py:921-968` |
| Sampler config defaults / release overrides | `config/manager_env/commands/terms/motion.yaml:16-30`; `config/exp/.../sonic_release.yaml:70-71` |
| PPO trainer (GAE, schedules, sampler sync) | `gear_sonic/trl/trainer/ppo_trainer.py:1702-1706, 1958-1970, 2095-2120` |
| schedule_dict engine (unused, fully wired) | `gear_sonic/trl/utils/scheduler.py:296-353` |
| Termination thresholds (the verifier) | `config/manager_env/terminations/tracking/base_adaptive_strict_ori_foot_xyz.yaml` |
| Reward terms | `config/manager_env/rewards/tracking/base_5point_local_feet_acc.yaml` + `rewards/terms/` |
| Preregistered research harness | `scripts/research/` (multiseed orchestrator, paired stats, dynamics sim, diagnosis classifier) |
| Research plan + guardrails | `fable-next.md`; status history `docs/research_track_b_sim_status.md` |
