# frontier_rl × Isaac Lab: design choices and guidance

*Response to the "Isaac Lab RL Infrastructure Guide" (2026-07-22). Their guide
is accurate on every seam we could cross-check against our SONIC review; this
doc records each design decision for the integration, the reasoning, what we
refactored in `frontier_rl` to match their contract, and the pre-registered
predictions for the experimental ladder. Companion code:
`frontier_rl/adapters/isaaclab_curriculum.py` (refactored, CPU-tested against
a stub env with torch tensors).*

---

## 0. What we changed in frontier_rl for this integration

Their §9.2 sketch calls `teacher.argmax_utility()` and `teacher.dead_fraction()`
— now implemented. Full refactor list:

1. **`FrontierBinTeacher` accepts torch tensors (any device)** in
   `observe_resets` — curriculum terms hand us CUDA tensors; we do one small
   D2H copy per reset batch (the teacher's state is ~2×n_bins floats; keeping
   it on CPU/numpy is deliberate — see D7).
2. **New methods**: `argmax_utility()` (the frontier bin), `dead_fraction()`,
   `mastered_fraction()`, `pass_rate_estimates()` — the telemetry their
   sketch logs.
3. **New class `FrontierTerrainTeacherTerm`** — the complete §9.2 integration
   as a stateful callable: observe outcomes of ending episodes → sample
   next-episode terrain levels → write `terrain_levels` + `env_origins` →
   return the telemetry dict (free `Curriculum/*` TB tags). `n_bins`
   auto-detected from `terrain.max_terrain_level` on first call; success
   signal pluggable via `success_fn` (default: `time_outs`). Duck-typed env,
   no isaaclab import — unit-tested on CPU with a stub (posterior tracked a
   latent 10-level difficulty gradient; frontier landed at p≈0.5; origins
   written; state round-trips).
4. Kept the thin `make_curriculum_term` for command-axis designs where
   assignment lives in a custom `CommandTerm` (their §9.4 option 2).

## 1. Design decisions (D1–D10), each with the reasoning

**D1 — Difficulty axis: terrain level first.** Their §9.4 ordering is right
and we adopt it: terrain level is already per-env state actuated at reset —
zero new machinery, and the stock `terrain_levels_vel` provides the greedy
scripted-adaptive baseline for free. Command-magnitude bins (the
MountainCar-pattern axis) are experiment 2, not 1 — they need a custom
CommandTerm and therefore a second diff.

**D2 — Success signal: task predicate over raw survival.** `time_outs` is
clean but saturates (their §9.3 caveat, and our EVIDENCE.md "starved regime"
row: a saturated signal gives every teacher nothing to discriminate). Default
in code is `time_outs` (works day one); the recommended config for Anymal is
a `success_fn` wrapping the same distance-vs-commanded predicate
`terrain_levels_vel` already uses — it stays informative across all levels
and makes the teacher-vs-greedy comparison signal-identical (both consume
the same predicate; only the *policy over bins* differs). That isolation is
worth more than any single-arm result.

**D3 — Utility: learnability p(1−p), not advantage-mass.** Their §9.1 already
resolves this per our regime table: no rollout groups ⇒ no N ⇒ the
advmass band-placement theorem has nothing to bind to, and learnability is
the N→1 member with zero knobs. (Same call as SONIC Q2 — this is now our
standing rule for reset-stream evidence.)

**D4 — Optimism: deterministic mean + 1·std, Thompson off by default.**
Their determinism pitfall (same seed + same GPU count only) plus our V3
result (the floor makes Thompson's exploration contribution small) →
determinism costs little and buys reproducible arms. `thompson=True` remains
one flag away for an ablation.

**D5 — Decay: half-life = 2048 episode-equivalents.** Evidence-scaled (per
SONIC Q4; exact aging verified in tests: 10 → 5.0 after one half-life).
Calibration: with 4096 envs and 20 s episodes at 50 Hz decimated control,
resets arrive at ~200/s-of-sim-time per 4096 envs; 2048 episode-equivalents
≈ a few learning iterations of memory — long enough to smooth batch noise,
short enough to track a policy that meaningfully improves every ~10
iterations. This is a first guess, flagged as such; their τ/noise-band
tooling should gate any retune.

**D6 — Floor: 0.1, plus keep the stock max-level randomization.** The
built-in "envs that beat max level get random levels" behavior in
`update_env_origins` is anti-forgetting replay we'd otherwise add ourselves;
our term reproduces the equivalent effect through the floor. Easy-decile
retention metric in every arm regardless (H6 lesson — retention risk is
objective-dependent, and dense-reward PPO's retention under frontier
sampling is unmeasured).

**D7 — Teacher state stays CPU/numpy.** The teacher touches only reset
batches (tens of envs at a time, per their §10 cadence note), and its state
is two small arrays. A GPU port would save microseconds and cost determinism
simplicity. Their "everything is batched" pitfall applies to *per-step term
functions*; the curriculum term runs on the reset stream where a D2H copy of
a ~32-element tensor is noise.

**D8 — Estimator/hindsight: env-side only in ladder steps 1–2.** Their §9.5
honesty matches ours: dense-reward PPO ⇒ teacher with learnability, skip
estimator changes, skip trajectory-level hindsight (statistics-only credit is
already what the posterior consumes). Estimator work (MaxRL weighting inside
rsl_rl) is gated behind a sparse-success manipulation task where the verifier
is real — their ladder step 3, unchanged.

**D9 — Baselines: all four arms, scripted first.** Their §8 caution ("adaptive
curricula have repeatedly failed to beat scripted schedules here") is the
strongest prior in the doc and we treat it as such: no-curriculum
(`terrain_levels=null`), stock greedy (`terrain_levels_vel`), a scripted
`modify_term_cfg` schedule, and the teacher — same seeds, fixed iterations,
gate metric frozen before launch. Our own SIM-M3-style negative on flat data
says the same thing from the other side.

**D10 — Telemetry contract.** Everything through the term's return dict:
`mean_bin`, `frontier_bin`, `dead_frac`, `mastered_frac`, `effective_bins` —
free `Curriculum/*` TB tags, diffable across arms via the auto-dumped
`params/env.yaml`. Plus the teacher's `state_dict` in checkpoint hooks.

## 2. Where we expect to win (pre-registered predictions)

Register these before any run; they follow from the regime map (EVIDENCE.md §2):

- **P-A (mechanism):** the teacher's sampling mass concentrates on levels with
  p̂ ∈ [0.2, 0.8] and *walks upward* over training; `dead_frac` stays below
  the greedy baseline's implied impossible-mass. (Mechanism gate — targeting
  ratio, NOT peakedness: ZPD utilities are diffuse by design.)
- **P-B (outcome, conditional):** wall-clock-matched terrain-level progression
  ≥ greedy `terrain_levels_vel`, with the gap growing when the terrain grid
  contains unlearnable-at-budget rows. On an easy grid we predict ≈ parity —
  the greedy ±1 walker is already near-optimal when every level is learnable
  (their prior-art caution is *expected* to hold there; our forecast says the
  teacher's edge is specifically impossible-bin avoidance and faster
  band-tracking, worth little when neither exists).
- **P-C (retention):** no easy-level regression beyond the τ noise band with
  floor 0.1 (dense-reward PPO retention is the open empirical question — if
  this fails, the ALP-term escalation from SONIC Q8 is next).
- **Honest null condition:** if the terrain grid's difficulty spread is
  narrow (all levels learnable within budget), we expect and will report
  parity with greedy — that outcome would *confirm* the regime map, not
  refute the method.

## 3. The experimental ladder (theirs, with our gates attached)

1. **Anymal-C rough, terrain axis** — `FrontierTerrainTeacherTerm` vs the 4
   arms (D9). Gates: P-A mechanism gate must pass before any outcome claim;
   P-B/P-C evaluated at ≥5 seeds per their own noise-band discipline.
2. **Commanded-speed bins on flat terrain** — custom CommandTerm sampling
   from per-env bins the teacher writes (their §9.4-2, our MountainCar
   pattern; conditioning is already in the obs vector, so the shared-policy
   contract holds by construction). This is the axis where we predict the
   *clearest* teacher win: speed difficulty is smooth, the frontier is real,
   and the stock alternative (global range walk) can't do per-env targeting.
3. **Sparse-success manipulation + estimator work** — only if 1–2 show
   mechanism + outcome signal; this is where MaxRL weighting and (maybe)
   trajectory hindsight re-enter, since the verifier is real and groups can
   be simulated by episode batching.

## 4. Open items on their side (small)

- Confirm the exact registration idiom for a stateful callable term in their
  fork (`CurrTerm(func=instance)` vs a `ManagerTermBase` subclass shim — the
  term is written to support either; a 5-line shim if needed).
- Confirm `success_fn` plumbing for the distance predicate (needs commanded
  speed at reset time — available from the command manager per their §3.2).
- Checkpoint hook location for teacher `state_dict` (their runner or an
  env-side callback — either works; eval must NOT restore teacher state,
  same rule as their SONIC finding).

---

*Verification status: the refactored adapter passes CPU tests against a stub
env (torch tensors, 10-level latent difficulty: posterior tracked truth,
frontier bin found at p≈0.5, origins written, state round-trip). Real-sim
validation requires the isaac-lab-base container — the ladder above is the
plan for it.*
