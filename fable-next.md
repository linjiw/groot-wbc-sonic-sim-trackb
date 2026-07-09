# fable-next.md — Track B research plan after the SIM-M3 negative result

Date: 2026-07-08
Author: Claude (Fable 5), synthesized from a multi-agent read of the repo at `d2d60a8` plus
adversarial verification of the sampler mechanism and a three-lens plan design
(diagnosis-scientist / mechanism-designer / paper-strategist) with a completeness-critic pass.

Scope: research plan for improving SONIC WBC RL training and the curriculum/sampling
method-framework, honoring GitHub issue #4 and all standing guardrails. **No deploy-contract
changes in any branch** (64D token + 7D×2 hands, obs ordering, ZMQ layout, LowCmd writer).

---

## 1. Where we are

### Milestone chain (all on `sample_data` = 2 motions, num_envs=8, seeds 0–2)

| Gate | Result | Reference |
|---|---|---|
| SIM-M1 bounded MPJPE-complete eval | PASS (harness validity) | tag `sim-m1-bounded-eval-metric-complete` |
| SIM-M2-pre variant-checkpoint eval + provenance | PASS | commit `6327804` |
| SIM-M2 3-seed paired micro (10 iters) | validity PASS, deltas tiny/mixed | tag `sim-m2-3seed-micro-causal-sanity` |
| SIM-M3 3-seed, 50 iters, preregistered effect gate | validity PASS, **effect gate FAIL** (mean adaptive−uniform MPJPE-G = +0.111, 1/3 seeds improved) | tag `sim-m3-bounded-effect-negative` |

Issue #4 (open) demands diagnosis, not scaling: telemetry summarizer for `Env/adp_samp/*`,
per-seed telemetry aggregation beside MPJPE/reward, an under-active / wrong-target /
active-but-not-useful classification, then either a bounded sampler-schedule probe
(if under-active) or a mechanism-family switch (if active-but-not-useful).

### What the adaptive sampler actually is (verified in code)

- **Bins:** every motion is split into 50-frame bins (`bin_size: 50`,
  `gear_sonic/config/manager_env/commands/terms/motion.yaml:16-25`;
  `motion_lib_base.py:2258-2388`). sample_data ≈ 2 × ~35 bins ≈ **70 bins**, matching the
  observed `effective_num_bins ≈ 69–70`.
- **Signal:** binary early-termination only. `commands.py:3212-3217` feeds
  `self._env.reset_terminated` into `motion_lib_base.update_adaptive_sampling(...):2462-2499`
  every sim step. Episode mass is length-normalized (`counts / bin_motion_length`, :2486).
- **Prior:** `init_num_failures: 1` seeds every bin with 1 failure / 1 episode → prior
  failure rate 1.0 per bin (`:2397-2424`).
- **Probability:** `failure_rate` clipped at `mean × adp_samp_failure_rate_max_over_mean`
  (release sets **200**, `sonic_release.yaml:70-71`), normalized, blended
  `0.9 × failure-based + 0.1 × uniform` (`uniform_sampling_rate=0.1`,
  `:2558-2589`), multiplied by length-agnostic bin weights, renormalized. `max_prob_per_bin`
  / `max_prob_per_motion` are unset in release (legacy skip path, `:2632-2635`).
- **Telemetry:** `manager_env_wrapper.py:921-968` emits 16 `adp_samp/*` keys per step
  (corrected from 17 during SIM-M4a implementation — the wrapper emits exactly 16, and the
  `prob_*` block plus `episodes_max_over_mean` are conditionally guarded, so adaptive logs may
  show fewer in early iterations); `trl/trainer/ppo_trainer.py:264-266` prints them per
  iteration as `Env/adp_samp/<key>: %.4f`.
  **Stale comment:** `:958-959` says the cap is 50× so "10× = 20% of cap"; release cap is 200×,
  so `num_concentrated_bins` (bins >10× uniform) is 5% of cap — fix the comment when touching
  this area.

### Diagnosis the telemetry already suggests (to be made official by SIM-M4)

`prob_max_over_uniform ≈ 3`, `num_concentrated_bins = 0`, `effective_num_bins ≈ 70/70`,
`episodes_max_over_mean ≈ 1.7–1.9`: the distribution stayed essentially flat. At
num_envs=8 × 50 iters × 24 steps/env (`sonic_release.yaml:79`) ≈ 9.6k env-steps over 70 bins,
with easy motions that mostly reach `motion_time_out` rather than terminate, observed failures
are ~0 and the failure rate stays **prior-dominated** (1/1 seed). The likely verdict is the
compound one: **under-active because the dataset produces no discriminative failure signal** —
which is simultaneously "under-active" and "nothing to discriminate." This matters because
issue #4 deliverables 4 and 5 prescribe different follow-ups; the compound verdict routes to a
**dataset-headroom gate first**, then the mechanism work. Amplifying a flat signal is a no-op.

### Load-bearing pitfalls discovered in this review (verified against code)

1. **Eval never restores sampler state.** Checkpoints DO save it
   (`env_state_dict['motion_lib']` = per-bin `adp_samp_num_episodes` / `adp_samp_num_failures`,
   `trl/callbacks/model_save_callback.py:66-136`), but `eval_agent_trl.py:439-440` loads only
   `policy_state_dict`; `load_env_state_dict` runs only on the training-resume path
   (`ppo_trainer.py:2215`). So the per-motion `sampling_prob` recorded in eval
   `metrics_eval.json` is the **uniform init**, not the trained distribution. Any
   "wrong-target" test must read the checkpoint's `env_state_dict`, not eval output.
2. **`max_unique_motions=1` in the micro eval commands**
   (`configs/research/sonic_paired_sample_micro_metric_complete.json:14,22` and the posttrain
   spec) selects 1 of 2 motions via `random.sample` (`motion_lib_base.py:450-458`). Per-motion
   difficulty scoring and any SIM-D1 headroom eval must drop this limiter and page through all
   motions. Also audit whether SIM-M3 uniform vs adaptive evals happened to score the same motion.
3. **Aggregator is hardcoded** to `uniform_sampling_micro` / `adaptive_sampling_micro`
   (`aggregate_sonic_comparisons.py:13-14`); any new variant name silently yields empty columns.
4. **3-seed statistics cannot reach significance.** The min sign-flip permutation p at n=3 is
   0.125. Three-seed gates are causal-sanity screens only; effect claims need ≥5 seeds
   (min p = 1/32 ≈ 0.031) plus bootstrap CIs.
5. **`outputs/` is gitignored** (`.gitignore:156-157`); the SIM-M2/M3 orchestrators
   (`run_sim_m2.py`/`run_sim_m3.py`) and all run artifacts live only on the robotixx machine.
   Anything the paper depends on needs a tracked home (`docs/artifacts/`) and a tracked
   orchestrator.
6. **Trainer `schedule_dict` is re-applied at eval** (`eval_agent_trl.py:466-470` calls
   `scheduler.update_scheduled_params` at the checkpoint's global step). Any
   termination-curriculum arm must scope its schedule to train-only attributes or strip it from
   the eval config, or eval terminations get silently mutated between arms.

---

## 2. The plan: SIM-M4 → SIM-M7

Naming continues the SIM-M convention. Every gate below is preregistered here, before results.
Heavy runs stay on robotixx `env_isaaclab`; this checkout does tooling, tests, and analysis
(tests are pure Python — install pytest into the local py3.11 venv).

### Phase 0 — SIM-M4a: telemetry + statistics tooling (no GPU, ~1 day)

Closes issue #4 deliverables 1–2. Build and test before touching any real log.

1. **`scripts/research/summarize_sampler_telemetry.py`** (new): block-based parse of train
   logs (split on `Learning iteration N`, then match `Env/adp_samp/<key>: <value>` per block —
   amended from a global-regex design because the guarded `prob_*`/`episodes_max_over_mean`
   keys appear late or drop out, which silently desyncs positional series); capture the **full
   per-iteration `[iteration, value]` series** for every emitted key plus `{first, last, min,
   max, slope over final 20 iters}` scalars. Series is mandatory — last-value cannot
   distinguish under-active from saturating. Absent telemetry (uniform logs) →
   `adaptive_telemetry_present: false`, never a warning; `nan`/`inf` recorded as nulls without
   shifting alignment.
2. **`scripts/research/dump_sampler_checkpoint_state.py`** (new, torch-only, no Isaac Lab):
   load each adaptive checkpoint's `env_state_dict['motion_lib']` and dump per-bin
   episodes/failures/failure-rate and the recomputed sampling distribution. This is the only
   artifact with the actual final sampled-bin distribution and directly satisfies the status
   doc's "audit whether adaptive sampling actually changes the sampled bin distribution."
3. **Plumb telemetry scalars** through the three hardcoded chokepoints:
   `summarize_sonic_logs.py` `_FIELD_PATTERNS`, `compare_sonic_manifests.py:16-29`
   `_METRIC_PATHS` (informational only — NOT `_PRIMARY_METRIC_PATHS`, so uniform runs can't
   break `ok_for_causal_comparison`), `aggregate_sonic_comparisons.py` `_METRIC_KEYS` +
   markdown columns.
4. **De-hardcode the aggregator**: `--variant-a/--variant-b` (delta = a−b, improved = delta<0
   unchanged). Required by every downstream multi-arm milestone. Implementation decision
   (2026-07-09): clean break to generic `a.*`/`b.*` row keys and `delta.<metric>.a_minus_b`,
   `schema_version: 2`, with A = treatment / B = control pinned in the docstring and defaults
   preserving the SIM-M3 pairing (a=adaptive_sampling_micro, b=uniform_sampling_micro). The
   SIM-M3 numbers are unchanged; only key names moved. Note this rename when re-aggregating in
   the status doc.
5. **Statistics upgrade** in the aggregate: mean ± sample std (existing), plus exact one-sided
   sign-flip permutation p-value on paired deltas and a 10k-resample paired bootstrap CI.
   Document the n=3 power limit in the output. Validity check: reproduce the recorded SIM-M3
   mean delta **+0.110667** from the synced comparison.json files.
6. **`scripts/research/run_sonic_multiseed.py`** (new, tracked): wraps
   `run_sonic_paired_experiment.py` per seed then aggregates — replaces the untracked
   robotixx-only `run_sim_m*.py`.
7. **Tests** under `tests/research/`: synthetic-log round-trip with known series, `%.4f`
   truncation case, uniform-log-yields-null, aggregator variant parametrization, permutation-p
   on known deltas, multiseed dry-run materialization. Keep the existing 39 green.
8. Fix the stale 10×/50× comment at `manager_env_wrapper.py:958-959` while in the area.

**SIM-M4a exit gate** (amended 2026-07-09 to match the verified emitter — 16 keys, not 17;
3 of the 6 SIM-M3 logs are uniform arms with zero adp_samp keys by design): all new/existing
tests pass locally and on robotixx; telemetry tool parses all emitted keys as series from the
3 adaptive SIM-M3 logs and reports `adaptive_telemetry_present: false` for the 3 uniform logs;
the +0.110667 reproduction matches (locally validated from the recorded status-doc numbers:
mean +0.110667, permutation p = 7/8, min achievable p = 1/8).

### Phase 1 — SIM-M4b: the diagnosis (log/checkpoint sync + analysis, no training)

Sync from robotixx into a tracked `docs/artifacts/sim_m4/`: the 6 SIM-M3 train logs, per-seed
summary/comparison JSONs, and the 3 adaptive checkpoints' `env_state_dict['motion_lib']`
tensors (dumped via item 2, small JSON — do not commit the 448 MB .pt files).

**Preregistered classification rule** (majority over the 3 adaptive seeds, applied only to
committed summarizer output):

- **Under-active / signal-starved** if, at final iteration: `prob_max_over_uniform < 5` AND
  `num_concentrated_bins == 0` AND `effective_num_bins ≥ 0.9 × 70`, with the starvation check
  that per-bin observed failures are prior-dominated (checkpoint dump:
  `num_failures − init ≤ 2` for ≥90% of bins). Record explicitly whether the compound verdict
  applies: failures are absent because episodes time out (easy data), i.e. *under-active because
  the dataset yields no failure signal*.
- **Wrong-target** if NOT under-active AND the concentrated bins' motions rank-disagree with
  per-motion difficulty from a fresh all-motions eval (n=2 ⇒ sign check only; note the
  degeneracy in the writeup). Source: checkpoint dump, **not** eval `sampling_prob` (pitfall 1).
- **Active-but-not-useful** if concentrated AND correctly targeted AND the SIM-M3 effect gate
  failed (it did).

**Decision routing:** plain under-active (discriminative data, sampler too damped) → SIM-M5a.
Compound verdict or wrong-target/not-useful → SIM-D1 first, then SIM-M5b. Post the
classification + telemetry table + checkpoint-dump evidence to issue #4 (closes deliverables
3–5's decision), update `docs/research_track_b_sim_status.md`, tag
`sim-m4-sampler-telemetry-diagnosis`, and open one new issue per next gate (matching the
one-issue-per-gate convention of #1–#4).

### Phase 2 — SIM-D1: dataset difficulty-headroom gate (eval-only GPU, ~1–2 days)

Tests the confound hypothesis directly: *2-motion sample_data has no difficulty spread, so ANY
sampling mechanism is a no-op at this scale — SIM-M3's negative may say nothing about the
mechanism family.*

**Data lanes (parallel):**

- **D-A (submit immediately, non-gating):** HF gated-access request for `bones-studio/seed`.
  Note: `nvidia/GEAR-SONIC`'s `bones_seed_smpl` tars are SMPL-side only; the robot-side motion
  lib needs the gated CSVs via `convert_soma_csv_to_motion_lib.py` + `filter_and_copy_bones_data.py`.
- **D-B (immediate bridge):** `scripts/research/build_synthetic_difficulty_set.py` (new) —
  ~24 variants of the 2 sample motions via playback-speed scaling
  (×{0.75, 1.0, 1.25, 1.5, 1.75, 2.0}) and forward/reversed, with the **same transform applied
  to robot and SMPL sides** so the universal-token encoder pairing and aux alignment losses
  stay valid. Speed is a physically meaningful difficulty axis for tracking. Labeled
  `synthetic` in every manifest; barred from headline claims — mechanism iteration only.
- **Rejected lane:** pairing external retargeted-G1 motions (e.g. LAFAN) with unrelated
  BONES-SEED SMPL clips — the multi-encoder training samples g1/smpl/teleop encoders and the
  latent-alignment aux losses require content-paired SMPL; mismatched pairing corrupts them.
  Only revisit with an explicit aux-loss/encoder redesign.

**Preregistered SIM-D1 gate (per candidate dataset, incl. retro-run on sample_data itself):**
one bounded eval of `sonic_release/last.pt` over ALL motions (no `max_unique_motions` limiter;
deterministic assignment). PASS iff (i) per-motion MPJPE-G spread p90−p10 ≥ 20 units or
p90/p10 ≥ 1.5, AND (ii) ≥20% of motions have success=0 or progress<0.9, AND (iii) ≥30% have
success=1 ∧ progress=1 (a frontier and a mastered anchor both exist). sample_data is expected
to FAIL — record "SIM-M3 is confounded by zero difficulty spread" in the status doc if so.
**No effect experiment may run on a dataset that failed SIM-D1.** If no candidate passes,
STOP mechanism work; the paper pivots to the methodology+diagnosis claims (§3).

### Phase 3 — SIM-M5: one branch-conditional mechanism probe (robotixx, micro budget)

All arms: 3 seeds (0–2), num_envs=8–16, **200 iters**, SIM-D1-passing dataset, specs under
`configs/research/`, launched by the tracked `run_sonic_multiseed.py`, frozen eval settings
across arms. Two-stage gating everywhere: a **mechanism-activation telemetry sub-gate must pass
before the effect sub-gate is scored**; an inactive-mechanism result is classified
*invalid-inactive*, not *negative*.

- **SIM-M5a — sampler-schedule probe** (only if plain under-active on discriminative data;
  this is issue #4 deliverable 4). Overrides only documented cfg keys:
  `adaptive_sampling.init_num_failures=0` (kill the flattening prior — the mechanistically
  correct knob), `uniform_sampling_rate: 0.1→0.05`, optionally `bin_size: 50→150`. Do **not**
  lead with `failure_counts_multiplier`: it multiplies observed failures, and 5 × ~0 ≈ 0.
  - Activation sub-gate: final `prob_max_over_uniform ≥ 10` AND `num_concentrated_bins ≥ 1`
    in ≥2/3 seeds. Effect sub-gate: mean tuned−uniform MPJPE-G ≤ −0.5, ≥2/3 seeds improved,
    all `ok_for_causal_comparison=true`. Activation fail ⇒ knobs exhausted ⇒ M5b. Max one knob
    revision total.
- **SIM-M5b — mechanism-family swap** (expected branch). Flag-gated, default-off, unit-tested
  mode key `adaptive_sampling.signal: {failure_rate | error_ema | staged}` in
  `motion_lib_base.py`:
  - `error_ema`: replace the binary `reset_terminated` signal with the continuous per-episode
    tracking error already computed in `command.metrics` (`commands.py:2396-2445`), threaded
    through the call site (`:3212-3217`) into the prob computation (`:2558-2589`), weighting
    bins by error-EMA percentile. Always discriminative even when nothing terminates —
    directly fixes the diagnosed root cause. (~150 LOC + prob-computation unit test.)
  - `staged` (competence-gated curriculum, the paper's title mechanism): port the CG-WBC math
    (`scripts/research/curriculum_sampler.py:144-178` — gated unlock + learning-progress
    `|ΔEMA|` softmax weighting + anchor mass) onto motion-lib bins/motions, gates defined in
    tracking terms (per-stage MPJPE-EMA ≤ threshold), state persisted via the existing
    `get_state_dict`/`load_state_dict` hooks (`motion_lib_base.py:2427-2459`).
  - Gate: activation sub-gate as above; effect sub-gate mean best-mechanism−uniform MPJPE-G
    ≤ −0.5, ≥2/3 seeds improved. Fail-but-active on a SIM-D1-passing dataset ⇒ reactive
    per-bin reweighting is retired for this paper ⇒ fallback paper path.
- **SIM-M5c — termination-threshold curriculum** (optional third arm, config-only, classic
  large effect): anneal `anchor_pos`/`ee_body_pos` termination thresholds 0.30→0.15 over
  iters 0–200 via trainer `schedule_dict`. It injects gradient from hard clips by *not killing
  episodes early* — independent of the failure-signal problem. **Mandatory guard:** scope the
  schedule to train-only attributes or strip it from eval (pitfall 6), and dry-run the
  scheduler path before launch.

### Phase 4 — SIM-M6: headline effect gate (only if an M5 arm passes)

5 seeds × 3 arms (uniform / default failure-rate / winning M5 mechanism) × 500–1000 iters,
num_envs as memory allows, on the best SIM-D1-passing dataset (BONES-SEED subset if access
landed; the synthetic bridge set cannot host this gate). **Preregistered gate:** all validity
gates pass; mean mechanism−uniform MPJPE-G ≤ −1.0; ≥4/5 seeds improved; paired bootstrap 95%
CI excludes 0 (exact permutation p ≤ 0.05 reported). Secondary, reported but non-gating: eval
success rate, sample-efficiency (iterations to reach uniform's final MPJPE). PASS ⇒ headline
claim C3. FAIL ⇒ fallback paper, and the remaining budget goes to a mechanism × dataset-size ×
budget boundary map (3 seeds per cell).

### Phase 5 — SIM-M7 (stretch): staged vs reactive head-to-head

Only if SIM-D1 passed AND ≥1 of M5's arms passed or showed directional signal with
CI-overlapping-zero: staged (competence-gated) vs best reactive mechanism, 3 seeds × 500 iters,
same gate template plus the preregistered sample-efficiency secondary
(iterations-to-baseline-final-MPJPE ≤ 0.75×). This is the experiment that makes the paper's
thesis (*competence-gated beats unstructured adaptation*) a measured claim rather than framing.

---

## 3. Paper strategy

**Claim architecture** (each independently publishable):

- **C1 — methodology (banked):** preregistered, validity-gated paired-experiment protocol for
  curriculum claims in humanoid WBC tracking — manifests, checkpoint provenance,
  control-mismatch/metric/checkpoint gates, effect gates fixed before results, preserved
  negative tags. SIM-M1…M3 are the demonstration.
- **C2 — diagnosis:** telemetry- and checkpoint-grounded triage showing PHC-style failure-rate
  resampling is signal-starved at small data/budget (prior-dominated bins, flat distribution,
  cap never binding), with the under-active / wrong-target / not-useful decision procedure.
  SIM-M4 delivers this.
- **C3 — effect (the risk claim):** competence-gated / learning-progress motion sampling
  improves tracking reliability at matched sample budget vs uniform AND vs the field-default
  failure-rate sampler, on data with demonstrated difficulty headroom, ≥4/5 seeds, CI excluding
  0. SIM-M6/M7 deliver this — or don't.
- **Fallback paper if C3 fails:** *"When does adaptive motion sampling help? A preregistered
  study"* — C1+C2 plus the boundary map. Every branch outcome is publishable, which removes
  the incentive to fish.

**Related-work positioning:** the built-in sampler is the PHC-style incumbent (Luo et al.
2023), so the study directly evaluates the field's default. Frame the alternatives within
UED/curriculum RL: Prioritized Level Replay (Jiang et al. 2021), ACCEL, ALP-GMM /
learning-progress (Portelas et al.), TSCL; motion-data curation in ASE/PULSE/MaskedMimic/
ExBody/OmniH2O; DeepMimic RSI for the init/termination-curriculum axes; rliable
(Agarwal et al. 2021) for the small-n statistics framing. Do not cite the unimplemented
20/60/20 curriculum split or motionbricks as capabilities.

**Timeline from 2026-07-08:** Jul 08–10 Phase 0 + D-A request + bridge set. Jul 10–12 SIM-M4
diagnosis, close issue #4. Jul 12–13 SIM-D1. Jul 14–18 SIM-M5 (branch by diagnosis). Jul 20–31
SIM-M6 (slips if no real dataset passes SIM-D1; fallback proceeds regardless). Aug 01 go/no-go:
full paper draft (CoRL/ICRA 2027 cycle) vs methodology+negative-results draft (RLC/workshop).

---

## 4. Standing guardrails (unchanged from the program docs, plus new ones)

1. Deploy contract frozen: action dims, obs ordering, ZMQ 1280-byte header, LowCmd generation.
2. Protected references preserved: tags `sim-micro-adaptive-sampling-v0`,
   `sim-m1-bounded-eval-metric-complete`, `sim-m2-3seed-micro-causal-sanity`,
   `sim-m3-bounded-effect-negative`; SIM-M3 stays on record as a negative result.
3. No BONES-SEED / large-budget scaling until a mechanism-activation gate AND SIM-D1 pass.
4. Eval settings frozen across arms within any comparison; never average away an invalid
   seed-level comparison.
5. Effect claims need ≥5 seeds + CI; 3-seed designs are screens.
6. Synthetic bridge data never supports headline claims and is always labeled in manifests.
7. New mechanisms ship flag-gated and default-off; `adaptive_sampling.signal` unset must be
   byte-identical to current behavior.
8. One issue per gate; each closes with artifacts, a tag, and a status-doc update.

## 5. Explicitly rejected next steps

- Re-running SIM-M3 with more seeds/iterations hoping the effect appears (uncontrolled fishing;
  the path goes through diagnosis + headroom gates).
- Leading the schedule probe with `failure_counts_multiplier` (multiplies a ~0 signal).
- Correlating eval-time `sampling_prob` with MPJPE for the wrong-target test (provably uniform
  at eval; use checkpoint `env_state_dict`).
- LAFAN-retargeted robot motions paired with unrelated BONES-SEED SMPL (corrupts aux
  alignment losses / encoder sampling).
- Pursuing the VLA/G0/GR00T fine-tune track for this paper (explicitly demoted; only its
  sampler/gating math is ported).
- Using motionbricks (vendored, unused, LFS pointers unpulled).
- 64+ GPU sonic_release-scale finetunes (budget caps at micro/mid runs).
