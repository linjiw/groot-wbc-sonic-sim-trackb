# fable-next.md — Track B research plan (SIM-M4a done → SIM-M4b diagnosis next)

Date: 2026-07-09 (v2 — reorganized after SIM-M4a landed at `5d48c72`)
Author: Claude (Fable 5). v1 (2026-07-08) was synthesized from a multi-agent read of the repo
at `d2d60a8` plus adversarial verification of the sampler mechanism and a three-lens plan
design; v2 folds in the SIM-M4a implementation results and its adversarial code review.

Scope: research plan for improving SONIC WBC RL training and the curriculum/sampling
method-framework, honoring GitHub issue #4 and all standing guardrails. **No deploy-contract
changes in any branch** (64D token + 7D×2 hands, obs ordering, ZMQ layout, LowCmd writer).

---

## 0. Status dashboard

| Gate | Result | Reference |
|---|---|---|
| SIM-M1 bounded MPJPE-complete eval | PASS (harness validity) | tag `sim-m1-bounded-eval-metric-complete` |
| SIM-M2-pre variant-checkpoint eval + provenance | PASS | commit `6327804` |
| SIM-M2 3-seed paired micro (10 iters) | validity PASS, deltas tiny/mixed | tag `sim-m2-3seed-micro-causal-sanity` |
| SIM-M3 3-seed, 50 iters, preregistered effect gate | validity PASS, **effect gate FAIL** (mean adaptive−uniform MPJPE-G = +0.111, 1/3 seeds improved; post-hoc exact permutation p = 7/8) | tag `sim-m3-bounded-effect-negative` |
| **SIM-M4a telemetry + statistics tooling** | **DONE** — local half of the exit gate passed (83 tests; +0.110667 / p = 7/8 reproduction) | commit `5d48c72` |
| SIM-M4b diagnosis | **NEXT** — blocked only on robotixx access | §2 Phase 1 |
| SIM-D1 dataset-headroom gate | pending (D-A HF request should go out now) | §2 Phase 2 |
| SIM-M5/M6/M7 | branch-conditional on M4b + D1 | §2 Phases 3–5 |

### What SIM-M4a delivered (commit `5d48c72`)

New tracked tools under `scripts/research/`, each with tests in `tests/research/` (83 total):

- `summarize_sampler_telemetry.py` — full per-iteration `[iteration, value]` series for every
  `Env/adp_samp/*` key (block-based parse on `Learning iteration N` headers; late-appearing
  guarded keys and `nan`/`inf` handled without desyncing series; uniform logs →
  `adaptive_telemetry_present: false`, never a warning).
- `dump_sampler_checkpoint_state.py` — per-bin episodes/failures/failure-rate from checkpoint
  `env_state_dict['motion_lib']` (the only artifact with the trained distribution — eval never
  restores sampler state), with prior-domination classification and self-documenting caveats
  (bin weights not checkpointed; no decay; all-bins vs active-bins clip base — identical on
  sample_data, which fits one batch). Run inside `env_isaaclab` on robotixx.
- `paired_stats.py` — exact one-sided sign-flip permutation p (improvement = negative delta;
  p ≥ 1/2^n by construction; non-finite inputs rejected — NaN would otherwise read as p=0.0)
  and a deterministic 10k-resample paired bootstrap CI.
- `run_sonic_multiseed.py` — tracked multi-seed orchestrator replacing the untracked robotixx
  `run_sim_m*.py`. A seed that yields no comparison **invalidates the run**
  (`ok_for_causal_comparison=false`) rather than silently shrinking the aggregate; variant
  names are validated against the spec template.
- `aggregate_sonic_comparisons.py` **schema_version 2** — `--variant-a` (treatment) /
  `--variant-b` (control; defaults preserve the SIM-M3 pairing), row keys `a.*`/`b.*`, delta
  key `delta.eval.all.mpjpe_g.a_minus_b`, effect summary carries permutation p + bootstrap CI
  + an explicit n=3 power note. New validity failures: one-sided variant-name mismatch,
  duplicate seeds, duplicate paths, unsupported effect metrics.
- Classification-telemetry final values flow through `summarize_sonic_logs.py`
  (`ADP_SAMP_CLASSIFICATION_KEYS` is the single source of truth) →
  `compare_sonic_manifests.py` `_METRIC_PATHS` (informational only, never
  `_PRIMARY_METRIC_PATHS`) → the aggregate telemetry table. Non-finite final values report as
  absent, not as stale finite fallbacks.

Naming break to know about: schema-2 aggregates use `a_minus_b` keys; the schema-1 artifacts
on robotixx use `adaptive_minus_uniform`, and the untracked `run_sim_m*.py` there call the OLD
aggregator signature (`--adaptive-minus-uniform-threshold`) — they will fail loudly against
this commit. Use `run_sonic_multiseed.py` for everything going forward.

### Immediate next steps — SIM-M4b checklist (needs robotixx, no training)

Run on robotixx (`conda activate env_isaaclab`, repo at `5d48c72` or later, repo root).
Adjust glob paths to the actual `outputs/research/paired_sample_micro_sim_m3/` layout.

1. **Telemetry series** over all 6 SIM-M3 train logs (3 adaptive + 3 uniform):

   ```bash
   python scripts/research/summarize_sampler_telemetry.py \
     --train-log outputs/research/paired_sample_micro_sim_m3/seed0/adaptive_sampling_micro/train.log \
     --train-log outputs/research/paired_sample_micro_sim_m3/seed1/adaptive_sampling_micro/train.log \
     --train-log outputs/research/paired_sample_micro_sim_m3/seed2/adaptive_sampling_micro/train.log \
     --train-log outputs/research/paired_sample_micro_sim_m3/seed0/uniform_sampling_micro/train.log \
     --train-log outputs/research/paired_sample_micro_sim_m3/seed1/uniform_sampling_micro/train.log \
     --train-log outputs/research/paired_sample_micro_sim_m3/seed2/uniform_sampling_micro/train.log \
     --output-json docs/artifacts/sim_m4/sampler_telemetry.json
   ```

   Remote half of the SIM-M4a exit gate: series parse for all emitted keys from the 3
   adaptive logs; `adaptive_telemetry_present: false` for the 3 uniform logs.

2. **Checkpoint dumps** of the 3 adaptive trained checkpoints (release knobs are the
   defaults: `--init-num-failures 1 --failure-rate-cap 200 --uniform-sampling-rate 0.1`):

   ```bash
   python scripts/research/dump_sampler_checkpoint_state.py \
     --checkpoint <adaptive_seed0>/last.pt \
     --checkpoint <adaptive_seed1>/last.pt \
     --checkpoint <adaptive_seed2>/last.pt \
     --output-json docs/artifacts/sim_m4/sampler_checkpoint_state.json
   ```

3. **Re-aggregation validity check** with the schema-2 aggregator over the 3 synced
   seed-level `comparison.json` files; confirm mean delta **+0.110667** and permutation
   **p = 7/8** match the local reproduction, and note the schema-1→2 key rename in the
   status doc.

4. **Sync** into tracked `docs/artifacts/sim_m4/`: the telemetry JSON, checkpoint-state JSON,
   aggregate JSON/MD, the per-seed summary/comparison JSONs, and the 6 raw train logs if
   size-reasonable (NOT the 448 MB `.pt` files). Also run the full test suite on robotixx
   (`pytest tests/research -q`) to close the "pass on robotixx" half of the exit gate.

5. **Apply the preregistered classification rule** (§2 Phase 1) to the committed artifacts,
   post classification + telemetry table + checkpoint-dump evidence to issue #4, update
   `docs/research_track_b_sim_status.md`, tag `sim-m4-sampler-telemetry-diagnosis`, and open
   one new issue for the routed next gate (SIM-M5a or SIM-D1→SIM-M5b).

6. **In parallel, today (no compute):** submit the D-A HF gated-access request for
   `bones-studio/seed` (§2 Phase 2) — its latency is the SIM-M6 critical path.

---

## 1. Context: what we know (verified in code)

### The adaptive sampler

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
  `0.9 × failure-based + 0.1 × uniform` (`uniform_sampling_rate=0.1`, `:2558-2589`),
  multiplied by length-agnostic bin weights, renormalized. `max_prob_per_bin` /
  `max_prob_per_motion` are unset in release (legacy skip path, `:2632-2635`). The clip base
  is the ACTIVE-bin mean — irrelevant for sample_data (one batch) but material for
  batched-loading datasets; the dump tool records this caveat.
- **Telemetry:** `manager_env_wrapper.py:921-968` emits **16** `adp_samp/*` keys per step
  (corrected from 17 during SIM-M4a; the `prob_*` block and `episodes_max_over_mean` are
  conditionally guarded, so adaptive logs may show fewer keys in early iterations);
  `trl/trainer/ppo_trainer.py:264-266` prints them per iteration as
  `Env/adp_samp/<key>: %.4f`. The stale 10×/50× cap comment was fixed in `5d48c72`.

### Working hypothesis for SIM-M4b (to be made official against committed artifacts)

`prob_max_over_uniform ≈ 3`, `num_concentrated_bins = 0`, `effective_num_bins ≈ 70/70`,
`episodes_max_over_mean ≈ 1.7–1.9`: the distribution stayed essentially flat. At
num_envs=8 × 50 iters × 24 steps/env (`sonic_release.yaml:79`) ≈ 9.6k env-steps over 70 bins,
with easy motions that mostly reach `motion_time_out` rather than terminate, observed failures
are ~0 and the failure rate stays **prior-dominated** (1/1 seed). The likely verdict is the
compound one: **under-active because the dataset produces no discriminative failure signal** —
simultaneously "under-active" and "nothing to discriminate." Issue #4 deliverables 4 and 5
prescribe different follow-ups; the compound verdict routes to a **dataset-headroom gate
first**, then the mechanism work. Amplifying a flat signal is a no-op.

### Load-bearing pitfalls (verified; status as of `5d48c72`)

1. **Eval never restores sampler state** — OPEN, worked around. Checkpoints DO save it
   (`env_state_dict['motion_lib']`, `trl/callbacks/model_save_callback.py:66-136`), but
   `eval_agent_trl.py:439-440` loads only `policy_state_dict`. Any "wrong-target" test must
   read the checkpoint via `dump_sampler_checkpoint_state.py`, never eval `sampling_prob`.
2. **`max_unique_motions=1` in the micro eval commands** — OPEN
   (`configs/research/sonic_paired_sample_micro_metric_complete.json:14,22` and the posttrain
   spec). Per-motion difficulty scoring and the SIM-D1 headroom eval must drop this limiter
   and page through all motions. Also audit whether SIM-M3 uniform vs adaptive evals happened
   to score the same motion.
3. ~~Aggregator hardcoded to two variant names~~ — **FIXED in `5d48c72`**
   (`--variant-a/--variant-b`, schema 2; unmatched names now fail validity instead of
   yielding silent empty columns).
4. **3-seed statistics cannot reach significance** — inherent. Min sign-flip permutation p at
   n=3 is 0.125; three-seed gates are causal-sanity screens only; effect claims need ≥5 seeds
   (min p = 1/32 ≈ 0.031) plus bootstrap CIs. The aggregate now prints this power note itself.
5. **`outputs/` is gitignored** — OPEN by design; mitigated: the orchestrator is now the
   tracked `run_sonic_multiseed.py`, and paper-relevant artifacts sync to `docs/artifacts/`.
6. **Trainer `schedule_dict` is re-applied at eval** (`eval_agent_trl.py:466-470`) — OPEN;
   mandatory guard for any SIM-M5c termination-curriculum arm: scope schedules to train-only
   attributes or strip them from the eval config.

---

## 2. The plan: SIM-M4b → SIM-M7

Every gate below is preregistered here, before results. Heavy runs stay on robotixx
`env_isaaclab`; this checkout does tooling, tests, and analysis (local py3.11 venv:
`.venv_research`, `pytest tests/research -q`).

### Phase 0 — SIM-M4a: telemetry + statistics tooling — DONE (`5d48c72`)

Summarized in §0. Local exit-gate half passed: 83 tests green; +0.110667 / p = 7/8
reproduction from the recorded status-doc numbers. Remote half (parse the 6 real SIM-M3 logs;
tests green on robotixx) folds into SIM-M4b step 1/4.

### Phase 1 — SIM-M4b: the diagnosis (log/checkpoint sync + analysis, no training)

Execution checklist: §0 "Immediate next steps." Artifacts land in tracked
`docs/artifacts/sim_m4/`.

**Preregistered classification rule** (majority over the 3 adaptive seeds, applied only to
committed summarizer output):

- **Under-active / signal-starved** if, at final iteration: `prob_max_over_uniform < 5` AND
  `num_concentrated_bins == 0` AND `effective_num_bins ≥ 0.9 × 70`, with the starvation check
  that per-bin observed failures are prior-dominated (checkpoint dump:
  `num_failures − init ≤ 2` for ≥90% of bins — the dump's `prior_dominated_fraction ≥ 0.9`
  with the default threshold). Record explicitly whether the compound verdict applies:
  failures are absent because episodes time out (easy data), i.e. *under-active because the
  dataset yields no failure signal*.
- **Wrong-target** if NOT under-active AND the concentrated bins' motions rank-disagree with
  per-motion difficulty from a fresh all-motions eval (n=2 ⇒ sign check only; note the
  degeneracy in the writeup). Source: checkpoint dump, **not** eval `sampling_prob`
  (pitfall 1).
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
  lib needs the gated CSVs via `convert_soma_csv_to_motion_lib.py` +
  `filter_and_copy_bones_data.py`.
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
`configs/research/` as `{seed}`-templated multiseed specs, launched by the tracked
`run_sonic_multiseed.py` (pass `--variant-a <mechanism> --variant-b uniform_...`; the schema-2
aggregator handles any pairing), frozen eval settings across arms. Two-stage gating
everywhere: a **mechanism-activation telemetry sub-gate must pass before the effect sub-gate
is scored**; an inactive-mechanism result is classified *invalid-inactive*, not *negative*.

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
  negative tags. SIM-M1…M3 are the demonstration; SIM-M4a's exact-permutation/bootstrap
  reporting and gate-integrity guards (missing seeds invalidate the run) strengthen it.
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

**Timeline from 2026-07-09** (Phase 0 landed on schedule): Jul 09 — D-A HF request out;
Jul 09–11 SIM-M4b (robotixx sync + diagnosis, close issue #4); Jul 11–13 SIM-D1 (+ D-B bridge
set if the compound verdict holds); Jul 14–18 SIM-M5 (branch by diagnosis); Jul 20–31 SIM-M6
(slips if no real dataset passes SIM-D1; fallback proceeds regardless). Aug 01 go/no-go: full
paper draft (CoRL/ICRA 2027 cycle) vs methodology+negative-results draft (RLC/workshop).

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
9. (new, from SIM-M4a review) Aggregates over a subset of the preregistered seeds are invalid
   by construction — `run_sonic_multiseed.py` and the aggregator enforce this; never bypass by
   hand-picking comparison files.
10. (new) Non-finite metric values are reported as absent/rejected, never silently coerced —
    a NaN delta must fail loudly, not read as a significant effect.

## 5. Explicitly rejected next steps

- Re-running SIM-M3 with more seeds/iterations hoping the effect appears (uncontrolled fishing;
  the path goes through diagnosis + headroom gates).
- Leading the schedule probe with `failure_counts_multiplier` (multiplies a ~0 signal).
- Correlating eval-time `sampling_prob` with MPJPE for the wrong-target test (provably uniform
  at eval; use the checkpoint dump).
- LAFAN-retargeted robot motions paired with unrelated BONES-SEED SMPL (corrupts aux
  alignment losses / encoder sampling).
- Pursuing the VLA/G0/GR00T fine-tune track for this paper (explicitly demoted; only its
  sampler/gating math is ported).
- Using motionbricks (vendored, unused, LFS pointers unpulled).
- 64+ GPU sonic_release-scale finetunes (budget caps at micro/mid runs).
- Reviving the untracked robotixx `run_sim_m*.py` against the schema-2 aggregator (old
  signature; superseded by `run_sonic_multiseed.py`).
