# Track B SONIC SIM status and next gate

Date: 2026-07-07

This document is the repo-level handoff for the current Track B / SONIC simulation research stack. It intentionally keeps the interpretation narrow: the current result is a trained-checkpoint evaluation validity gate, **not** an adaptive-sampling performance result.

## Protected references

| Reference | Commit / artifact | Purpose |
|---|---:|---|
| `sim-micro-adaptive-sampling-v0` | `d6bb536` | Original adaptive-sampling micro experiment reference. |
| `sim-m1-bounded-eval-metric-complete` | `98c5afc` | Metric-complete bounded eval gate. |
| SIM-M1 bundle | `/home/robotixx/sonic-sim-m1-bounded-eval-metric-complete.bundle` | Local protected handoff bundle, 378 MB. |
| SIM-M2-pre commit | `6327804` | Variant-specific post-training checkpoint eval gate. |

Recent research stack:

```text
6327804 Add SIM-M2-pre posttrain checkpoint eval
98c5afc Add SIM-M1 bounded eval metric smoke
e493d48 Add bounded SONIC eval micro metrics
d6bb536 Add SONIC adaptive sampling micro experiment
672c94b Add SONIC paired experiment launcher
c23f2ba Add SONIC manifest comparison harness
e668198 Add SONIC experiment manifest harness
f5d6554 Add SONIC log metric summarizer
```

## SIM-M2-pre result

Goal: verify that the paired harness evaluates each variant's own 10-iteration trained checkpoint instead of silently reusing `sonic_release/last.pt`.

Config:

```text
configs/research/sonic_paired_sample_micro_posttrain_eval.json
```

Artifacts generated locally:

```text
outputs/research/paired_sample_micro_posttrain_eval/uniform_sampling_micro/summary.json
outputs/research/paired_sample_micro_posttrain_eval/uniform_sampling_micro/manifest.json
outputs/research/paired_sample_micro_posttrain_eval/adaptive_sampling_micro/summary.json
outputs/research/paired_sample_micro_posttrain_eval/adaptive_sampling_micro/manifest.json
outputs/research/paired_sample_micro_posttrain_eval/comparison.json
```

Checkpoint provenance:

| Variant | Checkpoint | SHA256 | Release checkpoint? |
|---|---|---|---|
| `uniform_sampling_micro` | `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_uniform_sampling_micro_posttrain_seed0-20260701_022413/last.pt` | `f1803557a1b8735f2eb20bcab4ccb8fe3df93d78e06d21f89947d959ee4ec8eb` | no |
| `adaptive_sampling_micro` | `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_adaptive_sampling_micro_posttrain_seed0-20260701_022443/last.pt` | `9c82a8131faeb9954f7917e2cf19953a0a71301ea965487b8447a1bb8d94dc9d` | no |

Comparison result:

| Field | Value |
|---|---:|
| `manifest_count` | 2 |
| `control_mismatches` | 0 |
| `validation_errors` | 0 |
| `metric_warnings` | 0 |
| `checkpoint_warnings` | 0 |
| `ok_for_causal_comparison` | true |

Rows:

| Variant | `train.mean_rewards` | `eval.ok` | `eval.all.mpjpe_g` |
|---|---:|---:|---:|
| `uniform_sampling_micro` | 0.98515 | true | 31.901 |
| `adaptive_sampling_micro` | 1.02088 | true | 31.925 |

Interpretation: SIM-M2-pre passes as a trained-checkpoint evaluation validity gate. It proves the harness can train the two micro variants, save distinct post-training checkpoints, record checkpoint provenance, evaluate each checkpoint under bounded MPJPE-complete eval, and compare the resulting manifests without control, metric, validation, or checkpoint warnings.

Non-claim: this does **not** show adaptive-sampling benefit. It is one seed, 10 iterations, tiny `sample_data`, and bounded smoke eval. The MPJPE values are effectively tied.

## Verified test command

Run from repo root with the IsaacLab conda environment activated:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python -m pytest -q \
  tests/research/test_sonic_eval_metric_smoke.py \
  tests/research/test_run_sonic_paired_experiment.py \
  tests/research/test_compare_sonic_manifests.py \
  tests/research/test_im_eval_callback_config.py \
  tests/research/test_sonic_experiment_manifest.py \
  tests/research/test_sonic_log_summary.py \
  tests/research/test_curriculum_sampler.py \
  tests/research/test_curriculum_gates.py \
  tests/research/test_manifest_builder_fixture.py \
  tests/research/test_data_collection_launcher.py
```

Latest local verification: `39 passed in 3.55s` on 2026-07-07.

## Recommended next gate: SIM-M2

Next step: **3-seed paired micro causal sanity check**.

Design constraints:

| Dimension | Value |
|---|---|
| Seeds | 0, 1, 2 |
| `num_envs` | 8 |
| Learning iterations | 10 |
| Data | `sample_data/robot_filtered` + `sample_data/smpl_filtered` only |
| Only changed condition | `manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable` |
| Checkpoint saving | `++callbacks.model_save.save_last_frequency=10` |
| Eval | bounded MPJPE-complete eval per trained checkpoint |
| Aggregation | `aggregate_comparison.json` + `aggregate_table.md` |

SIM-M2 exit criteria:

- [ ] All seed-level comparisons emit `ok_for_causal_comparison=true`.
- [ ] All eval metrics are finite.
- [ ] No control mismatches.
- [ ] No metric warnings.
- [ ] No checkpoint warnings.
- [ ] `aggregate_comparison.json` exists.
- [ ] `aggregate_table.md` exists.
- [ ] Interpretation stays limited to micro causal-sanity validity unless the 3-seed result supports a stronger claim.

## Suggested implementation order

1. Add seed-specific configs or a multi-seed wrapper for seeds `0,1,2`.
2. Ensure every seed writes variant-specific checkpoints and provenance.
3. Run train+eval for each seed/variant pair.
4. Materialize per-seed `comparison.json` files.
5. Add an aggregate script/table over all seed-level comparisons.
6. Gate on warnings/errors before interpreting any metric deltas.

Do not scale to BONES-SEED, longer training, or performance claims until SIM-M2 passes.

## SIM-M2 result: 3-seed paired micro causal sanity

Date: 2026-07-07

Implementation added an aggregate comparison utility:

```text
scripts/research/aggregate_sonic_comparisons.py
tests/research/test_aggregate_sonic_comparisons.py
```

Runtime orchestrator used for this local gate:

```text
outputs/research/paired_sample_micro_sim_m2/run_sim_m2.py
```

Command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python outputs/research/paired_sample_micro_sim_m2/run_sim_m2.py --seeds 0 1 2 --execute
```

Artifacts:

```text
outputs/research/paired_sample_micro_sim_m2/seed0/comparison.json
outputs/research/paired_sample_micro_sim_m2/seed1/comparison.json
outputs/research/paired_sample_micro_sim_m2/seed2/comparison.json
outputs/research/paired_sample_micro_sim_m2/aggregate_comparison.json
outputs/research/paired_sample_micro_sim_m2/aggregate_table.md
```

Aggregate gate status:

| Field | Value |
|---|---:|
| `comparison_count` | 3 |
| `seeds` | `[0, 1, 2]` |
| `ok_for_causal_comparison` | true |
| `control_mismatches` | 0 |
| `metric_warnings` | 0 |
| `checkpoint_warnings` | 0 |
| `validation_errors` | 0 |

Seed rows:

| Seed | Uniform train mean reward | Adaptive train mean reward | Uniform MPJPE-G | Adaptive MPJPE-G | Adaptive - uniform MPJPE-G |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.98515 | 1.02088 | 31.901 | 31.925 | +0.024 |
| 1 | 0.88738 | 0.89901 | 31.637 | 31.596 | -0.041 |
| 2 | 0.95787 | 0.81643 | 34.257 | 34.312 | +0.055 |

Aggregate descriptive stats:

| Metric | Value |
|---|---:|
| mean uniform MPJPE-G | 32.5983 |
| mean adaptive MPJPE-G | 32.6110 |
| mean delta, adaptive - uniform | +0.0127 |
| sample std of delta | 0.0490 |

Interpretation: SIM-M2 passes as a 3-seed micro causal-sanity **validity gate**. The harness now survives seed expansion, preserves variant-specific checkpoint provenance, emits finite bounded eval metrics, and aggregates seed-level comparisons without control, metric, validation, or checkpoint warnings.

Non-claim: there is still no adaptive-sampling performance benefit here. Deltas are tiny and mixed-sign over a deliberately tiny 10-iteration sample-data smoke. The right conclusion is that the comparison/eval machinery is ready for a more meaningful next gate, not that adaptive sampling improves SONIC.

Recommended next gate after SIM-M2:

1. Preserve the SIM-M2 aggregate as the protected micro validity reference.
2. Add a slightly more meaningful but still bounded `SIM-M3` gate before BONES-SEED scaling, for example:
   - same 3 seeds,
   - modestly longer training budget,
   - still `sample_data` or a tiny fixed curated motion subset,
   - same checkpoint-provenance and bounded-eval requirements,
   - pre-register an effect-size threshold before looking at results.
3. Only if SIM-M3 shows stable, non-trivial directionality should we consider BONES-SEED or larger training.

## SIM-M3 pre-registered bounded effect-size gate

Date: 2026-07-07

Goal: test whether the SIM-M2-valid harness shows any stable adaptive-sampling direction under a still-small but less trivial training budget. This is **not** a BONES-SEED or paper-performance gate.

Controlled variables:

| Dimension | SIM-M3 value |
|---|---|
| Seeds | `0, 1, 2` |
| `num_envs` | 8 |
| Learning iterations | 50 |
| Dataset | `sample_data/robot_filtered` + `sample_data/smpl_filtered` only |
| Only changed condition | `manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable` |
| Adaptive extra knob | `adaptive_sampling.uniform_sampling_rate=0.1` |
| Checkpoint save cadence | `++callbacks.model_save.save_last_frequency=50` |
| Eval | same bounded MPJPE-complete eval as SIM-M2 |
| Aggregation | `scripts/research/aggregate_sonic_comparisons.py` with effect gate |

Pre-registered effect gate for a **candidate signal** only:

```text
metric = eval.all.mpjpe_g
lower_is_better = true
mean_delta_adaptive_minus_uniform <= -0.5 MPJPE-G
improved_seed_count >= 2 of 3
all seed-level comparisons ok_for_causal_comparison=true
control_mismatches = metric_warnings = checkpoint_warnings = validation_errors = 0
```

Decision rule:

- If the validity gate fails, fix the harness; do not interpret metrics.
- If validity passes but the effect gate fails, conclude no adaptive-sampling signal at this bounded budget.
- If validity and effect gate both pass, label it only as a SIM-M3 candidate signal and design the next fixed-data gate before any BONES-SEED scaling.
- No adaptive-sampling performance claim is allowed from SIM-M3 alone.

## SIM-M3 result: bounded effect-size probe

Date: 2026-07-07

Command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python outputs/research/paired_sample_micro_sim_m3/run_sim_m3.py --seeds 0 1 2 --execute
```

Artifacts:

```text
outputs/research/paired_sample_micro_sim_m3/seed0/comparison.json
outputs/research/paired_sample_micro_sim_m3/seed1/comparison.json
outputs/research/paired_sample_micro_sim_m3/seed2/comparison.json
outputs/research/paired_sample_micro_sim_m3/aggregate_comparison.json
outputs/research/paired_sample_micro_sim_m3/aggregate_table.md
```

Aggregate validity gate:

| Field | Value |
|---|---:|
| `comparison_count` | 3 |
| `seeds` | `[0, 1, 2]` |
| `ok_for_causal_comparison` | true |
| `control_mismatches` | 0 |
| `metric_warnings` | 0 |
| `checkpoint_warnings` | 0 |
| `validation_errors` | 0 |

Pre-registered effect gate:

| Field | Value |
|---|---:|
| metric | `eval.all.mpjpe_g` |
| threshold | adaptive - uniform <= -0.5 |
| minimum improved seeds | 2 of 3 |
| mean adaptive - uniform | +0.110667 |
| improved seeds | 1 of 3 |
| passes effect gate | false |

Seed rows:

| Seed | Uniform train mean reward | Adaptive train mean reward | Uniform MPJPE-G | Adaptive MPJPE-G | Adaptive - uniform MPJPE-G |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.98755 | 0.93542 | 31.835 | 32.037 | +0.202 |
| 1 | 0.93287 | 0.89537 | 31.554 | 31.696 | +0.142 |
| 2 | 0.96641 | 0.97600 | 34.244 | 34.232 | -0.012 |

Interpretation: SIM-M3 passes the harness validity gate but fails the pre-registered candidate-effect gate. This is a useful negative result: a modestly longer 50-iteration sample-data gate still does not show a stable adaptive-sampling benefit. Two seeds are worse for adaptive under MPJPE-G and the lone improved seed is effectively tied.

Decision: do not scale this adaptive-sampling mechanism to BONES-SEED as-is. The next step should shift from scaling to diagnosis. Candidate next diagnostics:

1. Audit whether adaptive sampling actually changes the sampled motion/bin distribution over 50 iterations (`Env/adp_samp/*` logs, per-seed concentration/effective-bin metrics).
2. Add an aggregate diagnostic table for adaptive-sampler telemetry, not just reward/MPJPE.
3. If the sampler is active but not helpful, test a different bounded mechanism or sampling schedule before any larger data/training expansion.
4. Preserve SIM-M3 as a negative bounded-effect reference.

## SIM-M4a: sampler-telemetry and statistics tooling (no GPU)

Date: 2026-07-09

Closes issue #4 deliverables 1–2 on the tooling side (log/checkpoint sync and the
SIM-M4b diagnosis remain). All work is local to this checkout; no training runs.

New tracked tools (`scripts/research/`):

| Tool | Purpose |
|---|---|
| `summarize_sampler_telemetry.py` | Full per-iteration `[iteration, value]` series for every `Env/adp_samp/*` key (block-based parse keyed on `Learning iteration N` headers), plus first/last/min/max and least-squares slope over the final 20 iterations. Uniform logs yield `adaptive_telemetry_present: false`, never a warning. `nan`/`inf` values are recorded as nulls without shifting series alignment. |
| `dump_sampler_checkpoint_state.py` | Dumps per-bin `adp_samp_num_episodes`/`adp_samp_num_failures` from checkpoint `env_state_dict['motion_lib']` (the only artifact with the trained distribution — eval never restores sampler state), with observed-failure/prior-domination classification and an unweighted recomputed distribution. Caveats (bin weights not checkpointed, no decay, all-bins vs active-bins clip base) are recorded in the output itself. Run inside `env_isaaclab` on robotixx. |
| `paired_stats.py` | Exact one-sided sign-flip permutation p (improvement = negative delta; p >= 1/2^n by construction; non-finite inputs rejected) and a deterministic 10k-resample paired bootstrap CI. |
| `run_sonic_multiseed.py` | Tracked multi-seed orchestrator replacing the robotixx-only `run_sim_m*.py`: renders a `{seed}`-templated spec per seed, materializes via `run_sonic_paired_experiment`, aggregates with the effect gate. A seed that yields no comparison invalidates the run (`ok_for_causal_comparison=false`); variant names are validated against the template. |

Pipeline changes:

- `summarize_sonic_logs.py` now extracts final values for all 16 `adp_samp_*` keys
  (non-finite final values are reported as absent, not silently replaced by an earlier
  finite iteration). The 6-key classification subset is exported as
  `ADP_SAMP_CLASSIFICATION_KEYS` and flows into `compare_sonic_manifests.py`
  `_METRIC_PATHS` (informational only — deliberately NOT in `_PRIMARY_METRIC_PATHS`, so
  uniform arms cannot fail `ok_for_causal_comparison`) and the aggregate telemetry table.
- `aggregate_sonic_comparisons.py` **schema_version 2**: variants parameterized as
  `--variant-a` (treatment) / `--variant-b` (control), row keys `a.*`/`b.*`, delta key
  `delta.eval.all.mpjpe_g.a_minus_b`, effect fields `a_minus_b_threshold` /
  `mean_delta_a_minus_b`. Defaults preserve the SIM-M3 pairing
  (a=`adaptive_sampling_micro`, b=`uniform_sampling_micro`), so `a_minus_b` ==
  the preregistered `adaptive_minus_uniform` for SIM-M2/M3 artifacts; schema-1 JSONs on
  robotixx keep the old key names. New validity guards: one-sided variant-name
  mismatches and duplicate seeds fail `ok_for_causal_comparison`; duplicate comparison
  paths and unsupported effect metrics raise.
- Effect summaries now carry a `statistics` block: exact permutation p, min achievable
  p, bootstrap 95% CI, and an explicit small-n power note (n=3 ⇒ min p = 0.125, screens
  only, per guardrail 5).

Preregistered validity reproduction (from the recorded SIM-M3 numbers above, run
locally through the schema-2 aggregator):

| Field | Value |
|---|---:|
| mean delta a−b (adaptive−uniform) | +0.110667 (matches) |
| exact permutation p (one-sided) | 7/8 = 0.875 |
| min achievable p at n=3 | 1/8 = 0.125 |
| bootstrap 95% CI | [-0.012, +0.202] (contains 0) |

The statistics confirm what the effect gate already said: SIM-M3 is a clean negative
with no hidden signal (p=0.875 is worse than chance toward improvement).

Also fixed while in the area: stale 10×/50× cap comment at
`manager_env_wrapper.py:958` (release cap is 200×, so 10× is a fixed concentration
marker, not a fraction of the cap).

Exit-gate status (amended in fable-next.md: 16 keys, not 17; 3 of 6 SIM-M3 logs are
uniform-arm logs with zero adp_samp keys by design): local tests (83, up from 39+43
baseline) and the +0.110667/p=0.875 reproduction pass. Remaining before closing the
gate: run the telemetry summarizer over the 3 adaptive SIM-M3 train logs and the
checkpoint dump over the 3 adaptive `last.pt` checkpoints on robotixx, and sync the
outputs into `docs/artifacts/sim_m4/`.

## SIM-M4-prep: preregistered classifier, dynamics simulation, sampler bug fix (no GPU)

Date: 2026-07-09

Built and adversarially reviewed on the tooling checkout ahead of the robotixx SIM-M4b
sync, so the diagnosis is preregistered in code and the mechanism path is de-risked.
117 research tests pass; ruff/black clean on touched files.

New/changed:

| Item | Purpose |
|---|---|
| `scripts/research/classify_sampler_diagnosis.py` | Mechanizes the §Phase 1 classification rule as tested logic. One command over the two SIM-M4a JSON artifacts emits the verdict (under-active / compound-starved / wrong-target / not-useful) + decision routing. Preregistration-in-code: thresholds are frozen constants (flatness floor `0.9×70=63`, majority over ALL adaptive seeds, dump required at prior-domination threshold 2.0). |
| `scripts/research/sampler_dynamics_sim.py` | Numerical experiment (labeled simulation, no MPJPE, never a headline) reimplementing the sampler update math to forecast mechanism behavior. |
| `gear_sonic/utils/motion_lib/motion_lib_base.py` | Fixed a latent `0/0` NaN hazard on `init_num_failures=0` (the SIM-M5a candidate knob): guarded both the per-bin failure-rate division and the all-zero probability normalization. Byte-identical for the release config (`init_num_failures=1`). |

Simulation findings (budget-independent 8→200 episodes/iter, seed-averaged; **validate against
real SIM-M3 telemetry before acting** — a validation FAIL voids them):

- **F1** — the release failure-rate sampler stays flat in the starved regime
  (prob_max_over_uniform ≈ 3.7, matching SIM-M3's ~3): reproduces the observed negative.
- **F2 (plan-altering)** — failure-rate resampling peaks **only on sparse extreme outliers**, not on
  broad difficulty spread. On a SIM-D1-shaped dataset it correctly targets the hard bins (hard-half
  mass ratio ≈ 2.9) while staying diffuse (prob_max_over_uniform ≈ 2.4, 0 concentrated bins). The
  original SIM-M5 peakedness-only activation gate would misclassify this as invalid-inactive.
  **Action: SIM-M5 activation gate amended in fable-next.md** to add a targeting criterion
  (hard-half/easy-half mass ratio ≥ 1.5) as an OR path.
- **F3** — in the truly starved regime neither failure_rate nor error_ema concentrates or targets;
  a mechanism swap cannot manufacture signal from a flat dataset. **Confirms SIM-D1 must precede
  mechanism work.**
- **error_ema is not claimed to beat failure_rate**: only a signal-density (targeting-at-low-budget)
  edge, which degrades with noise and reverses on sparse-hard outliers; the favorable ordering is
  an assumption of the sim's `error = difficulty + noise` model. Real error-SNR is a precondition to
  measure before the SIM-M5b error_ema arm.

This corrects the earlier working hypothesis (which attributed flatness solely to signal starvation):
flatness has two separable drivers — signal starvation AND low failure-rate contrast under an
outlier-only concentration metric — and the SIM-M4b checkpoint dump (`prior_dominated_fraction` and
per-bin failure spread) decides which holds on the real data.

## ZPD-teacher program locked + GPU-free tooling implemented (2026-07-23, no GPU)

The curriculum-MaxRL expert exchange closed (`docs/research_curriculum_maxrl_integration.md`
sent; `docs/external/SONIC_RESPONSE.md` received; reference implementation vendored at
`external_dependencies/curriculum-maxrl/`). Twelve decisions locked in
`docs/research_plan_zpd_teacher.md` (v1.1) — **the major research goal going forward**,
superseding fable-next.md §2 Phases 3–5. SIM-M4b / SIM-D1 remain the blocking prerequisites.

Commits `8a52afe`..`69b7e2b`, all flag-gated default-off, 182 research tests:

| Item | Purpose |
|---|---|
| `motion_lib_base.py` Change A | `adaptive_sampling.signal: learnability\|advantage_mass` — Beta posterior over per-bin survival (hazard complement), exact `E[p(1-p)]`, deterministic optimism interval-max (D3), evidence-half-life decay (D4), 20x-uniform tripwire replacing the mean×cap clip (D9), optional family kernel (D6). Byte-identical to release when `signal` unset (golden-tensor test). |
| `sampler_dynamics_sim.py` ZPD modes | Preregistered forecast Z1–Z6 + coupled threshold-controller scenarios CTRL1–3 (all reproduce the expert's Q7 findings). Artifacts frozen at `docs/artifacts/sim_m5/`. |
| **Z6 (plan-altering)** | The v1 hard-half/easy-half ≥ 1.5 activation criterion MISFIRES on ZPD utilities (≈ 0.94 for a correctly-working teacher — it down-weights impossible bins by design). Activation sub-gate re-frozen (plan §4, one revision, before any GPU run): ZPD arms gate on **frontier over-allocation ≥ 1.2 + posterior sanity Spearman ≥ 0.4**; failure_rate arms unchanged. M5-L screen leads with `optimism_k: 0` (forecast: k=1 mildly hurts frontier mass). |
| Eval-strip guard | `validate_spec` rejects specs whose commands use `trainer.schedule_dict` without stripping it from eval (the eval re-application pitfall, promoted from note to test per expert Q7). |
| Retention metric (D8) | `eval.easy_decile.*` from the eval callback's `metrics_eval.json` + a frozen SIM-D1 difficulty ranking; informational columns in `compare_sonic_manifests`. |
| Stall diagnostic (D12) | `scripts/research/stall_diagnostic.py` — frozen capacity-vs-curriculum classification rule; recorder side rides along with M5 eval runs. |

Not implemented, deliberately (gated): Change B schedule config (needs the robotixx
termination-manager path dry-run first), Change C disagreement probe (after M5-L verdict),
closed-loop controller M5.6 (after M5-T works).
