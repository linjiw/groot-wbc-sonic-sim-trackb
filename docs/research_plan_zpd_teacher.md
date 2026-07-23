# Research plan: ZPD-teacher motion sampling for SONIC — the curriculum-MaxRL integration

Date: 2026-07-22 (v1.1 amendment 2026-07-23 — §4 activation sub-gate re-frozen per
forecast finding Z6, BEFORE any GPU run; see §4 for the amendment record)
Status: **v1.1 — post-expert-review design plan, GPU-free tooling implemented.** This
document is the major research goal going forward. It amends `fable-next.md` §2
Phases 3–5 (SIM-M5/M6/M7) with a concrete, expert-reviewed mechanism family; Phases
0–2 (SIM-M4b diagnosis, SIM-D1 headroom gate) are unchanged and remain hard
prerequisites.

Provenance chain (all tracked in this repo):
1. Our analysis → `docs/research_curriculum_maxrl_integration.md` (the handoff doc, §8 = ten questions).
2. Expert response → `docs/external/SONIC_RESPONSE.md` (answers keyed Q1–Q10).
3. Reference implementation → `external_dependencies/curriculum-maxrl/` (vendored snapshot:
   `frontier_rl/teacher.py` FrontierTeacher, `frontier_rl/streaming.py` kernel teacher,
   `curriculum_maxrl/{THEORY,PROOFS,VALIDATION}.md`).
4. This plan = the design decisions locked from that exchange, the code design, and the
   preregistered experiment program.

Implementation status (2026-07-23, commits `8a52afe`..`dd30304`):
- §3.1 Change A: **implemented + tested** (`motion_lib_base.py`, 27 unit tests, byte-identity verified).
- §3.4.1 dynamics-sim modes + preregistered forecasts: **done** (Z1–Z6, CTRL1–3; artifacts
  under `docs/artifacts/sim_m5/`).
- §3.2 eval-strip unit test: **done** (guard in `validate_spec`, 6 tests).
- §3.4.3 retention metric plumbing: **done** (`eval.easy_decile.*`, 8 tests).
- §3.4.2 stall-diagnostic skeleton: **done** (`scripts/research/stall_diagnostic.py`, 8 tests;
  eval-time recorder side rides along with the M5 runs on robotixx).
- §3.2 Change B schedule config + dry-run of the termination-manager attribute path: **pending
  robotixx** (hard launch precondition, unchanged).
- §3.3 Change C disagreement probe: **deliberately not implemented** (gated on an M5-L verdict, D11).
- M5.6 closed-loop controller: **deliberately not implemented** (gated on M5-T working, D7).

---

## 0. Thesis (the major research goal)

**The field-default failure-rate motion sampler allocates by a difficulty-monotone
utility, so it over-commits sampling mass to unlearnable (impossible-at-budget) motion
bins and under-serves the learnable frontier. Replacing the utility with a
ZPD-shaped one — learnability `E[p(1−p)]` under a decayed Beta posterior over per-bin
survival — reallocates that mass to the frontier at near-zero code risk, and a
termination-threshold curriculum manufactures a frontier where the dataset provides
none. Together they form a theory-grounded, preregistered correction to the PHC-lineage
sampler that every current humanoid-tracking stack ships blind.**

Paper claim architecture (updates `fable-next.md` §3):

- **C1 (banked, unchanged):** preregistered validity-gated methodology.
- **C2 (strengthened):** the diagnosis now has a *mechanism-level theory*: failure-rate
  sampling is monotone in difficulty → mass on hazard≈1 bins is structural, not
  incidental. Forecast: 43–53% of non-uniform mass on impossible bins in mixed regimes
  (§5 of the handoff doc). The in-code comment block at `motion_lib_base.py:2620-2650`
  (max_prob rationale: "impossible motions … still get sampled heavily, wasting
  compute") shows the original authors knew the symptom; we supply the corrected
  utility family rather than a cap.
- **C3 (the risk claim, sharpened):** ZPD-utility sampling (± threshold curriculum)
  improves tracking at matched budget vs uniform AND vs failure-rate, on data with
  demonstrated difficulty headroom, ≥5 seeds, CI excluding 0.
- **Fallback (unchanged, still strong):** "When does adaptive motion sampling help?" —
  boundary map, now with the utility-shape axis added (monotone vs ZPD), which makes
  the negative branch *more* publishable, not less: even a null result contrasts two
  utility families under one preregistered harness.

---

## 1. Locked design decisions (from the expert exchange)

| # | Decision | Source | Supersedes |
|---|---|---|---|
| D1 | Lead utility = **learnability** `E[p(1−p)]` closed-form under Beta posterior; advantage-mass `u(p,N)` demoted to ablation (no natural N in our non-grouped setting → N is a tuned knob → loses the zero-hyperparameter property) | Q1, Q2 | handoff §7-A which defaulted to offering both equally |
| D2 | `p` = per-bin **survival (hazard complement)** from the existing occupancy counts — finer-grained than pass@1, difficulty-monotone, per-visit-normalized. Documented as a *deviation* from the MaxRL theory, not a violation | Q10.2 | — |
| D3 | **No Thompson sampling.** Deterministic optimism: utility maximized over the interval `[p̄−k·σ, p̄+k·σ]` (k=1) of the Beta posterior (u is unimodal → closed form: `u(p*)` if the interval contains the peak, else max of endpoints). Uniform floor already guarantees coverage | Q3 | — |
| D4 | Decay = **half-life in episode-equivalents** per bin: at recompute, `counts_old *= 0.5^(Δn_b / H)` where `Δn_b` = new episode-equivalents for bin b since last recompute. Scale-invariant across 8→4096 envs. Start H ≈ 3–5× mean per-bin visits per recompute at release scale. **Long memory when the threshold controller is active** (coupled-system finding: slow controller ⇒ quasi-stationary verifier ⇒ sampling variance dominates ⇒ fast forgetting only amplifies it) | Q4, Q7.2 | handoff sketch (per-recompute ×0.9) |
| D5 | **γ = 1** (no sharpening). Bins are directly settable via RSI → no unlock ordering → the compounding mechanism behind γ≈4 is absent; expert's own flat-pool ablation showed γ-sensitivity ≈ 0 with worse finals at high γ | Q5 | — |
| D6 | **No trajectory-level hindsight.** The per-step occupancy update *is* the on-policy port of hindsight (statistics half) and must be preserved when swapping utilities: the Beta posterior is built from occupancy (traversal, failure) counts. Optional remnant: **family-kernel evidence sharing** across speed-scaled variants (D-B synthetic set) — a success at ×1.5 is evidence about ×1.0's difficulty; posterior-level kernel, no conditioning rewrite, no off-policy issue | Q6 | handoff §5.3 |
| D7 | Threshold curriculum: **static one-sided schedule first** (loose→strict), closed-loop controller only after static works, and then with gain η ≈ 0.05, per-iteration Δτ cap, **one-sided (only tighten)** — the diagnosed failure mode is overshoot-and-pin, not oscillation. Target = frontier-band center (global early-termination rate 0.3–0.5), NOT ln N/N (inherits the no-N problem). No posterior reset on threshold moves | Q7 | handoff §7-B closed-loop-with-lnN/N-target |
| D8 | Forgetting: retention behavior of dense-reward PPO under frontier sampling is an **empirical unknown** (H6 doesn't transfer either direction). Easy-decile retention metric in every arm from day one; escalation ladder if it degrades: (a) floor 0.1→0.2, (b) anchor mass on mastered bins, (c) ALP-style `|Δp̂|` re-injection of regressing bins (~10 LOC, expert's best pure teacher) | Q8 | — |
| D9 | **Drop the mean×200 cap for ZPD utilities** (bounded by ¼, self-limiting); repurpose the dormant `max_prob_per_bin` knob (`:2632-2635`) as a loose safety tripwire (20× uniform) + alert threshold on existing `effective_num_bins` telemetry. Cap becomes a preregistered safety assertion, not a shaping mechanism | Q9 | — |
| D10 | §5.1 wording calibration: dense PPO lacks the *zero-gradient* pathology, but has the ***useful-gradient* pathology** (saturated bins → advantage variance ≈ 0; impossible bins → gradient mostly re-optimizes the known pre-failure prefix). This is the precise justification for the teacher and predicts *where* it helps | Q10.1 | handoff §5.1 |
| D11 | Empirical-|A| signal (Change C) = **disagreement probe after the lead arm has a verdict**, per-step-normalized (mean |A| per occupied step, never sum — episode-length and reward-weight confounds), expected to agree with learnability at the ends and differ mid-band; disagreement is the finding | Q1.2 | — |
| D12 | Stall diagnostic: if frontier bins stall under the new teacher, measure **per-step tracking-error growth rate within a bin vs bin length** before blaming the sampler — separates curriculum-problem from capacity-problem (expert: in their case the answer was capacity and the teacher was exonerated) | closing note | — |

---

## 2. Gate structure (amended program)

```
SIM-M4b  diagnosis (robotixx sync + one command)          [unchanged, NEXT, blocking]
   │
SIM-D1   dataset difficulty-headroom gate                  [unchanged, hard prerequisite]
   │        (D-A: BONES-SEED HF request; D-B: speed-scaled synthetic bridge)
   │
SIM-M5   mechanism probes (this plan's core)               [restructured]
   ├── M5-L   ZPD teacher, signal=learnability            [lead arm]
   ├── M5-T   static threshold schedule (config-only)     [co-lead, independent axis]
   ├── M5-LT  combination                                  [the expected winner]
   └── M5-A   advmass-N ablation                           [only if M5-L activates]
   │
SIM-M5.5 disagreement probe: per-step-normalized Σ|A| vs learnability   [D11]
SIM-M5.6 closed-loop threshold controller                  [only if M5-T works; D7 rules]
   │
SIM-M6   headline effect gate (5 seeds × 3 arms × 500–1000 iters)       [unchanged shape]
SIM-M7   (stretch) staged/competence-gated vs best reactive             [unchanged]
```

Standing guardrails (`fable-next.md` §4) all still apply — notably: flag-gated
default-off byte-identical mechanisms (7), no scaling before activation+SIM-D1 (3),
≥5 seeds for claims (5), simulations are forecasts never headlines (11), frozen
preregistration constants (12).

---

## 3. Code design

### 3.1 Change A — ZPD teacher in the sampler (M5-L; ~120–180 LOC + tests)

**Config** (all under `adaptive_sampling` in
`config/manager_env/commands/terms/motion.yaml`; defaults reproduce release exactly):

```yaml
adaptive_sampling:
  # existing keys unchanged ...
  signal: failure_rate        # failure_rate | learnability | advantage_mass
  evidence_half_life: null    # H in episode-equivalents; null = no decay (release)
  optimism_k: 0.0             # 0 = posterior mean only; 1.0 = interval-max optimism
  advmass_n: 16               # advantage_mass only (M5-A ablation)
  family_kernel: null         # optional {axis: speed, weights: {1: 0.25}} — D6 remnant
```

**Implementation** — all in `motion_lib_base.py`, tensor-level, no new call sites:

1. **Decay** (D4): in `sync_and_compute_adaptive_sampling` (`:2501`), before adding
   this-iteration counts is not possible (counts accumulate per step), so track
   `_adp_samp_counts_at_last_recompute`; at recompute, compute per-bin
   `delta_n = num_episodes − last_episodes`, then rescale the *old* portion:
   `old = last_counts * 0.5**(delta_n / H)`, `new_counts = old + delta_this_window`.
   Applied to both episodes and failures. Byte-identical when `evidence_half_life`
   is null. (Note the existing dormant `use_failure_rate_decay` is a *different*
   mechanism — backward propagation across bins — leave untouched.)

2. **Posterior + utility** (D1–D3): in `update_adaptive_sampling_probabilities`
   (`:2568`), branch on `signal`:

   ```python
   if signal in ("learnability", "advantage_mass"):
       succ = (num_episodes - num_failures).clamp(min=0)
       a = 1.0 + succ                     # Beta over per-bin survival prob
       b = 1.0 + num_failures
       p_mean = a / (a + b)
       if optimism_k > 0:                 # D3: interval-max over unimodal u
           var = a * b / ((a + b).square() * (a + b + 1.0))
           lo = (p_mean - optimism_k * var.sqrt()).clamp(0.0, 1.0)
           hi = (p_mean + optimism_k * var.sqrt()).clamp(0.0, 1.0)
           utility = torch.maximum(u_fn(lo), u_fn(hi))
           peak_inside = (lo <= P_STAR) & (P_STAR <= hi)   # learnability: P_STAR=0.5
           utility = torch.where(peak_inside, u_fn(P_STAR_T), utility)
       else:
           utility = u_fn_expected(a, b)  # learnability: a*b/((a+b)*(a+b+1))
       # NO mean×cap clip (D9); tripwire via max_prob_per_bin instead
   else:
       utility = clipped_failure_rate     # existing path, byte-identical
   ```

   with `u_fn(p) = p*(1−p)` for learnability (peak p\*=0.5),
   `u_fn(p) = (1−(1−p)^N) − p` for advantage_mass (peak `1 − N^(−1/(N−1))`).
   Note for learnability + optimism-off, `u_fn_expected` is the exact closed form
   `E[p(1−p)] = a·b/((a+b)(a+b+1))` — preferred over `u(p̄)` (Jensen).

3. **Downstream unchanged:** normalize → `0.9/0.1` uniform blend →
   `sequence_length_agnostic` bin weights → renorm. All 16 telemetry keys keep
   emitting; the tripwire (D9) sets `max_prob_per_bin: 20.0`-equivalent via the
   existing dormant path (`:2632+`, "float value" branch) **only for ZPD signals**.

4. **Family kernel** (D6, only for the D-B synthetic set): after count updates, add
   `κ`-weighted pseudo-counts from same-motion neighboring speed bins into the
   posterior (not into the raw counts — telemetry stays honest; a parallel
   `a_shared/b_shared` used only for utility). Flag-gated; default null; micro-scale
   only.

5. **New telemetry** (extends the 16 keys; same guarded-emit pattern
   `manager_env_wrapper.py:921-968`): `adp_samp/posterior_p_mean_{min,max,mean}`,
   `adp_samp/utility_{max_over_uniform,entropy}`, `adp_samp/decay_effective_window`,
   `adp_samp/tripwire_max_prob_binding` (0/1). Update
   `ADP_SAMP_CLASSIFICATION_KEYS` consumers only additively.

**State:** `get_state_dict`/`load_state_dict` (`:2426-2459`) gain the
`_counts_at_last_recompute` tensors (backward-compatible: absent keys → release
behavior). Eval still never restores sampler state — unchanged, and irrelevant since
eval forces uniform assignment.

**Tests** (`tests/research/` + a new `tests/` unit module colocated with existing
motion-lib tests if any; CPU-only): byte-identity when `signal` unset (golden-tensor
comparison across a scripted count sequence); learnability closed form vs Monte Carlo;
decay half-life invariance (same evidence in 1 vs 10 recomputes → same posterior);
optimism interval-max correctness incl. peak-inside case; tripwire binding; NaN-guard
regression (init=0 path stays fixed); family-kernel neutrality when null.

### 3.2 Change B — threshold curriculum (M5-T; config + ~40 LOC of guards)

**Static schedule** (first `schedule_dict` in the repo; engine
`trl/utils/scheduler.py:296-353`, applied at `ppo_trainer.py:1702-1706`):

```yaml
trainer:
  schedule_dict:
    "env@<verified-path>@get_term_cfg('anchor_pos')@params@threshold":
      { type: linear, seg_steps: [0, 150], seg_vals: [0.30, 0.15] }
    "env@<verified-path>@get_term_cfg('ee_body_pos')@params@threshold":
      { type: linear, seg_steps: [0, 150], seg_vals: [0.30, 0.15] }
```

One-sided by construction (monotone segments). **Before any run:** (i) dry-run the
attribute path against the live termination manager (scheduler `__main__` shows the
path syntax; the exact env wrapper chain must be verified once on robotixx);
(ii) **the eval-strip unit test** — expert-endorsed promotion from note to test:
assert that materialized eval configs contain no `schedule_dict` entries touching
termination params (pitfall: `eval_agent_trl.py:466-470` re-applies schedules;
`:134-142` strips only `train_only_events`-scoped ones). Simplest robust guard: the
research spec generator strips `trainer.schedule_dict` from eval invocations entirely,
with a test over the materialized commands.

**Closed-loop controller** (M5.6, conditional): ~30 LOC in the env wrapper reading the
global early-termination rate per iteration; `τ ← clip(τ − η·(fail_rate_target −
fail_rate_obs)·sign_convention, τ_min, τ)` — **one-sided (never loosens), η=0.05,
|Δτ| ≤ 0.002/iter cap** (D7). Target band 0.3–0.5. Runs only with long posterior
memory (large H). Flag-gated `terminations.curriculum_controller.enable`.

### 3.3 Change C — disagreement probe (M5.5; ~80 LOC, later)

Per-bin **mean |GAE advantage| per occupied step** (never sum — D11): the motion
command knows each env's bin per step; the trainer has per-step advantages;
accumulate `(Σ|A|, Σsteps)` per bin per iteration in the wrapper, expose as
`adp_samp/emp_advmass_*` telemetry **without driving sampling** in the first run —
log-only alongside M5-L, then compare rankings (Spearman + top/bottom-decile
agreement). Only if mid-band disagreement is large and interpretable does it become a
sampling arm.

### 3.4 Diagnostics and tooling (GPU-free, build before launch)

1. **Extend `sampler_dynamics_sim.py`** with `learnability`/`advantage_mass` modes +
   evidence-scaled decay + optimism, replacing the throwaway forecast script
   (handoff Appendix A); rerun the four-regime forecast as the **preregistered
   forecast artifact** with tests. Add the coupled threshold-controller simulation
   (expert's Q7 appendix) as a scenario, reproducing overshoot-and-pin at η≥0.3 as a
   validation target.
2. **Stall diagnostic** (D12): small analysis script over eval per-motion metrics +
   recorded per-step errors: per-bin error-growth-rate fit vs bin length →
   "capacity-limited" vs "curriculum-limited" classification. Build the skeleton now;
   it gates the *interpretation* of any M5 stall.
3. **Retention metric** (D8): extend `summarize_sonic_logs.py`/eval parsing with
   `eval.easy_decile.mpjpe_g` (easiest decile by the SIM-D1 all-motions difficulty
   ranking — frozen per dataset at D1 time). Wire into
   `compare_sonic_manifests.py` `_METRIC_PATHS` (informational) and give it a
   preregistered **non-inferiority bound** in the effect gates below.

---

## 4. Preregistered gates (frozen here, before any run)

All experiments: `run_sonic_multiseed.py`, paired seeds, frozen eval settings, all
existing validity gates. Screens = 3 seeds (0–2), 200 iters, num_envs 8–16; claims =
≥5 seeds. Dataset: SIM-D1-passing only.

**Activation sub-gate** (per arm, ≥2/3 seeds, from committed telemetry + checkpoint
dump). **AMENDED 2026-07-23 per preregistered forecast finding Z6**
(`docs/artifacts/sim_m5/zpd_forecast_preregistered.json`, before any GPU run): the
v1 targeting criterion (hard-half/easy-half mass ratio ≥ 1.5) is correct for
*failure-rate* arms but MISFIRES on ZPD utilities — the criterion is
difficulty-monotone while the ZPD utility deliberately down-weights the impossible
bins that populate the hard half (forecast: ratio ≈ 0.94 for a correctly-working
learnability teacher). Activation is therefore signal-family-specific:

- **failure_rate arms (unchanged):** peakedness (pmax/uniform ≥ 10 AND ≥ 1
  concentrated bin) OR hard-half/easy-half mass ratio ≥ 1.5.
- **ZPD arms (M5-L, M5-A): frontier over-allocation** — sampling mass on the
  SIM-D1 frontier band (middle difficulty tercile of evaluated bins) divided by
  that band's uniform share ≥ 1.2, **AND posterior sanity** — `posterior_p_mean`
  rank-correlates with the SIM-D1 per-motion difficulty ranking (Spearman ≥ 0.4
  over bins of evaluated motions; forecast confirms ≈ 0.86 is achievable in the
  mixed regime). Peakedness remains sufficient if it fires (not expected —
  forecast pmax/uniform 1.0–2.0).

Forecast-informed knob note (not a gate): optimism k=1 slightly *reduced* frontier
mass vs k=0 in the forecast (0.69 vs 0.76); the M5-L screen leads with
`optimism_k: 0` and treats k=1 as the one preregistered knob alternative.

**Effect sub-gate** (screen): mean MPJPE-G delta (arm − uniform) ≤ −0.5, ≥2/3 seeds
improved, `ok_for_causal_comparison=true` everywhere.

**Retention sub-gate** (every arm, D8): easy-decile MPJPE-G delta (arm − uniform)
≤ +0.5 (non-inferiority; mirrors the effect threshold). Violation ⇒ escalation ladder
D8(a)→(c), max one revision, re-preregistered.

**Tripwire assertion** (D9): `tripwire_max_prob_binding` must be 0 for ≥95% of
iterations; binding more often ⇒ run is *invalid-unstable*, not negative.

**M5-T-specific:** the schedule must produce a mid-run early-termination rate in
[0.15, 0.6] band for ≥50% of iterations (the "manufactured frontier" existence check);
outside ⇒ invalid-inactive (schedule mis-calibrated, one recalibration allowed).

**Decision routing:** M5-L activates + effect passes → M5-LT, then SIM-M6 with
{uniform, failure_rate, winner}. M5-L activates but effect fails → the boundary-map
result stands (utility family changes allocation but not outcome at this scale) —
publishable as-is; probe M5.5 for signal-quality explanations. M5-L fails to activate
on SIM-D1-passing data → check posterior sanity; if posterior is sane but diffuse,
run the stall diagnostic (D12) before any knob revision.

**SIM-M6 headline gate:** unchanged from `fable-next.md` Phase 4 (5 seeds, ≤ −1.0,
≥4/5, bootstrap CI excludes 0) plus the retention non-inferiority bound at the same
seed count.

---

## 5. Execution order and timeline (from 2026-07-22)

| When | What | Where |
|---|---|---|
| now | SIM-M4b: sync + `classify_sampler_diagnosis.py` + validate F1/F3; close issue #4; **push the 3 local commits**; D-A HF request out | robotixx + here |
| wk of Jul 27 | SIM-D1 all-motions eval (candidates: sample_data retro-run [expected FAIL], D-B speed-scaled set, BONES-SEED if access lands); freeze per-dataset difficulty ranking + easy decile | robotixx |
| parallel, GPU-free | §3.4 tooling: dynamics-sim modes + preregistered forecast; eval-strip unit test; retention metric plumbing; stall-diagnostic skeleton; Change A implementation + full test suite | this checkout |
| wk of Aug 03 | M5-L + M5-T screens (3 seeds × 200 iters); telemetry + checkpoint dumps synced to `docs/artifacts/sim_m5/` | robotixx |
| wk of Aug 10 | M5-LT (if either activates); M5.5 log-only probe rides along free | robotixx |
| wk of Aug 17+ | SIM-M6 (5 seeds) on the winner, or fallback boundary-map write-up begins | robotixx |
| Sep | paper draft: C1+C2(+C3) — RLC 2027 primary; ICBINB / CoRL-workshop interim | — |

Issue hygiene: one issue per gate (M5-L, M5-T get separate issues, template from #4);
each closes with artifacts + tag + status-doc update.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| SIM-D1 finds no dataset with headroom (BONES-SEED gated-access latency) | D-B synthetic speed-scaled set is the bridge (never headline); fallback paper path preregistered |
| ZPD utility activates but effect is null at micro scale | Still publishable (utility-family axis of the boundary map); M5.5 probes whether the binary hazard was the wrong signal |
| Retention degrades under frontier sampling (dense-PPO unknown, D8) | Metric in every arm from day one + preregistered escalation ladder |
| Threshold path into Isaac Lab termination manager differs from `get_term_cfg` sketch | Dry-run verification step is a hard launch precondition (§3.2) |
| Eval schedule re-application corrupts comparisons | Promoted to unit test before any M5-T run (§3.2) |
| Posterior flukes at tiny visit counts concentrate mass | Tripwire (D9) as validity assertion, not shaping |
| Expert's coupled-sim findings don't transfer (their competence model is a toy) | Controller is conditional (M5.6), one-sided, rate-limited; static schedule carries the claim |

---

## 7. What this plan deliberately does NOT do

- No MaxRL estimator / advantage-weighting changes in PPO (expert-endorsed drop).
- No trajectory-level hindsight relabeling (D6; on-policy contracts fail).
- No Thompson sampling (D3; determinism guardrail).
- No γ sharpening (D5).
- No `ln N/N` controller target (D7; no natural N).
- No pre-implementation of `staged`/competence-gated arms before their gate (unchanged
  from `fable-next.md` §5) — the CG-WBC port remains SIM-M7 stretch.
- No reward-term or DR changes in this program (separate axes; see 2026-07-21 review).
