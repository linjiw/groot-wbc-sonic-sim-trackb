# LACE × SONIC — Failure Geometry and Training Transfer

## Repository integration plan v0.3 — 2026-08-13

This document turns the LACE v0.2 proposal into a plan that can be executed in this
GEAR-SONIC checkout on one RTX 5090. It also records the changes required after checking
the actual release code and the August 2026 literature.

The previous content of this file was a separate predictive-latent-correction proposal.
It is preserved verbatim in `docs/research_plan_predictive_latent_correction_v3.md`.

## 0. Decision

Proceed with LACE, but do not implement v0.2 literally.

The strongest and cheapest paper is RQ1: establish whether a frozen, policy-conditioned
failure representation predicts cross-motion training transfer. RQ2 starts only if RQ1
passes. RQ3 starts only if a non-language LACE allocator beats SONIC's released adaptive
sampler.

Four corrections are mandatory:

1. Separate **failure type** from **failure frequency**. If the signature is
   `P(c | m)`, its magnitude contains scalar difficulty and makes the comparison against a
   scalar-difficulty baseline partly circular. Use a factorization instead:

   ```text
   a_m          = P(failure | m)                    # scalar difficulty
   a_m,resolved = P(resolved failure | m)           # attributed failure mass
   q_m(c)       = P(c | resolved failure, m)        # conditional mechanism
   f_m(c)       = a_m,resolved q_m(c)               # observed joint mass
   ```

   H1's mechanism-only comparison uses `q_m` on common resolved support. Report unresolved
   mass explicitly. The old `a_m q_m` construction assumes unresolved failures are missing at
   random and is sensitivity-only. Report `a_m`, `q_m`, and `[a_m, q_m]` separately.

2. Do not assume three official SONIC checkpoints exist. This checkout contains two names
   for one identical released checkpoint. Their SHA-256 is
   `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.
   The primary atlas will therefore use frozen early/middle/late checkpoints from a
   reproducible SONIC-Lite baseline trajectory. The released 42M checkpoint is an
   external-validity assignment and a cross-policy transport audit. If official weak/middle checkpoints
   become available, add them as an ablation.

3. Keep the mechanism vocabulary dimension `C` distinct from the comparison partition
   count `K`. Start with six programmatic mechanism channels (`C=6`). Cluster motion
   representations at matched `K in {4, 6, 8}` only for representation comparisons. Do not
   use bootstrap ARI to silently change the number of named mechanisms.

4. RQ3 needs a **matched structured-action non-language baseline**. Comparing a sampling-only
   bandit with an LLM that can also change augmentation and domain randomization would test
   action-space size, not language. The decisive comparison is a contextual bandit over the
   same whitelisted structured interventions versus LLM selection over that same list.

## 1. What is verified in this checkout

### 1.1 Repository and hardware state

As of 2026-08-13:

- Active branch: `research/cg-wbc-v0-golden-path`.
- The branch was fast-forwarded from `d2d60a8` through `f7dc739` to `7479297`.
  It matches `trackb/agent/official-bones-zpd-handoff` and is four commits ahead of
  `trackb/main`.
- All remotes were fetched and pruned; Git LFS objects were pulled.
- Host GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 590.48.01.
- Approximately 8.9 GiB VRAM is currently occupied, including a separate
  `critic_scorer_daemon.py` process using about 7.35 GiB. Do not stop it without explicit
  coordination.
- The repository filesystem has only about 3.7 GiB free, but `/data` has about 389 GiB free.
  All large LACE artifacts are rooted at
  `/data/robotixx/groot-wbc-sonic-research/lace`; repository-root storage is limited to code,
  compact configs, and digests.
- `/data/robotixx/groot-wbc-sonic-research/datasets` contains the verified source archives and
  a fully materialized, paired 512-motion BONES cohort (512 G1 PKLs and 512 SMPL PKLs). Its
  paired dataset SHA-256 is
  `0f0ad3a0f41b33ba5fdc28fdcdae33156f876769dba3218e64f0e914d17fb05a`.
- A five-way source-disjoint split for that cohort is frozen at
  `/data/robotixx/groot-wbc-sonic-research/lace/manifests/bones_seed_official_scale512_split_v1.json`.
  It contains 268 BONES actor/source groups and has path-independent selection digest
  `4a9d530513b15557a556a8a790e5dab49c11b796d84aa0384c3db0a283e555e7`.
  The tracked lock is `configs/research/lace/bones_seed_official_scale512_split_lock.json`.
  This cohort is suitable for implementation and atlas pilots. `D_geometry + D_test` contains
  only 61 source groups, so it is not the preregistered headline RQ1 inference target.
- The pre-outcome headline cohort is now frozen separately at 4,950 selected BONES motions.
  Selection used only official metadata, the release filename filter, category × duration
  strata, and source-group counts. On the declared 4,500–5,500 candidate grid (step 50), 4,950
  was the unique size satisfying `1,000 <= |D_atlas| <= 1,024` and at least 100 combined
  `D_geometry + D_test` source groups. The protocol is
  `configs/research/lace/headline_cohort_selection_v1.json` (schema 2, self digest
  `379c28a2...`). It explicitly excludes and records the prior two-motion instrumentation smoke,
  and binds the cohort builder, release-filter implementation, materializer, paired-manifest
  builder, metadata, member lists, and their exact bytes.
- The paired headline dataset is materialized under
  `/data/robotixx/groot-wbc-sonic-research/datasets/bones_seed_official_headline_scale4950`.
  It contains exactly 4,950 G1 and 4,950 SMPL files, occupies about 1.5 GiB, and has paired
  dataset SHA-256 `f93d6200...`. Its final source-disjoint split has 487 source groups:
  `D_atlas=1006/88`, `D_curriculum=1996/257`, `D_geometry=822/63`,
  `D_controller=557/38`, and `D_test=569/41`, where each pair is motions/source groups.
  `D_geometry + D_test` therefore contains 104 held-out source groups. The tracked lock is
  `configs/research/lace/bones_seed_official_headline_scale4950_split_lock.json`; the split
  digest is `53244389...`. Scientific split readiness is not optional: it rebuilds all 21
  declared candidate sizes, proves that 4,950 is the sole passing size, reconstructs the selected
  cohort and source-disjoint split, and revalidates the exact flat 4,950-pair dataset and aggregate
  hashes. An independently retained lock-file digest/timestamp is still required before outcomes;
  this deep local reconstruction does not pretend to be that external anchor. The verifier now
  requires that independent value via `--expected-headline-split-lock-file-sha256`; the current
  candidate lock bytes hash to `1fc4a14e...`, but that value becomes preregistration evidence only
  after it is recorded outside the mutable workspace.

### 1.2 SONIC's real baseline is adaptive, not uniform

The competitive-landscape dependency is resolved for SONIC itself:

- `gear_sonic/config/manager_env/commands/terms/motion.yaml` sets
  `adaptive_sampling.enable: true`.
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml` inherits that
  setting and raises `adp_samp_failure_rate_max_over_mean` to `200`.
- `MotionLibBase` partitions each motion into 50-frame/one-second bins, estimates early
  termination frequency, caps large failure rates, mixes the result with a 0.1 uniform
  floor, and samples bin/start-time pairs from the resulting distribution.

This matches the current [SONIC paper](https://arxiv.org/abs/2511.07820) and means the
primary RQ2 baseline is **SONIC native capped failure-rate sampling**, not uniform sampling.
Uniform remains a sanity floor.

### 1.3 Existing Track-B infrastructure to reuse

The update brought in reusable research machinery:

- `scripts/research/run_sonic_paired_experiment.py` and
  `scripts/research/run_sonic_multiseed.py` for controlled paired runs.
- `scripts/research/paired_stats.py` and
  `scripts/research/aggregate_sonic_comparisons.py` for paired analysis.
- `scripts/research/summarize_sampler_telemetry.py` and
  `scripts/research/dump_sampler_checkpoint_state.py` for sampler observability.
- `scripts/research/verify_schedule_path.py` plus the eval schedule-strip test for guarding
  train-only interventions.
- `scripts/research/build_bones_seed_official_cohort.py`,
  `materialize_bones_seed_official_cohort.py`, and the source-lock verifier for reproducible
  BONES subset construction.
- The ZPD signal family in `gear_sonic/utils/motion_lib/motion_lib_base.py`, including
  posterior utilities, evidence decay, probability tripwires, checkpoint state, and unit
  tests.

LACE should extend these paths rather than introduce a second experiment framework.

### 1.4 Existing official-cohort result changes the launch order

The newly accepted handoff contains a completed seed-0, 128-motion comparison between native
failure-rate sampling and ZPD learnability. Both 200-iteration fine-tunes regressed sharply from
the released initialization; ZPD also failed its preregistered MPJPE-L and AUC screens. This is
not evidence against LACE, but it is evidence that the current release-checkpoint fine-tuning
control is unsafe for detecting curriculum gains.

Before any Layer-C LACE comparison, establish a non-regressing control by testing a shorter
horizon, lower actor learning rate, or preserved optimizer state. Do not interpret a curriculum
delta when both arms are degrading from iteration zero. The exact result and hashes are in
`docs/research_official_bones_zpd_handoff.md`.

## 2. Updated scientific claim and novelty boundary

The claim under test is:

> Conditional failure-mechanism similarity predicts cross-motion training transfer beyond
> reference kinematics, motion semantics, and scalar difficulty; that geometry can improve
> fixed-budget allocation over SONIC's native failure-rate sampler.

The phrase **conditional failure mechanism** matters. LACE is not simply a richer hard-motion
miner.

The August 2026 landscape raises the novelty bar. `VERIFY.md` records the dated, primary-source
evidence and pinned release-code links:

- [SONIC](https://arxiv.org/abs/2511.07820) already uses bin-level capped failure-rate
  sampling by default.
- [EGM](https://arxiv.org/abs/2512.19043) already proposes bin-based cross-motion curriculum
  sampling based on tracking error.
- [Athena-WBC](https://arxiv.org/abs/2607.04837) already routes long-tail motions to
  capability-aligned experts with mechanism-specific training changes.

Therefore LACE must not claim that mechanism-aware specialization or adaptive cross-motion
sampling is new by itself. The defensible novelty is the **measured transfer geometry**:

1. identify policy-conditioned failure mechanisms using frozen probes;
2. test whether their distances predict causal cross-motion training gains;
3. allocate a fixed interaction budget in that coordinate system without adding experts or
   increasing model capacity.

Athena-WBC becomes a required related-work comparison and motivates an additional question:
does allocation alone recover gains that otherwise require capability-specific experts or
objective changes?

ASPIRE prevents a broad claim around language-based failure diagnosis and validated repair;
ZEST is prior art for structured hard-segment intervention; FAST is prior art for efficient
hard-motion residual adaptation. OmniTrack makes reference feasibility a mandatory control in
RQ1: reject corrupt references and include a reference-feasibility covariate so a mechanism
signature cannot win merely by rediscovering infeasible data.

## 3. Dependency graph and stop rules

```text
Operational readiness
  -> stable SONIC-Lite baseline and frozen checkpoint ladder
    -> stable failure atlas and cross-policy transport audit
      -> RQ1 transfer geometry
        -> RQ2 non-language allocation
          -> RQ3 structured interventions and language
```

Stop rules:

- No full atlas until storage, dataset provenance, and deterministic rollout replay pass.
- No scientific rollout until its external analysis protocol has an independently trusted
  pre-outcome lock (a reviewed Git commit/tag or append-only registry entry); a self-hash stored
  beside mutable output paths is not evidence of preregistration.
- No transfer matrix until signature stability and the cross-policy transport audit are reported.
- No curriculum headline until H1 beats the strongest non-failure representation out of
  sample.
- No LLM integration until a mechanism allocator beats native SONIC at the H2 gate.
- No real-robot test in v1; LACE v1 is simulator-first and uses privileged signals.

## 4. Data protocol: use five source-disjoint sets

The original four-way split still reuses `D_probe` for too many scientific decisions. Use five
sets, grouped by source recording identity before any clip-level transformation:

| Set | Purpose | Allowed consumers |
|---|---|---|
| `D_atlas` | Fit normalizers and freeze mechanism/representation parameters | Atlas builder only |
| `D_curriculum` | PPO training and mechanism upweighting | Trainer plus frozen, no-refit representation assignment |
| `D_geometry` | H1 model selection and transfer prediction | RQ1 analysis only |
| `D_controller` | Online utility/bandit feedback | RQ2/RQ3 controller only |
| `D_test` | One-time final reporting | Final evaluator only |

The split generator must:

- derive a canonical `source_group_id` from dataset metadata, not motion semantics in the
  filename;
- keep all crops, speed variants, mirrored clips, and `_M` variants in the same group;
- stratify only with information available before policy rollouts;
- write a manifest containing paths, source group, duration, available modalities, and a
  SHA-256 digest;
- freeze the manifest before atlas construction;
- make `D_test` inaccessible to controller configs and reject it in train commands.

Suggested proportions are 20/45/15/10/10 percent by source group, but freeze the exact split
only after checking motion count, duration, and semantic coverage. The 1–3k target refers to
`D_atlas` itself, so the total five-way source cohort is necessarily larger. Prefer at least 100
source groups in `D_geometry` and `D_test` combined; otherwise inference will be dominated by a
handful of recording sessions. The frozen 4,950-motion headline cohort meets both requirements
with 1,006 atlas motions and 104 geometry/test source groups.

## 5. Frozen Failure Atlas v1

### 5.1 Probe-policy set

Primary `Pi_atlas`:

- `pi_lite_early`: first checkpoint with nontrivial survival but substantial headroom;
- `pi_lite_mid`: approximately halfway between early and plateau on held-out success;
- `pi_lite_late`: frozen plateau checkpoint.

Freeze these checkpoints at preregistered optimizer-update fractions of the baseline budget
(pilot default: 10%, 50%, and 100%), not by selecting checkpoints on any held-out partition or
by choosing stages that make clustering look good. Record checkpoint hashes.

External audit:

- `pi_sonic_release`: `sonic_release/last.pt`, rollout only.

The full 107-motion released-policy external-audit schedule is frozen at
`/data/robotixx/groot-wbc-sonic-research/lace/manifests/release_d_atlas_external_schedule_v1.json`
and tracked by `configs/research/lace/release_d_atlas_external_schedule_lock.json`. It contains
856 exact tuples (107 motions x two DR conditions x two phases x two repeats), has schedule digest
`74f5b41d...`, and rebuilds byte-identically. It is a Layer-A external audit only; it must not be
substituted for the three-checkpoint Lite policy axis in the primary H1 representation.

The primary representation is the ordered policy-axis concatenation frozen before any transfer
outcomes are observed:

```text
q_m = concat(q_m^early, q_m^mid, q_m^late)
```

Policy-averaged signatures are a declared sensitivity analysis, never an outcome-adaptive
replacement for the primary representation.

`build_factorized_signatures` intentionally emits one row per `(motion, policy)`. The downstream
combination is now explicit in `gear_sonic/research/lace/policy_axis.py`: it requires the exact
motion × ordered-policy Cartesian product, concatenates only `[early, mid, late]` in that order,
and emits the equal-policy mean under a sensitivity-only label. If any policy has fewer than the
frozen minimum resolved failures, that motion has no primary feature row; it is neither averaged
over the remaining policies nor imputed. The same deep-rebuild contract carries the ordered
per-policy scalar-difficulty axis and unresolved-failure nuisance for matched common-support tests.
Policy-axis schema 2 additionally requires the actual signature atlas, frozen normalizer, analysis
protocol, and checkpoint-bearing schedule/atlas parent plus four independently supplied expected
digests. Copying parent digests out of the stored feature artifact is invalid; the expected values
must come from the external preregistration chain, and every scientific parent is deep-validated
before feature construction.

The 107-motion scale-512 primary Lite schedule will contain `107 × 3 × 2 × 2 × 2 = 2,568`
rollouts. It is not materialized until the early/mid/late checkpoint hashes and the externally
anchored measurement protocol exist. This is a pipeline-validation scale only; the headline atlas
uses the separately frozen 1,006-motion `D_atlas`: `1006 × 3 × 2 × 2 × 2 = 24,144` exact
episodes. That schedule is also intentionally unmaterialized until the checkpoint and protocol
gates pass.

### 5.2 Fixed rollout conditions

For every `(motion, policy, xi, initial_phase, repeat)` tuple:

- use a frozen domain-randomization seed list `Xi`;
- derive and freeze a distinct runtime RNG seed for each `(xi, phase, repeat)` cell, shared
  across motions and policies; `repeat` must never mean replaying an identical reseeded rollout;
- use common random numbers across policies and representation baselines;
- use the same reset-state and initial-phase schedule;
- use plane terrain for atlas v1 and exclude terrain geometry from `Xi`; BONES reference
  contact labels use absolute flat-world foot height, so rough terrain would turn terrain-label
  mismatch into a fake contact-timing mechanism;
- disable training and adaptive resampling;
- emit exactly one scheduled episode per vector environment and quiesce its recorder buffer after
  first completion while slower environments finish; Isaac auto-reset must never create a second
  record with the same frozen rollout ID;
- run enough repeats to estimate incidence, not only a single deterministic playback;
- store aggregates plus short event windows around the first failure; do not store all raw
  state at 50 Hz unless needed.

Start with `R=8` rollouts per motion for the pilot. Increase only if bootstrap stability is
limited by within-motion rollout noise.

### 5.3 Six programmatic mechanism channels

Use measurable, non-language channels first:

| Channel | SONIC/Isaac source | Episode score |
|---|---|---|
| Contact timing | Reference `feet_l/feet_r` versus the existing unfiltered foot-body contact-force sensor | mismatch incidence and one-to-one onset offset; validate body-name/order and flat-ground semantics |
| Foot slip | Foot-link tangential-velocity proxy while actual contact is active | duration and q90 proxy speed; do not call this exact sole contact-point slip |
| Base drift | Reference versus robot anchor translation | root error growth and max drift |
| Balance/orientation | Anchor/body orientation residual | angular RMS and tilt; termination/fall labels remain diagnostics until a physical predicate is independently recomputed |
| Actuation saturation | Implicit-actuator requested-PD estimate versus clipped applied command and effort limit | requested ratio, clip gap, clipped-joint fraction and duration; not measured PhysX torque |
| Joint/pose constraint | Actual soft-limit proximity beyond reference proximity plus local joint-pose divergence | excess-limit incidence and residual growth; absolute reference proximity is a feasibility covariate |

The first implementation must verify the exact Isaac Lab tensor names for applied torque and
effort limits in the installed version. A missing tensor is a failed probe, not a column of
zeros.

Also record non-mechanism covariates:

- MPJPE and local/global variants;
- linear and angular velocity residuals;
- time-to-first-failure, exact reference start/length, and absolute reference-clip progress;
- the combined failure/timeout masks and the installed manager's last-trigger term, explicitly
  labelled as non-multi-hot; add independently recomputed raw predicates before causal
  termination-attribution claims;
- episode length and reference duration;
- policy, seed, motion key, source group, start frame, and DR parameters.

Each scalar stream is summarized by mean, standard deviation, q90, maximum, incidence, onset
time, and pre-failure slope where meaningful. Primary episode severity uses a frozen two-second
window ending at first failure (or the censored episode end); export its exposure duration and
left-censoring flag. Full observed-episode summaries are sensitivity diagnostics, not the primary
mechanism score.

### 5.4 From episode scores to a soft signature

For episode `r`, first map physical residuals to six nonnegative **mechanism-evidence scores**
with frozen, predeclared thresholds. These are not calibrated probabilities. Within a channel,
combine normalized components by an equal-weight mean so channels with more diagnostics do not
receive mechanically larger values; fit alternative weights only on `D_atlas` and preregister
them. Then divide each channel by a positive calibration scale fit on
`D_atlas` only. Identity scales are allowed for contract/smoke runs, but the implementation
must actually apply the declared scales; they cannot be provenance-only metadata. Preserve
co-occurrence and do not force a single cause label.

```text
a_m             = mean_r[failed_mr]
a_m,resolved    = mean_r[failed_mr and resolved_mr]
a_m,unresolved  = a_m - a_m,resolved
z_mr(c)         = [s_mr(c) / scale_c] / sum_c' [s_mr(c') / scale_c']
q_m(c)          = mean_{r: failed and resolved} z_mr(c)
f_m(c)          = a_m,resolved q_m(c)
```

Per-episode normalization is intentional: it estimates the conditional mixture of mechanisms
without letting one high-amplitude failure dominate several lower-amplitude failures. Report a
pooled-severity aggregation as a sensitivity analysis, because it answers a different question.

If a motion has fewer than three resolved failures in the pilot, set `q_m` to missing rather
than a uniform mechanism distribution. Such motions remain available through the exploration
floor. H1 compares representations on common supported motion/policy pairs and reports coverage;
the full-failure product `a_m q_m` is reported only as a missing-at-random sensitivity.

Use a frozen symmetric Dirichlet prior for the pilot, export marginal credible intervals and
posterior concentration, and preregister any later empirical-Bayes prior fit without using
transfer outcomes. Fit channel scales as the positive failed-episode q90 on `D_atlas`, requiring
at least 20 positive observations per channel for a scientific atlas. The atlas artifact contains
the frozen normalizer input digest, thresholds, mechanism names, policy hashes, exact schedule,
seed/config binding, and per-motion posterior summaries.

### 5.5 Fit artifacts versus assignment artifacts

Only the primary three-policy `D_atlas` collection may fit channel scales or representation
parameters. Every other policy or partition is processed through a separate, parent-linked
**assignment artifact**:

- released SONIC on `D_atlas` for the cross-policy transport audit;
- the three frozen Lite policies on all of `D_curriculum` and `D_geometry` for RQ1 features;
- `D_controller` for RQ2/RQ3 feedback; and
- `D_test` only after the final one-time opening.

An assignment artifact binds the parent atlas/normalizer digest, applies its scales and signature
configuration byte-for-byte, and records its own frozen rollout schedule, measurement-family
digest, policy hashes, coverage, and source collection. It never calls a normalizer fitter and
never refits centroids. Released SONIC versus Lite is therefore evaluated in the Lite coordinate
system; independently normalizing each policy would erase exactly the scale transport being tested.

For primary RQ1 feature construction, freeze full assignment schedules with the same
`2 Xi x 2 phases x 2 repeats = 8` condition grid and the three Lite checkpoints:

| Partition | Motions | Policies | Episodes |
|---|---:|---:|---:|
| `D_curriculum` | 233 | 3 | 5,592 |
| `D_geometry` | 76 | 3 | 1,824 |

The schedules exactly cover the named partition, share the atlas measurement protocol and
common-random-number coordinates, and bind partition-specific reference-length inventories. A
smaller grid is not silently substituted: it changes evidence support relative to the atlas and
requires separately frozen validation. No `D_test` assignment schedule exists until the final
evaluation gate opens it.

Assignment outputs preserve missing `q`. Fixed source panels are never redrawn to improve failure
coverage. Primary H1 restricts target motions to an identical common-support set for every compared
representation, reports supported mass for each source panel, and includes signed unresolved source
exposure as a shared nuisance covariate. Posterior-mean imputation is sensitivity-only. Freeze
minimum source- and target-coverage thresholds before training; if they fail, stop H1 rather than
change support after observing transfer.

### 5.6 Stability gates

The atlas passes only if all of the following hold:

- split-half rank correlation of each channel across `Xi` is reported with source-group
  bootstrap intervals;
- the median Jensen-Shannon distance between split-half `q_m` estimates is below a
  preregistered threshold;
- mechanism incidence is neither degenerate nor dominated by a single source group;
- nearest neighbors are stable under seed bootstrap;
- frozen-atlas assignment works for unseen `D_geometry` motions without refitting;
- `q_m` is not almost perfectly reconstructible from `a_m` alone.

The CPU stability contract in `gear_sonic/research/lace/stability.py` now enforces common
motion/policy support without imputation and reports per-motion/policy Jensen-Shannon drift,
policy-axis distance-matrix Spearman correlation, deterministic nearest-neighbor overlap, and
every excluded motion. A separate source-group-cross-validated ridge diagnostic predicts the
ordered `q` vector from the ordered scalar-difficulty vector and reports held-out reconstruction
`R^2` against a training-fold intercept baseline. The fold map, policy order, ridge penalty,
inputs, and output are hash-bound.

The scale-512 screening rule is now frozen in
`configs/research/lace/signature_stability_gate_scale512_v1.json`, before primary atlas outcomes
exist. It fixes the repeat-index split, `R=8` full / four-rollout halves, complete three-policy
common support, ten-neighbor geometry, 2,000 source-group bootstrap replicates, and every numerical
threshold. In brief, the screen requires at least 40 motions, half of `D_atlas`, and 20 source
groups on common support; median split-half JS at most 0.1 bit; distance-rank correlation at least
0.5; mean neighbor overlap at least 0.4; every nondegenerate policy-channel point correlation at
least 0.4 with a nonnegative 95% bootstrap lower bound; nondegenerate/source-distributed mechanism
incidence; and source-group-cross-validated difficulty reconstruction `R^2 <= 0.8`. A failure that
is attributable only to within-motion noise permits one precommitted extension to `R=16`; a second
failure, an activation/coverage failure, or near-complete reconstruction from scalar difficulty
stops RQ1 training. `gear_sonic/research/lace/stability.py` applies the self-hashed config and fails
closed on analysis-setting drift.

Treat released SONIC versus SONIC-Lite as a **cross-policy transport** audit, because architecture,
training history, mastery, and capacity all change together. Compare their signatures with
per-channel rank correlation, neighbor-overlap, and distance-matrix correlation; do not describe
this contrast as a causal capacity test. If transport is low, continue with the trainee-policy
atlas and narrow external validity. A causal scale test, if H1 passes, uses matched Lite-S/Lite-M
architectures and training schedules that differ only in preregistered width/depth.

The report-only settings are frozen in
`configs/research/lace/cross_policy_transport_scale512_v1.json` (digest `23a12e4b...`). The analysis
applies one parent Lite normalizer to both policies without refitting, compares released SONIC with
each early/mid/late stage separately, uses complete pair support without imputation, and reports
source-group-bootstrap channel Spearman intervals, per-motion JS divergence, JS distance-matrix
Spearman correlation, and ten-neighbor overlap. Its 40-motion/50%/20-group threshold is a coverage
gate only. Low transport narrows external validity; it does not kill trainee-policy H1. The CPU CLI
is `scripts/research/analyze_lace_transport.py`.

## 6. Representation baselines for RQ1

Every representation is fit on `D_atlas`, assigned without refitting on `D_geometry`, and
partitioned at the same `K`:

1. failure mechanism `q_m`;
2. scalar difficulty `a_m`, MPJPE, or native SONIC failure rate;
3. reference kinematics: velocities, accelerations, contact schedule, root dynamics, pose
   range, and duration computed without policy rollout;
4. motion semantics from frozen metadata/labels only;
5. combined kinematics plus scalar difficulty;
6. reference-feasibility score/filter (reported alone and with kinematics+difficulty);
7. random matched-size partitions.

The combined baseline is non-optional. H1 is convincing only if mechanisms add predictive
information beyond both reference geometry and difficulty together.

The release's `filter_and_copy_bones_data.py` is only a filename-keyword eligibility filter; it
does not calculate physical feasibility. Keep that pass/fail flag, but build the feasibility
control from reference-only quantities: file/schema integrity, quaternion norm and temporal
discontinuity checks, G1 joint hard/soft-limit proximity, joint-speed ratios against the frozen
G1 actuator configuration, root linear/angular speed and acceleration, and reference foot-contact
versus foot-kinematics consistency. Bind the URDF/config hashes and fit any robust scaler or
outlier threshold on `D_atlas` only. Treat this as a **reference-feasibility proxy**, not proof of
dynamic feasibility: torque/contact feasibility would require inverse dynamics or a privileged
reference generator. Pre-register hard corruption exclusions separately from the continuous
feasibility features so outcome data cannot decide which references disappear.

The promoted scale-512 reference-only artifact is materialized at
`/data/robotixx/groot-wbc-sonic-research/lace/manifests/bones_seed_official_scale512_d_atlas_reference_feasibility_v2.json`
(file digest `32798713...`, manifest digest `ea4dd795...`). It exactly reproduces the pinned
SONIC float32 resampling and motion-MJCF forward kinematics for all 107 `D_atlas` motions and finds
zero hard-corruption exclusions, zero hard-limit violations, 82 soft-limit excursions, and two
configured actuator-speed excursions. The latter quantities remain covariates, not exclusions. A
two-row, all-runtime-ready live recorder JSONL now verifies the ordered articulation joint names and
hard/soft position/velocity limit tensors; v2 binds the evidence file SHA-256, row count, and common
readback digest and is `scientific_use=true`. The provisional v1 artifact remains immutable for
audit history. Reference contact is only consistency with SONIC's kinematic labeling rule, not an
independent physical-contact or dynamic-feasibility measurement.

Use identical clustering method, scaler-fitting scope, `K`, minimum cell size, and source-group
constraints. Report `K in {4, 6, 8}` as robustness; use `K=6` for the preregistered primary
analysis.

`gear_sonic/research/lace/representations.py` now implements that shared CPU contract: population
standardization fit only on explicit `D_atlas` rows, deterministic multi-start Lloyd clustering,
canonical centroid/tie ordering, frozen no-refit assignment, exact feature/source/provenance
digests, and seeded exact-cell-size random controls. Minimum cell and source-group support remain
explicit preregistration inputs. The exact scale-512 values are frozen in
`configs/research/lace/representation_protocol_scale512_v1.json` (digest `8c42d6a2...`): `K=4`
requires at least 12 motions/eight source groups per cell, `K=6` requires 8/5, and `K=8` requires
5/4, with seed 8,132,026, 32 starts, 300 iterations, and tolerance `1e-10`. The primary null uses
100 exact-cell-size motion-level random controls (seeds 8,133,000--8,133,099); source IDs stay bound
for downstream blocking because whole-source-group permutation cannot generally preserve the same
cell sizes. Whole-group randomization is reported separately as a size-unmatched sensitivity if it
is used.

## 7. SONIC-Lite

### 7.1 Architecture contract

Create a G1-only experiment profile; do not modify `sonic_release.yaml` in place.

SONIC-Lite keeps:

- Unitree G1 embodiment and simulator;
- current policy/critic observation contracts;
- 10-frame proprioception and action histories;
- G1 reference interface and future-frame horizon;
- reward, termination, reset, DR, PPO, and evaluation definitions;
- the same motion library and 50-frame binning semantics.

It removes:

- teleop and SMPL encoders;
- cross-modal encoder sampling and alignment losses;
- unused kinematic reconstruction heads unless required for a controlled auxiliary-loss
  ablation.

The existing G1 encoder and dynamic decoder are themselves large MLPs, so merely deleting two
encoders will not produce a 1.2M model. Define new width profiles and measure trainable parameter
count from the instantiated actor. Target counts are bands, not names:

- Lite-S: `1.0M–1.5M` actor parameters;
- Lite-M: `4M–6M` actor parameters.

Record actor, critic, and total trainable counts separately. The paper-scale comparison uses
actor count. Do not describe a profile as 1.2M until the instantiated count is checked in a
unit test or dry-run artifact.

Implemented Lite-S configuration files:

```text
gear_sonic/config/actor_critic/universal_token/g1_lite_s.yaml
gear_sonic/config/exp/manager/universal_token/g1_only/lace_lite_s.yaml
gear_sonic/config/manager_env/observations/tokenizer/unitoken_g1_noz.yaml
```

The CPU static audit and CPU-instantiated forward pass agree on `1,227,514` actor,
`1,171,329` critic, and `2,398,843` total trainable parameters. The actor is the paper's Lite-S
scale. The inherited 100k-iteration maximum gives exact 10k/50k/100k checkpoint locations, but
the final experimental budget remains gated on the throughput and learning-curve pilot; any
override is hashed and the same fractional rule is recomputed before outcomes are opened.

Deferred until H1 passes:

```text
gear_sonic/config/actor_critic/universal_token/g1_lite_m.yaml
gear_sonic/config/exp/manager/universal_token/g1_only/lace_lite_m.yaml
```

### 7.2 Throughput benchmark

After freeing disk and GPU resources, benchmark `N_env in {128, 256, 512, 1024}` with the exact
training workload for the experiment and at least 20 timed PPO iterations after warm-up. The
scale-512 engineering pilot uses 233 `D_curriculum` motions; the headline scale-4,950 split uses
all 1,996. `N*` selected on the 233-motion pilot is not transferable to the headline workload and
must not be reused without the headline sweep.

Every cell uses plane terrain and keeps its entire `D_curriculum` partition resident in the same
lexicographic key order, independent of `N_env`. The launch contract derives
`override_num_motions_to_load` from the locked split (233 or 1,996) and sets
`sort_motion_keys=true`; every runtime iteration must attest the subset digest, resident count and
unique count, complete-universe count, `all_motions_loaded`, ordered-key digest, set digest, and
active terrain before its telemetry is accepted. The headline protocol is frozen in
`configs/research/lace/throughput_lite_s_headline_scale4950_v1.json`; its materialized partition
contains 1,996 robot/SMPL pairs with subset digest `ef911e0e...` and resident-order digest
`e26ce695...`. It reuses the byte-verified pre-environment Lite-S initialization so that only the
resident motion cohort and `N_env` change.

After the source freeze, the CPU-only planner produced the immutable launch plan
`/data/robotixx/groot-wbc-sonic-research/lace/throughput/plans/lite_s_headline_scale4950_dcurriculum_env_sweep_v2.plan.json`
(canonical plan digest `9980b241...`, file digest `4f6bf6e1...`). It binds 197 executable Python
files and 118 Hydra sources and composes all four cells. This records a launch-ready *plan*, not a
completed benchmark: the current foreign GPU process and 23,246 MiB free-memory observation fail
the frozen zero-foreign-process / 28,672 MiB preflight, so no cell has been launched.
The immutable `v1` plan is retained only as an audit trail and is superseded because the final
repository-wide Black pass changed source bytes before any cell was launched.

- policy/control transitions/s and simulator substeps/s separately;
- rollout time and PPO update time separately;
- peak allocated and reserved VRAM;
- CPU RAM and disk growth;
- reset/termination rate;
- simulator errors and numerical failures.

All cells load one byte-identical Lite-S pre-update checkpoint. Create it once on CPU, before any
environment can consume an `N_env`-dependent number of random draws, with
`scripts/research/run_lace_throughput.py --protocol <protocol.json> --materialize-init`.
The command atomically writes `last.pt`, `config.yaml`, and a self-hashed receipt containing the
seed, exact actor/critic construction contract, parameter counts, and per-tensor state hashes;
planning and launch both strictly reload and verify the bundle.

Select `N*` by sustained timed end-to-end control transitions/s among cells that pass every gate,
with the smaller `N_env` breaking exact ties, and freeze it for every arm at a model scale.
Iterations/hour remains a diagnostic: because one iteration contains
`N_env * num_steps_per_env` transitions, maximizing iterations/hour would mechanically favor
smaller environment counts. Decimation is fixed, so end-to-end physics substeps/s has the same
ranking; report it alongside collection-only simulator throughput.

The headline resident motion bytes are about 8.1 times the scale-512 pilot workload. Feasibility of
the 1,996-motion resident set and the winning `N_env` are therefore empirical outputs, not assumed
facts. Sweep in ascending `N_env`, retain a valid OOM receipt if a cell fails, mark larger cells
`not_run_after_prior_oom`, and never substitute a smaller resident set to make a cell fit.

In this trainer, each PPO iteration collects a fixed
`N_env * num_steps_per_env` control transitions. Isaac advances the simulator `decimation`
substeps per transition, so these are two distinct counters. Matching `N_env`, world size,
iterations, rollout length, decimation, epochs, minibatches, microbatching, and gradient
accumulation matches the *planned* schedule; it does not prove that non-finite gradients or
accelerator synchronization did not change the realized parameter-update count. Headline arms
therefore also bind the same resolved-training-config SHA-256 and record optimizer calls,
synchronized parameter updates, and skipped calls. Episode length still changes the number and
distribution of resets, so report it, but not the fixed-horizon transition count. Reset causes are
made disjoint with timeout precedence: `timeout = done & time_out`,
`termination = done & ~time_out`, hence `reset = timeout + termination` exactly.

## 8. Transfer-matrix experiment: RQ1

### 8.1 Causal intervention

Do **not** define RQ1 treatment rows from `q_m`. Doing so would use the candidate failure
representation to construct the very interventions whose transfer it is asked to predict, while
the semantic and kinematic baselines receive no equivalent advantage.

Before any policy rollout is inspected, freeze `S=8` representation-blind source panels from
`D_curriculum`. Sample by source recording group with a fixed seed, match panel motion count and
duration as closely as possible, and use only provenance plus duration strata—not failure,
semantic, or kinematic embeddings—to form panels. For source panel `s`:

```text
r_s(b)       = p_base(b) / sum_{b' in panel_s} p_base(b')  if b in panel_s, else 0
p_plus_s     = (1 - rho_s) p_base + rho_s r_s
delta_p_s    = p_plus_s - p_base
sum_b p_plus_s(b) = 1
```

Using the base distribution conditional on panel membership changes only panel exposure; a
duration-corrected uniform `r_s` would introduce a second hidden within-panel reweighting treatment.

Choose `rho_s` by a preregistered one-dimensional solve so every row has the same target
`KL(p_plus_s || p_base)` while respecting the same per-bin ratio cap. Panels must also meet a
realized added-exposure tolerance; otherwise redraw them before training. Report KL, total
variation, the signed `delta_p_s`, per-panel duration, and realized visits. This prevents support
size or dose—not representation—from explaining a row's apparent transfer.

For RQ1, `p_base` is a fixed sequence-length-agnostic distribution. Do not allow SONIC's
native sampler to move underneath the intervention; otherwise a transfer row mixes the
mechanism intervention with dynamic hard mining. Native adaptive sampling returns in RQ2.

The fixed sampler operates at the **motion** level. At scale 512, all 233 canonically ordered
`D_curriculum` motions are resident simultaneously; the headline run uses all 1,996. The runner
derives this count and exact order from the locked split and never hard-codes 233. The immutable
base/panel probability vector is the sole motion-selection source. Each run writes planned and
realized per-motion visits, signed exposure shift, KL/TV dose, resident-key digest, initialization
digest, control-transition occupancy, completion causes, and realized optimizer-update counters.
Any sampler reload or mutation, key-order drift, incomplete resident set, hidden temporal-bin
weighting, or mismatch between draws, occupancy, completions, and PPO accounting invalidates the
row.

The opt-in runtime bridge is implemented in
`gear_sonic/research/lace/fixed_distribution.py`. `TrackingCommand` installs a vector only after
the intervention-plan file hash and self-digest match, every motion is resident in the exact plan
order, resident IDs are the identity mapping, and native adaptive sampling is disabled. Both
MotionLib probability tensors are then frozen, and every subsequent motion draw revalidates them.
This makes the treatment executable without changing the release path. The CPU-only RQ1 runner
contract in `gear_sonic/research/lace/rq1_training.py` constructs and dry-composes one-arm plans,
binds the full resolved Hydra configuration, split/materialization/checkpoint/source assets, and
requires exact draw/occupancy/completion/optimizer evidence plus a launcher-owned successful
completion record before a receipt can validate. Its CLI deliberately has no GPU-launch action.
An external append-only attempt claim, continuous exclusive-GPU evidence, and one end-to-end Isaac
callback validation remain mandatory before the first RQ1 GPU run.

The primary scale-512 dose is frozen before transfer outcomes in
`configs/research/lace/rq1_intervention_protocol_scale512_v1.json`: a uniform motion-level base,
target `KL(p_plus || p_base)=0.05` nats, per-motion ratio cap 3, and maximum allowed range 0.002
in added panel exposure. The resulting eight-row artifact is stored at
`/data/robotixx/groot-wbc-sonic-research/lace/manifests/bones_seed_official_scale512_rq1_intervention_plan_v1.json`
(plan digest `e726a7f2...`). It covers all 233 motions; the realized added-exposure range is
0.001389 and maximum probability ratios are below 1.93. Its storage lock deep-rebuilds the
representation-blind panels and every probability vector from the split and protocol, so a
self-rehashed treatment edit is rejected.

Train one base policy and one policy per frozen source panel from an identical initialization for
each paired seed. Every trained policy is evaluated on the same motion-level `D_geometry` targets
with common rollout seeds. Only after all treatment rows are frozen may each candidate
representation summarize a treatment and compare it with each target. The treatment summary is
the **signed** exposure shift, not the upweighted panel centroid:

```text
delta_z_s       = sum_m delta_p_s(m) z_m
x_sj(k)         = delta_z_s(k) * (z_j(k) - E_p_base[z](k))
```

For the confirmatory matched-`K` comparison, `z` is a soft or one-hot membership vector and the
ordered `K` interaction components are the model features. Their sum is a signed exposure-alignment
statistic. This accounts for both the exposure added inside the panel and the diffuse exposure
removed elsewhere; a simple panel-to-target distance would silently discard half of the causal
treatment. All candidate representations use the same construction and the same `K`.

```text
G_s->j       = M_j(pi_plus_s) - M_j(pi_base)
G_tilde_s->j = G_s->j / max(H_j, h_min)
```

The confirmatory model uses paired raw `G`, target baseline performance/headroom covariates, and
target-group blocking. This controls the headroom confound without placing the same noisy base
estimate in both numerator and denominator. `G_tilde` remains a required secondary robustness
view, with `H_j` estimated from an independent evaluation rollout set and a preregistered `h_min`;
targets below `h_min` are marked unsupported rather than allowed to explode.

### 8.2 Predictive test

Primary test: out-of-sample predictive loss of transfer gains under crossed outer folds that hold
out entire source intervention panels and target recording groups together. Inner folds repeat
the same panel/target blocking for hyperparameters and representation dimensionality. A model
therefore has to generalize to both a new training intervention and new target motions.

Compare models using mechanism distance, scalar difficulty, reference kinematics, feasibility,
semantics, kinematics+difficulty, and repeated seeded random controls on the **same intervention
rows and targets**. Compare `baseline covariates + q` against the same baseline covariates alone;
do not compare a q-only model against a stronger adjusted baseline. The causal source unit is the
intervention panel/training run, while targets are grouped by recording source and seeds are paired.
Use whole-panel/target block permutations plus a hierarchical panel/seed/target-source bootstrap;
asymptotic clustered standard errors are secondary with only eight rows. Baseline/headroom
covariates come from evaluation rollouts independent of those used to form `G`.

An `S x K` source-panel/target-cluster summary is a visualization, not the sample size. The
confirmatory predictive test retains motion-level outcomes grouped by recording source. Matched
`K in {4,6,8}` clustering remains a robustness analysis for discrete representations; it never
defines the causal treatment rows.

H1 passes only if failure mechanism features improve held-out predictive performance over the
strongest non-failure baseline with a source-group bootstrap interval excluding zero. Diagonal
dominance is neither required nor the target claim.

The implemented analysis contract keeps one record per
`source_panel x target_motion x paired_training_seed`. Raw paired gain is primary. The normalized
view uses independent headroom rollouts with a disjoint seed set and excludes a whole target motion
when any paired headroom is below the frozen `h_min`; it never clips a denominator. Crossed outer
and inner folds embargo the held source panel and the held target recording group. Uncertainty uses
motion-within-group aggregation followed by a hierarchical source-panel/global-seed/target-group
bootstrap, plus a whole-source-row failure-feature permutation that reruns nested model selection
and every augmented out-of-sample refit. Feature names and baseline/failure/evaluation artifact
hashes are required inputs, so the strongest comparator must be frozen before outcomes are opened.

## 9. LACE sampler integration: RQ2

### 9.1 Minimal core change

Extend the existing sampler rather than replace it. Add a `lace` configuration block under
`adaptive_sampling` and keep all defaults behavior-preserving:

```yaml
adaptive_sampling:
  enable: true
  signal: failure_rate
  lace:
    enable: false
    atlas_path: null
    utility_path: null
    composition: native_log_residual
    epsilon: 0.10
    max_kl: 0.05
    max_ratio: 2.0
    min_effective_bins: null
```

When disabled, checkpoint keys and sampling probabilities must remain byte-identical to the
current release path.

When enabled:

```text
score_t(b) = sum_c q_motion(b)(c) u_t(c)
log p'_t(b) = log p_native_t(b) + lambda score_t(b)
p_t = TrustRegionNormalize(p'_t)
```

This makes the primary H2 comparison `native` versus `native + mechanism residual`; it isolates
the incremental value of the failure geometry. Also retain a `uniform + mechanism` diagnostic
arm to show whether LACE merely recreates hard mining.

### 9.2 Controller feedback

Every `J` PPO iterations, evaluate the current checkpoint on `D_controller` using fixed seeds.
Aggregate performance by the frozen soft mechanism membership:

```text
loss_t(c) = sum_m q_m(c) loss_t(m) / sum_m q_m(c)
u_t(c)    = posterior expected improvement opportunity
```

The evaluator exports only the `C`-dimensional state and uncertainty to the controller. It
must not expose `D_geometry` or `D_test`. Probe control transitions and simulator substeps are
accounted separately, and the policy is not updated on probe transitions.

The first allocator should be deterministic failure utility or UCB/Thompson over six
mechanisms. No LLM is involved in RQ2.

### 9.3 Gate A must operate at bin and mechanism level

Before accepting a distribution:

```text
KL(p_new_bin || p_old_bin) <= delta
max_b p_new_bin(b) / p_old_bin(b) <= r_max
effective_bins(p_new_bin) >= n_eff_min
exposure_new(c) >= exposure_min(c) for every c
```

Here `exposure(c) = sum_b p(b) q_b(c)`. Because mechanisms overlap, a simple per-arm minimum
on a six-way simplex is not sufficient.

### 9.4 Required H2 arms

Run them in stages, not all at headline seed count immediately:

1. uniform;
2. SONIC native capped failure-rate sampler;
3. EGM-style composite scalar-error temporal-bin sampler, labelled as a paper-spec
   reimplementation unless official code becomes available;
4. ZPD/learnability sampler already implemented in this branch;
5. kinematic-cluster allocator at matched `K`;
6. random matched-size clusters;
7. failure-mechanism fixed utility;
8. failure-mechanism UCB/Thompson;
9. native + best failure-mechanism residual.

Static quality/diversity curation is included only if its implementation and fixed-budget
sampling semantics can be reproduced fairly. Do not create a weak straw-man version merely to
populate the table.

H2 passes when the best preregistered LACE arm improves learning-curve AUC over native SONIC,
does not regress the endpoint beyond a predeclared equivalence margin, and does not regress the
hard-tail source groups. Report success, MPJPE, normalized progress, easy-decile retention,
mechanism exposure, entropy, and reset rate.

## 10. Structured actions and language: RQ3

### 10.1 Current code cannot yet support the proposed action space safely

The release config exposes useful knobs:

- ground friction and restitution;
- periodic pushes;
- rigid-body mass and center-of-mass randomization;
- reference augmentations such as freeze frames and upper-body concatenation;
- termination-threshold schedules.

But several event terms are startup-only, and `ManagerEnvWrapper.reinit_dr()` is currently a
no-op. Changing a Hydra value during training therefore does not prove the active simulator
distribution changed. RQ3 requires an explicit `apply_intervention()` adapter with read-back
telemetry and a reset policy for every mutable knob.

Only interventions with a tested causal path enter the action library. A proposed action that
cannot be read back from live environments is illegal.

### 10.2 Matched action library

Build a small discrete library, for example:

- mechanism sampling residual only;
- sampling residual plus narrower/wider friction range;
- sampling residual plus lateral/forward push mix;
- sampling residual plus bounded mass/CoM range change;
- sampling residual plus one validated reference augmentation;
- no-op.

All actions receive the same bounds, trust region, duration, and cooldown. Compare:

1. random legal structured action;
2. hand-coded causal lookup table;
3. contextual UCB/Thompson over the structured library;
4. LLM selection over the identical library and identical observation summary.

Only comparison 4 versus the strongest of 2/3 addresses language value. Free-form code or
arbitrary YAML generation is out of scope.

### 10.3 Shadow-update verification

Gate B must branch from identical model, optimizer, scheduler, RNG, and sampler state, then:

1. train `B_probe` steps under the candidate action;
2. evaluate with common seeds on `D_controller`;
3. compare against a no-op shadow branch, not the stale pre-update policy;
4. accept/reject using a predeclared loss and tolerance.

The no-op shadow is necessary because ordinary learning drift over `B_probe` steps can be
mistaken for intervention benefit.

On this machine, do not retain dozens of 469 MB branch checkpoints. Run branches sequentially,
write compact result records, and delete only explicitly designated temporary artifacts after
successful result validation. Storage cleanup must be approved separately.

The dedicated instrumented run labels every proposal with Gate-B ground truth. Main headline
runs deploy Gate A only. Report intervention-level false accepts/rejects, accepted regret,
worst accepted action, and confidence intervals clustered by checkpoint time because adjacent
interventions are autocorrelated.

## 11. Code and artifact layout

Proposed implementation map:

```text
gear_sonic/research/lace/
  schema.py                  # versioned manifests and atlas validation
  probes.py                  # per-step mechanism measurements
  signatures.py              # smoothing, q/a/f construction, assignment
  allocator.py               # static utility and contextual bandits
  trust_region.py            # Gate A over bins and soft mechanisms
  interventions.py           # whitelisted, read-back structured actions

gear_sonic/trl/callbacks/
  failure_atlas_callback.py  # rollout aggregation/export
  lace_controller_callback.py

scripts/research/
  build_lace_split.py
  collect_failure_atlas.py
  analyze_lace_stability.py
  run_lace_transfer.py
  analyze_lace_transfer.py
  run_lace_multiseed.py

configs/research/lace/
  atlas.yaml
  transfer_matrix.yaml
  curriculum.yaml
  structured_actions.yaml

tests/research/
  test_lace_schema.py
  test_lace_signatures.py
  test_lace_sampler.py
  test_lace_trust_region.py
  test_lace_split_leakage.py
  test_lace_interventions.py
  test_lace_compute_accounting.py
```

Research outputs use `outputs/research/lace/<artifact_version>/` and include a manifest with
git commit, resolved config, input hashes, checkpoint hashes, seed lists, environment versions,
and schema version. Large rollout traces and checkpoints belong on a filesystem with adequate
capacity, not in Git.

On the current workstation, the logical output path above must resolve into
`/data/robotixx/groot-wbc-sonic-research/lace`; do not write full rollouts or checkpoints to the
repository filesystem. `configs/research/lace/storage_5090.json` records the machine-local roots.

## 12. Compute accounting

For every arm report:

```text
C_train       main-training policy/control transitions
C_probe       controller and evaluation policy/control transitions
C_shadow      Gate-B candidate plus no-op policy/control transitions
P_train       main-training simulator substeps (`C_train * decimation`)
P_probe       controller and evaluation simulator substeps
P_shadow      Gate-B candidate plus no-op simulator substeps
U_call        observed `optimizer.step()` calls at the microbatch boundary
U_update      observed synchronized parameter updates
U_skip        observed non-finite-gradient skips
C_total       C_train + C_probe + C_shadow
P_total       P_train + P_probe + P_shadow
T_gpu         wall-clock GPU time
T_controller  controller/LLM wall time and token/API cost
```

Do not call LLM inference free. Do not count only successful or nonterminated steps. Reject a
headline matched-budget comparison unless both arms have the same resolved config digest, zero
skipped optimizer calls, and complete realized `U_call` and `U_update` counters matching plan.

Seed policy:

- interface development: one seed;
- atlas and mechanism screening: two paired seeds, third only after passing activation;
- RQ1 confirmation: three paired seeds;
- RQ2 headline: five paired seeds only for finalists that pass the one- and three-seed gates;
- RQ3: one instrumented reliability run, then paired confirmation only if language beats the
  matched structured bandit.

The 4–6M second scale is required after H1 passes and before a broad scaling claim. It need not
repeat every baseline: repeat the base row, best non-failure representation, and failure
representation transfer interventions, then confirm the winning RQ2 arm.

## 13. Execution schedule with gates

Calendar estimates begin only after operational readiness.

| Phase | Work | Exit criterion |
|---|---|---|
| P0 | Lock `/data` storage; coordinate GPU availability; validate corpus; freeze pilot source groups; verify environment | `/data` has at least 100 GB free, dataset/split manifests valid, sample eval reproducible |
| P1 | Build Lite-S config; parameter-count check; benchmark `N_env`; train one stable baseline | Stable curve and frozen early/mid/late checkpoints |
| P2 | Implement probes; collect pilot atlas; verify tensor semantics and termination causes | No missing/silent-zero channels; split-half stability passes |
| P2b | Compare Lite and released-SONIC signatures in the frozen Lite coordinate system | Cross-policy transport result reported; external-validity scope frozen |
| P3 | Fit representation baselines on `D_atlas`; assign `D_geometry`; freeze `K` and analysis | Source-disjoint representation protocol passes |
| P4 | One-seed transfer matrix, then three paired seeds | H1 predictive improvement or stop |
| P5 | RQ2 allocator screen and finalists | H2 conjunctive AUC/endpoint/tail gate or stop |
| P6 | Lite-M confirmation; establish a non-regressing release fine-tune; then native versus LACE from the released checkpoint | Mechanism result is not Lite-S-only and Layer-C control is valid |
| P7 | Implement/read-back structured actions; matched bandit; Gate-B labels; optional LLM | H3 versus matched structured controller |

The first GPU experiment after readiness is not a 1–3k-motion atlas. It is the environment-count
benchmark and a small deterministic probe validation on `sample_data`, followed by a
scientifically sized data pilot.

The 233-motion scale-512 throughput sweep is an implementation gate, not the final workload
selection. Before headline training, repeat the winning candidate cells with all 1,996 ordered
`D_curriculum` motions resident; freeze the headline `N*` only from that workload. A value selected
with 233 resident motions may be reported as a pilot diagnostic but cannot be silently reused for
the headline runs.

## 14. Immediate implementation order

The first pull request should be CPU-testable and contain no research claim:

1. versioned split and atlas schemas;
2. source-group leakage checks;
3. factorized `a_m/q_m/f_m` construction with synthetic fixtures;
4. trust-region projection and soft-mechanism exposure tests;
5. compute-accounting helpers;
6. a no-op LACE sampler configuration proving release behavior is unchanged.

Items 1–6, the six-channel simulator probe, the schema-v2 exact rollout scheduler, independent
single-evaluation termination trace, runtime-DR readback, reference-feasibility proxy,
representation controls, transfer-analysis contract, and Lite-S profile now have CPU-tested
implementations under `gear_sonic/research/lace` and `scripts/research/`. Every live path remains
opt-in and the released SONIC configuration is unchanged by default.

The bounded two-motion release-checkpoint instrumentation smoke completed on 2026-08-14. Its two
episode records are stored at
`/data/robotixx/groot-wbc-sonic-research/lace/rollouts/release-pilot-dr101-start-r0.jsonl`
(file SHA-256 `b9a9c89f...`). Its articulation readback was sufficient to promote the separately
rebuilt reference-feasibility artifact, but the smoke predates the current scientific
receipt/instrument/protocol chain. It is therefore deliberately uncollectable and may not be
relabeled as scientific evidence or used to build the primary atlas.

The remaining pre-experiment boundary is explicit and fail-closed:

1. on an exclusive GPU, run the implemented protocol-preflight mode, which captures the live
   measurement contract after manager creation and exits before reset, checkpoint-weight load,
   policy action, or episode observation;
2. bind it to a separately trusted, pre-outcome registry/timestamp anchor rather than relying on a
   mutable local self-hash;
3. preregister one canonical output path per schedule cell and prove exhaustive, one-attempt
   receipt coverage from storage that cannot be deleted and rerun after inspecting an outcome;
4. require independently bound parent artifacts and receipt-driven reconstruction for every
   stored scientific atlas and downstream policy-axis feature artifact; and
5. run the consolidated CPU suite after the shared source bundle is frozen.

The full assignment-only reference inventories are frozen for both the scale-512 pipeline and the
headline cohort. Headline counts are 1,996 `D_curriculum`, 822 `D_geometry`, and 557
`D_controller`; `D_test` remains deliberately unopened. Their 3-policy schedules are intentionally
not materialized until the Lite early/mid/late checkpoint hashes, parent `D_atlas` normalizer, and
analysis-protocol anchor exist. The environment sweep is also launch-ready in code but not yet
executed: its immutable headline plan is frozen as `...env_sweep_v2.plan.json`, while the RTX 5090
currently has a foreign process and does not meet the preregistered exclusivity/free-memory gate.
Do not launch an atlas, throughput cell, or transfer intervention before these boundaries pass.

Headline data preparation is now complete through paired materialization, the five-way split, and
the 1,006-motion reference-length inventory. The independently rebuilt reference-only feasibility
artifact is also ready (`f40b64c4...`): zero hard corruption exclusions, zero hard joint-limit
violations, 816 soft-limit excursions, and 28 configured actuator-speed excursions. The latter two
remain covariates/sensitivity flags and do not remove motions. This does not authorize headline
rollouts: the exact three Lite checkpoints, analysis protocol, execution registry, and one-attempt
external anchor are still absent.

## 15. Kill criteria and honest outcomes

- If `q_m` is unstable across seeds, the failure geometry is measurement noise. Stop before
  curricula.
- If `q_m` is nearly determined by `a_m`, LACE has not separated mechanism from difficulty.
- If reference kinematics plus scalar difficulty predicts transfer equally well, report that
  privileged failure probes add no useful geometry.
- If cross-policy transport is low, use trainee-policy signatures and narrow the claim; do
  not describe the 42M atlas as universal.
- If native SONIC or the existing ZPD sampler matches LACE, stop before RQ3.
- If a structured contextual bandit matches the LLM, remove language from the method claim.
- If structured DR edits cannot be applied and read back during a live run, restrict RQ3 to
  sampling and augmentation rather than pretending config changes affected physics.
- If source-disjoint sample size is too small for blocked inference, call the transfer matrix
  exploratory and collect more groups before a confirmatory claim.

## 16. Bottom line

This repository is a good substrate for LACE because SONIC already exposes the precise causal
lever the proposal needs: a checkpointed, bin-level motion sampling distribution recomputed once
per PPO iteration. The current Track-B branch also has paired-run, telemetry, schedule, and
statistical infrastructure.

The project should begin as a **failure-geometry measurement paper**, not as an LLM curriculum
project. The decisive first result is whether conditional mechanism distance predicts held-out
training transfer beyond kinematics plus difficulty. Everything else is downstream.
