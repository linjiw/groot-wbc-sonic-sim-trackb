# Feasibility hygiene for SONIC motion tracking

**What this plans.** Three CPU-cheap modules in front of SONIC's WBC motion-tracking training —
a dynamic-feasibility screen, a contact-projection repair operator, and an exposure ledger over the
adaptive sampler — plus a matched-compute ablation designed so that each contrast identifies one
factor.

**Status: planned, partially built, nothing measured.** No training run has been launched for this.
Every number below is either (a) read off shipped code, (b) measured on CPU with the real sampler
code path, or (c) imported from a sibling project on a *different* motion bank and explicitly
marked as not transferable. Predictions are registered as P10-P12 in
`prediction_register.md`, before any arm runs.

## The directive, and the three things wrong with it

The directive was: screen every clip with a feasibility tool; repair the flagged ones by root
projection and prune what will not repair; and **replace SONIC's failure-adaptive sampler with a
normalised mixture carrying a 10% uniform floor**. Then run four arms — raw+adaptive,
pruned+grounded, repaired+grounded, repaired+uniform — and read stratified survival.

Three parts of that do not survive contact with this codebase.

### 1. The uniform floor is already shipped, at exactly 10%

`motion_lib_base.py:3235-3238` (per-reset distribution) and `:3399-3402` (clip-residency
distribution) both compute, verbatim:

```python
uniform_sampling_prob = torch.ones_like(failure_based_sampling_prob) / len(failure_based_sampling_prob)
p = failure_based_sampling_prob * (1 - self.uniform_sampling_rate) + uniform_sampling_prob * self.uniform_sampling_rate
```

`uniform_sampling_rate` is read at `:2457` and pinned at
`gear_sonic/config/manager_env/commands/terms/motion.yaml:26` to **0.1**. The adaptive component is
normalised before the blend (`:3225-3231`), with an exact-uniform fallback when the signal sums to
zero. Implementing "normalise then mix with a 10% floor" here is a no-op.

It is also approximately honest in practice. The blend happens *before* `p *= adp_samp_bin_weights`
(`:3239`) and the renormalisation (`:3240-3242`), so `a` is not guaranteed. Driving the real code
path over the real BONES-SEED length inventory (1,006 motions, 7,863 bins on the 50 Hz timeline),
the realized uniform mass lands at **0.098–0.163 against a nominal 0.10**. The upper end is reached
when the failing clip is the longest one: a long clip splits into many bins, each carrying a small
weight, so concentrating the adaptive mass there shrinks the weighted denominator and the uniform
term's relative share grows. A real but second-order distortion, not a broken floor.

### 2. A floor bounds probability from below; the problem is the top

That is the actual gap, and it is not what the directive proposed to fix. On that same real bank,
driven through `sync_and_compute_adaptive_sampling`, **one clip that fails every episode still
takes 3.4–4.5% of all sampling mass — 34–45× its fair share — with the 10% floor active.** The
floor guarantees coverage. It does nothing about concentration.

SONIC has two mechanisms that could bound the top, and neither is doing so:

| mechanism | where | shipped state | effect measured |
|---|---|---|---|
| heavy-tail clip `clip(failure_rate, 0, 200·mean)` | `:3216-3221`, `sonic_release.yaml:70-71` | active | **inert.** Failure rates are ≤ 1, so a bound of 200·mean only binds when the mean failure rate is below 0.005 — an almost-solved bank |
| `max_prob_per_motion` / `max_prob_per_bin` | `:2461-2462`, block at `:3285-3292` | **`None` in every shipped yaml** → whole block early-returns | `"auto"` resolves to 200/N, also inert. Explicit **5× fair share → 5.2× realized**; **2× fair → 2.1×** |

So the cheap sampler-side remedy is not new code. It is setting a config value that already exists
and has never been set. That becomes arm E, and it is the honest competitor that data hygiene has
to beat.

### 3. The prevalence numbers come from a different bank

`77.2% feasible / 22.8% infeasible / 3.2% rare ground-contact poses` are measurements on a
10,705-clip AMASS-derived bank retargeted by `whole_body_tracking`, screened in the sibling
project. SONIC trains on BONES-SEED (`data/motion_lib_bones_seed/robot_filtered`), a different
corpus through a different retargeter. Per-source infeasibility in that sibling measurement ranged
from **0.1% to 100%**, so a bank-level rate is not a property that travels. **The prevalence on
SONIC's bank is unmeasured and must be measured before any arm is sized.** That is the first thing
the screen produces.

## What already exists here, and must not be rebuilt

`gear_sonic/research/lace/` is a 35k-line research stack, and much of it overlaps the directive:

- `reference_feasibility.py` + `reference_feasibility_manifest.py` already screen every selected
  clip offline, CPU-only, without importing Isaac Lab: array-schema validation, the exact float32
  30→50 Hz resample, MJCF forward kinematics for both feet, SONIC's `foot_detect` contact rule,
  joint hard/soft/velocity-limit flags, and a 26-dim outcome-independent feature vector, all under
  a sha256 cascade. A real manifest exists for 1,006 motions.
- `atlas_probe_mode.py` pins `(motion_key, motion_id, start_step)` per environment from a frozen
  schedule. **That is a stratified-start evaluator**, and the plan uses it rather than adding one.
- `probes.py:680-725` is a complete, frozen-schema actuator-saturation probe
  (`clipped_joint_fraction`, `max_requested_torque_ratio`, `max_clip_gap_ratio`). It has **never
  been run** — no artifact exists.

The screen this plan adds is not a replacement for any of that. `reference_feasibility.py` says of
itself, in its own docstring, that it is *"a kinematic proxy, not a dynamic feasibility proof"*. It
asks whether the reference is a well-formed, reachable G1 configuration. The new screen asks a
different question — **could any controller supply the forces this reference demands, given the
contacts it offers?** — and answers it with inverse dynamics plus a friction-cone, torque-limited
contact solve. A clip can be perfectly reachable and still require a metre of unsupported descent.
The two screens are complementary and the plan keeps both.

## The pipeline

```
BONES-SEED motion bank (.pkl)
      │
      ├─ 1. dynamic feasibility screen        ~1 CPU-s/clip     gear_sonic/research/hygiene/screen.py
      │     per clip: airborne_frac, infeasible_frac, unsupported_force_N_{p50,p95,max},
      │               unsupported_impulse_per_weight_s, max_tau_ratio_p95, torque_infeasible_frac
      │
      ├─ 2. contact-projection repair         ~3-9 CPU-s/clip   gear_sonic/research/hygiene/repair.py
      │     root-z projection + smoothing, re-screened against a success budget
      │     (offset_max ≤ 0.15 m, infeasible_frac_after ≤ 0.05); refusals are named, not silent
      │
      └─ 3. exposure ledger over the sampler  free              .../hygiene/sampler_diagnostics.py
            normalized Shannon entropy, top-1 clip share, mass spent on flagged clips
```

Module 3 is deliberately *diagnostic*, not an intervention. The intervention SONIC is missing is
the per-motion cap of §2, which is config-only.

Two format facts make this cheaper here than in the sibling project. First, a pure root projection
in the SONIC format touches **only `root_trans_offset[:, 2]`** — `dof` and `pose_aa` are untouched,
so the `pose_aa[:, i+1] == DOF_AXIS[i] · dof[:, i]` redundancy that the LACE manifest asserts to
1e-6 is preserved by construction. Second, a repaired bank has to be a **new bank root of
real files**, not an overlay on the old one — and the reason is the opposite of the obvious one.
LACE materialises subsets *as absolute symlinks* (`throughput.py:336-356`) and verifies that each
entry resolves to its recorded source (`:401-407`), so a symlink view is exactly the right shape
for *selecting* from a bank, and a directory of copies would fail that check. But repaired content
does not exist anywhere to point at. The repaired clips become their own sources, which means the
split, length and feasibility artifacts have to be rebuilt against the new root so the sha256
cascade stays self-consistent.

## The ablation

The directive's four arms move two factors at once between A and B (bank *and* sampler), which
identifies neither. Since the floor is already shipped, the sampler does not need to change to get
a clean bank axis — so the matrix splits into two single-factor axes:

| arm | bank | sampler | isolates |
|---|---|---|---|
| **A** raw | as retargeted | SONIC default (`a=0.1`) | baseline |
| **B** pruned | flagged clips removed (N shrinks) | SONIC default | pruning, confounded by data volume |
| **C** repaired | flagged clips replaced (**N preserved**) | SONIC default | hygiene, *un*confounded by volume |
| **D** raw | as retargeted | `uniform_sampling_rate = 1.0` | the value of adaptivity itself |
| **E** raw | as retargeted | `max_prob_per_motion = 5×` fair share | whether capping substitutes for hygiene |

Arm D sets the rate to 1.0 rather than `enable=false`. Disabling the adaptive path also changes the
**start-time distribution** — the adaptive path draws a bin, then a frame, then shifts backward by
up to `pre_failure_sample_window=200` frames (`:3547-3549`), while the plain path draws
continuous-uniform phase over the clip (`sample_time`, `:2249-2258`). `enable=false` would change
three things at once.

Note what "uniform" then means, because it is not the directive's "plain uniform": with the rate at
1.0 the draw is uniform over *bins* weighted by `adp_samp_bin_weights`, and since those weights sum
to a constant per motion, that is uniform **over motions** — not uniform over time within a clip.
The backward shift of up to 200 frames also survives, so starts stay biased early by the clamp at
zero. Both effects are identical in every arm, which is the point; but arm D answers "is
failure-weighting better than equal weighting", not "is this the true uniform reference".

Arm C is the load-bearing one. It keeps N, the clip names, the durations and the fps identical to
A; only the *contents* of the flagged files differ. **Repaired-vs-raw is therefore a strictly
cleaner hygiene test than pruned-vs-raw**, and it is the reason C outranks B in priority.

### The confound, and the test that separates it

B trains on fewer clips, so at matched compute more gradient steps land per clip. "It converged
faster because there was less data" must be separable from "hygiene helped". The separator is a
concentration signature, not an average:

```
order feasible eval clips ascending by the BASELINE arm's offset-mean survival
worst = order[:max(1, n//10)]      best_half = order[n//2:]      easy = {c : baseline(c) >= 0.95}
claim passes iff   Δ_worst >= 2 · Δ_best_half   AND   |Δ_easy| <= 0.02
```

A data-volume effect lifts everything roughly uniformly: it fails the ratio test and moves the easy
stratum. The pre-listed null is that **a pass on the average with a fail on the signature is
reported as a volume effect, with no rescue analysis**.

The eval set stays **raw** in every arm. Evaluating a repaired policy on repaired references would
be a self-serving endpoint.

## Metrics

**Stratified survival** at fixed start offsets, via `atlas_probe_mode`. Strata: easy, worst decile,
and ground-contact/kneel/crawl. Prediction: easy is flat across all arms (Δ≈0); worst decile
C ≥ B ≫ A; ground-contact C > B, because repair recovers rare poses that pruning discards.

**Exposure ledger.** Normalized Shannon entropy `-Σp log p / log N`, top-1 clip share, and mass
spent on flagged clips. One trap: SONIC already logs `effective_num_bins = 1/Σp²`
(`manager_env_wrapper.py:1023`), which is **Rényi-2, not Shannon**. The two are not interchangeable
and must never be quoted side by side as if they were.

**Torque health.** Actuator saturation and contact-force discontinuity through tracking
transitions. Three incompatible saturation definitions are already in circulation — SONIC's
`requested_ratio ≥ 0.95 or clip_gap_ratio > 0.01` (`probes.py:47-51`), and two different ones in
the sibling project. **This plan uses SONIC's**, and any cross-project number must be recomputed,
not quoted. Contact-force magnitude is computed and then discarded into a bool at
`isaac_recorder.py:486-487`; recovering it is new instrumentation with an unverified solver-field
dependency, and is the least certain part of this plan.

## Order of work

1. **Screen SONIC's own bank.** Until the prevalence is measured here, arm sizes are guesses. — *built, running on a slice*
2. **Repair the flagged clips**, produce the pruned and repaired banks, census the refusals. — *built*
3. **Exposure ledger + the cap measurement.** Config-only; costs no GPU. — *built*
4. **Freeze the analysis and dry-run it on synthetic outcomes** before any arm exists. — *built*
5. Register predictions before any arm runs. — *done: P10 (prevalence), P11 (the cap may win),
   P12 (repair vs pruning) in `prediction_register.md`*
6. Only then, arms. Priority C > E > B > D, against A.

Steps 1–4 are CPU-only by design, so the expensive step is last and best-informed.

## Open and unverified

- Prevalence on BONES-SEED: **unmeasured**. Everything downstream is sized off a guess until step 1
  finishes.
- Whether repair helps at all on this bank: unmeasured. The sibling project's operator recovered
  65.8% of flagged clips, but on a different retargeter's failure modes.
- The repair operator triggers on *airborne* frames but scores on the full LP. Clips infeasible
  while in contact (friction-cone or torque reasons) get a zero offset and score as failures rather
  than out-of-scope. That is a mislabel, not a repair failure, and it is reported separately.
- `max_prob_per_motion` is an absolute probability, so "5× fair share" depends on bank size and
  cannot be written as a relative value in yaml today.
- Contact-force discontinuity has no working instrumentation in either project.
- Every arm is n=1 seed. Inference is per-clip paired permutation only; there is no seed-level
  power, and the write-up must not imply otherwise.

---

### Appendix: the directive as received

Recorded verbatim so the plan above can be checked against it.

> 将这套工具链（refeas + repair_contact_projection + stratified-start eval）接入 SONIC-style WBC
> Motion Tracking 训练框架，并用严格的因果对比证明其有效性。
>
> **一、接入管线：三步改造。** Step 1 离线清洗：用 refeas 跑一遍所有 Mocap clips，打上
> `airborne_ratio` 和 `unsupported_wrench` 标签。Step 2 几何投影修复：对被标记的 clip 进行根节点
> 投影修复；超出容差（如大幅度腾空跳跃）的动作直接 Prune 掉。Step 3 防塌缩调度器：将 SONIC 默认的
> Failure-adaptive Sampler 替换为带有 10% 均匀覆盖下限（Uniform floor on simplex）的规范化混合采样器。
>
> **二、验证矩阵：4 组消融（Matched Compute）**，固定总训练步数（如 4,000 iterations）和超参数：
> A. Baseline（Raw Corpus 含 22.8% 脏数据 + SONIC 默认 Failure-adaptive）；
> B. Cleaned-Prune（仅保留 Feasible Clips + 规范化 Grounded 采样）；
> C. Cleaned-Repair（Feasible + 修复后的 Clips 全库 + 规范化 Grounded 采样）；
> D. Oracle Control（Cleaned-Repair 动作库 + Plain Uniform）。
>
> **三、核心指标。** (1) 训练效率账本：达到相同平均跟踪奖励所需的 GPU-hours / 环境交互步数；
> Baseline 组 Top-1 难度片段会霸占 40%~80% 的采样权重，Cleaned-Repair 组采样分布熵显著提升。
> (2) 分层跟踪存活率：Easy Stratum 四组持平（Δ≈0）；Worst-Decile Stratum
> Cleaned-Repair ≥ Cleaned-Prune ≫ Baseline；Ground-Contact / Kneel / Crawl
> Cleaned-Repair 显著超越 Cleaned-Prune，挽回原本只占 3.2% 的稀有姿态覆盖度。
> (3) WBC 执行器与力矩健康度：跟踪过渡段的电机饱和率（Actuator Saturation %）和接触力突变（ΔGRF）；
> 消除悬空帧后 WBC 不再出现"空中打满力矩却无法落足、落地瞬间剧烈冲击"的病态震荡。

Field names in the directive do not exist under those spellings: `airborne_ratio` → `airborne_frac`;
`unsupported_wrench` is not a single scalar but `unsupported_force_N` / `unsupported_torque_Nm` per
frame, with `infeasible_frac` as the binary. The `22.8%` and `3.2%` figures are from the sibling
project's AMASS bank, not this one — see §3.
