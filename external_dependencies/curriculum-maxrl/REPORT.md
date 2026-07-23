# Curriculum-MaxRL: Experiment Report and Research Assessment

*Status: 2026-07-21. All numbers reproduce from this repo; provenance for each
table is the JSONL logs in `curriculum_maxrl/` and `curriculum_maxrl/maze_gpu/`.*

---

## 1. Research goal

Starting question: **can a teacher-curriculum improve reinforcement learning
with verifiable rewards, using MaxRL's likelihood objective (arXiv:2602.02710)
as the foundation — and can the combination surpass both plain MaxRL and
standard RL?**

Refined during the project into a sharper claim we set out to prove or refute:

> The right curriculum signal is not an external heuristic; it is computable
> from the estimator's own algebra. And the estimator's blind spot — prompts
> whose rollout groups all fail — defines exactly where the curriculum must be
> supplemented by a second mechanism (hindsight relabeling) that no sampling
> policy can replace.

## 2. The chain from math to algorithm to evidence

Every algorithmic component was derived from a proved identity, then validated
CPU-first (exact gradients), then GPU (1.26M-param transformer, 17×17 mazes).
The chain, component by component:

| Math result (PROOFS.md) | Algorithm component | CPU evidence | GPU evidence |
|---|---|---|---|
| **P1**: E[Σ\|w\|] = 2(pass@N − pass@1), exact | AdvMass teacher utility | ties hand-tuned ZPD with 0 hyperparams (V2: AUC 0.700 vs 0.688) | teacher beats uniform 6/6 paired seeds; dead groups 5.8→3.4/8 |
| **P2**: peak at p\* ≈ ln N/N, strictly concave | compute-indexed ZPD band; no band tuning | posterior tracks true p within ±0.03 | teacher p̂ matches eval per level; 60% of mass on true frontier |
| **P3**: greedy water-filling optimal for Σ mass | `allocate_rollouts_greedy` | +18% mass vs uniform split | (phase 2 — needs per-prompt n in rollout worker) |
| **P4**: RLOO mass = 2p(1−p) ≡ SFL learnability | unifies curriculum literature w/ estimator algebra | exact MC match | — (interpretive) |
| **P5**: MaxRL mass ≈ N× RLOO's as p→0 | why the teacher is safe with MaxRL specifically | — | GRPO+teacher collapses (H6 reversed); MaxRL+teacher grows pass@8 every seed |
| **P6**: hindsight update = ML gradient under shifted conditional; exact when laws match | dense hindsight relabeling | V1: per-group cosine = fresh-group cosine (0.956 vs 0.958); mean cosine 1.000 | dense hindsight = GPU champion (final 0.258, best 0.269) |
| **P7**: posterior lag & floor as staleness bound | decay 0.7, floor 0.1 | V2b: decay 0.7 closes ~19% of oracle gap; V3: floor flat 0–0.4 | defaults shipped in verl module |
| V6 (empirical): learning compounds ⇒ γ>1 | `power` knob, sample ∝ u^γ | γ=4: AUC 0.782 vs 0.728 | (not yet run on GPU) |

The two places the chain *broke* were as informative as where it held:

- **Adaptive truncation T** (decoupling T from N via the repo's subset
  estimator — mathematically valid, verified): slightly *hurts* (AUC 0.698 vs
  0.704). Diagnosis: at N=16–32 variance is not the binding constraint, so
  lowering T only forfeits hard-prompt upweighting. The math identified a
  degree of freedom; the experiments showed it is not the binding one.
- **Hindsight→teacher feedback**: V4 pre-registered "safe but redundant, risk
  = optimism inflation"; the GPU run reproduced that signature exactly
  (posterior p̂ 0.81 vs true 0.47 at level 2, worse final). Dropped. The
  posterior must see only requested-task evidence.

Both CPU pre-registrations that made GPU predictions were confirmed — the
toy → GPU pipeline is calibrated, which is the quiet methodological result of
the project.

## 3. Headline findings

### F1. The teacher works, and its mechanism is what the math says it is
At matched wall-clock, every teacher variant beats uniform sampling
(AUC 0.221–0.223 vs 0.211–0.216; 6/6 paired per-seed deltas positive). The
measured mechanism matches P1: dead-group rate drops 5.8→3.4 of 8, steps/sec
rises ~30% (frontier rollouts terminate early), and the sampling mass sits on
the levels where 2(pass@N − pass@1) is largest.

### F2. The curriculum requires the likelihood objective (H6 reversed)
The most instructive refutation: we predicted a frontier teacher would rescue
GRPO from pass@k collapse by retiring the mastered prompts its inverted w(p)
overweights. The opposite: frontier+GRPO collapsed *harder* (pass@8
0.332→0.269) than uniform+GRPO (0.351→0.312), losing easy-level retention,
while MaxRL under the identical teacher *grew* coverage in every seed
(0.316→0.348). GRPO's inversion was maintaining easy prompts; a curriculum
removes that maintenance. **Data-level curricula amplify objective-level
pathologies — a frontier curriculum needs likelihood-style weighting to be
safe.** This strengthens the MaxRL paper's central claim from the outside and
is, we believe, the finding most worth communicating.

### F3. Hindsight breaks the allocator's information ceiling
The oracle teacher (true pass rates — the best possible sampler) reaches CPU
AUC 0.851. Thompson+γ=4+hindsight reaches **0.890**. No contradiction: the
oracle bound constrains *sampling policies*; hindsight changes what a sample
yields, manufacturing verified signal from failures (V1: relabeled gradients
are exact on-structure, mean cosine 1.000). The categorical version: in a
frontier-heavy pool (max p = 10⁻⁵), uniform, DAPO dynamic sampling, and the
plain teacher all flatline at *exactly zero* while teacher+hindsight reaches
0.98 — there was nothing to sample toward; the signal had to be created.
Traced, hindsight invents the curriculum below the given pool, ignites
learnability within ~400 groups, then goes silent. **It is a cold-start
igniter, not a crutch.**

### F4. Proportional-to-mass sampling under-exploits (compounding)
Sampling ∝ u^γ with γ≈4 beats γ=1 (0.782 vs 0.728) and beats hard top-k.
P1 makes γ=1 optimal for signal *collected per draw*; but learning compounds —
progress on the highest-mass task unlocks its successors — so the optimal
temperature is sharper. The mass functional is the right *ordering*; the
right *concentration* over it depends on task-graph connectivity (γ→1 on
flat pools).

### F5. Where everything lands (GPU matched-clock leaderboard, seed 0)

| rank | config | final | best | AUC | pass@8 |
|---|---|---|---|---|---|
| 1 | frontier_alp + maxrl + **dense hindsight** | **0.258** | **0.269** | 0.236 | 0.361 |
| 2 | frontier_alp + maxrl | 0.244 | 0.257 | 0.233 | 0.361 |
| 3 | frontier + maxrl + hindsight | 0.230 | 0.256 | 0.234 | 0.356 |
| … | uniform + maxrl (reference) | 0.225 | 0.233 | 0.214 | — |
| … | uniform + grpo (reference) | 0.230 | 0.237 | 0.216 | 0.312↓ |

## 4. How far are we from the research goal?

**Achieved:**
1. A derived (not heuristic) curriculum — the advantage-mass teacher — with
   proofs, exact validation, and consistent GPU wins. ✅
2. The surpass-normal-RL goal: teacher+MaxRL configurations Pareto-dominate
   GRPO at matched compute on both average performance trend and coverage
   (which GRPO *loses* over training). ✅
3. The surpass-plain-MaxRL goal: +0.02–0.03 final / +0.02 AUC over
   uniform+MaxRL at matched wall-clock, consistent across seeds; and
   categorically (∞×) in frontier-heavy regimes via hindsight. ✅ with the
   caveat that the balanced-regime margin is modest (~5–10%).
4. A mechanistic account connecting every component to the estimator's
   algebra, with two honest negative results and two confirmed
   pre-registrations. ✅

**Not yet achieved:**
1. **The deep frontier remains uncrossed at fixed budget** — level 6+
   (distance ≥ 16) sits at ~0.01–0.02 after 2400 s in every config. Dense
   hindsight improved the approach (levels 0–3 all lifted) but one budget
   does not cash out 16-step credit assignment. This is now a
   compute-duration question, not a mechanism question — but that claim
   itself needs a long-run to verify.
2. **LLM-scale transfer** — the verl integration is production-ready and the
   data-scarce GSM8K regime is precisely where hindsight's fixed-prompt-set
   compounding should shine (the infinite-data maze understates it), but the
   single A10G cannot fit the 8-GPU recipe. Open until a larger node.
3. **Inference-efficiency currency** — the paper's samples-to-coverage
   speedup framing; the study is running now (3 matched checkpoints,
   `eval_efficiency.py`).
4. **P3 (optimal rollout allocation) has no GPU test** — needs per-prompt
   group sizes in the rollout path.

**Assessment: the conceptual goal is met; what remains is scale.** The
project's thesis is no longer "a curriculum might help MaxRL" but a specific,
proved, and doubly-validated statement: *the estimator's algebra defines the
curriculum (P1/P2), the objective determines whether a curriculum is safe at
all (F2), and beyond the allocator's ceiling only signal-creation — not
signal-allocation — makes progress (F3).*

## 5. The most promising next push (and why)

Ranked by expected-information-per-A10G-hour:

1. **Long-horizon run of the champion config** (dense hindsight, 4–6× the
   2400 s budget, one seed): directly tests the "level 6 is a duration
   question" claim — the only unverified load-bearing claim about the maze.
   If level 6 leaves zero, the frontier-march story is complete end-to-end;
   if not, the mechanism needs revision (e.g. hindsight-min-depth curriculum,
   or move budgets that scale with achieved depth).
2. **γ=4 concentration on GPU** (one matched run): the last CPU win without
   a GPU test; trivial to run (`power` knob already shipped).
3. **Efficiency table** (running): converts our wins into the paper's own
   currency; needed for any external comparison.
4. **Multi-seed for dense hindsight** (seeds 1–2): the champion's margin
   (0.258 vs 0.244) is a single-seed number; the prior multi-seed round
   showed ±0.01–0.015 seed noise on finals, so this is due diligence.
5. GSM8K 2×2 when hardware allows — the highest-value experiment overall,
   blocked on resources only.

## 6. Threats to validity (kept current)

- Maze GPU results are seed-0 except the confirmed multi-seed round; the
  champion's final-margin is within 2× seed noise (mitigation: #4 above).
- CPU effect sizes use exact gradients and exactly-relabelable structure —
  the most favorable case for hindsight (P6's conditional-law match holds
  exactly). Real verifiers relabel imperfectly; the maze's small gain vs the
  toy's large one brackets the expected range.
- Matched wall-clock on a shared machine: another process held the GPU for
  part of one window; all compared runs used exclusive windows (checked via
  dead/step + steps-per-second consistency).
- The teacher assumes a finite prompt pool with a stationary index; streaming
  pools need the parametric (ALP-GMM-style) variant sketched in GUIDE.md.
