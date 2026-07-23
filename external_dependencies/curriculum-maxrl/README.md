# Curriculum-MaxRL

**Teacher-guided curriculum learning driven by the MaxRL objective's own algebra.**

Research codebase exploring the integration of curriculum learning (teacher–student,
ZPD/learnability targeting) with Maximum Likelihood Reinforcement Learning
([MaxRL, arXiv:2602.02710](https://arxiv.org/abs/2602.02710)). Built against the
official MaxRL implementation (a [verl](https://github.com/verl-project/verl) fork).

## The idea in one paragraph

MaxRL reweights per-prompt gradients by ~1/pass-rate (a truncated Maclaurin expansion
of `log p`), which acts as an *implicit, gradient-level* curriculum — but it cannot
rescue prompts whose rollout groups come back all-fail (K=0 → group dropped, zero
gradient), and it wastes compute re-rolling mastered prompts. We add an *explicit,
data-level* teacher whose utility function is **derived from the estimator itself**:
the expected total |advantage| a prompt receives from a group of N rollouts is exactly

```
E[Σ|w|] = 2 · (pass@N(p) − pass@1(p)) = 2 · ((1−(1−p)^N) − p)
```

— twice the probability the prompt is *solvable within N attempts but not within one*.
This is a compute-indexed formalization of the zone of proximal development, peaking at
p* ≈ ln(N)/N. The teacher Thompson-samples a decayed Beta posterior over each prompt's
pass rate and samples prompts proportional to this utility; the optimal per-prompt
rollout allocation is greedy water-filling on the marginal `p(1−p)^N` (the probability
the next rollout is a group's first success).

## Repo map

| path | contents |
|---|---|
| `PAPER.md` | **The story** — 30-second pitch, why this direction, the three insights, what problem it resolves, real + hidden benefits |
| `GUIDE.md` | Design guide: approaches tried, verification status of each, and what's next |
| `REPORT.md` | Full experiment report: math→algorithm→evidence chain, findings, goal assessment |
| `SCHEDULE.md` | Live experiment tracking: executing queue, decision trees, next wave |
| `curriculum_maxrl/THEORY.md` | Exact advantage-mass formulas per estimator (MC-verified), derived teacher utility, optimal allocation, adaptive-T negative result |
| `curriculum_maxrl/DESIGN.md` | Original integration design, hypotheses H1–H5, CPU validation tables |
| `curriculum_maxrl/RESEARCH.md` | Deep-research synthesis of modern curriculum RL (PAIRED/PLR/ACCEL, ALP-GMM, SFL learnability, RLVR curricula) — 3-vote adversarially verified against primary sources |
| `curriculum_maxrl/*.py` | CPU prototype: skill-chain testbed, 5 estimators, 5 teachers, experiment runners |
| `curriculum_maxrl/maze_gpu/` | GPU testbed: 1.26M-param transformer on 17×17 mazes, goal-distance curriculum (13 levels), pass@k eval, matched wall-clock sweep protocol + logs |
| `verl_integration/` | Production integration for the MaxRL verl fork: `curriculum.py` (drop-in module), patches for `main_ppo.py` / `ray_trainer.py`, SmolLM+GSM8K launch script |

## Quick start (CPU, numpy only)

```bash
cd curriculum_maxrl
python3 run_experiment.py --steps 400 --seeds 5   # teacher × estimator sweep, ~1 min
python3 run_speed.py                              # learning-speed + adaptive-N comparison
python3 test_verl_curriculum.py                   # unit tests for the verl module
```

GPU maze testbed (needs torch + one ~24GB GPU):

```bash
cd curriculum_maxrl/maze_gpu
python3 train.py --teacher frontier --estimator maxrl --steps 300  # or --max-seconds 2400
python3 analyze.py matched_*.jsonl
```

## verl integration (into the MaxRL repo)

1. Copy `verl_integration/curriculum.py` to `verl/utils/curriculum.py`.
2. Apply `verl_integration/main_ppo.patch` and `ray_trainer.patch`
   (`git apply verl_integration/*.patch` from the MaxRL repo root).
3. Launch with:

```
+data.curriculum.enable=true
+data.curriculum.floor=0.1            # uniform replay floor (anti-forgetting)
+data.curriculum.decay=0.9            # posterior decay (tracks the moving policy)
+data.curriculum.utility=advmass      # derived utility; "frontier" = older heuristic
```

Teacher state is checkpointed/restored automatically; wandb gets
`curriculum/visited_frac`, `curriculum/frac_dead_p_lt_0.05`,
`curriculum/frac_mastered_p_gt_0.9`. See `verl_integration/smollm_curriculum.sh`
for a full GSM8K recipe.

## Headline validated results

On the CPU skill-chain testbed (36 tasks, initial pass rates 10^-level, 5 seeds):

- **Curriculum and MaxRL are complementary.** Teacher fixes the K=0 dead zone MaxRL
  can't reach; MaxRL extracts more per in-band group. `frontier+maxrl` is fastest to
  the deepest level (206 steps vs 248 uniform+maxrl vs 262 zpd+grpo) and best in the
  beyond-frontier-heavy regime (0.961 vs 0.871 / 0.847 for each alone).
- **MaxRL already does most of what a curriculum does on moderate distributions**
  (+0.01 from teacher) while GRPO needs the teacher badly (+0.23) — empirical support
  for the paper's "implicit curriculum" reading.
- **The derived advantage-mass utility matches the hand-tuned ZPD band with zero
  band hyperparameters.**

On the GPU maze testbed: uniform sampling wastes ~65% of rollout groups (dead K=0);
the frontier teacher cuts that to ~49% and runs ~2× more steps in the same wall-clock.
Matched-wall-clock sweep in progress; see `curriculum_maxrl/maze_gpu/EXPERIMENTS.md`.

## Citation / provenance

Builds on the MaxRL paper and codebase (Tajwar, Zeng et al., ICML 2026). The
curriculum design draws on PLR (Jiang et al.), PAIRED (Dennis et al.), ALP-GMM
(Portelas et al.), and SFL learnability (Rutherford et al., NeurIPS 2024) — see
`curriculum_maxrl/RESEARCH.md` for the verified literature synthesis.
