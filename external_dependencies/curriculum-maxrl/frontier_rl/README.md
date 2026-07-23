# frontier_rl — the curriculum-MaxRL schedule as a reusable framework

The validated training algorithm (see `../REPORT.md`), packaged so it can be
applied to gym environments, robotics simulators, or LLM prompt sets without
touching the core. numpy-only; no torch/gym dependency in the core.

## The algorithm (one screen)

```
teacher:   Beta(α,β) posterior per task (decay 0.7) → Thompson sample p̃
           → utility u(p̃) = (1−(1−p̃)^N) − p̃    [= MaxRL's expected advantage
             mass, proved exact; peak at p* ≈ ln N/N]
           → sample tasks ∝ u^γ  (γ≈4 if tasks share skills, 1 if independent)
           → mixed with a 10% uniform floor

estimator: MaxRL success-conditioned advantages  w_i = r_i/K − 1/N
           (K=0 groups dropped — this is what makes the teacher necessary)

hindsight: dead groups are relabeled by the ENV to the sub-goal actually
           achieved and trained as successes of that easier task
           (exact ML gradient where the env's relabel is exact — proved +
           measured; breaks the information ceiling of any pure sampler)
```

## Plugging in your environment

Implement `TaskSpace` (three methods) and `Policy` (one method) from
`interfaces.py`:

```python
class MyEnv:                                  # TaskSpace
    n_tasks: int                              # discrete task/goal bins
    def rollout_group(task_id, n) -> GroupResult   # N episodes, binary rewards
    def relabel(group) -> (task', rewards') | (task', rewards', trajs') | None

class MyPolicy:                               # Policy
    def update(task_id, trajectories, weights)     # one weighted PG step

trainer = FrontierTrainer(MyEnv(), MyPolicy(),
                          TrainerConfig(n_rollouts=16, hindsight=True))
trainer.train(steps=500)
```

**The two hindsight contracts** (from Proposition 6; violating either turns
the exact relabeled gradient into noise):

1. **Exactness** — a relabeled success must be a *true* success of the
   relabeled task under the env's own verifier.
2. **Conditioning** — if trajectories embed the goal (goal-relative features,
   `desired_goal` obs, goal tokens in a prompt), return rewritten
   trajectories with the achieved goal substituted. We measured the cost of
   skipping this on the gridworld: hindsight *hurts* without the rewrite
   (AUC 0.600 < teacher-only 0.658) and gives the best result with it
   (0.703). This is HER's observation-rewrite, surfaced as an interface
   contract.

## Adapters included

| adapter | what it shows | result (AUC, uniform → teacher → +hindsight) |
|---|---|---|
| `skill_chain` | regression anchor vs the validated testbed | 0.650 → 0.728 → **0.890** (matches REPORT.md) |
| `grid_reach` | goal-conditioned robotics pattern (goal bins = distance rings, relabel = reached cell, REINFORCE tabular policy) | 0.592 → 0.658 → **0.703** |
| `gym_classic` | **real gymnasium envs**: MountainCar positional curriculum (hard exploration) + CartPole survival curriculum | MC: 0.216 → 0.228 → **0.246**; CP: 0.190 → 0.225 → **0.246** (3 seeds) |
| `gym_goal` | gymnasium GoalEnv skeleton: where reset/step/is_success go, how to bin continuous goals, HER-style relabel via `achieved_goal` | (skeleton — bring your env) |

The gymnasium results reproduce the validated ordering
(uniform < teacher < teacher+hindsight) on real environments with a
deliberately weak tile-coded REINFORCE policy — MountainCar's sparse flag
success is the real-world twin of our frontier-heavy regime, and its
positional curriculum ("reach x ≥ x*, walking x* from valley to flag") is
exactly the pattern to copy for robotics reach tasks. Budgets in the demo
are small (~10 min CPU); scale `steps` for stronger separations.

### MountainCar case study: the flag, solved — and a transfer lesson

Scaled runs (600 steps) with **per-bin policy parameters** never reach the
flag (hardest bin stays 0.000 for every method): each bin's tile table
learns from scratch, so the curriculum has nothing to *transfer*. Giving
all bins one **shared** policy (the task enters only the success predicate)
changes everything — 150 steps, 3 seeds:

| shared-policy config | mean pass | FLAG bin |
|---|---|---|
| flag-only (no curriculum) | 0.028 | **0.000** |
| uniform over bins | 0.975 | 0.889 |
| teacher (γ=4) | 0.994 | 0.944 |
| **teacher (γ=4) + hindsight** | **1.000** | **1.000** |

Training on the flag alone — the standard sparse-reward setup — scores
exactly zero: MountainCar's classic exploration wall. *Any* mixture over
easier targets breaks the wall (energy-pumping transfers), the teacher
sharpens it, and the full stack solves the flag bin perfectly in every
seed. Two morals for practitioners:

1. **Curricula operate through shared parameters.** Difficulty bins must
   share the policy (condition on the goal, don't partition by it) or
   there is no channel for competence to flow through — the same
   generalization-cliff lesson as our maze-size finding, now in gym form.
2. With sharing in place, MountainCar reproduces the categorical result:
   flag-only 0.000 → full stack 1.000 at equal compute.

Run them:

```bash
python3 frontier_rl/test_framework.py                 # unit tests
python3 frontier_rl/examples/run_skill_chain.py       # regression anchor (~2 min)
python3 frontier_rl/examples/run_grid_reach.py        # robotics-style demo (~3 min)
python3 frontier_rl/examples/run_gym_benchmark.py     # gymnasium benchmark (~10 min, pip install gymnasium)
```

## Mapping to robotics / gym in practice

- **Task bins**: pick the axis your curriculum should walk (goal distance,
  obstacle count, object mass...). ~8–30 bins is plenty; the posterior needs
  a few groups per bin to localize the frontier.
- **Binary success**: use the env's own success predicate. Shaped rewards
  can coexist in your policy update; the *teacher* should only see binary
  outcomes (that is what the advantage-mass math assumes).
- **relabel**: gymnasium GoalEnvs give you `achieved_goal` for free — map it
  to its bin and rewrite `desired_goal` in the stored observations
  (contract 2). For non-goal envs with no meaningful relabel, return `None`;
  you keep the teacher benefits and lose only the hindsight term.
- **Group size N**: the teacher's band targets p ≈ ln N/N. N=16 targets
  ~17% success tasks; raise N to push the curriculum toward harder bins.
- **On/off-policy**: the schedule is estimator-agnostic at the interface
  level, but its guarantees are for the MaxRL weights; if you swap in a PPO
  update keep the weights as advantages and stay near-on-policy.

## Design: one schedule, five execution shapes

The algorithm is deliberately factored so each piece can be swapped to match
the training regime without touching the others — the flexibility is the
design, not an accident:

| training regime | teacher variant | evidence stream | hindsight | validated on |
|---|---|---|---|---|
| episodic groups, fixed task pool (RLVR/LLM prompts) | `FrontierTeacher` (Beta rows, Thompson) | group (task, K of N) | dense relabel via `TaskSpace.relabel` | skill chain, maze GPU, verl integration |
| episodic groups, procedural tasks | `StreamingFrontierTeacher` (kernel posterior) | (difficulty, K of N) | same | continuous-goal reach |
| goal-conditioned control (gym/robotics) | `FrontierTeacher` over goal bins | group | relabel + conditioning rewrite | gridworld, MountainCar, CartPole |
| massively parallel sim (IsaacLab, 4096 envs) | `FrontierBinTeacher` (vectorized, evidence-scaled decay, deterministic optimism) | per-reset Bernoulli stream | statistics-half only (occupancy credit) | adapter + unit tests; SONIC design doc |
| dense-reward PPO | any of the above with `utility="learnability"` | termination flag as verifier | usually skip (dense reward is the partial credit) | SONIC_RESPONSE.md analysis |

The swap points and what fixes each choice:

- **utility** — `advmass` when a real group size N exists (the band is then
  *derived*, peak ≈ ln N/N); `learnability` when evidence is a reset/hazard
  stream with no N (SONIC Q2).
- **posterior** — Beta rows for fixed pools; kernel over a difficulty axis
  for procedural sources; vectorized arrays with half-life-in-episode-
  equivalents decay when throughput varies by orders of magnitude (Q4).
- **optimism** — Thompson when stochasticity is fine; `mean + k·std` under
  determinism guardrails (Q3). Both validated equivalent when a floor exists.
- **γ** — 4 on tight chains (compounding), 1 everywhere else (measured,
  including the negative transfer on broad pools).
- **hindsight** — full trajectory relabel where the env verifies exactly and
  conditioning can be rewritten; statistics-only credit where it can't
  (on-policy PPO); off where dense reward already carries partial credit.

## IsaacLab / massively-parallel sim adapter

`adapters/isaaclab_curriculum.py` provides `FrontierBinTeacher`, mapping the
teacher onto IsaacLab's ManagerBasedRLEnv pattern (verified against a
production humanoid-tracking fork): task bins live in the *command manager*,
success lives in the *termination manager*, and the teacher consumes the
**reset stream** — every episode reset is one Bernoulli observation
(bin, terminated-early?). No groups needed; no isaaclab import required (the
curriculum-term wrapper imports it lazily), so the module unit-tests on CPU.

```python
teacher = FrontierBinTeacher(n_bins=n_motion_bins, utility="learnability",
                             decay_half_life=2048)   # episode-equivalents
# termination hook (each step or on resets):
teacher.observe_resets(bin_of_env[reset_ids], terminated_early[reset_ids])
# command hook (assigning tasks to reset envs):
new_bins = teacher.sample_bins(len(reset_ids))
```

Key adaptations for the parallel-sim regime, each traced to the SONIC
analysis: evidence-scaled decay (half-life invariant to env count — exact:
10 successes age to 5.0 after one half-life of events), deterministic
optimism bonus, learnability default, and a `max_prob` tripwire instead of a
shaping cap. See `SONIC_RESPONSE.md` for the full design rationale including
the closed-loop threshold-curriculum stability rules.

## Streaming / procedural task sources

`streaming.py` provides `StreamingFrontierTeacher` for sources with **no
fixed task pool** (every task fresh: generated mazes, sampled goals,
synthetic problems with a difficulty parameter d ∈ [0,1]). It replaces the
per-task Beta rows with a kernel (Nadaraya-Watson) pass-rate posterior over
the difficulty axis + Thompson sampling on a difficulty grid, with optional
isotonic projection when d orders pass rates. Validated on a continuous-goal
reach task (5 seeds, matched budgets): streaming **matches the discrete-bin
teacher exactly** (AUC 0.684 vs 0.684; final 0.922 vs 0.919; uniform 0.648)
— you lose nothing by dropping the pool assumption. Use it when your
task generator has a difficulty dial; use bins when you have a fixed
prompt set.

## What this does NOT do
- Replace your RL optimizer: `Policy.update` is yours; this package decides
  *what to train on and with what advantage weights*, not how to descend.
- GRPO-style std-normalized advantages under a curriculum — measured to
  amplify coverage collapse (REPORT.md F2). Use the MaxRL weights.
