# curriculum_maxrl — teacher-guided curriculum for MaxRL

Research prototype exploring the integration of curriculum learning
(teacher–student, ZPD/learnability targeting) with the MaxRL objective
(arXiv:2602.02710). Everything here runs on CPU with numpy only.

## Contents

| file | purpose |
|---|---|
| `RESEARCH.md` | verified deep-research synthesis of modern curriculum RL and where MaxRL fits |
| `THEORY.md` | advantage-mass analysis: exact E[Σ\|w\|] per estimator, derived teacher utility, optimal rollout allocation |
| `DESIGN.md` | integration design, hypotheses, and validation results |
| `maze_gpu/` | GPU testbed (tiny transformer on 17×17 mazes, goal-distance curriculum, pass@k eval) |
| `testbed.py` | skill-chain environment (binary verifier rewards, exact score functions) |
| `estimators.py` | REINFORCE / RLOO / GRPO / MaxRL per-group advantage weights |
| `teachers.py` | Uniform / ZPD-band / ALP / MaxRL-frontier teachers + adaptive rollout allocation |
| `run_experiment.py` | teacher × estimator sweep (final performance) |
| `run_speed.py` | learning-speed (AUC, steps-to-frontier) + adaptive-N comparison |
| `verl_curriculum.py` | drop-in `FrontierTeacher` + `CurriculumSampler` for the verl trainer in this repo |
| `test_verl_curriculum.py` | CPU unit tests for the verl integration module |

## Quick start

```bash
python3 run_experiment.py --steps 400 --seeds 5   # ~1 min on 8 cores
python3 run_speed.py
python3 test_verl_curriculum.py
```

## Core idea in one line

The MaxRL estimator's own math defines the curriculum: the expected total
|advantage| a prompt receives from a group of N rollouts is exactly
`2·(pass@N − pass@1)` — the probability it is solvable within N attempts but
not within one (THEORY.md). The teacher samples prompts proportional to this
derived utility (Thompson-sampled from a Beta posterior), and the optimal
rollout allocation is greedy water-filling on the marginal `p(1−p)^N` — the
probability the next rollout is a group's first success. At N=1 the utility
reduces to SFL's learnability `p(1−p)`; RLOO's advantage mass *is* `2p(1−p)`
exactly, unifying the learnability-curriculum literature with the estimator
algebra.
