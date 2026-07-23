"""frontier_rl — the curriculum-MaxRL training schedule as a reusable framework.

The validated algorithm (REPORT.md), environment-agnostic:

  1. TEACHER   — Thompson sampling over a decayed Beta posterior per task,
                 utility = expected MaxRL advantage mass u(p) = pass@N − pass@1
                 (PROOFS.md P1), concentration u^gamma (V6), uniform floor.
  2. ESTIMATOR — MaxRL success-conditioned advantages (w = r/K − 1/N, drop
                 K=0 groups); GRPO/RLOO included for comparison only.
  3. HINDSIGHT — dense relabeling of dead groups to achieved sub-goals
                 (P6/V1: exact ML gradient where the env's relabel is exact).
  4. LOOP      — group rollouts → teacher.observe → advantages (+relabels)
                 → user-supplied policy update.

To plug in a new environment (gym task, robotics sim), implement the
`TaskSpace` protocol in `interfaces.py` — the trainer never imports your
simulator. See `adapters/` for three references, from a 40-line toy to a
gym-style continuous-control task.
"""

from frontier_rl.interfaces import TaskSpace, GroupResult, Policy
from frontier_rl.teacher import FrontierTeacher
from frontier_rl.estimators import maxrl_weights, grpo_weights, rloo_weights
from frontier_rl.trainer import FrontierTrainer, TrainerConfig

__all__ = [
    "TaskSpace", "GroupResult", "Policy",
    "FrontierTeacher", "FrontierTrainer", "TrainerConfig",
    "maxrl_weights", "grpo_weights", "rloo_weights",
]
