"""The three contracts a new environment must satisfy.

Design constraints (why the interface looks like this):
- The teacher only ever sees (task_id, binary rewards) — matching what an
  RLVR trainer observes.  No env internals leak into the curriculum.
- Hindsight requires the env, not the trainer, to say what a failed rollout
  achieved: `relabel()` returns (new_task_id, is_success_under_new_task) —
  because only the env knows its goal structure (P6: relabeling quality is
  an env property; the trainer applies the same estimator either way).
- The policy update is a callback: this framework schedules *what to train
  on and with what advantage weights*; how gradients are applied (PPO step,
  REINFORCE, an LLM fine-tune) stays in user code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable


@dataclass
class GroupResult:
    """One group of N rollouts on one task."""
    task_id: int
    rewards: Any            # np.ndarray (N,) binary
    trajectories: list      # opaque to the framework; passed back to Policy
    infos: list = field(default_factory=list)  # per-rollout env info


@runtime_checkable
class TaskSpace(Protocol):
    """The environment side: a discrete set of tasks (goals, levels, prompts).

    For continuous goal spaces, discretize (bins/level sets) or subclass the
    teacher with a parametric density model — the trainer only needs
    task ids to be stable keys.
    """

    @property
    def n_tasks(self) -> int: ...

    def rollout_group(self, task_id: int, n_rollouts: int) -> GroupResult:
        """Run N episodes/attempts on the task; binary success rewards."""
        ...

    def relabel(self, group: GroupResult) -> Optional[tuple]:
        """Hindsight hook, called only on dead (all-fail) groups.

        Return (relabeled_task_id, new_rewards) or
        (relabeled_task_id, new_rewards, new_trajectories).  new_rewards
        marks which rollouts count as successes *of the relabeled task* —
        e.g. the goal actually reached, the deepest verified prefix.  Return
        None if the env has no meaningful relabeling (the trainer then just
        skips the group, plain-MaxRL style).

        Two contracts (both from P6 — the relabeled update is the ML gradient
        of the relabeled task only if the conditional laws match):

        1. EXACTNESS: a relabeled success must be a TRUE success of the
           relabeled task under the env's own verifier.  Never return
           "almost reached" as success.
        2. CONDITIONING: if trajectories embed the goal (goal-relative
           features, desired_goal in observations, goal tokens in a prompt),
           return new_trajectories with the goal REWRITTEN to the relabeled
           one — training goal-A-conditioned actions as successes of goal B
           mis-trains the conditioning (this is HER's observation-rewrite,
           and skipping it silently *hurts*: see the grid_reach adapter note).
        """
        ...


@runtime_checkable
class Policy(Protocol):
    """The learner side: applies one weighted policy-gradient-style update.

    weights[i] multiplies grad log pi(trajectories[i]); the framework has
    already encoded the estimator (and hindsight scale) in them.  task_id is
    the task the trajectories are *credited to* (differs from the sampled
    task for relabeled groups — conditioning/prompt must be rebuilt for it,
    which is why it is passed explicitly).
    """

    def update(self, task_id: int, trajectories: Sequence, weights: Any) -> None: ...
