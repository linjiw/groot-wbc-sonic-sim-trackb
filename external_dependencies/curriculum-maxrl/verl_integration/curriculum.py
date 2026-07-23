# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Curriculum teacher + weighted sampler for MaxRL-style RL training.

The teacher tracks a decayed Beta posterior over each prompt's pass rate and
samples prompts by the *expected advantage mass* of the MaxRL estimator: for
a group of N rollouts on a prompt with pass rate p, the expected total
|advantage| emitted (Algorithm 1 of arXiv:2602.02710, w_succ = 1/K - 1/N,
w_fail = -1/N, K=0 groups dropped) is exactly

    E[sum_j |w_j|] = 2 * (pass@N(p) - pass@1(p)) = 2 * ((1-(1-p)^N) - p),

i.e. twice the probability the prompt is solvable within N attempts but not
within one.  Sampling proportional to this utility maximizes the learning
signal the optimizer receives per group.  It peaks at p* ~ ln(N)/N, so
larger group sizes automatically target harder prompts.  At N=1 it reduces
to the "learnability" objective p(1-p) of Rutherford et al. (2024).  The
older heuristic frontier utility (1-(1-p)^N)(1-p) is available via
utility="frontier" and is numerically near-identical.

Enable via config:

    data.curriculum.enable=true
    data.curriculum.floor=0.1          # uniform replay floor
    data.curriculum.decay=0.9          # posterior decay per observation
    data.curriculum.success_threshold=0.5
    data.curriculum.utility=advmass    # or "frontier"

The sampler re-draws Thompson weights each epoch; the trainer feeds
observations back after every reward computation (see ray_trainer.fit).
"""

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class FrontierTeacher:
    def __init__(self, n_prompts, n_rollouts=16, decay=0.7, floor=0.1, seed=0,
                 success_threshold=0.5, utility="advmass", power=1.0):
        # decay=0.7 validated in VALIDATION.md V2b: the oracle-vs-Thompson gap
        # is a tracking problem; faster forgetting closes ~19% of it.
        # power: sample ∝ u^power (V6) — sharper concentration compounds on
        # chain-structured pools (γ≈4); keep 1.0 for flat prompt sets (GSM8K).
        assert utility in ("advmass", "frontier"), utility
        self.n_prompts = n_prompts
        self.n_rollouts = n_rollouts
        self.decay = decay
        self.floor = floor
        self.success_threshold = success_threshold
        self.utility_kind = utility
        self.power = power
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_prompts, dtype=np.float64)
        self.beta = np.ones(n_prompts, dtype=np.float64)
        self.visits = np.zeros(n_prompts, dtype=np.int64)

    def observe_batch(self, dataset_indices, uids, scores):
        """Update posteriors from one training batch.

        Args:
          dataset_indices: (bs,) original dataset row per rollout
          uids: (bs,) prompt-group id per rollout
          scores: (bs,) scalar reward per rollout
        """
        by_uid = defaultdict(list)
        uid_to_idx = {}
        for di, u, s in zip(dataset_indices, uids, scores):
            try:
                idx = int(di)
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < self.n_prompts):
                continue
            by_uid[u].append(float(s) > self.success_threshold)
            uid_to_idx[u] = idx
        for u, successes in by_uid.items():
            idx = uid_to_idx[u]
            k = sum(successes)
            n = len(successes)
            self.alpha[idx] = 1.0 + (self.alpha[idx] - 1.0) * self.decay + k
            self.beta[idx] = 1.0 + (self.beta[idx] - 1.0) * self.decay + (n - k)
            self.visits[idx] += 1

    def sampling_weights(self):
        p = self.rng.beta(self.alpha, self.beta)
        pass_at_n = 1.0 - (1.0 - p) ** self.n_rollouts
        if self.utility_kind == "advmass":
            u = np.maximum(pass_at_n - p, 0.0)
        else:  # frontier
            u = pass_at_n * (1.0 - p)
        if self.power != 1.0:
            u = u ** self.power
        total = u.sum()
        if total <= 1e-12:
            return np.full(self.n_prompts, 1.0 / self.n_prompts)
        probs = u / total
        uniform = np.full(self.n_prompts, 1.0 / self.n_prompts)
        return (1.0 - self.floor) * probs + self.floor * uniform

    def pass_rate_estimates(self):
        return self.alpha / (self.alpha + self.beta)

    def metrics(self):
        """Scalars for logging."""
        p = self.pass_rate_estimates()
        visited = self.visits > 0
        out = {
            "curriculum/visited_frac": float(visited.mean()),
            "curriculum/mean_p_hat_visited": float(p[visited].mean()) if visited.any() else 0.0,
        }
        if visited.any():
            pv = p[visited]
            out["curriculum/frac_dead_p_lt_0.05"] = float((pv < 0.05).mean())
            out["curriculum/frac_mastered_p_gt_0.9"] = float((pv > 0.9).mean())
        return out

    def state_dict(self):
        return {"alpha": self.alpha, "beta": self.beta, "visits": self.visits}

    def load_state_dict(self, state):
        self.alpha = np.asarray(state["alpha"], dtype=np.float64)
        self.beta = np.asarray(state["beta"], dtype=np.float64)
        self.visits = np.asarray(state["visits"], dtype=np.int64)


class CurriculumSampler(Sampler):
    """Weighted sampler (with replacement) driven by a FrontierTeacher.

    Weights are re-drawn from the teacher at the start of every epoch, so the
    curriculum adapts without rebuilding the dataloader.  Epoch length equals
    the dataset size, keeping total_training_steps bookkeeping unchanged.
    """

    def __init__(self, data_source, teacher, seed=1):
        self.n = len(data_source)
        self.teacher = teacher
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        w = self.teacher.sampling_weights()
        idx = self.rng.choice(self.n, size=self.n, replace=True, p=w)
        return iter(idx.tolist())
