"""Unit tests for frontier_rl. Run: python3 frontier_rl/test_framework.py"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from frontier_rl import (FrontierTeacher, FrontierTrainer, TrainerConfig,
                         maxrl_weights, grpo_weights, rloo_weights)
from frontier_rl.adapters.skill_chain import SkillChainSpace
from frontier_rl.adapters.grid_reach import GridReachSpace


def test_estimators():
    r = np.array([1., 0., 0., 1.])
    w = maxrl_weights(r)
    assert abs(w.sum()) < 1e-12 and abs(w[0] - (0.5 - 0.25)) < 1e-12
    assert not maxrl_weights(np.zeros(4)).any()
    assert abs(rloo_weights(r).sum()) < 1e-12
    assert abs(grpo_weights(r).sum()) < 1e-12
    print("estimators OK")


def test_teacher_posterior_and_utility():
    t = FrontierTeacher(n_tasks=20, n_rollouts=16, seed=0)
    for _ in range(6):
        t.observe(3, np.array([1.]*4 + [0.]*12))   # frontier p~0.25
        t.observe(4, np.ones(16))                   # mastered
        t.observe(5, np.zeros(16))                  # dead
    d = np.zeros(20)
    for _ in range(300):
        d += t.distribution()
    d /= 300
    assert d[3] > d[4] and d[3] > d[5], (d[3], d[4], d[5])
    assert abs(d.sum() - 1) < 1e-9
    # utility peak location ~ln(N)/N
    p = np.linspace(1e-4, 0.999, 4000)
    peak = p[np.argmax(t.utility(p))]
    assert abs(peak - (1 - 16 ** (-1/15))) < 2e-3
    print("teacher OK")


def test_teacher_state_roundtrip():
    t = FrontierTeacher(5, 8, seed=0)
    t.observe(2, np.array([1., 0., 1., 0., 0., 0., 0., 0.]))
    t2 = FrontierTeacher(5, 8, seed=0)
    t2.load_state_dict(t.state_dict())
    assert np.allclose(t.alpha, t2.alpha) and np.allclose(t.beta, t2.beta)
    print("state roundtrip OK")


def test_trainer_on_skill_chain():
    env = SkillChainSpace(seed=0)
    trainer = FrontierTrainer(env, env, TrainerConfig(seed=0, hindsight=True,
                                                       teacher_gamma=4.0))
    before = env.true_pass_rates().mean()
    stats = trainer.train(steps=60)
    after = env.true_pass_rates().mean()
    assert after > before + 0.15, (before, after)
    assert any(s.relabeled_groups > 0 for s in stats), "hindsight never fired"
    print(f"trainer on skill chain OK ({before:.3f} -> {after:.3f})")


def test_hindsight_contract_gridworld():
    env = GridReachSpace(radius=6, seed=0)
    g = env.rollout_group(5, 16)   # ring 6: hard from scratch
    if g.rewards.sum() == 0:
        rel = env.relabel(g)
        assert rel is not None
        new_task, new_r, new_trajs = rel
        assert 0 <= new_task < env.n_tasks
        assert new_r.sum() >= 1, "relabel must create at least one success"
        ring = new_task + 1
        for r, info, nt in zip(new_r, g.infos, new_trajs):
            end_ring = max(abs(info["final_pos"][0]), abs(info["final_pos"][1]))
            if r == 1.0:
                # exactness (P6 contract 1): success truly ended on that ring
                assert end_ring == ring
                # conditioning (contract 2): goal rewritten to achieved cell
                assert np.array_equal(nt["goal"], info["final_pos"])
    print("hindsight contract OK")


def test_dead_group_without_relabel_is_skipped():
    class NoRelabelEnv(SkillChainSpace):
        def relabel(self, group):
            return None
    env = NoRelabelEnv(seed=0)
    trainer = FrontierTrainer(env, env, TrainerConfig(seed=0, hindsight=True))
    stats = trainer.train(steps=10)
    assert all(s.relabeled_groups == 0 for s in stats)
    print("no-relabel fallback OK")


if __name__ == "__main__":
    test_estimators()
    test_teacher_posterior_and_utility()
    test_teacher_state_roundtrip()
    test_trainer_on_skill_chain()
    test_hindsight_contract_gridworld()
    test_dead_group_without_relabel_is_skipped()
    print("\nALL TESTS PASSED")
