"""CPU unit tests for verl_curriculum.py (no torch/verl required)."""

import numpy as np

from verl_curriculum import FrontierTeacher, CurriculumSampler, allocate_rollout_budget


def test_teacher_updates_and_weights():
    teacher = FrontierTeacher(n_prompts=10, n_rollouts=8, seed=0)

    # prompt 0: mastered (all successes); prompt 1: frontier (2/8);
    # prompt 2: dead (0/8). one batch = 3 prompt groups x 8 rollouts.
    idx, uids, scores = [], [], []
    for prompt, k in [(0, 8), (1, 2), (2, 0)]:
        for j in range(8):
            idx.append(prompt)
            uids.append(f"u{prompt}")
            scores.append(1.0 if j < k else 0.0)
    for _ in range(5):  # several batches to sharpen posteriors
        teacher.observe_batch(np.array(idx), np.array(uids), np.array(scores))

    p = teacher.pass_rate_estimates()
    assert p[0] > 0.85 and abs(p[1] - 0.25) < 0.15 and p[2] < 0.15, p[:3]

    w = np.zeros(10)
    for _ in range(200):  # average over Thompson draws
        w += teacher.sampling_weights()
    w /= 200

    # frontier prompt should dominate mastered and dead prompts
    assert w[1] > 2 * w[0], (w[0], w[1])
    assert w[1] > 2 * w[2], (w[1], w[2])
    # coverage floor keeps every prompt sampled
    assert w.min() > 0.1 / 10 * 0.5
    assert abs(w.sum() - 1.0) < 1e-9


def test_sampler_tracks_teacher():
    class FakeDataset(list):
        pass

    ds = FakeDataset(range(50))
    teacher = FrontierTeacher(n_prompts=50, n_rollouts=8, seed=0, floor=0.0)
    # make prompt 7 the lone frontier prompt, everything else mastered
    teacher.alpha[:] = 100.0
    teacher.beta[:] = 1.0
    teacher.alpha[7] = 3.0
    teacher.beta[7] = 9.0

    sampler = CurriculumSampler(ds, teacher, seed=1)
    counts = np.zeros(50)
    for _ in range(20):
        for i in sampler:
            counts[i] += 1
    assert counts[7] == counts.max()
    assert counts[7] > 5 * np.median(counts)


def test_budget_allocation():
    p_hat = np.array([0.9, 0.5, 0.05, 0.01])
    n = allocate_rollout_budget(p_hat, total_budget=64, n_min=4, n_max=32)
    assert n.sum() == 64
    assert n[3] >= n[2] >= n[1] >= n[0]
    assert n.min() >= 4 and n.max() <= 32


def test_teacher_checkpoint_roundtrip():
    t1 = FrontierTeacher(n_prompts=5, n_rollouts=8, seed=0)
    t1.observe_batch(np.array([0] * 8), np.array(["u"] * 8),
                     np.array([1.0] * 4 + [0.0] * 4))
    t2 = FrontierTeacher(n_prompts=5, n_rollouts=8, seed=0)
    t2.load_state_dict(t1.state_dict())
    assert np.allclose(t1.alpha, t2.alpha) and np.allclose(t1.beta, t2.beta)


if __name__ == "__main__":
    test_teacher_updates_and_weights()
    test_sampler_tracks_teacher()
    test_budget_allocation()
    test_teacher_checkpoint_roundtrip()
    print("all tests passed")
