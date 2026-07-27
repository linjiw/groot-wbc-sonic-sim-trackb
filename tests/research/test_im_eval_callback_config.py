from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback, mean_recorded_mpjpe_mm


def test_im_eval_callback_accepts_optional_max_eval_steps() -> None:
    signature = inspect.signature(ImEvalCallback)

    assert "max_eval_steps" in signature.parameters
    callback = ImEvalCallback(eval_frequency=1, max_eval_steps=128)
    assert callback.max_eval_steps == 128


def test_progress_mpjpe_preserves_recorded_millimetres() -> None:
    """The callback records MPJPE in mm, so display must not scale it again."""
    recorded_mm = [[torch.tensor(125.0), torch.tensor(175.0)]]

    assert mean_recorded_mpjpe_mm(recorded_mm) == pytest.approx(150.0)


def test_evaluate_policy_calls_policy_once_for_first_transition() -> None:
    initial_obs = {"actor_obs": torch.tensor([[7.0]])}

    class StubPolicy:
        def __init__(self):
            self.seen_obs = []

        def eval(self):
            return self

        def init_rollout(self):
            pass

        def clear_rollout(self):
            pass

        def act_inference(self, obs_dict, **_kwargs):
            self.seen_obs.append(obs_dict["actor_obs"].clone())
            return torch.tensor([[float(len(self.seen_obs))]])

    class StubEnv:
        num_envs = 1
        device = torch.device("cpu")
        config = SimpleNamespace(robot=SimpleNamespace(actions_dim=1))

        def reset_all(self, **_kwargs):
            return initial_obs

        def render_results(self):
            pass

    callback = ImEvalCallback(eval_frequency=2, eval_only=True)
    callback.accelerator = SimpleNamespace(wait_for_everyone=lambda: None)
    callback.args = SimpleNamespace(global_rank=0)
    callback.env = StubEnv()
    callback.model = SimpleNamespace(policy=StubPolicy())
    callback._eval_mode = lambda: None
    callback._train_mode = lambda: None
    callback._pre_evaluate_policy = lambda: None
    callback._post_evaluate_policy = lambda _actor_state: {"ok": True}

    actions_sent = []

    def env_step(actor_state):
        actions_sent.append(actor_state["actions"].clone())
        return actor_state

    callback.env_step = env_step
    callback._post_eval_env_step = lambda actor_state: {
        **actor_state,
        "end_eval": True,
    }

    callback.evaluate_policy()

    assert len(callback.model.policy.seen_obs) == 1
    assert torch.equal(callback.model.policy.seen_obs[0], initial_obs["actor_obs"])
    assert len(actions_sent) == 1
    assert actions_sent[0].item() == pytest.approx(1.0)


def test_first_action_is_identical_after_each_rollout_reset() -> None:
    first_obs = {"actor_obs": torch.tensor([[10.0]])}
    second_obs = {"actor_obs": torch.tensor([[20.0]])}

    class StubPolicy:
        def __init__(self):
            self.calls_since_reset = 0
            self.init_calls = 0
            self.seen_values = []

        def eval(self):
            return self

        def init_rollout(self):
            self.calls_since_reset = 0
            self.init_calls += 1

        def clear_rollout(self):
            pass

        def act_inference(self, obs_dict, **_kwargs):
            self.calls_since_reset += 1
            self.seen_values.append(float(obs_dict["actor_obs"].item()))
            return torch.tensor([[float(self.calls_since_reset)]])

    class StubEnv:
        num_envs = 1
        device = torch.device("cpu")
        config = SimpleNamespace(robot=SimpleNamespace(actions_dim=1))

        def reset_all(self, **_kwargs):
            return first_obs

        def forward_motion_samples(self, global_rank, world_size):
            assert (global_rank, world_size) == (0, 1)
            return second_obs

        def render_results(self):
            pass

    callback = ImEvalCallback(eval_frequency=2, eval_only=True)
    callback.accelerator = SimpleNamespace(wait_for_everyone=lambda: None)
    callback.args = SimpleNamespace(global_rank=0, world_size=1)
    callback.env = StubEnv()
    callback.model = SimpleNamespace(policy=StubPolicy())
    callback._eval_mode = lambda: None
    callback._train_mode = lambda: None
    callback._pre_evaluate_policy = lambda: None
    callback._post_evaluate_policy = lambda _actor_state: {"ok": True}

    actions_sent = []

    def env_step(actor_state):
        actions_sent.append(float(actor_state["actions"].item()))
        return actor_state

    def post_eval_env_step(actor_state):
        if len(actions_sent) == 1:
            return callback._advance_eval_motion(actor_state)
        actor_state["end_eval"] = True
        return actor_state

    callback.env_step = env_step
    callback._post_eval_env_step = post_eval_env_step

    callback.evaluate_policy()

    assert actions_sent == pytest.approx([1.0, 1.0])
    assert callback.model.policy.seen_values == pytest.approx([10.0, 20.0])
    assert callback.model.policy.init_calls == 2


def test_advancing_motion_uses_reset_observation_and_clears_policy_state() -> None:
    fresh_obs = {"actor_obs": torch.tensor([[1.0, 2.0]])}

    class StubEnv:
        num_envs = 1
        device = torch.device("cpu")

        def forward_motion_samples(self, global_rank, world_size):
            assert (global_rank, world_size) == (0, 1)
            return fresh_obs

    class StubPolicy:
        def __init__(self):
            self.init_calls = 0

        def init_rollout(self):
            self.init_calls += 1

    callback = ImEvalCallback(eval_frequency=1)
    callback.env = StubEnv()
    callback.args = SimpleNamespace(global_rank=0, world_size=1)
    callback.model = SimpleNamespace(policy=StubPolicy())
    actor_state = {
        "obs": {"actor_obs": torch.tensor([[-9.0, -9.0]])},
        "dones": torch.ones(1, dtype=torch.bool),
    }

    advanced = callback._advance_eval_motion(actor_state)

    assert advanced["obs"] is fresh_obs
    assert advanced["dones"].tolist() == [False]
    assert callback.model.policy.init_calls == 1


def test_advancing_motion_rejects_discarded_reset_observation() -> None:
    callback = ImEvalCallback(eval_frequency=1)
    callback.env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        forward_motion_samples=lambda *_args: None,
    )
    callback.args = SimpleNamespace(global_rank=0, world_size=1)
    callback.model = SimpleNamespace(policy=SimpleNamespace(init_rollout=lambda: None))

    with pytest.raises(RuntimeError, match="must return observations"):
        callback._advance_eval_motion({"obs": {}, "dones": torch.ones(1)})
