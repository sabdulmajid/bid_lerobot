import json

import pytest
import numpy as np
import torch
from torch import nn

from lerobot.scripts.eval import (
    _add_observation_noise,
    _compute_pass_at_k_for_eval,
    _direct_action_dict,
    load_pretrained_policy_hydra_config,
    rollout,
)


class _PlainPolicy:
    def select_action(self, observation):
        del observation
        return torch.ones(2, 3)


class _PlainModulePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def reset(self):
        pass

    def select_action(self, observation):
        batch_size = observation["observation.state"].shape[0]
        return torch.ones(batch_size, 2, device=self.anchor.device)


class _OneStepVectorEnv:
    num_envs = 1

    def __init__(self):
        self.actions = []

    def reset(self, seed=None):
        del seed
        return {"agent_pos": np.zeros((1, 2), dtype=np.float32)}, {}

    def call(self, name):
        if name != "_max_episode_steps":
            raise ValueError(name)
        return [1]

    def step(self, action):
        self.actions.append(action.copy())
        return (
            {"agent_pos": np.zeros((1, 2), dtype=np.float32)},
            np.zeros(1, dtype=np.float32),
            np.array([True]),
            np.array([False]),
            {},
        )


def test_direct_action_dict_supports_non_vqbet_policy_protocol():
    out = _direct_action_dict(_PlainPolicy(), {"observation.state": torch.zeros(2, 4)}, ah_test=1, temperature=1.0)

    assert out["action"].shape == (2, 1, 3)
    assert out["action_pred"].shape == (2, 1, 3)


def test_rollout_direct_single_step_noise_zero_preserves_action():
    env = _OneStepVectorEnv()
    policy = _PlainModulePolicy()

    out = rollout(env, policy, policy, sampler="direct", ah_test=1, noise_level=0.0)

    assert torch.equal(out["action"][0, 0], torch.ones(2))
    assert np.array_equal(env.actions[0], np.ones((1, 2), dtype=np.float32))


def test_add_observation_noise_perturbs_only_floating_tensors():
    observation = {
        "float": torch.zeros(3),
        "int": torch.tensor([1, 2, 3]),
    }
    generator = torch.Generator().manual_seed(0)

    noised = _add_observation_noise(observation, 0.1, generator)

    assert not torch.equal(noised["float"], observation["float"])
    assert torch.equal(noised["int"], observation["int"])


def test_add_observation_noise_zero_std_returns_original_mapping():
    observation = {"float": torch.zeros(3)}

    assert _add_observation_noise(observation, 0.0, torch.Generator()) is observation


def test_compute_pass_at_k_for_eval_requires_grouping_for_multi_attempt_metrics():
    pass_at_k, note = _compute_pass_at_k_for_eval([False, True, True], (1, 2, 4))

    assert pass_at_k == {"pass@1": 2 / 3}
    assert "single-attempt" in note


def test_compute_pass_at_k_for_eval_rejects_fake_grouping():
    with pytest.raises(ValueError, match="same restored start state"):
        _compute_pass_at_k_for_eval(
            [False, True, False, False, False, True],
            (1, 2, 3),
            pass_at_group_size=3,
        )


def test_config_json_loader_rejects_non_vqbet_policy_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"policy_type": "act"}))

    with pytest.raises(ValueError, match="not supported"):
        load_pretrained_policy_hydra_config(tmp_path)
