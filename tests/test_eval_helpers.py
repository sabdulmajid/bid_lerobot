import torch

from lerobot.scripts.eval import _add_observation_noise, _direct_action_dict


class _PlainPolicy:
    def select_action(self, observation):
        del observation
        return torch.ones(2, 3)


def test_direct_action_dict_supports_non_vqbet_policy_protocol():
    out = _direct_action_dict(_PlainPolicy(), {"observation.state": torch.zeros(2, 4)}, ah_test=1, temperature=1.0)

    assert out["action"].shape == (2, 1, 3)
    assert out["action_pred"].shape == (2, 1, 3)


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
