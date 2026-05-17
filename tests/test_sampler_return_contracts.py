import pytest
import torch

from lerobot.common.samplers.multi import bidirectional_plus_ema_sampler, bidirectional_sampler
from lerobot.common.samplers.single import coherence_sampler, ema_sampler, random_sampler


class _TuplePolicy:
    def __init__(self, action_dim: int = 2, chunk_len: int = 3, latent_dim: int = 2):
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.latent_dim = latent_dim
        self.calls = 0

    def select_action(self, batch, *args, **kwargs):
        self.calls += 1
        batch_size = batch["observation.state"].shape[0]
        values = torch.arange(
            1,
            batch_size * self.chunk_len * self.action_dim + 1,
            dtype=torch.float32,
        ).reshape(batch_size, self.chunk_len, self.action_dim)
        action = values[:, 0]
        latent = torch.arange(
            batch_size * self.chunk_len * self.latent_dim,
            dtype=torch.long,
        ).reshape(batch_size, self.chunk_len, self.latent_dim)
        return action, values, latent


def _observation(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.ones(batch_size, 4),
        "observation.image": torch.ones(batch_size, 3, 96, 96),
    }


def _assert_action_dict(out: dict[str, torch.Tensor], batch_size: int = 2) -> None:
    assert out["action"].shape == (batch_size, 1, 2)
    assert out["action_pred"].shape == (batch_size, 3, 2)


@pytest.mark.parametrize("sampler", [coherence_sampler, random_sampler, ema_sampler])
def test_single_policy_samplers_ignore_extra_select_action_metadata(sampler):
    policy = _TuplePolicy()

    out = sampler(policy, None, _observation(), ah_count=0, ah_test=1, num_sample=4)

    _assert_action_dict(out)
    assert policy.calls == 1


@pytest.mark.parametrize("sampler", [bidirectional_sampler, bidirectional_plus_ema_sampler])
def test_bidirectional_samplers_ignore_extra_select_action_metadata(sampler):
    strong = _TuplePolicy()
    weak = _TuplePolicy()

    out = sampler(strong, weak, None, _observation(), ah_count=0, ah_test=1, num_sample=4, factor=2)

    _assert_action_dict(out)
    assert strong.calls == 1
    assert weak.calls == 1
