import pytest
import torch
from omegaconf import OmegaConf

from lerobot.common.policies import factory


class _FakeConfig:
    def __init__(self, name: str = "fake"):
        self.name = name


class _FakeCheckpoint:
    def state_dict(self):
        return {"kept": torch.ones(1)}


class _FakePolicy(torch.nn.Module):
    name = "fake"

    def __init__(self, config, dataset_stats=None):
        super().__init__()
        del config, dataset_stats
        self.kept = torch.nn.Parameter(torch.zeros(1))
        self.missing = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, path):
        del path
        return _FakeCheckpoint()


class _FakeValueHeadMigrationPolicy(torch.nn.Module):
    name = "fake"

    def __init__(self, config, dataset_stats=None):
        super().__init__()
        del config, dataset_stats
        self.kept = torch.nn.Parameter(torch.zeros(1))
        self.vqbet = torch.nn.Module()
        self.vqbet.value_head = torch.nn.Linear(1, 1)

    @classmethod
    def from_pretrained(cls, path):
        del path
        return _FakeCheckpoint()


def test_make_policy_rejects_unexpected_missing_checkpoint_weights(monkeypatch):
    monkeypatch.setattr(
        factory,
        "get_policy_and_config_classes",
        lambda name: (_FakePolicy, _FakeConfig),
    )
    hydra_cfg = OmegaConf.create({"device": "cpu", "policy": {"name": "fake"}})

    with pytest.raises(RuntimeError, match="missing"):
        factory.make_policy(hydra_cfg, pretrained_policy_name_or_path="/tmp/fake")


def test_make_policy_allows_vqbet_value_head_checkpoint_migration(monkeypatch):
    monkeypatch.setattr(
        factory,
        "get_policy_and_config_classes",
        lambda name: (_FakeValueHeadMigrationPolicy, _FakeConfig),
    )
    hydra_cfg = OmegaConf.create({"device": "cpu", "policy": {"name": "fake"}})

    policy = factory.make_policy(hydra_cfg, pretrained_policy_name_or_path="/tmp/fake")

    assert isinstance(policy, _FakeValueHeadMigrationPolicy)
