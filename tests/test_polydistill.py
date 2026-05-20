from pathlib import Path

import torch

from lerobot.common.polyppo.distill import (
    collect_polydistill_dataset,
    flatten_selected_steps,
    select_top_attempt_indices,
    train_polydistill,
)
from lerobot.common.polyppo.distill_v2 import (
    attempt_weights,
    collect_polydistill_v2_dataset,
    flatten_weighted_candidate_steps,
    rank_attempt_indices,
    train_polydistill_v2,
)


def test_select_top_attempt_prefers_success_then_overlap_then_return():
    successes = torch.tensor([[False, True, True], [False, False, False]])
    overlaps = torch.tensor([[0.9, 0.2, 0.8], [0.1, 0.7, 0.7]])
    returns = torch.tensor([[10.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    selected = select_top_attempt_indices(successes, overlaps, returns)

    assert torch.equal(selected, torch.tensor([2, 2]))


def test_flatten_selected_steps_drops_padding():
    obs_state = torch.randn(2, 3, 2, 4)
    obs_image = torch.randn(2, 3, 2, 3, 8, 8)
    code_ids = torch.randint(0, 8, (2, 3, 2))
    valid = torch.tensor([[True, False, True], [False, True, False]])

    batch, flat_codes = flatten_selected_steps(obs_state, obs_image, code_ids, valid)

    assert batch["observation.state"].shape == (3, 2, 4)
    assert batch["observation.image"].shape == (3, 2, 3, 8, 8)
    assert torch.equal(flat_codes, code_ids[valid])


def test_collect_and_train_polydistill_mock_smoke(tmp_path: Path):
    dataset_config = tmp_path / "collect.yaml"
    train_config = tmp_path / "train.yaml"
    dataset_dir = tmp_path / "dataset"
    train_dir = tmp_path / "train"
    dataset_config.write_text(
        f"""
run:
  id: test_distill_collect
  output_dir: {dataset_dir}
  seed: 123
  device: cpu
distill:
  mock: true
  n_start_states: 4
  attempts_per_start: 3
  max_steps: 5
"""
    )
    train_config.write_text(
        f"""
run:
  id: test_distill_train
  output_dir: {train_dir}
  seed: 123
  device: cpu
train:
  dataset_path: {dataset_dir}
  mock_policy: true
  epochs: 2
"""
    )

    dataset = collect_polydistill_dataset(dataset_config)
    train = train_polydistill(train_config)

    assert dataset["validation_status"] == "passed"
    assert dataset["aggregated"]["selected_success_rate"] == 1.0
    assert all(train["checks"].values())
    assert Path(train["checkpoint_path"]).exists()


def test_rank_attempt_indices_prefers_success_then_overlap_then_return():
    successes = torch.tensor([[False, True, True, False]])
    overlaps = torch.tensor([[0.99, 0.2, 0.8, 0.7]])
    returns = torch.tensor([[100.0, 0.0, 1.0, 2.0]])

    ranked = rank_attempt_indices(successes, overlaps, returns)

    assert torch.equal(ranked[0, :4], torch.tensor([2, 1, 0, 3]))


def test_attempt_weights_modes_are_per_start_normalized():
    successes = torch.tensor([[False, True, True], [False, False, False]])
    overlaps = torch.tensor([[0.5, 0.6, 0.9], [0.1, 0.3, 0.2]])
    returns = torch.zeros_like(overlaps)

    top1 = attempt_weights(successes, overlaps, returns, objective="top1_ce", top_m=3, tau=0.1)
    topm = attempt_weights(successes, overlaps, returns, objective="topm_soft_ce", top_m=2, tau=0.1)
    success = attempt_weights(
        successes, overlaps, returns, objective="success_filtered_soft_ce", top_m=3, tau=0.1
    )
    adv = attempt_weights(successes, overlaps, returns, objective="adv_weighted_bc", top_m=3, tau=0.5)

    assert torch.equal(top1, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert torch.allclose(topm.sum(dim=1), torch.ones(2))
    assert success[0, 0] == 0
    assert success[1, 0] == 1
    assert torch.allclose(success.sum(dim=1), torch.ones(2))
    assert torch.allclose(adv.sum(dim=1), torch.ones(2))


def test_flatten_weighted_candidate_steps_drops_zero_weight_and_padding():
    obs_state = torch.randn(2, 2, 3, 2, 4)
    obs_image = torch.randn(2, 2, 3, 2, 3, 8, 8)
    code_ids = torch.randint(0, 8, (2, 2, 3, 2))
    valid = torch.tensor(
        [
            [[True, False, True], [True, True, False]],
            [[False, True, False], [True, False, False]],
        ]
    )
    weights = torch.tensor([[1.0, 0.0], [0.25, 0.75]])

    batch, flat_codes, flat_weights = flatten_weighted_candidate_steps(
        obs_state, obs_image, code_ids, valid, weights
    )
    positive = flat_weights > 0

    assert batch["observation.state"].shape[0] == int(valid.sum().item())
    assert batch["observation.image"].shape[1:] == (2, 3, 8, 8)
    assert torch.equal(flat_codes, code_ids[valid])
    assert positive.sum().item() == 4


def test_collect_and_train_polydistill_v2_mock_smoke(tmp_path: Path):
    dataset_config = tmp_path / "collect_v2.yaml"
    train_config = tmp_path / "train_v2.yaml"
    dataset_dir = tmp_path / "dataset_v2"
    train_dir = tmp_path / "train_v2"
    dataset_config.write_text(
        f"""
run:
  id: test_distill_v2_collect
  output_dir: {dataset_dir}
  seed: 123
  device: cpu
distill:
  mock: true
  n_start_states: 4
  attempts_per_start: 5
  keep_top_attempts: 3
  max_steps: 5
"""
    )
    train_config.write_text(
        f"""
run:
  id: test_distill_v2_train
  output_dir: {train_dir}
  seed: 123
  device: cpu
train:
  dataset_path: {dataset_dir}
  mock_policy: true
  epochs: 2
objective:
  name: topm_soft_ce
  top_m: 2
  tau: 0.1
"""
    )

    dataset = collect_polydistill_v2_dataset(dataset_config)
    train = train_polydistill_v2(train_config)

    assert dataset["artifact_kind"] == "polydistill_v2_dataset"
    assert dataset["validation_status"] == "passed"
    assert dataset["tensor_shapes"]["all_code_ids"][:2] == [4, 5]
    assert dataset["tensor_shapes"]["candidate_code_ids"][:2] == [4, 3]
    assert all(train["checks"].values())
