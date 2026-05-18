from pathlib import Path

import torch

from lerobot.common.polyppo.distill import (
    collect_polydistill_dataset,
    flatten_selected_steps,
    select_top_attempt_indices,
    train_polydistill,
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
