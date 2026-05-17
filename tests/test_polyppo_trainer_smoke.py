from pathlib import Path

import pytest

from lerobot.common.polyppo.rollout import collect_polyppo_rollouts
from lerobot.common.polyppo.trainer import train_polyppo_one_update


def _write_rollout_config(tmp_path: Path) -> Path:
    output_dir = tmp_path / "rollout"
    config = tmp_path / "rollout.yaml"
    config.write_text(
        f"""
run:
  id: test_rollout
  output_dir: {output_dir}
  seed: 123
  device: cpu
  gpu_id: 0
  validate_profile: smoke
policy:
  path: null
  temperature: 0.7
rollout:
  mock: true
  n_prefixes: 4
  attempts_per_prefix: 3
  prefix_steps: 2
  continuation_horizon: 4
  n_obs_steps: 2
  action_dim: 2
  action_chunk_size: 2
  n_code_layers: 2
  n_codes: 8
polyppo:
  diversity_kind: code
  lambda_div: 0.1
"""
    )
    return config


def test_train_polyppo_mock_one_update_checks_and_checkpoint(tmp_path):
    rollout_config = _write_rollout_config(tmp_path)
    rollout_payload = collect_polyppo_rollouts(rollout_config)
    train_output = tmp_path / "train"
    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        f"""
run:
  id: test_train
  output_dir: {train_output}
  seed: 123
  device: cpu
  gpu_id: 0
policy:
  path: null
  temperature: 0.7
train:
  rollout_path: {rollout_payload["output_path"]}
  collect_if_missing: false
  mock_policy: true
  lr: 0.01
  clip_ratio: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  kl_coef: 0.0
  ratio_tolerance: 1.0e-6
  post_update_eval_episodes: 10
polyppo:
  diversity_kind: code
  lambda_div: 0.1
"""
    )

    payload = train_polyppo_one_update(train_config)

    assert all(payload["checks"].values())
    assert payload["ppo_ratio_max_abs_error_before_update"] == 0.0
    assert Path(payload["checkpoint_path"]).exists()
    assert payload["checkpoint_hashes"]
    assert Path(payload["post_update_eval_path"], "eval_info.json").exists()
    assert payload["changed_parameters"]


def test_train_polyppo_mock_requires_base_log_probs_for_positive_kl(tmp_path):
    rollout_config = _write_rollout_config(tmp_path)
    rollout_payload = collect_polyppo_rollouts(rollout_config)
    train_config = tmp_path / "train_kl.yaml"
    train_config.write_text(
        f"""
run:
  id: test_train_kl
  output_dir: {tmp_path / "train_kl"}
  seed: 123
  device: cpu
policy:
  path: null
train:
  rollout_path: {rollout_payload["output_path"]}
  collect_if_missing: false
  mock_policy: true
  lr: 0.01
  clip_ratio: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  kl_coef: 0.1
  ratio_tolerance: 1.0e-6
polyppo:
  diversity_kind: code
  lambda_div: 0.1
"""
    )

    with pytest.raises(ValueError, match="base_log_probs"):
        train_polyppo_one_update(train_config)
