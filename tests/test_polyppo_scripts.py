import subprocess
import sys
from pathlib import Path


def test_collect_and_train_polyppo_scripts_run_mock_smoke(tmp_path):
    rollout_config = tmp_path / "rollout.yaml"
    train_config = tmp_path / "train.yaml"
    rollout_dir = tmp_path / "rollout"
    train_dir = tmp_path / "train"
    rollout_config.write_text(
        f"""
run:
  id: cli_rollout
  output_dir: {rollout_dir}
  seed: 123
  device: cpu
  gpu_id: 0
  validate_profile: smoke
policy:
  path: null
rollout:
  mock: true
  n_prefixes: 4
  attempts_per_prefix: 3
  continuation_horizon: 4
polyppo:
  diversity_kind: code
  lambda_div: 0.1
"""
    )
    train_config.write_text(
        f"""
run:
  id: cli_train
  output_dir: {train_dir}
  seed: 123
  device: cpu
  gpu_id: 0
policy:
  path: null
train:
  rollout_path: {rollout_dir}
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

    collect = subprocess.run(
        [sys.executable, "-m", "lerobot.scripts.collect_polyppo_rollouts", "--config", str(rollout_config)],
        check=False,
        text=True,
        capture_output=True,
    )
    train = subprocess.run(
        [sys.executable, "-m", "lerobot.scripts.train_polyppo", "--config", str(train_config)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert collect.returncode == 0, collect.stderr + collect.stdout
    assert train.returncode == 0, train.stderr + train.stdout
    assert Path(train_dir, "polyppo_train_info.json").exists()
