import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import torch

from lerobot.scripts import eval_grouped_passk as grouped_module
from lerobot.scripts import eval_polyppo_checkpoints as eval_ckpt_module


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


def test_eval_polyppo_checkpoints_passes_configured_stress_knobs(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {}}, checkpoint)
    config = tmp_path / "stress.yaml"
    output_dir = tmp_path / "stress_out"
    config.write_text(
        f"""
run:
  output_dir: {output_dir}
  seed: 11
  device: cpu
policy:
  path: fake-policy
  temperature: 0.2
checkpoints:
  - name: direct
    checkpoint_path:
  - name: method
    checkpoint_path: {checkpoint}
stress:
  variants:
    - name: obs_noise
      action_noise_std: 0.01
      observation_noise_std: 0.03
  pass_at_k: [1, 2, 4]
  pass_at_group_size: 3
  same_seed_across_checkpoints: true
  eval_episodes: 3
  eval_batch_size: 3
"""
    )
    calls = []

    class _Policy(torch.nn.Module):
        def reset(self):
            return None

    class _Env:
        def close(self):
            return None

    def _fake_eval_policy(*args, **kwargs):
        del args
        calls.append(kwargs)
        return {
            "per_episode": [
                {
                    "episode_ix": ix,
                    "seed": kwargs["start_seed"] + ix,
                    "sum_reward": 1.0,
                    "max_reward": 1.0,
                    "success": ix == 0,
                    "num_steps": 10,
                }
                for ix in range(3)
            ],
            "aggregated": {
                "avg_sum_reward": 1.0,
                "avg_max_reward": 1.0,
                "avg_num_steps": 10.0,
                "pass_at_k": {f"pass@{k}": 1.0 for k in kwargs["pass_at_ks"]},
            },
        }

    monkeypatch.setattr(eval_ckpt_module, "get_pretrained_policy_path", lambda path: Path(path))
    monkeypatch.setattr(
        eval_ckpt_module,
        "load_pretrained_policy_hydra_config",
        lambda path, overrides: SimpleNamespace(
            device="cpu",
            use_amp=False,
            seed=0,
            eval=SimpleNamespace(n_episodes=0, batch_size=0, use_async_envs=False),
            policy=SimpleNamespace(bet_softmax_temperature=1.0),
        ),
    )
    monkeypatch.setattr(eval_ckpt_module, "make_policy", lambda **kwargs: _Policy())
    monkeypatch.setattr(eval_ckpt_module, "make_env", lambda cfg: _Env())
    monkeypatch.setattr(eval_ckpt_module, "eval_policy", _fake_eval_policy)
    monkeypatch.setattr(
        eval_ckpt_module,
        "validate_artifact",
        lambda *args, **kwargs: SimpleNamespace(ok=True, errors=[]),
    )

    summary = eval_ckpt_module.eval_polyppo_checkpoints(config)

    assert len(calls) == 2
    assert calls[0]["start_seed"] == calls[1]["start_seed"] == 11
    assert calls[0]["noise_level"] == 0.01
    assert calls[0]["observation_noise_std"] == 0.03
    assert calls[0]["pass_at_ks"] == (1, 2, 4)
    assert calls[0]["pass_at_group_size"] == 3
    assert summary["rows"][0]["pass_at_k"] == {"pass@1": 1.0, "pass@2": 1.0, "pass@4": 1.0}


def test_eval_grouped_passk_pairs_start_seed_across_methods(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {}}, checkpoint)
    config = tmp_path / "grouped.yaml"
    output_dir = tmp_path / "grouped_out"
    config.write_text(
        f"""
run:
  output_dir: {output_dir}
  seed: 1234
  device: cpu
policy:
  path: fake-policy
checkpoints:
  - name: direct
    checkpoint_path:
  - name: method
    checkpoint_path: {checkpoint}
grouped_passk:
  n_start_states: 2
  attempts_per_start: 4
  same_start_states_across_methods: true
  pass_at_k: [1, 2, 4]
"""
    )
    calls = []

    class _Policy(torch.nn.Module):
        def reset(self):
            return None

    def _fake_eval_one_policy_grouped(**kwargs):
        calls.append({"method_name": kwargs["method_name"], "seed": kwargs["seed"]})
        return {
            "aggregated": {
                "success_rate": 1.0,
                "pass_at_k": {"pass@1": 1.0, "pass@2": 1.0, "pass@4": 1.0},
            }
        }

    monkeypatch.setattr(grouped_module, "get_pretrained_policy_path", lambda path: Path(path))
    monkeypatch.setattr(
        grouped_module,
        "load_pretrained_policy_hydra_config",
        lambda path, overrides: SimpleNamespace(
            device="cpu",
            use_amp=False,
            seed=0,
            eval=SimpleNamespace(use_async_envs=False),
            policy=SimpleNamespace(bet_softmax_temperature=1.0),
        ),
    )
    monkeypatch.setattr(grouped_module, "make_policy", lambda **kwargs: _Policy())
    monkeypatch.setattr(grouped_module, "_eval_one_policy_grouped", _fake_eval_one_policy_grouped)
    monkeypatch.setattr(
        grouped_module,
        "validate_artifact",
        lambda *args, **kwargs: SimpleNamespace(ok=True, errors=[]),
    )

    summary = grouped_module.eval_grouped_passk(config)

    assert [call["method_name"] for call in calls] == ["direct", "method"]
    assert [call["seed"] for call in calls] == [1234, 1234]
    assert summary["same_start_states_across_methods"] is True
