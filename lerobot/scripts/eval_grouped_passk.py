#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.common.artifacts.validator import validate_artifact
from lerobot.common.envs.pusht_state import restore_pusht_state, snapshot_pusht_state
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.polyppo.metrics import compute_pass_at_k, mean_confidence_interval
from lerobot.common.polyppo.rollout import ObservationHistory, _preprocess_single_observation, _sample_code_action
from lerobot.common.polyppo.utils import (
    checkpoint_hashes,
    command_line,
    config_to_dict,
    current_gpu_id,
    git_metadata,
    load_config,
    select_device,
    stable_hash,
    write_json,
)
from lerobot.common.utils.utils import set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path, load_pretrained_policy_hydra_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate grouped pass@k from repeated PushT starts.")
    parser.add_argument("--config", required=True, help="Path to grouped pass@k YAML config.")
    args = parser.parse_args()
    summary = eval_grouped_passk(args.config)
    print(f"wrote grouped pass@k artifact: {summary['output_path']}")
    return 0


def eval_grouped_passk(config_path: str | Path) -> dict[str, Any]:
    cfg = config_to_dict(load_config(config_path))
    run_cfg = cfg.get("run", {})
    policy_cfg = cfg.get("policy", {})
    eval_cfg = cfg.get("grouped_passk", {})
    output_dir = Path(run_cfg.get("output_dir", "outputs/polyppo/grouped_passk"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(run_cfg.get("device"), run_cfg.get("gpu_id"))
    pretrained_path = get_pretrained_policy_path(policy_cfg.get("path", "lerobot/vqbet_pusht"))

    checkpoints = cfg.get("checkpoints") or [{"name": "pretrained_vqbet", "checkpoint_path": None}]
    rows = []
    for method_ix, checkpoint_cfg in enumerate(checkpoints):
        method_name = checkpoint_cfg["name"]
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        seed = int(run_cfg.get("seed", 130000)) + method_ix * 10000
        hydra_cfg = load_pretrained_policy_hydra_config(pretrained_path, [])
        hydra_cfg.device = str(device)
        hydra_cfg.use_amp = False
        hydra_cfg.eval.use_async_envs = False
        hydra_cfg.seed = seed
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_path))
        checkpoint_path = checkpoint_cfg.get("checkpoint_path")
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            policy.load_state_dict(state["model_state"])
        policy.to(device)
        policy.eval()

        info = _eval_one_policy_grouped(
            policy=policy,
            hydra_cfg=hydra_cfg,
            cfg=cfg,
            method_name=method_name,
            method_dir=method_dir,
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else pretrained_path,
            seed=seed,
            device=device,
        )
        validation = validate_artifact(method_dir, profile=eval_cfg.get("validate_profile", "benchmark"))
        if not validation.ok:
            raise RuntimeError(f"grouped pass@k artifact rejected for {method_name}: {validation.errors}")
        row = dict(info["aggregated"])
        row.update(
            {
                "method": method_name,
                "path": str(method_dir),
                "validation_status": "passed",
            }
        )
        rows.append(row)

    summary = {
        "artifact_kind": "polyppo_grouped_passk_summary",
        "output_path": str(output_dir),
        "rows": rows,
    }
    write_json(output_dir / "grouped_passk_summary.json", summary)
    return summary


def _eval_one_policy_grouped(
    *,
    policy,
    hydra_cfg,
    cfg: dict[str, Any],
    method_name: str,
    method_dir: Path,
    checkpoint_path: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    eval_cfg = cfg.get("grouped_passk", {})
    policy_cfg = cfg.get("policy", {})
    env_cfg = cfg.get("env", {})
    n_starts = int(eval_cfg.get("n_start_states", 8))
    attempts = int(eval_cfg.get("attempts_per_start", 8))
    if n_starts <= 0 or attempts <= 1:
        raise ValueError("grouped_passk requires n_start_states > 0 and attempts_per_start > 1.")
    pass_at_ks = tuple(int(k) for k in eval_cfg.get("pass_at_k", [1, 2, 4, 8]))
    if max(pass_at_ks) > attempts:
        raise ValueError("grouped pass@k cannot request k larger than attempts_per_start.")

    max_steps = int(eval_cfg.get("max_steps", env_cfg.get("episode_length", hydra_cfg.env.episode_length)))
    temperature = float(policy_cfg.get("temperature", hydra_cfg.policy.bet_softmax_temperature))
    set_global_seed(seed)
    gym_kwargs = dict(hydra_cfg.env.get("gym", {}))
    gym_kwargs["max_episode_steps"] = max_steps
    env = gym.make(f"gym_{hydra_cfg.env.name}/{hydra_cfg.env.task}", disable_env_checker=True, **gym_kwargs)

    start_hashes = []
    successes = torch.zeros(n_starts, attempts, dtype=torch.bool)
    max_overlaps = torch.zeros(n_starts, attempts)
    episode_lengths = torch.zeros(n_starts, attempts, dtype=torch.long)
    sum_rewards = torch.zeros(n_starts, attempts)
    per_episode = []
    start_records = []
    wall_start = time.time()
    try:
        for start_ix in range(n_starts):
            start_seed = seed + start_ix
            obs, _ = env.reset(seed=start_seed)
            history = ObservationHistory(hydra_cfg.policy.n_obs_steps)
            history.seed(_preprocess_single_observation(preprocess_observation, obs))
            start_state = snapshot_pusht_state(env)
            start_history = history.clone()
            start_hash = stable_hash(start_state)
            start_hashes.append(start_hash)
            start_attempts = []
            for attempt_ix in range(attempts):
                restore_pusht_state(env, start_state)
                history = start_history.clone()
                if hasattr(policy, "reset"):
                    policy.reset()
                max_coverage = 0.0
                total_reward = 0.0
                success = False
                steps = 0
                for step_ix in range(max_steps):
                    action, _, _, _, _ = _sample_code_action(policy, history.batch(), device, temperature)
                    obs, reward, terminated, truncated, info = env.step(action)
                    history.append(_preprocess_single_observation(preprocess_observation, obs))
                    total_reward += float(reward)
                    max_coverage = max(max_coverage, float(info.get("coverage", 0.0)))
                    success = success or bool(info.get("is_success", False))
                    steps = step_ix + 1
                    if bool(terminated or truncated):
                        break
                successes[start_ix, attempt_ix] = success
                max_overlaps[start_ix, attempt_ix] = max_coverage
                episode_lengths[start_ix, attempt_ix] = steps
                sum_rewards[start_ix, attempt_ix] = total_reward
                episode_ix = start_ix * attempts + attempt_ix
                row = {
                    "episode_ix": episode_ix,
                    "seed": seed * 100000 + episode_ix,
                    "start_ix": start_ix,
                    "attempt_ix": attempt_ix,
                    "start_seed": start_seed,
                    "start_state_hash": start_hash,
                    "sum_reward": total_reward,
                    "max_reward": max_coverage,
                    "success": success,
                    "num_steps": steps,
                }
                per_episode.append(row)
                start_attempts.append(row)
            start_records.append({"start_ix": start_ix, "start_seed": start_seed, "start_state_hash": start_hash, "attempts": start_attempts})
    finally:
        env.close()

    pass_at_k = compute_pass_at_k(successes.numpy(), ks=pass_at_ks)
    coverage = {key.replace("pass@", "coverage@"): int(round(value * n_starts)) for key, value in pass_at_k.items()}
    best_overlap_at_k = {
        f"best_max_overlap@{k}": float(max_overlaps[:, : min(k, attempts)].max(dim=1).values.mean().item())
        for k in pass_at_ks
    }
    aggregated = {
        "method": method_name,
        "n_start_states": n_starts,
        "attempts_per_start": attempts,
        "n_episodes": n_starts * attempts,
        "grouped_same_start": True,
        "avg_sum_reward": float(sum_rewards.mean().item()),
        "avg_max_reward": float(max_overlaps.mean().item()),
        "avg_num_steps": float(episode_lengths.float().mean().item()),
        "success_rate": float(successes.float().mean().item()),
        "pass_at_k": pass_at_k,
        "coverage_at_k": coverage,
        "best_max_overlap_at_k": best_overlap_at_k,
        "success_ci95": mean_confidence_interval(successes.flatten().numpy()),
        "max_overlap_ci95": mean_confidence_interval(max_overlaps.flatten().numpy()),
        "wall_time_s": time.time() - wall_start,
    }
    metadata = git_metadata(Path.cwd())
    info = {
        "per_episode": per_episode,
        "starts": start_records,
        "aggregated": aggregated,
        "run_metadata": {
            "sampler": "grouped_passk_code_sampling",
            "n_episodes": len(per_episode),
            "n_start_states": n_starts,
            "attempts_per_start": attempts,
            "grouped_same_start": True,
            "seed": seed,
            "start_state_hashes": start_hashes,
            "command": command_line(),
            "git_sha": metadata["git_sha"],
            "git_dirty": metadata["git_dirty"],
            "git_branch": metadata["git_branch"],
            "device": str(device),
            "gpu_id": current_gpu_id(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "output_path": str(method_dir),
            "checkpoint": {
                "main_policy_path": str(checkpoint_path),
                "main_policy_hashes": checkpoint_hashes(checkpoint_path),
                "reference_policy_path": None,
                "reference_policy_hashes": [],
            },
            "temperature": temperature,
            "action_identity": "continuous env action from VQ-BeT; grouped pass@k samples RVQ code ids",
        },
    }
    write_json(method_dir / "eval_info.json", info)
    write_json(method_dir / "seed_manifest.json", [row["seed"] for row in per_episode])
    return info


if __name__ == "__main__":
    raise SystemExit(main())
