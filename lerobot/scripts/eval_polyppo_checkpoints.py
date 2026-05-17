#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from lerobot.common.artifacts.validator import validate_artifact
from lerobot.common.envs.factory import make_env
from lerobot.common.policies.factory import make_policy
from lerobot.common.polyppo.metrics import mean_confidence_interval
from lerobot.common.polyppo.utils import (
    checkpoint_hashes,
    command_line,
    config_to_dict,
    current_gpu_id,
    git_metadata,
    load_config,
    select_device,
    write_json,
)
from lerobot.common.utils.utils import set_global_seed
from lerobot.scripts.eval import eval_policy, get_pretrained_policy_path, load_pretrained_policy_hydra_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PolyPPO checkpoints on PushT stress variants.")
    parser.add_argument("--config", required=True, help="Path to a PolyPPO eval YAML config.")
    args = parser.parse_args()
    summary = eval_polyppo_checkpoints(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def eval_polyppo_checkpoints(config_path: str | Path) -> dict[str, Any]:
    cfg = config_to_dict(load_config(config_path))
    run_cfg = cfg.get("run", {})
    policy_cfg = cfg.get("policy", {})
    stress_cfg = cfg.get("stress", {})
    pass_at_ks = tuple(int(k) for k in stress_cfg.get("pass_at_k", [1]))
    pass_at_group_size = stress_cfg.get("pass_at_group_size")
    if pass_at_group_size is not None:
        pass_at_group_size = int(pass_at_group_size)
    output_dir = Path(run_cfg.get("output_dir", "outputs/polyppo/stress_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(run_cfg.get("device"), run_cfg.get("gpu_id"))
    pretrained_path = get_pretrained_policy_path(policy_cfg.get("path", "lerobot/vqbet_pusht"))
    checkpoints = cfg.get("checkpoints") or []
    if not checkpoints:
        raise ValueError("configs/polyppo stress eval requires at least one checkpoint entry.")

    rows = []
    for method_ix, checkpoint_cfg in enumerate(checkpoints):
        method_name = checkpoint_cfg["name"]
        checkpoint_path = Path(checkpoint_cfg["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {method_name}: {checkpoint_path}")

        hydra_cfg = load_pretrained_policy_hydra_config(pretrained_path, [])
        hydra_cfg.device = str(device)
        hydra_cfg.use_amp = False
        hydra_cfg.eval.n_episodes = int(stress_cfg.get("eval_episodes", 20))
        hydra_cfg.eval.batch_size = int(stress_cfg.get("eval_batch_size", hydra_cfg.eval.n_episodes))
        hydra_cfg.eval.use_async_envs = False
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_path))
        state = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(state["model_state"])
        policy.to(device)
        policy.eval()

        for variant_ix, variant in enumerate(stress_cfg.get("variants", [])):
            variant_name = variant["name"]
            observation_noise_std = float(variant.get("observation_noise_std", 0.0))
            variant_dir = output_dir / method_name / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            seed = int(run_cfg.get("seed", 120000)) + method_ix * 1000 + variant_ix * 100
            hydra_cfg.seed = seed
            set_global_seed(seed)
            if hasattr(policy, "reset"):
                policy.reset()

            env = make_env(hydra_cfg)
            try:
                info = eval_policy(
                    env,
                    policy,
                    policy,
                    n_episodes=hydra_cfg.eval.n_episodes,
                    max_episodes_rendered=0,
                    videos_dir=variant_dir / "videos",
                    start_seed=seed,
                    enable_progbar=False,
                    enable_inner_progbar=False,
                    sampler=variant.get("sampler", "direct"),
                    temperature=float(policy_cfg.get("temperature", hydra_cfg.policy.bet_softmax_temperature)),
                    noise_level=float(variant.get("action_noise_std", 0.0)),
                    observation_noise_std=observation_noise_std,
                    pass_at_ks=pass_at_ks,
                    pass_at_group_size=pass_at_group_size,
                )
            finally:
                env.close()

            metadata = git_metadata(Path.cwd())
            info["run_metadata"] = {
                "sampler": variant.get("sampler", "direct"),
                "n_episodes": hydra_cfg.eval.n_episodes,
                "seed": seed,
                "command": command_line(),
                "git_sha": metadata["git_sha"],
                "git_dirty": metadata["git_dirty"],
                "git_branch": metadata["git_branch"],
                "device": str(device),
                "gpu_id": current_gpu_id(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "output_path": str(variant_dir),
                "checkpoint": {
                    "main_policy_path": str(checkpoint_path),
                    "main_policy_hashes": checkpoint_hashes(checkpoint_path),
                    "reference_policy_path": None,
                    "reference_policy_hashes": [],
                },
                "temperature": float(policy_cfg.get("temperature", hydra_cfg.policy.bet_softmax_temperature)),
                "observation_noise_std": observation_noise_std,
                "pass_at_k": list(pass_at_ks),
                "pass_at_group_size": pass_at_group_size,
                "stress_variant": variant,
                "action_identity": (
                    "continuous env action from trained VQ-BeT; PPO update action identity is RVQ code ids"
                ),
            }
            write_json(variant_dir / "eval_info.json", info)
            write_json(variant_dir / "seed_manifest.json", [row["seed"] for row in info["per_episode"]])
            validation = validate_artifact(variant_dir, profile="benchmark")
            if not validation.ok:
                raise RuntimeError(f"Stress eval artifact rejected for {method_name}/{variant_name}: {validation.errors}")
            successes = [bool(row["success"]) for row in info["per_episode"]]
            max_rewards = [float(row["max_reward"]) for row in info["per_episode"]]
            rows.append(
                {
                    "method": method_name,
                    "variant": variant_name,
                    "path": str(variant_dir),
                    "n_episodes": hydra_cfg.eval.n_episodes,
                    "success_rate": float(sum(successes) / len(successes)),
                    "avg_max_reward": float(info["aggregated"]["avg_max_reward"]),
                    "avg_sum_reward": float(info["aggregated"]["avg_sum_reward"]),
                    "avg_num_steps": float(info["aggregated"]["avg_num_steps"]),
                    "pass_at_k": info["aggregated"]["pass_at_k"],
                    "pass_at_k_requested": info["aggregated"].get("pass_at_k_requested", list(pass_at_ks)),
                    "pass_at_k_note": info["aggregated"].get("pass_at_k_note"),
                    "success_ci95": mean_confidence_interval(successes),
                    "max_reward_ci95": mean_confidence_interval(max_rewards),
                    "validation_status": "passed",
                }
            )

    summary = {
        "artifact_kind": "polyppo_eval_summary",
        "output_path": str(output_dir),
        "checkpoint_count": len(checkpoints),
        "variant_count": len(stress_cfg.get("variants", [])),
        "rows": rows,
    }
    write_json(output_dir / "eval_polyppo_summary.json", summary)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
