#!/usr/bin/env python

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.common.policies.polyppo_scaffold import assign_set_advantages, pairwise_l1_diversity
from lerobot.common.polyppo.metrics import summarize_rollout_attempts
from lerobot.common.polyppo.storage import save_rollout_artifact
from lerobot.common.polyppo.utils import (
    checkpoint_hashes,
    command_line,
    config_to_dict,
    current_gpu_id,
    env_override,
    git_metadata,
    load_config,
    now_run_id,
    select_device,
    stable_hash,
)


class ObservationHistory:
    def __init__(self, n_obs_steps: int):
        self.n_obs_steps = n_obs_steps
        self._items: dict[str, deque[torch.Tensor]] = {}

    def seed(self, observation: dict[str, torch.Tensor]) -> None:
        self._items = {key: deque(maxlen=self.n_obs_steps) for key in observation}
        for _ in range(self.n_obs_steps):
            self.append(observation)

    def append(self, observation: dict[str, torch.Tensor]) -> None:
        for key, value in observation.items():
            self._items.setdefault(key, deque(maxlen=self.n_obs_steps)).append(value.detach().cpu().clone())

    def clone(self) -> "ObservationHistory":
        other = ObservationHistory(self.n_obs_steps)
        other._items = {
            key: deque([value.clone() for value in values], maxlen=self.n_obs_steps)
            for key, values in self._items.items()
        }
        return other

    def batch(self) -> dict[str, torch.Tensor]:
        return {key: torch.stack(list(values), dim=1) for key, values in self._items.items()}


def collect_polyppo_rollouts(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg_dict = config_to_dict(cfg)
    run_cfg = cfg_dict.get("run", {})
    rollout_cfg = cfg_dict.get("rollout", {})
    output_dir = Path(run_cfg.get("output_dir", "outputs/polyppo/rollout_smoke"))
    run_id = run_cfg.get("id") or now_run_id("polyppo_rollout")
    device = select_device(run_cfg.get("device"), run_cfg.get("gpu_id"))

    start = time.time()
    if rollout_cfg.get("mock", False):
        payload, tensors = _collect_mock_rollouts(cfg_dict, run_id, output_dir, device)
    else:
        payload, tensors = _collect_pusht_rollouts(cfg_dict, run_id, output_dir, device)

    payload["wall_time_s"] = time.time() - start
    return save_rollout_artifact(
        output_dir,
        payload,
        tensors,
        validate_profile=run_cfg.get("validate_profile", "smoke"),
    )


def _base_payload(cfg: dict[str, Any], run_id: str, output_dir: Path, device: torch.device) -> dict[str, Any]:
    run_cfg = cfg.get("run", {})
    policy_cfg = cfg.get("policy", {})
    seed = int(run_cfg.get("seed", 100000))
    metadata = git_metadata(Path.cwd())
    policy_path = policy_cfg.get("path")
    return {
        "artifact_kind": "polyppo_rollout",
        "run_id": run_id,
        "git_sha": metadata["git_sha"],
        "git_dirty": metadata["git_dirty"],
        "git_branch": metadata["git_branch"],
        "command": command_line(),
        "seed": seed,
        "seed_manifest": [],
        "output_path": str(output_dir),
        "gpu_id": current_gpu_id(device) if current_gpu_id(device) is not None else run_cfg.get("gpu_id"),
        "cuda_visible_devices": env_override("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "checkpoint": {
            "main_policy_path": policy_path,
            "main_policy_hashes": checkpoint_hashes(policy_path),
            "reference_policy_path": None,
            "reference_policy_hashes": [],
        },
        "config": cfg,
        "action_identity": "RVQ code-id sequence",
    }


def _collect_mock_rollouts(
    cfg: dict[str, Any], run_id: str, output_dir: Path, device: torch.device
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    del device
    run_cfg = cfg.get("run", {})
    rollout_cfg = cfg.get("rollout", {})
    seed = int(run_cfg.get("seed", 100000))
    generator = torch.Generator().manual_seed(seed)
    n_sets = int(rollout_cfg.get("n_prefixes", 4))
    n_attempts = int(rollout_cfg.get("attempts_per_prefix", 3))
    horizon = int(rollout_cfg.get("continuation_horizon", 4))
    n_obs_steps = int(rollout_cfg.get("n_obs_steps", 2))
    action_dim = int(rollout_cfg.get("action_dim", 2))
    action_chunk = int(rollout_cfg.get("action_chunk_size", 2))
    n_code_layers = int(rollout_cfg.get("n_code_layers", 2))
    n_codes = int(rollout_cfg.get("n_codes", 8))

    rewards = torch.randn(n_sets, n_attempts, horizon, generator=generator) * 0.05
    rewards += torch.linspace(0.0, 1.0, n_attempts).view(1, n_attempts, 1) * 0.1
    final_returns = rewards.sum(dim=-1)
    successes = final_returns > final_returns.median()
    max_overlaps = torch.clamp(final_returns + 0.5, 0.0, 1.0)
    code_ids = torch.randint(0, n_codes, (n_sets, n_attempts, horizon, n_code_layers), generator=generator)
    action_preds = torch.randn(n_sets, n_attempts, horizon, action_chunk, action_dim, generator=generator)
    old_log_probs = torch.randn(n_sets, n_attempts, horizon, generator=generator) * 0.1 - 1.0
    values = final_returns.unsqueeze(-1).expand_as(old_log_probs) * 0.5
    entropy = torch.ones_like(old_log_probs) * 0.7
    valid = torch.ones_like(old_log_probs, dtype=torch.bool)
    returns = final_returns.unsqueeze(-1).expand_as(old_log_probs)

    code_diversity = pairwise_l1_diversity(_masked_time_mean(code_ids.float(), valid))
    action_diversity = pairwise_l1_diversity(_masked_time_mean(action_preds.float(), valid))
    advantages = _assign_step_advantages(
        returns,
        code_ids=code_ids,
        action_preds=action_preds,
        diversity_kind=cfg.get("polyppo", {}).get("diversity_kind", "code"),
        poly_lambda=float(cfg.get("polyppo", {}).get("lambda_div", 0.1)),
        valid_mask=valid,
    )

    tensors = {
        "returns": returns,
        "rewards": rewards,
        "old_log_probs": old_log_probs,
        "values": values,
        "entropy": entropy,
        "code_ids": code_ids,
        "action_preds": action_preds,
        "valid": valid,
        "advantages": advantages,
        "successes": successes,
        "max_overlaps": max_overlaps,
        "code_diversity": code_diversity,
        "action_diversity": action_diversity,
    }
    payload = _base_payload(cfg, run_id, output_dir, torch.device("cpu"))
    payload.update(
        {
            "mock_policy": True,
            "n_sets": n_sets,
            "n_attempts": n_attempts,
            "horizon": horizon,
            "set_id": list(range(n_sets)),
            "attempt_id": [list(range(n_attempts)) for _ in range(n_sets)],
            "prefix_step": [int(rollout_cfg.get("prefix_steps", 2))] * n_sets,
            "prefix_env_state_hash": [f"mock-prefix-{ix}" for ix in range(n_sets)],
            "old_log_prob": old_log_probs.mean(dim=-1),
            "value": values.mean(dim=-1),
            "entropy": entropy.mean(dim=-1),
            "return": final_returns,
            "advantage": advantages.mean(dim=-1),
            "reward": rewards,
            "done": torch.ones_like(valid, dtype=torch.bool),
            "success": successes,
            "max_overlap": max_overlaps,
            "episode_length": torch.full((n_sets, n_attempts), horizon, dtype=torch.long),
            "action_summary": {
                "mean": float(action_preds.float().mean().item()),
                "std": float(action_preds.float().std(unbiased=False).item()),
            },
            "observation_summary": {
                "state_shape": [n_sets, n_attempts, horizon, n_obs_steps, 2],
                "image_shape": [n_sets, n_attempts, horizon, n_obs_steps, 3, 96, 96],
            },
            "code_ids": code_ids,
            "diversity_score": code_diversity,
            "code_diversity": code_diversity,
            "action_diversity": action_diversity,
            "poly_return": final_returns + float(cfg.get("polyppo", {}).get("lambda_div", 0.1)) * code_diversity,
            "poly_lambda": float(cfg.get("polyppo", {}).get("lambda_div", 0.1)),
            "seed_manifest": [seed + ix for ix in range(n_sets)],
            "aggregated": summarize_rollout_attempts(
                rewards=rewards,
                successes=successes,
                max_overlaps=max_overlaps,
                code_diversity=code_diversity,
                action_diversity=action_diversity,
            ),
        }
    )
    return payload, tensors


def _collect_pusht_rollouts(
    cfg: dict[str, Any], run_id: str, output_dir: Path, device: torch.device
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    from lerobot.common.envs.pusht_state import restore_pusht_state, snapshot_pusht_state
    from lerobot.common.envs.utils import preprocess_observation
    from lerobot.scripts.eval import get_pretrained_policy_path, load_pretrained_policy_hydra_config
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.utils.utils import set_global_seed

    import gymnasium as gym
    import gym_pusht  # noqa: F401

    run_cfg = cfg.get("run", {})
    env_cfg = cfg.get("env", {})
    policy_cfg = cfg.get("policy", {})
    rollout_cfg = cfg.get("rollout", {})
    policy_path = policy_cfg.get("path")
    if not policy_path:
        raise ValueError("policy.path is required for non-mock PolyPPO rollout collection.")

    seed = int(run_cfg.get("seed", 100000))
    set_global_seed(seed)
    pretrained_path = get_pretrained_policy_path(policy_path)
    hydra_cfg = load_pretrained_policy_hydra_config(pretrained_path, [])
    hydra_cfg.device = str(device)
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_path))
    policy.to(device)
    policy.eval()

    gym_kwargs = dict(hydra_cfg.env.get("gym", {}))
    if env_cfg.get("episode_length", hydra_cfg.env.get("episode_length")):
        gym_kwargs["max_episode_steps"] = int(env_cfg.get("episode_length", hydra_cfg.env.episode_length))
    env = gym.make(f"gym_{hydra_cfg.env.name}/{hydra_cfg.env.task}", disable_env_checker=True, **gym_kwargs)

    n_sets = int(rollout_cfg.get("n_prefixes", 4))
    n_attempts = int(rollout_cfg.get("attempts_per_prefix", 3))
    horizon = int(rollout_cfg.get("continuation_horizon", 8))
    prefix_steps = int(rollout_cfg.get("prefix_steps", hydra_cfg.policy.n_obs_steps))
    temperature = float(policy_cfg.get("temperature", hydra_cfg.policy.get("bet_softmax_temperature", 1.0)))

    rewards = torch.zeros(n_sets, n_attempts, horizon)
    old_log_probs = torch.zeros(n_sets, n_attempts, horizon)
    values = torch.zeros(n_sets, n_attempts, horizon)
    entropy = torch.zeros(n_sets, n_attempts, horizon)
    valid = torch.zeros(n_sets, n_attempts, horizon, dtype=torch.bool)
    successes = torch.zeros(n_sets, n_attempts, dtype=torch.bool)
    max_overlaps = torch.zeros(n_sets, n_attempts)
    episode_lengths = torch.zeros(n_sets, n_attempts, dtype=torch.long)
    obs_states: list[list[list[torch.Tensor]]] = [[[] for _ in range(n_attempts)] for _ in range(n_sets)]
    obs_images: list[list[list[torch.Tensor]]] = [[[] for _ in range(n_attempts)] for _ in range(n_sets)]
    code_ids: list[list[list[torch.Tensor]]] = [[[] for _ in range(n_attempts)] for _ in range(n_sets)]
    action_preds: list[list[list[torch.Tensor]]] = [[[] for _ in range(n_attempts)] for _ in range(n_sets)]
    prefix_hashes: list[str] = []
    seed_manifest: list[int] = []

    try:
        for set_ix in range(n_sets):
            episode_seed = seed + set_ix
            seed_manifest.append(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            history = ObservationHistory(hydra_cfg.policy.n_obs_steps)
            history.seed(_preprocess_single_observation(preprocess_observation, obs))

            done = False
            for _ in range(prefix_steps):
                action, _, _, _, _ = _sample_code_action(policy, history.batch(), device, temperature)
                obs, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                history.append(_preprocess_single_observation(preprocess_observation, obs))
                if done:
                    break

            prefix_state = snapshot_pusht_state(env)
            prefix_history = history.clone()
            prefix_hashes.append(stable_hash(prefix_state))

            for attempt_ix in range(n_attempts):
                restore_pusht_state(env, prefix_state)
                history = prefix_history.clone()
                max_coverage = 0.0
                for step_ix in range(horizon):
                    batch = history.batch()
                    obs_states[set_ix][attempt_ix].append(batch["observation.state"].squeeze(0).clone())
                    image_key = "observation.image" if "observation.image" in batch else next(
                        key for key in batch if key.startswith("observation.image")
                    )
                    obs_images[set_ix][attempt_ix].append(batch[image_key].squeeze(0).clone())
                    action, out, action_chunk, log_prob, value = _sample_code_action(
                        policy, batch, device, temperature
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    history.append(_preprocess_single_observation(preprocess_observation, obs))

                    rewards[set_ix, attempt_ix, step_ix] = float(reward)
                    old_log_probs[set_ix, attempt_ix, step_ix] = float(log_prob.item())
                    values[set_ix, attempt_ix, step_ix] = float(value.item())
                    entropy[set_ix, attempt_ix, step_ix] = float(out["entropy"].detach().cpu().item())
                    valid[set_ix, attempt_ix, step_ix] = True
                    code_ids[set_ix][attempt_ix].append(out["code_ids"].detach().cpu().squeeze(0))
                    action_preds[set_ix][attempt_ix].append(action_chunk.detach().cpu().squeeze(0))
                    max_coverage = max(max_coverage, float(info.get("coverage", 0.0)))
                    if bool(info.get("is_success", False)):
                        successes[set_ix, attempt_ix] = True
                    episode_lengths[set_ix, attempt_ix] = step_ix + 1
                    if done:
                        break
                max_overlaps[set_ix, attempt_ix] = max_coverage
    finally:
        env.close()

    code_tensor = _pad_nested_tensors(code_ids, (n_sets, n_attempts, horizon))
    action_tensor = _pad_nested_tensors(action_preds, (n_sets, n_attempts, horizon))
    obs_state_tensor = _pad_nested_tensors(obs_states, (n_sets, n_attempts, horizon))
    obs_image_tensor = _pad_nested_tensors(obs_images, (n_sets, n_attempts, horizon))
    returns = rewards.sum(dim=-1, keepdim=True).expand_as(rewards)
    code_diversity = pairwise_l1_diversity(_masked_time_mean(code_tensor.float(), valid))
    action_diversity = pairwise_l1_diversity(_masked_time_mean(action_tensor.float(), valid))
    advantages = _assign_step_advantages(
        returns,
        code_ids=code_tensor,
        action_preds=action_tensor,
        diversity_kind=cfg.get("polyppo", {}).get("diversity_kind", "code"),
        poly_lambda=float(cfg.get("polyppo", {}).get("lambda_div", 0.1)),
        valid_mask=valid,
    )
    tensors = {
        "returns": returns,
        "rewards": rewards,
        "old_log_probs": old_log_probs,
        "values": values,
        "entropy": entropy,
        "code_ids": code_tensor.long(),
        "action_preds": action_tensor,
        "valid": valid,
        "advantages": advantages,
        "successes": successes,
        "max_overlaps": max_overlaps,
        "episode_lengths": episode_lengths,
        "obs_state": obs_state_tensor,
        "obs_image": obs_image_tensor,
        "code_diversity": code_diversity,
        "action_diversity": action_diversity,
    }
    final_returns = rewards.sum(dim=-1)
    payload = _base_payload(cfg, run_id, output_dir, device)
    payload["checkpoint"]["main_policy_path"] = str(pretrained_path)
    payload["checkpoint"]["main_policy_hashes"] = checkpoint_hashes(pretrained_path)
    payload.update(
        {
            "mock_policy": False,
            "n_sets": n_sets,
            "n_attempts": n_attempts,
            "horizon": horizon,
            "set_id": list(range(n_sets)),
            "attempt_id": [list(range(n_attempts)) for _ in range(n_sets)],
            "prefix_step": [prefix_steps] * n_sets,
            "prefix_env_state_hash": prefix_hashes,
            "old_log_prob": old_log_probs.mean(dim=-1),
            "value": values.mean(dim=-1),
            "entropy": entropy.mean(dim=-1),
            "return": final_returns,
            "advantage": advantages.mean(dim=-1),
            "reward": rewards,
            "done": ~valid,
            "success": successes,
            "max_overlap": max_overlaps,
            "episode_length": episode_lengths,
            "action_summary": {
                "mean": float(action_tensor.float().mean().item()),
                "std": float(action_tensor.float().std(unbiased=False).item()),
            },
            "observation_summary": {
                "state_shape": list(obs_state_tensor.shape),
                "image_shape": list(obs_image_tensor.shape),
            },
            "code_ids": code_tensor.long(),
            "diversity_score": code_diversity,
            "code_diversity": code_diversity,
            "action_diversity": action_diversity,
            "poly_return": final_returns + float(cfg.get("polyppo", {}).get("lambda_div", 0.1)) * code_diversity,
            "poly_lambda": float(cfg.get("polyppo", {}).get("lambda_div", 0.1)),
            "seed_manifest": seed_manifest,
            "aggregated": summarize_rollout_attempts(
                rewards=rewards,
                successes=successes,
                max_overlaps=max_overlaps,
                code_diversity=code_diversity,
                action_diversity=action_diversity,
            ),
        }
    )
    return payload, tensors


def _preprocess_single_observation(preprocess_observation, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
    batched = {key: np.expand_dims(value, axis=0) for key, value in observation.items()}
    return preprocess_observation(batched)


def _assign_step_advantages(
    returns: torch.Tensor,
    *,
    code_ids: torch.Tensor,
    action_preds: torch.Tensor,
    diversity_kind: str,
    poly_lambda: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if diversity_kind == "none" or poly_lambda == 0.0:
        return assign_set_advantages(returns, normalize=True, valid_mask=valid_mask)
    if diversity_kind == "code":
        diversity = pairwise_l1_diversity(_masked_time_mean(code_ids.float(), valid_mask))
    elif diversity_kind == "action":
        diversity = pairwise_l1_diversity(_masked_time_mean(action_preds.float(), valid_mask))
    else:
        raise ValueError("diversity_kind must be one of {'none', 'code', 'action'}.")
    return assign_set_advantages(
        returns,
        diversity=diversity.unsqueeze(-1).expand_as(returns),
        poly_lambda=poly_lambda,
        valid_mask=valid_mask,
    )


def _masked_time_mean(values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return values.mean(dim=2)
    expand_shape = (*valid_mask.shape, *([1] * (values.ndim - valid_mask.ndim)))
    valid = valid_mask.reshape(expand_shape).to(dtype=values.dtype, device=values.device)
    count = valid.sum(dim=2).clamp_min(1.0)
    return (values * valid).sum(dim=2) / count


def _sample_code_action(policy, batch: dict[str, torch.Tensor], device: torch.device, temperature: float):
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        out = policy.evaluate_code_actions(batch, code_ids=None, temperature=temperature)
        action_chunk = policy.unnormalize_outputs({"action": out["action_pred"]})["action"]
    action = action_chunk[:, 0, :].detach().cpu().numpy()[0]
    return action, out, action_chunk, out["log_prob"], out["value"]


def _pad_nested_tensors(values: list[list[list[torch.Tensor]]], shape_prefix: tuple[int, int, int]) -> torch.Tensor:
    exemplar = None
    for set_values in values:
        for attempt_values in set_values:
            if attempt_values:
                exemplar = attempt_values[0]
                break
        if exemplar is not None:
            break
    if exemplar is None:
        raise ValueError("Cannot pad empty tensor collection.")
    output = torch.zeros(*shape_prefix, *exemplar.shape, dtype=exemplar.dtype)
    for set_ix, set_values in enumerate(values):
        for attempt_ix, attempt_values in enumerate(set_values):
            for step_ix, value in enumerate(attempt_values[: shape_prefix[2]]):
                output[set_ix, attempt_ix, step_ix] = value
    return output
