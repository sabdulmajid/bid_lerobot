#!/usr/bin/env python

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.common.polyppo.distill import (
    _changed_parameters,
    _code_head_modules,
    _grad_norms,
    _train_mock_distill,
)
from lerobot.common.polyppo.rollout import (
    ObservationHistory,
    _preprocess_single_observation,
    _sample_code_action,
)
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
    write_json,
)


def collect_polydistill_v2_dataset(config_path: str | Path) -> dict[str, Any]:
    cfg = config_to_dict(load_config(config_path))
    run_cfg = cfg.get("run", {})
    output_dir = Path(run_cfg.get("output_dir", "outputs/polyppo/polydistill_v2_dataset"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_cfg.get("id") or now_run_id("polydistill_v2_collect")
    device = select_device(run_cfg.get("device"), run_cfg.get("gpu_id"))
    start = time.time()
    if cfg.get("distill", {}).get("mock", False):
        payload, tensors = _collect_mock_v2_dataset(cfg, run_id, output_dir, device)
    else:
        payload, tensors = _collect_pusht_v2_dataset(cfg, run_id, output_dir, device)
    payload["wall_time_s"] = time.time() - start
    return _write_v2_dataset_artifact(output_dir, payload, tensors)


def train_polydistill_v2(config_path: str | Path) -> dict[str, Any]:
    cfg = config_to_dict(load_config(config_path))
    run_cfg = cfg.get("run", {})
    train_cfg = cfg.get("train", {})
    output_dir = Path(run_cfg.get("output_dir", "outputs/polyppo/polydistill_v2_train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_cfg.get("id") or now_run_id("polydistill_v2_train")
    device = select_device(run_cfg.get("device"), run_cfg.get("gpu_id"))
    dataset_path = train_cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("train.dataset_path is required.")
    payload, tensors = load_polydistill_v2_dataset(dataset_path)
    if payload.get("artifact_kind") not in {"polydistill_v2_dataset", "polydistill_dataset"}:
        raise ValueError(f"Unsupported PolyDistill artifact: {payload.get('artifact_kind')}.")
    if payload.get("validation_status") != "passed":
        raise RuntimeError(f"Refusing invalid PolyDistill dataset: {payload.get('validation_errors', [])}")
    if payload.get("mock_policy", False) or train_cfg.get("mock_policy", False):
        result = _train_mock_distill(cfg, payload, tensors, output_dir, run_id, device)
    else:
        result = _train_vqbet_v2_distill(cfg, payload, tensors, output_dir, run_id, device)
    write_json(output_dir / "polydistill_v2_train_info.json", result)
    return result


def load_polydistill_v2_dataset(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    path = Path(path)
    payload_path = path if path.is_file() else path / "polydistill_v2_dataset.json"
    if not payload_path.exists():
        payload_path = path / "polydistill_dataset.json"
    artifact_dir = payload_path.parent
    import json

    payload = json.loads(payload_path.read_text())
    tensors = torch.load(artifact_dir / payload["tensor_path"], map_location="cpu")
    return payload, tensors


def rank_attempt_indices(
    successes: torch.Tensor,
    max_overlaps: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    if not (successes.shape == max_overlaps.shape == returns.shape):
        raise ValueError("successes, max_overlaps, and returns must share shape.")
    score = max_overlaps.float() + returns.float() * 1e-6 + successes.float() * 1_000.0
    return score.argsort(dim=1, descending=True)


def attempt_weights(
    successes: torch.Tensor,
    max_overlaps: torch.Tensor,
    returns: torch.Tensor,
    *,
    objective: str,
    top_m: int,
    tau: float,
) -> torch.Tensor:
    """Return per-start/per-ranked-attempt weights for PolyDistill v2 objectives."""
    if top_m <= 0:
        raise ValueError("top_m must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if not (successes.shape == max_overlaps.shape == returns.shape):
        raise ValueError("successes, max_overlaps, and returns must share shape.")
    n_starts, n_kept = successes.shape
    m = min(top_m, n_kept)
    weights = torch.zeros(n_starts, n_kept, dtype=torch.float32)
    if objective == "top1_ce":
        weights[:, 0] = 1.0
        return weights

    score = max_overlaps.float() + returns.float() * 1e-6
    if objective == "topm_soft_ce":
        weights[:, :m] = torch.softmax(score[:, :m] / tau, dim=1)
        return weights

    if objective == "success_filtered_soft_ce":
        for start_ix in range(n_starts):
            mask = successes[start_ix, :m].bool()
            if not mask.any():
                mask = torch.zeros(m, dtype=torch.bool)
                mask[0] = True
            values = score[start_ix, :m][mask]
            weights[start_ix, :m][mask] = torch.softmax(values / tau, dim=0)
        return weights

    if objective == "adv_weighted_bc":
        values = score[:, :m]
        centered = values - values.mean(dim=1, keepdim=True)
        scaled = centered / values.std(dim=1, keepdim=True).clamp_min(1e-6)
        weights[:, :m] = torch.softmax(scaled / tau, dim=1)
        return weights

    raise ValueError(f"Unsupported PolyDistill v2 objective: {objective}")


def flatten_weighted_candidate_steps(
    obs_state: torch.Tensor,
    obs_image: torch.Tensor,
    code_ids: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if valid.shape != code_ids.shape[:3]:
        raise ValueError("valid must match candidate code-id leading shape.")
    if weights.shape != valid.shape[:2]:
        raise ValueError("weights must have shape (n_starts, n_kept_attempts).")
    flat_valid = valid.reshape(-1).bool()
    step_weights = weights[:, :, None].expand_as(valid).reshape(-1)[flat_valid].float()
    batch = {
        "observation.state": obs_state.reshape(-1, *obs_state.shape[-2:])[flat_valid].float(),
        "observation.image": obs_image.reshape(-1, *obs_image.shape[-4:])[flat_valid].float(),
    }
    return batch, code_ids.reshape(-1, code_ids.shape[-1])[flat_valid].long(), step_weights


def _collect_pusht_v2_dataset(
    cfg: dict[str, Any], run_id: str, output_dir: Path, device: torch.device
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    from lerobot.common.envs.pusht_state import restore_pusht_state, snapshot_pusht_state
    from lerobot.common.envs.utils import preprocess_observation
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.utils.utils import set_global_seed
    from lerobot.scripts.eval import get_pretrained_policy_path, load_pretrained_policy_hydra_config

    import gymnasium as gym
    import gym_pusht  # noqa: F401

    run_cfg = cfg.get("run", {})
    policy_cfg = cfg.get("policy", {})
    distill_cfg = cfg.get("distill", {})
    env_cfg = cfg.get("env", {})
    policy_path = policy_cfg.get("path")
    if not policy_path:
        raise ValueError("policy.path is required for PolyDistill v2 collection.")

    seed = int(run_cfg.get("seed", 210000))
    set_global_seed(seed)
    pretrained_path = get_pretrained_policy_path(policy_path)
    hydra_cfg = load_pretrained_policy_hydra_config(pretrained_path, [])
    hydra_cfg.device = str(device)
    hydra_cfg.use_amp = False
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_path))
    policy.to(device)
    policy.eval()

    max_steps = int(distill_cfg.get("max_steps", env_cfg.get("episode_length", hydra_cfg.env.episode_length)))
    n_starts = int(distill_cfg.get("n_start_states", 8))
    attempts = int(distill_cfg.get("attempts_per_start", 16))
    keep_top = int(distill_cfg.get("keep_top_attempts", min(4, attempts)))
    temperature = float(policy_cfg.get("temperature", hydra_cfg.policy.bet_softmax_temperature))
    if n_starts <= 0 or attempts <= 0 or keep_top <= 0:
        raise ValueError("n_start_states, attempts_per_start, and keep_top_attempts must be positive.")
    keep_top = min(keep_top, attempts)

    gym_kwargs = dict(hydra_cfg.env.get("gym", {}))
    gym_kwargs["max_episode_steps"] = max_steps
    env = gym.make(f"gym_{hydra_cfg.env.name}/{hydra_cfg.env.task}", disable_env_checker=True, **gym_kwargs)

    kept_traces_by_start: list[list[dict[str, list[torch.Tensor]]]] = []
    light_traces_by_start: list[list[dict[str, list[torch.Tensor]]]] = []
    successes = torch.zeros(n_starts, attempts, dtype=torch.bool)
    max_overlaps = torch.zeros(n_starts, attempts)
    returns = torch.zeros(n_starts, attempts)
    episode_lengths = torch.zeros(n_starts, attempts, dtype=torch.long)
    ranked_attempts = torch.zeros(n_starts, keep_top, dtype=torch.long)
    start_hashes: list[str] = []
    start_records: list[dict[str, Any]] = []
    seed_manifest: list[int] = []
    try:
        for start_ix in range(n_starts):
            start_seed = seed + start_ix
            seed_manifest.append(start_seed)
            obs, _ = env.reset(seed=start_seed)
            history = ObservationHistory(hydra_cfg.policy.n_obs_steps)
            history.seed(_preprocess_single_observation(preprocess_observation, obs))
            start_state = snapshot_pusht_state(env)
            start_history = history.clone()
            start_hash = stable_hash(start_state)
            start_hashes.append(start_hash)
            full_traces = []
            light_traces = []
            attempts_json = []
            for attempt_ix in range(attempts):
                restore_pusht_state(env, start_state)
                history = start_history.clone()
                if hasattr(policy, "reset"):
                    policy.reset()
                full_trace = {"obs_state": [], "obs_image": [], "code_ids": [], "action_chunks": [], "rewards": [], "coverages": []}
                light_trace = {"code_ids": [], "action_chunks": [], "rewards": [], "coverages": []}
                total_return = 0.0
                max_coverage = 0.0
                success = False
                steps = 0
                for step_ix in range(max_steps):
                    batch = history.batch()
                    full_trace["obs_state"].append(batch["observation.state"].squeeze(0).detach().cpu().clone())
                    image_key = "observation.image" if "observation.image" in batch else next(
                        key for key in batch if key.startswith("observation.image")
                    )
                    full_trace["obs_image"].append(batch[image_key].squeeze(0).detach().cpu().clone())
                    action, out, action_chunk, _, _ = _sample_code_action(policy, batch, device, temperature)
                    code_ids = out["code_ids"].squeeze(0).detach().cpu().clone()
                    action_chunk_cpu = action_chunk.squeeze(0).detach().cpu().clone()
                    full_trace["code_ids"].append(code_ids)
                    full_trace["action_chunks"].append(action_chunk_cpu)
                    light_trace["code_ids"].append(code_ids)
                    light_trace["action_chunks"].append(action_chunk_cpu)
                    obs, reward, terminated, truncated, info = env.step(action)
                    history.append(_preprocess_single_observation(preprocess_observation, obs))
                    reward_value = float(reward)
                    coverage_value = float(info.get("coverage", 0.0))
                    full_trace["rewards"].append(torch.tensor(reward_value, dtype=torch.float32))
                    full_trace["coverages"].append(torch.tensor(coverage_value, dtype=torch.float32))
                    light_trace["rewards"].append(torch.tensor(reward_value, dtype=torch.float32))
                    light_trace["coverages"].append(torch.tensor(coverage_value, dtype=torch.float32))
                    total_return += reward_value
                    max_coverage = max(max_coverage, coverage_value)
                    success = success or bool(info.get("is_success", False))
                    steps = step_ix + 1
                    if bool(terminated or truncated):
                        break
                successes[start_ix, attempt_ix] = success
                max_overlaps[start_ix, attempt_ix] = max_coverage
                returns[start_ix, attempt_ix] = total_return
                episode_lengths[start_ix, attempt_ix] = steps
                full_traces.append(full_trace)
                light_traces.append(light_trace)
                attempts_json.append(
                    {
                        "attempt_ix": attempt_ix,
                        "success": success,
                        "max_overlap": max_coverage,
                        "sum_reward": total_return,
                        "num_steps": steps,
                    }
                )
            ranked = rank_attempt_indices(
                successes[start_ix : start_ix + 1],
                max_overlaps[start_ix : start_ix + 1],
                returns[start_ix : start_ix + 1],
            )[0, :keep_top]
            ranked_attempts[start_ix] = ranked
            kept_traces_by_start.append([full_traces[int(ix)] for ix in ranked.tolist()])
            light_traces_by_start.append(light_traces)
            start_records.append(
                {
                    "start_ix": start_ix,
                    "start_seed": start_seed,
                    "start_state_hash": start_hash,
                    "ranked_attempt_ixs": ranked.tolist(),
                    "attempts": attempts_json,
                }
            )
    finally:
        env.close()

    storage_dtype = str(distill_cfg.get("storage_dtype", "float16"))
    kept = _pad_ranked_traces(kept_traces_by_start, max_steps, storage_dtype=storage_dtype)
    all_light = _pad_light_traces(light_traces_by_start, max_steps)
    selected_attempts = ranked_attempts[:, 0]
    selected_successes = successes[torch.arange(n_starts), selected_attempts]
    selected_overlaps = max_overlaps[torch.arange(n_starts), selected_attempts]
    selected_returns = returns[torch.arange(n_starts), selected_attempts]
    kept_successes = successes.gather(1, ranked_attempts)
    kept_overlaps = max_overlaps.gather(1, ranked_attempts)
    kept_returns = returns.gather(1, ranked_attempts)
    tensors = {
        **kept,
        **all_light,
        "selected_obs_state": kept["candidate_obs_state"][:, 0],
        "selected_obs_image": kept["candidate_obs_image"][:, 0],
        "selected_code_ids": kept["candidate_code_ids"][:, 0],
        "selected_valid": kept["candidate_valid"][:, 0],
        "attempt_successes": successes,
        "attempt_max_overlaps": max_overlaps,
        "attempt_returns": returns,
        "attempt_episode_lengths": episode_lengths,
        "ranked_attempts": ranked_attempts,
        "kept_successes": kept_successes,
        "kept_max_overlaps": kept_overlaps,
        "kept_returns": kept_returns,
        "selected_attempts": selected_attempts,
    }
    metadata = git_metadata(Path.cwd())
    payload = {
        "artifact_kind": "polydistill_v2_dataset",
        "run_id": run_id,
        "output_path": str(output_dir),
        "mock_policy": False,
        "git_sha": metadata["git_sha"],
        "git_dirty": metadata["git_dirty"],
        "git_branch": metadata["git_branch"],
        "command": command_line(),
        "seed": seed,
        "seed_manifest": seed_manifest,
        "device": str(device),
        "gpu_id": current_gpu_id(device),
        "cuda_visible_devices": env_override("CUDA_VISIBLE_DEVICES"),
        "config": cfg,
        "action_identity": "RVQ code-id sequence; continuous offsets/actions remain deterministic policy outputs",
        "checkpoint": {
            "main_policy_path": str(pretrained_path),
            "main_policy_hashes": checkpoint_hashes(pretrained_path),
            "reference_policy_path": None,
            "reference_policy_hashes": [],
        },
        "n_start_states": n_starts,
        "attempts_per_start": attempts,
        "keep_top_attempts": keep_top,
        "max_steps": max_steps,
        "storage_policy": {
            "all_attempts": "scalar metrics, code ids, action chunks, rewards, coverages, validity mask",
            "kept_attempts": "full observation histories plus code/action/reward traces",
            "storage_dtype": storage_dtype,
        },
        "start_state_hashes": start_hashes,
        "starts": start_records,
        "aggregated": {
            "candidate_success_rate": float(successes.float().mean().item()),
            "selected_success_rate": float(selected_successes.float().mean().item()),
            "selected_avg_max_overlap": float(selected_overlaps.float().mean().item()),
            "selected_avg_return": float(selected_returns.float().mean().item()),
            "kept_success_rate": float(kept_successes.float().mean().item()),
            "kept_avg_max_overlap": float(kept_overlaps.float().mean().item()),
            "kept_valid_steps": int(kept["candidate_valid"].sum().item()),
        },
    }
    return payload, tensors


def _collect_mock_v2_dataset(
    cfg: dict[str, Any], run_id: str, output_dir: Path, device: torch.device
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    del output_dir, device
    distill_cfg = cfg.get("distill", {})
    n_starts = int(distill_cfg.get("n_start_states", 4))
    attempts = int(distill_cfg.get("attempts_per_start", 5))
    keep_top = min(int(distill_cfg.get("keep_top_attempts", 3)), attempts)
    max_steps = int(distill_cfg.get("max_steps", 6))
    generator = torch.Generator().manual_seed(int(cfg.get("run", {}).get("seed", 123)))
    successes = torch.zeros(n_starts, attempts, dtype=torch.bool)
    successes[:, -2:] = True
    max_overlaps = torch.rand(n_starts, attempts, generator=generator)
    returns = torch.rand(n_starts, attempts, generator=generator)
    ranked = rank_attempt_indices(successes, max_overlaps, returns)[:, :keep_top]
    kept_successes = successes.gather(1, ranked)
    kept_overlaps = max_overlaps.gather(1, ranked)
    kept_returns = returns.gather(1, ranked)
    tensors = {
        "candidate_obs_state": torch.randn(n_starts, keep_top, max_steps, 2, 2, generator=generator),
        "candidate_obs_image": torch.randn(n_starts, keep_top, max_steps, 2, 3, 16, 16, generator=generator),
        "candidate_code_ids": torch.randint(0, 8, (n_starts, keep_top, max_steps, 2), generator=generator),
        "candidate_action_chunks": torch.randn(n_starts, keep_top, max_steps, 2, 2, generator=generator),
        "candidate_rewards": torch.randn(n_starts, keep_top, max_steps, generator=generator),
        "candidate_coverages": torch.rand(n_starts, keep_top, max_steps, generator=generator),
        "candidate_valid": torch.ones(n_starts, keep_top, max_steps, dtype=torch.bool),
        "all_code_ids": torch.randint(0, 8, (n_starts, attempts, max_steps, 2), generator=generator),
        "all_action_chunks": torch.randn(n_starts, attempts, max_steps, 2, 2, generator=generator),
        "all_rewards": torch.randn(n_starts, attempts, max_steps, generator=generator),
        "all_coverages": torch.rand(n_starts, attempts, max_steps, generator=generator),
        "all_valid": torch.ones(n_starts, attempts, max_steps, dtype=torch.bool),
        "attempt_successes": successes,
        "attempt_max_overlaps": max_overlaps,
        "attempt_returns": returns,
        "attempt_episode_lengths": torch.full((n_starts, attempts), max_steps, dtype=torch.long),
        "ranked_attempts": ranked,
        "kept_successes": kept_successes,
        "kept_max_overlaps": kept_overlaps,
        "kept_returns": kept_returns,
        "selected_attempts": ranked[:, 0],
    }
    tensors["selected_obs_state"] = tensors["candidate_obs_state"][:, 0]
    tensors["selected_obs_image"] = tensors["candidate_obs_image"][:, 0]
    tensors["selected_code_ids"] = tensors["candidate_code_ids"][:, 0]
    tensors["selected_valid"] = tensors["candidate_valid"][:, 0]
    metadata = git_metadata(Path.cwd())
    payload = {
        "artifact_kind": "polydistill_v2_dataset",
        "run_id": run_id,
        "output_path": str(cfg.get("run", {}).get("output_dir", "outputs/polyppo/polydistill_v2_mock")),
        "mock_policy": True,
        "git_sha": metadata["git_sha"],
        "git_dirty": metadata["git_dirty"],
        "git_branch": metadata["git_branch"],
        "command": command_line(),
        "seed": int(cfg.get("run", {}).get("seed", 123)),
        "seed_manifest": list(range(n_starts)),
        "device": "cpu",
        "gpu_id": None,
        "cuda_visible_devices": env_override("CUDA_VISIBLE_DEVICES"),
        "config": cfg,
        "n_start_states": n_starts,
        "attempts_per_start": attempts,
        "keep_top_attempts": keep_top,
        "max_steps": max_steps,
        "aggregated": {
            "candidate_success_rate": float(successes.float().mean().item()),
            "selected_success_rate": float(successes.gather(1, ranked[:, :1]).float().mean().item()),
            "kept_success_rate": float(kept_successes.float().mean().item()),
            "kept_valid_steps": int(tensors["candidate_valid"].sum().item()),
        },
    }
    return payload, tensors


def _train_vqbet_v2_distill(
    cfg: dict[str, Any],
    dataset_payload: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    output_dir: Path,
    run_id: str,
    device: torch.device,
) -> dict[str, Any]:
    from lerobot.common.policies.factory import make_policy
    from lerobot.scripts.eval import get_pretrained_policy_path, load_pretrained_policy_hydra_config

    start = time.time()
    train_cfg = cfg.get("train", {})
    objective_cfg = cfg.get("objective", {})
    policy_path = train_cfg.get("policy_path_override") or dataset_payload["checkpoint"].get("main_policy_path")
    pretrained_path = get_pretrained_policy_path(policy_path)
    hydra_cfg = load_pretrained_policy_hydra_config(pretrained_path, [])
    hydra_cfg.device = str(device)
    hydra_cfg.use_amp = False
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_path))
    policy.to(device)
    policy.train()

    base_policy = None
    kl_coef = float(objective_cfg.get("kl_coef", train_cfg.get("kl_coef", 0.0)))
    if kl_coef > 0:
        base_policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_path))
        base_policy.to(device)
        base_policy.eval()
        for param in base_policy.parameters():
            param.requires_grad = False

    for param in policy.parameters():
        param.requires_grad = False
    trainable_modules = _code_head_modules(policy, hydra_cfg.policy.sequentially_select)
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True

    weights = attempt_weights(
        tensors["kept_successes"],
        tensors["kept_max_overlaps"],
        tensors["kept_returns"],
        objective=str(objective_cfg.get("name", "topm_soft_ce")),
        top_m=int(objective_cfg.get("top_m", train_cfg.get("top_m", 4))),
        tau=float(objective_cfg.get("tau", train_cfg.get("tau", 0.07))),
    )
    batch_all, code_ids_all, step_weights = flatten_weighted_candidate_steps(
        tensors["candidate_obs_state"],
        tensors["candidate_obs_image"],
        tensors["candidate_code_ids"],
        tensors["candidate_valid"],
        weights,
    )
    positive = step_weights > 0
    batch_all = {key: value[positive].to(device) for key, value in batch_all.items()}
    code_ids_all = code_ids_all[positive].to(device)
    step_weights = step_weights[positive].to(device)
    if code_ids_all.numel() == 0:
        raise ValueError("PolyDistill v2 dataset produced no positive-weight code steps.")

    optimizer = torch.optim.Adam(
        [param for param in policy.parameters() if param.requires_grad],
        lr=float(train_cfg.get("lr", 3e-5)),
    )
    epochs = int(train_cfg.get("epochs", 3))
    batch_size = int(train_cfg.get("batch_size", 64))
    temperature = float(cfg.get("policy", {}).get("temperature", hydra_cfg.policy.bet_softmax_temperature))
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("train.epochs and train.batch_size must be positive.")

    before = {name: param.detach().clone() for name, param in policy.named_parameters() if param.requires_grad}
    loss_history = []
    grad_norms: dict[str, float] = {}
    for epoch in range(epochs):
        order = torch.randperm(code_ids_all.shape[0], device=device)
        losses = []
        ce_losses = []
        anchor_losses = []
        for start_ix in range(0, order.numel(), batch_size):
            indices = order[start_ix : start_ix + batch_size]
            micro_batch = {key: value[indices] for key, value in batch_all.items()}
            micro_weights = step_weights[indices]
            out = policy.evaluate_code_actions(micro_batch, code_ids=code_ids_all[indices], temperature=temperature)
            denom = micro_weights.sum().clamp_min(1e-8)
            ce_loss = -(out["log_prob"] * micro_weights).sum() / denom
            anchor_loss = torch.zeros((), device=device)
            if base_policy is not None and kl_coef > 0:
                with torch.no_grad():
                    base_out = base_policy.evaluate_code_actions(
                        micro_batch, code_ids=code_ids_all[indices], temperature=temperature
                    )
                anchor_loss = (((out["log_prob"] - base_out["log_prob"]) ** 2) * micro_weights).sum() / denom
            loss = ce_loss + kl_coef * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            grad_norms = _grad_norms(policy)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            ce_losses.append(float(ce_loss.detach().cpu().item()))
            anchor_losses.append(float(anchor_loss.detach().cpu().item()))
        loss_history.append(
            {
                "epoch": epoch,
                "loss": float(sum(losses) / max(len(losses), 1)),
                "ce_loss": float(sum(ce_losses) / max(len(ce_losses), 1)),
                "anchor_loss": float(sum(anchor_losses) / max(len(anchor_losses), 1)),
                "grad_norm_max": float(max(grad_norms.values())) if grad_norms else 0.0,
            }
        )

    changed = _changed_parameters(before, policy)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"model_state": policy.state_dict(), "config": cfg}, checkpoint_path)
    metadata = git_metadata(Path.cwd())
    checks = {
        "checkpoint_saved": checkpoint_path.exists(),
        "loss_finite": all(torch.isfinite(torch.tensor(row["loss"])) for row in loss_history),
        "nonzero_gradients": bool(any(value > 0 for value in grad_norms.values())),
        "parameters_changed": bool(changed),
        "positive_weight_steps": int(step_weights.numel()) > 0,
    }
    return {
        "artifact_kind": "polydistill_v2_train",
        "run_id": run_id,
        "output_path": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hashes": checkpoint_hashes(checkpoint_path),
        "config": cfg,
        "objective": {
            "name": objective_cfg.get("name", "topm_soft_ce"),
            "top_m": int(objective_cfg.get("top_m", train_cfg.get("top_m", 4))),
            "tau": float(objective_cfg.get("tau", train_cfg.get("tau", 0.07))),
            "kl_coef": kl_coef,
            "anchor": "squared selected-code log-prob delta to pretrained VQ-BeT" if kl_coef > 0 else "none",
        },
        "git_sha": metadata["git_sha"],
        "git_dirty": metadata["git_dirty"],
        "git_branch": metadata["git_branch"],
        "command": command_line(),
        "device": str(device),
        "gpu_id": current_gpu_id(device),
        "cuda_visible_devices": env_override("CUDA_VISIBLE_DEVICES"),
        "dataset_path": dataset_payload["output_path"],
        "dataset_summary": dataset_payload.get("aggregated", {}),
        "candidate_attempt_weights": weights,
        "positive_weight_steps": int(step_weights.numel()),
        "loss_history": loss_history,
        "changed_parameters": changed,
        "grad_norms": grad_norms,
        "checks": checks,
        "wall_time_s": time.time() - start,
    }


def _pad_ranked_traces(
    traces_by_start: list[list[dict[str, list[torch.Tensor]]]],
    max_steps: int,
    *,
    storage_dtype: str,
) -> dict[str, torch.Tensor]:
    if not traces_by_start or not traces_by_start[0]:
        raise ValueError("No ranked traces to pad.")
    state_exemplar = next(trace["obs_state"][0] for traces in traces_by_start for trace in traces if trace["obs_state"])
    image_exemplar = next(trace["obs_image"][0] for traces in traces_by_start for trace in traces if trace["obs_image"])
    code_exemplar = next(trace["code_ids"][0] for traces in traces_by_start for trace in traces if trace["code_ids"])
    action_exemplar = next(
        trace["action_chunks"][0] for traces in traces_by_start for trace in traces if trace["action_chunks"]
    )
    n_starts = len(traces_by_start)
    keep_top = len(traces_by_start[0])
    image_dtype = torch.float16 if storage_dtype == "float16" else image_exemplar.dtype
    obs_state = torch.zeros(n_starts, keep_top, max_steps, *state_exemplar.shape, dtype=state_exemplar.dtype)
    obs_image = torch.zeros(n_starts, keep_top, max_steps, *image_exemplar.shape, dtype=image_dtype)
    code_ids = torch.zeros(n_starts, keep_top, max_steps, *code_exemplar.shape, dtype=code_exemplar.dtype)
    action_chunks = torch.zeros(n_starts, keep_top, max_steps, *action_exemplar.shape, dtype=torch.float16 if storage_dtype == "float16" else action_exemplar.dtype)
    rewards = torch.zeros(n_starts, keep_top, max_steps)
    coverages = torch.zeros(n_starts, keep_top, max_steps)
    valid = torch.zeros(n_starts, keep_top, max_steps, dtype=torch.bool)
    for start_ix, traces in enumerate(traces_by_start):
        for keep_ix, trace in enumerate(traces):
            length = min(max_steps, len(trace["code_ids"]))
            for step_ix in range(length):
                obs_state[start_ix, keep_ix, step_ix] = trace["obs_state"][step_ix]
                obs_image[start_ix, keep_ix, step_ix] = trace["obs_image"][step_ix].to(image_dtype)
                code_ids[start_ix, keep_ix, step_ix] = trace["code_ids"][step_ix]
                action_chunks[start_ix, keep_ix, step_ix] = trace["action_chunks"][step_ix].to(action_chunks.dtype)
                rewards[start_ix, keep_ix, step_ix] = trace["rewards"][step_ix]
                coverages[start_ix, keep_ix, step_ix] = trace["coverages"][step_ix]
                valid[start_ix, keep_ix, step_ix] = True
    return {
        "candidate_obs_state": obs_state,
        "candidate_obs_image": obs_image,
        "candidate_code_ids": code_ids.long(),
        "candidate_action_chunks": action_chunks,
        "candidate_rewards": rewards,
        "candidate_coverages": coverages,
        "candidate_valid": valid,
    }


def _pad_light_traces(
    traces_by_start: list[list[dict[str, list[torch.Tensor]]]],
    max_steps: int,
) -> dict[str, torch.Tensor]:
    code_exemplar = next(trace["code_ids"][0] for traces in traces_by_start for trace in traces if trace["code_ids"])
    action_exemplar = next(
        trace["action_chunks"][0] for traces in traces_by_start for trace in traces if trace["action_chunks"]
    )
    n_starts = len(traces_by_start)
    attempts = len(traces_by_start[0])
    code_ids = torch.zeros(n_starts, attempts, max_steps, *code_exemplar.shape, dtype=code_exemplar.dtype)
    action_chunks = torch.zeros(n_starts, attempts, max_steps, *action_exemplar.shape, dtype=action_exemplar.dtype)
    rewards = torch.zeros(n_starts, attempts, max_steps)
    coverages = torch.zeros(n_starts, attempts, max_steps)
    valid = torch.zeros(n_starts, attempts, max_steps, dtype=torch.bool)
    for start_ix, traces in enumerate(traces_by_start):
        for attempt_ix, trace in enumerate(traces):
            length = min(max_steps, len(trace["code_ids"]))
            for step_ix in range(length):
                code_ids[start_ix, attempt_ix, step_ix] = trace["code_ids"][step_ix]
                action_chunks[start_ix, attempt_ix, step_ix] = trace["action_chunks"][step_ix]
                rewards[start_ix, attempt_ix, step_ix] = trace["rewards"][step_ix]
                coverages[start_ix, attempt_ix, step_ix] = trace["coverages"][step_ix]
                valid[start_ix, attempt_ix, step_ix] = True
    return {
        "all_code_ids": code_ids.long(),
        "all_action_chunks": action_chunks,
        "all_rewards": rewards,
        "all_coverages": coverages,
        "all_valid": valid,
    }


def _write_v2_dataset_artifact(
    output_dir: Path,
    payload: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    tensor_path = output_dir / "polydistill_v2_tensors.pt"
    torch.save({key: value.detach().cpu() for key, value in tensors.items()}, tensor_path)
    payload = dict(payload)
    payload["tensor_path"] = tensor_path.name
    payload["tensor_shapes"] = {key: list(value.shape) for key, value in tensors.items()}
    errors = _validate_v2_dataset_payload(payload, tensors)
    payload["validation_status"] = "passed" if not errors else "rejected"
    payload["validation_errors"] = errors
    write_json(output_dir / "polydistill_v2_dataset.json", payload)
    write_json(output_dir / "seed_manifest.json", payload.get("seed_manifest", []))
    return payload


def _validate_v2_dataset_payload(payload: dict[str, Any], tensors: dict[str, torch.Tensor]) -> list[str]:
    errors = []
    required = {
        "candidate_obs_state",
        "candidate_obs_image",
        "candidate_code_ids",
        "candidate_action_chunks",
        "candidate_rewards",
        "candidate_coverages",
        "candidate_valid",
        "all_code_ids",
        "all_action_chunks",
        "all_rewards",
        "all_coverages",
        "all_valid",
        "attempt_successes",
        "attempt_max_overlaps",
        "attempt_returns",
        "ranked_attempts",
        "kept_successes",
        "kept_max_overlaps",
        "kept_returns",
    }
    missing = sorted(required.difference(tensors))
    if missing:
        errors.append(f"missing tensors: {missing}")
        return errors
    candidate_shape = tensors["candidate_valid"].shape
    for key in ("candidate_obs_state", "candidate_obs_image", "candidate_code_ids", "candidate_action_chunks"):
        if tensors[key].shape[:3] != candidate_shape:
            errors.append(f"{key} leading shape {list(tensors[key].shape[:3])} does not match candidate_valid")
    all_shape = tensors["all_valid"].shape
    for key in ("all_code_ids", "all_action_chunks", "all_rewards", "all_coverages"):
        if tensors[key].shape[:3] != all_shape:
            errors.append(f"{key} leading shape {list(tensors[key].shape[:3])} does not match all_valid")
    if payload.get("n_start_states") != candidate_shape[0]:
        errors.append("n_start_states does not match candidate tensor shape")
    if payload.get("attempts_per_start") != all_shape[1]:
        errors.append("attempts_per_start does not match all-attempt tensor shape")
    if payload.get("keep_top_attempts") != candidate_shape[1]:
        errors.append("keep_top_attempts does not match candidate tensor shape")
    if tensors["candidate_valid"].sum().item() <= 0:
        errors.append("candidate_valid contains no valid steps")
    for key, value in tensors.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            errors.append(f"{key} contains non-finite values")
    return errors
