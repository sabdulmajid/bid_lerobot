#!/usr/bin/env python

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.common.policies.polyppo_scaffold import (
    RolloutBatch,
    assign_set_advantages,
    logprob_ratio,
    pairwise_l1_diversity,
    ppo_loss,
)
from lerobot.common.polyppo.rollout import collect_polyppo_rollouts
from lerobot.common.polyppo.storage import load_rollout_artifact, write_train_artifact
from lerobot.common.polyppo.utils import (
    checkpoint_hashes,
    command_line,
    config_to_dict,
    current_gpu_id,
    git_metadata,
    load_config,
    now_run_id,
    select_device,
)


class TrainableRolloutTable(nn.Module):
    """A tiny trainable table used for deterministic PPO smoke tests."""

    def __init__(self, old_log_probs: torch.Tensor, values: torch.Tensor, entropy: torch.Tensor):
        super().__init__()
        self.current_log_probs = nn.Parameter(old_log_probs.clone())
        self.values = nn.Parameter(values.clone())
        self.register_buffer("entropy", entropy.clone())

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.current_log_probs, self.values, self.entropy


def train_polyppo_one_update(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg_dict = config_to_dict(cfg)
    run_cfg = cfg_dict.get("run", {})
    train_cfg = cfg_dict.get("train", {})
    output_dir = Path(run_cfg.get("output_dir", "outputs/polyppo/one_update"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_cfg.get("id") or now_run_id("polyppo_train")
    device = select_device(run_cfg.get("device"), run_cfg.get("gpu_id"))

    rollout_path = train_cfg.get("rollout_path")
    if train_cfg.get("collect_if_missing", False) and (not rollout_path or not Path(rollout_path).exists()):
        rollout_config = train_cfg.get("rollout_config")
        if not rollout_config:
            raise ValueError("train.collect_if_missing requires train.rollout_config.")
        collected_payload = collect_polyppo_rollouts(rollout_config)
        if collected_payload.get("validation_status") != "passed":
            raise RuntimeError(
                "train.collect_if_missing collected a rollout artifact that failed validation: "
                f"{collected_payload.get('validation_errors', [])}"
            )
        rollout_path = config_to_dict(load_config(rollout_config)).get("run", {}).get("output_dir")
    if not rollout_path:
        raise ValueError("train.rollout_path is required.")

    payload, tensors = load_rollout_artifact(rollout_path)
    if payload.get("validation_status") not in (None, "passed"):
        raise RuntimeError(
            "Refusing to train from a rollout artifact that failed validation: "
            f"{payload.get('validation_errors', [])}"
        )
    start = time.time()
    if payload.get("mock_policy", False) or train_cfg.get("mock_policy", False):
        result = _train_mock_one_update(cfg_dict, payload, tensors, output_dir, run_id, device)
    else:
        result = _train_vqbet_one_update(cfg_dict, payload, tensors, output_dir, run_id, device)
    result["wall_time_s"] = time.time() - start
    write_train_artifact(output_dir, result)
    return result


def _train_mock_one_update(
    cfg: dict[str, Any],
    rollout_payload: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    output_dir: Path,
    run_id: str,
    device: torch.device,
) -> dict[str, Any]:
    del rollout_payload
    train_cfg = cfg.get("train", {})
    poly_cfg = cfg.get("polyppo", {})
    old_log_probs = tensors["old_log_probs"].to(device).float()
    returns = tensors["returns"].to(device).float()
    old_values = tensors["values"].to(device).float()
    entropy = tensors["entropy"].to(device).float()
    code_ids = tensors["code_ids"].to(device)
    action_preds = tensors["action_preds"].to(device).float()
    valid_mask = tensors.get("valid", torch.ones_like(old_log_probs, dtype=torch.bool)).to(device).bool()
    kl_coef = float(train_cfg.get("kl_coef", 0.0))
    num_updates = int(train_cfg.get("num_updates", 1))
    if num_updates <= 0:
        raise ValueError("train.num_updates must be positive.")
    base_log_probs = None
    if kl_coef > 0.0:
        if "base_log_probs" not in tensors:
            raise ValueError("train.kl_coef > 0 requires rollout tensors['base_log_probs'].")
        base_log_probs = tensors["base_log_probs"].to(device).float()

    model = TrainableRolloutTable(old_log_probs, old_values, entropy).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-2)))

    current_log_probs, values, entropy_values = model()
    ratio = logprob_ratio(current_log_probs, old_log_probs)
    ratio_error = float((ratio[valid_mask] - 1.0).abs().max().item())
    _assert_close_ratio(ratio_error, float(train_cfg.get("ratio_tolerance", 1e-6)))

    advantages = _assign_step_advantages(
        returns,
        code_ids=code_ids,
        action_preds=action_preds,
        diversity_kind=poly_cfg.get("diversity_kind", "none"),
        poly_lambda=float(poly_cfg.get("lambda_div", 0.0)),
        valid_mask=valid_mask,
    )

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    loss_history = []
    grad_norms = {}
    loss_out = None
    for update_ix in range(num_updates):
        current_log_probs, values, entropy_values = model()
        batch = RolloutBatch(
            returns=returns,
            old_log_probs=old_log_probs,
            values=values,
            current_log_probs=current_log_probs,
        )
        loss_out = ppo_loss(
            batch,
            advantages,
            clip_ratio=float(train_cfg.get("clip_ratio", 0.2)),
            value_coef=float(train_cfg.get("value_coef", 0.5)),
            entropy=entropy_values,
            entropy_coef=float(train_cfg.get("entropy_coef", 0.01)),
            base_log_probs=base_log_probs,
            kl_coef=kl_coef,
            valid_mask=valid_mask,
        )
        _assert_finite_loss(loss_out)
        optimizer.zero_grad()
        loss_out.total_loss.backward()
        grad_norms = _grad_norms(model)
        optimizer.step()
        loss_history.append(_loss_history_row(update_ix, loss_out, grad_norms))
    if loss_out is None:
        raise RuntimeError("No PPO updates were run.")
    changed = _changed_parameters(before, model)
    if not changed:
        raise RuntimeError("one-update smoke failed: no trainable parameters changed.")
    if not any(value > 0 for value in grad_norms.values()):
        raise RuntimeError("one-update smoke failed: all gradient norms are zero.")

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint_path)
    reloaded = TrainableRolloutTable(old_log_probs, old_values, entropy).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    _write_mock_post_update_eval(output_dir / "post_update_eval", int(train_cfg.get("post_update_eval_episodes", 10)))

    return _train_result_payload(
        cfg=cfg,
        run_id=run_id,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        ratio_error=ratio_error,
        loss_out=loss_out,
        grad_norms=grad_norms,
        changed_parameters=changed,
        post_update_eval_path=output_dir / "post_update_eval",
        device=device,
        num_updates=num_updates,
        loss_history=loss_history,
    )


def _train_vqbet_one_update(
    cfg: dict[str, Any],
    rollout_payload: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    output_dir: Path,
    run_id: str,
    device: torch.device,
) -> dict[str, Any]:
    from lerobot.common.policies.factory import make_policy
    from lerobot.scripts.eval import get_pretrained_policy_path, load_pretrained_policy_hydra_config

    train_cfg = cfg.get("train", {})
    poly_cfg = cfg.get("polyppo", {})
    policy_override = train_cfg.get("policy_path_override")
    policy_path = policy_override or rollout_payload["checkpoint"].get("main_policy_path") or cfg.get("policy", {}).get("path")
    if policy_path is None:
        raise ValueError("Training requires a rollout checkpoint main_policy_path or train.policy_path_override.")
    pretrained_path = get_pretrained_policy_path(policy_path)
    hydra_cfg = load_pretrained_policy_hydra_config(pretrained_path, [])
    hydra_cfg.device = str(device)
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_path))
    policy.to(device)
    policy.eval()

    for param in policy.parameters():
        param.requires_grad = False
    trainable_modules = [policy.vqbet.value_head]
    if hydra_cfg.policy.sequentially_select:
        trainable_modules.extend(
            [
                policy.vqbet.action_head.map_to_cbet_preds_primary_bin,
                policy.vqbet.action_head.map_to_cbet_preds_secondary_bin,
            ]
        )
    else:
        trainable_modules.append(policy.vqbet.action_head.map_to_cbet_preds_bin)
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        [param for param in policy.parameters() if param.requires_grad],
        lr=float(train_cfg.get("lr", 1e-5)),
    )
    obs_batch = {
        "observation.state": tensors["obs_state"].reshape(-1, *tensors["obs_state"].shape[-2:]).to(device),
        "observation.image": tensors["obs_image"].reshape(-1, *tensors["obs_image"].shape[-4:]).to(device),
    }
    code_ids = tensors["code_ids"].reshape(-1, tensors["code_ids"].shape[-1]).to(device)
    old_log_probs = tensors["old_log_probs"].to(device).float()
    returns = tensors["returns"].to(device).float()
    action_preds = tensors["action_preds"].to(device).float()
    code_ids_set = tensors["code_ids"].to(device)
    valid_mask = tensors.get("valid", torch.ones_like(old_log_probs, dtype=torch.bool)).to(device).bool()
    kl_coef = float(train_cfg.get("kl_coef", 0.0))
    num_updates = int(train_cfg.get("num_updates", 1))
    if num_updates <= 0:
        raise ValueError("train.num_updates must be positive.")

    out = _evaluate_rollout_codes(
        policy,
        obs_batch,
        code_ids,
        temperature=float(cfg.get("policy", {}).get("temperature", hydra_cfg.policy.bet_softmax_temperature)),
        batch_size=int(train_cfg.get("recompute_batch_size", 1)),
    )
    current_log_probs = out["log_prob"].reshape_as(old_log_probs)
    current_values = out["value"].reshape_as(returns)
    current_entropy = out["entropy"].reshape_as(returns)
    ratio = logprob_ratio(current_log_probs, old_log_probs)
    ratio_error = float((ratio[valid_mask] - 1.0).abs().max().item())
    _assert_close_ratio(ratio_error, float(train_cfg.get("ratio_tolerance", 1e-4)))
    base_log_probs = None
    if kl_coef > 0.0:
        base_policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_path))
        base_policy.to(device)
        base_policy.eval()
        for param in base_policy.parameters():
            param.requires_grad = False
        with torch.no_grad():
            base_out = _evaluate_rollout_codes(
                base_policy,
                obs_batch,
                code_ids,
                temperature=float(cfg.get("policy", {}).get("temperature", hydra_cfg.policy.bet_softmax_temperature)),
                batch_size=int(train_cfg.get("recompute_batch_size", 1)),
            )
        base_log_probs = base_out["log_prob"].reshape_as(old_log_probs).detach()

    advantages = _assign_step_advantages(
        returns,
        code_ids=code_ids_set,
        action_preds=action_preds,
        diversity_kind=poly_cfg.get("diversity_kind", "none"),
        poly_lambda=float(poly_cfg.get("lambda_div", 0.0)),
        valid_mask=valid_mask,
    )
    before = {name: param.detach().clone() for name, param in policy.named_parameters() if param.requires_grad}
    loss_history = []
    grad_norms = {}
    loss_out = None
    for update_ix in range(num_updates):
        out = _evaluate_rollout_codes(
            policy,
            obs_batch,
            code_ids,
            temperature=float(cfg.get("policy", {}).get("temperature", hydra_cfg.policy.bet_softmax_temperature)),
            batch_size=int(train_cfg.get("recompute_batch_size", 1)),
        )
        current_log_probs = out["log_prob"].reshape_as(old_log_probs)
        current_values = out["value"].reshape_as(returns)
        current_entropy = out["entropy"].reshape_as(returns)
        batch = RolloutBatch(
            returns=returns,
            old_log_probs=old_log_probs,
            values=current_values,
            current_log_probs=current_log_probs,
        )
        loss_out = ppo_loss(
            batch,
            advantages,
            clip_ratio=float(train_cfg.get("clip_ratio", 0.2)),
            value_coef=float(train_cfg.get("value_coef", 0.5)),
            entropy=current_entropy,
            entropy_coef=float(train_cfg.get("entropy_coef", 0.01)),
            base_log_probs=base_log_probs,
            kl_coef=kl_coef,
            valid_mask=valid_mask,
        )
        _assert_finite_loss(loss_out)
        optimizer.zero_grad()
        loss_out.total_loss.backward()
        grad_norms = _grad_norms(policy)
        optimizer.step()
        loss_history.append(_loss_history_row(update_ix, loss_out, grad_norms))
    if loss_out is None:
        raise RuntimeError("No PPO updates were run.")
    changed = _changed_parameters(before, policy)
    if not changed:
        raise RuntimeError("VQ-BeT PPO one-update smoke failed: no trainable parameters changed.")

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"model_state": policy.state_dict(), "config": cfg}, checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state["model_state"])
    _write_real_post_update_eval(
        policy,
        hydra_cfg,
        output_dir / "post_update_eval",
        checkpoint_path,
        cfg,
        device,
    )

    return _train_result_payload(
        cfg=cfg,
        run_id=run_id,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        ratio_error=ratio_error,
        loss_out=loss_out,
        grad_norms=grad_norms,
        changed_parameters=changed,
        post_update_eval_path=output_dir / "post_update_eval",
        device=device,
        num_updates=num_updates,
        loss_history=loss_history,
    )


def _loss_history_row(update_ix: int, loss_out, grad_norms: dict[str, float]) -> dict[str, Any]:
    return {
        "update_ix": update_ix,
        "total": float(loss_out.total_loss.detach().cpu().item()),
        "policy": float(loss_out.policy_loss.detach().cpu().item()),
        "value": float(loss_out.value_loss.detach().cpu().item()),
        "entropy": float(loss_out.entropy_bonus.detach().cpu().item()),
        "kl_to_base": float(loss_out.kl_to_base.detach().cpu().item()),
        "approx_kl": float(loss_out.approx_kl.detach().cpu().item()),
        "clip_fraction": float(loss_out.clip_fraction.detach().cpu().item()),
        "grad_norm_max": float(max(grad_norms.values())) if grad_norms else 0.0,
    }


def _assert_close_ratio(ratio_error: float, tolerance: float) -> None:
    if ratio_error > tolerance:
        raise RuntimeError(f"PPO ratio check failed before update: max |ratio-1|={ratio_error} > {tolerance}")


def _evaluate_rollout_codes(
    policy: nn.Module,
    obs_batch: dict[str, torch.Tensor],
    code_ids: torch.Tensor,
    *,
    temperature: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Evaluate rollout code ids with a configurable microbatch size.

    The default microbatch size is one because the hard PPO gate compares against
    old log-probs collected online one environment step at a time. Larger
    microbatches are valid only when rollout collection stores old log-probs
    recomputed through the same path.
    """
    if batch_size <= 0:
        raise ValueError("train.recompute_batch_size must be positive.")
    outputs: dict[str, list[torch.Tensor]] = {"log_prob": [], "value": [], "entropy": []}
    total = code_ids.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        micro_obs = {key: value[start:end] for key, value in obs_batch.items()}
        micro_out = policy.evaluate_code_actions(
            micro_obs,
            code_ids=code_ids[start:end],
            temperature=temperature,
        )
        for key in outputs:
            outputs[key].append(micro_out[key])
    return {key: torch.cat(values, dim=0) for key, values in outputs.items()}


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


def _assert_finite_loss(loss_out) -> None:
    values = {
        "total_loss": loss_out.total_loss,
        "policy_loss": loss_out.policy_loss,
        "value_loss": loss_out.value_loss,
        "entropy_bonus": loss_out.entropy_bonus,
        "kl_to_base": loss_out.kl_to_base,
        "approx_kl": loss_out.approx_kl,
        "clip_fraction": loss_out.clip_fraction,
    }
    bad = [name for name, value in values.items() if not torch.isfinite(value).all()]
    if bad:
        raise RuntimeError(f"Non-finite PPO loss fields: {bad}")


def _grad_norms(model: nn.Module) -> dict[str, float]:
    out = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            out[name] = float(param.grad.detach().norm().item())
    return out


def _changed_parameters(before: dict[str, torch.Tensor], model: nn.Module) -> list[str]:
    changed = []
    current = dict(model.named_parameters())
    for name, old_value in before.items():
        if name in current and not torch.equal(old_value.cpu(), current[name].detach().cpu()):
            changed.append(name)
    return changed


def _write_mock_post_update_eval(output_dir: Path, n_episodes: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_episode = [
        {
            "episode_ix": ix,
            "seed": 200000 + ix,
            "sum_reward": 1.0 + 0.01 * ix,
            "max_reward": 0.5 + 0.01 * ix,
            "success": bool(ix % 2 == 0),
            "num_steps": 10,
        }
        for ix in range(n_episodes)
    ]
    payload = {
        "per_episode": per_episode,
        "aggregated": {
            "avg_sum_reward": float(sum(row["sum_reward"] for row in per_episode) / n_episodes),
            "avg_max_reward": float(sum(row["max_reward"] for row in per_episode) / n_episodes),
            "pc_success": float(100.0 * sum(row["success"] for row in per_episode) / n_episodes),
            "pass_at_k": {"pass@1": 0.5},
        },
        "run_metadata": {
            "sampler": "polyppo_mock",
            "n_episodes": n_episodes,
            "checkpoint": {
                "main_policy_path": str(output_dir.parent / "checkpoint.pt"),
                "main_policy_hashes": checkpoint_hashes(output_dir.parent / "checkpoint.pt"),
                "reference_policy_path": None,
                "reference_policy_hashes": [],
            },
        },
    }
    (output_dir / "eval_info.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output_dir / "seed_manifest.json").write_text(json.dumps([row["seed"] for row in per_episode], indent=2))


def _write_real_post_update_eval(
    policy: nn.Module,
    hydra_cfg,
    output_dir: Path,
    checkpoint_path: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> None:
    from lerobot.common.artifacts.validator import validate_artifact
    from lerobot.common.envs.factory import make_env
    from lerobot.common.polyppo.utils import current_gpu_id
    from lerobot.common.utils.utils import set_global_seed
    from lerobot.scripts.eval import eval_policy

    train_cfg = cfg.get("train", {})
    output_dir.mkdir(parents=True, exist_ok=True)
    n_episodes = int(train_cfg.get("post_update_eval_episodes", 10))
    hydra_cfg.eval.n_episodes = n_episodes
    hydra_cfg.eval.batch_size = int(train_cfg.get("post_update_eval_batch_size", min(n_episodes, 10)))
    hydra_cfg.eval.use_async_envs = False
    hydra_cfg.seed = int(train_cfg.get("post_update_eval_seed", cfg.get("run", {}).get("seed", 100000) + 1000))
    hydra_cfg.device = str(device)
    hydra_cfg.use_amp = False
    sampler = train_cfg.get("post_update_sampler", "direct")
    temperature = float(cfg.get("policy", {}).get("temperature", hydra_cfg.policy.bet_softmax_temperature))

    set_global_seed(hydra_cfg.seed)
    env = make_env(hydra_cfg)
    try:
        info = eval_policy(
            env,
            policy,
            policy,
            n_episodes=n_episodes,
            max_episodes_rendered=0,
            videos_dir=output_dir / "videos",
            start_seed=hydra_cfg.seed,
            enable_progbar=False,
            enable_inner_progbar=False,
            sampler=sampler,
            temperature=temperature,
        )
    finally:
        env.close()

    metadata = git_metadata(Path.cwd())
    info["run_metadata"] = {
        "sampler": sampler,
        "n_episodes": n_episodes,
        "seed": hydra_cfg.seed,
        "command": command_line(),
        "git_sha": metadata["git_sha"],
        "git_dirty": metadata["git_dirty"],
        "git_branch": metadata["git_branch"],
        "device": str(device),
        "gpu_id": current_gpu_id(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "output_path": str(output_dir),
        "checkpoint": {
            "main_policy_path": str(checkpoint_path),
            "main_policy_hashes": checkpoint_hashes(checkpoint_path),
            "reference_policy_path": None,
            "reference_policy_hashes": [],
        },
        "temperature": temperature,
        "action_identity": "continuous env action from trained VQ-BeT; PPO update action identity is RVQ code ids",
    }
    (output_dir / "eval_info.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    (output_dir / "seed_manifest.json").write_text(
        json.dumps([episode["seed"] for episode in info["per_episode"]], indent=2)
    )
    validation = validate_artifact(output_dir, profile="smoke")
    if not validation.ok:
        raise RuntimeError(f"post-update eval artifact failed validation: {validation.errors}")


def _train_result_payload(
    *,
    cfg: dict[str, Any],
    run_id: str,
    output_dir: Path,
    checkpoint_path: Path,
    ratio_error: float,
    loss_out,
    grad_norms: dict[str, float],
    changed_parameters: list[str],
    post_update_eval_path: Path,
    device: torch.device,
    num_updates: int = 1,
    loss_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata = git_metadata(Path.cwd())
    return {
        "artifact_kind": "polyppo_train",
        "run_id": run_id,
        "git_sha": metadata["git_sha"],
        "git_dirty": metadata["git_dirty"],
        "git_branch": metadata["git_branch"],
        "command": command_line(),
        "output_path": str(output_dir),
        "gpu_id": current_gpu_id(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hashes": checkpoint_hashes(checkpoint_path),
        "post_update_eval_path": str(post_update_eval_path),
        "num_updates": num_updates,
        "ppo_ratio_max_abs_error_before_update": ratio_error,
        "loss": {
            "total": float(loss_out.total_loss.detach().cpu().item()),
            "policy": float(loss_out.policy_loss.detach().cpu().item()),
            "value": float(loss_out.value_loss.detach().cpu().item()),
            "entropy": float(loss_out.entropy_bonus.detach().cpu().item()),
            "kl_to_base": float(loss_out.kl_to_base.detach().cpu().item()),
            "approx_kl": float(loss_out.approx_kl.detach().cpu().item()),
            "clip_fraction": float(loss_out.clip_fraction.detach().cpu().item()),
        },
        "checks": {
            "ratio_is_one_before_update": ratio_error <= float(cfg.get("train", {}).get("ratio_tolerance", 1e-6)),
            "loss_finite": all(math.isfinite(v) for v in [
                float(loss_out.total_loss.detach().cpu().item()),
                float(loss_out.policy_loss.detach().cpu().item()),
                float(loss_out.value_loss.detach().cpu().item()),
                float(loss_out.entropy_bonus.detach().cpu().item()),
                float(loss_out.approx_kl.detach().cpu().item()),
            ]),
            "nonzero_gradients": any(value > 0 for value in grad_norms.values()),
            "parameters_changed": bool(changed_parameters),
            "checkpoint_saved": checkpoint_path.exists(),
            "post_update_eval_written": (post_update_eval_path / "eval_info.json").exists(),
        },
        "grad_norms": grad_norms,
        "changed_parameters": changed_parameters,
        "loss_history": loss_history or [],
        "config": cfg,
    }
