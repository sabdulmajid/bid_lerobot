#!/usr/bin/env python

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

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
    git_metadata,
    load_config,
    now_run_id,
    select_device,
    stable_hash,
    write_json,
)


def collect_polydistill_dataset(config_path: str | Path) -> dict[str, Any]:
    cfg = config_to_dict(load_config(config_path))
    run_cfg = cfg.get("run", {})
    output_dir = Path(run_cfg.get("output_dir", "outputs/polyppo/polydistill_dataset"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_cfg.get("id") or now_run_id("polydistill_collect")
    device = select_device(run_cfg.get("device"), run_cfg.get("gpu_id"))
    start = time.time()
    if cfg.get("distill", {}).get("mock", False):
        payload, tensors = _collect_mock_dataset(cfg, run_id, output_dir, device)
    else:
        payload, tensors = _collect_pusht_dataset(cfg, run_id, output_dir, device)
    payload["wall_time_s"] = time.time() - start
    return _write_dataset_artifact(output_dir, payload, tensors)


def train_polydistill(config_path: str | Path) -> dict[str, Any]:
    cfg = config_to_dict(load_config(config_path))
    run_cfg = cfg.get("run", {})
    train_cfg = cfg.get("train", {})
    output_dir = Path(run_cfg.get("output_dir", "outputs/polyppo/polydistill_train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_cfg.get("id") or now_run_id("polydistill_train")
    device = select_device(run_cfg.get("device"), run_cfg.get("gpu_id"))
    dataset_path = train_cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("train.dataset_path is required.")

    payload, tensors = load_polydistill_dataset(dataset_path)
    if payload.get("artifact_kind") != "polydistill_dataset":
        raise ValueError(f"Expected polydistill_dataset artifact, got {payload.get('artifact_kind')}.")
    if payload.get("validation_status") != "passed":
        raise RuntimeError(f"Refusing to train from invalid PolyDistill dataset: {payload.get('validation_errors', [])}")

    if payload.get("mock_policy", False) or train_cfg.get("mock_policy", False):
        result = _train_mock_distill(cfg, payload, tensors, output_dir, run_id, device)
    else:
        result = _train_vqbet_distill(cfg, payload, tensors, output_dir, run_id, device)
    result["wall_time_s"] = result.get("wall_time_s", 0.0)
    write_json(output_dir / "polydistill_train_info.json", result)
    return result


def load_polydistill_dataset(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    path = Path(path)
    payload_path = path if path.is_file() else path / "polydistill_dataset.json"
    artifact_dir = payload_path.parent
    import json

    payload = json.loads(payload_path.read_text())
    tensors = torch.load(artifact_dir / payload["tensor_path"], map_location="cpu")
    return payload, tensors


def select_top_attempt_indices(
    successes: torch.Tensor,
    max_overlaps: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Select the top attempt per start, preferring success then overlap then return."""
    if not (successes.shape == max_overlaps.shape == returns.shape):
        raise ValueError("successes, max_overlaps, and returns must share shape.")
    score = max_overlaps.float() + returns.float() * 1e-6 + successes.float() * 1_000.0
    return score.argmax(dim=1)


def flatten_selected_steps(
    obs_state: torch.Tensor,
    obs_image: torch.Tensor,
    code_ids: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if valid.shape != code_ids.shape[:2]:
        raise ValueError("valid must match selected code-id leading shape.")
    flat_valid = valid.reshape(-1).bool()
    batch = {
        "observation.state": obs_state.reshape(-1, *obs_state.shape[-2:])[flat_valid],
        "observation.image": obs_image.reshape(-1, *obs_image.shape[-4:])[flat_valid],
    }
    return batch, code_ids.reshape(-1, code_ids.shape[-1])[flat_valid]


def _collect_pusht_dataset(
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
        raise ValueError("policy.path is required for PolyDistill collection.")

    seed = int(run_cfg.get("seed", 190000))
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
    temperature = float(policy_cfg.get("temperature", hydra_cfg.policy.bet_softmax_temperature))
    if n_starts <= 0 or attempts <= 0:
        raise ValueError("n_start_states and attempts_per_start must be positive.")

    gym_kwargs = dict(hydra_cfg.env.get("gym", {}))
    gym_kwargs["max_episode_steps"] = max_steps
    env = gym.make(f"gym_{hydra_cfg.env.name}/{hydra_cfg.env.task}", disable_env_checker=True, **gym_kwargs)

    selected_traces = []
    successes = torch.zeros(n_starts, attempts, dtype=torch.bool)
    max_overlaps = torch.zeros(n_starts, attempts)
    returns = torch.zeros(n_starts, attempts)
    episode_lengths = torch.zeros(n_starts, attempts, dtype=torch.long)
    selected_attempts = torch.zeros(n_starts, dtype=torch.long)
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
            attempt_traces = []
            attempts_json = []
            for attempt_ix in range(attempts):
                restore_pusht_state(env, start_state)
                history = start_history.clone()
                if hasattr(policy, "reset"):
                    policy.reset()
                trace = {"obs_state": [], "obs_image": [], "code_ids": []}
                total_return = 0.0
                max_coverage = 0.0
                success = False
                steps = 0
                for step_ix in range(max_steps):
                    batch = history.batch()
                    trace["obs_state"].append(batch["observation.state"].squeeze(0).detach().cpu().clone())
                    image_key = "observation.image" if "observation.image" in batch else next(
                        key for key in batch if key.startswith("observation.image")
                    )
                    trace["obs_image"].append(batch[image_key].squeeze(0).detach().cpu().clone())
                    action, out, _, _, _ = _sample_code_action(policy, batch, device, temperature)
                    trace["code_ids"].append(out["code_ids"].squeeze(0).detach().cpu().clone())
                    obs, reward, terminated, truncated, info = env.step(action)
                    history.append(_preprocess_single_observation(preprocess_observation, obs))
                    total_return += float(reward)
                    max_coverage = max(max_coverage, float(info.get("coverage", 0.0)))
                    success = success or bool(info.get("is_success", False))
                    steps = step_ix + 1
                    if bool(terminated or truncated):
                        break
                successes[start_ix, attempt_ix] = success
                max_overlaps[start_ix, attempt_ix] = max_coverage
                returns[start_ix, attempt_ix] = total_return
                episode_lengths[start_ix, attempt_ix] = steps
                attempt_traces.append(trace)
                attempts_json.append(
                    {
                        "attempt_ix": attempt_ix,
                        "success": success,
                        "max_overlap": max_coverage,
                        "sum_reward": total_return,
                        "num_steps": steps,
                    }
                )
            selected_ix = int(
                select_top_attempt_indices(
                    successes[start_ix : start_ix + 1],
                    max_overlaps[start_ix : start_ix + 1],
                    returns[start_ix : start_ix + 1],
                ).item()
            )
            selected_attempts[start_ix] = selected_ix
            selected_traces.append(attempt_traces[selected_ix])
            start_records.append(
                {
                    "start_ix": start_ix,
                    "start_seed": start_seed,
                    "start_state_hash": start_hash,
                    "selected_attempt_ix": selected_ix,
                    "attempts": attempts_json,
                }
            )
    finally:
        env.close()

    selected_obs_state, selected_obs_image, selected_code_ids, selected_valid = _pad_selected_traces(
        selected_traces, max_steps
    )
    tensors = {
        "selected_obs_state": selected_obs_state,
        "selected_obs_image": selected_obs_image,
        "selected_code_ids": selected_code_ids.long(),
        "selected_valid": selected_valid,
        "attempt_successes": successes,
        "attempt_max_overlaps": max_overlaps,
        "attempt_returns": returns,
        "attempt_episode_lengths": episode_lengths,
        "selected_attempts": selected_attempts,
    }
    selected_successes = successes[torch.arange(n_starts), selected_attempts]
    selected_overlaps = max_overlaps[torch.arange(n_starts), selected_attempts]
    selected_returns = returns[torch.arange(n_starts), selected_attempts]
    metadata = git_metadata(Path.cwd())
    payload = {
        "artifact_kind": "polydistill_dataset",
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
        "config": cfg,
        "checkpoint": {
            "main_policy_path": str(pretrained_path),
            "main_policy_hashes": checkpoint_hashes(pretrained_path),
            "reference_policy_path": None,
            "reference_policy_hashes": [],
        },
        "n_start_states": n_starts,
        "attempts_per_start": attempts,
        "max_steps": max_steps,
        "start_state_hashes": start_hashes,
        "starts": start_records,
        "selected_attempts": selected_attempts,
        "selected_success": selected_successes,
        "selected_max_overlap": selected_overlaps,
        "selected_return": selected_returns,
        "aggregated": {
            "candidate_success_rate": float(successes.float().mean().item()),
            "selected_success_rate": float(selected_successes.float().mean().item()),
            "selected_avg_max_overlap": float(selected_overlaps.float().mean().item()),
            "selected_avg_return": float(selected_returns.float().mean().item()),
            "selected_valid_steps": int(selected_valid.sum().item()),
        },
    }
    return payload, tensors


def _collect_mock_dataset(
    cfg: dict[str, Any], run_id: str, output_dir: Path, device: torch.device
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    del output_dir, device
    distill_cfg = cfg.get("distill", {})
    n_starts = int(distill_cfg.get("n_start_states", 4))
    attempts = int(distill_cfg.get("attempts_per_start", 3))
    max_steps = int(distill_cfg.get("max_steps", 5))
    generator = torch.Generator().manual_seed(int(cfg.get("run", {}).get("seed", 123)))
    successes = torch.zeros(n_starts, attempts, dtype=torch.bool)
    successes[:, -1] = True
    max_overlaps = torch.rand(n_starts, attempts, generator=generator)
    returns = torch.rand(n_starts, attempts, generator=generator)
    selected = select_top_attempt_indices(successes, max_overlaps, returns)
    tensors = {
        "selected_obs_state": torch.randn(n_starts, max_steps, 2, 2, generator=generator),
        "selected_obs_image": torch.randn(n_starts, max_steps, 2, 3, 16, 16, generator=generator),
        "selected_code_ids": torch.randint(0, 8, (n_starts, max_steps, 2), generator=generator),
        "selected_valid": torch.ones(n_starts, max_steps, dtype=torch.bool),
        "attempt_successes": successes,
        "attempt_max_overlaps": max_overlaps,
        "attempt_returns": returns,
        "selected_attempts": selected,
    }
    metadata = git_metadata(Path.cwd())
    payload = {
        "artifact_kind": "polydistill_dataset",
        "run_id": run_id,
        "output_path": str(cfg.get("run", {}).get("output_dir", "outputs/polyppo/polydistill_mock")),
        "mock_policy": True,
        "git_sha": metadata["git_sha"],
        "git_dirty": metadata["git_dirty"],
        "git_branch": metadata["git_branch"],
        "command": command_line(),
        "seed": int(cfg.get("run", {}).get("seed", 123)),
        "seed_manifest": list(range(n_starts)),
        "device": "cpu",
        "gpu_id": None,
        "config": cfg,
        "n_start_states": n_starts,
        "attempts_per_start": attempts,
        "max_steps": max_steps,
        "selected_attempts": selected,
        "selected_success": successes[torch.arange(n_starts), selected],
        "selected_max_overlap": max_overlaps[torch.arange(n_starts), selected],
        "selected_return": returns[torch.arange(n_starts), selected],
        "aggregated": {
            "candidate_success_rate": float(successes.float().mean().item()),
            "selected_success_rate": float(successes[torch.arange(n_starts), selected].float().mean().item()),
            "selected_valid_steps": int(tensors["selected_valid"].sum().item()),
        },
    }
    return payload, tensors


def _train_vqbet_distill(
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
    policy_path = train_cfg.get("policy_path_override") or dataset_payload["checkpoint"].get("main_policy_path")
    pretrained_path = get_pretrained_policy_path(policy_path)
    hydra_cfg = load_pretrained_policy_hydra_config(pretrained_path, [])
    hydra_cfg.device = str(device)
    hydra_cfg.use_amp = False
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_path))
    policy.to(device)
    policy.train()

    for param in policy.parameters():
        param.requires_grad = False
    trainable_modules = _code_head_modules(policy, hydra_cfg.policy.sequentially_select)
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True

    batch_all, code_ids_all = flatten_selected_steps(
        tensors["selected_obs_state"],
        tensors["selected_obs_image"],
        tensors["selected_code_ids"],
        tensors["selected_valid"],
    )
    if code_ids_all.numel() == 0:
        raise ValueError("PolyDistill dataset has no valid selected code steps.")
    batch_all = {key: value.to(device) for key, value in batch_all.items()}
    code_ids_all = code_ids_all.to(device)

    optimizer = torch.optim.Adam(
        [param for param in policy.parameters() if param.requires_grad],
        lr=float(train_cfg.get("lr", 3e-5)),
    )
    epochs = int(train_cfg.get("epochs", 5))
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
        for start_ix in range(0, order.numel(), batch_size):
            indices = order[start_ix : start_ix + batch_size]
            micro_batch = {key: value[indices] for key, value in batch_all.items()}
            out = policy.evaluate_code_actions(micro_batch, code_ids=code_ids_all[indices], temperature=temperature)
            loss = -out["log_prob"].mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norms = _grad_norms(policy)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        loss_history.append(
            {
                "epoch": epoch,
                "ce_loss": float(sum(losses) / max(len(losses), 1)),
                "grad_norm_max": float(max(grad_norms.values())) if grad_norms else 0.0,
            }
        )

    changed = _changed_parameters(before, policy)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"model_state": policy.state_dict(), "config": cfg}, checkpoint_path)
    result = _distill_result_payload(
        cfg=cfg,
        run_id=run_id,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        changed_parameters=changed,
        loss_history=loss_history,
        grad_norms=grad_norms,
        dataset_payload=dataset_payload,
        valid_steps=int(code_ids_all.shape[0]),
        wall_time_s=time.time() - start,
    )
    return result


def _train_mock_distill(
    cfg: dict[str, Any],
    dataset_payload: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    output_dir: Path,
    run_id: str,
    device: torch.device,
) -> dict[str, Any]:
    start = time.time()
    _, code_ids = flatten_selected_steps(
        tensors["selected_obs_state"],
        tensors["selected_obs_image"],
        tensors["selected_code_ids"],
        tensors["selected_valid"],
    )
    model = nn.Embedding(16, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = code_ids[:, 0].to(device).clamp_max(15)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    loss_history = []
    for epoch in range(int(cfg.get("train", {}).get("epochs", 2))):
        logits = model.weight @ model.weight[target].mean(dim=0)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0).expand(target.numel(), -1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append({"epoch": epoch, "ce_loss": float(loss.detach().cpu().item()), "grad_norm_max": 1.0})
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint_path)
    return _distill_result_payload(
        cfg=cfg,
        run_id=run_id,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        changed_parameters=_changed_parameters(before, model),
        loss_history=loss_history,
        grad_norms={"mock": 1.0},
        dataset_payload=dataset_payload,
        valid_steps=int(code_ids.shape[0]),
        wall_time_s=time.time() - start,
    )


def _distill_result_payload(
    *,
    cfg: dict[str, Any],
    run_id: str,
    output_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    changed_parameters: list[str],
    loss_history: list[dict[str, Any]],
    grad_norms: dict[str, float],
    dataset_payload: dict[str, Any],
    valid_steps: int,
    wall_time_s: float,
) -> dict[str, Any]:
    metadata = git_metadata(Path.cwd())
    loss_finite = all(torch.isfinite(torch.tensor(row["ce_loss"])) for row in loss_history)
    checks = {
        "checkpoint_saved": checkpoint_path.exists(),
        "loss_finite": bool(loss_finite),
        "nonzero_gradients": bool(any(value > 0 for value in grad_norms.values())),
        "parameters_changed": bool(changed_parameters),
        "valid_selected_steps": valid_steps > 0,
    }
    return {
        "artifact_kind": "polydistill_train",
        "run_id": run_id,
        "output_path": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hashes": checkpoint_hashes(checkpoint_path),
        "config": cfg,
        "git_sha": metadata["git_sha"],
        "git_dirty": metadata["git_dirty"],
        "git_branch": metadata["git_branch"],
        "command": command_line(),
        "device": str(device),
        "gpu_id": current_gpu_id(device),
        "dataset_path": dataset_payload["output_path"],
        "dataset_summary": dataset_payload.get("aggregated", {}),
        "valid_selected_steps": valid_steps,
        "loss_history": loss_history,
        "changed_parameters": changed_parameters,
        "grad_norms": grad_norms,
        "checks": checks,
        "wall_time_s": wall_time_s,
    }


def _write_dataset_artifact(
    output_dir: Path,
    payload: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    tensor_path = output_dir / "polydistill_tensors.pt"
    torch.save({key: value.detach().cpu() for key, value in tensors.items()}, tensor_path)
    payload = dict(payload)
    payload["tensor_path"] = tensor_path.name
    payload["tensor_shapes"] = {key: list(value.shape) for key, value in tensors.items()}
    errors = _validate_dataset_payload(payload, tensors)
    payload["validation_status"] = "passed" if not errors else "rejected"
    payload["validation_errors"] = errors
    write_json(output_dir / "polydistill_dataset.json", payload)
    write_json(output_dir / "seed_manifest.json", payload.get("seed_manifest", []))
    return payload


def _validate_dataset_payload(payload: dict[str, Any], tensors: dict[str, torch.Tensor]) -> list[str]:
    errors = []
    required = {"selected_obs_state", "selected_obs_image", "selected_code_ids", "selected_valid"}
    missing = sorted(required.difference(tensors))
    if missing:
        errors.append(f"missing tensors: {missing}")
        return errors
    if tensors["selected_valid"].sum().item() <= 0:
        errors.append("selected_valid contains no valid steps")
    leading = tensors["selected_valid"].shape
    for key in ("selected_obs_state", "selected_obs_image", "selected_code_ids"):
        if tensors[key].shape[:2] != leading:
            errors.append(f"{key} leading shape {list(tensors[key].shape[:2])} does not match valid {list(leading)}")
    if payload.get("n_start_states") != leading[0]:
        errors.append("n_start_states does not match tensor shape")
    for key, value in tensors.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            errors.append(f"{key} contains non-finite values")
    return errors


def _pad_selected_traces(
    traces: list[dict[str, list[torch.Tensor]]],
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not traces:
        raise ValueError("No selected traces to pad.")
    state_exemplar = next(trace["obs_state"][0] for trace in traces if trace["obs_state"])
    image_exemplar = next(trace["obs_image"][0] for trace in traces if trace["obs_image"])
    code_exemplar = next(trace["code_ids"][0] for trace in traces if trace["code_ids"])
    n = len(traces)
    obs_state = torch.zeros(n, max_steps, *state_exemplar.shape, dtype=state_exemplar.dtype)
    obs_image = torch.zeros(n, max_steps, *image_exemplar.shape, dtype=image_exemplar.dtype)
    code_ids = torch.zeros(n, max_steps, *code_exemplar.shape, dtype=code_exemplar.dtype)
    valid = torch.zeros(n, max_steps, dtype=torch.bool)
    for ix, trace in enumerate(traces):
        length = min(max_steps, len(trace["code_ids"]))
        for step_ix in range(length):
            obs_state[ix, step_ix] = trace["obs_state"][step_ix]
            obs_image[ix, step_ix] = trace["obs_image"][step_ix]
            code_ids[ix, step_ix] = trace["code_ids"][step_ix]
            valid[ix, step_ix] = True
    return obs_state, obs_image, code_ids, valid


def _code_head_modules(policy: nn.Module, sequentially_select: bool) -> list[nn.Module]:
    if sequentially_select:
        return [
            policy.vqbet.action_head.map_to_cbet_preds_primary_bin,
            policy.vqbet.action_head.map_to_cbet_preds_secondary_bin,
        ]
    return [policy.vqbet.action_head.map_to_cbet_preds_bin]


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
