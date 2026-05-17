#!/usr/bin/env python

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def compute_pass_at_k(successes: Any, ks: list[int] | tuple[int, ...] = (1, 2, 4, 8)) -> dict[str, float]:
    """Compute set-level pass@k from a matrix shaped (n_sets, attempts_per_set)."""
    success = np.asarray(successes, dtype=bool)
    if success.ndim == 1:
        success = success[:, None]
    if success.ndim != 2:
        raise ValueError("successes must have shape (n_sets, attempts_per_set).")

    out: dict[str, float] = {}
    for k in ks:
        k_eff = min(k, success.shape[1])
        out[f"pass@{k}"] = float(success[:, :k_eff].any(axis=1).mean())
    return out


def mean_confidence_interval(values: Any, confidence_z: float = 1.96) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": 1}
    half_width = confidence_z * float(arr.std(ddof=1)) / math.sqrt(arr.size)
    return {"mean": mean, "ci95_low": mean - half_width, "ci95_high": mean + half_width, "n": int(arr.size)}


def tensor_shape_summary(tensors: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in tensors.items() if isinstance(value, torch.Tensor)}


def summarize_rollout_attempts(
    *,
    rewards: torch.Tensor,
    successes: torch.Tensor,
    max_overlaps: torch.Tensor,
    code_diversity: torch.Tensor,
    action_diversity: torch.Tensor,
) -> dict[str, Any]:
    final_returns = rewards.sum(dim=-1)
    return {
        "avg_return": float(final_returns.float().mean().item()),
        "avg_success": float(successes.float().mean().item()),
        "avg_max_overlap": float(max_overlaps.float().mean().item()),
        "avg_code_diversity": float(code_diversity.float().mean().item()),
        "avg_action_diversity": float(action_diversity.float().mean().item()),
        "pass_at_k": compute_pass_at_k(successes.detach().cpu().numpy()),
        "return_ci95": mean_confidence_interval(final_returns.detach().cpu().reshape(-1).numpy()),
        "max_overlap_ci95": mean_confidence_interval(max_overlaps.detach().cpu().reshape(-1).numpy()),
    }
