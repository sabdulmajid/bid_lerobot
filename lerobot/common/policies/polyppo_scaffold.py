#!/usr/bin/env python

"""Utility scaffolding for a minimal Polychromic PPO-style objective.

This module intentionally keeps implementation limited to pure tensor helpers so that it can be
unit-tested without touching the existing training loop.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RolloutBatch:
    """Container for a batch of rollout statistics used by PPO-like updates.

    The rollout tensor layout is intentionally explicit about set semantics:
    - first dimension: number of start states (`n`)
    - second dimension: number of attempts per start state (`N`)
    - remaining dimensions: trajectory or token dimensions (for example time horizon).

    The module requires that all tensor fields share identical shape.
    """

    returns: Tensor
    old_log_probs: Tensor
    values: Tensor
    current_log_probs: Tensor

    def __post_init__(self):
        if self.returns.ndim < 2:
            raise ValueError("RolloutBatch tensors must have at least shape (n, N).")

        if self.current_log_probs.shape != self.returns.shape:
            raise ValueError("`current_log_probs` must match `returns` shape.")
        if self.old_log_probs.shape != self.returns.shape:
            raise ValueError("`old_log_probs` must match `returns` shape.")
        if self.values.shape != self.returns.shape:
            raise ValueError("`values` must match `returns` shape.")

    @property
    def n_sets(self) -> int:
        return self.returns.shape[0]

    @property
    def n_attempts(self) -> int:
        return self.returns.shape[1]


@dataclass
class PPOLossOutput:
    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_bonus: Tensor
    kl_to_base: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


def f_poly(returns: Tensor, diversity: Tensor, poly_lambda: float = 0.0) -> Tensor:
    """Add an auxiliary diversity score to reward returns.

    This follows the PolyPPO score used for the PushT experiments:
    `score_i = return_i + lambda_div * diversity_i`.
    When `poly_lambda=0.0`, this is equivalent to returns-only ranking.
    """
    if returns.shape != diversity.shape:
        raise ValueError("`returns` and `diversity` must have identical shape.")
    if poly_lambda < 0.0:
        raise ValueError("`poly_lambda` must be non-negative.")
    return returns + poly_lambda * diversity


def pairwise_l1_diversity(values: Tensor) -> Tensor:
    """Average L1 distance from each attempt to other attempts in the same set."""
    if values.ndim < 2:
        raise ValueError("`values` must have at least shape (n_sets, n_attempts).")
    n_attempts = values.shape[1]
    if n_attempts <= 1:
        return torch.zeros(values.shape[:2], dtype=values.dtype, device=values.device)

    flat = values.reshape(values.shape[0], values.shape[1], -1).float()
    pairwise = (flat.unsqueeze(2) - flat.unsqueeze(1)).abs().mean(dim=-1)
    eye = torch.eye(n_attempts, dtype=torch.bool, device=values.device).unsqueeze(0)
    pairwise = pairwise.masked_fill(eye, 0.0)
    return pairwise.sum(dim=2) / (n_attempts - 1)


def polyppo_scores(
    returns: Tensor,
    *,
    code_ids: Tensor | None = None,
    action_preds: Tensor | None = None,
    diversity_kind: str = "none",
    poly_lambda: float = 0.0,
    quality_gate: str = "none",
    quality_quantile: float = 0.5,
    lambda_bad: float = 0.0,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Build the per-attempt score used for set-normalized PolyPPO advantages."""
    base_kind = diversity_kind
    if base_kind in {"none", "return_only"}:
        return returns
    if base_kind.startswith("quality_bad_"):
        base_kind = base_kind.removeprefix("quality_bad_")
        if quality_gate == "none":
            quality_gate = "mean"
    elif base_kind.startswith("quality_"):
        base_kind = base_kind.removeprefix("quality_")
        if quality_gate == "none":
            quality_gate = "mean"

    if base_kind == "code":
        if code_ids is None:
            raise ValueError("code_ids are required for code diversity.")
        diversity = pairwise_l1_diversity(code_ids.float())
    elif base_kind == "action":
        if action_preds is None:
            raise ValueError("action_preds are required for action diversity.")
        diversity = pairwise_l1_diversity(action_preds.float())
    else:
        raise ValueError(
            "diversity_kind must be one of {'none', 'return_only', 'code', 'action', "
            "'quality_code', 'quality_action', 'quality_bad_code', 'quality_bad_action'}."
        )
    return quality_gated_scores(
        returns,
        diversity,
        poly_lambda=poly_lambda,
        quality_gate=quality_gate,
        quality_quantile=quality_quantile,
        lambda_bad=lambda_bad,
        valid_mask=valid_mask,
    )


def quality_gated_scores(
    returns: Tensor,
    diversity: Tensor,
    *,
    poly_lambda: float = 0.0,
    quality_gate: str = "none",
    quality_quantile: float = 0.5,
    lambda_bad: float = 0.0,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Add diversity only to high-quality attempts and optionally penalize bad diversity.

    `returns` and `diversity` are per-attempt tensors with shape `(n_sets, n_attempts)`.
    Quality is computed within each set so the objective rewards coverage among attempts
    that are competitive for the same restored prefix instead of rewarding off-manifold
    variety uniformly.
    """
    if returns.shape != diversity.shape:
        raise ValueError("`returns` and `diversity` must have identical shape.")
    if poly_lambda < 0.0:
        raise ValueError("`poly_lambda` must be non-negative.")
    if lambda_bad < 0.0:
        raise ValueError("`lambda_bad` must be non-negative.")
    if valid_mask is not None and valid_mask.shape != returns.shape:
        raise ValueError("`valid_mask` must match return shape.")
    if quality_gate not in {"none", "mean", "quantile"}:
        raise ValueError("quality_gate must be one of {'none', 'mean', 'quantile'}.")
    if not (0.0 <= quality_quantile <= 1.0):
        raise ValueError("quality_quantile must be between 0 and 1.")

    if quality_gate == "none":
        good = torch.ones_like(returns, dtype=torch.bool)
        bad = torch.zeros_like(good)
    else:
        threshold = _quality_threshold(
            returns,
            quality_gate=quality_gate,
            quality_quantile=quality_quantile,
            valid_mask=valid_mask,
        )
        good = returns >= threshold
        bad = returns < threshold

    if valid_mask is not None:
        valid = valid_mask.to(dtype=torch.bool, device=returns.device)
        good = good & valid
        bad = bad & valid

    score = returns + poly_lambda * diversity * good.to(dtype=returns.dtype)
    if lambda_bad > 0.0:
        score = score - lambda_bad * diversity * bad.to(dtype=returns.dtype)
    return score


def _quality_threshold(
    returns: Tensor,
    *,
    quality_gate: str,
    quality_quantile: float,
    valid_mask: Tensor | None,
) -> Tensor:
    if quality_gate == "mean":
        if valid_mask is None:
            return returns.mean(dim=1, keepdim=True)
        valid = valid_mask.to(dtype=returns.dtype, device=returns.device)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (returns * valid).sum(dim=1, keepdim=True) / count

    thresholds = []
    valid_bool = (
        torch.ones_like(returns, dtype=torch.bool, device=returns.device)
        if valid_mask is None
        else valid_mask.to(dtype=torch.bool, device=returns.device)
    )
    for row, valid_row in zip(returns, valid_bool, strict=True):
        valid_values = row[valid_row]
        if valid_values.numel() == 0:
            thresholds.append(row.new_tensor(0.0))
        else:
            thresholds.append(torch.quantile(valid_values.float(), quality_quantile).to(dtype=row.dtype))
    return torch.stack(thresholds).view(returns.shape[0], 1)


def assign_set_advantages(
    returns: Tensor,
    diversity: Tensor | None = None,
    poly_lambda: float = 0.0,
    normalize: bool = True,
    eps: float = 1e-8,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Assign PPO-style advantages within each set of N attempts.

    Advantages are computed as centered and optionally variance-normalized scores where the score itself
    can include diversity via `f_poly`.
    """
    scores = returns if diversity is None else f_poly(returns, diversity, poly_lambda=poly_lambda)
    if valid_mask is None:
        centered = scores - scores.mean(dim=1, keepdim=True)
        if not normalize:
            return centered
        denom = centered.std(dim=1, keepdim=True, unbiased=False)
        return centered / (denom + eps)

    if valid_mask.shape != scores.shape:
        raise ValueError("`valid_mask` must match score shape.")
    valid = valid_mask.to(dtype=scores.dtype, device=scores.device)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (scores * valid).sum(dim=1, keepdim=True) / count
    centered = (scores - mean) * valid
    if not normalize:
        return centered
    denom = torch.sqrt(((centered**2) * valid).sum(dim=1, keepdim=True) / count)
    return (centered / (denom + eps)) * valid


def assign_polyppo_advantages(
    returns: Tensor,
    *,
    code_ids: Tensor | None = None,
    action_preds: Tensor | None = None,
    diversity_kind: str = "none",
    poly_lambda: float = 0.0,
    quality_gate: str = "none",
    quality_quantile: float = 0.5,
    lambda_bad: float = 0.0,
    normalize: bool = True,
    valid_mask: Tensor | None = None,
) -> Tensor:
    scores = polyppo_scores(
        returns,
        code_ids=code_ids,
        action_preds=action_preds,
        diversity_kind=diversity_kind,
        poly_lambda=poly_lambda,
        quality_gate=quality_gate,
        quality_quantile=quality_quantile,
        lambda_bad=lambda_bad,
        valid_mask=valid_mask,
    )
    return assign_set_advantages(scores, normalize=normalize, valid_mask=valid_mask)


def logprob_ratio(current_log_probs: Tensor, old_log_probs: Tensor) -> Tensor:
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("`current_log_probs` and `old_log_probs` must have identical shape.")
    return torch.exp(current_log_probs - old_log_probs)


def clipped_ppo_surrogate_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_ratio: float,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute the PPO clipped surrogate objective (negative sign for minimization)."""
    if not (0.0 < clip_ratio < 1.0):
        raise ValueError("`clip_ratio` must be between 0 and 1.")
    if not (current_log_probs.shape == old_log_probs.shape == advantages.shape):
        raise ValueError("`current_log_probs`, `old_log_probs`, and `advantages` must share shape.")
    ratio = torch.exp(current_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    loss = -torch.minimum(unclipped, clipped)
    return _masked_mean(loss, valid_mask)


def ppo_loss(
    batch: RolloutBatch,
    advantages: Tensor,
    *,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy: Tensor | None = None,
    entropy_coef: float = 0.0,
    base_log_probs: Tensor | None = None,
    kl_coef: float = 0.0,
    valid_mask: Tensor | None = None,
) -> PPOLossOutput:
    if advantages.shape != batch.returns.shape:
        raise ValueError("`advantages` must match rollout batch shape.")
    if valid_mask is not None and valid_mask.shape != batch.returns.shape:
        raise ValueError("`valid_mask` must match rollout batch shape.")
    policy_loss = clipped_ppo_surrogate_loss(
        batch.current_log_probs,
        batch.old_log_probs,
        advantages,
        clip_ratio,
        valid_mask=valid_mask,
    )
    value = value_loss(batch.values, batch.returns, valid_mask=valid_mask)

    if entropy is None:
        entropy_bonus = torch.zeros((), dtype=batch.returns.dtype, device=batch.returns.device)
    else:
        if entropy.shape != batch.returns.shape:
            raise ValueError("`entropy` must match rollout batch shape.")
        entropy_bonus = _masked_mean(entropy, valid_mask)

    if base_log_probs is None:
        if kl_coef > 0.0:
            raise ValueError("`base_log_probs` are required when `kl_coef` is positive.")
        base_kl = torch.zeros((), dtype=batch.returns.dtype, device=batch.returns.device)
    else:
        if base_log_probs.shape != batch.current_log_probs.shape:
            raise ValueError("`base_log_probs` must match current log-prob shape.")
        base_kl = kl_penalty(base_log_probs, batch.current_log_probs, valid_mask=valid_mask)

    ratio = logprob_ratio(batch.current_log_probs, batch.old_log_probs)
    approx_kl = kl_penalty(batch.old_log_probs, batch.current_log_probs, valid_mask=valid_mask)
    clip_fraction = _masked_mean(((ratio - 1.0).abs() > clip_ratio).float(), valid_mask)
    total = policy_loss + value_coef * value - entropy_coef * entropy_bonus + kl_coef * base_kl
    return PPOLossOutput(
        total_loss=total,
        policy_loss=policy_loss,
        value_loss=value,
        entropy_bonus=entropy_bonus,
        kl_to_base=base_kl,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


def value_loss(values: Tensor, returns: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Return MSE value loss."""
    if values.shape != returns.shape:
        raise ValueError("`values` and `returns` must have identical shape.")
    squared_error = (values - returns) ** 2
    return _masked_mean(squared_error, valid_mask)


def kl_penalty(old_log_probs: Tensor, current_log_probs: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Sample-based KL estimate KL(old || current)."""
    if old_log_probs.shape != current_log_probs.shape:
        raise ValueError("`old_log_probs` and `current_log_probs` must have identical shape.")
    return _masked_mean(old_log_probs - current_log_probs, valid_mask)


def _masked_mean(values: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    if valid_mask is None:
        return values.mean()
    if valid_mask.shape != values.shape:
        raise ValueError("`valid_mask` must match tensor shape.")
    valid = valid_mask.to(dtype=values.dtype, device=values.device)
    count = valid.sum()
    if count.item() <= 0:
        raise ValueError("`valid_mask` must contain at least one valid entry.")
    return (values * valid).sum() / count
