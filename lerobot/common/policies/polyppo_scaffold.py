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
) -> Tensor:
    """Build the per-attempt score used for set-normalized PolyPPO advantages."""
    if diversity_kind == "none":
        return returns
    if diversity_kind == "code":
        if code_ids is None:
            raise ValueError("code_ids are required for code diversity.")
        diversity = pairwise_l1_diversity(code_ids.float())
    elif diversity_kind == "action":
        if action_preds is None:
            raise ValueError("action_preds are required for action diversity.")
        diversity = pairwise_l1_diversity(action_preds.float())
    else:
        raise ValueError("diversity_kind must be one of {'none', 'code', 'action'}.")
    return f_poly(returns, diversity, poly_lambda=poly_lambda)


def assign_set_advantages(
    returns: Tensor,
    diversity: Tensor | None = None,
    poly_lambda: float = 0.0,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Assign PPO-style advantages within each set of N attempts.

    Advantages are computed as centered and optionally variance-normalized scores where the score itself
    can include diversity via `f_poly`.
    """
    scores = returns if diversity is None else f_poly(returns, diversity, poly_lambda=poly_lambda)
    centered = scores - scores.mean(dim=1, keepdim=True)
    if not normalize:
        return centered
    denom = centered.std(dim=1, keepdim=True, unbiased=False)
    return centered / (denom + eps)


def assign_polyppo_advantages(
    returns: Tensor,
    *,
    code_ids: Tensor | None = None,
    action_preds: Tensor | None = None,
    diversity_kind: str = "none",
    poly_lambda: float = 0.0,
    normalize: bool = True,
) -> Tensor:
    scores = polyppo_scores(
        returns,
        code_ids=code_ids,
        action_preds=action_preds,
        diversity_kind=diversity_kind,
        poly_lambda=poly_lambda,
    )
    return assign_set_advantages(scores, normalize=normalize)


def logprob_ratio(current_log_probs: Tensor, old_log_probs: Tensor) -> Tensor:
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("`current_log_probs` and `old_log_probs` must have identical shape.")
    return torch.exp(current_log_probs - old_log_probs)


def clipped_ppo_surrogate_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_ratio: float,
) -> Tensor:
    """Compute the PPO clipped surrogate objective (negative sign for minimization)."""
    if not (0.0 < clip_ratio < 1.0):
        raise ValueError("`clip_ratio` must be between 0 and 1.")
    if not (current_log_probs.shape == old_log_probs.shape == advantages.shape):
        raise ValueError("`current_log_probs`, `old_log_probs`, and `advantages` must share shape.")
    ratio = torch.exp(current_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.minimum(unclipped, clipped).mean()


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
) -> PPOLossOutput:
    if advantages.shape != batch.returns.shape:
        raise ValueError("`advantages` must match rollout batch shape.")
    policy_loss = clipped_ppo_surrogate_loss(
        batch.current_log_probs,
        batch.old_log_probs,
        advantages,
        clip_ratio,
    )
    value = value_loss(batch.values, batch.returns)

    if entropy is None:
        entropy_bonus = torch.zeros((), dtype=batch.returns.dtype, device=batch.returns.device)
    else:
        if entropy.shape != batch.returns.shape:
            raise ValueError("`entropy` must match rollout batch shape.")
        entropy_bonus = entropy.mean()

    if base_log_probs is None:
        base_kl = torch.zeros((), dtype=batch.returns.dtype, device=batch.returns.device)
    else:
        if base_log_probs.shape != batch.current_log_probs.shape:
            raise ValueError("`base_log_probs` must match current log-prob shape.")
        base_kl = kl_penalty(base_log_probs, batch.current_log_probs)

    ratio = logprob_ratio(batch.current_log_probs, batch.old_log_probs)
    approx_kl = kl_penalty(batch.old_log_probs, batch.current_log_probs)
    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
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


def value_loss(values: Tensor, returns: Tensor) -> Tensor:
    """Return MSE value loss."""
    if values.shape != returns.shape:
        raise ValueError("`values` and `returns` must have identical shape.")
    return nn.functional.mse_loss(values, returns)


def kl_penalty(old_log_probs: Tensor, current_log_probs: Tensor) -> Tensor:
    """Sample-based KL estimate KL(old || current)."""
    if old_log_probs.shape != current_log_probs.shape:
        raise ValueError("`old_log_probs` and `current_log_probs` must have identical shape.")
    return torch.mean(old_log_probs - current_log_probs)
