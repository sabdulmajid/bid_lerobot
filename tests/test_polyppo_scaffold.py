import torch
import pytest

from lerobot.common.policies.polyppo_scaffold import (
    RolloutBatch,
    assign_set_advantages,
    assign_polyppo_advantages,
    clipped_ppo_surrogate_loss,
    f_poly,
    kl_penalty,
    logprob_ratio,
    pairwise_l1_diversity,
    ppo_loss,
    polyppo_scores,
    quality_gated_scores,
    value_loss,
)


def test_rollout_batch_shape_contracts_for_n4_n8():
    n_sets = 4
    n_attempts = 8
    n_steps = 3
    returns = torch.randn(n_sets, n_attempts, n_steps)
    old_log_probs = torch.zeros_like(returns)
    values = torch.randn_like(returns)
    current_log_probs = torch.zeros_like(returns)

    batch = RolloutBatch(
        returns=returns,
        old_log_probs=old_log_probs,
        values=values,
        current_log_probs=current_log_probs,
    )

    assert batch.n_sets == n_sets
    assert batch.n_attempts == n_attempts
    assert batch.returns.shape == (n_sets, n_attempts, n_steps)


def test_f_poly_blends_returns_and_diversity():
    returns = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    diversity = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    out = f_poly(returns, diversity, poly_lambda=0.25)
    expected = torch.tensor(
        [
            [1.0, 2.25],
            [3.25, 4.0],
        ]
    )
    assert torch.allclose(out, expected)


def test_set_advantage_assignment_shape_and_set_normalization_for_n4_n8():
    n_sets = 4
    n_attempts = 8
    returns = torch.arange(n_sets * n_attempts, dtype=torch.float32).reshape(n_sets, n_attempts)
    normalized_adv = assign_set_advantages(returns, normalize=True)

    assert normalized_adv.shape == (n_sets, n_attempts)
    assert torch.allclose(normalized_adv.mean(dim=1), torch.zeros(n_sets), atol=1e-6)
    # With unbiased=False in std normalization, std of each set should be approx 1.
    assert torch.allclose(normalized_adv.std(dim=1, unbiased=False), torch.ones(n_sets), atol=1e-4)


def test_set_advantages_sum_to_zero_per_set():
    returns = torch.randn(4, 8)
    advantages = assign_set_advantages(returns, normalize=False)

    assert torch.allclose(advantages.sum(dim=1), torch.zeros(4), atol=1e-6)


def test_set_advantages_masked_sum_to_zero_per_set_and_pads_are_zero():
    returns = torch.tensor([[1.0, 2.0, 100.0], [4.0, 8.0, 12.0]])
    valid_mask = torch.tensor([[True, True, False], [True, False, True]])

    advantages = assign_set_advantages(returns, normalize=False, valid_mask=valid_mask)

    assert torch.equal(advantages[~valid_mask], torch.zeros_like(advantages[~valid_mask]))
    assert torch.allclose((advantages * valid_mask).sum(dim=1), torch.zeros(2), atol=1e-6)


def test_lambda_zero_matches_ppo_without_diversity_advantages():
    returns = torch.randn(4, 8)
    diversity = torch.randn(4, 8)

    without_diversity = assign_set_advantages(returns, diversity=None, normalize=True)
    lambda_zero = assign_set_advantages(returns, diversity=diversity, poly_lambda=0.0, normalize=True)

    assert torch.equal(lambda_zero, without_diversity)


def test_polyppo_lambda_zero_matches_return_only_ppo_for_code_and_action_modes():
    returns = torch.randn(4, 8)
    code_ids = torch.randint(0, 16, (4, 8, 2))
    action_preds = torch.randn(4, 8, 5, 2)

    return_only = assign_polyppo_advantages(returns, diversity_kind="none", poly_lambda=0.0)
    code_zero = assign_polyppo_advantages(
        returns,
        code_ids=code_ids,
        diversity_kind="code",
        poly_lambda=0.0,
    )
    action_zero = assign_polyppo_advantages(
        returns,
        action_preds=action_preds,
        diversity_kind="action",
        poly_lambda=0.0,
    )

    assert torch.equal(code_zero, return_only)
    assert torch.equal(action_zero, return_only)


def test_pairwise_l1_diversity_is_finite_for_degenerate_attempts():
    values = torch.zeros(3, 4, 2)

    diversity = pairwise_l1_diversity(values)

    assert torch.equal(diversity, torch.zeros(3, 4))


def test_polyppo_scores_require_matching_diversity_inputs():
    returns = torch.randn(2, 3)

    try:
        polyppo_scores(returns, diversity_kind="code", poly_lambda=0.3)
    except ValueError as exc:
        assert "code_ids" in str(exc)
    else:
        raise AssertionError("Expected missing code_ids to raise ValueError.")


def test_quality_gated_scores_reward_good_diversity_and_penalize_bad_diversity():
    returns = torch.tensor([[0.0, 1.0, 2.0]])
    diversity = torch.tensor([[10.0, 10.0, 10.0]])

    scores = quality_gated_scores(
        returns,
        diversity,
        poly_lambda=0.1,
        quality_gate="mean",
        lambda_bad=0.2,
    )

    assert torch.allclose(scores, torch.tensor([[-2.0, 2.0, 3.0]]))


def test_quality_gated_scores_uses_quantile_and_valid_mask_per_set():
    returns = torch.tensor([[0.0, 1.0, 100.0], [2.0, 4.0, 8.0]])
    diversity = torch.ones_like(returns)
    valid_mask = torch.tensor([[True, True, False], [True, True, True]])

    scores = quality_gated_scores(
        returns,
        diversity,
        poly_lambda=0.5,
        quality_gate="quantile",
        quality_quantile=0.75,
        valid_mask=valid_mask,
    )

    assert torch.allclose(scores[0], torch.tensor([0.0, 1.5, 100.0]))
    assert torch.allclose(scores[1], torch.tensor([2.0, 4.0, 8.5]))


def test_ppo_ratio_is_one_when_log_probs_are_unchanged():
    old_log_probs = torch.randn(4, 8)
    advantages = torch.randn(4, 8)
    ratio = logprob_ratio(old_log_probs, old_log_probs)

    assert torch.equal(ratio, torch.ones_like(ratio))
    assert torch.isfinite(clipped_ppo_surrogate_loss(old_log_probs, old_log_probs, advantages, 0.2))


def test_clipped_ppo_surrogate_loss_on_toy_n4_n8_data():
    n_sets = 4
    n_attempts = 8
    old = torch.log(torch.full((n_sets, n_attempts), 0.5))
    current = old + torch.randn(n_sets, n_attempts) * 0.1
    advantages = torch.linspace(-1.0, 1.0, n_sets * n_attempts).reshape(n_sets, n_attempts)
    clip_ratio = 0.2

    loss = clipped_ppo_surrogate_loss(current, old, advantages, clip_ratio)

    ratio = torch.exp(current - old)
    expected = -torch.minimum(
        ratio * advantages,
        torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages,
    ).mean()

    assert torch.allclose(loss, expected)


def test_clipped_ppo_surrogate_loss_valid_mask_excludes_padded_steps():
    old = torch.zeros(1, 3)
    current = torch.tensor([[0.0, 0.1, 10.0]])
    advantages = torch.tensor([[1.0, 2.0, 1000.0]])
    valid_mask = torch.tensor([[True, True, False]])
    clip_ratio = 0.2

    loss = clipped_ppo_surrogate_loss(current, old, advantages, clip_ratio, valid_mask=valid_mask)
    ratio = torch.exp(current[:, :2] - old[:, :2])
    expected = -torch.minimum(
        ratio * advantages[:, :2],
        torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[:, :2],
    ).mean()

    assert torch.allclose(loss, expected)


def test_value_loss_and_kl_penalty_smoke_for_n4_n8():
    n_sets = 4
    n_attempts = 8
    values = torch.arange(n_sets * n_attempts, dtype=torch.float32).reshape(n_sets, n_attempts)
    returns = values + torch.tensor(0.5)

    expected_value_loss = ((values - returns) ** 2).mean()
    assert value_loss(values, returns).item() == expected_value_loss.item()

    probs_old = torch.full((n_sets, n_attempts), 0.125)
    probs_current = probs_old + torch.linspace(-0.002, 0.002, n_sets * n_attempts).reshape(n_sets, n_attempts)
    probs_current = probs_current / probs_current.sum(dim=1, keepdim=True)
    old_log_probs = torch.log(probs_old)
    current_log_probs = torch.log(probs_current)

    expected_kl = torch.mean(old_log_probs - current_log_probs)
    assert torch.allclose(kl_penalty(old_log_probs, current_log_probs), expected_kl)


def test_value_loss_and_kl_penalty_respect_valid_mask():
    values = torch.tensor([[1.0, 10.0, 1000.0]])
    returns = torch.tensor([[2.0, 12.0, -1000.0]])
    old_log_probs = torch.tensor([[0.0, -0.1, 99.0]])
    current_log_probs = torch.tensor([[0.2, -0.3, -99.0]])
    valid_mask = torch.tensor([[True, True, False]])

    assert torch.allclose(value_loss(values, returns, valid_mask=valid_mask), torch.tensor(2.5))
    assert torch.allclose(kl_penalty(old_log_probs, current_log_probs, valid_mask=valid_mask), torch.tensor(0.0))


def test_ppo_loss_combines_policy_value_entropy_and_base_kl():
    n_sets = 4
    n_attempts = 8
    returns = torch.randn(n_sets, n_attempts)
    old_log_probs = torch.randn(n_sets, n_attempts)
    current_log_probs = old_log_probs + 0.05
    values = returns * 0.5
    advantages = assign_set_advantages(returns)
    entropy = torch.ones_like(returns) * 0.7
    base_log_probs = old_log_probs - 0.02
    batch = RolloutBatch(
        returns=returns,
        old_log_probs=old_log_probs,
        values=values,
        current_log_probs=current_log_probs,
    )

    out = ppo_loss(
        batch,
        advantages,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy=entropy,
        entropy_coef=0.01,
        base_log_probs=base_log_probs,
        kl_coef=0.1,
    )

    expected = out.policy_loss + 0.5 * out.value_loss - 0.01 * out.entropy_bonus + 0.1 * out.kl_to_base
    assert torch.allclose(out.total_loss, expected)
    assert torch.isfinite(out.approx_kl)
    assert torch.isfinite(out.clip_fraction)


def test_ppo_loss_requires_base_log_probs_when_kl_coef_positive():
    returns = torch.randn(2, 3)
    batch = RolloutBatch(
        returns=returns,
        old_log_probs=torch.zeros_like(returns),
        values=torch.zeros_like(returns),
        current_log_probs=torch.zeros_like(returns),
    )
    advantages = assign_set_advantages(returns)

    with pytest.raises(ValueError, match="base_log_probs"):
        ppo_loss(batch, advantages, clip_ratio=0.2, kl_coef=0.1)
