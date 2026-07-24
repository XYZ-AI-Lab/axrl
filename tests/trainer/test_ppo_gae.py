from __future__ import annotations

import torch

from axrl.trainer.ppo_utils import build_terminal_token_rewards, compute_gae, normalize_over_valid_tokens_in_batch


def test_build_terminal_token_rewards_places_score_on_last_valid_token() -> None:
    scores = torch.tensor([1.5, 2.5])
    loss_mask = torch.tensor(
        [
            [False, True, True, False],
            [False, False, False, False],
        ]
    )

    rewards = build_terminal_token_rewards(scores, loss_mask)

    torch.testing.assert_close(
        rewards,
        torch.tensor(
            [
                [0.0, 0.0, 1.5, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_compute_gae_terminal_reward_with_zero_values() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    values = torch.zeros_like(rewards)
    loss_mask = torch.tensor([[False, True, True, False]])

    advantages, returns = compute_gae(rewards, values, loss_mask, gamma=1.0, gae_lambda=1.0)

    expected = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_compute_gae_respects_gamma_lambda_and_terminal_bootstrap() -> None:
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.2, 0.4]])
    loss_mask = torch.tensor([[True, True]])

    advantages, returns = compute_gae(rewards, values, loss_mask, gamma=0.5, gae_lambda=0.5)

    torch.testing.assert_close(advantages, torch.tensor([[0.15, 0.6]]))
    torch.testing.assert_close(returns, torch.tensor([[0.35, 1.0]]))


def test_compute_gae_resets_across_mask_gaps() -> None:
    rewards = torch.tensor([[1.0, 0.0, 2.0]])
    values = torch.zeros_like(rewards)
    loss_mask = torch.tensor([[True, False, True]])

    advantages, returns = compute_gae(rewards, values, loss_mask, gamma=1.0, gae_lambda=1.0)

    torch.testing.assert_close(advantages, torch.tensor([[1.0, 0.0, 2.0]]))
    torch.testing.assert_close(returns, torch.tensor([[1.0, 0.0, 2.0]]))


def test_normalize_over_valid_tokens_in_batch_uses_one_batch_statistic() -> None:
    values = torch.tensor([[1.0, 2.0, 100.0], [3.0, -100.0, -200.0]])
    loss_mask = torch.tensor([[True, True, False], [True, False, False]])

    normalized = normalize_over_valid_tokens_in_batch(values, loss_mask)

    valid = values[loss_mask]
    expected = (values - valid.mean()) * torch.rsqrt(valid.var(unbiased=False) + 1e-8)
    expected = expected.masked_fill(~loss_mask, 0.0)
    torch.testing.assert_close(normalized, expected)


def test_normalize_over_valid_tokens_in_batch_returns_zero_without_valid_tokens() -> None:
    values = torch.tensor([[1.0, 2.0]])
    loss_mask = torch.tensor([[False, False]])

    normalized = normalize_over_valid_tokens_in_batch(values, loss_mask)

    torch.testing.assert_close(normalized, torch.zeros_like(values))
