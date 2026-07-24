from __future__ import annotations

import torch


def build_terminal_token_rewards(
    scores: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Build token rewards with the scalar score placed on the final valid token."""
    mask = loss_mask.bool()
    rewards = torch.zeros(mask.shape, device=scores.device, dtype=torch.float32)
    mask = mask.to(device=scores.device)
    lengths = mask.long().sum(dim=1)
    has_tokens = lengths > 0
    if bool(has_tokens.any()):
        batch_idx = torch.arange(scores.shape[0], device=scores.device)[has_tokens]
        positions = torch.arange(mask.shape[1], device=scores.device).expand_as(mask)
        last_idx = positions.masked_fill(~mask, 0).amax(dim=1)[has_tokens]
        rewards[batch_idx, last_idx] += scores.float()[has_tokens]
    return rewards.masked_fill(~mask, 0.0)


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute tokenwise GAE advantages and returns over valid loss tokens."""
    assert rewards.shape == values.shape == loss_mask.shape, (
        f"GAE shape mismatch: rewards={tuple(rewards.shape)}, values={tuple(values.shape)}, loss_mask={tuple(loss_mask.shape)}"
    )
    rewards = rewards.float()
    values = values.float()
    mask = loss_mask.bool()

    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(seq_len)):
        if t + 1 < seq_len:
            next_value = values[:, t + 1]
            next_valid = mask[:, t + 1].float()
        else:
            next_value = torch.zeros_like(last_gae)
            next_valid = torch.zeros_like(last_gae)

        valid = mask[:, t].float()
        delta = rewards[:, t] + gamma * next_value * next_valid - values[:, t]
        last_gae = delta + gamma * gae_lambda * last_gae * next_valid
        last_gae = last_gae * valid
        advantages[:, t] = last_gae

    returns = (advantages + values).masked_fill(~mask, 0.0)
    advantages = advantages.masked_fill(~mask, 0.0)
    return advantages, returns


def normalize_over_valid_tokens_in_batch(values: torch.Tensor, loss_mask: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Normalize one tensor using one mean/std over all valid tokens in the batch."""
    assert values.shape == loss_mask.shape, f"values shape {tuple(values.shape)} must match loss_mask shape {tuple(loss_mask.shape)}"
    mask = loss_mask.bool()
    if not bool(mask.any()):
        return torch.zeros_like(values)
    valid_values = values.float().masked_select(mask)
    mean = valid_values.mean()
    var = valid_values.var(unbiased=False)
    normalized = (values.float() - mean) * torch.rsqrt(var + eps)
    return normalized.masked_fill(~mask, 0.0)


def clipped_value_loss_per_token(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    *,
    value_clip: float | None = 0.2,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return PPO's per-token clipped value loss.

    ``old_values`` anchors the critic prediction from rollout/value-forward time.
    When clipping is enabled, the critic is penalized by the larger squared
    error between the raw prediction and the prediction clipped to stay within
    ``value_clip`` of that anchor. This discourages large critic shifts on one
    PPO update while ``clipfrac`` reports how often the raw shift exceeded the
    clip range.
    """
    assert values.shape == old_values.shape == returns.shape, (
        f"value-loss shape mismatch: values={tuple(values.shape)}, old_values={tuple(old_values.shape)}, returns={tuple(returns.shape)}"
    )
    values = values.float()
    old_values = old_values.float()
    returns = returns.float()
    loss_unclipped = (values - returns).pow(2)
    if value_clip is None:
        return loss_unclipped, None

    clipfrac = (torch.abs(values - old_values) > value_clip).to(torch.float32)
    values_clipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
    loss_clipped = (values_clipped - returns).pow(2)
    loss = torch.maximum(loss_unclipped, loss_clipped)
    return loss, clipfrac
