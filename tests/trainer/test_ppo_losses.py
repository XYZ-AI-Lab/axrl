from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from axrl.configs import GrpoTrainerConfig, PPOValueConfig
from axrl.trainer.grpo_trainer import GrpoTrainer
from axrl.trainer.ppo_utils import clipped_value_loss_per_token
from axrl.trainer.value_trainer import ValueTrainer


def test_grpo_trainer_ppo_loss_uses_clipped_policy_ratio() -> None:
    trainer = GrpoTrainer(
        GrpoTrainerConfig(
            loss_type="ppo",
            clip_ratio_high=0.2,
            clip_ratio_low=0.2,
            dual_clip_neg_adv_factor=None,
            kl_control_alpha=0.0,
            is_base_logprobs="old_logprobs",
        )
    )
    log_prob = torch.log(torch.tensor([[1.5, 0.5]]))
    loss_mask = torch.tensor([[True, True]])
    batch = TensorDict(
        {
            "loss_mask": loss_mask,
            "labels": torch.zeros_like(log_prob, dtype=torch.long),
            "advantage": torch.tensor([[1.0, -1.0]]),
            "rollout_logprobs": torch.zeros_like(log_prob),
            "old_logprobs": torch.zeros_like(log_prob),
            "ref_logprobs": torch.zeros_like(log_prob),
        },
        batch_size=1,
    )

    loss, denom, metrics = trainer.loss_func({"log_prob": log_prob, "entropy": torch.zeros_like(log_prob)}, batch)

    torch.testing.assert_close(loss, torch.tensor(-0.2))
    assert denom.item() == 2
    assert metrics["pg_loss"] == pytest.approx(-0.2)
    assert metrics["ppo_policy_loss"] == pytest.approx(-0.2)
    assert metrics["clip_frac__all"] == pytest.approx(1.0)


def test_grpo_trainer_ppo_requires_old_logprobs_as_base() -> None:
    trainer = GrpoTrainer(GrpoTrainerConfig(loss_type="ppo", is_base_logprobs="rollout_logprobs"))

    with pytest.raises(AssertionError, match="old_logprobs"):
        trainer.get_loss_func()


def test_clipped_value_loss_per_token_uses_more_conservative_objective() -> None:
    values = torch.tensor([[1.5]])
    old_values = torch.tensor([[1.0]])
    returns = torch.tensor([[2.0]])

    losses, clipfrac = clipped_value_loss_per_token(values, old_values, returns, value_clip=0.2)

    torch.testing.assert_close(losses, torch.tensor([[0.64]]))
    assert clipfrac is not None
    torch.testing.assert_close(clipfrac, torch.tensor([[1.0]]))


def test_value_clipfrac_tracks_value_movement_not_selected_loss_branch() -> None:
    values = torch.tensor([[1.3]])
    old_values = torch.tensor([[1.0]])
    returns = torch.tensor([[0.0]])

    losses, clipfrac = clipped_value_loss_per_token(values, old_values, returns, value_clip=0.2)

    torch.testing.assert_close(losses, torch.tensor([[1.69]]))
    assert clipfrac is not None
    torch.testing.assert_close(clipfrac, torch.tensor([[1.0]]))


def test_clipped_value_loss_can_be_disabled() -> None:
    values = torch.tensor([[1.5]])
    old_values = torch.tensor([[1.0]])
    returns = torch.tensor([[2.0]])

    losses, clipfrac = clipped_value_loss_per_token(values, old_values, returns, value_clip=None)

    torch.testing.assert_close(losses, torch.tensor([[0.25]]))
    assert clipfrac is None


def test_value_trainer_loss_func_masks_and_logs_metrics() -> None:
    trainer = ValueTrainer(PPOValueConfig(value_clip=0.2, value_loss_coef=0.5))
    values = torch.tensor([[1.5, 1.0, 100.0]])
    batch = TensorDict(
        {
            "loss_mask": torch.tensor([[True, True, False]]),
            "old_values": torch.tensor([[1.0, 1.0, 0.0]]),
            "returns": torch.tensor([[2.0, 1.25, 0.0]]),
        },
        batch_size=1,
    )

    loss, denom, metrics = trainer.loss_func({"values": values.unsqueeze(-1)}, batch)

    torch.testing.assert_close(loss, torch.tensor((0.64 + 0.0625) / 2 * 0.5))
    assert denom.item() == 2
    assert metrics["value_loss"] == pytest.approx(float(loss.item()))
    assert metrics["value_clipfrac"] == pytest.approx(0.5)
    assert metrics["value_mean"] == pytest.approx(1.25)
    assert metrics["return_mean"] == pytest.approx(1.625)


def test_value_trainer_no_loss_tokens_does_not_emit_nan() -> None:
    trainer = ValueTrainer(PPOValueConfig())
    values = torch.tensor([[1.0, 2.0]])
    batch = TensorDict(
        {
            "loss_mask": torch.tensor([[False, False]]),
            "old_values": torch.zeros_like(values),
            "returns": torch.zeros_like(values),
        },
        batch_size=1,
    )

    loss, denom, metrics = trainer.loss_func({"values": values}, batch)

    assert denom.item() == 0
    assert math.isfinite(float(loss.item()))
    assert all(math.isfinite(float(value)) for value in metrics.values())
