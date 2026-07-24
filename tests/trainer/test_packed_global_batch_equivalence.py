"""Trainer-side equivalence between packing layouts.

These tests verify that the SFT and GRPO trainers' aggregated losses depend
only on the trainable tokens in a global batch, not on how those tokens are
split across packed samples. Combined with the iterator invariants in
``tests/data/test_global_batch_iterator.py``, this confirms that
``rollout_trace.to_packed_samples()`` no longer perturbs the gradient
magnitude.

The tests are CPU-only — they call ``trainer.loss_func`` directly on
synthetic batches rather than spinning up Megatron.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from axrl.configs import GrpoTrainerConfig, SftTrainerConfig
from axrl.trainer.grpo_trainer import GrpoTrainer
from axrl.trainer.sft_trainer import SftTrainer


def _sft_loss_for(log_prob: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    trainer = SftTrainer(SftTrainerConfig())
    batch = TensorDict(
        {
            "labels": torch.zeros_like(log_prob, dtype=torch.long),
            "loss_mask": loss_mask,
        },
        batch_size=log_prob.shape[0],
    )
    loss, _, _ = trainer.loss_func({"log_prob": log_prob}, batch)
    return loss


def test_sft_loss_is_invariant_to_packing_split() -> None:
    """One row of trainable tokens, vs the same tokens spread over two rows: same loss + same gradient."""
    log_prob = torch.linspace(-1.0, 1.0, steps=8).reshape(1, 8).clone()
    log_prob.requires_grad_()
    loss_mask = torch.tensor([[True, True, True, True, False, False, False, False]])
    one_row_loss = _sft_loss_for(log_prob, loss_mask)
    one_row_loss.backward()
    assert log_prob.grad is not None
    one_row_grad = log_prob.grad.detach().clone()

    split_log_prob = log_prob.detach().clone()
    split_log_prob.requires_grad_()
    # Same trainable tokens, just spread across two rows ⇒ identical token-mean.
    split_view = split_log_prob.view(2, 4)
    split_mask = loss_mask.view(2, 4)
    split_loss = _sft_loss_for(split_view, split_mask)
    split_loss.backward()
    assert split_log_prob.grad is not None
    torch.testing.assert_close(split_loss.detach(), one_row_loss.detach())
    torch.testing.assert_close(split_log_prob.grad.view(1, 8), one_row_grad)


def test_sft_loss_unaffected_by_zero_loss_padding_rows() -> None:
    """Appending zero-loss padding samples to a batch leaves the loss + gradient unchanged."""
    log_prob = torch.linspace(-0.5, 0.5, steps=6).reshape(2, 3).clone()
    log_prob.requires_grad_()
    loss_mask = torch.tensor([[True, True, False], [True, False, False]])

    base_loss = _sft_loss_for(log_prob, loss_mask)
    base_loss.backward()
    assert log_prob.grad is not None
    base_grad = log_prob.grad.detach().clone()

    padded_log_prob = torch.cat([log_prob.detach().clone(), torch.full((2, 3), 7.0)], dim=0)
    padded_log_prob.requires_grad_()
    padded_mask = torch.cat([loss_mask, torch.zeros(2, 3, dtype=torch.bool)], dim=0)

    padded_loss = _sft_loss_for(padded_log_prob, padded_mask)
    padded_loss.backward()
    assert padded_log_prob.grad is not None
    torch.testing.assert_close(padded_loss.detach(), base_loss.detach())
    torch.testing.assert_close(padded_log_prob.grad[:2], base_grad)
    # Zero-loss padding rows must produce zero gradient.
    assert torch.equal(padded_log_prob.grad[2:], torch.zeros_like(padded_log_prob.grad[2:]))


def _grpo_loss_for(
    log_prob: torch.Tensor,
    loss_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    trainer = GrpoTrainer(GrpoTrainerConfig(loss_type="grpo2", loss_agg_type="token-mean", kl_control_alpha=0.0))
    batch = TensorDict(
        {
            "loss_mask": loss_mask,
            "labels": torch.zeros_like(log_prob, dtype=torch.long),
            "advantage": advantages,
            "rollout_logprobs": torch.zeros_like(log_prob),
            "old_logprobs": torch.zeros_like(log_prob),
            "ref_logprobs": torch.zeros_like(log_prob),
        },
        batch_size=log_prob.shape[0],
    )
    loss, _, _ = trainer.loss_func({"log_prob": log_prob, "entropy": torch.zeros_like(log_prob)}, batch)
    return loss


@pytest.mark.parametrize("loss_agg_type", ["token-mean"])
def test_grpo_loss_unaffected_by_zero_loss_padding_rows(loss_agg_type: str) -> None:
    del loss_agg_type
    log_prob = torch.linspace(-0.01, 0.01, steps=12).reshape(3, 4).clone()
    log_prob.requires_grad_()
    loss_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, False],
        ]
    )
    advantages = torch.tensor(
        [
            [0.5, -0.25, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.75, 0.125, -0.5, 0.0],
        ]
    )

    base_loss = _grpo_loss_for(log_prob, loss_mask, advantages)
    base_loss.backward()
    assert log_prob.grad is not None
    base_grad = log_prob.grad.detach().clone()

    padded_log_prob = torch.cat([log_prob.detach().clone(), torch.full((2, 4), 3.0)], dim=0)
    padded_log_prob.requires_grad_()
    padded_mask = torch.cat([loss_mask, torch.zeros(2, 4, dtype=torch.bool)], dim=0)
    padded_advantages = torch.cat([advantages, torch.zeros(2, 4)], dim=0)

    padded_loss = _grpo_loss_for(padded_log_prob, padded_mask, padded_advantages)
    padded_loss.backward()
    assert padded_log_prob.grad is not None
    torch.testing.assert_close(padded_loss.detach(), base_loss.detach())
    torch.testing.assert_close(padded_log_prob.grad[:3], base_grad)
    assert torch.equal(padded_log_prob.grad[3:], torch.zeros_like(padded_log_prob.grad[3:]))


def test_grpo_icepop_masks_policy_loss_and_logs_mask_ratios() -> None:
    trainer = GrpoTrainer(
        GrpoTrainerConfig(
            loss_type="grpo",
            loss_agg_type="token-mean",
            kl_control_alpha=0.0,
            icepop_masking_low=0.5,
            icepop_masking_high=5.0,
        )
    )
    mismatch_ratio = torch.tensor([[0.25, 1.0, 10.0]])
    rollout_logprobs = torch.full_like(mismatch_ratio, -10.0)
    old_logprobs = rollout_logprobs + mismatch_ratio.log()
    log_prob = old_logprobs.clone()
    loss_mask = torch.ones_like(log_prob, dtype=torch.bool)
    advantages = torch.ones_like(log_prob)
    batch = TensorDict(
        {
            "loss_mask": loss_mask,
            "labels": torch.zeros_like(log_prob, dtype=torch.long),
            "advantage": advantages,
            "rollout_logprobs": rollout_logprobs,
            "old_logprobs": old_logprobs,
            "ref_logprobs": torch.zeros_like(log_prob),
        },
        batch_size=log_prob.shape[0],
    )

    loss, denom, metrics = trainer.loss_func({"log_prob": log_prob, "entropy": torch.zeros_like(log_prob)}, batch)

    torch.testing.assert_close(loss, torch.tensor(-1.0 / 3.0))
    assert denom.item() == 3
    assert metrics["icepop_mask_low_ratio__all"] == pytest.approx(1.0 / 3.0)
    assert metrics["icepop_mask_high_ratio__all"] == pytest.approx(1.0 / 3.0)
    assert metrics["icepop_mask_ratio__all"] == pytest.approx(2.0 / 3.0)
    assert metrics["icepop_keep_ratio__all"] == pytest.approx(1.0 / 3.0)
    assert metrics["policy_loss_mask_sum"] == pytest.approx(1.0)


def test_grpo_sequence_mask_only_gates_policy_gradient_not_kl() -> None:
    trainer = GrpoTrainer(
        GrpoTrainerConfig(
            loss_type="grpo",
            loss_agg_type="token-mean",
            kl_control_alpha=1.0,
            kl_base_logprobs="ref_logprobs",
            mismatch_seq_masking_high=0.5,
        )
    )
    log_prob = torch.ones(1, 3)
    loss_mask = torch.ones_like(log_prob, dtype=torch.bool)
    batch = TensorDict(
        {
            "loss_mask": loss_mask,
            "labels": torch.zeros_like(log_prob, dtype=torch.long),
            "advantage": torch.ones_like(log_prob),
            "rollout_logprobs": torch.zeros_like(log_prob),
            "old_logprobs": torch.zeros_like(log_prob),
            "ref_logprobs": torch.zeros_like(log_prob),
        },
        batch_size=log_prob.shape[0],
    )

    loss, denom, metrics = trainer.loss_func({"log_prob": log_prob, "entropy": torch.zeros_like(log_prob)}, batch)

    torch.testing.assert_close(loss, torch.tensor(0.5))
    assert denom.item() == 3
    assert metrics["kl_loss"] == pytest.approx(0.5)
    assert metrics["ratio_mean"] == pytest.approx(0.0)
    assert metrics["seq_mask_high_keep_rate__all"] == pytest.approx(0.0)
    assert metrics["policy_loss_mask_sum"] == pytest.approx(0.0)
    assert metrics["loss_mask_sum"] == pytest.approx(3.0)
