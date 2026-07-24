from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

import axrl.trainer.grpo_trainer as grpo_trainer_module
from axrl.configs import EntropyControlConfig, GrpoTrainerConfig, ModelConfig, OPDConfig
from axrl.trainer.grpo_trainer import GrpoTrainer


def test_logits_processor_skips_entropy_when_entropy_control_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_entropy(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("entropy helper should not be called when entropy alpha is disabled")

    def fake_log_prob(vocab_parallel_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(target, dtype=vocab_parallel_logits.dtype)

    monkeypatch.setattr(grpo_trainer_module, "vocab_parallel_entropy", fail_entropy)
    monkeypatch.setattr(grpo_trainer_module, "vocab_parallel_log_prob", fake_log_prob)

    trainer = GrpoTrainer(GrpoTrainerConfig())
    labels = torch.zeros((2, 3), dtype=torch.long)
    outputs = trainer.logits_processor(torch.randn(2, 3, 5), labels)

    assert torch.equal(outputs["entropy"], torch.zeros_like(labels, dtype=torch.float32))
    assert torch.equal(outputs["log_prob"], torch.zeros_like(labels, dtype=torch.float32))


def test_logits_processor_computes_entropy_when_metric_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_entropy(logits: torch.Tensor) -> torch.Tensor:
        return torch.full(logits.shape[:2], 7.0, dtype=logits.dtype, device=logits.device)

    def fake_log_prob(vocab_parallel_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(target, dtype=vocab_parallel_logits.dtype)

    monkeypatch.setattr(grpo_trainer_module, "vocab_parallel_entropy", fake_entropy)
    monkeypatch.setattr(grpo_trainer_module, "vocab_parallel_log_prob", fake_log_prob)

    trainer = GrpoTrainer(GrpoTrainerConfig(entropy_control=EntropyControlConfig(compute_entropy=True)))
    labels = torch.zeros((2, 3), dtype=torch.long)
    outputs = trainer.logits_processor(torch.randn(2, 3, 5), labels)

    assert torch.equal(outputs["entropy"], torch.full_like(labels, 7.0, dtype=torch.float32))
    assert torch.equal(outputs["log_prob"], torch.zeros_like(labels, dtype=torch.float32))


def test_logits_processor_computes_entropy_when_entropy_control_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_entropy(logits: torch.Tensor) -> torch.Tensor:
        return torch.full(logits.shape[:2], 3.0, dtype=logits.dtype, device=logits.device)

    def fake_log_prob(vocab_parallel_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(target, dtype=vocab_parallel_logits.dtype)

    monkeypatch.setattr(grpo_trainer_module, "vocab_parallel_entropy", fake_entropy)
    monkeypatch.setattr(grpo_trainer_module, "vocab_parallel_log_prob", fake_log_prob)

    trainer = GrpoTrainer(GrpoTrainerConfig(entropy_control=EntropyControlConfig(alpha=0.1)))
    labels = torch.zeros((2, 3), dtype=torch.long)
    outputs = trainer.logits_processor(torch.randn(2, 3, 5), labels)

    assert torch.equal(outputs["entropy"], torch.full_like(labels, 3.0, dtype=torch.float32))
    assert torch.equal(outputs["log_prob"], torch.zeros_like(labels, dtype=torch.float32))


def _trainer(
    *,
    opd_alpha: float = 0.5,
    reverse_kl_clip: float | None = 10.0,
    normalize_student_logprob_scale: bool = False,
) -> GrpoTrainer:
    return GrpoTrainer(
        GrpoTrainerConfig(
            opd=OPDConfig(
                enabled=True,
                backend="megatron",
                teacher_model=ModelConfig(name="teacher"),
                opd_alpha=opd_alpha,
                reverse_kl_clip=reverse_kl_clip,
                normalize_student_logprob_scale=normalize_student_logprob_scale,
            )
        )
    )


def test_opd_disabled_leaves_advantages_unchanged() -> None:
    trainer = GrpoTrainer(GrpoTrainerConfig())
    advantages = torch.tensor([[1.0, 0.0]])
    adjusted, metrics = trainer.apply_opd_to_advantages(
        advantages=advantages,
        loss_mask=torch.tensor([[True, False]]),
        cur_logprobs=torch.tensor([[-1.0, 0.0]]),
        rollout_logprobs=torch.tensor([[-1.0, 0.0]]),
        old_logprobs=torch.tensor([[-1.0, 0.0]]),
        teacher_logprobs=torch.tensor([[-2.0, 0.0]]),
    )

    assert torch.equal(adjusted, advantages)
    assert metrics == {}


def test_relative_reverse_kl_adjusts_advantages() -> None:
    trainer = _trainer(opd_alpha=1.0)

    adjusted, metrics = trainer.apply_opd_to_advantages(
        advantages=torch.tensor([[1.0, -1.0]]),
        loss_mask=torch.tensor([[True, True]]),
        cur_logprobs=torch.tensor([[-1.0, -3.0]]),
        rollout_logprobs=torch.tensor([[-1.0, -3.0]]),
        old_logprobs=torch.tensor([[-1.0, -3.0]]),
        teacher_logprobs=torch.tensor([[-2.0, -2.0]]),
    )

    assert torch.allclose(adjusted, torch.tensor([[-1.0, 1.0]]))
    assert metrics["opd/reverse_kl_mean"] == pytest.approx(0.0)
    assert metrics["opd/reverse_kl_std"] == pytest.approx(1.0)
    assert metrics["opd/reverse_kl_abs_mean"] == pytest.approx(1.0)


def test_opd_reverse_kl_clipping() -> None:
    trainer = _trainer(opd_alpha=1.0, reverse_kl_clip=2.0)

    adjusted, metrics = trainer.apply_opd_to_advantages(
        advantages=torch.tensor([[1.0, -1.0]]),
        loss_mask=torch.tensor([[True, True]]),
        cur_logprobs=torch.tensor([[10.0, 0.0]]),
        rollout_logprobs=torch.tensor([[10.0, 0.0]]),
        old_logprobs=torch.tensor([[10.0, 0.0]]),
        teacher_logprobs=torch.tensor([[0.0, 0.0]]),
    )

    assert torch.allclose(adjusted, torch.tensor([[-2.0, 0.0]]))
    assert metrics["opd/reverse_kl_mean"] == pytest.approx(1.0)
    assert metrics["opd/reverse_kl_clip_rate"] == pytest.approx(0.5)


def test_opd_alpha_blends_grpo_advantage_and_teacher_reverse_kl() -> None:
    common = dict(
        advantages=torch.tensor([[1.0, -1.0]]),
        loss_mask=torch.tensor([[True, True]]),
        cur_logprobs=torch.tensor([[-1.0, -3.0]]),
        rollout_logprobs=torch.tensor([[-1.0, -3.0]]),
        old_logprobs=torch.tensor([[-1.0, -3.0]]),
        teacher_logprobs=torch.tensor([[-2.0, -2.0]]),
    )

    pure_grpo, _ = _trainer(opd_alpha=0.0).apply_opd_to_advantages(**common)
    pure_opd, _ = _trainer(opd_alpha=1.0).apply_opd_to_advantages(**common)

    assert torch.allclose(pure_grpo, torch.tensor([[1.0, -1.0]]))
    assert torch.allclose(pure_opd, torch.tensor([[-1.0, 1.0]]))


def test_pure_opd_uses_raw_teacher_signal_when_reward_advantages_are_tiny() -> None:
    adjusted, metrics = _trainer(opd_alpha=1.0).apply_opd_to_advantages(
        advantages=torch.tensor([[0.1, -0.1]]),
        loss_mask=torch.tensor([[True, True]]),
        cur_logprobs=torch.tensor([[-1.0, -3.0]]),
        rollout_logprobs=torch.tensor([[-1.0, -3.0]]),
        old_logprobs=torch.tensor([[-1.0, -3.0]]),
        teacher_logprobs=torch.tensor([[-2.0, -2.0]]),
    )

    assert torch.allclose(adjusted, torch.tensor([[-1.0, 1.0]]))
    assert metrics["opd/base_advantage_std"] == pytest.approx(0.1)


def test_opd_can_use_current_logprobs_as_student_source() -> None:
    trainer = _trainer(opd_alpha=1.0)
    trainer.config.opd.student_logprob_source = "cur_logprobs"

    adjusted, metrics = trainer.apply_opd_to_advantages(
        advantages=torch.tensor([[1.0, -1.0]]),
        loss_mask=torch.tensor([[True, True]]),
        cur_logprobs=torch.tensor([[-1.0, -3.0]]),
        rollout_logprobs=torch.tensor([[-5.0, -5.0]]),
        old_logprobs=torch.tensor([[-5.0, -5.0]]),
        teacher_logprobs=torch.tensor([[-2.0, -2.0]]),
    )

    assert torch.allclose(adjusted, torch.tensor([[-1.0, 1.0]]))
    assert metrics["opd/reverse_kl_mean"] == pytest.approx(0.0)


def test_opd_reports_student_teacher_abs_logprob_scale_ratio() -> None:
    trainer = _trainer()
    trainer.config.opd.student_logprob_source = "cur_logprobs"

    _, metrics = trainer.apply_opd_to_advantages(
        advantages=torch.tensor([[1.0, -1.0, 100.0]]),
        loss_mask=torch.tensor([[True, True, False]]),
        cur_logprobs=torch.tensor([[-2.0, -4.0, -100.0]]),
        rollout_logprobs=torch.tensor([[-10.0, -10.0, -100.0]]),
        old_logprobs=torch.tensor([[-10.0, -10.0, -100.0]]),
        teacher_logprobs=torch.tensor([[-1.0, -3.0, -100.0]]),
    )

    assert metrics["opd/student_teacher_logprob_abs_mean_ratio"] == pytest.approx(1.5)
    assert metrics["opd/raw_student_teacher_logprob_abs_mean_ratio"] == pytest.approx(1.5)
    assert metrics["opd/student_logprob_scale"] == pytest.approx(1.0)
    assert metrics["opd/base_advantage_std"] == pytest.approx(1.0)


def test_opd_can_normalize_student_logprob_scale_to_teacher() -> None:
    trainer = _trainer(opd_alpha=1.0, normalize_student_logprob_scale=True)
    trainer.config.opd.student_logprob_source = "cur_logprobs"

    adjusted, metrics = trainer.apply_opd_to_advantages(
        advantages=torch.tensor([[5.0, 5.0, 999.0]]),
        loss_mask=torch.tensor([[True, True, False]]),
        cur_logprobs=torch.tensor([[-2.0, -6.0, -1000.0]]),
        rollout_logprobs=torch.tensor([[-2.0, -6.0, -1000.0]]),
        old_logprobs=torch.tensor([[-2.0, -6.0, -1000.0]]),
        teacher_logprobs=torch.tensor([[-1.0, -3.0, -1000.0]]),
    )

    assert torch.allclose(adjusted, torch.tensor([[0.0, 0.0, 999.0]]))
    assert metrics["opd/student_logprob_scale"] == pytest.approx(0.5)
    assert metrics["opd/raw_student_teacher_logprob_abs_mean_ratio"] == pytest.approx(2.0)
    assert metrics["opd/student_teacher_logprob_abs_mean_ratio"] == pytest.approx(1.0)
    assert metrics["opd/reverse_kl_abs_mean"] == pytest.approx(0.0)


def test_matching_teacher_student_logprobs_have_zero_reverse_kl() -> None:
    adjusted, metrics = _trainer(opd_alpha=0.5).apply_opd_to_advantages(
        advantages=torch.tensor([[2.0]]),
        loss_mask=torch.tensor([[True]]),
        cur_logprobs=torch.tensor([[-1.0]]),
        rollout_logprobs=torch.tensor([[-1.0]]),
        old_logprobs=torch.tensor([[-1.0]]),
        teacher_logprobs=torch.tensor([[-1.0]]),
    )

    assert torch.allclose(adjusted, torch.tensor([[1.0]]))
    assert metrics["opd/reverse_kl_mean"] == pytest.approx(0.0)


def test_opd_ignores_non_loss_teacher_logprobs_and_rejects_loss_nan() -> None:
    trainer = _trainer()

    adjusted, metrics = trainer.apply_opd_to_advantages(
        advantages=torch.tensor([[2.0, 0.0, 5.0]]),
        loss_mask=torch.tensor([[True, True, False]]),
        cur_logprobs=torch.tensor([[-1.0, -3.0, float("nan")]]),
        rollout_logprobs=torch.tensor([[-1.0, -3.0, float("nan")]]),
        old_logprobs=torch.tensor([[-1.0, -3.0, float("nan")]]),
        teacher_logprobs=torch.tensor([[-2.0, -2.0, float("nan")]]),
    )

    assert torch.allclose(adjusted, torch.tensor([[0.5, 0.5, 5.0]]))
    assert metrics["opd/reverse_kl_mean"] == pytest.approx(0.0)
    assert metrics["opd/teacher_logprobs_present_rate"] == pytest.approx(1.0)

    with pytest.raises(AssertionError, match="finite"):
        trainer.apply_opd_to_advantages(
            advantages=torch.tensor([[1.0]]),
            loss_mask=torch.tensor([[True]]),
            cur_logprobs=torch.tensor([[-1.0]]),
            rollout_logprobs=torch.tensor([[-1.0]]),
            old_logprobs=torch.tensor([[-1.0]]),
            teacher_logprobs=torch.tensor([[float("nan")]]),
        )


def test_loss_func_requires_teacher_logprobs_when_opd_enabled() -> None:
    trainer = _trainer()
    outputs = {
        "log_prob": torch.zeros((1, 2)),
        "entropy": torch.zeros((1, 2)),
    }
    batch = TensorDict(
        {
            "loss_mask": torch.tensor([[True, False]]),
            "labels": torch.tensor([[1, -100]]),
            "advantage": torch.tensor([[1.0, 0.0]]),
            "rollout_logprobs": torch.zeros((1, 2)),
            "old_logprobs": torch.zeros((1, 2)),
            "ref_logprobs": torch.zeros((1, 2)),
        },
        batch_size=1,
    )

    with pytest.raises(AssertionError, match="missing teacher_logprobs"):
        trainer.loss_func(outputs, batch)
