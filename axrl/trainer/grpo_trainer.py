import logging
from collections.abc import Iterator
from functools import partial
from typing import Protocol, override

import torch
from megatron.core.models.gpt import GPTModel
from tensordict import TensorDict

from axrl.configs import GrpoTrainerConfig, ValueAggType
from axrl.trainer.base_trainer import BaseTrainer
from axrl.utils import kl_utils
from axrl.utils.logger import LoggerBuffer
from axrl.utils.megatron.model_forward import GPTModelForwardFn
from axrl.utils.megatron.prefix_tree import extract_merge_info_from_batch
from axrl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_prob, vocab_parallel_top_k_mask
from axrl.utils.timer import Timer

logger = logging.getLogger(__name__)


def get_num_turns(turn_index: torch.Tensor, mask: torch.Tensor) -> int:
    """Count unique assistant turns within ``mask`` across the batch.

    ``turn_index`` is per-token; assistant tokens carry a non-negative turn id,
    non-assistant tokens carry -1. Returns the total number of unique non-negative
    turn ids summed across sequences (not batched-unique across sequences).
    """
    total = 0
    for i in range(turn_index.shape[0]):
        seq_turns = turn_index[i][mask[i]]
        total += int(seq_turns[seq_turns >= 0].unique().numel())
    return total


class LossFunc(Protocol):
    """Only for typing-hints of loss function."""

    def __call__(
        self,
        *,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        base_logprobs: torch.Tensor,
        token_mismatch_tir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class GrpoTrainer(BaseTrainer):
    def __init__(self, config: GrpoTrainerConfig) -> None:
        super().__init__()
        self.config = config

    def compute_micro_batch_denominator(self, mask: torch.Tensor, num_turns: int = 0) -> torch.Tensor:
        denominator_type = self.config.micro_batch_denominator_type
        assert denominator_type in ("sequence", "token", "seq_turn"), f"Unknown micro_batch_denominator_type: {denominator_type}"

        if denominator_type == "sequence":
            token_count = torch.sum(mask, dim=-1)
            valid_seq = token_count > 0
            return torch.sum(valid_seq, dtype=torch.int)

        if denominator_type == "seq_turn":
            token_count = torch.sum(mask, dim=-1)
            valid_seq = token_count > 0
            num_sequences = torch.sum(valid_seq, dtype=torch.float)
            alpha = self.config.seq_turn_alpha if self.config.seq_turn_alpha is not None else self.config.turn_reward_alpha
            return (num_sequences + alpha * num_turns).to(torch.int)

        return torch.sum(mask, dtype=torch.int)

    def set_metric_agg_type(self, logger_buffer: LoggerBuffer) -> None:
        pass

    def group_metrics(
        self, values: torch.Tensor, prefix: str, mask: torch.Tensor, adv: torch.Tensor, mode: ValueAggType = "token-mean"
    ) -> dict[str, float]:
        """Group metrics by positive and negative advantages."""
        assert adv.shape == mask.shape, f"adv/mask shape mismatch: {adv.shape} vs {mask.shape}"
        pos_adv_mask = (adv > 0) & mask
        neg_adv_mask = (adv < 0) & mask
        result: dict[str, float] = {
            f"{prefix}__all": self.aggregate_with_mask(values, mask, mode=mode).item(),
        }
        if pos_adv_mask.any():
            result[f"{prefix}__pos"] = self.aggregate_with_mask(values, pos_adv_mask, mode=mode).item()
        if neg_adv_mask.any():
            result[f"{prefix}__neg"] = self.aggregate_with_mask(values, neg_adv_mask, mode=mode).item()
        return result

    def calculate_metrics(
        self,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        rollout_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        entropy: torch.Tensor,
        ratio: torch.Tensor,
        clipped_ratio: torch.Tensor,
        control_entropy: float,
        pg_loss: float,
        entropy_loss: float,
        kl_loss: float,
        loss: float,
    ) -> dict[str, float]:
        """Metrics for GRPO trainer."""

        def add_adv_split_stats(
            values: torch.Tensor, prefix: str, all_mask: torch.Tensor | None = None, mode: ValueAggType = "token-mean"
        ) -> dict[str, float]:
            if all_mask is None:
                all_mask = loss_mask
            return self.group_metrics(
                values=values,
                prefix=prefix,
                mask=all_mask,
                adv=advantages,
                mode=mode,
            )

        agg = self.aggregate_with_mask

        def add_stats(values: torch.Tensor, prefix: str, mask: torch.Tensor | None = None) -> None:
            """Add mean, std, min, max, p05, p95 for a metric."""
            m = mask if mask is not None else loss_mask
            metrics[f"{prefix}_mean"] = agg(values, m, mode="token-mean").item()
            metrics[f"{prefix}_std"] = agg(values, m, mode="token-std").item()
            metrics[f"{prefix}_min"] = agg(values, m, mode="token-min").item()
            metrics[f"{prefix}_max"] = agg(values, m, mode="token-max").item()
            metrics[f"{prefix}_p05"] = agg(values, m, mode="token-p05").item()
            metrics[f"{prefix}_p95"] = agg(values, m, mode="token-p95").item()

        num_samples = advantages.shape[0]
        probs = torch.exp(logprobs)
        perplexity = torch.exp(-agg(logprobs, loss_mask, mode="token-mean"))
        ratio_mismatch = self.get_ratio(logprobs=old_logprobs, base_logprobs=rollout_logprobs)

        metrics: dict[str, float] = {
            "pg_loss": pg_loss,
            "entropy_loss": entropy_loss,
            "kl_loss": kl_loss,
            "loss": loss,
            "control_entropy": control_entropy,
            "entropy_target": self.config.entropy_control.target_entropy,
            "entropy_alpha": self.config.entropy_control.alpha,
            "entropy_top_quantile": self.config.entropy_control.top_quantile,
            "kl_control_alpha": self.config.kl_control_alpha,
            "num_samples": num_samples,
            "perplexity": perplexity.item(),
        }
        if self.config.loss_type == "ppo":
            metrics["ppo_policy_loss"] = pg_loss

        # ---- Clipping metrics (pos/neg advantage split) ----
        metrics.update(add_adv_split_stats((clipped_ratio - ratio).abs() > 1e-4, "clip_frac", mode="token-mean"))
        metrics.update(add_adv_split_stats(ratio - clipped_ratio > 1e-4, "clip_frac_high", mode="token-mean"))
        metrics.update(add_adv_split_stats(clipped_ratio - ratio > 1e-4, "clip_frac_low", mode="token-mean"))

        # ---- Scalar metrics (mean/std/min/max/p05/p95, no pos/neg split) ----
        add_stats(entropy, "entropy")
        add_stats(ratio, "ratio")
        add_stats(probs, "prob")
        add_stats(ratio_mismatch, "ratio_mismatch")

        # Importance sampling ratio to probability (binned)
        is_ranges = [(x * 0.1, (x + 2) * 0.1) for x in range(0, 18, 2)]
        is_ranges.append((2.0, float("100")))
        for lower, upper in is_ranges:
            is_mask = (ratio >= lower) & (ratio < upper) & loss_mask
            if is_mask.any():
                metrics[f"prob_with_ratio_{lower:.1f}_{upper:.1f}"] = agg(probs, is_mask, mode="token-mean").item()

        # Approx KL divergence (k2)
        add_stats(kl_utils.kl_divergence(rollout_logprobs, old_logprobs, kl_type="k1"), "kl_k1_rollout_old")
        add_stats(kl_utils.kl_divergence(rollout_logprobs, old_logprobs, kl_type="k3"), "kl_k3_rollout_old")
        add_stats(kl_utils.kl_divergence(rollout_logprobs, old_logprobs, kl_type="k2"), "kl_k2_rollout_old")
        add_stats(kl_utils.kl_divergence(rollout_logprobs, logprobs, kl_type="k2"), "kl_k2_rollout_cur")
        add_stats(kl_utils.kl_divergence(old_logprobs, logprobs, kl_type="k2"), "kl_k2_old_cur")
        add_stats(kl_utils.kl_divergence(ref_logprobs, logprobs, kl_type="k2"), "kl_k2_ref_cur")

        # Intra-sequence ratio std
        metrics["intra_seq_ratio_std"] = agg(ratio, loss_mask, mode="seq-mean-token-std").item()

        return metrics

    def estimate_clipped_ratio(self, pg_losses: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        clipped_ratio = torch.where(
            advantages.abs() < 1e-4,
            torch.ones_like(pg_losses),
            pg_losses / (-advantages),
        )
        return clipped_ratio

    def grpo_loss(
        self,
        *,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        base_logprobs: torch.Tensor,
        token_mismatch_tir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_ratio = logprobs - base_logprobs
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)

        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio_low, 1 + self.config.clip_ratio_high)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * clipped_ratio
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # original PPO clip loss

        if self.config.dual_clip_neg_adv_factor is None:
            pg_losses = clip_pg_losses1
        else:
            # clip on negative advantages (dual-clip PPO)
            assert self.config.dual_clip_neg_adv_factor > 1.0
            pg_losses3 = -self.config.dual_clip_neg_adv_factor * advantages
            clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
            pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

        final_clipped_ratio = self.estimate_clipped_ratio(pg_losses, advantages)
        pg_losses = pg_losses * token_mismatch_tir
        pg_loss = self.aggregate_with_mask(pg_losses, loss_mask, mode=self.config.loss_agg_type)
        return pg_loss, ratio, final_clipped_ratio

    def tis_loss(
        self,
        *,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        base_logprobs: torch.Tensor,
        token_mismatch_tir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reinforce with truncated importance sampling.

        Reference:
        - https://arxiv.org/pdf/2510.13786
        - https://arxiv.org/abs/2506.13585
        """
        log_ratio = logprobs - base_logprobs
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio).detach()
        loss, clipped_ratio = self.calculate_pg_loss_and_clip_ratio(
            ratio=ratio,
            logprobs=logprobs,
            advantages=advantages,
            mask=loss_mask,
            clip_high=1 + self.config.clip_ratio_high,
            clip_low=1 - self.config.clip_ratio_low,
            hard_clip=False,
        )
        loss = loss * token_mismatch_tir
        pg_loss = self.aggregate_with_mask(loss, loss_mask, mode=self.config.loss_agg_type)
        return pg_loss, ratio, clipped_ratio

    def kimi2_5_loss(
        self,
        *,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        base_logprobs: torch.Tensor,
        token_mismatch_tir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Similar to TIS, but with hard clipping.

        Reference:
        - https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf
        """
        log_ratio = logprobs - base_logprobs
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio).detach()
        loss, clipped_ratio = self.calculate_pg_loss_and_clip_ratio(
            ratio=ratio,
            logprobs=logprobs,
            advantages=advantages,
            mask=loss_mask,
            clip_high=1 + self.config.clip_ratio_high,
            clip_low=1 - self.config.clip_ratio_low,
            hard_clip=True,
        )
        loss = loss * token_mismatch_tir
        pg_loss = self.aggregate_with_mask(loss, loss_mask, mode=self.config.loss_agg_type)
        return pg_loss, ratio, clipped_ratio

    def topr_loss(
        self,
        *,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        base_logprobs: torch.Tensor,
        token_mismatch_tir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_ratio = logprobs - base_logprobs
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio).detach()

        # Case 1: Adv >= 0
        loss1, clipped_ratio1 = self.calculate_pg_loss_and_clip_ratio(
            ratio=ratio,
            logprobs=logprobs,
            advantages=advantages,
            mask=(advantages >= 0) & loss_mask,
            clip_high=1,
            clip_low=1,
            hard_clip=False,
        )

        # Case 2: Adv < 0
        loss2, clipped_ratio2 = self.calculate_pg_loss_and_clip_ratio(
            ratio=ratio,
            logprobs=logprobs,
            advantages=advantages,
            mask=(advantages < 0) & loss_mask,
            clip_high=1,
            clip_low=0,
            hard_clip=False,
        )

        loss = loss1 + loss2
        clipped_ratio = clipped_ratio1 + clipped_ratio2
        loss = loss * token_mismatch_tir
        pg_loss = self.aggregate_with_mask(loss, loss_mask, mode=self.config.loss_agg_type)
        return pg_loss, ratio, clipped_ratio

    def calculate_pg_loss_and_clip_ratio(
        self,
        ratio: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        clip_high: float | None,
        clip_low: float | None,
        *,
        hard_clip: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_low = 0.0 if clip_low is None else clip_low
        clipped_ratio = torch.clamp(ratio, min=clip_low, max=clip_high).detach()
        is_clipped = (ratio - clipped_ratio).abs() > 1e-5
        loss = -advantages * clipped_ratio * logprobs
        if hard_clip:
            loss = torch.where(is_clipped, loss.detach(), loss)
        loss = torch.where(mask, loss, torch.zeros_like(loss))
        clipped_ratio = torch.where(mask, clipped_ratio, torch.zeros_like(clipped_ratio))
        return loss, clipped_ratio

    def gspo(
        self,
        *,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        base_logprobs: torch.Tensor,
        token_mismatch_tir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        negative_approx_kl = logprobs - base_logprobs
        seq_lengths = torch.sum(loss_mask, dim=-1).clamp(min=1)  # B
        negative_approx_kl_seq = torch.sum(negative_approx_kl * loss_mask, dim=-1) / seq_lengths  # B
        log_seq_ratio = logprobs - logprobs.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
        log_seq_ratio = torch.clamp(log_seq_ratio, max=10.0)
        seq_ratio = torch.exp(log_seq_ratio)
        pg_loss1 = -advantages * seq_ratio
        #  In GSPO, we set the left and right clipping ranges in Equation(5) to 3e-4 and 4e-4,respectively.
        pg_loss2 = -advantages * torch.clamp(seq_ratio, max=1 + self.config.clip_ratio_high, min=1 - self.config.clip_ratio_low)
        pg_losses = torch.maximum(pg_loss1, pg_loss2)
        final_clipped_ratio = self.estimate_clipped_ratio(pg_losses, advantages)
        pg_losses = pg_losses * token_mismatch_tir
        assert self.config.loss_agg_type == "seq-mean-token-mean", "GSPO only supports 'seq-mean-token-mean' aggregation."
        pg_loss = self.aggregate_with_mask(pg_losses, loss_mask, mode=self.config.loss_agg_type)
        return pg_loss, seq_ratio, final_clipped_ratio

    def get_ratio(self, logprobs: torch.Tensor, base_logprobs: torch.Tensor) -> torch.Tensor:
        log_ratio = logprobs - base_logprobs
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)
        return ratio

    def grpo2_loss(
        self,
        *,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        base_logprobs: torch.Tensor,
        token_mismatch_tir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GRPO2 loss function."""
        ratio = self.get_ratio(logprobs=logprobs, base_logprobs=base_logprobs).detach()
        assert self.config.dual_soft_clip is not None

        # Case 1: Adv >= 0, Ratio > 1. Tokens were previously encouraged.
        # Hard-clip gradients if tokens deviate too far from the original model.
        loss1, clipped_ratio1 = self.calculate_pg_loss_and_clip_ratio(
            ratio=ratio,
            logprobs=logprobs,
            advantages=advantages,
            mask=(advantages >= 0) & (ratio > 1) & loss_mask,
            clip_high=1 + self.config.clip_ratio_high,
            clip_low=None,
            hard_clip=True,
        )

        # Case 2: Adv >= 0, Ratio <= 1. Tokens were previously punished.
        # positive_weight >= 1. Use soft-clip to allow recovery towards the original model (Ratio -> 1).
        loss2, clipped_ratio2 = self.calculate_pg_loss_and_clip_ratio(
            ratio=ratio,
            logprobs=logprobs,
            advantages=advantages,
            mask=(advantages >= 0) & (ratio <= 1) & loss_mask,
            clip_high=None,
            clip_low=0.0,  # NO clip at low side
            hard_clip=False,
        )

        # Case 3: Adv < 0, Ratio > 1. Tokens were previously encouraged.
        # Use soft-clip to allow correction towards the original model (Ratio -> 1).
        loss3, clipped_ratio3 = self.calculate_pg_loss_and_clip_ratio(
            ratio=ratio,
            logprobs=logprobs,
            advantages=advantages,
            mask=(advantages < 0) & (ratio > 1) & loss_mask,
            clip_high=self.config.dual_soft_clip,
            clip_low=None,
            hard_clip=False,
        )

        # Case 4: Adv < 0, Ratio <= 1. Tokens were previously punished.
        # Hard-clip gradients if tokens deviate too far from the original model.
        loss4, clipped_ratio4 = self.calculate_pg_loss_and_clip_ratio(
            ratio=ratio,
            logprobs=logprobs,
            advantages=advantages,
            mask=(advantages < 0) & (ratio <= 1) & loss_mask,
            clip_high=None,
            clip_low=1 - self.config.clip_ratio_low,
            hard_clip=True,
        )

        loss = loss1 + loss2 + loss3 + loss4
        clipped_ratio = clipped_ratio1 + clipped_ratio2 + clipped_ratio3 + clipped_ratio4
        loss = loss * token_mismatch_tir
        pg_loss = self.aggregate_with_mask(loss, loss_mask, mode=self.config.loss_agg_type)

        return pg_loss, ratio, clipped_ratio

    def get_top_entropy_mask(self, loss_mask: torch.Tensor, entropy: torch.Tensor, top_quantile: float) -> torch.Tensor:
        """Select top-entropy tokens per sequence."""
        assert 0.0 < top_quantile < 1.0
        entropy_vals = entropy.detach().clone()
        entropy_vals[~loss_mask] = torch.finfo(entropy.dtype).min

        valid_lens = loss_mask.sum(dim=1)
        ks = torch.ceil(valid_lens.float() * top_quantile).to(torch.long)
        ks = torch.clamp(ks, min=1)

        max_k: int = ks.max().item()  # type: ignore
        topk_vals, _ = torch.topk(entropy_vals, k=max_k, dim=1)

        # Gather the threshold value for each row (k_i-th largest)
        row_indices = torch.arange(topk_vals.size(0), device=topk_vals.device)
        threshold = topk_vals[row_indices, ks - 1]  # (B,)
        peak_mask = (entropy_vals >= threshold.unsqueeze(1)) & loss_mask
        return peak_mask

    def get_entropy_loss(self, entropy: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.config.entropy_control.top_quantile < 1.0:
            top_quantile = self.config.entropy_control.top_quantile
            loss_mask = self.get_top_entropy_mask(loss_mask=loss_mask, entropy=entropy, top_quantile=top_quantile)

        batch_entropy = self.aggregate_with_mask(entropy, loss_mask, mode=self.config.loss_agg_type)

        if self.config.entropy_control.alpha <= 0.0:
            return torch.tensor(0.0, device=entropy.device), batch_entropy.item()

        entropy_loss = (self.config.entropy_control.target_entropy - batch_entropy) ** 2
        return entropy_loss, batch_entropy.item()

    def get_kl_loss(self, logprobs: torch.Tensor, base_logprobs: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        kl_div = kl_utils.kl_divergence(logprobs, base_logprobs, kl_type="k2")
        kl_loss = self.aggregate_with_mask(kl_div, loss_mask, mode=self.config.loss_agg_type)
        return kl_loss

    def apply_opd_to_advantages(
        self,
        *,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        cur_logprobs: torch.Tensor,
        rollout_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        opd_config = self.config.opd
        if not opd_config.enabled:
            return advantages, {}

        assert teacher_logprobs.shape == advantages.shape == loss_mask.shape, (
            f"OPD tensor shape mismatch: teacher={teacher_logprobs.shape}, advantages={advantages.shape}, mask={loss_mask.shape}"
        )
        mask = loss_mask.to(torch.bool)
        assert torch.isfinite(teacher_logprobs[mask]).all(), "OPD teacher_logprobs must be finite over loss_mask."
        if opd_config.student_logprob_source == "rollout_logprobs":
            student_logprobs = rollout_logprobs
        elif opd_config.student_logprob_source == "old_logprobs":
            student_logprobs = old_logprobs
        else:
            assert opd_config.student_logprob_source == "cur_logprobs"
            student_logprobs = cur_logprobs
        assert torch.isfinite(student_logprobs[mask]).all(), "OPD student_logprobs must be finite over loss_mask."

        agg = self.aggregate_with_mask
        raw_student_logprob_abs_mean = agg(student_logprobs.abs(), loss_mask, mode="token-mean")
        raw_teacher_logprob_abs_mean = agg(teacher_logprobs.abs(), loss_mask, mode="token-mean")
        student_logprob_scale = torch.tensor(1.0, device=student_logprobs.device, dtype=student_logprobs.dtype)
        if opd_config.normalize_student_logprob_scale:
            student_logprob_scale = raw_teacher_logprob_abs_mean / raw_student_logprob_abs_mean.clamp_min(1e-6)
            student_logprobs = student_logprobs * student_logprob_scale.detach()

        reverse_kl = torch.zeros_like(advantages)
        reverse_kl[mask] = (student_logprobs[mask] - teacher_logprobs[mask]).detach()
        unclipped_reverse_kl = reverse_kl
        if opd_config.reverse_kl_clip is not None:
            reverse_kl = reverse_kl.clamp(-opd_config.reverse_kl_clip, opd_config.reverse_kl_clip)

        adjusted_advantages = (1.0 - opd_config.opd_alpha) * advantages - opd_config.opd_alpha * reverse_kl
        adjusted_advantages = torch.where(mask, adjusted_advantages, advantages)

        reverse_kl_abs = reverse_kl.abs()
        reverse_kl_mean = agg(reverse_kl, loss_mask, mode="token-mean")
        reverse_kl_std = agg(reverse_kl, loss_mask, mode="token-std")
        clip_rate = (reverse_kl != unclipped_reverse_kl).to(torch.float32)
        masked_teacher = torch.zeros_like(teacher_logprobs)
        masked_teacher[mask] = teacher_logprobs[mask]
        masked_student = torch.zeros_like(student_logprobs)
        masked_student[mask] = student_logprobs[mask]
        present = torch.zeros_like(advantages)
        present[mask] = torch.isfinite(teacher_logprobs[mask]).to(torch.float32)
        advantage_delta = adjusted_advantages - advantages
        base_advantage_mean = agg(advantages, loss_mask, mode="token-mean")
        base_advantage_std = agg(advantages, loss_mask, mode="token-std")
        student_logprob_abs_mean = agg(masked_student.abs(), loss_mask, mode="token-mean")
        teacher_logprob_abs_mean = agg(masked_teacher.abs(), loss_mask, mode="token-mean")
        metrics = {
            "opd/reverse_kl_mean": reverse_kl_mean.item(),
            "opd/reverse_kl_std": reverse_kl_std.item(),
            "opd/reverse_kl_abs_mean": agg(reverse_kl_abs, loss_mask, mode="token-mean").item(),
            "opd/reverse_kl_p05": agg(reverse_kl, loss_mask, mode="token-p05").item(),
            "opd/reverse_kl_p95": agg(reverse_kl, loss_mask, mode="token-p95").item(),
            "opd/reverse_kl_clip_rate": agg(clip_rate, loss_mask, mode="token-mean").item(),
            "opd/teacher_logprob_mean": agg(masked_teacher, loss_mask, mode="token-mean").item(),
            "opd/student_logprob_mean": agg(masked_student, loss_mask, mode="token-mean").item(),
            "opd/student_logprob_scale": student_logprob_scale.item(),
            "opd/raw_student_teacher_logprob_abs_mean_ratio": (raw_student_logprob_abs_mean / raw_teacher_logprob_abs_mean.clamp_min(1e-6)).item(),
            "opd/student_teacher_logprob_abs_mean_ratio": (student_logprob_abs_mean / teacher_logprob_abs_mean.clamp_min(1e-6)).item(),
            "opd/advantage_delta_mean": agg(advantage_delta, loss_mask, mode="token-mean").item(),
            "opd/base_advantage_mean": base_advantage_mean.item(),
            "opd/base_advantage_std": base_advantage_std.item(),
            "opd/adjusted_advantage_mean": agg(adjusted_advantages, loss_mask, mode="token-mean").item(),
            "opd/adjusted_advantage_std": agg(adjusted_advantages, loss_mask, mode="token-std").item(),
            "opd/teacher_logprobs_present_rate": agg(present, loss_mask, mode="token-mean").item(),
        }
        return adjusted_advantages, metrics

    def get_loss_func(
        self,
    ) -> LossFunc:
        if self.config.loss_type == "grpo":
            return self.grpo_loss
        if self.config.loss_type == "ppo":
            assert self.config.is_base_logprobs == "old_logprobs", "PPO actor loss must use old_logprobs as the policy-ratio base."
            return self.grpo_loss
        if self.config.loss_type == "tis":
            return self.tis_loss
        if self.config.loss_type == "grpo2":
            return self.grpo2_loss
        if self.config.loss_type == "topr":
            return self.topr_loss
        if self.config.loss_type == "gspo":
            return self.gspo
        if self.config.loss_type == "kimi2_5":
            assert self.config.is_base_logprobs == "rollout_logprobs"
            assert self.config.mismatch_token_clip_max is None
            return self.kimi2_5_loss
        raise ValueError(f"Unsupported loss type: {self.config.loss_type}")

    def get_seq_ratio(
        self,
        logprobs: torch.Tensor,
        base_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        log_ratio = (logprobs - base_logprobs) * loss_mask
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        seq_len = loss_mask.sum(dim=-1).clamp(min=1.0)
        seq_log_ratio = torch.sum(log_ratio, dim=-1) / seq_len
        seq_log_ratio = torch.clamp(seq_log_ratio, -20.0, 20.0).unsqueeze(-1)
        seq_ratio = torch.exp(seq_log_ratio).expand_as(log_ratio)  # (B,S)
        return seq_ratio

    def truncate_mismatch_token_ratio(
        self, old_logprobs: torch.Tensor, rollout_logprobs: torch.Tensor, loss_mask: torch.Tensor, advantages: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        # Truncated Rollout/Trainer IS: https://fengyao.notion.site/off-policy-rl
        token_mismatch_ir = self.get_ratio(logprobs=old_logprobs, base_logprobs=rollout_logprobs).detach()
        token_mismatch_ir = torch.where(loss_mask, token_mismatch_ir, torch.ones_like(token_mismatch_ir))  # set to 1 for padding tokens
        token_mismatch_tir = (
            torch.ones_like(token_mismatch_ir)
            if self.config.mismatch_token_clip_max is None
            else torch.clamp(token_mismatch_ir, max=self.config.mismatch_token_clip_max)
        )

        clip_applied = (
            torch.zeros_like(token_mismatch_tir)
            if self.config.mismatch_token_clip_max is None
            else (token_mismatch_tir - token_mismatch_ir).abs() > 1e-4
        )

        # metrics
        metrics = {}
        metrics.update(self.group_metrics(token_mismatch_ir, "token_mismatch_ir", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(token_mismatch_ir, "token_mismatch_ir_batch_max", mode="token-max", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(token_mismatch_ir, "token_mismatch_ir_batch_min", mode="token-min", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(clip_applied.float(), "token_mismatch_ir_clip_rate", mode="token-mean", adv=advantages, mask=loss_mask))

        return token_mismatch_ir, token_mismatch_tir, metrics

    def get_icepop_token_mask(
        self, token_mismatch_ir: torch.Tensor, loss_mask: torch.Tensor, advantages: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # ICEPOP token masking: https://hijkzzz.notion.site/online-ice-pop
        icepop_mask_low = (
            torch.zeros_like(token_mismatch_ir, dtype=torch.bool)
            if self.config.icepop_masking_low is None
            else token_mismatch_ir < self.config.icepop_masking_low
        )
        icepop_mask_high = (
            torch.zeros_like(token_mismatch_ir, dtype=torch.bool)
            if self.config.icepop_masking_high is None
            else token_mismatch_ir > self.config.icepop_masking_high
        )
        icepop_mask = icepop_mask_low | icepop_mask_high
        icepop_token_mask = ~icepop_mask

        metrics = {}
        metrics.update(self.group_metrics(icepop_mask_low.float(), "icepop_mask_low_ratio", mode="token-mean", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(icepop_mask_high.float(), "icepop_mask_high_ratio", mode="token-mean", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(icepop_mask.float(), "icepop_mask_ratio", mode="token-mean", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(icepop_token_mask.float(), "icepop_keep_ratio", mode="token-mean", adv=advantages, mask=loss_mask))
        return icepop_token_mask, metrics

    def get_mismatch_sequence_mask(
        self,
        token_mismatch_ir: torch.Tensor,
        logprobs: torch.Tensor,
        rollout_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Geometric sequence masking: https://richardli.xyz/rl-collapse
        # Use logprobs=logprobs, instead of old_logprobs, to filter sequence during model updates
        seq_ratio = self.get_seq_ratio(logprobs=logprobs, base_logprobs=rollout_logprobs, loss_mask=loss_mask).detach()
        seq_mask_low = (
            torch.ones_like(seq_ratio, dtype=torch.bool)
            if self.config.mismatch_seq_masking_low is None
            else (seq_ratio >= self.config.mismatch_seq_masking_low)
        )
        seq_mask_high = (
            torch.ones_like(seq_ratio, dtype=torch.bool)
            if self.config.mismatch_seq_masking_high is None
            else (seq_ratio <= self.config.mismatch_seq_masking_high)
        )
        # min token_mismatch_ir in a sequence
        seq_min_token_is = token_mismatch_ir.min(dim=-1, keepdim=True).values
        seq_mask_veto = (
            torch.ones_like(seq_min_token_is, dtype=torch.bool)
            if self.config.mismatch_token_veto_threshold is None
            else (seq_min_token_is >= self.config.mismatch_token_veto_threshold)
        )
        seq_mask_veto = seq_mask_veto.expand_as(loss_mask)
        seq_mask = seq_mask_low & seq_mask_high & seq_mask_veto  # (B,S)

        # metrics
        metrics: dict[str, float] = {}
        metrics.update(self.group_metrics(seq_mask_low.float(), "seq_mask_low_keep_rate", mode="token-mean", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(seq_mask_high.float(), "seq_mask_high_keep_rate", mode="token-mean", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(seq_mask_veto.float(), "seq_mask_veto_keep_rate", mode="token-mean", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(seq_mask.float(), "seq_mask_keep_rate", mode="token-mean", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(seq_ratio, "seq_ratio_batch_max", mode="token-max", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(seq_ratio, "seq_ratio_batch_min", mode="token-min", adv=advantages, mask=loss_mask))
        metrics.update(self.group_metrics(seq_ratio, "seq_ratio", mode="token-mean", adv=advantages, mask=loss_mask))
        return seq_mask, metrics

    def loss_func(
        self,
        outputs: dict[str, torch.Tensor],
        batch: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | TensorDict] | dict[str, float]]:
        loss_mask: torch.Tensor = batch["loss_mask"].bool()
        labels: torch.Tensor = batch["labels"]
        advantages: torch.Tensor = batch["advantage"]
        rollout_logprobs: torch.Tensor = batch["rollout_logprobs"]
        old_logprobs: torch.Tensor = batch["old_logprobs"]
        ref_logprobs: torch.Tensor = batch["ref_logprobs"]
        teacher_logprobs: torch.Tensor | None = batch.get("teacher_logprobs", None)
        logprobs: torch.Tensor = outputs["log_prob"]
        entropy: torch.Tensor = outputs["entropy"]
        turn_index: torch.Tensor | None = batch.get("turn_index", None)
        assert advantages.dim() == 2 and advantages.shape == logprobs.shape  # per-token advantage: (B, T)
        if self.config.opd.enabled:
            assert teacher_logprobs is not None, "OPD is enabled but batch is missing teacher_logprobs."
            advantages, opd_metrics = self.apply_opd_to_advantages(
                advantages=advantages,
                loss_mask=loss_mask,
                cur_logprobs=logprobs,
                rollout_logprobs=rollout_logprobs,
                old_logprobs=old_logprobs,
                teacher_logprobs=teacher_logprobs,
            )
        else:
            opd_metrics = {}

        with torch.no_grad():
            token_mismatch_ir, token_mismatch_tir, token_mismatch_metrics = self.truncate_mismatch_token_ratio(
                old_logprobs=old_logprobs,
                rollout_logprobs=rollout_logprobs,
                loss_mask=loss_mask,
                advantages=advantages,
            )

            icepop_token_mask, icepop_metrics = self.get_icepop_token_mask(
                token_mismatch_ir=token_mismatch_ir,
                loss_mask=loss_mask,
                advantages=advantages,
            )

            seq_mask, seq_mismatch_metrics = self.get_mismatch_sequence_mask(
                token_mismatch_ir=token_mismatch_ir,
                logprobs=logprobs,
                rollout_logprobs=rollout_logprobs,
                loss_mask=loss_mask,
                advantages=advantages,
            )

        # checking
        assert loss_mask.shape == labels.shape == logprobs.shape == old_logprobs.shape == ref_logprobs.shape == entropy.shape

        loss_func = self.get_loss_func()
        # Mismatch filters gate only the policy-gradient term; KL/entropy keep the original loss mask.
        policy_mismatch_tir = token_mismatch_tir * seq_mask.to(token_mismatch_tir.dtype) * icepop_token_mask.to(token_mismatch_tir.dtype)
        policy_loss_mask = loss_mask & seq_mask & icepop_token_mask
        pg_loss, ratio, clipped_ratio = loss_func(
            loss_mask=loss_mask,
            advantages=advantages,
            logprobs=logprobs,
            base_logprobs=old_logprobs if self.config.is_base_logprobs == "old_logprobs" else rollout_logprobs,
            token_mismatch_tir=policy_mismatch_tir,
        )

        entropy_loss, control_entropy = self.get_entropy_loss(
            entropy=entropy,
            loss_mask=loss_mask,
        )

        kl_loss = self.get_kl_loss(
            logprobs=logprobs,
            base_logprobs=old_logprobs if self.config.kl_base_logprobs == "old_logprobs" else ref_logprobs,
            loss_mask=loss_mask,
        )

        loss = pg_loss + entropy_loss * self.config.entropy_control.alpha + kl_loss * self.config.kl_control_alpha
        num_turns = get_num_turns(turn_index, loss_mask) if turn_index is not None else 0
        denom = self.compute_micro_batch_denominator(mask=loss_mask, num_turns=num_turns)

        with torch.no_grad(), Timer("Calculate GRPO metrics", verbose=False):
            metrics = self.calculate_metrics(
                loss_mask=policy_loss_mask,
                advantages=advantages,
                logprobs=logprobs,
                rollout_logprobs=rollout_logprobs,
                old_logprobs=old_logprobs,
                ref_logprobs=ref_logprobs,
                entropy=entropy,
                ratio=ratio,
                clipped_ratio=clipped_ratio,
                control_entropy=control_entropy,
                pg_loss=pg_loss.item(),
                entropy_loss=entropy_loss.item(),
                kl_loss=kl_loss.item(),
                loss=loss.item(),
            )
            metrics.update(token_mismatch_metrics)
            metrics.update(icepop_metrics)
            metrics.update(seq_mismatch_metrics)
            metrics.update(opd_metrics)
            metrics["loss_mask_sum"] = float(loss_mask.sum().item())
            metrics["policy_loss_mask_sum"] = float(policy_loss_mask.sum().item())
            metrics["denom"] = denom.item()
        return loss, denom, metrics

    def logits_processor(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = logits.float()
        results: dict[str, torch.Tensor] = {}

        compute_entropy = self.config.entropy_control.alpha > 0.0 or self.config.entropy_control.compute_entropy
        if not compute_entropy:
            results["entropy"] = torch.zeros(labels.shape, dtype=logits.dtype, device=logits.device)
            results["log_prob"] = vocab_parallel_log_prob(vocab_parallel_logits=logits, target=labels)
            return results

        if self.config.entropy_control.top_k > 0:
            entropy_logits = vocab_parallel_top_k_mask(vocab_parallel_logits=logits, top_k=self.config.entropy_control.top_k)
        else:
            entropy_logits = logits

        results["entropy"] = vocab_parallel_entropy(entropy_logits)
        # vocab_parallel_log_prob() will change the value of logits in-place, not sure if it's OK for the entropy loss backward.
        # So here we clone it to avoid potential issue.
        logits = logits.clone()
        results["log_prob"] = vocab_parallel_log_prob(vocab_parallel_logits=logits, target=labels)
        return results

    @override
    def forward_step(
        self,
        data_iterator: Iterator[TensorDict],
        model: GPTModel,
        model_forward_fn: GPTModelForwardFn,
    ) -> None:
        batch = next(data_iterator)
        routed_experts = batch.get("routed_experts", None)
        merge_info = extract_merge_info_from_batch(batch)
        batch = batch.exclude("routed_experts").to(torch.cuda.current_device())
        labels: torch.Tensor = batch["labels"]
        outputs: dict[str, torch.Tensor] = model_forward_fn(  # type: ignore[assignment]
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"].to(torch.bool),
            logits_processor=self.logits_processor,
            logits_processor_args={"labels": labels},
            routed_experts=routed_experts,
            merge_info=merge_info,
            loss_mask=batch["loss_mask"].to(torch.bool),
        )
        results = outputs, partial(self.loss_func, batch=batch)
        return results  # type: ignore
