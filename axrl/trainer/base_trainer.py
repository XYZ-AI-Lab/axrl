import logging
from collections.abc import Iterator
from functools import partial
from typing import Any

import torch
from megatron.core.models.gpt import GPTModel
from tensordict import TensorDict

from axrl.configs import ValueAggType
from axrl.utils.logger import LoggerBuffer
from axrl.utils.megatron.model_forward import GPTModelForwardFn
from axrl.utils.megatron.prefix_tree import extract_merge_info_from_batch
from axrl.utils.megatron.tensor_parallel import vocab_parallel_log_prob

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self) -> None:
        pass

    def set_metric_agg_type(self, logger_buffer: LoggerBuffer) -> None:
        """Set aggregation types for various metrics in the logger buffer.

        The logger buffer will aggregate metrics based on these agg-types every log interval.

        Example: `logger_buffer.set_metric_agg_type("cpu_time_s", ["mean"])` will report `cpu_time_s_mean` as the mean of `cpu_time_s` in the buffer.
        """
        logger_buffer.set_metric_agg_type("peak_mem_reserved_gbs", ["min", "max"])
        logger_buffer.set_metric_agg_type("peak_mem_gbs", ["min", "max"])
        logger_buffer.set_metric_agg_type("cpu_time_s", ["mean", "std"])

    def loss_func(
        self,
        outputs: dict[str, torch.Tensor],
        batch: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | TensorDict] | dict[str, float]]:
        raise NotImplementedError

    def forward_step(
        self,
        data_iterator: Iterator[TensorDict],
        model: GPTModel,
        model_forward_fn: GPTModelForwardFn,
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def aggregate_with_mask(value: torch.Tensor, mask: torch.Tensor, mode: ValueAggType) -> torch.Tensor:
        """Aggregate value across tokens.

        Input shapes: value (B, T), mask (B, T)
        Output shape: loss (1,)

        Example: loss = aggregate_with_mask(losses, loss_mask, mode="token-mean")
        """
        epsilon = 1e-10
        assert value.shape == mask.shape
        if mode == "token-mean":
            denom = BaseTrainer.compute_denominator_count(mask, mode=mode)
            aggregated_value = torch.sum(value * mask) / (denom + epsilon)
        elif mode == "seq-mean-token-mean":
            token_sum = torch.sum(value * mask, dim=-1)  # (B,)
            token_count = torch.sum(mask, dim=-1)
            valid_seq = token_count > 0
            token_mean = token_sum / (token_count + epsilon)  # (B,)
            denom = BaseTrainer.compute_denominator_count(mask, mode=mode)
            aggregated_value = torch.sum(token_mean * valid_seq) / (denom + epsilon)
        elif mode == "token-max":
            masked_value = value.masked_fill(mask == 0, float("-inf"))
            aggregated_value = torch.max(masked_value)
        elif mode == "token-min":
            masked_value = value.masked_fill(mask == 0, float("inf"))
            aggregated_value = torch.min(masked_value)
        elif mode == "token-std":
            denom = BaseTrainer.compute_denominator_count(mask, mode="token-mean")
            mean = torch.sum(value * mask) / (denom + epsilon)
            sq_diff = (value - mean) ** 2
            variance = torch.sum(sq_diff * mask) / (denom + epsilon)
            aggregated_value = torch.sqrt(variance + epsilon)
        elif mode == "token-p05":
            valid = value[mask.bool()]
            if valid.numel() == 0:
                aggregated_value = torch.tensor(0.0, device=value.device)
            else:
                aggregated_value = torch.quantile(valid.float(), 0.05)
        elif mode == "token-p95":
            valid = value[mask.bool()]
            if valid.numel() == 0:
                aggregated_value = torch.tensor(0.0, device=value.device)
            else:
                aggregated_value = torch.quantile(valid.float(), 0.95)
        elif mode == "seq-mean-token-std":
            # Per-sequence std of masked tokens, averaged across sequences.
            mask_f = mask.float()
            seq_len = mask_f.sum(dim=-1).clamp(min=1.0)  # (B,)
            seq_mean = (value * mask_f).sum(dim=-1) / seq_len  # (B,)
            sq_diff = ((value - seq_mean.unsqueeze(-1)) ** 2) * mask_f
            seq_var = sq_diff.sum(dim=-1) / seq_len  # (B,)
            seq_std = torch.sqrt(seq_var + epsilon)  # (B,)
            valid_seq = mask_f.sum(dim=-1) > 0
            denom = valid_seq.sum().clamp(min=1)
            aggregated_value = (seq_std * valid_seq).sum() / denom
        else:
            raise ValueError(f"Unknown loss aggregation mode: [{mode}]")
        return aggregated_value

    @staticmethod
    def compute_denominator_count(mask: torch.Tensor, mode: ValueAggType) -> torch.Tensor:
        """Return the normalization factor that matches the specified aggregation mode."""
        dtype = torch.int
        if mode == "token-mean":
            return torch.sum(mask, dtype=dtype)
        if mode == "seq-mean-token-mean":
            token_count = torch.sum(mask, dim=-1)
            valid_seq = token_count > 0
            return torch.sum(valid_seq, dtype=dtype)
        raise ValueError(f"Unknown loss aggregation mode: [{mode}]")

    def _logprob_loss_func(
        self,
        outputs: dict[str, torch.Tensor],
        batch: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float | TensorDict] | dict[str, float]]:
        """Loss function for logprob outputs."""
        metrics: dict[str, float | TensorDict] = {}
        loss = torch.tensor(0.0, device=outputs["log_prob"].device)
        metrics["loss"] = float(loss.item())
        results: dict[str, torch.Tensor] = {k: v.clone().detach().cpu() for k, v in outputs.items()}
        results["index"] = batch["index"].clone().detach().cpu()
        metrics["output"] = TensorDict(results, batch_size=batch["labels"].shape[0])
        return loss, metrics

    def _logprob_logits_processor(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = logits.float()
        results: dict[str, torch.Tensor] = {}
        results["log_prob"] = vocab_parallel_log_prob(vocab_parallel_logits=logits, target=labels)
        return results

    def logprob_forward_step(
        self,
        data_iterator: Iterator[TensorDict],
        model: GPTModel,
        model_forward_fn: GPTModelForwardFn,
    ) -> Any:
        batch = next(data_iterator)
        routed_experts = batch.get("routed_experts", None)
        merge_info = extract_merge_info_from_batch(batch)
        batch = batch.exclude("routed_experts").to(torch.cuda.current_device())
        labels: torch.Tensor = batch["labels"]
        outputs: dict[str, torch.Tensor] = model_forward_fn(  # type: ignore[assignment]
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"].to(torch.bool),
            logits_processor=self._logprob_logits_processor,
            logits_processor_args={"labels": labels},
            routed_experts=routed_experts,
            merge_info=merge_info,
            loss_mask=batch["loss_mask"].to(torch.bool),
        )
        return outputs, partial(self._logprob_loss_func, batch=batch)  # type: ignore
