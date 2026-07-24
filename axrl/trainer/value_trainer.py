from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, override

import torch
from tensordict import TensorDict

from axrl.trainer.base_trainer import BaseTrainer
from axrl.trainer.ppo_utils import clipped_value_loss_per_token
from axrl.utils.megatron.prefix_tree import extract_merge_info_from_batch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from megatron.core.models.gpt import GPTModel

    from axrl.configs import PPOValueConfig
    from axrl.utils.logger import LoggerBuffer
    from axrl.utils.megatron.model_forward import GPTModelForwardFn


class ValueTrainer(BaseTrainer):
    def __init__(self, config: PPOValueConfig) -> None:
        super().__init__()
        self.config = config

    @override
    def set_metric_agg_type(self, logger_buffer: LoggerBuffer) -> None:
        super().set_metric_agg_type(logger_buffer)
        logger_buffer.set_metric_agg_type("denom", ["mean", "std"])
        logger_buffer.set_metric_agg_type("num_samples", ["sum"])

    @staticmethod
    def _values_from_output(output: torch.Tensor) -> torch.Tensor:
        if output.dim() == 3:
            assert output.shape[-1] == 1, f"value output with 3 dims must have final dim 1, got {tuple(output.shape)}"
            return output.squeeze(-1)
        assert output.dim() == 2, f"value output must have shape (B, T) or (B, T, 1), got {tuple(output.shape)}"
        return output

    def values_processor(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"values": self._values_from_output(values).float()}

    def _value_output_loss_func(
        self,
        outputs: dict[str, torch.Tensor],
        batch: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float | TensorDict] | dict[str, float]]:
        metrics: dict[str, float | TensorDict] = {}
        loss = torch.tensor(0.0, device=outputs["values"].device)
        results: dict[str, torch.Tensor] = {k: v.clone().detach().cpu() for k, v in outputs.items()}
        results["index"] = batch["index"].clone().detach().cpu()
        metrics["output"] = TensorDict(results, batch_size=batch["input_ids"].shape[0])
        return loss, metrics

    def _masked_metric(self, values: torch.Tensor, mask: torch.Tensor, mode: str) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(self.aggregate_with_mask(values, mask, mode=mode).item())  # type: ignore[arg-type]

    @override
    def loss_func(
        self,
        outputs: dict[str, torch.Tensor],
        batch: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | TensorDict] | dict[str, float]]:
        values = self._values_from_output(outputs["values"])
        loss_mask: torch.Tensor = batch["loss_mask"].bool()
        old_values: torch.Tensor = batch["old_values"]
        returns: torch.Tensor = batch["returns"]
        assert values.shape == old_values.shape == returns.shape == loss_mask.shape, (
            f"value batch shape mismatch: values={tuple(values.shape)}, old_values={tuple(old_values.shape)}, "
            f"returns={tuple(returns.shape)}, loss_mask={tuple(loss_mask.shape)}"
        )

        losses, clipfrac = clipped_value_loss_per_token(
            values=values,
            old_values=old_values,
            returns=returns,
            value_clip=self.config.value_clip,
        )
        value_loss = self.aggregate_with_mask(losses, loss_mask, mode="token-mean") * self.config.value_loss_coef
        denom = self.compute_denominator_count(mask=loss_mask, mode="token-mean")

        metrics: dict[str, float] = {
            "loss": float(value_loss.item()),
            "value_loss": float(value_loss.item()),
            "denom": float(denom.item()),
            "num_samples": float(values.shape[0]),
            "value_mean": self._masked_metric(values.float(), loss_mask, "token-mean"),
            "value_std": self._masked_metric(values.float(), loss_mask, "token-std"),
            "return_mean": self._masked_metric(returns.float(), loss_mask, "token-mean"),
            "return_std": self._masked_metric(returns.float(), loss_mask, "token-std"),
        }
        if clipfrac is not None:
            metrics["value_clipfrac"] = self._masked_metric(clipfrac, loss_mask, "token-mean")
        return value_loss, denom, metrics

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
        outputs: dict[str, torch.Tensor] = model_forward_fn(  # type: ignore[assignment]
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"].to(torch.bool),
            logits_processor=self.values_processor,
            routed_experts=routed_experts,
            merge_info=merge_info,
            loss_mask=batch["loss_mask"].to(torch.bool),
        )
        return outputs, partial(self.loss_func, batch=batch)  # type: ignore

    def value_forward_step(
        self,
        data_iterator: Iterator[TensorDict],
        model: GPTModel,
        model_forward_fn: GPTModelForwardFn,
    ) -> None:
        batch = next(data_iterator)
        routed_experts = batch.get("routed_experts", None)
        merge_info = extract_merge_info_from_batch(batch)
        batch = batch.exclude("routed_experts").to(torch.cuda.current_device())
        outputs: dict[str, torch.Tensor] = model_forward_fn(  # type: ignore[assignment]
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"].to(torch.bool),
            logits_processor=self.values_processor,
            routed_experts=routed_experts,
            merge_info=merge_info,
            loss_mask=batch["loss_mask"].to(torch.bool),
        )
        return outputs, partial(self._value_output_loss_func, batch=batch)  # type: ignore
