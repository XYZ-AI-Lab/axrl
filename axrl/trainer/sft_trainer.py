import logging
from collections.abc import Iterator
from functools import partial
from typing import override

import torch
from megatron.core.models.gpt import GPTModel
from tensordict import TensorDict

from axrl.configs import SftTrainerConfig
from axrl.trainer.base_trainer import BaseTrainer
from axrl.utils.logger import LoggerBuffer
from axrl.utils.megatron.model_forward import GPTModelForwardFn
from axrl.utils.megatron.prefix_tree import extract_merge_info_from_batch
from axrl.utils.megatron.tensor_parallel import vocab_parallel_argmax, vocab_parallel_entropy, vocab_parallel_log_prob

logger = logging.getLogger(__name__)


class SftTrainer(BaseTrainer):
    def __init__(self, config: SftTrainerConfig) -> None:
        super().__init__()
        self.config = config

    @override
    def set_metric_agg_type(self, logger_buffer: LoggerBuffer) -> None:
        super().set_metric_agg_type(logger_buffer)
        logger_buffer.set_metric_agg_type("denom", ["mean", "std"])
        logger_buffer.set_metric_agg_type("num_samples", ["sum"])

    @override
    def loss_func(
        self,
        outputs: dict[str, torch.Tensor],
        batch: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | TensorDict] | dict[str, float]]:
        """Loss function.

        Returns:
            the loss scalar for this micro-batch
            the number of non-padded tokens in this microbatch
            a dict containing reporting metrics on the loss and number of tokens across
                the data parallel ranks
        """
        labels: torch.Tensor = batch["labels"]
        loss_mask: torch.Tensor = batch["loss_mask"]

        metrics: dict[str, float | TensorDict] = {}
        if self.config.compute_entropy:
            entropies = outputs["entropy"]
            entropy: float = self.aggregate_with_mask(entropies, loss_mask, mode="token-mean").item()
            metrics["entropy"] = entropy

        if self.config.compute_accuracy:
            argmax = outputs["argmax"]
            corrects = argmax == labels
            accuracy: float = self.aggregate_with_mask(corrects.float(), loss_mask, mode="token-mean").item()
            metrics["accuracy"] = accuracy

        log_prob = outputs["log_prob"]

        loss = self.aggregate_with_mask(-log_prob, loss_mask, mode="token-mean")
        denom = self.compute_denominator_count(mask=loss_mask, mode="token-mean")
        metrics["denom"] = float(denom.item())
        metrics["loss"] = float(loss.item())
        return loss, denom, metrics

    def logits_processor(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}
        if self.config.compute_entropy:
            results["entropy"] = vocab_parallel_entropy(logits).detach()
        if self.config.compute_accuracy:
            results["argmax"] = vocab_parallel_argmax(logits).detach()
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
        return outputs, partial(self.loss_func, batch=batch)  # type: ignore
