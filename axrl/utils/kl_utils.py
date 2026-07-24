import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import torch
import torch.nn.functional

logger = logging.getLogger(__name__)


KLType = Literal["k1", "k2", "k3"]


def kl_divergence(logprobs: torch.Tensor, logprobs_ref: torch.Tensor, kl_type: KLType = "k2") -> torch.Tensor:
    """KL divergence approximations.

    ``logprobs`` are the sampled/current-policy logprobs and ``logprobs_ref``
    are the base/reference logprobs on the same tokens.

    Reference:
    - http://joschu.net/blog/kl-approx.html
    """
    logr = logprobs.float() - logprobs_ref.float()
    if kl_type == "k1":
        return logr
    if kl_type == "k2":
        return logr**2 / 2
    if kl_type == "k3":
        logr = -logr
        return logr.exp() - 1 - logr
    raise ValueError(f"Unknown kl_type: {kl_type}")


@dataclass
class LogprobsDiffResult:
    k1: float
    k1_max: float
    k2: float
    k2_max: float
    k3: float
    k3_max: float
    cosine_similarity: float
    base_logprobs_mean: float
    test_logprobs_mean: float


def _filter_logprobs_with_loss_mask(base_seq: list[float], test_seq: list[float], loss_mask: list[bool]) -> tuple[list[float], list[float]]:
    seq_len = len(base_seq)
    assert seq_len == len(test_seq)
    assert sum(loss_mask[seq_len:]) == 0
    loss_mask = loss_mask[:seq_len]
    assert sum(loss_mask) > 0
    base_seq_masked = [lp for lp, lm in zip(base_seq, loss_mask, strict=True) if lm]
    test_seq_masked = [lp for lp, lm in zip(test_seq, loss_mask, strict=True) if lm]
    assert len(base_seq_masked) == len(test_seq_masked)
    return base_seq_masked, test_seq_masked


def compare_logprobs(
    loss_masks: torch.Tensor,
    base_logprobs: torch.Tensor,
    test_logprobs: torch.Tensor,
) -> LogprobsDiffResult:
    assert len(base_logprobs) == len(test_logprobs), "Number of sentences must match"
    items: list[dict[str, float]] = []
    for i, (base_seq, test_seq, loss_mask) in enumerate(zip(base_logprobs, test_logprobs, loss_masks, strict=True)):
        base_seq_masked, test_seq_masked = _filter_logprobs_with_loss_mask(base_seq.tolist(), test_seq.tolist(), loss_mask.tolist())
        if len(base_seq_masked) == 0:
            logger.warning(f"Skipping sentence {i} with no valid tokens for consistency checking.")
            continue
        base_tensor = torch.tensor(base_seq_masked, dtype=torch.float32)
        test_tensor = torch.tensor(test_seq_masked, dtype=torch.float32)
        k1 = kl_divergence(test_tensor, base_tensor, "k1")
        k2 = kl_divergence(test_tensor, base_tensor, "k2")
        k3 = kl_divergence(test_tensor, base_tensor, "k3")

        items.append(
            {
                "k1": k1.abs().mean().item(),
                "k1_max": k1.abs().max().item(),
                "k2": k2.mean().item(),
                "k2_max": k2.abs().max().item(),
                "k3": k3.mean().item(),
                "k3_max": k3.abs().max().item(),
                "base_seq_mean": base_tensor.mean().item(),
                "test_seq_mean": test_tensor.mean().item(),
                "cosine_similarity": torch.nn.functional.cosine_similarity(base_tensor.unsqueeze(0), test_tensor.unsqueeze(0)).item(),
            }
        )

    data = pd.DataFrame(items)
    result = LogprobsDiffResult(
        base_logprobs_mean=data["base_seq_mean"].mean(),
        test_logprobs_mean=data["test_seq_mean"].mean(),
        cosine_similarity=data["cosine_similarity"].mean(),
        k1=data["k1"].mean(),
        k1_max=data["k1_max"].max(),
        k2=data["k2"].mean(),
        k2_max=data["k2_max"].max(),
        k3=data["k3"].mean(),
        k3_max=data["k3_max"].max(),
    )
    logger.info(f"Consistency result: {result}")
    return result
