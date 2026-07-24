from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel


def gather_vocab_parallel_logits(logits: torch.Tensor) -> torch.Tensor:
    """Gather vocab-parallel logits across TP to form full-vocab logits.

    - Input shape: [B, S, V_part]
    - Output shape: [B, S, V_full]
    """
    tp_world = mpu.get_tensor_model_parallel_world_size()
    if tp_world == 1:
        return logits
    tp_group = mpu.get_tensor_model_parallel_group()
    parts = [torch.empty_like(logits) for _ in range(tp_world)]
    dist.all_gather(parts, logits, group=tp_group)
    return torch.cat(parts, dim=-1)


class _VocabParallelEntropy(torch.autograd.Function):
    # from https://github.com/volcengine/verl/blob/6f559540e7932e2e778bb566a52778d2aee81154/verl/utils/megatron/tensor_parallel.py
    @staticmethod
    def forward(ctx: Any, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[return]
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks.

    Args:
        vocab_parallel_logits: (B, S, V_part)

    Returns: (B, S)
    """
    assert vocab_parallel_logits.dim() == 3, "Expected logits with shape (B, S, V_part)"
    b, s = vocab_parallel_logits.shape[:2]
    vocab_parallel_logits = vocab_parallel_logits.view(b * s, -1)
    output: torch.Tensor = _VocabParallelEntropy.apply(vocab_parallel_logits)  # type: ignore
    output = output.view(b, s)
    return output


def vocab_parallel_log_prob(vocab_parallel_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute log-probability when the logits are sharded in tp ranks.

    Args:
        vocab_parallel_logits: (B, S, V_part)
        target: (B, S)
    Returns: (B, S)
    """
    cross_entropy: torch.Tensor = tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=vocab_parallel_logits, target=target)  # type: ignore
    return -cross_entropy


def vocab_parallel_top_k_mask(vocab_parallel_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    tp_group = mpu.get_tensor_model_parallel_group()
    world_size = mpu.get_tensor_model_parallel_world_size()
    rank = mpu.get_tensor_model_parallel_rank()
    assert top_k > 0, "top_k must be positive"

    *prefix_shape, vocab_size_per_rank = vocab_parallel_logits.shape
    logits_flat = vocab_parallel_logits.reshape(-1, vocab_size_per_rank)

    # local top k
    local_k = min(top_k, vocab_size_per_rank)
    local_vals, local_idx = torch.topk(logits_flat, local_k, dim=-1, largest=True, sorted=False)

    vocab_offset = rank * vocab_size_per_rank
    local_idx = local_idx + vocab_offset  # now global indices

    gather_vals = [torch.empty_like(local_vals) for _ in range(world_size)]
    gather_idx = [torch.empty_like(local_idx) for _ in range(world_size)]
    dist.all_gather(gather_vals, local_vals, group=tp_group)
    dist.all_gather(gather_idx, local_idx, group=tp_group)

    combined_vals = torch.cat(gather_vals, dim=-1)
    combined_idx = torch.cat(gather_idx, dim=-1)

    # global top k
    global_k = min(top_k, combined_vals.size(-1))
    _, topk_pos = torch.topk(combined_vals, global_k, dim=-1, largest=True, sorted=False)
    topk_idx = torch.gather(combined_idx, dim=-1, index=topk_pos)

    local_idx = topk_idx - vocab_offset
    valid = (local_idx >= 0) & (local_idx < vocab_size_per_rank)

    mask = torch.zeros_like(logits_flat, dtype=torch.bool)
    if torch.any(valid):
        rows = torch.arange(mask.size(0), device=mask.device).unsqueeze(1).expand_as(local_idx)
        mask[rows[valid], local_idx[valid]] = True

    fill_value = torch.finfo(logits_flat.dtype).min
    masked_logits = logits_flat.masked_fill(~mask, fill_value)
    return masked_logits.reshape(*prefix_shape, vocab_size_per_rank)


def vocab_parallel_argmax(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute argmax when the logits are sharded in tp ranks.

    Args:
        vocab_parallel_logits: (B, S, V_part)

    Returns: (B, S)
    """
    # Support arbitrary leading dims; last dim is the partitioned vocab.
    assert vocab_parallel_logits.dim() >= 1, "Expected logits with last dim as vocab partition"
    tp_world = mpu.get_tensor_model_parallel_world_size()
    if tp_world == 1:
        return vocab_parallel_logits.argmax(dim=-1)

    # Local maxima and indices over the partitioned vocab axis.
    local_max_vals, local_argmax = torch.max(vocab_parallel_logits, dim=-1)

    # Compute global max values across TP ranks.
    global_max_vals = local_max_vals.clone()
    dist.all_reduce(global_max_vals, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())

    # Build candidate global indices only for ranks that hit the global max.
    # Tie-breaker: choose the smallest global index among equal maxima.
    vpart = vocab_parallel_logits.size(-1)
    offset = mpu.get_tensor_model_parallel_rank() * vpart
    candidate_idx = local_argmax.to(torch.long) + offset
    is_winner = local_max_vals == global_max_vals
    big = torch.iinfo(torch.long).max  # max value of int64
    masked_candidate = torch.where(is_winner, candidate_idx, torch.full_like(candidate_idx, big))

    # Reduce by MIN to pick the earliest global index among winners.
    dist.all_reduce(masked_candidate, op=dist.ReduceOp.MIN, group=mpu.get_tensor_model_parallel_group())
    return masked_candidate
