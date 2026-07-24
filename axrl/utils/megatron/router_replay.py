"""Utility for Rollout Routing Replay (R3).

References:
- RouterReplay design doc:
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/docs/api-guide/router_replay.md#L15-L77
- RouterReplay implementation:
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router_replay.py#L8-L151
- Top-k routing with replay hook:
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/moe_utils.py#L608-L749
- Router construction and routing path:
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router.py#L131-L215
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router.py#L545-L646
- veRL router replay helpers referenced by this module:
    https://github.com/verl-project/verl/blob/f88ed61abe1e0f10febd61c06228ca0db030fdd8/verl/utils/megatron/router_replay_utils.py#L269-L324
    https://github.com/verl-project/verl/blob/f88ed61abe1e0f10febd61c06228ca0db030fdd8/verl/utils/megatron/router_replay_utils.py#L476-L540
    https://github.com/verl-project/verl/blob/f88ed61abe1e0f10febd61c06228ca0db030fdd8/verl/models/mcore/util.py
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from axrl.utils.megatron.pack_utils import preprocess_packed_seqs
from axrl.utils.megatron.utils import unwrap_model

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

    from axrl.utils.megatron.prefix_tree import PrefixMergeInfo


@runtime_checkable
class _EmbeddingSequenceParallelConfig(Protocol):
    scatter_embedding_sequence_parallel: bool


def _uses_gdn_without_embedding_sequence_parallel(tf_config: TransformerConfig) -> bool:
    if tf_config.experimental_attention_variant != "gated_delta_net":
        return False
    if not isinstance(tf_config, _EmbeddingSequenceParallelConfig):
        return False
    return not tf_config.scatter_embedding_sequence_parallel


def _split_routed_experts_samples(routed_experts: torch.Tensor) -> list[torch.Tensor]:
    if routed_experts.is_nested:
        return list(routed_experts.unbind())
    assert routed_experts.dim() == 4, f"Expected routed_experts with shape [B, T, L, K], got {tuple(routed_experts.shape)}"
    return [routed_experts[index] for index in range(routed_experts.size(0))]


_AXRL_ORIGINAL_ROUTER_REPLAY_GET_TOPK_ATTR = "_axrl_original_get_replay_topk"
if not hasattr(RouterReplay, _AXRL_ORIGINAL_ROUTER_REPLAY_GET_TOPK_ATTR):
    setattr(RouterReplay, _AXRL_ORIGINAL_ROUTER_REPLAY_GET_TOPK_ATTR, RouterReplay.get_replay_topk)
_ORIGINAL_ROUTER_REPLAY_GET_TOPK = getattr(RouterReplay, _AXRL_ORIGINAL_ROUTER_REPLAY_GET_TOPK_ATTR)
_AXRL_REPLAY_TOKEN_MASK_ATTR = "_axrl_replay_token_mask"


def _axrl_router_replay_get_topk(
    self: RouterReplay,
    scores: torch.Tensor,
    topk: int,
    num_groups: int | None = None,
    group_topk: int | None = None,
    default_compute_topk: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    replay_token_mask = getattr(self, _AXRL_REPLAY_TOKEN_MASK_ATTR, None)
    if replay_token_mask is None or self.router_replay_action not in {
        RouterReplayAction.REPLAY_FORWARD,
        RouterReplayAction.REPLAY_BACKWARD,
    }:
        return _ORIGINAL_ROUTER_REPLAY_GET_TOPK(
            self,
            scores,
            topk,
            num_groups=num_groups,
            group_topk=group_topk,
            default_compute_topk=default_compute_topk,
        )

    if self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
        top_indices = self.replay_backward_list.pop(0).to(scores.device)
        return scores.gather(1, top_indices), top_indices

    assert self.target_topk_idx is not None
    replay_indices = self.target_topk_idx.to(scores.device)
    replay_token_mask = replay_token_mask.to(device=scores.device, dtype=torch.bool).view(-1, 1)
    assert replay_token_mask.shape[0] == scores.shape[0], (
        f"Replay token mask length {replay_token_mask.shape[0]} does not match router score length {scores.shape[0]}"
    )
    if bool(replay_token_mask.all().item()):
        top_indices = replay_indices
    elif bool((~replay_token_mask).all().item()):
        _, top_indices = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
    else:
        _, current_indices = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
        top_indices = torch.where(replay_token_mask, replay_indices, current_indices)
    if self.replay_backward_list:
        self.replay_backward_list[-1] = top_indices.detach()
    return scores.gather(1, top_indices), top_indices


if RouterReplay.get_replay_topk is not _axrl_router_replay_get_topk:
    RouterReplay.get_replay_topk = _axrl_router_replay_get_topk


def _set_router_replay_token_masks(local_routers: list[RouterReplay], replay_token_mask: torch.Tensor | None) -> None:
    for router in local_routers:
        if replay_token_mask is None:
            if hasattr(router, _AXRL_REPLAY_TOKEN_MASK_ATTR):
                delattr(router, _AXRL_REPLAY_TOKEN_MASK_ATTR)
            continue
        setattr(router, _AXRL_REPLAY_TOKEN_MASK_ATTR, replay_token_mask)


def _is_moe_layer(tf_config: TransformerConfig, layer_idx: int) -> bool:
    moe_layer_freq = tf_config.moe_layer_freq
    if isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0
    if isinstance(moe_layer_freq, Sequence):
        return bool(moe_layer_freq[layer_idx])
    raise TypeError(f"Unsupported moe_layer_freq type: {type(moe_layer_freq)}")


def get_moe_num_layers_to_build(
    tf_config: TransformerConfig,
    *,
    vp_rank: int | None = None,
    pp_rank: int | None = None,
) -> int:
    """Count how many local layers are MoE layers on the current PP or VPP shard.

    RouterReplay instances exist only for MoE router layers, so replay tensors
    must be sliced using the MoE-only layer count instead of the total number of
    transformer layers.

        Reference material:
        - Megatron TopKRouter construction:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router.py#L131-L215
    """
    total_layers = get_num_layers_to_build(tf_config, vp_stage=vp_rank, pp_rank=pp_rank)
    layer_offset = get_transformer_layer_offset(tf_config, vp_stage=vp_rank, pp_rank=pp_rank)
    local_global_indices = range(layer_offset, layer_offset + total_layers)
    return sum(1 for layer_idx in local_global_indices if _is_moe_layer(tf_config, layer_idx))


def get_current_rank_layer_info(
    tf_config: TransformerConfig,
    *,
    vp_rank: int | None = None,
    pp_rank: int | None = None,
) -> dict[str, int]:
    """Return the global transformer layer range assigned to the current rank.

    The returned start and end indices describe the rank-local window in global
    layer numbering. Downstream replay helpers use this range to decide which
    replay tensors belong to the current PP or VPP stage.

        Reference material:
        - Megatron layer offset helper:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/transformer_layer.py#L39-L195
    """
    if pp_rank is None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
    if vp_rank is None:
        vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_rank, pp_rank=pp_rank)
    offset = get_transformer_layer_offset(tf_config, vp_stage=vp_rank, pp_rank=pp_rank)
    return {
        "start": offset,
        "end": offset + num_layers_to_build,
        "count": num_layers_to_build,
    }


def get_micro_batch_router_list(
    tf_config: TransformerConfig,
    *,
    vp_rank: int | None = None,
    pp_rank: int | None = None,
) -> list[RouterReplay]:
    """Return the RouterReplay instances that belong to the current micro-batch stage.

    Megatron registers RouterReplay instances in MoE layer construction order.
    When virtual pipeline parallelism is enabled, this helper computes the local
    offset so axrl can address only the routers belonging to the active VP stage.

        Reference material:
        - veRL RouterReplayHelper.get_micro_batch_router_list:
            https://github.com/verl-project/verl/blob/f88ed61abe1e0f10febd61c06228ca0db030fdd8/verl/utils/megatron/router_replay_utils.py#L476-L505
        - RouterReplay lifecycle:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router_replay.py#L18-L151
        - TopKRouter replay attachment:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router.py#L131-L215
    """
    vpp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_rank is None:
        vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()

    if vpp_size is not None:
        assert vp_rank is not None
        offset = 0
        for previous_vp_rank in range(vpp_size):
            if previous_vp_rank == vp_rank:
                break
            offset += get_moe_num_layers_to_build(tf_config, vp_rank=previous_vp_rank, pp_rank=pp_rank)
    else:
        offset = 0

    num_local_moe_layers = get_moe_num_layers_to_build(tf_config, vp_rank=vp_rank, pp_rank=pp_rank)
    return RouterReplay.global_router_replay_instances[offset : offset + num_local_moe_layers]


def select_local_router_replay_tensors(
    layers_topk_idx: torch.Tensor,
    tf_config: TransformerConfig,
    *,
    vp_rank: int | None = None,
    pp_rank: int | None = None,
) -> list[torch.Tensor]:
    """Slice packed replay indices down to the MoE routers local to this rank.

    The input can be indexed either by all transformer layers or by MoE layers
    only. This helper resolves that ambiguity and returns one replay tensor per
    local RouterReplay instance in the order Megatron expects.

        Reference material:
        - RouterReplay.set_replay_data:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router_replay.py#L29-L42
        - TopKRouter routing order:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router.py#L545-L638
    """
    local_rank_info = get_current_rank_layer_info(tf_config, vp_rank=vp_rank, pp_rank=pp_rank)

    index_by_layer = len(layers_topk_idx) == tf_config.num_layers
    moe_layer_idx = sum(1 for layer_idx in range(local_rank_info["start"]) if _is_moe_layer(tf_config, layer_idx))

    replay_tensors: list[torch.Tensor] = []
    for layer_idx in range(local_rank_info["start"], local_rank_info["end"]):
        if not _is_moe_layer(tf_config, layer_idx):
            continue
        router_layer_idx = layer_idx if index_by_layer else moe_layer_idx
        replay_tensors.append(layers_topk_idx[router_layer_idx].to(torch.int64))
        moe_layer_idx += 1
    return replay_tensors


def expand_routed_experts_to_token_positions(
    routed_experts: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Expand compact seq-1 routing into token-aligned [B, S, L, K] tensors.

    SGLang records routed experts for next-token predictions, so compact routing has
    length seq-1 and starts at expanded index 0: entry i contains the routing used
    to predict token i+1 from the prefix ending at token i. When expanded back to a
    [B, S, L, K] tensor, the final valid token position has no next-token target, so
    it is filled with placeholder expert ids together with padded positions before
    packing to Megatron's token order.

        Reference material:
        - RouterReplay design note for forward and backward replay semantics:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/docs/api-guide/router_replay.md#L15-L58
    """
    attention_mask_cpu = attention_mask.to(device="cpu", dtype=torch.bool)
    samples = _split_routed_experts_samples(routed_experts)
    batch_size, max_seq_len = attention_mask_cpu.shape
    assert len(samples) == batch_size, f"Expected {batch_size} routed_experts samples, got {len(samples)}"
    num_layers, topk = samples[0].shape[-2:]
    filler = torch.arange(topk, dtype=samples[0].dtype).view(1, 1, 1, topk).expand(batch_size, max_seq_len, num_layers, topk)
    expanded = filler.clone()
    for sample_idx, sample_routing in enumerate(samples):
        effective_seq_len = int(attention_mask_cpu[sample_idx].sum().item())
        expected_routing_len = max(effective_seq_len - 1, 0)
        assert sample_routing.shape[0] == expected_routing_len, (
            f"Expected routed_experts length {expected_routing_len} for sample {sample_idx}, got {sample_routing.shape[0]}"
        )
        if expected_routing_len > 0:
            expanded[sample_idx, :expected_routing_len] = sample_routing[:expected_routing_len].to(device="cpu")
    return expanded


def prepare_router_replay_tensors(
    routed_experts: torch.Tensor,
    attention_mask: torch.Tensor,
    tf_config: TransformerConfig,
    *,
    vp_rank: int | None = None,
    pp_rank: int | None = None,
    loss_mask: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[RouterReplay]]:
    """Convert rollout routing into rank-local replay tensors for Megatron routers.

    This is the core layout bridge in axrl R3. It performs four steps:
    1. expand compact seq-1 routing to token-aligned tensors
    2. pack tokens using the same packed-sequence rules as Megatron or veRL-style input prep
    3. shard the packed tokens over the tensor-parallel sequence dimension
    4. select only the MoE layers that belong to the current PP or VPP stage

    Returns:
        A tuple of (replay_tensors, local_routers) where both lists are aligned
        1:1 with the MoE RouterReplay instances for the current rank.

        Reference material:
        - veRL set_router_replay_data:
            https://github.com/verl-project/verl/blob/f88ed61abe1e0f10febd61c06228ca0db030fdd8/verl/utils/megatron/router_replay_utils.py#L269-L324
        - veRL RouterReplayHelper.get_micro_batch_router_list:
            https://github.com/verl-project/verl/blob/f88ed61abe1e0f10febd61c06228ca0db030fdd8/verl/utils/megatron/router_replay_utils.py#L476-L505
        - RouterReplay class:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router_replay.py#L18-L151
        - Sequence-parallel scatter wrappers:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/tensor_parallel/mappings.py#L276-L294
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/tensor_parallel/mappings.py#L493-L520
        - veRL packed-sequence preprocessing:
            https://github.com/verl-project/verl/blob/f88ed61abe1e0f10febd61c06228ca0db030fdd8/verl/models/mcore/util.py
    """
    attention_mask_cpu = attention_mask.to(device="cpu", dtype=torch.bool)
    expanded = expand_routed_experts_to_token_positions(routed_experts, attention_mask_cpu)  # [B, S, L, K]
    pad_value = torch.arange(expanded.shape[-1], dtype=expanded.dtype).unsqueeze(0).expand(expanded.shape[-2], -1)
    packed_replay, _ = preprocess_packed_seqs(expanded, attention_mask_cpu, pre_process=True, pad_value=pad_value)
    # packed_replay: [1, T_packed, L, K]
    packed_replay = packed_replay.contiguous()
    device = attention_mask.device
    packed_replay = packed_replay.to(device)
    sp_replay = scatter_to_sequence_parallel_region(packed_replay.squeeze(0))
    # sp_replay: [T_local, L, K]
    assert sp_replay is not None
    sp_replay = sp_replay.unsqueeze(0)  # [1, T_local, L, K]
    layers_topk_idx = sp_replay.permute(0, 2, 1, 3).squeeze(0)  # [L, T_local, K]
    loss_token_mask: torch.Tensor | None = None
    if loss_mask is not None and getattr(tf_config, "moe_replay_routing_for_loss_tokens_only", False):
        packed_loss_mask, _ = preprocess_packed_seqs(loss_mask.to(device="cpu", dtype=torch.bool), attention_mask_cpu, pre_process=True)
        packed_loss_mask = packed_loss_mask.contiguous().to(device)
        sp_loss_mask = scatter_to_sequence_parallel_region(packed_loss_mask.squeeze(0))
        assert sp_loss_mask is not None
        loss_token_mask = sp_loss_mask.to(torch.bool)

    local_routers = get_micro_batch_router_list(tf_config, vp_rank=vp_rank, pp_rank=pp_rank)
    _set_router_replay_token_masks(local_routers, loss_token_mask)
    replay_tensors = select_local_router_replay_tensors(layers_topk_idx, tf_config, vp_rank=vp_rank, pp_rank=pp_rank)
    # replay_tensors: list[[T_local, K]], one tensor per local MoE router
    assert len(replay_tensors) == len(local_routers), f"Expected {len(local_routers)} replay tensors, got {len(replay_tensors)}"
    return replay_tensors, local_routers


def pack_routing_for_magi(
    merged_per_traj: list[torch.Tensor],
    merge_info_list: list[PrefixMergeInfo],
    device: torch.device,
) -> torch.Tensor:
    """Pack per-trajectory merged routing into ``(sum_total_padded, L, K)``.

    Input ``merged_per_traj[i]`` is trajectory ``i``'s gathered routing,
    shape ``(real_total_i - 1, L, K)`` — one row per visited packed-sequence
    position (except the first, which has no routing). Output is the flat
    tensor Megatron's router consumes: all trajectories right-padded to
    ``total_padded_i`` and concatenated, shape ``(Σ total_padded_i, L, K)``.

    Per trajectory:
    1. Assert the merged tensor has exactly ``real_total_i - 1`` rows. A
       mismatch means the upstream routing and ``PrefixMergeInfo`` have
       desynced (e.g., the Sample was silently truncated after sglang
       captured routing over the longer pre-truncation sequence) — we fail
       loudly rather than slice, because any silent reshaping here would
       corrupt R3 replay.
    2. Right-pad to ``total_padded_i`` with arange-K placeholder rows
       (matches sglang's padding-routing convention; padding-position
       replay is a no-op).
    3. Concatenate along the token dimension.

    Finally move the packed tensor to ``device`` with ``non_blocking=True``
    so the H2D copy can overlap with the forward's compute on CUDA streams.
    """
    assert merged_per_traj, "pack_routing_for_magi: empty merged list"
    assert len(merged_per_traj) == len(merge_info_list)
    sample0 = merged_per_traj[0]
    num_layers, topk = sample0.shape[-2:]
    placeholder_row = torch.arange(topk, dtype=sample0.dtype).view(1, 1, topk).expand(1, num_layers, topk)
    parts: list[torch.Tensor] = []
    for traj_idx, (merged, mi) in enumerate(zip(merged_per_traj, merge_info_list, strict=True)):
        target_len = max(mi.real_total - 1, 0)
        assert merged.shape[0] == target_len, (
            f"pack_routing_for_magi[{traj_idx}]: merged has {merged.shape[0]} rows, expected {target_len} "
            f"(real_total={mi.real_total}, total_padded={mi.total_padded}). "
            "merge_info and routing are out of sync."
        )
        n_pad = mi.total_padded - target_len
        assert n_pad >= 0, f"real_total {mi.real_total} > total_padded {mi.total_padded} — impossible by construction"
        if n_pad > 0:
            parts.append(torch.cat([merged, placeholder_row.expand(n_pad, num_layers, topk)], dim=0))
        else:
            parts.append(merged)
    packed_replay = torch.cat(parts, dim=0).contiguous()
    return packed_replay.to(device, non_blocking=True)


def prepare_magi_router_replay_tensors(
    routed_experts: torch.Tensor,
    merge_info: list[PrefixMergeInfo],
    magi_key: Any,
    tf_config: TransformerConfig,
    *,
    device: torch.device,
    vp_rank: int | None = None,
    pp_rank: int | None = None,
    loss_mask: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[RouterReplay]]:
    """Magi-merged-forward variant of :func:`prepare_router_replay_tensors`.

    Routes the routing tensor through the SAME magi-attention dispatch the
    input forward uses, so per-CP-rank token positions match. For GDN the
    caller passes a Magi key built with Megatron's packed-CP partition
    (forward chunk plus mirrored tail chunk); for transformer attention the
    caller passes the normal Magi flex key.

    ``routed_experts`` is the per-trajectory merged routing produced by the
    R3 materialiser — a jagged nested tensor of ``[real_total_i - 1, L, K]``
    rows that already encodes the trie scatter. We unbind to a list and pack
    directly via :func:`pack_routing_for_magi`, skipping the dense
    ``[B, S, L, K]`` filler.
    """
    from magi_attention.api import dispatch

    merged_per_traj = list(routed_experts.unbind(0))
    packed_replay = pack_routing_for_magi(merged_per_traj, merge_info, device=device)
    # The key is the partition contract. In the Qwen3.6 GDN path it is the
    # Megatron-compatible CP key, so routing follows the same CP rows as the
    # hidden states before any TP sequence-parallel scatter.
    cp_local_replay = dispatch(packed_replay, magi_key)
    gdn_without_embedding_sp = _uses_gdn_without_embedding_sequence_parallel(tf_config)
    if gdn_without_embedding_sp:
        # Full VL GDN keeps router scores at the Magi-dispatched CP-local token
        # length because embedding SP scatter is disabled. Text-only Qwen3.6
        # enables embedding scatter and follows the normal SP-sharded replay path.
        sp_local_replay = cp_local_replay
    else:
        # Standard transformer blocks TP-shard along the sequence dim
        # (sequence_parallel=True), so replay must match that sharding.
        sp_local_replay = scatter_to_sequence_parallel_region(cp_local_replay)
        assert sp_local_replay is not None
    layers_topk_idx = sp_local_replay.transpose(0, 1).contiguous()
    loss_token_mask: torch.Tensor | None = None
    if loss_mask is not None and getattr(tf_config, "moe_replay_routing_for_loss_tokens_only", False):
        packed_loss_mask = torch.cat([loss_mask[i, : mi.total_padded] for i, mi in enumerate(merge_info)], dim=0).to(
            device=device,
            dtype=torch.bool,
        )
        cp_local_mask = dispatch(packed_loss_mask, magi_key)
        if gdn_without_embedding_sp:
            sp_local_mask = cp_local_mask
        else:
            sp_local_mask = scatter_to_sequence_parallel_region(cp_local_mask)
            assert sp_local_mask is not None
        loss_token_mask = sp_local_mask.to(torch.bool)

    local_routers = get_micro_batch_router_list(tf_config, vp_rank=vp_rank, pp_rank=pp_rank)
    _set_router_replay_token_masks(local_routers, loss_token_mask)
    replay_tensors = select_local_router_replay_tensors(layers_topk_idx, tf_config, vp_rank=vp_rank, pp_rank=pp_rank)
    assert len(replay_tensors) == len(local_routers)
    return replay_tensors, local_routers


def prepare_forward_router_replay(
    model: Any,
    attention_mask: torch.Tensor,
    routed_experts: torch.Tensor | None,
    *,
    loss_mask: torch.Tensor | None = None,
) -> list[RouterReplay] | None:
    """Enable replay for the next forward pass and load per-layer top-k indices.

    When routed_experts is present, this helper switches the local Megatron
    RouterReplay instances (for the current VP stage) into REPLAY_FORWARD mode
    and installs the prepared top-k index tensors. When routed_experts is absent,
    it explicitly clears any lingering replay state.

    Returns:
        The list of local RouterReplay instances when replay is activated,
        or None when replay is not active.

        Reference material:
        - RouterReplay actions and buffers:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router_replay.py#L8-L151
        - topk_routing_with_score_function replay hook:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/moe_utils.py#L608-L749
        - TopKRouter.routing:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router.py#L545-L638
    """
    tf_config = unwrap_model(model).config
    if not getattr(tf_config, "moe_enable_routing_replay", False):
        return None
    if routed_experts is None:
        RouterReplay.clear_global_router_replay_action()
        RouterReplay.clear_global_indices()
        _set_router_replay_token_masks(RouterReplay.global_router_replay_instances, None)
        return None

    unwrapped_model = unwrap_model(model)
    vp_rank = unwrapped_model.vp_stage
    replay_tensors, local_routers = prepare_router_replay_tensors(
        routed_experts,
        attention_mask,
        tf_config,
        vp_rank=vp_rank,
        loss_mask=loss_mask,
    )
    for router, tensor in zip(local_routers, replay_tensors, strict=True):
        router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        router.set_target_indices(tensor)
    return local_routers


def finish_forward_router_replay(*, replay_routers: list[RouterReplay] | None) -> None:
    """Switch replay state from forward replay to backward or recompute replay.

    Only the routers that were activated during the forward pass are switched
    to REPLAY_BACKWARD. This is VPP-safe: routers belonging to other VP stages
    are not modified.

    Megatron uses a separate REPLAY_BACKWARD action so activation recomputation
    during backward can consume the same routing indices as the forward pass.

        Reference material:
        - REPLAY_BACKWARD behavior in design doc:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/docs/api-guide/router_replay.md#L54-L58
        - REPLAY_BACKWARD implementation:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router_replay.py#L116-L151
    """
    if replay_routers is not None:
        for router in replay_routers:
            router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)


def clear_router_replay_state() -> None:
    """Clear global replay actions and cached indices after a train or eval step.

    This is the final cleanup point for R3 lifecycle management and prevents
    stale replay tensors from leaking into subsequent micro-batches.

        Reference material:
        - RouterReplay cleanup helpers:
            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/moe/router_replay.py#L43-L69
    """
    RouterReplay.clear_global_router_replay_action()
    RouterReplay.clear_global_indices()
    _set_router_replay_token_masks(RouterReplay.global_router_replay_instances, None)
