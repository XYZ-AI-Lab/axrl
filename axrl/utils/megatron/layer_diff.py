"""Three-way per-layer hidden-state diff between TE, Magi-flat, Magi-merged forwards.

Used as a diagnostic to locate the first submodule where a Magi path diverges
from the TE baseline. Single-rank only (TP=PP=CP=DP=1).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from magi_attention.api import magi_attn_flex_key
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType as MagiAttnMaskType
from megatron.core import parallel_state as mpu
from megatron.core.utils import divide
from tensordict import TensorDict

from axrl.utils.megatron.magi_forward import (
    _build_single_path_merge_info,
    magi_flat_gptmodel_forward,
    magi_merged_gptmodel_forward,
)
from axrl.utils.megatron.model_forward import gptmodel_forward
from axrl.utils.megatron.pack_utils import postprocess_packed_seqs, preprocess_packed_seqs
from axrl.utils.megatron.prefix_tree import (
    PrefixTreeNode,
    build_prefix_merge_info,
    build_prefix_tree_ranges,
    extract_paths_from_batch,
    merge_prefix_merge_infos,
    pack_tree_aligned_as_list,
    scatter_packed_to_batch,
)
from axrl.utils.megatron.utils import unwrap_model

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.models.gpt import GPTModel
    from megatron.core.transformer.module import Float16Module


def magi_prefix_merging_layer_diff(  # noqa: PLR0915 - diagnostic threads three forwards + many hooks
    model_chunks: Sequence[GPTModel | Float16Module | DDP],
    samples: Any,
) -> dict[str, Any]:
    """Three-way per-layer hidden-state diff (``te`` / ``magi_flat`` / ``magi_merged``).

    Runs three forward paths on the same samples with hooks on every decoder
    layer + every submodule in layer 0 + a pre-hook on ``core_attention``
    (captures post-RoPE Q/K/V), then returns per-hook pairwise
    ``max_abs``/``mean_abs`` deltas at ``loss_mask=True`` positions:

    - ``te_to_flat``: kernel-implementation delta (TE FA3 THD vs Magi
      ``calc_attn`` with per-sample causal flex ranges).
    - ``flat_to_merged``: pure prefix-merging delta (same ``calc_attn``
      kernel on both sides, different ``(q_ranges, k_ranges)``).
    - ``te_to_merged``: end-to-end delta (both switches together).

    ``model_chunks`` is the worker's ``self.model`` list. Single-rank only.
    """
    assert mpu.get_tensor_model_parallel_world_size() == 1
    assert mpu.get_pipeline_model_parallel_world_size() == 1
    assert mpu.get_context_parallel_world_size() == 1
    assert mpu.get_data_parallel_world_size() == 1

    model = model_chunks[0]
    for chunk in model_chunks:
        chunk.eval()

    tensor_dict = TensorDict(dict(samples), batch_size=len(samples)).to(torch.cuda.current_device())
    input_ids = tensor_dict["input_ids"]
    attention_mask = tensor_dict["attention_mask"].to(torch.bool)
    loss_mask = tensor_dict["loss_mask"].to(torch.bool)
    batch_size, seq_len = input_ids.shape
    device: torch.device = input_ids.device  # type: ignore[assignment]

    unwrapped = unwrap_model(model)
    hook_points = _hook_points_layer_and_layer0_submodules(unwrapped)

    def run(fwd: Callable[..., Any]) -> dict[str, torch.Tensor | None]:
        store: dict[str, torch.Tensor | None] = {}
        handles = _install_hooks(store, unwrapped, hook_points)
        with torch.no_grad():
            fwd(model=model, input_ids=input_ids, attention_mask=attention_mask)
        for h in handles:
            h.remove()
        return store

    # The unified ``magi_merged_gptmodel_forward`` requires a pre-built ``PrefixMergeInfo``;
    # build one from the batch so the diagnostic continues to drive the merged path.
    paths = extract_paths_from_batch(input_ids, attention_mask)
    cp = mpu.get_context_parallel_world_size()
    tp = mpu.get_tensor_model_parallel_world_size()
    align = max(tp, 1) * max(cp, 1)
    merged_packed_list, merged_nodes, merged_path_to_leaf, merged_total_padded = pack_tree_aligned_as_list(paths, align_size=align)
    merged_packed_tensor = torch.tensor(merged_packed_list, dtype=torch.long, device=device)
    diag_merge_info = build_prefix_merge_info(
        nodes=merged_nodes,
        path_to_leaf=merged_path_to_leaf,
        total_padded=merged_total_padded,
        turn_sample_lens=[len(p) for p in paths],
        # Diagnostic-only: uses an all-ones attention_mask (below), so every
        # packed slot counts as visited.
        real_total=merged_total_padded,
    )
    merged_input_ids = merged_packed_tensor.unsqueeze(0)
    merged_attention_mask = torch.ones_like(merged_input_ids, dtype=torch.bool)

    def run_merged() -> dict[str, torch.Tensor | None]:
        store: dict[str, torch.Tensor | None] = {}
        handles = _install_hooks(store, unwrapped, hook_points)
        with torch.no_grad():
            partial(magi_merged_gptmodel_forward, merge_info=[diag_merge_info])(
                model=model, input_ids=merged_input_ids, attention_mask=merged_attention_mask
            )
        for h in handles:
            h.remove()
        return store

    te_store = run(gptmodel_forward)
    flat_store = run(magi_flat_gptmodel_forward)
    merged_store = run_merged()

    # Build unpack contexts for each layout (TE packed, Magi flat, Magi merged)
    # so we can put each captured hidden state back into (B, S, H).
    te_ctx = _TeUnpackCtx.build(input_ids, attention_mask)
    # Flat path is now per-trajectory aligned (matches ``magi_flat_gptmodel_forward``,
    # which builds K single-path PrefixMergeInfos and concatenates them via
    # ``merge_prefix_merge_infos``). Mirror that here so unpack uses the same layout.
    flat_combined = merge_prefix_merge_infos([_build_single_path_merge_info(len(p), align=align) for p in paths])
    flat_ctx = _MagiUnpackCtx.from_merge_info(flat_combined, device, unwrapped.config)
    # Reuse ``diag_merge_info`` (already built above for the merged forward) so we
    # don't repeat the trie construction.
    merged_ctx = _MagiUnpackCtx.from_merge_info(diag_merge_info, device, unwrapped.config)

    def pair(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, float | None]:
        if a is None or b is None:
            return {"max_abs_at_loss_mask": None, "mean_abs_at_loss_mask": None, "max_abs_at_valid": None}
        diff = (a.float() - b.float()).abs()
        at_loss = diff[loss_mask]
        at_valid = diff[attention_mask]
        return {
            "max_abs_at_loss_mask": float(at_loss.max().item()) if at_loss.numel() else 0.0,
            "mean_abs_at_loss_mask": float(at_loss.mean().item()) if at_loss.numel() else 0.0,
            "max_abs_at_valid": float(at_valid.max().item()) if at_valid.numel() else 0.0,
        }

    rows: list[dict[str, Any]] = []
    names = [n for n, _ in hook_points] + [k for k in te_store if k.startswith("layer[0].core_attention.in_")]
    for name in names:
        te_raw = te_store.get(name)
        flat_raw = flat_store.get(name)
        merged_raw = merged_store.get(name)
        te_up = te_ctx.unpack(te_raw, batch_size, seq_len, attention_mask) if te_raw is not None else None
        flat_up = flat_ctx.unpack(flat_raw, batch_size, seq_len, attention_mask) if flat_raw is not None else None
        merged_up = merged_ctx.unpack(merged_raw, batch_size, seq_len, attention_mask) if merged_raw is not None else None

        rows.append(
            {
                "name": name,
                "shape_te": tuple(te_raw.shape) if te_raw is not None else None,
                "shape_flat": tuple(flat_raw.shape) if flat_raw is not None else None,
                "shape_merged": tuple(merged_raw.shape) if merged_raw is not None else None,
                "te_to_flat": pair(te_up, flat_up),
                "flat_to_merged": pair(flat_up, merged_up),
                "te_to_merged": pair(te_up, merged_up),
            }
        )
    return {"rows": rows}


# --------------------------------------------------------------------- #
# Hook installation
# --------------------------------------------------------------------- #


def _hook_points_layer_and_layer0_submodules(unwrapped: Any) -> list[tuple[str, Any]]:
    hook_points: list[tuple[str, Any]] = []
    for layer_idx, layer in enumerate(unwrapped.decoder.layers):
        hook_points.append((f"layer[{layer_idx}]", layer))
    for name, module in unwrapped.decoder.layers[0].named_modules():
        if name == "":
            continue
        hook_points.append((f"layer[0].{name}", module))
    return hook_points


def _install_hooks(
    store: dict[str, torch.Tensor | None],
    unwrapped: Any,
    hook_points: list[tuple[str, Any]],
) -> list[Any]:
    handles: list[Any] = []
    for hook_name, module in hook_points:
        store[hook_name] = None

        def _hook(_module: Any, _inp: Any, out: Any, key: str = hook_name) -> None:
            if isinstance(out, tuple):
                for item in out:
                    if isinstance(item, torch.Tensor):
                        store[key] = item.detach()
                        return
                return
            if isinstance(out, torch.Tensor):
                store[key] = out.detach()

        handles.append(module.register_forward_hook(_hook))

    # Pre-hook on the attention kernel to capture post-RoPE Q, K, V.
    kernel = getattr(unwrapped.decoder.layers[0].self_attention, "core_attention", None)
    if kernel is not None:
        for which in ("in_q", "in_k", "in_v"):
            store[f"layer[0].core_attention.{which}"] = None

        def _pre(_module: Any, inp: Any) -> None:
            args = inp if isinstance(inp, tuple) else (inp,)
            if len(args) >= 3:
                for idx, which in enumerate(("in_q", "in_k", "in_v")):
                    t = args[idx]
                    if isinstance(t, torch.Tensor):
                        store[f"layer[0].core_attention.{which}"] = t.detach()

        handles.append(kernel.register_forward_pre_hook(_pre))
    return handles


# --------------------------------------------------------------------- #
# Layout-specific unpackers
# --------------------------------------------------------------------- #


def _to_thd(h: torch.Tensor) -> torch.Tensor | None:
    """Normalize a captured hidden state to ``(T, *feat)``."""
    if h.dim() == 3 and h.size(1) == 1:
        return h.squeeze(1)
    if h.dim() == 4 and h.size(1) == 1:
        return h.squeeze(1)
    if h.dim() == 3:  # (T, heads, head_dim) — core_attention pre-hook
        return h
    if h.dim() == 2:  # (T, feat) — core_attention output
        return h
    return None


class _TeUnpackCtx:
    """Unpack TE packed layout back to (B, S, *feat)."""

    def __init__(self, packed_params: Any) -> None:
        self.params = packed_params

    @classmethod
    def build(cls, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _TeUnpackCtx:
        _, params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        return cls(params)

    def unpack(self, h: torch.Tensor, batch_size: int, seq_len: int, attention_mask: torch.Tensor) -> torch.Tensor | None:
        thd = _to_thd(h)
        if thd is None:
            return None
        feat = thd.shape[1:]
        flat = thd.reshape(thd.size(0), -1).unsqueeze(0)  # (1, T, F)
        unpacked = postprocess_packed_seqs(flat, self.params, attention_mask, batch_size, seq_len, post_process=True)
        return unpacked.reshape(batch_size, seq_len, *feat)


class _MagiUnpackCtx:
    """Unpack Magi-packed layout (flat or merged trie) back to (B, S, *feat)."""

    def __init__(self, magi_key: Any, nodes: list[PrefixTreeNode], path_to_leaf: list[tuple[int, int]]) -> None:
        self.key = magi_key
        self.nodes = nodes
        self.path_to_leaf = path_to_leaf

    @classmethod
    def from_merge_info(cls, info: Any, device: torch.device, cfg: Any) -> _MagiUnpackCtx:
        """Build the unpack context directly from a pre-computed ``PrefixMergeInfo``.

        Both the flat and merged diagnostics build a ``PrefixMergeInfo`` upstream
        and call this — keeping the unpack metadata in lock-step with whatever
        the actual forward saw.
        """
        nodes = info.nodes
        path_to_leaf = info.path_to_leaf
        total_padded = info.total_padded

        q_t, k_t, attn_type_map = build_prefix_tree_ranges(nodes, device=device)
        q_ranges = AttnRanges.from_ranges(q_t.cpu().tolist())
        k_ranges = AttnRanges.from_ranges(k_t.cpu().tolist())
        mask_list = [MagiAttnMaskType.FULL if int(x) == 0 else MagiAttnMaskType.CAUSAL for x in attn_type_map.cpu().tolist()]
        num_query_groups = cfg.num_query_groups if cfg.num_query_groups is not None else cfg.num_attention_heads
        num_heads_q = divide(cfg.num_attention_heads, 1)
        num_heads_kv = 1 if num_query_groups < 1 else divide(num_query_groups, 1)
        head_dim = cfg.kv_channels if cfg.kv_channels is not None else (cfg.hidden_size // cfg.num_attention_heads)
        cp_group = mpu.get_context_parallel_group()
        assert cp_group is not None, "Magi requires an initialized context-parallel group"
        magi_key = magi_attn_flex_key(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=mask_list,
            total_seqlen_q=total_padded,
            total_seqlen_k=total_padded,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            pad_size=0,
            cp_group_or_mesh=cp_group,  # pyright: ignore[reportArgumentType]
        )
        return cls(magi_key, nodes, path_to_leaf)

    def unpack(self, h: torch.Tensor, batch_size: int, seq_len: int, attention_mask: torch.Tensor) -> torch.Tensor | None:
        from magi_attention.api import undispatch

        thd = _to_thd(h)
        if thd is None:
            return None
        feat = thd.shape[1:]
        flat = thd.reshape(thd.size(0), -1)
        global_out = undispatch(flat, self.key)
        unpacked = scatter_packed_to_batch(global_out, self.nodes, self.path_to_leaf, attention_mask)
        return unpacked.reshape(batch_size, seq_len, *feat)
