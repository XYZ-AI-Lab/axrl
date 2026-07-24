from __future__ import annotations

import logging
import math
import os
import shutil
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from megatron.core import parallel_state as mpu

from axrl.utils.megatron.prefix_tree import (
    PrefixMergeInfo,
    PrefixTreeNode,
    build_prefix_merge_info,
    compute_tree_rel_positions,
    extract_paths_from_batch,
)
from axrl.utils.megatron.utils import unwrap_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeKey
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.models.gpt import GPTModel
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


@dataclass
class MagiForwardContext:
    """Per-forward-call context consumed by the Magi attention & RoPE patches.

    Attributes:
        magi_key: DistAttnRuntimeKey driving `calc_attn` inside the
            monkey-patched attention layer.
        position_ids: Local shard's per-token position-within-path;
            consumed by the patched RoPE to pick `freqs[position_ids]`.
    """

    magi_key: DistAttnRuntimeKey
    position_ids: torch.Tensor


_current: ContextVar[MagiForwardContext | None] = ContextVar("axrl_magi_forward_context", default=None)


def current_magi_context() -> MagiForwardContext | None:
    """Return the active Magi forward context (or None)."""
    return _current.get()


# Mutable container so `install_magi_attention_patch` doesn't need a
# `global` statement — shared idempotency flag.
_patch_installed: list[bool] = [False]
_ffa_head_dim_patch_installed: list[bool] = [False]


def install_magi_attention_patch() -> None:
    """Install the TE-attention + RoPE + checkpoint monkey-patches exactly once."""
    if _patch_installed[0]:
        return
    _patch_magi_ffa_forward_head_dim_256()
    _patch_te_attention()
    _patch_rotary_pos_emb()
    _patch_checkpoint_functions()
    _patch_installed[0] = True
    logger.info("Installed Magi Attention monkey-patches (attention + RoPE + checkpoint).")


def _patch_magi_ffa_forward_head_dim_256() -> None:
    """Allow Magi FFA forward JIT for Qwen3.6's 256-dim attention heads."""
    if _ffa_head_dim_patch_installed[0]:
        return

    import inspect

    from magi_attention.functional import _flex_flash_attn_jit as ffa_jit

    _patch_magi_ffa_common_sources_for_head_dim_256(ffa_jit)
    _patch_magi_ffa_forward_head_dim_256_tile(ffa_jit)

    original_sanity_check = ffa_jit.sanity_check
    if getattr(original_sanity_check, "_axrl_allows_fwd_head_dim_256", False):
        _ffa_head_dim_patch_installed[0] = True
        return

    original_signature = inspect.signature(original_sanity_check)

    def patched_sanity_check(*args: Any, **kwargs: Any) -> Any:
        bound = original_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        direction = values["direction"]
        head_dim = values["head_dim"]
        if direction != "fwd" or head_dim <= 128:
            return original_sanity_check(*args, **kwargs)

        ffa_jit.check_cuda_compute_capability(values["arch"])
        assert head_dim <= 256, "AXRL Magi FFA forward only supports head_dim <= 256"
        assert ffa_jit.round_up_headdim(head_dim) in (
            192,
            256,
        ), "AXRL Magi FFA forward head_dim patch only supports rounded head_dim 192 or 256"
        assert values["compute_dtype"] in (
            torch.float16,
            torch.bfloat16,
        ), "compute_dtype must be float16 or bfloat16"
        assert values["output_dtype"] in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ), "output_dtype must be float16, bfloat16 or float32"
        assert values["dq_dtype"] is None, "dq_dtype must be None when direction == 'fwd'"
        assert values["dkv_dtype"] is None, "dkv_dtype must be None when direction == 'fwd'"

        ref_block_size = values["ref_block_size"]
        if values["swap_ab"]:
            assert ref_block_size in (
                (8, 64),
                (16, 64),
                (32, 64),
                (64, 64),
            ), "ref_block_size must be (8, 64), (16, 64), (32, 64) or (64, 64) when swap_ab == True"
        elif ref_block_size is not None:
            kblock_m, kblock_n = ref_block_size
            assert kblock_m in (
                64,
                128,
                192,
            ), "ref_block_size: (kblock_m, kblock_n), kblock_m must be 64, 128 or 192 when swapab == False"
            assert kblock_n % 16 == 0 and kblock_n <= 256, "ref_block_size: (kblock_m, kblock_n), kblock_n <= 256 and kblock_n % 16 == 0 must be True"

        assert not values["swap_bwd_qk_loop"], "swap_bwd_qk_loop only take effect when direction == 'bwd'"
        assert not (values["pack_gqa"] and values["cat_gqa"]), "pack_gqa and cat_gqa cannot be both True"
        assert not values["cat_gqa"], "cat_gqa only take effect when direction == 'bwd'"
        return None

    patched_sanity_check._axrl_allows_fwd_head_dim_256 = True  # type: ignore[attr-defined]
    ffa_jit.sanity_check = patched_sanity_check
    _ffa_head_dim_patch_installed[0] = True


def _patch_magi_ffa_forward_head_dim_256_tile(ffa_jit: Any) -> None:
    """Use a launchable FFA tile for Qwen3.6's float32-output 256-head forward."""
    import importlib
    import inspect

    flex_flash_attn_mod = cast("Any", importlib.import_module("magi_attention.functional.flex_flash_attn"))
    original_get_ffa_jit_mod = ffa_jit.get_ffa_jit_mod
    if getattr(original_get_ffa_jit_mod, "_axrl_forces_fwd_head_dim_256_tile", False):
        flex_flash_attn_mod.get_ffa_jit_mod = original_get_ffa_jit_mod
        return

    original_signature = inspect.signature(original_get_ffa_jit_mod)

    def patched_get_ffa_jit_mod(*args: Any, **kwargs: Any) -> Any:
        bound = original_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if (
            values["direction"] == "fwd"
            and values["head_dim"] == 256
            and values["output_dtype"] is torch.float32
            and values["ref_block_size"] is None
        ):
            # Magi's default 128x64 tile compiles for this path, but H200 rejects
            # its dynamic shared-memory launch for float32 output. 64x64 launches
            # successfully while staying on native FFA.
            values["ref_block_size"] = (64, 64)
        return original_get_ffa_jit_mod(**values)

    patched_get_ffa_jit_mod._axrl_forces_fwd_head_dim_256_tile = True  # type: ignore[attr-defined]
    cache_clear = getattr(original_get_ffa_jit_mod, "cache_clear", None)
    if cache_clear is not None:
        patched_get_ffa_jit_mod.cache_clear = cache_clear  # type: ignore[attr-defined]
    cache_info = getattr(original_get_ffa_jit_mod, "cache_info", None)
    if cache_info is not None:
        patched_get_ffa_jit_mod.cache_info = cache_info  # type: ignore[attr-defined]
    ffa_jit.get_ffa_jit_mod = patched_get_ffa_jit_mod
    flex_flash_attn_mod.get_ffa_jit_mod = patched_get_ffa_jit_mod


def _patch_magi_ffa_common_sources_for_head_dim_256(ffa_jit: Any) -> None:
    """Point Magi's JIT at a patched FFA common source with fwd hdim-256 symbols."""
    from filelock import FileLock
    from magi_attention.common.jit import env as jit_env

    original_dir = jit_env.FLEXIBLE_FLASH_ATTENTION_CSRC_DIR
    original_source = original_dir / "flash_fwd_postprocess.cu"
    original_text = original_source.read_text(encoding="utf-8")
    if "run_flash_fwd_post_process_<cutlass::half_t, 256>" in original_text:
        return

    patch_dir = Path(
        os.environ.get(
            "AXRL_MAGI_FFA_PATCH_DIR",
            str(Path.home() / ".cache" / "axrl" / "magi_attention" / "flexible_flash_attention_hdim256"),
        )
    )
    marker = patch_dir / ".axrl_hdim256_fwd_postprocess"
    lock_path = patch_dir.parent / f"{patch_dir.name}.lock"
    patch_dir.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), thread_local=False):
        patched_source = patch_dir / "flash_fwd_postprocess.cu"
        if not marker.exists() or not patched_source.exists():
            shutil.copytree(original_dir, patch_dir, dirs_exist_ok=True)
            patched_text = original_text
            replacements = {
                "template void run_flash_fwd_post_process_<float, 192>(Flash_fwd_params& params, cudaStream_t stream);": (
                    "template void run_flash_fwd_post_process_<float, 192>(Flash_fwd_params& params, cudaStream_t stream);\n"
                    "template void run_flash_fwd_post_process_<float, 256>(Flash_fwd_params& params, cudaStream_t stream);"
                ),
                "template void run_flash_fwd_post_process_<cutlass::bfloat16_t, 192>(Flash_fwd_params& params, cudaStream_t stream);": (
                    "template void run_flash_fwd_post_process_<cutlass::bfloat16_t, 192>(Flash_fwd_params& params, cudaStream_t stream);\n"
                    "template void run_flash_fwd_post_process_<cutlass::bfloat16_t, 256>(Flash_fwd_params& params, cudaStream_t stream);"
                ),
                "template void run_flash_fwd_post_process_<cutlass::half_t, 192>(Flash_fwd_params& params, cudaStream_t stream);": (
                    "template void run_flash_fwd_post_process_<cutlass::half_t, 192>(Flash_fwd_params& params, cudaStream_t stream);\n"
                    "template void run_flash_fwd_post_process_<cutlass::half_t, 256>(Flash_fwd_params& params, cudaStream_t stream);"
                ),
            }
            for old, new in replacements.items():
                assert old in patched_text, f"Unexpected Magi FFA postprocess source format, missing: {old}"
                patched_text = patched_text.replace(old, new, 1)
            patched_source.write_text(patched_text, encoding="utf-8")

            # The old 256-head forward module can exist but fail at dlopen because
            # its common object lacks the forward postprocess symbol. Rebuild only
            # the affected JIT entries.
            for cache_dir in [jit_env.MAGI_ATTENTION_JIT_DIR / "256hd_common"]:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
            for cache_dir in jit_env.MAGI_ATTENTION_JIT_DIR.glob("flex_flash_attn_sm_*_fwd_256hd_*"):
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
            marker.write_text("patched\n", encoding="utf-8")

    jit_env.FLEXIBLE_FLASH_ATTENTION_CSRC_DIR = patch_dir
    cache_clear = getattr(ffa_jit.get_ffa_jit_mod, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def _patch_checkpoint_functions() -> None:
    """Preserve the magi context across activation recompute.

    Both megatron-core's ``CheckpointFunction`` (bf16/fp16 path) and TE's
    ``_CheckpointFunction`` (fp8/fp4 path) re-run the forward closure inside
    ``backward``. That closure reads ``_current`` (a ContextVar) which has
    already been ``reset()`` by the time backward fires, so the patched
    RoPE/attention asserts. We capture the active context onto each
    per-microbatch ``ctx`` during forward and re-set the ContextVar before
    the recompute fires in backward.
    """
    from megatron.core.tensor_parallel.random import CheckpointFunction

    _patch_checkpoint_class(CheckpointFunction)

    # TE's class is only used on the fp8/fp4 path. Skip silently when TE
    # isn't importable — that environment can't take the fp8 path anyway.
    try:
        from transformer_engine.pytorch.distributed import _CheckpointFunction as TECheckpointFunction
    except ImportError:
        return
    _patch_checkpoint_class(TECheckpointFunction)


def _patch_checkpoint_class(cls: type) -> None:
    """Wrap ``cls.forward`` / ``cls.backward`` so the magi context survives recompute.

    Read the originals from ``cls.__dict__`` rather than ``cls.forward.__func__``:
    on ``torch.autograd.Function`` subclasses the descriptor protocol returns a
    bare function with no ``__func__`` attribute, so the latter raises
    ``AttributeError`` at install time and the error gets swallowed inside
    autograd's C++ stack on the next forward.
    """
    raw_forward = cls.__dict__["forward"]
    raw_backward = cls.__dict__["backward"]
    original_forward = raw_forward.__func__ if isinstance(raw_forward, staticmethod) else raw_forward
    original_backward = raw_backward.__func__ if isinstance(raw_backward, staticmethod) else raw_backward

    def patched_forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        ctx.magi_ctx = current_magi_context()
        return original_forward(ctx, *args, **kwargs)

    def patched_backward(ctx: Any, *args: Any) -> Any:
        magi_ctx = getattr(ctx, "magi_ctx", None)
        assert magi_ctx is not None, (
            f"{cls.__qualname__}.backward fired with no saved magi context. "
            "patched_forward should have captured ``current_magi_context()`` onto ``ctx`` — "
            "if it didn't, either the forward bypassed the Magi forward fn entirely "
            "(use_magi_merged_forward off?) or install order is wrong."
        )
        token = _current.set(magi_ctx)
        result = original_backward(ctx, *args)
        _current.reset(token)
        return result

    cls.forward = staticmethod(patched_forward)
    cls.backward = staticmethod(patched_backward)


def _patch_te_attention() -> None:
    from megatron.core.extensions import transformer_engine as te_ext

    def patched_forward(
        self: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        attn_mask_type: AttnMaskType,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        num_splits: int | None = None,
    ) -> torch.Tensor:
        del self, attention_mask, attn_mask_type, attention_bias, packed_seq_params, num_splits
        ctx = current_magi_context()
        assert ctx is not None, (
            "Magi Attention patch is installed but no MagiForwardContext is active. "
            "Every forward must be driven through `magi_merged_gptmodel_forward` "
            "(use_magi_merged_forward=True) or `magi_flat_gptmodel_forward` "
            "(use_magi_flat_forward=True)."
        )
        return _magi_attention_forward(query, key, value, ctx=ctx)

    te_ext.TEDotProductAttention.forward = patched_forward


def _patch_rotary_pos_emb() -> None:
    """Redirect Megatron's ``apply_rotary_pos_emb`` through the Magi context.

    Uses the same TE fused sbhd RoPE kernel that the baseline path hits,
    just with the ``freqs`` tensor pre-gathered by ``position_ids`` so
    ``freqs[s]`` in the kernel returns the correct per-token frequency
    for each post-dispatch packed-tree position.
    """
    from megatron.core.models.common.embeddings import rope_utils

    def patched_apply(
        t: torch.Tensor,
        freqs: torch.Tensor,
        config: TransformerConfig,
        cu_seqlens: torch.Tensor | None = None,
        mscale: float = 1.0,
        cp_group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        del cu_seqlens, cp_group
        ctx = current_magi_context()
        assert ctx is not None, (
            "Magi RoPE patch is installed but no MagiForwardContext is active. "
            "Every forward must be driven through `magi_merged_gptmodel_forward` "
            "(use_magi_merged_forward=True) or `magi_flat_gptmodel_forward` "
            "(use_magi_flat_forward=True)."
        )
        assert not config.multi_latent_attention, "Multi-latent attention is not yet supported by the fused RoPE path."
        return _apply_rope_te_fused(
            t,
            freqs,
            position_ids=ctx.position_ids,
            rotary_interleaved=config.rotary_interleaved,
            mscale=mscale,
        )

    rope_utils.apply_rotary_pos_emb = patched_apply
    from megatron.core.transformer import attention as attn_mod

    if hasattr(attn_mod, "apply_rotary_pos_emb"):
        attn_mod.apply_rotary_pos_emb = patched_apply


def _apply_rope_te_fused(
    t: torch.Tensor,
    freqs: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    rotary_interleaved: bool,
    mscale: float,
) -> torch.Tensor:
    """Fused RoPE via TE's ``fused_apply_rotary_pos_emb`` (sbhd).

    Pre-gathers ``freqs[position_ids]`` so the TE sbhd kernel — which
    indexes ``freqs[s]`` by the packed layout position — sees the
    right per-token frequency, then calls the exact same kernel the
    baseline uses via ``apply_rope_fusion=True``.

    ``mscale != 1.0`` (used by YaRN-style RoPE) scales the **rotation
    magnitude**, not the angle — the previous unfused implementation
    multiplied cos/sin by mscale, which is not equivalent to multiplying
    ``freqs`` by mscale. Until we add a proper unfused fallback, refuse
    rather than silently produce wrong output. Standard non-MLA RoPE
    always uses ``mscale=1.0`` so this is currently inert.
    """
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb

    assert mscale == 1.0, f"TE-fused RoPE patch only supports mscale=1.0, got {mscale}"

    squeezed = t.dim() == 4 and t.size(1) == 1
    t3 = t.squeeze(1) if squeezed else t  # (T, H, D)
    # TE sbhd expects (S, B, H, D). Treat dispatched sequence as (T, 1, H, D).
    t_sbhd = t3.unsqueeze(1)

    # freqs incoming shape: (seq_len, 1, 1, D). index_select gathers per-token.
    pos = position_ids.to(dtype=torch.long, device=t3.device)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if pos.numel() != t3.size(0) and tp_size > 1 and pos.numel() * tp_size == t3.size(0):
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

        # Sequence-parallel QKV projections all-gather the sequence dimension before
        # RoPE. Keep Magi's per-token positions in the same TP-rank concatenation.
        pos = gather_from_sequence_parallel_region(pos).to(dtype=torch.long)
    freqs_flat = freqs.reshape(-1, freqs.shape[-1])
    gathered = freqs_flat.index_select(0, pos)
    # TE sbhd kernel expects freqs of shape (S, 1, 1, D).
    freqs_sbhd = gathered.unsqueeze(1).unsqueeze(1)

    out = fused_apply_rotary_pos_emb(t_sbhd, freqs_sbhd, interleaved=rotary_interleaved)
    out3 = out.squeeze(1)
    return out3.unsqueeze(1) if squeezed else out3


def _magi_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ctx: MagiForwardContext,
) -> torch.Tensor:
    """Route the attention call to Magi's distributed ``calc_attn``.

    Input shape (THD): ``(t, 1, np, hn)`` or ``(t, np, hn)``. Output
    shape: ``(t, 1, np*hn)`` — matches the reshape Megatron applies
    right after the core-attention call for THD.
    """
    from magi_attention.api import calc_attn

    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        assert x.dim() == 3, f"Expected 3D tensor after squeeze, got {tuple(x.shape)}"
        return x

    q = _flatten(query).contiguous()
    k = _flatten(key).contiguous()
    v = _flatten(value).contiguous()

    # Note: we should check how to work with fp8 (FA3 supports FP8
    # attention on Hopper+; widening this gate + threading amax/scales
    # through `calc_attn` is the likely path once Magi exposes it).
    orig_dtype = q.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    out, _ = calc_attn(q=q, k=k, v=v, key=ctx.magi_key)
    out = out.to(orig_dtype)
    return out.reshape(out.size(0), 1, -1)


def _build_megatron_cp_partitions(*, sequence_lengths: list[int], cp_size: int) -> tuple[int, list[list[int]]]:
    """Return the packed-CP chunk order used by ``preprocess_packed_seqs``.

    GDN consumes the packed sequence in Megatron's CP layout: each CP rank gets
    its forward chunk and the mirrored tail chunk for every packed sequence.
    Routing replay must use this same partition when the Magi key is built for
    GDN, otherwise replayed experts align to the wrong token rows.
    """
    assert sequence_lengths and all(length > 0 for length in sequence_lengths), f"Invalid sequence lengths: {sequence_lengths}"
    total_padded = sum(sequence_lengths)
    if cp_size == 1:
        return total_padded, [[0]]

    half_lengths: list[int] = []
    for length in sequence_lengths:
        assert length % (2 * cp_size) == 0, f"Packed length {length} is not divisible by 2*CP={2 * cp_size}"
        half_lengths.append(length // (2 * cp_size))
    chunk_size = half_lengths[0]
    for half_length in half_lengths[1:]:
        chunk_size = math.gcd(chunk_size, half_length)
    assert chunk_size > 0

    partitions: list[list[int]] = [[] for _ in range(cp_size)]
    chunk_offset = 0
    for half_length in half_lengths:
        chunks_per_half = half_length // chunk_size
        chunks_per_seq = 2 * cp_size * chunks_per_half
        for rank in range(cp_size):
            first_start = chunk_offset + rank * chunks_per_half
            tail_start = chunk_offset + (2 * cp_size - rank - 1) * chunks_per_half
            partitions[rank].extend(range(first_start, first_start + chunks_per_half))
            partitions[rank].extend(range(tail_start, tail_start + chunks_per_half))
        chunk_offset += chunks_per_seq
    assert chunk_offset * chunk_size == total_padded, (chunk_offset, chunk_size, total_padded)
    return chunk_size, partitions


def _build_megatron_cp_magi_key(
    *,
    q_ranges: Any,
    k_ranges: Any,
    mask_type_list: list[Any],
    total_padded: int,
    sequence_boundaries: list[int] | None,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cp_group: ProcessGroup,
) -> Any:
    """Build a Magi key whose token shards match Megatron packed-CP layout."""
    import torch.distributed as dist
    from magi_attention.api.magi_attn_interface import dist_attn_runtime_dict_mgr
    from magi_attention.common.enum import AttnRole, AttnType
    from magi_attention.config import DispatchConfig, DistAttnConfig, SequentialDispatchAlg
    from magi_attention.dist_attn_runtime_mgr import init_dist_attn_runtime_key, init_dist_attn_runtime_mgr
    from magi_attention.meta.collection import DispatchMeta
    from magi_attention.utils import flatten_nested_list, perm_idxs2unperm_idxs

    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    boundaries = sequence_boundaries if sequence_boundaries is not None else [0, total_padded]
    lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    assert lengths and all(length > 0 for length in lengths), f"Invalid packed sequence boundaries: {boundaries}"
    chunk_size, partitions = _build_megatron_cp_partitions(sequence_lengths=lengths, cp_size=cp_size)

    partitions_perm_idxs = flatten_nested_list(partitions)
    partitions_unperm_idxs = perm_idxs2unperm_idxs(partitions_perm_idxs)
    num_chunks = len(partitions_perm_idxs)
    shard_seqlen = total_padded // cp_size
    dispatch_meta_q = DispatchMeta(
        attn_role=AttnRole.QUERY,
        attn_type=AttnType.SELF_ATTN,
        total_seqlen=total_padded,
        shard_seqlen=shard_seqlen,
        max_valid_ids=total_padded,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        cp_rank=cp_rank,
        cp_size=cp_size,
        partitions=partitions,
        partitions_perm_idxs=partitions_perm_idxs,
        partitions_unperm_idxs=partitions_unperm_idxs,
    )
    dispatch_meta_k = DispatchMeta(
        attn_role=AttnRole.KEY,
        attn_type=AttnType.SELF_ATTN,
        total_seqlen=total_padded,
        shard_seqlen=shard_seqlen,
        max_valid_ids=total_padded,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        cp_rank=cp_rank,
        cp_size=cp_size,
        partitions=partitions,
        partitions_perm_idxs=partitions_perm_idxs,
        partitions_unperm_idxs=partitions_unperm_idxs,
    )

    dist_attn_config = DistAttnConfig(dispatch_config=DispatchConfig(chunk_size=chunk_size, alg=SequentialDispatchAlg()))
    magi_key = init_dist_attn_runtime_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=mask_type_list,
        total_seqlen_q=total_padded,
        total_seqlen_k=total_padded,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=0,
        chunk_size=chunk_size,
        cp_group=cp_group,
        cp_mesh=None,
        dist_attn_config=dist_attn_config,
    )
    if magi_key not in dist_attn_runtime_dict_mgr:
        dist_attn_runtime_dict_mgr[magi_key] = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=mask_type_list,
            total_seqlen_q=total_padded,
            total_seqlen_k=total_padded,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            chunk_size=chunk_size,
            cp_group=cp_group,
            cp_mesh=None,
            dist_attn_config=dist_attn_config,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            ref_dispatch_meta_q=dispatch_meta_q,
            ref_dispatch_meta_k=dispatch_meta_k,
        )
    return magi_key


# --------------------------------------------------------------------- #
# Main entrypoint.
# --------------------------------------------------------------------- #


def magi_merged_gptmodel_forward(
    model: GPTModel | Float16Module | DDP,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_processor: Callable | None = None,
    logits_processor_args: dict | None = None,
    *,
    routed_experts: torch.Tensor | None = None,
    merge_info: list[PrefixMergeInfo] | None = None,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Magi forward driven by K pre-built :class:`PrefixMergeInfo` (one per trajectory).

    ``input_ids`` shape is ``(K, S)`` where K is the number of merged
    trajectories in the microbatch and S is ``model.seq_length``. Each
    row's first ``merge_info[i].total_padded`` positions hold trajectory
    i's DFS-pre-order packed layout. The K rows are concatenated
    internally into one packed sequence with combined q/k ranges (one
    ``calc_attn`` call), then per-row outputs are scattered back to
    ``(K, S, *F)`` so the loss layer matches the flat path's interface.

    When ``routed_experts`` is provided and the model has
    ``moe_enable_routing_replay=True``, the routing tensor is dispatched
    through the SAME ``magi_key`` as the input — required for correct
    per-CP-rank position alignment under prefix-tree merging.
    """
    install_magi_attention_patch()
    if merge_info is None:
        paths = extract_paths_from_batch(input_ids, attention_mask)
        tp = mpu.get_tensor_model_parallel_world_size()
        cp = mpu.get_context_parallel_world_size()
        align = max(tp, 1) * max(cp, 1)
        merge_info = [_build_single_path_merge_info(path_len=len(p), align=align) for p in paths]
    return _magi_gptmodel_forward_from_merge_info(
        model=model,
        merge_info=merge_info,
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_processor=logits_processor,
        logits_processor_args=logits_processor_args,
        routed_experts=routed_experts,
        loss_mask=loss_mask,
    )


def magi_flat_gptmodel_forward(
    model: GPTModel | Float16Module | DDP,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_processor: Callable | None = None,
    logits_processor_args: dict | None = None,
    *,
    routed_experts: torch.Tensor | None = None,
    merge_info: list[PrefixMergeInfo] | None = None,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Magi forward with a FLAT trie (per-sample causal, no merging).

    Drop-in replacement for ``gptmodel_forward`` that goes through the
    same ``calc_attn`` kernel as :func:`magi_merged_gptmodel_forward` by building
    K single-path :class:`PrefixMergeInfo` (one per row), so the flex
    ranges reduce to per-sample causal attention — equivalent to the TE
    FA3 THD path.

    Verified bit-exact against :func:`gptmodel_forward` (see
    ``tests/mcore/test_magi_attention.py``), so it acts as a
    "canonical Magi" reference for isolating the pure prefix-merging
    delta in :func:`magi_merged_gptmodel_forward`.

    R3: when ``routed_experts`` is provided and the model has
    ``moe_enable_routing_replay=True``, routing is dispatched through
    the same ``magi_key`` as the input (single-path trie still requires
    the magi-aware CP split).
    """
    assert merge_info is None, "magi_flat_gptmodel_forward builds its own per-row merge_info"
    install_magi_attention_patch()
    paths = extract_paths_from_batch(input_ids, attention_mask)
    tp = mpu.get_tensor_model_parallel_world_size()
    cp = mpu.get_context_parallel_world_size()
    align = max(tp, 1) * max(cp, 1)
    merge_info = [_build_single_path_merge_info(path_len=len(p), align=align) for p in paths]
    return _magi_gptmodel_forward_from_merge_info(
        model=model,
        merge_info=merge_info,
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_processor=logits_processor,
        logits_processor_args=logits_processor_args,
        routed_experts=routed_experts,
        loss_mask=loss_mask,
    )


def _pack_rows_by_merge_info(t: torch.Tensor, merge_info: list[PrefixMergeInfo]) -> torch.Tensor:
    """Slice each row of ``t`` to its trajectory's ``total_padded`` and concat along seq dim.

    ``t`` shape ``(K, S, *F)`` — data-loader's row-stacked layout where each
    row was right-padded to a common width S. Returns ``(sum_total_padded, *F)``,
    matching the kernel's packed-sequence input layout.
    """
    slices = [t[i, : merge_info[i].total_padded].contiguous() for i in range(len(merge_info))]
    return torch.cat(slices, dim=0).contiguous()


def _build_single_path_merge_info(path_len: int, align: int) -> PrefixMergeInfo:
    """Build a 1-path :class:`PrefixMergeInfo` for the flat (non-merged) case.

    Each row in the flat path is one independent trajectory: no prefix
    sharing, just a single causal segment over ``path_len`` tokens. This
    helper aligns ``path_len`` to ``lcm(align, 128)`` (matching
    :func:`pack_tree_aligned_as_list`'s convention) so the merged forward
    sees a kernel-friendly packed length.
    """
    total_align = math.lcm(align, 128)
    total_padded = path_len + (-path_len) % total_align
    nodes = [PrefixTreeNode(start=0, end=total_padded, parent=-1)]
    return build_prefix_merge_info(
        nodes=nodes,
        path_to_leaf=[(0, 0)],
        total_padded=total_padded,
        turn_sample_lens=[path_len],
        real_total=path_len,
    )


def _run_magi_packed_forward(  # noqa: PLR0915
    *,
    model: GPTModel | Float16Module | DDP,
    nodes: list[PrefixTreeNode],
    q_list: list[tuple[int, int]],
    k_list: list[tuple[int, int]],
    mask_list: list[int],
    total_padded: int,
    max_path_len: int,
    packed_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    sequence_boundaries: list[int] | None,
    routed_experts: torch.Tensor | None,
    routing_merge_info: list[PrefixMergeInfo] | None,
    loss_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, Any]:
    """Run the Magi-driven model forward on a pre-built packed trie.

    Sets up ``magi_attn_flex_key`` from per-node ``(q, k, mask)`` ranges,
    dispatches the packed sequence and per-token RoPE positions, runs
    ``model()`` inside a :class:`MagiForwardContext`, and returns
    ``(output_orig, magi_key)``. Callers handle output unpacking — this
    is the only place where the from-merge-info and from-tree paths
    diverge.
    """
    from magi_attention.api import dispatch, magi_attn_flex_key
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType as MagiAttnMaskType
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.utils import divide

    device = input_ids.device

    q_ranges = AttnRanges.from_ranges([list(r) for r in q_list])
    k_ranges = AttnRanges.from_ranges([list(r) for r in k_list])
    mask_type_list = [MagiAttnMaskType.FULL if int(x) == 0 else MagiAttnMaskType.CAUSAL for x in mask_list]

    cfg = unwrap_model(model).config
    tp = mpu.get_tensor_model_parallel_world_size()
    # Mirror Megatron's per-TP-partition head counting so the shapes we pass
    # to Magi match the tensors the kernel actually receives. See
    # `Attention.__init__` (commit-pinned):
    # https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/transformer/attention.py#L284-L305
    num_query_groups = cfg.num_query_groups if cfg.num_query_groups is not None else cfg.num_attention_heads
    num_heads_q = divide(cfg.num_attention_heads, tp)
    num_heads_kv = 1 if num_query_groups < tp else divide(num_query_groups, tp)
    head_dim = cfg.kv_channels if cfg.kv_channels is not None else (cfg.hidden_size // cfg.num_attention_heads)

    cp_group_obj = mpu.get_context_parallel_group()
    assert cp_group_obj is not None, "Magi requires an initialized context-parallel group"
    cp_group = cast("ProcessGroup", cp_group_obj)
    cfg = unwrap_model(model).config
    use_megatron_cp_partition = getattr(cfg, "experimental_attention_variant", None) == "gated_delta_net"
    if use_megatron_cp_partition:
        magi_key = _build_megatron_cp_magi_key(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_type_list=mask_type_list,
            total_padded=total_padded,
            sequence_boundaries=sequence_boundaries,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            cp_group=cp_group,
        )
    else:
        magi_key = magi_attn_flex_key(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=mask_type_list,
            total_seqlen_q=total_padded,
            total_seqlen_k=total_padded,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            pad_size=0,
            cp_group_or_mesh=cp_group,  # pyright: ignore[reportArgumentType]
        )

    global_rel_positions = compute_tree_rel_positions(nodes, total_padded, device)

    # Later PP stages ignore `input_ids` for the main decoder, but Qwen3.6 MTP
    # still uses them in postprocess. Keep token/position metadata in the same
    # Magi CP-local packed layout as the pipeline hidden states.
    local_input_ids = dispatch(packed_tensor, magi_key).unsqueeze(0).contiguous()
    local_rel_positions = dispatch(global_rel_positions, magi_key).to(torch.int64)
    model_position_ids = local_rel_positions.unsqueeze(0).contiguous()

    total_local = int(local_rel_positions.numel())
    # GDN consumes PackedSeqParams after Megatron CP/SP sharding. Preserve the
    # packed row boundaries so GDN recurrence resets between independent samples;
    # a single [0, total] boundary would let state flow across concatenated rows.
    cu_values = [0, total_local]
    if use_megatron_cp_partition:
        boundary_scale = 1
        if getattr(cfg, "sequence_parallel", False) and not getattr(cfg, "scatter_embedding_sequence_parallel", True):
            boundary_scale = mpu.get_tensor_model_parallel_world_size()
        if sequence_boundaries is None:
            cu_values = [0, total_padded * boundary_scale]
        else:
            cu_values = [int(boundary * boundary_scale) for boundary in sequence_boundaries]
    cu_seqlens = torch.tensor(cu_values, dtype=torch.int32, device=device)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_path_len,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_kv=max_path_len,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
    )

    # When R3 is active, dispatch routing through the same magi_key as tokens
    # and RoPE positions. For GDN, that key uses Megatron's packed-CP
    # first-plus-mirrored-tail partition, so replayed experts stay aligned with
    # the GDN hidden rows before the TP sequence-parallel scatter.
    replay_routers: list[Any] | None = None
    if routed_experts is not None and getattr(cfg, "moe_enable_routing_replay", False):
        from axrl.utils.megatron.router_replay import RouterReplayAction, prepare_magi_router_replay_tensors

        assert routing_merge_info is not None
        unwrapped = unwrap_model(model)
        replay_tensors, replay_routers = prepare_magi_router_replay_tensors(
            routed_experts=routed_experts,
            merge_info=routing_merge_info,
            magi_key=magi_key,
            tf_config=cfg,
            device=device,
            vp_rank=unwrapped.vp_stage,
            loss_mask=loss_mask,
        )
        for router, replay_tensor in zip(replay_routers, replay_tensors, strict=True):
            router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            router.set_target_indices(replay_tensor)

    ctx = MagiForwardContext(magi_key=magi_key, position_ids=local_rel_positions)
    token = _current.set(ctx)
    try:
        output_orig = model(
            input_ids=local_input_ids,
            attention_mask=None,
            position_ids=model_position_ids,
            packed_seq_params=packed_seq_params,
        )
    finally:
        _current.reset(token)
        from axrl.utils.megatron.router_replay import finish_forward_router_replay

        finish_forward_router_replay(replay_routers=replay_routers)
    return output_orig, magi_key


def _magi_gptmodel_forward_from_merge_info(
    model: GPTModel | Float16Module | DDP,
    merge_info: list[PrefixMergeInfo],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_processor: Callable | None,
    logits_processor_args: dict | None,
    routed_experts: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Magi forward from K pre-built :class:`PrefixMergeInfo` (one per trajectory).

    ``input_ids`` shape ``(K, S)``. The K rows are concatenated row-by-row up to
    each row's ``total_padded`` and combined via
    :func:`merge_prefix_merge_infos`, so the kernel sees one packed sequence
    with offset-shifted q/k ranges. Outputs are scattered back to ``(K, S, *F)``
    so the loss layer matches the flat path's interface.

    Worked example — K=2 trajectories with S=8, T0=5, T1=3 (toy sizes; real
    workloads have S in the thousands and Ti in the hundreds)::

        input_ids =
          row 0: [a0 a1 a2 a3 a4 .  .  . ]   T0=5, then 3 right-pad cells
          row 1: [b0 b1 b2 .  .  .  .  . ]   T1=3, then 5 right-pad cells

        # 1. Slice each row to its own Ti and concat:
        packed_tensor = [a0 a1 a2 a3 a4 b0 b1 b2]   length 8 = sum(Ti)

        # 2. ``merge_prefix_merge_infos`` shifts trie 1's q/k ranges by T0=5:
        merge_info[0].q_ranges (within trie 0): [(0, 5)]
        merge_info[1].q_ranges (within trie 1): [(0, 3)]
        fused_info.q_ranges = [(0, 5), (5, 8)]   # no cross-tree attention
        fused_info.k_ranges = [(0, 5), (5, 8)]

        # 3. Single ``calc_attn`` call on packed_tensor with the fused ranges.
        # output_orig (CP=1 view) = [o0 o1 o2 o3 o4 o5 o6 o7]   length 8

        # 4. ``_unpack_packed_to_batch``: split per row, right-pad each to S=8,
        #    stack along new dim 0:
        result =
          row 0: [o0 o1 o2 o3 o4  0  0  0]
          row 1: [o5 o6 o7  0  0  0  0  0]
        result.shape = (2, 8) = (K, S)

    Right-pad zeros never affect the loss because ``loss_mask`` is False on
    those positions.

    CP dispatch — what ``dispatch``/``undispatch`` do
    ------------------------------------------------

    Magi's ``calc_attn`` runs distributed across the context-parallel (CP)
    group: each CP rank holds one shard of the packed sequence. Two
    primitives from ``magi_attention.api`` move tensors between the global
    packed view (``(sum_total_padded, *F)``) and the per-CP-rank shard view:

    - ``dispatch(global_tensor, magi_key)`` — splits a global packed
      tensor along the seq axis and returns the slice this CP rank owns.
      The split is the same one the kernel uses internally, so the
      returned shard is positionally aligned with ``output_orig``. Used
      here to scatter ``logits_processor_args`` (e.g. ``labels``) into
      the same per-rank view as the kernel output, so that
      ``logits_processor`` does element-wise work on aligned positions
      without any cross-rank coordination.

    - ``undispatch(local_shard, magi_key)`` — the inverse: gathers each CP
      rank's shard via NCCL all-gather and reassembles the global packed
      tensor. Used inside ``_unpack_packed_to_batch`` so we can split the
      output by trajectory.

    ``magi_key`` carries the dispatch metadata (which token went to which
    rank, padding offsets, etc.). Every tensor that participates in this
    forward — ``packed_tensor``, ``logits_processor_args`` values,
    ``output_orig`` — uses the same ``magi_key`` so positions stay
    in lock-step. With CP=1, ``dispatch``/``undispatch`` are essentially
    identity operations (only one rank, one shard).
    """
    from magi_attention.api import dispatch, undispatch

    from axrl.utils.megatron.prefix_tree import merge_prefix_merge_infos

    assert merge_info, "merge_info list must be non-empty"
    K = len(merge_info)  # one PrefixMergeInfo per trajectory in the microbatch
    assert input_ids.dim() >= 2 and input_ids.shape[0] == K, (
        f"input_ids must be ``(K, S)`` shaped to match merge_info length K={K}, got {tuple(input_ids.shape)}"
    )

    post_process = unwrap_model(model).post_process  # last PP stage only
    # S = data-loader's batch width — also the shape the trainer's ``loss_mask`` /
    # ``labels`` use, so the forward output must match it. Typically S equals
    # ``model.seq_length``, but the data loader may pad rows wider when
    # ``max(total_padded_i)`` exceeds it. We use that width directly here so
    # the function stays agnostic to the loader's choice.
    output_seq_len = attention_mask.shape[1]
    assert all(mi.total_padded <= output_seq_len for mi in merge_info), (
        f"merge_info has total_padded > batch width {output_seq_len}: "
        f"{[mi.total_padded for mi in merge_info]} — the data loader must pad rows "
        f"to >= max(total_padded)."
    )

    # Per-row slice up to total_padded_i (drops the data-loader's right pad), then concat.
    # slices[i]:    (total_padded_i, *F)
    # packed_tensor: (sum_total_padded, *F)  — the kernel's input layout.
    packed_tensor = _pack_rows_by_merge_info(input_ids, merge_info)
    # fused_info.{q_ranges,k_ranges,attn_type_map} use offsets [0, T0, T0+T1, ...]
    # so the kernel sees K independent trees in one packed sequence.
    fused_info = merge_prefix_merge_infos(merge_info)
    total_padded_values = [mi.total_padded for mi in merge_info]
    sequence_boundaries = [0]
    for total_padded_i in total_padded_values:
        sequence_boundaries.append(sequence_boundaries[-1] + total_padded_i)

    # output_orig: per-CP-rank shard; magi_key carries the dispatch metadata used below.
    output_orig, magi_key = _run_magi_packed_forward(
        model=model,
        nodes=fused_info.nodes,
        q_list=fused_info.q_ranges,
        k_list=fused_info.k_ranges,
        mask_list=fused_info.attn_type_map,
        total_padded=fused_info.total_padded,
        max_path_len=fused_info.max_path_len,
        packed_tensor=packed_tensor,
        input_ids=input_ids,
        sequence_boundaries=sequence_boundaries,
        routed_experts=routed_experts,
        routing_merge_info=merge_info if routed_experts is not None else None,
        loss_mask=loss_mask,
    )

    def _unpack_packed_to_batch(local_output: torch.Tensor) -> torch.Tensor:
        """Undispatch + scatter packed (sum_total_padded,) → batched (K, S, *F)."""
        if not post_process:
            return local_output  # non-last PP stage: pass activations through unchanged
        # Drop a leading batch dim of 1 if present, then gather across CP ranks.
        x = local_output.squeeze(0) if local_output.dim() >= 2 and local_output.size(0) == 1 else local_output
        global_out = undispatch(x, magi_key)  # (sum_total_padded, *F)
        per_row: list[torch.Tensor] = []
        offset = 0
        for mi in merge_info:
            n = mi.total_padded
            row = global_out[offset : offset + n]  # (n, *F)
            pad_size = output_seq_len - n
            if pad_size > 0:
                # Right-pad with zeros to S; loss_mask is False on the tail so it never contributes.
                tail_shape = (pad_size, *row.shape[1:])
                tail = torch.zeros(tail_shape, dtype=row.dtype, device=row.device)
                row = torch.cat([row, tail], dim=0)  # (S, *F)
            per_row.append(row)
            offset += n
        return torch.stack(per_row, dim=0)  # (K, S, *F)

    if post_process and logits_processor is not None:
        # `logits_processor` (e.g. SFT/GRPO) consumes logits + per-token side inputs.
        # Side inputs arrive in data-loader layout (K, S, *F); we must repack them to
        # the same (sum_total_padded, *F) layout the kernel produced so positions align.
        logits_processor_args = logits_processor_args or {}
        args: dict[str, torch.Tensor] = {}
        for k, v in logits_processor_args.items():
            # v: (K, S, *F)  →  packed_v: (sum_total_padded, *F)  →  dispatched: per-CP shard.
            assert v.dim() >= 2, f"logits_processor arg {k!r} must be ``(K, S, *F)`` shaped, got {tuple(v.shape)}"
            args[k] = dispatch(_pack_rows_by_merge_info(v, merge_info), magi_key)  # match output_orig's CP-rank shard layout
        output_dict = logits_processor(output_orig, **args)  # e.g. {"log_prob", "entropy"}
        # Each tensor in output_dict is per-CP shard; unpack back to (K, S, *F) for the loss layer.
        return {k: _unpack_packed_to_batch(v) for k, v in output_dict.items()}

    return _unpack_packed_to_batch(output_orig)
