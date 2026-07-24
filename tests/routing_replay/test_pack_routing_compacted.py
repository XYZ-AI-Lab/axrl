"""CPU tests for ``pack_routing_for_magi`` on compacted trajectories.

Targets the off-by-one the R3 benchmark hit
(``ValueError: merged length 1281 > total_padded 1280``) and parity with
the legacy expand+pack path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from axrl.utils.megatron.magi_forward import _build_megatron_cp_partitions
from axrl.utils.megatron.prefix_tree import (
    gather_merged_routing_per_path,
    merge_trajectory_samples,
)
from axrl.utils.megatron.router_replay import (
    expand_routed_experts_to_token_positions,
    pack_routing_for_magi,
)
from tests.routing_replay._compacted_fixture import (
    NUM_LAYERS,
    TOPK,
    build_compacted_fixture,
    make_routing_payloads,
)

if TYPE_CHECKING:
    from axrl.data.generation import TensorHandle
    from axrl.utils.megatron.prefix_tree import PrefixMergeInfo
    from tests.routing_replay._compacted_fixture import CompactedFixture


def _gather_merged_tensor(
    f: CompactedFixture,
    payloads: dict[TensorHandle, np.ndarray],
) -> tuple[torch.Tensor, PrefixMergeInfo]:
    merged = merge_trajectory_samples(f.turn_samples)
    assert merged.merge_info is not None
    per_path: list[np.ndarray] = []
    for sample in f.turn_samples:
        assert sample.routing_handles_per_path is not None
        per_path.append(np.concatenate([payloads[h] for h in sample.routing_handles_per_path[0]], axis=0))
    merged_np = gather_merged_routing_per_path(per_path, merged.merge_info)
    return torch.from_numpy(merged_np), merged.merge_info


def test_pack_single_compacted_trajectory_no_overflow() -> None:
    """Merged tensor + merge_info pack correctly for one compacted trajectory."""
    f = build_compacted_fixture()
    merged_tensor, merge_info = _gather_merged_tensor(f, make_routing_payloads(f))
    packed = pack_routing_for_magi([merged_tensor], [merge_info], device=torch.device("cpu"))
    assert packed.shape[0] == merge_info.total_padded
    assert packed.shape[1:] == (NUM_LAYERS, TOPK)


def test_pack_multi_trajectory_batched_no_overflow() -> None:
    """Two compacted trajectories batched — each packed to its own total_padded."""
    f1, f2 = build_compacted_fixture(), build_compacted_fixture()
    merged1, mi1 = _gather_merged_tensor(f1, make_routing_payloads(f1))
    merged2, mi2 = _gather_merged_tensor(f2, make_routing_payloads(f2))
    packed = pack_routing_for_magi([merged1, merged2], [mi1, mi2], device=torch.device("cpu"))
    assert packed.shape[0] == mi1.total_padded + mi2.total_padded


def test_megatron_cp_partitions_match_packed_gdn_cp_split() -> None:
    """GDN Magi routing uses the same CP row order as ``preprocess_packed_seqs``."""
    lengths = [128, 256]
    cp_size = 2
    chunk_size, partitions = _build_megatron_cp_partitions(sequence_lengths=lengths, cp_size=cp_size)

    actual: list[list[int]] = []
    for partition in partitions:
        rows: list[int] = []
        for chunk in partition:
            rows.extend(range(chunk * chunk_size, (chunk + 1) * chunk_size))
        actual.append(rows)

    expected: list[list[int]] = [[] for _ in range(cp_size)]
    offset = 0
    for length in lengths:
        per_rank = length // cp_size
        half = per_rank // 2
        for rank in range(cp_size):
            expected[rank].extend(range(offset + half * rank, offset + half * (rank + 1)))
            expected[rank].extend(range(offset + length - half * (rank + 1), offset + length - half * rank))
        offset += length

    assert actual == expected


def test_pack_parity_with_legacy_expand_path_on_compacted() -> None:
    """Bit-exact parity between ``pack_routing_for_magi`` and the legacy ``expand + _pack_rows_by_merge_info``.

    Covers the compacted case; ``test_r3_routing_helpers.py::test_pack_routing_for_magi_parity``
    covers the no-compaction case.
    """
    f = build_compacted_fixture()
    payloads = make_routing_payloads(f)
    merged_tensor, merge_info = _gather_merged_tensor(f, payloads)
    merged_sample = merge_trajectory_samples(f.turn_samples)
    n_real = merged_tensor.shape[0]
    assert n_real == merge_info.real_total - 1

    routed_experts = torch.stack([merged_tensor[:n_real]], dim=0)
    expand_mask = torch.tensor([merged_sample.attention_mask], dtype=torch.bool)
    expanded = expand_routed_experts_to_token_positions(
        routed_experts.view(1, n_real, NUM_LAYERS, TOPK),
        expand_mask,
    )
    from axrl.utils.megatron.magi_forward import _pack_rows_by_merge_info

    reference = _pack_rows_by_merge_info(expanded, [merge_info])
    packed = pack_routing_for_magi([merged_tensor], [merge_info], device=torch.device("cpu"))
    assert torch.equal(packed, reference.to(packed.dtype))
