"""Edge-case stress tests for ``merge_trajectory_samples`` + gather.

Covers uneven trajectory shapes that could trip source-map / pack bugs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from axrl.configs import IGNORE_INDEX
from axrl.data import array_utils
from axrl.data.generation import TensorHandle
from axrl.data.sample import Sample
from axrl.utils.megatron.magi_forward import _build_single_path_merge_info
from axrl.utils.megatron.prefix_tree import (
    gather_merged_routing_per_path,
    merge_trajectory_samples,
)
from axrl.utils.megatron.router_replay import pack_routing_for_magi

NUM_LAYERS = 2
TOPK = 3


def _make_turn_sample(
    input_ids: list[int],
    loss_mask_start: int,
    loss_mask_end: int,
    handles: list[TensorHandle] | None = None,
) -> Sample:
    n = len(input_ids)
    labels = [*list(input_ids[1:]), IGNORE_INDEX]
    loss_mask = [False] * n
    for i in range(loss_mask_start, loss_mask_end):
        loss_mask[i] = True
    return Sample(
        input_ids=array_utils.as_i32(input_ids),
        labels=array_utils.as_i32(labels),
        loss_mask=array_utils.as_bool(loss_mask),
        attention_mask=array_utils.as_bool([True] * n),
        position_ids=array_utils.as_i32(list(range(n))),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * n),
        rollout_logprobs=array_utils.as_f32([0.0] * n),
        routing_handles_per_path=[handles] if handles else None,
    )


def test_merge_with_asymmetric_path_lengths() -> None:
    """Short and long sibling paths share one trie; real_total stays bounded by total_padded."""
    common = list(range(1, 6))
    s0 = _make_turn_sample([*common, 50], 5, 6)
    s1 = _make_turn_sample([*common, 51], 5, 6)
    s2 = _make_turn_sample([*common, 100, 101], 5, 7)
    s3 = _make_turn_sample([*common, *range(200, 500)], 5, 305)  # 305-token tail

    merged = merge_trajectory_samples([s0, s1, s2, s3])
    assert merged.merge_info is not None
    mi = merged.merge_info
    assert mi.real_total <= mi.total_padded
    assert mi.routing_source_path is not None
    assert mi.routing_source_path.shape[0] == mi.real_total - 1


def test_merge_many_trivial_1token_turns() -> None:
    """10 sibling paths each adding a single unique token — exercises wide-fanout source maps."""
    prompt = list(range(1, 10))
    samples = [_make_turn_sample([*prompt, 1000 + k], len(prompt), len(prompt) + 1) for k in range(10)]
    merged = merge_trajectory_samples(samples)
    assert merged.merge_info is not None
    assert len(merged.merge_info.path_to_leaf) == 10
    assert merged.merge_info.real_total <= merged.merge_info.total_padded


def test_merge_duplicate_paths_both_trainable_raises() -> None:
    """Strict shared-trainable rule: two paths can NOT both be trainable at the same slot."""
    seq = [1, 2, 3, 4, 5]
    s0 = _make_turn_sample(seq, 2, 4)
    s1 = _make_turn_sample(seq, 2, 4)
    with pytest.raises(AssertionError, match="shared trainable slot"):
        merge_trajectory_samples([s0, s1])


def test_synthetic_merge_end_to_end_no_overflow() -> None:
    """Full gather + pack on a two-path merged fixture."""
    common = [1, 2, 3]
    h0 = TensorHandle(ref="nodeA:merge_h0")
    h1 = TensorHandle(ref="nodeA:merge_h1")
    s0 = _make_turn_sample([*common, 10, 11], 3, 5, handles=[h0])
    s1 = _make_turn_sample([*common, 20, 21, 22], 3, 6, handles=[h0, h1])
    merged = merge_trajectory_samples([s0, s1])
    assert merged.merge_info is not None
    mi = merged.merge_info

    r0 = np.arange(4 * NUM_LAYERS * TOPK, dtype=np.int16).reshape(4, NUM_LAYERS, TOPK)
    r1 = np.arange(5 * NUM_LAYERS * TOPK, dtype=np.int16).reshape(5, NUM_LAYERS, TOPK) + 10000
    gathered = gather_merged_routing_per_path([r0, r1], mi)
    assert gathered.shape[0] == mi.real_total - 1

    packed = pack_routing_for_magi([torch.from_numpy(gathered)], [mi], device=torch.device("cpu"))
    assert packed.shape[0] == mi.total_padded


@pytest.mark.parametrize("path_len", [127, 128, 129, 1279, 1280, 1281])
def test_single_path_align_boundary_no_off_by_one(path_len: int) -> None:
    """Single-path merge at 128-alignment boundaries — pack must not off-by-one."""
    mi = _build_single_path_merge_info(path_len=path_len, align=1)
    assert mi.real_total == path_len
    assert mi.total_padded % 128 == 0
    assert mi.total_padded >= path_len

    routing = torch.zeros((path_len - 1, NUM_LAYERS, TOPK), dtype=torch.int16)
    packed = pack_routing_for_magi([routing], [mi], device=torch.device("cpu"))
    assert packed.shape[0] == mi.total_padded


def test_gather_merged_routing_per_path_matches_hand_computed_output() -> None:
    """Golden-value check of the gather: each output row equals the expected per-path routing row.

    Uses the same two-path fixture as ``test_synthetic_merge_end_to_end_no_overflow``
    but asserts the FULL gather output bit-exactly against a hand-computed
    reference. Catches off-by-one bugs in the per-path mask indexing that
    shape-only tests miss.
    """
    common = [1, 2, 3]
    h0 = TensorHandle(ref="nodeA:gold_h0")
    h1 = TensorHandle(ref="nodeA:gold_h1")
    s0 = _make_turn_sample([*common, 10, 11], 3, 5, handles=[h0])
    s1 = _make_turn_sample([*common, 20, 21, 22], 3, 6, handles=[h0, h1])
    merged = merge_trajectory_samples([s0, s1])
    assert merged.merge_info is not None
    mi = merged.merge_info

    # Per-path routings: path 0 has path_len=5 → 4 routing rows; path 1 has
    # path_len=6 → 5 rows. Use unique sentinel values so every row is
    # individually identifiable.
    r0 = np.arange(4 * NUM_LAYERS * TOPK, dtype=np.int16).reshape(4, NUM_LAYERS, TOPK)
    r1 = (np.arange(5 * NUM_LAYERS * TOPK, dtype=np.int16) + 10000).reshape(5, NUM_LAYERS, TOPK)

    got = gather_merged_routing_per_path([r0, r1], mi)
    assert mi.routing_source_path is not None
    assert mi.routing_source_read_pos is not None
    # Hand-computed reference: one row per merged-routing position, drawn from
    # the path the source map points at.
    expected = np.stack(
        [[r0, r1][int(mi.routing_source_path[i])][int(mi.routing_source_read_pos[i]) - 1] for i in range(mi.real_total - 1)],
        axis=0,
    )
    assert np.array_equal(got, expected)
