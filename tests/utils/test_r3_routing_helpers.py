"""Tests for the per-path routing source-map and gather in ``axrl/utils/megatron/prefix_tree.py``.

``merge_trajectory_samples`` builds the source map (which path's routing
to use at each merged trie position) under the trainable-wins /
lowest-path-idx-wins rule. ``gather_merged_routing_per_path`` applies it
given each path's cumulative routing array.
"""

import numpy as np
import pytest

from axrl.data import array_utils
from axrl.data.sample import Sample
from axrl.utils.megatron.prefix_tree import (
    compute_tree_rel_positions_as_list,
    gather_merged_routing_per_path,
    merge_trajectory_samples,
)

L = 4
K = 2


def _routing_for_prefix(seq_len: int) -> np.ndarray:
    """Synthetic routing: row i carries int16(pos*1000 + linear_index) at position pos = i + 1."""
    rows = max(seq_len - 1, 0)
    out = np.zeros((rows, L, K), dtype=np.int16)
    for i in range(rows):
        pos = i + 1
        out[i] = (pos * 1000 + np.arange(L * K, dtype=np.int64)).reshape(L, K).astype(np.int16)
    return out


def _make_per_turn_sample(input_ids: list[int], loss_mask_bool: list[bool]) -> Sample:
    n = len(input_ids)
    return Sample(
        input_ids=array_utils.as_i32(input_ids),
        labels=array_utils.as_i32([-100] * n),
        loss_mask=array_utils.as_bool(loss_mask_bool),
        attention_mask=array_utils.as_bool([True] * n),
        position_ids=array_utils.as_i32(list(range(n))),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * n),
    )


@pytest.mark.parametrize(
    ("turn_lens", "prompt_len"),
    [
        ((4,), 3),
        ((4, 5), 2),
        ((4, 5, 6, 7), 3),
        ((4,), 0),
    ],
)
def test_source_map_aligns_with_trie_positions(turn_lens: tuple[int, ...], prompt_len: int) -> None:
    """Each merged routing slot reads from the right path's row.

    The chosen path's token at the read position must match the merged
    token's expected global position.
    """
    cumulative_input_ids: list[int] = list(range(prompt_len))
    per_turn_samples: list[Sample] = []
    cum_lens = []
    for t, turn_len in enumerate(turn_lens):
        cumulative_input_ids = [*cumulative_input_ids, *(prompt_len + 1000 * (t + 1) + np.arange(turn_len)).tolist()]
        n = len(cumulative_input_ids)
        cum_lens.append(n)
        loss_mask = [False] * (n - turn_len) + [True] * turn_len
        per_turn_samples.append(_make_per_turn_sample(cumulative_input_ids, loss_mask))

    merged = merge_trajectory_samples(per_turn_samples)
    info = merged.merge_info
    assert info is not None
    assert info.routing_source_path is not None
    assert info.routing_source_read_pos is not None

    # Each path's routing for this synthetic test is a positional encoding,
    # so we can build per-path arrays from cumulative lengths.
    per_path_routings = [_routing_for_prefix(n) for n in cum_lens]
    gathered = gather_merged_routing_per_path(per_path_routings, info)
    assert gathered.shape == (info.real_total - 1, L, K)

    rel_positions = compute_tree_rel_positions_as_list(info.nodes, info.total_padded)
    # Merged slot p stores the routing for predicting the token at trie position p+1.
    # Since per-path routing is positional, the value at gathered[p] must equal
    # full_routing(p+1) = (p+1)*1000 + arange(L*K).
    for p in range(info.real_total - 1):
        global_pos = rel_positions[p + 1]
        np.testing.assert_array_equal(gathered[p], _routing_for_prefix(global_pos + 1)[global_pos - 1])


def test_source_map_size() -> None:
    samples = [_make_per_turn_sample([0, 1, 2, 3], [False, False, False, True])]
    merged = merge_trajectory_samples(samples)
    info = merged.merge_info
    assert info is not None
    assert info.routing_source_path is not None
    assert info.routing_source_path.dtype == np.int64
    assert info.routing_source_path.shape == (info.real_total - 1,)


def test_real_total_matches_attention_mask() -> None:
    samples = [
        _make_per_turn_sample([10, 11, 12, 20], [False, False, False, True]),
        _make_per_turn_sample([10, 11, 12, 30, 31], [False, False, False, True, True]),
    ]
    merged = merge_trajectory_samples(samples)
    assert merged.merge_info is not None
    assert merged.merge_info.real_total == sum(merged.attention_mask)
