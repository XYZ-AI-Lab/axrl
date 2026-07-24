"""CPU tests exercising ``merge_trajectory_samples`` on a compacted 4-turn trajectory.

Covers the core invariants the R3 benchmark can violate:
- ``real_total <= total_padded`` (benchmark crashed with real_total > total_padded)
- routing source map length equals ``real_total - 1``
- gather output fits into ``total_padded``
- per-path handle concatenation length equals ``path_len - 1``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from axrl.utils.megatron.prefix_tree import (
    gather_merged_routing_per_path,
    merge_trajectory_samples,
)
from tests.routing_replay._compacted_fixture import (
    build_compacted_fixture,
    make_routing_payloads,
)

if TYPE_CHECKING:
    from axrl.data.generation import TensorHandle
    from tests.routing_replay._compacted_fixture import CompactedFixture


def test_merged_real_total_le_total_padded() -> None:
    """The benchmark's failing invariant: real_total must never exceed total_padded."""
    f = build_compacted_fixture()
    merged = merge_trajectory_samples(f.turn_samples)
    mi = merged.merge_info
    assert mi is not None
    assert mi.real_total <= mi.total_padded


def test_merged_routing_source_path_length_equals_real_total_minus_1() -> None:
    """``gather_merged_routing_per_path`` reads exactly ``real_total - 1`` rows from the source map."""
    f = build_compacted_fixture()
    merged = merge_trajectory_samples(f.turn_samples)
    mi = merged.merge_info
    assert mi is not None
    assert mi.routing_source_path is not None
    assert mi.routing_source_read_pos is not None
    assert mi.routing_source_path.shape[0] == mi.real_total - 1
    assert mi.routing_source_read_pos.shape[0] == mi.real_total - 1


def test_merged_source_map_every_position_has_a_source() -> None:
    """No position at write_pos >= 1 should have ``source_path == -1`` (would fail gather)."""
    f = build_compacted_fixture()
    merged = merge_trajectory_samples(f.turn_samples)
    mi = merged.merge_info
    assert mi is not None
    assert mi.routing_source_path is not None
    assert (mi.routing_source_path >= 0).all()


def test_merged_has_4_leaf_paths_with_handle_chains() -> None:
    """After compaction, all 4 leaf paths share prefix h0 and end at h{0,1,2,3}."""
    f = build_compacted_fixture()
    merged = merge_trajectory_samples(f.turn_samples)
    assert merged.routing_handles_per_path is not None
    assert len(merged.routing_handles_per_path) == 4
    h0 = f.all_minted_handles[0]
    for chain in merged.routing_handles_per_path:
        assert chain[0] == h0
    assert [chain[-1] for chain in merged.routing_handles_per_path] == f.all_minted_handles


def _build_per_path_routings(f: CompactedFixture, payloads: dict[TensorHandle, np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for sample in f.turn_samples:
        assert sample.routing_handles_per_path is not None
        out.append(np.concatenate([payloads[h] for h in sample.routing_handles_per_path[0]], axis=0))
    return out


def test_gather_output_fits_into_total_padded() -> None:
    """Regression for benchmark bug: gather output must fit in total_padded."""
    f = build_compacted_fixture()
    merged = merge_trajectory_samples(f.turn_samples)
    assert merged.merge_info is not None
    per_path = _build_per_path_routings(f, make_routing_payloads(f))
    merged_np = gather_merged_routing_per_path(per_path, merged.merge_info)
    assert merged_np.shape[0] == merged.merge_info.real_total - 1
    assert merged_np.shape[0] <= merged.merge_info.total_padded


@pytest.mark.parametrize("max_recent", [0, 1, 2])
def test_per_path_concat_routing_length_invariant_under_compaction_policies(max_recent: int) -> None:
    """For each leaf path, concat of its handles has exactly ``len(input_ids) - 1`` rows.

    Must hold under any ``max_recent_tool_results`` policy — the fixture's
    ``expected_payload_rows`` is computed from the live trace state, so the
    length invariant is the thing under test, not a hand-derived constant.
    """
    f = build_compacted_fixture(max_recent_tool_results=max_recent)
    payloads = make_routing_payloads(f)
    for k, sample in enumerate(f.turn_samples):
        assert sample.routing_handles_per_path is not None
        handles = sample.routing_handles_per_path[0]
        total = sum(payloads[h].shape[0] for h in handles)
        expected = len(sample.input_ids) - 1
        assert total == expected, f"max_recent={max_recent} path {k}: {total} != {expected}"
