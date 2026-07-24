"""CPU tests: SampleTensorDict roundtrip + balanced microbatch split on compacted trajectories.

Targets potential misalignment between ``merge_info`` /
``routing_handles_per_path`` (non-tensor fields) and the tensor fields
after serialization, balanced split, or simple split.
"""

from __future__ import annotations

import numpy as np

from axrl.data.sample import Sample, SampleTensorDict, samples_from_tensor_dict
from axrl.utils.megatron.prefix_tree import merge_trajectory_samples
from axrl.utils.megatron.seqlen_balancing import (
    realign_non_tensor_keys_after_split,
    split_into_balanced_microbatches,
)
from tests.routing_replay._compacted_fixture import build_compacted_fixture


def _build_merged_samples(n: int) -> list[Sample]:
    samples = []
    for i in range(n):
        f = build_compacted_fixture()
        merged = merge_trajectory_samples(f.turn_samples)
        assert merged.merge_info is not None
        merged.reward = float(i)  # sentinel for alignment tracking
        merged.merge_info.turn_sample_lens = list(merged.merge_info.turn_sample_lens)
        samples.append(merged)
    max_len = max(len(s.input_ids) for s in samples)
    from axrl.data.sample import _pad_sample_to  # pyright: ignore[reportPrivateUsage]

    return [_pad_sample_to(s, max_len) for s in samples]


def test_sampletensordict_roundtrip_preserves_compacted_merge_info_and_handles() -> None:
    """from_samples → samples_from_tensor_dict preserves merge_info fields + routing handles."""
    originals = _build_merged_samples(3)
    td = SampleTensorDict.from_samples(originals)
    roundtripped = samples_from_tensor_dict(td)
    assert len(roundtripped) == len(originals)
    for orig, rt in zip(originals, roundtripped, strict=True):
        assert rt.merge_info is not None
        assert orig.merge_info is not None
        assert rt.merge_info.total_padded == orig.merge_info.total_padded
        assert rt.merge_info.real_total == orig.merge_info.real_total
        assert rt.merge_info.routing_source_path is not None
        assert orig.merge_info.routing_source_path is not None
        assert np.array_equal(rt.merge_info.routing_source_path, orig.merge_info.routing_source_path)
        assert rt.routing_handles_per_path == orig.routing_handles_per_path


def test_balanced_split_non_tensor_fields_align_with_tensor_permutation() -> None:
    """After balanced split, each row's non-tensor fields align with its tensor row.

    Verified via per-sample ``reward`` sentinels: ``merge_info[row]`` and
    ``routing_handles_per_path[row]`` must match the original sample whose
    reward equals ``reward[row]``.
    """
    originals = _build_merged_samples(4)
    td = SampleTensorDict.from_samples(originals)
    micro_batches, index_map = split_into_balanced_microbatches(
        batch=td,
        max_token_len=100,
        same_micro_num_in_dp=False,
    )
    for mb_idx, mb in enumerate(micro_batches):
        mb_merge_infos = mb.get_non_tensor("merge_info")
        mb_handles = mb.get_non_tensor("routing_handles_per_path") if "routing_handles_per_path" in mb.keys() else None  # noqa: SIM118
        mb_rewards = mb["reward"].tolist()
        for row, (mi, rew) in enumerate(zip(mb_merge_infos, mb_rewards, strict=True)):
            orig = originals[index_map[mb_idx][row]]
            assert orig.merge_info is not None
            assert mi.real_total == orig.merge_info.real_total
            assert rew == orig.reward
            if mb_handles is not None:
                assert mb_handles[row] == orig.routing_handles_per_path


def test_simple_split_with_realign_preserves_alignment() -> None:
    """Raw ``TensorDict.split`` broadcasts non-tensor fields; ``realign_non_tensor_keys_after_split`` fixes it."""
    originals = _build_merged_samples(4)
    td = SampleTensorDict.from_samples(originals)
    micro_batches = list(td.split(2))
    realign_non_tensor_keys_after_split(td, micro_batches)
    offset = 0
    for mb in micro_batches:
        mb_merge_infos = mb.get_non_tensor("merge_info")
        mb_rewards = mb["reward"].tolist()
        for row, (mi, rew) in enumerate(zip(mb_merge_infos, mb_rewards, strict=True)):
            orig = originals[offset + row]
            assert orig.merge_info is not None
            assert mi.total_padded == orig.merge_info.total_padded
            assert rew == orig.reward
        offset += int(mb.batch_size[0])
