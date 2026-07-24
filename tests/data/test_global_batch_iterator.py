"""CPU tests for the trajectory-grouped global-batch iterator.

These tests exercise the data layout invariants the trainer relies on:
- packed samples with the same ``trajectory_id`` always land in the same global batch,
- the number of yielded global batches equals
  ``num_trajectories // global_batch_size`` regardless of how many packed samples
  each trajectory split into,
- each yielded shard is sized so the union across dp ranks reconstructs the
  padded global batch.
"""

from __future__ import annotations

import pytest

from axrl.configs import IGNORE_INDEX
from axrl.data import array_utils
from axrl.data.global_batch_iterator import build_global_batches_for_dp_rank
from axrl.data.sample import Sample, SampleTensorDict


def _sample(trajectory_id: int, *, seq_len: int = 4) -> Sample:
    return Sample(
        input_ids=array_utils.as_i32(list(range(1, seq_len + 1))),
        labels=array_utils.as_i32([IGNORE_INDEX, *range(2, seq_len + 1)]),
        loss_mask=array_utils.as_bool([False, True, True, False][:seq_len]),
        attention_mask=array_utils.as_bool([True] * seq_len),
        position_ids=array_utils.as_i32(list(range(seq_len))),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * seq_len),
        rollout_logprobs=array_utils.as_f32([0.0] * seq_len),
        trajectory_id=trajectory_id,
    )


def _all_dp_batches(
    samples: SampleTensorDict,
    *,
    global_batch_size: int,
    dp_size: int,
    padding_sample_length: int = 2,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> list[list[SampleTensorDict]]:
    """Run the iterator for every dp_rank, returning batches[rank][global_batch_idx]."""
    return [
        build_global_batches_for_dp_rank(
            samples,
            global_batch_size=global_batch_size,
            dp_size=dp_size,
            dp_rank=rank,
            padding_sample_length=padding_sample_length,
            padding_routing_handle=None,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
        )
        for rank in range(dp_size)
    ]


def test_iterator_yields_one_batch_per_original_global_batch_when_no_split() -> None:
    samples = [_sample(traj) for traj in range(4)]
    td = SampleTensorDict.from_samples(samples)

    by_rank = _all_dp_batches(td, global_batch_size=2, dp_size=1)
    assert len(by_rank) == 1
    assert len(by_rank[0]) == 2  # 4 trajectories / global_batch_size=2 ⇒ 2 batches
    assert by_rank[0][0]["trajectory_id"].tolist() == [0, 1]
    assert by_rank[0][1]["trajectory_id"].tolist() == [2, 3]


def test_iterator_keeps_split_trajectory_samples_in_one_batch() -> None:
    # Trajectory 0 splits into 3 packed samples; trajectory 1 stays at 1.
    samples = [_sample(0), _sample(0), _sample(0), _sample(1), _sample(2), _sample(2), _sample(3)]
    td = SampleTensorDict.from_samples(samples)

    by_rank = _all_dp_batches(td, global_batch_size=2, dp_size=1)
    # 4 trajectories / 2 ⇒ 2 global batches; same number of gradient updates as no-split.
    assert len(by_rank[0]) == 2
    batch_0_ids = sorted(by_rank[0][0]["trajectory_id"].tolist())
    batch_1_ids = sorted(by_rank[0][1]["trajectory_id"].tolist())
    assert batch_0_ids == [0, 0, 0, 1]
    assert batch_1_ids == [2, 2, 3]


def test_iterator_round_robins_across_dp_ranks_and_pads_to_dp_multiple() -> None:
    # 3 packed samples in a single global batch with dp_size=2 → 1 padding row.
    samples = [_sample(0), _sample(0), _sample(1)]
    td = SampleTensorDict.from_samples(samples)

    by_rank = _all_dp_batches(td, global_batch_size=2, dp_size=2)
    assert len(by_rank) == 2
    assert all(len(rank_batches) == 1 for rank_batches in by_rank)
    rank0_batch = by_rank[0][0]
    rank1_batch = by_rank[1][0]
    assert len(rank0_batch) == len(rank1_batch) == 2  # (3 reals + 1 padding) / 2

    # Padding rows carry index == -1 and trajectory_id == -1.
    combined_indices = rank0_batch["index"].tolist() + rank1_batch["index"].tolist()
    combined_trajectory_ids = rank0_batch["trajectory_id"].tolist() + rank1_batch["trajectory_id"].tolist()
    real_indices = sorted(i for i in combined_indices if i >= 0)
    assert real_indices == [0, 1, 2]
    assert combined_indices.count(-1) == 1
    assert combined_trajectory_ids.count(-1) == 1


def test_iterator_shuffle_is_deterministic_and_groups_by_batch() -> None:
    samples = [_sample(0), _sample(0), _sample(1), _sample(2), _sample(2), _sample(3)]
    td = SampleTensorDict.from_samples(samples)

    seed_a_run_1 = _all_dp_batches(td, global_batch_size=2, dp_size=2, shuffle=True, shuffle_seed=17)
    seed_a_run_2 = _all_dp_batches(td, global_batch_size=2, dp_size=2, shuffle=True, shuffle_seed=17)
    seed_b = _all_dp_batches(td, global_batch_size=2, dp_size=2, shuffle=True, shuffle_seed=42)

    # Same seed ⇒ exact same per-rank slices.
    for rank in range(2):
        for gb_idx in range(len(seed_a_run_1[rank])):
            assert seed_a_run_1[rank][gb_idx]["trajectory_id"].tolist() == seed_a_run_2[rank][gb_idx]["trajectory_id"].tolist()

    # Different seeds typically reshuffle within a batch; tolerate degenerate equality but
    # require the per-batch trajectory composition to be identical regardless of seed.
    for gb_idx in range(2):
        composition_a = sorted(t for rank_batches in seed_a_run_1 for t in rank_batches[gb_idx]["trajectory_id"].tolist() if t >= 0)
        composition_b = sorted(t for rank_batches in seed_b for t in rank_batches[gb_idx]["trajectory_id"].tolist() if t >= 0)
        assert composition_a == composition_b


def test_iterator_trailing_batch_holds_remainder_trajectories() -> None:
    # 3 trajectories, gbs=2 ⇒ 2 global batches: [0, 1] then [2].
    samples = [_sample(0), _sample(1), _sample(2)]
    td = SampleTensorDict.from_samples(samples)

    by_rank = _all_dp_batches(td, global_batch_size=2, dp_size=1)
    assert len(by_rank) == 1
    assert len(by_rank[0]) == 2
    assert by_rank[0][0]["trajectory_id"].tolist() == [0, 1]
    assert by_rank[0][1]["trajectory_id"].tolist() == [2]


def test_iterator_rejects_missing_trajectory_id_column() -> None:
    samples = [_sample(0), _sample(1)]
    td = SampleTensorDict.from_samples(samples)
    # Drop the trajectory_id column to simulate samples produced outside the new pipeline.
    td = td.exclude("trajectory_id")  # type: ignore[assignment]

    with pytest.raises(AssertionError, match="trajectory_id"):
        build_global_batches_for_dp_rank(
            td,
            global_batch_size=2,
            dp_size=1,
            dp_rank=0,
            padding_sample_length=2,
            padding_routing_handle=None,
            shuffle=False,
            shuffle_seed=0,
        )
