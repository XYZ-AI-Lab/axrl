"""Tests for sequence-length balancing with non-tensor routing payloads."""

from axrl.data import array_utils
from axrl.data.generation import TensorHandle
from axrl.data.sample import Sample, SampleTensorDict
from axrl.utils.megatron.seqlen_balancing import split_into_balanced_microbatches


def _make_sample(
    seq_len: int,
    effective_seq_len: int,
    *,
    routing_handles_per_path: list[list[TensorHandle]] | None = None,
) -> Sample:
    padding = seq_len - effective_seq_len
    return Sample(
        input_ids=array_utils.as_i32(list(range(seq_len))),
        labels=array_utils.as_i32(list(range(seq_len))),
        loss_mask=array_utils.as_bool(([True] * effective_seq_len) + ([False] * padding)),
        attention_mask=array_utils.as_bool(([True] * effective_seq_len) + ([False] * padding)),
        position_ids=array_utils.as_i32(list(range(seq_len))),
        reward=1.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([1.0] * seq_len),
        routing_handles_per_path=routing_handles_per_path,
    )


def test_split_into_balanced_microbatches_slices_routing_handles_per_row() -> None:
    handles_per_row = [
        [[TensorHandle(ref="nodeA:t0-0")]],
        [[TensorHandle(ref="nodeA:t1-0")], [TensorHandle(ref="nodeA:t1-0"), TensorHandle(ref="nodeA:t1-1")]],
        [[TensorHandle(ref="nodeA:t2-0")]],
    ]
    batch = SampleTensorDict.from_samples(
        [
            _make_sample(seq_len=6, effective_seq_len=6, routing_handles_per_path=handles_per_row[0]),
            _make_sample(seq_len=6, effective_seq_len=5, routing_handles_per_path=handles_per_row[1]),
            _make_sample(seq_len=6, effective_seq_len=4, routing_handles_per_path=handles_per_row[2]),
        ]
    )

    micro_batches, _ = split_into_balanced_microbatches(batch, max_token_len=9)
    seen: list[list[list[TensorHandle]]] = []
    for mb in micro_batches:
        per_row = mb.get_non_tensor("routing_handles_per_path")
        assert len(per_row) == int(mb.batch_size[0])
        for entry in per_row:
            row_per_path = list(entry.tolist()) if hasattr(entry, "tolist") else list(entry)
            seen.append([list(p) for p in row_per_path])
    # The split reorders rows; compare as a dict keyed by the trailing path's handle.
    by_handle = {row[-1][0]: row for row in seen}
    expected_by_handle = {row[-1][0]: row for row in handles_per_row}
    assert by_handle == expected_by_handle
