import asyncio

import pytest

from axrl.configs import IGNORE_INDEX
from axrl.data import array_utils
from axrl.data.conversation import Conversation, Message
from axrl.data.generation import TensorHandle
from axrl.data.rollout_trace import (
    RolloutTrace,
    pack_rollout_traces_for_train_batches,
)
from axrl.data.rollout_trace_packing import (
    RolloutTracePackingProcessor,
    RolloutTracePackRequest,
    pack_rollout_trace_samples_to_tensor_dict,
)
from axrl.data.sample import (
    Sample,
    SampleTensorDict,
    pad_sample_tensor_dict_to_multiple,
    remove_padding_from_sample_tensor_dict,
    samples_from_tensor_dict,
)
from axrl.processor.processor_pool import ProcessorPool
from axrl.utils.megatron.prefix_tree import MergingTree, add_sample_to_merging_tree, get_packed_len_if_merge


def _sample(input_ids: list[int], *, trainable_positions: set[int] | None = None) -> Sample:
    if trainable_positions is None:
        trainable_positions = set(range(1, len(input_ids) - 1))
    return Sample(
        input_ids=array_utils.as_i32(input_ids),
        labels=array_utils.as_i32([*input_ids[1:], IGNORE_INDEX]),
        loss_mask=array_utils.as_bool([i in trainable_positions for i in range(len(input_ids))]),
        attention_mask=array_utils.as_bool([True] * len(input_ids)),
        position_ids=array_utils.as_i32(list(range(len(input_ids)))),
        reward=1.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([1.0 if i in trainable_positions else 0.0 for i in range(len(input_ids))]),
        rollout_logprobs=array_utils.as_f32([0.0] * len(input_ids)),
        turn_index=array_utils.as_i32([0 if i in trainable_positions else -1 for i in range(len(input_ids))]),
        turn_reward=array_utils.as_f32([0.0] * len(input_ids)),
    )


def _trace_with_turn_samples(samples: list[Sample]) -> RolloutTrace:
    trace = RolloutTrace(
        Conversation(messages=[Message(role="user", content="prompt")]),
        token_in_token_out=False,
    )
    trace.turn_samples = samples
    return trace


def test_to_packed_samples_preserves_loss_tokens_and_respects_max_length() -> None:
    trace = _trace_with_turn_samples(
        [
            _sample([1, 2, 3, 4]),
            _sample([1, 2, 5, 6]),
            _sample([1, 2, 7, 8]),
        ]
    )

    packed = trace.to_packed_samples(max_pack_length=6)

    assert packed
    # ``merge_trajectory_samples`` rounds ``total_padded`` up to ``lcm(align_size, 128) == 128``,
    # so check the real (non-padding) content length stays within ``max_pack_length``.
    assert all(sum(sample.attention_mask) <= 6 for sample in packed)
    assert sum(sum(sample.loss_mask) for sample in packed) == sum(sum(sample.loss_mask) for sample in trace.turn_samples)


def test_to_packed_samples_greedily_merges_until_next_sample_would_exceed_limit() -> None:
    trace = _trace_with_turn_samples(
        [
            _sample([1, 2, 3, 4], trainable_positions={2}),
            _sample([1, 2, 3, 4, 5], trainable_positions={3}),
            _sample([9, 10, 11]),
        ]
    )

    packed = trace.to_packed_samples(max_pack_length=6)

    assert [info.turn_sample_lens for info in (sample.merge_info for sample in packed if sample.merge_info is not None)] == [[4, 5], [3]]


def test_to_packed_samples_raises_when_single_pre_merge_sample_exceeds_limit() -> None:
    trace = _trace_with_turn_samples([_sample([1, 2, 3, 4, 5])])

    with pytest.raises(ValueError, match="pre-merge turn sample length"):
        trace.to_packed_samples(max_pack_length=4)


def test_merging_tree_len_estimate_matches_full_merge() -> None:
    samples = [
        _sample([1, 2, 3, 4], trainable_positions=set()),
        _sample([1, 2, 5, 6], trainable_positions=set()),
        _sample([1, 2, 3, 4, 7], trainable_positions=set()),
        _sample([1, 2], trainable_positions=set()),
        _sample([9, 10, 11], trainable_positions=set()),
    ]
    tree = MergingTree()

    for sample in samples:
        # ``get_packed_len_if_merge`` returns the raw tree-packed length (no alignment),
        # which must match ``tree.packed_len`` after the same insertion.
        estimate = get_packed_len_if_merge(tree, sample)
        add_sample_to_merging_tree(tree, sample)
        assert estimate == tree.packed_len


def test_merging_tree_same_leading_siblings_estimate_matches_insertion() -> None:
    # Regression test for label-aware packed-trie behavior: paths can share
    # their leading token yet still require coexisting as same-leading
    # siblings under the same parent when the label-aware common prefix
    # collapses to zero (because both paths diverge at the next token).
    # The path triple below exercises this: the second and third paths both
    # start with token ``2`` after the shared ``1`` prefix, but they fork at
    # position ``k=1`` so ``_label_aware_common_prefix_len`` returns 0 there.
    # The estimator (`_added_len_if_insert*`) and the inserter (`_insert_path`)
    # must both ``continue`` past the matching child and append a new sibling,
    # otherwise the predicted length diverges from the actual packed length.
    samples = [
        _sample([1, 2, 3, 4], trainable_positions=set()),  # A, X, ...
        _sample([1, 5, 6, 7], trainable_positions=set()),  # A, Y, ...
        _sample([1, 5, 6, 8, 9], trainable_positions=set()),  # A, Y, Y', Z, ...
        _sample([1, 5, 10], trainable_positions=set()),  # extra branch off the Y sibling
    ]
    tree = MergingTree()

    for sample in samples:
        estimate = get_packed_len_if_merge(tree, sample)
        add_sample_to_merging_tree(tree, sample)
        assert estimate == tree.packed_len


def test_to_packed_samples_materializes_only_final_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    from axrl.utils.megatron import prefix_tree

    trace = _trace_with_turn_samples(
        [
            _sample([1, 2, 3, 4], trainable_positions={2}),
            _sample([1, 2, 3, 4, 5], trainable_positions={3}),
            _sample([9, 10, 11]),
        ]
    )
    call_sizes: list[int] = []
    original_merge = prefix_tree.merge_trajectory_samples

    def counting_merge(samples: list[Sample], align_size: int = 128) -> Sample:
        call_sizes.append(len(samples))
        return original_merge(samples, align_size=align_size)

    monkeypatch.setattr(prefix_tree, "merge_trajectory_samples", counting_merge)

    trace.to_packed_samples(max_pack_length=6)

    assert call_sizes == [2, 1]


def test_to_packed_samples_leaves_trajectory_id_unset() -> None:
    trace = _trace_with_turn_samples(
        [
            _sample([1, 2, 3], trainable_positions={1}),
            _sample([9, 10, 11, 12, 13], trainable_positions={1, 2, 3}),
        ]
    )

    packed = trace.to_packed_samples(max_pack_length=5)

    assert len(packed) == 2
    assert all(sample.trajectory_id == -1 for sample in packed)


def test_to_packed_samples_requires_normalized_reward_baseline_within_group() -> None:
    samples = [
        _sample([1, 2, 3], trainable_positions={1}),
        _sample([1, 2, 4], trainable_positions={1}),
    ]
    samples[1].reward_baseline = 1.0
    trace = _trace_with_turn_samples(samples)

    with pytest.raises(AssertionError, match="reward_baseline"):
        trace.to_packed_samples(max_pack_length=8)


def test_pack_rollout_traces_for_train_batches_assigns_trajectory_ids() -> None:
    traces = [
        _trace_with_turn_samples(
            [
                _sample([1, 2, 3, 4, 5], trainable_positions={1, 2, 3}),
                _sample([9, 10, 11, 12], trainable_positions={1, 2}),
            ]
        ),
        _trace_with_turn_samples(
            [
                _sample([21, 22, 23, 24, 25], trainable_positions={1, 2, 3}),
                _sample([29, 30, 31, 32], trainable_positions={1, 2}),
            ]
        ),
    ]

    packed = pack_rollout_traces_for_train_batches(traces, max_pack_length=5, global_batch_size=2)

    assert [sum(sample.loss_mask) for sample in packed] == [3, 2, 3, 2]
    # Two packed samples per trace, ids 0 and 1.
    assert [sample.trajectory_id for sample in packed] == [0, 0, 1, 1]


def test_pack_rollout_trace_samples_to_tensor_dict_assigns_trajectory_id() -> None:
    trace = _trace_with_turn_samples([_sample([1, 2, 3], trainable_positions={1})])

    actual = pack_rollout_trace_samples_to_tensor_dict(
        trajectory_id=12,
        turn_samples=trace.turn_samples,
        max_pack_length=128,
        allow_prefix_sharing=True,
    )

    assert actual["trajectory_id"].tolist() == [12]
    assert actual["index"].tolist() == [0]
    assert actual["input_ids"].shape == (1, 128)
    assert int(actual["loss_mask"].sum().item()) == 1


def test_sample_tensor_dict_preserves_teacher_logprobs() -> None:
    sample = _sample([1, 2, 3], trainable_positions={1})
    sample.teacher_logprobs = array_utils.as_f32([0.0, -0.7, 0.0])

    tensor_dict = SampleTensorDict.from_samples([sample], max_length=5)
    roundtripped = samples_from_tensor_dict(tensor_dict)

    assert "teacher_logprobs" in tensor_dict.keys()  # noqa: SIM118 - tensordict semantics
    assert tensor_dict["teacher_logprobs"].shape == (1, 5)
    assert tensor_dict["teacher_logprobs"][0].tolist() == pytest.approx([0.0, -0.7, 0.0, 0.0, 0.0])
    assert roundtripped[0].teacher_logprobs is not None
    assert roundtripped[0].teacher_logprobs.tolist() == pytest.approx([0.0, -0.7, 0.0, 0.0, 0.0])


def test_padding_helpers_preserve_teacher_logprobs() -> None:
    sample = _sample([1, 2, 3], trainable_positions={1})
    sample.teacher_logprobs = array_utils.as_f32([0.0, -0.5, 0.0])
    tensor_dict = SampleTensorDict.from_samples([sample], max_length=4)

    padded, original_len = pad_sample_tensor_dict_to_multiple(tensor_dict, 2, padding_sample_length=1)
    trimmed = remove_padding_from_sample_tensor_dict(padded, original_len)

    assert "teacher_logprobs" in padded.keys()  # noqa: SIM118 - tensordict semantics
    assert padded["teacher_logprobs"].shape == (2, 4)
    assert padded["teacher_logprobs"][1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert trimmed["teacher_logprobs"].tolist() == tensor_dict["teacher_logprobs"].tolist()


def test_to_packed_samples_preserves_teacher_logprobs() -> None:
    first = _sample([1, 2, 3], trainable_positions={1})
    second = _sample([1, 4, 5], trainable_positions={1})
    first.teacher_logprobs = array_utils.as_f32([0.0, -0.1, 0.0])
    second.teacher_logprobs = array_utils.as_f32([0.0, -0.2, 0.0])
    trace = _trace_with_turn_samples([first, second])

    packed = trace.to_packed_samples(max_pack_length=8, allow_prefix_sharing=True)

    assert len(packed) == 1
    assert packed[0].teacher_logprobs is not None
    actual = packed[0].teacher_logprobs[packed[0].loss_mask].tolist()
    assert actual == pytest.approx([-0.1, -0.2])


def test_pack_rollout_trace_processor_returns_teacher_logprobs() -> None:
    sample = _sample([1, 2, 3], trainable_positions={1})
    sample.teacher_logprobs = array_utils.as_f32([0.0, -0.4, 0.0])

    actual = pack_rollout_trace_samples_to_tensor_dict(
        trajectory_id=3,
        turn_samples=[sample],
        max_pack_length=128,
        allow_prefix_sharing=True,
    )

    assert "teacher_logprobs" in actual.keys()  # noqa: SIM118 - tensordict semantics
    assert actual["teacher_logprobs"][0][actual["loss_mask"][0]].tolist() == pytest.approx([-0.4])


def test_rollout_trace_packing_processor_pool_assigns_trajectory_id() -> None:
    trace = _trace_with_turn_samples([_sample([1, 2, 3], trainable_positions={1})])
    request = RolloutTracePackRequest(
        trajectory_id=7,
        turn_samples=trace.turn_samples,
        max_pack_length=128,
        allow_prefix_sharing=True,
    )

    with ProcessorPool[RolloutTracePackRequest, SampleTensorDict](
        RolloutTracePackingProcessor,
        config=None,
        num_processors=1,
        timeout_seconds=60,
    ) as pool:
        actual = asyncio.run(pool.generate(request))

    assert actual["trajectory_id"].tolist() == [7]
    assert actual["input_ids"].shape == (1, 128)
    assert int(actual["loss_mask"].sum().item()) == 1


def test_pack_rollout_traces_for_train_batches_packs_short_trajectories_into_one_sample() -> None:
    traces = [
        _trace_with_turn_samples([_sample([1, 2, 3, 4, 5], trainable_positions={1, 2, 3})]),
        _trace_with_turn_samples([_sample([11, 12, 13, 14, 15], trainable_positions={1, 2, 3})]),
        _trace_with_turn_samples([_sample([21, 22, 23], trainable_positions={1})]),
        _trace_with_turn_samples([_sample([31, 32, 33], trainable_positions={1})]),
    ]

    packed = pack_rollout_traces_for_train_batches(traces, max_pack_length=5, global_batch_size=2)

    assert [sum(sample.loss_mask) for sample in packed] == [3, 3, 1, 1]
    assert [sample.trajectory_id for sample in packed] == [0, 1, 2, 3]


def test_pack_rollout_traces_for_train_batches_with_variable_split_count_per_trace() -> None:
    traces = [
        _trace_with_turn_samples(
            [
                _sample([1, 2, 3, 4, 5], trainable_positions={1, 2, 3}),
                _sample([9, 10, 11, 12], trainable_positions={1, 2}),
            ]
        ),
        _trace_with_turn_samples(
            [
                _sample([21, 22, 23], trainable_positions={1}),
            ]
        ),
    ]

    packed = pack_rollout_traces_for_train_batches(traces, max_pack_length=5, global_batch_size=2)

    assert len(packed) == 3
    assert [sum(sample.loss_mask) for sample in packed] == [3, 2, 1]
    # Trace 0 split into two packed samples, trace 1 stayed as one.
    assert [sample.trajectory_id for sample in packed] == [0, 0, 1]


def test_to_packed_samples_single_sample_keeps_default_trajectory_id() -> None:
    trace = _trace_with_turn_samples(
        [
            _sample([1, 2, 3], trainable_positions={1}),
            _sample([1, 2, 3, 4], trainable_positions={2}),
        ]
    )

    packed = trace.to_packed_samples(max_pack_length=4)

    assert len(packed) == 1
    assert packed[0].trajectory_id == -1


def test_to_packed_samples_flat_mode_emits_single_flat_sample_for_linear_chain() -> None:
    """allow_prefix_sharing=False on a linear-prefix chain emits one flat sample (merge_info=None)."""
    trace = _trace_with_turn_samples(
        [
            _sample([1, 2, 3], trainable_positions={1, 2}),
            _sample([1, 2, 3, 4, 5], trainable_positions={3, 4}),
            _sample([1, 2, 3, 4, 5, 6, 7], trainable_positions={5, 6}),
        ]
    )

    packed = trace.to_packed_samples(max_pack_length=8, allow_prefix_sharing=False)

    assert len(packed) == 1
    flat = packed[0]
    assert flat.merge_info is None
    # ``merge_trajectory_samples`` aligns ``total_padded`` to 128; the real content lives in the
    # ``attention_mask`` window.
    real_len = sum(flat.attention_mask)
    assert real_len == 7
    assert flat.input_ids[:real_len].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert sum(flat.loss_mask) == sum(sum(s.loss_mask) for s in trace.turn_samples)


def test_to_packed_samples_flat_mode_rejects_branching_trace() -> None:
    """allow_prefix_sharing=False on diverging turn samples raises ValueError."""
    trace = _trace_with_turn_samples(
        [
            _sample([1, 2, 3], trainable_positions={1}),
            _sample([1, 2, 4], trainable_positions={1}),
        ]
    )

    with pytest.raises(ValueError, match="linear prefix chain"):
        trace.to_packed_samples(max_pack_length=16, allow_prefix_sharing=False)


def test_to_packed_samples_flat_mode_rejects_overflow() -> None:
    """allow_prefix_sharing=False does not split: a too-long trace raises rather than emitting multiple samples."""
    trace = _trace_with_turn_samples(
        [
            _sample([1, 2, 3, 4, 5], trainable_positions={1, 2}),
            _sample([1, 2, 3, 4, 5, 6, 7, 8], trainable_positions={6}),
        ]
    )

    with pytest.raises(ValueError, match="exceeds max_pack_length"):
        trace.to_packed_samples(max_pack_length=6, allow_prefix_sharing=False)


def test_to_packed_samples_flat_mode_collapses_routing_handles_to_single_path() -> None:
    h0 = TensorHandle(ref="t0")
    h1 = TensorHandle(ref="t1")
    s0 = _sample([1, 2, 3], trainable_positions={1, 2})
    s0.routing_handles_per_path = [[h0]]
    s1 = _sample([1, 2, 3, 4, 5], trainable_positions={3, 4})
    s1.routing_handles_per_path = [[h0, h1]]
    trace = _trace_with_turn_samples([s0, s1])

    packed = trace.to_packed_samples(max_pack_length=8, allow_prefix_sharing=False)

    assert len(packed) == 1
    assert packed[0].merge_info is None
    assert packed[0].routing_handles_per_path is not None
    assert len(packed[0].routing_handles_per_path) == 1
    assert packed[0].routing_handles_per_path == [[h0, h1]]


def test_pack_rollout_traces_for_train_batches_flat_mode_passes_through_flag() -> None:
    """allow_prefix_sharing=False at batch level → every emitted sample has merge_info=None."""
    traces = [
        _trace_with_turn_samples(
            [
                _sample([1, 2, 3], trainable_positions={1}),
                _sample([1, 2, 3, 4], trainable_positions={2}),
            ]
        ),
        _trace_with_turn_samples(
            [
                _sample([5, 6, 7], trainable_positions={1}),
            ]
        ),
    ]

    packed = pack_rollout_traces_for_train_batches(
        traces,
        max_pack_length=8,
        global_batch_size=2,
        allow_prefix_sharing=False,
    )

    assert len(packed) == 2
    assert all(sample.merge_info is None for sample in packed)


def test_sample_tensor_dict_tensorizes_trajectory_id() -> None:
    samples = [_sample([1, 2, 3], trainable_positions={1}), _sample([4, 5, 6], trainable_positions={1})]
    for trajectory_id, sample in enumerate(samples):
        sample.trajectory_id = trajectory_id

    td = SampleTensorDict.from_samples(samples)

    assert td["trajectory_id"].tolist() == [0, 1]


def test_pad_sample_tensor_dict_to_multiple_does_not_rebuild_real_samples() -> None:
    samples = [_sample([1, 2, 3]), _sample([4, 5, 6]), _sample([7, 8, 9])]
    td = SampleTensorDict.from_samples(samples)

    padded, original_len = pad_sample_tensor_dict_to_multiple(td, multiple=4, padding_sample_length=3)

    assert original_len == 3
    assert len(padded) == 4
    assert padded["input_ids"][:3].tolist() == td["input_ids"].tolist()
    assert padded["attention_mask"][-1].tolist() == [True, True, True]
    assert not padded["loss_mask"][-1].any()
    assert padded["turn_index"][-1].tolist() == [-1, -1, -1]
    # Real-sample indices are preserved; padding rows use ``-1`` as a sentinel.
    assert padded["index"].tolist() == [0, 1, 2, -1]


def test_pad_sample_tensor_dict_to_multiple_preserves_merged_metadata() -> None:
    real = _trace_with_turn_samples([_sample([1, 2, 3])]).to_packed_samples(max_pack_length=128)[0]
    td = SampleTensorDict.from_samples([real])

    padded, original_len = pad_sample_tensor_dict_to_multiple(td, multiple=2, padding_sample_length=3)

    assert original_len == 1
    assert len(padded) == 2
    merge_infos = padded.get_non_tensor("merge_info")
    assert len(merge_infos) == 2
    # ``merge_trajectory_samples`` aligns the padding sample's ``total_padded`` to 128.
    assert merge_infos[-1].total_padded == 128
    assert padded.get_non_tensor("routing_handles_per_path", default=None) is None
    assert not padded["loss_mask"][-1].any()


def test_pad_sample_tensor_dict_to_multiple_adds_shared_padding_routing_info() -> None:
    padding_handle = TensorHandle(ref="pad-routing")
    turn_sample = _sample([1, 2, 3])
    turn_sample.routing_handles_per_path = [[TensorHandle(ref="real-routing")]]
    real = _trace_with_turn_samples([turn_sample]).to_packed_samples(max_pack_length=8)[0]
    td = SampleTensorDict.from_samples([real])

    padded, original_len = pad_sample_tensor_dict_to_multiple(
        td,
        multiple=4,
        padding_sample_length=3,
        padding_routing_handle=padding_handle,
    )

    assert original_len == 1
    routing_handles = padded.get_non_tensor("routing_handles_per_path")
    assert routing_handles[0] == real.routing_handles_per_path
    assert routing_handles[1:] == [[[padding_handle]]] * 3
    merge_infos = padded.get_non_tensor("merge_info")
    assert merge_infos[0] == real.merge_info
    assert all(mi is not None for mi in merge_infos[1:])
    assert [mi.real_total for mi in merge_infos[1:]] == [3, 3, 3]
    assert not padded["loss_mask"][1:].any()


def test_pad_sample_tensor_dict_to_multiple_adds_padding_routing_info_for_flat_routing() -> None:
    padding_handle = TensorHandle(ref="pad-routing")
    real = _sample([1, 2, 3, 4])
    real.routing_handles_per_path = [[TensorHandle(ref="real-routing")]]
    td = SampleTensorDict.from_samples([real])
    assert "merge_info" not in td.keys()  # noqa: SIM118

    padded, _ = pad_sample_tensor_dict_to_multiple(
        td,
        multiple=2,
        padding_sample_length=4,
        padding_routing_handle=padding_handle,
    )

    assert "merge_info" not in padded.keys()  # noqa: SIM118
    assert padded.get_non_tensor("routing_handles_per_path")[1] == [[padding_handle]]


def test_pad_sample_tensor_dict_to_multiple_uses_tensor_width_for_padding_attention_tokens() -> None:
    real = _trace_with_turn_samples([_sample([1, 2, 3, 4, 5, 6])]).to_packed_samples(max_pack_length=128)[0]
    td = SampleTensorDict.from_samples([real])

    padded, _ = pad_sample_tensor_dict_to_multiple(td, multiple=2, padding_sample_length=6)

    # ``merge_trajectory_samples`` aligns ``total_padded`` to 128; only the first
    # ``padding_sample_length`` positions have ``attention_mask=True`` — the alignment
    # tail stays ``False``.
    pad_attention = padded["attention_mask"][-1].tolist()
    assert pad_attention[:6] == [True, True, True, True, True, True]
    assert not any(pad_attention[6:])
    assert padded.get_non_tensor("merge_info")[-1].total_padded == 128


def test_remove_padding_from_sample_tensor_dict_trims_non_tensor_metadata() -> None:
    real = _trace_with_turn_samples([_sample([1, 2, 3])]).to_packed_samples(max_pack_length=128)[0]
    td = SampleTensorDict.from_samples([real])
    padded, original_len = pad_sample_tensor_dict_to_multiple(td, multiple=2, padding_sample_length=3)

    trimmed = remove_padding_from_sample_tensor_dict(padded, original_len)

    assert len(trimmed) == 1
    assert trimmed["input_ids"].tolist() == td["input_ids"].tolist()
    assert len(trimmed.get_non_tensor("merge_info")) == 1
    assert trimmed.get_non_tensor("merge_info")[0] == real.merge_info


# ---------------------------------------------------------------------------
# Split-trace invariants
#
# When ``to_packed_samples`` splits one trace into multiple packed samples
# (because the trace's merged length exceeds ``max_pack_length``), the resulting
# samples must satisfy invariants the downstream model + R3 cache silently
# require:
#
# 1. Every ``merge_info.total_padded`` is a multiple of 128 — TP+sequence-
#    parallel and MoE token-dispatcher require 128 alignment for
#    ``all_to_all_single`` split-sizes to add up.
#
# 2. The R3 routing cache key (``traj_key_of(routing_handles_per_path)``) is
#    unique per packed sample — even though the per-turn cumulative routing
#    handle list means every packed sample of a split trace shares the same
#    FIRST handle, they MUST have distinct cache keys, otherwise the materialiser
#    returns stale tensor shapes across packed samples.
#
# 3. Trainable token count is preserved across the split.
#
# Both of these invariants were broken once during development and produced
# obscure production failures: the alignment bug surfaced as
# ``RuntimeError: Split sizes doesn't match total dim 0 size`` in MoE
# all_to_all; the cache-key bug surfaced as
# ``AssertionError: pack_routing_for_magi[N]: merged has X rows, expected Y``.
# This test runs at unit-test speed and catches both classes of regression.
# ---------------------------------------------------------------------------


def _trace_for_split_invariants() -> RolloutTrace:
    """Build a multi-turn trace that forces ``to_packed_samples`` to split.

    Turns share a common prefix then diverge — under the label-aware trie this
    accumulates merged length so any reasonable ``max_pack_length`` will split
    the output into multiple packed samples. Routing handles follow the
    cumulative-from-trace-start convention from ``TokenTrace.to_last_turn_sample``.
    """
    samples: list[Sample] = []
    # Each turn extends or diverges; trie merged length grows with each divergent tail.
    sequences = [
        [*range(1, 11), 100],
        [*range(1, 11), 200, 201, 202],
        [*range(1, 11), 300, 301, 302, 303, 304],
        [*range(1, 11), 400, 401, 402, 403, 404, 405, 406],
    ]
    for t, ids in enumerate(sequences):
        sample = _sample(ids, trainable_positions={len(ids) - 1})
        sample.routing_handles_per_path = [[TensorHandle(ref=f"node:turn{j}-chunk0") for j in range(t + 1)]]
        samples.append(sample)
    return _trace_with_turn_samples(samples)


def test_to_packed_samples_split_total_padded_is_128_aligned() -> None:
    """Every emitted packed sample's ``merge_info.total_padded`` must be a multiple of 128.

    Required by TP + sequence-parallel and MoE token-dispatcher
    ``all_to_all_single`` split-sizes to add up to the per-rank token count.
    """
    trace = _trace_for_split_invariants()
    # ``max_pack_length=20`` admits any single turn but forces a split because the
    # merged 4-turn trie expands beyond it via the divergent tails.
    packed = trace.to_packed_samples(max_pack_length=20)
    assert len(packed) >= 2, f"fixture must split into ≥2 packed samples to exercise the split path, got {len(packed)}"
    for i, sample in enumerate(packed):
        assert sample.merge_info is not None, f"packed[{i}] missing merge_info"
        assert sample.merge_info.total_padded % 128 == 0, (
            f"packed[{i}].merge_info.total_padded={sample.merge_info.total_padded} is not a multiple of 128 — "
            "MoE token-dispatcher all_to_all_single will reject this shape."
        )


def test_to_packed_samples_split_routing_cache_keys_are_unique() -> None:
    """Every packed sample of a split trace must have a unique R3 routing cache key.

    The cumulative ``routing_handles_per_path[0]`` convention means every packed
    sample of a split trace shares the same FIRST handle; the cache key must
    derive from a position that's unique per packed sample so the materialiser
    doesn't return a stale merged tensor of the wrong shape.
    """
    from axrl.utils.megatron.routing_caches import traj_key_of

    trace = _trace_for_split_invariants()
    packed = trace.to_packed_samples(max_pack_length=20)
    assert len(packed) >= 2, "fixture must split into ≥2 packed samples to exercise the split path"

    cache_keys = []
    for p in packed:
        assert p.routing_handles_per_path is not None
        cache_keys.append(traj_key_of(p.routing_handles_per_path))
    assert len(set(cache_keys)) == len(cache_keys), (
        f"R3 cache keys collide across packed samples of a split trace: {cache_keys}. "
        "Different packed samples of the same trace would share a cache entry and read stale routing data."
    )
    # Sanity check: every packed sample DOES share its first handle (so the
    # naive ``handles_per_path[0][0]`` key would collide — keep this in the
    # test to document the invariant ``traj_key_of`` is correcting for).
    first_handles = []
    for p in packed:
        assert p.routing_handles_per_path is not None
        first_handles.append(p.routing_handles_per_path[0][0])
    assert len(set(first_handles)) == 1, "fixture sanity: split packed samples should share their first handle"


def test_to_packed_samples_split_preserves_trainable_token_count() -> None:
    """Splitting must not lose or duplicate trainable tokens."""
    trace = _trace_for_split_invariants()
    expected = sum(sum(s.loss_mask) for s in trace.turn_samples)
    for max_pack_length in (20, 30):
        packed = trace.to_packed_samples(max_pack_length=max_pack_length)
        actual = sum(sum(p.loss_mask) for p in packed)
        assert actual == expected, f"max_pack_length={max_pack_length}: trainable tokens {actual} != expected {expected}"
