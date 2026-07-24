"""Multi-turn multi-turn tests for RolloutTrace.

Asserts that per-turn ``TensorHandle``s are accumulated in append order and
land on the merged ``Sample.routing_handles_per_path`` as one inner list per
leaf path. Does not exercise GPU / sglang / Megatron — only the CPU plumbing
between rollout and the merged sample.
"""

from __future__ import annotations

from axrl.data import array_utils
from axrl.data.conversation import Conversation, GenerationState, Message
from axrl.data.generation import GenerationInput, GenerationOutput, TensorHandle
from axrl.data.rollout_trace import RolloutTrace


def _make_conv(prompt_tokens: list[int], *, capture_routing: bool = False) -> Conversation:
    conv = Conversation(
        conversation_id="test",
        messages=[Message(role="user", content="prompt")],
        gen_state=GenerationState(input_ids=array_utils.as_i32(prompt_tokens), capture_routing=capture_routing),
    )
    return conv


def _generation_input(prompt_tokens: list[int]) -> GenerationInput:
    return GenerationInput(session_id="session", input_ids=array_utils.as_i32(prompt_tokens))


def _generation_output(tokens: list[int], *, routing_handle: TensorHandle | None = None) -> GenerationOutput:
    return GenerationOutput(
        session_id="session",
        output_ids=array_utils.as_i32(tokens),
        output_logprobs=array_utils.as_f32([0.0] * len(tokens)),
        output_text="assistant",
        output_text_with_special_tokens="assistant",
        cached_tokens=0,
        finish_reason="stop",
        e2e_elapsed_seconds=0.0,
        stop_reason=None,
        retry=0,
        routing_handle=routing_handle,
    )


def test_routing_handles_accumulate_in_call_order() -> None:
    conv = _make_conv([1, 2, 3])
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=64)

    h0 = TensorHandle(ref="nodeA:call0")
    trace._append_assistant_message(text="a0", tokens=array_utils.as_i32([10, 11, 12]), logprobs=array_utils.as_f32([0.0] * 3), routing_handle=h0)
    trace.append_user_or_tool_message(content="tr0", tokens=array_utils.as_i32([20, 21]))
    h1 = TensorHandle(ref="nodeA:call1")
    trace._append_assistant_message(text="a1", tokens=array_utils.as_i32([30, 31, 32, 33]), logprobs=array_utils.as_f32([0.0] * 4), routing_handle=h1)
    h2 = TensorHandle(ref="nodeA:call2")
    trace._append_assistant_message(text="a2", tokens=array_utils.as_i32([40, 41]), logprobs=array_utils.as_f32([0.0] * 2), routing_handle=h2)

    merged = trace.to_sample()
    assert merged.routing_handles_per_path is not None
    # 3 leaf paths (one per assistant turn), each with the cumulative chain
    # of handles available when that turn was appended.
    assert merged.routing_handles_per_path == [
        [h0],
        [h0, h1],
        [h0, h1, h2],
    ]


def test_routing_handles_skipped_when_none() -> None:
    """No routing_handle passed ⇒ merged sample carries no per-path handles."""
    conv = _make_conv([1, 2])
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=32)
    trace._append_assistant_message(text="a0", tokens=array_utils.as_i32([10, 11]), logprobs=array_utils.as_f32([0.0] * 2))
    merged = trace.to_sample()
    assert merged.routing_handles_per_path is None


def test_routing_handles_preserved_in_call_order_on_last_path() -> None:
    """The trace records exactly the handles the caller supplied, in order."""
    conv = _make_conv([1])
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=32)
    handles = [TensorHandle(ref=f"nodeA:preserve-{t}") for t in range(4)]
    for t, h in enumerate(handles):
        trace._append_assistant_message(text=f"a{t}", tokens=array_utils.as_i32([100 + t]), logprobs=array_utils.as_f32([0.0]), routing_handle=h)
    merged = trace.to_sample()
    assert merged.routing_handles_per_path is not None
    # Last leaf path should carry every handle in the order it was appended.
    assert merged.routing_handles_per_path[-1] == handles


def test_prepare_generation_input_first_prompt_has_no_routing_reuse() -> None:
    trace = RolloutTrace(_make_conv([1, 2, 3], capture_routing=True), token_in_token_out=True, max_length=32)
    generation_input = _generation_input([4, 5, 6])

    trace.prepare_generation_input(generation_input)

    assert generation_input.routed_expert_start_index == 0
    assert trace.conversation.gen_state.captured_routing_rows == 0
    assert trace.token_ids.tolist() == [4, 5, 6]
    assert trace.routing_handles == []


def test_prepare_generation_input_without_routing_capture_drops_handles_and_syncs_tokens() -> None:
    trace = RolloutTrace(_make_conv([1, 2, 3], capture_routing=False), token_in_token_out=True, max_length=32)
    handle = TensorHandle(ref="stale")
    assert trace.token_trace is not None
    trace.token_trace.routing_handles[:] = [handle]
    trace.token_trace.token_info_index_per_handle[:] = [0]
    trace.token_trace.routing_row_count_per_handle[:] = [2]
    generation_input = _generation_input([1, 2, 9])

    trace.prepare_generation_input(generation_input)

    assert generation_input.routed_expert_start_index == 0
    assert trace.conversation.gen_state.captured_routing_rows == 0
    assert trace.token_ids.tolist() == [1, 2, 9]
    assert trace.routing_handles == []


def test_prepare_generation_input_preserves_full_handles_then_slices_next_handle() -> None:
    old_prompt = list(range(1, 701))
    trace = RolloutTrace(_make_conv(old_prompt, capture_routing=True), token_in_token_out=True, max_length=1024)
    handle_a = TensorHandle(ref="A", row_count=80)
    handle_b = TensorHandle(ref="B")
    handle_c = TensorHandle(ref="C")
    assert trace.token_trace is not None
    trace.token_trace.routing_handles[:] = [handle_a, handle_b, handle_c]
    trace.token_trace.token_info_index_per_handle[:] = [0, 0, 0]
    trace.token_trace.routing_row_count_per_handle[:] = [80, 300, 200]
    trace.conversation.gen_state.captured_routing_rows = 580
    new_prompt = [*old_prompt[:181], 9001, 9002]
    generation_input = _generation_input(new_prompt)

    trace.prepare_generation_input(generation_input)

    assert generation_input.routed_expert_start_index == 180
    assert trace.conversation.gen_state.captured_routing_rows == 180
    assert trace.token_ids.tolist() == new_prompt
    assert trace.routing_handles == [handle_a, handle_b.prefix(100)]
    assert trace.token_trace.routing_row_count_per_handle == [80, 100]


def test_append_assistant_after_prepare_uses_preserved_routing_rows_for_new_handle() -> None:
    old_prompt = list(range(1, 701))
    trace = RolloutTrace(_make_conv(old_prompt, capture_routing=True), token_in_token_out=True, max_length=1024)
    handle_a = TensorHandle(ref="A", row_count=80)
    handle_b = TensorHandle(ref="B")
    assert trace.token_trace is not None
    trace.token_trace.routing_handles[:] = [handle_a, handle_b]
    trace.token_trace.token_info_index_per_handle[:] = [0, 0]
    trace.token_trace.routing_row_count_per_handle[:] = [80, 300]
    trace.conversation.gen_state.captured_routing_rows = 380
    new_prompt = [*old_prompt[:181], 9001, 9002]
    trace.prepare_generation_input(_generation_input(new_prompt))
    handle_current = TensorHandle(ref="current")

    trace.append_assistant_message(_generation_output([42, 43, 44], routing_handle=handle_current))

    assert trace.token_trace is not None
    assert trace.token_trace.routing_handles == [handle_a, handle_b.prefix(100), handle_current]
    assert trace.token_trace.routing_row_count_per_handle == [80, 100, 5]
    assert trace.conversation.gen_state.captured_routing_rows == 185
    sample = trace.turn_samples[-1]
    assert sample.routing_handles_per_path == [[handle_a, handle_b.prefix(100), handle_current]]
    assert int(sum(sample.loss_mask)) == 3
