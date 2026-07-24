"""CPU tests for RolloutTrace's compaction bookkeeping + handle invalidation.

Covers ``_compact_token_trace`` behaviour:
- Handles with ``chunk_idx >= earliest_newly_masked_idx`` are dropped or sliced
  to keep valid prefix rows.
- ``captured_routing_rows`` resets to the first row whose input token changed
  (or 0 if no valid handles remain).
- Turn sample snapshots taken pre-compact are NOT retroactively mutated.
"""

from __future__ import annotations

import numpy as np

from axrl.data import array_utils
from axrl.data.conversation import Conversation, GenerationState, Message
from axrl.data.generation import TensorHandle
from axrl.data.rollout_trace import RolloutTrace

PLACEHOLDER = array_utils.as_i32([999, 998])


def _make_trace() -> RolloutTrace:
    conv = Conversation(
        conversation_id="bookkeeping",
        messages=[Message(role="user", content="p")],
        gen_state=GenerationState(input_ids=array_utils.as_i32([1, 2, 3])),
    )
    conv.gen_state.capture_routing = True
    return RolloutTrace(conv, token_in_token_out=True, max_length=1024)


def _append_turn(
    trace: RolloutTrace,
    turn_idx: int,
    asst_tokens: list[int],
    tool_tokens: list[int] | None,
) -> TensorHandle:
    handle = TensorHandle(ref=f"nodeA:opk{turn_idx}")
    trace._append_assistant_message(
        text=f"a{turn_idx}",
        tokens=array_utils.as_i32(asst_tokens),
        logprobs=np.zeros(len(asst_tokens), dtype=np.float32),
        routing_handle=handle,
    )
    if tool_tokens is not None:
        trace.append_user_or_tool_message(content=f"t{turn_idx}", tokens=array_utils.as_i32(tool_tokens))
    return handle


def test_compact_drops_handles_whose_chunk_idx_ge_earliest_masked() -> None:
    """Handles whose assistant chunk index falls inside the masked window are dropped."""
    trace = _make_trace()
    h0 = _append_turn(trace, 0, [10, 11], [100, 101, 102])
    _append_turn(trace, 1, [20, 21], [110, 111])
    _append_turn(trace, 2, [30, 31], [120, 121])
    trace.compact(max_recent_tool_results=1, placeholder_tokens=PLACEHOLDER, placeholder_text="<omitted>")
    h1_prefix = TensorHandle(ref="nodeA:opk1", row_start=0, row_count=1)
    assert trace.routing_handles == [h0, h1_prefix]
    assert h1_prefix.row_count == 1


def test_compact_resets_captured_routing_rows_to_end_of_last_valid_asst_minus_1() -> None:
    """After slicing stale handles, ``captured_routing_rows`` advances to the compacted chunk start."""
    trace = _make_trace()
    n_prompt = 3
    asst0 = [10, 11, 12]
    _append_turn(trace, 0, asst0, [100, 101])
    _append_turn(trace, 1, [20, 21], [110])
    trace.compact(max_recent_tool_results=1, placeholder_tokens=PLACEHOLDER, placeholder_text="x")
    # The first row affected by the compacted tool chunk starts at end_of_asst0.
    assert trace.conversation.gen_state.captured_routing_rows == n_prompt + len(asst0)


def test_compact_falls_back_to_zero_when_no_valid_handles_remain() -> None:
    """With all handles invalidated (pathological state), ``captured_routing_rows`` resets to 0."""
    trace = _make_trace()
    _append_turn(trace, 0, [10, 11], [100, 101])
    _append_turn(trace, 1, [20, 21], [110])
    assert trace.token_trace is not None
    del trace.token_trace.routing_handles[:]
    del trace.token_trace.token_info_index_per_handle[:]
    del trace.token_trace.routing_row_count_per_handle[:]
    trace.compact(max_recent_tool_results=1, placeholder_tokens=PLACEHOLDER, placeholder_text="x")
    assert trace.conversation.gen_state.captured_routing_rows == 0


def test_turn_sample_snapshot_pre_compact_is_not_retroactively_changed() -> None:
    """Per-turn samples snapshot handles at build time; later compactions don't mutate them."""
    trace = _make_trace()
    h0 = _append_turn(trace, 0, [10, 11], [100, 101, 102])
    h1 = _append_turn(trace, 1, [20, 21], [110])
    trace.compact(max_recent_tool_results=1, placeholder_tokens=PLACEHOLDER, placeholder_text="x")
    assert trace.routing_handles == [h0, TensorHandle(ref="nodeA:opk1", row_start=0, row_count=1)]  # trace state advanced
    assert trace.turn_samples[1].routing_handles_per_path is not None
    assert trace.turn_samples[1].routing_handles_per_path[0] == [h0, h1]  # snapshot preserved


def test_compact_reduces_running_token_count_by_token_delta() -> None:
    """Tool chunk (len X) → placeholder (len Y) reduces total running tokens by X-Y."""
    trace = _make_trace()
    tool0 = [100, 101, 102, 103]
    _append_turn(trace, 0, [10, 11], tool0)
    _append_turn(trace, 1, [20, 21], [110])
    assert trace.token_trace is not None
    before = sum(len(ti.tokens) for ti in trace.token_trace.token_infos)
    trace.compact(max_recent_tool_results=1, placeholder_tokens=PLACEHOLDER, placeholder_text="x")
    after = sum(len(ti.tokens) for ti in trace.token_trace.token_infos)
    assert before - after == len(tool0) - len(PLACEHOLDER)


def test_compact_owns_placeholder_tokens() -> None:
    """Compacted token chunks must not alias caller-owned placeholder arrays."""
    trace = _make_trace()
    _append_turn(trace, 0, [10, 11], [100, 101, 102])
    _append_turn(trace, 1, [20, 21], [110])
    placeholder = array_utils.as_i32([999, 998])

    trace.compact(max_recent_tool_results=1, placeholder_tokens=placeholder, placeholder_text="x")
    placeholder[0] = 123

    assert trace.token_ids.tolist() == [1, 2, 3, 10, 11, 999, 998, 20, 21, 110]


def test_compact_rewrites_tool_role_messages_and_leaves_user_messages() -> None:
    """Only tool-role chunks are compacted; appended user context stays intact."""
    trace = _make_trace()
    trace._append_assistant_message(text="a0", tokens=array_utils.as_i32([10, 11]), logprobs=array_utils.as_f32([0.0, 0.0]))
    trace.append_user_or_tool_message(content="t0", tokens=array_utils.as_i32([100, 101, 102]), tool_call_id="call_0")
    trace._append_assistant_message(text="a1", tokens=array_utils.as_i32([20, 21]), logprobs=array_utils.as_f32([0.0, 0.0]))
    trace.append_user_or_tool_message(content="extra user context", tokens=array_utils.as_i32([120]), role="user")
    trace._append_assistant_message(text="a2", tokens=array_utils.as_i32([30, 31]), logprobs=array_utils.as_f32([0.0, 0.0]))
    trace.append_user_or_tool_message(content="t1", tokens=array_utils.as_i32([110, 111]), tool_call_id="call_1")

    trace.compact(max_recent_tool_results=1, placeholder_tokens=PLACEHOLDER, placeholder_text="<omitted>")

    assert trace.conversation.messages[2].role == "tool"
    assert trace.conversation.messages[2].content == "<omitted>"
    assert trace.conversation.messages[4].role == "user"
    assert trace.conversation.messages[4].content == "extra user context"
    assert trace.conversation.messages[6].content == "t1"


def test_compact_summary_replaces_live_prompt_and_preserves_turn_samples() -> None:
    """Summary compaction resets the live token trace but keeps already-built turn samples."""
    trace = _make_trace()
    h0 = _append_turn(trace, 0, [10, 11], [100, 101])
    _append_turn(trace, 1, [20, 21], [110, 111])
    assert len(trace.turn_samples) == 2
    assert trace.routing_handles

    summary_conv = Conversation(
        conversation_id="bookkeeping",
        messages=[Message(role="user", content="p\n\nsummary")],
        gen_state=GenerationState(input_ids=array_utils.as_i32([7, 8, 9]), capture_routing=True, captured_routing_rows=123),
    )

    trace.compact(conv_with_summary=summary_conv)

    assert len(trace.turn_samples) == 2
    assert trace.turn_samples[0].routing_handles_per_path == [[h0]]
    assert trace.token_ids.tolist() == [7, 8, 9]
    assert trace.routing_handles == []
    assert trace.conversation.messages == summary_conv.messages
    assert trace.conversation.gen_state.captured_routing_rows == 0
