"""Shared CPU fixtures for R3 + compaction tests.

Build a fully-populated 4-turn ``RolloutTrace`` with per-turn compactions,
without spinning up sglang or Ray. Routing handles are minted deterministically;
synthetic per-handle routing tensors are produced via ``make_routing_payloads``
so the end-to-end fetch/gather/pack pipeline can be exercised purely on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from axrl.data import array_utils
from axrl.data.conversation import Conversation, GenerationState, Message
from axrl.data.generation import TensorHandle
from axrl.data.rollout_trace import RolloutTrace

if TYPE_CHECKING:
    from axrl.data.sample import Sample

    pass

# Synthetic token vocab — values chosen to make each chunk uniquely identifiable
# so prefix-tree branching is predictable in the tests.
PROMPT_TOKENS = [1, 2, 3, 4, 5]  # 5 tokens
ASST_TOKENS = {
    0: [10, 11, 12, 13],  # 4 tokens
    1: [20, 21, 22, 23, 24],  # 5 tokens
    2: [30, 31, 32],  # 3 tokens
    3: [40, 41, 42, 43, 44, 45],  # 6 tokens
}
TOOL_TOKENS = {
    0: [100, 101, 102],  # 3 tokens
    1: [110, 111],  # 2 tokens
    2: [120, 121, 122, 123],  # 4 tokens
}
PLACEHOLDER_TOKENS = [999, 998]  # 2 tokens
PAD_TOKEN_ID = 0

# Routing geometry (small so synthetic routing is cheap).
NUM_LAYERS = 2
TOPK = 3


@dataclass
class CompactedFixture:
    """Pre-built 4-turn RolloutTrace with compaction applied between turns.

    Attributes:
        trace: The populated ``RolloutTrace`` (token_in_token_out=True).
        handles: Handle sequence actually captured (after compactions — only
            the handles that survived in ``trace.routing_handles``).
        all_minted_handles: Every handle ever minted and "put to the tensor store" — i.e.,
            h0..h3. Some may have been dropped from ``trace.routing_handles``
            by compactions, but they're still in the tensor store (test needs to fetch them).
        turn_samples: The 4 per-turn ``Sample`` objects as snapshotted at
            ``append_assistant_message`` time (with their per-path handle
            snapshots intact).
        expected_payload_rows: ``expected_payload_rows[k]`` is the number of
            routing rows sglang would have returned for handle ``h_k``, given
            the fixture's compaction policy. Used by
            ``make_routing_payloads`` to mint synthetic tensors of the
            correct shape for any ``max_recent_tool_results``.
        max_recent_tool_results: The policy used when building the fixture.
    """

    trace: RolloutTrace
    handles: list[TensorHandle]
    all_minted_handles: list[TensorHandle]
    turn_samples: list[Sample]
    expected_payload_rows: list[int]
    max_recent_tool_results: int


def _seed_conversation() -> Conversation:
    conv = Conversation(
        conversation_id="compacted-fixture",
        messages=[Message(role="user", content="synthetic prompt")],
        gen_state=GenerationState(input_ids=array_utils.as_i32(PROMPT_TOKENS)),
    )
    conv.gen_state.capture_routing = True
    return conv


def build_compacted_fixture(max_recent_tool_results: int = 1) -> CompactedFixture:
    """Build a compacted 4-turn RolloutTrace fixture.

    Turn layout (with ``max_recent_tool_results=1``):
        - Turn 0: gen asst0 (handle h0). Append tool0. Compact → no-op.
        - Turn 1: gen asst1 (handle h1). Append tool1. Compact → drop tool0
          to placeholder; drop h1 (stale: tool0 in its prefix).
        - Turn 2: gen asst2 (handle h2). Append tool2. Compact → drop tool1 to
          placeholder; drop h2 (stale).
        - Turn 3: gen asst3 (handle h3). No compact (last turn).

    Final ``trace.routing_handles == [h0, h3]`` (2 survivors).
    Per-turn samples snapshotted BEFORE the compactions that would have dropped
    their handles retain the original chain — so
    ``turn_samples[1].routing_handles_per_path == [[h0, h1]]`` etc.

    For any ``max_recent_tool_results``, ``expected_payload_rows`` on the
    returned fixture records the exact row count sglang would have returned
    for each turn's handle, computed from the trace state machine.
    """
    seed = _seed_conversation()
    trace = RolloutTrace(seed, token_in_token_out=True, max_length=2048)
    all_minted: list[TensorHandle] = []
    expected_payload_rows: list[int] = []

    for turn_idx in range(4):
        # Simulate the handle sglang would return for this turn's call.
        handle = TensorHandle(ref=f"nodeA:opk{turn_idx}")
        all_minted.append(handle)
        # Capture the state sglang sees at this call: prior_rows is the
        # trace's current captured_routing_rows (advanced after each turn,
        # reset after each compact).
        prior_rows_before_turn = trace.conversation.gen_state.captured_routing_rows
        asst_toks = array_utils.as_i32(ASST_TOKENS[turn_idx])
        trace._append_assistant_message(
            text=f"asst{turn_idx}",
            tokens=asst_toks,
            logprobs=np.zeros(len(asst_toks), dtype=np.float32),
            routing_handle=handle,
        )
        running_len_after = len(trace.token_trace.token_ids) if trace.token_trace is not None else 0
        expected_payload_rows.append(max(0, running_len_after - 1 - prior_rows_before_turn))
        if turn_idx == 3:
            break
        tool_toks = array_utils.as_i32(TOOL_TOKENS[turn_idx])
        trace.append_user_or_tool_message(content=f"tool{turn_idx}", tokens=tool_toks)
        trace.compact(
            max_recent_tool_results=max_recent_tool_results,
            placeholder_tokens=array_utils.as_i32(PLACEHOLDER_TOKENS),
            placeholder_text="<omitted>",
        )

    return CompactedFixture(
        trace=trace,
        handles=list(trace.routing_handles),
        all_minted_handles=all_minted,
        turn_samples=list(trace.turn_samples),
        expected_payload_rows=expected_payload_rows,
        max_recent_tool_results=max_recent_tool_results,
    )


def make_routing_payloads(
    fixture: CompactedFixture,
    num_layers: int = NUM_LAYERS,
    topk: int = TOPK,
) -> dict[TensorHandle, np.ndarray]:
    """Mint deterministic synthetic routing tensors for every minted handle.

    Handle ``h_turn`` gets a tensor of shape ``(rows_turn, L, K)`` where
    ``rows_turn`` comes from ``fixture.expected_payload_rows[turn]`` —
    matching what sglang would have returned given the fixture's
    ``captured_routing_rows`` state at each turn under the fixture's
    compaction policy.

    Tensor values encode ``(turn_idx, row_idx)`` so tests can assert which
    handle and row a merged-trie position was gathered from:
        ``data[handle][row] == turn_idx * 10_000 + row`` (broadcast to all
        (L, K) cells).
    """
    payloads: dict[TensorHandle, np.ndarray] = {}
    for turn_idx, (handle, rows) in enumerate(zip(fixture.all_minted_handles, fixture.expected_payload_rows, strict=True)):
        arr = np.empty((rows, num_layers, topk), dtype=np.int16)
        for row in range(rows):
            arr[row, :, :] = turn_idx * 10_000 + row
        payloads[handle] = arr
    for sample in fixture.turn_samples:
        if sample.routing_handles_per_path is None:
            continue
        for path_handles in sample.routing_handles_per_path:
            for handle in path_handles:
                _add_sliced_payload_if_needed(handle, payloads)
    for handle in fixture.trace.routing_handles:
        _add_sliced_payload_if_needed(handle, payloads)
    return payloads


def _add_sliced_payload_if_needed(handle: TensorHandle, payloads: dict[TensorHandle, np.ndarray]) -> None:
    if handle in payloads:
        return
    base_handle = TensorHandle(ref=handle.ref)
    base = payloads[base_handle]
    end = None if handle.row_count is None else handle.row_start + handle.row_count
    payloads[handle] = base[handle.row_start : end]


# --------------------------------------------------------------------- #
# Helpers for asserting trie structure
# --------------------------------------------------------------------- #


def expected_path_input_ids() -> list[list[int]]:
    """Tokens for each of the 4 compacted turn samples.

    Mirrors what ``to_last_turn_sample`` produces at each turn's add time.
    """
    ph = list(PLACEHOLDER_TOKENS)
    return [
        # turn 0: built BEFORE any compact — tokens are [prompt, asst0].
        PROMPT_TOKENS + ASST_TOKENS[0],
        # turn 1: built BEFORE turn-1 compact — tokens are [prompt, asst0, tool0_orig, asst1].
        PROMPT_TOKENS + ASST_TOKENS[0] + TOOL_TOKENS[0] + ASST_TOKENS[1],
        # turn 2: built AFTER turn-1 compact — tool0 → placeholder; then full state is
        # [prompt, asst0, ph, asst1, tool1_orig, asst2].
        PROMPT_TOKENS + ASST_TOKENS[0] + ph + ASST_TOKENS[1] + TOOL_TOKENS[1] + ASST_TOKENS[2],
        # turn 3: built AFTER turn-2 compact — tool1 → placeholder; state is
        # [prompt, asst0, ph, asst1, ph, asst2, tool2_orig, asst3].
        PROMPT_TOKENS + ASST_TOKENS[0] + ph + ASST_TOKENS[1] + ph + ASST_TOKENS[2] + TOOL_TOKENS[2] + ASST_TOKENS[3],
    ]
