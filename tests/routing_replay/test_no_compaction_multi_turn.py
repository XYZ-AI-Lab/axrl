"""CPU tests for the NO-COMPACTION multi-turn case (the R3 benchmark's scenario).

Paths form a nested chain (each turn's sample is strictly longer than the
previous). Handle chains: ``[h0]``, ``[h0,h1]``, ``[h0,h1,h2]``,
``[h0,h1,h2,h3]``. The benchmark hit
``merged length 1281 > total_padded 1280`` here — these tests reproduce the
full flow and lock in the invariants.

Basic merge-info / source-map invariants are covered in
``test_merge_compacted.py``; this file focuses on the end-to-end gather+pack
and the aligned-boundary edge case specific to the no-compaction layout.
"""

from __future__ import annotations

import numpy as np
import torch

from axrl.data import array_utils
from axrl.data.conversation import Conversation, GenerationState, Message
from axrl.data.generation import TensorHandle
from axrl.data.rollout_trace import RolloutTrace
from axrl.utils.megatron.prefix_tree import (
    gather_merged_routing_per_path,
    merge_trajectory_samples,
)
from axrl.utils.megatron.router_replay import pack_routing_for_magi

PROMPT_TOKENS = [1, 2, 3, 4, 5]
ASST_TOKENS = {
    0: [10, 11, 12, 13],
    1: [20, 21, 22, 23, 24],
    2: [30, 31, 32],
    3: [40, 41, 42, 43, 44, 45],
}
TOOL_TOKENS = {
    0: [100, 101, 102],
    1: [110, 111],
    2: [120, 121, 122, 123],
}
NUM_LAYERS = 2
TOPK = 3


def _build_no_compaction_fixture(
    num_turns: int = 4,
) -> tuple[RolloutTrace, list[TensorHandle], dict[TensorHandle, np.ndarray]]:
    conv = Conversation(
        conversation_id="no-compact",
        messages=[Message(role="user", content="p")],
        gen_state=GenerationState(input_ids=array_utils.as_i32(PROMPT_TOKENS)),
    )
    conv.gen_state.capture_routing = True
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=2048)
    handles: list[TensorHandle] = []
    payloads: dict[TensorHandle, np.ndarray] = {}
    prefix_len = len(PROMPT_TOKENS)
    prior_rows = 0
    for t in range(num_turns):
        h = TensorHandle(ref=f"nodeA:opk{t}")
        handles.append(h)
        asst = ASST_TOKENS[t]
        total_after = prefix_len + len(asst)
        rows = max(total_after - 1 - prior_rows, 0)
        payloads[h] = np.zeros((rows, NUM_LAYERS, TOPK), dtype=np.int16)
        trace._append_assistant_message(
            text=f"a{t}",
            tokens=array_utils.as_i32(asst),
            logprobs=array_utils.as_f32([0.0] * len(asst)),
            routing_handle=h,
        )
        prior_rows = max(0, total_after - 1)
        prefix_len = total_after
        if t < num_turns - 1:
            tool = TOOL_TOKENS[t]
            trace.append_user_or_tool_message(content=f"tr{t}", tokens=array_utils.as_i32(tool))
            prefix_len += len(tool)
    return trace, handles, payloads


def _per_path_routings(samples: list, payloads: dict[TensorHandle, np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for sample in samples:
        assert sample.routing_handles_per_path is not None
        out.append(np.concatenate([payloads[h] for h in sample.routing_handles_per_path[0]], axis=0))
    return out


def test_no_compaction_4_turn_handle_chains_are_strict_prefix() -> None:
    """Each turn sample's chain extends the previous by exactly one handle."""
    trace, handles, _ = _build_no_compaction_fixture(4)
    for s in trace.turn_samples:
        assert s.routing_handles_per_path is not None
    chains = [s.routing_handles_per_path[0] for s in trace.turn_samples if s.routing_handles_per_path is not None]
    assert [len(c) for c in chains] == [1, 2, 3, 4]
    for i in range(4):
        assert chains[i] == handles[: i + 1]


def test_no_compaction_gather_then_pack_no_overflow() -> None:
    """End-to-end: the benchmark's failing invariant must hold — gather ≤ total_padded, pack OK."""
    trace, _, payloads = _build_no_compaction_fixture(4)
    merged = merge_trajectory_samples(trace.turn_samples)
    assert merged.merge_info is not None
    per_path = _per_path_routings(trace.turn_samples, payloads)
    merged_np = gather_merged_routing_per_path(per_path, merged.merge_info)
    merged_tensor = torch.from_numpy(merged_np)
    assert merged_tensor.shape[0] <= merged.merge_info.total_padded
    packed = pack_routing_for_magi([merged_tensor], [merged.merge_info], device=torch.device("cpu"))
    assert packed.shape[0] == merged.merge_info.total_padded


def test_aligned_boundary_multi_turn_no_off_by_one() -> None:
    """Multi-turn trajectory whose total length lands exactly on the 128-alignment boundary."""
    # 5 (prompt) + 4 (a0) + 3 (t0) + 5 (a1) + 2 (t1) + 3 (a2) + 4 (t2) + 102 (a3) = 128
    conv = Conversation(
        conversation_id="bdy-mt",
        messages=[Message(role="user", content="p")],
        gen_state=GenerationState(input_ids=array_utils.as_i32(PROMPT_TOKENS)),
    )
    conv.gen_state.capture_routing = True
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=256)

    prefix_len = 5
    prior_rows = 0
    payloads: dict[TensorHandle, np.ndarray] = {}
    custom_assts = {0: ASST_TOKENS[0], 1: ASST_TOKENS[1], 2: ASST_TOKENS[2], 3: list(range(200, 200 + 102))}
    for t in range(4):
        h = TensorHandle(ref=f"nodeA:bdy{t}")
        a = custom_assts[t]
        total_after = prefix_len + len(a)
        rows = max(total_after - 1 - prior_rows, 0)
        payloads[h] = np.zeros((rows, NUM_LAYERS, TOPK), dtype=np.int16)
        trace._append_assistant_message(text=f"a{t}", tokens=array_utils.as_i32(a), logprobs=array_utils.as_f32([0.0] * len(a)), routing_handle=h)
        prior_rows = max(0, total_after - 1)
        prefix_len = total_after
        if t < 3:
            tool = TOOL_TOKENS[t]
            trace.append_user_or_tool_message(content="t", tokens=array_utils.as_i32(tool))
            prefix_len += len(tool)
    assert len(trace.turn_samples[-1].input_ids) == 128

    merged = merge_trajectory_samples(trace.turn_samples)
    assert merged.merge_info is not None
    mi = merged.merge_info
    assert mi.total_padded == 128
    assert mi.real_total == 128

    per_path = _per_path_routings(trace.turn_samples, payloads)
    merged_np = gather_merged_routing_per_path(per_path, mi)
    assert merged_np.shape[0] == mi.real_total - 1 == 127
    merged_tensor = torch.from_numpy(merged_np)
    packed = pack_routing_for_magi([merged_tensor], [mi], device=torch.device("cpu"))
    assert packed.shape[0] == 128
