from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm import tqdm

from axrl.data import array_utils
from axrl.data.conversation import Conversation, Message
from axrl.data.token_trace import TokenTrace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from axrl.data.conversation import ToolCall
    from axrl.data.generation import GenerationInput, GenerationOutput, TensorHandle
    from axrl.data.sample import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationInputPreparation:
    shared_prefix_tokens: int
    previous_routing_rows: int
    preserved_routing_rows: int
    dropped_routing_rows: int
    preserved_routing_handles: int


class RolloutTrace:
    """Rollout-side trajectory recorder.

    Owns one chunk-based ``TokenTrace`` per rollout plus a list of per-turn
    ``Sample``s for the merged-forward training path.
    """

    def __init__(
        self,
        conv: Conversation,
        *,
        token_in_token_out: bool = True,
        max_length: int | None = None,
    ) -> None:
        if token_in_token_out:
            assert max_length is not None, "token_in_token_out requires max_length"
            assert conv.gen_state.input_ids is not None, "token_in_token_out requires the seed conversation to carry tokens"
        self.token_in_token_out = token_in_token_out
        self.max_length = max_length
        self.initialize(conv)

    def initialize(self, conv: Conversation) -> None:
        self.conversation = copy.deepcopy(conv)
        self.turn_samples: list[Sample] = []
        self.token_trace: TokenTrace | None = None
        if not self.token_in_token_out:
            return
        assert conv.gen_state.input_ids is not None and self.max_length is not None
        self.token_trace = TokenTrace()
        self.token_trace.extend_tokens(conv.gen_state.input_ids, token_type="init")
        seed_len = self.token_trace.token_count
        assert seed_len <= self.max_length, f"seed conversation tokens ({seed_len}) exceed max_length ({self.max_length})"
        self._sync_conv_gen_state_from_token_trace()

    def _token_mode_state(self) -> tuple[TokenTrace, int]:
        assert self.token_trace is not None, "token_trace not initialized"
        assert self.max_length is not None, "max_length not set"
        return self.token_trace, self.max_length

    @property
    def running_len(self) -> int:
        assert self.token_trace is not None
        return self.token_trace.token_count

    @property
    def token_ids(self) -> NDArray[np.int32]:
        assert self.token_trace is not None
        return self.token_trace.token_ids

    @property
    def routing_handles(self) -> list[TensorHandle]:
        assert self.token_trace is not None
        return self.token_trace.routing_handles

    def prepare_generation_input(self, generation_input: GenerationInput) -> GenerationInputPreparation:
        """Sync a full-rendered generation prompt and set routing reuse metadata.

        Black-box runtimes send each model call as a freshly rendered prompt. The
        prompt may share a prefix with the previous live trace, so keep routing
        handles for the token-identical prefix and drop the stale suffix before
        generation. Existing ``turn_samples`` are immutable; only the live prompt
        for the next assistant turn is rewritten.
        """
        assert self.token_in_token_out, "prepare_generation_input requires token_in_token_out=True"

        token_trace, max_length = self._token_mode_state()
        input_ids = array_utils.as_i32(generation_input.input_ids)
        assert len(input_ids) <= max_length, f"generation prompt tokens ({len(input_ids)}) exceed max_length ({max_length})"

        shared_prefix_tokens = _common_prefix_len(token_trace.token_ids, input_ids)
        preserved_rows = 0
        previous_routing_rows = 0
        preserved_handles: list[TensorHandle] = []
        preserved_row_counts: list[int] = []
        if self.conversation.gen_state.capture_routing:
            previous_routing_rows = token_trace.captured_routing_rows
            preserved_rows = min(max(shared_prefix_tokens - 1, 0), previous_routing_rows)
            preserved_handles, preserved_row_counts = self._routing_prefix_by_rows(preserved_rows)

        self.token_trace = TokenTrace()
        self.token_trace.extend_tokens(input_ids, token_type="init")
        self.token_trace.routing_handles[:] = preserved_handles
        self.token_trace.routing_row_count_per_handle[:] = preserved_row_counts
        self.token_trace.token_info_index_per_handle[:] = [0] * len(preserved_handles)

        self.conversation.gen_state.input_ids = self.token_trace.token_ids
        self.conversation.gen_state.captured_routing_rows = preserved_rows
        generation_input.input_ids = self.token_trace.token_ids
        generation_input.routed_expert_start_index = preserved_rows
        return GenerationInputPreparation(
            shared_prefix_tokens=shared_prefix_tokens,
            previous_routing_rows=previous_routing_rows,
            preserved_routing_rows=preserved_rows,
            dropped_routing_rows=max(previous_routing_rows - preserved_rows, 0),
            preserved_routing_handles=len(preserved_handles),
        )

    def append_assistant_message(self, generation_output: GenerationOutput) -> Conversation:
        """Append an assistant turn from a rollout generation output."""
        return self._append_assistant_message(
            text=generation_output.output_text,
            tokens=generation_output.output_ids,
            logprobs=generation_output.output_logprobs,
            tool_calls=generation_output.tool_calls,
            routing_handle=generation_output.routing_handle,
            assistant_boundary_token_id=generation_output.assistant_boundary_token_id,
        )

    def _append_assistant_message(
        self,
        *,
        text: str,
        tokens: NDArray[np.int32] | None = None,
        logprobs: NDArray[np.float32] | None = None,
        tool_calls: list[ToolCall] | None = None,
        routing_handle: TensorHandle | None = None,
        assistant_boundary_token_id: int | None = None,
    ) -> Conversation:
        """Append an assistant turn; build a per-turn Sample by chunk-view masking."""
        self.conversation.add_message(Message(role="assistant", content=text, tool_calls=tool_calls))
        if not self.token_in_token_out:
            return self.conversation
        token_trace, max_length = self._token_mode_state()
        assert tokens is not None and logprobs is not None
        prior_routing_rows = self.conversation.gen_state.captured_routing_rows if self.conversation.gen_state.capture_routing else 0
        routing_row_count = None
        if routing_handle is not None:
            routing_row_count = token_trace.new_routing_row_count(len(tokens), prior_routing_rows=prior_routing_rows)
        token_trace.extend_tokens(
            tokens,
            logprobs,
            token_type="assistant",
            routing_handle=routing_handle,
            routing_row_count=routing_row_count,
        )
        running_len = token_trace.token_count
        assert running_len <= max_length, f"running token length {running_len} exceeds max_length {max_length}"
        sample = token_trace.to_last_turn_sample(max_length=running_len, pad_token_id=0)
        self.turn_samples.append(sample)
        self._append_assistant_boundary_if_needed(token_trace, tokens, assistant_boundary_token_id, max_length=max_length)
        self._sync_conv_gen_state_from_token_trace(after_assistant_turn=True)
        return self.conversation

    def append_user_or_tool_message(
        self,
        content: str,
        tokens: NDArray[np.int32] | None = None,
        tool_call_id: str | None = None,
        role: Literal["tool", "user"] = "tool",
    ) -> Conversation:
        self.conversation.add_message(Message(role=role, content=content, tool_call_id=tool_call_id))
        if not self.token_in_token_out:
            return self.conversation
        token_trace, max_length = self._token_mode_state()
        assert tokens is not None
        token_type = "tool_result" if role == "tool" else "user"
        token_trace.extend_tokens(tokens, token_type=token_type)
        running_len = token_trace.token_count
        assert running_len <= max_length, f"running token length {running_len} exceeds max_length {max_length}"
        self._sync_conv_gen_state_from_token_trace()
        return self.conversation

    def _sync_conv_gen_state_from_token_trace(self, *, after_assistant_turn: bool = False) -> None:
        if not self.token_in_token_out:
            return
        assert self.token_trace is not None
        self.conversation.gen_state.input_ids = self.token_trace.token_ids
        if after_assistant_turn and self.conversation.gen_state.capture_routing:
            self.conversation.gen_state.captured_routing_rows = self.token_trace.captured_routing_rows

    def _append_assistant_boundary_if_needed(
        self,
        token_trace: TokenTrace,
        tokens: NDArray[np.int32],
        assistant_boundary_token_id: int | None,
        *,
        max_length: int,
    ) -> None:
        if not self._needs_assistant_boundary(tokens, assistant_boundary_token_id):
            return
        assert assistant_boundary_token_id is not None
        token_count_before = token_trace.token_count
        boundary_len = token_count_before + 1
        if boundary_len > max_length:
            logger.info(
                "Skipping assistant boundary token append because it would exceed max_length: token_id=%s, token_count_before=%s, max_length=%s",
                assistant_boundary_token_id,
                token_count_before,
                max_length,
            )
            return
        logger.info(
            "Appending assistant boundary token after creating the assistant turn sample: token_id=%s, token_count_before=%s",
            assistant_boundary_token_id,
            token_count_before,
        )
        token_trace.extend_tokens(
            array_utils.as_i32([assistant_boundary_token_id]),
            token_type="assistant_boundary",
        )

    @staticmethod
    def _needs_assistant_boundary(tokens: NDArray[np.int32], assistant_boundary_token_id: int | None) -> bool:
        return assistant_boundary_token_id is not None and (len(tokens) == 0 or int(tokens[-1]) != assistant_boundary_token_id)

    def observation(self) -> Conversation:
        """Return the conversation reflecting the current trace state."""
        return self.conversation

    def set_turn_reward(self, turn_index: int, reward: float) -> None:
        assert 0 <= turn_index < len(self.turn_samples)
        sample = self.turn_samples[turn_index]
        sample.turn_reward = np.where(sample.loss_mask, np.float32(reward), np.float32(0.0)).astype(np.float32, copy=False)

    def to_sample(self) -> Sample:
        """Return one merged ``Sample`` (with ``merge_info`` set) covering this trajectory.

        Length equals ``merge_info.total_padded`` (variable per trajectory).
        Padding to a uniform batch width happens in ``SampleTensorDict.from_samples``.
        """
        from axrl.utils.megatron.prefix_tree import merge_trajectory_samples  # local import to avoid cycle

        assert self.turn_samples, "RolloutTrace.to_sample requires at least one assistant turn"
        return merge_trajectory_samples(self.turn_samples)

    def to_packed_samples(self, max_pack_length: int, *, allow_prefix_sharing: bool = True) -> list[Sample]:
        """Greedily split this trace into merged samples bounded by ``max_pack_length``.

        Each emitted sample contains one or more consecutive entries from
        ``turn_samples``. Adding the next turn starts a new emitted sample when
        the merged sample would exceed ``max_pack_length``.

        When ``allow_prefix_sharing`` is False the trace must form a linear
        prefix chain (no compaction / branching) and is exported as a single
        flat ``Sample`` with ``merge_info=None`` — the only shape the baseline
        ``GPTModel`` forward can consume. Splitting and tree-structured packing
        both require ``allow_prefix_sharing=True`` (i.e. Magi merged forward).
        """
        from axrl.utils.megatron.prefix_tree import (  # local import to avoid cycle
            MergingTree,
            add_sample_to_merging_tree,
            get_packed_len_if_merge,
            merge_trajectory_samples,
        )

        assert max_pack_length > 0, f"max_pack_length must be positive, got {max_pack_length}"
        assert self.turn_samples, "RolloutTrace.to_packed_samples requires at least one assistant turn"

        if not allow_prefix_sharing:
            return [self._to_flat_single_sample(max_pack_length=max_pack_length)]

        def merge_group(group: list[Sample]) -> Sample:
            assert all(sample.reward_baseline == group[0].reward_baseline for sample in group), (
                "All samples in a packed group must have the same reward_baseline. Normalize group rewards before packing rollout traces."
            )
            merged = merge_trajectory_samples(group, align_size=1)
            merged.reward = group[0].reward
            merged.reward_baseline = group[0].reward_baseline
            return merged

        packed: list[Sample] = []
        current_group: list[Sample] = []
        current_tree = MergingTree()
        for turn_sample in self.turn_samples:
            single_len = len(turn_sample.input_ids)
            if single_len > max_pack_length:
                raise ValueError(f"pre-merge turn sample length {single_len} exceeds max_pack_length {max_pack_length}")

            if get_packed_len_if_merge(current_tree, turn_sample) <= max_pack_length:
                current_group.append(turn_sample)
                add_sample_to_merging_tree(current_tree, turn_sample)
                continue

            assert current_group
            packed.append(merge_group(current_group))
            current_group = [turn_sample]
            current_tree = MergingTree()
            add_sample_to_merging_tree(current_tree, turn_sample)

        assert current_group
        packed.append(merge_group(current_group))
        return packed

    def _to_flat_single_sample(self, *, max_pack_length: int) -> Sample:
        """Export this trace as one flat ``Sample`` with ``merge_info=None``.

        Used by ``to_packed_samples`` when prefix sharing is disallowed (i.e.
        the trainer's forward path is the baseline ``GPTModel`` forward, which
        cannot consume ``merge_info``-bearing samples). Requires the trace's
        turns to form a strict linear prefix chain (compaction / branching is
        rejected) and the full sequence to fit in ``max_pack_length`` (no
        splitting allowed — splitting requires merge bookkeeping that only the
        merged forward understands).
        """
        from axrl.utils.megatron.prefix_tree import merge_trajectory_samples  # local import to avoid cycle

        longest_turn_idx = max(range(len(self.turn_samples)), key=lambda i: len(self.turn_samples[i].input_ids))
        longest_input_ids = self.turn_samples[longest_turn_idx].input_ids
        for sample in self.turn_samples:
            sample_len = len(sample.input_ids)
            if sample_len <= len(longest_input_ids) and np.array_equal(sample.input_ids, longest_input_ids[:sample_len]):
                continue
            raise ValueError(
                "RolloutTrace turns do not form a linear prefix chain "
                "(at least one turn is not a prefix of the longest turn); "
                "compacted or branching traces require allow_prefix_sharing=True."
            )

        merged = merge_trajectory_samples(self.turn_samples, align_size=1)
        # ``merge_trajectory_samples`` aligns ``total_padded`` to ``lcm(align_size, 128)``,
        # so compare on the real (non-padding) length to verify the linear-chain invariant.
        real_len = sum(merged.attention_mask)
        longest_turn_len = max(len(s.input_ids) for s in self.turn_samples)
        if real_len != longest_turn_len:
            raise ValueError(
                "RolloutTrace turns do not form a linear prefix chain "
                f"(merged real length {real_len} != longest turn length {longest_turn_len}); "
                "compacted or branching traces require allow_prefix_sharing=True."
            )
        if real_len > max_pack_length:
            raise ValueError(f"flat trace length {real_len} exceeds max_pack_length {max_pack_length}; splitting requires allow_prefix_sharing=True.")

        assert all(sample.reward_baseline == self.turn_samples[0].reward_baseline for sample in self.turn_samples), (
            "All turn samples in a flat-export trace must share reward_baseline. Normalize group rewards before packing."
        )
        merged.reward = self.turn_samples[-1].reward
        merged.reward_baseline = self.turn_samples[-1].reward_baseline
        merged.merge_info = None
        if merged.routing_handles_per_path is not None:
            # Flat export only: the Magi merged-forward path keeps per-turn
            # handle paths through ``merge_trajectory_samples`` above.
            # Each per-turn sample already carries the cumulative routing chain
            # up to that turn. A flat linear-chain sample is therefore represented
            # by the longest path's chain; concatenating every turn's cumulative
            # chain would duplicate prefix routing and desync R3 replay lengths.
            source_handles = self.turn_samples[longest_turn_idx].routing_handles_per_path
            assert source_handles is not None and len(source_handles) == 1
            merged.routing_handles_per_path = [list(source_handles[0])]
        return merged

    def compact(
        self,
        *,
        max_recent_tool_results: int | None = None,
        placeholder_tokens: NDArray[np.int32] | None = None,
        placeholder_text: str | None = None,
        conv_with_summary: Conversation | None = None,
    ) -> Conversation:
        if conv_with_summary is not None:
            self._compact_to_summary_conversation(conv_with_summary)
            return self.conversation

        assert max_recent_tool_results is not None
        assert placeholder_tokens is not None
        assert placeholder_text is not None
        assert max_recent_tool_results >= 0
        if self.token_in_token_out:
            self._compact_token_trace(max_recent_tool_results, placeholder_tokens, placeholder_text)
        else:
            self._compact_conversation_only(max_recent_tool_results, placeholder_text)
        self._sync_conv_gen_state_from_token_trace()
        return self.conversation

    def _compact_to_summary_conversation(self, conv_with_summary: Conversation) -> None:
        """Replace the live prompt with a pre-tokenized summary conversation.

        Existing per-turn samples stay immutable; future assistant samples branch
        from the summarized prompt.
        """
        summary_conv = copy.deepcopy(conv_with_summary)
        if not self.token_in_token_out:
            self.conversation = summary_conv
            return

        assert summary_conv.gen_state.input_ids is not None, "summary conversation must carry input_ids"
        assert self.max_length is not None
        self.conversation = summary_conv
        self.token_trace = TokenTrace()
        self.token_trace.extend_tokens(summary_conv.gen_state.input_ids, token_type="init")
        summary_len = self.token_trace.token_count
        assert summary_len <= self.max_length, f"summary conversation tokens ({summary_len}) exceed max_length ({self.max_length})"
        self.conversation.gen_state.captured_routing_rows = 0
        self._sync_conv_gen_state_from_token_trace()

    def _compact_conversation_only(self, max_recent_tool_results: int, placeholder_text: str) -> None:
        """Conversation-only compaction: rewrite older tool message content. Idempotent."""
        tool_messages = [msg for msg in self.conversation.messages if msg.role == "tool"]
        num_to_mask = max(0, len(tool_messages) - max_recent_tool_results)
        for tool_message in tool_messages[:num_to_mask]:
            tool_message.content = placeholder_text

    def _compact_token_trace(
        self,
        max_recent_tool_results: int,
        placeholder_tokens: NDArray[np.int32],
        placeholder_text: str,
    ) -> None:
        """Token-mode compaction. Mutates the token trace AND the conversation in lockstep."""
        token_trace, max_length = self._token_mode_state()
        uncompacted_tool_chunks = [
            (idx, token_info)
            for idx, token_info in enumerate(token_trace.token_infos)
            if token_info.token_type == "tool_result" and not token_info.compacted_to_placeholder
        ]
        num_to_mask = max(0, len(uncompacted_tool_chunks) - max_recent_tool_results)
        if num_to_mask == 0:
            return
        chunks_to_mask = uncompacted_tool_chunks[:num_to_mask]
        earliest_newly_masked_idx = chunks_to_mask[0][0]

        self._invalidate_stale_r3_handles(earliest_newly_masked_idx)

        for _idx, token_info in chunks_to_mask:
            token_info.tokens = array_utils.as_i32(placeholder_tokens).copy()
            token_info.logprobs = np.zeros(len(placeholder_tokens), dtype=np.float32)
            token_info.compacted_to_placeholder = True

        uncompacted_tool_messages = [msg for msg in self.conversation.messages if msg.role == "tool" and msg.content != placeholder_text]
        assert len(uncompacted_tool_messages) >= num_to_mask, (
            f"found {len(uncompacted_tool_messages)} uncompacted tool messages for {num_to_mask} compacted tool-result chunks"
        )
        for tool_message in uncompacted_tool_messages[:num_to_mask]:
            tool_message.content = placeholder_text

        running_len = sum(len(token_info.tokens) for token_info in token_trace.token_infos)
        assert running_len < max_length, f"running token length {running_len} exceeds max_length {max_length}"

    def _invalidate_stale_r3_handles(self, earliest_newly_masked_idx: int) -> None:
        """Drop or slice R3 handles whose prefill prefix included a rewritten chunk.

        A handle is stale iff its assistant chunk index is
        ``>= earliest_newly_masked_idx``. The prefix of the first stale handle
        can still be valid when it covers rows before the rewritten chunk, so
        preserve that prefix as a first-axis handle slice and only recapture
        from the first row whose input token changed.
        """
        gen_state = self.conversation.gen_state
        if not gen_state.capture_routing:
            return
        token_trace = self.token_trace
        assert token_trace is not None
        assert len(token_trace.routing_handles) == len(token_trace.token_info_index_per_handle) == len(token_trace.routing_row_count_per_handle)

        preserve_rows = min(
            sum(len(ti.tokens) for ti in token_trace.token_infos[:earliest_newly_masked_idx]),
            gen_state.captured_routing_rows,
        )
        new_handles: list[TensorHandle] = []
        new_chunk_indices: list[int] = []
        new_row_counts: list[int] = []
        rows_kept = 0

        for handle, chunk_idx, row_count in zip(
            token_trace.routing_handles,
            token_trace.token_info_index_per_handle,
            token_trace.routing_row_count_per_handle,
            strict=True,
        ):
            if rows_kept >= preserve_rows:
                break
            rows_remaining = preserve_rows - rows_kept
            if chunk_idx < earliest_newly_masked_idx and row_count <= rows_remaining:
                new_handles.append(handle)
                new_chunk_indices.append(chunk_idx)
                new_row_counts.append(row_count)
                rows_kept += row_count
                continue

            prefix_rows = min(row_count, rows_remaining)
            if prefix_rows > 0:
                new_handles.append(handle.prefix(prefix_rows))
                new_chunk_indices.append(max(0, earliest_newly_masked_idx - 1))
                new_row_counts.append(prefix_rows)
                rows_kept += prefix_rows
            break

        token_trace.routing_handles[:] = new_handles
        token_trace.token_info_index_per_handle[:] = new_chunk_indices
        token_trace.routing_row_count_per_handle[:] = new_row_counts
        gen_state.captured_routing_rows = rows_kept

    def _routing_prefix_by_rows(self, preserve_rows: int) -> tuple[list[TensorHandle], list[int]]:
        assert preserve_rows >= 0, f"preserve_rows must be non-negative, got {preserve_rows}"
        token_trace = self.token_trace
        assert token_trace is not None
        new_handles: list[TensorHandle] = []
        new_row_counts: list[int] = []
        rows_kept = 0
        for handle, row_count in zip(token_trace.routing_handles, token_trace.routing_row_count_per_handle, strict=True):
            if rows_kept >= preserve_rows:
                break
            rows_remaining = preserve_rows - rows_kept
            if row_count <= rows_remaining:
                new_handles.append(handle)
                new_row_counts.append(row_count)
                rows_kept += row_count
                continue
            prefix_rows = min(row_count, rows_remaining)
            if prefix_rows > 0:
                new_handles.append(handle.prefix(prefix_rows))
                new_row_counts.append(prefix_rows)
                rows_kept += prefix_rows
            break
        assert rows_kept == preserve_rows, f"kept {rows_kept} routing rows, expected {preserve_rows}"
        return new_handles, new_row_counts


def _common_prefix_len(left: NDArray[np.int32], right: NDArray[np.int32]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def pack_rollout_traces_for_train_batches(
    traces: Sequence[RolloutTrace],
    *,
    max_pack_length: int,
    global_batch_size: int,
    allow_prefix_sharing: bool = True,
) -> list[Sample]:
    """Pack traces and stamp each packed sample with its source ``trajectory_id``.

    ``trajectory_id`` is the index of the originating trace in ``traces``; downstream
    iterators chunk consecutive trajectory ids into global batches of
    ``global_batch_size`` trajectories so a trajectory's packed samples always land
    in the same gradient update regardless of how many packed samples it split into.
    """
    assert global_batch_size > 0, f"global_batch_size must be positive, got {global_batch_size}"
    assert len(traces) % global_batch_size == 0, f"trace count {len(traces)} must be divisible by {global_batch_size}"

    total_original_trainable_tokens = sum(sum(sum(sample.loss_mask) for sample in trace.turn_samples) for trace in traces)
    assert total_original_trainable_tokens > 0, "Packed traces must contain trainable tokens."

    packed_samples: list[Sample] = []
    for trajectory_id, trace in tqdm(enumerate(traces), desc="Packing rollout traces into training samples", unit="trace"):
        trace_samples = trace.to_packed_samples(
            max_pack_length=max_pack_length,
            allow_prefix_sharing=allow_prefix_sharing,
        )
        for sample in trace_samples:
            sample.trajectory_id = trajectory_id
        packed_samples.extend(trace_samples)

    assert sum(sum(sample.loss_mask) for sample in packed_samples) == total_original_trainable_tokens
    return packed_samples
