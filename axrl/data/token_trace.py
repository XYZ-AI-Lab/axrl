from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from axrl.configs import IGNORE_INDEX
from axrl.data import array_utils
from axrl.data.generation import TensorHandle
from axrl.data.sample import Sample

TokenInfoType = Literal["init", "assistant", "assistant_boundary", "tool_result", "user"]


@dataclass(init=False)
class TokenInfo:
    """One contiguous chunk with a single role/type."""

    tokens: NDArray[np.int32]
    token_type: TokenInfoType
    compacted_to_placeholder: bool = False
    logprobs: NDArray[np.float32] = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def __init__(
        self,
        tokens: NDArray[np.int32],
        token_type: TokenInfoType,
        *,
        compacted_to_placeholder: bool = False,
        logprobs: NDArray[np.float32] | None = None,
    ) -> None:
        self.tokens = array_utils.as_i32(tokens).copy()
        self.token_type = token_type
        self.compacted_to_placeholder = compacted_to_placeholder
        self.logprobs = np.empty(0, dtype=np.float32) if logprobs is None else array_utils.as_f32(logprobs).copy()


@dataclass
class TokenTrace:
    """Multi-turn token stream as a list of typed chunks; flat fields are derived views."""

    token_infos: list[TokenInfo] = field(default_factory=list)
    # One ``TensorHandle`` per assistant turn when R3 is on, in append order.
    # ``token_info_index_per_handle[i]`` is the ``token_infos`` index of the assistant
    # chunk that handle ``routing_handles[i]`` was captured for; needed because
    # compaction drops stale handles (the leading-prefix mapping otherwise
    # implied by handle index would no longer hold).
    routing_handles: list[TensorHandle] = field(default_factory=list)
    token_info_index_per_handle: list[int] = field(default_factory=list)
    # Number of first-axis routing rows contributed by each handle in
    # ``routing_handles``. This lets compaction keep a valid prefix slice from
    # the first stale handle instead of recapturing the entire suffix.
    routing_row_count_per_handle: list[int] = field(default_factory=list)
    _turn_rewards: dict[int, float] = field(default_factory=dict, repr=False)

    @property
    def token_ids(self) -> NDArray[np.int32]:
        if not self.token_infos:
            return np.empty(0, dtype=np.int32)
        return np.concatenate([token_info.tokens for token_info in self.token_infos])

    @property
    def token_count(self) -> int:
        return sum(len(token_info.tokens) for token_info in self.token_infos)

    @property
    def token_logprobs(self) -> NDArray[np.float32]:
        if not self.token_infos:
            return np.empty(0, dtype=np.float32)
        chunks: list[NDArray[np.float32]] = []
        for token_info in self.token_infos:
            if len(token_info.logprobs) > 0:
                assert len(token_info.logprobs) == len(token_info.tokens)
                chunks.append(token_info.logprobs)
            else:
                chunks.append(np.zeros(len(token_info.tokens), dtype=np.float32))
        return np.concatenate(chunks)

    @property
    def loss_mask(self) -> NDArray[np.bool_]:
        if not self.token_infos:
            return np.empty(0, dtype=np.bool_)
        return np.concatenate(
            [np.full(len(token_info.tokens), token_info.token_type == "assistant", dtype=np.bool_) for token_info in self.token_infos]
        )

    @property
    def turn_index(self) -> NDArray[np.int32]:
        chunks: list[NDArray[np.int32]] = []
        assistant_turns_seen = 0
        for token_info in self.token_infos:
            if token_info.token_type == "assistant":
                chunks.append(np.full(len(token_info.tokens), assistant_turns_seen, dtype=np.int32))
                assistant_turns_seen += 1
            else:
                chunks.append(np.full(len(token_info.tokens), -1, dtype=np.int32))
        if not chunks:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(chunks)

    @property
    def num_assistant_turns(self) -> int:
        return sum(1 for token_info in self.token_infos if token_info.token_type == "assistant")

    @property
    def captured_routing_rows(self) -> int:
        return sum(self.routing_row_count_per_handle)

    def extend_tokens(
        self,
        new_tokens: NDArray[np.int32],
        logprobs: NDArray[np.float32] | None = None,
        *,
        token_type: TokenInfoType,
        routing_handle: TensorHandle | None = None,
        routing_row_count: int | None = None,
    ) -> None:
        if logprobs is None:
            logprobs = np.zeros(len(new_tokens), dtype=np.float32)
        else:
            logprobs = array_utils.as_f32(logprobs)
        assert len(logprobs) == len(new_tokens), f"Expected {len(new_tokens)} logprobs, got {len(logprobs)}"
        if token_type == "init":
            assert not self.token_infos, "token_type='init' is only allowed for the first chunk"
        else:
            assert self.token_infos, "first chunk must be token_type='init'"
        if routing_handle is not None:
            assert token_type == "assistant", "routing_handle is only meaningful on assistant chunks"
            if routing_row_count is None:
                routing_row_count = self.new_routing_row_count(len(new_tokens))
            self.routing_handles.append(routing_handle)
            # The handle's asst chunk is the one we're about to append.
            self.token_info_index_per_handle.append(len(self.token_infos))
            self.routing_row_count_per_handle.append(routing_row_count)
        self.token_infos.append(TokenInfo(tokens=new_tokens, token_type=token_type, logprobs=logprobs))

    def new_routing_row_count(self, new_token_count: int, *, prior_routing_rows: int | None = None) -> int:
        assert new_token_count >= 0, f"new_token_count must be non-negative, got {new_token_count}"
        prior_rows = sum(self.routing_row_count_per_handle) if prior_routing_rows is None else prior_routing_rows
        assert prior_rows >= 0, f"prior routing rows must be non-negative, got {prior_rows}"
        total_after = self.token_count + new_token_count
        routing_row_count = total_after - 1 - prior_rows
        assert routing_row_count >= 0, (
            f"routing row count would be negative: total_after={total_after}, prior_routing_rows={prior_rows}, new_token_count={new_token_count}"
        )
        return routing_row_count

    def set_turn_reward(self, turn_index: int, reward: float) -> None:
        """Assign reward to assistant turn ``turn_index`` (0-based, only counts assistant chunks)."""
        assert 0 <= turn_index < self.num_assistant_turns, f"turn_index {turn_index} out of range [0, {self.num_assistant_turns})"
        self._turn_rewards[turn_index] = reward

    def _shift_left(self, values: NDArray[Any], padding_value: Any) -> NDArray[Any]:
        if len(values) == 0:
            return values.copy()
        return np.concatenate([values[1:], np.asarray([padding_value], dtype=values.dtype)])

    def to_last_turn_sample(self, max_length: int, pad_token_id: int) -> Sample:
        """Build a Sample where only the LAST assistant chunk is loss-bearing.

        Used by RolloutTrace to emit one per-turn Sample per assistant message;
        earlier turns already produced their own samples.
        """
        assert self.token_infos, "to_last_turn_sample requires at least one chunk"
        last_chunk = self.token_infos[-1]
        assert last_chunk.token_type == "assistant", f"last chunk must be assistant, got {last_chunk.token_type!r}"

        input_ids = self.token_ids
        logprobs = self.token_logprobs
        seq_length = len(input_ids)
        assert seq_length > 0, f"Invalid input length: {seq_length}"
        assert len(logprobs) == seq_length
        assert seq_length <= max_length, (
            f"TokenTrace length {seq_length} exceeds max_length {max_length}. Rollout env must terminate before the trace overflows."
        )

        loss_mask = np.zeros(seq_length, dtype=np.bool_)
        turn_index = np.full(seq_length, -1, dtype=np.int32)
        turn_reward = np.zeros(seq_length, dtype=np.float32)
        last_start = seq_length - len(last_chunk.tokens)
        if len(last_chunk.tokens) > 0:
            loss_mask[last_start:] = True
            turn_index[last_start:] = 0
            turn_reward[last_start:] = np.float32(self._turn_rewards.get(0, 0.0))
        assert not bool(loss_mask[0]), "TokenTrace must start with prompt/context tokens excluded from loss"

        # Left-shift to align with labels: at position t these describe input_ids[t+1].
        labels = self._shift_left(input_ids, IGNORE_INDEX)
        logprobs = self._shift_left(logprobs, 0.0)
        loss_mask = self._shift_left(loss_mask, padding_value=False)
        turn_index = self._shift_left(turn_index, padding_value=-1)
        turn_reward = self._shift_left(turn_reward, padding_value=0.0)

        if seq_length < max_length:
            padding_length = max_length - seq_length
            input_ids = np.pad(input_ids, (0, padding_length), constant_values=pad_token_id)
            logprobs = np.pad(logprobs, (0, padding_length), constant_values=0.0)
            labels = np.pad(labels, (0, padding_length), constant_values=IGNORE_INDEX)
            loss_mask = np.pad(loss_mask, (0, padding_length), constant_values=False)
            turn_index = np.pad(turn_index, (0, padding_length), constant_values=-1)
            turn_reward = np.pad(turn_reward, (0, padding_length), constant_values=0.0)

        attention_mask = np.concatenate(
            [np.ones(seq_length, dtype=np.bool_), np.zeros(max_length - seq_length, dtype=np.bool_)],
        )
        position_ids = np.arange(max_length, dtype=np.int32)
        return Sample(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=loss_mask,
            rollout_logprobs=logprobs,
            reward=0.0,
            reward_baseline=0.0,
            advantage=np.zeros(max_length, dtype=np.float32),
            turn_index=turn_index,
            turn_reward=turn_reward,
            routing_handles_per_path=[list(self.routing_handles)] if self.routing_handles else None,
        )

    def to_sample(self, max_length: int, pad_token_id: int) -> Sample:
        input_ids = self.token_ids
        logprobs = self.token_logprobs
        loss_mask = self.loss_mask
        turn_index = self.turn_index

        assert len(input_ids) > 0, f"Invalid input length: {len(input_ids)}"
        assert len(input_ids) == len(logprobs)
        assert len(input_ids) == len(loss_mask)
        assert len(input_ids) == len(turn_index)
        assert not bool(loss_mask[0]), "TokenTrace must start with prompt/context tokens excluded from loss"

        turn_reward = np.asarray([self._turn_rewards.get(int(idx), 0.0) if idx >= 0 else 0.0 for idx in turn_index], dtype=np.float32)

        assert len(input_ids) <= max_length, (
            f"TokenTrace length {len(input_ids)} exceeds max_length {max_length}. Rollout env must terminate before the trace overflows."
        )

        # Left-shift to align with labels: at position t these describe input_ids[t+1].
        labels = self._shift_left(input_ids, IGNORE_INDEX)
        logprobs = self._shift_left(logprobs, 0.0)
        loss_mask = self._shift_left(loss_mask, padding_value=False)
        turn_index = self._shift_left(turn_index, padding_value=-1)
        turn_reward = self._shift_left(turn_reward, padding_value=0.0)

        seq_length = len(input_ids)
        if seq_length < max_length:
            padding_length = max_length - seq_length
            input_ids = np.pad(input_ids, (0, padding_length), constant_values=pad_token_id)
            logprobs = np.pad(logprobs, (0, padding_length), constant_values=0.0)
            labels = np.pad(labels, (0, padding_length), constant_values=IGNORE_INDEX)
            loss_mask = np.pad(loss_mask, (0, padding_length), constant_values=False)
            turn_index = np.pad(turn_index, (0, padding_length), constant_values=-1)
            turn_reward = np.pad(turn_reward, (0, padding_length), constant_values=0.0)

        attention_mask = np.concatenate(
            [np.ones(seq_length, dtype=np.bool_), np.zeros(max_length - seq_length, dtype=np.bool_)],
        )
        position_ids = np.arange(max_length, dtype=np.int32)
        return Sample(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=loss_mask,
            rollout_logprobs=logprobs,
            reward=0.0,
            reward_baseline=0.0,
            advantage=np.zeros(max_length, dtype=np.float32),
            turn_index=turn_index,
            turn_reward=turn_reward,
            routing_handles_per_path=[list(self.routing_handles)] if self.routing_handles else None,
        )
