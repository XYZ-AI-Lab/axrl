from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from tensordict import TensorDict

from axrl.configs import IGNORE_INDEX
from axrl.data import array_utils

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from axrl.data.generation import TensorHandle
    from axrl.utils.megatron.prefix_tree import PrefixMergeInfo


@dataclass
class Sample:
    input_ids: NDArray[np.int32]
    labels: NDArray[np.int32]
    loss_mask: NDArray[np.bool_]
    attention_mask: NDArray[np.bool_]
    position_ids: NDArray[np.int32]
    reward: float
    reward_baseline: float
    advantage: NDArray[np.float32]
    rollout_logprobs: NDArray[np.float32] | None = None
    old_logprobs: NDArray[np.float32] | None = None
    ref_logprobs: NDArray[np.float32] | None = None  # logprobs from the initial policy (before any updates)
    teacher_logprobs: NDArray[np.float32] | None = None
    old_values: NDArray[np.float32] | None = None
    returns: NDArray[np.float32] | None = None
    turn_index: NDArray[np.int32] | None = None  # label-aligned; assistant turns >= 0, others -1
    turn_reward: NDArray[np.float32] | None = None  # label-aligned; 0 for non-assistant tokens
    # Per-leaf-path routing handles. Outer list indexes leaf paths in
    # ``merge_info.path_to_leaf`` order; inner list is the chain of
    # ``TensorHandle``s whose routing chunks concat into that path's
    # cumulative routing. Flat (non-merged) samples have a single inner
    # list. ``None`` ⇒ R3 off.
    routing_handles_per_path: list[list[TensorHandle]] | None = None
    # Prefix-tree merge metadata. None ⇒ flat trajectory; otherwise the flat fields above
    # hold the trie's DFS-pre-order packed layout for one merged trajectory.
    merge_info: PrefixMergeInfo | None = None
    # Index of the originating RolloutTrace within the model-sync training
    # window. Packed samples from the same trace share the same id; padding
    # samples added by the global-batch iterator use ``-1``.
    trajectory_id: int = -1

    def __post_init__(self) -> None:
        self.input_ids = array_utils.as_i32(self.input_ids)
        self.labels = array_utils.as_i32(self.labels)
        self.loss_mask = array_utils.as_bool(self.loss_mask)
        self.attention_mask = array_utils.as_bool(self.attention_mask)
        self.position_ids = array_utils.as_i32(self.position_ids)
        self.advantage = array_utils.as_f32(self.advantage)
        self.rollout_logprobs = array_utils.optional_as_f32(self.rollout_logprobs)
        self.old_logprobs = array_utils.optional_as_f32(self.old_logprobs)
        self.ref_logprobs = array_utils.optional_as_f32(self.ref_logprobs)
        self.teacher_logprobs = array_utils.optional_as_f32(self.teacher_logprobs)
        self.old_values = array_utils.optional_as_f32(self.old_values)
        self.returns = array_utils.optional_as_f32(self.returns)
        self.turn_index = array_utils.optional_as_i32(self.turn_index)
        self.turn_reward = array_utils.optional_as_f32(self.turn_reward)


class SampleTensorDict(TensorDict):
    @staticmethod
    def from_samples(samples: list[Sample], *, max_length: int | None = None) -> SampleTensorDict:
        return _tensorize_sample_batch(samples, max_length=max_length)


_PER_ROW_NON_TENSOR_KEYS = ("merge_info", "routing_handles_per_path")


def _pad_sample_to(sample: Sample, max_length: int) -> Sample:
    """Right-pad every per-position field of ``sample`` out to ``max_length``.

    Pad token id defaults to 0; padded positions have ``attention_mask=False`` /
    ``loss_mask=False`` so the value is never read by the model or the loss.
    """
    cur = len(sample.input_ids)
    assert cur <= max_length, f"sample length {cur} exceeds max_length {max_length}"
    pad = max_length - cur
    if pad == 0:
        return sample
    return Sample(
        input_ids=np.pad(sample.input_ids, (0, pad), constant_values=0),
        labels=np.pad(sample.labels, (0, pad), constant_values=IGNORE_INDEX),
        loss_mask=np.pad(sample.loss_mask, (0, pad), constant_values=False),
        attention_mask=np.pad(sample.attention_mask, (0, pad), constant_values=False),
        position_ids=np.concatenate([sample.position_ids, np.arange(cur, max_length, dtype=np.int32)]),
        reward=sample.reward,
        reward_baseline=sample.reward_baseline,
        advantage=np.pad(sample.advantage, (0, pad), constant_values=0.0),
        rollout_logprobs=(np.pad(sample.rollout_logprobs, (0, pad), constant_values=0.0) if sample.rollout_logprobs is not None else None),
        old_logprobs=(np.pad(sample.old_logprobs, (0, pad), constant_values=0.0) if sample.old_logprobs is not None else None),
        ref_logprobs=(np.pad(sample.ref_logprobs, (0, pad), constant_values=0.0) if sample.ref_logprobs is not None else None),
        teacher_logprobs=(np.pad(sample.teacher_logprobs, (0, pad), constant_values=0.0) if sample.teacher_logprobs is not None else None),
        old_values=(np.pad(sample.old_values, (0, pad), constant_values=0.0) if sample.old_values is not None else None),
        returns=(np.pad(sample.returns, (0, pad), constant_values=0.0) if sample.returns is not None else None),
        turn_index=(np.pad(sample.turn_index, (0, pad), constant_values=-1) if sample.turn_index is not None else None),
        turn_reward=(np.pad(sample.turn_reward, (0, pad), constant_values=0.0) if sample.turn_reward is not None else None),
        routing_handles_per_path=sample.routing_handles_per_path,
        merge_info=sample.merge_info,
        trajectory_id=sample.trajectory_id,
    )


def _tensorize_sample_batch(samples: list[Sample], *, max_length: int | None = None) -> SampleTensorDict:
    assert samples
    if max_length is None:
        max_length = max(len(sample.input_ids) for sample in samples)
    samples = [_pad_sample_to(s, max_length) for s in samples]
    batch: dict[str, torch.Tensor] = {
        "index": torch.arange(len(samples), dtype=torch.long),
        "input_ids": torch.from_numpy(np.stack([s.input_ids for s in samples])).long(),
        "labels": torch.from_numpy(np.stack([s.labels for s in samples])).long(),
        "loss_mask": torch.from_numpy(np.stack([s.loss_mask for s in samples])),
        "attention_mask": torch.from_numpy(np.stack([s.attention_mask for s in samples])),
        "position_ids": torch.from_numpy(np.stack([s.position_ids for s in samples])).long(),
        "reward": torch.from_numpy(np.asarray([s.reward for s in samples], dtype=np.float32)),
        "reward_baseline": torch.from_numpy(np.asarray([s.reward_baseline for s in samples], dtype=np.float32)),
        "advantage": torch.from_numpy(np.stack([s.advantage for s in samples])),
        "trajectory_id": torch.tensor([s.trajectory_id for s in samples], dtype=torch.long),
    }

    if all(s.turn_index is not None for s in samples):
        batch["turn_index"] = torch.from_numpy(np.stack([s.turn_index for s in samples if s.turn_index is not None])).long()

    if all(s.turn_reward is not None for s in samples):
        batch["turn_reward"] = torch.from_numpy(np.stack([s.turn_reward for s in samples if s.turn_reward is not None]))

    if all(sample.rollout_logprobs is not None for sample in samples):
        batch["rollout_logprobs"] = torch.from_numpy(np.stack([sample.rollout_logprobs for sample in samples if sample.rollout_logprobs is not None]))

    if all(sample.old_logprobs is not None for sample in samples):
        batch["old_logprobs"] = torch.from_numpy(np.stack([sample.old_logprobs for sample in samples if sample.old_logprobs is not None]))

    if all(sample.ref_logprobs is not None for sample in samples):
        batch["ref_logprobs"] = torch.from_numpy(np.stack([sample.ref_logprobs for sample in samples if sample.ref_logprobs is not None]))

    if all(sample.teacher_logprobs is not None for sample in samples):
        batch["teacher_logprobs"] = torch.from_numpy(np.stack([sample.teacher_logprobs for sample in samples if sample.teacher_logprobs is not None]))

    if all(sample.old_values is not None for sample in samples):
        batch["old_values"] = torch.from_numpy(np.stack([sample.old_values for sample in samples if sample.old_values is not None]))

    if all(sample.returns is not None for sample in samples):
        batch["returns"] = torch.from_numpy(np.stack([sample.returns for sample in samples if sample.returns is not None]))

    td = SampleTensorDict(batch, batch_size=len(samples))
    if any(s.merge_info is not None for s in samples):
        td.set_non_tensor("merge_info", [s.merge_info for s in samples])
    if any(s.routing_handles_per_path is not None for s in samples):
        td.set_non_tensor("routing_handles_per_path", [s.routing_handles_per_path for s in samples])
    return td


def samples_from_tensor_dict(sd: SampleTensorDict) -> list[Sample]:
    """Inverse of :func:`_tensorize_sample_batch`.

    Recovers a flat ``list[Sample]`` from a ``SampleTensorDict``. Optional
    fields are propagated when present; ``merge_info`` is read from the
    per-row list stored via ``set_non_tensor`` when available.
    """
    from axrl.utils.megatron.prefix_tree import PrefixMergeInfo

    batch_size = sd.shape[0]
    input_ids: torch.Tensor = sd["input_ids"]  # type: ignore[assignment]
    labels: torch.Tensor = sd["labels"]  # type: ignore[assignment]
    loss_mask: torch.Tensor = sd["loss_mask"]  # type: ignore[assignment]
    attention_mask: torch.Tensor = sd["attention_mask"]  # type: ignore[assignment]
    position_ids: torch.Tensor = sd["position_ids"]  # type: ignore[assignment]
    reward: torch.Tensor = sd["reward"]  # type: ignore[assignment]
    reward_baseline: torch.Tensor = sd["reward_baseline"]  # type: ignore[assignment]
    advantage: torch.Tensor = sd["advantage"]  # type: ignore[assignment]

    rollout_logprobs = sd.get("rollout_logprobs", None)
    old_logprobs = sd.get("old_logprobs", None)
    ref_logprobs = sd.get("ref_logprobs", None)
    teacher_logprobs = sd.get("teacher_logprobs", None)
    old_values = sd.get("old_values", None)
    returns = sd.get("returns", None)
    turn_index = sd.get("turn_index", None)
    turn_reward = sd.get("turn_reward", None)
    trajectory_id = sd.get("trajectory_id", None)
    merge_info_list: list[PrefixMergeInfo | None] | None = sd.get_non_tensor("merge_info", default=None)
    routing_handles_per_path_list: list[list[list[TensorHandle]] | None] | None = sd.get_non_tensor("routing_handles_per_path", default=None)

    def tensor_row_to_i32(tensor: torch.Tensor, row_idx: int) -> NDArray[np.int32]:
        return tensor[row_idx].detach().cpu().numpy().astype(np.int32, copy=True)

    def tensor_row_to_f32(tensor: torch.Tensor, row_idx: int) -> NDArray[np.float32]:
        return tensor[row_idx].detach().cpu().numpy().astype(np.float32, copy=True)

    def tensor_row_to_bool(tensor: torch.Tensor, row_idx: int) -> NDArray[np.bool_]:
        return tensor[row_idx].detach().cpu().numpy().astype(np.bool_, copy=True)

    out: list[Sample] = []
    for i in range(batch_size):
        mi: PrefixMergeInfo | None = None
        if merge_info_list is not None:
            mi_value = merge_info_list[i]
            if isinstance(mi_value, PrefixMergeInfo):
                mi = mi_value
        routing_handles_per_path: list[list[TensorHandle]] | None = None
        if routing_handles_per_path_list is not None:
            row_per_path = routing_handles_per_path_list[i]
            if row_per_path is not None:
                routing_handles_per_path = [list(path_handles) for path_handles in row_per_path]
        out.append(
            Sample(
                input_ids=tensor_row_to_i32(input_ids, i),
                labels=tensor_row_to_i32(labels, i),
                loss_mask=tensor_row_to_bool(loss_mask, i),
                attention_mask=tensor_row_to_bool(attention_mask, i),
                position_ids=tensor_row_to_i32(position_ids, i),
                reward=float(reward[i].item()),
                reward_baseline=float(reward_baseline[i].item()),
                advantage=tensor_row_to_f32(advantage, i),
                rollout_logprobs=tensor_row_to_f32(rollout_logprobs, i) if rollout_logprobs is not None else None,
                old_logprobs=tensor_row_to_f32(old_logprobs, i) if old_logprobs is not None else None,
                ref_logprobs=tensor_row_to_f32(ref_logprobs, i) if ref_logprobs is not None else None,
                teacher_logprobs=tensor_row_to_f32(teacher_logprobs, i) if teacher_logprobs is not None else None,
                old_values=tensor_row_to_f32(old_values, i) if old_values is not None else None,
                returns=tensor_row_to_f32(returns, i) if returns is not None else None,
                turn_index=tensor_row_to_i32(turn_index, i) if turn_index is not None else None,
                turn_reward=tensor_row_to_f32(turn_reward, i) if turn_reward is not None else None,
                routing_handles_per_path=routing_handles_per_path,
                merge_info=mi,
                trajectory_id=(int(trajectory_id[i].item()) if trajectory_id is not None else -1),
            )
        )
    return out


def _make_padding_sample(
    *,
    seq_len: int,
    padding_sample_length: int,
    tensor_keys: set[str],
    needs_merge_info: bool,
    padding_routing_handle: TensorHandle | None,
) -> Sample:
    assert seq_len > 0, f"padding sample needs a positive seq_len, got {seq_len}"
    assert 0 < padding_sample_length <= seq_len, f"padding_sample_length must be in (0, {seq_len}], got {padding_sample_length}"
    active_len = padding_sample_length
    sample = Sample(
        input_ids=np.zeros(active_len, dtype=np.int32),
        labels=np.full(active_len, IGNORE_INDEX, dtype=np.int32),
        loss_mask=np.zeros(active_len, dtype=np.bool_),
        attention_mask=np.ones(active_len, dtype=np.bool_),
        position_ids=np.arange(active_len, dtype=np.int32),
        reward=0.0,
        reward_baseline=0.0,
        advantage=np.zeros(active_len, dtype=np.float32),
        routing_handles_per_path=[[padding_routing_handle]] if padding_routing_handle is not None else None,
        trajectory_id=-1,
    )

    if needs_merge_info:
        from axrl.utils.megatron.prefix_tree import merge_trajectory_samples

        sample = merge_trajectory_samples([sample], align_size=1)
        assert sample.merge_info is not None

    if "rollout_logprobs" in tensor_keys:
        sample.rollout_logprobs = np.zeros(len(sample.input_ids), dtype=np.float32)
    if "old_logprobs" in tensor_keys:
        sample.old_logprobs = np.zeros(len(sample.input_ids), dtype=np.float32)
    if "ref_logprobs" in tensor_keys:
        sample.ref_logprobs = np.zeros(len(sample.input_ids), dtype=np.float32)
    if "teacher_logprobs" in tensor_keys:
        sample.teacher_logprobs = np.zeros(len(sample.input_ids), dtype=np.float32)
    if "old_values" in tensor_keys:
        sample.old_values = np.zeros(len(sample.input_ids), dtype=np.float32)
    if "returns" in tensor_keys:
        sample.returns = np.zeros(len(sample.input_ids), dtype=np.float32)
    if "turn_index" in tensor_keys:
        sample.turn_index = np.full(len(sample.input_ids), -1, dtype=np.int32)
    if "turn_reward" in tensor_keys:
        sample.turn_reward = np.zeros(len(sample.input_ids), dtype=np.float32)
    return _pad_sample_to(sample, seq_len)


def _make_padding_tensor_dict(
    samples: SampleTensorDict,
    pad_count: int,
    *,
    padding_sample_length: int,
    padding_routing_handle: TensorHandle | None,
) -> SampleTensorDict:
    seq_len = int(samples["input_ids"].shape[1])
    tensor_keys = {key for key, value in samples.items() if isinstance(value, torch.Tensor)}
    has_routing_info = "routing_handles_per_path" in samples.keys()  # noqa: SIM118 - tensordict semantics
    padding_routing_handle = padding_routing_handle if has_routing_info else None

    padding_sample = _make_padding_sample(
        seq_len=seq_len,
        padding_sample_length=padding_sample_length,
        tensor_keys=tensor_keys,
        needs_merge_info="merge_info" in samples.keys(),  # noqa: SIM118 - tensordict semantics
        padding_routing_handle=padding_routing_handle,
    )
    padding = SampleTensorDict.from_samples([padding_sample] * pad_count, max_length=seq_len)
    # Padding rows use ``index = -1`` as a sentinel so per-batch padding (added
    # by the trajectory-grouped iterator) never collides with real-sample indices.
    padding["index"] = torch.full((pad_count,), -1, dtype=padding["index"].dtype)
    if has_routing_info and padding_routing_handle is None:
        padding.set_non_tensor("routing_handles_per_path", [None] * pad_count)
    return padding


def _concat_sample_tensor_dicts(sample_batches: Sequence[SampleTensorDict]) -> SampleTensorDict:
    assert sample_batches, "_concat_sample_tensor_dicts requires at least one batch"
    if len(sample_batches) == 1:
        return sample_batches[0]

    first_batch = sample_batches[0]
    tensor_keys = [key for key, value in first_batch.items() if isinstance(value, torch.Tensor)]
    device = first_batch["input_ids"].device
    selected_batches: list[SampleTensorDict] = []
    for batch in sample_batches:
        batch_tensor_keys = {key for key, value in batch.items() if isinstance(value, torch.Tensor)}
        assert batch_tensor_keys == set(tensor_keys), f"tensor keys differ across sample batches: {batch_tensor_keys} vs {set(tensor_keys)}"
        for key in tensor_keys:
            assert batch[key].shape[1:] == first_batch[key].shape[1:], (
                f"tensor shape differs for key {key!r}: {batch[key].shape} vs {first_batch[key].shape}"
            )
        selected_batches.append(batch.to(device).select(*tensor_keys))

    concatenated = cast(
        "SampleTensorDict",
        TensorDict.cat(selected_batches, dim=0),
    )
    for key in _PER_ROW_NON_TENSOR_KEYS:
        if any(key in batch.keys() for batch in sample_batches):  # noqa: SIM118 - tensordict semantics
            values = []
            for batch in sample_batches:
                if key in batch.keys():  # noqa: SIM118 - tensordict semantics
                    values.extend(list(batch.get_non_tensor(key)))
                else:
                    values.extend([None] * len(batch))
            concatenated.set_non_tensor(key, values)
    return concatenated


def pad_sample_tensor_dict_to_multiple(
    samples: SampleTensorDict,
    multiple: int,
    *,
    padding_sample_length: int,
    padding_routing_handle: TensorHandle | None = None,
) -> tuple[SampleTensorDict, int]:
    """Pad ``samples`` so the row count is a multiple of ``multiple`` with zero-loss rows.

    Padding rows are marked with ``index == -1`` so callers can distinguish
    them from real samples after later reordering. Used by the trajectory-grouped
    global-batch iterator and by routing-materialiser tests that need fixed-size
    padded batches.
    """
    assert multiple > 0, f"multiple must be positive, got {multiple}"
    original_len = len(samples)
    if original_len == 0 or original_len % multiple == 0:
        return samples, original_len

    padding = _make_padding_tensor_dict(
        samples,
        multiple - (original_len % multiple),
        padding_sample_length=padding_sample_length,
        padding_routing_handle=padding_routing_handle,
    )
    return _concat_sample_tensor_dicts([samples, padding]), original_len


def remove_padding_from_sample_tensor_dict(samples: SampleTensorDict, original_len: int) -> SampleTensorDict:
    """Drop trailing padding rows (keeps the first ``original_len`` rows)."""
    assert 0 <= original_len <= len(samples), f"original_len must be in [0, {len(samples)}], got {original_len}"
    if original_len == len(samples):
        return samples

    tensor_keys = [key for key, value in samples.items() if isinstance(value, torch.Tensor)]
    trimmed = cast("SampleTensorDict", samples.select(*tensor_keys)[:original_len])
    for key in _PER_ROW_NON_TENSOR_KEYS:
        if key in samples.keys():  # noqa: SIM118 - tensordict semantics
            trimmed.set_non_tensor(key, list(samples.get_non_tensor(key))[:original_len])
    return trimmed


def select_sample_tensor_dict_rows(samples: SampleTensorDict, row_indices: list[int]) -> SampleTensorDict:
    """Build a new ``SampleTensorDict`` containing the rows at ``row_indices`` (in order).

    Both tensor columns and the per-row non-tensor lists are reordered consistently.
    """
    n = len(samples)
    assert all(0 <= i < n for i in row_indices), f"row_indices out of range for batch of size {n}"
    tensor_keys = [key for key, value in samples.items() if isinstance(value, torch.Tensor)]
    if len(row_indices) == 0:
        empty = cast("SampleTensorDict", samples.select(*tensor_keys)[:0])
        for key in _PER_ROW_NON_TENSOR_KEYS:
            if key in samples.keys():  # noqa: SIM118 - tensordict semantics
                empty.set_non_tensor(key, [])
        return empty
    idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=samples["input_ids"].device)
    selected_tensors = {key: samples[key].index_select(0, idx_tensor) for key in tensor_keys}
    out = cast("SampleTensorDict", SampleTensorDict(selected_tensors, batch_size=len(row_indices)))
    for key in _PER_ROW_NON_TENSOR_KEYS:
        if key in samples.keys():  # noqa: SIM118 - tensordict semantics
            full_list = list(samples.get_non_tensor(key))
            out.set_non_tensor(key, [full_list[i] for i in row_indices])
    return out


def _get_token_aligned_mask(sample: Sample) -> list[bool]:
    """Reconstruct the token-aligned loss mask from a sample.

    Sample.loss_mask is left-shifted to align with labels. This function
    undoes that shift to get a mask aligned with input_ids.
    """
    return [False, *sample.loss_mask[:-1].tolist()]


def get_prompt_ids(sample: Sample) -> list[int]:
    """Extract the prompt token IDs from a sample.

    Returns all leading tokens that are not included in the loss computation,
    i.e., tokens before the first response token.
    """
    token_aligned_mask = _get_token_aligned_mask(sample)

    prompt_ids = []
    for token_id, in_loss in zip(sample.input_ids, token_aligned_mask, strict=True):
        if in_loss:
            break
        prompt_ids.append(int(token_id))
    return prompt_ids


def get_response_ids(sample: Sample) -> list[int]:
    """Extract the response token IDs from a sample.

    Returns all tokens that are included in the loss computation,
    i.e., the model-generated response tokens (excluding padding).
    """
    token_aligned_mask = _get_token_aligned_mask(sample)

    response_ids = []
    for token_id, in_loss in zip(sample.input_ids, token_aligned_mask, strict=True):
        if in_loss:
            response_ids.append(int(token_id))
    return response_ids


def collect_unique_handles_from_samples(samples: list[Sample]) -> list[TensorHandle]:
    """Dedup handles across a batch of samples, preserving first-seen order.

    Used by the controller to build the delete list at step end.
    """
    seen: set[TensorHandle] = set()
    ordered: list[TensorHandle] = []
    for sample in samples:
        if not sample.routing_handles_per_path:
            continue
        for path_handles in sample.routing_handles_per_path:
            for handle in path_handles:
                if handle not in seen:
                    seen.add(handle)
                    ordered.append(handle)
    return ordered


def collect_unique_handles_from_sample_tensor_dict(samples: SampleTensorDict) -> list[TensorHandle]:
    """Dedup routing handles from a tensorized sample batch."""
    if "routing_handles_per_path" not in samples.keys():  # noqa: SIM118 - tensordict semantics
        return []
    rows = samples.get_non_tensor("routing_handles_per_path", default=None)
    if rows is None:
        return []

    seen: set[TensorHandle] = set()
    ordered: list[TensorHandle] = []
    for row_handles in rows:
        if not row_handles:
            continue
        for path_handles in row_handles:
            for handle in path_handles:
                if handle not in seen:
                    seen.add(handle)
                    ordered.append(handle)
    return ordered
