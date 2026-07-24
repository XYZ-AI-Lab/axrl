"""Trajectory-grouped global-batch iteration over packed samples.

Replaces the legacy ``DistributedSampler``-based dataloader for the training and
forward-only paths. Packed samples carrying the same ``trajectory_id`` always land
in the same global batch so that the number of gradient updates over a training
window is independent of how many packed samples each trajectory split into.

Per-batch layout:

1. The trajectories in the training window are chunked into
   ``num_global_batches = num_trajectories // global_batch_size`` contiguous groups
   of ``global_batch_size`` trajectories. Each group is one global batch.
2. The packed samples for one global batch are padded with zero-loss padding
   samples so the total count is divisible by ``dp_size``.
3. The padded samples (real + padding) are shuffled together with a deterministic
   per-batch seed when ``shuffle=True``.
4. Samples are distributed round-robin across ``dp_size`` ranks; rank ``r``
   receives positions ``[r, r + dp_size, r + 2 * dp_size, ...]``.

For the forward-only path (``shuffle=False``, ``dp_rank=0`` collecting outputs by
``index``), the same layout is produced deterministically so logprob results can
be matched back to the original sample positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from axrl.data.sample import (
    SampleTensorDict,
    _concat_sample_tensor_dicts,
    _make_padding_tensor_dict,
    select_sample_tensor_dict_rows,
)

if TYPE_CHECKING:
    from axrl.data.generation import TensorHandle


def assert_trajectory_count_divisible(samples: SampleTensorDict, global_batch_size: int) -> int:
    """Training-side invariant: total trajectory count must equal an exact multiple of ``global_batch_size``.

    The iterator itself accepts non-divisible counts (forward-only callers
    benefit from a smaller trailing batch). Training callers MUST enforce
    divisibility so the gradient-update count matches
    ``model_sync_every_n_global_updates`` from the controller.

    Returns the number of trajectories.
    """
    trajectory_id_tensor = samples.get("trajectory_id", None)
    assert isinstance(trajectory_id_tensor, torch.Tensor), "training samples must carry trajectory_id"
    assert int(trajectory_id_tensor.min().item()) >= 0, (
        "training samples must have non-negative trajectory_id; pack with pack_rollout_traces_for_train_batches."
    )
    num_trajectories = int(trajectory_id_tensor.max().item()) + 1
    assert num_trajectories % global_batch_size == 0, (
        f"num_trajectories ({num_trajectories}) must be divisible by global_batch_size "
        f"({global_batch_size}) for training to produce a deterministic number of gradient updates."
    )
    return num_trajectories


def _group_row_indices_by_global_batch(
    trajectory_ids: list[int],
    *,
    global_batch_size: int,
) -> list[list[int]]:
    """Return, for each global batch, the row indices that belong to it.

    ``trajectory_ids`` must use the contiguous id range ``0..num_trajectories-1``
    (padding samples with ``trajectory_id == -1`` are not expected here — padding
    is added by the iterator itself, after grouping). If ``num_trajectories`` is
    not a multiple of ``global_batch_size``, the trailing global batch holds the
    remainder trajectories (``num_trajectories % global_batch_size`` of them).
    Training callers ensure exact divisibility upstream; forward-only callers
    (compute_logprobs / eval) may pass any sample count.
    """
    assert global_batch_size > 0, f"global_batch_size must be positive, got {global_batch_size}"
    if not trajectory_ids:
        return []
    num_trajectories = max(trajectory_ids) + 1
    assert min(trajectory_ids) >= 0, "padding samples (trajectory_id == -1) are not expected at iterator input"
    assert set(trajectory_ids) == set(range(num_trajectories)), (
        f"trajectory_id range must be contiguous 0..{num_trajectories - 1}, got {sorted(set(trajectory_ids))}"
    )

    num_global_batches = (num_trajectories + global_batch_size - 1) // global_batch_size
    grouped: list[list[int]] = [[] for _ in range(num_global_batches)]
    for row_idx, traj_id in enumerate(trajectory_ids):
        grouped[traj_id // global_batch_size].append(row_idx)
    return grouped


def _pad_to_dp_multiple(
    batch: SampleTensorDict,
    *,
    dp_size: int,
    padding_sample_length: int,
    padding_routing_handle: TensorHandle | None,
) -> SampleTensorDict:
    """Pad ``batch`` with zero-loss samples so its row count is a multiple of ``dp_size``."""
    n = len(batch)
    assert n > 0, "global batch must contain at least one packed sample"
    remainder = n % dp_size
    if remainder == 0:
        return batch
    pad_count = dp_size - remainder
    padding = _make_padding_tensor_dict(
        batch,
        pad_count,
        padding_sample_length=padding_sample_length,
        padding_routing_handle=padding_routing_handle,
    )
    return _concat_sample_tensor_dicts([batch, padding])


def build_global_batches_for_dp_rank(
    samples: SampleTensorDict,
    *,
    global_batch_size: int,
    dp_size: int,
    dp_rank: int,
    padding_sample_length: int,
    padding_routing_handle: TensorHandle | None,
    shuffle: bool,
    shuffle_seed: int,
) -> list[SampleTensorDict]:
    """Group ``samples`` by trajectory and return this dp rank's slice for each global batch.

    Returns one :class:`SampleTensorDict` per global batch. Each returned batch has
    ``len(global_batch_after_padding) // dp_size`` rows assigned to the current
    ``dp_rank`` (round-robin distribution after the per-batch shuffle).
    """
    assert 0 <= dp_rank < dp_size, f"dp_rank {dp_rank} must be in [0, {dp_size})"
    trajectory_ids_tensor = samples.get("trajectory_id", None)
    assert isinstance(trajectory_ids_tensor, torch.Tensor), (
        "SampleTensorDict is missing the 'trajectory_id' column; pack rollout traces with pack_rollout_traces_for_train_batches before training."
    )
    trajectory_ids: list[int] = trajectory_ids_tensor.tolist()
    # Convenience for forward-only callers that pass raw ``Sample``s without
    # trajectory grouping (e.g. HF↔Mcore logprob consistency tests, eval):
    # treat each sample as its own trajectory.
    if all(tid == -1 for tid in trajectory_ids):
        trajectory_ids = list(range(len(trajectory_ids)))

    grouped_indices = _group_row_indices_by_global_batch(
        trajectory_ids,
        global_batch_size=global_batch_size,
    )

    per_batch_dp_slices: list[SampleTensorDict] = []
    for gb_idx, row_indices in enumerate(grouped_indices):
        assert row_indices, f"global batch {gb_idx} produced zero packed samples"
        batch = select_sample_tensor_dict_rows(samples, row_indices)
        batch = _pad_to_dp_multiple(
            batch,
            dp_size=dp_size,
            padding_sample_length=padding_sample_length,
            padding_routing_handle=padding_routing_handle,
        )
        total = len(batch)
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(shuffle_seed + gb_idx)
            perm = torch.randperm(total, generator=generator).tolist()
        else:
            perm = list(range(total))
        # Round-robin across DP ranks so per-shard token counts stay roughly balanced.
        local_positions = perm[dp_rank::dp_size]
        per_batch_dp_slices.append(select_sample_tensor_dict_rows(batch, local_positions))
    return per_batch_dp_slices
