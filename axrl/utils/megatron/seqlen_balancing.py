# Adapted from https://github.com/volcengine/verl/blob/bbd1288353d1349d4ce2a8e1d9e88ed63b9a0ab6/verl/utils/seqlen_balancing.py

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import heapq
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase
from torch import distributed as dist
from torch.distributed import ProcessGroup

from axrl.utils import gpu_utils

if TYPE_CHECKING:
    from tensordict.utils import NestedKey

logger = logging.getLogger(__name__)


_MERGE_INFO_KEY = "merge_info"
_ROUTING_HANDLES_PER_PATH_KEY = "routing_handles_per_path"
_NON_TENSOR_KEYS = (_MERGE_INFO_KEY, _ROUTING_HANDLES_PER_PATH_KEY)


def realign_non_tensor_keys_after_split(batch: TensorDict, micro_batches: Sequence[TensorDictBase]) -> None:
    """Slice each microbatch's per-row non-tensor lists to its own row range.

    ``TensorDict.split`` (the static-microbatch path in
    :meth:`MegatronWorker.forward_backward_batch`) carries each non-tensor
    list as a per-TD value, which means every chunk gets the full list
    rather than its row slice. Walk the chunks and overwrite each one's
    ``merge_info`` / ``routing_handles`` with the correct slice so the
    downstream consumer sees only its own trajectories.
    """
    for key in _NON_TENSOR_KEYS:
        if key not in batch.keys():  # noqa: SIM118
            continue
        full = batch.get_non_tensor(key)
        offset = 0
        for mb in micro_batches:
            n = int(mb.batch_size[0])
            mb.set_non_tensor(key, full[offset : offset + n])
            offset += n


def _index_batch(batch: TensorDict, indices: list[int] | torch.Tensor) -> TensorDict:
    """Select a subset of samples while preserving nested-tensor + non-tensor fields.

    Reference material:
    - veRL sequence-length balancing helpers:
      https://github.com/verl-project/verl/blob/7402ca73bcf2b85b5337393b4ccc9ec45ea96b6d/verl/utils/seqlen_balancing.py#L1-L220
    """
    if isinstance(indices, torch.Tensor):
        index_list = indices.tolist()
    else:
        index_list = indices

    indexed: dict[NestedKey, object] = {}
    for key, value in batch.items():
        if key in _NON_TENSOR_KEYS:
            continue
        if getattr(value, "is_nested", False):
            indexed[key] = torch.nested.as_nested_tensor([value[idx] for idx in index_list], layout=torch.jagged)
        else:
            indexed[key] = value[index_list]
    out = TensorDict(indexed, batch_size=len(index_list))
    for key in _NON_TENSOR_KEYS:
        if key in batch.keys():  # noqa: SIM118
            full_list = batch.get_non_tensor(key)
            out.set_non_tensor(key, [full_list[idx] for idx in index_list])
    return out


def _concat_micro_batches(micro_batches: list[TensorDict]) -> TensorDict:
    concatenated: dict[NestedKey, torch.Tensor] = {}
    for key in list(micro_batches[0].keys()):
        values = [micro_batch[key] for micro_batch in micro_batches]
        if getattr(values[0], "is_nested", False):
            flattened = []
            for value in values:
                flattened.extend(list(value.unbind(0)))
            concatenated[key] = torch.nested.as_nested_tensor(flattened, layout=torch.jagged)
        else:
            concatenated[key] = torch.cat(values, dim=0)
    total_batch_size = sum(int(micro_batch.batch_size[0]) for micro_batch in micro_batches)
    return TensorDict(concatenated, batch_size=total_batch_size)


def _karmarkar_karp(seqlen_list: list[int], k_partitions: int, *, equal_size: bool) -> list[list[int]]:  # noqa: C901
    """Partition indices of `seqlen_list` into `k_partitions` to balance their sums."""

    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items: list[tuple[int, int]] = []

        def add(self, idx: int, val: int) -> None:
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other: "Set") -> None:
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other: "Set") -> bool:
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self) -> list[list[int]]:
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other: "State") -> None:
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other: "State") -> bool:
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq: list[State] = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for _, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def _get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, *, equal_size: bool) -> list[list[int]]:
    """Calculates partitions of indices from seqlen_list such that the sum of sequence lengths in each partition is balanced.

    Uses the Karmarkar-Karp differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                           Requires len(seqlen_list) to be divisible by k_partitions.
                           If False, partitions can have varying numbers of items, focusing
                           only on balancing the sum of sequence lengths.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                         original indices of the items assigned to that partition. The indices
                         within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions: list[list[int]]) -> list[list[int]]:
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions: list[list[int]] = []
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions.append(sorted(partition))
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = _karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def _ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def _roundup_divisible(a: int, b: int) -> int:
    return ((a + b - 1) // b) * b


def split_into_balanced_microbatches(
    batch: TensorDict,
    max_token_len: int,
    dp_group: ProcessGroup | None = None,
    vpp_size: int | None = None,
    min_num_micro_batch: int | None = None,
    microbatch_group_size: int | None = None,
    *,
    same_micro_num_in_dp: bool = True,
    use_dynamic_bsz_balance: bool = True,
    verbose: bool = False,
) -> tuple[list[TensorDict], list[list[int]]]:
    """Split a batch into micro-batches by total token count, with optional DP sync and padding.

    Args:
        batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
        max_token_len (int): max sum of attention_mask per micro-batch.
        dp_group (optional): torch.distributed group for data-parallel sync.
        vpp_size (optional): virtual pipeline parallel size.
        microbatch_group_size (int, optional): for VPP, round num_micro_batches up to be divisible
            by this value. Corresponds to ``microbatch_group_size_per_vp_stage`` in Megatron, which
            defaults to ``pipeline_parallel_size``.
            Reference (Megatron VPP schedule divisibility check):
                https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/pipeline_parallel/schedules.py#L996-L997
            Reference (slime VPP divisibility enforcement):
                https://github.com/THUDM/slime/blob/f71f7103dfdffdc65064d5798a4b68a242461084/slime/backends/megatron_utils/data.py#L360-L363
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
        min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).
        use_dynamic_bsz_balance (bool, optional): balance the computational workload between micro-batches
        verbose (bool, optional): if True, log detailed micro-batch info.

    Returns:
        List[TensorDict]: the micro-batches.
        List[List[int]]: index lists mapping each micro-batch back to original positions.
    """
    # this is per local micro_bsz
    is_merged = _MERGE_INFO_KEY in batch.keys()  # noqa: SIM118
    if is_merged:
        merge_infos = batch.get_non_tensor(_MERGE_INFO_KEY)
        seq_len_effective = [int(mi.total_padded) for mi in merge_infos]
        max_seq_len = max(seq_len_effective)
    else:
        max_seq_len = batch["attention_mask"].shape[-1]
        seq_len_effective = batch["attention_mask"].sum(dim=1).tolist()
    if not is_merged:
        assert max_token_len >= max_seq_len, f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    effective_max_token_len = max(max_token_len, max_seq_len)
    total_seqlen: int = sum(seq_len_effective)
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    num_micro_batches: int = min(len(seq_len_effective), _ceildiv(total_seqlen, effective_max_token_len))
    if min_num_micro_batch is not None:
        # used to support pp
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    if dist.is_initialized() and same_micro_num_in_dp:
        num_micro_batches_tensor = torch.tensor([num_micro_batches], device=gpu_utils.get_current_device())
        dist.all_reduce(num_micro_batches_tensor, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches_tensor.cpu().item()  # type: ignore
    if vpp_size is not None:
        num_micro_batches = _roundup_divisible(num_micro_batches, vpp_size)
    if microbatch_group_size is not None:
        num_micro_batches = _roundup_divisible(num_micro_batches, microbatch_group_size)

    assert num_micro_batches <= len(seq_len_effective), (
        f"num_micro_batches ({num_micro_batches}) must be less than or equal to batch size ({len(seq_len_effective)})"
    )

    micro_bsz_idx = _get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)

    if use_dynamic_bsz_balance:
        # Use the sum of squared sequence lengths to approximate attention computation workload
        micro_bsz_idx.sort(
            key=lambda partition: (
                sum(seq_len_effective[idx] ** 2 for idx in partition),
                min(partition) if partition else 0,
            ),
            reverse=True,
        )

    micro_batches = [_index_batch(batch, partition) for partition in micro_bsz_idx]

    if verbose:
        log_microbatch_info(micro_batches, micro_bsz_idx)

    return micro_batches, micro_bsz_idx


def _get_reverse_idx(idx_map: list[int]) -> list[int]:
    """Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map


def reconstruct_batch_from_microbatches(micro_batches: list[TensorDict], micro_bsz_idx_lst: list[list[int]]) -> TensorDict:
    batch = _concat_micro_batches(micro_batches)
    micro_bsz_idx: list[int] = []
    for idx in micro_bsz_idx_lst:
        micro_bsz_idx.extend(idx)
    reverse_idx_map = torch.tensor(_get_reverse_idx(micro_bsz_idx))
    new_batch = _index_batch(batch, reverse_idx_map)
    return new_batch


def log_microbatch_info(micro_batches: list[TensorDict], micro_bsz_idx_lst: list[list[int]]) -> None:
    logger.info(f"Splitted into {len(micro_batches)} micro batches, indexes: {micro_bsz_idx_lst}")
    for i, micro_batch in enumerate(micro_batches):
        seqlens = micro_batch["attention_mask"].sum(dim=1).tolist()
        logger.info(f"Micro batch {i}: samples: {micro_batch['attention_mask'].shape[0]}, seq lens: {seqlens}, sum: {sum(seqlens)}")


def test_seqlen_balancing() -> None:
    max_seq_len: int = 2048
    max_token_len: int = 2048 * 2
    batch_size: int = 17
    num_input_ids: list[int] = [(x + 1) * 100 for x in range(batch_size)]
    input_ids = torch.Tensor([[1] * seq_len + [0] * (max_seq_len - seq_len) for seq_len in num_input_ids]).long()
    attention_mask = input_ids.ne(0).long()
    data: TensorDict = TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=batch_size)
    micro_batches, micro_bsz_idx_lst = split_into_balanced_microbatches(data, max_token_len=max_token_len, vpp_size=3, verbose=True)
    new_batch = reconstruct_batch_from_microbatches(micro_batches, micro_bsz_idx_lst)
    torch.testing.assert_close(new_batch, data)
    logger.info("test_seqlen_balancing passed!")


if __name__ == "__main__":
    from axrl.utils import setup_logger

    setup_logger("info")
    test_seqlen_balancing()
