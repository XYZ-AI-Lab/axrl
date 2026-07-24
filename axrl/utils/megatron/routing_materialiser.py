"""Trainer-side resolution of routing handles into the nested-tensor form the ``prepare_*_router_replay_tensors`` functions expect.

Sits between the data iterator (which yields a ``TensorDict`` with
``routing_handles_per_path`` as a non-tensor field) and the existing layout
bridge in ``axrl/utils/megatron/router_replay.py``.

For each trajectory (= sample row) the materialiser:
- Fetches all unique ``TensorHandle``s for the row (across paths) from the tensor store.
- Builds one routing array per leaf path (``len(path_handles_i) - 1`` chunks
  concatenated along axis 0).
- For trie-merged trajectories, gathers via the source map computed in
  ``merge_trajectory_samples`` (trainable wins / lowest path_idx wins).
- For flat trajectories, picks the single path's routing array as-is.

Returns a jagged nested tensor (one row per trajectory) of shape
``(real_total_i - 1, L, K)`` ready for ``prepare_*_router_replay_tensors``.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from axrl.utils.megatron.prefix_tree import gather_merged_routing_per_path
from axrl.utils.megatron.routing_caches import RoutingMergedCache, traj_key_of
from axrl.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tensordict import TensorDict

    from axrl.data.generation import TensorHandle
    from axrl.utils.megatron.prefix_tree import PrefixMergeInfo

logger = logging.getLogger(__name__)


class RoutingMaterialiser:
    """Per-step caches for trainer-side routing resolution.

    Caches per-trajectory merged tensors so the three forwards of a step
    (ref / old / train) each only fetch from the tensor store once per trajectory.
    Cleared by the controller's per-step delete.
    """

    def __init__(self) -> None:
        self._merged_cache = RoutingMergedCache()

    def materialise(
        self,
        routing_handles_per_path_per_traj: list[list[list[TensorHandle]]],
        merge_info_list: list[PrefixMergeInfo | None],
    ) -> torch.Tensor:
        """Resolve per-path handles → per-trajectory merged tensors → jagged nested tensor.

        ``merge_info_list[i]`` is ``None`` for flat samples (single path);
        otherwise it provides the trie source map.
        """
        from axrl.utils import tensor_store as store

        unique_handles_to_fetch: list[TensorHandle] = []
        seen: set[TensorHandle] = set()
        for handles_per_path, _mi in zip(routing_handles_per_path_per_traj, merge_info_list, strict=True):
            assert handles_per_path, "every trajectory must have at least one path"
            traj_key = traj_key_of(handles_per_path)
            if self._merged_cache.get(traj_key) is not None:
                continue
            for path_handles in handles_per_path:
                for h in path_handles:
                    if h not in seen:
                        seen.add(h)
                        unique_handles_to_fetch.append(h)

        with Timer(f"store.get_batch ({len(unique_handles_to_fetch)} handles)"):
            fetched: dict[TensorHandle, np.ndarray] = store.get_batch(unique_handles_to_fetch) if unique_handles_to_fetch else {}

        t_concat = 0.0
        t_merge = 0.0
        t_cache_put = 0.0
        t_from_numpy = 0.0
        n_trajs_fresh = 0
        n_trajs_cached = 0
        per_traj_tensors: list[torch.Tensor] = []
        for handles_per_path, mi in zip(routing_handles_per_path_per_traj, merge_info_list, strict=True):
            traj_key = traj_key_of(handles_per_path)
            merged_np = self._merged_cache.get(traj_key)
            if merged_np is None:
                n_trajs_fresh += 1
                t0 = time.perf_counter()
                per_path_routings = [
                    np.concatenate([fetched[h] for h in path_handles], axis=0) if path_handles else np.empty((0,), dtype=np.int16)
                    for path_handles in handles_per_path
                ]
                t_concat += time.perf_counter() - t0

                t0 = time.perf_counter()
                if mi is None:
                    assert len(per_path_routings) == 1, "flat sample must have exactly one path"
                    merged_np = per_path_routings[0]
                else:
                    merged_np = gather_merged_routing_per_path(per_path_routings, mi)
                t_merge += time.perf_counter() - t0

                t0 = time.perf_counter()
                self._merged_cache.put(traj_key, merged_np)
                t_cache_put += time.perf_counter() - t0
            else:
                n_trajs_cached += 1
            t0 = time.perf_counter()
            per_traj_tensors.append(torch.from_numpy(merged_np))
            t_from_numpy += time.perf_counter() - t0

        t0 = time.perf_counter()
        out = torch.nested.as_nested_tensor(per_traj_tensors, layout=torch.jagged)
        t_nested = time.perf_counter() - t0

        logger.debug(
            "materialise breakdown (ms): concat=%.1f merge=%.1f cache_put=%.1f from_numpy=%.1f nested=%.1f (fresh=%d cached=%d)",
            t_concat * 1000,
            t_merge * 1000,
            t_cache_put * 1000,
            t_from_numpy * 1000,
            t_nested * 1000,
            n_trajs_fresh,
            n_trajs_cached,
        )

        return out

    def clear(self) -> None:
        self._merged_cache.clear()


def materialise_routed_experts_from_batch(
    batch: TensorDict,
    materialiser: RoutingMaterialiser,
) -> torch.Tensor | None:
    """Pull per-path routing handles from ``batch`` and return a nested int16 tensor.

    Returns ``None`` when the batch carries no routing handles (R3 off).
    """
    if "routing_handles_per_path" not in batch.keys():  # noqa: SIM118
        return None
    raw_obj = _unwrap_non_tensor(batch.get_non_tensor("routing_handles_per_path"))
    raw_per_traj = list(raw_obj) if raw_obj is not None else []
    handles_per_path_per_traj: list[list[list[TensorHandle]]] = []
    for row in raw_per_traj:
        if row is None:
            continue
        unwrapped_row = _unwrap_non_tensor(row)
        handles_per_path_per_traj.append([list(_unwrap_non_tensor(p)) for p in unwrapped_row])
    if not handles_per_path_per_traj:
        return None
    if len(handles_per_path_per_traj) != len(raw_per_traj):
        msg = "routing_handles_per_path must be present on every trajectory in the microbatch (mixed None disallowed)"
        raise RuntimeError(msg)
    merge_info_list: list[PrefixMergeInfo | None]
    if "merge_info" in batch.keys():  # noqa: SIM118
        merge_info_list = [_unwrap_non_tensor(mi) for mi in batch.get_non_tensor("merge_info")]
    else:
        merge_info_list = [None] * len(handles_per_path_per_traj)
    return materialiser.materialise(handles_per_path_per_traj, merge_info_list)


def _unwrap_non_tensor(value: Any) -> Any:
    """Coerce a tensordict NonTensorData/NonTensorStack-shaped value back to its python data."""
    if hasattr(value, "data") and not isinstance(value, (list, tuple)):
        try:
            return value.data
        except AttributeError:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, type(None))):
        try:
            return value.tolist()
        except (AttributeError, RuntimeError):
            pass
    return value


# ----------------------------------------------------------------------
# Microbatch prefetch — overlaps R3 materialise with forward compute.
# ----------------------------------------------------------------------


def iter_microbatches_with_prefetched_routing(
    micro_batches: Sequence[TensorDict],
    materialiser: RoutingMaterialiser,
    prefetch_depth: int = 2,
) -> Iterator[TensorDict]:
    """Yield microbatches with routing prefetched in the background.

    Launches the fetch for ``mb[i + prefetch_depth]`` as ``mb[i]`` is
    yielded. The ``.result()`` call for ``mb[i]`` happens just before
    Megatron consumes ``mb[i]`` on the GPU, so while the GPU runs forward
    on ``mb[i-1]`` the CPU is already prefetching
    ``mb[i+prefetch_depth-1]``. Total wall becomes
    ``max(sum(fetches), sum(forwards)) + tail`` instead of
    ``sum(fetches) + sum(forwards)``. Peak routing memory drops from
    ``N x routing/mb`` to ``prefetch_depth x routing/mb``.

    Single-pass. For vpp, construct one generator per vpp stage — the
    first stage pays the real materialise cost; subsequent stages hit
    :class:`RoutingMaterialiser`'s per-trajectory cache.
    """
    if not micro_batches:
        return

    pool = ThreadPoolExecutor(
        max_workers=max(1, prefetch_depth),
        thread_name_prefix="r3_materialise",
    )
    in_flight: deque[Future[torch.Tensor | None]] = deque()
    for mb in micro_batches[:prefetch_depth]:
        in_flight.append(pool.submit(materialise_routed_experts_from_batch, mb, materialiser))

    try:
        for i, mb in enumerate(micro_batches):
            nxt = i + prefetch_depth
            if nxt < len(micro_batches):
                in_flight.append(
                    pool.submit(materialise_routed_experts_from_batch, micro_batches[nxt], materialiser),
                )
            routed_experts = in_flight.popleft().result()
            if routed_experts is not None:
                mb["routed_experts"] = routed_experts
            yield mb
    finally:
        pool.shutdown(wait=False)
