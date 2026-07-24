"""Tests for the trainer-side R3 caches in axrl/utils/megatron/routing_caches.py.

Covers:
- traj_key_of returns the handle itself (opaque key; different handles for
  different trajectories).
- RoutingMergedCache: get returns None until put; idempotent; cleared on clear().
- R3PrefetchRing: depth-2 issue/consume order; shutdown drains.
"""

from typing import Any

import numpy as np
import pytest

from axrl.data.generation import TensorHandle
from axrl.utils.megatron.routing_caches import (
    R3PrefetchRing,
    RoutingMergedCache,
    traj_key_of,
)


def test_traj_key_of_uses_last_path_last_handle() -> None:
    # ``handles_per_path[0][0]`` is shared across all packed samples of a split
    # trace, so the key must differ at the LAST path's LAST chunk.
    shared_first = TensorHandle(ref="nodeA:traj-A-call0")
    handles_a = [[shared_first, TensorHandle(ref="nodeA:traj-A-t0-last")], [TensorHandle(ref="nodeA:traj-A-t1")]]
    handles_b = [[shared_first, TensorHandle(ref="nodeA:traj-A-t2-last")], [TensorHandle(ref="nodeA:traj-A-t3")]]
    key_a = traj_key_of(handles_a)
    key_b = traj_key_of(handles_b)
    assert key_a is handles_a[-1][-1]
    assert key_b is handles_b[-1][-1]
    assert key_a != key_b


def test_different_trajectories_produce_distinct_handles() -> None:
    """Two trajectories at the same turn-order must yield different opaque keys."""
    a = TensorHandle(ref="nodeA:traj-A-call0")
    b = TensorHandle(ref="nodeA:traj-B-call0")
    assert a != b
    assert hash(a) != hash(b)


def test_merged_cache_round_trip() -> None:
    cache = RoutingMergedCache()
    key = TensorHandle(ref="nodeA:traj-A-t0")
    assert cache.get(key) is None
    assert key not in cache

    arr = np.zeros((5, 4, 2), dtype=np.int16)
    cache.put(key, arr)
    assert key in cache
    assert cache.get(key) is arr

    cache.clear()
    assert cache.get(key) is None


def test_prefetch_ring_fifo_consume() -> None:
    fetched_payloads = [
        {TensorHandle(ref="nodeA:pf0"): np.array([1])},
        {TensorHandle(ref="nodeA:pf1"): np.array([2])},
        {TensorHandle(ref="nodeA:pf2"): np.array([3])},
    ]
    requests: list[list[TensorHandle]] = []

    class _ImmediateFuture:
        def __init__(self, payload: dict[TensorHandle, Any]) -> None:
            self._payload = payload

        def result(self) -> dict[TensorHandle, Any]:
            return self._payload

    def fake_async_get(handles: list[TensorHandle]) -> _ImmediateFuture:
        idx = len(requests)
        requests.append(list(handles))
        return _ImmediateFuture(fetched_payloads[idx])

    ring = R3PrefetchRing(fake_async_get, depth=2)
    for payload in fetched_payloads:
        h = next(iter(payload.keys()))
        ring.issue([h])
    consumed = [ring.consume() for _ in range(len(fetched_payloads))]
    assert consumed == fetched_payloads


def test_prefetch_ring_empty_issue_returns_empty_dict() -> None:
    ring = R3PrefetchRing(lambda _h: pytest.fail("should not be called for empty issue"))
    ring.issue([])
    assert ring.consume() == {}


def test_prefetch_ring_shutdown_drains_pending() -> None:
    cancelled: list[bool] = []

    class _CancellableFuture:
        def cancel(self) -> None:
            cancelled.append(True)

    ring = R3PrefetchRing(lambda _h: _CancellableFuture())
    ring.issue([TensorHandle(ref="nodeA:drain0")])
    ring.issue([TensorHandle(ref="nodeA:drain1")])
    ring.shutdown()
    assert len(ring) == 0
    assert cancelled == [True, True]
