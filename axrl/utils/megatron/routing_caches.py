"""Caches for the R3 store path on the trainer side.

These are scoped to one global step, keyed by a handle that uniquely identifies
the packed-sample (the LAST chunk handle of the LAST path). The key survives the
``compute_logprobs`` (eval mbs) → ``train_step`` (train mbs) microbatching
boundary, whereas ``mb_idx`` does not.

The controller calls ``clear()`` at the end of each global step,
alongside the tensor store's ``delete``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from axrl.data.generation import TensorHandle

# Each ``TensorHandle`` is a frozen dataclass (hashable) and uuid4-minted, so
# any single handle is globally unique. When a trace produces a SINGLE merged
# sample (e.g. ``RolloutTrace.to_sample()``), its first-handle is unique to
# that trace.
#
# But ``RolloutTrace.to_packed_samples()`` can split one trace into multiple
# packed samples, and the per-turn cumulative ``routing_handles_per_path[0]``
# means every packed sample of a trace shares the SAME first handle (the
# trace's very first chunk). Keying on the first handle would collide those
# packed samples into one cache entry. Keying on the LAST chunk handle of the
# LAST path is unique per packed sample because the last turn-in-group differs
# across the packed samples of a split trace.


def traj_key_of(handles_per_path: list[list[TensorHandle]]) -> TensorHandle:
    """Uniquely identify a packed sample via its last-path's last-chunk handle."""
    assert handles_per_path, "handles_per_path must be non-empty"
    last_path = handles_per_path[-1]
    assert last_path, "every path must have at least one handle"
    return last_path[-1]


class RoutingMergedCache:
    """Per-trajectory merged routing tensor, shared across ref/old/train.

    First forward populates; subsequent forwards short-circuit to the cached
    bytes without re-fetching from the tensor store or re-running the gather.
    """

    def __init__(self) -> None:
        self._by_traj: dict[TensorHandle, np.ndarray] = {}

    def get(self, traj_key: TensorHandle) -> np.ndarray | None:
        return self._by_traj.get(traj_key)

    def put(self, traj_key: TensorHandle, merged: np.ndarray) -> None:
        self._by_traj[traj_key] = merged

    def __contains__(self, traj_key: TensorHandle) -> bool:
        return traj_key in self._by_traj

    def clear(self) -> None:
        self._by_traj.clear()


class R3PrefetchRing:
    """Two-deep async store prefetch for a per-microbatch loop.

    Issuing a fetch is non-blocking; ``consume`` awaits the in-flight future
    for the next microbatch. Same-node handles resolve eagerly with a
    zero-copy view, so single-node deployments observe near-zero wait.
    """

    def __init__(self, store_async_get: Callable[[list[TensorHandle]], object], depth: int = 2) -> None:
        self._store_async_get = store_async_get
        self.depth = depth
        self._pending: list[object] = []  # list[Future-like]

    def issue(self, handles: list[TensorHandle]) -> None:
        if not handles:
            self._pending.append(_PrefetchedEmpty())
            return
        self._pending.append(self._store_async_get(handles))

    def consume(self) -> dict[TensorHandle, np.ndarray]:
        future = self._pending.pop(0)
        if isinstance(future, _PrefetchedEmpty):
            return {}
        # ``.result()`` blocks here until the prefetched store fetch resolves.
        # In the intended caller pattern, the previous microbatch's forward
        # ran concurrently with this fetch, so the wait is usually near-zero.
        result = future.result() if hasattr(future, "result") else future  # type: ignore[union-attr]
        assert isinstance(result, dict), f"store_async_get must return dict[TensorHandle, np.ndarray], got {type(result).__name__}"
        return result

    def shutdown(self) -> None:
        import contextlib

        for f in self._pending:
            if hasattr(f, "cancel"):
                with contextlib.suppress(Exception):
                    f.cancel()  # type: ignore[union-attr]
        self._pending.clear()

    def __len__(self) -> int:
        return len(self._pending)


class _PrefetchedEmpty:
    """Sentinel for empty issues (no handles to fetch)."""
