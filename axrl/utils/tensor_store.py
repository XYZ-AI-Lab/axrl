"""Tensor store backed by Ray plasma. Deadly simple.

Usage::

    from axrl.utils import tensor_store as store

    h = store.put(tensor)              # producer
    tensors = store.get_batch([h])     # consumer; dict keyed by handle
    store.delete_batch([h])            # free plasma objects

Module-level functions (no class, no singleton) — each call dispatches
against whatever Ray context the caller lives in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import ray
import ray._private.internal_api

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class TensorHandle:
    """Opaque handle wrapping a Ray ``ObjectRef``.

    Production callers get an ``ObjectRef`` from :func:`put`. Tests
    that only need a hashable sentinel (e.g. routing-tree logic that
    never fetches real tensors) can pass a short ``str`` instead.

    ``row_start`` / ``row_count`` are an optional first-axis view over the
    stored tensor. R3 compaction uses this to keep the still-valid prefix of a
    routing tensor when a later tool-result chunk is rewritten.
    """

    ref: str | ray.ObjectRef
    row_start: int = 0
    row_count: int | None = None

    def prefix(self, row_count: int) -> TensorHandle:
        assert row_count >= 0, f"row_count must be non-negative, got {row_count}"
        if self.row_count is not None:
            assert row_count <= self.row_count, f"prefix row_count {row_count} exceeds handle row_count {self.row_count}"
        return TensorHandle(ref=self.ref, row_start=self.row_start, row_count=row_count)


def put(tensor: object) -> TensorHandle:
    return TensorHandle(ref=ray.put(tensor))


def get_batch(handles: list[TensorHandle]) -> dict[TensorHandle, NDArray]:
    if not handles:
        return {}
    refs: list[ray.ObjectRef] = []
    seen: set[ray.ObjectRef] = set()
    for handle in handles:
        ref = cast("ray.ObjectRef", handle.ref)
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    tensors_by_ref = dict(zip(refs, ray.get(refs), strict=True))
    return {handle: _apply_row_slice(tensors_by_ref[cast("ray.ObjectRef", handle.ref)], handle) for handle in handles}


def delete_batch(handles: list[TensorHandle]) -> None:
    if not handles:
        return
    refs = []
    seen: set[ray.ObjectRef] = set()
    for handle in handles:
        if isinstance(handle.ref, str):
            continue
        if handle.ref in seen:
            continue
        seen.add(handle.ref)
        refs.append(handle.ref)
    if not refs:
        return
    ray._private.internal_api.free(refs)


def _apply_row_slice(tensor: NDArray, handle: TensorHandle) -> NDArray:
    if handle.row_start == 0 and handle.row_count is None:
        return tensor
    end = None if handle.row_count is None else handle.row_start + handle.row_count
    return tensor[handle.row_start : end]
