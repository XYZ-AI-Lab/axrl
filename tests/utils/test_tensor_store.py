"""Smoke tests for axrl.utils.tensor_store.

Runs a single-node Ray cluster so they're ~3s slower than pure unit
tests, but they exercise the real plasma round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest
import ray

from axrl.utils import tensor_store as store
from axrl.utils.tensor_store import TensorHandle


@pytest.fixture(autouse=True)
def _ray_cluster():  # noqa: ANN202 (pytest yield fixture)
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=2, include_dashboard=False)
    yield
    ray.shutdown()


def test_put_get_round_trip_preserves_bytes() -> None:
    payload = np.arange(256, dtype=np.int16).reshape(16, 16)
    h = store.put(payload)
    fetched = store.get_batch([h])
    np.testing.assert_array_equal(fetched[h], payload)


def test_get_batch_returns_all_handles() -> None:
    h1 = store.put(np.full((8,), 1, dtype=np.int32))
    h2 = store.put(np.full((8,), 2, dtype=np.int32))
    h3 = store.put(np.full((8,), 3, dtype=np.int32))
    out = store.get_batch([h1, h3, h2])
    assert out[h1][0] == 1
    assert out[h2][0] == 2
    assert out[h3][0] == 3


def test_get_batch_applies_row_slice_views() -> None:
    payload = np.arange(10, dtype=np.int32)
    h = store.put(payload)
    sliced = TensorHandle(ref=h.ref, row_start=2, row_count=4)
    out = store.get_batch([sliced])
    np.testing.assert_array_equal(out[sliced], payload[2:6])


def test_empty_get_and_delete_are_cheap() -> None:
    assert store.get_batch([]) == {}
    assert store.delete_batch([]) is None  # type: ignore[func-returns-value]


def test_handle_equality_by_ref() -> None:
    # With ref typed as str | ObjectRef, tests can use string sentinels.
    h1 = TensorHandle(ref="opk0")
    h2 = TensorHandle(ref="opk0")
    h3 = TensorHandle(ref="opk1")
    assert h1 == h2 and hash(h1) == hash(h2)
    assert h1 != h3
    assert {h1: "ok"}[h2] == "ok"
