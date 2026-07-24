from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import ray
import torch
from tensordict import TensorDict

from axrl.data.generation import TensorHandle
from axrl.ray import ray_utils
from axrl.utils import tensor_store as store
from axrl.utils.megatron.routing_materialiser import RoutingMaterialiser, materialise_routed_experts_from_batch
from axrl.utils.megatron.spike_snapshot_routing import (
    collect_unique_routing_handles_from_batch,
    restore_spike_snapshot_routing,
    save_spike_snapshot_routing,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


NUM_LAYERS = 2
TOPK = 2


@pytest.fixture
def _ray_cluster() -> Iterator[None]:
    if ray.is_initialized():
        ray_utils.stop()
    ray.init(num_cpus=2, include_dashboard=False)
    try:
        yield
    finally:
        ray_utils.stop()


def _routing_payload(start: int, rows: int) -> np.ndarray:
    return (np.arange(rows * NUM_LAYERS * TOPK, dtype=np.int16).reshape(rows, NUM_LAYERS, TOPK) + start).astype(np.int16)


def _batch(routing_handles_per_path: list[list[list[TensorHandle]]]) -> TensorDict:
    batch = TensorDict(
        {"input_ids": torch.ones((len(routing_handles_per_path), 4), dtype=torch.long)},
        batch_size=len(routing_handles_per_path),
    )
    batch.set_non_tensor("routing_handles_per_path", routing_handles_per_path)
    return batch


def _nested_rows(nested: torch.Tensor) -> list[np.ndarray]:
    return [row.numpy().copy() for row in nested.unbind(0)]


@pytest.mark.usefixtures("_ray_cluster")
def test_spike_snapshot_routing_payloads_restore_materialiser_equivalent(tmp_path: Path) -> None:
    h0 = store.put(_routing_payload(10, rows=2))
    h1_base = store.put(_routing_payload(30, rows=4))
    h1 = TensorHandle(ref=h1_base.ref, row_start=1, row_count=2)
    h2 = store.put(_routing_payload(50, rows=3))
    batch = _batch([[[h0, h1]], [[h0, h2]]])
    before = materialise_routed_experts_from_batch(batch, RoutingMaterialiser())
    assert before is not None
    before_rows = _nested_rows(before)
    old_refs = {h0.ref, h1.ref, h2.ref}

    payload_path = tmp_path / "routing_payload_rank0.pt"
    assert save_spike_snapshot_routing(batch, payload_path) == 3

    store.delete_batch([h0, h1, h2])
    assert restore_spike_snapshot_routing(batch, payload_path) == 3
    restored_handles = collect_unique_routing_handles_from_batch(batch)
    assert len(restored_handles) == 3
    assert all(handle.ref not in old_refs for handle in restored_handles)

    rows = batch.get_non_tensor("routing_handles_per_path")
    assert rows[0][0] == [restored_handles[0], restored_handles[1]]
    assert rows[1][0] == [restored_handles[0], restored_handles[2]]

    after = materialise_routed_experts_from_batch(batch, RoutingMaterialiser())
    assert after is not None
    for restored, expected in zip(_nested_rows(after), before_rows, strict=True):
        assert np.array_equal(restored, expected)


def test_restore_spike_snapshot_routing_rejects_payload_count_mismatch(tmp_path: Path) -> None:
    h0 = TensorHandle(ref="nodeA:h0")
    h1 = TensorHandle(ref="nodeA:h1")
    batch = _batch([[[h0, h1]]])
    payload_path = tmp_path / "routing_payload_rank0.pt"
    torch.save([_routing_payload(10, rows=1)], payload_path)

    with pytest.raises(RuntimeError, match="payload count mismatch"):
        restore_spike_snapshot_routing(batch, payload_path)
