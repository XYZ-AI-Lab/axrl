"""CPU tests for ``RoutingMaterialiser``.

Stubs store via monkeypatch; no Ray. Covers the most likely bug area:
the materialiser that turns per-path handles into the merged routing tensor
the Megatron forward consumes.
"""

from __future__ import annotations

import numpy as np
import pytest

from axrl.data.generation import TensorHandle
from axrl.utils.megatron.prefix_tree import (
    PrefixMergeInfo,
    gather_merged_routing_per_path,
    merge_trajectory_samples,
)
from axrl.utils.megatron.routing_materialiser import RoutingMaterialiser, materialise_routed_experts_from_batch
from tests.routing_replay._compacted_fixture import (
    NUM_LAYERS,
    TOPK,
    CompactedFixture,
    build_compacted_fixture,
    make_routing_payloads,
)


class _Stub:
    def __init__(self) -> None:
        self.store: dict[TensorHandle, np.ndarray] = {}
        self.fetch_calls = 0
        self.fetched_handles: list[list[TensorHandle]] = []

    def __call__(self, handles: list[TensorHandle]) -> dict[TensorHandle, np.ndarray]:
        self.fetch_calls += 1
        self.fetched_handles.append(list(handles))
        out: dict[TensorHandle, np.ndarray] = {}
        for h in handles:
            if h not in self.store:
                raise KeyError(f"missing handle {h}")
            out[h] = self.store[h]
        return out


@pytest.fixture
def stub_tq(monkeypatch: pytest.MonkeyPatch) -> _Stub:
    stub = _Stub()

    monkeypatch.setattr("axrl.utils.tensor_store.get_batch", stub)
    return stub


def _populate_stub(stub: _Stub, fixture: CompactedFixture) -> dict[TensorHandle, np.ndarray]:
    payloads = make_routing_payloads(fixture)
    stub.store.update(payloads)
    return payloads


def test_materialiser_fetches_each_unique_handle_exactly_once(stub_tq: _Stub) -> None:
    """A compacted trajectory fetches each distinct handle view in one batch."""
    f = build_compacted_fixture()
    _populate_stub(stub_tq, f)
    merged = merge_trajectory_samples(f.turn_samples)
    assert merged.routing_handles_per_path is not None
    assert merged.merge_info is not None
    materialiser = RoutingMaterialiser()

    rhpp: list[list[list[TensorHandle]]] = [merged.routing_handles_per_path]
    mi_list: list[PrefixMergeInfo | None] = [merged.merge_info]
    materialiser.materialise(rhpp, mi_list)

    assert stub_tq.fetch_calls == 1
    fetched = stub_tq.fetched_handles[0]
    assert len(set(fetched)) == len(fetched)
    assert {h.ref for h in fetched} == {h.ref for h in f.all_minted_handles}


def test_materialiser_gather_output_matches_hand_computed_reference(stub_tq: _Stub) -> None:
    """Materialiser output equals an external-reference per-path concat + gather."""
    f = build_compacted_fixture()
    payloads = _populate_stub(stub_tq, f)
    merged = merge_trajectory_samples(f.turn_samples)
    assert merged.routing_handles_per_path is not None
    assert merged.merge_info is not None

    got = RoutingMaterialiser().materialise([merged.routing_handles_per_path], [merged.merge_info]).unbind(0)[0].numpy()
    per_path = [np.concatenate([payloads[h] for h in chain], axis=0) for chain in merged.routing_handles_per_path]
    reference = gather_merged_routing_per_path(per_path, merged.merge_info)
    assert np.array_equal(got, reference)


def test_materialiser_cache_hits_on_repeat_call_for_same_trajectory(stub_tq: _Stub) -> None:
    """ref/old/train forwards on the same trajectory re-use the cached merged tensor (no extra store fetches)."""
    f = build_compacted_fixture()
    _populate_stub(stub_tq, f)
    merged = merge_trajectory_samples(f.turn_samples)
    assert merged.routing_handles_per_path is not None
    assert merged.merge_info is not None
    materialiser = RoutingMaterialiser()

    materialiser.materialise([merged.routing_handles_per_path], [merged.merge_info])
    calls_after_first = stub_tq.fetch_calls
    materialiser.materialise([merged.routing_handles_per_path], [merged.merge_info])
    assert stub_tq.fetch_calls == calls_after_first


def test_materialiser_flat_trajectory_bypasses_gather(stub_tq: _Stub) -> None:
    """When ``mi is None`` (flat single-path trajectory), per-path concat is returned verbatim."""
    h = TensorHandle(ref="nodeA:flat0")
    rows = 7
    payload = np.empty((rows, NUM_LAYERS, TOPK), dtype=np.int16)
    for r in range(rows):
        payload[r, :, :] = r + 100
    stub_tq.store[h] = payload

    nested = RoutingMaterialiser().materialise([[[h]]], [None])
    assert np.array_equal(nested.unbind(0)[0].numpy(), payload)


def test_materialiser_missing_handle_raises_loudly(stub_tq: _Stub) -> None:
    """Missing handles must raise, not silently zero-substitute."""
    f = build_compacted_fixture()
    payloads = make_routing_payloads(f)
    # Leave the third minted handle (turn 2) missing to mimic a dropped PUT.
    missing = f.all_minted_handles[2]
    for h, v in payloads.items():
        if h != missing:
            stub_tq.store[h] = v
    merged = merge_trajectory_samples(f.turn_samples)
    assert merged.routing_handles_per_path is not None
    assert merged.merge_info is not None
    with pytest.raises(KeyError, match="missing handle"):
        RoutingMaterialiser().materialise([merged.routing_handles_per_path], [merged.merge_info])


def test_materialiser_reads_routing_handles_through_sampletensordict_roundtrip(stub_tq: _Stub) -> None:
    """Merged sample → SampleTensorDict → back → materialise: matches reference gather."""
    from axrl.data.sample import SampleTensorDict, samples_from_tensor_dict

    f = build_compacted_fixture()
    payloads = _populate_stub(stub_tq, f)
    merged = merge_trajectory_samples(f.turn_samples)
    assert merged.routing_handles_per_path is not None
    assert merged.merge_info is not None

    td = SampleTensorDict.from_samples([merged])
    rt = samples_from_tensor_dict(td)[0]
    assert rt.routing_handles_per_path is not None

    got = RoutingMaterialiser().materialise([rt.routing_handles_per_path], [rt.merge_info]).unbind(0)[0].numpy()
    per_path = [np.concatenate([payloads[h] for h in chain], axis=0) for chain in merged.routing_handles_per_path]
    reference = gather_merged_routing_per_path(per_path, merged.merge_info)
    assert np.array_equal(got, reference)


def test_materialiser_zero_fills_shared_padding_routing_handle(stub_tq: _Stub) -> None:
    from axrl.configs import IGNORE_INDEX
    from axrl.data import array_utils
    from axrl.data.sample import Sample, SampleTensorDict, pad_sample_tensor_dict_to_multiple

    h = TensorHandle(ref="nodeA:real")
    padding_handle = TensorHandle(ref="nodeA:padding")
    real = Sample(
        input_ids=array_utils.as_i32([1, 2, 3, 4]),
        labels=array_utils.as_i32([2, 3, 4, IGNORE_INDEX]),
        loss_mask=array_utils.as_bool([False, True, True, False]),
        attention_mask=array_utils.as_bool([True, True, True, True]),
        position_ids=array_utils.as_i32([0, 1, 2, 3]),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0, 1.0, 1.0, 0.0]),
        routing_handles_per_path=[[h]],
    )
    payload = np.arange(3 * NUM_LAYERS * TOPK, dtype=np.int16).reshape(3, NUM_LAYERS, TOPK)
    stub_tq.store[h] = payload
    stub_tq.store[padding_handle] = np.zeros((3, NUM_LAYERS, TOPK), dtype=np.int16)
    batch, _ = pad_sample_tensor_dict_to_multiple(
        SampleTensorDict.from_samples([real]),
        multiple=2,
        padding_sample_length=4,
        padding_routing_handle=padding_handle,
    )

    nested = materialise_routed_experts_from_batch(batch, RoutingMaterialiser())
    assert nested is not None
    real_routing, padding_routing = nested.unbind(0)

    assert np.array_equal(real_routing.numpy(), payload)
    assert padding_routing.shape == payload.shape
    assert not bool(padding_routing.any().item())
    assert padding_handle in stub_tq.fetched_handles[0]
