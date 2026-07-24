"""Tests for the FLAT-trajectory R3 path (single-turn rollouts).

Samples come from ``TokenTrace.to_sample`` → flat samples with
``merge_info=None`` and a single-element ``routing_handles_per_path=[[h0]]``.
At forward time, ``_build_single_path_merge_info`` synthesizes per-row
merge_info from ``attention_mask``; the materialiser sees ``mi=None`` and
returns the concat-of-one-handle directly (no gather).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from axrl.data import array_utils
from axrl.data.generation import TensorHandle
from axrl.data.sample import Sample, SampleTensorDict, samples_from_tensor_dict
from axrl.data.token_trace import TokenTrace
from axrl.utils.megatron.magi_forward import _build_single_path_merge_info
from axrl.utils.megatron.router_replay import pack_routing_for_magi
from axrl.utils.megatron.routing_materialiser import RoutingMaterialiser

NUM_LAYERS = 2
TOPK = 3


class _Stub:
    def __init__(self) -> None:
        self.store: dict[TensorHandle, np.ndarray] = {}

    def __call__(self, handles: list[TensorHandle]) -> dict[TensorHandle, np.ndarray]:
        return {h: self.store[h] for h in handles if h in self.store}


@pytest.fixture
def stub_tq(monkeypatch: pytest.MonkeyPatch) -> _Stub:
    stub = _Stub()

    monkeypatch.setattr("axrl.utils.tensor_store.get_batch", stub)
    return stub


def _build_flat_single_turn_sample(
    prompt_tokens: list[int],
    output_tokens: list[int],
    max_length: int,
    handle: TensorHandle | None = None,
) -> Sample:
    trace = TokenTrace()
    trace.extend_tokens(array_utils.as_i32(prompt_tokens), token_type="init")
    trace.extend_tokens(
        array_utils.as_i32(output_tokens),
        logprobs=array_utils.as_f32([0.0] * len(output_tokens)),
        token_type="assistant",
        routing_handle=handle,
    )
    return trace.to_sample(max_length=max_length, pad_token_id=0)


def test_flat_sample_forward_merge_info_consistent_with_handle_payload(stub_tq: _Stub) -> None:
    """End-to-end flat-path check: sglang payload → materialiser → pack must not raise."""
    prompt_len, output_len, max_length = 50, 78, 128  # running_len=128 aligned
    prompt = list(range(1, 1 + prompt_len))
    output = list(range(1000, 1000 + output_len))
    h0 = TensorHandle(ref="nodeA:opk0")
    sample = _build_flat_single_turn_sample(prompt, output, max_length=max_length, handle=h0)
    path_len = int(sum(sample.attention_mask))

    h0_rows = prompt_len + output_len - 1
    stub_tq.store[h0] = np.arange(h0_rows * NUM_LAYERS * TOPK, dtype=np.int16).reshape(h0_rows, NUM_LAYERS, TOPK)

    materialiser = RoutingMaterialiser()
    merged_tensor = materialiser.materialise([[[h0]]], [None]).unbind(0)[0]
    assert merged_tensor.shape[0] == h0_rows

    mi = _build_single_path_merge_info(path_len=path_len, align=1)
    packed = pack_routing_for_magi([merged_tensor], [mi], device=torch.device("cpu"))
    assert packed.shape[0] == mi.total_padded


def test_to_sample_fails_loudly_when_trace_exceeds_max_length() -> None:
    """Upstream invariant: ``TokenTrace.to_sample`` must reject over-length traces.

    Silently truncating here would desync the Sample from sglang's stored
    routing (captured over the PRE-truncation sequence) and corrupt R3
    replay. The rollout env is responsible for keeping the trace within
    ``max_length``; if it doesn't, we fail fast here rather than let the
    mismatch leak into ``pack_routing_for_magi``.
    """
    prompt_len, output_len, max_length = 100, 50, 128  # running_len=150 > max_length
    h0 = TensorHandle(ref="nodeA:opk0")
    with pytest.raises(AssertionError, match="exceeds max_length"):
        _build_flat_single_turn_sample(
            list(range(1, 1 + prompt_len)),
            list(range(1000, 1000 + output_len)),
            max_length=max_length,
            handle=h0,
        )


def test_pack_routing_for_magi_fails_loudly_on_row_count_mismatch() -> None:
    """Downstream guard: ``pack_routing_for_magi`` must assert on row-count mismatch.

    This mirrors the upstream invariant above. If routing rows and
    ``merge_info.real_total - 1`` ever disagree at pack time, the two
    were built over different sequences — silently slicing to fit would
    hide the bug, so we assert.
    """
    mi = _build_single_path_merge_info(path_len=100, align=1)
    over_long = torch.zeros((149, NUM_LAYERS, TOPK), dtype=torch.int16)
    with pytest.raises(AssertionError, match="merge_info and routing are out of sync"):
        pack_routing_for_magi([over_long], [mi], device=torch.device("cpu"))


def test_flat_sample_routing_handles_survive_tensordict_roundtrip() -> None:
    """Merge_info stays None and the per-path handle survives SampleTensorDict roundtrip."""
    h0 = TensorHandle(ref="nodeA:opk0")
    sample = _build_flat_single_turn_sample([1, 2, 3], [100, 101], max_length=16, handle=h0)
    assert sample.routing_handles_per_path == [[h0]]
    assert sample.merge_info is None

    td = SampleTensorDict.from_samples([sample])
    rt = samples_from_tensor_dict(td)[0]
    assert rt.routing_handles_per_path == [[h0]]
    assert rt.merge_info is None
