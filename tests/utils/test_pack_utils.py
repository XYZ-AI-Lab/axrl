"""Tests for Megatron packed-sequence helpers."""

import pytest
import torch

from axrl.utils.megatron import pack_utils


def _mock_cp_env(monkeypatch: pytest.MonkeyPatch, *, cp_size: int, cp_rank_ref: dict[str, int]) -> None:
    monkeypatch.setattr(pack_utils.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pack_utils.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(pack_utils.mpu, "get_context_parallel_rank", lambda: cp_rank_ref["value"])


def test_preprocess_packed_seqs_uses_custom_pad_value_for_cp_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify custom sentinels survive CP padding instead of being zero-filled.

    Router replay relies on non-zero placeholders for padded tokens so missing
    routing slots remain distinguishable after packed-sequence preprocessing.
    """
    cp_rank_ref = {"value": 0}
    _mock_cp_env(monkeypatch, cp_size=2, cp_rank_ref=cp_rank_ref)

    input_ids = torch.tensor([[[[10, 11]], [[20, 21]], [[30, 31]]]], dtype=torch.int16)
    attention_mask = torch.tensor([[True, True, False]])
    pad_value = torch.tensor([[0, 1]], dtype=torch.int16)

    packed, params = pack_utils.preprocess_packed_seqs(input_ids, attention_mask, pre_process=True, pad_value=pad_value)

    expected = pad_value.view(1, 1, 1, 2).expand(1, 128, 1, 2).clone()
    expected[0, 0] = torch.tensor([[10, 11]], dtype=torch.int16)
    expected[0, 1] = torch.tensor([[20, 21]], dtype=torch.int16)
    torch.testing.assert_close(packed, expected)
    assert params.cu_seqlens_q.tolist() == [0, 256]


def test_preprocess_packed_seqs_handles_short_cp_chunk_after_total_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for CP packing when total alignment inflates the last sequence.

    The last sequence may be padded up to the global total alignment, which can
    make the nominal CP half-chunk much larger than the number of valid tokens
    available in the first chunk.
    """
    cp_rank_ref = {"value": 0}
    _mock_cp_env(monkeypatch, cp_size=2, cp_rank_ref=cp_rank_ref)

    input_ids = torch.tensor([[10, 11, 12, 99]], dtype=torch.int64)
    attention_mask = torch.tensor([[True, True, True, False]])

    cp_rank_ref["value"] = 0
    packed_rank0, params_rank0 = pack_utils.preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
    expected_rank0 = torch.zeros((1, 128), dtype=torch.int64)
    expected_rank0[0, :3] = torch.tensor([10, 11, 12], dtype=torch.int64)
    torch.testing.assert_close(packed_rank0, expected_rank0)
    assert params_rank0.cu_seqlens_q.tolist() == [0, 256]

    cp_rank_ref["value"] = 1
    packed_rank1, params_rank1 = pack_utils.preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
    expected_rank1 = torch.zeros((1, 128), dtype=torch.int64)
    torch.testing.assert_close(packed_rank1, expected_rank1)
    assert params_rank1.cu_seqlens_q.tolist() == [0, 256]


def test_postprocess_packed_seqs_round_trips_short_cp_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Packed CP outputs reconstruct correctly even when one front chunk is short."""
    cp_rank_ref = {"value": 0}
    _mock_cp_env(monkeypatch, cp_size=2, cp_rank_ref=cp_rank_ref)
    monkeypatch.setattr(pack_utils.mpu, "get_context_parallel_group", lambda: None)

    input_ids = torch.tensor([[10, 11, 12, 99]], dtype=torch.int64)
    attention_mask = torch.tensor([[True, True, True, False]])
    expected = torch.tensor([[10, 11, 12, 0]], dtype=torch.int64)

    local_outputs: list[torch.Tensor] = []
    local_params: list = []
    for rank in range(2):
        cp_rank_ref["value"] = rank
        packed, params = pack_utils.preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        local_outputs.append(packed)
        local_params.append(params)

    def fake_all_gather(output_list: list[torch.Tensor], _output: torch.Tensor, **_kwargs: object) -> None:
        for idx, local_output in enumerate(local_outputs):
            output_list[idx].copy_(local_output)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    for rank in range(2):
        cp_rank_ref["value"] = rank
        restored = pack_utils.postprocess_packed_seqs(
            local_outputs[rank],
            local_params[rank],
            attention_mask,
            batch_size=1,
            seq_len=4,
            post_process=True,
        )
        torch.testing.assert_close(restored, expected)
