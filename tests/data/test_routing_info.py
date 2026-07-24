"""Tests for routing-handle plumbing in the data pipeline (post-legacy).

After the legacy in-Sample ``routed_experts`` numpy was deleted in favour of
``TensorHandle``s, the only data-side concerns are:

- ``TokenTrace`` builds the right multi-turn token/loss structure.
- ``Sample.routing_handles_per_path`` survives the ``SampleTensorDict`` round-trip.
- A rollout's single-call ``routing_handle`` ends up on
  ``Sample.routing_handles_per_path`` when ``TokenTrace.to_sample`` is used to
  convert ``GenerationOutput`` → ``Sample`` (the pattern that replaced the
  dedicated ``RolloutSampleConverter``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from axrl.data import array_utils
from axrl.data.generation import GenerationInput, GenerationOutput, TensorHandle
from axrl.data.sample import Sample, SampleTensorDict, get_prompt_ids, samples_from_tensor_dict
from axrl.data.token_trace import TokenTrace


class TestTokenTraceStructure:
    def test_basic_full_trace(self) -> None:
        prompt_len = 3
        response_len = 5
        trace = TokenTrace()
        trace.extend_tokens(array_utils.as_i32(list(range(prompt_len))), token_type="init")
        trace.extend_tokens(array_utils.as_i32(list(range(response_len))), token_type="assistant")

        sample = trace.to_sample(max_length=16, pad_token_id=0)
        seq = prompt_len + response_len
        assert sample.input_ids[:seq].tolist() == list(range(prompt_len)) + list(range(response_len))
        assert sample.loss_mask[: prompt_len - 1].tolist() == [False] * (prompt_len - 1)
        assert all(sample.loss_mask[prompt_len - 1 : seq - 1])
        assert sample.routing_handles_per_path is None

    def test_overlength_trace_asserts(self) -> None:
        """Truncation was replaced by a loud assert — silent slicing corrupts R3 replay."""
        trace = TokenTrace()
        trace.extend_tokens(array_utils.as_i32([100, 101, 102]), token_type="init")
        trace.extend_tokens(array_utils.as_i32([200, 201, 202]), token_type="assistant")

        with pytest.raises(AssertionError, match="exceeds max_length"):
            trace.to_sample(max_length=5, pad_token_id=0)

    def test_first_token_must_be_context(self) -> None:
        trace = TokenTrace()
        with pytest.raises(AssertionError, match="first chunk must be token_type='init'"):
            trace.extend_tokens(array_utils.as_i32([1, 2]), token_type="assistant")

    def test_trace_owns_appended_arrays(self) -> None:
        trace = TokenTrace()
        prompt = array_utils.as_i32([1, 2, 3])
        response = array_utils.as_i32([4, 5])
        logprobs = array_utils.as_f32([0.1, 0.2])

        trace.extend_tokens(prompt, token_type="init")
        trace.extend_tokens(response, logprobs=logprobs, token_type="assistant")

        prompt[0] = 99
        response[0] = 88
        logprobs[0] = 9.9

        assert trace.token_ids.tolist() == [1, 2, 3, 4, 5]
        assert trace.token_logprobs[:3].tolist() == [0.0, 0.0, 0.0]
        assert trace.token_logprobs[3:].tolist() == pytest.approx([0.1, 0.2])

    def test_last_turn_sample_only_trains_latest_assistant_chunk(self) -> None:
        h0 = TensorHandle(ref="nodeA:h0")
        h1 = TensorHandle(ref="nodeA:h1")
        trace = TokenTrace()
        trace.extend_tokens(array_utils.as_i32([1, 2, 3]), token_type="init")
        trace.extend_tokens(
            array_utils.as_i32([4, 5]),
            logprobs=array_utils.as_f32([0.1, 0.2]),
            token_type="assistant",
            routing_handle=h0,
        )
        trace.extend_tokens(array_utils.as_i32([6, 7]), token_type="tool_result")
        trace.extend_tokens(
            array_utils.as_i32([8, 9, 10]),
            logprobs=array_utils.as_f32([0.3, 0.4, 0.5]),
            token_type="assistant",
            routing_handle=h1,
        )

        assert trace.token_count == 10
        sample = trace.to_last_turn_sample(max_length=12, pad_token_id=0)

        assert sample.input_ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0]
        assert sample.labels.tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 10, -100, -100, -100]
        assert sample.loss_mask.tolist() == [False, False, False, False, False, False, True, True, True, False, False, False]
        assert sample.turn_index is not None
        assert sample.turn_index.tolist() == [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1]
        assert sample.attention_mask.tolist() == [True, True, True, True, True, True, True, True, True, True, False, False]
        assert sample.routing_handles_per_path == [[h0, h1]]


def test_sample_normalizes_legacy_list_inputs() -> None:
    raw: dict[str, Any] = {
        "input_ids": [1, 2, 3, 4],
        "labels": [2, 3, 4, -100],
        "loss_mask": [False, True, True, False],
        "attention_mask": [True, True, True, True],
        "position_ids": [0, 1, 2, 3],
        "reward": 1.0,
        "reward_baseline": 0.0,
        "advantage": [0.0, 1.0, 1.0, 0.0],
        "rollout_logprobs": [0.0, 0.1, 0.2, 0.0],
    }
    sample = Sample(**raw)

    assert sample.input_ids.dtype == np.int32
    assert sample.loss_mask.dtype == np.bool_
    assert sample.advantage.dtype == np.float32
    assert get_prompt_ids(sample) == [1, 2]

    td = SampleTensorDict.from_samples([sample])
    assert td["advantage"].dtype == torch.float32
    assert td["rollout_logprobs"].dtype == torch.float32


def _make_sample(seq_len: int, *, routing_handles_per_path: list[list[TensorHandle]] | None = None) -> Sample:
    return Sample(
        input_ids=array_utils.as_i32(list(range(seq_len))),
        labels=array_utils.as_i32(list(range(seq_len))),
        loss_mask=array_utils.as_bool([True] * seq_len),
        attention_mask=array_utils.as_bool([True] * seq_len),
        position_ids=array_utils.as_i32(list(range(seq_len))),
        reward=1.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([1.0] * seq_len),
        routing_handles_per_path=routing_handles_per_path,
    )


class TestRoutingHandlesRoundTrip:
    def test_handles_survive_tensorize(self) -> None:
        h0 = [[TensorHandle(ref="nodeA:A0")]]
        h1 = [[TensorHandle(ref="nodeA:B0")], [TensorHandle(ref="nodeA:B0"), TensorHandle(ref="nodeA:B1")]]
        td = SampleTensorDict.from_samples([_make_sample(8, routing_handles_per_path=h0), _make_sample(8, routing_handles_per_path=h1)])
        recovered = samples_from_tensor_dict(td)
        assert recovered[0].routing_handles_per_path == h0
        assert recovered[1].routing_handles_per_path == h1

    def test_no_handles(self) -> None:
        td = SampleTensorDict.from_samples([_make_sample(8) for _ in range(3)])
        assert "routing_handles_per_path" not in td.keys()  # noqa: SIM118
        recovered = samples_from_tensor_dict(td)
        assert all(s.routing_handles_per_path is None for s in recovered)


def _gen_output(output_ids: list[int], *, routing_handle: TensorHandle | None = None) -> GenerationOutput:
    return GenerationOutput(
        session_id="s1",
        output_ids=array_utils.as_i32(output_ids),
        output_logprobs=array_utils.as_f32([0.1] * len(output_ids)),
        output_text="t",
        output_text_with_special_tokens="t",
        cached_tokens=0,
        finish_reason="stop",
        e2e_elapsed_seconds=0.1,
        stop_reason=None,
        retry=0,
        routing_handle=routing_handle,
    )


def _rollout_to_sample(req: GenerationInput, output: GenerationOutput, *, max_length: int, pad_token_id: int) -> Sample:
    """Inlined single-call rollout → Sample conversion.

    Mirrors the logic the former ``RolloutSampleConverter.process`` used to
    run as a Ray worker.
    """
    trace = TokenTrace()
    trace.extend_tokens(req.input_ids, token_type="init")
    assert output.output_logprobs is not None
    assert len(output.output_logprobs) == len(output.output_ids)
    trace.extend_tokens(
        output.output_ids,
        logprobs=output.output_logprobs,
        token_type="assistant",
        routing_handle=output.routing_handle,
    )
    return trace.to_sample(max_length=max_length, pad_token_id=pad_token_id)


class TestRolloutOutputToSampleRouting:
    def test_stamps_handle_when_present(self) -> None:
        gen_input = GenerationInput(session_id="s1", input_ids=array_utils.as_i32([1, 2, 3, 4, 5]))
        handle = TensorHandle(ref="nodeA:stamp-t23")
        gen_output = _gen_output([10, 11, 12], routing_handle=handle)
        sample = _rollout_to_sample(gen_input, gen_output, max_length=32, pad_token_id=0)
        assert sample.routing_handles_per_path == [[handle]]

    def test_no_handle_means_none(self) -> None:
        gen_input = GenerationInput(session_id="s1", input_ids=array_utils.as_i32([1, 2, 3, 4, 5]))
        gen_output = _gen_output([10, 11, 12])
        sample = _rollout_to_sample(gen_input, gen_output, max_length=32, pad_token_id=0)
        assert sample.routing_handles_per_path is None
