from __future__ import annotations

import asyncio
from typing import override

import numpy as np
import pytest
from pydantic import ValidationError

from axrl.configs import IGNORE_INDEX, GrpoTrainerConfig, ModelConfig, OAIClientConfig, OPDConfig, RolloutWorkerConfig, SamplingConfig
from axrl.data import Conversation, GenerationInput, GenerationOutput, Message, RolloutResult, Sample, array_utils
from axrl.data.rollout_trace import RolloutTrace
from axrl.metrics.response_metric import ResponseMetric
from axrl.opd.teacher_logprobs import (
    aggregate_teacher_metrics,
    align_input_logprobs_to_sample,
    annotate_sglang_teacher_logprobs,
    get_input_logprob_start_index,
    initialize_local_teacher_oai_client,
)
from axrl.pipeline.config import PipelineExperimentConfig
from axrl.worker.oai_client import OAICompatibleGenerationClient


def _sample() -> Sample:
    input_ids = array_utils.as_i32([10, 11, 12, 13])
    return Sample(
        input_ids=input_ids,
        labels=array_utils.as_i32([11, 12, 13, IGNORE_INDEX]),
        loss_mask=array_utils.as_bool([False, True, True, False]),
        attention_mask=array_utils.as_bool([True, True, True, True]),
        position_ids=array_utils.as_i32([0, 1, 2, 3]),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0, 1.0, 1.0, 0.0]),
    )


def _metric() -> ResponseMetric:
    return ResponseMetric(
        token_count=1,
        token_unique_ratio=1.0,
        word_length_max=1,
        line_length_max=1,
        ngram_repetition=0.0,
        reasoning_behavior_backtracking=0.0,
        reasoning_behavior_verification=0.0,
        reasoning_behavior_causal=0.0,
        rollout_cached_tokens=0,
        rollout_num_retry=0,
        rollout_e2e_elapsed_seconds=0.0,
        rollout_finish_reason_stop=1,
        rollout_finish_reason_length=0,
        rollout_finish_reason_tool_calls=0,
        rollout_finish_reason_function_call=0,
        rollout_finish_reason_content_filter=0,
        score=1.0,
    )


@pytest.mark.parametrize("opd_alpha", [0.0, 0.7])
def test_initialize_local_teacher_oai_client_uses_opd_config(opd_alpha: float) -> None:
    teacher_model = ModelConfig(name="teacher", seq_length=128)
    config = PipelineExperimentConfig(
        train_sampling_config=SamplingConfig(max_total_tokens=128, temperature=0.7),
        grpo=GrpoTrainerConfig(
            opd=OPDConfig(
                enabled=True,
                backend="sglang",
                teacher_model=teacher_model,
                sglang_worker=RolloutWorkerConfig(model=teacher_model),
                sglang_host="127.0.0.1",
                sglang_port=30000,
                opd_alpha=opd_alpha,
            )
        ),
    )

    client = initialize_local_teacher_oai_client(config)

    assert isinstance(client, OAICompatibleGenerationClient)
    assert client.config.base_url == "http://127.0.0.1:30000"
    assert client.config.sampling_config.temperature == pytest.approx(0.7)
    assert client.config.max_connections is None


def test_initialize_local_teacher_oai_client_disabled_returns_none() -> None:
    config = PipelineExperimentConfig()

    assert initialize_local_teacher_oai_client(config) is None


def test_opd_config_validates_sglang_teacher_settings() -> None:
    teacher_model = ModelConfig(name="teacher", seq_length=128)

    with pytest.raises(ValidationError, match="teacher_model"):
        OPDConfig(enabled=True, backend="sglang")

    with pytest.raises(ValidationError, match="must match"):
        OPDConfig(
            enabled=True,
            backend="sglang",
            teacher_model=teacher_model,
            sglang_worker=RolloutWorkerConfig(model=ModelConfig(name="other", seq_length=128)),
            sglang_host="127.0.0.1",
            sglang_port=30000,
        )

    config = OPDConfig(
        enabled=True,
        backend="sglang",
        teacher_model=teacher_model,
        sglang_worker=RolloutWorkerConfig(model=teacher_model),
        sglang_port=30000,
    )
    assert config.sglang_host is None

    with pytest.raises(ValidationError, match="sglang_port"):
        OPDConfig(
            enabled=True,
            backend="sglang",
            teacher_model=teacher_model,
            sglang_worker=RolloutWorkerConfig(model=teacher_model),
        )


def test_opd_config_allows_megatron_teacher_without_sglang_endpoint() -> None:
    teacher_model = ModelConfig(name="teacher", seq_length=128)

    config = OPDConfig(
        enabled=True,
        backend="megatron",
        teacher_model=teacher_model,
        teacher_weight_name="teacher_snapshot",
    )

    assert config.backend == "megatron"
    assert config.teacher_weight_name == "teacher_snapshot"


def test_align_input_logprobs_to_sample_writes_label_aligned_values() -> None:
    sample = _sample()
    output = GenerationOutput(
        session_id="s",
        output_ids=array_utils.as_i32([]),
        output_logprobs=array_utils.as_f32([]),
        output_text="",
        output_text_with_special_tokens="",
        cached_tokens=0,
        finish_reason="stop",
        e2e_elapsed_seconds=0.0,
        stop_reason=None,
        retry=0,
        input_logprobs=array_utils.as_f32([-0.2, -0.3]),
        input_logprob_token_ids=array_utils.as_i32([12, 13]),
        input_logprob_start_index=2,
    )

    assert get_input_logprob_start_index(sample) == 2
    actual = align_input_logprobs_to_sample(sample, output)

    assert actual.tolist() == pytest.approx([0.0, -0.2, -0.3, 0.0])


def test_align_input_logprobs_to_sample_rejects_token_mismatch() -> None:
    sample = _sample()
    output = GenerationOutput(
        session_id="s",
        output_ids=array_utils.as_i32([]),
        output_logprobs=array_utils.as_f32([]),
        output_text="",
        output_text_with_special_tokens="",
        cached_tokens=0,
        finish_reason="stop",
        e2e_elapsed_seconds=0.0,
        stop_reason=None,
        retry=0,
        input_logprobs=array_utils.as_f32([-0.2]),
        input_logprob_token_ids=array_utils.as_i32([99]),
        input_logprob_start_index=2,
    )

    with pytest.raises(AssertionError, match="do not match"):
        align_input_logprobs_to_sample(sample, output)


def test_annotate_sglang_teacher_logprobs_requests_zero_new_tokens() -> None:
    requests: list[GenerationInput] = []

    class RecordingClient(OAICompatibleGenerationClient):
        def __init__(self) -> None:
            self.config = OAIClientConfig(base_url="http://127.0.0.1:30000", sampling_config=SamplingConfig(max_total_tokens=64))

        @override
        async def generate(self, req: GenerationInput) -> GenerationOutput:
            requests.append(req)
            token_ids = array_utils.to_int_list(req.input_ids[req.input_logprob_start_index :])
            return GenerationOutput(
                session_id=req.session_id,
                output_ids=array_utils.as_i32([]),
                output_logprobs=array_utils.as_f32([]),
                output_text="",
                output_text_with_special_tokens="",
                cached_tokens=0,
                finish_reason="stop",
                e2e_elapsed_seconds=0.0,
                stop_reason=None,
                retry=0,
                input_logprobs=np.asarray([-0.1] * len(token_ids), dtype=np.float32),
                input_logprob_token_ids=array_utils.as_i32(token_ids),
                input_logprob_start_index=req.input_logprob_start_index,
            )

    conv = Conversation(
        conversation_id="conv",
        messages=[Message(role="user", content="q")],
        extra={"group_id": "g"},
    )
    trace = RolloutTrace(conv, token_in_token_out=False)
    trace.turn_samples = [_sample()]
    result = RolloutResult(conversation=conv, trace=trace, metric=_metric())

    metrics = asyncio.run(annotate_sglang_teacher_logprobs(result, RecordingClient()))

    assert requests
    assert requests[0].capture_input_logprobs
    assert requests[0].input_logprob_start_index == 2
    assert requests[0].sampling_config is not None
    assert requests[0].sampling_config.max_new_tokens == 0
    assert result.trace is not None
    assert result.trace.turn_samples[0].teacher_logprobs is not None
    assert result.trace.turn_samples[0].teacher_logprobs.tolist() == pytest.approx([0.0, -0.1, -0.1, 0.0])
    assert metrics["opd/teacher_logprob_tokens"] == 2.0


def test_aggregate_teacher_metrics_averages_numeric_values() -> None:
    results = [
        RolloutResult(
            conversation=Conversation(
                conversation_id="conv-1",
                messages=[Message(role="user", content="q")],
                extra={"teacher_metrics": {"opd/a": 1.0, "opd/b": 3, "text": "skip"}},
            ),
            trace=None,
            metric=_metric(),
        ),
        RolloutResult(
            conversation=Conversation(
                conversation_id="conv-2",
                messages=[Message(role="user", content="q")],
                extra={"teacher_metrics": {"opd/a": 3.0}},
            ),
            trace=None,
            metric=_metric(),
        ),
        RolloutResult(
            conversation=Conversation(conversation_id="conv-3", messages=[Message(role="user", content="q")]),
            trace=None,
            metric=_metric(),
        ),
    ]

    assert aggregate_teacher_metrics(results) == {"opd/a": 2.0, "opd/b": 3.0}
