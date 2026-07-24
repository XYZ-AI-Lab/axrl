from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np

from axrl.configs import OAIClientConfig
from axrl.data import GenerationInput, GenerationOutput, RolloutResult, Sample, array_utils
from axrl.worker.oai_client import OAICompatibleGenerationClient

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from axrl.pipeline.config import PipelineExperimentConfig


def initialize_local_teacher_oai_client(config: PipelineExperimentConfig) -> OAICompatibleGenerationClient | None:
    """Create the actor-local SGLang teacher client when OPD needs one."""
    opd = config.grpo.opd
    if not opd.enabled or opd.backend != "sglang":
        return None

    assert opd.sglang_host is not None
    assert opd.sglang_port is not None
    return OAICompatibleGenerationClient(
        OAIClientConfig(
            base_url=f"http://{opd.sglang_host}:{opd.sglang_port}",
            sampling_config=config.train_sampling_config,
            max_connections=None,
        )
    )


def get_input_logprob_start_index(sample: Sample) -> int:
    """Return the first input-token index whose label-aligned logprob is trainable."""
    trainable_label_indices = np.flatnonzero(sample.loss_mask)
    assert len(trainable_label_indices) > 0, "OPD teacher logprob annotation requires at least one trainable label."
    # Label position i is the logprob for input token i + 1 in causal LM training.
    start_index = int(trainable_label_indices[0]) + 1
    real_input_len = int(sample.attention_mask.sum())
    assert 0 < start_index <= real_input_len, (
        f"input_logprob_start_index={start_index} must be in (0, {real_input_len}] for non-empty trainable labels."
    )
    return start_index


def align_input_logprobs_to_sample(sample: Sample, output: GenerationOutput) -> NDArray[np.float32]:
    """Convert input-token logprobs into the sample's label-aligned layout."""
    assert output.input_logprobs is not None, "Teacher GenerationOutput is missing input_logprobs."
    assert output.input_logprob_token_ids is not None, "Teacher GenerationOutput is missing input_logprob_token_ids."
    assert output.input_logprob_start_index is not None, "Teacher GenerationOutput is missing input_logprob_start_index."

    real_input_len = int(sample.attention_mask.sum())
    input_ids = array_utils.to_int_list(sample.input_ids[:real_input_len])
    start_index = output.input_logprob_start_index
    expected_token_ids = input_ids[start_index:]
    actual_token_ids = array_utils.to_int_list(output.input_logprob_token_ids)
    assert actual_token_ids == expected_token_ids, (
        f"teacher input logprob token ids {actual_token_ids} do not match input ids slice {expected_token_ids}."
    )
    assert len(output.input_logprobs) == len(expected_token_ids), (
        f"teacher input_logprobs length {len(output.input_logprobs)} does not match expected token count {len(expected_token_ids)}."
    )

    teacher_logprobs = np.zeros_like(sample.advantage, dtype=np.float32)
    # SGLang returns logprobs for input_ids[start_index:]; each input token k
    # predicts the label at position k - 1, which should be exactly the
    # trainable assistant-token span for this turn sample.
    label_positions = np.arange(start_index - 1, start_index - 1 + len(output.input_logprobs))
    assert sample.loss_mask[label_positions].all(), "Teacher input logprob window must cover only trainable labels."
    teacher_logprobs[label_positions] = output.input_logprobs
    assert np.isfinite(teacher_logprobs[sample.loss_mask]).all(), "Teacher logprobs must be finite over trainable labels."
    return teacher_logprobs


async def annotate_sglang_teacher_logprobs(
    result: RolloutResult,
    oai_client: OAICompatibleGenerationClient,
) -> dict[str, float]:
    """Annotate all turn samples in one rollout result with SGLang teacher logprobs."""
    assert result.trace is not None, "OPD teacher annotation requires RolloutResult.trace."
    start_time = time.perf_counter()
    retries = 0
    token_count = 0

    async def annotate_sample(sample_index: int, sample: Sample) -> None:
        nonlocal retries, token_count
        start_index = get_input_logprob_start_index(sample)

        real_input_len = int(sample.attention_mask.sum())
        logprob_input_ids = sample.input_ids[:real_input_len]
        token_count += int(real_input_len - start_index)
        output = await oai_client.generate(
            GenerationInput(
                session_id=f"{result.conversation.gen_state.session_id or result.conversation.conversation_id}:opd:{sample_index}",
                input_ids=logprob_input_ids,
                sampling_config=oai_client.config.sampling_config.model_copy(update={"max_new_tokens": 0}),
                capture_input_logprobs=True,
                input_logprob_start_index=start_index,
            )
        )
        retries += output.retry
        sample.teacher_logprobs = align_input_logprobs_to_sample(sample, output)

    await asyncio.gather(*(annotate_sample(index, sample) for index, sample in enumerate(result.trace.turn_samples)))
    elapsed_seconds = time.perf_counter() - start_time
    return {
        "opd/teacher_logprob_seconds": elapsed_seconds,
        "opd/teacher_logprob_tokens": float(token_count),
        "opd/teacher_logprob_tokens_per_second": (token_count / elapsed_seconds if elapsed_seconds > 0 else 0.0),
        "opd/teacher_logprob_retries": float(retries),
        "opd/teacher_backend_sglang": 1.0,
    }


def aggregate_teacher_metrics(results: Sequence[RolloutResult]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        raw_metrics = result.conversation.extra.get("teacher_metrics")
        if not isinstance(raw_metrics, dict):
            continue
        for key, value in raw_metrics.items():
            if isinstance(value, int | float):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}
