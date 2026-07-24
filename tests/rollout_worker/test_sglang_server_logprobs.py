from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from transformers import AutoProcessor

from axrl.configs import ModelConfig, OAIClientConfig, RolloutWorkerConfig, SamplingConfig
from axrl.data import GenerationInput, array_utils
from axrl.data.conversation import Conversation, Message
from axrl.ray import ray_utils
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.runner import CgroupRunner
from axrl.utils.sglang_launch_utils import SGLangServiceHandle, start_sglang_router
from axrl.worker.oai_client import OAICompatibleGenerationClient

if TYPE_CHECKING:
    from collections.abc import Iterator

_TEST_MODEL_NAME = "Qwen/Qwen3-0.6B"
_TEST_SEQ_LENGTH = 2048
_TEST_TP_SIZE = 1
_TEST_GPU_MEMORY_UTILIZATION = 0.45
_SGLANG_PROCESS_PATTERNS = (
    "sglang.launch_server",
    "sglang_router",
    "RemoteSGLangWorker",
    "axrl.worker.sglang_worker",
)


def _sglang_processes() -> list[str]:
    proc = subprocess.run(["/usr/bin/ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True)
    lines = [line.strip() for line in proc.stdout.splitlines()]
    return [line for line in lines if any(pattern in line for pattern in _SGLANG_PROCESS_PATTERNS)]


def _assert_no_sglang_processes(timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    leaked = _sglang_processes()
    while leaked and time.monotonic() < deadline:
        time.sleep(1.0)
        leaked = _sglang_processes()
    assert not leaked, "SGLang server/worker processes leaked after test:\n" + "\n".join(leaked)


@pytest.fixture(autouse=True)
def _assert_sglang_processes_are_cleaned_up() -> Iterator[None]:
    yield
    _assert_no_sglang_processes()


def _skip_if_real_sglang_test_cannot_run(config: RolloutWorkerConfig) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < config.tp_size:
        pytest.skip(f"Need >= {config.tp_size} GPU(s), found {torch.cuda.device_count()}.")
    if not config.model.get_full_path().exists():
        pytest.skip(f"Model path does not exist: {config.model.get_full_path()}.")
    if not CgroupRunner.is_supported():
        pytest.skip("CgroupRunner is not supported on this host.")


def _comparison_config() -> RolloutWorkerConfig:
    return RolloutWorkerConfig(
        model=ModelConfig(name=_TEST_MODEL_NAME, seq_length=_TEST_SEQ_LENGTH, trust_remote_code=True),
        sampling_config=_sampling_config(_TEST_SEQ_LENGTH),
        tp_size=_TEST_TP_SIZE,
        pp_size=1,
        dp_size=1,
        num_workers=1,
        gpu_memory_utilization=_TEST_GPU_MEMORY_UTILIZATION,
        max_running_requests=8,
        enable_metrics=False,
        log_level="warning",
    )


def _sampling_config(seq_length: int) -> SamplingConfig:
    return SamplingConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_total_tokens=min(seq_length, 512),
        max_new_tokens=8,
    )


def _prompt_input_ids(config: RolloutWorkerConfig) -> list[int]:
    processor: Any = AutoProcessor.from_pretrained(config.model.get_full_path(), use_fast=True)
    conversation = Conversation(messages=[Message(role="user", content="Write one short sentence about Beijing.")])
    prompt = processor.apply_chat_template(conversation.to_dict()["messages"], add_generation_prompt=True, tokenize=False)
    return processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()


async def _generate_with_real_sglang_router(
    config: RolloutWorkerConfig,
    *,
    prompt_ids: list[int],
) -> tuple[list[int], np.ndarray]:
    resource_group: ResourceGroup | None = None
    handle: SGLangServiceHandle | None = None
    client: OAICompatibleGenerationClient | None = None
    ray_utils.restart()
    try:
        resource_group = ResourceGroup(requests=[Request(cpu=2, gpu=config.tp_size)])
        handle = await start_sglang_router(resource_group, config)
        client = OAICompatibleGenerationClient(
            OAIClientConfig(
                base_url=handle.base_url,
                sampling_config=config.sampling_config,
                request_timeout_seconds=300.0,
                retry_initial_sleep_seconds=1.0,
                retry_max_sleep_seconds=5.0,
            )
        )
        output = await client.generate(
            GenerationInput(
                session_id="router-sglang-generate",
                input_ids=array_utils.as_i32(prompt_ids),
                sampling_config=config.sampling_config,
            )
        )
        output_ids = output.output_ids.tolist()
        assert output_ids, "SGLang router should generate at least one token."
        assert len(output.output_logprobs) == len(output_ids)
        assert np.isfinite(output.output_logprobs).all()
        return output_ids, output.output_logprobs.copy()
    finally:
        if client is not None:
            await client.close()
        if handle is not None:
            with contextlib.suppress(Exception):
                await handle.terminate()
        if resource_group is not None:
            resource_group.shutdown()
        ray_utils.stop()


async def _score_with_real_sglang_router(
    config: RolloutWorkerConfig,
    *,
    input_ids: list[int],
    start_index: int,
) -> tuple[list[int], np.ndarray]:
    resource_group: ResourceGroup | None = None
    handle: SGLangServiceHandle | None = None
    client: OAICompatibleGenerationClient | None = None
    ray_utils.restart()
    try:
        resource_group = ResourceGroup(requests=[Request(cpu=2, gpu=config.tp_size)])
        handle = await start_sglang_router(resource_group, config)
        client = OAICompatibleGenerationClient(
            OAIClientConfig(
                base_url=handle.base_url,
                sampling_config=config.sampling_config,
                request_timeout_seconds=300.0,
                retry_initial_sleep_seconds=1.0,
                retry_max_sleep_seconds=5.0,
            )
        )
        output = await client.generate(
            GenerationInput(
                session_id="router-sglang-score",
                input_ids=array_utils.as_i32(input_ids),
                sampling_config=config.sampling_config.model_copy(update={"max_new_tokens": 0}),
                capture_input_logprobs=True,
                input_logprob_start_index=start_index,
            )
        )
        assert output.output_ids.tolist() == []
        assert output.input_logprob_token_ids is not None
        assert output.input_logprobs is not None
        assert np.isfinite(output.input_logprobs).all()
        return output.input_logprob_token_ids.tolist(), output.input_logprobs.copy()
    finally:
        if client is not None:
            await client.close()
        if handle is not None:
            with contextlib.suppress(Exception):
                await handle.terminate()
        if resource_group is not None:
            resource_group.shutdown()
        ray_utils.stop()


async def _generate_with_ray_rollout_worker(
    config: RolloutWorkerConfig,
    *,
    prompt_ids: list[int],
) -> tuple[list[int], np.ndarray]:
    resource_group: ResourceGroup | None = None
    worker: Any | None = None
    ray_utils.restart()
    try:
        from axrl.ray.ray_rollout_worker import RayRolloutWorker

        resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=config.tp_size)])
        worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(config, resource_group))
        output = await worker.generate(
            GenerationInput(
                session_id="ray-rollout-generate",
                input_ids=array_utils.as_i32(prompt_ids),
                sampling_config=config.sampling_config,
            )
        )
        output_ids = output.output_ids.tolist()
        assert output_ids, "RayRolloutWorker should generate at least one token."
        assert len(output.output_logprobs) == len(output_ids)
        assert np.isfinite(output.output_logprobs).all()
        return output_ids, output.output_logprobs.copy()
    finally:
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.shutdown()
        if resource_group is not None:
            resource_group.shutdown()
        ray_utils.stop()


async def _score_with_ray_rollout_worker(
    config: RolloutWorkerConfig,
    *,
    input_ids: list[int],
    start_index: int,
) -> tuple[list[int], np.ndarray]:
    resource_group: ResourceGroup | None = None
    worker: Any | None = None
    ray_utils.restart()
    try:
        from axrl.ray.ray_rollout_worker import RayRolloutWorker

        resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=config.tp_size)])
        worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(config, resource_group))
        score_output = await worker.generate(
            GenerationInput(
                session_id="ray-rollout-score",
                input_ids=array_utils.as_i32(input_ids),
                sampling_config=config.sampling_config.model_copy(update={"max_new_tokens": 0}),
                capture_input_logprobs=True,
                input_logprob_start_index=start_index,
            )
        )
        assert score_output.output_ids.tolist() == []
        assert score_output.input_logprob_token_ids is not None
        assert score_output.input_logprobs is not None
        assert np.isfinite(score_output.input_logprobs).all()
        return score_output.input_logprob_token_ids.tolist(), score_output.input_logprobs.copy()
    finally:
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.shutdown()
        if resource_group is not None:
            resource_group.shutdown()
        ray_utils.stop()


def test_real_sglang_router_output_logprobs_match_ray_rollout_worker_input_logprobs() -> None:
    config = _comparison_config()
    _skip_if_real_sglang_test_cannot_run(config)
    prompt_ids = _prompt_input_ids(config)

    async def run() -> None:
        generated_ids, server_output_logprobs = await _generate_with_real_sglang_router(config, prompt_ids=prompt_ids)
        scored_ids = [*prompt_ids, *generated_ids]
        ray_token_ids, ray_input_logprobs = await _score_with_ray_rollout_worker(
            config,
            input_ids=scored_ids,
            start_index=len(prompt_ids),
        )

        assert ray_token_ids == generated_ids
        max_abs_diff = float(np.max(np.abs(server_output_logprobs - ray_input_logprobs)))
        assert max_abs_diff < 5e-2, (
            f"SGLang router output logprobs should match RayRolloutWorker input logprobs for the generated tokens; max_abs_diff={max_abs_diff}"
        )

    asyncio.run(run())


def test_ray_rollout_worker_output_logprobs_match_real_sglang_router_input_logprobs() -> None:
    config = _comparison_config()
    _skip_if_real_sglang_test_cannot_run(config)
    prompt_ids = _prompt_input_ids(config)

    async def run() -> None:
        generated_ids, ray_output_logprobs = await _generate_with_ray_rollout_worker(config, prompt_ids=prompt_ids)
        scored_ids = [*prompt_ids, *generated_ids]
        server_token_ids, server_input_logprobs = await _score_with_real_sglang_router(
            config,
            input_ids=scored_ids,
            start_index=len(prompt_ids),
        )

        assert server_token_ids == generated_ids
        max_abs_diff = float(np.max(np.abs(ray_output_logprobs - server_input_logprobs)))
        assert max_abs_diff < 5e-2, (
            f"RayRolloutWorker output logprobs should match SGLang router input logprobs for the generated tokens; max_abs_diff={max_abs_diff}"
        )

    asyncio.run(run())


def test_ray_rollout_worker_input_logprobs_start_at_requested_token_index() -> None:
    config = _comparison_config()
    _skip_if_real_sglang_test_cannot_run(config)
    prompt_ids = _prompt_input_ids(config)
    assert len(prompt_ids) > 2

    async def run() -> None:
        start_index = 1
        token_ids, input_logprobs = await _score_with_ray_rollout_worker(
            config,
            input_ids=prompt_ids,
            start_index=start_index,
        )

        assert token_ids == prompt_ids[start_index:]
        assert len(input_logprobs) == len(token_ids)
        assert np.isfinite(input_logprobs).all()

    asyncio.run(run())
