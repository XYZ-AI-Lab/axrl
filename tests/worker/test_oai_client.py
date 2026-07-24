from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from axrl.configs import OAIClientConfig
from axrl.data import GenerationInput, GenerationOutput, array_utils
from axrl.ray import ray_utils
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils.sglang_launch_utils import SGLangServiceHandle, start_sglang_router
from axrl.worker.oai_client import OAICompatibleGenerationClient
from tests.utils.sglang_integration_utils import (
    assert_no_sglang_processes,
    prompt_input_ids,
    sglang_test_config,
    skip_if_real_sglang_cannot_run,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


@pytest.fixture(autouse=True)
def _assert_sglang_processes_are_cleaned_up() -> Iterator[None]:
    assert_no_sglang_processes()
    yield
    assert_no_sglang_processes()


def _empty_generation_output(req: GenerationInput, *, retry: int) -> GenerationOutput:
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
        retry=retry,
        event_timing=req.event_timing,
    )


def test_oai_client_uses_fresh_backend_rid_per_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OAICompatibleGenerationClient(
        OAIClientConfig(
            base_url="http://127.0.0.1:30000",
            retry_initial_sleep_seconds=0.0,
            retry_max_sleep_seconds=0.0,
        )
    )
    seen_payloads: list[dict[str, Any]] = []

    async def request_once(*, req: GenerationInput, payload: dict[str, Any], retry: int) -> GenerationOutput:
        seen_payloads.append(payload)
        if retry == 0:
            raise RuntimeError("force retry")
        return _empty_generation_output(req, retry=retry)

    monkeypatch.setattr(client, "_request_once", request_once)
    req = GenerationInput(session_id="shared-session", input_ids=array_utils.as_i32([1, 2, 3]))

    output = asyncio.run(client.generate(req))

    rids = [str(payload["rid"]) for payload in seen_payloads]
    assert output.session_id == "shared-session"
    assert rids != ["shared-session", "shared-session"]
    assert len(rids) == 2
    assert len(set(rids)) == 2
    assert all(rid.startswith("sglang-") for rid in rids)
    assert all("session_id" not in payload for payload in seen_payloads)


def test_oai_client_uses_fresh_backend_rid_for_concurrent_same_session(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OAICompatibleGenerationClient(OAIClientConfig(base_url="http://127.0.0.1:30000"))
    seen_payloads: list[dict[str, Any]] = []

    async def request_once(*, req: GenerationInput, payload: dict[str, Any], retry: int) -> GenerationOutput:
        await asyncio.sleep(0)
        seen_payloads.append(payload)
        return _empty_generation_output(req, retry=retry)

    async def run() -> list[GenerationOutput]:
        monkeypatch.setattr(client, "_request_once", request_once)
        reqs = [
            GenerationInput(session_id="shared-session", input_ids=array_utils.as_i32([1, 2, 3])),
            GenerationInput(session_id="shared-session", input_ids=array_utils.as_i32([4, 5, 6])),
        ]
        return list(await asyncio.gather(*(client.generate(req) for req in reqs)))

    outputs = asyncio.run(run())

    rids = [str(payload["rid"]) for payload in seen_payloads]
    assert [output.session_id for output in outputs] == ["shared-session", "shared-session"]
    assert len(rids) == 2
    assert len(set(rids)) == 2
    assert all(rid != "shared-session" for rid in rids)


def test_oai_client_generates_and_scores_against_real_sglang_router() -> None:
    config = sglang_test_config()
    skip_if_real_sglang_cannot_run(config)
    prompt_ids = prompt_input_ids(config)

    async def run() -> None:
        resource_group: ResourceGroup | None = None
        handle: SGLangServiceHandle | None = None
        client: OAICompatibleGenerationClient | None = None
        ray_utils.restart()
        try:
            resource_group = ResourceGroup(requests=[Request(cpu=2, gpu=1)])
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

            generation = await client.generate(
                GenerationInput(
                    session_id="real-oai-client-generate",
                    input_ids=array_utils.as_i32(prompt_ids),
                    sampling_config=config.sampling_config,
                )
            )
            generated_ids = generation.output_ids.tolist()
            assert generated_ids
            assert generation.output_logprobs.shape == generation.output_ids.shape
            assert np.isfinite(generation.output_logprobs).all()

            scored_ids = [*prompt_ids, *generated_ids]
            score = await client.generate(
                GenerationInput(
                    session_id="real-oai-client-score",
                    input_ids=array_utils.as_i32(scored_ids),
                    sampling_config=config.sampling_config.model_copy(update={"max_new_tokens": 0}),
                    capture_input_logprobs=True,
                    input_logprob_start_index=len(prompt_ids),
                )
            )

            assert score.output_ids.tolist() == []
            assert score.output_logprobs.tolist() == []
            assert score.input_logprob_token_ids is not None
            assert score.input_logprobs is not None
            assert score.input_logprob_token_ids.tolist() == generated_ids
            assert len(score.input_logprobs) == len(generated_ids)
            assert np.isfinite(score.input_logprobs).all()
            assert score.input_logprob_start_index == len(prompt_ids)
        finally:
            if client is not None:
                await client.close()
            if handle is not None:
                with contextlib.suppress(Exception):
                    await handle.terminate()
            if resource_group is not None:
                resource_group.shutdown()
            ray_utils.stop()

    asyncio.run(run())
