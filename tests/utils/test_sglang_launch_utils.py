from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from axrl.configs import OAIClientConfig
from axrl.data import GenerationInput, array_utils
from axrl.ray import ray_utils
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils.network_utils import get_available_port
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


@pytest.fixture(autouse=True)
def _assert_sglang_processes_are_cleaned_up() -> Iterator[None]:
    assert_no_sglang_processes()
    yield
    assert_no_sglang_processes()


def test_start_sglang_router_serves_real_generate_with_two_workers() -> None:
    config = sglang_test_config(num_workers=2)
    skip_if_real_sglang_cannot_run(config, min_gpus=2)
    prompt_ids = prompt_input_ids(config)

    async def run() -> None:
        resource_group: ResourceGroup | None = None
        handle: SGLangServiceHandle | None = None
        client: OAICompatibleGenerationClient | None = None
        ray_utils.restart()
        try:
            router_host = "127.0.0.1"
            router_port = get_available_port()
            resource_group = ResourceGroup(requests=[Request(cpu=4, gpu=2)])
            handle = await start_sglang_router(
                resource_group,
                config,
                router_host=router_host,
                router_port=router_port,
            )
            assert "sglang_router" in handle.command
            assert len(handle.children) == 2
            assert all("sglang.launch_server" in child.command for child in handle.children)
            assert handle.host == router_host
            assert handle.port == router_port
            assert handle.base_url == f"http://{router_host}:{router_port}"

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
                    session_id="real-sglang-router",
                    input_ids=array_utils.as_i32(prompt_ids),
                    sampling_config=config.sampling_config,
                )
            )

            assert output.output_ids.tolist()
            assert len(output.output_logprobs) == len(output.output_ids)
            assert np.isfinite(output.output_logprobs).all()
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
