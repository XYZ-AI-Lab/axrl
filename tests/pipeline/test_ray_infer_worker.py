from __future__ import annotations

import asyncio
from typing import override

from axrl.processor.base_processor import BaseProcessor
from axrl.ray import ray_utils
from axrl.ray.ray_infer_worker import RayInferWorker


class EchoProcessor(BaseProcessor[str, str]):
    @override
    def process(self, item: str) -> str:
        return f"{self.config}:{item}"


def test_ray_infer_worker_processes_single_and_batch_requests() -> None:
    ray_utils.restart()

    async def run() -> None:
        worker = RayInferWorker[str, str](
            RayInferWorker.initialize_remote_actor(
                EchoProcessor,
                config="echo",
                num_processes=2,
                num_cpus=2,
                timeout_seconds=60,
            )
        )
        try:
            assert await worker.generate("a") == "echo:a"
            assert await worker.batch_generate(["c", "d"]) == ["echo:c", "echo:d"]
        finally:
            worker.shutdown()

    try:
        asyncio.run(run())
    finally:
        ray_utils.stop()
