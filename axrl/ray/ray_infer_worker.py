from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast, override

import ray

from axrl.processor.processor_pool import ProcessorPool
from axrl.worker.infer_worker import InferWorker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ray.actor import ActorHandle

    from axrl.processor.base_processor import BaseProcessor


@ray.remote
class RemoteRayInferWorker[InT, OutT]:
    def __init__(
        self,
        processor_cls: type[BaseProcessor[InT, OutT]],
        config: Any,
        num_processes: int,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.process_pool: ProcessorPool[InT, OutT] = ProcessorPool(
            processor_cls,
            config=config,
            num_processors=num_processes,
            timeout_seconds=timeout_seconds,
        )

    async def generate(self, req: InT) -> OutT:
        return await self.process_pool.generate(req)

    async def batch_generate(self, reqs: Sequence[InT]) -> Sequence[OutT]:
        return await self.process_pool.batch_generate(reqs)

    def shutdown(self) -> None:
        self.process_pool.shutdown()


class RayInferWorker[InT, OutT](InferWorker[InT, OutT]):
    """Typed wrapper around a Ray actor that owns a local ``ProcessorPool``."""

    def __init__(self, remote_actor: ActorHandle) -> None:
        super().__init__()
        self._remote_actor = remote_actor

    @override
    async def generate(self, req: InT) -> OutT:
        return cast("OutT", await self._remote_actor.generate.remote(req))

    @override
    async def batch_generate(self, reqs: Sequence[InT]) -> Sequence[OutT]:
        return cast("Sequence[OutT]", await self._remote_actor.batch_generate.remote(reqs))

    @override
    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            ray.get(self._remote_actor.shutdown.remote())
        with contextlib.suppress(Exception):
            ray.kill(self._remote_actor, no_restart=True)

    @staticmethod
    def initialize_remote_actor(
        processor_cls: type[BaseProcessor[InT, OutT]],
        config: Any,
        *,
        num_processes: int,
        num_cpus: int | None = None,
        timeout_seconds: float = 60.0,
    ) -> ActorHandle:
        assert num_processes > 0, "num_processes must be greater than zero."
        if num_cpus is None:
            num_cpus = num_processes
        assert num_cpus > 0, "num_cpus must be greater than zero."
        return cast(
            "ActorHandle",
            RemoteRayInferWorker.options(num_cpus=num_cpus).remote(  # type: ignore[attr-defined]
                processor_cls,
                config,
                num_processes,
                timeout_seconds,
            ),
        )
