from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

import ray

from axrl.processor.processor_pool import ProcessorPool
from axrl.worker.infer_router import InferenceRouter

if TYPE_CHECKING:
    from axrl.processor.base_processor import BaseProcessor
    from axrl.ray.resource_group import ResourceGroup

logger = logging.getLogger(__name__)


@ray.remote
class RemoteProcessorPool[InT, OutT](ProcessorPool[InT, OutT]):
    pass


class RayCPUWorker[InT, OutT](InferenceRouter[InT, OutT]):
    """CPU-based worker group with router for multiple `ProcessorPool` instances."""

    def __init__(
        self,
        processor_cls: type[BaseProcessor[InT, OutT]],
        config: Any,
        resource_group: ResourceGroup,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__()
        self.processor_cls = processor_cls
        self.config = config
        self.resource_group = resource_group
        self.num_workers = len(resource_group.bundle_infos)
        self.timeout_seconds = timeout_seconds

    async def generate(self, req: InT) -> OutT:
        session_id = getattr(req, "session_id", None)
        worker_index = await self._assign_worker_index(session_id)
        logger.debug(f"Assigned worker {worker_index} for session {session_id}, workloads: {self._workloads}")
        try:
            return await self._remote_workers[worker_index].generate.remote(req)
        finally:
            await self._on_conversations_finish(session_id, worker_index)
            logger.debug(f"Finished worker {worker_index} for session {session_id}, workloads: {self._workloads}")

    @override
    def initialize(self) -> None:
        super().initialize()
        workers: list[RemoteProcessorPool[InT, OutT]] = []
        for i in range(self.num_workers):
            worker = self._start_remote_worker(i)
            workers.append(worker)
        self._set_workers(workers)  # type: ignore[arg-type]

    def _start_remote_worker(self, bundle_index: int) -> RemoteProcessorPool[InT, OutT]:
        num_cpus = self.resource_group.requests[bundle_index].cpu
        assert num_cpus > 0
        worker = RemoteProcessorPool.options(
            scheduling_strategy=self.resource_group.get_scheduling_strategy(bundle_index),
        ).remote(self.processor_cls, self.config, num_cpus, self.timeout_seconds)
        return worker
