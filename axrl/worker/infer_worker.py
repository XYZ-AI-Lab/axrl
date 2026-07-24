import asyncio
import logging
from collections.abc import AsyncGenerator, Sequence

from axrl.worker.worker import Worker

logger = logging.getLogger(__name__)


class InferWorker[InT, OutT](Worker):
    """General base worker for inference tasks.

    Inference engines (like SGLang), reward models, and tools should subclass InferWorker.
    Different inference workers can be implemented for specific tasks (e.g., VLMInferWorker, LLMInferWorker).
    This class remains general—`InT`/`OutT` let type checkers enforce request/response shapes without
    constraining runtime flexibility.
    """

    def __init__(self) -> None:
        super().__init__()

    async def generate(self, req: InT) -> OutT:
        """Generate a response for a single inference request."""
        raise NotImplementedError

    async def batch_generate(self, reqs: Sequence[InT]) -> Sequence[OutT]:
        """Generate responses for multiple inference requests concurrently.

        Executes all requests in parallel using asyncio.gather, supporting
        partial rollout strategy with model updates.

        """
        if not reqs:
            return []

        tasks = [self.generate(req) for req in reqs]
        return await asyncio.gather(*tasks)

    async def stream_conversation_results(
        self,
        reqs: Sequence[InT],
        max_concurrent_requests: int = 512,
        results_per_chunk: int = -1,
    ) -> AsyncGenerator[Sequence[tuple[int, OutT]], None]:
        """Generate responses for inference requests with streaming results and concurrency control.

        Processes requests with limited concurrency, yielding results in chunks
        as they complete. Useful for large batches to control memory usage and
        provide progressive results.

        Args:
            reqs: List of inference requests to process.
            max_concurrent_requests: Maximum number of concurrent generation requests.
            results_per_chunk: Number of results per yielded chunk, -1 for all at once.

        Yields:
            Chunks of completed inference results.
        """
        if not reqs:
            return

        if results_per_chunk <= 0:
            results_per_chunk = len(reqs)

        reqs = list(reqs)
        pending_reqs: dict[int, InT] = dict(enumerate(reqs))
        accumulated_results: list[tuple[int, OutT]] = []
        completed_count = 0
        total_count = len(reqs)
        active_tasks: list[asyncio.Task[OutT]] = []

        while completed_count < total_count:
            # Create new tasks up to concurrency limit
            slots_available = max_concurrent_requests - len(active_tasks)
            if slots_available > 0 and pending_reqs:
                num_new_tasks = min(slots_available, len(pending_reqs))
                for _ in range(num_new_tasks):
                    req_index, req = pending_reqs.popitem()
                    task = asyncio.create_task(self.generate(req))
                    task.req_index = req_index  # type: ignore[attr-defined]
                    active_tasks.append(task)

            # Wait for at least one task to complete
            done_tasks, pending_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            active_tasks = list(pending_tasks)

            # Process completed tasks
            for task in done_tasks:
                result = task.result()
                req_index = task.req_index  # type: ignore[attr-defined]
                accumulated_results.append((req_index, result))
                completed_count += 1

                if len(accumulated_results) == results_per_chunk or completed_count == total_count:
                    yield accumulated_results
                    accumulated_results = []

    async def stream_conversation_group_results(
        self,
        req_groups: Sequence[Sequence[InT]],
        max_concurrent_requests: int = 512,
        groups_per_chunk: int = -1,
    ) -> AsyncGenerator[Sequence[tuple[int, Sequence[OutT]]], None]:
        """Generate responses for groups of inference requests with streaming results.

        Processes groups of requests with concurrency control based on total
        request count across all groups. Yields completed groups in chunks.

        Args:
            req_groups: List of groups of inference requests to process.
            groups_per_chunk: Number of groups per yielded chunk, -1 for all at once.
            max_concurrent_requests: Maximum total requests processing concurrently.

        Yields:
            Chunks of completed group results (list of lists).
        """
        if not req_groups:
            return

        if groups_per_chunk <= 0:
            groups_per_chunk = len(req_groups)

        req_groups = list(req_groups)
        pending_groups: dict[int, Sequence[InT]] = dict(enumerate(req_groups))
        accumulated_results: list[tuple[int, Sequence[OutT]]] = []
        completed_count = 0
        total_count = len(req_groups)
        active_tasks: list[asyncio.Task[Sequence[OutT]]] = []
        current_request_count = 0

        while completed_count < total_count:
            # Create new tasks up to concurrency limit
            new_group_ids: list[int] = []
            for group_id, group in list(pending_groups.items()):
                if current_request_count + len(group) <= max_concurrent_requests:
                    new_group_ids.append(group_id)
                    current_request_count += len(group)
                if current_request_count >= max_concurrent_requests:
                    break
            for group_id in new_group_ids:
                group = pending_groups.pop(group_id)
                task = asyncio.create_task(self.batch_generate(group))
                task.group_id = group_id  # type: ignore[attr-defined]
                active_tasks.append(task)

            # Wait for at least one task to complete
            done_tasks, pending_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            active_tasks = list(pending_tasks)

            # Process completed tasks
            for task in done_tasks:
                result = task.result()
                group_id = task.group_id  # type: ignore[attr-defined]
                accumulated_results.append((group_id, result))
                completed_count += 1
                current_request_count -= len(result)
                # Yield chunk if ready
                if len(accumulated_results) == groups_per_chunk or completed_count == total_count:
                    yield accumulated_results
                    accumulated_results = []

    async def release_gpu_memory(self, *, backup_weights_on_cpu: bool = True) -> None:
        raise NotImplementedError

    async def resume_gpu_memory(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    async def is_gpu_memory_released(self) -> bool:
        raise NotImplementedError

    async def flush_cache(self) -> None:
        pass
