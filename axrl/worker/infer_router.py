from __future__ import annotations

import asyncio
import logging
from typing import TypeVar, override

import numpy as np
import ray

from axrl.data.event_timing import EventTiming
from axrl.ray import ray_utils
from axrl.utils.timer import SessionTimer
from axrl.worker.infer_worker import InferWorker

InT = TypeVar("InT")
OutT = TypeVar("OutT")

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CACHED_SESSION_IDS = 10_000
_EVENT_TIMING_WARNING_THRESHOLD_SECONDS = 3.0


def _get_event_timing(obj: object) -> EventTiming | None:
    event_timing = getattr(obj, "event_timing", None)
    return event_timing if isinstance(event_timing, EventTiming) else None


def _warn_if_event_timing_slow(session_id: str, event_timing: EventTiming) -> None:
    driver_worker_overhead_seconds = event_timing.driver_worker_overhead_seconds
    if driver_worker_overhead_seconds is not None and driver_worker_overhead_seconds > _EVENT_TIMING_WARNING_THRESHOLD_SECONDS:
        logger.warning(
            "Generation routing overhead is %.3fs for session_id=%s; schedule_to_driver=%.3fs, worker_runtime=%.3fs, timing=%s",
            driver_worker_overhead_seconds,
            session_id,
            event_timing.schedule_to_driver_seconds or 0.0,
            event_timing.worker_runtime_seconds or 0.0,
            event_timing,
        )


@ray.remote
class RemoteInferWorker(InferWorker[InT, OutT]):
    pass


class InferenceRouter(InferWorker[InT, OutT]):
    """Routes general inference requests to a pool of remote workers with load balancing.

    This router is agnostic to the request and result types, supporting any inference scenario.
    """

    def __init__(self, max_imbalance: int = 4, max_cached_session_ids: int = _DEFAULT_MAX_CACHED_SESSION_IDS) -> None:
        super().__init__()
        self.max_imbalance = max_imbalance
        self.max_cached_session_ids = max_cached_session_ids
        self._workload_update_lock = asyncio.Lock()

    def _set_workers(self, remote_workers: list[RemoteInferWorker[InT, OutT]]) -> None:
        self._remote_workers = remote_workers
        self._num_workers = len(remote_workers)

        # Active counts drive load balancing; the session-worker map keeps idle
        # affinity so later turns in the same rollout can reuse worker KV cache.
        self._workloads: np.ndarray = np.zeros(self._num_workers, dtype=int)
        self._session_worker: dict[str, int] = {}  # session id -> worker_index
        self._session_count: dict[str, int] = {}  # active request count by cached session id

    @override
    async def generate(self, req: InT) -> OutT:
        """Generate a response with load balancing across the worker pool.

        Requests with the same session_id are routed to the same worker to leverage
        cache, with `self.max_imbalance` to avoid overloading a single worker.

        Args:
            req: The inference request. Rollout requests must provide a non-empty `session_id`.

        Returns:
            The result of the inference (type Any).
        """
        session_id = getattr(req, "session_id", None)
        assert isinstance(session_id, str) and session_id, "Request must set a non-empty session_id before routing."
        worker_index = await self._assign_worker_index(session_id)
        event_timing = _get_event_timing(req)
        owns_event_timing = event_timing is not None and event_timing.scheduled_at is None
        if event_timing is not None:
            if owns_event_timing:
                event_timing.mark_scheduled()
        logger.debug(f"Assigned worker {worker_index} for session {session_id}, workloads: {self._workloads}")
        with SessionTimer(session_id, "async", "Router: await rollout worker"):
            result: OutT = await self._remote_workers[worker_index].generate.remote(req)
        event_timing = _get_event_timing(result)
        if event_timing is not None and owns_event_timing:
            event_timing.mark_driver_received()
            _warn_if_event_timing_slow(session_id, event_timing)
        await self._on_conversations_finish(session_id, worker_index)
        logger.debug(f"Finished worker {worker_index} for session {session_id}, workloads: {self._workloads}")
        return result

    @override
    def shutdown(self) -> None:
        refs = [worker.shutdown.remote() for worker in self._remote_workers]
        ray.get(refs)
        ray_utils.kill_remote_workers(self._remote_workers)
        self._remote_workers = []
        # logger.info("All remote inference worker engines have been shut down and killed.")

    @override
    async def release_gpu_memory(self, *, backup_weights_on_cpu: bool = True) -> None:
        handles = [worker.release_gpu_memory.remote(backup_weights_on_cpu=backup_weights_on_cpu) for worker in self._remote_workers]
        await asyncio.gather(*handles)
        await self.clear_session_worker_mapping()

    @override
    async def resume_gpu_memory(self, tags: list[str] | None = None) -> None:
        handles = [worker.resume_gpu_memory.remote(tags=tags) for worker in self._remote_workers]
        await asyncio.gather(*handles)

    @override
    async def flush_cache(self) -> None:
        handles = [worker.flush_cache.remote() for worker in self._remote_workers]
        await asyncio.gather(*handles)
        await self.clear_session_worker_mapping()

    async def _assign_worker_index(self, session_id: str | None) -> int:
        async with self._workload_update_lock:
            min_workload_index = int(np.argmin(self._workloads))
            if not session_id:
                self._workloads[min_workload_index] += 1
                return min_workload_index

            # update active session count
            if session_id not in self._session_count:
                self._session_count[session_id] = 0
            self._session_count[session_id] += 1

            # assign worker, preferring cached affinity when load balance permits it
            if session_id in self._session_worker:
                prefer_index = self._session_worker[session_id]
                if self._workloads[prefer_index] < self._workloads[min_workload_index] + self.max_imbalance:
                    self._workloads[prefer_index] += 1
                    return prefer_index
            self._session_worker[session_id] = min_workload_index
            self._workloads[min_workload_index] += 1
            self._cleanup_idle_session_worker_mapping_if_needed()
            return min_workload_index

    async def _on_conversations_finish(self, session_id: str | None, worker_index: int) -> None:
        async with self._workload_update_lock:
            self._workloads[worker_index] -= 1
            if not session_id:
                return
            self._session_count[session_id] -= 1

    async def clear_session_worker_mapping(self) -> None:
        """Clear idle session affinity while preserving active request counts."""
        async with self._workload_update_lock:
            self._drop_idle_session_worker_mapping()

    def _cleanup_idle_session_worker_mapping_if_needed(self) -> None:
        if self.max_cached_session_ids < 0 or len(self._session_worker) < self.max_cached_session_ids:
            return
        old_size = len(self._session_worker)
        self._drop_idle_session_worker_mapping()
        logger.warning(
            "Session-worker affinity mapping reached max_cached_session_ids=%d; "
            "cleaned idle session mappings from %d to %d entries with %d active sessions.",
            self.max_cached_session_ids,
            old_size,
            len(self._session_worker),
            len(self._session_count),
        )

    def _drop_idle_session_worker_mapping(self) -> None:
        self._session_worker = {
            session_id: worker_index for session_id, worker_index in self._session_worker.items() if self._session_count.get(session_id, 0) > 0
        }
        self._session_count = {session_id: count for session_id, count in self._session_count.items() if count > 0}
