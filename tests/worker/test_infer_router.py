import asyncio
from dataclasses import dataclass, field
from typing import Any, cast

from axrl.data import EventTiming
from axrl.worker.infer_router import InferenceRouter


@dataclass
class _TimedRequest:
    session_id: str
    event_timing: EventTiming = field(default_factory=EventTiming)


@dataclass
class _TimedResponse:
    session_id: str
    event_timing: EventTiming


class _TimedGenerateMethod:
    async def remote(self, req: _TimedRequest) -> _TimedResponse:
        req.event_timing.mark_worker_received(102.0)
        req.event_timing.mark_worker_returned(104.0)
        return _TimedResponse(session_id=req.session_id, event_timing=req.event_timing)


class _TimedRemoteWorker:
    def __init__(self) -> None:
        self.generate = _TimedGenerateMethod()


def _make_router(*, max_cached_session_ids: int = 10) -> InferenceRouter[object, object]:
    router: InferenceRouter[object, object] = InferenceRouter(max_imbalance=1, max_cached_session_ids=max_cached_session_ids)
    router._set_workers(cast("Any", [object(), object()]))
    return router


def test_session_worker_affinity_survives_finished_turn() -> None:
    async def run() -> None:
        router = _make_router()

        filler_worker = await router._assign_worker_index("filler")
        assert filler_worker == 0
        sticky_worker = await router._assign_worker_index("sticky")
        assert sticky_worker == 1

        await router._on_conversations_finish("sticky", sticky_worker)
        await router._on_conversations_finish("filler", filler_worker)
        assert router._workloads.tolist() == [0, 0]

        next_sticky_worker = await router._assign_worker_index("sticky")
        assert next_sticky_worker == sticky_worker

    asyncio.run(run())


def test_idle_session_worker_affinity_is_batch_cleaned_when_over_limit() -> None:
    async def run() -> None:
        router = _make_router(max_cached_session_ids=2)

        for session_id in ["s1", "s2", "s3"]:
            worker_index = await router._assign_worker_index(session_id)
            await router._on_conversations_finish(session_id, worker_index)

        assert router._session_worker == {"s3": 0}
        assert router._session_count == {"s3": 0}

    asyncio.run(run())


def test_finished_sessions_are_kept_until_next_over_limit_cleanup() -> None:
    async def run() -> None:
        router = _make_router(max_cached_session_ids=1)

        s1_worker = await router._assign_worker_index("s1")
        s2_worker = await router._assign_worker_index("s2")
        assert set(router._session_worker) == {"s1", "s2"}

        await router._on_conversations_finish("s1", s1_worker)
        assert set(router._session_worker) == {"s1", "s2"}

        s3_worker = await router._assign_worker_index("s3")
        assert set(router._session_worker) == {"s2", "s3"}

        await router._on_conversations_finish("s2", s2_worker)
        await router._on_conversations_finish("s3", s3_worker)
        assert set(router._session_worker) == {"s2", "s3"}
        assert router._session_count == {"s2": 0, "s3": 0}

    asyncio.run(run())


def test_cache_lifecycle_clears_session_affinity_without_dropping_active_counts() -> None:
    async def run() -> None:
        router = _make_router()

        worker_index = await router._assign_worker_index("s1")
        await router._on_conversations_finish("s1", worker_index)
        await router.clear_session_worker_mapping()

        assert router._session_worker == {}
        assert router._session_count == {}

    asyncio.run(run())


def test_generate_preserves_existing_driver_timing_for_outer_router() -> None:
    async def run() -> None:
        router: InferenceRouter[_TimedRequest, _TimedResponse] = InferenceRouter(max_imbalance=1)
        router._set_workers(cast("Any", [_TimedRemoteWorker()]))
        req = _TimedRequest(session_id="session-1")
        req.event_timing.mark_scheduled(100.0)

        result = await router.generate(req)

        assert result.event_timing.scheduled_at == 100.0
        assert result.event_timing.worker_received_at == 102.0
        assert result.event_timing.worker_returned_at == 104.0
        assert result.event_timing.driver_received_at is None

    asyncio.run(run())
