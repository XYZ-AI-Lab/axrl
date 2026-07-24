from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
import ray

from axrl.utils.ray_task_queue import DEFAULT_QUEUE_NAME, RayTaskQueue, TaskAssignment

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _ray_cluster() -> Iterator[None]:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=2, include_dashboard=False)
    try:
        yield
    finally:
        ray.shutdown()


async def _get_assignment(
    task_queue: RayTaskQueue[str],
    worker_id: str,
    *,
    key: str = DEFAULT_QUEUE_NAME,
) -> TaskAssignment[str]:
    assignment = await task_queue.get(worker_id, key=key)
    assert assignment is not None
    return assignment


async def _take_and_complete(task_queue: RayTaskQueue[str], worker_id: str) -> str:
    assignment = await _get_assignment(task_queue, worker_id)
    await task_queue.complete(assignment.assignment_id)
    return assignment.task


def test_max_running_tasks_is_global_until_completion() -> None:
    async def run() -> None:
        queue = RayTaskQueue[str](RayTaskQueue.initialize_remote_actor(max_running_tasks=2))
        await queue.put_many(["task-a", "task-b"])

        first = await _get_assignment(queue, "worker-1")
        assert first.task == "task-a"
        assert first.queued_at <= first.assigned_at
        second = await _get_assignment(queue, "worker-2")
        assert second.task == "task-b"
        assert second.queued_at <= second.assigned_at

        third_task = asyncio.create_task(_get_assignment(queue, "worker-3"))
        await asyncio.sleep(0.2)
        assert not third_task.done()

        stats = await queue.stats()
        assert stats.pending_count == 0
        assert stats.running_count == 2

        await queue.complete(first.assignment_id)
        await queue.put("task-c")
        third = await asyncio.wait_for(third_task, timeout=5)
        assert third.task == "task-c"
        assert third.queued_at <= third.assigned_at

    asyncio.run(run())


def test_get_nowait_is_available() -> None:
    async def run() -> None:
        queue = RayTaskQueue[str](RayTaskQueue.initialize_remote_actor(max_running_tasks=1))

        assert await queue.get_nowait("worker-1") is None

        await queue.put("task-a")
        assignment = await queue.get_nowait("worker-1")
        assert assignment is not None
        assert assignment.task == "task-a"
        assert assignment.queued_at <= assignment.assigned_at

    asyncio.run(run())


def test_complete_asserts_known_assignment_id() -> None:
    async def run() -> None:
        queue = RayTaskQueue[str](RayTaskQueue.initialize_remote_actor(max_running_tasks=1))

        with pytest.raises(AssertionError, match="Unknown assignment_id"):
            await queue.complete("missing")

    asyncio.run(run())


def test_mailbox_mode_does_not_require_completion() -> None:
    async def run() -> None:
        queue = RayTaskQueue[str](RayTaskQueue.initialize_remote_actor(max_running_tasks=None))
        await queue.put_many(["task-a", "task-b"])

        first = await _get_assignment(queue, "worker-a")
        second = await _get_assignment(queue, "worker-b")
        assert first.task == "task-a"
        assert second.task == "task-b"

        stats = await queue.stats()
        assert stats.pending_count == 0
        assert stats.running_count == 0
        assert stats.max_running_tasks is None
        assert stats.completed_count == 0
        with pytest.raises(AssertionError, match="only valid when max_running_tasks is set"):
            await queue.complete(first.assignment_id)

    asyncio.run(run())


def test_multiple_local_consumers_can_share_the_same_queue() -> None:
    async def run() -> None:
        queue = RayTaskQueue[str](RayTaskQueue.initialize_remote_actor(max_running_tasks=2))
        await queue.put_many(["shared-a", "shared-b"])

        results = await asyncio.gather(
            _take_and_complete(queue, "worker-a"),
            _take_and_complete(queue, "worker-b"),
        )
        assert sorted(results) == ["shared-a", "shared-b"]

        stats = await queue.stats()
        assert stats.pending_count == 0
        assert stats.running_count == 0
        assert stats.completed_count == 2

    asyncio.run(run())


def test_blocking_gets_do_not_starve_put_many() -> None:
    async def run() -> None:
        queue = RayTaskQueue[str](RayTaskQueue.initialize_remote_actor(max_running_tasks=2))
        await asyncio.wait_for(queue.stats(), timeout=30)
        first_task = asyncio.create_task(_take_and_complete(queue, "worker-a"))
        second_task = asyncio.create_task(_take_and_complete(queue, "worker-b"))
        await asyncio.sleep(0.2)
        assert not first_task.done()
        assert not second_task.done()

        await asyncio.wait_for(queue.put_many(["task-a", "task-b"]), timeout=5)
        results = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=5)
        assert sorted(results) == ["task-a", "task-b"]

    asyncio.run(run())


def test_queue_keys_are_independent_pending_streams_with_global_running_limit() -> None:
    async def run() -> None:
        queue = RayTaskQueue[str](RayTaskQueue.initialize_remote_actor(max_running_tasks=2))
        await queue.put_many(["a0", "a1"], key="dp:0")
        await queue.put("b0", key="dp:1")

        first = await _get_assignment(queue, "worker-a", key="dp:0")
        second = await _get_assignment(queue, "worker-b", key="dp:1")
        assert first.task == "a0"
        assert first.key == "dp:0"
        assert second.task == "b0"
        assert second.key == "dp:1"

        blocked = asyncio.create_task(_get_assignment(queue, "worker-c", key="dp:0"))
        await asyncio.sleep(0.2)
        assert not blocked.done()

        stats_default = await queue.stats()
        stats_dp0 = await queue.stats(key="dp:0")
        stats_dp1 = await queue.stats(key="dp:1")
        assert stats_default.pending_count == 0
        assert stats_default.running_count == 0
        assert stats_dp0.pending_count == 1
        assert stats_dp0.running_count == 1
        assert stats_dp1.pending_count == 0
        assert stats_dp1.running_count == 1

        await queue.complete(first.assignment_id)
        third = await asyncio.wait_for(blocked, timeout=5)
        assert third.task == "a1"
        assert third.key == "dp:0"

    asyncio.run(run())
