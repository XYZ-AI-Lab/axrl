from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import ray

from axrl.configs import RAY_TASK_QUEUE_MAX_CONCURRENCY

if TYPE_CHECKING:
    from ray.actor import ActorHandle

TaskT = TypeVar("TaskT")

DEFAULT_QUEUE_NAME = "default"


@dataclass(frozen=True)
class TaskAssignment[TaskT]:
    """A task currently assigned to one worker.

    When ``max_running_tasks`` is set, the queue keeps this assignment in its
    running table until the worker calls ``complete``. Rollout loops should
    catch expected errors and publish explicit failure results; otherwise the
    driver may keep waiting for output from a running assignment.

    When ``max_running_tasks=None``, the queue works as a mailbox: ``get``
    consumes the task immediately and ``complete`` is not required.
    """

    assignment_id: str
    task: TaskT
    worker_id: str
    queued_at: float
    assigned_at: float
    key: str = DEFAULT_QUEUE_NAME


@dataclass(frozen=True)
class _QueuedTask[TaskT]:
    task: TaskT
    queued_at: float


@dataclass(frozen=True)
class RayTaskQueueStats:
    pending_count: int
    running_count: int
    max_running_tasks: int | None
    completed_count: int


@ray.remote
class RemoteRayTaskQueue[TaskT]:
    """Ray actor task queue with global running-task limits.

    A single ``RayTaskQueue`` actor can be shared by rollout workers on
    different processes or nodes. The actor exposes both ``get`` and
    ``get_nowait``:

    - ``get`` is the normal async blocking API for rollout workers. It can
      also take ``timeout_seconds`` and return ``None`` if no task is ready.
    - ``get_nowait`` is useful for polling, diagnostics, or custom scheduling.

    Set ``max_running_tasks`` to an integer for task-queue mode: assignments
    count as running until workers call ``complete``. Set it to ``None`` for
    mailbox mode: ``get`` consumes tasks immediately and ``complete`` is not
    required.
    """

    def __init__(self, max_running_tasks: int | None) -> None:
        if max_running_tasks is not None and max_running_tasks <= 0:
            raise ValueError("max_running_tasks must be greater than zero.")
        self._max_running_tasks = max_running_tasks
        self._pending: dict[str, deque[_QueuedTask[TaskT]]] = {}
        self._running: dict[str, TaskAssignment[TaskT]] = {}
        self._completed_count = 0
        self._condition = asyncio.Condition()

    async def put(self, task: TaskT, *, key: str = DEFAULT_QUEUE_NAME) -> None:
        async with self._condition:
            self._pending.setdefault(key, deque()).append(_QueuedTask(task=task, queued_at=time.monotonic()))
            self._condition.notify_all()

    async def put_many(self, tasks: list[TaskT], *, key: str = DEFAULT_QUEUE_NAME) -> None:
        async with self._condition:
            queued_at = time.monotonic()
            self._pending.setdefault(key, deque()).extend(_QueuedTask(task=task, queued_at=queued_at) for task in tasks)
            self._condition.notify_all()

    async def get_nowait(self, worker_id: str, *, key: str = DEFAULT_QUEUE_NAME) -> TaskAssignment[TaskT] | None:
        """Return a task assignment if one can run now, otherwise return ``None``."""
        async with self._condition:
            return self._take_next_assignment_locked(worker_id, key=key)

    async def get(
        self,
        worker_id: str,
        *,
        key: str = DEFAULT_QUEUE_NAME,
        timeout_seconds: float | None = None,
    ) -> TaskAssignment[TaskT] | None:
        """Wait until this worker can receive the next runnable task."""
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        async with self._condition:
            while True:
                assignment = self._take_next_assignment_locked(worker_id, key=key)
                if assignment is not None:
                    return assignment
                if deadline is None:
                    await self._condition.wait()
                    continue
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    return None
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining_seconds)
                except TimeoutError:
                    return None

    async def complete(self, assignment_id: str) -> None:
        """Mark an assigned task as finished and release its running slot."""
        assert self._max_running_tasks is not None, "RayTaskQueue.complete() is only valid when max_running_tasks is set."
        async with self._condition:
            assignment = self._running.pop(assignment_id, None)
            assert assignment is not None, f"Unknown assignment_id: {assignment_id}"
            self._completed_count += 1
            self._condition.notify_all()

    async def stats(self, key: str = DEFAULT_QUEUE_NAME) -> RayTaskQueueStats:
        async with self._condition:
            pending = self._pending.get(key)
            pending_count = len(pending) if pending is not None else 0
            running_count = sum(1 for assignment in self._running.values() if assignment.key == key)
            return RayTaskQueueStats(
                pending_count=pending_count,
                running_count=running_count,
                max_running_tasks=self._max_running_tasks,
                completed_count=self._completed_count,
            )

    async def running_assignments(self) -> list[TaskAssignment[TaskT]]:
        async with self._condition:
            return list(self._running.values())

    def _take_next_assignment_locked(self, worker_id: str, *, key: str) -> TaskAssignment[TaskT] | None:
        pending = self._pending.get(key)
        if not pending:
            return None
        if self._max_running_tasks is not None and len(self._running) >= self._max_running_tasks:
            return None
        queued_task = pending.popleft()
        assignment_id = f"{worker_id}-{uuid.uuid4().hex}"
        assignment = TaskAssignment(
            assignment_id=assignment_id,
            task=queued_task.task,
            worker_id=worker_id,
            queued_at=queued_task.queued_at,
            assigned_at=time.monotonic(),
            key=key,
        )
        if self._max_running_tasks is not None:
            self._running[assignment_id] = assignment
        return assignment


class RayTaskQueue[TaskT]:
    """Typed local wrapper around a shared ``RemoteRayTaskQueue`` actor."""

    def __init__(self, actor: ActorHandle) -> None:
        self._actor = actor

    def get_actor_handle(self) -> ActorHandle:
        return self._actor

    async def put(self, task: TaskT, *, key: str = DEFAULT_QUEUE_NAME) -> None:
        await self._actor.put.remote(task, key=key)

    async def put_many(self, tasks: list[TaskT], *, key: str = DEFAULT_QUEUE_NAME) -> None:
        await self._actor.put_many.remote(tasks, key=key)

    async def get_nowait(self, worker_id: str, *, key: str = DEFAULT_QUEUE_NAME) -> TaskAssignment[TaskT] | None:
        return cast("TaskAssignment[TaskT] | None", await self._actor.get_nowait.remote(worker_id, key=key))

    async def get(
        self,
        worker_id: str,
        *,
        key: str = DEFAULT_QUEUE_NAME,
        timeout_seconds: float | None = None,
    ) -> TaskAssignment[TaskT] | None:
        return cast(
            "TaskAssignment[TaskT] | None",
            await self._actor.get.remote(worker_id, key=key, timeout_seconds=timeout_seconds),
        )

    async def complete(self, assignment_id: str) -> None:
        await self._actor.complete.remote(assignment_id)

    async def stats(self, key: str = DEFAULT_QUEUE_NAME) -> RayTaskQueueStats:
        return cast("RayTaskQueueStats", await self._actor.stats.remote(key=key))

    async def running_assignments(self) -> list[TaskAssignment[TaskT]]:
        return cast("list[TaskAssignment[TaskT]]", await self._actor.running_assignments.remote())

    @staticmethod
    def initialize_remote_actor(max_running_tasks: int | None) -> ActorHandle:
        return cast(
            "ActorHandle",
            RemoteRayTaskQueue.options(max_concurrency=RAY_TASK_QUEUE_MAX_CONCURRENCY, num_cpus=1).remote(max_running_tasks),  # type: ignore[attr-defined]
        )
