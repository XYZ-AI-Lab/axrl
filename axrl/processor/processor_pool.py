from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import threading
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from axrl.worker.infer_worker import InferWorker

logger = logging.getLogger(__name__)
_LINUX_PR_SET_PDEATHSIG = 1


class ProcessorPoolTaskError(RuntimeError):
    pass


class ProcessorPoolTaskTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True)
class _Started:
    worker_index: int
    pid: int
    generation: int


@dataclass(frozen=True)
class _Failure:
    processor: str
    task_id: str
    session_id: str | None
    input_repr: str
    exc_name: str
    exc_message: str
    traceback: str


@dataclass
class _Task[OutT]:
    started: asyncio.Future[_Started]
    result: asyncio.Future[OutT]
    loop: asyncio.AbstractEventLoop


class ProcessorPool[InT, OutT](InferWorker[InT, OutT]):
    def __init__(
        self,
        processor_cls: Any,
        config: Any,
        num_processors: int,
        timeout_seconds: float = 60.0,
    ) -> None:
        if num_processors <= 0:
            raise ValueError("num_processors must be > 0")
        self.processor_cls = processor_cls
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.ctx = mp.get_context("spawn")
        self.in_queue = self.ctx.SimpleQueue()
        self.out_queue = self.ctx.SimpleQueue()
        self.procs: list[Any] = []
        self.generations = [0] * num_processors
        self.tasks: dict[str, _Task[OutT]] = {}
        self.closed = False
        for index in range(num_processors):
            self.procs.append(self._start_worker(index))
        self.listener = threading.Thread(target=self._listen, name="ProcessorPoolListener", daemon=True)
        self.listener.start()

    async def generate(self, req: InT) -> OutT:
        if self.closed:
            raise RuntimeError("ProcessorPool is closed")
        task_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        state = _Task[OutT](started=loop.create_future(), result=loop.create_future(), loop=loop)
        self.tasks[task_id] = state
        self.in_queue.put((task_id, req, _get_session_id(req)))
        started: _Started | None = None
        try:
            started = await state.started
            return await asyncio.wait_for(state.result, timeout=self.timeout_seconds)
        except TimeoutError as exc:
            assert started is not None
            self.tasks.pop(task_id, None)
            state.result.cancel()
            self._restart_worker_for_timeout(task_id, started)
            raise ProcessorPoolTaskTimeoutError(
                f"ProcessorPool task timed out "
                f"(processor={self.processor_cls.__name__}, task_id={task_id}, "
                f"session_id={_get_session_id(req)}, timeout_seconds={self.timeout_seconds})\nInput: {_safe_repr(req)}"
            ) from exc
        except BaseException:
            self.tasks.pop(task_id, None)
            raise

    async def batch_generate(self, reqs: Any) -> Any:
        return await asyncio.gather(*(self.generate(req) for req in reqs))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for _ in self.procs:
            self.in_queue.put(None)
        for index, proc in enumerate(self.procs):
            _stop_process(proc, worker_index=index, reason="pool close", graceful_timeout_seconds=5.0)
        self.out_queue.put(None)
        self.listener.join(timeout=2.0)
        error = RuntimeError("ProcessorPool closed")
        for state in self.tasks.values():
            state.loop.call_soon_threadsafe(_set_exception, state.started, error)
            state.loop.call_soon_threadsafe(_set_exception, state.result, error)
        self.tasks.clear()
        self.procs.clear()
        with contextlib.suppress(Exception):
            self.in_queue.close()
            self.out_queue.close()

    shutdown = close

    def __enter__(self) -> ProcessorPool[InT, OutT]:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        self.close()

    def _start_worker(self, index: int) -> Any:
        self.generations[index] += 1
        generation = self.generations[index]
        proc = self.ctx.Process(
            target=_worker,
            args=(index, generation, self.processor_cls, self.config, self.in_queue, self.out_queue),
            daemon=True,
        )
        proc.start()
        return proc

    def _listen(self) -> None:
        while True:
            msg = self.out_queue.get()
            if msg is None:
                return
            task_id, kind, payload = msg
            state = self.tasks.get(task_id)
            if state is None:
                logger.warning("ProcessorPool received stale %s for task_id=%s", kind, task_id)
                continue
            if kind == "started":
                state.loop.call_soon_threadsafe(_set_result, state.started, payload)
                continue
            self.tasks.pop(task_id, None)
            if kind == "ok":
                state.loop.call_soon_threadsafe(_set_result, state.result, payload)
            else:
                state.loop.call_soon_threadsafe(_set_exception, state.result, ProcessorPoolTaskError(_format_failure(payload)))

    def _restart_worker_for_timeout(self, task_id: str, started: _Started) -> None:
        if self.closed:
            return
        proc = self.procs[started.worker_index]
        if proc.pid != started.pid or self.generations[started.worker_index] != started.generation:
            logger.warning("ProcessorPool ignored stale timeout assignment for task_id=%s", task_id)
            return
        _stop_process(proc, worker_index=started.worker_index, reason="task timeout", graceful_timeout_seconds=0.0)
        self.procs[started.worker_index] = self._start_worker(started.worker_index)


def _worker[InT, OutT](
    worker_index: int,
    generation: int,
    processor_cls: Any,
    config: Any,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
) -> None:
    _exit_with_parent()
    processor_name = processor_cls.__name__
    try:
        processor = processor_cls(config)
        init_error: BaseException | None = None
    except BaseException as exc:
        processor = None
        init_error = exc
    while True:
        item = in_queue.get()
        if item is None:
            return
        task_id, req, session_id = item
        out_queue.put((task_id, "started", _Started(worker_index, os.getpid(), generation)))
        if init_error is not None:
            init_exc = RuntimeError(f"processor initialization: {init_error}")
            out_queue.put((task_id, "error", _failure(init_exc, processor_name, task_id, session_id, req)))
            continue
        try:
            assert processor is not None
            out_queue.put((task_id, "ok", processor.process(req)))
        except BaseException as exc:
            out_queue.put((task_id, "error", _failure(exc, processor_name, task_id, session_id, req)))


def _exit_with_parent() -> None:
    """Make spawned workers exit if their owning Ray actor/process dies."""
    if os.name != "posix":
        return
    with contextlib.suppress(BaseException):
        import ctypes

        libc = ctypes.CDLL(None)
        libc.prctl(_LINUX_PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() == 1:
        raise SystemExit(0)


def _stop_process(process: Any, *, worker_index: int, reason: str, graceful_timeout_seconds: float) -> None:
    if graceful_timeout_seconds > 0:
        process.join(timeout=graceful_timeout_seconds)
    if process.is_alive():
        logger.warning("ProcessorPool terminating worker: reason=%s worker_index=%d pid=%s", reason, worker_index, process.pid)
        process.terminate()
        process.join(timeout=1.0)
    if process.is_alive():
        logger.warning("ProcessorPool killing worker: reason=%s worker_index=%d pid=%s", reason, worker_index, process.pid)
        process.kill()
        process.join(timeout=1.0)


def _failure(exc: BaseException, processor: str, task_id: str, session_id: str | None, req: Any) -> _Failure:
    return _Failure(processor, task_id, session_id, _safe_repr(req), type(exc).__name__, str(exc), traceback.format_exc())


def _format_failure(failure: _Failure) -> str:
    return (
        f"ProcessorPool task failed (processor={failure.processor}, task_id={failure.task_id}, "
        f"session_id={failure.session_id}, exception={failure.exc_name}: {failure.exc_message})"
        f"\nInput: {failure.input_repr}\nWorker traceback:\n{failure.traceback}"
    )


def _set_result(future: asyncio.Future[Any], value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception(future: asyncio.Future[Any], exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


def _get_session_id(req: Any) -> str | None:
    session_id = getattr(req, "session_id", None) or getattr(getattr(req, "gen_state", None), "session_id", None)
    return str(session_id) if session_id else None


def _safe_repr(value: Any) -> str:
    with contextlib.suppress(Exception):
        return repr(value)[:200000]
    return f"<unrepresentable {type(value).__module__}.{type(value).__name__}>"
