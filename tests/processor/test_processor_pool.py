import asyncio
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import override

import pytest

from axrl.processor import processor_pool as processor_pool_module
from axrl.processor.base_processor import BaseProcessor
from axrl.processor.processor_pool import ProcessorPool, ProcessorPoolTaskError, ProcessorPoolTaskTimeoutError

_NORMAL_TIMEOUT_SECONDS = 30.0


class _EchoProcessor(BaseProcessor[str, str]):
    @override
    def process(self, item: str) -> str:
        return f"ok:{item}"


class _FailingProcessor(BaseProcessor[str, str]):
    @override
    def process(self, item: str) -> str:
        raise ValueError(f"boom: {item}")


class _SlowProcessor(BaseProcessor[str, str]):
    @override
    def process(self, item: str) -> str:
        time.sleep(0.5)
        return item


class _SometimesSlowProcessor(BaseProcessor[str, str]):
    @override
    def process(self, item: str) -> str:
        if item == "slow":
            time.sleep(2.0)
        return f"ok:{item}"


class _SleepForeverProcessor(BaseProcessor[str, str]):
    @override
    def process(self, item: str) -> str:
        if item == "forever":
            while True:
                time.sleep(60)
        return f"ok:{item}"


class _CrashProcessor(BaseProcessor[str, str]):
    @override
    def process(self, item: str) -> str:
        if item == "crash":
            os._exit(17)
        return f"ok:{item}"


class _InitFailingProcessor(BaseProcessor[str, str]):
    def __init__(self, config: None = None) -> None:
        super().__init__(config)
        raise RuntimeError("init failed")

    @override
    def process(self, item: str) -> str:
        return item


class _Request:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class _FailingRequestProcessor(BaseProcessor[_Request, str]):
    @override
    def process(self, item: _Request) -> str:
        raise RuntimeError("request failed")


def test_processor_pool_returns_result() -> None:
    with ProcessorPool[str, str](_EchoProcessor, config=None, num_processors=1, timeout_seconds=_NORMAL_TIMEOUT_SECONDS) as pool:
        result = asyncio.run(pool.generate("x"))

    assert result == "ok:x"


def test_processor_pool_propagates_worker_exception() -> None:
    with (
        ProcessorPool[str, str](_FailingProcessor, config=None, num_processors=1, timeout_seconds=_NORMAL_TIMEOUT_SECONDS) as pool,
        pytest.raises(ProcessorPoolTaskError) as exc_info,
    ):
        asyncio.run(pool.generate("x"))

    message = str(exc_info.value)
    assert "processor=_FailingProcessor" in message
    assert "ValueError: boom: x" in message
    assert "Input: 'x'" in message
    assert "Worker traceback" in message


def test_processor_pool_propagates_processor_init_exception() -> None:
    with (
        ProcessorPool[str, str](_InitFailingProcessor, config=None, num_processors=1, timeout_seconds=_NORMAL_TIMEOUT_SECONDS) as pool,
        pytest.raises(ProcessorPoolTaskError) as exc_info,
    ):
        asyncio.run(pool.generate("x"))

    message = str(exc_info.value)
    assert "processor=_InitFailingProcessor" in message
    assert "RuntimeError: processor initialization: init failed" in message
    assert "Input: 'x'" in message


def test_processor_pool_exception_includes_session_id() -> None:
    with (
        ProcessorPool[_Request, str](_FailingRequestProcessor, config=None, num_processors=1, timeout_seconds=_NORMAL_TIMEOUT_SECONDS) as pool,
        pytest.raises(ProcessorPoolTaskError) as exc_info,
    ):
        asyncio.run(pool.generate(_Request(session_id="session-1")))

    assert "session_id=session-1" in str(exc_info.value)


def test_processor_pool_times_out() -> None:
    with ProcessorPool[str, str](_SlowProcessor, config=None, num_processors=1, timeout_seconds=_NORMAL_TIMEOUT_SECONDS) as pool:
        assert asyncio.run(pool.generate("warmup")) == "warmup"
        pool.timeout_seconds = 0.05
        with pytest.raises(ProcessorPoolTaskTimeoutError) as exc_info:
            asyncio.run(pool.generate("x"))

    message = str(exc_info.value)
    assert "processor=_SlowProcessor" in message
    assert "timeout_seconds=0.05" in message
    assert "Input: 'x'" in message


def test_processor_pool_timeout_excludes_queue_wait() -> None:
    async def run_two_requests() -> list[str]:
        with ProcessorPool[str, str](_SlowProcessor, config=None, num_processors=1, timeout_seconds=0.75) as pool:
            results = await asyncio.gather(pool.generate("first"), pool.generate("second"))
        return list(results)

    assert asyncio.run(run_two_requests()) == ["first", "second"]


def test_processor_pool_recovers_after_timed_out_task() -> None:
    with ProcessorPool[str, str](_SometimesSlowProcessor, config=None, num_processors=1, timeout_seconds=_NORMAL_TIMEOUT_SECONDS) as pool:
        assert asyncio.run(pool.generate("warmup")) == "ok:warmup"
        pool.timeout_seconds = 1.0
        # The first request intentionally times out; the assertion below is that the pool stays usable afterward.
        with pytest.raises(ProcessorPoolTaskTimeoutError):
            asyncio.run(pool.generate("slow"))

        pool.timeout_seconds = _NORMAL_TIMEOUT_SECONDS
        result = asyncio.run(pool.generate("fast"))

    assert result == "ok:fast"


def test_processor_pool_ignores_stale_worker_assignment_for_timeout() -> None:
    with ProcessorPool[str, str](_EchoProcessor, config=None, num_processors=1, timeout_seconds=_NORMAL_TIMEOUT_SECONDS) as pool:
        current_process = pool.procs[0]
        assert current_process.pid is not None
        stale_assignment = processor_pool_module._Started(
            worker_index=0,
            pid=current_process.pid,
            generation=pool.generations[0] - 1,
        )

        pool._restart_worker_for_timeout("stale-task", stale_assignment)
        assert pool.procs[0].pid == current_process.pid
        result = asyncio.run(pool.generate("x"))

    assert result == "ok:x"


def test_processor_pool_restarts_after_sleep_forever_timeout() -> None:
    with ProcessorPool[str, str](_SleepForeverProcessor, config=None, num_processors=1, timeout_seconds=0.2) as pool:
        old_pid = pool.procs[0].pid
        with pytest.raises(ProcessorPoolTaskTimeoutError):
            asyncio.run(pool.generate("forever"))
        assert pool.procs[0].pid != old_pid
        assert asyncio.run(pool.generate("fast")) == "ok:fast"


def test_processor_pool_restarts_after_worker_crash() -> None:
    with ProcessorPool[str, str](_CrashProcessor, config=None, num_processors=1, timeout_seconds=0.2) as pool:
        old_pid = pool.procs[0].pid
        with pytest.raises(ProcessorPoolTaskTimeoutError):
            asyncio.run(pool.generate("crash"))
        assert pool.procs[0].pid != old_pid
        assert asyncio.run(pool.generate("fast")) == "ok:fast"


def test_processor_pool_shutdown_stops_workers() -> None:
    pool = ProcessorPool[str, str](_EchoProcessor, config=None, num_processors=2, timeout_seconds=_NORMAL_TIMEOUT_SECONDS)
    pids = [proc.pid for proc in pool.procs]
    pool.close()

    assert pool.closed
    for pid in pids:
        assert pid is not None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.skipif(os.name != "posix", reason="parent-death signal is POSIX-only")
def test_processor_pool_workers_exit_when_parent_dies(tmp_path: Path) -> None:
    pid_path = tmp_path / "worker-pids.txt"
    script_path = tmp_path / "start_pool_and_exit.py"
    script_path.write_text(
        f"""
import asyncio
import os
import time
from pathlib import Path
from typing import override

from axrl.processor.base_processor import BaseProcessor
from axrl.processor.processor_pool import ProcessorPool


class SleepProcessor(BaseProcessor[str, str]):
    @override
    def process(self, item: str) -> str:
        if item == "sleep":
            time.sleep(1000)
        return f"ok:{{item}}"


async def main() -> None:
    pool = ProcessorPool(SleepProcessor, config=None, num_processors=2, timeout_seconds=30.0)
    await pool.batch_generate(["warmup-0", "warmup-1"])
    Path({str(pid_path)!r}).write_text("\\n".join(str(proc.pid) for proc in pool.procs), encoding="utf-8")
    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
""",
        encoding="utf-8",
    )

    completed = subprocess.run([sys.executable, str(script_path)], check=False, timeout=60.0)
    assert completed.returncode == 0
    pids = [int(line) for line in pid_path.read_text(encoding="utf-8").splitlines()]
    assert pids
    try:
        for pid in pids:
            _wait_until_dead(pid)
    finally:
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, 9)


def test_processor_pool_performance_smoke() -> None:
    async def run_batch() -> tuple[list[str], float]:
        with ProcessorPool[str, str](_EchoProcessor, config=None, num_processors=4, timeout_seconds=_NORMAL_TIMEOUT_SECONDS) as pool:
            await pool.generate("warmup")
            start = time.monotonic()
            results = await pool.batch_generate([str(i) for i in range(200)])
            return list(results), time.monotonic() - start

    results, elapsed_seconds = asyncio.run(run_batch())

    assert results == [f"ok:{i}" for i in range(200)]
    assert elapsed_seconds < 10.0


def _wait_until_dead(pid: int, *, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_is_live(pid):
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} is still alive")


def _process_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        with contextlib.suppress(OSError, IndexError):
            return proc_stat.read_text(encoding="utf-8").split()[2] != "Z"
    return True
