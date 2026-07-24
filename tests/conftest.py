"""Shared test setup: assert each module reaps its own ray/sglang/megatron workers.

Each test must call ``worker.shutdown()`` and (if it used Ray)
``ray_utils.stop()``. If a test forgets, the assertion in this fixture fails
loudly at module teardown with the surviving (pid, cmdline) so the leak is
fixed at the source. Without this, a leak silently poisons the next test
(e.g., the next module's sglang bootstrap aborts via ``os._exit(1)`` and
pytest exits 1 with no FAILED line).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

_LEAKED_WORKER_PATTERNS = (
    "ray::",
    "raylet",
    "sglang_scheduler",
    "sgl_router",
    "sglang::",
    "megatron",
)
_LEFTOVER_KILL_RETRIES = 5
_LEFTOVER_KILL_RETRY_WAIT_SEC = 60.0


def _parent_pid(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat")
        with stat.open() as f:
            after_comm = f.read().rsplit(")", maxsplit=1)[1].split()
    except OSError:
        return None
    if len(after_comm) < 2 or not after_comm[1].isdigit():
        return None
    return int(after_comm[1])


def _protected_process_pids() -> set[int]:
    """Return pytest and ancestor PIDs that leak cleanup must not kill."""
    protected: set[int] = set()
    pid: int | None = os.getpid()
    while pid is not None and pid > 1 and pid not in protected:
        protected.add(pid)
        pid = _parent_pid(pid)
    return protected


def _list_leaked_workers() -> list[tuple[int, str]]:
    """Return [(pid, cmdline)] for any sglang/ray/megatron process still alive.

    Skips lines whose cmdline contains ``pgrep`` so this scan never matches
    its own subprocess.
    """
    found: list[tuple[int, str]] = []
    seen: set[int] = set()
    protected_pids = _protected_process_pids()
    for pattern in _LEAKED_WORKER_PATTERNS:
        result = subprocess.run(
            ["pgrep", "-af", pattern],  # noqa: S607
            check=False,
            capture_output=True,
            timeout=5,
            text=True,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            pid = int(parts[0])
            cmdline = parts[1]
            if "pgrep" in cmdline or pid in seen or pid in protected_pids:
                continue
            seen.add(pid)
            found.append((pid, cmdline))
    return found


def _kill_leaked_workers(leaked: list[tuple[int, str]]) -> None:
    for pid, _cmdline in leaked:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def _reap_zombie_children() -> None:
    """Clear zombie children of the pytest process.

    The defunct sglang scheduler/detokenizer processes that ``pgrep``
    matches are children of this pytest process (it is a subreaper).
    ``pkill -9`` is a no-op on zombies — only ``waitpid`` clears them.
    """
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _assert_no_leaked_workers() -> None:
    """SIGKILL leaked ray/sglang/megatron workers; retry up to 5x.

    Each attempt kills only the worker PIDs returned by the leak scanner, reaps
    any resulting zombies, then waits 60s for the OS to settle. If 5
    attempts pass and survivors remain, raise so the offending test is
    obvious from the failure message.
    """
    leaked: list[tuple[int, str]] = []
    for _ in range(_LEFTOVER_KILL_RETRIES):
        leaked = _list_leaked_workers()
        if not leaked:
            return
        _kill_leaked_workers(leaked)
        with contextlib.suppress(OSError):
            _reap_zombie_children()
        time.sleep(_LEFTOVER_KILL_RETRY_WAIT_SEC)
    leaked = _list_leaked_workers()
    if not leaked:
        return
    formatted = "\n".join(f"  pid={pid} cmd={cmd[:160]}" for pid, cmd in leaked)
    raise AssertionError(f"Workers still alive after {_LEFTOVER_KILL_RETRIES} pkill -9 retries:\n{formatted}")


@pytest.fixture(autouse=True, scope="module")
def _assert_no_leaks_at_module_teardown() -> None:  # type: ignore
    """Assert at module teardown that the module reaped its own workers."""
    yield  # type: ignore
    _assert_no_leaked_workers()
