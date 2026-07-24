from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import shutil
import socket
import uuid
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from axis_recipe.blackbox_rl.config import OpenHandsLauncherConfig
from axis_recipe.blackbox_rl.openhands_utils import get_openhand_launch_script
from axrl.runner import CgroupRunner
from axrl.runner.e2b_runner import E2BRunnerConfig

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def test_cgroup_runner_raises_when_unsupported(tmp_path: Path) -> None:
    if CgroupRunner.is_supported():
        pytest.skip("CgroupRunner is supported on this host.")
    runtime = CgroupRunner(name=f"axrl-test-{uuid.uuid4().hex}")
    with pytest.raises(RuntimeError, match="requires permission to mount"):
        asyncio.run(runtime.start("true", tmp_path))


def test_cgroup_runner_terminates_bash_sleep_child(tmp_path: Path) -> None:
    _skip_if_cgroup_is_unavailable()
    asyncio.run(_test_cgroup_runner_terminates_bash_sleep_child(tmp_path))


def test_cgroup_runner_does_not_depend_on_asyncio_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_if_cgroup_is_unavailable()

    async def _raise_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CgroupRunner should not use asyncio subprocess APIs.")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _raise_if_called)
    asyncio.run(_test_cgroup_runner_runs_true(tmp_path))


def test_cgroup_runner_allows_reused_logical_name(tmp_path: Path) -> None:
    _skip_if_cgroup_is_unavailable()
    asyncio.run(_test_cgroup_runner_allows_reused_logical_name(tmp_path))


def test_cgroup_runner_terminates_python_sleep_child(tmp_path: Path) -> None:
    _skip_if_cgroup_is_unavailable()
    asyncio.run(_test_cgroup_runner_terminates_python_sleep_child(tmp_path))


def test_cgroup_runner_memory_limit_stops_runaway_python(tmp_path: Path) -> None:
    _skip_if_cgroup_is_unavailable()
    asyncio.run(_test_cgroup_runner_memory_limit_stops_runaway_python(tmp_path))


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required for this test.")
def test_cgroup_runner_terminates_tmux_bash_sleep_child(tmp_path: Path) -> None:
    _skip_if_cgroup_is_unavailable()
    asyncio.run(_test_cgroup_runner_terminates_tmux_bash_sleep_child(tmp_path))


@pytest.mark.skipif(shutil.which("openhands") is None, reason="openhands is required for this test.")
def test_cgroup_runner_terminates_openhands(tmp_path: Path) -> None:
    _skip_if_cgroup_is_unavailable()
    asyncio.run(_test_cgroup_runner_terminates_openhands(tmp_path))


async def _test_cgroup_runner_terminates_bash_sleep_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "sleep.pid"
    runtime = CgroupRunner(name=f"axrl-test-{uuid.uuid4().hex}")
    await runtime.start(f"/usr/bin/sleep 100d & echo $! > {shlex.quote(str(pid_path))}; wait", tmp_path)
    try:
        await _wait_until(pid_path.exists)
        child_pid = int(pid_path.read_text(encoding="utf-8"))

        await runtime.terminate(timeout_seconds=2.0)

        assert runtime.returncode is not None
        await _wait_until(lambda: not _process_is_live(child_pid))
    finally:
        await runtime.terminate()


async def _test_cgroup_runner_runs_true(tmp_path: Path) -> None:
    runtime = CgroupRunner(name=f"axrl-test-{uuid.uuid4().hex}")
    await runtime.start("true", tmp_path)
    try:
        await _wait_until(lambda: runtime.returncode is not None)
        assert runtime.returncode == 0
    finally:
        await runtime.terminate()


async def _test_cgroup_runner_allows_reused_logical_name(tmp_path: Path) -> None:
    first = CgroupRunner(name="axrl-test-reused-name")
    second = CgroupRunner(name="axrl-test-reused-name")
    try:
        await first.start("sleep 100d", tmp_path)
        await second.start("sleep 100d", tmp_path)
    finally:
        await second.terminate()
        await first.terminate()


async def _test_cgroup_runner_terminates_python_sleep_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "sleep.pid"
    script_path = tmp_path / "spawn_sleep.py"
    script_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import subprocess",
                "process = subprocess.Popen(['/usr/bin/sleep', '100d'])",
                f"Path({str(pid_path)!r}).write_text(str(process.pid), encoding='utf-8')",
                "process.wait()",
            ]
        ),
        encoding="utf-8",
    )
    runtime = CgroupRunner(name=f"axrl-test-{uuid.uuid4().hex}")
    await runtime.start(f"python {shlex.quote(str(script_path))}", tmp_path)
    try:
        await _wait_until(pid_path.exists)
        child_pid = int(pid_path.read_text(encoding="utf-8"))

        await runtime.terminate(timeout_seconds=2.0)

        assert runtime.returncode is not None
        await _wait_until(lambda: not _process_is_live(child_pid))
    finally:
        await runtime.terminate()


async def _test_cgroup_runner_memory_limit_stops_runaway_python(tmp_path: Path) -> None:
    child_path = tmp_path / "allocate_forever.py"
    child_path.write_text(
        "import time\nchunks = []\nwhile True:\n    chunks.append(bytearray(32 * 1024 * 1024))\n    time.sleep(0.01)\n",
        encoding="utf-8",
    )
    script_path = tmp_path / "spawn_allocator.py"
    script_path.write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        f"process = subprocess.Popen([sys.executable, {str(child_path)!r}])\n"
        "try:\n"
        "    sys.exit(process.wait(timeout=10.0))\n"
        "except subprocess.TimeoutExpired:\n"
        "    process.kill()\n"
        "    raise\n",
        encoding="utf-8",
    )
    runtime = CgroupRunner(
        name=f"axrl-test-{uuid.uuid4().hex}",
        memory_limit_bytes=128 * 1024 * 1024,
    )

    returncode, _stdout = await runtime.run(f"python {shlex.quote(str(script_path))}", tmp_path, timeout_seconds=10.0)

    assert returncode != 0


async def _test_cgroup_runner_terminates_tmux_bash_sleep_child(tmp_path: Path) -> None:
    socket_name = f"axrl-test-{uuid.uuid4().hex}"
    session_name = f"axrl-test-{uuid.uuid4().hex}"
    sleep_pid_path = tmp_path / "tmux-sleep.pid"
    pane_script_path = tmp_path / "tmux-pane.sh"
    pane_script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "/usr/bin/sleep 100d &",
                f"echo $! > {shlex.quote(str(sleep_pid_path))}",
                "wait",
            ]
        ),
        encoding="utf-8",
    )
    pane_script_path.chmod(0o755)
    runtime = CgroupRunner(name=f"axrl-test-{uuid.uuid4().hex}")
    command = f"tmux -L{shlex.quote(socket_name)} new-session -d -s {shlex.quote(session_name)} {shlex.quote(str(pane_script_path))}; sleep 100d"
    await runtime.start(command, tmp_path)
    try:
        await _wait_until_async(partial(_tmux_has_session, socket_name, session_name))
        await _wait_until(sleep_pid_path.exists)
        sleep_pid = int(sleep_pid_path.read_text(encoding="utf-8"))

        await runtime.terminate(timeout_seconds=2.0)

        assert runtime.returncode is not None
        await _wait_until_async(_tmux_session_is_gone(socket_name, session_name))
        await _wait_until(lambda: not _process_is_live(sleep_pid))
    finally:
        await runtime.terminate()
        await _kill_tmux_server(socket_name)


async def _test_cgroup_runner_terminates_openhands(tmp_path: Path) -> None:
    server, request_seen, stop_server, port = await _start_hanging_http_server()
    try:
        tmux_wrapper_dir = _write_tmux_wrapper(tmp_path)
        runtime = CgroupRunner(name=f"axrl-test-{uuid.uuid4().hex}", stop_timeout_seconds=2.0)
        command = get_openhand_launch_script(
            OpenHandsLauncherConfig(
                llm_timeout_seconds=60,
                e2b=E2BRunnerConfig(workdir=str(tmp_path)),
                extra_env={
                    "LD_LIBRARY_PATH_ORIG": os.environ.get("LD_LIBRARY_PATH", ""),
                    "PATH": f"{tmux_wrapper_dir}:$PATH",
                    "TMUX_PROGRAM": str(tmux_wrapper_dir / "tmux"),
                },
            ),
            task="Say hello once.",
            llm_base_url=f"http://127.0.0.1:{port}/v1",
            llm_model="test-model",
        )
        await runtime.start(command, tmp_path)
        try:
            await asyncio.wait_for(request_seen.wait(), timeout=60.0)

            await runtime.terminate(timeout_seconds=2.0)

            assert runtime.returncode is not None
        finally:
            await runtime.terminate()
    finally:
        stop_server.set()
        server.close()
        await server.wait_closed()


def _skip_if_cgroup_is_unavailable() -> None:
    if not CgroupRunner.is_supported():
        pytest.skip("private writable cgroup v2 is not available in this environment.")


def _tmux_session_is_gone(socket_name: str, session_name: str) -> Callable[[], Awaitable[bool]]:
    async def _check() -> bool:
        return not await _tmux_has_session(socket_name, session_name)

    return _check


async def _tmux_has_session(socket_name: str, session_name: str) -> bool:
    process = await asyncio.create_subprocess_exec(
        "tmux",
        f"-L{socket_name}",
        "has-session",
        "-t",
        session_name,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await process.wait()
    return process.returncode == 0


async def _kill_tmux_server(socket_name: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "tmux",
        f"-L{socket_name}",
        "kill-server",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await process.wait()


def _write_tmux_wrapper(tmp_path: Path) -> Path:
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    wrapper_path = wrapper_dir / "tmux"
    wrapper_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${LD_LIBRARY_PATH_ORIG:-}" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_ORIG}"
fi
exec /usr/bin/tmux "$@"
""",
        encoding="utf-8",
    )
    wrapper_path.chmod(0o755)
    return wrapper_dir


async def _start_hanging_http_server() -> tuple[asyncio.AbstractServer, asyncio.Event, asyncio.Event, int]:
    request_seen = asyncio.Event()
    stop_server = asyncio.Event()
    port = _free_port()

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_seen.set()
        await reader.read(1024)
        await stop_server.wait()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(_handle, host="127.0.0.1", port=port)
    return server, request_seen, stop_server, port


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_until(predicate: Callable[[], bool], *, timeout_seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for condition.")


async def _wait_until_async(predicate: Callable[[], Awaitable[bool]], *, timeout_seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for async condition.")


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
