from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import uuid
from contextlib import suppress
from functools import cache
from pathlib import Path
from typing import IO

from axrl.runner.base_runner import DEFAULT_TERMINATE_TIMEOUT_SECONDS, BaseRunner

_START_TIMEOUT_SECONDS = 5.0


class CgroupRunner(BaseRunner):
    """Run a command in a Linux cgroup so its process tree can be cleaned up.

    A cgroup, short for "control group", is a Linux kernel feature that groups
    processes under one lifecycle/resource-control boundary. We use it when a
    runtime may start extra processes outside the direct child tree, such as
    tmux sessions, shell grandchildren, or long-lived helper commands. Killing
    the cgroup lets us stop those related processes together.

    This is not a security sandbox. The command still has the same filesystem,
    network, environment, and system-call access as the Python process that
    launches it. CgroupRunner provides process cleanup and best-effort memory
    limiting through an inherited ``ulimit -v``. It does not provide isolation
    from untrusted code.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        stop_timeout_seconds: float = DEFAULT_TERMINATE_TIMEOUT_SECONDS,
        memory_limit_bytes: int | None = None,
    ) -> None:
        if memory_limit_bytes is not None and memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be greater than zero.")
        self.name = _unique_cgroup_name(name or "axrl")
        self.stop_timeout_seconds = stop_timeout_seconds
        self.memory_limit_bytes = memory_limit_bytes
        self.process: subprocess.Popen[bytes] | None = None
        self._work_dir: Path | None = None
        self._mount_path: Path | None = None
        self._cgroup_path: Path | None = None

    @staticmethod
    def is_supported() -> bool:
        return _can_mount_private_cgroup2()

    async def start(self, command: str, cwd: Path) -> None:
        assert self.process is None, "CgroupRunner.start() can only be called once."
        if not self.is_supported():
            raise RuntimeError("CgroupRunner requires permission to mount a private writable cgroup v2 filesystem.")
        self._work_dir = Path(tempfile.mkdtemp(prefix="axrl-cgroup-runner-"))
        self._mount_path = self._work_dir / "cgroup"
        self._mount_path.mkdir()
        await _run_command(_mount_bin(), "-t", "cgroup2", "none", str(self._mount_path))
        self._cgroup_path = self._mount_path / self.name
        self._cgroup_path.mkdir()
        ready_path = self._work_dir / "runner.pid"
        go_path = self._work_dir / "go"
        wrapper = _start_wrapper(
            command=command,
            ready_path=ready_path,
            go_path=go_path,
            memory_limit_bytes=self.memory_limit_bytes,
        )
        self.process = await asyncio.to_thread(
            subprocess.Popen,
            [_bash_bin(), "-lc", wrapper],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            await _wait_until(ready_path.exists, timeout_seconds=_START_TIMEOUT_SECONDS)
            pid = int(ready_path.read_text(encoding="utf-8").strip())
            (self._cgroup_path / "cgroup.procs").write_text(f"{pid}\n", encoding="utf-8")
            go_path.touch()
        except Exception:
            await self.terminate()
            raise

    async def terminate(self, *, timeout_seconds: float = DEFAULT_TERMINATE_TIMEOUT_SECONDS) -> None:
        await self._kill_cgroup()
        await self._wait_process_after_kill(timeout_seconds=timeout_seconds)
        await self._cleanup()

    async def run(self, command: str, cwd: Path, *, timeout_seconds: float | None = None) -> tuple[int, str]:
        await self.start(command, cwd)
        try:
            assert self.process is not None
            try:
                # communicate() is blocking; stderr is already merged into stdout
                # by Popen(stderr=STDOUT), so the second return value is unused.
                stdout, _stderr = await asyncio.to_thread(self.process.communicate, timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Timed out waiting for CgroupRunner command.") from exc
            return int(self.process.returncode), stdout.decode(errors="replace")
        finally:
            await self.terminate()

    @property
    def stdout(self) -> IO[bytes] | None:
        return None if self.process is None else self.process.stdout

    @property
    def returncode(self) -> int | None:
        return None if self.process is None else self.process.poll()

    async def _kill_cgroup(self) -> None:
        if self._cgroup_path is None or not self._cgroup_path.exists():
            return
        kill_path = self._cgroup_path / "cgroup.kill"
        if kill_path.exists():
            kill_path.write_text("1", encoding="utf-8")
            return
        for pid_text in (self._cgroup_path / "cgroup.procs").read_text(encoding="utf-8").splitlines():
            with suppress(ProcessLookupError, ValueError):
                os.kill(int(pid_text), signal.SIGKILL)

    async def _wait_process_after_kill(self, *, timeout_seconds: float) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        try:
            await asyncio.to_thread(self.process.wait, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                self.process.kill()
            with suppress(subprocess.TimeoutExpired):
                await asyncio.to_thread(self.process.wait, timeout=timeout_seconds)

    async def _cleanup(self) -> None:
        if self._mount_path is not None and self._mount_path.exists():
            with suppress(RuntimeError):
                await _run_command(_umount_bin(), str(self._mount_path))
        if self._work_dir is not None:
            with suppress(OSError):
                shutil.rmtree(self._work_dir)
        self._work_dir = None
        self._mount_path = None
        self._cgroup_path = None


def _start_wrapper(*, command: str, ready_path: Path, go_path: Path, memory_limit_bytes: int | None) -> str:
    setup_lines = ["set -euo pipefail"]
    if memory_limit_bytes is not None:
        # This inherited per-process limit is the reliable cap for the launched
        # runtime and its subprocesses.
        setup_lines.append(f"ulimit -v {max(1, memory_limit_bytes // 1024)}")
    return "\n".join(
        [
            *setup_lines,
            f"printf '%s\\n' \"$$\" > {shlex.quote(str(ready_path))}",
            f"while [ ! -e {shlex.quote(str(go_path))} ]; do sleep 0.01; done",
            f"exec bash -lc {shlex.quote(command)}",
        ]
    )


async def _run_command(*args: str) -> None:
    result = await asyncio.to_thread(
        subprocess.run,
        list(args),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Command failed: {' '.join(args)}: {message}")


async def _wait_until(predicate: object, *, timeout_seconds: float) -> None:
    assert callable(predicate)
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("Timed out waiting for runner startup.")


@cache
def _can_mount_private_cgroup2() -> bool:
    mount_bin = shutil.which("mount")
    umount_bin = shutil.which("umount")
    if mount_bin is None or umount_bin is None:
        return False
    work_dir = Path(tempfile.mkdtemp(prefix="axrl-cgroup-probe-"))
    try:
        result = subprocess.run(
            [mount_bin, "-t", "cgroup2", "none", str(work_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            return False
        return (work_dir / "cgroup.kill").exists()
    finally:
        subprocess.run(
            [umount_bin, str(work_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        shutil.rmtree(work_dir, ignore_errors=True)


def _mount_bin() -> str:
    mount_bin = shutil.which("mount")
    assert mount_bin is not None, "mount must exist when CgroupRunner.is_supported() is true."
    return mount_bin


def _bash_bin() -> str:
    bash_bin = shutil.which("bash")
    assert bash_bin is not None, "bash must exist to run CgroupRunner commands."
    return bash_bin


def _umount_bin() -> str:
    umount_bin = shutil.which("umount")
    assert umount_bin is not None, "umount must exist when CgroupRunner.is_supported() is true."
    return umount_bin


def _safe_cgroup_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)
    return safe[:120] or f"axrl-{uuid.uuid4().hex}"


def _unique_cgroup_name(value: str) -> str:
    suffix = uuid.uuid4().hex[:12]
    max_prefix_len = 120 - len(suffix) - 1
    prefix = _safe_cgroup_name(value)[:max_prefix_len].rstrip("-_.") or "axrl"
    return f"{prefix}-{suffix}"
