"""Utilities for launching standalone SGLang HTTP servers."""

from __future__ import annotations

import asyncio
import logging
import shlex
import socket
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import httpx
import ray

from axrl.runner import BaseRunner, CgroupRunner
from axrl.utils.network_utils import get_available_port

if TYPE_CHECKING:
    from collections.abc import Sequence

    from axrl.configs import RolloutWorkerConfig
    from axrl.ray.resource_group import ResourceGroup

logger = logging.getLogger(__name__)

__all__ = ["SGLangServiceHandle", "start_sglang_router"]

_DEFAULT_WORKER_HOST = "0.0.0.0"  # noqa: S104 - remote workers must be reachable from the router.
_DEFAULT_ROUTER_POLICY = "cache_aware"
_DEFAULT_WAIT_READY_TIMEOUT_SECONDS = 600.0


@dataclass
class SGLangServiceHandle:
    """Handle for an SGLang HTTP service launched by AXRL."""

    runner: BaseRunner
    host: str
    port: int
    base_url: str
    command: str
    name: str
    children: list[SGLangServiceHandle] = field(default_factory=list)
    stdout_lines: list[str] = field(default_factory=list)
    _stdout_task: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/generate"

    async def wait_ready(self, *, timeout_seconds: float = 600.0) -> None:
        await _wait_http_ready(
            base_url=self.base_url,
            runner=self.runner,
            stdout_lines=self.stdout_lines,
            timeout_seconds=timeout_seconds,
            name=self.name,
        )

    async def terminate(self) -> None:
        try:
            await self.runner.terminate()
            if self._stdout_task is not None:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stdout_task, timeout=5.0)
        finally:
            for handle in reversed(self.children):
                await handle.terminate()


@dataclass(frozen=True)
class _SGLangWorkerSpec:
    bundle_index: int
    host: str
    port: int
    num_gpus: int


@ray.remote
class _RemoteCgroupRunner:
    def __init__(self, *, name: str) -> None:
        self.runner = CgroupRunner(name=name)
        self.stdout_lines: list[str] = []
        self._stdout_task: asyncio.Task[None] | None = None

    async def start(self, command: str, cwd: str) -> None:
        await self.runner.start(command, Path(cwd))
        stdout = self.runner.stdout
        if stdout is not None:
            self._stdout_task = asyncio.create_task(_drain_stdout(stdout, self.stdout_lines))

    async def terminate(self, *, timeout_seconds: float = 5.0) -> None:
        try:
            await self.runner.terminate(timeout_seconds=timeout_seconds)
            if self._stdout_task is not None:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stdout_task, timeout=5.0)
        finally:
            self._stdout_task = None

    def returncode(self) -> int | None:
        return self.runner.returncode

    def stdout_tail(self, max_lines: int = 20) -> list[str]:
        return self.stdout_lines[-max_lines:]


class _RayCgroupRunner(BaseRunner):
    """Local ``BaseRunner`` wrapper for a CGroup runner owned by a Ray actor."""

    def __init__(self, actor: Any) -> None:
        self.actor = actor
        self._terminated = False

    async def start(self, command: str, cwd: Path) -> None:
        await asyncio.to_thread(ray.get, self.actor.start.remote(command, str(cwd)))

    async def terminate(self, *, timeout_seconds: float = 5.0) -> None:
        if self._terminated:
            return
        try:
            await asyncio.to_thread(ray.get, self.actor.terminate.remote(timeout_seconds=timeout_seconds))
        finally:
            with suppress(Exception):
                ray.kill(self.actor, no_restart=True)
            self._terminated = True

    @property
    def stdout(self) -> asyncio.StreamReader | IO[bytes] | None:
        return None

    @property
    def returncode(self) -> int | None:
        try:
            return ray.get(self.actor.returncode.remote(), timeout=1.0)
        except Exception:
            logger.debug("Could not read remote CGroup runner return code.", exc_info=True)
            return None

    def stdout_tail(self, max_lines: int = 20) -> list[str]:
        try:
            return ray.get(self.actor.stdout_tail.remote(max_lines), timeout=1.0)
        except Exception:
            logger.debug("Could not read remote CGroup runner stdout.", exc_info=True)
            return []


def _build_sglang_server_command(
    config: RolloutWorkerConfig,
    *,
    host: str,
    port: int,
    gpus: Sequence[int] | None = None,
) -> str:
    """Build the shell command used to launch a standalone SGLang server."""
    env = {
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_NVLS_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_DEBUG": "WARN",
    }
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)

    args = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(config.model.get_full_path()),
        "--host",
        host,
        "--port",
        str(port),
        "--mem-fraction-static",
        str(config.gpu_memory_utilization),
        "--tp-size",
        str(config.tp_size),
        "--pp-size",
        str(config.pp_size),
        "--dp-size",
        str(config.dp_size),
        "--ep-size",
        str(config.ep_size),
        "--moe-a2a-backend",
        config.moe_a2a_backend,
        "--load-format",
        "dummy" if config.load_dummy_weights else "auto",
        "--max-running-requests",
        str(config.max_running_requests),
        "--dtype",
        config.dtype,
        "--kv-cache-dtype",
        config.kv_cache_dtype,
        "--log-level",
        config.log_level,
        "--decode-log-interval",
        "8192",
    ]
    if config.model.trust_remote_code:
        args.append("--trust-remote-code")
    if config.enable_fp32_lm_head:
        args.append("--enable-fp32-lm-head")
    if config.enable_metrics:
        args.append("--enable-metrics")
    if config.enable_routing_replay:
        args.append("--enable-return-routed-experts")
    if config.moe_runner_backend is not None:
        args.extend(["--moe-runner-backend", config.moe_runner_backend])
    if config.attention_backend is not None:
        args.extend(["--attention-backend", config.attention_backend])
    if config.prefill_max_requests is not None:
        args.extend(["--prefill-max-requests", str(config.prefill_max_requests)])

    return shlex.join(["env", *[f"{key}={value}" for key, value in env.items()], *args])


def _build_sglang_router_command(
    *,
    worker_urls: Sequence[str],
    host: str,
    port: int,
) -> str:
    """Build the shell command used to launch an SGLang router."""
    if not worker_urls:
        raise ValueError("SGLang router requires at least one worker URL.")
    args = [
        sys.executable,
        "-m",
        "sglang_router",
        "launch",
        "--host",
        host,
        "--port",
        str(port),
        "--worker-urls",
        *worker_urls,
        "--policy",
        _DEFAULT_ROUTER_POLICY,
    ]
    return shlex.join(args)


async def _start_sglang_server(
    config: RolloutWorkerConfig,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    gpus: Sequence[int] | None = None,
    cwd: Path | None = None,
) -> SGLangServiceHandle:
    """Start a standalone SGLang HTTP server for native ``/generate`` calls."""
    selected_port = get_available_port() if port is None else port
    selected_runner = CgroupRunner(name=f"sglang-server-{selected_port}")
    command = _build_sglang_server_command(
        config,
        host=host,
        port=selected_port,
        gpus=gpus,
    )
    handle = SGLangServiceHandle(
        runner=selected_runner,
        host=host,
        port=selected_port,
        base_url=f"http://{host}:{selected_port}",
        command=command,
        name="SGLang server",
    )
    logger.info("Starting SGLang server: %s", command)
    await selected_runner.start(command, cwd or Path.cwd())
    stdout = selected_runner.stdout
    if stdout is not None:
        handle._stdout_task = asyncio.create_task(_drain_stdout(stdout, handle.stdout_lines))
    try:
        await handle.wait_ready(timeout_seconds=_DEFAULT_WAIT_READY_TIMEOUT_SECONDS)
    except Exception:
        await handle.terminate()
        raise
    return handle


async def start_sglang_router(
    resource_group: ResourceGroup,
    config: RolloutWorkerConfig,
    *,
    router_host: str | None = None,
    router_port: int | None = None,
) -> SGLangServiceHandle:
    """Start SGLang workers from a Ray resource group and route through ``sglang_router``.

    ``router_host`` and ``router_port`` define the deterministic endpoint used
    by both the router bind address and ``SGLangServiceHandle.base_url``.
    """
    worker_bundle_indices = list(range(len(resource_group.bundle_infos)))
    if not worker_bundle_indices:
        raise ValueError("start_sglang_router requires at least one worker bundle.")

    router_bundle_index = 0
    selected_router_host = router_host if router_host is not None else resource_group.bundle_infos[router_bundle_index].ip
    selected_router_port = router_port if router_port is not None else _get_remote_available_port(resource_group, router_bundle_index)
    worker_specs = _build_worker_specs(
        resource_group,
        config,
        bundle_indices=worker_bundle_indices,
    )
    worker_handles: list[SGLangServiceHandle] = []
    router_handle: SGLangServiceHandle | None = None
    selected_cwd = Path.cwd()
    try:
        worker_handles = [
            _build_remote_worker_handle(
                resource_group,
                config,
                index=index,
                spec=spec,
                worker_host=_DEFAULT_WORKER_HOST,
            )
            for index, spec in enumerate(worker_specs)
        ]
        await asyncio.gather(
            *[
                _start_handle(
                    handle,
                    cwd=selected_cwd,
                    wait_ready_timeout_seconds=_DEFAULT_WAIT_READY_TIMEOUT_SECONDS,
                )
                for handle in worker_handles
            ]
        )

        router_command = _build_sglang_router_command(
            worker_urls=[handle.base_url for handle in worker_handles],
            host=selected_router_host,
            port=selected_router_port,
        )
        selected_router_runner = _create_ray_cgroup_runner(
            resource_group,
            bundle_index=router_bundle_index,
            name=f"sglang-router-{selected_router_port}",
            num_gpus=0,
        )
        router_handle = SGLangServiceHandle(
            runner=selected_router_runner,
            host=selected_router_host,
            port=selected_router_port,
            base_url=f"http://{selected_router_host}:{selected_router_port}",
            command=router_command,
            name="SGLang router",
            children=worker_handles,
        )
        await _start_handle(
            router_handle,
            cwd=selected_cwd,
            wait_ready_timeout_seconds=_DEFAULT_WAIT_READY_TIMEOUT_SECONDS,
        )
        return router_handle
    except Exception:
        if router_handle is not None:
            with suppress(Exception):
                await router_handle.runner.terminate()
        for handle in reversed(worker_handles):
            with suppress(Exception):
                await handle.terminate()
        raise


def _create_ray_cgroup_runner(
    resource_group: ResourceGroup,
    *,
    bundle_index: int,
    name: str,
    num_gpus: float,
) -> _RayCgroupRunner:
    actor = _RemoteCgroupRunner.options(  # type: ignore[attr-defined]
        num_cpus=0,
        num_gpus=num_gpus,
        scheduling_strategy=resource_group.get_scheduling_strategy(bundle_index),
    ).remote(name=name)
    return _RayCgroupRunner(actor)


def _get_remote_available_port(resource_group: ResourceGroup, bundle_index: int) -> int:
    return int(resource_group.run_task(func=get_available_port, bundle_index=bundle_index, num_gpus=0, num_cpus=0))


def _get_available_ports(count: int) -> list[int]:
    if count <= 0:
        return []
    sockets: list[socket.socket] = []
    try:
        ports: list[int] = []
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", 0))
            sock.listen(1)
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
        return ports
    finally:
        for sock in sockets:
            sock.close()


def _get_remote_available_ports(resource_group: ResourceGroup, bundle_index: int, count: int) -> list[int]:
    return list(resource_group.run_task(func=_get_available_ports, bundle_index=bundle_index, num_gpus=0, num_cpus=0, count=count))


def _build_worker_specs(
    resource_group: ResourceGroup,
    config: RolloutWorkerConfig,
    *,
    bundle_indices: Sequence[int],
) -> list[_SGLangWorkerSpec]:
    gpus_per_worker = config.gpus_per_worker()
    if gpus_per_worker <= 0:
        raise ValueError(f"gpus_per_worker must be greater than zero, got {gpus_per_worker}.")

    specs_without_ports: list[tuple[int, str, int]] = []
    for bundle_index in bundle_indices:
        request = resource_group.requests[bundle_index]
        bundle_gpus = int(request.gpu)
        if bundle_gpus <= 0:
            raise ValueError(f"Worker bundle {bundle_index} must have at least one GPU.")
        if bundle_gpus % gpus_per_worker != 0:
            raise ValueError(f"Worker bundle {bundle_index} has {bundle_gpus} GPUs, which is not divisible by gpus_per_worker={gpus_per_worker}.")
        host = resource_group.bundle_infos[bundle_index].ip
        for _ in range(bundle_gpus // gpus_per_worker):
            specs_without_ports.append((bundle_index, host, gpus_per_worker))

    if len(specs_without_ports) != config.num_workers:
        raise ValueError(f"Resource group layout creates {len(specs_without_ports)} SGLang workers, but config.num_workers={config.num_workers}.")
    bundle_index_by_host: dict[str, int] = {}
    count_by_host: dict[str, int] = {}
    for bundle_index, host, _num_gpus in specs_without_ports:
        bundle_index_by_host.setdefault(host, bundle_index)
        count_by_host[host] = count_by_host.get(host, 0) + 1
    ports_by_host = {
        host: iter(_get_remote_available_ports(resource_group, bundle_index_by_host[host], count)) for host, count in count_by_host.items()
    }
    selected_worker_ports = [next(ports_by_host[host]) for _bundle_index, host, _num_gpus in specs_without_ports]
    return [
        _SGLangWorkerSpec(bundle_index=bundle_index, host=host, port=port, num_gpus=num_gpus)
        for (bundle_index, host, num_gpus), port in zip(specs_without_ports, selected_worker_ports, strict=True)
    ]


def _build_remote_worker_handle(
    resource_group: ResourceGroup,
    config: RolloutWorkerConfig,
    *,
    index: int,
    spec: _SGLangWorkerSpec,
    worker_host: str,
) -> SGLangServiceHandle:
    command = _build_sglang_server_command(
        config,
        host=worker_host,
        port=spec.port,
        gpus=None,
    )
    runner = _create_ray_cgroup_runner(
        resource_group,
        bundle_index=spec.bundle_index,
        name=f"sglang-router-worker-{index}-{spec.port}",
        num_gpus=spec.num_gpus,
    )
    return SGLangServiceHandle(
        runner=runner,
        host=spec.host,
        port=spec.port,
        base_url=f"http://{spec.host}:{spec.port}",
        command=command,
        name=f"SGLang worker {index}",
    )


async def _start_handle(
    handle: SGLangServiceHandle,
    *,
    cwd: Path,
    wait_ready_timeout_seconds: float,
) -> None:
    logger.info("Starting %s: %s", handle.name, handle.command)
    await handle.runner.start(handle.command, cwd)
    stdout = handle.runner.stdout
    if stdout is not None:
        handle._stdout_task = asyncio.create_task(_drain_stdout(stdout, handle.stdout_lines))
    await handle.wait_ready(timeout_seconds=wait_ready_timeout_seconds)


async def _wait_http_ready(
    *,
    base_url: str,
    runner: BaseRunner,
    stdout_lines: list[str],
    timeout_seconds: float,
    name: str,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    health_url = f"{base_url}/health"
    async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
        while asyncio.get_running_loop().time() < deadline:
            if runner.returncode is not None:
                tail_lines = stdout_lines[-20:]
                if not tail_lines and isinstance(runner, _RayCgroupRunner):
                    tail_lines = runner.stdout_tail()
                stdout_tail = "\n".join(tail_lines)
                raise RuntimeError(f"{name} exited before becoming ready: returncode={runner.returncode}\n{stdout_tail}")
            try:
                response = await client.get(health_url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {name} at {health_url}.")


async def _drain_stdout(stream: IO[bytes] | asyncio.StreamReader, lines: list[str]) -> None:
    while True:
        if isinstance(stream, asyncio.StreamReader):
            line = await stream.readline()
        else:
            line = await asyncio.to_thread(stream.readline)
        if not line:
            return
        text = line.decode(errors="replace").rstrip()
        lines.append(text)
        if len(lines) > 2000:
            del lines[:1000]
        logger.debug("SGLang server: %s", text)
