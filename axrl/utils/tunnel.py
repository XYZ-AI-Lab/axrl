from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import re
import shutil
import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import Field

from axrl.configs import StrictBaseModel

if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


class TunnelConfig(StrictBaseModel):
    command: list[str] = Field(
        default_factory=lambda: [
            "cloudflared",
            "tunnel",
            "--url",
            "http://127.0.0.1:{port}",
        ]
    )
    ready_url_regex: str = r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com"
    startup_timeout_seconds: float = 30.0


class Tunnel:
    def __init__(self, *, process: Process, base_url: str, stdout_lines: list[str], drain_task_name: str) -> None:
        self.process = process
        self.base_url = base_url.rstrip("/")
        self.stdout_lines = stdout_lines
        self.drain_task_name = drain_task_name
        self._drain_task: asyncio.Task[None] | None = None

    @classmethod
    async def start(
        cls,
        config: TunnelConfig,
        *,
        template_vars: Mapping[str, Any],
        drain_task_name: str = "axrl-tunnel-drain",
    ) -> Tunnel:
        if not config.command:
            raise ValueError("tunnel.command must be non-empty.")
        command = format_template_sequence(config.command, template_vars)
        if shutil.which(command[0]) is None:
            raise FileNotFoundError(f"Tunnel executable {command[0]!r} was not found.")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_lines: list[str] = []
        pattern = re.compile(config.ready_url_regex)
        try:
            base_url = await _wait_for_tunnel_url(
                process=process,
                stdout_lines=stdout_lines,
                pattern=pattern,
                timeout_seconds=config.startup_timeout_seconds,
            )
        except Exception:
            tunnel = cls(process=process, base_url="", stdout_lines=stdout_lines, drain_task_name=drain_task_name)
            await tunnel.stop()
            raise
        tunnel = cls(process=process, base_url=base_url, stdout_lines=stdout_lines, drain_task_name=drain_task_name)
        tunnel._drain_task = asyncio.create_task(tunnel._drain_stdout(), name=drain_task_name)
        return tunnel

    async def stop(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
        if self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5.0)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()

    async def _drain_stdout(self) -> None:
        assert self.process.stdout is not None
        while True:
            line = await self.process.stdout.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            self.stdout_lines.append(text)
            logger.debug("Tunnel output: %s", text)


def discover_local_ip() -> str:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def format_template(value: str, template_vars: Mapping[str, Any]) -> str:
    return value.format(**template_vars).rstrip("/")


def format_template_sequence(values: Sequence[str], template_vars: Mapping[str, Any]) -> list[str]:
    return [part.format(**template_vars) for part in values]


def allow_out_for_base_url(base_url: str, configured_allow_out: Sequence[str], *, context: str = "network allow_out") -> tuple[str, ...]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Exposed base URL must be an http(s) URL, got {base_url!r}.")
    entries = [parsed.hostname, *configured_allow_out]
    return dedupe_allow_out(entries, context=context)


def dedupe_allow_out(entries: Sequence[str], *, context: str = "network allow_out") -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        value = entry.strip()
        if not value or value in seen:
            continue
        lowered = value.lower()
        if lowered in {"*", "all", "0.0.0.0/0", "::/0"}:
            raise ValueError(f"Refusing broad {context} entry: {entry!r}.")
        if "/" in lowered:
            try:
                network = ipaddress.ip_network(lowered, strict=False)
            except ValueError:
                pass
            else:
                if network.prefixlen == 0:
                    raise ValueError(f"Refusing broad {context} entry: {entry!r}.")
        deduped.append(value)
        seen.add(value)
    return tuple(deduped)


def is_public_routable_host(host: str | None) -> bool:
    if host is None:
        return False
    if host.lower() in {"localhost", "localhost.localdomain"}:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return not host.endswith(".local")
    return not (address.is_loopback or address.is_link_local or address.is_private or address.is_reserved or address.is_unspecified)


async def _wait_for_tunnel_url(
    *,
    process: Process,
    stdout_lines: list[str],
    pattern: re.Pattern[str],
    timeout_seconds: float,
) -> str:
    assert process.stdout is not None
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            output = _preview_tunnel_output(stdout_lines)
            raise TimeoutError(f"Timed out waiting for tunnel URL. Recent tunnel output: {output}")
        line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
        if not line:
            returncode = await process.wait()
            output = _preview_tunnel_output(stdout_lines)
            raise RuntimeError(f"Tunnel exited before exposing a URL; returncode={returncode}; output={output}")
        text = line.decode("utf-8", errors="replace").rstrip("\n")
        stdout_lines.append(text)
        match = pattern.search(text)
        if match is not None:
            url = match.groupdict().get("url")
            if url is None:
                url = match.group(1) if match.groups() else match.group(0)
            return url


def _preview_tunnel_output(lines: list[str]) -> str:
    output = "\n".join(lines[-20:])
    return output[-2000:] if output else "<empty>"
