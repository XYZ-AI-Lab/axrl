from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, override

from e2b import ALL_TRAFFIC, AsyncCommandHandle, AsyncSandbox, CommandExitException, FileNotFoundException
from pydantic import Field

from axrl.configs import StrictBaseModel
from axrl.runner.base_runner import DEFAULT_TERMINATE_TIMEOUT_SECONDS, BaseRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_BROAD_ALLOW_OUT = {"*", "all", "0.0.0.0/0", "::/0"}


class E2BRunnerConfig(StrictBaseModel):
    template: str = "axrl-openhands"
    timeout_seconds: int = 1800
    command_timeout_seconds: float | None = 0.0
    request_timeout_seconds: float | None = 60.0
    secure: bool = True
    api_key_env: str = "E2B_API_KEY"
    dotenv_path: str | None = ".env"
    workdir: str = "/workspace"
    solution_root: str = "/workspace"
    metadata: dict[str, str] = Field(default_factory=dict)
    envs: dict[str, str] = Field(default_factory=dict)
    user: str | None = None


class E2BRunner(BaseRunner):
    """Run a command inside an E2B sandbox with restricted outbound network access."""

    def __init__(
        self,
        *,
        name: str,
        config: E2BRunnerConfig | None = None,
        allow_out: Sequence[str],
        setup_dirs: Sequence[str] = (),
    ) -> None:
        self.name = name
        self.config = config or E2BRunnerConfig()
        self.allow_out = _normalize_allow_out(allow_out)
        self.setup_dirs = tuple(str(path) for path in setup_dirs)
        self._stdout: asyncio.StreamReader | None = None
        self._stdout_closed = False
        self._sandbox: AsyncSandbox | None = None
        self._command: AsyncCommandHandle | None = None
        self._wait_task: asyncio.Task[None] | None = None
        self._returncode: int | None = None
        self._terminated = False

    @override
    async def start(self, command: str, cwd: Path) -> None:
        assert self._sandbox is None, "E2BRunner.start() can only be called once."
        _validate_allow_out(self.allow_out)
        self._stdout = asyncio.StreamReader()
        api_key = resolve_e2b_api_key(self.config)
        metadata = {"axrl_runner": self.name, **self.config.metadata}
        sandbox = await AsyncSandbox.create(
            template=self.config.template,
            timeout=self.config.timeout_seconds,
            metadata=metadata,
            envs=self.config.envs,
            secure=self.config.secure,
            network={"allow_out": list(self.allow_out), "deny_out": [ALL_TRAFFIC]},
            api_key=api_key,
            request_timeout=self.config.request_timeout_seconds,
        )
        self._sandbox = sandbox
        try:
            await self._ensure_remote_dirs(cwd)
            self._command = await sandbox.commands.run(
                command,
                background=True,
                cwd=str(cwd),
                user=self.config.user,
                on_stdout=self._feed_text,
                on_stderr=self._feed_text,
                timeout=self.config.command_timeout_seconds,
                request_timeout=self.config.request_timeout_seconds,
            )
        except Exception:
            await self.terminate()
            raise
        self._wait_task = asyncio.create_task(self._wait_command(), name=f"{self.name}-e2b-wait")

    @override
    async def terminate(self, *, timeout_seconds: float = DEFAULT_TERMINATE_TIMEOUT_SECONDS) -> None:
        if self._terminated:
            return
        self._terminated = True
        if self._command is not None and self.returncode is None:
            with suppress(Exception):
                await asyncio.wait_for(self._command.kill(), timeout=timeout_seconds)
        if self._wait_task is not None and not self._wait_task.done():
            with suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(self._wait_task), timeout=timeout_seconds)
            if not self._wait_task.done():
                self._wait_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._wait_task
        if self._sandbox is not None:
            with suppress(Exception):
                await self._sandbox.kill(request_timeout=self.config.request_timeout_seconds)
        self._feed_eof()

    @property
    @override
    def stdout(self) -> asyncio.StreamReader | IO[bytes] | None:
        return self._stdout

    @property
    @override
    def returncode(self) -> int | None:
        if self._returncode is not None:
            return self._returncode
        if self._command is None:
            return None
        exit_code = getattr(self._command, "exit_code", None)
        return None if exit_code is None else int(exit_code)

    async def read_text(self, path: str) -> str:
        if self._sandbox is None:
            return ""
        try:
            content = await self._sandbox.files.read(
                path,
                request_timeout=self.config.request_timeout_seconds,
            )
        except FileNotFoundException:
            return ""
        if isinstance(content, bytes | bytearray):
            return bytes(content).decode("utf-8", errors="replace")
        return str(content)

    async def _ensure_remote_dirs(self, cwd: Path) -> None:
        assert self._sandbox is not None
        for directory in _unique_remote_dirs([str(cwd), *self.setup_dirs]):
            await self._sandbox.files.make_dir(
                directory,
                user=self.config.user,
                request_timeout=self.config.request_timeout_seconds,
            )

    async def _wait_command(self) -> None:
        assert self._command is not None
        try:
            result = await self._command.wait()
            self._returncode = int(result.exit_code)
        except CommandExitException as exc:
            self._returncode = int(exc.exit_code)
            logger.warning("E2B command %s exited with code %s.", self.name, self._returncode)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if _is_e2b_closed_transport_error(exc):
                logger.warning("E2B command %s wait failed after the E2B SDK transport closed: %s", self.name, exc)
                logger.debug("E2B command %s wait transport-close traceback.", self.name, exc_info=True)
            else:
                logger.warning("E2B command %s wait failed.", self.name, exc_info=True)
        finally:
            self._feed_eof()

    def _feed_text(self, data: str) -> None:
        if self._stdout_closed or self._stdout is None:
            return
        self._stdout.feed_data(data.encode("utf-8", errors="replace"))

    def _feed_eof(self) -> None:
        if self._stdout_closed or self._stdout is None:
            return
        self._stdout.feed_eof()
        self._stdout_closed = True


def resolve_e2b_api_key(config: E2BRunnerConfig) -> str | None:
    api_key = os.environ.get(config.api_key_env)
    if api_key:
        return api_key
    api_key = _read_dotenv_value(config.dotenv_path, config.api_key_env)
    if api_key:
        return api_key
    return None


def _read_dotenv_value(dotenv_path: str | None, key: str) -> str | None:
    if dotenv_path is None:
        return None
    path = Path(dotenv_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        name, sep, value = line.partition("=")
        if sep != "=" or name.strip() != key:
            continue
        return _strip_dotenv_quotes(value.strip())
    return None


def _strip_dotenv_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _normalize_allow_out(allow_out: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in allow_out:
        value = entry.strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return tuple(normalized)


def _validate_allow_out(allow_out: Sequence[str]) -> None:
    if not allow_out:
        raise ValueError("E2BRunner requires a non-empty network allow_out list; omitting it would allow public internet egress.")
    for entry in allow_out:
        lowered = entry.strip().lower()
        if lowered in _BROAD_ALLOW_OUT:
            raise ValueError(f"E2BRunner refuses broad outbound network allowlist entry: {entry!r}.")
        if "/" not in lowered:
            continue
        try:
            network = ipaddress.ip_network(lowered, strict=False)
        except ValueError:
            continue
        if network.prefixlen == 0:
            raise ValueError(f"E2BRunner refuses broad outbound network allowlist entry: {entry!r}.")


def _is_e2b_closed_transport_error(exc: BaseException) -> bool:
    message = str(exc)
    return "ConnectionState.CLOSED" in message


def _unique_remote_dirs(paths: Sequence[str]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for path in paths:
        value = str(PurePosixPath(path))
        if not value or value == "." or value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return tuple(unique)
