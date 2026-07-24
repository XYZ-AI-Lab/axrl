from __future__ import annotations

from abc import ABC, abstractmethod
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio
    from pathlib import Path

DEFAULT_TERMINATE_TIMEOUT_SECONDS = 5.0


class BaseRunner(ABC):
    @abstractmethod
    async def start(self, command: str, cwd: Path) -> None:
        """Start one managed runtime."""

    @abstractmethod
    async def terminate(self, *, timeout_seconds: float = DEFAULT_TERMINATE_TIMEOUT_SECONDS) -> None:
        """Stop the runtime and clean up any remaining child processes."""

    @property
    @abstractmethod
    def stdout(self) -> asyncio.StreamReader | IO[bytes] | None:
        """Return the live stdout stream, if this runner captures one."""

    @property
    @abstractmethod
    def returncode(self) -> int | None:
        """Return the launcher exit code after it exits."""
