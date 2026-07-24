from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from axrl.runner import e2b_runner as e2b_runner_module
from axrl.runner.e2b_runner import E2BRunner, E2BRunnerConfig


class FakeCommandExitError(Exception):
    def __init__(self, exit_code: int) -> None:
        super().__init__(f"exit {exit_code}")
        self.exit_code = exit_code


class FakeFileNotFoundError(Exception):
    pass


class FakeCommandHandle:
    def __init__(self, *, exit_code: int = 0, raise_exit: bool = False) -> None:
        self.exit_code: int | None = None
        self.final_exit_code = exit_code
        self.raise_exit = raise_exit
        self.killed = False
        self.wait_started = asyncio.Event()
        self.release_wait = asyncio.Event()

    async def wait(self) -> Any:
        self.wait_started.set()
        await self.release_wait.wait()
        self.exit_code = self.final_exit_code
        if self.raise_exit:
            raise FakeCommandExitError(self.final_exit_code)
        return SimpleNamespace(exit_code=self.final_exit_code)

    async def kill(self) -> bool:
        self.killed = True
        self.exit_code = -9
        self.release_wait.set()
        return True


class FakeCommands:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.handle = FakeCommandHandle()

    async def run(self, command: str, **kwargs: Any) -> FakeCommandHandle:
        self.sandbox.command = command
        self.sandbox.command_kwargs = kwargs
        kwargs["on_stdout"]("stdout line\n")
        kwargs["on_stderr"]("stderr line\n")
        return self.handle


class FakeFiles:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.contents: dict[str, str] = {}

    async def make_dir(self, path: str, **_: Any) -> bool:
        self.sandbox.dirs.append(path)
        return True

    async def read(self, path: str, **_: Any) -> str:
        if path not in self.contents:
            raise FakeFileNotFoundError
        return self.contents[path]


class FakeSandbox:
    created: ClassVar[list[FakeSandbox]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.create_kwargs = kwargs
        self.commands = FakeCommands(self)
        self.files = FakeFiles(self)
        self.dirs: list[str] = []
        self.command = ""
        self.command_kwargs: dict[str, Any] = {}
        self.killed = False

    @classmethod
    async def create(cls, **kwargs: Any) -> FakeSandbox:
        sandbox = cls(**kwargs)
        cls.created.append(sandbox)
        return sandbox

    async def kill(self, **_: Any) -> bool:
        self.killed = True
        return True


def test_e2b_runner_creates_sandbox_with_restricted_network(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_test_e2b_runner_creates_sandbox_with_restricted_network(monkeypatch))


async def _test_e2b_runner_creates_sandbox_with_restricted_network(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.created = []
    monkeypatch.setattr(e2b_runner_module, "AsyncSandbox", FakeSandbox)
    monkeypatch.setattr(e2b_runner_module, "CommandExitException", FakeCommandExitError)
    monkeypatch.setattr(e2b_runner_module, "FileNotFoundException", FakeFileNotFoundError)
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    runner = E2BRunner(
        name="test-openhands",
        config=E2BRunnerConfig(template="template-id", timeout_seconds=123, workdir="/workspace"),
        allow_out=["proxy.example.com", "proxy.example.com"],
        setup_dirs=["/workspace/tmp"],
    )

    await runner.start("echo hello", Path("/workspace"))
    sandbox = FakeSandbox.created[0]
    assert sandbox.create_kwargs["template"] == "template-id"
    assert sandbox.create_kwargs["timeout"] == 123
    assert sandbox.create_kwargs["network"] == {"allow_out": ["proxy.example.com"], "deny_out": ["0.0.0.0/0"]}
    assert sandbox.create_kwargs["api_key"] == "test-key"
    assert sandbox.dirs == ["/workspace", "/workspace/tmp"]
    assert sandbox.command == "echo hello"
    assert sandbox.command_kwargs["background"] is True
    assert sandbox.command_kwargs["cwd"] == "/workspace"
    stdout = runner.stdout
    assert stdout is not None
    assert await stdout.read(1024) == b"stdout line\nstderr line\n"

    sandbox.commands.handle.release_wait.set()
    await runner.terminate()
    assert sandbox.killed


def test_e2b_runner_rejects_empty_or_broad_allow_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e2b_runner_module, "AsyncSandbox", FakeSandbox)
    empty_runner = E2BRunner(name="empty", allow_out=[])
    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(empty_runner.start("true", Path("/workspace")))

    broad_runner = E2BRunner(name="broad", allow_out=["0.0.0.0/0"])
    with pytest.raises(ValueError, match="broad"):
        asyncio.run(broad_runner.start("true", Path("/workspace")))


def test_e2b_runner_records_nonzero_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_test_e2b_runner_records_nonzero_returncode(monkeypatch))


async def _test_e2b_runner_records_nonzero_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.created = []
    monkeypatch.setattr(e2b_runner_module, "AsyncSandbox", FakeSandbox)
    monkeypatch.setattr(e2b_runner_module, "CommandExitException", FakeCommandExitError)
    sandbox = await FakeSandbox.create()
    sandbox.commands.handle = FakeCommandHandle(exit_code=7, raise_exit=True)
    FakeSandbox.created = []

    async def create(**kwargs: Any) -> FakeSandbox:
        sandbox.create_kwargs = kwargs
        FakeSandbox.created.append(sandbox)
        return sandbox

    monkeypatch.setattr(FakeSandbox, "create", staticmethod(create))
    runner = E2BRunner(name="nonzero", allow_out=["proxy.example.com"])
    await runner.start("exit 7", Path("/workspace"))

    sandbox.commands.handle.release_wait.set()
    assert runner._wait_task is not None
    await runner._wait_task

    assert runner.returncode == 7


def test_e2b_runner_reads_text_from_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_test_e2b_runner_reads_text_from_sandbox(monkeypatch))


async def _test_e2b_runner_reads_text_from_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.created = []
    monkeypatch.setattr(e2b_runner_module, "AsyncSandbox", FakeSandbox)
    monkeypatch.setattr(e2b_runner_module, "CommandExitException", FakeCommandExitError)
    monkeypatch.setattr(e2b_runner_module, "FileNotFoundException", FakeFileNotFoundError)
    runner = E2BRunner(name="read", allow_out=["proxy.example.com"])
    await runner.start("sleep 100", Path("/workspace"))
    sandbox = FakeSandbox.created[0]
    sandbox.files.contents["/workspace/tmp/solution.py"] = "class Solution: pass"

    assert await runner.read_text("/workspace/tmp/solution.py") == "class Solution: pass"
    assert await runner.read_text("/workspace/tmp/missing.py") == ""

    await runner.terminate()
