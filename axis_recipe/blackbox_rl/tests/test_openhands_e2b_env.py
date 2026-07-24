from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, cast

from axis_recipe.blackbox_rl import openhands_env as openhands_env_module
from axis_recipe.blackbox_rl.config import OpenHandsEnvConfig, OpenHandsLauncherConfig
from axis_recipe.blackbox_rl.openhands_env import OpenHandsEnv
from axrl.data import Conversation, Message

if TYPE_CHECKING:
    import pytest


class FakeE2BRunner:
    instances: ClassVar[list[FakeE2BRunner]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.start_command = ""
        self.start_cwd: Path | None = None
        self.terminated = False
        self.read_paths: list[str] = []
        self.stdout_reader = asyncio.StreamReader()
        self.stdout_reader.feed_eof()
        FakeE2BRunner.instances.append(self)

    async def start(self, command: str, cwd: Path) -> None:
        self.start_command = command
        self.start_cwd = cwd

    async def terminate(self) -> None:
        self.terminated = True

    async def read_text(self, path: str) -> str:
        assert not self.terminated
        self.read_paths.append(path)
        return "class Solution: pass"

    @property
    def stdout(self) -> asyncio.StreamReader:
        return self.stdout_reader

    @property
    def returncode(self) -> int | None:
        return None


def test_openhands_launch_uses_e2b_runner_without_host_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_test_openhands_launch_uses_e2b_runner_without_host_wrappers(monkeypatch))


async def _test_openhands_launch_uses_e2b_runner_without_host_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeE2BRunner.instances = []
    monkeypatch.setattr(openhands_env_module, "E2BRunner", FakeE2BRunner)
    env = cast("OpenHandsEnv", object.__new__(OpenHandsEnv))
    env.openhands_config = OpenHandsLauncherConfig()
    env.config = OpenHandsEnvConfig()
    env.llm_base_url = "https://proxy.example.com/sessions/session/v1"
    env.llm_model = "served-model"
    env.session_id = "session"
    env.original_conv = Conversation(
        messages=[Message(role="user", content="solve")],
        conversation_id="case",
        extra={"answer": "ok", "entry_point": "twoSum", "starter_code": "class Solution: pass"},
    )
    env.test_file_path = PurePosixPath("/workspace/tmp/openhands-case-study/case.py")
    env.test_file_rel = PurePosixPath("tmp/openhands-case-study/case.py")
    env.e2b_allow_out = ("proxy.example.com",)

    await env.launch_runtime()

    runner = FakeE2BRunner.instances[0]
    assert runner.kwargs["allow_out"] == ("proxy.example.com",)
    assert runner.kwargs["setup_dirs"] == ("/workspace/tmp/openhands-case-study",)
    assert runner.start_cwd == Path("/workspace")
    assert "LLM_BASE_URL=https://proxy.example.com/sessions/session/v1" in runner.start_command
    assert "TMUX_PROGRAM=" not in runner.start_command
    assert "HOME=/workspace/.openhands-home" in runner.start_command


def test_openhands_terminate_reads_solution_before_killing_sandbox() -> None:
    asyncio.run(_test_openhands_terminate_reads_solution_before_killing_sandbox())


async def _test_openhands_terminate_reads_solution_before_killing_sandbox() -> None:
    env = cast("OpenHandsEnv", object.__new__(OpenHandsEnv))
    env.config = OpenHandsEnvConfig()
    env.openhands_config = OpenHandsLauncherConfig()
    env.conv = Conversation(messages=[Message(role="user", content="solve")], conversation_id="case")
    env.trace = None
    env.stdout_lines = []
    env.json_events = []
    env._solution_text = ""
    env._runtime_terminated = False
    env.test_file_path = PurePosixPath("/workspace/tmp/openhands-case-study/case.py")
    env.test_file_rel = PurePosixPath("tmp/openhands-case-study/case.py")
    runner = FakeE2BRunner(name="fake", config=env.openhands_config.e2b, allow_out=("proxy.example.com",))
    env.runner = cast("Any", runner)
    env.stdout_reader_task = None

    await env.terminate_runtime()
    await env.terminate_runtime()

    assert env._solution_text == "class Solution: pass"
    assert env.conv.extra["openhands_solution"] == "class Solution: pass"
    assert runner.read_paths == ["/workspace/tmp/openhands-case-study/case.py"]
    assert runner.terminated
