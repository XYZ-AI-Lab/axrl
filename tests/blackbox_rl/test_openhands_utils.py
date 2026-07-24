from __future__ import annotations

import asyncio
from pathlib import PurePosixPath
from typing import Any, cast

from axis_recipe.blackbox_rl.config import OpenHandsEnvConfig, OpenHandsLauncherConfig
from axis_recipe.blackbox_rl.openhands_env import OpenHandsEnv
from axis_recipe.blackbox_rl.openhands_utils import get_openhand_launch_script, read_stdout
from axrl.data import Conversation, GenerationOutput, Message, array_utils


def test_get_openhand_launch_script_sets_litellm_timeout() -> None:
    script = get_openhand_launch_script(
        OpenHandsLauncherConfig(llm_timeout_seconds=1200.0),
        task="say hi",
        llm_base_url="http://proxy/v1",
        llm_model="served-model",
    )

    assert "LLM_TIMEOUT=1200" in script


def test_get_openhand_launch_script_disables_public_skills_by_default() -> None:
    script = get_openhand_launch_script(
        OpenHandsLauncherConfig(),
        task="say hi",
        llm_base_url="http://proxy/v1",
        llm_model="served-model",
    )

    assert "HOME=/workspace/.openhands-home" in script
    assert "public-skills" in script
    assert "axrl-empty" in script
    assert '"skills":[]' in script


def test_get_openhand_launch_script_can_load_public_skills_when_enabled() -> None:
    script = get_openhand_launch_script(
        OpenHandsLauncherConfig(load_public_skills=True),
        task="say hi",
        llm_base_url="http://proxy/v1",
        llm_model="served-model",
    )

    assert "HOME=" not in script
    assert "public-skills" not in script


def test_get_openhand_launch_script_omits_local_wrappers_for_e2b() -> None:
    script = get_openhand_launch_script(
        OpenHandsLauncherConfig(),
        task="say hi",
        llm_base_url="http://proxy/v1",
        llm_model="served-model",
    )

    assert "LLM_BASE_URL=http://proxy/v1" in script
    assert "TMUX_PROGRAM=" not in script


def test_read_stdout_accepts_long_json_line() -> None:
    asyncio.run(_test_read_stdout_accepts_long_json_line())


def test_openhands_solution_file_uses_e2b_solution_root() -> None:
    env = cast("OpenHandsEnv", object.__new__(OpenHandsEnv))
    env.config = OpenHandsEnvConfig()
    env.openhands_config = OpenHandsLauncherConfig()
    conv = Conversation(messages=[Message(role="user", content="solve")], conversation_id="two-sum")

    path = env._make_unique_test_file_path(conv)

    assert path.parent == PurePosixPath("/workspace/tmp/openhands-case-study")
    assert "two-sum" in path.name
    assert path.suffix == ".py"


def test_openhands_termination_caches_sandbox_solution() -> None:
    env = cast("OpenHandsEnv", object.__new__(OpenHandsEnv))
    env.config = OpenHandsEnvConfig()
    env.openhands_config = OpenHandsLauncherConfig()
    env.conv = Conversation(messages=[Message(role="user", content="solve")], conversation_id="two-sum")
    env.trace = None
    env.stdout_lines = []
    env.json_events = []
    env._solution_text = ""
    env._runtime_terminated = False
    env.test_file_rel = env._make_unique_test_file_path(env.conv)
    env.test_file_path = PurePosixPath("/workspace/tmp/two-sum.py")
    runner = _FakeOpenHandsRunner("class Solution: ...")
    env.runner = cast("Any", runner)
    env.stdout_reader_task = None

    asyncio.run(env.terminate_runtime())

    assert env.conv.extra["openhands_solution"] == "class Solution: ..."
    assert "openhands_exit_code" in env.conv.extra
    assert env._verifier_text() == "class Solution: ..."
    assert runner.terminate_calls == 1


def test_openhands_verifier_text_prefers_collected_solution() -> None:
    solution = "class Solution:\n    def answer(self):\n        return 1"
    env = _VerifierTextOpenHandsEnv(
        solution=solution,
        outputs=[_generation_output("```python\nclass Wrong:\n    pass\n```")],
        stdout_lines=["```python\nclass AlsoWrong:\n    pass\n```"],
    )

    assert env._verifier_text() == solution


def test_openhands_verifier_text_falls_back_to_model_outputs_without_solution() -> None:
    env = _VerifierTextOpenHandsEnv(
        solution="",
        outputs=[_generation_output("first"), _generation_output("second")],
        stdout_lines=["stdout should not be verifier input"],
    )

    assert env._verifier_text() == "first\n\nsecond"


async def _test_read_stdout_accepts_long_json_line() -> None:
    reader = asyncio.StreamReader(limit=64)
    payload = "x" * 70_000
    reader.feed_data((f'{{"payload": "{payload}"}}\n').encode())
    reader.feed_eof()
    stdout_lines: list[str] = []
    json_events: list[dict[str, object]] = []

    await read_stdout(reader, session_id="test-long-line", stdout_lines=stdout_lines, json_events=json_events)

    assert len(stdout_lines) == 1
    assert json_events == [{"payload": payload}]


class _VerifierTextOpenHandsEnv(OpenHandsEnv):
    def __init__(self, *, solution: str, outputs: list[GenerationOutput], stdout_lines: list[str]) -> None:
        self._solution_text = solution
        self.outputs = outputs
        self.stdout_lines = stdout_lines


class _FakeOpenHandsRunner:
    def __init__(self, solution: str) -> None:
        self.solution = solution
        self.terminate_calls = 0

    async def read_text(self, _path: str) -> str:
        return self.solution

    async def terminate(self) -> None:
        self.terminate_calls += 1

    @property
    def returncode(self) -> int | None:
        return None


def _generation_output(text: str) -> GenerationOutput:
    return GenerationOutput(
        session_id="session",
        output_ids=array_utils.as_i32([1]),
        output_logprobs=array_utils.as_f32([-0.1]),
        output_text=text,
        output_text_with_special_tokens=text,
        cached_tokens=0,
        finish_reason="stop",
        e2e_elapsed_seconds=0.1,
        stop_reason=None,
        retry=0,
    )
