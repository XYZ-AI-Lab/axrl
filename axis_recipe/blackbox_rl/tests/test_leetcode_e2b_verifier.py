from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from axis_recipe.blackbox_rl import leetcode_verifier as leetcode_module
from axis_recipe.blackbox_rl.leetcode_dataset import make_leetcode_label
from axis_recipe.blackbox_rl.leetcode_verifier import LeetCodeVerifier
from axrl.verifier.base_verifier import VerifierInput
from axrl.verifier.leetcode_executor import SUBPROCESS_RESULT_PREFIX


class FakeFiles:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox

    async def make_dir(self, path: str, **_: Any) -> bool:
        self.sandbox.dirs.append(path)
        return True

    async def write(self, path: str, data: str, **_: Any) -> bool:
        self.sandbox.file_contents[path] = data
        return True


class FakeCommands:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox

    async def run(self, command: str, **kwargs: Any) -> SimpleNamespace:
        self.sandbox.command = command
        self.sandbox.command_kwargs = kwargs
        payload = json.loads(self.sandbox.file_contents[leetcode_module._E2B_PAYLOAD_PATH])
        result = "passed" if "return [0, 1]" in payload["completion"] else "failed: wrong answer"
        return SimpleNamespace(exit_code=0, stdout=f"{SUBPROCESS_RESULT_PREFIX}{json.dumps(result)}\n", stderr="")


class FailingCommands(FakeCommands):
    async def run(self, command: str, **kwargs: Any) -> SimpleNamespace:
        self.sandbox.command = command
        self.sandbox.command_kwargs = kwargs
        raise RuntimeError("temporary E2B 502")


class FakeSandbox:
    created: ClassVar[list[FakeSandbox]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.create_kwargs = kwargs
        self.file_contents: dict[str, str] = {}
        self.dirs: list[str] = []
        self.command = ""
        self.command_kwargs: dict[str, Any] = {}
        self.killed = False
        self.files = FakeFiles(self)
        self.commands = FakeCommands(self)

    @classmethod
    async def create(cls, **kwargs: Any) -> FakeSandbox:
        sandbox = cls(**kwargs)
        cls.created.append(sandbox)
        return sandbox

    async def kill(self, **_: Any) -> bool:
        self.killed = True
        return True


def _two_sum_label() -> str:
    return make_leetcode_label(
        task_id="two-sum",
        prompt="from typing import *\n",
        test=("def check(candidate):\n    assert candidate(nums=[2, 7], target=9) == [0, 1]\n"),
        entry_point="Solution().twoSum",
    )


def test_leetcode_verifier_can_run_in_no_internet_e2b_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.created = []
    monkeypatch.setattr(leetcode_module, "AsyncSandbox", FakeSandbox)
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    verifier = LeetCodeVerifier(
        config={
            "timeout": 3.0,
            "e2b": {
                "template": "axrl-openhands",
                "timeout_seconds": 120,
                "request_timeout_seconds": 60.0,
            },
        }
    )
    output_text = """
```python
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        return [0, 1]
```
"""

    assert verifier.verify(_two_sum_label(), output_text) == 1.0

    sandbox = FakeSandbox.created[0]
    assert sandbox.create_kwargs["allow_internet_access"] is False
    assert sandbox.create_kwargs["api_key"] == "test-key"
    assert leetcode_module._E2B_WORKDIR in sandbox.dirs
    assert leetcode_module._E2B_EXECUTOR_PATH in sandbox.file_contents
    assert "PYTHONPATH=/tmp/axrl-verifier-src" in sandbox.command
    assert sandbox.command_kwargs["cwd"] == leetcode_module._E2B_WORKDIR
    assert sandbox.killed


def test_leetcode_verifier_rejects_non_e2b_runner() -> None:
    with pytest.raises(ValueError, match="only supports runner='e2b'"):
        LeetCodeVerifier(config={"runner": "cgroup"})


def test_leetcode_verifier_treats_e2b_command_failure_as_failed_check(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.created = []
    monkeypatch.setattr(leetcode_module, "AsyncSandbox", FakeSandbox)
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    verifier = LeetCodeVerifier(config={"timeout": 3.0})
    output_text = """
```python
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        return [0, 1]
```
"""

    sandbox = FakeSandbox()
    sandbox.commands = FailingCommands(sandbox)

    async def create(**kwargs: Any) -> FakeSandbox:
        sandbox.create_kwargs = kwargs
        FakeSandbox.created.append(sandbox)
        return sandbox

    monkeypatch.setattr(FakeSandbox, "create", staticmethod(create))

    result = verifier.process(
        VerifierInput(
            label=_two_sum_label(),
            output_text=output_text,
        )
    )

    assert result.score == 0.0
    assert result.infos["result"].startswith("failed: verifier infrastructure error: RuntimeError")
    assert sandbox.killed
