from __future__ import annotations

import asyncio
import logging
import random
import re
import string
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, override

from axis_recipe.blackbox_rl.blackbox_env import BlackBoxEnv
from axis_recipe.blackbox_rl.data import BlackBoxResponseMetric
from axis_recipe.blackbox_rl.openhands_utils import get_openhand_launch_script, read_stdout
from axrl.runner.e2b_runner import E2BRunner
from axrl.verifier.base_verifier import VerifierInput

if TYPE_CHECKING:
    from axis_recipe.blackbox_rl.config import OpenHandsEnvConfig, OpenHandsLauncherConfig
    from axrl.data import Conversation, GenerationOutput
    from axrl.metrics.response_metric import ResponseMetric
    from axrl.openai_proxy import OpenAIProxySessionRegistry
    from axrl.openai_proxy.chat_adapter import OpenAIChatAdapterInput, OpenAIChatAdapterOutput
    from axrl.processor.processor_pool import ProcessorPool
    from axrl.verifier.base_verifier import VerifierOutput
    from axrl.worker.infer_worker import InferWorker

logger = logging.getLogger(__name__)

_OPENHANDS_FINISH_DRAIN_SECONDS = 5.0


@dataclass
class OpenHandsResponseMetric(BlackBoxResponseMetric):
    openhands_exit_code: int | None = None
    openhands_stdout_lines: int = 0
    collected_solution_chars: int = 0


class OpenHandsEnv(BlackBoxEnv):
    """OpenHands runtime implementation for the generic black-box env loop."""

    def __init__(
        self,
        *,
        conv: Conversation,
        label: str | list[str],
        registry: OpenAIProxySessionRegistry,
        adapter: ProcessorPool[OpenAIChatAdapterInput, OpenAIChatAdapterOutput],
        openhands_config: OpenHandsLauncherConfig,
        llm_base_url: str,
        e2b_allow_out: tuple[str, ...],
        llm_model: str,
        score_provider: InferWorker[VerifierInput, VerifierOutput],
        metric_calculator: InferWorker[GenerationOutput, ResponseMetric],
        config: OpenHandsEnvConfig,
        max_length: int,
        pad_token_id: int = 0,
    ) -> None:
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.config = config
        self.openhands_config = openhands_config
        self.e2b_allow_out = e2b_allow_out
        self.test_file_path = self._make_unique_test_file_path(conv)
        self.test_file_rel = PurePosixPath(self._display_test_file_path())
        self.runner: E2BRunner | None = None
        self.stdout_reader_task: asyncio.Task[None] | None = None
        self.stdout_lines: list[str] = []
        self.json_events: list[dict[str, object]] = []
        self._solution_text = ""
        self._runtime_terminated = False
        conv.extra["openhands_test_file"] = self._display_test_file_path()
        super().__init__(
            conv=conv,
            label=label,
            registry=registry,
            adapter=adapter,
            score_provider=score_provider,
            metric_calculator=metric_calculator,
            initial_request_timeout_seconds=config.initial_request_timeout_seconds,
            request_timeout_seconds=config.request_timeout_seconds,
            max_model_calls=config.max_model_calls,
            max_length=max_length,
            pad_token_id=pad_token_id,
            runtime_name="OpenHands",
        )

    @override
    async def launch_runtime(self) -> None:
        self.runner = None
        self.stdout_reader_task = None
        self.stdout_lines = []
        self.json_events = []
        self._runtime_terminated = False
        runtime_cwd = Path(self.openhands_config.e2b.workdir)
        task = build_openhands_leetcode_prompt(self.original_conv, test_file_rel=self.test_file_rel)
        script = get_openhand_launch_script(
            self.openhands_config,
            task=task,
            llm_base_url=self.llm_base_url,
            llm_model=self.llm_model,
        )
        self.runner = E2BRunner(
            name=f"openhands-{self.session_id}",
            config=self.openhands_config.e2b,
            allow_out=self.e2b_allow_out,
            setup_dirs=(str(self.test_file_path.parent),),
        )
        await self.runner.start(script, runtime_cwd)
        stdout = self.runner.stdout
        if stdout is not None:
            self.stdout_reader_task = asyncio.create_task(
                read_stdout(
                    stdout,
                    session_id=self.session_id,
                    stdout_lines=self.stdout_lines,
                    json_events=self.json_events,
                )
            )

    @override
    async def terminate_runtime(self) -> None:
        if not self._runtime_terminated:
            self._runtime_terminated = True
            await self._collect_solution_file()
            if self.runner is not None:
                await self.runner.terminate()
            if self.stdout_reader_task is not None:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.stdout_reader_task, timeout=_OPENHANDS_FINISH_DRAIN_SECONDS)
        self._attach_openhands_final_extra()

    @override
    async def calculate_trajectory_score(self) -> float:
        output = await self.score_provider.generate(VerifierInput(label=self.label, output_text=self._verifier_text()))
        return output.score

    def _verifier_text(self) -> str:
        return self._solution_text or self._model_output_text()

    def _attach_openhands_final_extra(self) -> None:
        target = self.trace.conversation if self.trace is not None else self.conv
        target.extra.update(
            {
                "openhands_test_file": self._display_test_file_path(),
                "openhands_solution": self._solution_text,
                "openhands_stdout_lines": list(self.stdout_lines),
                "openhands_json_events": list(self.json_events),
                "openhands_exit_code": self._runtime_exit_code_for_metric(),
            }
        )

    @override
    def _is_normal_finish_action(self, action: GenerationOutput) -> bool:
        return any(tool_call.name == "finish" for tool_call in action.tool_calls or [])

    @override
    def build_response_metric(self, base_metric: ResponseMetric, score: float) -> OpenHandsResponseMetric:
        blackbox_metric = super().build_response_metric(base_metric, score)
        metric = OpenHandsResponseMetric(
            **blackbox_metric.to_dict(),
            openhands_exit_code=self._runtime_exit_code_for_metric(),
            openhands_stdout_lines=len(self.stdout_lines),
            collected_solution_chars=len(self._solution_text),
        )
        metric.score = score
        return metric

    def _runtime_exit_code_for_metric(self) -> int | None:
        status = getattr(self, "_status", None)
        if status is not None and status.normal_finish:
            return 0
        if self.runner is None:
            return None
        return self.runner.returncode

    async def _collect_solution_file(self) -> None:
        if not self.config.collect_file_on_finish or self._solution_text or self.runner is None:
            return
        try:
            self._solution_text = await self.runner.read_text(str(self.test_file_path))
        except Exception as exc:
            logger.warning("Failed to read OpenHands solution file from E2B: %s: %s", self.test_file_path, exc)
            logger.debug("OpenHands E2B solution-file read traceback.", exc_info=True)

    def _make_unique_test_file_path(self, conv: Conversation) -> PurePosixPath:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        short_name = _slug(conv.conversation_id or "leetcode")[:32]
        rand = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        filename = f"{stamp}-{short_name}-{rand}.py"
        return _sandbox_dir(self.openhands_config.e2b.solution_root, self.config.test_file_dir) / filename

    def _display_test_file_path(self) -> str:
        try:
            return str(self.test_file_path.relative_to(PurePosixPath(self.openhands_config.e2b.workdir)))
        except ValueError:
            return str(self.test_file_path)


def build_openhands_leetcode_prompt(conv: Conversation, *, test_file_rel: PurePosixPath) -> str:
    problem = conv.extra.get("problem_description") or conv.messages[-1].content
    starter_code = conv.extra.get("starter_code", "")
    entry_point = conv.extra.get("entry_point", "")
    return f"""Solve this LeetCode-style Python task.

Use only Python-related actions:
- Use `file_editor` to create or edit the solution file.
- You may use `terminal` only to run `python {test_file_rel}` or inspect Python errors from that command.
- Do not run shell scripts, package installs, version-control commands, or broad bash exploration.
- Do not create any other Python files, including helper files in the repository root; put all helper code in the solution file.
- Before calling `finish`, run `python {test_file_rel}` and make sure the lightweight tests in `__main__` pass.
- Call the `finish` tool when done.

Create exactly one Python test/solution file at:
{test_file_rel}

The file should define the requested `Solution` class/method and include lightweight self-checks in an `if __name__ == "__main__":` block.

Entry point:
{entry_point}

Problem:
{problem}

Starter code:
```python
{starter_code}
```
"""


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "leetcode"


def _sandbox_dir(solution_root: str, test_file_dir: str) -> PurePosixPath:
    root = PurePosixPath(solution_root)
    directory = PurePosixPath(test_file_dir)
    if directory.is_absolute():
        return directory
    return root / directory
