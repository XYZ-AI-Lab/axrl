from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, override

import numpy as np
import pytest
import ray

from axis_recipe.blackbox_rl.blackbox_env import BlackBoxEnv
from axrl.data import Conversation, GenerationInput, GenerationOutput, Message, ToolCall, array_utils
from axrl.metrics.response_metric import ResponseMetric, ResponseMetricCalculator
from axrl.openai_proxy import (
    OpenAIChatBuildResponseRequest,
    OpenAIChatConvertedRequest,
    OpenAIChatConvertRequest,
    OpenAIChatResponseContext,
    OpenAIChatResponseResult,
    OpenAIProxySessionRegistry,
)
from axrl.verifier.base_verifier import VerifierInput, VerifierOutput

if TYPE_CHECKING:
    from collections.abc import Iterator

    from axrl.openai_proxy.chat_adapter import OpenAIChatAdapterInput, OpenAIChatAdapterOutput


@pytest.fixture(scope="module", autouse=True)
def _ray_for_portable_proxy_registry() -> Iterator[None]:
    started_here = False
    if not ray.is_initialized():
        ray.init(num_cpus=2, include_dashboard=False)
        started_here = True
    yield
    if started_here and ray.is_initialized():
        ray.shutdown()


class _FakeAdapter:
    def __init__(self, *, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens

    async def generate(self, req: OpenAIChatAdapterInput) -> OpenAIChatAdapterOutput:
        if isinstance(req, OpenAIChatBuildResponseRequest):
            return OpenAIChatResponseResult(response_json={"id": "chatcmpl-test", "choices": []})
        assert isinstance(req, OpenAIChatConvertRequest)
        input_ids = np.arange(self.prompt_tokens, dtype=np.int32)
        return OpenAIChatConvertedRequest(
            session_id=req.session_id,
            generation_input=GenerationInput(session_id=req.session_id, input_ids=input_ids),
            messages=[Message(role="user", content="hello")],
            context=OpenAIChatResponseContext(
                request_json=req.request_json,
                prompt_tokens=len(input_ids),
                response_id="chatcmpl-test",
            ),
        )


class _FakeMetricCalculator:
    def __init__(self) -> None:
        self.calculator = ResponseMetricCalculator()

    async def generate(self, req: GenerationOutput) -> ResponseMetric:
        return self.calculator.process(req)


class _FakeScoreProvider:
    def __init__(self, *, score: float = 0.0) -> None:
        self.score = score
        self.calls = 0

    async def generate(self, _req: VerifierInput) -> VerifierOutput:
        self.calls += 1
        return VerifierOutput(score=self.score)


async def _warm_registry(registry: OpenAIProxySessionRegistry) -> None:
    await registry.create_session("warmup")
    await registry.close_session("warmup")


class _FakeBlackBoxEnv(BlackBoxEnv):
    def __init__(
        self,
        *,
        registry: OpenAIProxySessionRegistry,
        adapter: _FakeAdapter,
        max_length: int,
        max_model_calls: int = 4,
        initial_request_timeout_seconds: float = 1.0,
        score_provider: _FakeScoreProvider | None = None,
    ) -> None:
        conv = Conversation(messages=[Message(role="user", content="seed")], conversation_id="session")
        conv.gen_state.session_id = "session"
        self.terminated = False
        self.terminate_count = 0
        self.launch_count = 0
        self.first_launch_started = asyncio.Event()
        self.second_launch_started = asyncio.Event()
        self.runtime_exit = asyncio.Event()
        self.fake_score_provider = score_provider or _FakeScoreProvider()
        super().__init__(
            conv=conv,
            label="answer",
            registry=registry,
            adapter=adapter,  # type: ignore[arg-type]
            score_provider=self.fake_score_provider,  # type: ignore[arg-type]
            metric_calculator=_FakeMetricCalculator(),  # type: ignore[arg-type]
            initial_request_timeout_seconds=initial_request_timeout_seconds,
            request_timeout_seconds=1.0,
            max_model_calls=max_model_calls,
            max_length=max_length,
            runtime_name="FakeRuntime",
        )

    @override
    async def launch_runtime(self) -> None:
        self.launch_count += 1
        if self.launch_count == 1:
            self.first_launch_started.set()
        if self.launch_count >= 2:
            self.second_launch_started.set()
        return None

    @override
    async def terminate_runtime(self) -> None:
        self.terminated = True
        self.terminate_count += 1
        self.runtime_exit.set()

    @override
    async def calculate_trajectory_score(self) -> float:
        output = await self.fake_score_provider.generate(VerifierInput(label="answer", output_text=""))
        return output.score

    @override
    def _is_normal_finish_action(self, action: GenerationOutput) -> bool:
        return any(tool_call.name == "finish" for tool_call in action.tool_calls or [])


def test_prompt_too_long_closes_session_and_finishes_with_forced_zero_score() -> None:
    asyncio.run(_test_prompt_too_long_closes_session_and_finishes_with_forced_zero_score())


def test_invalid_tool_call_at_max_model_calls_fails_pending_request_and_verifies() -> None:
    asyncio.run(_test_invalid_tool_call_at_max_model_calls_fails_pending_request_and_verifies())


def test_finish_tool_call_does_not_set_request_timeout() -> None:
    asyncio.run(_test_finish_tool_call_does_not_set_request_timeout())


def test_initial_request_timeout_retry_does_not_set_request_timeout() -> None:
    asyncio.run(_test_initial_request_timeout_retry_does_not_set_request_timeout())


async def _test_prompt_too_long_closes_session_and_finishes_with_forced_zero_score() -> None:
    registry = OpenAIProxySessionRegistry(request_timeout_seconds=5.0)
    try:
        await asyncio.wait_for(_warm_registry(registry), timeout=60.0)
        adapter = _FakeAdapter(prompt_tokens=3)
        env = _FakeBlackBoxEnv(registry=registry, adapter=adapter, max_length=4)

        start_task = asyncio.create_task(env.start())
        await asyncio.wait_for(env.first_launch_started.wait(), timeout=10.0)
        first_response_task = asyncio.create_task(
            registry.submit_chat_completion(
                session_id="session",
                body={"messages": [{"role": "user", "content": "hello"}]},
                headers={},
            )
        )
        observation = await start_task
        assert not first_response_task.done()
        assert not env.done
        assert observation.conversation_id == "session"

        adapter.prompt_tokens = 5
        second_response_task = asyncio.create_task(
            registry.submit_chat_completion(
                session_id="session",
                body={"messages": [{"role": "user", "content": "too long"}]},
                headers={},
            )
        )
        final_observation, score, done, trace, metric = await env.step(
            GenerationOutput(
                session_id="session",
                output_ids=array_utils.as_i32([10]),
                output_logprobs=array_utils.as_f32([-0.1]),
                output_text="continue",
                output_text_with_special_tokens="continue",
                cached_tokens=0,
                finish_reason="stop",
                e2e_elapsed_seconds=0.01,
                stop_reason=None,
                retry=0,
            )
        )
        first_response = await first_response_task
        second_response = await second_response_task

        assert first_response.status_code == 200
        assert second_response.status_code == 413
        assert second_response.body["error"]["type"] == "openai_proxy_error"
        assert "prompt exceeded max_length" in second_response.body["error"]["message"]
        assert env.done
        assert env.terminated
        assert done
        assert score == 0.0
        assert trace is not None
        assert metric is not None
        assert metric.score == 0.0
        assert env.fake_score_provider.calls == 0
        assert final_observation.extra["blackbox_forced_score_reason"] == second_response.body["error"]["message"]

        closed_response = await registry.submit_chat_completion(
            session_id="session",
            body={"messages": [{"role": "user", "content": "retry"}]},
            headers={},
        )
        assert closed_response.status_code == 404
    finally:
        registry.shutdown()


async def _test_invalid_tool_call_at_max_model_calls_fails_pending_request_and_verifies() -> None:
    registry = OpenAIProxySessionRegistry(request_timeout_seconds=5.0)
    try:
        await asyncio.wait_for(_warm_registry(registry), timeout=60.0)
        score_provider = _FakeScoreProvider(score=0.25)
        env = _FakeBlackBoxEnv(
            registry=registry,
            adapter=_FakeAdapter(prompt_tokens=3),
            max_length=8,
            max_model_calls=1,
            score_provider=score_provider,
        )

        start_task = asyncio.create_task(env.start())
        await asyncio.wait_for(env.first_launch_started.wait(), timeout=10.0)
        response_task = asyncio.create_task(
            registry.submit_chat_completion(
                session_id="session",
                body={"messages": [{"role": "user", "content": "hello"}]},
                headers={},
            )
        )
        observation = await start_task
        assert not response_task.done()

        observation, score, done, trace, metric = await env.step(
            GenerationOutput(
                session_id="session",
                output_ids=array_utils.as_i32([10]),
                output_logprobs=array_utils.as_f32([-0.1]),
                output_text="",
                output_text_with_special_tokens="",
                cached_tokens=0,
                finish_reason="tool_calls",
                e2e_elapsed_seconds=0.01,
                stop_reason=None,
                retry=0,
                tool_calls=[ToolCall(id="call_1", index=0, name="finish", arguments="{bad-json")],
            )
        )
        response = await response_task

        assert response.status_code == 400
        assert "max_model_calls=1" in response.body["error"]["message"]
        assert done
        assert score == 0.25
        assert trace is not None
        assert metric is not None
        assert metric.score == 0.25
        assert env.terminated
        assert score_provider.calls == 1
        assert observation.conversation_id == "session"
    finally:
        registry.shutdown()


async def _test_finish_tool_call_does_not_set_request_timeout() -> None:
    registry = OpenAIProxySessionRegistry(request_timeout_seconds=5.0)
    try:
        await asyncio.wait_for(_warm_registry(registry), timeout=60.0)
        score_provider = _FakeScoreProvider(score=1.0)
        env = _FakeBlackBoxEnv(
            registry=registry,
            adapter=_FakeAdapter(prompt_tokens=3),
            max_length=8,
            max_model_calls=4,
            score_provider=score_provider,
        )

        start_task = asyncio.create_task(env.start())
        await asyncio.wait_for(env.first_launch_started.wait(), timeout=10.0)
        response_task = asyncio.create_task(
            registry.submit_chat_completion(
                session_id="session",
                body={"messages": [{"role": "user", "content": "hello"}]},
                headers={},
            )
        )
        await start_task

        observation, score, done, trace, metric = await env.step(
            GenerationOutput(
                session_id="session",
                output_ids=array_utils.as_i32([10]),
                output_logprobs=array_utils.as_f32([-0.1]),
                output_text="",
                output_text_with_special_tokens="",
                cached_tokens=0,
                finish_reason="tool_calls",
                e2e_elapsed_seconds=0.01,
                stop_reason=None,
                retry=0,
                tool_calls=[ToolCall(id="call_1", index=0, name="finish", arguments="{}")],
            )
        )
        response = await response_task

        assert response.status_code == 200
        assert done
        assert score == 1.0
        assert trace is not None
        assert metric is not None
        assert metric.score == 1.0
        assert metric.normal_finish == 1
        assert metric.request_timeout == 0
        assert observation.extra["blackbox_normal_finish"]
        assert observation.extra["blackbox_normal_finish_reason"] == "FakeRuntime received finish action for session session."
        assert score_provider.calls == 1
    finally:
        registry.shutdown()


async def _test_initial_request_timeout_retry_does_not_set_request_timeout() -> None:
    registry = OpenAIProxySessionRegistry(request_timeout_seconds=5.0)
    try:
        await asyncio.wait_for(_warm_registry(registry), timeout=60.0)
        score_provider = _FakeScoreProvider(score=1.0)
        env = _FakeBlackBoxEnv(
            registry=registry,
            adapter=_FakeAdapter(prompt_tokens=3),
            max_length=8,
            initial_request_timeout_seconds=0.2,
            score_provider=score_provider,
        )

        start_task = asyncio.create_task(env.start())
        await asyncio.wait_for(env.second_launch_started.wait(), timeout=10.0)
        assert env.terminate_count >= 1
        response_task = asyncio.create_task(
            registry.submit_chat_completion(
                session_id="session",
                body={"messages": [{"role": "user", "content": "hello after retry"}]},
                headers={},
            )
        )
        await start_task

        observation, score, done, trace, metric = await env.step(
            GenerationOutput(
                session_id="session",
                output_ids=array_utils.as_i32([10]),
                output_logprobs=array_utils.as_f32([-0.1]),
                output_text="",
                output_text_with_special_tokens="",
                cached_tokens=0,
                finish_reason="tool_calls",
                e2e_elapsed_seconds=0.01,
                stop_reason=None,
                retry=0,
                tool_calls=[ToolCall(id="call_1", index=0, name="finish", arguments="{}")],
            )
        )
        response = await response_task

        assert response.status_code == 200
        assert done
        assert score == 1.0
        assert trace is not None
        assert metric is not None
        assert metric.score == 1.0
        assert metric.initial_request_timeout == 0
        assert metric.request_timeout == 0
        assert metric.normal_finish == 1
        assert observation.extra["blackbox_normal_finish"]
    finally:
        registry.shutdown()
