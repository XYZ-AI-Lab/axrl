from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest
import ray
import torch

from axrl.agent.rollout_agent import RolloutAgent
from axrl.configs import ModelConfig, SamplingConfig
from axrl.data import Conversation, GenerationInput, GenerationOutput, Message, ToolCall, array_utils
from axrl.openai_proxy import (
    OpenAIChatAdapter,
    OpenAIChatAdapterConfig,
    OpenAIChatBuildResponseRequest,
    OpenAIChatConvertedRequest,
    OpenAIChatConvertRequest,
    OpenAIChatResponseResult,
    OpenAIProxyServer,
    OpenAIProxySessionRegistry,
)
from axrl.processor.processor_pool import ProcessorPool

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.types import Message as ASGIMessage
    from starlette.types import Scope

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


class TinyChatTokenizer:
    bos_token_id = 0
    chat_template = "{{ messages }}"
    model_max_length = 4096

    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(ch) + 1 for ch in text]

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return "".join(chr(int(token_id) - 1) for token_id in token_ids if int(token_id) > 0)

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        return_tensors: str | None = None,
        return_dict: bool = False,
        **_: Any,
    ) -> str | list[int] | dict[str, Any] | torch.Tensor:
        chunks: list[str] = []
        if tools:
            tool_names = ", ".join(tool["function"]["name"] for tool in tools)
            chunks.append(f"<tools>{tool_names}</tools>")
        for message in conversation:
            chunks.append(f"<{message['role']}>{message.get('content') or ''}</{message['role']}>")
        if add_generation_prompt:
            chunks.append("<assistant>")
        rendered = "\n".join(chunks)
        if return_dict:
            return {"input_ids": self.encode(rendered)}
        if return_tensors == "pt":
            return torch.tensor([self.encode(rendered)], dtype=torch.long)
        if tokenize:
            return self.encode(rendered)
        return rendered


class TinyOpenAIChatAdapter(OpenAIChatAdapter):
    def _load_tokenizer(self) -> TinyChatTokenizer:
        return TinyChatTokenizer()


class QwenLikeTinyChatTokenizer(TinyChatTokenizer):
    chat_template = "{% if not enable_thinking is defined %}{% set enable_thinking = true %}{% endif %}<think>{{ messages }}</think><tool_call>"


class QwenToolOnlyTinyChatTokenizer(TinyChatTokenizer):
    chat_template = '{%- if tools %}<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>{%- endif %}'


class DeepSeekThinkTagTinyChatTokenizer(TinyChatTokenizer):
    chat_template = "{{ messages }}</think>"


class QwenLikeOpenAIChatAdapter(OpenAIChatAdapter):
    def _load_tokenizer(self) -> TinyChatTokenizer:
        return QwenLikeTinyChatTokenizer()


class QwenToolOnlyOpenAIChatAdapter(OpenAIChatAdapter):
    def _load_tokenizer(self) -> TinyChatTokenizer:
        return QwenToolOnlyTinyChatTokenizer()


class DeepSeekThinkTagOpenAIChatAdapter(OpenAIChatAdapter):
    def _load_tokenizer(self) -> TinyChatTokenizer:
        return DeepSeekThinkTagTinyChatTokenizer()


def _tiny_adapter() -> OpenAIChatAdapter:
    return TinyOpenAIChatAdapter(OpenAIChatAdapterConfig(model=ModelConfig(name="tiny", seq_length=4096)))


def _tiny_adapter_pool(*, tool_call_parser: str | None = None) -> ProcessorPool[OpenAIChatAdapterInput, OpenAIChatAdapterOutput]:
    return ProcessorPool(
        TinyOpenAIChatAdapter,
        config=OpenAIChatAdapterConfig(model=ModelConfig(name="tiny", seq_length=4096), tool_call_parser=tool_call_parser),
        num_processors=1,
        timeout_seconds=60.0,
    )


def _qwen_like_adapter() -> OpenAIChatAdapter:
    return QwenLikeOpenAIChatAdapter(
        OpenAIChatAdapterConfig(
            model=ModelConfig(name="tiny", seq_length=4096),
            tool_call_parser="auto",
            reasoning_parser="auto",
        )
    )


def _qwen_tool_only_adapter(*, tool_call_parser: str | None) -> OpenAIChatAdapter:
    return QwenToolOnlyOpenAIChatAdapter(
        OpenAIChatAdapterConfig(
            model=ModelConfig(name="tiny", seq_length=4096),
            tool_call_parser=tool_call_parser,
        )
    )


def _deepseek_think_tag_adapter() -> OpenAIChatAdapter:
    return DeepSeekThinkTagOpenAIChatAdapter(
        OpenAIChatAdapterConfig(
            model=ModelConfig(name="tiny", seq_length=4096),
            reasoning_parser="auto",
        )
    )


def _generation_output(
    *,
    session_id: str = "session",
    output_text: str = "done",
    tool_calls: list[ToolCall] | None = None,
) -> GenerationOutput:
    return GenerationOutput(
        session_id=session_id,
        output_ids=array_utils.as_i32([101, 102]),
        output_logprobs=array_utils.as_f32([-0.1, -0.2]),
        output_text=output_text,
        output_text_with_special_tokens=output_text,
        cached_tokens=0,
        finish_reason="stop",
        e2e_elapsed_seconds=0.01,
        stop_reason=None,
        retry=0,
        tool_calls=tool_calls,
    )


def test_openai_chat_adapter_ignores_openai_sampling_and_builds_response() -> None:
    adapter = _tiny_adapter()
    converted = adapter.process(
        OpenAIChatConvertRequest(
            session_id="session",
            request_json={
                "model": "tiny",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "text", "text": "world"},
                        ],
                    }
                ],
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "stream": False,
            },
        )
    )

    assert isinstance(converted, OpenAIChatConvertedRequest)
    assert converted.generation_input.sampling_config is None
    assert converted.generation_input.session_id == "session"
    assert len(converted.messages) == 1
    assert converted.messages[0].role == "user"
    content = converted.messages[0].content
    assert not isinstance(content, str)
    assert [part.text for part in content] == ["hello", "world"]
    assert len(converted.generation_input.input_ids) > 0
    decoded = TinyChatTokenizer().decode(converted.generation_input.input_ids.tolist())
    assert "hello world" in decoded

    response = adapter.process(
        OpenAIChatBuildResponseRequest(
            context=converted.context,
            generation_output=_generation_output(session_id="session"),
        )
    )
    assert isinstance(response, OpenAIChatResponseResult)
    assert response.response_json["object"] == "chat.completion"
    assert response.response_json["choices"][0]["message"]["content"] == "done"
    assert response.response_json["usage"]["completion_tokens"] == 2


def test_openai_chat_adapter_rejects_streaming_until_rollout_supports_it() -> None:
    adapter = _tiny_adapter()
    with pytest.raises(ValueError, match="stream=true"):
        adapter.process(
            OpenAIChatConvertRequest(
                session_id="session",
                request_json={
                    "model": "tiny",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        )


def test_openai_chat_adapter_preserves_parsed_tool_calls_in_sglang_response() -> None:
    adapter = _tiny_adapter()
    converted = adapter.process(
        OpenAIChatConvertRequest(
            session_id="session",
            request_json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Finish the task.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )
    )
    assert isinstance(converted, OpenAIChatConvertedRequest)

    response = adapter.process(
        OpenAIChatBuildResponseRequest(
            context=converted.context,
            generation_output=_generation_output(
                session_id="session",
                output_text="",
                tool_calls=[ToolCall(id="call_test", index=0, name="finish", arguments='{"ok": true}')],
            ),
        )
    )

    assert isinstance(response, OpenAIChatResponseResult)
    choice = response.response_json["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["id"] == "call_test"
    assert choice["message"]["tool_calls"][0]["index"] == 0
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "finish"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"ok": true}'


def test_openai_chat_adapter_uses_sglang_template_parser_detection_and_reasoning_parser() -> None:
    adapter = _qwen_like_adapter()
    converted = adapter.process(
        OpenAIChatConvertRequest(
            session_id="session",
            request_json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Finish the task.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )
    )
    assert isinstance(converted, OpenAIChatConvertedRequest)
    assert converted.generation_input.tool_call_parser == "qwen"
    assert converted.context.resolved_tool_call_parser == "qwen"
    assert converted.context.resolved_reasoning_parser == "qwen3"

    response = adapter.process(
        OpenAIChatBuildResponseRequest(
            context=converted.context,
            generation_output=_generation_output(session_id="session", output_text="<think>plan</think>done"),
        )
    )

    assert isinstance(response, OpenAIChatResponseResult)
    message = response.response_json["choices"][0]["message"]
    assert message["reasoning_content"] == "plan"
    assert message["content"] == "done"


def test_openai_chat_adapter_uses_sglang_reasoning_request_policy() -> None:
    adapter = _deepseek_think_tag_adapter()
    converted = adapter.process(
        OpenAIChatConvertRequest(
            session_id="session",
            request_json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert isinstance(converted, OpenAIChatConvertedRequest)
    assert converted.context.resolved_reasoning_parser == "deepseek-r1"
    assert not converted.context.force_reasoning

    response = adapter.process(
        OpenAIChatBuildResponseRequest(
            context=converted.context,
            generation_output=_generation_output(session_id="session", output_text="plan</think>done"),
        )
    )

    assert isinstance(response, OpenAIChatResponseResult)
    message = response.response_json["choices"][0]["message"]
    assert message["reasoning_content"] == "plan"
    assert message["content"] == "done"


def test_openai_chat_adapter_uses_explicit_qwen_tool_parser() -> None:
    adapter = _qwen_tool_only_adapter(tool_call_parser="qwen")
    converted = adapter.process(
        OpenAIChatConvertRequest(
            session_id="session",
            request_json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Finish the task.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )
    )

    assert isinstance(converted, OpenAIChatConvertedRequest)
    assert converted.context.resolved_reasoning_parser is None
    assert converted.context.resolved_tool_call_parser == "qwen"
    assert converted.generation_input.tool_call_parser == "qwen"


def test_openai_chat_adapter_auto_does_not_guess_from_tool_markers() -> None:
    adapter = _qwen_tool_only_adapter(tool_call_parser="auto")
    converted = adapter.process(
        OpenAIChatConvertRequest(
            session_id="session",
            request_json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Finish the task.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )
    )

    assert isinstance(converted, OpenAIChatConvertedRequest)
    assert converted.context.resolved_reasoning_parser is None
    assert converted.context.resolved_tool_call_parser is None
    assert converted.generation_input.tool_call_parser is None


def test_openai_chat_adapter_renders_tools_with_tokenizer() -> None:
    adapter = _tiny_adapter()
    converted = adapter.process(
        OpenAIChatConvertRequest(
            session_id="session",
            request_json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Finish the task.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
    )

    assert isinstance(converted, OpenAIChatConvertedRequest)
    decoded = TinyChatTokenizer().decode(converted.generation_input.input_ids.tolist())
    assert "<tools>finish</tools>" in decoded
    assert converted.generation_input.tool_choice is None
    assert converted.generation_input.tool_call_parser is None
    assert converted.generation_input.tools is not None
    assert converted.generation_input.tools[0]["function"]["name"] == "finish"


def test_openai_proxy_registry_waits_for_env_response() -> None:
    asyncio.run(_test_openai_proxy_registry_waits_for_env_response())


def test_openai_proxy_server_accepts_scoped_chat_request() -> None:
    asyncio.run(_test_openai_proxy_server_accepts_scoped_chat_request())


def test_openai_proxy_server_requires_bearer_token_when_configured() -> None:
    asyncio.run(_test_openai_proxy_server_requires_bearer_token_when_configured())


def test_openai_proxy_server_handles_client_disconnect_while_reading_body() -> None:
    asyncio.run(_test_openai_proxy_server_handles_client_disconnect_while_reading_body())


async def _test_openai_proxy_registry_waits_for_env_response() -> None:
    registry = OpenAIProxySessionRegistry(request_timeout_seconds=5.0)
    try:
        await registry.create_session("session")
        submit_task = asyncio.create_task(
            registry.submit_chat_completion(
                session_id="session",
                body={"model": "tiny", "messages": []},
                headers={"x-test": "1"},
            )
        )

        pending = await registry.wait_for_request("session", timeout_seconds=1.0)
        assert pending.body["model"] == "tiny"
        assert pending.headers["x-test"] == "1"
        await pending.respond({"id": "chatcmpl-test"})

        response = await submit_task
        assert response.status_code == 200
        assert response.body == {"id": "chatcmpl-test"}
    finally:
        registry.shutdown()


def test_remote_openai_proxy_registry_can_handoff_across_wrappers() -> None:
    asyncio.run(_test_remote_openai_proxy_registry_can_handoff_across_wrappers())


async def _test_remote_openai_proxy_registry_can_handoff_across_wrappers() -> None:
    submit_registry = OpenAIProxySessionRegistry(request_timeout_seconds=5.0)
    try:
        env_registry = OpenAIProxySessionRegistry.from_remote_actor(submit_registry.get_actor_handle())
        await env_registry.create_session("remote-session")

        submit_task = asyncio.create_task(
            submit_registry.submit_chat_completion(
                session_id="remote-session",
                body={"model": "tiny", "messages": []},
                headers={"x-test": "remote"},
            )
        )
        pending = await env_registry.wait_for_request("remote-session", timeout_seconds=5.0)

        assert pending.body["model"] == "tiny"
        assert pending.headers["x-test"] == "remote"
        await pending.respond({"id": "chatcmpl-remote"})

        response = await submit_task
        assert response.status_code == 200
        assert response.body == {"id": "chatcmpl-remote"}
    finally:
        submit_registry.shutdown()


async def _test_openai_proxy_server_accepts_scoped_chat_request() -> None:
    session_id = "session:with:colon"
    registry = OpenAIProxySessionRegistry(request_timeout_seconds=5.0)
    try:
        await registry.create_session(session_id)
        server = OpenAIProxyServer(host="127.0.0.1", port=0, registry=registry, request_timeout_seconds=5.0)
        await server.start()
        try:
            post_task = asyncio.create_task(
                asyncio.to_thread(
                    _post_json,
                    f"{server.session_base_url(session_id)}/chat/completions",
                    {"model": "tiny", "messages": [{"role": "user", "content": "hello"}]},
                )
            )
            pending = await registry.wait_for_request(session_id, timeout_seconds=1.0)
            assert pending.body["model"] == "tiny"
            await pending.respond({"id": "chatcmpl-route"})

            status_code, response = await post_task
            assert status_code == 200
            assert response == {"id": "chatcmpl-route"}
        finally:
            await server.stop()
    finally:
        registry.shutdown()


async def _test_openai_proxy_server_requires_bearer_token_when_configured() -> None:
    session_id = "auth-session"
    registry = OpenAIProxySessionRegistry(request_timeout_seconds=5.0)
    try:
        await registry.create_session(session_id)
        server = OpenAIProxyServer(host="127.0.0.1", port=0, registry=registry, request_timeout_seconds=5.0, auth_token="secret")
        await server.start()
        try:
            url = f"{server.session_base_url(session_id)}/chat/completions"
            status_code, response = await asyncio.to_thread(
                _post_json,
                url,
                {"model": "tiny", "messages": [{"role": "user", "content": "hello"}]},
            )
            assert status_code == 401
            assert response["error"]["type"] == "unauthorized"

            post_task = asyncio.create_task(
                asyncio.to_thread(
                    _post_json,
                    url,
                    {"model": "tiny", "messages": [{"role": "user", "content": "hello"}]},
                    {"authorization": "Bearer secret"},
                )
            )
            pending = await registry.wait_for_request(session_id, timeout_seconds=1.0)
            assert pending.body["model"] == "tiny"
            await pending.respond({"id": "chatcmpl-auth"})

            status_code, response = await post_task
            assert status_code == 200
            assert response == {"id": "chatcmpl-auth"}
        finally:
            await server.stop()
    finally:
        registry.shutdown()


async def _test_openai_proxy_server_handles_client_disconnect_while_reading_body() -> None:
    registry = OpenAIProxySessionRegistry(request_timeout_seconds=5.0)
    try:
        server = OpenAIProxyServer(host="127.0.0.1", port=0, registry=registry, request_timeout_seconds=5.0)
        app = server._build_app()
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/sessions/disconnect/v1/chat/completions",
            "raw_path": b"/sessions/disconnect/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 0),
        }
        receive_messages = iter(
            [
                {"type": "http.request", "body": b'{"model":', "more_body": True},
                {"type": "http.disconnect"},
            ]
        )
        sent_messages: list[ASGIMessage] = []

        async def receive() -> ASGIMessage:
            return next(receive_messages)

        async def send(message: ASGIMessage) -> None:
            sent_messages.append(message)

        await app(scope, receive, send)

        start = next(message for message in sent_messages if message["type"] == "http.response.start")
        assert start["status"] == 499
    finally:
        registry.shutdown()


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8")
    request_headers = {"content-type": "application/json", **(headers or {})}
    request = urllib.request.Request(  # noqa: S310 - local test server.
        url,
        data=data,
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:  # noqa: S310 - local test server.
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as response:
        return response.code, json.loads(response.read().decode("utf-8"))


def test_conversation_serializes_generation_sampling_config() -> None:
    conv = Conversation(
        conversation_id="conv",
        messages=[Message(role="user", content="hello")],
    )
    conv.gen_state.session_id = "session"
    conv.gen_state.sampling_config = SamplingConfig(temperature=0.3, top_p=0.8, max_total_tokens=256)

    restored = Conversation.from_dict(conv.to_dict())

    assert restored.gen_state.sampling_config is not None
    assert restored.gen_state.sampling_config.temperature == 0.3
    assert restored.gen_state.sampling_config.top_p == 0.8
    assert restored.gen_state.sampling_config.max_total_tokens == 256


def test_rollout_agent_uses_conversation_sampling_override() -> None:
    asyncio.run(_test_rollout_agent_uses_conversation_sampling_override())


async def _test_rollout_agent_uses_conversation_sampling_override() -> None:
    class FakeRolloutWorker:
        def __init__(self) -> None:
            self.request: GenerationInput | None = None

        async def generate(self, req: GenerationInput) -> GenerationOutput:
            self.request = req
            return _generation_output(session_id=req.session_id)

    worker = FakeRolloutWorker()
    agent = RolloutAgent(worker)  # type: ignore[arg-type]
    conv = Conversation(
        conversation_id="conv",
        messages=[Message(role="user", content="hello")],
    )
    conv.gen_state.session_id = "session"
    conv.gen_state.input_ids = array_utils.as_i32([1, 2, 3])
    conv.gen_state.sampling_config = SamplingConfig(temperature=0.2, top_p=0.7, max_total_tokens=128)

    await agent.act(conv, SamplingConfig(temperature=1.0, top_p=1.0, max_total_tokens=4096))

    assert worker.request is not None
    assert worker.request.sampling_config is not None
    assert worker.request.sampling_config.temperature == 0.2
    assert worker.request.sampling_config.top_p == 0.7
    assert worker.request.sampling_config.max_total_tokens == 128
