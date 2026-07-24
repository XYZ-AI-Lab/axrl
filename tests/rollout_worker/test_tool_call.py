from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from rich.pretty import pprint
from transformers import AutoTokenizer

from axrl.configs import ModelConfig, RolloutWorkerConfig, SamplingConfig
from axrl.data import Conversation, GenerationInput, Message
from axrl.data.conversation import GenerationState
from axrl.processor.conversation_tokenizer import ConversationTokenizer
from axrl.ray import ray_utils
from axrl.ray.ray_rollout_worker import RayRolloutWorker
from axrl.ray.resource_group import Request, ResourceGroup

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for, in the format 'City, Country'.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get the temperature at a location on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for, in the format 'City, Country'.",
                    },
                    "date": {
                        "type": "string",
                        "description": "The date in 'YYYY-MM-DD' format.",
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]


@dataclass
class ModelSpec:
    name: str
    tool_call_parser: str


QWEN25 = ModelSpec(name="Qwen/Qwen2.5-1.5B-Instruct", tool_call_parser="qwen")
QWEN3 = ModelSpec(name="Qwen/Qwen3-1.7B", tool_call_parser="qwen")

ALL_MODELS = [
    pytest.param(QWEN25, id="qwen2.5-1.5b"),
    pytest.param(QWEN3, id="qwen3-1.7b"),
]


def make_config(model_spec: ModelSpec) -> RolloutWorkerConfig:
    return RolloutWorkerConfig(
        engine_type="sglang",
        model=ModelConfig(name=model_spec.name, seq_length=4096),
        gpu_memory_utilization=0.5,
        num_workers=1,
        tp_size=1,
        enable_metrics=False,
        sampling_config=SamplingConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_total_tokens=4096,
        ),
    )


def build_input(
    model_config: ModelConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    tool_call_parser: str = "qwen",
    tool_choice: Literal["auto", "required", "none"] | None = None,
) -> GenerationInput:
    conversation = Conversation(
        conversation_id="test_tool_call",
        messages=[Message.from_dict(msg) for msg in messages],
        gen_state=GenerationState(
            tools=tools,
            tool_choice=tool_choice,
            tool_call_parser=tool_call_parser,
        ),
    )
    tokenizer = ConversationTokenizer(model_config)
    return tokenizer.process(conversation)


def _make_ray_worker(config: RolloutWorkerConfig) -> RayRolloutWorker:
    # Run sglang Engine in a Ray actor process so each test gets a fresh
    # interpreter — avoids in-process state pollution across Engine inits.
    ray_utils.restart()
    resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=1)])
    return RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(config, resource_group))


async def _test_tool_call_auto(model_spec: ModelSpec) -> None:
    """Test tool call with tool_choice=auto (default). Model decides to call a tool."""
    config = make_config(model_spec)
    worker = _make_ray_worker(config)

    messages = [{"role": "user", "content": "What is the temperature in Paris right now?"}]
    gen_input = build_input(config.model, messages, TOOLS, tool_call_parser=model_spec.tool_call_parser)

    gen_output = await worker.generate(gen_input)
    pprint(gen_output)
    # Decode full input + output with special tokens to see the complete conversation
    tokenizer = AutoTokenizer.from_pretrained(str(config.model.get_full_path()))
    full_text = tokenizer.decode([*gen_input.input_ids.tolist(), *gen_output.output_ids.tolist()], skip_special_tokens=False)
    print(f"\n=== Full conversation (with special tokens) ===\n{full_text}")

    # The model should produce a tool call for get_current_temperature
    assert gen_output.tool_calls is not None, f"Expected tool calls but got none. Output: {gen_output.output_text}"
    assert len(gen_output.tool_calls) >= 1
    tc = gen_output.tool_calls[0]
    assert tc.name == "get_current_temperature", f"Expected get_current_temperature, got {tc.name}"
    args = json.loads(tc.arguments)
    assert "location" in args, f"Expected 'location' in arguments, got {args}"
    assert "Paris" in args["location"], f"Expected 'Paris' in location, got {args['location']}"
    assert gen_output.finish_reason == "tool_calls"
    print(f"PASS: test_tool_call_auto ({model_spec.name})")

    worker.shutdown()
    ray_utils.stop()


async def _test_tool_call_required(model_spec: ModelSpec) -> None:
    """Test tool call with tool_choice=required. Model must call a tool."""
    config = make_config(model_spec)
    worker = _make_ray_worker(config)

    messages = [{"role": "user", "content": "Temperature in Tokyo on 2025-01-01?"}]
    gen_input = build_input(
        config.model,
        messages,
        TOOLS,
        tool_call_parser=model_spec.tool_call_parser,
        tool_choice="required",
    )

    gen_output = await worker.generate(gen_input)
    pprint(gen_output)
    tokenizer = AutoTokenizer.from_pretrained(str(config.model.get_full_path()))
    full_text = tokenizer.decode([*gen_input.input_ids.tolist(), *gen_output.output_ids.tolist()], skip_special_tokens=False)
    print(f"\n=== Full conversation (with special tokens) ===\n{full_text}")

    assert gen_output.tool_calls is not None, f"Expected tool calls but got none. Output: {gen_output.output_text}"
    assert len(gen_output.tool_calls) >= 1
    tc = gen_output.tool_calls[0]
    assert tc.name in ("get_current_temperature", "get_temperature_date"), f"Unexpected tool: {tc.name}"
    args = json.loads(tc.arguments)
    assert "location" in args, f"Expected 'location' in arguments, got {args}"
    assert gen_output.finish_reason == "tool_calls"
    print(f"PASS: test_tool_call_required ({model_spec.name})")

    worker.shutdown()
    ray_utils.stop()


async def _test_parse_function_calls() -> None:
    """Test parse_function_calls with known text."""
    from axrl.worker.sglang_worker import SGLangWorker

    # Simulate Qwen-style output
    text = '<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Paris, France"}}\n</tool_call>'
    output_text, tool_calls, finish_reason = SGLangWorker.parse_function_calls(
        text,
        tools=TOOLS,
        tool_call_parser="qwen",
        finish_reason="stop",
    )
    assert tool_calls is not None
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.name == "get_current_temperature"
    args = json.loads(tc.arguments)
    assert args == {"location": "Paris, France"}
    assert finish_reason == "tool_calls"
    assert output_text == ""
    print("PASS: test_parse_function_calls_deterministic")

    # Test with no tool call in text
    text2 = "The weather is nice today."
    output_text2, tool_calls2, finish_reason2 = SGLangWorker.parse_function_calls(
        text2,
        tools=TOOLS,
        tool_call_parser="qwen",
        finish_reason="stop",
    )
    assert tool_calls2 is None
    assert output_text2 == text2
    assert finish_reason2 == "stop"
    print("PASS: test_parse_no_tool_call")

    # Test required mode with raw JSON
    json_text = '[{"name": "get_current_temperature", "parameters": {"location": "London, UK"}}]'
    output_text3, tool_calls3, finish_reason3 = SGLangWorker.parse_function_calls(
        json_text,
        tools=TOOLS,
        tool_choice="required",
        finish_reason="stop",
    )
    assert tool_calls3 is not None
    assert len(tool_calls3) == 1
    assert tool_calls3[0].name == "get_current_temperature"
    assert json.loads(tool_calls3[0].arguments) == {"location": "London, UK"}
    assert finish_reason3 == "tool_calls"
    assert output_text3 == ""
    print("PASS: test_parse_required_mode")


# ── pytest entry points ─────────────────────────────────────────


def test_parse_function_calls_deterministic() -> None:
    """Deterministic parse tests — no GPU needed."""
    asyncio.run(_test_parse_function_calls())


@pytest.mark.parametrize("model_spec", ALL_MODELS)
def test_tool_call_auto(model_spec: ModelSpec) -> None:
    asyncio.run(_test_tool_call_auto(model_spec))


@pytest.mark.parametrize("model_spec", ALL_MODELS)
def test_tool_call_required(model_spec: ModelSpec) -> None:
    asyncio.run(_test_tool_call_required(model_spec))


# ── direct invocation ───────────────────────────────────────────


def main() -> None:
    # Deterministic tests (no GPU)
    asyncio.run(_test_parse_function_calls())

    # GPU tests for both models
    for spec in (QWEN25, QWEN3):
        asyncio.run(_test_tool_call_auto(spec))
        asyncio.run(_test_tool_call_required(spec))


if __name__ == "__main__":
    main()
