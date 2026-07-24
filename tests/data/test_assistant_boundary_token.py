import logging
from typing import Any

import pytest
from transformers import AutoProcessor, AutoTokenizer

from axrl.configs import ModelConfig
from axrl.data import array_utils
from axrl.data.conversation import Conversation, GenerationState, Message
from axrl.data.generation import TensorHandle
from axrl.data.rollout_trace import RolloutTrace
from axrl.processor.appended_message_tokenizer import AppendedMessageTokenizer
from axrl.processor.chat_template_utils import get_single_token_assistant_boundary_id

_MODEL_CONFIGS: list[ModelConfig] = [
    ModelConfig(name="Qwen/Qwen3-0.6B-Base"),
    ModelConfig(name="Qwen/Qwen3-0.6B"),
    ModelConfig(name="Qwen/Qwen3-1.7B"),
    ModelConfig(name="Qwen/Qwen3-8B"),
    ModelConfig(name="Qwen/Qwen3-4B-Instruct-2507"),
    ModelConfig(name="Qwen/Qwen2.5-1.5B-Instruct"),
    ModelConfig(name="Qwen/Qwen2.5-3B-Instruct"),
    ModelConfig(name="Qwen/Qwen2.5-VL-7B-Instruct"),
    # DeepSeek strips the generation-time think prefix from assistant history before tool output.
    # ModelConfig(name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ModelConfig(name="Qwen/Qwen3-30B-A3B-Instruct-2507"),
    ModelConfig(name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"),
    ModelConfig(name="Qwen/Qwen3-30B-A3B-Base"),
    ModelConfig(name="Qwen/Qwen3-30B-A3B-Thinking-2507"),
]


def _load_processor(config: ModelConfig) -> Any:
    model_path = config.get_full_path()
    try:
        return AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    except Exception as processor_error:
        try:
            return AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        except Exception as tokenizer_error:
            pytest.skip(f"Could not load tokenizer/processor for {config.name}: {processor_error}; {tokenizer_error}")


def _tokenize(processor: Any, text: str) -> list[int]:
    encoded = processor(text=[text], return_tensors="pt", add_special_tokens=False)
    return array_utils.to_int_list(encoded["input_ids"][0])


def _decode(processor: Any, token_ids: list[int]) -> str:
    try:
        return processor.decode(token_ids=token_ids, skip_special_tokens=False)
    except TypeError:
        return processor.decode(token_ids, skip_special_tokens=False)


def _assistant_content_for_model(model_name: str) -> str:
    if model_name.endswith("Thinking-2507"):
        return "I should answer directly.\n</think>\n\nThe answer is Paris."
    return "The answer is Paris."


@pytest.mark.parametrize("config", _MODEL_CONFIGS, ids=[config.name for config in _MODEL_CONFIGS])
def test_boundary_token_restores_assistant_to_tool_template(config: ModelConfig) -> None:
    """Boundary token restores the exact chat-template text before tool tokens.

    Example:
        decode(prompt_tokens + assistant_tokens + boundary_id + tool_tokens)
        == apply_chat_template([user, assistant, tool], add_generation_prompt=True)

    This covers truncated generation where ``assistant_tokens`` contain only
    the assistant content and do not already include the chat-template
    assistant boundary.
    """
    processor = _load_processor(config)
    boundary_id = get_single_token_assistant_boundary_id(processor)

    user_msg = Message(role="user", content="What is the capital of France?")
    assistant_content = _assistant_content_for_model(config.name)
    assistant_msg = Message(role="assistant", content=assistant_content)
    tool_msg = Message(role="tool", content="<information>Paris is in France.</information>", tool_call_id="call_1")

    prompt_text = processor.apply_chat_template([user_msg.to_dict()], add_generation_prompt=True, tokenize=False)
    prompt_tokens = _tokenize(processor, prompt_text)
    assistant_tokens = _tokenize(processor, assistant_content)
    assert assistant_tokens[-1] != boundary_id

    appended_tool_tokens = AppendedMessageTokenizer(config).process(tool_msg).tolist()
    decoded = _decode(processor, [*prompt_tokens, *assistant_tokens, boundary_id, *appended_tool_tokens])
    expected = processor.apply_chat_template(
        [user_msg.to_dict(), assistant_msg.to_dict(), tool_msg.to_dict()],
        add_generation_prompt=True,
        tokenize=False,
    )

    assert decoded == expected


def test_rollout_trace_keeps_boundary_as_context_not_current_turn_training(caplog: pytest.LogCaptureFixture) -> None:
    """Append the boundary after creating the current turn's training sample.

    Example:
        prompt [1, 2] + assistant [3, 4] + boundary [99]
        -> trace tokens [1, 2, 3, 4, 99]
        -> first turn sample input_ids [1, 2, 3, 4]

    The boundary becomes context for future turns but is not trained as part of
    the assistant turn that caused it to be appended.
    """
    conv = Conversation(
        messages=[Message(role="user", content="prompt")],
        gen_state=GenerationState(input_ids=array_utils.as_i32([1, 2]), capture_routing=True),
    )
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=16)
    h0 = TensorHandle(ref="nodeA:h0")

    with caplog.at_level(logging.INFO):
        trace._append_assistant_message(
            text="a0",
            tokens=array_utils.as_i32([3, 4]),
            logprobs=array_utils.as_f32([0.1, 0.2]),
            routing_handle=h0,
            assistant_boundary_token_id=99,
        )

    assert trace.token_ids.tolist() == [1, 2, 3, 4, 99]
    assert trace.turn_samples[0].input_ids.tolist() == [1, 2, 3, 4]
    assert trace.turn_samples[0].routing_handles_per_path == [[h0]]
    assert trace.conversation.gen_state.input_ids is not None
    assert trace.conversation.gen_state.input_ids.tolist() == [1, 2, 3, 4, 99]
    assert trace.conversation.gen_state.captured_routing_rows == 3
    assert "Appending assistant boundary token after creating the assistant turn sample" in caplog.text

    trace.append_user_or_tool_message(content="tool", tokens=array_utils.as_i32([5]))
    h1 = TensorHandle(ref="nodeA:h1")
    trace._append_assistant_message(
        text="a1",
        tokens=array_utils.as_i32([6]),
        logprobs=array_utils.as_f32([0.3]),
        routing_handle=h1,
        assistant_boundary_token_id=99,
    )

    assert trace.token_ids.tolist() == [1, 2, 3, 4, 99, 5, 6, 99]
    assert trace.token_trace is not None
    assert trace.token_trace.routing_row_count_per_handle == [3, 3]
    assert trace.conversation.gen_state.captured_routing_rows == 6


def test_multiturn_boundary_keeps_r3_routing_rows_aligned() -> None:
    """Boundary tokens stay in multi-turn context without shifting R3 rows.

    Example:
        [prompt 1,2] [assistant 3,4] [boundary 99] [tool 5]
        [assistant 6] [boundary 99]
        -> final trace [1, 2, 3, 4, 99, 5, 6, 99]

    The last turn sample includes the first boundary in its context
    ``[1, 2, 3, 4, 99, 5, 6]``. With routing handles from both assistant
    turns, the captured R3 row count must still match the sample's active
    token count minus one.
    """
    boundary_id = 99
    conv = Conversation(
        messages=[Message(role="user", content="prompt")],
        gen_state=GenerationState(input_ids=array_utils.as_i32([1, 2]), capture_routing=True),
    )
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=16)
    h0 = TensorHandle(ref="nodeA:h0")
    h1 = TensorHandle(ref="nodeA:h1")

    trace._append_assistant_message(
        text="a0",
        tokens=array_utils.as_i32([3, 4]),
        logprobs=array_utils.as_f32([0.1, 0.2]),
        routing_handle=h0,
        assistant_boundary_token_id=boundary_id,
    )
    trace.append_user_or_tool_message(content="tool", tokens=array_utils.as_i32([5]))
    trace._append_assistant_message(
        text="a1",
        tokens=array_utils.as_i32([6]),
        logprobs=array_utils.as_f32([0.3]),
        routing_handle=h1,
        assistant_boundary_token_id=boundary_id,
    )

    assert trace.token_ids.tolist() == [1, 2, 3, 4, boundary_id, 5, 6, boundary_id]
    assert trace.token_ids.tolist().count(boundary_id) == 2
    r3_sample = trace.turn_samples[-1]
    assert r3_sample.input_ids.tolist() == [1, 2, 3, 4, boundary_id, 5, 6]
    assert boundary_id in r3_sample.input_ids.tolist()
    assert r3_sample.routing_handles_per_path == [[h0, h1]]
    assert trace.token_trace is not None
    expected_routing_rows = int(r3_sample.attention_mask.sum()) - 1
    assert trace.token_trace.captured_routing_rows == expected_routing_rows

    no_r3_conv = Conversation(
        messages=[Message(role="user", content="prompt")],
        gen_state=GenerationState(input_ids=array_utils.as_i32([1, 2])),
    )
    no_r3_trace = RolloutTrace(no_r3_conv, token_in_token_out=True, max_length=16)
    no_r3_trace._append_assistant_message(
        text="a0",
        tokens=array_utils.as_i32([3, 4]),
        logprobs=array_utils.as_f32([0.1, 0.2]),
        assistant_boundary_token_id=boundary_id,
    )
    no_r3_trace.append_user_or_tool_message(content="tool", tokens=array_utils.as_i32([5]))
    no_r3_trace._append_assistant_message(
        text="a1",
        tokens=array_utils.as_i32([6]),
        logprobs=array_utils.as_f32([0.3]),
        assistant_boundary_token_id=boundary_id,
    )

    assert no_r3_trace.token_ids.tolist() == [1, 2, 3, 4, boundary_id, 5, 6, boundary_id]
    assert no_r3_trace.token_ids.tolist().count(boundary_id) == 2
    no_r3_sample = no_r3_trace.turn_samples[-1]
    assert no_r3_sample.input_ids.tolist() == [1, 2, 3, 4, boundary_id, 5, 6]
    assert boundary_id in no_r3_sample.input_ids.tolist()
    assert no_r3_sample.routing_handles_per_path is None


def test_rollout_trace_does_not_duplicate_existing_boundary(caplog: pytest.LogCaptureFixture) -> None:
    """Do not append a second boundary when the generated output already has it.

    Example:
        prompt [1, 2] + assistant output [3, 99]
        -> trace tokens [1, 2, 3, 99]

    This protects normal non-truncated generation, where SGLang already
    returned the boundary token in ``output_ids``.
    """
    conv = Conversation(
        messages=[Message(role="user", content="prompt")],
        gen_state=GenerationState(input_ids=array_utils.as_i32([1, 2])),
    )
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=16)

    with caplog.at_level(logging.INFO):
        trace._append_assistant_message(
            text="a0",
            tokens=array_utils.as_i32([3, 99]),
            logprobs=array_utils.as_f32([0.1, 0.2]),
            assistant_boundary_token_id=99,
        )

    assert trace.token_ids.tolist() == [1, 2, 3, 99]
    assert "Appending assistant boundary token after creating the assistant turn sample" not in caplog.text


def test_rollout_trace_appends_boundary_that_exactly_reaches_max_length(caplog: pytest.LogCaptureFixture) -> None:
    """Append boundary insertion when it exactly reaches max length.

    Example:
        prompt [1, 2] + assistant [3, 4] with max_length=5
        -> appending boundary [99] would produce length 5 == max_length
        -> append the boundary because the trace remains within max_length
    """
    conv = Conversation(
        messages=[Message(role="user", content="prompt")],
        gen_state=GenerationState(input_ids=array_utils.as_i32([1, 2])),
    )
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=5)

    with caplog.at_level(logging.INFO):
        trace._append_assistant_message(
            text="a0",
            tokens=array_utils.as_i32([3, 4]),
            logprobs=array_utils.as_f32([0.1, 0.2]),
            assistant_boundary_token_id=99,
        )

    assert trace.token_ids.tolist() == [1, 2, 3, 4, 99]
    assert trace.turn_samples[0].input_ids.tolist() == [1, 2, 3, 4]
    assert "Appending assistant boundary token after creating the assistant turn sample" in caplog.text


def test_rollout_trace_skips_boundary_that_would_exceed_max_length(caplog: pytest.LogCaptureFixture) -> None:
    """Skip boundary insertion when it would exceed max length.

    Example:
        prompt [1, 2] + assistant [3, 4] exactly fills max_length=4
        -> appending boundary [99] would produce length 5 > max_length
        -> keep the assistant sample and skip the optional boundary token
    """
    conv = Conversation(
        messages=[Message(role="user", content="prompt")],
        gen_state=GenerationState(input_ids=array_utils.as_i32([1, 2])),
    )
    trace = RolloutTrace(conv, token_in_token_out=True, max_length=4)

    with caplog.at_level(logging.INFO):
        trace._append_assistant_message(
            text="a0",
            tokens=array_utils.as_i32([3, 4]),
            logprobs=array_utils.as_f32([0.1, 0.2]),
            assistant_boundary_token_id=99,
        )

    assert trace.token_ids.tolist() == [1, 2, 3, 4]
    assert trace.turn_samples[0].input_ids.tolist() == [1, 2, 3, 4]
    assert "Skipping assistant boundary token append because it would exceed max_length" in caplog.text
