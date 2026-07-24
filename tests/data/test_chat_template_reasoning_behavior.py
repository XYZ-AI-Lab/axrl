from typing import Any

import pytest
from transformers import AutoTokenizer

from axrl.configs import ModelConfig
from axrl.data import Message

QWEN3_NAME = "Qwen/Qwen3-1.7B"
QWEN25_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEEPSEEK_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEEPSEEK_ASSISTANT_THINK_PREFIX = "<\uff5cAssistant\uff5c><think>\n"


def _load_tokenizer(model_name: str) -> Any:
    return AutoTokenizer.from_pretrained(ModelConfig(name=model_name).get_full_path(), trust_remote_code=True, use_fast=True)


def _assistant_with_reasoning(reasoning: str, answer: str) -> Message:
    return Message(role="assistant", content=f"<think>\n{reasoning}\n</think>\n\n{answer}")


@pytest.fixture(scope="module")
def qwen3_tokenizer() -> Any:
    return _load_tokenizer(QWEN3_NAME)


@pytest.fixture(scope="module")
def qwen25_tokenizer() -> Any:
    return _load_tokenizer(QWEN25_NAME)


@pytest.fixture(scope="module")
def deepseek_tokenizer() -> Any:
    return _load_tokenizer(DEEPSEEK_NAME)


def test_qwen3_generation_prompt_changes_with_enable_thinking(qwen3_tokenizer: Any) -> None:
    r"""Qwen3 uses two different generation-prompt forms for thinking vs non-thinking mode.

    This test illustrates the subtle point that ``enable_thinking=False`` does
    not remove all think markup. Instead, the template emits an empty think block
    as part of the open assistant prompt:

    ``<|im_start|>assistant\n<think>\n\n</think>\n\n``

    The default path leaves the assistant prompt open without that empty block.
    """
    messages = [{"role": "user", "content": "What is the capital of France?"}]

    prompt_default = qwen3_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_no_thinking = qwen3_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    assert "<think>\n\n</think>\n\n" not in prompt_default
    assert prompt_default.endswith("<|im_start|>assistant\n")
    assert prompt_no_thinking.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen3_assistant_only_conversation_serialization(qwen3_tokenizer: Any) -> None:
    r"""Qwen3 serializes an assistant-only conversation as closed history, not an open prompt.

    This is the minimal case that often looks surprising when debugging append
    logic. For a conversation containing only ``assistant: Hi!``, the template
    renders a completed assistant history turn:

    ``<|im_start|>assistant\nHi!<|im_end|>\n``

    There is no trailing generation prompt and no think block in this history
    form.
    """
    messages = [{"role": "assistant", "content": "Hi!"}]

    prompt = qwen3_tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

    assert prompt == "<|im_start|>assistant\nHi!<|im_end|>\n"


def test_qwen3_strips_reasoning_for_assistant_before_last_user(qwen3_tokenizer: Any) -> None:
    """Qwen3 removes stored reasoning for assistant turns that are before the last user query.

    The important behavior is not "keep only the single last thinking block".
    Instead, Qwen3 finds the last real user query and treats earlier assistant
    turns as settled history. For those earlier turns it strips the historical
    ``<think>...</think>`` block and keeps only the final answer text.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        _assistant_with_reasoning("reason about Paris", "Paris is the capital of France.").to_dict(),
        {"role": "user", "content": "Can you also tell me about Germany?"},
    ]

    prompt = qwen3_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    assert "Paris is the capital of France." in prompt
    assert "reason about Paris" not in prompt
    assert "<think>\nreason about Paris\n</think>" not in prompt


def test_qwen3_keeps_reasoning_for_assistant_after_last_user(qwen3_tokenizer: Any) -> None:
    """Qwen3 preserves assistant reasoning in the current post-user segment.

    When the assistant message is after the last real user query, its stored
    reasoning remains in the rendered prompt. This is why a conversation ending
    with ``assistant -> tool`` behaves differently from ``assistant -> user``.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        _assistant_with_reasoning("reason about Paris", "Paris is the capital of France.").to_dict(),
        {"role": "tool", "content": "<information>Paris is the capital of France.</information>", "tool_call_id": "call_1"},
    ]

    prompt = qwen3_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    assert "<think>\nreason about Paris\n</think>" in prompt
    assert "Paris is the capital of France." in prompt


def test_qwen3_can_keep_multiple_reasoning_blocks_in_same_post_user_segment(qwen3_tokenizer: Any) -> None:
    """Qwen3 can preserve more than one reasoning block after the same last user message.

    This guards against the overly simplified interpretation that Qwen3 keeps
    only one global "last thinking block". If multiple assistant messages appear
    after the same last user query, each assistant message with reasoning in that
    segment is preserved by the template.
    """
    messages = [
        {"role": "user", "content": "Plan a search."},
        _assistant_with_reasoning("first reasoning", "First answer.").to_dict(),
        _assistant_with_reasoning("second reasoning", "Second answer.").to_dict(),
        {"role": "tool", "content": "<information>search result</information>", "tool_call_id": "call_1"},
    ]

    prompt = qwen3_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    assert "<think>\nfirst reasoning\n</think>" in prompt
    assert "<think>\nsecond reasoning\n</think>" in prompt


def test_deepseek_strips_historical_reasoning_but_starts_generation_with_think(deepseek_tokenizer: Any) -> None:
    r"""DeepSeek removes historical ``<think>`` blocks and uses a fresh think prefix for generation.

    DeepSeek's template strips any stored reasoning from assistant history before
    rendering previous turns. However, when ``add_generation_prompt=True`` opens a
    new assistant turn, the template starts that generation prompt with
    the assistant-think prefix token. This differs from Qwen3, which preserves some
    post-user historical reasoning blocks.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        _assistant_with_reasoning("deepseek reasoning", "Paris is the capital of France.").to_dict(),
        {"role": "user", "content": "Can you also tell me about Germany?"},
    ]

    prompt = deepseek_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    assert "deepseek reasoning" not in prompt
    assert "Paris is the capital of France." in prompt
    assert prompt.endswith(DEEPSEEK_ASSISTANT_THINK_PREFIX)


def test_qwen25_has_no_reasoning_specific_template_toggle(qwen25_tokenizer: Any) -> None:
    """Qwen2.5 serves as the non-reasoning contrast model for these template tests.

    Unlike Qwen3 and DeepSeek, the Qwen2.5 template does not have dedicated logic
    for splitting assistant content into reasoning and final-answer components.
    If a caller stores literal ``<think>...</think>`` text in the assistant
    content, the template keeps that text as-is.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        _assistant_with_reasoning("qwen25 reasoning", "Paris is the capital of France.").to_dict(),
        {"role": "user", "content": "Can you also tell me about Germany?"},
    ]

    prompt = qwen25_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    assert "<think>\nqwen25 reasoning\n</think>" in prompt
    assert "Paris is the capital of France." in prompt
