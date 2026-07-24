"""Tests that AppendedMessageTokenizer returns the suffix for an appended turn."""

from typing import Any

import pytest
from transformers import AutoProcessor

from axrl.configs import ModelConfig
from axrl.data import Message
from axrl.processor.appended_message_tokenizer import AppendedMessageTokenizer
from axrl.processor.chat_template_utils import get_single_token_assistant_boundary_id

MODEL_NAMES = [
    pytest.param("Qwen/Qwen2.5-1.5B-Instruct", id="qwen2.5"),
    pytest.param("Qwen/Qwen3-1.7B", id="qwen3"),
]


ROLES = [
    pytest.param("user", id="user"),
    pytest.param("tool", id="tool"),
]


def _make_follow_up_message(role: str) -> Message:
    if role == "user":
        return Message(role="user", content="Can you also tell me about Germany?")
    return Message(role="tool", content="<information>Paris is the capital of France.</information>", tool_call_id="call_1")


def _tokenize_messages(processor: Any, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()


def _tokenize_text(processor: Any, text: str) -> list[int]:
    return processor(text=[text], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()


def _decode(processor: Any, token_ids: list[int]) -> str:
    try:
        return processor.decode(token_ids=token_ids, skip_special_tokens=False)
    except TypeError:
        return processor.decode(token_ids, skip_special_tokens=False)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("role", ROLES)
def test_new_tokens_append_after_generated_assistant_boundary(model_name: str, role: str) -> None:
    """Appending a user or tool turn should preserve generated assistant content."""
    config = ModelConfig(name=model_name)
    processor: Any = AutoProcessor.from_pretrained(config.get_full_path(), use_fast=True)
    appended_msg_tok = AppendedMessageTokenizer(config)

    msg = _make_follow_up_message(role)
    prompt_tokens = _tokenize_messages(
        processor,
        [Message(role="user", content="France capital?").to_dict()],
        add_generation_prompt=True,
    )
    assistant_content = "The capital of France is Paris."
    assistant_tokens = _tokenize_text(processor, assistant_content)
    boundary_id = get_single_token_assistant_boundary_id(processor)
    new_tokens = appended_msg_tok.process(msg)
    decoded = _decode(processor, [*prompt_tokens, *assistant_tokens, boundary_id, *new_tokens.tolist()])

    assert assistant_content in decoded
    assert isinstance(msg.content, str)
    assert msg.content in decoded
    assert decoded.index(assistant_content) < decoded.index(msg.content)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_assistant_tokens_append_to_open_generation_prompt(model_name: str) -> None:
    """Appending an assistant turn should return the generated content tokens."""
    config = ModelConfig(name=model_name)
    processor: Any = AutoProcessor.from_pretrained(config.get_full_path(), use_fast=True)
    appended_msg_tok = AppendedMessageTokenizer(config)

    user_msg = Message(role="user", content="Please compare A100 and H100 briefly.")
    assistant_content = "<think>\ncompare the request\n</think>\n\nH100 is typically faster and more efficient."
    assistant_msg = Message(role="assistant", content=assistant_content)

    prompt_tokens = _tokenize_messages(processor, [user_msg.to_dict()], add_generation_prompt=True)
    new_tokens = appended_msg_tok.process(assistant_msg).tolist()

    assert new_tokens == _tokenize_text(processor, assistant_content)
    assert _decode(processor, [*prompt_tokens, *new_tokens]).endswith(assistant_content)


if __name__ == "__main__":
    for model_name in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-1.7B"):
        for role in ("user", "tool"):
            test_new_tokens_append_after_generated_assistant_boundary(model_name=model_name, role=role)
        test_assistant_tokens_append_to_open_generation_prompt(model_name=model_name)
