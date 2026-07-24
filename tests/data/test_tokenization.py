"""Tests for tokenization to ensure no BOS token duplication."""

from typing import Any

import pytest
from transformers import AutoProcessor, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# One representative per tokenizer family axrl trains/serves. Guards against
# auto-class regressions like the transformers 5.6.0 silent slow-tokenizer
# downgrade that mangled prompts on this exact code path.
_ROUND_TRIP_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-30B-A3B-Base",
]


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from axrl.configs import ModelConfig

    path = ModelConfig(name=MODEL_NAME).get_full_path()
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def test_chat_template_includes_bos(tokenizer: Any) -> None:
    """Verify that apply_chat_template already includes BOS token in output."""
    messages = [{"role": "user", "content": "Hello"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    assert prompt.startswith(tokenizer.bos_token), f"Chat template should start with BOS token. Got: {prompt[:50]!r}"


def test_no_duplicate_bos_with_add_special_tokens_false(tokenizer: Any) -> None:
    """Verify that add_special_tokens=False prevents BOS duplication."""
    messages = [{"role": "user", "content": "Hello"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Correct way: add_special_tokens=False
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

    # First token should be BOS, second should NOT be BOS
    assert input_ids[0] == tokenizer.bos_token_id, "First token should be BOS"
    assert input_ids[1] != tokenizer.bos_token_id, f"Second token should NOT be BOS (got duplicate). Tokens: {input_ids[:5]}"


def test_no_duplicate_bos_even_without_add_special_tokens_false(tokenizer: Any) -> None:
    """Verify transformers >=5.6.0 does not duplicate BOS during re-tokenization.

    Earlier versions duplicated BOS when ``add_special_tokens=False`` was
    omitted; this guards against a regression that would re-introduce the
    historical double-BOS bug.
    """
    messages = [{"role": "user", "content": "Hello"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Default add_special_tokens=True path.
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

    assert input_ids[0] == tokenizer.bos_token_id, "First token should be BOS"
    assert input_ids[1] != tokenizer.bos_token_id, f"BOS should not be duplicated in transformers >=5.6.0. Got: {input_ids[:5]}"


@pytest.mark.parametrize("model_name", _ROUND_TRIP_MODELS)
def test_encode_decode_round_trip(model_name: str) -> None:
    from axrl.configs import ModelConfig

    text = "Solve the following.\nThe answer is 42."
    tok = AutoProcessor.from_pretrained(ModelConfig(name=model_name).get_full_path(), use_fast=True)
    ids = tok.encode(text, add_special_tokens=False)
    assert tok.decode(token_ids=ids, skip_special_tokens=True) == text
