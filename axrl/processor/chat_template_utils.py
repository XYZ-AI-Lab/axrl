from typing import Any

from axrl.data import array_utils

_ASSISTANT_BOUNDARY_SENTINEL = "AXRL_ASSISTANT_BOUNDARY_SENTINEL_9f9d7a"


def _tokenize_text(processor: Any, text: str) -> list[int]:
    encoded = processor(text=[text], return_tensors="pt", add_special_tokens=False)
    return array_utils.to_int_list(encoded["input_ids"][0])


def _strip_trailing_newline_token(processor: Any, token_ids: list[int]) -> list[int]:
    newline_ids = _tokenize_text(processor, "\n")
    if len(newline_ids) == 1 and token_ids and token_ids[-1] == newline_ids[0]:
        return token_ids[:-1]
    return token_ids


def get_single_token_assistant_boundary_id(processor: Any) -> int:
    r"""Return the single chat-template token that terminates assistant history.

    Qwen-style templates render an assistant history suffix like
    ``<|im_end|>\n``. The separator newline belongs to the following appended
    message tokenization path, so this helper strips it and asserts that the
    remaining assistant boundary is exactly one token.
    """
    messages = [
        {"role": "user", "content": "AXRL boundary probe"},
        {"role": "assistant", "content": _ASSISTANT_BOUNDARY_SENTINEL},
    ]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    sentinel_index = rendered.rfind(_ASSISTANT_BOUNDARY_SENTINEL)
    assert sentinel_index >= 0, "assistant boundary probe sentinel was not preserved by chat template"

    suffix = rendered[sentinel_index + len(_ASSISTANT_BOUNDARY_SENTINEL) :]
    boundary_ids = _strip_trailing_newline_token(processor, _tokenize_text(processor, suffix))
    assert len(boundary_ids) == 1, (
        f"assistant chat-template boundary must be a single token after newline stripping, got {boundary_ids} from suffix {suffix!r}"
    )
    return int(boundary_ids[0])
