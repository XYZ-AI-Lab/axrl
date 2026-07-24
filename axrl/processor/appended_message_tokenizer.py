import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from transformers import AutoProcessor

from axrl.configs import ModelConfig
from axrl.data import Conversation, Message, array_utils
from axrl.processor.base_processor import BaseProcessor
from axrl.processor.chat_template_utils import get_single_token_assistant_boundary_id

logger = logging.getLogger(__name__)

_ASSISTANT_CONTENT_SENTINEL = "AXRL_APPENDED_MESSAGE_TOKENIZER_SENTINEL_7d7f1e"


class AppendedMessageTokenizer(BaseProcessor[Message, NDArray[np.int32]]):
    """Returns the tokens for a message appended to a chat conversation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        model_dir = config.get_full_path()
        self._processor: Any = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            use_fast=True,
        )
        self.assistant_boundary_token_id = get_single_token_assistant_boundary_id(self._processor)

    def get_newline_token_id(self) -> int:
        new_line_token_ids = self._processor(text=["\n"], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        assert len(new_line_token_ids) == 1
        return new_line_token_ids[0]

    def _get_user_prefix_conv(self) -> Conversation:
        """A minimal valid prefix before extracting appended user-turn tokens.

        Regression note: Qwen3 templates may rewrite assistant history when a
        later user turn is rendered. Use a sentinel assistant and only keep the
        suffix after its boundary so generated thinking/content stays untouched.
        """
        return Conversation(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello."),
                Message(role="assistant", content=_ASSISTANT_CONTENT_SENTINEL),
            ],
        )

    def _get_tool_prefix_conv(self) -> Conversation:
        """A minimal valid prefix before an appended tool turn.

        Regression note: Qwen3.5/3.6 native tool templates reject a bare
        assistant->tool sequence; the fake user turn keeps tool-result
        tokenization aligned with normal user->assistant->tool history.
        """
        conv = Conversation(
            messages=[
                Message(role="user", content="Hello."),
                Message(role="assistant", content=_ASSISTANT_CONTENT_SENTINEL),
            ],
        )
        return conv

    def _render_chat_template(self, conv: Conversation, *, add_generation_prompt: bool) -> str:
        prompt = self._processor.apply_chat_template(
            conv.to_dict()["messages"],
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        return prompt

    def _tokenize_text(self, text: str) -> list[int]:
        input_ids = self._processor(text=[text], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        return input_ids

    def process(self, item: Message) -> NDArray[np.int32]:
        """Tokenize the appended message."""
        if item.role == "assistant":
            if not isinstance(item.content, str):
                raise ValueError("AppendedMessageTokenizer only supports text assistant messages.")
            return array_utils.as_i32(self._tokenize_text(item.content))

        if item.role == "user":
            conv = self._get_user_prefix_conv()
        elif item.role == "tool":
            conv = self._get_tool_prefix_conv()
        else:
            raise ValueError(f"AppendedMessageTokenizer only supports assistant/user/tool messages, got role={item.role!r}.")

        conv.add_message(item)
        prompt = self._render_chat_template(conv, add_generation_prompt=True)
        sentinel_index = prompt.rfind(_ASSISTANT_CONTENT_SENTINEL)
        assert sentinel_index >= 0, "Appended-message tokenizer sentinel was not preserved by chat template."
        suffix = prompt[sentinel_index + len(_ASSISTANT_CONTENT_SENTINEL) :]
        input_ids = self._tokenize_text(suffix)
        assert input_ids and input_ids[0] == self.assistant_boundary_token_id, (
            "Appended-message tokenizer suffix should start with the assistant boundary token."
        )
        input_ids = input_ids[1:]
        return array_utils.as_i32(input_ids)
