from typing import Any

from transformers import AutoProcessor

from axrl.configs import ModelConfig
from axrl.data import Conversation, array_utils
from axrl.data.generation import GenerationInput
from axrl.processor.base_processor import BaseProcessor


class ConversationTokenizer(BaseProcessor[Conversation, GenerationInput]):
    """Tokenizes conversation messages using a specified tokenizer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        model_dir = config.get_full_path()
        self._processor: Any = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            use_fast=True,
        )

    def process(self, item: Conversation) -> GenerationInput:
        """Tokenize the messages in the conversation."""
        gen_state = item.gen_state
        messages = [msg.to_dict() for msg in item.messages]
        assert messages[-1]["role"] == "user"
        prompt = self._processor.apply_chat_template(
            messages,
            tools=gen_state.tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = self._processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        session_id = gen_state.session_id or item.conversation_id or "conversation_tokenizer"
        return GenerationInput(
            session_id=session_id,
            input_ids=array_utils.as_i32(input_ids),
            tools=gen_state.tools,
            tool_choice=gen_state.tool_choice,
            tool_call_parser=gen_state.tool_call_parser,
        )
