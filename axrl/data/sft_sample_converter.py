from typing import override

import numpy as np
import torch
from transformers import AutoTokenizer

from axrl.configs import IGNORE_INDEX, ModelConfig
from axrl.data.conversation import Conversation
from axrl.data.sample import Sample
from axrl.processor.base_processor import BaseProcessor


class SftSampleConverter(BaseProcessor[Conversation, Sample]):
    """Converts a Conversation into a Sample for Supervised Fine-Tuning (SFT)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get_full_path(),
            trust_remote_code=True,
        )
        self.max_length = config.seq_length
        self.pad_token_id = self.tokenizer.pad_token_id
        assert self.pad_token_id is not None, "pad_token_id should not be None. Please set pad_token_id in the tokenizer."

    def _shift(self, tensor: torch.Tensor, pad: float | None = None) -> torch.Tensor:
        if pad is None:
            pad = self.pad_token_id
        shifted = tensor.clone()
        shifted[:-1] = tensor[1:]
        if pad is not None:
            shifted[-1] = pad
        return shifted

    @override
    def process(self, item: Conversation) -> Sample:
        conversation = item
        messages = [x.to_dict() for x in conversation.messages]
        assert messages[-2]["role"] == "user"
        assert messages[-1]["role"] == "assistant"

        # transformers >=5.6 returns a BatchEncoding from apply_chat_template
        # when return_tensors is set; older versions returned a raw tensor.
        input_ids: torch.Tensor = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=False,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            tokenize=True,
            return_dict=False,
        )  # type: ignore
        prompt_ids: torch.Tensor = self.tokenizer.apply_chat_template(
            conversation=messages[:-1],
            add_generation_prompt=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            tokenize=True,
            return_dict=False,
        )  # type: ignore

        input_ids = input_ids.squeeze(0)
        prompt_ids = prompt_ids.squeeze(0)
        assert len(prompt_ids) > 0, "prompt_ids should have at least one token."
        assert input_ids.shape[0] >= prompt_ids.shape[0]

        # Remove last token of prompt_ids as it is often a newline (\n), which can have
        # different token IDs when combined with other whitespace in the full input_ids.
        prompt_ids = prompt_ids[:-1]

        # assert input_ids starts with prompt_tokens
        assert torch.equal(input_ids[: prompt_ids.shape[0]], prompt_ids), (
            f"All tokens should start with the prompt tokens, {input_ids=}, {prompt_ids=}"
        )

        assert 0 < input_ids.shape[0] <= self.max_length

        # Create labels
        labels = input_ids.clone()
        labels = self._shift(labels, pad=IGNORE_INDEX)
        num_prompt_labels = prompt_ids.shape[0] - 1  # -1 because of the shift

        seq_length = input_ids.shape[0]
        assert seq_length <= self.max_length, "Input sequence length exceeds maximum length."

        # Handling padding
        if seq_length < self.max_length:
            padding_length = self.max_length - seq_length
            padded_input_ids = torch.full(size=(padding_length,), fill_value=self.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, padded_input_ids])
            padded_label = torch.full(size=(padding_length,), fill_value=IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([labels, padded_label])

        # Create attention mask and position IDs
        attention_mask = torch.arange(self.max_length, device=input_ids.device) < seq_length
        position_ids = torch.arange(0, input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        loss_mask = labels.ne(IGNORE_INDEX).bool()
        is_prompt_label = torch.arange(0, self.max_length, dtype=torch.long, device=input_ids.device) < num_prompt_labels
        loss_mask = loss_mask & (~is_prompt_label)
        sample = Sample(
            input_ids=input_ids.detach().cpu().numpy().astype(np.int32, copy=True),
            labels=labels.detach().cpu().numpy().astype(np.int32, copy=True),
            attention_mask=attention_mask.detach().cpu().numpy().astype(np.bool_, copy=True),
            position_ids=position_ids.detach().cpu().numpy().astype(np.int32, copy=True),
            loss_mask=loss_mask.detach().cpu().numpy().astype(np.bool_, copy=True),
            reward=0.0,
            reward_baseline=0.0,
            advantage=np.zeros(len(input_ids), dtype=np.float32),
        )
        return sample
