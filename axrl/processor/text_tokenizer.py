import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from transformers import AutoProcessor

from axrl.configs import ModelConfig
from axrl.data import array_utils
from axrl.processor.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class TextTokenizer(BaseProcessor[str, NDArray[np.int32]]):
    """Tokenizes conversation messages using a specified tokenizer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        model_dir = config.get_full_path()
        self._processor: Any = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            use_fast=True,
        )

    def process(self, item: str) -> NDArray[np.int32]:
        """Tokenize the messages in the conversation."""
        input_ids = self._processor(text=[item], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        return array_utils.as_i32(input_ids)
