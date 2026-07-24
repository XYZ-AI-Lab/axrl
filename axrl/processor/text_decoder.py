import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from transformers import AutoProcessor

from axrl.configs import ModelConfig
from axrl.data import array_utils
from axrl.processor.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class TextDecoder(BaseProcessor[NDArray[np.int32], str]):
    """Tokenizes conversation messages using a specified tokenizer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        model_dir = config.get_full_path()
        self._processor: Any = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            use_fast=True,
        )

    def process(self, item: NDArray[np.int32]) -> str:
        return self._processor.decode(token_ids=array_utils.to_int_list(item), skip_special_tokens=False)
