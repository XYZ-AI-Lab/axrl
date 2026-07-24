import logging
from typing import TYPE_CHECKING, override

import pandas as pd

from axrl.configs import HfDataConfig
from axrl.data import Conversation, Message
from axrl.datasets.base_dataset import BaseDataset

if TYPE_CHECKING:
    from axrl.verifier.base_verifier import BaseVerifier

logger = logging.getLogger(__name__)


class AIME2024(BaseDataset):
    @override
    def initialize(self) -> None:
        self.source = "BytedTsinghua-SIA/AIME-2024"
        data = self._load_from_hf(HfDataConfig(repo_id=self.source, filename="data/aime-2024.parquet"))
        # create hash for prompt for de-duplication
        data["prompt_hash"] = data["prompt"].apply(lambda x: hash(x[0]["content"]))
        data = data.drop_duplicates(subset=["prompt_hash"]).reset_index(drop=True)
        self._initialize(data)

    @override
    def get_verifier(self) -> type["BaseVerifier"]:
        from axrl.verifier.dapo_verifier import DapoVerifier

        return DapoVerifier

    def _initialize(self, data: pd.DataFrame) -> None:
        conversations: list[Conversation] = []
        tag = self.__class__.__name__
        for i, row in data.iterrows():
            content: str = row["prompt"][0]["content"].strip()
            answer = row["reward_model"]["ground_truth"]
            content = f"{content}\n\n" + r"Please reason step by step, and put your final answer within \boxed{}."
            if len(content) == 0:
                logger.error(f"Row {i} has empty problem content; skipping: {row}.")
                continue
            conv = Conversation(
                messages=[
                    Message(role="user", content=content),
                ]
            )
            conv.conversation_id = f"{tag}_sample_{i}"
            conv.extra["answer"] = str(answer)
            conv.source = self.source
            if len(conv.extra["answer"]) == 0:
                logger.error(f"Row {i} has empty answer; skipping: {row}.")
                continue
            conversations.append(conv)
        logger.info(f"Initialized {len(conversations)}/{len(data)} conversations from data.")
        self._conversations = conversations
        self._label = [str(conv.extra["answer"]) for conv in self._conversations]
        self._score_history = [[] for _ in range(len(self._conversations))]
        self._length_history = [[] for _ in range(len(self._conversations))]
        self._conversation_id_to_index = {conv.conversation_id: i for i, conv in enumerate(self._conversations)}
        self._check_initialized()


if __name__ == "__main__":
    from axrl.utils import setup_logger

    setup_logger("info")
    dataset = AIME2024()
    dataset.initialize()
    samples = dataset.sample_conversations(10)
    for i, sample in enumerate(samples):
        print(f"Sample {i}:")
        print(sample)
