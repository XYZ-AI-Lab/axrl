import logging
import re
from typing import override

import pandas as pd

from axrl.configs import DatasetConfig, HfDataConfig
from axrl.data import Conversation, Message
from axrl.datasets.base_dataset import BaseDataset
from axrl.verifier.base_verifier import BaseVerifier

logger = logging.getLogger(__name__)


class GSM8K(BaseDataset):
    def __init__(self, config: DatasetConfig | None = None, *, is_train: bool) -> None:
        super().__init__(config)
        self.is_train = is_train

    @override
    def initialize(self) -> None:
        self.source = "openai/gsm8k"
        filename = "main/train-00000-of-00001.parquet" if self.is_train else "main/test-00000-of-00001.parquet"
        data = self._load_from_hf(HfDataConfig(repo_id=self.source, filename=filename))
        self._initialize(data)

    @override
    def get_verifier(self) -> type[BaseVerifier]:
        from axrl.verifier.gsm8k import GSM8KVerifier

        return GSM8KVerifier

    def _initialize(self, data: pd.DataFrame) -> None:
        conversations: list[Conversation] = []
        tag = self.__class__.__name__
        instruction_following = 'Let\'s think step by step and output the final answer after "####".'
        for i, row in data.iterrows():
            problem_raw: str = row["question"]
            problem = problem_raw + " " + instruction_following
            answer = str(row["answer"]).strip()
            answer = self.extract_solution(answer)
            conv = Conversation(
                messages=[
                    Message(role="user", content=problem),
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

    @staticmethod
    def extract_solution(solution_str: str) -> str:
        # Following https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        assert solution is not None
        final_solution = solution.group(0)
        final_solution = final_solution.split("#### ")[1].replace(",", "")
        return final_solution


class GSM8KTrain(GSM8K):
    def __init__(self, config: DatasetConfig | None = None) -> None:
        super().__init__(config, is_train=True)


class GSM8KTest(GSM8K):
    def __init__(self, config: DatasetConfig | None = None) -> None:
        super().__init__(config, is_train=False)


if __name__ == "__main__":
    from rich.pretty import pprint

    train_dataset = GSM8KTrain()
    train_dataset.initialize()
    print(f"Loaded {len(train_dataset)} training samples.")
    test_dataset = GSM8KTest()
    test_dataset.initialize()
    print(f"Loaded {len(test_dataset)} test samples.")
    merged_dataset = BaseDataset.concat([train_dataset, test_dataset])
    print(f"Merged dataset has {len(merged_dataset)} samples.")
    print("First sample in merged dataset:")
    pprint(merged_dataset._conversations[0])
