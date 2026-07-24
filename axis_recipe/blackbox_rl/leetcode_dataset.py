from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Literal, override

if TYPE_CHECKING:
    import pandas as pd

from axrl.configs import DatasetConfig, HfDataConfig
from axrl.data import Conversation, Message
from axrl.datasets import register_dataset
from axrl.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

Split = Literal["train", "test"]


class LeetCodeDataset(BaseDataset):
    source = "newfacade/LeetCodeDataset"

    def __init__(self, config: DatasetConfig | None = None, *, split: Split) -> None:
        super().__init__(config)
        self.split = split

    @override
    def get_verifier(self) -> None:
        return None

    @override
    def initialize(self) -> None:
        filename = f"LeetCodeDataset-{self.split}.jsonl"
        data = self._load_from_hf(HfDataConfig(repo_id=self.source, filename=filename))
        self._initialize(data)

    def _initialize(self, data: pd.DataFrame) -> None:
        conversations: list[Conversation] = []
        tag = self.__class__.__name__
        for i, row in data.iterrows():
            query = str(row["query"]).strip()
            completion = str(row["completion"]).strip()
            if not query or not completion:
                logger.warning("Skipping empty LeetCode row at split=%s index=%s", self.split, i)
                continue

            task_id = str(row["task_id"]).strip()
            conv = Conversation(
                messages=[Message(role="user", content=query)],
                conversation_id=f"{tag}_{self.split}_{task_id or i}",
                source=self.source,
            )
            conv.extra["answer"] = completion
            conv.extra["split"] = self.split
            conv.extra["task_id"] = task_id
            conv.extra["question_id"] = int(row["question_id"])
            conv.extra["difficulty"] = str(row["difficulty"])
            conv.extra["tags"] = list(row["tags"])
            conv.extra["problem_description"] = str(row["problem_description"])
            conv.extra["starter_code"] = str(row["starter_code"])
            conv.extra["entry_point"] = str(row["entry_point"])
            conv.extra["test"] = str(row["test"])
            conv.extra["input_output"] = row["input_output"]
            conv.extra["reference_response"] = str(row["response"])
            conv.extra["verifier_label"] = make_leetcode_label(
                task_id=task_id,
                prompt=str(row["prompt"]),
                test=conv.extra["test"],
                entry_point=conv.extra["entry_point"],
            )
            conversations.append(conv)

        logger.info("Initialized %s/%s LeetCode conversations for split=%s.", len(conversations), len(data), self.split)
        self._conversations = conversations
        self._label = [str(conv.extra["verifier_label"]) for conv in self._conversations]
        self._score_history = [[] for _ in range(len(self._conversations))]
        self._length_history = [[] for _ in range(len(self._conversations))]
        self._conversation_id_to_index = {conv.conversation_id: i for i, conv in enumerate(self._conversations)}
        self._check_initialized()


class LeetCodeTrain(LeetCodeDataset):
    def __init__(self, config: DatasetConfig | None = None) -> None:
        super().__init__(config, split="train")


class LeetCodeTest(LeetCodeDataset):
    def __init__(self, config: DatasetConfig | None = None) -> None:
        super().__init__(config, split="test")


def make_leetcode_label(*, task_id: str, prompt: str, test: str, entry_point: str) -> str:
    return json.dumps(
        {
            "task_id": task_id,
            "prompt": prompt,
            "test": test,
            "entry_point": entry_point,
        },
        ensure_ascii=False,
    )


def register_leetcode_datasets() -> None:
    for name, dataset_cls in (
        ("newfacade/LeetCodeDataset/train", LeetCodeTrain),
        ("newfacade/LeetCodeDataset/test", LeetCodeTest),
    ):
        try:
            register_dataset(name, dataset_cls)
        except ValueError as exc:
            if "already registered" not in str(exc):
                raise
