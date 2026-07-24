"""FlashRAG Natural Questions dataset for train/test splits."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, override

from axrl.configs import DatasetConfig, HfDataConfig
from axrl.data.conversation import Conversation, GenerationState, Message
from axrl.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

Split = Literal["train", "test"]

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information relevant to answering the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

if TYPE_CHECKING:
    import pandas as pd

    from axrl.verifier.base_verifier import BaseVerifier


class FlashRAGNQ(BaseDataset):
    def __init__(self, config: DatasetConfig | None = None, *, split: Split) -> None:
        super().__init__(config)
        self.split: Split = split

    @override
    def get_verifier(self) -> type[BaseVerifier] | None:
        return None

    def _make_prompt(self, question: str) -> str:
        question = question.strip()
        if question and question[-1] != "?":
            question += "?"
        return (
            "You are an information retrieval assistant who solves fact questions via multi-turn reasoning. "
            "You must begin with <think> ... </think> to outline your reasoning. "
            "Immediately after thinking, choose exactly one action: "
            "(a) use the search tool if you still need evidence, or "
            "(b) output <answer> your final reply </answer> and NOTHING else if you are confident. "
            "Keep the answer short without detailed illustrations (e.g., <answer> Beijing </answer>)."
            "Never add text after the action, and wait for the next message before searching again. "
            "After receiving search results, always start a new <think> block to analyze them, "
            "then decide whether to search again or to <answer> directly. "
            f"Question: {question}\n"
        )

    @override
    def initialize(self) -> None:
        self.source = "RUC-NLPIR/FlashRAG_datasets"
        data = self._load_from_hf(HfDataConfig(repo_id=self.source, filename=f"nq/{self.split}.jsonl"))
        self._initialize(data)

    def _initialize(self, data: pd.DataFrame) -> None:
        conversations: list[Conversation] = []
        tag = f"FlashRAGNQ_{self.split}"
        for idx, row in data.iterrows():
            question: str = str(row["question"])
            prompt = self._make_prompt(question)
            answer = row["golden_answers"]
            assert isinstance(answer, list)
            conv = Conversation(
                conversation_id=f"{tag}_{idx}",
                messages=[
                    Message(role="system", content="You are a helpful assistant."),
                    Message(role="user", content=prompt),
                ],
                extra={
                    "data_source": "nq",
                    "ability": "fact-reasoning",
                    "answer": answer,
                    "split": self.split,
                    "index": idx,
                },
                source=self.source,
                gen_state=GenerationState(tools=SEARCH_TOOLS, tool_call_parser="qwen"),
            )
            conversations.append(conv)
        logger.info(f"Initialized {len(conversations)}/{len(data)} conversations from data.")
        self._conversations = conversations
        self._label = [conv.extra["answer"] for conv in self._conversations]
        self._score_history = [[] for _ in conversations]
        self._length_history = [[] for _ in conversations]
        self._conversation_id_to_index = {conv.conversation_id: i for i, conv in enumerate(conversations)}
        self._check_initialized(verbose=True)


class FlashRAGNQTrain(FlashRAGNQ):
    def __init__(self, config: DatasetConfig | None = None) -> None:
        super().__init__(config, split="train")


class FlashRAGNQTest(FlashRAGNQ):
    def __init__(self, config: DatasetConfig | None = None) -> None:
        super().__init__(config, split="test")


if __name__ == "__main__":
    from axrl.utils import setup_logger

    setup_logger("info")
    train_dataset = FlashRAGNQTrain()
    train_dataset.initialize()

    test_dataset = FlashRAGNQTest()
    test_dataset.initialize()
