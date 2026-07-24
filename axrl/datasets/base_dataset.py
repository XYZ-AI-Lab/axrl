import logging
from collections import Counter

import numpy as np
import pandas as pd

from axrl.configs import DatasetConfig, HfDataConfig, SampleType
from axrl.data.conversation import Conversation, Message
from axrl.utils.hf.download_data_from_hf import download_data_file
from axrl.verifier.base_verifier import BaseVerifier

logger = logging.getLogger(__name__)


class BaseDataset:
    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config: DatasetConfig | None = config
        self._conversations: list[Conversation] = []
        self._label: list[str | list[str]] = []
        self._score_history: list[list[float]] = []
        self._length_history: list[list[int]] = []
        self._history_capacity: int = 128
        self._conversation_id_to_index: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._conversations)

    def get_verifier(self) -> type[BaseVerifier] | None:
        return None

    @staticmethod
    def concat(datasets: list["BaseDataset"]) -> "BaseDataset":
        conversation_ids = set()
        dataset = BaseDataset()
        for ds in datasets:
            for conv in ds._conversations:
                assert conv.conversation_id, "Conversation ID should not be empty."
                assert conv.conversation_id not in conversation_ids, f"Duplicate conversation_id found during concat: {conv.conversation_id}"
                conversation_ids.add(conv.conversation_id)
            dataset._conversations.extend(ds._conversations)
            dataset._label.extend(ds._label)
            dataset._score_history.extend(ds._score_history)
            dataset._length_history.extend(ds._length_history)
        dataset._history_capacity = max(ds._history_capacity for ds in datasets)
        dataset._conversation_id_to_index = {conv.conversation_id: i for i, conv in enumerate(dataset._conversations)}
        dataset._check_initialized()
        source_counter = Counter([conv.source for conv in dataset._conversations])
        logger.info(f"Concatenated dataset has {len(dataset)} conversations from sources: {source_counter}")

        return dataset

    def filter_by_max_prompt_length(self, max_length: int) -> None:
        assert max_length > 0
        filtered_conversations: list[Conversation] = []
        filtered_labels: list[str | list[str]] = []
        filtered_score_history: list[list[float]] = []
        filtered_length_history: list[list[int]] = []
        for conv, label, scores, lengths in zip(self._conversations, self._label, self._score_history, self._length_history, strict=True):
            assert conv.gen_state.input_ids is not None
            if len(conv.gen_state.input_ids) > max_length:
                continue
            filtered_conversations.append(conv)
            filtered_labels.append(label)
            filtered_score_history.append(scores)
            filtered_length_history.append(lengths)
        logger.info(f"Filtered conversations by max_prompt_length={max_length}: {len(self._conversations)} -> {len(filtered_conversations)}")
        self._conversations = filtered_conversations
        self._label = filtered_labels
        self._score_history = filtered_score_history
        self._length_history = filtered_length_history
        self._conversation_id_to_index = {conv.conversation_id: i for i, conv in enumerate(self._conversations)}
        self._check_initialized()

    def _load_from_hf(self, config: HfDataConfig) -> pd.DataFrame:
        local_path = config.get_full_path()
        if local_path.exists():
            logger.info(f"Data file already exists at {local_path}, skipping download.")
        else:
            download_data_file(config)
        if local_path.suffix == ".parquet":
            data = pd.read_parquet(local_path)
        elif local_path.suffix == ".csv":
            data = pd.read_csv(local_path)
        elif local_path.suffix == ".json" or local_path.suffix == ".jsonl":
            data = pd.read_json(local_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {local_path.suffix}")

        logger.info(f"Loaded {len(data)} records from {local_path}.")
        logger.info("First row:")
        for col in data.columns:
            logger.info(f"  [{col}]: {data.iloc[0][col]!r}")
        return data

    def _initialize_from_data(self, data: pd.DataFrame, problem_col_name: str, answer_col_name: str, source: str) -> None:
        conversations: list[Conversation] = []
        prompt_end = r" Please reason step by step, and put your final answer within \boxed{}."
        tag = self.__class__.__name__
        for i, row in data.iterrows():
            content: str = row[problem_col_name].strip()
            if len(content) == 0:
                logger.warning(f"Row {i} has empty problem content; skipping: {row}.")
                continue
            conv = Conversation(
                messages=[
                    Message(role="system", content="You are a helpful assistant."),
                    Message(role="user", content=content + prompt_end),
                ]
            )
            conv.conversation_id = f"{tag}_sample_{i}"
            conv.extra["answer"] = str(row[answer_col_name])
            conv.source = source
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

    def _check_initialized(self, *, verbose: bool = True) -> None:
        assert len(self._conversations) > 0
        assert len(self._conversations) == len(self._score_history)
        assert len(self._conversations) == len(self._length_history)
        assert len(self._conversations) == len(self._label)
        assert len(self._conversation_id_to_index) == len(self._conversations)
        for i, conv in enumerate(self._conversations):
            assert conv.conversation_id, "Conversation ID should not be empty."
            assert conv.conversation_id in self._conversation_id_to_index
            assert len(conv.source) > 0, "Conversation source should not be empty."
            assert self._conversation_id_to_index[conv.conversation_id] == i

        if verbose:
            logger.info(f"Initialized {self.__class__.__name__} with {len(self._conversations)} conversations.")
            logger.info(f"First conversation sample: {self._conversations[0]}")

    def sample_conversations(
        self, num_samples: int, sample_type: SampleType = "uniform", num_moving_avg_steps: int = 64, epsilon: float = 0.05
    ) -> list[Conversation]:
        sampled_indices = self.sample(num_samples, sample_type, num_moving_avg_steps, epsilon)
        sampled_conversations = [self._conversations[i] for i in sampled_indices]
        return sampled_conversations

    def sample(self, num_samples: int, sample_type: SampleType = "uniform", num_moving_avg_steps: int = 64, epsilon: float = 0.05) -> list[int]:
        weights: list[float] = []
        for scores in self._score_history:
            avg_reward: float = 0.0 if len(scores) == 0 else float(np.mean(scores[-num_moving_avg_steps:]))
            if sample_type == "uniform":
                weights.append(1.0)
            elif sample_type == "uniform-no-easy":
                assert 0.0 <= avg_reward <= 1.0
                weights.append(0.0 if avg_reward >= 0.9 else 1.0)
            elif sample_type == "low-success-rate":
                assert 0.0 <= avg_reward <= 1.0
                weights.append(1.0 - avg_reward)
            elif sample_type == "intermediate-success-rate":
                assert 0.0 <= avg_reward <= 1.0
                weights.append(1.0 - abs(0.5 - avg_reward) * 2.0)
            else:
                raise ValueError(f"Unknown sample_type: {sample_type}")
        assert len(weights) == len(self._conversations)
        assert len(weights) > 0, "No conversations to sample from."
        probs: np.ndarray = np.array(weights, dtype=np.float32)
        assert epsilon > 0.0, "Epsilon must be positive."
        probs += epsilon  # to avoid any zero probs
        probs /= np.sum(probs)
        replace = num_samples > len(weights)
        if replace:
            logger.warning(f"Sampling with replacement since {num_samples} > dataset size {len(weights)}.")
        sampled_indices = np.random.choice(a=list(range(len(weights))), size=num_samples, replace=replace, p=probs)
        return sampled_indices.tolist()

    def sort_by_mean_lengths(self, sampled_indices: list[int], num_latest_lengths: int = 8) -> list[int]:
        """Sort the sampled indices by the mean of their latest lengths in descending order."""
        mean_lengths: list[tuple[int, int | None]] = []
        for index in sampled_indices:
            mean_length = self._get_latest_mean_length(index, num_latest_lengths)
            mean_lengths.append((index, mean_length))
        valid_lengths = [length for _, length in mean_lengths if length is not None]
        if len(valid_lengths) == 0:
            logger.warning("All mean lengths are None; returning original order.")
            return sampled_indices

        overall_mean_length = int(np.mean(valid_lengths))
        logger.info(f"There are {len(valid_lengths)}/{len(mean_lengths)} valid mean lengths; overall mean length: {overall_mean_length}.")
        valid_mean_lengths = [(index, length if length is not None else overall_mean_length) for index, length in mean_lengths]
        valid_mean_lengths.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, _ in valid_mean_lengths]
        return sorted_indices

    def get_hist_score_metrics(self, num_latest_scores: int) -> dict[str, float]:
        mean_scores: list[float] = []
        for score_history in self._score_history:
            scores = score_history[-num_latest_scores:]
            if not scores:
                continue
            mean_score: float = float(np.mean(scores))
            mean_scores.append(mean_score)
        tag = self.__class__.__name__
        info = {
            f"hist_mean_score__{tag}": float(np.mean(mean_scores)) if mean_scores else 0.0,
            f"hist_unique_count__{tag}": len(mean_scores),
        }
        return info

    def get_latest_scores(self, conversation_id: str, num_latest_scores: int) -> list[float]:
        assert num_latest_scores > 0
        index = self._get_index_by_conversation_id(conversation_id)
        score_history = self._score_history[index]
        return score_history[-num_latest_scores:]

    def _get_latest_mean_length(self, index: int, num_latest_lengths: int) -> int | None:
        assert num_latest_lengths > 0
        length_history = self._length_history[index]
        lengths = length_history[-num_latest_lengths:]
        if not lengths:
            return None

        return int(np.mean(lengths))

    def _get_index_by_conversation_id(self, conversation_id: str) -> int:
        assert conversation_id in self._conversation_id_to_index, f"Conversation ID {conversation_id} not found in dataset."
        return self._conversation_id_to_index[conversation_id]

    def get_conversation_by_conversation_id(self, conversation_id: str) -> Conversation:
        index = self._get_index_by_conversation_id(conversation_id)
        return self._conversations[index]

    def update_score_history(self, conversation_id: str, scores: list[float]) -> None:
        index = self._get_index_by_conversation_id(conversation_id)
        assert index < len(self._score_history)
        for score in scores:
            self._score_history[index].append(score)
            if len(self._score_history[index]) > self._history_capacity:
                self._score_history[index].pop(0)

    def update_length_history(self, conversation_id: str, lengths: list[int]) -> None:
        index = self._get_index_by_conversation_id(conversation_id)
        assert index < len(self._length_history)
        for length in lengths:
            self._length_history[index].append(length)
            if len(self._length_history[index]) > self._history_capacity:
                self._length_history[index].pop(0)

    def initialize(self) -> None:
        raise NotImplementedError

    def get_conv(self, index: int) -> Conversation:
        return self._conversations[index]

    def get_label(self, index: int) -> str | list[str]:
        return self._label[index]
