from dataclasses import dataclass, field

import numpy as np

from axrl.metrics.response_metric import ResponseMetric


@dataclass
class ConversationMetrics:
    """Aggregates and recent response metrics for one conversation."""

    dataset_name: str
    index: int
    num_rollouts: int = 0
    mean_moving_score: float | None = None
    mean_moving_tokens: int | None = None
    response_metrics: list[ResponseMetric] = field(default_factory=list)


class ConversationMetricsStore:
    """In-memory store of per conversation metrics."""

    def __init__(self, num_recent_metrics: int = 64) -> None:
        """Create a store that keeps the most recent response metrics."""
        self.dataset_conv_metrics: dict[str, list[ConversationMetrics]] = {}
        self.num_recent_metrics = num_recent_metrics

    def initialize(self, dataset_name: str, conv_count: int) -> None:
        """Initialize per-conversation metrics for a dataset."""
        assert dataset_name not in self.dataset_conv_metrics
        self.dataset_conv_metrics[dataset_name] = [
            ConversationMetrics(
                dataset_name=dataset_name,
                index=i,
            )
            for i in range(conv_count)
        ]

    def update(self, dataset_name: str, index: int, response_metrics: list[ResponseMetric]) -> None:
        """Update aggregates and append a response metric."""
        conv_metrics = self.get_conversation_metrics(dataset_name, index)

        # Update for each response metric
        for response_metric in response_metrics:
            assert response_metric.score is not None
            conv_metrics.num_rollouts += 1
            conv_metrics.response_metrics.append(response_metric)
            if len(conv_metrics.response_metrics) > self.num_recent_metrics:
                conv_metrics.response_metrics.pop(0)

        # Update moving averages
        conv_metrics.mean_moving_score = float(np.mean([m.score for m in conv_metrics.response_metrics]))  # type: ignore
        conv_metrics.mean_moving_tokens = int(np.mean([m.token_count for m in conv_metrics.response_metrics]))

    def get_conversation_metrics(self, dataset_name: str, index: int) -> ConversationMetrics:
        """Return metrics for a conversation."""
        assert dataset_name in self.dataset_conv_metrics
        dataset_conv_metrics = self.dataset_conv_metrics[dataset_name]
        assert 0 <= index < len(dataset_conv_metrics)
        return dataset_conv_metrics[index]

    def summary(self) -> dict[str, float]:
        """Return summary statistics for all conversations."""
        summary: dict[str, float] = {}
        for dataset_name, conv_metrics_list in self.dataset_conv_metrics.items():
            conv_metrics = [cm for cm in conv_metrics_list if cm.num_rollouts > 0]
            summary[f"{dataset_name}__num_conversations"] = float(len(conv_metrics))
            if not conv_metrics:
                continue
            mean_scores = np.mean([cm.mean_moving_score for cm in conv_metrics])  # type: ignore
            mean_tokens = np.mean([cm.mean_moving_tokens for cm in conv_metrics])  # type: ignore
            summary[f"{dataset_name}__mean_moving_score"] = float(mean_scores)
            summary[f"{dataset_name}__mean_moving_tokens"] = float(mean_tokens)
        return summary
