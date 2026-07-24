from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, override

import numpy as np
import pandas as pd
from scipy import stats

from axrl.data import GenerationOutput, RolloutResult, array_utils
from axrl.processor.base_processor import BaseProcessor


@dataclass
class ResponseMetric:
    token_count: int  # total number of tokens in the response
    # content filtering metrics
    token_unique_ratio: float  # lexical diversity: fraction of unique tokens
    word_length_max: int  # length of the longest word; can indicate anomalies or complex terms
    line_length_max: int  # length of the longest line; very long lines may hurt readability
    ngram_repetition: float  # proportion of repeated n-grams; detects degenerate or repetitive text
    # reasoning behavior metrics
    reasoning_behavior_backtracking: float  # frequency of backtracking cues (rethink, reconsider, start over)
    reasoning_behavior_verification: float  # frequency of verification cues (check, verify, ensure)
    reasoning_behavior_causal: float  # frequency of causal reasoning cues (because, leads to, results in)
    # rollout-specific (rollout engine) metrics
    rollout_cached_tokens: int  # number of cached tokens
    rollout_num_retry: int  # number of retries
    rollout_e2e_elapsed_seconds: float  # end-to-end elapsed time in seconds
    rollout_finish_reason_stop: int
    rollout_finish_reason_length: int
    rollout_finish_reason_tool_calls: int
    rollout_finish_reason_function_call: int
    rollout_finish_reason_content_filter: int
    # label
    is_low_quality: float | None = None  # tag for low-quality response
    score: float | None = None
    score_mean: float | None = None
    score_std: float | None = None
    driver_worker_overhead_seconds: float = 0.0  # skew-safe driver/worker handoff overhead

    def to_dict(self) -> dict[str, Any]:
        """Flatten this metric to a dict for aggregation.

        The default implementation uses ``dataclasses.asdict``.  Subclasses
        that carry dynamic or non-field metrics (e.g. tool-specific metrics
        stored in a dict) should override this to inject them as flat keys.
        """
        return asdict(self)


def is_low_quality(metric: ResponseMetric) -> bool:
    """Determine if the response is low quality based on token count and repetition metrics."""
    if metric.token_count >= 4096 and metric.ngram_repetition >= 0.4 and metric.token_unique_ratio <= 0.1:  # high repetition
        return True
    if metric.line_length_max >= 8192 and metric.ngram_repetition >= 0.2:  # very long line with repetition
        return True
    return False


class ResponseMetricCalculator(BaseProcessor[GenerationOutput, ResponseMetric]):
    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config)

        self.key_words: dict[str, set[str]] = {
            "backtracking": {"wait", "second thought", "re-examine", "rethink", "reconsider", "start over", "try another", "backtrack"},
            "verification": {"check", "verify", "confirm", "ensure", "validate", "cross-check"},
            "causal": {"because", "leads to", "due to", "consequence", "since", "therefore", "thus"},
        }

    def _calculate_entropy(self, items: list) -> float:
        if not items:
            return 0.0
        counter = Counter(items)
        counts = np.array(list(counter.values()))
        probs = counts / counts.sum()
        entropy = float(stats.entropy(probs))
        return entropy

    def _has_key_words(self, response: str, target_words: set[str]) -> float:
        for word in target_words:
            if word in response:
                return 1.0
        return 0.0

    def _get_ngram_repetition(self, tokens: list[int], n: int = 5) -> float:
        """Calculate the n-gram repetition by splitting the token list into 4K non-overlapping chunks to save memory.

        Returns the length-weighted mean of the repetition ratio.
        """
        if len(tokens) < n:
            return 0.0

        # split into chunks of 4k tokens to avoid memory issues
        chunk_size = 4000
        chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        repetitions = []
        length: list[int] = []
        for chunk in chunks:
            ngrams = ["_".join([str(x) for x in chunk[i : i + n]]) for i in range(len(chunk) - n + 1)]
            counter = Counter(ngrams)
            total_ngrams = sum(counter.values())
            repeated_ngrams = sum(count for count in counter.values() if count > 1)
            repeatition_ratio = repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0
            repetitions.append(repeatition_ratio)
            length.append(len(chunk))

        weighted_avg_repetition = float(np.average(repetitions, weights=length))
        return weighted_avg_repetition

    def _get_simple_words(self, words: list[str]) -> list[str]:
        # for each word, remove non-alphabetic characters
        simple_words: list = []
        for word in words:
            simple_word = "".join([c for c in word if c.isalpha() or c == "-"])
            if simple_word:
                simple_words.append(simple_word)
        return simple_words

    @override
    def process(self, item: GenerationOutput) -> ResponseMetric:
        assert isinstance(item, GenerationOutput)
        tokens = array_utils.to_int_list(item.output_ids)
        text: str = item.output_text
        text_lower = text.lower()
        lines = [line for line in text.splitlines() if line.strip()]
        line_lengths = [len(line) for line in lines]
        raw_words = [word for word in text.split() if word.strip()]
        lower_words = [word.lower() for word in raw_words]
        word_lengths = [len(word) for word in lower_words]
        metric = ResponseMetric(
            token_count=len(tokens),
            token_unique_ratio=len(set(tokens)) / (len(tokens) if tokens else 1),
            word_length_max=int(np.max(word_lengths)) if word_lengths else 0,
            line_length_max=int(np.max(line_lengths)) if line_lengths else 0,
            ngram_repetition=self._get_ngram_repetition(tokens, n=20),
            reasoning_behavior_backtracking=self._has_key_words(text_lower, self.key_words["backtracking"]),
            reasoning_behavior_verification=self._has_key_words(text_lower, self.key_words["verification"]),
            reasoning_behavior_causal=self._has_key_words(text_lower, self.key_words["causal"]),
            rollout_cached_tokens=item.cached_tokens,
            rollout_num_retry=item.retry,
            rollout_e2e_elapsed_seconds=item.e2e_elapsed_seconds,
            rollout_finish_reason_stop=1 if item.finish_reason == "stop" else 0,
            rollout_finish_reason_length=1 if item.finish_reason == "length" else 0,
            rollout_finish_reason_tool_calls=1 if item.finish_reason == "tool_calls" else 0,
            rollout_finish_reason_function_call=1 if item.finish_reason == "function_call" else 0,
            rollout_finish_reason_content_filter=1 if item.finish_reason == "content_filter" else 0,
            driver_worker_overhead_seconds=item.event_timing.driver_worker_overhead_seconds or 0.0,
        )
        metric.is_low_quality = float(is_low_quality(metric))
        return metric


def aggregate_response_metrics(
    response_metrics: list[ResponseMetric],
    total_seconds: float | None = None,
    prefix: str = "Training-Response",
) -> dict[str, float]:
    def _add_metric(data: pd.DataFrame, pos_prefix: str, metrics: dict[str, float]) -> None:
        if data.empty:
            return
        metrics_to_add = data.mean().to_dict()
        metrics_to_add["rollout_number"] = len(data)
        for key, value in metrics_to_add.items():
            metrics[f"{prefix}/{key}__{pos_prefix}"] = float(value)
        metrics[f"{prefix}/token_count_max__{pos_prefix}"] = float(data["token_count"].max())
        metrics[f"{prefix}/token_count_min__{pos_prefix}"] = float(data["token_count"].min())

    data = pd.DataFrame([m.to_dict() for m in response_metrics])

    data_pos = data[(data["score"].notna()) & (data["score"] > 0)]
    data_neg = data[(data["score"].notna()) & (data["score"] <= 0)]
    metrics: dict[str, float] = {}
    _add_metric(data, "all", metrics)
    _add_metric(data_pos, "pos", metrics)
    _add_metric(data_neg, "neg", metrics)
    all_tokens = data["token_count"].sum()
    if total_seconds is not None and total_seconds > 0:
        metrics[f"{prefix}/throughput_tokens_per_second"] = all_tokens / total_seconds
    return metrics


def aggregate_response_metrics_by_subset(
    results: Sequence[RolloutResult],
    subset_key: str,
    prefix: str,
) -> dict[str, float]:
    """Aggregate metrics per subset based on ``conv.extra[subset_key]``.

    Returns an empty dict when fewer than two subsets are present.
    """
    subset_to_metrics: dict[str, list[ResponseMetric]] = defaultdict(list)
    for result in results:
        conv = result.conversation
        metric = result.metric
        subset_value = conv.extra.get(subset_key) if conv is not None else None
        if subset_value is not None:
            subset_to_metrics[str(subset_value)].append(metric)

    if len(subset_to_metrics) <= 1:
        return {}

    all_metrics: dict[str, float] = {}
    for subset_value, subset_metrics in sorted(subset_to_metrics.items()):
        sub_prefix = f"{prefix}/{subset_value}"
        all_metrics.update(aggregate_response_metrics(subset_metrics, None, prefix=sub_prefix))
    return all_metrics


if __name__ == "__main__":
    from rich.pretty import pprint

    calculator = ResponseMetricCalculator()
    sample_output = GenerationOutput(
        session_id="test_session",
        output_text=(
            "This is a sample response. It is meant to test the ResponseMetricCalculator. \n\n"
            "Some keywords matters such as 'Wait', 're-examine', 'commonly', 'cross-check', and 'rethink'. "
            "Let's see how well it performs!"
        ),
        output_text_with_special_tokens="",
        output_ids=array_utils.as_i32([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 11, 12, 1, 2, 3, 4, 5]),
        output_logprobs=array_utils.as_f32([]),
        cached_tokens=0,
        finish_reason="stop",
        e2e_elapsed_seconds=0,
        stop_reason=None,
        retry=0,
    )
    result = calculator.process(sample_output)
    pprint(result)
