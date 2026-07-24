from typing import Literal, NamedTuple

import pandas as pd
import torch.distributed as dist
from megatron.core import mpu

from axrl.utils import dist_utils


class LogInfo(NamedTuple):
    step: int
    tag: str
    metrics: list[dict[str, float]]


BufferAggType = Literal["mean", "sum", "max", "min", "std"]


class LoggerBuffer:
    def __init__(
        self,
        rank: int,
        metric_agg_types: dict[str, list[BufferAggType]] | None = None,
        default_agg_type: BufferAggType = "mean",
    ) -> None:
        self._rank = rank
        self._metric_agg_types = metric_agg_types or {}
        self._default_agg_type = default_agg_type
        self._cached_metrics_all: dict[str, list[LogInfo]] = {}  # metrics that should gather from all ranks
        self._cached_metrics_dp: dict[str, list[LogInfo]] = {}  # metrics that should gather from dp group
        self._synced_metrics: dict[str, list[LogInfo]] = {}  # synced metrics

    def set_metric_agg_type(self, metric_name: str, agg_types: list[BufferAggType]) -> None:
        self._metric_agg_types[metric_name] = agg_types

    def update_metrics(self, step: int, tag: str, metrics: list[dict[str, float]], gather_group: Literal["all", "dp"] = "all") -> None:
        cached_metrics = self._cached_metrics_all if gather_group == "all" else self._cached_metrics_dp
        if tag not in cached_metrics:
            cached_metrics[tag] = []
        cached_metrics[tag].append(LogInfo(step, tag, metrics))

    def sync_and_flush(self) -> tuple[int, dict[str, float]]:
        self.sync()
        step, metrics_to_log = self._merge_synced_metrics()
        self._reset_metrics()
        return step, metrics_to_log

    def sync(self) -> None:
        self._synced_metrics = {}
        if dist.is_initialized() and dist.get_world_size() > 1:
            metrics_from_all = dist_utils.all_gather_object(self._cached_metrics_all)
            metrics_from_dp_group = dist_utils.all_gather_object(self._cached_metrics_dp, group=mpu.get_data_parallel_group())  # pyright: ignore[reportArgumentType]
            all_metrics = metrics_from_all + metrics_from_dp_group
        else:
            all_metrics = [self._cached_metrics_all, self._cached_metrics_dp]

        for rank_metrics in all_metrics:
            for tag, log_infos in rank_metrics.items():
                if tag not in self._synced_metrics:
                    self._synced_metrics[tag] = []
                self._synced_metrics[tag].extend(log_infos)

        # sort by step
        for tag in self._synced_metrics:
            self._synced_metrics[tag].sort(key=lambda x: x.step)

    def _reset_metrics(self) -> None:
        self._cached_metrics_all = {}
        self._cached_metrics_dp = {}
        self._synced_metrics = {}

    def _merge_synced_metrics(self) -> tuple[int, dict[str, float]]:
        metrics_to_log = {}
        step: int = 0
        for tag in list(self._synced_metrics.keys()):
            all_metrics = []
            for log_info in self._synced_metrics[tag]:
                all_metrics.extend(log_info.metrics)
                step = max(step, log_info.step)
            if not all_metrics:
                continue
            data = pd.DataFrame(all_metrics)
            for metric_name in data.columns:
                if metric_name not in self._metric_agg_types:
                    if metric_name.endswith("_max") or "batch_max" in metric_name:
                        self._metric_agg_types[metric_name] = ["max"]
                    elif metric_name.endswith("_min") or "batch_min" in metric_name:
                        self._metric_agg_types[metric_name] = ["min"]
                    elif metric_name.endswith("_sum") or "batch_sum" in metric_name:
                        self._metric_agg_types[metric_name] = ["sum"]
                    else:
                        self._metric_agg_types[metric_name] = [self._default_agg_type]  # type: ignore

                agg_types = self._metric_agg_types[metric_name]
                multiple_agg_types = len(agg_types) > 1
                for agg_type in agg_types:
                    suffix = f"_{agg_type}" if multiple_agg_types and agg_type != "mean" else ""
                    key = f"{tag}/{metric_name}{suffix}"
                    if agg_type == "mean":
                        metrics_to_log[key] = data[metric_name].mean()
                    elif agg_type == "sum":
                        metrics_to_log[key] = data[metric_name].sum()
                    elif agg_type == "max":
                        metrics_to_log[key] = data[metric_name].max()
                    elif agg_type == "min":
                        metrics_to_log[key] = data[metric_name].min()
                    elif agg_type == "std":
                        metrics_to_log[key] = data[metric_name].std()
                    else:
                        raise ValueError(f"Unknown aggregation type: {agg_type}")
        return step, metrics_to_log

    def aggregate_step_metrics(self, step_metrics: list[dict[str, float]]) -> dict[str, float]:
        if not step_metrics:
            return {}
        data = pd.DataFrame(step_metrics)
        aggregated_metrics = {}
        for metric_name in data.columns:
            if metric_name.endswith("_max"):
                aggregated_metrics[metric_name] = data[metric_name].max()
            elif metric_name.endswith("_min"):
                aggregated_metrics[metric_name] = data[metric_name].min()
            elif metric_name.endswith("_sum"):
                aggregated_metrics[metric_name] = data[metric_name].sum()
            else:
                aggregated_metrics[metric_name] = data[metric_name].mean()
        return aggregated_metrics
