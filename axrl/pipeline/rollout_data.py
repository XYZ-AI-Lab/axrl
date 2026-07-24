from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from axrl.data import Conversation, RolloutResult
    from axrl.ray.ray_infer_worker import RayInferWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.utils.ray_task_queue import RayTaskQueue
    from axrl.worker.infer_worker import InferWorker

GroupFilterType = Literal[
    "pass",
    "zero_std_all_fail",
    "zero_std_all_success",
]


@dataclass(frozen=True)
class RolloutGroup:
    results: list[RolloutResult]
    filter_type: GroupFilterType

    @property
    def is_valid(self) -> bool:
        return self.filter_type == "pass"


@dataclass(frozen=True)
class TrainGroupBatch:
    valid_groups: list[RolloutGroup]
    skipped_groups: list[RolloutGroup] = field(default_factory=list)
    filter_type_counts: Counter[GroupFilterType] = field(default_factory=Counter)


@dataclass(frozen=True)
class RolloutRuntime:
    """Runtime handles passed to recipe rollout loops.

    Use ``local_workers`` for actor-local inference helpers, usually
    ``ProcessorPool`` instances, that handle short CPU requests such as
    tokenization or metric calculation. Keeping those workers beside the rollout
    actor avoids extra Ray object transfers.

    Use ``shared_workers`` for shared Ray worker wrappers such as LLM verifiers.
    They are better for longer requests where the network hop is small compared
    with the request cost.
    """

    rollout_worker: RayRolloutWorker
    rollout_queue: RayTaskQueue[Conversation]
    result_queue: RayTaskQueue[RolloutResult]
    local_workers: dict[str, InferWorker[Any, Any]] = field(default_factory=dict)
    shared_workers: dict[str, RayRolloutWorker | RayInferWorker[Any, Any]] = field(default_factory=dict)

    def get_local_worker(self, key: str) -> InferWorker[Any, Any]:
        assert key in self.local_workers, f"Rollout runtime is missing local worker {key!r}."
        return self.local_workers[key]

    def get_shared_worker(self, key: str) -> RayRolloutWorker | RayInferWorker[Any, Any]:
        assert key in self.shared_workers, f"Rollout runtime is missing shared worker {key!r}."
        return self.shared_workers[key]
