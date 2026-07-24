from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ray

    from axrl.data.conversation import Conversation
    from axrl.data.rollout_trace import RolloutTrace
    from axrl.data.sample import SampleTensorDict
    from axrl.metrics.response_metric import ResponseMetric
    from axrl.utils.tensor_store import TensorHandle


@dataclass(slots=True)
class RolloutResult:
    conversation: Conversation
    trace: RolloutTrace | None
    metric: ResponseMetric
    packed_samples: SampleTensorDict | None = None
    packed_samples_ref: ray.ObjectRef[SampleTensorDict] | None = None
    trainable_token_count: int | None = None
    routing_handles: list[TensorHandle] | None = None
    reward: float | None = None
    reward_baseline: float | None = None
    scalar_advantage: float | None = None
