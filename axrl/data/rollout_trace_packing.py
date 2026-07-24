from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from axrl.data.sample import SampleTensorDict
from axrl.processor.base_processor import BaseProcessor

if TYPE_CHECKING:
    from axrl.data.sample import Sample

logger = logging.getLogger(__name__)


@dataclass
class RolloutTracePackRequest:
    trajectory_id: int
    turn_samples: list[Sample]
    max_pack_length: int
    allow_prefix_sharing: bool


def _limit_pack_worker_threads() -> None:
    """Keep each packing process from oversubscribing CPU threads."""
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover - best-effort process-local tuning
        logger.debug("Could not configure torch thread count in rollout packing worker.", exc_info=True)


class RolloutTracePackingProcessor(BaseProcessor[RolloutTracePackRequest, SampleTensorDict]):
    def __init__(self, config: None = None) -> None:
        super().__init__(config)
        _limit_pack_worker_threads()

    @override
    def process(self, item: RolloutTracePackRequest) -> SampleTensorDict:
        return pack_rollout_trace_samples_to_tensor_dict(
            trajectory_id=item.trajectory_id,
            turn_samples=item.turn_samples,
            max_pack_length=item.max_pack_length,
            allow_prefix_sharing=item.allow_prefix_sharing,
        )


def pack_rollout_trace_samples_to_tensor_dict(
    *,
    trajectory_id: int,
    turn_samples: list[Sample],
    max_pack_length: int,
    allow_prefix_sharing: bool,
) -> SampleTensorDict:
    """Pack one rollout trace's turn samples into a tensorized batch."""
    # Bypass RolloutTrace.__init__; to_packed_samples only needs turn_samples here.
    from axrl.data.rollout_trace import RolloutTrace

    trace = RolloutTrace.__new__(RolloutTrace)
    trace.turn_samples = turn_samples
    trace_samples = trace.to_packed_samples(
        max_pack_length=max_pack_length,
        allow_prefix_sharing=allow_prefix_sharing,
    )
    for sample in trace_samples:
        sample.trajectory_id = trajectory_id
    assert trace_samples, "packing chunk produced no samples"
    return SampleTensorDict.from_samples(trace_samples, max_length=max_pack_length)
