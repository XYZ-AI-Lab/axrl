"""Regression for the handle-collection helper used by end-of-step delete.

``collect_unique_handles_from_samples`` is the pure dedup helper that
feeds ``store.delete_batch`` on the retire path in
``PipelineController.delete_r3_handles_and_caches``. Prior to the fix,
filtered-out groups leaked their R3 handles until step-end; this
helper plus the retire call site close that leak.
"""

from __future__ import annotations

from axrl.data import array_utils
from axrl.data.generation import TensorHandle
from axrl.data.sample import Sample, collect_unique_handles_from_samples


def _sample_with_handles(handles_per_path: list[list[TensorHandle]] | None) -> Sample:
    length = 4
    return Sample(
        input_ids=array_utils.as_i32(list(range(length))),
        labels=array_utils.as_i32(list(range(1, length + 1))),
        loss_mask=array_utils.as_bool([True] * length),
        attention_mask=array_utils.as_bool([True] * length),
        position_ids=array_utils.as_i32(list(range(length))),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * length),
        rollout_logprobs=array_utils.as_f32([0.0] * length),
        routing_handles_per_path=handles_per_path,
    )


def test_collect_dedupes_shared_handles_across_samples_and_paths() -> None:
    """Shared handles across samples and compacted-merge paths appear exactly once."""
    h0, h1, h2 = (TensorHandle(ref=f"nodeA:opk{i}") for i in range(3))
    samples = [
        _sample_with_handles([[h0, h1], [h0, h1]]),  # compacted: h0/h1 shared across 2 paths
        _sample_with_handles([[h2]]),
        _sample_with_handles([[h0]]),  # h0 already seen across samples
    ]
    assert collect_unique_handles_from_samples(samples) == [h0, h1, h2]


def test_collect_skips_samples_without_routing_handles() -> None:
    """Samples from R3-disabled rollouts (``routing_handles_per_path=None``) contribute nothing."""
    h0 = TensorHandle(ref="nodeA:opk0")
    samples = [
        _sample_with_handles(None),
        _sample_with_handles([[h0]]),
        _sample_with_handles(None),
    ]
    assert collect_unique_handles_from_samples(samples) == [h0]
