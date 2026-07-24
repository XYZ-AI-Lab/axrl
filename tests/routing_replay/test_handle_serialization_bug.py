"""TensorDict roundtrip regressions for FLAT samples with distinct routing handles.

The R3 benchmark's KL1-explosion bug manifested as: after serialization,
all samples in a batch ended up with the SAME handle (the last sample's),
not their own. Handles at the non-tensor field layer must survive
(a) ``SampleTensorDict`` roundtrip, (b) ``TensorDict.split`` + realign,
(c) ``save_zst`` / ``load_zst``.
"""

from __future__ import annotations

from axrl.data import array_utils
from axrl.data.generation import TensorHandle
from axrl.data.sample import Sample, SampleTensorDict, samples_from_tensor_dict


def _make_flat_sample(length: int, handle: TensorHandle) -> Sample:
    return Sample(
        input_ids=array_utils.as_i32(list(range(length))),
        labels=array_utils.as_i32(list(range(1, length + 1))),
        loss_mask=array_utils.as_bool([False] * (length // 2) + [True] * (length - length // 2)),
        attention_mask=array_utils.as_bool([True] * length),
        position_ids=array_utils.as_i32(list(range(length))),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * length),
        rollout_logprobs=array_utils.as_f32([0.0] * length),
        routing_handles_per_path=[[handle]],
    )


def test_512_flat_samples_distinct_handles_survive_tensordict_roundtrip() -> None:
    """Benchmark-scale regression: 64 groups x 8 rollouts must all keep their own handle."""
    n_groups, n_rollouts = 64, 8
    samples = []
    for g in range(n_groups):
        for r in range(n_rollouts):
            samples.append(_make_flat_sample(10, TensorHandle(ref=f"nodeA:g{g}-r{r}")))
    assert len(samples) == 512
    td = SampleTensorDict.from_samples(samples)
    roundtripped = samples_from_tensor_dict(td)
    assert len(roundtripped) == 512
    for rt in roundtripped:
        assert rt.routing_handles_per_path is not None
    distinct = {rt.routing_handles_per_path[0][0] for rt in roundtripped if rt.routing_handles_per_path is not None}
    assert len(distinct) == 512, f"distinct handles collapsed: only {len(distinct)} of 512 remain — serialization benchmark bug"


def test_roundtrip_then_split_with_realign_preserves_per_row_handles() -> None:
    """``TensorDict.split`` broadcasts non-tensor fields; the ``realign`` helper reslices per-row.

    This locks in the mandatory fixup; without it, every microbatch gets the
    full-batch handle list and later materialisation corrupts.
    """
    from axrl.utils.megatron.seqlen_balancing import realign_non_tensor_keys_after_split

    n = 16
    samples = [_make_flat_sample(10, TensorHandle(ref=f"nodeA:split{i}")) for i in range(n)]
    td = SampleTensorDict.from_samples(samples)
    mbs = list(td.split(4))
    realign_non_tensor_keys_after_split(td, mbs)
    offset = 0
    for mb in mbs:
        mb_handles = mb.get_non_tensor("routing_handles_per_path")
        assert len(mb_handles) == int(mb.batch_size[0])
        for row, handles in enumerate(mb_handles):
            assert handles == [[TensorHandle(ref=f"nodeA:split{offset + row}")]]
        offset += int(mb.batch_size[0])


def test_zst_roundtrip_preserves_distinct_handles() -> None:
    """End-to-end production path: ``save_zst`` + ``load_zst`` must preserve distinct handles."""
    import tempfile
    from pathlib import Path

    from axrl.utils.zst_utils import load_zst, save_zst

    n = 32
    samples = [_make_flat_sample(10, TensorHandle(ref=f"nodeA:zst{i}")) for i in range(n)]
    td = SampleTensorDict.from_samples(samples)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "samples.zst"
        save_zst(td, path)
        loaded = load_zst(path)
    loaded_samples = samples_from_tensor_dict(loaded)
    for s in loaded_samples:
        assert s.routing_handles_per_path is not None
    distinct = {s.routing_handles_per_path[0][0] for s in loaded_samples if s.routing_handles_per_path is not None}
    assert len(distinct) == n
