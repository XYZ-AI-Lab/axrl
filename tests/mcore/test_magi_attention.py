"""Tests for `use_magi_merged_forward` path (Magi Attention + prefix tree).

Naming convention: ``test_<path_under_test>_<aspect>_matches_<baseline>[_<dataset>]``.
``magi_merged`` = prefix-tree-merging path; ``magi_flat`` = same Magi
``calc_attn`` kernel with a flat trie (no merging); ``te_baseline`` = the
default TE FA3 THD path.

Test catalog (one sentence each: name — aspect — assert condition):

- ``test_magi_merged_logprob_matches_te_baseline_realistic`` — realistic
  4-turn Hide-Tool-Result samples (``_REALISTIC_CASES_SINGLE_TRAJ``,
  DP=1 only) through ``compute_logprobs`` against TE; cosine > 0.999 and
  max abs diff < 3.0 at ``loss_mask=True`` positions.
- ``test_magi_merged_train_step_matches_te_baseline_realistic`` — same
  single-trajectory realistic samples through ``train_step``; loss rel
  diff < 3e-3 and grad-norm rel diff < 2e-2.
- ``test_magi_flat_logprob_matches_te_baseline_realistic`` — TE FA3 THD vs
  Magi ``calc_attn`` flat-trie forward (kernel-switch leg, no merging);
  bit-exact (max abs == 0.0) when CP=1, max abs < 3.0 + cosine > 0.999 when
  CP>1.
- ``test_magi_flat_train_step_matches_te_baseline_realistic`` — same
  kernel-switch leg through ``train_step``; bit-exact loss + grad-norm rel
  diff < 1e-3 when CP=1, loss + grad-norm rel diff < 3e-3 / 1.5e-2 when CP>1.
- ``test_magi_three_way_te_flat_merged_layer_diff_cp1_bi`` — single-rank
  diagnostic on realistic_cp1_bi running all three of TE / Magi-flat /
  Magi-merged with hooks on layer-0 sub-modules; ``te → flat`` bit-exact
  everywhere, pre-attention hooks bit-exact, ``flat → merged`` at
  ``core_attention`` bounded at 1 unit-scale bf16 spacing (≤8e-3).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import pytest
import torch

from axrl.data import Sample, SampleTensorDict
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from tests.mcore._context_management_fixture import (
    ParallelCase as Case,
)
from tests.mcore._context_management_fixture import (
    make_hide_tool_result_samples,
    make_megatron_worker_config,
    make_realistic_tokens,
)

if TYPE_CHECKING:
    from axrl.configs import MegatronWorkerConfig

logger = logging.getLogger(__name__)


def _make_config(case: Case) -> MegatronWorkerConfig:
    """Test-specific config: inference-only, gbs=8, mbs=2 (overrides shared defaults)."""
    config = make_megatron_worker_config(case, seq_length=1024)
    config.global_batch_size = 8
    config.train_micro_batch_size = 2
    config.eval_micro_batch_size = 4
    config.inference_only = True
    return config


def _compute_logprobs(
    config: MegatronWorkerConfig,
    samples: list[Sample],
    rg: ResourceGroup,
) -> torch.Tensor:
    worker = RayMegatronWorker(config=config, resource_group=rg)
    worker.initialize()
    logprobs, _ = worker.compute_logprobs(
        samples=SampleTensorDict.from_samples(samples),
        batch_size=config.global_batch_size,
    )
    worker.shutdown()
    return logprobs.cpu()


def _run_one_train_step(
    config: MegatronWorkerConfig,
    samples: list[Sample],
    rg: ResourceGroup,
) -> tuple[float, float]:
    """Return (loss, grad_norm) for a single `train_step` on the provided samples.

    Each sample is assigned its own ``trajectory_id`` so the new trajectory-grouped
    iterator treats ``config.global_batch_size`` as the trajectory count per gradient
    update.
    """
    samples = [Sample(**s.__dict__) for s in samples]
    for trajectory_id, sample in enumerate(samples):
        sample.trajectory_id = trajectory_id
    worker = RayMegatronWorker(config=config, resource_group=rg)
    worker.initialize()
    tensor_dict = SampleTensorDict.from_samples(samples)
    _global_step, metrics = worker.train(
        samples=tensor_dict,
        global_step=0,
        data_shuffle_seed=0,
        compute_logprobs=False,
    )
    worker.shutdown()
    return float(metrics["actor_train/loss"]), float(metrics["actor_train/grad_norm"])


def _make_realistic_hide_tool_result_samples(max_length: int = 2048) -> list[Sample]:
    """Build 4 realistic SFT samples using the Hide-Tool-Result strategy.

    Wrapper that delegates to the shared fixture. See
    :func:`tests.mcore._context_management_fixture.make_hide_tool_result_samples`
    for the full conversation structure (system + user + 4 assistant
    turns + 3 tool results) and Hide-Tool-Result trimming pattern.
    """
    return make_hide_tool_result_samples(make_realistic_tokens(max_length=max_length), max_length=max_length)


# Single combined parallel config exercises every parallel dimension at once
# instead of running a matrix; tp * cp * pp = 8 ranks (single-traj) and
# tp * cp * dp = 8 (multi-traj). The bi=True variant is kept for exact-match
# investigation.
_REALISTIC_CASES = [
    Case(name="realistic_tp2_cp2_pp2", tp=2, cp=2, pp=2),
    Case(name="realistic_tp2_cp2_dp2", tp=2, cp=2, dp=2),
    Case(name="realistic_tp2_cp2_pp2_bi", tp=2, cp=2, pp=2, batch_invariant=True),
]

_REALISTIC_CASES_SINGLE_TRAJ = [c for c in _REALISTIC_CASES if c.dp == 1]


@pytest.mark.parametrize("case", _REALISTIC_CASES_SINGLE_TRAJ, ids=lambda c: c.name)
def test_magi_merged_logprob_matches_te_baseline_realistic(case: Case) -> None:
    """Realistic 4-turn tool-using conversation with Hide-Tool-Result.

    Verifies that Magi prefix-tree forward matches the TE baseline for
    ``compute_logprobs()``.
    """
    if case.world_size() > torch.cuda.device_count():
        pytest.skip(f"Requires {case.world_size()} GPUs")

    samples = _make_realistic_hide_tool_result_samples(max_length=2048)

    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=case.world_size())])
    base_config = _make_config(case)
    base_config.model.seq_length = 2048
    base_config.use_magi_merged_forward = False
    base_logp = _compute_logprobs(base_config, samples, rg)

    # Magi merged path: merge the 4 per-turn samples into one merged ``Sample``
    # (with ``merge_info`` set) and feed it as a single trajectory.
    from axrl.utils.megatron.prefix_tree import merge_trajectory_samples, unpack_tensor_from_merged

    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=case.world_size())])
    magi_config = _make_config(case)
    magi_config.model.seq_length = 2048
    magi_config.use_magi_merged_forward = True
    merged_sample = merge_trajectory_samples(samples)
    magi_logp = _compute_logprobs(magi_config, [merged_sample], rg)

    # Unpack merged logprobs back to per-sample order (matches input ``samples`` order).
    assert merged_sample.merge_info is not None
    total_padded = merged_sample.merge_info.total_padded
    packed_logprobs = magi_logp[0, :total_padded].float()
    per_sample = unpack_tensor_from_merged(packed_logprobs, merged_sample.merge_info)
    assert len(per_sample) == len(samples)

    # Per-position comparison at trainable slots.
    base_train_per_sample: list[torch.Tensor] = []
    magi_train_per_sample: list[torch.Tensor] = []
    for i, s in enumerate(samples):
        loss_mask_i = torch.tensor(s.loss_mask, dtype=torch.bool)
        base_train_per_sample.append(base_logp[i].float()[loss_mask_i])
        magi_train_per_sample.append(torch.tensor(per_sample[i], dtype=torch.float)[loss_mask_i])

    base_train = torch.cat(base_train_per_sample)
    magi_train = torch.cat(magi_train_per_sample)
    assert base_train.numel() == magi_train.numel(), f"{case.name}: trainable count mismatch"
    diff = (base_train - magi_train).abs()
    max_abs_loss = diff.max().item() if diff.numel() > 0 else 0.0
    cos = torch.nn.functional.cosine_similarity(base_train.unsqueeze(0), magi_train.unsqueeze(0)).item()
    logger.info(f"realistic[{case.name}] max_abs(loss_mask, per-pos)={max_abs_loss:.4e} cosine(loss_mask, per-pos)={cos:.6f}")
    assert max_abs_loss < 3.0, f"{case.name}: max_abs(loss_mask)={max_abs_loss} too large"
    assert cos > 0.999, f"{case.name}: cosine(loss_mask)={cos} too low"
    ray_utils.stop()


@pytest.mark.parametrize("case", _REALISTIC_CASES_SINGLE_TRAJ, ids=lambda c: c.name)
def test_magi_merged_train_step_matches_te_baseline_realistic(case: Case) -> None:
    """Same realistic samples, now through ``train_step()``.

    Verifies equivalent loss + grad_norm.
    """
    if case.world_size() > torch.cuda.device_count():
        pytest.skip(f"Requires {case.world_size()} GPUs")

    samples = _make_realistic_hide_tool_result_samples(max_length=2048)

    # Megatron requires global_batch_size % (train_micro_batch_size * DP) == 0.
    # Sizing micro_batch = len(samples) // DP gives one microbatch per DP rank.
    assert len(samples) % case.dp == 0, f"sample count {len(samples)} must be divisible by DP={case.dp}"
    per_dp_micro = len(samples) // case.dp

    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=case.world_size())])
    base_config = _make_config(case)
    base_config.model.seq_length = 2048
    base_config.inference_only = False
    base_config.global_batch_size = len(samples)
    base_config.train_micro_batch_size = per_dp_micro
    base_config.log_every_k_steps = 1
    base_config.use_magi_merged_forward = False
    base_loss, base_gn = _run_one_train_step(base_config, samples, rg)

    # Magi merged path: merge the 4 per-turn samples into one merged ``Sample``.
    from axrl.utils.megatron.prefix_tree import merge_trajectory_samples

    merged_sample = merge_trajectory_samples(samples)

    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=case.world_size())])
    magi_config = _make_config(case)
    magi_config.model.seq_length = 2048
    magi_config.inference_only = False
    # Trajectory-aware merging: global_batch_size / train_micro_batch_size count
    # trajectories. The 4 per-turn samples become 1 merged trajectory.
    magi_config.global_batch_size = 1
    magi_config.train_micro_batch_size = 1
    magi_config.log_every_k_steps = 1
    magi_config.use_magi_merged_forward = True
    magi_loss, magi_gn = _run_one_train_step(magi_config, [merged_sample], rg)

    logger.info(f"realistic[{case.name}] base loss={base_loss:.6f} gn={base_gn:.6f} magi loss={magi_loss:.6f} gn={magi_gn:.6f}")
    # Observed worst-case across realistic cases on H20/FA3/Magi kernels:
    # loss rel diff 1.05e-3, grad-norm rel diff 1.633e-2.
    assert abs(base_loss - magi_loss) / max(abs(base_loss), 1e-6) < 3e-3
    assert abs(base_gn - magi_gn) / max(abs(base_gn), 1e-6) < 2e-2
    ray_utils.stop()


@pytest.mark.parametrize("case", _REALISTIC_CASES_SINGLE_TRAJ, ids=lambda c: c.name)
def test_magi_merged_train_step_full_recompute_realistic(case: Case) -> None:
    """Merged forward + ``train_step`` with full activation recompute must complete.

    With ``recompute_granularity='full'``, megatron's ``CheckpointFunction.backward``
    re-runs the forward closure during backward. The closure goes through the
    patched RoPE/attention, which read the ``_current`` ContextVar — by then
    already reset at end of the original forward. Without the
    ``CheckpointFunction`` monkey-patch in ``install_magi_attention_patch`` the
    patched RoPE asserts ``Magi RoPE patch is installed but no MagiForwardContext
    is active``. Test passes iff train_step finishes (no assert) and produces
    finite loss + grad-norm.
    """
    if case.world_size() > torch.cuda.device_count():
        pytest.skip(f"Requires {case.world_size()} GPUs")

    samples = _make_realistic_hide_tool_result_samples(max_length=2048)

    from axrl.utils.megatron.prefix_tree import merge_trajectory_samples

    merged_sample = merge_trajectory_samples(samples)

    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=case.world_size())])
    config = _make_config(case)
    config.model.seq_length = 2048
    config.inference_only = False
    # One merged trajectory ⇒ gbs=mbs=1.
    config.global_batch_size = 1
    config.train_micro_batch_size = 1
    config.log_every_k_steps = 1
    config.use_magi_merged_forward = True
    # Full activation recompute — the failure mode from the prod yaml.
    config.recompute_granularity = "full"
    config.recompute_method = "uniform"
    config.recompute_num_layers = 1
    loss, gn = _run_one_train_step(config, [merged_sample], rg)

    logger.info(f"recompute[{case.name}] loss={loss:.6f} gn={gn:.6f}")
    assert math.isfinite(loss), f"{case.name}: loss is non-finite: {loss}"
    assert math.isfinite(gn), f"{case.name}: grad_norm is non-finite: {gn}"
    ray_utils.stop()


@pytest.mark.parametrize("case", _REALISTIC_CASES, ids=lambda c: c.name)
def test_magi_flat_logprob_matches_te_baseline_realistic(case: Case) -> None:
    """TE FA3 THD forward vs Magi ``calc_attn`` with a flat trie.

    With CP=1 both paths use the same kernel arrangement (per-sample causal),
    so we expect bit-exact match. With CP>1 the two paths use entirely
    different distributed-attention schemes (TE Megatron CP ring vs Magi
    dispatch), so the bf16 ULP differences compound through 28 layers and
    we only require a loose absolute-bound + tight cosine-similarity match
    on the trainable tokens.
    """
    if case.world_size() > torch.cuda.device_count():
        pytest.skip(f"Requires {case.world_size()} GPUs")

    samples = _make_realistic_hide_tool_result_samples(max_length=2048)

    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=case.world_size())])
    te_config = _make_config(case)
    te_config.model.seq_length = 2048
    te_config.use_magi_merged_forward = False
    te_config.use_magi_flat_forward = False
    te_logp = _compute_logprobs(te_config, samples, rg)

    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=case.world_size())])
    flat_config = _make_config(case)
    flat_config.model.seq_length = 2048
    flat_config.use_magi_merged_forward = False
    flat_config.use_magi_flat_forward = True
    flat_logp = _compute_logprobs(flat_config, samples, rg)

    attn_mask = torch.tensor([s.attention_mask for s in samples])
    loss_mask = torch.tensor([s.loss_mask for s in samples])
    diff = (te_logp - flat_logp).abs()
    max_abs_valid = diff[attn_mask].max().item()
    max_abs_loss = diff[loss_mask].max().item() if loss_mask.any() else 0.0
    cos = torch.nn.functional.cosine_similarity(
        te_logp[loss_mask].flatten().float().unsqueeze(0),
        flat_logp[loss_mask].flatten().float().unsqueeze(0),
    ).item()
    logger.info(f"te_vs_flat[{case.name}] max_abs(valid)={max_abs_valid:.4e} max_abs(loss_mask)={max_abs_loss:.4e} cosine(loss_mask)={cos:.6f}")
    if case.cp == 1:
        # CP=1: same kernel arrangement on both sides → bit-exact.
        assert max_abs_valid == 0.0, f"{case.name}: expected bit-exact match, got max_abs={max_abs_valid}"
    else:
        # CP>1: TE and Magi use different CP attention schemes; bf16 ULP
        # noise compounds. Observed range across cp2/cp4/tp2_cp2/cp2_dp2:
        # ~5e-1 to ~1.0. Allow ~3x headroom and require a tight cosine.
        assert max_abs_valid < 3.0, f"{case.name}: max_abs_valid={max_abs_valid} too large"
        assert cos > 0.999, f"{case.name}: cosine_similarity={cos} too low"
    ray_utils.stop()


@pytest.mark.parametrize("case", _REALISTIC_CASES, ids=lambda c: c.name)
def test_magi_flat_train_step_matches_te_baseline_realistic(case: Case) -> None:
    """TE and Magi-flat must also agree very tightly on train_step loss and grad-norm.

    Forward is bit-exact (see ``test_magi_flat_logprob_matches_te_baseline_realistic``);
    backward uses kernel-specific autograd (FA3 backward vs Magi
    ``calc_attn`` backward), so the grad-norm can pick up small bf16 ULP
    noise. We use a tight-but-not-zero tolerance.
    """
    if case.world_size() > torch.cuda.device_count():
        pytest.skip(f"Requires {case.world_size()} GPUs")

    samples = _make_realistic_hide_tool_result_samples(max_length=2048)
    assert len(samples) % case.dp == 0
    per_dp_micro = len(samples) // case.dp

    def _run(*, use_magi_flat: bool) -> tuple[float, float]:
        ray_utils.restart()
        rg = ResourceGroup([Request(cpu=1, gpu=case.world_size())])
        config = _make_config(case)
        config.model.seq_length = 2048
        config.inference_only = False
        config.global_batch_size = len(samples)
        config.train_micro_batch_size = per_dp_micro
        config.log_every_k_steps = 1
        config.use_magi_merged_forward = False
        config.use_magi_flat_forward = use_magi_flat
        return _run_one_train_step(config, samples, rg)

    te_loss, te_gn = _run(use_magi_flat=False)
    flat_loss, flat_gn = _run(use_magi_flat=True)
    logger.info(f"te_vs_flat[{case.name}] te loss={te_loss:.6f} gn={te_gn:.6f} flat loss={flat_loss:.6f} gn={flat_gn:.6f}")
    if case.cp == 1:
        # Forward is bit-exact for cp=1 so loss is too.
        assert te_loss == flat_loss, f"{case.name}: te_loss={te_loss} flat_loss={flat_loss}"
        # Backward kernels differ (FA3 vs Magi calc_attn backward); both
        # paths also pick up ~2e-4 of run-to-run cuBLAS noise on grad-norm
        # at this scale, so allow 1e-3.
        assert abs(te_gn - flat_gn) / max(abs(te_gn), 1e-6) < 1e-3, f"{case.name}: te_gn={te_gn} flat_gn={flat_gn}"
    else:
        # CP>1: distributed-attention schemes differ. Observed worst-case
        # across realistic cases: loss rel diff 1.05e-3, grad-norm rel diff
        # 5.95e-3. Bands set with ~3x headroom.
        assert abs(te_loss - flat_loss) / max(abs(te_loss), 1e-6) < 3e-3, f"{case.name}: te_loss={te_loss} flat_loss={flat_loss}"
        assert abs(te_gn - flat_gn) / max(abs(te_gn), 1e-6) < 1.5e-2, f"{case.name}: te_gn={te_gn} flat_gn={flat_gn}"
    ray_utils.stop()


def test_magi_three_way_te_flat_merged_layer_diff_cp1_bi() -> None:
    """Run the three-way layer-diff diagnostic on realistic_cp1_bi and verify the expected pattern.

    - ``te → flat``: bit-exact (kernel switch is a no-op on identical inputs).
    - ``flat → merged``: bounded at 1 bf16 ULP at ``core_attention``, and
      everything strictly upstream of attention (layernorm, linear_qkv,
      q/k-norm, post-RoPE Q/K/V) is bit-exact.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Requires at least 1 GPU")

    samples = _make_realistic_hide_tool_result_samples(max_length=2048)
    case = Case(name="realistic_cp1_bi", batch_invariant=True)

    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=1)])
    config = _make_config(case)
    config.model.seq_length = 2048
    config.global_batch_size = len(samples)
    config.train_micro_batch_size = len(samples)
    config.eval_micro_batch_size = len(samples)
    config.inference_only = True
    config.use_magi_merged_forward = True  # ensures Magi patches are installed
    worker = RayMegatronWorker(config=config, resource_group=rg)
    worker.initialize()
    result = worker.magi_prefix_merging_layer_diff(SampleTensorDict.from_samples(samples))
    worker.shutdown()
    ray_utils.stop()

    by_name = {r["name"]: r for r in result["rows"]}

    # 1. Every pre-attention hook should be bit-exact on all three pairs.
    pre_attn_hooks = [
        "layer[0].input_layernorm",
        "layer[0].self_attention.linear_qkv",
        "layer[0].self_attention.q_layernorm",
        "layer[0].self_attention.k_layernorm",
        "layer[0].core_attention.in_q",
        "layer[0].core_attention.in_k",
        "layer[0].core_attention.in_v",
    ]
    for name in pre_attn_hooks:
        assert name in by_name, f"missing hook: {name}"
        row = by_name[name]
        for pair in ("te_to_flat", "flat_to_merged", "te_to_merged"):
            v = row[pair]["max_abs_at_loss_mask"]
            assert v == 0.0, f"{name} {pair} expected 0, got {v}"

    # 2. te → flat must be bit-exact everywhere (kernel switch is a no-op).
    for name, row in by_name.items():
        if row["te_to_flat"]["max_abs_at_loss_mask"] is None:
            continue  # skipped (shape mismatch for some submodules)
        assert row["te_to_flat"]["max_abs_at_loss_mask"] == 0.0, f"te_to_flat not bit-exact at {name}: {row['te_to_flat']}"

    # 3. flat → merged at core_attention must stay within 1 unit-scale
    # bf16 spacing (2^-7 = 0.0078125).
    core = by_name["layer[0].self_attention.core_attention"]
    core_delta = core["flat_to_merged"]["max_abs_at_loss_mask"]
    assert 0.0 < core_delta <= 8e-3, f"core_attention flat→merged delta out of band: {core_delta}"

    logger.info(f"three-way diff: layer[0].core_attention flat→merged = {core_delta:.3e}")


if __name__ == "__main__":
    for case in _REALISTIC_CASES:
        test_magi_merged_logprob_matches_te_baseline_realistic(case)
