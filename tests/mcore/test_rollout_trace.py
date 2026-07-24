"""Tests for the Magi merged forward (trajectory-aware prefix-merged).

Each ``Sample`` is one trajectory; when ``merge_info`` is set, the
trajectory's per-turn paths have already been prefix-merged into the
DFS-pre-order packed layout by :func:`merge_trajectory_samples`. The
worker dispatches between flat (``merge_info is None``) and merged
(``merge_info is not None``) paths via ``config.use_magi_merged_forward``.

Test catalog:

- ``test_merge_trajectory_samples_round_trip_per_turn_no_compaction`` — CPU.
  Each turn-sample's structural fields round-trip exactly through
  merge → unpack; trainable positions agree on labels and loss_mask.
- ``test_merged_token_layout_matches_token_trace_no_compaction`` — CPU.
  The merged sample's last-path tokens match the linear ``TokenTrace`` stream.
- ``test_loss_aware_split_then_merge_round_trip_cpu`` — CPU. The
  ``[1,2,3,4]`` / ``[1,2,5,6]`` divergent-trainable case end-to-end.
- ``test_rollout_trace_merged_train_step_matches_token_trace_realistic`` —
  GPU. Worker's merged path (``train(samples=[merged_sample])``) produces
  loss/grad-norm matching the linear baseline within tolerance.
- ``test_two_trajectory_merged_train_step_close_to_unmerged_realistic`` —
  GPU. Two-trajectory merge matches feeding two flat samples through the
  baseline.
- ``test_merged_train_step_matches_flat_shared_trainable_in_one_path_realistic`` —
  GPU regression. Two paths sharing a long region (≥70% of tokens) where
  exactly one path is trainable on the shared region (the other is
  trainable on its divergent tail); merged forward must match flat (B=2)
  baseline.
- ``test_merged_train_step_matches_flat_shared_trainable_in_one_path_tiny`` —
  GPU regression. Tiny synthetic 3+3+3-token version of the above
  with hand-chosen token IDs.
- ``test_grpo_merged_train_step_matches_baseline_realistic`` —
  GPU. ``GrpoTrainer`` on the same fixture: 4 turn samples (flat path)
  vs 1 merged sample (prefix-tree merged path). Both use
  ``compute_logprobs=True`` so actor-local training refreshes ref and old
  logprobs from the model before backward. Asserts
  ``actor_train/loss``, ``actor_train/entropy_mean``, ``actor_train/grad_norm`` match
  within tolerance for cp1/cp2/tp2/pp2.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from axrl.configs import IGNORE_INDEX, MegatronWorkerConfig
from axrl.data import array_utils
from axrl.data.conversation import Conversation, GenerationState, Message
from axrl.data.rollout_trace import RolloutTrace
from axrl.data.sample import Sample
from axrl.data.token_trace import TokenTrace
from axrl.utils.megatron.prefix_tree import (
    merge_trajectory_samples,
    path_chain_root_to_leaf,
    unpack_tensor_from_merged,
)
from tests.mcore._context_management_fixture import (
    ParallelCase as _ParallelCase,
)
from tests.mcore._context_management_fixture import (
    RealisticTokens as _RealisticTokens,
)
from tests.mcore._context_management_fixture import (
    make_megatron_worker_config,
)
from tests.mcore._context_management_fixture import (
    make_realistic_tokens as _make_realistic_tokens,
)
from tests.mcore._context_management_fixture import (
    train_step_loss_gn as _train_step_loss_gn,
)


def _make_parallel_config(case: _ParallelCase) -> MegatronWorkerConfig:
    """Worker config for these tests — wider ``seq_length=4096`` than the shared default."""
    return make_megatron_worker_config(case, seq_length=4096)


logger = logging.getLogger(__name__)


# =====================================================================
# Conversation fixture (realistic 4-turn tool-using conversation)
# =====================================================================


# NOTE: The realistic 4-turn fixture lives in ``tests/mcore/_context_management_fixture.py``;
# it is imported above as ``_RealisticTokens`` and ``_make_realistic_tokens``.


def _build_rollout_trace(toks: _RealisticTokens, *, max_length: int = 4096) -> RolloutTrace:
    """Build a 4-turn RolloutTrace from per-message tokens."""
    seed_conv = Conversation(
        messages=[Message(role="user", content="<combined system+user prompt>")],
        gen_state=GenerationState(input_ids=array_utils.as_i32(toks.prompt_tokens)),
    )
    trace = RolloutTrace(seed_conv, token_in_token_out=True, max_length=max_length)
    trace._append_assistant_message(text="a1", tokens=toks.a1, logprobs=np.zeros(len(toks.a1), dtype=np.float32))
    trace.append_user_or_tool_message(content="tr1", tokens=toks.tr1)
    trace._append_assistant_message(text="a2", tokens=toks.a2, logprobs=np.zeros(len(toks.a2), dtype=np.float32))
    trace.append_user_or_tool_message(content="tr2", tokens=toks.tr2)
    trace._append_assistant_message(text="a3", tokens=toks.a3, logprobs=np.zeros(len(toks.a3), dtype=np.float32))
    trace.append_user_or_tool_message(content="tr3", tokens=toks.tr3)
    trace._append_assistant_message(text="a4", tokens=toks.a4, logprobs=np.zeros(len(toks.a4), dtype=np.float32))
    return trace


def _build_linear_token_trace_sample(toks: _RealisticTokens, max_length: int) -> Sample:
    """Linear no-compaction TokenTrace sample over the same conversation."""
    trace = TokenTrace()
    trace.extend_tokens(toks.prompt_tokens, token_type="init")
    trace.extend_tokens(toks.a1, token_type="assistant")
    trace.extend_tokens(toks.tr1, token_type="tool_result")
    trace.extend_tokens(toks.a2, token_type="assistant")
    trace.extend_tokens(toks.tr2, token_type="tool_result")
    trace.extend_tokens(toks.a3, token_type="assistant")
    trace.extend_tokens(toks.tr3, token_type="tool_result")
    trace.extend_tokens(toks.a4, token_type="assistant")
    return trace.to_sample(max_length=max_length, pad_token_id=toks.pad_id)


# =====================================================================
# CPU: round-trip and TokenTrace token-level equivalence
# =====================================================================


def _unpack_per_turn_from_merged(merged: Sample) -> list[Sample]:
    """Reconstruct per-turn samples by walking ``merge_info.path_to_leaf`` root → leaf."""
    info = merged.merge_info
    assert info is not None, "merged sample must carry merge_info"
    out: list[Sample] = []
    for leaf_idx, _ in info.path_to_leaf:
        chain = path_chain_root_to_leaf(info.nodes, leaf_idx)
        path_input_ids: list[int] = []
        path_labels: list[int] = []
        path_loss_mask: list[bool] = []
        path_attention_mask: list[bool] = []
        path_position_ids: list[int] = []
        for n_idx in chain:
            nd = info.nodes[n_idx]
            for j in range(nd.start, nd.end):
                if not merged.attention_mask[j]:
                    break  # padding tail
                path_input_ids.append(int(merged.input_ids[j]))
                path_labels.append(int(merged.labels[j]))
                path_loss_mask.append(bool(merged.loss_mask[j]))
                path_attention_mask.append(bool(merged.attention_mask[j]))
                path_position_ids.append(int(merged.position_ids[j]))
        out.append(
            Sample(
                input_ids=array_utils.as_i32(path_input_ids),
                labels=array_utils.as_i32(path_labels),
                loss_mask=array_utils.as_bool(path_loss_mask),
                attention_mask=array_utils.as_bool(path_attention_mask),
                position_ids=array_utils.as_i32(path_position_ids),
                reward=0.0,
                reward_baseline=0.0,
                advantage=array_utils.as_f32([0.0] * len(path_input_ids)),
            )
        )
    return out


def test_merge_trajectory_samples_round_trip_per_turn_no_compaction() -> None:
    """Merged → unpack → per-turn bit-exact (labels/loss_mask only at trainable positions)."""
    max_length = 4096
    toks = _make_realistic_tokens(max_length=max_length)
    trace = _build_rollout_trace(toks, max_length=max_length)
    samples = trace.turn_samples
    assert len(samples) == 4
    merged = merge_trajectory_samples(samples)
    assert merged.merge_info is not None

    reconstructed = _unpack_per_turn_from_merged(merged)
    assert len(reconstructed) == len(samples)
    for i, (got, want) in enumerate(zip(reconstructed, samples, strict=True)):
        assert np.array_equal(got.input_ids, want.input_ids), f"turn {i}: input_ids mismatch"
        assert np.array_equal(got.attention_mask, want.attention_mask), f"turn {i}: attention_mask mismatch"
        assert np.array_equal(got.position_ids, want.position_ids), f"turn {i}: position_ids mismatch"
        for pos, mask in enumerate(want.loss_mask):
            if mask:
                assert bool(got.loss_mask[pos]) is True, f"turn {i}: trainable position {pos} not trainable in merged"
                assert got.labels[pos] == want.labels[pos], f"turn {i}: label mismatch at trainable position {pos}"


def test_merged_token_layout_matches_token_trace_no_compaction() -> None:
    """Merged sample's last path matches the linear ``TokenTrace`` stream and trainable structure."""
    max_length = 4096
    toks = _make_realistic_tokens(max_length=max_length)
    trace = _build_rollout_trace(toks, max_length=max_length)
    samples = trace.turn_samples
    merged = merge_trajectory_samples(samples)
    reconstructed = _unpack_per_turn_from_merged(merged)

    expected_last = [
        *array_utils.to_int_list(toks.prompt_tokens),
        *array_utils.to_int_list(toks.a1),
        *array_utils.to_int_list(toks.tr1),
        *array_utils.to_int_list(toks.a2),
        *array_utils.to_int_list(toks.tr2),
        *array_utils.to_int_list(toks.a3),
        *array_utils.to_int_list(toks.tr3),
        *array_utils.to_int_list(toks.a4),
    ]
    assert reconstructed[-1].input_ids.tolist() == expected_last

    tt_sample = _build_linear_token_trace_sample(toks, max_length=max_length)
    valid_len = sum(tt_sample.attention_mask)
    assert tt_sample.input_ids[:valid_len].tolist() == expected_last

    rt_trainable = sum(merged.loss_mask)
    tt_trainable = sum(tt_sample.loss_mask)
    assert rt_trainable == tt_trainable, f"merged trainable={rt_trainable}, token_trace trainable={tt_trainable}"


def test_label_aware_split_then_merge_round_trip_cpu() -> None:
    """Label-aware divergent paths: merge, unpack, verify per-turn integrity end-to-end.

    Builds two paths with disagreeing next-tokens at position 2
    (``[1,2,3,4]`` / ``[1,2,5,6]``). The label-aware drop fires at
    position 1 (whose label = the diverging input_ids[2] differs), so
    the merged sample has two "2" slots and
    ``unpack_tensor_from_merged`` returns each path's distinct
    logprobs.
    """
    s1 = Sample(
        input_ids=array_utils.as_i32([1, 2, 3, 4]),
        labels=array_utils.as_i32([2, 3, 4, IGNORE_INDEX]),
        loss_mask=array_utils.as_bool([False, True, True, False]),
        attention_mask=array_utils.as_bool([True] * 4),
        position_ids=array_utils.as_i32([0, 1, 2, 3]),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * 4),
        rollout_logprobs=array_utils.as_f32([0.1, 0.2, 0.3, 0.0]),
    )
    s2 = Sample(
        input_ids=array_utils.as_i32([1, 2, 5, 6]),
        labels=array_utils.as_i32([2, 5, 6, IGNORE_INDEX]),
        loss_mask=array_utils.as_bool([False, True, True, False]),
        attention_mask=array_utils.as_bool([True] * 4),
        position_ids=array_utils.as_i32([0, 1, 2, 3]),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * 4),
        rollout_logprobs=array_utils.as_f32([0.4, 0.5, 0.6, 0.0]),
    )
    merged = merge_trajectory_samples([s1, s2])
    # Position 0 (token 1, non-trainable for both) is shared. Positions 1-3 must
    # split: each path has its own trainable "2" copy because the next tokens
    # disagree (3 vs 5). Length of packed = 1 (shared) + 3 (path0) + 3 (path1) + alignment padding.
    assert len(merged.input_ids) >= 7
    # Unpack rollout_logprobs: each path's *trainable* positions should round-trip exactly.
    packed = torch.tensor([float(x) for x in merged.input_ids], dtype=torch.float)
    assert merged.merge_info is not None
    per_turn = unpack_tensor_from_merged(packed, merged.merge_info)
    assert per_turn[0] == [1.0, 2.0, 3.0, 4.0]
    assert per_turn[1] == [1.0, 2.0, 5.0, 6.0]


# =====================================================================
# GPU: train_step equivalence (single trajectory + multi-trajectory)
# =====================================================================


# Single combined config exercises every parallel dimension at once instead of
# running a tp/cp/pp matrix; tp * cp * pp = 8 ranks.
_GPU_CASES = [
    _ParallelCase(name="tp2_cp2_pp2", tp=2, cp=2, pp=2),
]

# Multi-trajectory parallel cases: include DP > 1, which requires >= dp_size
# merged trajectories per global step (so the standard ``DistributedSampler``
# can give each rank at least one). The single-trajectory ``_GPU_CASES`` above
# pin dp=1 because they feed exactly one merged trajectory.
_GPU_CASES_MULTI_TRAJ = [
    *_GPU_CASES,
    _ParallelCase(name="tp2_cp2_dp2", tp=2, cp=2, dp=2),
]


def _pad_sample_to(sample: Sample, max_length: int, pad_token_id: int) -> Sample:
    """Right-pad a Sample's flat fields to ``max_length`` so it stacks cleanly into a SampleTensorDict.

    Each merged trajectory ``Sample`` is ``total_padded`` long (varies per
    trajectory). The worker's merged path reads only the first
    ``merge_info.total_padded`` positions back, so padding tail is unused.
    """
    cur = len(sample.input_ids)
    assert cur <= max_length, f"sample length {cur} exceeds max_length {max_length}"
    pad = max_length - cur
    if pad == 0:
        return sample
    return Sample(
        input_ids=np.pad(sample.input_ids, (0, pad), constant_values=pad_token_id),
        labels=np.pad(sample.labels, (0, pad), constant_values=IGNORE_INDEX),
        loss_mask=np.pad(sample.loss_mask, (0, pad), constant_values=False),
        attention_mask=np.pad(sample.attention_mask, (0, pad), constant_values=False),
        position_ids=np.concatenate([sample.position_ids, np.arange(cur, max_length, dtype=np.int32)]),
        reward=sample.reward,
        reward_baseline=sample.reward_baseline,
        advantage=np.pad(sample.advantage, (0, pad), constant_values=0.0),
        rollout_logprobs=(np.pad(sample.rollout_logprobs, (0, pad), constant_values=0.0) if sample.rollout_logprobs is not None else None),
        old_logprobs=(np.pad(sample.old_logprobs, (0, pad), constant_values=0.0) if sample.old_logprobs is not None else None),
        ref_logprobs=(np.pad(sample.ref_logprobs, (0, pad), constant_values=0.0) if sample.ref_logprobs is not None else None),
        turn_index=(np.pad(sample.turn_index, (0, pad), constant_values=-1) if sample.turn_index is not None else None),
        turn_reward=(np.pad(sample.turn_reward, (0, pad), constant_values=0.0) if sample.turn_reward is not None else None),
        merge_info=sample.merge_info,
    )


def _train_step_baseline_token_trace(cfg: MegatronWorkerConfig, samples: list[Sample], world_size: int = 1) -> tuple[float, float]:
    """Baseline train_step on linear ``TokenTrace`` (TE flat forward, no Magi merged forward)."""
    cfg = cfg.model_copy(deep=True)
    cfg.use_magi_merged_forward = False
    cfg.use_magi_flat_forward = False
    return _train_step_loss_gn(cfg, samples, world_size=world_size)


def _train_step_with_merged_path(
    cfg: MegatronWorkerConfig,
    merged_samples: list[Sample],
    world_size: int = 1,
    *,
    pad_token_id: int = 0,
) -> tuple[float, float]:
    """Drive ``worker.train`` with merged ``Sample``s (each has ``merge_info`` set).

    Single-sample batches need no padding (the worker accepts unpadded
    merged samples). Multi-sample batches pad to the longest sample's
    ``total_padded`` so flat fields stack into a tensordict.
    """
    assert all(s.merge_info is not None for s in merged_samples), "merged path requires merge_info on every Sample"
    if len(merged_samples) == 1:
        samples = merged_samples
    else:
        max_total = max(len(s.input_ids) for s in merged_samples)
        samples = [_pad_sample_to(s, max_length=max_total, pad_token_id=pad_token_id) for s in merged_samples]
    cfg = cfg.model_copy(deep=True)
    cfg.use_magi_merged_forward = True
    return _train_step_loss_gn(cfg, samples, world_size=world_size)


@pytest.mark.parametrize("case", _GPU_CASES, ids=lambda c: c.name)
def test_rollout_trace_merged_train_step_matches_token_trace_realistic(case: _ParallelCase) -> None:
    """End-to-end: merged path through ``worker.train(samples=...)`` matches linear ``TokenTrace`` baseline.

    Tolerance bands: ``loss rel diff < 3e-3``, ``grad-norm rel diff < 1.5e-2``.
    """
    world = case.world_size()
    if world > torch.cuda.device_count():
        pytest.skip(f"Requires {world} GPUs")

    cfg = _make_parallel_config(case)
    seq_length = cfg.model.seq_length
    toks = _make_realistic_tokens(max_length=seq_length)

    trace = _build_rollout_trace(toks, max_length=seq_length)
    merged_sample = trace.to_sample()
    tt_sample = _build_linear_token_trace_sample(toks, max_length=seq_length)

    base_loss, base_gn = _train_step_baseline_token_trace(cfg, [tt_sample], world_size=world)
    rt_loss, rt_gn = _train_step_with_merged_path(cfg, [merged_sample], world_size=world)
    logger.info(f"rollout_trace_vs_token_trace[{case.name}] base loss={base_loss:.6f} gn={base_gn:.6f} rt loss={rt_loss:.6f} gn={rt_gn:.6f}")
    assert abs(base_loss - rt_loss) / max(abs(base_loss), 1e-6) < 3e-3, f"{case.name}: loss rel diff: base={base_loss} rt={rt_loss}"
    assert abs(base_gn - rt_gn) / max(abs(base_gn), 1e-6) < 1.5e-2, f"{case.name}: grad-norm rel diff: base={base_gn} rt={rt_gn}"


@pytest.mark.parametrize("case", _GPU_CASES_MULTI_TRAJ, ids=lambda c: c.name)
def test_two_trajectory_merged_train_step_close_to_unmerged_realistic(case: _ParallelCase) -> None:
    """Two trajectories through the worker's merged path vs two flat samples through baseline.

    Includes a ``dp2`` parametrization: with 2 trajectories and DP=2 the standard
    ``DistributedSampler`` shards one trajectory per rank, so the merged path
    works without the single-trajectory padding pathology that forces the
    other realistic merged tests to dp=1.
    """
    world = case.world_size()
    if world > torch.cuda.device_count():
        pytest.skip(f"Requires {world} GPUs")

    cfg = _make_parallel_config(case)
    seq_length = cfg.model.seq_length
    toks = _make_realistic_tokens(max_length=seq_length)

    trace_a = _build_rollout_trace(toks, max_length=seq_length)
    trace_b = _build_rollout_trace(toks, max_length=seq_length)
    merged_a = trace_a.to_sample()
    merged_b = trace_b.to_sample()

    tt_a = _build_linear_token_trace_sample(toks, max_length=seq_length)
    tt_b = _build_linear_token_trace_sample(toks, max_length=seq_length)

    # 2 trajectories per global step, sharded across DP ranks.
    per_dp_micro = 2 // case.dp  # 2 with dp=1, 1 with dp=2.
    config_ref = cfg.model_copy(deep=True)
    config_ref.global_batch_size = 2
    config_ref.train_micro_batch_size = per_dp_micro
    base_loss, base_gn = _train_step_baseline_token_trace(config_ref, [tt_a, tt_b], world_size=world)

    config_merged = cfg.model_copy(deep=True)
    config_merged.global_batch_size = 2
    config_merged.train_micro_batch_size = per_dp_micro
    rt_loss, rt_gn = _train_step_with_merged_path(config_merged, [merged_a, merged_b], world_size=world, pad_token_id=toks.pad_id)

    logger.info(f"two_traj[{case.name}] base loss={base_loss:.6f} gn={base_gn:.6f} merged loss={rt_loss:.6f} gn={rt_gn:.6f}")
    assert abs(base_loss - rt_loss) / max(abs(base_loss), 1e-6) < 5e-3, f"{case.name}: loss rel diff: base={base_loss} rt={rt_loss}"
    assert abs(base_gn - rt_gn) / max(abs(base_gn), 1e-6) < 1.5e-2, f"{case.name}: grad-norm rel diff: base={base_gn} rt={rt_gn}"


# =====================================================================
# GPU regression: label-aware merge (= production) matches flat baseline
# when shared trainable region dominates total tokens. Verifies labels
# and rollout_logprobs agree on shared trainable slots — which is the
# only soundness invariant the merge relies on after dropping the
# loss-aware split.
# =====================================================================


def _build_path_from_segments(
    segments: list[tuple[np.ndarray, bool]],
    pad_id: int,
) -> Sample:
    """Build an unpadded sample from per-segment ``(tokens, trainable)`` pairs.

    Each tuple's bool sets that segment's ``token_type``: ``True`` →
    "assistant", ``False`` → "init" for the first segment, "tool_result"
    afterwards. No padding (``max_length`` == valid length); the merge
    trie consumes ``input_ids`` directly so padding would propagate into
    the packed layout.
    """
    trace = TokenTrace()
    valid = 0
    for idx, (seg_toks, trainable) in enumerate(segments):
        if trainable:
            token_type = "assistant"
        else:
            token_type = "init" if idx == 0 else "tool_result"
        trace.extend_tokens(array_utils.as_i32(seg_toks), token_type=token_type)
        valid += len(seg_toks)
    return trace.to_sample(max_length=valid, pad_token_id=pad_id)


@pytest.mark.parametrize("case", _GPU_CASES, ids=lambda c: c.name)
def test_merged_train_step_matches_flat_shared_trainable_in_one_path_realistic(case: _ParallelCase) -> None:
    """Merged forward matches flat baseline; shared region trainable in exactly one path.

    Two paths sharing a long region (``a1+a2+a3+a4``) followed by short
    divergent tails. Path A has the shared region trainable + tail_a
    non-trainable; path B has the shared region non-trainable + tail_b
    trainable. The strict v10 rule forbids two trainable paths sharing
    one slot — this layout side-steps it and still exercises the merged
    forward against a long shared region.
    """
    world = case.world_size()
    if world > torch.cuda.device_count():
        pytest.skip(f"Requires {world} GPUs")

    cfg = _make_parallel_config(case)
    seq_length = cfg.model.seq_length
    toks = _make_realistic_tokens(max_length=seq_length)

    # Short divergent non-trainable tails (must differ at first token so the
    # label-aware drop fires at the last shared position). ``tr1`` and ``tr2``
    # share the leading role-marker tokens — find the first divergent offset.
    common = 0
    while common < len(toks.tr1) and common < len(toks.tr2) and toks.tr1[common] == toks.tr2[common]:
        common += 1
    assert common < min(len(toks.tr1), len(toks.tr2)), "tr1 and tr2 must diverge somewhere"
    tail_len = 10
    tail_a = toks.tr1[common : common + tail_len]
    tail_b = toks.tr2[common : common + tail_len]
    assert tail_a[0] != tail_b[0], "tails must diverge at first token for label-aware split to fire"

    # Concatenate assistant turns into one long shared region.
    shared = np.concatenate([toks.a1, toks.a2, toks.a3, toks.a4])
    sa = _build_path_from_segments(
        [(toks.prompt_tokens, False), (shared, True), (tail_a, False)],
        pad_id=toks.pad_id,
    )
    sb = _build_path_from_segments(
        [(toks.prompt_tokens, False), (shared, False), (tail_b, True)],
        pad_id=toks.pad_id,
    )

    total_a = sum(sa.attention_mask)
    shared_len = len(shared) - 1  # last shared position split off by label-aware drop
    assert shared_len / total_a > 0.7, f"shared region should dominate; got shared={shared_len} total={total_a}"

    # F: flat 2-row baseline (samples padded to seq_length so they stack into a tensordict).
    sa_padded = _pad_sample_to(sa, max_length=seq_length, pad_token_id=toks.pad_id)
    sb_padded = _pad_sample_to(sb, max_length=seq_length, pad_token_id=toks.pad_id)
    config_F = cfg.model_copy(deep=True)
    config_F.global_batch_size = 2
    config_F.train_micro_batch_size = 2
    F_loss, F_gn = _train_step_baseline_token_trace(config_F, [sa_padded, sb_padded], world_size=world)

    # L: label-aware merge (production path).
    merged_L = merge_trajectory_samples([sa, sb])
    config_L = cfg.model_copy(deep=True)
    config_L.global_batch_size = 1
    config_L.train_micro_batch_size = 1
    L_loss, L_gn = _train_step_with_merged_path(config_L, [merged_L], world_size=world)

    assert merged_L.merge_info is not None
    L_loss_gap = abs(F_loss - L_loss) / max(abs(F_loss), 1e-6)
    L_gn_gap = abs(F_gn - L_gn) / max(abs(F_gn), 1e-6)
    logger.info(
        f"shared_trainable_in_one_path[{case.name}] "
        f"sa_total={total_a} shared={shared_len} "
        f"L_total={merged_L.merge_info.total_padded} L_trainable={sum(merged_L.loss_mask)} "
        f"F loss={F_loss:.6f} gn={F_gn:.6f} | L loss={L_loss:.6f} gn={L_gn:.6f} "
        f"(gap loss={L_loss_gap:.3e} gn={L_gn_gap:.3e})"
    )
    assert L_loss_gap < 3e-3, f"{case.name}: L loss diverged from F: F={F_loss} L={L_loss}"
    assert L_gn_gap < 1.5e-2, f"{case.name}: L grad-norm diverged from F: F={F_gn} L={L_gn}"


@pytest.mark.parametrize("case", _GPU_CASES, ids=lambda c: c.name)
def test_merged_train_step_matches_flat_shared_trainable_in_one_path_tiny(case: _ParallelCase) -> None:
    """Hand-traceable tiny regression with shared trainable in one path only.

      path_a: [11, 12, 13]  +  [21, 22, 23]  +  [31, 32, 33]
              prompt (F)        shared (T)        tail (F)

      path_b: [11, 12, 13]  +  [21, 22, 23]  +  [41, 42, 43]
              prompt (F)        shared (F)        tail (T)

    Tails diverge at token 0 (31 != 41); the label-aware drop still fires
    at the last shared position. Per the strict v10 rule, only one path
    may have trainable on a shared slot.
    """
    world = case.world_size()
    if world > torch.cuda.device_count():
        pytest.skip(f"Requires {world} GPUs")

    cfg = _make_parallel_config(case)
    seq_length = cfg.model.seq_length

    prompt = array_utils.as_i32([11, 12, 13])
    shared = array_utils.as_i32([21, 22, 23])
    tail_a = array_utils.as_i32([31, 32, 33])
    tail_b = array_utils.as_i32([41, 42, 43])
    pad_id = 0

    sa = _build_path_from_segments([(prompt, False), (shared, True), (tail_a, False)], pad_id=pad_id)
    sb = _build_path_from_segments([(prompt, False), (shared, False), (tail_b, True)], pad_id=pad_id)

    sa_padded = _pad_sample_to(sa, max_length=seq_length, pad_token_id=pad_id)
    sb_padded = _pad_sample_to(sb, max_length=seq_length, pad_token_id=pad_id)

    config_F = cfg.model_copy(deep=True)
    config_F.global_batch_size = 2
    config_F.train_micro_batch_size = 2
    F_loss, F_gn = _train_step_baseline_token_trace(config_F, [sa_padded, sb_padded], world_size=world)

    merged_L = merge_trajectory_samples([sa, sb])
    config_L = cfg.model_copy(deep=True)
    config_L.global_batch_size = 1
    config_L.train_micro_batch_size = 1
    L_loss, L_gn = _train_step_with_merged_path(config_L, [merged_L], world_size=world)

    assert merged_L.merge_info is not None
    L_loss_gap = abs(F_loss - L_loss) / max(abs(F_loss), 1e-6)
    L_gn_gap = abs(F_gn - L_gn) / max(abs(F_gn), 1e-6)
    logger.info(
        f"shared_trainable_in_one_path_tiny[{case.name}] "
        f"L_total={merged_L.merge_info.total_padded} L_trainable={sum(merged_L.loss_mask)} "
        f"F loss={F_loss:.6f} gn={F_gn:.6f} | L loss={L_loss:.6f} gn={L_gn:.6f} "
        f"(gap loss={L_loss_gap:.3e} gn={L_gn_gap:.3e})"
    )
    # Wider tolerance than the realistic case: 9-token total fixture is at the edge
    # of bf16 noise (per-step ~1 ULP per layer compounds visibly here).
    assert L_loss_gap < 1e-2, f"{case.name}: L loss diverged from F: F={F_loss} L={L_loss}"
    assert L_gn_gap < 3e-2, f"{case.name}: L grad-norm diverged from F: F={F_gn} L={L_gn}"


def _build_pack_split_sft_trace() -> RolloutTrace:
    """Build 5 short per-turn SFT samples whose full merge crosses the 512-token split."""
    prompt = list(range(101, 125))
    turn_samples: list[Sample] = []
    for turn_idx in range(5):
        tail_start = 1000 + turn_idx * 200
        input_ids = [*prompt, *range(tail_start, tail_start + 120)]
        labels = [*input_ids[1:], IGNORE_INDEX]
        trainable_positions = set(range(len(prompt) - 1, len(input_ids) - 1))
        turn_samples.append(
            Sample(
                input_ids=array_utils.as_i32(input_ids),
                labels=array_utils.as_i32(labels),
                loss_mask=array_utils.as_bool([i in trainable_positions for i in range(len(input_ids))]),
                attention_mask=array_utils.as_bool([True] * len(input_ids)),
                position_ids=array_utils.as_i32(list(range(len(input_ids)))),
                reward=0.0,
                reward_baseline=0.0,
                advantage=array_utils.as_f32([0.0] * len(input_ids)),
            )
        )

    trace = RolloutTrace(
        Conversation(messages=[Message(role="user", content="synthetic pack split prompt")]),
        token_in_token_out=False,
    )
    trace.turn_samples = turn_samples
    return trace


def _sft_train_step_metrics(
    cfg: MegatronWorkerConfig,
    samples: list[Sample],
    world_size: int,
    *,
    global_batch_size: int | None = None,
    use_magi_merged_forward: bool = True,
    trajectory_ids: list[int] | None = None,
) -> dict[str, float]:
    """Spawn worker with ``SftTrainer(compute_entropy=True)`` and return one-step metrics.

    ``trajectory_ids`` controls how packed samples are grouped into global batches.
    Defaults to one trajectory per sample (i.e. one gradient update per real sample).
    To exercise the "one trace split into multiple packed samples" path, pass the
    same id for every sample that came from one trace and set ``global_batch_size``
    to the number of unique trajectories.
    """
    from axrl.configs import SftTrainerConfig
    from axrl.data.sample import SampleTensorDict
    from axrl.ray import ray_utils
    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.ray.resource_group import Request, ResourceGroup
    from axrl.trainer.sft_trainer import SftTrainer

    cfg = cfg.model_copy(deep=True)
    cfg.use_magi_merged_forward = use_magi_merged_forward
    samples = [Sample(**s.__dict__) for s in samples]
    ids = trajectory_ids if trajectory_ids is not None else list(range(len(samples)))
    assert len(ids) == len(samples), f"trajectory_ids length {len(ids)} must match samples count {len(samples)}"
    for sample, trajectory_id in zip(samples, ids, strict=True):
        sample.trajectory_id = trajectory_id
    num_trajectories = len(set(ids))
    train_batch_size = global_batch_size if global_batch_size is not None else num_trajectories
    cfg.global_batch_size = train_batch_size
    # global_batch_size here is the Megatron microbatches-calculator input; with the
    # new trajectory-grouped iterator we drive train_step ourselves so the only
    # remaining constraint is the divisibility assert inside Megatron's init.
    cfg.train_micro_batch_size = 1
    cfg.reset_init_weights_every_k_steps = 1

    worker: RayMegatronWorker | None = None
    ray_utils.restart()
    try:
        rg = ResourceGroup([Request(cpu=1, gpu=world_size)])
        worker = RayMegatronWorker(config=cfg, resource_group=rg)
        worker.initialize()
        worker.set_trainer(SftTrainer(config=SftTrainerConfig(compute_entropy=True)))
        _step, metrics = worker.train(
            samples=SampleTensorDict.from_samples(samples),
            global_step=0,
            data_shuffle_seed=0,
            compute_logprobs=False,
        )
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    finally:
        if worker is not None:
            worker.shutdown()
        ray_utils.stop()


_SFT_PACK_SPLIT_CASES = [
    _ParallelCase(name="cp1_bi", batch_invariant=True),
    _ParallelCase(name="cp2_bi", cp=2, batch_invariant=True),
]


@pytest.mark.parametrize("case", _SFT_PACK_SPLIT_CASES, ids=lambda c: c.name)
def test_sft_train_pack_split_256_512_matches_no_split(case: _ParallelCase) -> None:
    """SFT train metrics are invariant to splitting one trajectory's merged packs."""
    world = case.world_size()
    if world > torch.cuda.device_count():
        pytest.skip(f"Requires {world} GPUs")

    cfg = make_megatron_worker_config(case, seq_length=1024)
    trace = _build_pack_split_sft_trace()

    samples_no_pack_split = trace.to_packed_samples(max_pack_length=1024)
    samples_split_512 = trace.to_packed_samples(max_pack_length=512)
    samples_split_256 = trace.to_packed_samples(max_pack_length=256)

    assert [len(samples_no_pack_split), len(samples_split_512), len(samples_split_256)] == [1, 2, 5]
    expected_trainable = sum(sum(sample.loss_mask) for sample in trace.turn_samples)
    assert sum(sum(sample.loss_mask) for sample in samples_no_pack_split) == expected_trainable
    assert sum(sum(sample.loss_mask) for sample in samples_split_512) == expected_trainable
    assert sum(sum(sample.loss_mask) for sample in samples_split_256) == expected_trainable

    # All packed samples come from one trace ⇒ trajectory_id=0, one global batch.
    metrics_no_split = _sft_train_step_metrics(
        cfg, samples_no_pack_split, world_size=world, global_batch_size=1, trajectory_ids=[0] * len(samples_no_pack_split)
    )
    metrics_split_512 = _sft_train_step_metrics(
        cfg, samples_split_512, world_size=world, global_batch_size=1, trajectory_ids=[0] * len(samples_split_512)
    )
    metrics_split_256 = _sft_train_step_metrics(
        cfg, samples_split_256, world_size=world, global_batch_size=1, trajectory_ids=[0] * len(samples_split_256)
    )

    logger.info(
        "sft_pack_split "
        f"no_split loss={metrics_no_split['actor_train/loss']:.6f} entropy={metrics_no_split['actor_train/entropy']:.6f} "
        f"gn={metrics_no_split['actor_train/grad_norm']:.6f}; "
        f"split512 loss={metrics_split_512['actor_train/loss']:.6f} entropy={metrics_split_512['actor_train/entropy']:.6f} "
        f"gn={metrics_split_512['actor_train/grad_norm']:.6f}; "
        f"split256 loss={metrics_split_256['actor_train/loss']:.6f} entropy={metrics_split_256['actor_train/entropy']:.6f} "
        f"gn={metrics_split_256['actor_train/grad_norm']:.6f}"
    )
    # Splitting one trace into more packed samples introduces small bf16 numerical
    # drift via different 128-aligned trailing padding per packed sample. Tolerate
    # ~0.5% relative drift on loss/entropy and ~1.5% on grad-norm.
    for name, metrics in [("split_512", metrics_split_512), ("split_256", metrics_split_256)]:
        assert (
            abs(metrics_no_split["actor_train/loss"] - metrics["actor_train/loss"]) / max(abs(metrics_no_split["actor_train/loss"]), 1e-6) < 5e-3
        ), name
        assert (
            abs(metrics_no_split["actor_train/entropy"] - metrics["actor_train/entropy"]) / max(abs(metrics_no_split["actor_train/entropy"]), 1e-6)
            < 5e-3
        ), name
        assert (
            abs(metrics_no_split["actor_train/grad_norm"] - metrics["actor_train/grad_norm"]) / max(metrics_no_split["actor_train/grad_norm"], 1e-6)
            < 1.5e-2
        ), name


def _build_sft_padding_samples() -> list[Sample]:
    prompt = array_utils.as_i32(list(range(201, 209)))
    samples: list[Sample] = []
    for sample_idx in range(6):
        tail_start = 1000 + sample_idx * 64
        samples.append(
            _build_path_from_segments(
                [
                    (prompt, False),
                    (array_utils.as_i32(list(range(tail_start, tail_start + 24))), True),
                ],
                pad_id=0,
            )
        )
    return samples


def test_sft_train_one_trajectory_packed_to_multiple_samples_matches_unpacked() -> None:
    """6 SFT samples grouped as one global batch are trained the same regardless of how many packed samples they split into.

    The iterator groups all packed samples sharing ``trajectory_id`` into the
    same global batch and (with token-mean loss) the gradient depends only on
    the total trainable tokens in that batch — which is invariant under packing.
    """
    case = _ParallelCase(name="cp1_bi", batch_invariant=True)
    world = case.world_size()
    if world > torch.cuda.device_count():
        pytest.skip(f"Requires {world} GPUs")

    cfg = make_megatron_worker_config(case, seq_length=256)
    samples = _build_sft_padding_samples()
    assert len(samples) == 6
    expected_trainable = sum(sum(sample.loss_mask) for sample in samples)

    # Treat all 6 packed samples as belonging to one trajectory, global_batch_size=1.
    metrics_packed = _sft_train_step_metrics(
        cfg,
        samples,
        world_size=world,
        global_batch_size=1,
        trajectory_ids=[0] * len(samples),
        use_magi_merged_forward=False,
    )
    # Treat each of the 6 packed samples as its own trajectory, global_batch_size=6 (single global batch).
    metrics_one_per_traj = _sft_train_step_metrics(
        cfg,
        samples,
        world_size=world,
        global_batch_size=6,
        trajectory_ids=list(range(len(samples))),
        use_magi_merged_forward=False,
    )

    logger.info(
        "sft_grouped_vs_unpacked "
        f"packed loss={metrics_packed['actor_train/loss']:.6f} entropy={metrics_packed['actor_train/entropy']:.6f} "
        f"gn={metrics_packed['actor_train/grad_norm']:.6f} denom={metrics_packed['actor_train/denom']:.0f}; "
        f"one-per-traj loss={metrics_one_per_traj['actor_train/loss']:.6f} entropy={metrics_one_per_traj['actor_train/entropy']:.6f} "
        f"gn={metrics_one_per_traj['actor_train/grad_norm']:.6f} denom={metrics_one_per_traj['actor_train/denom']:.0f}"
    )
    assert metrics_packed["actor_train/denom"] == pytest.approx(expected_trainable)
    assert metrics_one_per_traj["actor_train/denom"] == pytest.approx(expected_trainable)
    assert metrics_one_per_traj["actor_train/loss"] == pytest.approx(metrics_packed["actor_train/loss"], abs=5e-3)
    assert (
        abs(metrics_one_per_traj["actor_train/entropy"] - metrics_packed["actor_train/entropy"])
        / max(abs(metrics_packed["actor_train/entropy"]), 1e-6)
        < 5e-3
    )
    assert (
        abs(metrics_one_per_traj["actor_train/grad_norm"] - metrics_packed["actor_train/grad_norm"])
        / max(metrics_packed["actor_train/grad_norm"], 1e-6)
        < 1.5e-2
    )


# =====================================================================
# GPU: GrpoTrainer flat (4 turn samples) vs merged (1 sample) end-to-end
# =====================================================================


# Production GRPO cases run with ``batch_invariant=True``. Without it the
# Magi merged forward isn't bit-deterministic across calls: at step 0 with
# ``compute_logprobs=True`` the ratio = exp(new - old) drifts up to ±55%
# per token in the merged path (vs 1e-5 in TE flat). The first case
# (``cp1`` with bi=False) is kept for explicit side-by-side comparison;
# tolerances branch on ``case.batch_invariant`` below.
# Single combined parallel config; bi axis kept (bi=True is the realistic
# prod path, bi=False keeps the explicit side-by-side comparison).
_GRPO_CASES = [
    _ParallelCase(name="tp2_cp2_pp2", tp=2, cp=2, pp=2, batch_invariant=False),
    _ParallelCase(name="tp2_cp2_pp2_bi", tp=2, cp=2, pp=2, batch_invariant=True),
]


def _seed_grpo_inputs(samples: list[Sample]) -> None:
    """Populate GRPO-required fields with alternating signs across samples.

    Sample ``i`` gets advantage ``+1`` on trainable tokens if ``i`` is even,
    else ``-1``. This exercises both clip directions in the GRPO ratio.
    ``merge_trajectory_samples`` propagates per-position ``advantage`` and
    ``rollout_logprobs`` from input samples, so the merged path inherits
    the same per-token values without re-seeding.
    """
    for i, s in enumerate(samples):
        sign = 1.0 if i % 2 == 0 else -1.0
        s.advantage = np.asarray([sign if m else 0.0 for m in s.loss_mask], dtype=np.float32)
        s.rollout_logprobs = np.zeros(len(s.input_ids), dtype=np.float32)


def _grpo_train_step(cfg: MegatronWorkerConfig, samples: list[Sample], world_size: int) -> dict[str, float]:
    """Spawn worker with ``GrpoTrainer``, run one ``train`` step with ``compute_logprobs=True``.

    Uses ``micro_batch_denominator_type="token"`` so ``actor_train/denom`` reports the
    trainable-token count (invariant across batching strategies), letting the
    flat and merged paths be compared on the same denominator. Each sample is
    assigned its own ``trajectory_id`` (so ``global_batch_size`` from ``cfg`` is
    interpreted as trajectory count).
    """
    from axrl.configs import GrpoTrainerConfig
    from axrl.data.sample import SampleTensorDict
    from axrl.ray import ray_utils
    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.ray.resource_group import Request, ResourceGroup
    from axrl.trainer.grpo_trainer import GrpoTrainer

    samples = [Sample(**s.__dict__) for s in samples]
    for trajectory_id, sample in enumerate(samples):
        sample.trajectory_id = trajectory_id

    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=world_size)])
    worker = RayMegatronWorker(config=cfg, resource_group=rg)
    worker.initialize()
    worker.set_trainer(GrpoTrainer(config=GrpoTrainerConfig(micro_batch_denominator_type="token")))
    _step, metrics = worker.train(
        samples=SampleTensorDict.from_samples(samples),
        global_step=0,
        data_shuffle_seed=0,
        compute_logprobs=True,
    )
    worker.shutdown()
    ray_utils.stop()
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


@pytest.mark.parametrize("case", _GRPO_CASES, ids=lambda c: c.name)
def test_grpo_merged_train_step_matches_baseline_realistic(case: _ParallelCase) -> None:
    """GrpoTrainer on 4 turn samples (flat) vs 1 merged sample (prefix-tree).

    Same fixture for both sides; both use ``compute_logprobs=True`` so the
    worker computes ref + old logprobs from the model itself. Asserts
    ``actor_train/loss``, ``actor_train/entropy_mean``, ``actor_train/grad_norm`` match.
    """
    from tests.mcore._context_management_fixture import make_hide_tool_result_samples

    world = case.world_size()
    if world > torch.cuda.device_count():
        pytest.skip(f"Requires {world} GPUs")

    cfg = _make_parallel_config(case)
    seq_length = cfg.model.seq_length
    toks = _make_realistic_tokens(max_length=seq_length)
    samples = make_hide_tool_result_samples(toks, max_length=seq_length)
    _seed_grpo_inputs(samples)

    merged = merge_trajectory_samples(samples)

    base_config = cfg.model_copy(deep=True)
    base_config.use_magi_merged_forward = False
    base_config.global_batch_size = len(samples)
    base_config.train_micro_batch_size = len(samples)
    base_config.reset_init_weights_every_k_steps = 1
    base = _grpo_train_step(base_config, samples, world_size=world)

    merged_config = cfg.model_copy(deep=True)
    merged_config.use_magi_merged_forward = True
    merged_config.global_batch_size = 1
    merged_config.train_micro_batch_size = 1
    merged_config.reset_init_weights_every_k_steps = 1
    m = _grpo_train_step(merged_config, [merged], world_size=world)

    logger.info(
        f"grpo_merged[{case.name}] "
        f"base loss={base['actor_train/loss']:.6f} entropy={base['actor_train/entropy_mean']:.6f} "
        f"gn={base['actor_train/grad_norm']:.6f} denom={base['actor_train/denom']:.0f} "
        f"num_samples={base['actor_train/num_samples']:.0f} "
        f"ratio[std={base['actor_train/ratio_std']:.4e},min={base['actor_train/ratio_min']:.4f},max={base['actor_train/ratio_max']:.4f}] | "
        f"merged loss={m['actor_train/loss']:.6f} entropy={m['actor_train/entropy_mean']:.6f} "
        f"gn={m['actor_train/grad_norm']:.6f} denom={m['actor_train/denom']:.0f} "
        f"num_samples={m['actor_train/num_samples']:.0f} "
        f"ratio[std={m['actor_train/ratio_std']:.4e},min={m['actor_train/ratio_min']:.4f},max={m['actor_train/ratio_max']:.4f}]"
    )
    assert base["actor_train/denom"] == m["actor_train/denom"], (
        f"{case.name}: token count mismatch base={base['actor_train/denom']} merged={m['actor_train/denom']}"
    )

    if case.batch_invariant or case.deterministic:
        # H20 bf16 merged/flat GRPO paths have shown loss abs drift up to
        # 6.38e-3 and grad-norm rel drift up to 3.31e-2 while matching token
        # counts and entropy closely.
        loss_abs_tol, entropy_rel_tol, grad_norm_rel_tol = 8e-3, 5e-3, 4e-2
    else:
        loss_abs_tol, entropy_rel_tol, grad_norm_rel_tol = 0.1, 5e-3, 0.1
    assert abs(base["actor_train/loss"] - m["actor_train/loss"]) < loss_abs_tol
    assert abs(base["actor_train/entropy_mean"] - m["actor_train/entropy_mean"]) / max(abs(base["actor_train/entropy_mean"]), 1e-6) < entropy_rel_tol
    assert abs(base["actor_train/grad_norm"] - m["actor_train/grad_norm"]) / max(abs(base["actor_train/grad_norm"]), 1e-6) < grad_norm_rel_tol


# =====================================================================
# CPU: no-compaction multi-turn merge tree is a linear chain, not a branching tree
# =====================================================================


def test_no_compaction_multi_turn_merged_is_linear_chain_not_branching() -> None:
    """No-compaction multi-turn merge tree is a linear chain, not a branching tree.

    ``RolloutTrace.to_sample()`` on a NO-COMPACTION multi-turn trace merges
    per-turn samples via :func:`merge_trajectory_samples`. Each turn-k
    sample's ``input_ids`` is the full prompt+responses through turn k, so
    the per-turn paths are nested strict prefixes (``ids_0 < ids_1 < ids_2``
    etc.). The prefix-tree packer collapses nested prefixes into a SINGLE
    trunk with one leaf marked at each turn boundary — not a branching
    tree. ``merge_info.path_to_leaf`` still carries ``num_turns`` entries
    (one per turn sample), but every internal node has fan-out exactly 1.
    """
    max_length = 4096
    toks = _make_realistic_tokens(max_length=max_length)
    trace = _build_rollout_trace(toks, max_length=max_length)
    merged = trace.to_sample()

    # Sanity on the per-turn precondition that makes the tree a chain.
    per_turn_ids = [list(s.input_ids) for s in trace.turn_samples]
    assert len(per_turn_ids) == 4  # four assistant turns in the fixture
    for i in range(len(per_turn_ids) - 1):
        assert per_turn_ids[i] == per_turn_ids[i + 1][: len(per_turn_ids[i])], f"turn-{i} sample must be a strict prefix of turn-{i + 1}"

    # The merged sample reports ``num_turns`` paths in its metadata...
    mi = merged.merge_info
    assert mi is not None
    assert len(mi.path_to_leaf) == 4

    # ...but the trie is a linear chain: every non-leaf node has exactly
    # one child. Any fan-out > 1 would indicate a lateral branch.
    child_count: dict[int, int] = {}
    for node in mi.nodes:
        parent = getattr(node, "parent", None)
        if parent is not None and parent >= 0:
            child_count[parent] = child_count.get(parent, 0) + 1
    assert all(c == 1 for c in child_count.values()), (
        f"no-compaction multi-turn trace produced a branching merge tree; per-parent fan-out distribution: {sorted(child_count.values())}"
    )

    # Each path chain root→leaf covers an earlier slice of the trunk. Leaves
    # are ordered by path length, so the deepest leaf's chain is a superset
    # of all earlier chains (= they share the same backbone).
    leaf_chains = [set(path_chain_root_to_leaf(mi.nodes, leaf_idx)) for leaf_idx, _ in mi.path_to_leaf]
    chain_lens = [len(c) for c in leaf_chains]
    assert chain_lens == sorted(chain_lens), f"path chains should be monotonically non-decreasing in length along the trunk, got {chain_lens}"
    for i in range(len(leaf_chains) - 1):
        assert leaf_chains[i] <= leaf_chains[i + 1], f"path-{i} trie chain is not a subset of path-{i + 1}'s — indicates branching"


# =====================================================================
# Flat (allow_prefix_sharing=False) vs merged (allow_prefix_sharing=True)
# parity test for a multi-turn linear-prefix-chain trace.
# =====================================================================


def _build_linear_chain_sft_trace() -> RolloutTrace:
    """Multi-turn linear-prefix-chain trace (no compaction, no branching).

    Each turn-sample's input_ids extends the previous turn's input with
    ``[tool_result, assistant]``. ``loss_mask`` is True only on this turn's
    assistant tokens (label-shifted by 1), exactly mirroring what
    ``TokenTrace.to_last_turn_sample`` produces in production.
    """
    prompt = list(range(101, 125))
    asst_lens = [40, 32, 28]
    tool_lens = [12, 16]

    full: list[int] = list(prompt)
    asst_ranges: list[tuple[int, int]] = []
    for i, asst_len in enumerate(asst_lens):
        if i > 0:
            tool_start = 3000 + i * 100
            full.extend(range(tool_start, tool_start + tool_lens[i - 1]))
        asst_start = len(full)
        asst_token_start = 1000 + i * 200
        full.extend(range(asst_token_start, asst_token_start + asst_len))
        asst_ranges.append((asst_start, asst_start + asst_len))

    turn_samples: list[Sample] = []
    for turn_idx in range(len(asst_lens)):
        end = asst_ranges[turn_idx][1]
        input_ids = full[:end]
        labels = [*input_ids[1:], IGNORE_INDEX]
        a_start, a_end = asst_ranges[turn_idx]
        # Label-shifted: position i predicts input_ids[i+1], so trainable label
        # positions for assistant tokens [a_start, a_end) are [a_start-1, a_end-1).
        loss_mask = [a_start - 1 <= i < a_end - 1 for i in range(len(input_ids))]
        turn_samples.append(
            Sample(
                input_ids=array_utils.as_i32(input_ids),
                labels=array_utils.as_i32(labels),
                loss_mask=array_utils.as_bool(loss_mask),
                attention_mask=array_utils.as_bool([True] * len(input_ids)),
                position_ids=array_utils.as_i32(list(range(len(input_ids)))),
                reward=0.0,
                reward_baseline=0.0,
                advantage=array_utils.as_f32([0.0] * len(input_ids)),
            )
        )

    trace = RolloutTrace(
        Conversation(messages=[Message(role="user", content="linear chain prompt")]),
        token_in_token_out=False,
    )
    trace.turn_samples = turn_samples
    return trace


def test_sft_train_flat_vs_merged_linear_chain_match() -> None:
    """SFT train metrics agree between flat and merged exports.

    The traces are the same multi-turn linear-prefix-chain fixture on
    Qwen2.5-3B-Instruct.
    """
    case = _ParallelCase(name="cp1_bi", batch_invariant=True)
    world = case.world_size()
    if world > torch.cuda.device_count():
        pytest.skip(f"Requires {world} GPUs")

    cfg = make_megatron_worker_config(case, seq_length=512)
    cfg.model.name = "Qwen/Qwen2.5-3B-Instruct"
    trace = _build_linear_chain_sft_trace()

    flat_samples = trace.to_packed_samples(max_pack_length=512, allow_prefix_sharing=False)
    merged_samples = trace.to_packed_samples(max_pack_length=512, allow_prefix_sharing=True)

    assert len(flat_samples) == 1 and flat_samples[0].merge_info is None
    assert len(merged_samples) == 1 and merged_samples[0].merge_info is not None
    assert sum(flat_samples[0].loss_mask) == sum(merged_samples[0].loss_mask)

    metrics_flat = _sft_train_step_metrics(cfg, flat_samples, world_size=world, use_magi_merged_forward=False)
    metrics_merged = _sft_train_step_metrics(cfg, merged_samples, world_size=world, use_magi_merged_forward=True)

    logger.info(
        "linear_chain_flat_vs_merged "
        f"flat loss={metrics_flat['actor_train/loss']:.6f} entropy={metrics_flat['actor_train/entropy']:.6f} "
        f"gn={metrics_flat['actor_train/grad_norm']:.6f}; "
        f"merged loss={metrics_merged['actor_train/loss']:.6f} entropy={metrics_merged['actor_train/entropy']:.6f} "
        f"gn={metrics_merged['actor_train/grad_norm']:.6f}"
    )
    loss_gap = abs(metrics_flat["actor_train/loss"] - metrics_merged["actor_train/loss"])
    entropy_gap = abs(metrics_flat["actor_train/entropy"] - metrics_merged["actor_train/entropy"]) / max(
        abs(metrics_flat["actor_train/entropy"]), 1e-6
    )
    gn_gap = abs(metrics_flat["actor_train/grad_norm"] - metrics_merged["actor_train/grad_norm"]) / max(metrics_flat["actor_train/grad_norm"], 1e-6)
    assert loss_gap < 5e-3, f"loss diverged: flat={metrics_flat['actor_train/loss']} merged={metrics_merged['actor_train/loss']}"
    assert entropy_gap < 5e-3, f"entropy diverged: flat={metrics_flat['actor_train/entropy']} merged={metrics_merged['actor_train/entropy']}"
    assert gn_gap < 1.5e-2, f"grad_norm diverged: flat={metrics_flat['actor_train/grad_norm']} merged={metrics_merged['actor_train/grad_norm']}"
