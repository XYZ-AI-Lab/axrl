"""Pressure test for the magi merged forward.

Runs both inference (``compute_logprobs``, ``inference_only=True``) and
training (``train`` with optimizer CPU-offload, ``inference_only=False``)
on multi-turn trajectories built via ``RolloutTrace`` with hide-tool-results
compaction (``max_recent_tool_results=1``).

Workload:
    - 4 trajectories per global step.
    - turns ∈ {256, 128, 64, 32} (largest first), ~1K tokens/turn.
    - mbs=1, dp=1, recompute=full/uniform/1, bf16.

Cluster:
    - 4 nodes x 8 GPUs = 32 GPUs total. Driver runs on the head node.
    - The launch script (`run_pressure_test.sh`) brings up ray; here we
      ``ray.init(address=...)`` against the existing cluster.

Procedure (per phase):
    1. Iterate the parallel-config matrix at MATRIX_TURNS (256). Take
       the fastest config that fits.
    2. With the chosen config, sweep all turn counts and record
       (median time, peak mem reserved, sum of ``total_padded``).

Reports CSV / JSON to ``--output-dir``; plots are produced by
``plot_pressure_results.py`` in the same dir.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import ray

from axrl.configs import (
    DataloaderConfig,
    MCoreLrSchedulerConfig,
    MCoreOptimizerConfig,
    MegatronWorkerConfig,
    ModelConfig,
    SFTConfig,
)
from axrl.data import GenerationOutput, GenerationState, Sample, SampleTensorDict
from axrl.data.conversation import Conversation, Message
from axrl.data.rollout_trace import RolloutTrace
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.trainer.sft_trainer import SftTrainer
from axrl.utils import setup_logger

if TYPE_CHECKING:
    from axrl.utils.gpu_utils import GpuUsageInfo

logger = logging.getLogger(__name__)


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
NUM_NODES = 4
GPUS_PER_NODE = 8
TOTAL_GPUS = NUM_NODES * GPUS_PER_NODE  # 32
TRAJECTORIES_PER_BATCH = 4
TOKENS_PER_TURN = 1024
SEED_PROMPT_TOKENS = 64
TOOL_RESULTS_KEPT_IN_CONTEXT = 1
PLACEHOLDER_TEXT = "Tool result is omitted to save tokens."
NUM_BUILD_WORKERS = 32


def _synthetic_generation_output(output_ids: np.ndarray) -> GenerationOutput:
    return GenerationOutput(
        session_id="synthetic",
        output_ids=output_ids,
        output_logprobs=np.zeros(len(output_ids), dtype=np.float32),
        output_text="",
        output_text_with_special_tokens="",
        cached_tokens=0,
        finish_reason="stop",
        e2e_elapsed_seconds=0.0,
        stop_reason=None,
        retry=0,
    )


# Both inference and training share the same matrix turn count and turn
# sweep. The 32-512 range was chosen so the training matrix (16 cells *
# ~12-17 min/cell at 512 turns) fits a reasonable wall budget while still
# probing the high end where most parallel configs OOM.
MATRIX_TURNS: int = 512
TURN_COUNTS: tuple[int, ...] = (512, 256, 128, 64, 32)
# Quick fixed-config cell run before each phase's matrix walk; the result
# is discarded. Burns the first ~minute of init + first-pass JIT/NCCL
# warmup latency on a small turn count so the matrix-cell timings aren't
# polluted by one-time bring-up costs.
WARMUP_TURNS: int = 128


@dataclass(frozen=True)
class ParallelConfig:
    """One (tp, cp, pp, ep) point in the search matrix."""

    name: str
    tp: int
    cp: int
    pp: int
    ep: int

    def world_size(self) -> int:
        return self.tp * self.cp * self.pp  # dp = 1

    def expert_group_size(self) -> int:
        # ep * etp * pp must divide world; etp=1, dp=1 -> ep * pp <= 32.
        return self.ep * self.pp


# Constraints: ``tp * cp * pp = 32`` (dp=1, world=32), ``ep * pp <= 32``,
# ``etp=1``. EP is fixed at 8 or 4 (per user — Qwen3-30B-A3B has 128
# experts, ep=8 gives 16 experts/rank and ep=4 gives 32/rank; smaller ep
# values aren't load-balanced enough to be worth measuring).
PARALLEL_CONFIGS: tuple[ParallelConfig, ...] = (
    # cp-heavy
    ParallelConfig(name="cp32-pp1-ep8", tp=1, cp=32, pp=1, ep=8),
    ParallelConfig(name="cp32-pp1-ep4", tp=1, cp=32, pp=1, ep=4),
    # cp + pp
    ParallelConfig(name="cp16-pp2-ep8", tp=1, cp=16, pp=2, ep=8),
    ParallelConfig(name="cp16-pp2-ep4", tp=1, cp=16, pp=2, ep=4),
    ParallelConfig(name="cp8-pp4-ep8", tp=1, cp=8, pp=4, ep=8),
    ParallelConfig(name="cp8-pp4-ep4", tp=1, cp=8, pp=4, ep=4),
    ParallelConfig(name="cp4-pp8-ep4", tp=1, cp=4, pp=8, ep=4),
    # tp + cp + pp / tp + pp / tp + cp
    ParallelConfig(name="tp2-cp8-pp2-ep4", tp=2, cp=8, pp=2, ep=4),
    ParallelConfig(name="tp2-cp16-pp1-ep8", tp=2, cp=16, pp=1, ep=8),
    ParallelConfig(name="tp2-cp4-pp4-ep4", tp=2, cp=4, pp=4, ep=4),
    ParallelConfig(name="tp4-cp4-pp2-ep4", tp=4, cp=4, pp=2, ep=4),
    ParallelConfig(name="tp4-cp8-pp1-ep8", tp=4, cp=8, pp=1, ep=8),
    ParallelConfig(name="tp4-cp2-pp4-ep4", tp=4, cp=2, pp=4, ep=4),
    ParallelConfig(name="tp8-cp2-pp2-ep4", tp=8, cp=2, pp=2, ep=4),
    ParallelConfig(name="tp8-cp4-pp1-ep8", tp=8, cp=4, pp=1, ep=8),
    ParallelConfig(name="tp8-pp4-ep8", tp=8, cp=1, pp=4, ep=8),
)


# =====================================================================
# Synthetic trajectories — same construction style as the earlier sweep.
# =====================================================================


def _next_pow2_at_least(value: int, *, floor: int = 128) -> int:
    p = 1
    while p < value:
        p <<= 1
    return max(p, floor)


def _build_merged_trajectories(
    *,
    num_trajectories: int,
    turns_per_conversation: int,
    pad_to_max_length: int,
    rng: np.random.Generator,
) -> list[RolloutTrace]:
    """Multi-turn trajectories with hide-tool-results compaction.

    Each conversation has ``turns_per_conversation`` assistant turns separated
    by tool results; ``compact(max_recent_tool_results=3)`` runs after every
    tool result so the merged path exercises real prefix sharing.
    """
    placeholder_token_ids = rng.integers(low=10, high=10000, size=8, dtype=np.int32)
    traces: list[RolloutTrace] = []
    for _ in range(num_trajectories):
        seed_token_ids = rng.integers(low=10, high=10000, size=SEED_PROMPT_TOKENS, dtype=np.int32)
        seed_conversation = Conversation(
            messages=[Message(role="user", content="seed")],
            gen_state=GenerationState(input_ids=seed_token_ids),
        )
        trace = RolloutTrace(seed_conversation, token_in_token_out=True, max_length=pad_to_max_length)
        for turn_idx in range(turns_per_conversation):
            assistant_token_count = TOKENS_PER_TURN // 2
            assistant_token_ids = rng.integers(low=10, high=10000, size=assistant_token_count, dtype=np.int32)
            trace.append_assistant_message(_synthetic_generation_output(assistant_token_ids))
            if turn_idx == turns_per_conversation - 1:
                break
            tool_token_count = TOKENS_PER_TURN - assistant_token_count
            tool_token_ids = rng.integers(low=10, high=10000, size=tool_token_count, dtype=np.int32)
            trace.append_user_or_tool_message(content="t", tokens=tool_token_ids)
            trace.compact(
                max_recent_tool_results=TOOL_RESULTS_KEPT_IN_CONTEXT,
                placeholder_tokens=placeholder_token_ids,
                placeholder_text=PLACEHOLDER_TEXT,
            )
        traces.append(trace)
    return traces


def _build_samples_chunk(
    turns: int,
    num_trajectories: int,
    pad_to_max_length: int,
    chunk_seed: int,
) -> list[Sample]:
    """Worker entry: build one chunk of merged samples.

    ``merge_trajectory_samples`` builds the prefix trie via a recursive
    ``_insert``; recursion depth equals the path length (= number of turns).
    Python's default recursion limit (1000) can be hit at long turn
    counts, so bump it before calling ``to_sample()``.
    """
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))
    rng = np.random.default_rng(chunk_seed)
    traces = _build_merged_trajectories(
        num_trajectories=num_trajectories,
        turns_per_conversation=turns,
        pad_to_max_length=pad_to_max_length,
        rng=rng,
    )
    return [trace.to_sample() for trace in traces]


def _split_trajectories(num_trajectories: int, num_workers: int) -> list[int]:
    base, extra = divmod(num_trajectories, num_workers)
    return [base + (1 if i < extra else 0) for i in range(num_workers)]


def _build_merged_samples(turns: int, *, seed: int) -> tuple[list[Sample], int]:
    """Return ``(samples, pad_to_max_length)`` for the merged sweep."""
    pad_to_max_length = _next_pow2_at_least(turns * TOKENS_PER_TURN + 4096)
    effective_workers = min(NUM_BUILD_WORKERS, TRAJECTORIES_PER_BATCH)
    counts = _split_trajectories(TRAJECTORIES_PER_BATCH, effective_workers)
    args = [(turns, count, pad_to_max_length, seed + chunk_idx) for chunk_idx, count in enumerate(counts)]
    logger.info(
        "build: turns=%d trajectories=%d workers=%d pad_to=%d",
        turns,
        TRAJECTORIES_PER_BATCH,
        effective_workers,
        pad_to_max_length,
    )
    t_start = time.perf_counter()
    with mp.get_context("fork").Pool(effective_workers) as pool:
        chunks = pool.starmap(_build_samples_chunk, args)
    samples = [s for chunk in chunks for s in chunk]
    elapsed = time.perf_counter() - t_start
    logger.info("build: turns=%d -> %d samples in %.1fs", turns, len(samples), elapsed)
    return samples, pad_to_max_length


def _sum_total_padded(samples: list[Sample]) -> int:
    """Sum of merged-sample packed lengths — total work the kernel sees per step."""
    total = 0
    for s in samples:
        assert s.merge_info is not None, "magi merged sample requires merge_info"
        total += s.merge_info.total_padded
    return total


def _max_path_length(samples: list[Sample]) -> int:
    """Longest leaf-to-root path in tokens across the merged trees.

    For ``magi_merged_forward`` each sample carries a ``PrefixMergeInfo``
    whose ``max_path_len`` is the longest root→leaf chain — i.e. the
    longest unmerged conversation snapshot inside the trie.
    """
    return max((s.merge_info.max_path_len for s in samples if s.merge_info is not None), default=0)


def _max_total_padded(samples: list[Sample]) -> int:
    """Largest single-sample packed length = per-microbatch (mbs=1) length.

    Memory peak is governed by this, not the sum, because each microbatch
    only holds one merged sample's worth of activations / logits.
    """
    return max((s.merge_info.total_padded for s in samples if s.merge_info is not None), default=0)


# =====================================================================
# Worker config
# =====================================================================


RecomputeGranularity = Literal["full", "selective"]
RecomputeMethod = Literal["uniform", "block"]


@dataclass(frozen=True)
class RecomputeStrategy:
    """One recompute setting for ``MegatronWorkerConfig``.

    The default values match the original baseline (``full`` granularity
    with ``uniform`` over 1 layer at a time = recompute every layer).
    Setting ``granularity=None`` disables recompute entirely.
    """

    name: str
    granularity: RecomputeGranularity | None = "full"
    method: RecomputeMethod | None = "uniform"  # only used when granularity == "full"
    num_layers: int | None = 1
    modules: tuple[str, ...] | None = None  # for "selective"


# Module-level singleton so it can be a default argument without B008.
_DEFAULT_RECOMPUTE = RecomputeStrategy(name="full-uniform-1")


def _make_megatron_config(
    config: ParallelConfig,
    *,
    seq_length: int,
    global_batch_size: int,
    training: bool = False,
    optimizer_offload_fraction: float = 1.0,
    recompute: RecomputeStrategy = _DEFAULT_RECOMPUTE,
) -> MegatronWorkerConfig:
    """Worker config; ``training=True`` flips on optimizer + offload.

    ``optimizer_offload_fraction`` controls how much of the optimizer state
    lives on CPU (1.0 = all, 0.0 = none = on-GPU optimizer). ``recompute``
    parameterises activation recompute.
    """
    assert config.world_size() == TOTAL_GPUS, f"{config.name}: tp*cp*pp = {config.world_size()} != {TOTAL_GPUS} (dp must be 1)"
    if training:
        optimizer_cfg = MCoreOptimizerConfig(
            lr=1e-5,
            min_lr=1e-6,
            optimizer_cpu_offload=optimizer_offload_fraction > 0.0,
            optimizer_offload_fraction=optimizer_offload_fraction,
        )
    else:
        optimizer_cfg = MCoreOptimizerConfig(optimizer_cpu_offload=False)
    return MegatronWorkerConfig(
        model=ModelConfig(name=MODEL_NAME, trust_remote_code=True, seq_length=seq_length),
        seed=42,
        tp_size=config.tp,
        cp_size=config.cp,
        pp_size=config.pp,
        dp_size=1,
        ep_size=config.ep,
        etp_size=1,
        use_magi_merged_forward=True,
        enable_routing_replay=False,
        bf16=True,
        # verl-derived ``cast_output_layer_to_fp32``: the LM-head matmul
        # runs in fp32 and the output skips ``Float16Module``'s bf16->fp32
        # cast (which otherwise allocates fp32 logits *while* bf16 logits
        # are still alive). Trades a 1.5 GB transient fp32 weight copy
        # for ~one bf16-logits tensor of peak savings.
        enable_fp32_lm_head=True,
        global_batch_size=global_batch_size,
        train_micro_batch_size=1,
        eval_micro_batch_size=1,
        log_every_k_steps=1,
        log_gpu_usaegs=True,
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",
        recompute_granularity=recompute.granularity,
        recompute_method=recompute.method if recompute.granularity == "full" else None,
        recompute_num_layers=recompute.num_layers if recompute.granularity == "full" else None,
        recompute_modules=list(recompute.modules) if recompute.modules else None,
        data_loader=DataloaderConfig(num_workers=0),
        optimizer=optimizer_cfg,
        lr_scheduler=MCoreLrSchedulerConfig(
            lr_warmup_steps=10,
            lr_decay_steps=1000,
            lr_decay_style="constant",
        ),
        inference_only=not training,
    )


# =====================================================================
# Cell execution
# =====================================================================


@dataclass
class CellResult:
    parallel_config: str
    turns: int
    median_time_s: float | None = None
    peak_mem_reserved_gb: float | None = None
    sum_total_padded: int | None = None
    max_total_padded: int | None = None
    max_path_length: int | None = None
    seq_length: int | None = None
    notes: str | None = None
    error: str | None = None


def _max_peak_reserved_gb(gpu_usage_per_rank: list[GpuUsageInfo]) -> float:
    return max(info.peak_mem_reserved_gbs for info in gpu_usage_per_rank)


def _spawn_worker(config: MegatronWorkerConfig) -> RayMegatronWorker:
    # One bundle per node (8 GPUs each). A single big ``Request(gpu=32)``
    # bundle can never be placed because no single node has 32 GPUs.
    requests = [Request(cpu=1, gpu=GPUS_PER_NODE) for _ in range(NUM_NODES)]
    resource_group = ResourceGroup(requests, strategy="STRICT_SPREAD")
    return RayMegatronWorker(config=config, resource_group=resource_group)


def _kill_worker_hard(worker: RayMegatronWorker) -> None:
    """Tear down a worker without waiting for graceful shutdown.

    ``RayMegatronWorker.shutdown()`` does ``ray.get(refs)`` on each actor's
    `shutdown.remote()`, which hangs forever if any actor is stuck after a
    CUDA OOM. The matrix search needs to advance to the next config quickly,
    so we kill via ``ray.kill`` (force, no_restart) directly and drop the
    placement group.
    """
    actors = list(getattr(worker, "_remote_workers", []) or [])
    for actor in actors:
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)
    worker._remote_workers = []
    pg = getattr(getattr(worker, "resource_group", None), "pg", None)
    if pg is not None:
        with contextlib.suppress(Exception):
            ray.util.remove_placement_group(pg)


def _is_oom_error(exc: BaseException) -> bool:
    msg = repr(exc).lower()
    return "outofmemoryerror" in msg or "cuda out of memory" in msg or "rayoutofmemory" in msg


def run_one_cell(
    parallel_config: ParallelConfig,
    turns: int,
    *,
    measured_passes: int,
    samples: list[Sample],
    seq_length: int,
    label_extra: str = "",
) -> CellResult:
    """Run one cell: spawn worker, 1 warmup + ``measured_passes`` measured passes."""
    label = f"[{parallel_config.name} | turns={turns}{label_extra}]"
    cfg = _make_megatron_config(
        parallel_config,
        seq_length=seq_length,
        global_batch_size=len(samples),
    )
    sum_packed = _sum_total_padded(samples)
    max_packed = _max_total_padded(samples)
    max_path_len = _max_path_length(samples)
    logger.info(
        "%s starting. n_samples=%d seq_length=%d sum_total_padded=%d max_total_padded=%d max_path_length=%d",
        label,
        len(samples),
        seq_length,
        sum_packed,
        max_packed,
        max_path_len,
    )

    worker = _spawn_worker(cfg)
    row = CellResult(
        parallel_config=parallel_config.name,
        turns=turns,
        sum_total_padded=sum_packed,
        max_total_padded=max_packed,
        max_path_length=max_path_len,
        seq_length=seq_length,
    )

    try:
        worker.initialize()
        worker.set_trainer(SftTrainer(SFTConfig()))
        batch = SampleTensorDict.from_samples(samples)

        logger.info("%s warmup pass", label)
        worker.compute_logprobs(samples=batch, batch_size=len(samples))

        wall_times_s: list[float] = []
        peak_mems_gb: list[float] = []
        for pass_idx in range(measured_passes):
            t_start = time.perf_counter()
            _, gpu_usage_per_rank = worker.compute_logprobs(samples=batch, batch_size=len(samples))
            t_end = time.perf_counter()
            wall_s = t_end - t_start
            peak_gb = _max_peak_reserved_gb(gpu_usage_per_rank)
            wall_times_s.append(wall_s)
            peak_mems_gb.append(peak_gb)
            logger.info(
                "%s pass %d/%d: wall=%.3fs peak_mem_reserved=%.2fGB",
                label,
                pass_idx + 1,
                measured_passes,
                wall_s,
                peak_gb,
            )
        row.median_time_s = statistics.median(wall_times_s)
        row.peak_mem_reserved_gb = max(peak_mems_gb)
        logger.info(
            "%s median_time=%.3fs peak_mem=%.2fGB",
            label,
            row.median_time_s,
            row.peak_mem_reserved_gb,
        )
    except Exception as exc:
        if _is_oom_error(exc):
            logger.warning("%s OOM: %r", label, exc)
        else:
            logger.exception("%s crashed", label)
        row.error = repr(exc)
    finally:
        # Skip graceful shutdown — after OOM some actors hang forever in
        # ``shutdown.remote()``; force-kill so the matrix search can advance.
        _kill_worker_hard(worker)
        # Give the cluster a beat to release GPU memory before the next attempt.
        time.sleep(5)
    return row


def run_one_train_cell(
    parallel_config: ParallelConfig,
    turns: int,
    *,
    measured_passes: int,
    samples: list[Sample],
    seq_length: int,
    label_extra: str = "",
    optimizer_offload_fraction: float = 1.0,
    recompute: RecomputeStrategy = _DEFAULT_RECOMPUTE,
) -> CellResult:
    """Train cell: ``forward + backward + optimizer step`` with optimizer offload.

    Mirrors :func:`run_one_cell`; reports median wall time over
    ``measured_passes`` (1 warmup first) and peak GPU memory reserved.
    """
    label = f"[{parallel_config.name} | turns={turns} | TRAIN | offload={optimizer_offload_fraction} | recompute={recompute.name}{label_extra}]"
    cfg = _make_megatron_config(
        parallel_config,
        seq_length=seq_length,
        global_batch_size=len(samples),
        training=True,
        optimizer_offload_fraction=optimizer_offload_fraction,
        recompute=recompute,
    )
    sum_packed = _sum_total_padded(samples)
    max_packed = _max_total_padded(samples)
    max_path_len = _max_path_length(samples)
    logger.info(
        "%s starting. n_samples=%d seq_length=%d sum_total_padded=%d max_total_padded=%d max_path_length=%d",
        label,
        len(samples),
        seq_length,
        sum_packed,
        max_packed,
        max_path_len,
    )
    worker = _spawn_worker(cfg)
    row = CellResult(
        parallel_config=parallel_config.name,
        turns=turns,
        sum_total_padded=sum_packed,
        max_total_padded=max_packed,
        max_path_length=max_path_len,
        seq_length=seq_length,
        notes="training",
    )
    try:
        worker.initialize()
        worker.set_trainer(SftTrainer(SFTConfig()))
        batch = SampleTensorDict.from_samples(samples)

        logger.info("%s warmup train step", label)
        worker.train(global_step=0, samples=batch, data_shuffle_seed=0, compute_logprobs=False)

        wall_times_s: list[float] = []
        peak_mems_gb: list[float] = []
        for pass_idx in range(measured_passes):
            t_start = time.perf_counter()
            _, metrics = worker.train(
                global_step=pass_idx + 1,
                samples=batch,
                data_shuffle_seed=0,
                compute_logprobs=False,
            )
            t_end = time.perf_counter()
            wall_s = t_end - t_start
            # Per-rank GPU usage is logged under
            # ``train-GPU/<rank-name>/peak_mem_reserved_gbs`` and aggregated
            # with the ``["min", "max"]`` rule from
            # ``BaseTrainer.set_metric_agg_type``; the multi-agg suffix
            # appends ``_max``/``_min``. Match the ``_max`` keys and take
            # the cluster-wide max.
            rank_peaks = [float(v) for k, v in metrics.items() if k.endswith("peak_mem_reserved_gbs_max") and isinstance(v, (int, float))]
            peak_gb = max(rank_peaks) if rank_peaks else float("nan")
            wall_times_s.append(wall_s)
            if not math.isnan(peak_gb):
                peak_mems_gb.append(peak_gb)
            logger.info(
                "%s train pass %d/%d: wall=%.3fs peak_mem=%.2fGB",
                label,
                pass_idx + 1,
                measured_passes,
                wall_s,
                peak_gb,
            )
        row.median_time_s = statistics.median(wall_times_s)
        row.peak_mem_reserved_gb = max(peak_mems_gb) if peak_mems_gb else None
        logger.info(
            "%s median_time=%.3fs peak_mem=%.2fGB",
            label,
            row.median_time_s,
            row.peak_mem_reserved_gb if row.peak_mem_reserved_gb is not None else float("nan"),
        )
    except Exception as exc:
        if _is_oom_error(exc):
            logger.warning("%s OOM: %r", label, exc)
        else:
            logger.exception("%s crashed", label)
        row.error = repr(exc)
    finally:
        _kill_worker_hard(worker)
        time.sleep(5)
    return row


# =====================================================================
# Matrix search + sweep
# =====================================================================


@dataclass
class PressurePlan:
    output_dir: Path = Path("tmp/magi-bench-pressure")
    measured_passes: int = 5
    seed: int = 42
    # Matrix search runs at this turn count; the winner is then swept
    # across all turn counts in ``turns``.
    longest_turns: int = MATRIX_TURNS
    turns: tuple[int, ...] = TURN_COUNTS
    parallel_configs: tuple[ParallelConfig, ...] = PARALLEL_CONFIGS
    ray_address: str = ""  # if empty, the launcher set RAY_ADDRESS in env


def _connect_ray(address: str) -> None:
    if ray.is_initialized():
        return
    if address:
        ray.init(address=address)
    else:
        ray.init(address="auto")


def find_fitting_config(
    plan: PressurePlan,
    samples: list[Sample],
    seq_length: int,
) -> tuple[ParallelConfig | None, list[CellResult]]:
    """Walk the full matrix on the longest turn count and pick the fastest fit.

    Each attempt uses ``measured_passes=1`` (one warmup + one timed pass) so
    the matrix sweep is bounded; the winning config is then re-run by
    ``run_full_sweep`` with the full ``plan.measured_passes`` for stable
    medians. Returns ``(winner, attempts)``; ``winner`` is the config with
    the smallest measured wall-clock time (or ``None`` if every config OOMs).
    """
    attempts: list[CellResult] = []
    successes: list[tuple[ParallelConfig, CellResult]] = []
    for cfg in plan.parallel_configs:
        result = run_one_cell(
            cfg,
            plan.longest_turns,
            measured_passes=1,
            samples=samples,
            seq_length=seq_length,
            label_extra=" [matrix-search]",
        )
        attempts.append(result)
        _persist_results(attempts, plan.output_dir, filename="matrix_search.csv")
        if result.error is None and result.median_time_s is not None:
            successes.append((cfg, result))
        else:
            logger.warning("config %s failed at %d turns; trying next", cfg.name, plan.longest_turns)
    if not successes:
        return None, attempts
    winner_cfg, winner_result = min(successes, key=lambda pair: pair[1].median_time_s or float("inf"))
    logger.info(
        "matrix winner: %s at %d turns (time=%.3fs, %d/%d configs fit)",
        winner_cfg.name,
        plan.longest_turns,
        winner_result.median_time_s or 0.0,
        len(successes),
        len(plan.parallel_configs),
    )
    return winner_cfg, attempts


def run_full_sweep(
    plan: PressurePlan,
    parallel_config: ParallelConfig,
    longest_samples: list[Sample],
    longest_seq_length: int,
) -> list[CellResult]:
    """Run all turn counts with the chosen config, full ``measured_passes`` each."""
    rows: list[CellResult] = []

    longest_result = run_one_cell(
        parallel_config,
        plan.longest_turns,
        measured_passes=plan.measured_passes,
        samples=longest_samples,
        seq_length=longest_seq_length,
    )
    longest_result.notes = "winner of matrix search"
    rows.append(longest_result)
    _persist_results(rows, plan.output_dir)

    for turns in plan.turns:
        if turns == plan.longest_turns:
            continue
        rng_seed = plan.seed + turns * NUM_BUILD_WORKERS
        samples, seq_length = _build_merged_samples(turns, seed=rng_seed)
        result = run_one_cell(
            parallel_config,
            turns,
            measured_passes=plan.measured_passes,
            samples=samples,
            seq_length=seq_length,
        )
        rows.append(result)
        _persist_results(rows, plan.output_dir)
    return rows


def run_training_phase(plan: PressurePlan) -> list[CellResult]:
    """Walk the full parallel matrix at ``MATRIX_TURNS`` then sweep training.

    Training is much slower per pass than inference because of the
    optimizer-offload CPU↔GPU transfer (~12-17 minutes per pass at 512
    turns vs ~1 minute for inference). Each cell uses ``measured_passes=1``
    (1 warmup + 1 measured) so the full 16-cell matrix fits in ~3 hours.
    """
    rows: list[CellResult] = []
    matrix_turns = MATRIX_TURNS
    logger.info(
        "training matrix at %d turns over %d configs",
        matrix_turns,
        len(plan.parallel_configs),
    )
    rng_seed = plan.seed + matrix_turns * NUM_BUILD_WORKERS + 1
    matrix_samples, matrix_seq_length = _build_merged_samples(matrix_turns, seed=rng_seed)

    successes: list[tuple[ParallelConfig, CellResult]] = []
    for cfg in plan.parallel_configs:
        result = run_one_train_cell(
            cfg,
            matrix_turns,
            measured_passes=1,
            samples=matrix_samples,
            seq_length=matrix_seq_length,
            label_extra=" [train-matrix]",
        )
        rows.append(result)
        _persist_results(rows, plan.output_dir, filename="train_matrix_search.csv")
        if result.error is None and result.median_time_s is not None:
            successes.append((cfg, result))

    if not successes:
        logger.error("no parallel config fit training at %d turns", matrix_turns)
        _persist_results(rows, plan.output_dir, filename="train_results.csv")
        return rows

    winner_cfg, winner_result = min(successes, key=lambda pair: pair[1].median_time_s or float("inf"))
    logger.info(
        "train matrix winner: %s at %d turns (time=%.3fs, %d/%d configs fit)",
        winner_cfg.name,
        matrix_turns,
        winner_result.median_time_s or 0.0,
        len(successes),
        len(plan.parallel_configs),
    )

    sweep_rows: list[CellResult] = [winner_result]
    sweep_rows[0].notes = f"matrix winner ({matrix_turns} turns)"
    for turns in plan.turns:
        if turns == matrix_turns:
            continue
        rng_seed = plan.seed + turns * NUM_BUILD_WORKERS + 1
        samples, seq_length = _build_merged_samples(turns, seed=rng_seed)
        result = run_one_train_cell(
            winner_cfg,
            turns,
            measured_passes=1,
            samples=samples,
            seq_length=seq_length,
            label_extra=" [train-sweep]",
        )
        sweep_rows.append(result)
        _persist_results(sweep_rows, plan.output_dir, filename="train_results.csv")
    return rows + sweep_rows


# =====================================================================
# Optimization sweeps (offload fraction, recompute strategy, final tune)
# =====================================================================

# Optimizer-offload fraction sweep at the training winner.
OFFLOAD_FRACTIONS_TO_SWEEP: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# Recompute-strategy sweep at the training winner. ``selective`` keeps
# MLP activations and recomputes only what's named in ``modules`` —
# usually attention; megatron's default selective set is ``["core_attn"]``.
# ``full / uniform / 1`` is the original baseline (recompute every layer).
RECOMPUTE_STRATEGIES_TO_SWEEP: tuple[RecomputeStrategy, ...] = (
    RecomputeStrategy(name="none", granularity=None, method=None, num_layers=None),
    RecomputeStrategy(name="selective-core_attn", granularity="selective", modules=("core_attn",)),
    RecomputeStrategy(name="selective-mlp", granularity="selective", modules=("mlp",)),
    RecomputeStrategy(name="selective-core_attn+mlp", granularity="selective", modules=("core_attn", "mlp")),
    RecomputeStrategy(name="full-uniform-1", granularity="full", method="uniform", num_layers=1),
)

# Optimization phase target: bring train-step time at 512 turns down to
# ~3x the inference time at the same config (so a single train step is
# roughly forward + backward + offload-tax ~= 3x forward).
OPTIMIZATION_TURNS: int = 512
OPTIMIZATION_CONFIG_NAME: str = "tp4-cp8-pp1-ep8"  # current training matrix winner

# Final-pass candidate combinations. Each entry is
# ``(parallel-config-name, offload_fraction, recompute_strategy)``; the
# driver runs them in order at OPTIMIZATION_TURNS and reports the time
# vs the inference baseline so we can pick the best 3x-target config.
OPTIMIZATION_FINAL_CANDIDATES: tuple[tuple[str, float, RecomputeStrategy], ...] = (
    ("tp4-cp8-pp1-ep8", 0.0, RecomputeStrategy(name="selective-core_attn", granularity="selective", modules=("core_attn",))),
    ("tp4-cp8-pp1-ep8", 0.0, RecomputeStrategy(name="none", granularity=None, method=None, num_layers=None)),
    ("tp2-cp16-pp1-ep8", 0.0, RecomputeStrategy(name="selective-core_attn", granularity="selective", modules=("core_attn",))),
    ("tp8-cp4-pp1-ep8", 0.0, RecomputeStrategy(name="selective-core_attn", granularity="selective", modules=("core_attn",))),
    ("tp4-cp8-pp1-ep8", 0.25, RecomputeStrategy(name="selective-core_attn", granularity="selective", modules=("core_attn",))),
)


def _resolve_config_by_name(name: str) -> ParallelConfig:
    for cfg in PARALLEL_CONFIGS:
        if cfg.name == name:
            return cfg
    raise ValueError(f"unknown parallel config: {name}")


def run_optimization_phase(plan: PressurePlan) -> list[CellResult]:
    """Sweep optimizer-offload fraction and recompute strategy, then tune.

    Three rounds, all at ``OPTIMIZATION_TURNS`` turns and the current
    training matrix winner ``OPTIMIZATION_CONFIG_NAME``:

    1. Offload-fraction sweep with the baseline recompute. Persists
       ``train_offload_sweep.csv``.
    2. Recompute-strategy sweep with the best offload fraction from
       step 1. Persists ``train_recompute_sweep.csv``.
    3. ``OPTIMIZATION_FINAL_CANDIDATES`` — explicit (config, offload,
       recompute) combinations to try to hit the ~3x-of-inference target.
       Persists ``train_optimization_final.csv``.
    """
    rows: list[CellResult] = []
    base_cfg = _resolve_config_by_name(OPTIMIZATION_CONFIG_NAME)
    rng_seed = plan.seed + OPTIMIZATION_TURNS * NUM_BUILD_WORKERS + 2  # different seed than training matrix
    samples, seq_length = _build_merged_samples(OPTIMIZATION_TURNS, seed=rng_seed)
    baseline_recompute = RecomputeStrategy(name="full-uniform-1", granularity="full", method="uniform", num_layers=1)

    # ------------------------------------------------------------------
    # 1. Optimizer-offload-fraction sweep
    # ------------------------------------------------------------------
    logger.info("optimization: offload-fraction sweep at %d turns / %s", OPTIMIZATION_TURNS, base_cfg.name)
    offload_rows: list[CellResult] = []
    for fraction in OFFLOAD_FRACTIONS_TO_SWEEP:
        result = run_one_train_cell(
            base_cfg,
            OPTIMIZATION_TURNS,
            measured_passes=1,
            samples=samples,
            seq_length=seq_length,
            label_extra=" [opt-offload]",
            optimizer_offload_fraction=fraction,
            recompute=baseline_recompute,
        )
        result.notes = f"offload={fraction}"
        offload_rows.append(result)
        rows.append(result)
        _persist_results(offload_rows, plan.output_dir, filename="train_offload_sweep.csv")

    offload_successes = [r for r in offload_rows if r.error is None and r.median_time_s is not None and r.notes]
    if offload_successes:
        # Pick the fastest fitting fraction; fall back to 1.0 (always fits).
        best_offload_row = min(offload_successes, key=lambda r: r.median_time_s or float("inf"))
        assert best_offload_row.notes is not None
        best_offload_fraction = float(best_offload_row.notes.split("=", 1)[1])
    else:
        best_offload_fraction = 1.0
    logger.info("offload sweep best fraction: %s", best_offload_fraction)

    # ------------------------------------------------------------------
    # 2. Recompute-strategy sweep at best offload fraction
    # ------------------------------------------------------------------
    logger.info("optimization: recompute-strategy sweep with offload=%s", best_offload_fraction)
    recompute_rows: list[CellResult] = []
    for strategy in RECOMPUTE_STRATEGIES_TO_SWEEP:
        result = run_one_train_cell(
            base_cfg,
            OPTIMIZATION_TURNS,
            measured_passes=1,
            samples=samples,
            seq_length=seq_length,
            label_extra=" [opt-recompute]",
            optimizer_offload_fraction=best_offload_fraction,
            recompute=strategy,
        )
        result.notes = f"recompute={strategy.name};offload={best_offload_fraction}"
        recompute_rows.append(result)
        rows.append(result)
        _persist_results(recompute_rows, plan.output_dir, filename="train_recompute_sweep.csv")

    # ------------------------------------------------------------------
    # 3. Final candidate combinations targeting ~3x inference time
    # ------------------------------------------------------------------
    logger.info("optimization: final candidate combinations")
    final_rows: list[CellResult] = []
    for cfg_name, offload, strategy in OPTIMIZATION_FINAL_CANDIDATES:
        try:
            cfg = _resolve_config_by_name(cfg_name)
        except ValueError as exc:
            logger.warning("skipping final candidate %s: %s", cfg_name, exc)
            continue
        result = run_one_train_cell(
            cfg,
            OPTIMIZATION_TURNS,
            measured_passes=1,
            samples=samples,
            seq_length=seq_length,
            label_extra=" [opt-final]",
            optimizer_offload_fraction=offload,
            recompute=strategy,
        )
        result.notes = f"config={cfg_name};offload={offload};recompute={strategy.name}"
        final_rows.append(result)
        rows.append(result)
        _persist_results(final_rows, plan.output_dir, filename="train_optimization_final.csv")
    return rows


def _persist_results(
    rows: list[CellResult],
    output_dir: Path,
    *,
    filename: str = "results.csv",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename
    json_path = output_dir / filename.replace(".csv", ".json")
    with csv_path.open("w") as f:
        f.write(
            "parallel_config,turns,median_time_s,peak_mem_reserved_gb,sum_total_padded,max_total_padded,max_path_length,seq_length,notes,error\n",
        )
        for r in rows:
            f.write(
                f"{r.parallel_config},{r.turns},"
                f"{'' if r.median_time_s is None else f'{r.median_time_s:.6f}'},"
                f"{'' if r.peak_mem_reserved_gb is None else f'{r.peak_mem_reserved_gb:.4f}'},"
                f"{'' if r.sum_total_padded is None else r.sum_total_padded},"
                f"{'' if r.max_total_padded is None else r.max_total_padded},"
                f"{'' if r.max_path_length is None else r.max_path_length},"
                f"{'' if r.seq_length is None else r.seq_length},"
                f"{r.notes or ''},"
                f"{r.error or ''}\n",
            )
    with json_path.open("w") as f:
        json.dump([r.__dict__ for r in rows], f, indent=2)
    logger.info("persisted %d rows -> %s", len(rows), csv_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/magi-bench-pressure")
    parser.add_argument("--measured-passes", type=int, default=5)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", ""))
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--phase",
        choices=("all", "inference", "training", "optimization"),
        default="all",
        help=(
            "'inference' = matrix + turn sweep; 'training' = same flow with "
            "optimizer offload; 'optimization' = offload-fraction + recompute "
            "sweeps + final candidates; 'all' = inference + training."
        ),
    )
    args = parser.parse_args()
    setup_logger(args.log_level)

    plan = PressurePlan(
        output_dir=Path(args.output_dir),
        measured_passes=args.measured_passes,
        ray_address=args.ray_address,
    )
    plan.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("connecting to ray at %r", plan.ray_address or "auto")
    _connect_ray(plan.ray_address)
    logger.info("ray cluster resources: %s", ray.cluster_resources())

    failures: list[CellResult] = []

    warmup_cfg = plan.parallel_configs[0]
    warmup_samples, warmup_seq_length = _build_merged_samples(
        WARMUP_TURNS,
        seed=plan.seed + WARMUP_TURNS * NUM_BUILD_WORKERS - 1,
    )

    if args.phase in {"all", "inference"}:
        logger.info("inference warmup: %s @ %d turns (discarded)", warmup_cfg.name, WARMUP_TURNS)
        run_one_cell(
            warmup_cfg,
            WARMUP_TURNS,
            measured_passes=1,
            samples=warmup_samples,
            seq_length=warmup_seq_length,
            label_extra=" [warmup-discard]",
        )
        # Build the longest-turn samples once and reuse them across the matrix
        # search (the input data is identical across parallel configs).
        longest_samples, longest_seq_length = _build_merged_samples(
            plan.longest_turns,
            seed=plan.seed + plan.longest_turns * NUM_BUILD_WORKERS,
        )
        winner, attempts = find_fitting_config(plan, longest_samples, longest_seq_length)
        if winner is None:
            logger.error("no parallel config fit %d turns. attempts:", plan.longest_turns)
            for a in attempts:
                logger.error("  %s: %s", a.parallel_config, a.error)
            return 1

        logger.info("winner config for %d turns: %s", plan.longest_turns, winner.name)
        rows = run_full_sweep(plan, winner, longest_samples, longest_seq_length)
        failures.extend(r for r in rows if r.error)
        logger.info("inference sweep done. %d/%d cells succeeded.", len(rows) - len(failures), len(rows))

    if args.phase in {"all", "training"}:
        logger.info("training warmup: %s @ %d turns (discarded)", warmup_cfg.name, WARMUP_TURNS)
        run_one_train_cell(
            warmup_cfg,
            WARMUP_TURNS,
            measured_passes=1,
            samples=warmup_samples,
            seq_length=warmup_seq_length,
            label_extra=" [warmup-discard]",
        )
        train_rows = run_training_phase(plan)
        train_failures = [r for r in train_rows if r.error]
        failures.extend(train_failures)
        logger.info("training phase done. %d/%d cells succeeded.", len(train_rows) - len(train_failures), len(train_rows))

    if args.phase == "optimization":
        logger.info("optimization warmup: %s @ %d turns (discarded)", warmup_cfg.name, WARMUP_TURNS)
        run_one_train_cell(
            warmup_cfg,
            WARMUP_TURNS,
            measured_passes=1,
            samples=warmup_samples,
            seq_length=warmup_seq_length,
            label_extra=" [warmup-discard]",
        )
        opt_rows = run_optimization_phase(plan)
        opt_failures = [r for r in opt_rows if r.error]
        failures.extend(opt_failures)
        logger.info("optimization phase done. %d/%d cells succeeded.", len(opt_rows) - len(opt_failures), len(opt_rows))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
