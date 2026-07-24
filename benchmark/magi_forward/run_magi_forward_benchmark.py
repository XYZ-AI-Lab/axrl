"""Benchmark Magi merged forward vs gptmodel_forward.

Two sweeps over ``compute_logprobs`` (correctness already covered by
``tests/mcore/test_rollout_trace.py`` and ``tests/magi-forward/``). Both use
cp=8, Qwen3-4B, full activation recompute, no optimizer offload, mbs=2.

  Sweep "flat" (Fig 1, 2):
      x: ``seq_length`` per sample in {8192, 4096, 1024, 128}
      Each trajectory is single-turn; the magi side wraps it as a 1-path
      ``PrefixMergeInfo`` (via ``RolloutTrace.to_sample()``) so it goes
      through ``magi_merged_gptmodel_forward`` exactly the same way a
      multi-turn merged sample does.

  Sweep "merged" (Fig 3, 4):
      x: ``turns`` per conversation in {256, 128, 64, 16, 4}, ~1K tokens/turn
      Hide-tool-results compaction keeps the most recent 3 tool results so
      the merged side actually exercises real prefix-tree merging. Each
      trajectory uses a unique random seed prompt so trajectories share no
      prefix across the batch.

The forward variant is selected by ``MegatronWorkerConfig.use_magi_merged_forward``.
The flat side feeds in ``RolloutTrace.turn_samples`` (one Sample per assistant
turn, padded together) so both paths cover the same total work.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import multiprocessing as mp
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from axrl.configs import (
    DataloaderConfig,
    MCoreLrSchedulerConfig,
    MCoreOptimizerConfig,
    MegatronWorkerConfig,
    ModelConfig,
)
from axrl.data import GenerationOutput, GenerationState, Sample, SampleTensorDict
from axrl.data.conversation import Conversation, Message
from axrl.data.rollout_trace import RolloutTrace
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import setup_logger

if TYPE_CHECKING:
    from axrl.utils.gpu_utils import GpuUsageInfo

logger = logging.getLogger(__name__)


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
TOTAL_GPUS = 8  # cp=8, tp=pp=dp=ep=1 -> world_size = 8
TRAJECTORIES_PER_BATCH = 128
TOKENS_PER_TURN = 1024
SEED_PROMPT_TOKENS = 64
TOOL_RESULTS_KEPT_IN_CONTEXT = 3
PAD_TOKEN_ID = 0
PLACEHOLDER_TEXT = "Tool result is omitted to save tokens."
NUM_BUILD_WORKERS = 64

# Sweep names.
FLAT = "flat"
MERGED = "merged"

# Forward methods.
MAGI_MERGED = "magi_merged"
GPTMODEL_FORWARD = "gptmodel_forward"


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


# Flat sweep: seq_length per sample, largest first so OOM (if any) shows up early.
FLAT_SEQ_LENGTHS: tuple[int, ...] = (262144, 131072, 65536, 32768, 8192, 4096, 1024)
# Merged sweep: turns per conversation, largest first.
MERGED_TURN_COUNTS: tuple[int, ...] = (256, 128, 64, 16, 4)


def _next_pow2_at_least(value: int, *, floor: int = 128) -> int:
    p = 1
    while p < value:
        p <<= 1
    return max(p, floor)


def _build_flat_trajectories(
    *,
    num_trajectories: int,
    seq_length: int,
    pad_to_max_length: int,
    rng: np.random.Generator,
) -> list[RolloutTrace]:
    """Single-turn trajectories with unique random tokens, total = ``seq_length``."""
    traces: list[RolloutTrace] = []
    for _ in range(num_trajectories):
        seed_token_count = min(SEED_PROMPT_TOKENS, seq_length // 4)
        seed_token_ids = rng.integers(low=10, high=10000, size=seed_token_count, dtype=np.int32)
        seed_conversation = Conversation(
            messages=[Message(role="user", content="seed")],
            gen_state=GenerationState(input_ids=seed_token_ids),
        )
        trace = RolloutTrace(seed_conversation, token_in_token_out=True, max_length=pad_to_max_length)
        assistant_token_count = seq_length - seed_token_count
        assert assistant_token_count > 0, f"seq_length {seq_length} too small for seed length {seed_token_count}"
        assistant_token_ids = rng.integers(low=10, high=10000, size=assistant_token_count, dtype=np.int32)
        trace.append_assistant_message(_synthetic_generation_output(assistant_token_ids))
        assert trace.running_len == seq_length, f"flat trajectory len {trace.running_len} != target {seq_length}"
        traces.append(trace)
    return traces


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
            is_last_turn = turn_idx == turns_per_conversation - 1
            if is_last_turn:
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


@dataclass
class BenchmarkCase:
    """One (sweep, size, method) point — produces one CSV row."""

    sweep: str
    size: int  # seq_length when sweep == FLAT, turns when sweep == MERGED
    method: str

    @property
    def label(self) -> str:
        size_name = "seq_length" if self.sweep == FLAT else "turns"
        return f"[{self.sweep} | {size_name}={self.size} | {self.method}]"


@dataclass
class BenchmarkRow:
    sweep: str
    size: int
    method: str
    median_time_s: float | None = None
    peak_mem_reserved_gb: float | None = None
    error: str | None = None


def _make_megatron_config(
    *,
    use_magi_merged_forward: bool,
    seq_length: int,
    global_batch_size: int,
    micro_batch_size: int,
) -> MegatronWorkerConfig:
    """Worker config shared by every benchmark case.

    cp=8, tp=pp=dp=ep=1, full activation recompute, on-GPU optimizer (no
    offload), bf16, ``inference_only=True`` so the worker skips
    optimizer/scheduler init (we only call ``compute_logprobs``).
    """
    return MegatronWorkerConfig(
        model=ModelConfig(name=MODEL_NAME, trust_remote_code=True, seq_length=seq_length),
        seed=42,
        tp_size=1,
        cp_size=8,
        pp_size=1,
        dp_size=1,
        ep_size=1,
        etp_size=1,
        use_magi_merged_forward=use_magi_merged_forward,
        bf16=True,
        global_batch_size=global_batch_size,
        train_micro_batch_size=micro_batch_size,
        eval_micro_batch_size=micro_batch_size,
        log_every_k_steps=1,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        data_loader=DataloaderConfig(num_workers=0),
        optimizer=MCoreOptimizerConfig(optimizer_cpu_offload=False),
        lr_scheduler=MCoreLrSchedulerConfig(
            lr_warmup_steps=10,
            lr_decay_steps=1000,
            lr_decay_style="constant",
        ),
        inference_only=True,
    )


def _build_samples_chunk(
    sweep: str,
    size: int,
    method: str,
    num_trajectories: int,
    pad_to_max_length: int,
    chunk_seed: int,
) -> list[Sample]:
    """Worker entry: build one chunk of trajectories and return per-method samples.

    Each worker owns its slice of trajectories, has its own ``np.random.Generator``
    seeded deterministically off the case key + chunk index, and returns the
    samples its slice produced. The driver concatenates chunks in order.
    """
    rng = np.random.default_rng(chunk_seed)
    if sweep == FLAT:
        traces = _build_flat_trajectories(
            num_trajectories=num_trajectories,
            seq_length=size,
            pad_to_max_length=pad_to_max_length,
            rng=rng,
        )
    else:
        traces = _build_merged_trajectories(
            num_trajectories=num_trajectories,
            turns_per_conversation=size,
            pad_to_max_length=pad_to_max_length,
            rng=rng,
        )
    if method == MAGI_MERGED:
        return [trace.to_sample() for trace in traces]
    return [turn_sample for trace in traces for turn_sample in trace.turn_samples]


def _split_trajectories(num_trajectories: int, num_workers: int) -> list[int]:
    """Split ``num_trajectories`` across ``num_workers`` as evenly as possible."""
    base, extra = divmod(num_trajectories, num_workers)
    return [base + (1 if i < extra else 0) for i in range(num_workers)]


def _build_samples_for_case(
    case: BenchmarkCase,
    *,
    seed: int,
    num_workers: int = NUM_BUILD_WORKERS,
) -> tuple[list[Sample], int]:
    """Return ``(samples, seq_length_for_worker_config)``.

    The magi side returns one merged ``Sample`` per trajectory (with
    ``merge_info`` set). The gptmodel_forward side returns one ``Sample`` per
    assistant turn so the same total work is covered.

    Trajectories are produced in parallel across ``num_workers`` processes — the
    O(turns^2) Python-level cost of building per-turn snapshots and then
    trie-packing each merged sample bottlenecks the driver otherwise.
    """
    if case.sweep == FLAT:
        pad_to_max_length = _next_pow2_at_least(case.size + 256)
    else:
        assert case.sweep == MERGED, f"unknown sweep {case.sweep}"
        # Each turn ~1K tokens; merged sample ~ turns * 1K plus alignment + seed.
        pad_to_max_length = _next_pow2_at_least(case.size * TOKENS_PER_TURN + 4096)

    effective_workers = min(num_workers, TRAJECTORIES_PER_BATCH)
    per_worker_counts = _split_trajectories(TRAJECTORIES_PER_BATCH, effective_workers)
    args = [
        (
            case.sweep,
            case.size,
            case.method,
            count,
            pad_to_max_length,
            seed + chunk_idx,
        )
        for chunk_idx, count in enumerate(per_worker_counts)
    ]
    logger.info(
        "%s building %d trajectories with %d workers",
        case.label,
        TRAJECTORIES_PER_BATCH,
        effective_workers,
    )
    t_start = time.perf_counter()
    with mp.get_context("fork").Pool(effective_workers) as pool:
        chunks = pool.starmap(_build_samples_chunk, args)
    elapsed = time.perf_counter() - t_start
    samples = [s for chunk in chunks for s in chunk]
    logger.info("%s built %d samples in %.1fs", case.label, len(samples), elapsed)
    return samples, pad_to_max_length


def _restart_ray_with_retry(max_attempts: int = 5, backoff_seconds: float = 30.0) -> None:
    """``ray.init()`` occasionally times out after many shutdown/init cycles.

    (the GCS reports "deadline exceeded" / "raylet failed to startup"). Wrap it
    with retries + backoff so a transient startup hiccup doesn't lose the run.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            ray_utils.restart()
            return
        except Exception as exc:
            if attempt == max_attempts:
                raise
            logger.warning(
                "ray restart attempt %d/%d failed (%s); sleeping %.0fs",
                attempt,
                max_attempts,
                exc,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)


def run_case(
    case: BenchmarkCase,
    *,
    measured_passes: int,
    micro_batch_size: int,
    seed: int,
) -> BenchmarkRow:
    """Spawn a fresh worker, run 1 warmup pass + ``measured_passes`` measured passes."""
    samples, seq_length = _build_samples_for_case(case, seed=seed)
    config = _make_megatron_config(
        use_magi_merged_forward=case.method == MAGI_MERGED,
        seq_length=seq_length,
        # gbs == total samples: one outer iteration; the worker's internal
        # microbatch loop drives the actual mbs stride.
        global_batch_size=len(samples),
        micro_batch_size=micro_batch_size,
    )

    logger.info(
        "%s starting. n_samples=%d seq_length=%d",
        case.label,
        len(samples),
        seq_length,
    )

    _restart_ray_with_retry()
    resource_group = ResourceGroup([Request(cpu=1, gpu=TOTAL_GPUS)])
    worker = RayMegatronWorker(config=config, resource_group=resource_group)
    row = BenchmarkRow(sweep=case.sweep, size=case.size, method=case.method)

    try:
        worker.initialize()
        batch = SampleTensorDict.from_samples(samples)

        logger.info("%s warmup pass", case.label)
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
                case.label,
                pass_idx + 1,
                measured_passes,
                wall_s,
                peak_gb,
            )

        row.median_time_s = statistics.median(wall_times_s)
        row.peak_mem_reserved_gb = max(peak_mems_gb)
        logger.info(
            "%s median_time=%.3fs peak_mem=%.2fGB",
            case.label,
            row.median_time_s,
            row.peak_mem_reserved_gb,
        )
    except Exception as exc:
        logger.exception("%s crashed", case.label)
        row.error = repr(exc)
    finally:
        with contextlib.suppress(Exception):
            worker.shutdown()
    return row


def _max_peak_reserved_gb(gpu_usage_per_rank: list[GpuUsageInfo]) -> float:
    return max(info.peak_mem_reserved_gbs for info in gpu_usage_per_rank)


def _build_cases_for_sweep(sweep: str) -> list[BenchmarkCase]:
    if sweep == FLAT:
        sizes: tuple[int, ...] = FLAT_SEQ_LENGTHS
    elif sweep == MERGED:
        sizes = MERGED_TURN_COUNTS
    else:
        raise ValueError(f"unknown sweep {sweep}")
    return [BenchmarkCase(sweep=sweep, size=size, method=method) for size in sizes for method in (MAGI_MERGED, GPTMODEL_FORWARD)]


@dataclass
class BenchmarkPlan:
    output_dir: Path = Path("tmp/magi-bench")
    measured_passes: int = 5
    micro_batch_size: int = 2
    seed: int = 42
    sweeps: list[str] = field(default_factory=lambda: [FLAT, MERGED])


def _load_existing_rows(output_dir: Path) -> list[BenchmarkRow]:
    """Read previously persisted rows from ``output_dir/results.csv`` if it exists."""
    csv_path = output_dir / "results.csv"
    if not csv_path.exists():
        return []
    rows: list[BenchmarkRow] = []
    with csv_path.open() as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\n").split(",", 5)
            if len(parts) != 6:
                continue
            sweep, size_str, method, time_str, mem_str, error = parts
            rows.append(
                BenchmarkRow(
                    sweep=sweep,
                    size=int(size_str),
                    method=method,
                    median_time_s=float(time_str) if time_str else None,
                    peak_mem_reserved_gb=float(mem_str) if mem_str else None,
                    error=error or None,
                ),
            )
    return rows


def run_benchmark_plan(plan: BenchmarkPlan, *, resume: bool = False) -> list[BenchmarkRow]:
    plan.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = _load_existing_rows(plan.output_dir) if resume else []
    done_keys = {(r.sweep, r.size, r.method) for r in rows}
    if rows:
        logger.info("resuming with %d previously persisted rows", len(rows))

    for sweep in plan.sweeps:
        for case_idx, case in enumerate(_build_cases_for_sweep(sweep)):
            if (case.sweep, case.size, case.method) in done_keys:
                logger.info("%s already done (resume) — skipping", case.label)
                continue
            # Bump the seed deterministically per case so every case sees fresh
            # random tokens but the run is reproducible.
            case_seed = plan.seed + case_idx * NUM_BUILD_WORKERS * 4
            row = run_case(
                case,
                measured_passes=plan.measured_passes,
                micro_batch_size=plan.micro_batch_size,
                seed=case_seed,
            )
            rows.append(row)
            _persist_rows(rows, plan.output_dir)
    return rows


def _persist_rows(rows: list[BenchmarkRow], output_dir: Path) -> None:
    """Write CSV + JSON after every case so partial sweeps are recoverable."""
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    with csv_path.open("w") as f:
        f.write("sweep,size,method,median_time_s,peak_mem_reserved_gb,error\n")
        for r in rows:
            f.write(
                f"{r.sweep},{r.size},{r.method},"
                f"{'' if r.median_time_s is None else f'{r.median_time_s:.6f}'},"
                f"{'' if r.peak_mem_reserved_gb is None else f'{r.peak_mem_reserved_gb:.4f}'},"
                f"{r.error or ''}\n",
            )
    with json_path.open("w") as f:
        json.dump([r.__dict__ for r in rows], f, indent=2)
    logger.info("persisted %d rows -> %s", len(rows), csv_path)


def run_smoke() -> None:
    """Tiny end-to-end smoke test (both methods, both sweeps) with 4 trajectories."""
    global TRAJECTORIES_PER_BATCH
    saved = TRAJECTORIES_PER_BATCH
    TRAJECTORIES_PER_BATCH = 4
    try:
        out_dir = Path("tmp/magi-bench-smoke")
        out_dir.mkdir(parents=True, exist_ok=True)
        smoke_cases = [
            BenchmarkCase(sweep=FLAT, size=128, method=MAGI_MERGED),
            BenchmarkCase(sweep=FLAT, size=128, method=GPTMODEL_FORWARD),
            BenchmarkCase(sweep=MERGED, size=2, method=MAGI_MERGED),
            BenchmarkCase(sweep=MERGED, size=2, method=GPTMODEL_FORWARD),
        ]
        rows: list[BenchmarkRow] = []
        for case_idx, case in enumerate(smoke_cases):
            row = run_case(
                case,
                measured_passes=1,
                micro_batch_size=2,
                seed=case_idx * NUM_BUILD_WORKERS,
            )
            rows.append(row)
            _persist_rows(rows, out_dir)
            assert row.error is None, f"smoke case failed: {row.error}"
        logger.info("smoke OK")
    finally:
        TRAJECTORIES_PER_BATCH = saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Magi merged forward vs gptmodel_forward.")
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke test only.")
    parser.add_argument("--sweep", choices=[FLAT, MERGED, "both"], default="both")
    parser.add_argument("--output-dir", default="tmp/magi-bench", help="Output dir for CSV / JSON / plots.")
    parser.add_argument("--measured-passes", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already present in <output-dir>/results.csv.",
    )
    args = parser.parse_args()
    setup_logger(args.log_level)

    if args.smoke:
        run_smoke()
        return 0

    sweeps = [FLAT, MERGED] if args.sweep == "both" else [args.sweep]
    plan = BenchmarkPlan(
        output_dir=Path(args.output_dir),
        measured_passes=args.measured_passes,
        micro_batch_size=args.micro_batch_size,
        sweeps=sweeps,
    )
    rows = run_benchmark_plan(plan, resume=args.resume)
    failures = [r for r in rows if r.error]
    if failures:
        logger.error("%d cases failed:", len(failures))
        for r in failures:
            logger.error("  %s/%d/%s: %s", r.sweep, r.size, r.method, r.error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
