"""Weight sync benchmark: BF16 vs FP8.

Measures weight sync time for:
1. BF16 → BF16 (standard path)
2. BF16 → FP8 (with blockwise quantization)

Uses the same Megatron model (BF16) for both, varying only the rollout model.
"""

from __future__ import annotations

import asyncio
import logging
from statistics import fmean, stdev
from typing import TYPE_CHECKING

from axrl.configs import MegatronWorkerConfig, MetricLoggerConfig, ModelConfig, RolloutWorkerConfig, SamplingConfig
from axrl.controller.stage_manager import ColocatedStageManager
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.ray_rollout_worker import RayRolloutWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import setup_logger
from axrl.utils.timer import Timer

if TYPE_CHECKING:
    from axrl.utils.gpu_utils import GpuUsageInfo

logger = logging.getLogger(__name__)

BF16_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FP8_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

WARMUP_UPDATES = 2
MEASURED_UPDATES = 10
BUCKET_SIZE_GB = 2.0


def _get_peak_reserved_gb(usage_infos: list[GpuUsageInfo]) -> float:
    return max(info.peak_mem_reserved_gbs for info in usage_infos)


async def run_benchmark(
    *,
    rollout_model_name: str,
    megatron_model_name: str,
    label: str,
) -> dict:
    """Run a single weight sync benchmark."""
    ray_utils.restart()

    megatron_model = ModelConfig(name=megatron_model_name, trust_remote_code=True, seq_length=64)
    rollout_model = ModelConfig(name=rollout_model_name, trust_remote_code=True, seq_length=64)

    rollout_config = RolloutWorkerConfig(
        engine_type="sglang",
        model=rollout_model,
        sampling_config=SamplingConfig(temperature=0.0, max_total_tokens=64),
        tp_size=8,
        ep_size=8,
        num_workers=1,
        gpu_memory_utilization=0.7,
        load_dummy_weights=False,
        dtype="auto",
    )

    megatron_config = MegatronWorkerConfig(
        model=megatron_model,
        tp_size=2,
        dp_size=1,
        cp_size=4,
        ep_size=8,
        etp_size=1,
        pp_size=1,
        vpp_size=None,
        inference_only=True,
        metric_logger_config=MetricLoggerConfig(logger_type="console"),
    )

    resource_group = ResourceGroup([Request(gpu=8, cpu=1)])
    rollout_worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(rollout_config, resource_group))
    megatron_worker = RayMegatronWorker(config=megatron_config, resource_group=resource_group)
    stage_manager = ColocatedStageManager(rollout_worker=rollout_worker, megatron_worker=megatron_worker)

    with Timer("Startup", verbose=True) as startup_timer:
        megatron_worker.initialize()
        megatron_worker.build_weight_updater(rollout_worker, bucket_size_gb=BUCKET_SIZE_GB)
        megatron_worker.connect_rollout_worker()
        await stage_manager.switch_to_weight_sync()

    update_times: list[float] = []
    peak_mems: list[float] = []

    total_updates = WARMUP_UPDATES + MEASURED_UPDATES
    for i in range(total_updates):
        with Timer(f"Update {i + 1}", verbose=True) as t:
            usage_infos = megatron_worker.update_rollout_model_weights()

        peak_gb = _get_peak_reserved_gb(usage_infos)
        if i < WARMUP_UPDATES:
            logger.info(f"[{label}] Warmup {i + 1}/{WARMUP_UPDATES}: {t.elapsed_seconds:.3f}s, peak={peak_gb:.2f}GB")
        else:
            update_times.append(t.elapsed_seconds)
            peak_mems.append(peak_gb)
            logger.info(f"[{label}] Measured {len(update_times)}/{MEASURED_UPDATES}: {t.elapsed_seconds:.3f}s, peak={peak_gb:.2f}GB")

    await stage_manager.switch_to_rollout()
    rollout_worker.shutdown()
    megatron_worker.shutdown()

    result = {
        "label": label,
        "rollout_model": rollout_model_name,
        "megatron_model": megatron_model_name,
        "startup_seconds": startup_timer.elapsed_seconds,
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": len(update_times),
        "update_mean_seconds": fmean(update_times),
        "update_std_seconds": stdev(update_times) if len(update_times) > 1 else 0.0,
        "update_min_seconds": min(update_times),
        "update_max_seconds": max(update_times),
        "peak_mem_max_gb": max(peak_mems),
        "all_update_times": update_times,
    }
    return result


async def main() -> None:
    setup_logger("info")

    results = []

    # BF16 benchmark
    logger.info("=" * 60)
    logger.info("Running BF16 weight sync benchmark")
    logger.info("=" * 60)
    bf16_result = await run_benchmark(
        rollout_model_name=BF16_MODEL,
        megatron_model_name=BF16_MODEL,
        label="BF16→BF16",
    )
    results.append(bf16_result)

    # FP8 benchmark
    logger.info("=" * 60)
    logger.info("Running FP8 weight sync benchmark")
    logger.info("=" * 60)
    fp8_result = await run_benchmark(
        rollout_model_name=FP8_MODEL,
        megatron_model_name=BF16_MODEL,
        label="BF16→FP8",
    )
    results.append(fp8_result)

    # Print comparison
    print("\n" + "=" * 80)
    print("WEIGHT SYNC BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model: {BF16_MODEL} (30B MoE, 128 experts)")
    print("Config: tp=2, cp=4, ep=8 (Megatron) → tp=8, ep=8 (SGLang)")
    print(f"Bucket size: {BUCKET_SIZE_GB} GB")
    print(f"Warmup: {WARMUP_UPDATES}, Measured: {MEASURED_UPDATES}")
    print()
    print(f"{'Metric':<35} {'BF16→BF16':>15} {'BF16→FP8':>15} {'Speedup':>10}")
    print("-" * 80)

    bf16 = results[0]
    fp8 = results[1]

    speedup = bf16["update_mean_seconds"] / fp8["update_mean_seconds"] if fp8["update_mean_seconds"] > 0 else 0

    print(f"{'Update time (mean)':<35} {bf16['update_mean_seconds']:>14.3f}s {fp8['update_mean_seconds']:>14.3f}s {speedup:>9.2f}x")
    print(f"{'Update time (std)':<35} {bf16['update_std_seconds']:>14.3f}s {fp8['update_std_seconds']:>14.3f}s")
    print(f"{'Update time (min)':<35} {bf16['update_min_seconds']:>14.3f}s {fp8['update_min_seconds']:>14.3f}s")
    print(f"{'Update time (max)':<35} {bf16['update_max_seconds']:>14.3f}s {fp8['update_max_seconds']:>14.3f}s")
    print(f"{'Peak GPU reserved (max)':<35} {bf16['peak_mem_max_gb']:>13.2f}GB {fp8['peak_mem_max_gb']:>13.2f}GB")
    print(f"{'Startup time':<35} {bf16['startup_seconds']:>14.1f}s {fp8['startup_seconds']:>14.1f}s")
    print("=" * 80)

    # Also print individual update times
    for r in results:
        print(f"\n{r['label']} individual times: {[f'{t:.3f}s' for t in r['all_update_times']]}")


if __name__ == "__main__":
    asyncio.run(main())
