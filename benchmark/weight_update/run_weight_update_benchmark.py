from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from statistics import fmean

from weight_update_benchmark_config import WeightUpdateBenchmarkConfig

from axrl.controller.stage_manager import ColocatedStageManager
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.ray_rollout_worker import RayRolloutWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import setup_logger
from axrl.utils.config_utils import load_and_validate_config
from axrl.utils.gpu_utils import GpuUsageInfo, assert_all_gpus_empty
from axrl.utils.logger import MetricLogger, get_metric_logger
from axrl.utils.timer import Timer

logger = logging.getLogger(__name__)


def _validate_config(config: WeightUpdateBenchmarkConfig) -> None:
    total_rollout_gpus = config.rollout_worker.gpus_per_worker() * config.rollout_worker.num_workers
    total_megatron_gpus = config.megatron_worker.world_size()

    assert config.warmup_updates >= 0, "warmup_updates must be >= 0"
    assert config.measured_updates > 0, "measured_updates must be > 0"
    assert config.max_runtime_seconds > 0, "max_runtime_seconds must be > 0"
    assert total_rollout_gpus == total_megatron_gpus, (
        f"Colocated benchmark requires matching total GPUs, got rollout={total_rollout_gpus}, megatron={total_megatron_gpus}"
    )
    assert config.rollout_worker.model == config.megatron_worker.model, "Rollout and Megatron models must match"
    assert not config.rollout_worker.load_dummy_weights, "Benchmark must use real rollout weights"


def _build_resource_group(config: WeightUpdateBenchmarkConfig) -> ResourceGroup:
    requests = [Request(gpu=config.rollout_worker.gpus_per_worker(), cpu=1) for _ in range(config.rollout_worker.num_workers)]
    return ResourceGroup(requests)


def _get_peak_reserved_gb(usage_infos: list[GpuUsageInfo]) -> float:
    return max(info.peak_mem_reserved_gbs for info in usage_infos)


def _log_update_metrics(metric_logger: MetricLogger, step: int, update_seconds: float, peak_reserved_gb: float) -> None:
    metric_logger.log_scalars(
        {
            "benchmark/time/update_seconds": update_seconds,
            "benchmark/gpu/update_peak_reserved_gb": peak_reserved_gb,
        },
        step=step,
    )


async def main() -> None:  # noqa: PLR0915
    config_path = Path(__file__).with_name("weight_update_benchmark.yaml")
    config = load_and_validate_config(
        WeightUpdateBenchmarkConfig,
        config_path=str(config_path),
        print_configs=True,
    )
    setup_logger(config.log_level)
    _validate_config(config)

    if config.restart_ray:
        ray_utils.restart()

    config.megatron_worker.metric_logger_config = config.logger.model_copy()
    metric_logger = get_metric_logger(config.logger)
    metric_logger.log_config(config)

    rollout_worker: RayRolloutWorker | None = None
    megatron_worker: RayMegatronWorker | None = None
    stage_manager: ColocatedStageManager | None = None
    startup_seconds = 0.0
    total_wall_start = time.perf_counter()
    update_times: list[float] = []
    peak_reserved_gbs: list[float] = []
    truncated = False
    summary_metrics: dict[str, float] = {}

    try:
        resource_group = _build_resource_group(config)
        rollout_worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(config.rollout_worker, resource_group))
        megatron_worker = RayMegatronWorker(config=config.megatron_worker, resource_group=resource_group)
        stage_manager = ColocatedStageManager(rollout_worker=rollout_worker, megatron_worker=megatron_worker)

        with Timer("Benchmark startup", verbose=True) as startup_timer:
            megatron_worker.initialize()
            megatron_worker.build_weight_updater(rollout_worker, bucket_size_gb=config.bucket_size_gb)
            megatron_worker.connect_rollout_worker()
            await stage_manager.switch_to_weight_sync()
        startup_seconds = startup_timer.elapsed_seconds

        metric_logger.log_scalars(
            {
                "benchmark/time/startup_seconds": startup_seconds,
                "benchmark/config/warmup_updates": float(config.warmup_updates),
                "benchmark/config/measured_updates": float(config.measured_updates),
                "benchmark/config/bucket_size_gb": config.bucket_size_gb,
            },
            step=0,
        )

        total_updates = config.warmup_updates + config.measured_updates
        for update_idx in range(total_updates):
            elapsed = time.perf_counter() - total_wall_start
            if elapsed >= config.max_runtime_seconds:
                truncated = True
                logger.warning("Stopping benchmark early because the runtime budget was exhausted.")
                break

            with Timer(f"Weight update {update_idx + 1}", verbose=True) as update_timer:
                usage_infos = megatron_worker.update_rollout_model_weights()

            if update_idx < config.warmup_updates:
                logger.info(f"Warmup update {update_idx + 1}/{config.warmup_updates} finished in {update_timer.elapsed_seconds:.2f}s.")
                continue

            step = len(update_times) + 1
            peak_reserved_gb = _get_peak_reserved_gb(usage_infos)
            update_times.append(update_timer.elapsed_seconds)
            peak_reserved_gbs.append(peak_reserved_gb)
            _log_update_metrics(
                metric_logger=metric_logger,
                step=step,
                update_seconds=update_timer.elapsed_seconds,
                peak_reserved_gb=peak_reserved_gb,
            )

        summary_metrics = {
            "benchmark/summary/completed_updates": float(len(update_times)),
            "benchmark/summary/target_updates": float(config.measured_updates),
            "benchmark/summary/truncated": 1.0 if truncated else 0.0,
            "benchmark/summary/startup_seconds": startup_seconds,
        }
        if update_times:
            summary_metrics["benchmark/summary/update_seconds_mean"] = fmean(update_times)
            summary_metrics["benchmark/summary/update_seconds_max"] = max(update_times)
            summary_metrics["benchmark/summary/update_peak_reserved_gb_max"] = max(peak_reserved_gbs)
    finally:
        if stage_manager is not None and rollout_worker is not None and megatron_worker is not None:
            with Timer("Restore rollout placement", verbose=True):
                await stage_manager.switch_to_rollout()
        with Timer("Benchmark shutdown", verbose=True) as shutdown_timer:
            if rollout_worker is not None:
                rollout_worker.shutdown()
            if megatron_worker is not None:
                megatron_worker.shutdown()
        shutdown_seconds = shutdown_timer.elapsed_seconds

        summary_metrics["benchmark/summary/shutdown_seconds"] = shutdown_seconds
        summary_metrics["benchmark/summary/total_wall_seconds"] = time.perf_counter() - total_wall_start
        metric_logger.log_scalars(summary_metrics, step=len(update_times))

        if rollout_worker is not None and megatron_worker is not None:
            logger.info(f"Benchmark completed with {len(update_times)} measured updates.")
            assert_all_gpus_empty(max_used_gb=2)
        metric_logger.close()


if __name__ == "__main__":
    setup_logger("info")
    asyncio.run(main())
