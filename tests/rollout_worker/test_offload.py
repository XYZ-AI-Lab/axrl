from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import pytest

from axrl.utils import gpu_utils, setup_logger
from tests.test_configs import all_engine_types, default_engine_type, make_worker
from tests.test_configs import qwen25_config as rollout_config

if TYPE_CHECKING:
    from axrl.configs import EngineType
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.worker.rollout_worker import RolloutWorker

from axrl.ray import ray_utils

logger = logging.getLogger(__name__)


def get_total_gpu_used_gb() -> float:
    return sum(info.used_gb for info in gpu_utils.get_gpu_memory_info())


async def _test_model_offload(
    worker: RolloutWorker | RayRolloutWorker,
) -> None:
    worker.initialize()
    assert not await worker.is_gpu_memory_released()

    initial_usage = get_total_gpu_used_gb()
    logger.info("GPU usage after init: %s", gpu_utils.get_gpu_memory_info())

    await worker.release_gpu_memory(backup_weights_on_cpu=False)
    assert await worker.is_gpu_memory_released()
    after_weights_offload = get_total_gpu_used_gb()
    logger.info("GPU usage after offloading weights: %s", gpu_utils.get_gpu_memory_info())
    assert after_weights_offload < initial_usage

    await worker.resume_gpu_memory(["weights"])
    assert await worker.is_gpu_memory_released()
    after_weights_load = get_total_gpu_used_gb()
    logger.info("GPU usage after loading weights back: %s", gpu_utils.get_gpu_memory_info())
    assert after_weights_load > after_weights_offload

    await worker.resume_gpu_memory(["kv_cache"])
    assert not await worker.is_gpu_memory_released()
    after_kvcache_load = get_total_gpu_used_gb()
    logger.info("GPU usage after loading kv_cache back: %s", gpu_utils.get_gpu_memory_info())
    assert after_kvcache_load > after_weights_load

    # test repeated release
    await worker.release_gpu_memory(backup_weights_on_cpu=False)
    await worker.release_gpu_memory(backup_weights_on_cpu=False)
    assert await worker.is_gpu_memory_released()

    # test repeated resume
    await worker.resume_gpu_memory(tags=["weights"])
    await worker.resume_gpu_memory(tags=["weights"])
    await worker.resume_gpu_memory(tags=["kv_cache"])
    await worker.resume_gpu_memory(tags=["kv_cache"])
    assert not await worker.is_gpu_memory_released()

    worker.shutdown()


async def _test_offload_rollout_worker(engine_type: EngineType, *, use_ray_worker: bool) -> None:
    config = rollout_config.model_copy(deep=True)
    config.engine_type = engine_type
    worker = make_worker(
        config=config,
        use_ray_worker=use_ray_worker,
    )
    await _test_model_offload(worker)
    if use_ray_worker:
        ray_utils.stop()


@pytest.mark.parametrize("engine_type", all_engine_types)
@pytest.mark.parametrize("use_ray_worker", [True])
def test_offload_rollout_worker(engine_type: EngineType, *, use_ray_worker: bool) -> None:
    asyncio.run(_test_offload_rollout_worker(engine_type=engine_type, use_ray_worker=use_ray_worker))


if __name__ == "__main__":
    setup_logger("info")
    test_offload_rollout_worker(default_engine_type, use_ray_worker=False)
    test_offload_rollout_worker(default_engine_type, use_ray_worker=True)
