import logging

import pytest
import torch

from axrl.example.config_examples import get_megatron_trainer_config
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import gpu_utils, setup_logger

logger = logging.getLogger(__name__)


def get_total_gpu_used_gb() -> float:
    return sum(info.used_gb for info in gpu_utils.get_gpu_memory_info())


def test_offload_ray_megatron_worker() -> None:
    setup_logger("info")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("CUDA GPU is required for RayMegatronWorker offload test")

    ray_utils.restart()

    config = get_megatron_trainer_config(tp_size=2, dp_size=2, pp_size=1, cp_size=1)
    resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=4)])
    worker = RayMegatronWorker(config=config, resource_group=resource_group)

    worker.initialize()
    initial_usage = get_total_gpu_used_gb()
    logger.info("GPU usage after init: %s", gpu_utils.get_gpu_memory_info())

    worker.to_cpu()
    after_offload = get_total_gpu_used_gb()
    logger.info("GPU usage after to_cpu: %s", gpu_utils.get_gpu_memory_info())
    assert after_offload < initial_usage, "GPU memory should be freed after offloading to CPU"

    worker.to_gpu()
    after_reload = get_total_gpu_used_gb()
    logger.info("GPU usage after to_gpu: %s", gpu_utils.get_gpu_memory_info())
    assert after_reload > after_offload, "GPU memory should be used after loading back to GPU"

    worker.shutdown()
    ray_utils.stop()


if __name__ == "__main__":
    setup_logger("info")
    test_offload_ray_megatron_worker()
