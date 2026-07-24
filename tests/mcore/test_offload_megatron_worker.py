import logging
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from torch_memory_saver.utils import get_binary_path_from_package

from axrl.configs import MegatronWorkerConfig
from axrl.example.config_examples import get_megatron_trainer_config
from axrl.utils import dist_utils, gpu_utils, setup_logger
from axrl.worker.megatron_worker import MegatronWorker

logger = logging.getLogger(__name__)


@contextmanager
def torch_memory_saver_env():  # noqa: ANN201
    dynlib = Path(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))
    prev = {k: os.environ.get(k) for k in ("LD_PRELOAD", "TMS_INIT_ENABLE", "TMS_INIT_ENABLE_CPU_BACKUP")}
    os.environ["LD_PRELOAD"] = str(dynlib)
    os.environ["TMS_INIT_ENABLE"] = "1"
    os.environ["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
    try:
        yield
    finally:
        for key, val in prev.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def run_trainer(rank: int, world_size: int, num_gpus_per_node: int, config: MegatronWorkerConfig) -> None:
    worker = MegatronWorker(
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=rank % num_gpus_per_node,
        master_ip="127.0.0.1",
        master_port=12345,
    )
    worker.initialize()
    # loaded to GPU
    gpu_infos1 = gpu_utils.get_gpu_memory_info()
    logger.info(f"Initial GPU memory info: {gpu_infos1}")

    # offload to CPU
    worker.to_cpu()
    gpu_infos2 = gpu_utils.get_gpu_memory_info()
    assert gpu_infos2[0].used_ratio < gpu_infos1[0].used_ratio, "GPU memory should be freed after offloading to CPU"

    # load back to GPU
    worker.to_gpu()
    gpu_infos3 = gpu_utils.get_gpu_memory_info()
    assert gpu_infos3[0].used_ratio > gpu_infos2[0].used_ratio, "GPU memory should be used after loading back to GPU"

    dist_utils.cleanup_distributed()
    logger.info(f"Training completed for {worker.mcore_dist_info}.")


def _test_train(config: MegatronWorkerConfig) -> None:
    world_size = config.tp_size * config.dp_size * config.pp_size * config.cp_size
    num_gpus_per_node: int = 4
    logger.info(f"World size: {world_size}, {num_gpus_per_node} gpus per node")
    assert world_size <= torch.cuda.device_count()
    torch.multiprocessing.spawn(  # type: ignore
        run_trainer,
        args=(world_size, num_gpus_per_node, config),
        nprocs=world_size,
        join=True,
    )


def test_offload_megatron_worker() -> None:
    setup_logger("info")
    with torch_memory_saver_env():
        config = get_megatron_trainer_config(pp_size=1, vpp_size=None, dp_size=1)
        _test_train(config.model_copy())


if __name__ == "__main__":
    test_offload_megatron_worker()
