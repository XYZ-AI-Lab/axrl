import logging
import shutil

import pytest

from axrl.example.config_examples import get_megatron_trainer_config
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import setup_logger
from tests.test_configs import checkpoint_test_configs

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "model_name",
    list(checkpoint_test_configs.keys()),
)
def test_ray_megatron_worker_checkpoint_save_and_load_no_crash(model_name: str) -> None:
    """Checkpoint + load should not crash.

    This is intentionally a smoke test (no weight-consistency checks).
    """
    setup_logger("info")

    rollout_cfg = checkpoint_test_configs[model_name]

    megatron_cfg = get_megatron_trainer_config(
        tp_size=2,
        dp_size=1,
        pp_size=1,
        cp_size=2,
        model_config=rollout_cfg.model,
    )

    # Reuse a single checkpoint folder to avoid accumulating test output.
    # Clean it before and after each test case.
    megatron_cfg.checkpoint_dir = "pytest/test_ray_megatron_checkpoint"
    checkpoint_dir = megatron_cfg.get_checkpoint_dir()
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    assert not checkpoint_dir.exists()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ray_utils.restart()

    resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=4)])
    worker: RayMegatronWorker | None = None
    worker = RayMegatronWorker(config=megatron_cfg, resource_group=resource_group)
    worker.initialize()
    worker.save_checkpoint(0)
    loaded_step = worker.load_checkpoint()
    logger.info(f"Loaded checkpoint for {model_name}: step={loaded_step}")
    worker.shutdown()
    ray_utils.stop()
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    assert not checkpoint_dir.exists()


if __name__ == "__main__":
    for _model_name in checkpoint_test_configs:
        test_ray_megatron_worker_checkpoint_save_and_load_no_crash(_model_name)
