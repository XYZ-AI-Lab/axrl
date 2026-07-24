from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from axrl.configs import EngineType, MegatronWorkerConfig, ModelConfig, RolloutWorkerConfig
from axrl.example.config_examples import get_megatron_trainer_config
from axrl.ray import ray_utils
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.worker.rollout_worker import RolloutWorker

if TYPE_CHECKING:
    from axrl.ray.ray_rollout_worker import RayRolloutWorker

default_engine_type: EngineType = "sglang"
all_engine_types: list[EngineType] = ["sglang"]


qwen25_config = RolloutWorkerConfig(
    engine_type=default_engine_type,
    model=ModelConfig(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        seq_length=8192,
    ),
    gpu_memory_utilization=0.6,
    num_workers=2,
    tp_size=2,
    enable_metrics=False,
)

qwen3_config = RolloutWorkerConfig(
    engine_type=default_engine_type,
    model=ModelConfig(
        name="Qwen/Qwen3-1.7B",
        seq_length=8192,
    ),
    gpu_memory_utilization=0.6,
    num_workers=2,
    tp_size=2,
    enable_metrics=False,
)

deepseek_distill_qwen_config = RolloutWorkerConfig(
    engine_type=default_engine_type,
    model=ModelConfig(
        name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        seq_length=8192,
    ),
    gpu_memory_utilization=0.6,
    num_workers=2,
    tp_size=2,
    enable_metrics=False,
)

checkpoint_test_configs = {
    "qwen2.5": qwen25_config,
    "qwen3": qwen3_config,
    "deepseek_distill_qwen_1.5B": deepseek_distill_qwen_config,
}


def make_worker(config: RolloutWorkerConfig, *, use_ray_worker: bool) -> RolloutWorker | RayRolloutWorker:
    if not use_ray_worker:
        return RolloutWorker.get_worker(config)
    from axrl.ray.ray_rollout_worker import RayRolloutWorker

    ray_utils.restart()
    resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=2)] * 2)
    return RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(config, resource_group))


@dataclass
class RunConfig:
    tp_size: int = 1
    pp_size: int = 2
    vpp_size: int | None = None
    cp_size: int = 1
    dp_size: int = 1

    def total_gpus(self) -> int:
        return self.tp_size * self.pp_size * self.cp_size * self.dp_size


def get_consistency_checking_configs(model_config: ModelConfig | None = None) -> list[MegatronWorkerConfig]:
    import torch

    run_configs = [
        RunConfig(tp_size=1, pp_size=1, vpp_size=None, cp_size=1, dp_size=1),
        # pp
        RunConfig(tp_size=1, pp_size=2, vpp_size=None, cp_size=1, dp_size=1),
        # VPP testing is disabled until Megatron Bridge stage checks handle it.
        # Check /opt/conda/lib/python3.12/site-packages/megatron/bridge/models/conversion/model_bridge.py
        # RunConfig(tp_size=1, pp_size=2, vpp_size=2, cp_size=1, dp_size=1),
        # tp
        RunConfig(tp_size=2, pp_size=1, vpp_size=None, cp_size=1, dp_size=1),
        RunConfig(tp_size=4, pp_size=1, vpp_size=None, cp_size=1, dp_size=1),
        # cp
        RunConfig(tp_size=1, pp_size=1, vpp_size=None, cp_size=2, dp_size=1),
        RunConfig(tp_size=1, pp_size=1, vpp_size=None, cp_size=4, dp_size=1),
        # dp
        RunConfig(tp_size=1, pp_size=1, vpp_size=None, cp_size=1, dp_size=2),
        RunConfig(tp_size=1, pp_size=1, vpp_size=None, cp_size=1, dp_size=4),
        # mix (pp, tp)
        # RunConfig(tp_size=2, pp_size=2, vpp_size=2, cp_size=1, dp_size=1),
        # mix (pp, cp)
        # RunConfig(tp_size=1, pp_size=2, vpp_size=2, cp_size=2, dp_size=1),
        # mix (pp, dp)
        # RunConfig(tp_size=1, pp_size=2, vpp_size=2, cp_size=1, dp_size=2),
        # mix (tp, cp)
        RunConfig(tp_size=2, pp_size=1, vpp_size=None, cp_size=2, dp_size=1),
        # mix (tp, dp)
        RunConfig(tp_size=2, pp_size=1, vpp_size=None, cp_size=1, dp_size=2),
        # mix (cp, dp)
        RunConfig(tp_size=1, pp_size=1, vpp_size=None, cp_size=2, dp_size=2),
    ]
    available_gpus = torch.cuda.device_count()
    worker_configs: list[MegatronWorkerConfig] = []
    for run_config in run_configs:
        assert run_config.total_gpus() <= available_gpus, f"Not enough GPUs for the configuration: {run_config}."
        config = get_megatron_trainer_config(
            pp_size=run_config.pp_size,
            vpp_size=run_config.vpp_size,
            dp_size=run_config.dp_size,
            cp_size=run_config.cp_size,
            tp_size=run_config.tp_size,
            model_config=model_config,
        )
        config.inference_only = True
        worker_configs.append(config)
    return worker_configs
